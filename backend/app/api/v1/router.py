import asyncio
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Response,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel

from app.api.dependencies import current_user
from app.core.config import get_settings
from app.core.security import create_access_token, verify_password
from app.db.artifacts import ArtifactStore
from app.db.client import check_connection, get_database
from app.repositories.events import EventRepository
from app.repositories.experiments import ExperimentRepository
from app.research.dataset_loader import inspect_directory
from app.schemas.experiment import Exchange, ExperimentCreate, ExperimentState
from app.services.hsg_adapter import AlgorithmStatus, HSGAdapter
from app.services.market_calendar import MarketCalendarService
from app.services.reporting import generate_pdf, generate_research_archive

router = APIRouter(prefix="/api/v1")
settings = get_settings()
adapter = HSGAdapter()
india_calendar = MarketCalendarService(settings.configuration_directory / "market_holidays.json")
calendars = {
    Exchange.NSE: india_calendar,
    Exchange.BSE: india_calendar,
    Exchange.NASDAQ: MarketCalendarService(
        settings.configuration_directory / "nasdaq_market_holidays.json"
    ),
}
User = Annotated[str, Depends(current_user)]


class LoginRequest(BaseModel):
    username: str
    password: str


def serialize(document: dict[str, Any]) -> dict[str, Any]:
    output = dict(document)
    if "_id" in output:
        output["id"] = str(output.pop("_id"))
    return output


def checkpoint_exchanges(checkpoint: dict[str, Any]) -> set[str]:
    configured = checkpoint.get("trained_exchanges")
    return set(configured) if configured else {Exchange.NSE.value, Exchange.BSE.value}


@router.post("/auth/login")
async def login(payload: LoginRequest) -> dict[str, str]:
    if not settings.admin_password_hash:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ADMIN_PASSWORD_HASH_NOT_CONFIGURED",
        )
    if payload.username != settings.admin_username or not verify_password(
        payload.password, settings.admin_password_hash
    ):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="INVALID_CREDENTIALS")
    return {"access_token": create_access_token(payload.username), "token_type": "bearer"}


@router.get("/auth/me")
async def me(user: User) -> dict[str, str]:
    return {"username": user}


@router.get("/system/algorithm")
async def algorithm_status() -> dict[str, Any]:
    return adapter.status()


@router.get("/system/market")
async def market_status(exchange: Exchange = Query(default=Exchange.NSE)) -> dict[str, Any]:
    now = datetime.now(UTC)
    calendar = calendars[exchange]
    session = calendar.current_session(now)
    return {
        "exchange": exchange.value,
        "status": "OPEN" if calendar.is_market_open(now) else "CLOSED",
        "timestamp": now,
        "timezone": str(calendar.timezone),
        "session": {
            "name": session.name,
            "opens_at": session.opens_at,
            "closes_at": session.closes_at,
        }
        if session
        else None,
        "next_market_open": calendar.next_market_open(now),
    }


@router.get("/system/worker")
async def worker_status() -> dict[str, Any]:
    if not await check_connection():
        return {"status": "UNAVAILABLE", "heartbeat": None}
    state = await get_database().system_state.find_one({"_id": "worker"})
    return serialize(state) if state else {"status": "NOT_STARTED", "heartbeat": None}


@router.get("/system/model")
async def model_status() -> dict[str, Any]:
    if not await check_connection():
        return {"status": "UNAVAILABLE", "model_version": None, "created_at": None}
    checkpoint = await get_database().model_checkpoints.find_one(
        {"active": True}, sort=[("created_at", -1)]
    )
    if not checkpoint:
        return {"status": "NOT_TRAINED", "model_version": None, "created_at": None}
    if checkpoint.get("algorithm_sha256") != adapter.algorithm_sha256():
        return {
            "status": "INCOMPATIBLE",
            "model_version": checkpoint.get("model_version"),
            "created_at": checkpoint.get("created_at"),
            "algorithm_sha256": checkpoint.get("algorithm_sha256"),
            "trained_exchanges": sorted(checkpoint_exchanges(checkpoint)),
        }
    return {
        "status": "READY",
        "model_version": checkpoint.get("model_version"),
        "created_at": checkpoint.get("created_at"),
        "algorithm_sha256": checkpoint.get("algorithm_sha256"),
        "trained_exchanges": sorted(checkpoint_exchanges(checkpoint)),
    }


@router.get("/system/settings")
async def safe_settings(_: User) -> dict[str, Any]:
    return {
        "app_env": settings.app_env,
        "market_timezone": settings.market_timezone,
        "quote_stale_seconds": settings.quote_stale_seconds,
        "scan_interval_seconds": settings.scan_interval_seconds,
        "transaction_cost_bps": settings.transaction_cost_bps,
        "slippage_bps": settings.slippage_bps,
        "max_positions": settings.max_positions,
        "max_position_pct": settings.max_position_pct,
        "model_confidence_threshold": settings.model_confidence_threshold,
        "raw_tick_persistence": settings.raw_tick_persistence,
    }


@router.get("/datasets")
async def datasets() -> list[dict[str, Any]]:
    return [
        item.to_dict()
        for item in await asyncio.to_thread(inspect_directory, settings.data_directory)
    ]


@router.get("/datasets/{filename}")
async def dataset(filename: str) -> dict[str, Any]:
    allowed = {path.name: path for path in settings.data_directory.glob("*.csv")}
    if filename not in allowed:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="DATASET_NOT_FOUND")
    items = await asyncio.to_thread(inspect_directory, settings.data_directory)
    return next(item.to_dict() for item in items if item.filename == filename)


@router.post("/experiments", status_code=status.HTTP_201_CREATED)
async def create_experiment(payload: ExperimentCreate, _: User) -> dict[str, Any]:
    if not await check_connection():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="MONGODB_UNAVAILABLE")
    checkpoint = await get_database().model_checkpoints.find_one(
        {"active": True}, sort=[("created_at", -1)]
    )
    if payload.exchange.value not in checkpoint_exchanges(checkpoint or {}):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail=f"MODEL_NOT_TRAINED_FOR_{payload.exchange.value}",
        )
    return await ExperimentRepository(get_database()).create(payload)


@router.get("/experiments")
async def list_experiments(_: User) -> list[dict[str, Any]]:
    if not await check_connection():
        return []
    return await ExperimentRepository(get_database()).list()


@router.get("/experiments/{experiment_id}")
async def get_experiment(experiment_id: str, _: User) -> dict[str, Any]:
    if not await check_connection():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="MONGODB_UNAVAILABLE")
    result = await ExperimentRepository(get_database()).get(experiment_id)
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="EXPERIMENT_NOT_FOUND")
    return result


@router.post("/experiments/{experiment_id}/start")
async def start_experiment(experiment_id: str, _: User) -> dict[str, Any]:
    if not await check_connection():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="MONGODB_UNAVAILABLE")
    repository = ExperimentRepository(get_database())
    initializing = await repository.transition(
        experiment_id, ExperimentState.INITIALIZING, started_at=datetime.now(UTC)
    )
    if not initializing:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="EXPERIMENT_NOT_FOUND")
    algorithm = adapter.validate_contract()
    if algorithm.status is not AlgorithmStatus.INSTALLED:
        failed = await repository.transition(
            experiment_id,
            ExperimentState.FAILED,
            ended_at=datetime.now(UTC),
            error_code="ALGORITHM_NOT_INSTALLED"
            if algorithm.status is AlgorithmStatus.NOT_INSTALLED
            else "ALGORITHM_CONTRACT_INVALID",
        )
        await EventRepository(get_database()).emit(
            experiment_id,
            "HSGAgent",
            "HSG_INFERENCE_FAILED",
            severity="ERROR",
            details={"algorithm_status": algorithm.status.value},
        )
        return failed or initializing
    checkpoint = await get_database().model_checkpoints.find_one({"active": True})
    if not checkpoint:
        failed = await repository.transition(
            experiment_id,
            ExperimentState.FAILED,
            ended_at=datetime.now(UTC),
            error_code="MODEL_CHECKPOINT_NOT_FOUND",
        )
        await EventRepository(get_database()).emit(
            experiment_id,
            "HSGAgent",
            "MODEL_CHECKPOINT_NOT_FOUND",
            severity="ERROR",
        )
        return failed or initializing
    if checkpoint.get("algorithm_sha256") != algorithm.algorithm_sha256:
        failed = await repository.transition(
            experiment_id,
            ExperimentState.FAILED,
            ended_at=datetime.now(UTC),
            error_code="MODEL_CHECKPOINT_INCOMPATIBLE",
        )
        await EventRepository(get_database()).emit(
            experiment_id,
            "HSGAgent",
            "MODEL_CHECKPOINT_INCOMPATIBLE",
            severity="ERROR",
        )
        return failed or initializing
    exchange = Exchange(initializing.get("exchange", Exchange.NSE.value))
    if exchange.value not in checkpoint_exchanges(checkpoint):
        failed = await repository.transition(
            experiment_id,
            ExperimentState.FAILED,
            ended_at=datetime.now(UTC),
            error_code=f"MODEL_NOT_TRAINED_FOR_{exchange.value}",
        )
        await EventRepository(get_database()).emit(
            experiment_id,
            "HSGAgent",
            "MODEL_MARKET_UNSUPPORTED",
            severity="ERROR",
            details={"exchange": exchange.value},
        )
        return failed or initializing
    target = (
        ExperimentState.RUNNING
        if calendars[exchange].is_market_open(datetime.now(UTC))
        else ExperimentState.WAITING_FOR_MARKET
    )
    return await repository.transition(experiment_id, target) or initializing


@router.post("/experiments/{experiment_id}/pause")
async def pause_experiment(experiment_id: str, _: User) -> dict[str, Any]:
    result = await ExperimentRepository(get_database()).transition(
        experiment_id, ExperimentState.PAUSED
    )
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="EXPERIMENT_NOT_FOUND")
    return result


@router.post("/experiments/{experiment_id}/resume")
async def resume_experiment(experiment_id: str, _: User) -> dict[str, Any]:
    repository = ExperimentRepository(get_database())
    experiment = await repository.get(experiment_id)
    if not experiment:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="EXPERIMENT_NOT_FOUND")
    target = (
        ExperimentState.RUNNING
        if calendars[Exchange(experiment.get("exchange", Exchange.NSE.value))].is_market_open(
            datetime.now(UTC)
        )
        else ExperimentState.WAITING_FOR_MARKET
    )
    result = await repository.transition(experiment_id, target)
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="EXPERIMENT_NOT_FOUND")
    return result


@router.post("/experiments/{experiment_id}/cancel")
async def cancel_experiment(experiment_id: str, _: User) -> dict[str, Any]:
    result = await ExperimentRepository(get_database()).transition(
        experiment_id, ExperimentState.CANCELLED, ended_at=datetime.now(UTC)
    )
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="EXPERIMENT_NOT_FOUND")
    return result


@router.post("/experiments/{experiment_id}/reports", status_code=status.HTTP_202_ACCEPTED)
async def generate_report(experiment_id: str, _: User) -> dict[str, Any]:
    if not await check_connection():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail="MONGODB_UNAVAILABLE")
    database = get_database()
    experiment = await ExperimentRepository(database).get(experiment_id)
    if not experiment:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="EXPERIMENT_NOT_FOUND")

    async def records(collection_name: str) -> list[dict[str, Any]]:
        cursor = database[collection_name].find({"experiment_id": experiment_id})
        return [serialize(item) async for item in cursor]

    metrics_rows = await records("model_metrics")
    metrics = metrics_rows[-1] if metrics_rows else {}
    pdf = await asyncio.to_thread(generate_pdf, experiment, metrics)
    archive_payload = {
        "report_pdf": pdf,
        "experiment": experiment,
        "configuration": experiment,
        "dataset_manifest": experiment.get("dataset_hashes", {}),
        "trades": await records("trades"),
        "orders": await records("orders"),
        "predictions": await records("predictions"),
        "scanner_snapshots": await records("scanner_snapshots"),
        "portfolio_equity": await records("portfolios"),
        "financial_metrics": metrics,
        "model_metrics": metrics,
        "data_quality_events": await records("data_quality_events"),
        "agent_events": await records("agent_events"),
        "model_manifest": experiment.get("model_manifest", {}),
    }
    archive = await asyncio.to_thread(generate_research_archive, archive_payload)
    metadata = {"experiment_id": experiment_id, "created_at": datetime.now(UTC)}
    pdf_id = await ArtifactStore(database, "reports").put(
        f"{experiment_id}-report.pdf", pdf, metadata
    )
    archive_id = await ArtifactStore(database, "archives").put(
        f"{experiment_id}-research.zip", archive, metadata
    )
    report = {
        "experiment_id": experiment_id,
        "created_at": datetime.now(UTC),
        "status": "COMPLETED",
        "pdf_artifact_id": pdf_id,
        "archive_artifact_id": archive_id,
    }
    result = await database.reports.insert_one(report)
    return {"id": str(result.inserted_id), **report}


@router.get("/reports/{artifact_type}/{artifact_id}")
async def download_report(artifact_type: str, artifact_id: str, _: User) -> Response:
    if artifact_type not in {"pdf", "archive"}:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="REPORT_NOT_FOUND")
    bucket = "reports" if artifact_type == "pdf" else "archives"
    media_type = "application/pdf" if artifact_type == "pdf" else "application/zip"
    extension = "pdf" if artifact_type == "pdf" else "zip"
    try:
        content = await ArtifactStore(get_database(), bucket).read(artifact_id)
    except Exception as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="REPORT_NOT_FOUND") from exc
    return Response(
        content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="hsg-mt-report.{extension}"'},
    )


COLLECTION_ENDPOINTS = {
    "positions": "positions",
    "portfolio-equity": "portfolios",
    "scanner": "scanner_snapshots",
    "predictions": "predictions",
    "orders": "orders",
    "trades": "trades",
    "events": "agent_events",
    "metrics/financial": "model_metrics",
    "metrics/model": "model_metrics",
    "data-health": "data_quality_events",
    "reports": "reports",
}


@router.get("/experiments/{experiment_id}/{resource:path}")
async def experiment_resource(
    experiment_id: str,
    resource: str,
    _: User,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=250),
) -> dict[str, Any]:
    collection_name = COLLECTION_ENDPOINTS.get(resource)
    if not collection_name:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="RESOURCE_NOT_FOUND")
    if not await check_connection():
        return {"items": [], "page": page, "page_size": page_size, "total": 0}
    collection = get_database()[collection_name]
    query: dict[str, Any] = {"experiment_id": experiment_id}
    total = await collection.count_documents(query)
    cursor = (
        collection.find(query).sort("timestamp", -1).skip((page - 1) * page_size).limit(page_size)
    )
    return {
        "items": [serialize(item) async for item in cursor],
        "page": page,
        "page_size": page_size,
        "total": total,
    }


@router.websocket("/ws/experiments/{experiment_id}")
async def experiment_stream(websocket: WebSocket, experiment_id: str) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(
                {
                    "type": "heartbeat",
                    "experiment_id": experiment_id,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
            await asyncio.sleep(15)
    except WebSocketDisconnect:
        return
