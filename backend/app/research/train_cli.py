import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.config import get_settings
from app.db.artifacts import ArtifactStore
from app.db.client import check_connection, close_mongo_client, get_database
from app.db.indexes import ensure_indexes
from app.research.dataset_loader import inspect_directory
from app.research.pretraining import PretrainingOrchestrator
from app.services.hsg_adapter import AlgorithmStatus, HSGAdapter


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain HSG-MT locally and publish the checkpoint to MongoDB Atlas."
    )
    parser.add_argument(
        "filenames",
        nargs="*",
        help="CSV filenames from data/. Defaults to every detected stock OHLCV file.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--markets",
        nargs="+",
        choices=("NSE", "BSE", "NASDAQ"),
        default=["NSE", "BSE"],
        help="Markets represented by the training data (default: NSE BSE).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON file whose values override the command-line defaults.",
    )
    return parser


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    configuration: dict[str, Any] = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "early_stopping_patience": args.patience,
        "device": args.device,
        "split": [0.7, 0.15, 0.15],
        "walk_forward": True,
    }
    if args.config:
        loaded = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("--config must contain a JSON object")
        configuration.update(loaded)
    return configuration


def _show_progress(event: dict[str, Any]) -> None:
    phase = str(event.get("phase", ""))
    if phase == "LOADING_DATA":
        print(f"Loading {event['file_count']} dataset file(s)...", flush=True)
    elif phase == "PREPARING_SPLITS":
        print("Preparing chronological train/validation/test splits...", flush=True)
    elif phase == "TRAINING":
        epoch = int(event["epoch"])
        total = int(event["total_epochs"])
        train_loss = float(event["train_loss"])
        validation_loss = float(event["validation_loss"])
        stale = int(event["stale_epochs"])
        patience = int(event["early_stopping_patience"])
        print(
            f"Epoch {epoch:>3}/{total} | train_loss={train_loss:.6f} "
            f"| validation_loss={validation_loss:.6f} | patience={stale}/{patience}",
            flush=True,
        )
    elif phase == "SAVING_CHECKPOINT":
        print("Training complete. Uploading checkpoint to Atlas GridFS...", flush=True)


async def _paths(filenames: list[str]) -> list[Path]:
    data_directory = get_settings().data_directory
    available = {path.name: path for path in data_directory.glob("*.csv")}
    if filenames:
        missing = sorted(set(filenames) - set(available))
        if missing:
            raise FileNotFoundError(f"Dataset file(s) not found: {', '.join(missing)}")
        return [available[name] for name in filenames]
    inspections = await asyncio.to_thread(inspect_directory, data_directory)
    selected = [
        available[item.filename]
        for item in inspections
        if item.detected_type == "STOCK_OHLCV"
    ]
    if not selected:
        raise FileNotFoundError("No STOCK_OHLCV dataset was detected in data/")
    return selected


async def _run(args: argparse.Namespace) -> int:
    if not await check_connection():
        print(
            "MongoDB is unavailable; training was not started because the checkpoint "
            "cannot be saved."
        )
        return 1
    database = get_database()
    await ensure_indexes(database)
    adapter = HSGAdapter()
    contract = adapter.validate_contract()
    if contract.status is not AlgorithmStatus.INSTALLED:
        print(f"Algorithm is not ready: {contract.status.value}")
        for error in contract.errors:
            print(f"  {error}")
        return 1

    paths = await _paths(list(args.filenames))
    configuration = _configuration(args)
    print("HSG-MT offline pretraining")
    print(f"Datasets: {', '.join(path.name for path in paths)}")
    print(
        f"Epochs: {configuration['epochs']} | batch size: {configuration['batch_size']} "
        f"| device: {configuration['device']}"
    )
    result = await PretrainingOrchestrator(adapter).run(
        paths, configuration, progress_callback=_show_progress
    )
    if result["status"] != "COMPLETED":
        print(
            f"Training failed: {result.get('error_code', result['status'])} - "
            f"{result.get('message', '')}"
        )
        return 1

    algorithm_result = dict(result["result"])
    checkpoint = algorithm_result.pop("checkpoint", None)
    if not isinstance(checkpoint, bytes) or not checkpoint:
        print("Training failed: the algorithm did not return checkpoint bytes.")
        return 1
    model_version = str(algorithm_result["model_version"])
    checkpoint_metadata = dict(
        algorithm_result.get("metadata", {}).get("checkpoint_metadata", {})
    )
    model_id = str(uuid4())
    created_at = datetime.now(UTC)
    artifact_id = await ArtifactStore(database, "model_checkpoints").put(
        f"{model_version}.pt",
        checkpoint,
        {
            "model_id": model_id,
            "model_version": model_version,
            "algorithm_sha256": result["algorithm_sha256"],
            "created_at": created_at,
        },
    )
    model_document = {
        "_id": model_id,
        "active": True,
        "activated_at": created_at,
        "artifact_id": artifact_id,
        "model_version": model_version,
        "algorithm_sha256": result["algorithm_sha256"],
        "checkpoint_metadata": checkpoint_metadata,
        "metrics": algorithm_result.get("metrics", {}),
        "configuration": configuration,
        "dataset_hashes": result["dataset_hashes"],
        "trained_exchanges": sorted(set(args.markets)),
        "row_count": result["row_count"],
        "created_at": created_at,
        "source": "OFFLINE_CLI",
    }
    await database.model_checkpoints.insert_one(model_document)
    await database.model_checkpoints.update_many(
        {"active": True, "_id": {"$ne": model_id}},
        {"$set": {"active": False, "deactivated_at": created_at}},
    )
    audit_result = {**algorithm_result, "checkpoint_artifact_id": artifact_id}
    await database.training_runs.insert_one(
        {
            "_id": result["id"],
            **{key: value for key, value in result.items() if key not in {"id", "result"}},
            "configuration": configuration,
            "trained_exchanges": sorted(set(args.markets)),
            "result": audit_result,
            "source": "OFFLINE_CLI",
        }
    )
    print(f"Active model: {model_version}")
    print(f"Checkpoint artifact: {artifact_id}")
    print("The worker will load this model automatically on its next cycle.")
    return 0


async def _entry(args: argparse.Namespace) -> int:
    try:
        return await _run(args)
    finally:
        await close_mongo_client()


def main() -> None:
    args = _parser().parse_args()
    try:
        raise SystemExit(asyncio.run(_entry(args)))
    except KeyboardInterrupt:
        print("\nTraining interrupted; no model was activated.")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
