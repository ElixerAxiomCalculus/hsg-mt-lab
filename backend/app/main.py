from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import adapter, router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.client import check_connection, close_mongo_client, get_database
from app.db.indexes import ensure_indexes

settings = get_settings()
configure_logging(settings.log_level)


@asynccontextmanager
async def lifespan(_: FastAPI) -> Any:
    if await check_connection():
        await ensure_indexes(get_database())
    yield
    await close_mongo_client()


app = FastAPI(
    title="HSG-MT Lab API",
    version=settings.app_version,
    description="Autonomous paper-trading research platform for Indian equities.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)
app.include_router(router)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {"errors": exc.errors()},
                "correlation_id": str(uuid4()),
            }
        },
    )


@app.exception_handler(Exception)
async def unhandled_error(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": str(exc) if settings.app_env == "development" else "Unexpected error",
                "details": {},
                "correlation_id": str(uuid4()),
            }
        },
    )


@app.get("/health/live")
async def live() -> dict[str, str]:
    return {"status": "live", "service": "hsg-mt-api"}


@app.get("/health/ready")
async def ready() -> JSONResponse:
    mongo = await check_connection()
    status_code = 200 if mongo else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if mongo else "degraded",
            "mongodb": "connected" if mongo else "unavailable",
            "indexes": "configured" if mongo else "unknown",
            "algorithm": adapter.status(),
        },
    )
