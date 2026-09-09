from datetime import UTC, datetime, timedelta
from typing import Any


def target_ready_timestamp(prediction_timestamp: datetime, horizon_seconds: int) -> datetime:
    if prediction_timestamp.tzinfo is None:
        raise ValueError("Naive datetimes are prohibited")
    if horizon_seconds <= 0:
        raise ValueError("Prediction horizon must be positive")
    return prediction_timestamp.astimezone(UTC) + timedelta(seconds=horizon_seconds)


async def mature_pending_labels(database: Any, now: datetime) -> list[dict[str, Any]]:
    if now.tzinfo is None:
        raise ValueError("Naive datetimes are prohibited")
    cursor = database.online_updates.find(
        {"status": "PENDING", "target_ready_timestamp": {"$lte": now.astimezone(UTC)}}
    ).sort("target_ready_timestamp", 1)
    return [document async for document in cursor]
