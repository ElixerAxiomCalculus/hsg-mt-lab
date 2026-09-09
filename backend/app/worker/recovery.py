from typing import Any

ACTIVE_STATES = ["INITIALIZING", "WAITING_FOR_MARKET", "RUNNING", "PAUSED", "COMPLETING"]


async def recoverable_experiments(database: Any) -> list[dict[str, Any]]:
    cursor = database.experiments.find({"state": {"$in": ACTIVE_STATES}}).sort("updated_at", 1)
    return [document async for document in cursor]
