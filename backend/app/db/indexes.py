from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel

COLLECTIONS = (
    "users",
    "experiments",
    "experiment_leases",
    "portfolios",
    "positions",
    "orders",
    "trades",
    "predictions",
    "signals",
    "scanner_snapshots",
    "market_bars",
    "market_quotes",
    "model_runs",
    "model_checkpoints",
    "model_metrics",
    "training_runs",
    "online_updates",
    "agent_events",
    "data_quality_events",
    "dataset_registry",
    "market_holidays",
    "trading_universe",
    "reports",
    "system_state",
)


INDEXES: dict[str, list[IndexModel]] = {
    "users": [IndexModel("username", unique=True)],
    "experiments": [IndexModel([("state", ASCENDING), ("created_at", DESCENDING)])],
    "experiment_leases": [IndexModel("experiment_id", unique=True), IndexModel("expires_at")],
    "orders": [
        IndexModel("idempotency_key", unique=True),
        IndexModel([("experiment_id", 1), ("created_at", -1)]),
    ],
    "trades": [
        IndexModel("order_id", unique=True),
        IndexModel([("experiment_id", 1), ("fill_timestamp", -1)]),
    ],
    "portfolios": [IndexModel([("experiment_id", 1), ("timestamp", -1)])],
    "positions": [IndexModel([("experiment_id", 1), ("symbol", 1)], unique=True)],
    "predictions": [
        IndexModel([("experiment_id", 1), ("timestamp", -1)]),
        IndexModel("model_version"),
    ],
    "scanner_snapshots": [IndexModel([("experiment_id", 1), ("timestamp", -1)])],
    "market_bars": [IndexModel([("symbol", 1), ("timestamp", 1)], unique=True)],
    "market_quotes": [IndexModel([("symbol", 1), ("provider_timestamp", -1)])],
    "model_checkpoints": [
        IndexModel([("active", ASCENDING), ("created_at", DESCENDING)]),
        IndexModel("model_version", unique=True),
    ],
    "agent_events": [IndexModel([("experiment_id", 1), ("timestamp", -1)])],
    "data_quality_events": [IndexModel([("experiment_id", 1), ("timestamp", -1)])],
    "dataset_registry": [IndexModel("sha256", unique=True)],
    "market_holidays": [IndexModel("date", unique=True)],
    "reports": [IndexModel([("experiment_id", 1), ("created_at", -1)])],
}


async def ensure_indexes(database: Any) -> None:
    for name in COLLECTIONS:
        collection = database[name]
        models = INDEXES.get(name)
        if models:
            await collection.create_indexes(models)
