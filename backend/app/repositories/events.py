from datetime import UTC, datetime
from typing import Any
from uuid import uuid4


class EventRepository:
    def __init__(self, database: Any) -> None:
        self.collection = database.agent_events

    async def emit(
        self,
        experiment_id: str,
        component: str,
        event_type: str,
        *,
        severity: str = "INFO",
        details: dict[str, Any] | None = None,
        symbol: str | None = None,
        model_version: str | None = None,
        correlation_id: str | None = None,
        latency_ms: float | None = None,
    ) -> str:
        event_id = str(uuid4())
        await self.collection.insert_one(
            {
                "_id": event_id,
                "event_id": event_id,
                "experiment_id": experiment_id,
                "timestamp": datetime.now(UTC),
                "component": component,
                "event_type": event_type,
                "severity": severity,
                "symbol": symbol,
                "model_version": model_version,
                "correlation_id": correlation_id,
                "input_summary": None,
                "output_summary": None,
                "structured_details": details or {},
                "latency_ms": latency_ms,
            }
        )
        return event_id
