from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from app.schemas.experiment import ExperimentCreate, ExperimentState, validate_transition


class ExperimentRepository:
    def __init__(self, database: Any) -> None:
        self.collection = database.experiments

    @staticmethod
    def public(document: dict[str, Any]) -> dict[str, Any]:
        output = dict(document)
        output["id"] = str(output.pop("_id"))
        return output

    async def create(self, payload: ExperimentCreate) -> dict[str, Any]:
        now = datetime.now(UTC)
        document = {
            "_id": str(uuid4()),
            **payload.model_dump(mode="json"),
            "state": ExperimentState.DRAFT.value,
            "created_at": now,
            "updated_at": now,
            "active_market_minutes": 0,
            "sessions_encountered": 0,
        }
        await self.collection.insert_one(document)
        return self.public(document)

    async def get(self, experiment_id: str) -> dict[str, Any] | None:
        document = await self.collection.find_one({"_id": experiment_id})
        return self.public(document) if document else None

    async def list(self, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self.collection.find({}).sort("created_at", -1).limit(limit)
        return [self.public(document) async for document in cursor]

    async def transition(
        self, experiment_id: str, target: ExperimentState, **fields: Any
    ) -> dict[str, Any] | None:
        document = await self.collection.find_one({"_id": experiment_id})
        if not document:
            return None
        current = ExperimentState(document["state"])
        validate_transition(current, target)
        updates = {"state": target.value, "updated_at": datetime.now(UTC), **fields}
        result = await self.collection.find_one_and_update(
            {"_id": experiment_id, "state": current.value}, {"$set": updates}, return_document=True
        )
        return self.public(result) if result else None


class LeaseRepository:
    def __init__(self, database: Any, lease_seconds: int = 90) -> None:
        self.collection = database.experiment_leases
        self.lease_seconds = lease_seconds

    async def acquire(self, experiment_id: str, worker_id: str) -> bool:
        now = datetime.now(UTC)
        expires = now + timedelta(seconds=self.lease_seconds)
        result = await self.collection.find_one_and_update(
            {
                "experiment_id": experiment_id,
                "$or": [{"expires_at": {"$lte": now}}, {"worker_id": worker_id}],
            },
            {
                "$set": {
                    "worker_id": worker_id,
                    "acquired_at": now,
                    "heartbeat_at": now,
                    "expires_at": expires,
                }
            },
            upsert=True,
            return_document=True,
        )
        return bool(result and result["worker_id"] == worker_id)

    async def heartbeat(self, experiment_id: str, worker_id: str) -> bool:
        now = datetime.now(UTC)
        result = await self.collection.update_one(
            {"experiment_id": experiment_id, "worker_id": worker_id},
            {
                "$set": {
                    "heartbeat_at": now,
                    "expires_at": now + timedelta(seconds=self.lease_seconds),
                }
            },
        )
        return bool(result.modified_count == 1)

    async def release(self, experiment_id: str, worker_id: str) -> None:
        await self.collection.delete_one({"experiment_id": experiment_id, "worker_id": worker_id})
