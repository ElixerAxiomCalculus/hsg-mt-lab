import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from app.research.dataset_loader import fingerprint, load_validated_csv
from app.research.walk_forward import chronological_split
from app.services.hsg_adapter import AlgorithmError, HSGAdapter


class PretrainingOrchestrator:
    def __init__(self, adapter: HSGAdapter) -> None:
        self.adapter = adapter

    async def run(
        self,
        paths: list[Path],
        configuration: dict[str, Any],
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        run_id = str(uuid4())
        started_at = datetime.now(UTC)
        try:
            if progress_callback:
                progress_callback({"phase": "LOADING_DATA", "file_count": len(paths)})
            frames = await asyncio.to_thread(lambda: [load_validated_csv(path) for path in paths])
            stock_frames = [frame for frame in frames if "symbol" in frame and "close" in frame]
            if not stock_frames:
                raise ValueError("No stock OHLCV dataset found")
            if progress_callback:
                progress_callback({"phase": "PREPARING_SPLITS"})

            def prepare() -> tuple[pd.DataFrame, Any]:
                combined = pd.concat(stock_frames, ignore_index=True).sort_values(
                    "date", kind="stable"
                )
                return combined, chronological_split(combined)

            combined, split = await asyncio.to_thread(prepare)
            training_configuration = dict(configuration)
            if progress_callback:
                training_configuration["_progress_callback"] = progress_callback
            result = await self.adapter.pretrain(
                dataset={"train": split.train, "validation": split.validation, "test": split.test},
                configuration=training_configuration,
            )
            if progress_callback:
                progress_callback({"phase": "SAVING_CHECKPOINT"})
            return {
                "id": run_id,
                "status": "COMPLETED",
                "started_at": started_at,
                "completed_at": datetime.now(UTC),
                "dataset_hashes": {path.name: fingerprint(path) for path in paths},
                "row_count": len(combined),
                "algorithm_sha256": self.adapter.algorithm_sha256(),
                "result": result,
            }
        except AlgorithmError as exc:
            return {
                "id": run_id,
                "status": "ALGORITHM_NOT_INSTALLED"
                if exc.code == "ALGORITHM_NOT_INSTALLED"
                else "FAILED",
                "error_code": exc.code,
                "message": str(exc),
                "started_at": started_at,
                "completed_at": datetime.now(UTC),
            }
