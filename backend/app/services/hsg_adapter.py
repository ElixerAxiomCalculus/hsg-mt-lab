import asyncio
import hashlib
import importlib.util
import inspect
import math
import sys
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import ModuleType
from typing import Any

from app.schemas.trading import Prediction


class AlgorithmStatus(StrEnum):
    NOT_INSTALLED = "NOT_INSTALLED"
    CONTRACT_INVALID = "CONTRACT_INVALID"
    INSTALLED = "INSTALLED"


class AlgorithmError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ContractStatus:
    status: AlgorithmStatus
    algorithm_sha256: str
    capabilities: dict[str, bool]
    errors: list[str]


class HSGAdapter:
    _capabilities = (
        "pretrain",
        "load_checkpoint",
        "save_checkpoint",
        "predict",
        "online_update",
        "get_metrics",
        "get_trajectory_state",
    )

    def __init__(self, algorithm_path: Path | None = None) -> None:
        self.algorithm_path = algorithm_path or Path(__file__).resolve().parents[2] / "algo.py"
        self._module: ModuleType | None = None
        self._signature: tuple[int, int] | None = None

    def _content(self) -> bytes:
        return self.algorithm_path.read_bytes() if self.algorithm_path.exists() else b""

    def algorithm_sha256(self) -> str:
        return hashlib.sha256(self._content()).hexdigest()

    def _load(self) -> ModuleType | None:
        content = self._content()
        if not content.strip():
            self._module = None
            self._signature = None
            return None
        stat = self.algorithm_path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if self._module is not None and signature == self._signature:
            return self._module
        module_name = "hsg_mt_user_algorithm"
        spec = importlib.util.spec_from_file_location(module_name, self.algorithm_path)
        if spec is None or spec.loader is None:
            raise AlgorithmError("ALGORITHM_CONTRACT_INVALID", "Cannot load algo.py")
        module = importlib.util.module_from_spec(spec)
        previous_module = sys.modules.get(module_name)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module
            raise AlgorithmError("ALGORITHM_CONTRACT_INVALID", str(exc)) from exc
        self._module = module
        self._signature = signature
        return module

    def validate_contract(self) -> ContractStatus:
        try:
            module = self._load()
        except AlgorithmError as exc:
            return ContractStatus(
                AlgorithmStatus.CONTRACT_INVALID,
                self.algorithm_sha256(),
                {name: False for name in self._capabilities},
                [str(exc)],
            )
        if module is None:
            return ContractStatus(
                AlgorithmStatus.NOT_INSTALLED,
                self.algorithm_sha256(),
                {name: False for name in self._capabilities},
                [],
            )
        capabilities = {name: callable(getattr(module, name, None)) for name in self._capabilities}
        missing = [name for name in ("pretrain", "predict") if not capabilities[name]]
        return ContractStatus(
            AlgorithmStatus.CONTRACT_INVALID if missing else AlgorithmStatus.INSTALLED,
            self.algorithm_sha256(),
            capabilities,
            [f"Missing required callable: {name}" for name in missing],
        )

    def status(self) -> dict[str, Any]:
        contract = self.validate_contract()
        return {
            "status": contract.status.value,
            "algorithm_sha256": contract.algorithm_sha256,
            "capabilities": contract.capabilities,
            "errors": contract.errors,
        }

    async def _call(self, name: str, **kwargs: Any) -> Any:
        contract = self.validate_contract()
        if contract.status is AlgorithmStatus.NOT_INSTALLED:
            raise AlgorithmError("ALGORITHM_NOT_INSTALLED", "HSG-MT algorithm is not installed")
        if contract.status is AlgorithmStatus.CONTRACT_INVALID:
            raise AlgorithmError("ALGORITHM_CONTRACT_INVALID", "; ".join(contract.errors))
        module = self._load()
        function: Callable[..., Any] | None = getattr(module, name, None) if module else None
        if function is None:
            code = (
                "ONLINE_LEARNING_UNSUPPORTED"
                if name == "online_update"
                else "CAPABILITY_UNSUPPORTED"
            )
            raise AlgorithmError(code, f"Algorithm capability '{name}' is unavailable")
        if inspect.iscoroutinefunction(function):
            result = await function(**kwargs)
        else:
            result = await asyncio.to_thread(function, **kwargs)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def pretrain(self, dataset: Any, configuration: dict[str, Any]) -> dict[str, Any]:
        result = await self._call("pretrain", dataset=dataset, configuration=configuration)
        if not isinstance(result, dict) or not result.get("model_version"):
            raise AlgorithmError("ALGORITHM_CONTRACT_INVALID", "pretrain must return model_version")
        return result

    async def predict(self, **kwargs: Any) -> Prediction:
        result = await self._call("predict", **kwargs)
        if not isinstance(result, dict):
            raise AlgorithmError("ALGORITHM_CONTRACT_INVALID", "predict must return a mapping")
        try:
            prediction = Prediction.model_validate(result)
        except Exception as exc:
            raise AlgorithmError(
                "ALGORITHM_CONTRACT_INVALID", f"Invalid prediction: {exc}"
            ) from exc
        values = [prediction.confidence, prediction.expected_return]
        values.extend(prediction.regime_probabilities.values())
        if any(not math.isfinite(value) for value in values):
            raise AlgorithmError(
                "ALGORITHM_CONTRACT_INVALID", "Prediction contains non-finite values"
            )
        if prediction.timestamp.tzinfo is None:
            raise AlgorithmError(
                "ALGORITHM_CONTRACT_INVALID", "Prediction timestamp must be timezone-aware"
            )
        return prediction

    async def load_checkpoint(self, checkpoint: bytes, metadata: dict[str, Any]) -> Any:
        return await self._call("load_checkpoint", checkpoint=checkpoint, metadata=metadata)

    async def save_checkpoint(self, context: dict[str, Any]) -> Any:
        return await self._call("save_checkpoint", context=context)

    async def online_update(self, observation: dict[str, Any], context: dict[str, Any]) -> Any:
        return await self._call("online_update", observation=observation, context=context)

    async def get_metrics(self, context: dict[str, Any]) -> dict[str, Any]:
        result = await self._call("get_metrics", context=context)
        if not isinstance(result, dict):
            raise AlgorithmError("ALGORITHM_CONTRACT_INVALID", "get_metrics must return a mapping")
        return result

    async def get_trajectory_state(self, context: dict[str, Any]) -> dict[str, Any]:
        result = await self._call("get_trajectory_state", context=context)
        if not isinstance(result, dict):
            raise AlgorithmError(
                "ALGORITHM_CONTRACT_INVALID", "get_trajectory_state must return a mapping"
            )
        return result
