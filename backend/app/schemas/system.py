from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AlgorithmState(StrEnum):
    NOT_INSTALLED = "NOT_INSTALLED"
    CONTRACT_INVALID = "CONTRACT_INVALID"
    INSTALLED = "INSTALLED"


class AlgorithmStatus(BaseModel):
    state: AlgorithmState
    algorithm_sha256: str
    file_size: int
    capabilities: dict[str, bool]
    errors: list[str] = Field(default_factory=list)


class HealthStatus(BaseModel):
    status: str
    timestamp: datetime
    version: str
    checks: dict[str, Any] = Field(default_factory=dict)
