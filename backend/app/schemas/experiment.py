from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExperimentState(StrEnum):
    DRAFT = "DRAFT"
    INITIALIZING = "INITIALIZING"
    WAITING_FOR_MARKET = "WAITING_FOR_MARKET"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETING = "COMPLETING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


ALLOWED_TRANSITIONS: dict[ExperimentState, set[ExperimentState]] = {
    ExperimentState.DRAFT: {ExperimentState.INITIALIZING, ExperimentState.CANCELLED},
    ExperimentState.INITIALIZING: {
        ExperimentState.WAITING_FOR_MARKET,
        ExperimentState.RUNNING,
        ExperimentState.FAILED,
        ExperimentState.CANCELLED,
    },
    ExperimentState.WAITING_FOR_MARKET: {
        ExperimentState.RUNNING,
        ExperimentState.PAUSED,
        ExperimentState.CANCELLED,
        ExperimentState.FAILED,
    },
    ExperimentState.RUNNING: {
        ExperimentState.PAUSED,
        ExperimentState.COMPLETING,
        ExperimentState.CANCELLED,
        ExperimentState.FAILED,
    },
    ExperimentState.PAUSED: {
        ExperimentState.RUNNING,
        ExperimentState.WAITING_FOR_MARKET,
        ExperimentState.CANCELLED,
    },
    ExperimentState.COMPLETING: {ExperimentState.COMPLETED, ExperimentState.FAILED},
    ExperimentState.COMPLETED: set(),
    ExperimentState.FAILED: set(),
    ExperimentState.CANCELLED: set(),
}


def validate_transition(current: ExperimentState, target: ExperimentState) -> None:
    if target not in ALLOWED_TRANSITIONS[current]:
        raise ValueError(f"INVALID_STATE_TRANSITION: {current} -> {target}")


class DurationUnit(StrEnum):
    CALENDAR_HOURS = "CALENDAR_HOURS"
    CALENDAR_DAYS = "CALENDAR_DAYS"
    MARKET_SESSIONS = "MARKET_SESSIONS"


class EndPolicy(StrEnum):
    MARK_TO_MARKET = "MARK_TO_MARKET"
    LIQUIDATE_AT_NEXT_AVAILABLE_MARKET = "LIQUIDATE_AT_NEXT_AVAILABLE_MARKET"


class Exchange(StrEnum):
    NSE = "NSE"
    BSE = "BSE"
    NASDAQ = "NASDAQ"


class ExperimentCreate(BaseModel):
    name: str = Field(min_length=3, max_length=120)
    initial_capital: Decimal = Field(gt=0)
    exchange: Exchange = Exchange.NSE
    base_currency: str = ""
    duration: int = Field(default=2, ge=1)
    duration_unit: DurationUnit = DurationUnit.CALENDAR_DAYS
    universe: list[str] = Field(default_factory=list)
    max_positions: int = Field(default=5, ge=1)
    max_position_pct: Decimal = Field(default=Decimal("0.20"), gt=0, le=1)
    minimum_cash_reserve_pct: Decimal = Field(default=Decimal("0.05"), ge=0, lt=1)
    scanner_frequency_seconds: int = Field(default=300, ge=30)
    confidence_threshold: Decimal = Field(default=Decimal("0.65"), ge=0, le=1)
    transaction_cost_bps: Decimal = Field(default=Decimal("10"), ge=0)
    slippage_bps: Decimal = Field(default=Decimal("5"), ge=0)
    risk_guardrails: dict[str, Any] = Field(default_factory=dict)
    end_policy: EndPolicy = EndPolicy.MARK_TO_MARKET

    @model_validator(mode="after")
    def currency_matches_exchange(self) -> "ExperimentCreate":
        expected = "USD" if self.exchange is Exchange.NASDAQ else "INR"
        if self.base_currency and self.base_currency.upper() != expected:
            raise ValueError(f"{self.exchange.value} experiments require {expected}")
        self.base_currency = expected
        return self


class ExperimentRecord(ExperimentCreate):
    model_config = ConfigDict(extra="allow")
    id: str
    state: ExperimentState
    created_at: datetime
    started_at: datetime | None = None
    expected_end_at: datetime | None = None
    ended_at: datetime | None = None
    active_market_minutes: int = 0
    sessions_encountered: int = 0
    error_code: str | None = None
