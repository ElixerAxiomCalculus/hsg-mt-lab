from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Action(StrEnum):
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


class OrderState(StrEnum):
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    REJECTED = "REJECTED"
    WAITING_FOR_QUOTE = "WAITING_FOR_QUOTE"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class Quote(BaseModel):
    symbol: str
    exchange: str
    provider: str = "yfinance"
    provider_timestamp: datetime
    ingestion_timestamp: datetime
    last_price: Decimal = Field(gt=0)
    sequence: int | None = None
    source_type: str = "stream"


class Prediction(BaseModel):
    symbol: str
    timestamp: datetime
    action: Action
    confidence: float = Field(ge=0, le=1)
    expected_return: float
    prediction_horizon: str
    regime: str
    regime_probabilities: dict[str, float]
    geometric_state: dict[str, Any]
    trajectory_metrics: dict[str, Any]
    model_version: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RiskDecision(BaseModel):
    approved: bool
    proposed_action: Action
    proposed_quantity: int
    final_action: Action
    violated_rule: str | None = None
    override_reason: str | None = None


class Position(BaseModel):
    symbol: str
    quantity: int = Field(ge=0)
    average_entry_price: Decimal = Field(ge=0)
    current_market_price: Decimal = Field(ge=0)
    realized_pnl: Decimal = Decimal("0")
    first_entry_timestamp: datetime
    last_update_timestamp: datetime

    @property
    def market_value(self) -> Decimal:
        return self.current_market_price * self.quantity

    @property
    def unrealized_pnl(self) -> Decimal:
        return (self.current_market_price - self.average_entry_price) * self.quantity
