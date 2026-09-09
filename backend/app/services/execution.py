from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from app.schemas.trading import Action, Quote
from app.services.portfolio import money


@dataclass(frozen=True)
class Fill:
    symbol: str
    action: Action
    quantity: int
    decision_timestamp: datetime
    fill_timestamp: datetime
    reference_price: Decimal
    slippage: Decimal
    execution_price: Decimal
    gross_value: Decimal
    transaction_cost: Decimal


class ExecutionService:
    def __init__(self, slippage_bps: Decimal, transaction_cost_bps: Decimal) -> None:
        self.slippage_bps = slippage_bps
        self.transaction_cost_bps = transaction_cost_bps

    def fill_next_quote(
        self,
        symbol: str,
        action: Action,
        quantity: int,
        decision_timestamp: datetime,
        quotes: list[Quote],
    ) -> Fill:
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        eligible = sorted(
            (
                quote
                for quote in quotes
                if quote.symbol == symbol and quote.provider_timestamp > decision_timestamp
            ),
            key=lambda quote: quote.provider_timestamp,
        )
        if not eligible:
            raise ValueError("WAITING_FOR_QUOTE")
        quote = eligible[0]
        direction = Decimal("1") if action is Action.BUY else Decimal("-1")
        slippage = money(quote.last_price * self.slippage_bps / Decimal("10000"))
        execution_price = money(quote.last_price + direction * slippage)
        gross = money(execution_price * quantity)
        cost = money(gross * self.transaction_cost_bps / Decimal("10000"))
        return Fill(
            symbol,
            action,
            quantity,
            decision_timestamp,
            quote.provider_timestamp,
            quote.last_price,
            slippage,
            execution_price,
            gross,
            cost,
        )


def order_idempotency_key(
    experiment_id: str, symbol: str, prediction_id: str, action: Action
) -> str:
    return f"{experiment_id}:{symbol}:{prediction_id}:{action.value}"
