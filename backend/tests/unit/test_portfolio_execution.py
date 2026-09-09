from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from app.schemas.trading import Action, Quote
from app.services.execution import ExecutionService, order_idempotency_key
from app.services.portfolio import Portfolio
from app.services.risk import RiskConfig, RiskService

NOW = datetime(2026, 9, 9, 4, 0, tzinfo=UTC)


def test_buy_and_weighted_average_and_sell() -> None:
    portfolio = Portfolio.create(Decimal("10000"))
    portfolio.buy("INFY.NS", 10, Decimal("100"), Decimal("1"), NOW)
    portfolio.buy("INFY.NS", 10, Decimal("120"), Decimal("1.2"), NOW)
    assert portfolio.positions["INFY.NS"].average_entry_price == Decimal("110.00")
    assert portfolio.cash == Decimal("7797.80")
    portfolio.sell("INFY.NS", 5, Decimal("130"), Decimal("0.65"), NOW)
    assert portfolio.positions["INFY.NS"].quantity == 15
    assert portfolio.realized_pnl == Decimal("99.35")


def test_no_negative_cash_or_position() -> None:
    portfolio = Portfolio.create(Decimal("100"))
    with pytest.raises(ValueError, match="INSUFFICIENT_CAPITAL"):
        portfolio.buy("TCS.NS", 1, Decimal("101"), Decimal("0"), NOW)
    with pytest.raises(ValueError, match="INSUFFICIENT_POSITION"):
        portfolio.sell("TCS.NS", 1, Decimal("10"), Decimal("0"), NOW)


def test_execution_uses_first_quote_after_decision() -> None:
    before = Quote(
        symbol="INFY.NS",
        exchange="NSE",
        provider_timestamp=NOW - timedelta(seconds=1),
        ingestion_timestamp=NOW,
        last_price=Decimal("90"),
    )
    first = Quote(
        symbol="INFY.NS",
        exchange="NSE",
        provider_timestamp=NOW + timedelta(seconds=1),
        ingestion_timestamp=NOW,
        last_price=Decimal("100"),
    )
    future = Quote(
        symbol="INFY.NS",
        exchange="NSE",
        provider_timestamp=NOW + timedelta(seconds=20),
        ingestion_timestamp=NOW,
        last_price=Decimal("80"),
    )
    fill = ExecutionService(Decimal("10"), Decimal("10")).fill_next_quote(
        "INFY.NS", Action.BUY, 2, NOW, [future, before, first]
    )
    assert fill.reference_price == Decimal("100")
    assert fill.execution_price == Decimal("100.10")
    assert fill.transaction_cost == Decimal("0.20")


def test_risk_rejects_stale_and_fractional_capital() -> None:
    portfolio = Portfolio.create(Decimal("100"))
    decision = RiskService().evaluate(
        Action.BUY, "TCS.NS", Decimal(".9"), Decimal("500"), portfolio, False, RiskConfig()
    )
    assert decision.violated_rule == "INSUFFICIENT_CAPITAL_FOR_ONE_SHARE"
    stale = RiskService().evaluate(
        Action.BUY, "TCS.NS", Decimal(".9"), Decimal("10"), portfolio, True, RiskConfig()
    )
    assert stale.violated_rule == "DATA_STALE"


def test_order_idempotency_is_stable() -> None:
    assert order_idempotency_key("e", "INFY.NS", "p", Action.BUY) == order_idempotency_key(
        "e", "INFY.NS", "p", Action.BUY
    )
