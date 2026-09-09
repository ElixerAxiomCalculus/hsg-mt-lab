import pytest
from pydantic import ValidationError

from app.schemas.experiment import Exchange, ExperimentCreate


def test_indian_experiment_defaults_to_inr() -> None:
    experiment = ExperimentCreate(name="NSE study", initial_capital="100000")
    assert experiment.exchange is Exchange.NSE
    assert experiment.base_currency == "INR"


def test_nasdaq_experiment_uses_usd() -> None:
    experiment = ExperimentCreate(
        name="NASDAQ study",
        initial_capital="100000",
        exchange=Exchange.NASDAQ,
        base_currency="USD",
    )
    assert experiment.base_currency == "USD"


def test_market_currency_mismatch_is_rejected() -> None:
    with pytest.raises(ValidationError, match="NASDAQ experiments require USD"):
        ExperimentCreate(
            name="Invalid NASDAQ study",
            initial_capital="100000",
            exchange=Exchange.NASDAQ,
            base_currency="INR",
        )
