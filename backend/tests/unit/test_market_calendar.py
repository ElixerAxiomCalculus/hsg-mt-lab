from datetime import UTC, datetime
from pathlib import Path

from app.services.market_calendar import MarketCalendarService


def calendar(tmp_path: Path) -> MarketCalendarService:
    path = tmp_path / "calendar.json"
    path.write_text(
        '{"holidays":["2026-01-26"],"session_overrides":[{"date":"2026-11-08","enabled":true,"open":"18:00","close":"19:00","name":"MUHURAT"}]}',
        encoding="utf-8",
    )
    return MarketCalendarService(path)


def test_regular_session(tmp_path: Path) -> None:
    service = calendar(tmp_path)
    assert service.is_market_open(datetime(2026, 9, 9, 5, 0, tzinfo=UTC))


def test_weekend_and_holiday_rejected(tmp_path: Path) -> None:
    service = calendar(tmp_path)
    assert not service.is_trading_day(datetime(2026, 9, 12).date())
    assert not service.is_trading_day(datetime(2026, 1, 26).date())


def test_special_session_override(tmp_path: Path) -> None:
    service = calendar(tmp_path)
    session = service.current_session(datetime(2026, 11, 8, 13, 0, tzinfo=UTC))
    assert session is not None and session.name == "MUHURAT"


def test_naive_datetime_rejected(tmp_path: Path) -> None:
    service = calendar(tmp_path)
    try:
        service.is_market_open(datetime(2026, 9, 9, 10))
    except ValueError as exc:
        assert "Naive" in str(exc)
    else:
        raise AssertionError("Naive datetime was accepted")


def test_nasdaq_timezone_and_session_defaults(tmp_path: Path) -> None:
    path = tmp_path / "nasdaq.json"
    path.write_text(
        '{"timezone":"America/New_York","open":"09:30","close":"16:00",'
        '"holidays":["2026-09-07"],"session_overrides":[]}',
        encoding="utf-8",
    )
    service = MarketCalendarService(path)
    assert service.is_market_open(datetime(2026, 9, 9, 13, 30, tzinfo=UTC))
    assert not service.is_trading_day(datetime(2026, 9, 7).date())
