import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class MarketSession:
    session_date: date
    opens_at: datetime
    closes_at: datetime
    name: str = "REGULAR"


class MarketCalendarService:
    def __init__(self, config_path: Path, timezone: str = "Asia/Kolkata") -> None:
        payload = (
            json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        )
        self.timezone = ZoneInfo(str(payload.get("timezone", timezone)))
        self.default_open = time.fromisoformat(str(payload.get("open", "09:15")))
        self.default_close = time.fromisoformat(str(payload.get("close", "15:30")))
        self.holidays = {date.fromisoformat(value) for value in payload.get("holidays", [])}
        self.overrides = {item["date"]: item for item in payload.get("session_overrides", [])}

    def _local(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("Naive datetimes are prohibited")
        return value.astimezone(self.timezone)

    def is_trading_day(self, value: date) -> bool:
        override = self.overrides.get(value.isoformat())
        if override:
            return bool(override.get("enabled", True))
        return value.weekday() < 5 and value not in self.holidays

    def current_session(self, value: datetime) -> MarketSession | None:
        local = self._local(value)
        if not self.is_trading_day(local.date()):
            return None
        override = self.overrides.get(local.date().isoformat(), {})
        open_time = time.fromisoformat(override.get("open", self.default_open.isoformat()))
        close_time = time.fromisoformat(override.get("close", self.default_close.isoformat()))
        return MarketSession(
            local.date(),
            datetime.combine(local.date(), open_time, self.timezone),
            datetime.combine(local.date(), close_time, self.timezone),
            override.get("name", "REGULAR"),
        )

    def is_market_open(self, value: datetime) -> bool:
        session = self.current_session(value)
        local = self._local(value)
        return bool(session and session.opens_at <= local < session.closes_at)

    def next_market_open(self, value: datetime) -> datetime:
        local = self._local(value)
        for offset in range(0, 370):
            candidate = local.date() + timedelta(days=offset)
            if not self.is_trading_day(candidate):
                continue
            probe = datetime.combine(candidate, time(12), self.timezone)
            session = self.current_session(probe)
            if session and session.opens_at > local:
                return session.opens_at
        raise RuntimeError("No market session found within one year")

    def previous_market_close(self, value: datetime) -> datetime:
        local = self._local(value)
        for offset in range(0, 370):
            candidate = local.date() - timedelta(days=offset)
            if not self.is_trading_day(candidate):
                continue
            probe = datetime.combine(candidate, time(12), self.timezone)
            session = self.current_session(probe)
            if session and session.closes_at < local:
                return session.closes_at
        raise RuntimeError("No previous market session found within one year")

    def minutes_until_close(self, value: datetime) -> int:
        session = self.current_session(value)
        local = self._local(value)
        if not session or local >= session.closes_at:
            return 0
        return max(0, int((session.closes_at - local).total_seconds() // 60))
