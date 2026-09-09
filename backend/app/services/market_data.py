import asyncio
import random
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from app.schemas.trading import Quote


def quote_is_stale(quote: Quote, now: datetime, threshold_seconds: int) -> bool:
    if now.tzinfo is None or quote.provider_timestamp.tzinfo is None:
        raise ValueError("Naive datetimes are prohibited")
    return (
        now.astimezone(UTC) - quote.provider_timestamp.astimezone(UTC)
    ).total_seconds() > threshold_seconds


class MarketDataService:
    """Streaming boundary. The worker persists normalized bars and explicit fallback events."""

    def __init__(self, stale_seconds: int = 60) -> None:
        self.stale_seconds = stale_seconds
        self._stop = asyncio.Event()

    async def stream(self, symbols: list[str]) -> AsyncIterator[dict[str, Any]]:
        attempt = 0
        while not self._stop.is_set():
            try:
                import yfinance as yf

                async with yf.AsyncWebSocket() as websocket:
                    await websocket.subscribe(symbols)
                    attempt = 0
                    async for message in websocket.listen():
                        yield {"type": "quote", "payload": message}
                        if self._stop.is_set():
                            break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempt += 1
                yield {
                    "type": "data_quality_event",
                    "code": "DATA_PROVIDER_ERROR",
                    "details": {"provider": "yfinance", "error": str(exc), "attempt": attempt},
                    "timestamp": datetime.now(UTC),
                }
                delay = min(60.0, 2 ** min(attempt, 5)) + random.uniform(0, 1)
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                except TimeoutError:
                    pass

    def stop(self) -> None:
        self._stop.set()
