import asyncio
import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast
from uuid import uuid4

import pandas as pd

from app.core.config import Settings
from app.db.artifacts import ArtifactStore
from app.db.client import get_database
from app.repositories.events import EventRepository
from app.repositories.experiments import ExperimentRepository, LeaseRepository
from app.schemas.experiment import Exchange, ExperimentState
from app.schemas.trading import Action, Quote
from app.services.execution import ExecutionService, order_idempotency_key
from app.services.feature_engineering import build_features
from app.services.hsg_adapter import AlgorithmStatus, HSGAdapter
from app.services.market_calendar import MarketCalendarService
from app.services.market_data import quote_is_stale
from app.services.portfolio import Holding, Portfolio
from app.services.risk import RiskConfig, RiskService
from app.services.scanner import rank_bearish_candidates
from app.worker.recovery import recoverable_experiments

logger = logging.getLogger(__name__)


class WorkerOrchestrator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.worker_id = str(uuid4())
        self.stop_event = asyncio.Event()
        self.database = get_database()
        self.leases = LeaseRepository(self.database, settings.worker_lease_seconds)
        self.experiments = ExperimentRepository(self.database)
        self.events = EventRepository(self.database)
        self.adapter = HSGAdapter()
        indian_calendar = MarketCalendarService(
            settings.configuration_directory / "market_holidays.json", settings.market_timezone
        )
        self.calendars = {
            Exchange.NSE: indian_calendar,
            Exchange.BSE: indian_calendar,
            Exchange.NASDAQ: MarketCalendarService(
                settings.configuration_directory / "nasdaq_market_holidays.json"
            ),
        }
        self._owned: set[str] = set()
        self._loaded_checkpoint_id: str | None = None
        self._loaded_exchanges: set[str] = set()

    async def heartbeat(self, status: str = "RUNNING") -> None:
        now = datetime.now(UTC)
        await self.database.system_state.update_one(
            {"_id": "worker"},
            {"$set": {"status": status, "worker_id": self.worker_id, "heartbeat": now}},
            upsert=True,
        )

    async def recover(self) -> None:
        for document in await recoverable_experiments(self.database):
            experiment_id = str(document["_id"])
            if await self.leases.acquire(experiment_id, self.worker_id):
                self._owned.add(experiment_id)
                await self.events.emit(
                    experiment_id,
                    "Worker",
                    "WORKER_RECOVERED",
                    details={"worker_id": self.worker_id, "previous_state": document["state"]},
                )

    async def _ensure_checkpoint_loaded(self) -> bool:
        checkpoint = await self.database.model_checkpoints.find_one(
            {"active": True}, sort=[("created_at", -1)]
        )
        if not checkpoint:
            return False
        if checkpoint.get("algorithm_sha256") != self.adapter.algorithm_sha256():
            logger.error(
                "model_checkpoint_incompatible",
                extra={"service": "worker", "checkpoint_id": str(checkpoint["_id"])},
            )
            return False
        checkpoint_id = str(checkpoint["_id"])
        if checkpoint_id == self._loaded_checkpoint_id:
            return True
        try:
            content = await ArtifactStore(self.database, "model_checkpoints").read(
                str(checkpoint["artifact_id"])
            )
            await self.adapter.load_checkpoint(
                content, dict(checkpoint.get("checkpoint_metadata", {}))
            )
        except Exception:
            logger.exception(
                "model_checkpoint_load_failed",
                extra={"service": "worker", "checkpoint_id": checkpoint_id},
            )
            return False
        self._loaded_checkpoint_id = checkpoint_id
        self._loaded_exchanges = set(
            checkpoint.get("trained_exchanges") or [Exchange.NSE.value, Exchange.BSE.value]
        )
        logger.info(
            "model_checkpoint_loaded",
            extra={
                "service": "worker",
                "checkpoint_id": checkpoint_id,
                "model_version": checkpoint.get("model_version"),
            },
        )
        return True

    async def _history(self, symbols: list[str]) -> pd.DataFrame:
        def download() -> pd.DataFrame:
            import yfinance as yf

            result = yf.download(
                symbols,
                period="5d",
                interval="5m",
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=False,
            )
            return cast(pd.DataFrame, result)

        return await asyncio.to_thread(download)

    @staticmethod
    def _symbol_features(history: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            try:
                source = (
                    history[symbol].reset_index() if len(symbols) > 1 else history.reset_index()
                )
                source.columns = [
                    str(column).lower().replace(" ", "_") for column in source.columns
                ]
                source = source.rename(columns={source.columns[0]: "date"})
                features = build_features(source)
                latest = features.dropna(subset=["return_5", "return_20", "rsi"]).iloc[-1]
                values = {str(key): value for key, value in latest.to_dict().items()}
                rows.append({"symbol": symbol, **values})
            except (KeyError, IndexError, ValueError):
                continue
        return pd.DataFrame(rows)

    @staticmethod
    def _aware_timestamp(value: Any) -> datetime:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        return timestamp.tz_convert("UTC").to_pydatetime()

    async def _load_portfolio(self, experiment_id: str, initial_capital: Decimal) -> Portfolio:
        snapshot = await self.database.portfolios.find_one(
            {"experiment_id": experiment_id}, sort=[("timestamp", -1)]
        )
        portfolio = Portfolio.create(initial_capital)
        if snapshot:
            portfolio.cash = Decimal(str(snapshot["cash"]))
            portfolio.transaction_costs = Decimal(str(snapshot.get("transaction_costs", "0")))
            portfolio.realized_pnl = Decimal(str(snapshot.get("realized_pnl", "0")))
        cursor = self.database.positions.find({"experiment_id": experiment_id, "active": True})
        async for item in cursor:
            portfolio.positions[item["symbol"]] = Holding(
                symbol=item["symbol"],
                quantity=int(item["quantity"]),
                average_entry_price=Decimal(str(item["average_entry_price"])),
                current_market_price=Decimal(str(item["current_market_price"])),
                first_entry_timestamp=item["first_entry_timestamp"],
                last_update_timestamp=item["last_update_timestamp"],
                realized_pnl=Decimal(str(item.get("realized_pnl", "0"))),
            )
        return portfolio

    async def _persist_portfolio(
        self, experiment_id: str, portfolio: Portfolio, timestamp: datetime
    ) -> None:
        snapshot = portfolio.snapshot()
        positions = cast(list[dict[str, object]], snapshot.pop("positions"))
        await self.database.portfolios.insert_one(
            {"experiment_id": experiment_id, "timestamp": timestamp, **snapshot}
        )
        active_symbols: list[str] = []
        for position in positions:
            symbol = str(position["symbol"])
            active_symbols.append(symbol)
            holding = portfolio.positions[symbol]
            await self.database.positions.update_one(
                {"experiment_id": experiment_id, "symbol": symbol},
                {
                    "$set": {
                        **position,
                        "active": True,
                        "first_entry_timestamp": holding.first_entry_timestamp,
                        "last_update_timestamp": holding.last_update_timestamp,
                        "realized_pnl": str(holding.realized_pnl),
                    }
                },
                upsert=True,
            )
        await self.database.positions.update_many(
            {
                "experiment_id": experiment_id,
                "symbol": {"$nin": active_symbols},
                "active": True,
            },
            {"$set": {"active": False, "closed_at": timestamp}},
        )

    async def _persist_bars(
        self, experiment_id: str, snapshot: pd.DataFrame, ingested_at: datetime
    ) -> None:
        for _, row in snapshot.iterrows():
            timestamp = self._aware_timestamp(row["date"])
            symbol = str(row["symbol"])
            bar = {
                "experiment_id": experiment_id,
                "symbol": symbol,
                "exchange": self._symbol_exchange(symbol).value,
                "timestamp": timestamp,
                "ingestion_timestamp": ingested_at,
                "provider": "yfinance",
                "source_type": "intraday_fallback",
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
            }
            await self.database.market_bars.update_one(
                {"symbol": symbol, "timestamp": timestamp}, {"$set": bar}, upsert=True
            )

    async def _execute_pending_orders(
        self,
        experiment_id: str,
        document: dict[str, Any],
        snapshot: pd.DataFrame,
        now: datetime,
    ) -> Portfolio:
        portfolio = await self._load_portfolio(
            experiment_id, Decimal(str(document["initial_capital"]))
        )
        for _, row in snapshot.iterrows():
            portfolio.mark(
                str(row["symbol"]),
                Decimal(str(row["close"])),
                self._aware_timestamp(row["date"]),
            )
        execution = ExecutionService(
            Decimal(str(document.get("slippage_bps", self.settings.slippage_bps))),
            Decimal(str(document.get("transaction_cost_bps", self.settings.transaction_cost_bps))),
        )
        cursor = self.database.orders.find(
            {"experiment_id": experiment_id, "state": "WAITING_FOR_QUOTE"}
        ).sort("created_at", 1)
        async for order in cursor:
            symbol_rows = snapshot.loc[snapshot["symbol"] == order["symbol"]]
            if symbol_rows.empty:
                continue
            row = symbol_rows.iloc[0]
            quote = Quote(
                symbol=order["symbol"],
                exchange=self._symbol_exchange(str(order["symbol"])).value,
                provider_timestamp=self._aware_timestamp(row["date"]),
                ingestion_timestamp=now,
                last_price=Decimal(str(row["close"])),
                source_type="intraday_fallback",
            )
            if quote_is_stale(quote, now, self.settings.quote_stale_seconds):
                await self.events.emit(
                    experiment_id,
                    "ExecutionAgent",
                    "QUOTE_STALE",
                    severity="WARNING",
                    symbol=order["symbol"],
                )
                continue
            try:
                fill = execution.fill_next_quote(
                    order["symbol"],
                    Action(order["action"]),
                    int(order["quantity"]),
                    order["decision_timestamp"],
                    [quote],
                )
                if fill.action is Action.BUY:
                    portfolio.buy(
                        fill.symbol,
                        fill.quantity,
                        fill.execution_price,
                        fill.transaction_cost,
                        fill.fill_timestamp,
                    )
                else:
                    portfolio.sell(
                        fill.symbol,
                        fill.quantity,
                        fill.execution_price,
                        fill.transaction_cost,
                        fill.fill_timestamp,
                    )
                trade = {
                    "experiment_id": experiment_id,
                    "order_id": str(order["_id"]),
                    **fill.__dict__,
                    "resulting_cash": str(portfolio.cash),
                }
                await self.database.trades.update_one(
                    {"order_id": str(order["_id"])}, {"$setOnInsert": trade}, upsert=True
                )
                await self.database.orders.update_one(
                    {"_id": order["_id"], "state": "WAITING_FOR_QUOTE"},
                    {
                        "$set": {
                            "state": "FILLED",
                            "fill_timestamp": fill.fill_timestamp,
                            "execution_price": str(fill.execution_price),
                            "transaction_cost": str(fill.transaction_cost),
                        }
                    },
                )
                await self.events.emit(
                    experiment_id,
                    "ExecutionAgent",
                    "ORDER_FILLED",
                    symbol=fill.symbol,
                    details={
                        "action": fill.action.value,
                        "quantity": fill.quantity,
                        "execution_price": str(fill.execution_price),
                    },
                )
            except ValueError as exc:
                if str(exc) != "WAITING_FOR_QUOTE":
                    await self.database.orders.update_one(
                        {"_id": order["_id"]},
                        {"$set": {"state": "REJECTED", "rejection_reason": str(exc)}},
                    )
        await self._persist_portfolio(experiment_id, portfolio, now)
        return portfolio

    async def process_experiment(self, document: dict[str, Any]) -> None:
        experiment_id = str(document["_id"])
        if experiment_id not in self._owned and not await self.leases.acquire(
            experiment_id, self.worker_id
        ):
            return
        self._owned.add(experiment_id)
        await self.leases.heartbeat(experiment_id, self.worker_id)
        status = self.adapter.validate_contract()
        if status.status is not AlgorithmStatus.INSTALLED:
            if document["state"] == ExperimentState.INITIALIZING.value:
                await self.experiments.transition(
                    experiment_id,
                    ExperimentState.FAILED,
                    error_code="ALGORITHM_NOT_INSTALLED",
                    ended_at=datetime.now(UTC),
                )
            return
        if not await self._ensure_checkpoint_loaded():
            if document["state"] == ExperimentState.INITIALIZING.value:
                await self.experiments.transition(
                    experiment_id,
                    ExperimentState.FAILED,
                    error_code="MODEL_CHECKPOINT_NOT_FOUND",
                    ended_at=datetime.now(UTC),
                )
            await self.events.emit(
                experiment_id,
                "HSGAgent",
                "MODEL_CHECKPOINT_NOT_FOUND",
                severity="ERROR",
            )
            return
        exchange = Exchange(document.get("exchange", Exchange.NSE.value))
        if exchange.value not in self._loaded_exchanges:
            if document["state"] == ExperimentState.INITIALIZING.value:
                await self.experiments.transition(
                    experiment_id,
                    ExperimentState.FAILED,
                    error_code=f"MODEL_NOT_TRAINED_FOR_{exchange.value}",
                    ended_at=datetime.now(UTC),
                )
            await self.events.emit(
                experiment_id,
                "HSGAgent",
                "MODEL_MARKET_UNSUPPORTED",
                severity="ERROR",
                details={"exchange": exchange.value},
            )
            return
        now = datetime.now(UTC)
        calendar = self.calendars[exchange]
        market_open = calendar.is_market_open(now)
        if not market_open:
            if document["state"] == ExperimentState.RUNNING.value:
                await self.experiments.transition(experiment_id, ExperimentState.WAITING_FOR_MARKET)
            await self.events.emit(experiment_id, "MarketDataAgent", "MARKET_CLOSED")
            return
        if document["state"] == ExperimentState.WAITING_FOR_MARKET.value:
            await self.experiments.transition(experiment_id, ExperimentState.RUNNING)
            await self.events.emit(experiment_id, "Worker", "MARKET_OPENED")
        if document["state"] not in {
            ExperimentState.RUNNING.value,
            ExperimentState.WAITING_FOR_MARKET.value,
        }:
            return
        symbols = document.get("universe") or self._default_universe(exchange)
        history = await self._history(symbols)
        snapshot = self._symbol_features(history, symbols)
        if snapshot.empty:
            await self.events.emit(
                experiment_id,
                "MarketDataAgent",
                "DATA_PROVIDER_ERROR",
                severity="ERROR",
                details={"reason": "No feature-ready observations"},
            )
            return
        await self._persist_bars(experiment_id, snapshot, now)
        await self.events.emit(
            experiment_id,
            "MarketDataAgent",
            "MARKET_DATA_FALLBACK",
            severity="WARNING",
            details={"provider": "yfinance", "source_type": "intraday_fallback"},
        )
        portfolio = await self._execute_pending_orders(experiment_id, document, snapshot, now)
        candidates = rank_bearish_candidates(snapshot, now)
        await self.database.scanner_snapshots.insert_one(
            {
                "experiment_id": experiment_id,
                "timestamp": now,
                "candidates": candidates,
                "universe_size": len(snapshot),
            }
        )
        await self.events.emit(
            experiment_id,
            "ScannerAgent",
            "CANDIDATES_SELECTED",
            details={"symbols": [item["symbol"] for item in candidates]},
        )
        for candidate in candidates:
            symbol = str(candidate["symbol"])
            feature_row = snapshot.loc[snapshot["symbol"] == symbol].iloc[0].to_dict()
            try:
                prediction = await self.adapter.predict(
                    symbol=symbol,
                    timestamp=now,
                    features=feature_row,
                    context={"experiment_id": experiment_id},
                )
                prediction_id = str(uuid4())
                prediction_document = {
                    "_id": prediction_id,
                    "experiment_id": experiment_id,
                    "timestamp": now,
                    "decision_timestamp": now,
                    "algorithm_sha256": status.algorithm_sha256,
                    "feature_version": "market-features-v1",
                    **prediction.model_dump(mode="json"),
                }
                await self.database.predictions.insert_one(prediction_document)
                row = snapshot.loc[snapshot["symbol"] == symbol].iloc[0]
                quote = Quote(
                    symbol=symbol,
                    exchange=self._symbol_exchange(symbol).value,
                    provider_timestamp=self._aware_timestamp(row["date"]),
                    ingestion_timestamp=now,
                    last_price=Decimal(str(row["close"])),
                    source_type="intraday_fallback",
                )
                risk_config = RiskConfig(
                    max_positions=int(document.get("max_positions", self.settings.max_positions)),
                    max_position_pct=Decimal(
                        str(document.get("max_position_pct", self.settings.max_position_pct))
                    ),
                    minimum_cash_reserve_pct=Decimal(
                        str(document.get("minimum_cash_reserve_pct", "0.05"))
                    ),
                    minimum_confidence=Decimal(
                        str(
                            document.get(
                                "confidence_threshold",
                                self.settings.model_confidence_threshold,
                            )
                        )
                    ),
                )
                risk_decision = RiskService().evaluate(
                    prediction.action,
                    symbol,
                    Decimal(str(prediction.confidence)),
                    quote.last_price,
                    portfolio,
                    quote_is_stale(quote, now, self.settings.quote_stale_seconds),
                    risk_config,
                )
                risk_id = str(uuid4())
                await self.database.signals.insert_one(
                    {
                        "_id": risk_id,
                        "experiment_id": experiment_id,
                        "prediction_id": prediction_id,
                        "timestamp": now,
                        "raw_action": prediction.action.value,
                        **risk_decision.model_dump(mode="json"),
                    }
                )
                if not risk_decision.approved:
                    await self.events.emit(
                        experiment_id,
                        "RiskAgent",
                        "RISK_REJECTED",
                        severity="WARNING",
                        symbol=symbol,
                        model_version=prediction.model_version,
                        details=risk_decision.model_dump(mode="json"),
                    )
                    continue
                if risk_decision.final_action is Action.HOLD:
                    await self.events.emit(
                        experiment_id,
                        "HSGAgent",
                        "HOLD",
                        symbol=symbol,
                        model_version=prediction.model_version,
                        details={"confidence": prediction.confidence},
                    )
                    continue
                key = order_idempotency_key(
                    experiment_id, symbol, prediction_id, risk_decision.final_action
                )
                order_id = str(uuid4())
                result = await self.database.orders.update_one(
                    {"idempotency_key": key},
                    {
                        "$setOnInsert": {
                            "_id": order_id,
                            "experiment_id": experiment_id,
                            "prediction_id": prediction_id,
                            "risk_decision_id": risk_id,
                            "symbol": symbol,
                            "action": risk_decision.final_action.value,
                            "quantity": risk_decision.proposed_quantity,
                            "state": "WAITING_FOR_QUOTE",
                            "decision_timestamp": now,
                            "created_at": now,
                            "idempotency_key": key,
                        }
                    },
                    upsert=True,
                )
                if result.upserted_id is not None:
                    await self.events.emit(
                        experiment_id,
                        "ExecutionAgent",
                        "ORDER_CREATED",
                        symbol=symbol,
                        details={
                            "action": risk_decision.final_action.value,
                            "quantity": risk_decision.proposed_quantity,
                            "execution_policy": "FIRST_VALID_POST_DECISION_QUOTE",
                        },
                    )
            except Exception as exc:
                await self.events.emit(
                    experiment_id,
                    "HSGAgent",
                    "HSG_INFERENCE_FAILED",
                    severity="ERROR",
                    symbol=symbol,
                    details={"error": str(exc)},
                )

    @staticmethod
    def _symbol_exchange(symbol: str) -> Exchange:
        if symbol.endswith(".BO"):
            return Exchange.BSE
        if symbol.endswith(".NS"):
            return Exchange.NSE
        return Exchange.NASDAQ

    def _default_universe(self, exchange: Exchange) -> list[str]:
        universe = pd.read_csv(self.settings.configuration_directory / "trading_universe.csv")
        enabled = universe[universe["enabled"].astype(str).str.lower() == "true"]
        enabled = enabled[enabled["exchange"].astype(str).str.upper() == exchange.value]
        preferred = enabled[enabled["preferred"].astype(str).str.lower() == "true"]
        selected = preferred if not preferred.empty else enabled
        return selected["symbol"].astype(str).tolist()

    async def run(self) -> None:
        await self.recover()
        while not self.stop_event.is_set():
            await self.heartbeat()
            documents = await recoverable_experiments(self.database)
            for document in documents:
                if self.stop_event.is_set():
                    break
                try:
                    await self.process_experiment(document)
                except Exception:
                    logger.exception(
                        "experiment_cycle_failed",
                        extra={"service": "worker", "experiment_id": str(document["_id"])},
                    )
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(), timeout=self.settings.scan_interval_seconds
                )
            except TimeoutError:
                pass

    async def shutdown(self) -> None:
        self.stop_event.set()
        await self.heartbeat("STOPPING")
        for experiment_id in list(self._owned):
            await self.events.emit(
                experiment_id,
                "Worker",
                "WORKER_SHUTDOWN",
                details={"worker_id": self.worker_id},
            )
            await self.leases.release(experiment_id, self.worker_id)
        self._owned.clear()
        await self.heartbeat("STOPPED")
