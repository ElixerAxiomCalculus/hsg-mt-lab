from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

MONEY = Decimal("0.01")


def money(value: Decimal) -> Decimal:
    return value.quantize(MONEY, rounding=ROUND_HALF_UP)


@dataclass
class Holding:
    symbol: str
    quantity: int
    average_entry_price: Decimal
    current_market_price: Decimal
    first_entry_timestamp: datetime
    last_update_timestamp: datetime
    realized_pnl: Decimal = Decimal("0")

    @property
    def market_value(self) -> Decimal:
        return money(self.current_market_price * self.quantity)

    @property
    def unrealized_pnl(self) -> Decimal:
        return money((self.current_market_price - self.average_entry_price) * self.quantity)


@dataclass
class Portfolio:
    initial_capital: Decimal
    cash: Decimal
    positions: dict[str, Holding] = field(default_factory=dict)
    transaction_costs: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")

    @classmethod
    def create(cls, initial_capital: Decimal) -> "Portfolio":
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        return cls(money(initial_capital), money(initial_capital))

    @property
    def invested_value(self) -> Decimal:
        return money(sum((item.market_value for item in self.positions.values()), Decimal("0")))

    @property
    def equity(self) -> Decimal:
        return money(self.cash + self.invested_value)

    @property
    def unrealized_pnl(self) -> Decimal:
        return money(sum((item.unrealized_pnl for item in self.positions.values()), Decimal("0")))

    def mark(self, symbol: str, price: Decimal, at: datetime) -> None:
        if price <= 0:
            raise ValueError("Price must be positive")
        if symbol in self.positions:
            holding = self.positions[symbol]
            holding.current_market_price = price
            holding.last_update_timestamp = at

    def buy(
        self, symbol: str, quantity: int, execution_price: Decimal, cost: Decimal, at: datetime
    ) -> None:
        if quantity <= 0 or quantity != int(quantity):
            raise ValueError("Whole-share positive quantity required")
        debit = money(execution_price * quantity + cost)
        if debit > self.cash:
            raise ValueError("INSUFFICIENT_CAPITAL")
        existing = self.positions.get(symbol)
        if existing:
            total_quantity = existing.quantity + quantity
            average = (
                existing.average_entry_price * existing.quantity + execution_price * quantity
            ) / total_quantity
            existing.quantity = total_quantity
            existing.average_entry_price = money(average)
            existing.current_market_price = execution_price
            existing.last_update_timestamp = at
        else:
            self.positions[symbol] = Holding(
                symbol, quantity, money(execution_price), money(execution_price), at, at
            )
        self.cash = money(self.cash - debit)
        self.transaction_costs = money(self.transaction_costs + cost)

    def sell(
        self, symbol: str, quantity: int, execution_price: Decimal, cost: Decimal, at: datetime
    ) -> None:
        holding = self.positions.get(symbol)
        if not holding or quantity <= 0 or quantity > holding.quantity:
            raise ValueError("INSUFFICIENT_POSITION")
        proceeds = money(execution_price * quantity - cost)
        pnl = money((execution_price - holding.average_entry_price) * quantity - cost)
        self.cash = money(self.cash + proceeds)
        self.realized_pnl = money(self.realized_pnl + pnl)
        self.transaction_costs = money(self.transaction_costs + cost)
        holding.quantity -= quantity
        holding.realized_pnl = money(holding.realized_pnl + pnl)
        holding.current_market_price = execution_price
        holding.last_update_timestamp = at
        if holding.quantity == 0:
            del self.positions[symbol]

    def snapshot(self) -> dict[str, object]:
        return {
            "initial_capital": str(self.initial_capital),
            "cash": str(self.cash),
            "invested_value": str(self.invested_value),
            "equity": str(self.equity),
            "realized_pnl": str(self.realized_pnl),
            "unrealized_pnl": str(self.unrealized_pnl),
            "transaction_costs": str(self.transaction_costs),
            "positions": [
                {
                    "symbol": holding.symbol,
                    "quantity": holding.quantity,
                    "average_entry_price": str(holding.average_entry_price),
                    "current_market_price": str(holding.current_market_price),
                    "market_value": str(holding.market_value),
                    "unrealized_pnl": str(holding.unrealized_pnl),
                }
                for holding in self.positions.values()
            ],
        }
