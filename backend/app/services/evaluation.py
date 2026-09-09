import math
from decimal import Decimal


def maximum_drawdown(values: list[Decimal]) -> Decimal | None:
    if not values:
        return None
    peak = values[0]
    worst = Decimal("0")
    for value in values:
        peak = max(peak, value)
        if peak:
            worst = min(worst, value / peak - 1)
    return worst


def financial_metrics(
    initial_capital: Decimal,
    ending_equity: Decimal,
    realized_pnl: Decimal,
    unrealized_pnl: Decimal,
    transaction_costs: Decimal,
    equity_curve: list[Decimal],
) -> dict[str, object]:
    total_return = (ending_equity / initial_capital - 1) if initial_capital else Decimal("0")
    return {
        "starting_capital": str(initial_capital),
        "ending_portfolio_value": str(ending_equity),
        "total_return": str(total_return),
        "realized_pnl": str(realized_pnl),
        "unrealized_pnl": str(unrealized_pnl),
        "gross_pnl": str(realized_pnl + unrealized_pnl + transaction_costs),
        "transaction_costs": str(transaction_costs),
        "net_pnl": str(ending_equity - initial_capital),
        "maximum_drawdown": str(maximum_drawdown(equity_curve) or Decimal("0")),
        "warning": "Short experiments are not statistically conclusive"
        if len(equity_curve) < 30
        else None,
        "finite": math.isfinite(float(total_return)),
    }


def performance_metrics(
    initial: Decimal,
    ending: Decimal,
    equity_curve: list[Decimal],
    closed_trade_pnls: list[Decimal],
    benchmark_return: Decimal | None = None,
) -> dict[str, object]:
    winners = [value for value in closed_trade_pnls if value > 0]
    losers = [value for value in closed_trade_pnls if value < 0]
    gross_profit = sum(winners, Decimal("0"))
    gross_loss = abs(sum(losers, Decimal("0")))
    total_return = ending / initial - 1 if initial else Decimal("0")
    return {
        "starting_capital": str(initial),
        "ending_portfolio_value": str(ending),
        "total_return": str(total_return),
        "trade_count": len(closed_trade_pnls),
        "winning_closed_trades": len(winners),
        "losing_closed_trades": len(losers),
        "win_rate": len(winners) / len(closed_trade_pnls) if closed_trade_pnls else None,
        "average_winner": str(gross_profit / len(winners)) if winners else None,
        "average_loser": str(sum(losers, Decimal("0")) / len(losers)) if losers else None,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss else None,
        "maximum_drawdown": str(maximum_drawdown(equity_curve)),
        "benchmark_return": str(benchmark_return) if benchmark_return is not None else None,
        "excess_return": str(total_return - benchmark_return)
        if benchmark_return is not None
        else None,
        "statistical_warning": "Too few observations for reliable annualized statistics"
        if len(equity_curve) < 30
        else None,
        "finite": math.isfinite(float(total_return)),
    }
