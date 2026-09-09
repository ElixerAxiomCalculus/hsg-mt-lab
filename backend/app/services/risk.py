from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

from app.schemas.trading import Action, RiskDecision
from app.services.portfolio import Portfolio


@dataclass(frozen=True)
class RiskConfig:
    max_positions: int = 5
    max_position_pct: Decimal = Decimal("0.20")
    max_total_exposure_pct: Decimal = Decimal("0.95")
    minimum_cash_reserve_pct: Decimal = Decimal("0.05")
    minimum_confidence: Decimal = Decimal("0.65")


class RiskService:
    def evaluate(
        self,
        action: Action,
        symbol: str,
        confidence: Decimal,
        reference_price: Decimal,
        portfolio: Portfolio,
        stale: bool,
        config: RiskConfig,
    ) -> RiskDecision:
        def reject(rule: str, quantity: int = 0) -> RiskDecision:
            return RiskDecision(
                approved=False,
                proposed_action=action,
                proposed_quantity=quantity,
                final_action=Action.HOLD,
                violated_rule=rule,
                override_reason=rule,
            )

        if stale:
            return reject("DATA_STALE")
        if confidence < config.minimum_confidence:
            return reject("BELOW_CONFIDENCE_THRESHOLD")
        if action is Action.HOLD:
            return RiskDecision(
                approved=True,
                proposed_action=action,
                proposed_quantity=0,
                final_action=action,
            )
        if action is Action.SELL:
            holding = portfolio.positions.get(symbol)
            if not holding:
                return reject("NO_OWNED_POSITION")
            return RiskDecision(
                approved=True,
                proposed_action=action,
                proposed_quantity=holding.quantity,
                final_action=action,
            )
        if symbol not in portfolio.positions and len(portfolio.positions) >= config.max_positions:
            return reject("MAX_POSITIONS")
        nav = portfolio.equity
        allocation = min(
            nav * config.max_position_pct,
            portfolio.cash - nav * config.minimum_cash_reserve_pct,
        )
        quantity = int((allocation / reference_price).to_integral_value(rounding=ROUND_DOWN))
        if quantity < 1:
            return reject("INSUFFICIENT_CAPITAL_FOR_ONE_SHARE")
        projected_exposure = portfolio.invested_value + reference_price * quantity
        if projected_exposure > nav * config.max_total_exposure_pct:
            return reject("MAX_TOTAL_EXPOSURE", quantity)
        return RiskDecision(
            approved=True,
            proposed_action=action,
            proposed_quantity=quantity,
            final_action=action,
        )
