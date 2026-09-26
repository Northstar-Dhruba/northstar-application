"""Application translation of one produced recommendation into paper-trading intent.

Translation is the single place where advice becomes an intention to act. It is
deliberately explicit about doing nothing: an advised HOLD, or a SELL the
portfolio cannot support, returns a decision that says so rather than returning
nothing at all, so a caller can always account for every recommendation it saw.

Translation applies no position sizing policy -- the caller supplies the
quantity -- and creates no order, fill or portfolio change. It reads no clock:
the decision instant comes from the recommendation's own observed evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from northstar_core.foundation.value_objects import Quantity
from northstar_core.paper_trading import (
    ExecutionIntent,
    OrderSide,
    PaperPortfolioIdentity,
    Position,
)

from northstar_application.application_services._recommendation_action import executable_side
from northstar_application.application_services._result_coherence import (
    validate_result_coherence,
)
from northstar_application.application_services.analyze_asset_result import AnalyzeAssetResult


class ExecutionIntentNoIntentReason(StrEnum):
    """Closed vocabulary for why a recommendation produced no execution intent.

    HOLD means the strategy advised no action. INSUFFICIENT_POSITION means it
    advised selling more than the portfolio holds, or holds at all; paper
    trading is long-only, so a sale can never open a short.
    """

    HOLD = "HOLD"
    INSUFFICIENT_POSITION = "INSUFFICIENT_POSITION"


@dataclass(frozen=True, slots=True)
class ExecutionIntentDecision:
    """Outcome of translating one recommendation into paper-trading intent.

    Exactly one of ``intent`` and ``no_intent_reason`` is present. A decision is
    always returned so a recommendation is never silently dropped, and a
    deliberate non-action is never indistinguishable from a missing result.
    """

    intent: ExecutionIntent | None = None
    no_intent_reason: ExecutionIntentNoIntentReason | None = None

    def __post_init__(self) -> None:
        if self.intent is not None and not isinstance(self.intent, ExecutionIntent):
            raise TypeError("ExecutionIntentDecision intent must be an ExecutionIntent or None.")
        if self.no_intent_reason is not None and not isinstance(
            self.no_intent_reason, ExecutionIntentNoIntentReason
        ):
            raise TypeError(
                "ExecutionIntentDecision no-intent reason must be an "
                "ExecutionIntentNoIntentReason or None."
            )
        if (self.intent is None) == (self.no_intent_reason is None):
            raise ValueError(
                "ExecutionIntentDecision must carry exactly one of intent or no-intent reason."
            )

    @property
    def has_intent(self) -> bool:
        """Return whether the recommendation produced an intention to act."""
        return self.intent is not None


class CreateExecutionIntentUseCase:
    """Translate one produced recommendation into a paper-trading decision.

    The use case holds no dependencies: translation needs only the decision-time
    evidence it is given, which is what makes it reproducible.
    """

    def execute(
        self,
        result: AnalyzeAssetResult,
        portfolio_identity: PaperPortfolioIdentity,
        quantity: Quantity,
        position: Position | None = None,
    ) -> ExecutionIntentDecision:
        """Return the execution decision for one produced recommendation.

        ``position`` is required only to validate a SELL; it is ignored for a
        BUY and is never read for a HOLD. It is never mutated, and no holding is
        reduced here.
        """
        self._validate_inputs(result, portfolio_identity, quantity, position)
        validate_result_coherence(result, "CreateExecutionIntentUseCase")

        recommendation = result.recommendation
        side = executable_side(recommendation.action, "CreateExecutionIntentUseCase")
        if side is None:
            return ExecutionIntentDecision(no_intent_reason=ExecutionIntentNoIntentReason.HOLD)

        listing_reference = recommendation.asset_analysis.listing_reference
        if side is OrderSide.SELL and not self._position_supports_sale(position, result, quantity):
            return ExecutionIntentDecision(
                no_intent_reason=ExecutionIntentNoIntentReason.INSUFFICIENT_POSITION
            )

        return ExecutionIntentDecision(
            intent=ExecutionIntent(
                portfolio_identity=portfolio_identity,
                listing_reference=listing_reference,
                side=side,
                quantity=quantity,
                strategy_identity=recommendation.strategy_identity,
                decided_at=recommendation.point_in_time,
            )
        )

    @staticmethod
    def _position_supports_sale(
        position: Position | None,
        result: AnalyzeAssetResult,
        quantity: Quantity,
    ) -> bool:
        """Return whether an existing holding covers the advised sale."""
        if position is None:
            return False
        if position.listing_reference != result.recommendation.asset_analysis.listing_reference:
            return False
        return quantity <= position.quantity

    @staticmethod
    def _validate_inputs(
        result: AnalyzeAssetResult,
        portfolio_identity: PaperPortfolioIdentity,
        quantity: Quantity,
        position: Position | None,
    ) -> None:
        if result is None:
            raise TypeError("CreateExecutionIntentUseCase result cannot be None.")
        if not isinstance(result, AnalyzeAssetResult):
            raise TypeError("CreateExecutionIntentUseCase result must be an AnalyzeAssetResult.")
        if portfolio_identity is None:
            raise TypeError("CreateExecutionIntentUseCase portfolio identity cannot be None.")
        if not isinstance(portfolio_identity, PaperPortfolioIdentity):
            raise TypeError(
                "CreateExecutionIntentUseCase portfolio identity must be a PaperPortfolioIdentity."
            )
        if quantity is None:
            raise TypeError("CreateExecutionIntentUseCase quantity cannot be None.")
        if not isinstance(quantity, Quantity):
            raise TypeError("CreateExecutionIntentUseCase quantity must be a Quantity.")
        if position is not None and not isinstance(position, Position):
            raise TypeError("CreateExecutionIntentUseCase position must be a Position or None.")
