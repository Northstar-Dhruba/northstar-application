"""Application simulation of executing one approved paper-trading intent.

Simulated execution is immediate and priced from the evidence the decision was
made on: the same observation that produced the recommendation supplies the fill
price and the fill instant. Nothing is looked up, no later bar is consulted, and
no clock is read, so executing the same intent against the same evidence always
produces the same result.

Pricing at the decision observation also makes paper execution agree with
forward research by construction rather than by coincidence: a recommendation
outcome measures movement from exactly this price.

This simulator models no slippage, commission, delay, partial fill, rejection
policy, short sale, portfolio change or profit and loss, and persists nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.paper_trading import (
    ExecutionIntent,
    OrderSide,
    PaperFill,
    PaperFillIdentity,
    PaperOrder,
    PaperOrderIdentity,
    PaperOrderStatus,
)

from northstar_application.application_services.analyze_asset_result import AnalyzeAssetResult

_HOLD_ACTION = "HOLD"

# The Core recommendation vocabulary and the paper-trading direction vocabulary
# are separate types with overlapping spellings, so the correspondence between
# them is stated explicitly rather than inferred from the shared text.
_ACTION_SIDES: dict[str, OrderSide] = {
    "BUY": OrderSide.BUY,
    "SELL": OrderSide.SELL,
}


@dataclass(frozen=True, slots=True)
class PaperExecution:
    """One simulated execution: a terminal order and the fill it produced.

    Only a filled execution is representable. PaperOrderStatus.REJECTED remains
    reserved for a later rejection policy, so this value never pairs a rejected
    order with a fill, and a rejected order is never produced by simulation.
    """

    order: PaperOrder
    fill: PaperFill

    def __post_init__(self) -> None:
        if not isinstance(self.order, PaperOrder):
            raise TypeError("PaperExecution order must be a PaperOrder.")
        if not isinstance(self.fill, PaperFill):
            raise TypeError("PaperExecution fill must be a PaperFill.")

        if self.order.status is not PaperOrderStatus.FILLED:
            raise ValueError("PaperExecution order must be filled.")
        if self.fill.order_identity != self.order.identity:
            raise ValueError("PaperExecution fill must reference its own order identity.")
        if self.fill.intent != self.order.intent:
            raise ValueError("PaperExecution fill must execute the order's own intent.")
        if self.fill.quantity != self.order.intent.quantity:
            raise ValueError("PaperExecution fill must execute the full intended quantity.")


class SimulatePaperExecutionUseCase:
    """Execute one approved intent against the evidence it was decided on.

    The use case holds no dependencies. It reads no repository and no clock,
    which is what makes a simulated execution reproducible from its inputs
    alone.
    """

    def execute(
        self,
        intent: ExecutionIntent,
        result: AnalyzeAssetResult,
        order_identity: PaperOrderIdentity,
        fill_identity: PaperFillIdentity,
    ) -> PaperExecution:
        """Fill one approved intent at its decision-time observed price."""
        self._validate_inputs(intent, result, order_identity, fill_identity)
        self._validate_result_coherence(result)
        self._validate_intent_matches_evidence(intent, result)

        context = result.market_observation_context
        order = PaperOrder(
            identity=order_identity,
            intent=intent,
            status=PaperOrderStatus.FILLED,
        )
        fill = PaperFill(
            identity=fill_identity,
            order_identity=order.identity,
            intent=intent,
            quantity=intent.quantity,
            price=context.latest_price,
            filled_at=context.observed_at,
        )
        return PaperExecution(order=order, fill=fill)

    @staticmethod
    def _validate_inputs(
        intent: ExecutionIntent,
        result: AnalyzeAssetResult,
        order_identity: PaperOrderIdentity,
        fill_identity: PaperFillIdentity,
    ) -> None:
        if intent is None:
            raise TypeError("SimulatePaperExecutionUseCase intent cannot be None.")
        if not isinstance(intent, ExecutionIntent):
            raise TypeError("SimulatePaperExecutionUseCase intent must be an ExecutionIntent.")
        if result is None:
            raise TypeError("SimulatePaperExecutionUseCase result cannot be None.")
        if not isinstance(result, AnalyzeAssetResult):
            raise TypeError("SimulatePaperExecutionUseCase result must be an AnalyzeAssetResult.")
        if order_identity is None:
            raise TypeError("SimulatePaperExecutionUseCase order identity cannot be None.")
        if not isinstance(order_identity, PaperOrderIdentity):
            raise TypeError(
                "SimulatePaperExecutionUseCase order identity must be a PaperOrderIdentity."
            )
        if fill_identity is None:
            raise TypeError("SimulatePaperExecutionUseCase fill identity cannot be None.")
        if not isinstance(fill_identity, PaperFillIdentity):
            raise TypeError(
                "SimulatePaperExecutionUseCase fill identity must be a PaperFillIdentity."
            )

    @staticmethod
    def _validate_result_coherence(result: AnalyzeAssetResult) -> None:
        """Reject evidence that disagrees with itself before pricing from it."""
        context = result.market_observation_context
        recommendation = result.recommendation
        analysis = recommendation.asset_analysis

        if recommendation.point_in_time.compare(context.observed_at) != 0:
            raise ValueError(
                "SimulatePaperExecutionUseCase recommendation instant must match the "
                "observed market context instant."
            )
        if analysis.point_in_time.compare(context.observed_at) != 0:
            raise ValueError(
                "SimulatePaperExecutionUseCase asset analysis instant must match the "
                "observed market context instant."
            )
        if analysis.listing_reference != context.listing_reference:
            raise ValueError(
                "SimulatePaperExecutionUseCase asset analysis listing must match the "
                "observed market context listing."
            )

    @staticmethod
    def _validate_intent_matches_evidence(
        intent: ExecutionIntent, result: AnalyzeAssetResult
    ) -> None:
        """Require the intent to be the one this exact decision produced.

        Executing an intent against someone else's evidence would price it from
        an observation it was never derived from, so correspondence is checked
        on listing, strategy, decision instant and direction before any order
        exists.
        """
        recommendation = result.recommendation

        if intent.listing_reference != recommendation.asset_analysis.listing_reference:
            raise ValueError(
                "SimulatePaperExecutionUseCase intent listing must match the "
                "recommendation listing."
            )
        if intent.strategy_identity != recommendation.strategy_identity:
            raise ValueError(
                "SimulatePaperExecutionUseCase intent strategy must match the "
                "recommendation strategy."
            )
        if intent.decided_at.compare(recommendation.point_in_time) != 0:
            raise ValueError(
                "SimulatePaperExecutionUseCase intent decision instant must match the "
                "recommendation instant."
            )

        action = recommendation.action.value
        if action == _HOLD_ACTION:
            raise ValueError("SimulatePaperExecutionUseCase cannot execute a HOLD recommendation.")

        side = _ACTION_SIDES.get(action)
        if side is None:
            raise ValueError(
                f"SimulatePaperExecutionUseCase cannot execute recommendation action {action!r}."
            )
        if intent.side is not side:
            raise ValueError(
                "SimulatePaperExecutionUseCase intent side must match the recommendation action."
            )
