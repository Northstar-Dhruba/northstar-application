"""Application orchestration of one decision through paper trading.

One produced recommendation is translated into an intent, executed against the
evidence it was decided on, persisted as a fill, and folded back into a
portfolio. Every step is already defined elsewhere; this module only sequences
them and owns the one thing sequencing introduces: what a retry means.

A retry must converge, not accumulate. Because execution identities are derived
from the decision boundary rather than generated, re-running the same decision
produces the same fill and the store accepts it idempotently. The subtle part
is reconstruction: the portfolio a decision is judged against must exclude that
decision's own fill, or a repeated SELL would see the holding it already
reduced and refuse itself.

This orchestration reads no clock, generates no identity of its own, mutates no
holding, persists nothing but the fill, and calculates no profit and loss.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.foundation.value_objects import PointInTime, Quantity
from northstar_core.paper_trading import (
    OrderSide,
    PaperFill,
    PaperPortfolio,
    PaperPortfolioIdentity,
    Position,
)

from northstar_application.application_services._recommendation_action import executable_side
from northstar_application.application_services.analyze_asset_result import AnalyzeAssetResult
from northstar_application.application_services.build_paper_portfolio import (
    BuildPaperPortfolioUseCase,
    InvalidPaperFillHistoryError,
)
from northstar_application.application_services.create_execution_intent import (
    CreateExecutionIntentUseCase,
    ExecutionIntentDecision,
    ExecutionIntentNoIntentReason,
)
from northstar_application.application_services.create_paper_execution_identities import (
    CreatePaperExecutionIdentitiesUseCase,
)
from northstar_application.application_services.simulate_paper_execution import (
    PaperExecution,
    SimulatePaperExecutionUseCase,
)
from northstar_application.ports import (
    PaperFillConflictError,
    PaperFillQuery,
    PaperFillRepository,
    PaperFillStore,
)


class PaperTradingContractViolationError(ValueError):
    """Raised when a paper trading port violates its contract."""


@dataclass(frozen=True, slots=True)
class PaperTradingResult:
    """The complete outcome of running one decision through paper trading.

    ``portfolio_identity`` records which paper portfolio the run was asked
    about, so the result is attributable even when the decision intended
    nothing. Without it a no-intent result carries no statement of ownership at
    all, and a portfolio belonging to someone else would be indistinguishable
    from the right one.

    ``result`` is the decision-time evidence the run acted on, retained so the
    outcome stays auditable even when nothing was executed: a hold and a refused
    sale still record which listing, strategy, recommendation and instant they
    concerned.

    ``execution`` is present exactly when the decision produced an intent, so a
    caller can never read an execution that no decision authorised, nor lose an
    execution that one did. ``portfolio`` is the holdings as of the decision
    instant, after any fill this run persisted.
    """

    portfolio_identity: PaperPortfolioIdentity
    result: AnalyzeAssetResult
    decision: ExecutionIntentDecision
    execution: PaperExecution | None
    portfolio: PaperPortfolio

    def __post_init__(self) -> None:
        self._validate_types()

        if self.portfolio.identity != self.portfolio_identity:
            raise ValueError(
                "PaperTradingResult portfolio must belong to the requested paper portfolio."
            )
        if self.decision.has_intent and self.execution is None:
            raise ValueError("PaperTradingResult must carry an execution for an intended decision.")
        if not self.decision.has_intent and self.execution is not None:
            raise ValueError(
                "PaperTradingResult cannot carry an execution for a decision that intended none."
            )

        self._validate_decision_follows_the_recommendation()
        self._validate_intent_matches_the_recommendation()

    def _validate_types(self) -> None:
        if not isinstance(self.portfolio_identity, PaperPortfolioIdentity):
            raise TypeError(
                "PaperTradingResult portfolio identity must be a PaperPortfolioIdentity."
            )
        if not isinstance(self.result, AnalyzeAssetResult):
            raise TypeError("PaperTradingResult result must be an AnalyzeAssetResult.")
        if not isinstance(self.decision, ExecutionIntentDecision):
            raise TypeError("PaperTradingResult decision must be an ExecutionIntentDecision.")
        if self.execution is not None and not isinstance(self.execution, PaperExecution):
            raise TypeError("PaperTradingResult execution must be a PaperExecution or None.")
        if not isinstance(self.portfolio, PaperPortfolio):
            raise TypeError("PaperTradingResult portfolio must be a PaperPortfolio.")

    def _validate_decision_follows_the_recommendation(self) -> None:
        """Require a decision this recommendation could actually have produced.

        A HOLD can only decline; a BUY can only buy; a SELL either sells or
        declines for want of a holding. Any other pairing means the retained
        evidence describes a different decision from the one recorded, which
        would make the result unauditable exactly where auditing matters most.
        """
        advised_side = executable_side(self.result.recommendation.action, "PaperTradingResult")

        if advised_side is None:
            if self.decision.has_intent:
                raise ValueError(
                    "PaperTradingResult cannot intend execution for a HOLD recommendation."
                )
            if self.decision.no_intent_reason is not ExecutionIntentNoIntentReason.HOLD:
                raise ValueError("PaperTradingResult must decline a HOLD recommendation as a hold.")
            return

        if not self.decision.has_intent:
            if advised_side is OrderSide.BUY:
                raise ValueError(
                    "PaperTradingResult must intend execution for a BUY recommendation."
                )
            if (
                self.decision.no_intent_reason
                is not ExecutionIntentNoIntentReason.INSUFFICIENT_POSITION
            ):
                raise ValueError(
                    "PaperTradingResult may only decline a SELL recommendation for want of "
                    "a position."
                )
            return

        if self.decision.intent.side is not advised_side:
            raise ValueError("PaperTradingResult intent side must match the recommended action.")

    def _validate_intent_matches_the_recommendation(self) -> None:
        """Require an intent to be the one this exact decision produced."""
        intent = self.decision.intent
        if intent is None:
            return

        recommendation = self.result.recommendation
        if intent.portfolio_identity != self.portfolio_identity:
            raise ValueError(
                "PaperTradingResult decision must intend execution in the requested "
                "paper portfolio."
            )
        if intent.listing_reference != recommendation.asset_analysis.listing_reference:
            raise ValueError(
                "PaperTradingResult intent listing must match the recommendation listing."
            )
        if intent.strategy_identity != recommendation.strategy_identity:
            raise ValueError(
                "PaperTradingResult intent strategy must match the recommendation strategy."
            )
        if intent.decided_at.compare(recommendation.point_in_time) != 0:
            raise ValueError(
                "PaperTradingResult intent decision instant must match the recommendation instant."
            )
        if self.execution.order.intent != intent:
            raise ValueError("PaperTradingResult execution must execute the decision's own intent.")


class RunPaperTradingDecisionUseCase:
    """Run one produced decision through translation, execution and folding.

    Ports are injected because they reach outside the Application. The
    dependency-free use cases default, because constructing them takes no
    decision a caller could get wrong.
    """

    def __init__(
        self,
        fill_repository: PaperFillRepository,
        fill_store: PaperFillStore,
        create_execution_intent: CreateExecutionIntentUseCase | None = None,
        create_execution_identities: CreatePaperExecutionIdentitiesUseCase | None = None,
        simulate_execution: SimulatePaperExecutionUseCase | None = None,
        build_portfolio: BuildPaperPortfolioUseCase | None = None,
    ) -> None:
        if fill_repository is None:
            raise TypeError("RunPaperTradingDecisionUseCase fill repository cannot be None.")
        if not isinstance(fill_repository, PaperFillRepository):
            raise TypeError(
                "RunPaperTradingDecisionUseCase fill repository must be a PaperFillRepository."
            )
        if fill_store is None:
            raise TypeError("RunPaperTradingDecisionUseCase fill store cannot be None.")
        if not isinstance(fill_store, PaperFillStore):
            raise TypeError("RunPaperTradingDecisionUseCase fill store must be a PaperFillStore.")

        self._fill_repository = fill_repository
        self._fill_store = fill_store
        self._create_execution_intent = self._dependency(
            create_execution_intent, CreateExecutionIntentUseCase, "create_execution_intent"
        )
        self._create_execution_identities = self._dependency(
            create_execution_identities,
            CreatePaperExecutionIdentitiesUseCase,
            "create_execution_identities",
        )
        self._simulate_execution = self._dependency(
            simulate_execution, SimulatePaperExecutionUseCase, "simulate_execution"
        )
        self._build_portfolio = self._dependency(
            build_portfolio, BuildPaperPortfolioUseCase, "build_portfolio"
        )

    def execute(
        self,
        result: AnalyzeAssetResult,
        portfolio_identity: PaperPortfolioIdentity,
        quantity: Quantity,
    ) -> PaperTradingResult:
        """Run one decision to a persisted fill and a rebuilt portfolio."""
        self._validate_inputs(result, portfolio_identity, quantity)

        decision_instant = result.market_observation_context.observed_at
        fills = self._load_fills(portfolio_identity)
        as_of_portfolio = self._build(portfolio_identity, fills, decision_instant)

        already_executed = self._fill_for_this_decision(fills, result, portfolio_identity)
        pre_decision_portfolio = (
            as_of_portfolio
            if already_executed is None
            else self._build(
                portfolio_identity,
                self._without(fills, already_executed),
                decision_instant,
            )
        )

        decision = self._create_execution_intent.execute(
            result,
            portfolio_identity,
            quantity,
            self._held_position(pre_decision_portfolio, result),
        )

        if not decision.has_intent:
            if already_executed is not None:
                raise PaperFillConflictError(
                    "A paper fill is already persisted for this decision, which now intends "
                    "no execution."
                )
            return PaperTradingResult(
                portfolio_identity=portfolio_identity,
                result=result,
                decision=decision,
                execution=None,
                portfolio=as_of_portfolio,
            )

        identities = self._create_execution_identities.execute(decision.intent)
        execution = self._simulate_execution.execute(
            decision.intent,
            result,
            identities.order_identity,
            identities.fill_identity,
        )
        self._persist(execution.fill)

        reloaded = self._load_fills(portfolio_identity)
        return PaperTradingResult(
            portfolio_identity=portfolio_identity,
            result=result,
            decision=decision,
            execution=execution,
            portfolio=self._build(portfolio_identity, reloaded, decision_instant),
        )

    # -- collaboration -----------------------------------------------------

    def _load_fills(self, portfolio_identity: PaperPortfolioIdentity) -> tuple[PaperFill, ...]:
        """Read one portfolio's history, checking the shape the port promised."""
        fills = self._fill_repository.get_fills(PaperFillQuery(portfolio_identity))
        if not isinstance(fills, tuple):
            raise PaperTradingContractViolationError(
                "PaperFillRepository must return a tuple of PaperFill."
            )
        if not all(isinstance(fill, PaperFill) for fill in fills):
            raise PaperTradingContractViolationError(
                "PaperFillRepository must return PaperFill instances."
            )
        return fills

    def _persist(self, fill: PaperFill) -> None:
        """Store exactly one fill, requiring an honest accepted count."""
        stored = self._fill_store.store((fill,))
        if isinstance(stored, bool) or not isinstance(stored, int):
            raise PaperTradingContractViolationError(
                "PaperFillStore must return an integer count of accepted fills."
            )
        if stored != 1:
            raise PaperTradingContractViolationError(
                f"PaperFillStore must accept exactly one fill, but reported {stored}."
            )

    def _build(
        self,
        portfolio_identity: PaperPortfolioIdentity,
        fills: tuple[PaperFill, ...],
        as_of: PointInTime,
    ) -> PaperPortfolio:
        """Fold a history, which also validates its coherence and ordering."""
        return self._build_portfolio.execute(portfolio_identity, fills, as_of)

    # -- this decision's own fill ------------------------------------------

    @staticmethod
    def _fill_for_this_decision(
        fills: tuple[PaperFill, ...],
        result: AnalyzeAssetResult,
        portfolio_identity: PaperPortfolioIdentity,
    ) -> PaperFill | None:
        """Return the fill already persisted for this decision boundary, if any.

        The boundary is the portfolio, listing, strategy and decision instant.
        Side and quantity are excluded on purpose: a retry that changed them is
        still the same decision, and must be recognised as one so it converges
        or conflicts rather than executing twice.
        """
        recommendation = result.recommendation
        listing_reference = recommendation.asset_analysis.listing_reference

        matches = [
            fill
            for fill in fills
            if fill.portfolio_identity == portfolio_identity
            and fill.listing_reference == listing_reference
            and fill.strategy_identity == recommendation.strategy_identity
            and fill.decided_at.compare(recommendation.point_in_time) == 0
        ]
        if len(matches) > 1:
            raise InvalidPaperFillHistoryError(
                "More than one paper fill is persisted for one decision boundary."
            )
        return matches[0] if matches else None

    @staticmethod
    def _without(fills: tuple[PaperFill, ...], excluded: PaperFill) -> tuple[PaperFill, ...]:
        return tuple(fill for fill in fills if fill.identity != excluded.identity)

    @staticmethod
    def _held_position(portfolio: PaperPortfolio, result: AnalyzeAssetResult) -> Position | None:
        return portfolio.get_position(result.recommendation.asset_analysis.listing_reference)

    # -- validation --------------------------------------------------------

    @staticmethod
    def _dependency(value: object, expected: type, name: str) -> object:
        if value is None:
            return expected()
        if not isinstance(value, expected):
            raise TypeError(f"RunPaperTradingDecisionUseCase {name} must be a {expected.__name__}.")
        return value

    @staticmethod
    def _validate_inputs(
        result: AnalyzeAssetResult,
        portfolio_identity: PaperPortfolioIdentity,
        quantity: Quantity,
    ) -> None:
        if result is None:
            raise TypeError("RunPaperTradingDecisionUseCase result cannot be None.")
        if not isinstance(result, AnalyzeAssetResult):
            raise TypeError("RunPaperTradingDecisionUseCase result must be an AnalyzeAssetResult.")
        if portfolio_identity is None:
            raise TypeError("RunPaperTradingDecisionUseCase portfolio identity cannot be None.")
        if not isinstance(portfolio_identity, PaperPortfolioIdentity):
            raise TypeError(
                "RunPaperTradingDecisionUseCase portfolio identity must be a "
                "PaperPortfolioIdentity."
            )
        if quantity is None:
            raise TypeError("RunPaperTradingDecisionUseCase quantity cannot be None.")
        if not isinstance(quantity, Quantity):
            raise TypeError("RunPaperTradingDecisionUseCase quantity must be a Quantity.")
