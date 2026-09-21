"""Factual counts of what one paper trading run decided and executed.

Metrics here count events. They express no return, no profit and loss, no win
rate, accuracy or risk measure, and they never describe an execution as
successful or unsuccessful. A count of executions is not a score.

Holds and refused sales are counted rather than filtered, because how often a
strategy declined to act is part of what it did. Counts are grouped by listing
and strategy so one strategy's behaviour on one asset is never pooled with
another's.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.domain.value_objects import ListingReference
from northstar_core.paper_trading import OrderSide
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services.create_execution_intent import (
    ExecutionIntentNoIntentReason,
)
from northstar_application.application_services.paper_trading_run import PaperTradingRun
from northstar_application.application_services.run_paper_trading_decision import (
    PaperTradingResult,
)

_BUY_ACTION = "BUY"
_HOLD_ACTION = "HOLD"
_SELL_ACTION = "SELL"


def group_key(
    listing_reference: ListingReference, strategy_identity: StrategyIdentity
) -> tuple[str, str, str]:
    """Return the canonical group order key: listing, then strategy.

    Metrics calculation and report correspondence share this one definition so
    they can never disagree about what canonical order means.
    """
    return (
        listing_reference.symbol.value,
        listing_reference.exchange_code.value,
        strategy_identity.identity,
    )


@dataclass(frozen=True, slots=True)
class PaperTradingStrategyListingMetrics:
    """What one strategy decided and executed on one listing.

    Recommendation counts partition the group's decisions by what was advised.
    Outcome counts partition the same decisions by what followed: an execution,
    a hold, or a sale the portfolio could not support. Both partitions cover
    every decision exactly once, so nothing is silently dropped.
    """

    listing_reference: ListingReference
    strategy_identity: StrategyIdentity
    total_decisions: int
    buy_recommendations: int
    hold_recommendations: int
    sell_recommendations: int
    execution_count: int
    buy_execution_count: int
    sell_execution_count: int
    hold_no_intent_count: int
    insufficient_position_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.listing_reference, ListingReference):
            raise TypeError(
                "PaperTradingStrategyListingMetrics listing reference must be a ListingReference."
            )
        if not isinstance(self.strategy_identity, StrategyIdentity):
            raise TypeError(
                "PaperTradingStrategyListingMetrics strategy identity must be a StrategyIdentity."
            )
        for name in (
            "total_decisions",
            "buy_recommendations",
            "hold_recommendations",
            "sell_recommendations",
            "execution_count",
            "buy_execution_count",
            "sell_execution_count",
            "hold_no_intent_count",
            "insufficient_position_count",
        ):
            count = getattr(self, name)
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(f"PaperTradingStrategyListingMetrics {name} must be an integer.")
            if count < 0:
                raise ValueError(f"PaperTradingStrategyListingMetrics {name} cannot be negative.")

        advised = self.buy_recommendations + self.hold_recommendations + self.sell_recommendations
        if advised != self.total_decisions:
            raise ValueError(
                "PaperTradingStrategyListingMetrics recommendation counts must equal the "
                "total decision count."
            )
        followed = (
            self.execution_count + self.hold_no_intent_count + self.insufficient_position_count
        )
        if followed != self.total_decisions:
            raise ValueError(
                "PaperTradingStrategyListingMetrics outcome counts must equal the total "
                "decision count."
            )
        if self.buy_execution_count + self.sell_execution_count != self.execution_count:
            raise ValueError(
                "PaperTradingStrategyListingMetrics execution sides must equal the execution count."
            )


class CalculatePaperTradingMetricsUseCase:
    """Count one paper trading run's decisions per listing and strategy."""

    def execute(self, run: PaperTradingRun) -> tuple[PaperTradingStrategyListingMetrics, ...]:
        """Return one metrics value per listing and strategy pair, in canonical order."""
        if run is None:
            raise TypeError("CalculatePaperTradingMetricsUseCase run cannot be None.")
        if not isinstance(run, PaperTradingRun):
            raise TypeError("CalculatePaperTradingMetricsUseCase run must be a PaperTradingRun.")

        grouped: dict[tuple[str, str, str], list[PaperTradingResult]] = {}
        for result in run.results:
            recommendation = result.result.recommendation
            key = group_key(
                recommendation.asset_analysis.listing_reference,
                recommendation.strategy_identity,
            )
            grouped.setdefault(key, []).append(result)

        return tuple(self._metrics(tuple(grouped[key])) for key in sorted(grouped))

    @staticmethod
    def _metrics(
        results: tuple[PaperTradingResult, ...],
    ) -> PaperTradingStrategyListingMetrics:
        """Count one group, reading only the facts each result retained."""
        actions = [result.result.recommendation.action.value for result in results]
        executions = [result.execution for result in results]
        reasons = [result.decision.no_intent_reason for result in results]

        executed = [execution for execution in executions if execution is not None]
        recommendation = results[0].result.recommendation
        return PaperTradingStrategyListingMetrics(
            listing_reference=recommendation.asset_analysis.listing_reference,
            strategy_identity=recommendation.strategy_identity,
            total_decisions=len(results),
            buy_recommendations=actions.count(_BUY_ACTION),
            hold_recommendations=actions.count(_HOLD_ACTION),
            sell_recommendations=actions.count(_SELL_ACTION),
            execution_count=len(executed),
            buy_execution_count=sum(
                1 for execution in executed if execution.fill.side is OrderSide.BUY
            ),
            sell_execution_count=sum(
                1 for execution in executed if execution.fill.side is OrderSide.SELL
            ),
            hold_no_intent_count=sum(
                1 for reason in reasons if reason is ExecutionIntentNoIntentReason.HOLD
            ),
            insufficient_position_count=sum(
                1
                for reason in reasons
                if reason is ExecutionIntentNoIntentReason.INSUFFICIENT_POSITION
            ),
        )
