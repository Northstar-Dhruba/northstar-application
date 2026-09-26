"""One portfolio's sequence of paper trading decisions.

A run is the ordered record of what a portfolio was asked and what happened,
including every decision that executed nothing. Holds and refused sales stay in
the sequence rather than being filtered out, because a run that showed only
executions could not answer how often a strategy declined to act.

The run is an in-memory artifact. It aggregates results, orders them and
guarantees their coherence; it stores nothing, reads no clock, and interprets
no outcome as good or bad.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.foundation.value_objects import PointInTime
from northstar_core.paper_trading import PaperPortfolio, PaperPortfolioIdentity

from northstar_application.application_services.run_paper_trading_decision import (
    PaperTradingResult,
)


def _decision_order_key(result: PaperTradingResult) -> tuple[str, str, str]:
    """Return the tie-break key applied within one decision instant."""
    recommendation = result.result.recommendation
    listing_reference = recommendation.asset_analysis.listing_reference
    return (
        listing_reference.symbol.value,
        listing_reference.exchange_code.value,
        recommendation.strategy_identity.identity,
    )


def _decision_boundary(result: PaperTradingResult) -> tuple[str, str, str, str]:
    """Return the identity of the decision one result records.

    The portfolio is excluded because every result in a run already belongs to
    the same one. Side and quantity are excluded because they describe what was
    decided, not which decision it was.
    """
    symbol, exchange_code, strategy = _decision_order_key(result)
    return (symbol, exchange_code, strategy, result.result.recommendation.point_in_time.value)


@dataclass(frozen=True, slots=True)
class PaperTradingRun:
    """Every decision one paper portfolio made, in decision order.

    Results are ordered by decision instant, then listing, then strategy.
    That order is required of the caller rather than imposed here: silently
    sorting would let a caller's sequencing mistake pass unnoticed, and the
    producer is the only party that knows how the sequence was assembled.

    An empty run is valid: a portfolio that has been asked nothing has made no
    decisions, which is different from having decided to do nothing.
    """

    portfolio_identity: PaperPortfolioIdentity
    results: tuple[PaperTradingResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.portfolio_identity, PaperPortfolioIdentity):
            raise TypeError("PaperTradingRun portfolio identity must be a PaperPortfolioIdentity.")
        if not isinstance(self.results, tuple):
            raise TypeError("PaperTradingRun results must be a tuple.")
        if not all(isinstance(result, PaperTradingResult) for result in self.results):
            raise TypeError("PaperTradingRun results must contain PaperTradingResult values.")

        self._validate_results()

    def _validate_results(self) -> None:
        seen: set[tuple[str, str, str, str]] = set()
        previous: PaperTradingResult | None = None
        for result in self.results:
            if result.portfolio_identity != self.portfolio_identity:
                raise ValueError(
                    "PaperTradingRun results must all belong to the run's paper portfolio."
                )

            boundary = _decision_boundary(result)
            if boundary in seen:
                raise ValueError(
                    "PaperTradingRun cannot contain two results for one decision boundary."
                )
            seen.add(boundary)

            if previous is not None:
                instant = previous.result.recommendation.point_in_time.compare(
                    result.result.recommendation.point_in_time
                )
                if instant > 0 or (
                    instant == 0 and _decision_order_key(previous) > _decision_order_key(result)
                ):
                    raise ValueError(
                        "PaperTradingRun results must be ordered by decision instant, "
                        "then listing, then strategy."
                    )
            previous = result

    @property
    def decision_count(self) -> int:
        """Return how many decisions the run records."""
        return len(self.results)

    @property
    def first_decision_instant(self) -> PointInTime | None:
        """Return the earliest decision instant, or None when the run is empty."""
        if not self.results:
            return None
        return self.results[0].result.recommendation.point_in_time

    @property
    def last_decision_instant(self) -> PointInTime | None:
        """Return the latest decision instant, or None when the run is empty."""
        if not self.results:
            return None
        return self.results[-1].result.recommendation.point_in_time

    @property
    def final_portfolio(self) -> PaperPortfolio | None:
        """Return the holdings after the last decision, or None when empty."""
        if not self.results:
            return None
        return self.results[-1].portfolio
