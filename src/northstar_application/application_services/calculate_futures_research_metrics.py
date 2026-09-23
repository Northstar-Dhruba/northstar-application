"""Application calculation of factual futures historical research metrics.

Metrics summarise recorded forward market movement only. They apply no BUY,
SELL or HOLD interpretation, and express no accuracy, win rate, success or
failure, sign inversion, profit and loss, or execution meaning: a forward
return is what the market did after a decision, whichever way it leaned.

Grouping is by strategy and horizon, as in forward research, although a
historical run holds one strategy. Keeping the strategy in the key means a
summary of one strategy's movement is labelled as such and can never be pooled
with another's by a later aggregation that forgot to check.

Every requested horizon is summarised, including one at which no decision was
taken, so an empty run still reports each horizon it was asked about.

Unmeasured outcomes are counted, never summarised
-------------------------------------------------
A futures outcome can be unmeasured for two different reasons, and both are
counted separately: the horizon had not been reached by the run's cutoff, or
the decision quote was zero or negative so a percentage return is undefined.
Neither contributes a value to the statistics -- not a zero, not anything -- and
when nothing at a horizon was measured, every statistic is None.

Averages and medians use the shared return statistics under their explicit
Decimal context, the same definition the equity research metrics use.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.foundation.value_objects import Percentage
from northstar_core.strategy import (
    FuturesRecommendationOutcomeUnavailableReason,
    ResearchHorizon,
    StrategyIdentity,
)

from northstar_application.application_services._return_statistics import average, median
from northstar_application.application_services.run_futures_historical_research import (
    FuturesHistoricalResearchRun,
)

_INSUFFICIENT = FuturesRecommendationOutcomeUnavailableReason.INSUFFICIENT_FUTURE_OBSERVATIONS
_UNDEFINED_BASIS = FuturesRecommendationOutcomeUnavailableReason.UNDEFINED_RETURN_BASIS

_COUNT_FIELDS = (
    "total_count",
    "measured_count",
    "insufficient_future_observations_count",
    "undefined_return_basis_count",
)
_STATISTIC_FIELDS = (
    "average_forward_return",
    "median_forward_return",
    "minimum_forward_return",
    "maximum_forward_return",
)


@dataclass(frozen=True, slots=True)
class FuturesHistoricalResearchStrategyHorizonMetrics:
    """Factual summary of one strategy's recorded futures movement at one horizon.

    ``total_count`` is every outcome at this strategy and horizon, and it is
    exactly the sum of the measured, insufficient and undefined-basis counts.

    Return statistics summarise only measured outcomes. Returns below -100% are
    ordinary: a positive decision quote can be followed by a negative one.
    """

    strategy_identity: StrategyIdentity
    horizon: ResearchHorizon
    total_count: int
    measured_count: int
    insufficient_future_observations_count: int
    undefined_return_basis_count: int
    average_forward_return: Percentage | None
    median_forward_return: Percentage | None
    minimum_forward_return: Percentage | None
    maximum_forward_return: Percentage | None

    def __post_init__(self) -> None:
        subject = "FuturesHistoricalResearchStrategyHorizonMetrics"
        if not isinstance(self.strategy_identity, StrategyIdentity):
            raise TypeError(f"{subject} strategy identity must be a StrategyIdentity.")
        if not isinstance(self.horizon, ResearchHorizon):
            raise TypeError(f"{subject} horizon must be a ResearchHorizon.")
        for name in _COUNT_FIELDS:
            count = getattr(self, name)
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(f"{subject} {name} must be an integer.")
            if count < 0:
                raise ValueError(f"{subject} {name} cannot be negative.")
        for name in _STATISTIC_FIELDS:
            statistic = getattr(self, name)
            if statistic is not None and not isinstance(statistic, Percentage):
                raise TypeError(f"{subject} {name} must be a Percentage or None.")

        partition = (
            self.measured_count
            + self.insufficient_future_observations_count
            + self.undefined_return_basis_count
        )
        if partition != self.total_count:
            raise ValueError(
                f"{subject} measured and unavailable counts must equal the total count."
            )
        statistics = tuple(getattr(self, name) for name in _STATISTIC_FIELDS)
        if self.measured_count == 0 and any(statistic is not None for statistic in statistics):
            raise ValueError(
                f"{subject} return statistics must be None when no outcome was measured."
            )
        if self.measured_count > 0 and any(statistic is None for statistic in statistics):
            raise ValueError(
                f"{subject} return statistics are required when an outcome was measured."
            )


class CalculateFuturesHistoricalResearchMetricsUseCase:
    """Summarise one futures historical research run per strategy and horizon."""

    def execute(
        self, run: FuturesHistoricalResearchRun
    ) -> tuple[FuturesHistoricalResearchStrategyHorizonMetrics, ...]:
        """Return one metrics value per strategy and horizon, in run horizon order."""
        if not isinstance(run, FuturesHistoricalResearchRun):
            raise TypeError(
                "CalculateFuturesHistoricalResearchMetricsUseCase run must be a "
                "FuturesHistoricalResearchRun."
            )

        return tuple(
            self._strategy_horizon_metrics(run, run.strategy_identity, horizon)
            for horizon in run.horizons
        )

    @staticmethod
    def _strategy_horizon_metrics(
        run: FuturesHistoricalResearchRun,
        strategy_identity: StrategyIdentity,
        horizon: ResearchHorizon,
    ) -> FuturesHistoricalResearchStrategyHorizonMetrics:
        outcomes = tuple(
            outcome
            for outcome in run.outcomes
            if outcome.recommendation.strategy_identity == strategy_identity
            and outcome.horizon == horizon
        )
        returns = tuple(
            outcome.forward_return for outcome in outcomes if outcome.forward_return is not None
        )
        ordered = tuple(sorted(returns))

        return FuturesHistoricalResearchStrategyHorizonMetrics(
            strategy_identity=strategy_identity,
            horizon=horizon,
            total_count=len(outcomes),
            measured_count=len(returns),
            insufficient_future_observations_count=sum(
                1 for outcome in outcomes if outcome.unavailable_reason is _INSUFFICIENT
            ),
            undefined_return_basis_count=sum(
                1 for outcome in outcomes if outcome.unavailable_reason is _UNDEFINED_BASIS
            ),
            average_forward_return=average(ordered),
            median_forward_return=median(ordered),
            minimum_forward_return=ordered[0] if ordered else None,
            maximum_forward_return=ordered[-1] if ordered else None,
        )
