"""Application calculation of factual forward research metrics per strategy.

Metrics summarise recorded forward market movement only. They apply no BUY,
SELL or HOLD interpretation, and express no accuracy, win rate, success or
failure, sign inversion, profit and loss, or execution meaning.

Unlike a historical run, a forward run is not homogeneous by strategy: one
observation series can carry decisions frozen by several strategies at the same
instants. Metrics are therefore grouped by strategy as well as by horizon, so
one strategy's recorded movement is never pooled with another's.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.foundation.value_objects import Percentage
from northstar_core.strategy import ResearchHorizon, StrategyIdentity

from northstar_application.application_services._return_statistics import average, median
from northstar_application.application_services.measure_forward_research_record import (
    ForwardResearchMeasurementState,
)
from northstar_application.application_services.run_forward_research import ForwardResearchRun


def distinct_strategies(run: ForwardResearchRun) -> tuple[StrategyIdentity, ...]:
    """Return the run's distinct strategies in canonical grouping order.

    Ordering is ascending by identity rather than by first appearance, so the
    grouping is a total order that does not depend on which strategies happened
    to decide at the run's earliest instant. Metrics calculation and report
    coherence share this one definition so they can never disagree.
    """
    identities = {
        record.strategy_identity.identity: record.strategy_identity for record in run.records
    }
    return tuple(identities[identity] for identity in sorted(identities))


@dataclass(frozen=True, slots=True)
class ForwardResearchStrategyHorizonMetrics:
    """Factual summary of one strategy's recorded movement at one horizon.

    Counts describe every measurement attempted for this strategy and horizon.
    ``pending_count`` and ``unavailable_count`` both describe unmeasured
    decisions but differ in expectation: a pending decision is expected to
    become measurable as further observations arrive, whereas an unavailable one
    will not. A forward summary is therefore provisional by nature, and
    ``pending_count`` is what tells a reader how much of the horizon has yet to
    mature.

    Return statistics summarise only the measurements that produced a factual
    outcome; when none did, every statistic is ``None`` rather than zero, so an
    absence of evidence is never reported as a neutral movement.
    """

    strategy_identity: StrategyIdentity
    horizon: ResearchHorizon
    total_count: int
    measured_count: int
    pending_count: int
    unavailable_count: int
    average_forward_return: Percentage | None
    median_forward_return: Percentage | None
    minimum_forward_return: Percentage | None
    maximum_forward_return: Percentage | None

    def __post_init__(self) -> None:
        if not isinstance(self.strategy_identity, StrategyIdentity):
            raise TypeError(
                "ForwardResearchStrategyHorizonMetrics strategy identity must be a "
                "StrategyIdentity."
            )
        if not isinstance(self.horizon, ResearchHorizon):
            raise TypeError(
                "ForwardResearchStrategyHorizonMetrics horizon must be a ResearchHorizon."
            )
        for name in ("total_count", "measured_count", "pending_count", "unavailable_count"):
            count = getattr(self, name)
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(f"ForwardResearchStrategyHorizonMetrics {name} must be an integer.")
            if count < 0:
                raise ValueError(
                    f"ForwardResearchStrategyHorizonMetrics {name} cannot be negative."
                )
        for name in (
            "average_forward_return",
            "median_forward_return",
            "minimum_forward_return",
            "maximum_forward_return",
        ):
            statistic = getattr(self, name)
            if statistic is not None and not isinstance(statistic, Percentage):
                raise TypeError(
                    f"ForwardResearchStrategyHorizonMetrics {name} must be a Percentage or None."
                )

        partition = self.measured_count + self.pending_count + self.unavailable_count
        if partition != self.total_count:
            raise ValueError(
                "ForwardResearchStrategyHorizonMetrics measured, pending and unavailable "
                "counts must equal the total count."
            )
        statistics = (
            self.average_forward_return,
            self.median_forward_return,
            self.minimum_forward_return,
            self.maximum_forward_return,
        )
        if self.measured_count == 0 and any(statistic is not None for statistic in statistics):
            raise ValueError(
                "ForwardResearchStrategyHorizonMetrics return statistics must be None when "
                "no outcome was measured."
            )
        if self.measured_count > 0 and any(statistic is None for statistic in statistics):
            raise ValueError(
                "ForwardResearchStrategyHorizonMetrics return statistics are required when "
                "an outcome was measured."
            )


class CalculateForwardResearchMetricsUseCase:
    """Summarise one forward research run into per-strategy, per-horizon metrics."""

    def execute(self, run: ForwardResearchRun) -> tuple[ForwardResearchStrategyHorizonMetrics, ...]:
        """Return one metrics value per run strategy and horizon pair.

        Entries are strategy-major and horizon-minor: every strategy's entries
        appear together, in run horizon order. A run holding no frozen decision
        yields no metrics, because there is no strategy to summarise.
        """
        if run is None:
            raise TypeError("CalculateForwardResearchMetricsUseCase run cannot be None.")
        if not isinstance(run, ForwardResearchRun):
            raise TypeError(
                "CalculateForwardResearchMetricsUseCase run must be a ForwardResearchRun."
            )

        return tuple(
            self._strategy_horizon_metrics(run, strategy_identity, horizon)
            for strategy_identity in distinct_strategies(run)
            for horizon in run.horizons
        )

    @staticmethod
    def _strategy_horizon_metrics(
        run: ForwardResearchRun,
        strategy_identity: StrategyIdentity,
        horizon: ResearchHorizon,
    ) -> ForwardResearchStrategyHorizonMetrics:
        measurements = tuple(
            measurement
            for measurement in run.measurements
            if measurement.horizon == horizon
            and measurement.record.strategy_identity == strategy_identity
        )
        states = tuple(measurement.state for measurement in measurements)
        returns = tuple(
            measurement.measurement.outcome.forward_return
            for measurement, state in zip(measurements, states, strict=True)
            if state is ForwardResearchMeasurementState.MEASURED
        )
        pending = sum(1 for state in states if state is ForwardResearchMeasurementState.PENDING)
        unavailable = sum(
            1 for state in states if state is ForwardResearchMeasurementState.UNAVAILABLE
        )

        ordered = tuple(sorted(returns))
        return ForwardResearchStrategyHorizonMetrics(
            strategy_identity=strategy_identity,
            horizon=horizon,
            total_count=len(measurements),
            measured_count=len(returns),
            pending_count=pending,
            unavailable_count=unavailable,
            average_forward_return=average(ordered),
            median_forward_return=median(ordered),
            minimum_forward_return=ordered[0] if ordered else None,
            maximum_forward_return=ordered[-1] if ordered else None,
        )
