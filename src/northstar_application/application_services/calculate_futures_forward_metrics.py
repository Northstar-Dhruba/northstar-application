"""Application calculation of factual futures forward research metrics.

Metrics summarise recorded forward market movement only. They apply no BUY,
SELL or HOLD interpretation, and express no accuracy, win rate, success or
failure, sign inversion, profit and loss, or execution meaning.

A forward run can hold decisions from several strategies, so metrics are
grouped by strategy and horizon and one strategy's movement is never pooled
with another's. Strategies are ordered ascending by identity -- the equity
forward-research rule -- and each strategy's horizons follow the run's own
horizon order. A strategy appears only if the run holds a decision of it, so a
run with no frozen decision yields no metrics: there is no strategy to report
on, and none is invented.

A forward summary is provisional
--------------------------------
PENDING and UNAVAILABLE are both unmeasured but differ in expectation. A
pending measurement is expected to become measurable as further sessions are
persisted; an unavailable one will not, because its decision quote was zero or
negative and a percentage return of it is undefined. The two are counted
separately, never pooled and never turned into a return. Only measured returns
are summarised, through the shared return statistics under their explicit
Decimal context, and when nothing was measured every statistic is None.
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
from northstar_application.application_services.measure_forward_research_record import (
    ForwardResearchMeasurementState,
)
from northstar_application.application_services.measure_futures_forward_research_record import (
    FuturesForwardResearchMeasurement,
)
from northstar_application.application_services.run_futures_forward_research import (
    FuturesForwardResearchRun,
)

_UNDEFINED_BASIS = FuturesRecommendationOutcomeUnavailableReason.UNDEFINED_RETURN_BASIS

_COUNT_FIELDS = (
    "total_count",
    "measured_count",
    "pending_count",
    "undefined_return_basis_count",
)
_STATISTIC_FIELDS = (
    "average_forward_return",
    "median_forward_return",
    "minimum_forward_return",
    "maximum_forward_return",
)


def _distinct_strategies(run: FuturesForwardResearchRun) -> tuple[StrategyIdentity, ...]:
    """Return the run's distinct strategies ascending by identity.

    The equity forward-research ordering rule, so grouping is a total order that
    does not depend on which strategy happened to decide first. Metrics and
    report coherence share this one definition.
    """
    identities = {
        record.strategy_identity.identity: record.strategy_identity for record in run.records
    }
    return tuple(identities[identity] for identity in sorted(identities))


@dataclass(frozen=True, slots=True)
class FuturesForwardResearchStrategyHorizonMetrics:
    """Factual summary of one strategy's forward futures movement at one horizon.

    ``total_count`` is every measurement attempted for this strategy and horizon,
    and exactly the sum of the measured, pending and undefined-basis counts.
    Return statistics summarise measured outcomes only; returns below -100% are
    ordinary, since a positive decision quote can be followed by a negative one.
    """

    strategy_identity: StrategyIdentity
    horizon: ResearchHorizon
    total_count: int
    measured_count: int
    pending_count: int
    undefined_return_basis_count: int
    average_forward_return: Percentage | None
    median_forward_return: Percentage | None
    minimum_forward_return: Percentage | None
    maximum_forward_return: Percentage | None

    def __post_init__(self) -> None:
        subject = "FuturesForwardResearchStrategyHorizonMetrics"
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

        partition = self.measured_count + self.pending_count + self.undefined_return_basis_count
        if partition != self.total_count:
            raise ValueError(
                f"{subject} measured, pending and undefined-basis counts must equal the total "
                "count."
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


class CalculateFuturesForwardResearchMetricsUseCase:
    """Summarise one futures forward research run per strategy and horizon."""

    def execute(
        self, run: FuturesForwardResearchRun
    ) -> tuple[FuturesForwardResearchStrategyHorizonMetrics, ...]:
        """Return one metrics value per run strategy and horizon pair.

        Entries are strategy-major, strategies ascending by identity, and
        horizon-minor in run horizon order. A run with no frozen decision
        yields no metrics.
        """
        if not isinstance(run, FuturesForwardResearchRun):
            raise TypeError(
                "CalculateFuturesForwardResearchMetricsUseCase run must be a "
                "FuturesForwardResearchRun."
            )

        return tuple(
            self._strategy_horizon_metrics(run, strategy_identity, horizon)
            for strategy_identity in _distinct_strategies(run)
            for horizon in run.horizons
        )

    @staticmethod
    def _strategy_horizon_metrics(
        run: FuturesForwardResearchRun,
        strategy_identity: StrategyIdentity,
        horizon: ResearchHorizon,
    ) -> FuturesForwardResearchStrategyHorizonMetrics:
        measurements = tuple(
            measurement
            for measurement in run.measurements
            if measurement.horizon == horizon
            and measurement.record.strategy_identity == strategy_identity
        )
        returns: list[Percentage] = []
        pending = 0
        undefined_basis = 0
        for measurement in measurements:
            state = _state(measurement)
            if state is ForwardResearchMeasurementState.MEASURED:
                returns.append(measurement.outcome.forward_return)  # type: ignore[arg-type]
            elif state is ForwardResearchMeasurementState.PENDING:
                pending += 1
            else:
                undefined_basis += 1

        ordered = tuple(sorted(returns))
        return FuturesForwardResearchStrategyHorizonMetrics(
            strategy_identity=strategy_identity,
            horizon=horizon,
            total_count=len(measurements),
            measured_count=len(ordered),
            pending_count=pending,
            undefined_return_basis_count=undefined_basis,
            average_forward_return=average(ordered),
            median_forward_return=median(ordered),
            minimum_forward_return=ordered[0] if ordered else None,
            maximum_forward_return=ordered[-1] if ordered else None,
        )


def _state(measurement: FuturesForwardResearchMeasurement) -> ForwardResearchMeasurementState:
    """Return a measurement's state, refusing any combination metrics cannot count.

    UNAVAILABLE is counted as an undefined return basis, so it must be exactly
    that; a future unavailable reason must fail here rather than be miscounted.
    """
    state = measurement.state
    if (
        state is ForwardResearchMeasurementState.UNAVAILABLE
        and measurement.outcome.unavailable_reason is not _UNDEFINED_BASIS
    ):
        raise ValueError(
            "CalculateFuturesForwardResearchMetricsUseCase cannot count an unavailable "
            f"measurement with reason {measurement.outcome.unavailable_reason}."
        )
    return state
