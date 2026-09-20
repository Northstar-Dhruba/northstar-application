"""Application calculation of factual per-horizon historical research metrics.

Metrics summarise recorded forward market movement only. They apply no BUY,
SELL or HOLD interpretation, and express no accuracy, win rate, success or
failure, sign inversion, profit and loss, or execution meaning.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.foundation.value_objects import Percentage
from northstar_core.strategy import ResearchHorizon

from northstar_application.application_services._return_statistics import average, median
from northstar_application.application_services.measure_recommendation_outcome import (
    RecommendationOutcomeUnavailableReason,
)
from northstar_application.application_services.run_historical_research import (
    HistoricalResearchRun,
)


@dataclass(frozen=True, slots=True)
class HistoricalResearchHorizonMetrics:
    """Factual summary of recorded forward movement at one research horizon.

    Counts describe every measurement attempted at this horizon. Return
    statistics summarise only the measurements that produced a factual outcome;
    when none did, every statistic is ``None`` rather than zero, so an absence
    of evidence is never reported as a neutral movement.
    """

    horizon: ResearchHorizon
    total_count: int
    measured_count: int
    insufficient_future_observations_count: int
    currency_mismatch_count: int
    average_forward_return: Percentage | None
    median_forward_return: Percentage | None
    minimum_forward_return: Percentage | None
    maximum_forward_return: Percentage | None

    def __post_init__(self) -> None:
        if not isinstance(self.horizon, ResearchHorizon):
            raise TypeError("HistoricalResearchHorizonMetrics horizon must be a ResearchHorizon.")
        for name in (
            "total_count",
            "measured_count",
            "insufficient_future_observations_count",
            "currency_mismatch_count",
        ):
            count = getattr(self, name)
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError(f"HistoricalResearchHorizonMetrics {name} must be an integer.")
            if count < 0:
                raise ValueError(f"HistoricalResearchHorizonMetrics {name} cannot be negative.")
        for name in (
            "average_forward_return",
            "median_forward_return",
            "minimum_forward_return",
            "maximum_forward_return",
        ):
            statistic = getattr(self, name)
            if statistic is not None and not isinstance(statistic, Percentage):
                raise TypeError(
                    f"HistoricalResearchHorizonMetrics {name} must be a Percentage or None."
                )

        partition = (
            self.measured_count
            + self.insufficient_future_observations_count
            + self.currency_mismatch_count
        )
        if partition != self.total_count:
            raise ValueError(
                "HistoricalResearchHorizonMetrics measured and unavailable counts must "
                "equal the total count."
            )
        statistics = (
            self.average_forward_return,
            self.median_forward_return,
            self.minimum_forward_return,
            self.maximum_forward_return,
        )
        if self.measured_count == 0 and any(statistic is not None for statistic in statistics):
            raise ValueError(
                "HistoricalResearchHorizonMetrics return statistics must be None when "
                "no outcome was measured."
            )
        if self.measured_count > 0 and any(statistic is None for statistic in statistics):
            raise ValueError(
                "HistoricalResearchHorizonMetrics return statistics are required when "
                "an outcome was measured."
            )


class CalculateHistoricalResearchMetricsUseCase:
    """Summarise one historical research run into factual per-horizon metrics."""

    def execute(self, run: HistoricalResearchRun) -> tuple[HistoricalResearchHorizonMetrics, ...]:
        """Return one metrics value per run horizon, in run horizon order."""
        if run is None:
            raise TypeError("CalculateHistoricalResearchMetricsUseCase run cannot be None.")
        if not isinstance(run, HistoricalResearchRun):
            raise TypeError(
                "CalculateHistoricalResearchMetricsUseCase run must be a HistoricalResearchRun."
            )

        return tuple(self._horizon_metrics(run, horizon) for horizon in run.horizons)

    @staticmethod
    def _horizon_metrics(
        run: HistoricalResearchRun, horizon: ResearchHorizon
    ) -> HistoricalResearchHorizonMetrics:
        measurements = tuple(
            measurement for measurement in run.measurements if measurement.horizon == horizon
        )
        returns = tuple(
            measurement.outcome.forward_return
            for measurement in measurements
            if measurement.outcome is not None
        )
        insufficient = sum(
            1
            for measurement in measurements
            if measurement.unavailable
            is RecommendationOutcomeUnavailableReason.INSUFFICIENT_FUTURE_OBSERVATIONS
        )
        currency_mismatch = sum(
            1
            for measurement in measurements
            if measurement.unavailable is RecommendationOutcomeUnavailableReason.CURRENCY_MISMATCH
        )

        ordered = tuple(sorted(returns))
        return HistoricalResearchHorizonMetrics(
            horizon=horizon,
            total_count=len(measurements),
            measured_count=len(returns),
            insufficient_future_observations_count=insufficient,
            currency_mismatch_count=currency_mismatch,
            average_forward_return=average(ordered),
            median_forward_return=median(ordered),
            minimum_forward_return=ordered[0] if ordered else None,
            maximum_forward_return=ordered[-1] if ordered else None,
        )
