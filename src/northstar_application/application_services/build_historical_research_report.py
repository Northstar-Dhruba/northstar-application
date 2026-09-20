"""Application assembly of one structured historical research report.

A report is a structured Application result: it pairs one measured run with its
per-horizon metrics and exposes the metadata a reader needs. It performs no
rendering, serialization or persistence, introduces no new metric, and applies
no BUY, SELL or HOLD success interpretation, profit and loss, or execution
meaning.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services.calculate_historical_research_metrics import (
    CalculateHistoricalResearchMetricsUseCase,
    HistoricalResearchHorizonMetrics,
)
from northstar_application.application_services.run_historical_research import (
    HistoricalResearchRun,
)


@dataclass(frozen=True, slots=True)
class HistoricalResearchReport:
    """One measured research run paired with its per-horizon metrics.

    ``metrics`` holds exactly one entry per run horizon, at the same position,
    so a reader can pair ``run.horizons[i]`` with ``metrics[i]`` without
    searching. Run configuration and identity are not duplicated here; they
    remain owned by the run and its evaluations, and are surfaced through
    read-only derived properties.
    """

    run: HistoricalResearchRun
    metrics: tuple[HistoricalResearchHorizonMetrics, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run, HistoricalResearchRun):
            raise TypeError("HistoricalResearchReport run must be a HistoricalResearchRun.")
        if not isinstance(self.metrics, tuple):
            raise TypeError("HistoricalResearchReport metrics must be a tuple.")
        if not all(isinstance(entry, HistoricalResearchHorizonMetrics) for entry in self.metrics):
            raise TypeError(
                "HistoricalResearchReport metrics must contain "
                "HistoricalResearchHorizonMetrics values."
            )
        if len(self.metrics) != len(self.run.horizons):
            raise ValueError(
                "HistoricalResearchReport must contain one metrics entry for every run horizon."
            )
        for entry, horizon in zip(self.metrics, self.run.horizons, strict=True):
            if entry.horizon != horizon:
                raise ValueError(
                    "HistoricalResearchReport metrics must follow the run horizon order."
                )

    @property
    def listing_reference(self) -> ListingReference | None:
        """Return the researched listing, or None when the run has no evaluations."""
        if not self.run.evaluations:
            return None
        recommendation = self.run.evaluations[0].result.recommendation
        return recommendation.asset_analysis.listing_reference

    @property
    def strategy_identity(self) -> StrategyIdentity | None:
        """Return the researched strategy, or None when the run has no evaluations."""
        if not self.run.evaluations:
            return None
        return self.run.evaluations[0].result.recommendation.strategy_identity

    @property
    def timeframe(self) -> Timeframe:
        """Return the observation interval a research horizon counts."""
        return self.run.timeframe

    @property
    def available_through(self) -> PointInTime:
        """Return the boundary beyond which no observation was available."""
        return self.run.available_through

    @property
    def first_replay_instant(self) -> PointInTime | None:
        """Return the earliest replay instant, or None when the run is empty."""
        if not self.run.evaluations:
            return None
        return self.run.evaluations[0].replay_instant

    @property
    def last_replay_instant(self) -> PointInTime | None:
        """Return the latest replay instant, or None when the run is empty."""
        if not self.run.evaluations:
            return None
        return self.run.evaluations[-1].replay_instant

    @property
    def evaluation_count(self) -> int:
        """Return how many evaluated decisions the run measured."""
        return len(self.run.evaluations)


class BuildHistoricalResearchReportUseCase:
    """Assemble one structured report from a measured historical research run.

    Metrics are produced by CalculateHistoricalResearchMetricsUseCase so metric
    semantics remain defined in exactly one place.
    """

    def __init__(
        self,
        calculate_metrics: CalculateHistoricalResearchMetricsUseCase | None = None,
    ) -> None:
        if calculate_metrics is None:
            calculate_metrics = CalculateHistoricalResearchMetricsUseCase()
        if not isinstance(calculate_metrics, CalculateHistoricalResearchMetricsUseCase):
            raise TypeError(
                "BuildHistoricalResearchReportUseCase calculate_metrics must be a "
                "CalculateHistoricalResearchMetricsUseCase."
            )
        self._calculate_metrics = calculate_metrics

    def execute(self, run: HistoricalResearchRun) -> HistoricalResearchReport:
        """Return the structured report for one measured research run."""
        if run is None:
            raise TypeError("BuildHistoricalResearchReportUseCase run cannot be None.")
        if not isinstance(run, HistoricalResearchRun):
            raise TypeError(
                "BuildHistoricalResearchReportUseCase run must be a HistoricalResearchRun."
            )

        return HistoricalResearchReport(run=run, metrics=self._calculate_metrics.execute(run))
