"""Application assembly of one structured futures historical research report.

A report pairs one frozen research run with its metrics and exposes the facts a
reader needs to know what was researched and up to when. It is derived from the
run alone: it reads no repository, replays nothing, measures nothing and
consults no clock. It performs no rendering, serialisation or persistence,
introduces no metric of its own, and applies no BUY, SELL or HOLD success
interpretation, profit and loss, or execution meaning.

The run is referenced rather than copied, as in the equity report. Unlike the
equity report, identity never needs to be inferred from the first decision:
the futures run records its contract, strategy and cutoff itself, so even a
run with no decisions reports exactly what it covered.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.futures import FuturesContract
from northstar_core.strategy import ResearchHorizon, StrategyIdentity

from northstar_application.application_services.calculate_futures_research_metrics import (
    CalculateFuturesHistoricalResearchMetricsUseCase,
    FuturesHistoricalResearchStrategyHorizonMetrics,
)
from northstar_application.application_services.run_futures_historical_research import (
    FuturesHistoricalResearchRun,
)


@dataclass(frozen=True, slots=True)
class FuturesHistoricalResearchReport:
    """One frozen futures research run paired with its strategy-horizon metrics.

    ``metrics`` holds exactly one entry per run horizon, at the same position,
    for the run's strategy, so ``run.horizons[i]`` pairs with ``metrics[i]``
    without searching and in the caller's horizon order.
    """

    run: FuturesHistoricalResearchRun
    metrics: tuple[FuturesHistoricalResearchStrategyHorizonMetrics, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run, FuturesHistoricalResearchRun):
            raise TypeError(
                "FuturesHistoricalResearchReport run must be a FuturesHistoricalResearchRun."
            )
        if not isinstance(self.metrics, tuple):
            raise TypeError("FuturesHistoricalResearchReport metrics must be a tuple.")
        if not all(
            isinstance(entry, FuturesHistoricalResearchStrategyHorizonMetrics)
            for entry in self.metrics
        ):
            raise TypeError(
                "FuturesHistoricalResearchReport metrics must contain "
                "FuturesHistoricalResearchStrategyHorizonMetrics values."
            )
        if len(self.metrics) != len(self.run.horizons):
            raise ValueError(
                "FuturesHistoricalResearchReport must contain one metrics entry for every "
                "run horizon."
            )
        for entry, horizon in zip(self.metrics, self.run.horizons, strict=True):
            if entry.strategy_identity != self.run.strategy_identity:
                raise ValueError(
                    "FuturesHistoricalResearchReport metrics must describe the run strategy."
                )
            if entry.horizon != horizon:
                raise ValueError(
                    "FuturesHistoricalResearchReport metrics must follow the run horizon order."
                )
        if sum(entry.total_count for entry in self.metrics) != len(self.run.outcomes):
            raise ValueError(
                "FuturesHistoricalResearchReport metrics must account for every run outcome."
            )

    @property
    def contract(self) -> FuturesContract:
        """Return the one concrete contract the run researched."""
        return self.run.contract

    @property
    def strategy_identity(self) -> StrategyIdentity:
        """Return the strategy whose decisions the run recorded."""
        return self.run.strategy_identity

    @property
    def timeframe(self) -> Timeframe:
        """Return the observation interval a research horizon counts."""
        return self.run.timeframe

    @property
    def available_through(self) -> PointInTime:
        """Return the run's evidence cutoff, exactly as the run recorded it."""
        return self.run.available_through

    @property
    def horizons(self) -> tuple[ResearchHorizon, ...]:
        """Return the requested horizons in the caller's order."""
        return self.run.horizons

    @property
    def first_decision_instant(self) -> PointInTime | None:
        """Return the earliest decision instant, or None when the run took none."""
        if not self.run.analysis_results:
            return None
        return self.run.analysis_results[0].recommendation.point_in_time

    @property
    def last_decision_instant(self) -> PointInTime | None:
        """Return the latest decision instant, or None when the run took none."""
        if not self.run.analysis_results:
            return None
        return self.run.analysis_results[-1].recommendation.point_in_time

    @property
    def decision_count(self) -> int:
        """Return how many decisions the run recorded."""
        return len(self.run.analysis_results)


class BuildFuturesHistoricalResearchReportUseCase:
    """Assemble one structured report from a frozen futures research run.

    Metrics are produced by CalculateFuturesHistoricalResearchMetricsUseCase so
    metric semantics remain defined in exactly one place.
    """

    def __init__(
        self,
        calculate_metrics: CalculateFuturesHistoricalResearchMetricsUseCase | None = None,
    ) -> None:
        if calculate_metrics is None:
            calculate_metrics = CalculateFuturesHistoricalResearchMetricsUseCase()
        if not isinstance(calculate_metrics, CalculateFuturesHistoricalResearchMetricsUseCase):
            raise TypeError(
                "BuildFuturesHistoricalResearchReportUseCase calculate_metrics must be a "
                "CalculateFuturesHistoricalResearchMetricsUseCase."
            )
        self._calculate_metrics = calculate_metrics

    def execute(self, run: FuturesHistoricalResearchRun) -> FuturesHistoricalResearchReport:
        """Return the structured report for one frozen futures research run."""
        if not isinstance(run, FuturesHistoricalResearchRun):
            raise TypeError(
                "BuildFuturesHistoricalResearchReportUseCase run must be a "
                "FuturesHistoricalResearchRun."
            )

        return FuturesHistoricalResearchReport(
            run=run, metrics=self._calculate_metrics.execute(run)
        )
