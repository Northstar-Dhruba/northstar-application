"""Application assembly of one structured futures forward research report.

A report pairs one forward run with its per-strategy, per-horizon metrics and
exposes the facts a reader needs. It is derived from the run alone: it reads no
repository, measures nothing, freezes nothing and consults no clock. It performs
no rendering, serialisation or persistence, introduces no metric of its own, and
applies no BUY, SELL or HOLD success interpretation, profit and loss, or
execution meaning.

A forward report is a snapshot as of one evidence cutoff, not a settled result.
The same frozen decisions reported at a later cutoff can read differently --
pending measurements become measured or unavailable -- and that is expected.
Nothing here is stored.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.futures import FuturesContract
from northstar_core.strategy import ResearchHorizon, StrategyIdentity

from northstar_application.application_services.calculate_futures_forward_metrics import (
    CalculateFuturesForwardResearchMetricsUseCase,
    FuturesForwardResearchStrategyHorizonMetrics,
    _distinct_strategies,
)
from northstar_application.application_services.measure_forward_research_record import (
    ForwardResearchMeasurementState,
)
from northstar_application.application_services.run_futures_forward_research import (
    FuturesForwardResearchRun,
)


@dataclass(frozen=True, slots=True)
class FuturesForwardResearchReport:
    """One futures forward run paired with its per-strategy, per-horizon metrics.

    ``metrics`` holds exactly one entry per run strategy and horizon pair,
    strategy-major and horizon-minor, so a reader can walk the entries in a known
    order without searching. A forward run may carry several strategies, so no
    single report-level strategy exists; the strategies are surfaced as a tuple
    and each metrics entry names its own.

    Configuration and identity are not duplicated; they remain owned by the run
    and its records and are surfaced through read-only properties.
    """

    run: FuturesForwardResearchRun
    metrics: tuple[FuturesForwardResearchStrategyHorizonMetrics, ...]

    def __post_init__(self) -> None:
        subject = "FuturesForwardResearchReport"
        if not isinstance(self.run, FuturesForwardResearchRun):
            raise TypeError(f"{subject} run must be a FuturesForwardResearchRun.")
        if not isinstance(self.metrics, tuple):
            raise TypeError(f"{subject} metrics must be a tuple.")
        if not all(
            isinstance(entry, FuturesForwardResearchStrategyHorizonMetrics)
            for entry in self.metrics
        ):
            raise TypeError(
                f"{subject} metrics must contain FuturesForwardResearchStrategyHorizonMetrics "
                "values."
            )

        strategies = _distinct_strategies(self.run)
        if len(self.metrics) != len(strategies) * len(self.run.horizons):
            raise ValueError(
                f"{subject} must contain one metrics entry for every run strategy and horizon pair."
            )
        expected_pairs = (
            (strategy_identity, horizon)
            for strategy_identity in strategies
            for horizon in self.run.horizons
        )
        for entry, (strategy_identity, horizon) in zip(self.metrics, expected_pairs, strict=True):
            if entry.strategy_identity != strategy_identity:
                raise ValueError(
                    f"{subject} metrics must follow ascending strategy order, grouping each "
                    "strategy's horizons together."
                )
            if entry.horizon != horizon:
                raise ValueError(
                    f"{subject} metrics must follow the run horizon order within each strategy."
                )
            attempted = sum(
                1
                for measurement in self.run.measurements
                if measurement.horizon == horizon
                and measurement.record.strategy_identity == strategy_identity
            )
            if entry.total_count != attempted:
                raise ValueError(
                    f"{subject} metrics for {strategy_identity} at horizon {horizon} count "
                    f"{entry.total_count} measurements; the run holds {attempted}."
                )

    @property
    def contract(self) -> FuturesContract:
        """Return the concrete contract the run was queried for."""
        return self.run.query.contract

    @property
    def timeframe(self) -> Timeframe:
        """Return the observation interval a research horizon counts."""
        return self.run.query.timeframe

    @property
    def available_through(self) -> PointInTime:
        """Return the run's evidence cutoff, exactly as the run recorded it."""
        return self.run.available_through

    @property
    def horizons(self) -> tuple[ResearchHorizon, ...]:
        """Return the requested horizons in the caller's order."""
        return self.run.horizons

    @property
    def strategy_identities(self) -> tuple[StrategyIdentity, ...]:
        """Return the distinct monitored strategies, ascending by identity."""
        return _distinct_strategies(self.run)

    @property
    def record_count(self) -> int:
        """Return how many frozen decisions the run measured."""
        return len(self.run.records)

    @property
    def pending_count(self) -> int:
        """Return how many measurement attempts await further observations."""
        return sum(
            1
            for measurement in self.run.measurements
            if measurement.state is ForwardResearchMeasurementState.PENDING
        )

    @property
    def first_decision_instant(self) -> PointInTime | None:
        """Return the earliest frozen decision instant, or None when empty."""
        if not self.run.records:
            return None
        return self.run.records[0].decision_instant

    @property
    def last_decision_instant(self) -> PointInTime | None:
        """Return the latest frozen decision instant, or None when empty."""
        if not self.run.records:
            return None
        return self.run.records[-1].decision_instant


class BuildFuturesForwardResearchReportUseCase:
    """Assemble one structured report from a futures forward research run.

    Metrics are produced by CalculateFuturesForwardResearchMetricsUseCase so
    metric semantics and strategy grouping remain defined in exactly one place.
    """

    def __init__(
        self,
        calculate_metrics: CalculateFuturesForwardResearchMetricsUseCase | None = None,
    ) -> None:
        if calculate_metrics is None:
            calculate_metrics = CalculateFuturesForwardResearchMetricsUseCase()
        if not isinstance(calculate_metrics, CalculateFuturesForwardResearchMetricsUseCase):
            raise TypeError(
                "BuildFuturesForwardResearchReportUseCase calculate_metrics must be a "
                "CalculateFuturesForwardResearchMetricsUseCase."
            )
        self._calculate_metrics = calculate_metrics

    def execute(self, run: FuturesForwardResearchRun) -> FuturesForwardResearchReport:
        """Return the structured report for one futures forward research run."""
        if not isinstance(run, FuturesForwardResearchRun):
            raise TypeError(
                "BuildFuturesForwardResearchReportUseCase run must be a FuturesForwardResearchRun."
            )

        return FuturesForwardResearchReport(run=run, metrics=self._calculate_metrics.execute(run))
