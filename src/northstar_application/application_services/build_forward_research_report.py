"""Application assembly of one structured forward research report.

A report is a structured Application result: it pairs one measured forward run
with its per-strategy, per-horizon metrics and exposes the metadata a reader
needs. It performs no rendering, serialization or persistence, introduces no new
metric, and applies no BUY, SELL or HOLD success interpretation, profit and loss,
or execution meaning.

A forward report is a snapshot as of one data boundary, not a settled result:
pending measurements become measured as further observations arrive, so the same
run re-measured later will summarise differently. Nothing here is stored.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services.calculate_forward_research_metrics import (
    CalculateForwardResearchMetricsUseCase,
    ForwardResearchStrategyHorizonMetrics,
    distinct_strategies,
)
from northstar_application.application_services.measure_forward_research_record import (
    ForwardResearchMeasurementState,
)
from northstar_application.application_services.run_forward_research import ForwardResearchRun


@dataclass(frozen=True, slots=True)
class ForwardResearchReport:
    """One measured forward run paired with its per-strategy metrics.

    ``metrics`` holds exactly one entry per run strategy and horizon pair,
    strategy-major and horizon-minor, so a reader can walk the entries in a
    known order without searching. A forward run may carry decisions from
    several strategies, so no single report-level strategy exists; the
    researched strategies are surfaced as a set and each metrics entry names
    its own.

    Run configuration and identity are not duplicated here; they remain owned by
    the run and its records, and are surfaced through read-only properties.
    """

    run: ForwardResearchRun
    metrics: tuple[ForwardResearchStrategyHorizonMetrics, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run, ForwardResearchRun):
            raise TypeError("ForwardResearchReport run must be a ForwardResearchRun.")
        if not isinstance(self.metrics, tuple):
            raise TypeError("ForwardResearchReport metrics must be a tuple.")
        if not all(
            isinstance(entry, ForwardResearchStrategyHorizonMetrics) for entry in self.metrics
        ):
            raise TypeError(
                "ForwardResearchReport metrics must contain "
                "ForwardResearchStrategyHorizonMetrics values."
            )

        strategies = distinct_strategies(self.run)
        if len(self.metrics) != len(strategies) * len(self.run.horizons):
            raise ValueError(
                "ForwardResearchReport must contain one metrics entry for every run "
                "strategy and horizon pair."
            )

        expected_pairs = (
            (strategy_identity, horizon)
            for strategy_identity in strategies
            for horizon in self.run.horizons
        )
        for entry, (strategy_identity, horizon) in zip(self.metrics, expected_pairs, strict=True):
            if entry.strategy_identity != strategy_identity:
                raise ValueError(
                    "ForwardResearchReport metrics must follow ascending strategy order, "
                    "grouping each strategy's horizons together."
                )
            if entry.horizon != horizon:
                raise ValueError(
                    "ForwardResearchReport metrics must follow the run horizon order "
                    "within each strategy."
                )

    @property
    def listing_reference(self) -> ListingReference:
        """Return the monitored listing the run was queried for."""
        return ListingReference(self.run.query.symbol, self.run.query.exchange_code)

    @property
    def timeframe(self) -> Timeframe:
        """Return the observation interval a research horizon counts."""
        return self.run.query.timeframe

    @property
    def available_through(self) -> PointInTime:
        """Return the boundary beyond which no observation was available."""
        return self.run.available_through

    @property
    def strategy_identities(self) -> tuple[StrategyIdentity, ...]:
        """Return the distinct monitored strategies, ascending by identity."""
        return distinct_strategies(self.run)

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


class BuildForwardResearchReportUseCase:
    """Assemble one structured report from a measured forward research run.

    Metrics are produced by CalculateForwardResearchMetricsUseCase so metric
    semantics and strategy grouping remain defined in exactly one place.
    """

    def __init__(
        self,
        calculate_metrics: CalculateForwardResearchMetricsUseCase | None = None,
    ) -> None:
        if calculate_metrics is None:
            calculate_metrics = CalculateForwardResearchMetricsUseCase()
        if not isinstance(calculate_metrics, CalculateForwardResearchMetricsUseCase):
            raise TypeError(
                "BuildForwardResearchReportUseCase calculate_metrics must be a "
                "CalculateForwardResearchMetricsUseCase."
            )
        self._calculate_metrics = calculate_metrics

    def execute(self, run: ForwardResearchRun) -> ForwardResearchReport:
        """Return the structured report for one measured forward research run."""
        if run is None:
            raise TypeError("BuildForwardResearchReportUseCase run cannot be None.")
        if not isinstance(run, ForwardResearchRun):
            raise TypeError("BuildForwardResearchReportUseCase run must be a ForwardResearchRun.")

        return ForwardResearchReport(run=run, metrics=self._calculate_metrics.execute(run))
