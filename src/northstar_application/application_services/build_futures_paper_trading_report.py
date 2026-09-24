"""Application assembly of one structured futures paper-trading report.

A report pairs one futures paper-trading run with its per-contract execution
metrics and exposes facts derived from the run. It reads no repository,
executes nothing and consults no clock. It performs no rendering, serialisation
or persistence and calculates no profit and loss.

The report is an as-of view at the run cutoff; the same run at a later cutoff
may read differently as pending orders fill.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.futures import FuturesContract
from northstar_core.paper_trading import (
    FuturesContractCount,
    FuturesPaperPortfolio,
    FuturesPosition,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services.calculate_futures_paper_trading_metrics import (
    CalculateFuturesPaperTradingMetricsUseCase,
    FuturesPaperTradingStrategyContractMetrics,
)
from northstar_application.application_services.run_futures_paper_trading import (
    FuturesPaperTradingRun,
)


@dataclass(frozen=True, slots=True)
class FuturesPaperTradingReport:
    """One futures paper-trading run and the execution metrics describing it.

    ``metrics`` must be exactly what CalculateFuturesPaperTradingMetricsUseCase
    derives from ``run``: one entry per decided contract, by contract natural
    key, with truthful counts. Run configuration and holdings are not
    duplicated; they are surfaced from the run.
    """

    run: FuturesPaperTradingRun
    metrics: tuple[FuturesPaperTradingStrategyContractMetrics, ...]

    def __post_init__(self) -> None:
        subject = "FuturesPaperTradingReport"
        if not isinstance(self.run, FuturesPaperTradingRun):
            raise TypeError(f"{subject} run must be a FuturesPaperTradingRun.")
        if not isinstance(self.metrics, tuple):
            raise TypeError(f"{subject} metrics must be a tuple.")
        if not all(
            isinstance(entry, FuturesPaperTradingStrategyContractMetrics) for entry in self.metrics
        ):
            raise TypeError(
                f"{subject} metrics must contain FuturesPaperTradingStrategyContractMetrics values."
            )

        expected = CalculateFuturesPaperTradingMetricsUseCase().execute(self.run)
        if len(self.metrics) != len(expected):
            raise ValueError(
                f"{subject} must contain exactly one metrics entry for every decided contract."
            )
        for entry, canonical in zip(self.metrics, expected, strict=True):
            if entry.contract != canonical.contract:
                raise ValueError(f"{subject} metrics must follow canonical contract order.")
            if entry.strategy_identity != canonical.strategy_identity:
                raise ValueError(f"{subject} metrics must belong to the run strategy.")
            if entry != canonical:
                raise ValueError(f"{subject} metrics for {entry.contract} do not describe the run.")

    @property
    def portfolio_identity(self) -> PaperPortfolioIdentity:
        """Return the paper portfolio the run executed into."""
        return self.run.portfolio_identity

    @property
    def strategy_identity(self) -> StrategyIdentity:
        """Return the one strategy whose decisions the run executed."""
        return self.run.strategy_identity

    @property
    def contract(self) -> FuturesContract:
        """Return the contract the run was queried for."""
        return self.run.query.contract

    @property
    def timeframe(self) -> Timeframe:
        """Return the forward decisions' observation interval."""
        return self.run.query.timeframe

    @property
    def available_through(self) -> PointInTime:
        """Return the run cutoff exactly as the run recorded it."""
        return self.run.available_through

    @property
    def target_contracts(self) -> FuturesContractCount:
        """Return the target size every decision was executed with."""
        return self.run.target_contracts

    @property
    def decision_count(self) -> int:
        """Return how many frozen decisions the run executed."""
        return self.run.decision_count

    @property
    def order_count(self) -> int:
        """Return how many paper orders the run's decisions produced."""
        return len(self.run.orders)

    @property
    def fill_count(self) -> int:
        """Return how many of those orders are filled at the cutoff."""
        return len(self.run.fills)

    @property
    def pending_order_count(self) -> int:
        """Return how many of those orders are still pending at the cutoff."""
        return self.order_count - self.fill_count

    @property
    def portfolio(self) -> FuturesPaperPortfolio:
        """Return the whole portfolio as of the cutoff, across every contract."""
        return self.run.portfolio

    @property
    def positions(self) -> tuple[FuturesPosition, ...]:
        """Return every open position at the cutoff, in canonical contract order."""
        return self.run.portfolio.positions

    @property
    def position_count(self) -> int:
        """Return how many contracts hold an open position at the cutoff."""
        return self.run.portfolio.position_count

    @property
    def first_decision_instant(self) -> PointInTime | None:
        """Return the earliest executed decision instant, or None when empty."""
        return self.run.first_decision_instant

    @property
    def last_decision_instant(self) -> PointInTime | None:
        """Return the latest executed decision instant, or None when empty."""
        return self.run.last_decision_instant


class BuildFuturesPaperTradingReportUseCase:
    """Assemble one structured report from a futures paper-trading run."""

    def __init__(
        self, calculate_metrics: CalculateFuturesPaperTradingMetricsUseCase | None = None
    ) -> None:
        if calculate_metrics is None:
            calculate_metrics = CalculateFuturesPaperTradingMetricsUseCase()
        if not isinstance(calculate_metrics, CalculateFuturesPaperTradingMetricsUseCase):
            raise TypeError(
                "BuildFuturesPaperTradingReportUseCase calculate_metrics must be a "
                "CalculateFuturesPaperTradingMetricsUseCase."
            )
        self._calculate_metrics = calculate_metrics

    def execute(self, run: FuturesPaperTradingRun) -> FuturesPaperTradingReport:
        """Return the structured report for one futures paper-trading run."""
        if not isinstance(run, FuturesPaperTradingRun):
            raise TypeError(
                "BuildFuturesPaperTradingReportUseCase run must be a FuturesPaperTradingRun."
            )
        return FuturesPaperTradingReport(run=run, metrics=self._calculate_metrics.execute(run))
