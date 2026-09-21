"""One paper trading run paired with its factual counts.

A report is a structured Application result: it performs no rendering, no
serialization and no persistence, introduces no metric of its own, and applies
no success, ranking or performance interpretation. It is an in-memory artifact
describing what a portfolio did.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import PointInTime
from northstar_core.paper_trading import PaperPortfolio, PaperPortfolioIdentity
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services.calculate_paper_trading_metrics import (
    CalculatePaperTradingMetricsUseCase,
    PaperTradingStrategyListingMetrics,
    group_key,
)
from northstar_application.application_services.paper_trading_run import PaperTradingRun


@dataclass(frozen=True, slots=True)
class PaperTradingReport:
    """One paper trading run and the counts describing it.

    ``metrics`` holds exactly one entry per listing and strategy pair present
    in the run, in canonical order, so a reader can walk the entries without
    searching. Run configuration and holdings are not duplicated here; they
    remain owned by the run and are surfaced as read-only properties.
    """

    run: PaperTradingRun
    metrics: tuple[PaperTradingStrategyListingMetrics, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.run, PaperTradingRun):
            raise TypeError("PaperTradingReport run must be a PaperTradingRun.")
        if not isinstance(self.metrics, tuple):
            raise TypeError("PaperTradingReport metrics must be a tuple.")
        if not all(isinstance(entry, PaperTradingStrategyListingMetrics) for entry in self.metrics):
            raise TypeError(
                "PaperTradingReport metrics must contain PaperTradingStrategyListingMetrics values."
            )

        expected = self._expected_groups()
        if len(self.metrics) != len(expected):
            raise ValueError(
                "PaperTradingReport must contain one metrics entry for every run listing "
                "and strategy pair."
            )
        for entry, key in zip(self.metrics, expected, strict=True):
            if group_key(entry.listing_reference, entry.strategy_identity) != key:
                raise ValueError(
                    "PaperTradingReport metrics must follow canonical listing and strategy order."
                )

    def _expected_groups(self) -> tuple[tuple[str, str, str], ...]:
        keys = {
            group_key(
                result.result.recommendation.asset_analysis.listing_reference,
                result.result.recommendation.strategy_identity,
            )
            for result in self.run.results
        }
        return tuple(sorted(keys))

    @property
    def portfolio_identity(self) -> PaperPortfolioIdentity:
        """Return the paper portfolio the run describes."""
        return self.run.portfolio_identity

    @property
    def decision_count(self) -> int:
        """Return how many decisions the run records."""
        return self.run.decision_count

    @property
    def execution_count(self) -> int:
        """Return how many decisions produced a simulated execution.

        Counted from the run rather than summed from the metrics. Correspondence
        validation checks which groups are present, not that their counts are
        truthful, so a hand-built metrics tuple could carry correct group keys
        and understated counts. A report-level fact is read from the
        authoritative record instead.
        """
        return sum(1 for result in self.run.results if result.execution is not None)

    @property
    def final_portfolio(self) -> PaperPortfolio | None:
        """Return the holdings after the last decision, or None when empty."""
        return self.run.final_portfolio

    @property
    def first_decision_instant(self) -> PointInTime | None:
        """Return the earliest decision instant, or None when the run is empty."""
        return self.run.first_decision_instant

    @property
    def last_decision_instant(self) -> PointInTime | None:
        """Return the latest decision instant, or None when the run is empty."""
        return self.run.last_decision_instant

    @property
    def listing_references(self) -> tuple[ListingReference, ...]:
        """Return the distinct listings decided upon, in canonical order."""
        seen: dict[tuple[str, str], ListingReference] = {}
        for entry in self.metrics:
            listing_reference = entry.listing_reference
            seen[(listing_reference.symbol.value, listing_reference.exchange_code.value)] = (
                listing_reference
            )
        return tuple(seen[key] for key in sorted(seen))

    @property
    def strategy_identities(self) -> tuple[StrategyIdentity, ...]:
        """Return the distinct strategies that decided, ascending by identity."""
        seen = {entry.strategy_identity.identity: entry.strategy_identity for entry in self.metrics}
        return tuple(seen[identity] for identity in sorted(seen))


class BuildPaperTradingReportUseCase:
    """Assemble one structured report from a paper trading run.

    Metrics are produced by CalculatePaperTradingMetricsUseCase so counting
    semantics and canonical grouping remain defined in exactly one place.
    """

    def __init__(
        self, calculate_metrics: CalculatePaperTradingMetricsUseCase | None = None
    ) -> None:
        if calculate_metrics is None:
            calculate_metrics = CalculatePaperTradingMetricsUseCase()
        if not isinstance(calculate_metrics, CalculatePaperTradingMetricsUseCase):
            raise TypeError(
                "BuildPaperTradingReportUseCase calculate_metrics must be a "
                "CalculatePaperTradingMetricsUseCase."
            )
        self._calculate_metrics = calculate_metrics

    def execute(self, run: PaperTradingRun) -> PaperTradingReport:
        """Return the structured report for one paper trading run."""
        if run is None:
            raise TypeError("BuildPaperTradingReportUseCase run cannot be None.")
        if not isinstance(run, PaperTradingRun):
            raise TypeError("BuildPaperTradingReportUseCase run must be a PaperTradingRun.")

        return PaperTradingReport(run=run, metrics=self._calculate_metrics.execute(run))
