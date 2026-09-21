"""Tests for aggregating paper trading decisions into a run and report.

Results come from the real orchestration use case driven by the real strategy,
so every counted fact was actually produced rather than asserted into place.
"""

from __future__ import annotations

import dataclasses
from functools import cmp_to_key

import pytest
from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import (
    Currency,
    ExchangeCode,
    PointInTime,
    Price,
    Quantity,
    Symbol,
)
from northstar_core.paper_trading import PaperFill, PaperPortfolioIdentity
from northstar_core.strategy import (
    AssetAnalysisGenerator,
    MarketObservationContext,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services import (
    AnalyzeAssetResult,
    AnalyzeMarketObservationContextService,
    BuildPaperTradingReportUseCase,
    CalculatePaperTradingMetricsUseCase,
    PaperTradingReport,
    PaperTradingResult,
    PaperTradingRun,
    PaperTradingStrategyListingMetrics,
    RunPaperTradingDecisionUseCase,
)
from northstar_application.ports import (
    PaperFillConflictError,
    PaperFillQuery,
    PaperFillRepository,
    PaperFillStore,
)

_USD = Currency("USD")
_NASDAQ = ExchangeCode("NASDAQ")
_LSE = ExchangeCode("LSE")
_AAPL = ListingReference(Symbol("AAPL"), _NASDAQ)
_AAPL_LSE = ListingReference(Symbol("AAPL"), _LSE)
_MSFT = ListingReference(Symbol("MSFT"), _NASDAQ)
_ALPHA = StrategyIdentity("alpha")
_ZETA = StrategyIdentity("zeta")
_PORTFOLIO = PaperPortfolioIdentity("paper-1")
_OTHER_PORTFOLIO = PaperPortfolioIdentity("paper-2")


def _instant(day: int, fraction: str = "") -> PointInTime:
    return PointInTime(f"2026-01-{day:02d}T16:00:00{fraction}Z")


# ---------------------------------------------------------------------------
# Real evidence and real orchestration
# ---------------------------------------------------------------------------


def _context(
    closes: list[int],
    latest: str,
    previous: str,
    *,
    volume: str = "5000",
    observed_at: PointInTime,
    listing: ListingReference = _AAPL,
) -> MarketObservationContext:
    return MarketObservationContext(
        listing,
        observed_at,
        Price(latest, _USD),
        Price(previous, _USD),
        Quantity(volume),
        Price("9000", _USD),
        Price("1", _USD),
        tuple(Price(str(close), _USD) for close in closes),
        tuple(Quantity("1000") for _ in range(20)),
    )


def _analyse(
    context: MarketObservationContext, strategy: StrategyIdentity = _ALPHA
) -> AnalyzeAssetResult:
    return AnalyzeMarketObservationContextService(
        strategy=Strategy(strategy),
        analysis_generator=AssetAnalysisGenerator(),
    ).execute(context)


def _at(day: int, kwargs: dict) -> PointInTime:
    """Let an explicit instant win over the one implied by the day."""
    return kwargs.pop("observed_at", _instant(day))


def _buy(day: int, **kwargs: object) -> AnalyzeAssetResult:
    strategy = kwargs.pop("strategy", _ALPHA)
    return _analyse(
        _context(list(range(180, 200)), "250", "199", observed_at=_at(day, kwargs), **kwargs),
        strategy,
    )


def _sell(day: int, **kwargs: object) -> AnalyzeAssetResult:
    strategy = kwargs.pop("strategy", _ALPHA)
    return _analyse(
        _context(list(range(200, 180, -1)), "150", "181", observed_at=_at(day, kwargs), **kwargs),
        strategy,
    )


def _hold(day: int, **kwargs: object) -> AnalyzeAssetResult:
    strategy = kwargs.pop("strategy", _ALPHA)
    return _analyse(
        _context([100] * 20, "100", "100", volume="100", observed_at=_at(day, kwargs), **kwargs),
        strategy,
    )


def _compare_fills(left: PaperFill, right: PaperFill) -> int:
    instant = left.filled_at.compare(right.filled_at)
    if instant:
        return instant
    a, b = left.identity.identity, right.identity.identity
    return (a > b) - (a < b)


class MemoryFills(PaperFillStore, PaperFillRepository):
    """One in-memory pair honouring the documented persistence semantics."""

    def __init__(self) -> None:
        self.by_fill: dict[str, PaperFill] = {}
        self.by_order: dict[str, str] = {}

    def store(self, fills: tuple[PaperFill, ...]) -> int:
        for fill in fills:
            key = fill.identity.identity
            existing = self.by_fill.get(key)
            if existing is not None:
                if existing != fill:
                    raise PaperFillConflictError("Conflicting fill under one identity.")
                continue
            self.by_fill[key] = fill
            self.by_order[fill.order_identity.identity] = key
        return len(fills)

    def get_fills(self, query: PaperFillQuery) -> tuple[PaperFill, ...]:
        owned = [
            fill
            for fill in self.by_fill.values()
            if fill.portfolio_identity == query.portfolio_identity
        ]
        return tuple(sorted(owned, key=cmp_to_key(_compare_fills)))


class Decisions:
    """Drives real orchestration so every result is genuinely produced."""

    def __init__(self, portfolio: PaperPortfolioIdentity = _PORTFOLIO) -> None:
        self.storage = MemoryFills()
        self.portfolio = portfolio
        self.use_case = RunPaperTradingDecisionUseCase(self.storage, self.storage)

    def run(self, result: AnalyzeAssetResult, quantity: str = "10") -> PaperTradingResult:
        return self.use_case.execute(result, self.portfolio, Quantity(quantity))


def _run(*results: PaperTradingResult, portfolio: PaperPortfolioIdentity = _PORTFOLIO):
    return PaperTradingRun(portfolio_identity=portfolio, results=tuple(results))


def _report(run: PaperTradingRun) -> PaperTradingReport:
    return BuildPaperTradingReportUseCase().execute(run)


# ---------------------------------------------------------------------------
# Empty
# ---------------------------------------------------------------------------


def test_an_empty_run_is_valid() -> None:
    run = _run()

    assert run.decision_count == 0
    assert run.first_decision_instant is None
    assert run.last_decision_instant is None
    assert run.final_portfolio is None


def test_an_empty_run_produces_no_metrics() -> None:
    assert CalculatePaperTradingMetricsUseCase().execute(_run()) == ()


def test_an_empty_report_is_valid() -> None:
    report = _report(_run())

    assert report.metrics == ()
    assert report.decision_count == 0
    assert report.execution_count == 0
    assert report.listing_references == ()
    assert report.strategy_identities == ()
    assert report.final_portfolio is None
    assert report.portfolio_identity == _PORTFOLIO


# ---------------------------------------------------------------------------
# Single-decision runs
# ---------------------------------------------------------------------------


def test_one_buy_execution_is_counted() -> None:
    decisions = Decisions()

    report = _report(_run(decisions.run(_buy(20))))

    only = report.metrics[0]
    assert only.listing_reference == _AAPL
    assert only.strategy_identity == _ALPHA
    assert only.total_decisions == 1
    assert only.buy_recommendations == 1
    assert only.execution_count == 1
    assert only.buy_execution_count == 1
    assert only.sell_execution_count == 0
    assert only.hold_no_intent_count == 0
    assert only.insufficient_position_count == 0


def test_one_hold_is_counted_as_a_visible_non_action() -> None:
    decisions = Decisions()

    report = _report(_run(decisions.run(_hold(20))))

    only = report.metrics[0]
    assert only.total_decisions == 1
    assert only.hold_recommendations == 1
    assert only.hold_no_intent_count == 1
    assert only.execution_count == 0
    assert report.execution_count == 0


def test_one_sell_execution_is_counted() -> None:
    decisions = Decisions()
    decisions.run(_buy(20), quantity="10")

    report = _report(_run(decisions.run(_buy(20)), decisions.run(_sell(21), quantity="4")))

    only = report.metrics[0]
    assert only.total_decisions == 2
    assert only.buy_recommendations == 1
    assert only.sell_recommendations == 1
    assert only.execution_count == 2
    assert only.buy_execution_count == 1
    assert only.sell_execution_count == 1


def test_an_insufficient_position_sell_stays_visible() -> None:
    decisions = Decisions()

    report = _report(_run(decisions.run(_sell(20), quantity="10")))

    only = report.metrics[0]
    assert only.sell_recommendations == 1
    assert only.insufficient_position_count == 1
    assert only.execution_count == 0
    assert only.sell_execution_count == 0


def test_execution_is_never_inferred_from_the_recommendation() -> None:
    """A SELL recommendation that executed nothing must not count as an execution."""
    decisions = Decisions()

    only = _report(_run(decisions.run(_sell(20), quantity="10"))).metrics[0]

    assert only.sell_recommendations == 1
    assert only.sell_execution_count == 0


# ---------------------------------------------------------------------------
# Partition invariants
# ---------------------------------------------------------------------------


def test_recommendation_and_outcome_counts_both_partition_the_total() -> None:
    decisions = Decisions()
    results = (
        decisions.run(_buy(20), quantity="10"),
        decisions.run(_hold(21)),
        decisions.run(_sell(22), quantity="4"),
        decisions.run(_sell(23), quantity="999"),
    )

    only = _report(_run(*results)).metrics[0]

    assert only.total_decisions == 4
    assert (
        only.buy_recommendations + only.hold_recommendations + only.sell_recommendations
        == only.total_decisions
    )
    assert (
        only.execution_count + only.hold_no_intent_count + only.insufficient_position_count
        == only.total_decisions
    )
    assert only.buy_execution_count + only.sell_execution_count == only.execution_count
    assert (only.buy_recommendations, only.hold_recommendations, only.sell_recommendations) == (
        1,
        1,
        2,
    )
    assert (only.execution_count, only.hold_no_intent_count, only.insufficient_position_count) == (
        2,
        1,
        1,
    )


def _valid_metrics(**overrides: object) -> PaperTradingStrategyListingMetrics:
    values: dict[str, object] = {
        "listing_reference": _AAPL,
        "strategy_identity": _ALPHA,
        "total_decisions": 2,
        "buy_recommendations": 1,
        "hold_recommendations": 1,
        "sell_recommendations": 0,
        "execution_count": 1,
        "buy_execution_count": 1,
        "sell_execution_count": 0,
        "hold_no_intent_count": 1,
        "insufficient_position_count": 0,
    }
    values.update(overrides)
    return PaperTradingStrategyListingMetrics(**values)


def test_metrics_reject_invalid_member_types() -> None:
    with pytest.raises(TypeError, match="listing reference must be a ListingReference"):
        _valid_metrics(listing_reference="AAPL@NASDAQ")
    with pytest.raises(TypeError, match="strategy identity must be a StrategyIdentity"):
        _valid_metrics(strategy_identity="alpha")


@pytest.mark.parametrize(
    "name",
    [
        "total_decisions",
        "buy_recommendations",
        "hold_recommendations",
        "sell_recommendations",
        "execution_count",
        "buy_execution_count",
        "sell_execution_count",
        "hold_no_intent_count",
        "insufficient_position_count",
    ],
)
def test_metrics_reject_non_integer_and_boolean_counts(name: str) -> None:
    with pytest.raises(TypeError, match=f"{name} must be an integer"):
        _valid_metrics(**{name: "1"})
    with pytest.raises(TypeError, match=f"{name} must be an integer"):
        _valid_metrics(**{name: True})


def test_metrics_reject_negative_counts() -> None:
    with pytest.raises(ValueError, match="sell_recommendations cannot be negative"):
        _valid_metrics(sell_recommendations=-1)


def test_metrics_require_the_recommendation_partition() -> None:
    with pytest.raises(ValueError, match="recommendation counts must equal"):
        _valid_metrics(buy_recommendations=2)


def test_metrics_require_the_outcome_partition() -> None:
    with pytest.raises(ValueError, match="outcome counts must equal"):
        _valid_metrics(hold_no_intent_count=0, total_decisions=2, execution_count=1)


def test_metrics_require_execution_sides_to_sum() -> None:
    with pytest.raises(ValueError, match="execution sides must equal"):
        _valid_metrics(buy_execution_count=0)


def test_metrics_are_immutable() -> None:
    metrics = _valid_metrics()

    with pytest.raises(AttributeError):
        metrics.total_decisions = 9


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def test_multiple_listings_are_grouped_separately() -> None:
    decisions = Decisions()
    results = (
        decisions.run(_buy(20), quantity="10"),
        decisions.run(_buy(20, listing=_MSFT), quantity="5"),
    )

    report = _report(_run(*results))

    assert [entry.listing_reference for entry in report.metrics] == [_AAPL, _MSFT]
    assert all(entry.total_decisions == 1 for entry in report.metrics)


def test_multiple_strategies_are_grouped_separately() -> None:
    decisions = Decisions()
    results = (
        decisions.run(_buy(20), quantity="10"),
        decisions.run(_buy(20, strategy=_ZETA), quantity="5"),
    )

    report = _report(_run(*results))

    assert [entry.strategy_identity for entry in report.metrics] == [_ALPHA, _ZETA]


def test_one_listing_under_two_strategies_stays_separated() -> None:
    decisions = Decisions()
    results = (
        decisions.run(_buy(20), quantity="10"),
        decisions.run(_hold(20, strategy=_ZETA)),
    )

    report = _report(_run(*results))

    assert len(report.metrics) == 2
    alpha, zeta = report.metrics
    assert alpha.buy_recommendations == 1
    assert zeta.hold_recommendations == 1
    assert zeta.execution_count == 0


def test_one_strategy_across_two_listings_stays_separated() -> None:
    decisions = Decisions()
    results = (
        decisions.run(_buy(20), quantity="10"),
        decisions.run(_hold(20, listing=_MSFT)),
    )

    report = _report(_run(*results))

    assert len(report.metrics) == 2
    assert [entry.listing_reference for entry in report.metrics] == [_AAPL, _MSFT]
    assert report.metrics[0].execution_count == 1
    assert report.metrics[1].execution_count == 0


def test_groups_follow_canonical_listing_then_strategy_order() -> None:
    decisions = Decisions()
    results = (
        decisions.run(_buy(20, listing=_AAPL_LSE), quantity="1"),
        decisions.run(_buy(20, strategy=_ZETA), quantity="1"),
        decisions.run(_buy(20), quantity="1"),
        decisions.run(_buy(20, listing=_MSFT), quantity="1"),
    )

    report = _report(_run(*sorted(results, key=_result_order)))

    assert [
        (
            e.listing_reference.symbol.value,
            e.listing_reference.exchange_code.value,
            e.strategy_identity.identity,
        )
        for e in report.metrics
    ] == [
        ("AAPL", "LSE", "alpha"),
        ("AAPL", "NASDAQ", "alpha"),
        ("AAPL", "NASDAQ", "zeta"),
        ("MSFT", "NASDAQ", "alpha"),
    ]


def _result_order(result: PaperTradingResult) -> tuple[str, str, str, str]:
    recommendation = result.result.recommendation
    listing = recommendation.asset_analysis.listing_reference
    return (
        recommendation.point_in_time.value,
        listing.symbol.value,
        listing.exchange_code.value,
        recommendation.strategy_identity.identity,
    )


# ---------------------------------------------------------------------------
# Run ordering and identity
# ---------------------------------------------------------------------------


def test_results_must_be_ordered_chronologically() -> None:
    decisions = Decisions()
    first = decisions.run(_buy(20), quantity="10")
    second = decisions.run(_hold(21))

    assert _run(first, second).decision_count == 2
    with pytest.raises(ValueError, match="ordered by decision instant"):
        _run(second, first)


def test_sub_second_ordering_is_chronological_not_textual() -> None:
    """'.1Z' sorts before 'Z' as text while being the later instant."""
    decisions = Decisions()
    whole = decisions.run(_buy(20), quantity="10")
    fractional = decisions.run(_hold(20, observed_at=_instant(20, ".1")))

    assert _instant(20, ".1").value < _instant(20).value
    assert _run(whole, fractional).decision_count == 2
    with pytest.raises(ValueError, match="ordered by decision instant"):
        _run(fractional, whole)


def test_equal_instants_tie_break_by_listing_then_strategy() -> None:
    decisions = Decisions()
    aapl_alpha = decisions.run(_buy(20), quantity="1")
    aapl_zeta = decisions.run(_hold(20, strategy=_ZETA))
    msft_alpha = decisions.run(_hold(20, listing=_MSFT))

    assert _run(aapl_alpha, aapl_zeta, msft_alpha).decision_count == 3
    with pytest.raises(ValueError, match="then listing, then strategy"):
        _run(aapl_zeta, aapl_alpha, msft_alpha)
    with pytest.raises(ValueError, match="then listing, then strategy"):
        _run(msft_alpha, aapl_alpha, aapl_zeta)


def test_the_run_never_silently_sorts() -> None:
    decisions = Decisions()
    first = decisions.run(_buy(20), quantity="10")
    second = decisions.run(_hold(21))

    with pytest.raises(ValueError):
        _run(second, first)
    assert _run(first, second).results == (first, second)


def test_a_duplicate_decision_boundary_is_rejected() -> None:
    decisions = Decisions()
    result = decisions.run(_buy(20), quantity="10")

    with pytest.raises(ValueError, match="two results for one decision boundary"):
        _run(result, result)


def test_a_foreign_portfolio_result_is_rejected() -> None:
    decisions = Decisions()
    result = decisions.run(_buy(20), quantity="10")

    with pytest.raises(ValueError, match="must all belong to the run's paper portfolio"):
        _run(result, portfolio=_OTHER_PORTFOLIO)


def test_run_rejects_invalid_member_types() -> None:
    with pytest.raises(TypeError, match="portfolio identity must be a PaperPortfolioIdentity"):
        PaperTradingRun(portfolio_identity="paper-1", results=())
    with pytest.raises(TypeError, match="results must be a tuple"):
        PaperTradingRun(portfolio_identity=_PORTFOLIO, results=[])
    with pytest.raises(TypeError, match="must contain PaperTradingResult values"):
        PaperTradingRun(portfolio_identity=_PORTFOLIO, results=("result",))


def test_run_exposes_its_decision_span_and_final_holdings() -> None:
    decisions = Decisions()
    first = decisions.run(_buy(20), quantity="10")
    last = decisions.run(_sell(22), quantity="4")
    run = _run(first, last)

    assert run.decision_count == 2
    assert run.first_decision_instant == _instant(20)
    assert run.last_decision_instant == _instant(22)
    assert run.final_portfolio is last.portfolio
    assert run.final_portfolio.get_position(_AAPL).quantity == Quantity("6")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_report_exposes_derived_properties() -> None:
    decisions = Decisions()
    results = (
        decisions.run(_buy(20), quantity="10"),
        decisions.run(_hold(21, listing=_MSFT)),
        decisions.run(_sell(22), quantity="4"),
    )
    run = _run(*results)

    report = _report(run)

    assert report.portfolio_identity == _PORTFOLIO
    assert report.decision_count == 3
    assert report.execution_count == 2
    assert report.first_decision_instant == _instant(20)
    assert report.last_decision_instant == _instant(22)
    assert report.final_portfolio is results[-1].portfolio
    assert report.listing_references == (_AAPL, _MSFT)
    assert report.strategy_identities == (_ALPHA,)


def test_report_lists_distinct_strategies_ascending() -> None:
    decisions = Decisions()
    results = (
        decisions.run(_buy(20), quantity="1"),
        decisions.run(_hold(20, strategy=_ZETA)),
        decisions.run(_hold(21), quantity="1"),
    )

    report = _report(_run(*results))

    assert report.strategy_identities == (_ALPHA, _ZETA)
    assert report.listing_references == (_AAPL,)


def test_report_rejects_a_missing_metrics_entry() -> None:
    decisions = Decisions()
    run = _run(decisions.run(_buy(20), quantity="1"), decisions.run(_hold(20, listing=_MSFT)))
    metrics = CalculatePaperTradingMetricsUseCase().execute(run)

    with pytest.raises(ValueError, match="one metrics entry for every run listing"):
        PaperTradingReport(run=run, metrics=metrics[:-1])


def test_report_rejects_a_surplus_metrics_entry() -> None:
    decisions = Decisions()
    run = _run(decisions.run(_buy(20), quantity="1"))
    metrics = CalculatePaperTradingMetricsUseCase().execute(run)

    with pytest.raises(ValueError, match="one metrics entry for every run listing"):
        PaperTradingReport(run=run, metrics=metrics + metrics)


def test_report_rejects_reordered_metrics() -> None:
    decisions = Decisions()
    run = _run(decisions.run(_buy(20), quantity="1"), decisions.run(_hold(20, listing=_MSFT)))
    first, second = CalculatePaperTradingMetricsUseCase().execute(run)

    with pytest.raises(ValueError, match="canonical listing and strategy order"):
        PaperTradingReport(run=run, metrics=(second, first))


def test_report_rejects_a_group_absent_from_the_run() -> None:
    decisions = Decisions()
    run = _run(decisions.run(_buy(20), quantity="1"))
    (only,) = CalculatePaperTradingMetricsUseCase().execute(run)
    impostor = dataclasses.replace(only, listing_reference=_MSFT)

    with pytest.raises(ValueError, match="canonical listing and strategy order"):
        PaperTradingReport(run=run, metrics=(impostor,))


def test_report_rejects_a_wrong_strategy_group() -> None:
    decisions = Decisions()
    run = _run(decisions.run(_buy(20), quantity="1"))
    (only,) = CalculatePaperTradingMetricsUseCase().execute(run)
    impostor = dataclasses.replace(only, strategy_identity=_ZETA)

    with pytest.raises(ValueError, match="canonical listing and strategy order"):
        PaperTradingReport(run=run, metrics=(impostor,))


def test_report_rejects_invalid_member_types() -> None:
    decisions = Decisions()
    run = _run(decisions.run(_buy(20), quantity="1"))
    metrics = CalculatePaperTradingMetricsUseCase().execute(run)

    with pytest.raises(TypeError, match="run must be a PaperTradingRun"):
        PaperTradingReport(run="run", metrics=metrics)
    with pytest.raises(TypeError, match="metrics must be a tuple"):
        PaperTradingReport(run=run, metrics=list(metrics))
    with pytest.raises(TypeError, match="must contain PaperTradingStrategyListingMetrics"):
        PaperTradingReport(run=run, metrics=("metrics",))


def test_report_is_immutable() -> None:
    report = _report(_run())

    with pytest.raises(AttributeError):
        report.metrics = ()


# ---------------------------------------------------------------------------
# Builder and determinism
# ---------------------------------------------------------------------------


def test_builder_rejects_an_invalid_calculator() -> None:
    with pytest.raises(TypeError, match="must be a CalculatePaperTradingMetricsUseCase"):
        BuildPaperTradingReportUseCase("calculator")


def test_builder_requires_a_run() -> None:
    use_case = BuildPaperTradingReportUseCase()

    with pytest.raises(TypeError, match="run cannot be None"):
        use_case.execute(None)
    with pytest.raises(TypeError, match="run must be a PaperTradingRun"):
        use_case.execute("run")


def test_calculator_requires_a_run() -> None:
    use_case = CalculatePaperTradingMetricsUseCase()

    with pytest.raises(TypeError, match="run cannot be None"):
        use_case.execute(None)
    with pytest.raises(TypeError, match="run must be a PaperTradingRun"):
        use_case.execute("run")


def test_repeated_builds_return_an_equal_report() -> None:
    decisions = Decisions()
    run = _run(decisions.run(_buy(20), quantity="10"), decisions.run(_hold(21)))

    assert _report(run) == _report(run)


def test_an_injected_calculator_is_used() -> None:
    decisions = Decisions()
    run = _run(decisions.run(_buy(20), quantity="10"))

    assert BuildPaperTradingReportUseCase(CalculatePaperTradingMetricsUseCase()).execute(
        run
    ) == _report(run)


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------


def test_the_report_expresses_no_performance_interpretation() -> None:
    decisions = Decisions()
    report = _report(_run(decisions.run(_buy(20), quantity="10")))

    for absent in (
        "return",
        "pnl",
        "profit",
        "win_rate",
        "hit_rate",
        "accuracy",
        "sharpe",
        "cash",
        "balance",
        "ranking",
        "best_strategy",
    ):
        assert not hasattr(report, absent)
        assert not hasattr(report.metrics[0], absent)


# ---------------------------------------------------------------------------
# execution_count is a run-derived fact
# ---------------------------------------------------------------------------


def test_execution_count_is_correct_across_a_mixed_run() -> None:
    decisions = Decisions()
    results = (
        decisions.run(_buy(20), quantity="10"),
        decisions.run(_hold(21)),
        decisions.run(_sell(22), quantity="4"),
        decisions.run(_sell(23), quantity="999"),
    )

    report = _report(_run(*results))

    assert report.decision_count == 4
    assert report.execution_count == 2
    assert [result.execution is not None for result in results] == [True, False, True, False]


def test_execution_count_counts_executions_across_every_group() -> None:
    decisions = Decisions()
    # At one instant the run orders by listing, then strategy: AAPL/alpha,
    # AAPL/zeta, MSFT/alpha.
    results = (
        decisions.run(_buy(20), quantity="10"),
        decisions.run(_hold(20, strategy=_ZETA)),
        decisions.run(_buy(20, listing=_MSFT), quantity="5"),
    )

    report = _report(_run(*results))

    assert len(report.metrics) == 3
    assert report.execution_count == 2


def test_execution_count_is_zero_when_nothing_executed() -> None:
    decisions = Decisions()
    results = (decisions.run(_hold(20)), decisions.run(_sell(21), quantity="999"))

    assert _report(_run(*results)).execution_count == 0


def test_understated_metrics_cannot_lower_the_reported_execution_count() -> None:
    """Group keys stay correct, so correspondence passes; the count still cannot lie."""
    decisions = Decisions()
    run = _run(decisions.run(_buy(20), quantity="10"), decisions.run(_sell(21), quantity="4"))
    (only,) = CalculatePaperTradingMetricsUseCase().execute(run)
    understated = dataclasses.replace(
        only,
        execution_count=0,
        buy_execution_count=0,
        sell_execution_count=0,
        hold_no_intent_count=0,
        insufficient_position_count=2,
    )

    report = PaperTradingReport(run=run, metrics=(understated,))

    assert report.metrics[0].execution_count == 0
    assert report.execution_count == 2


def test_overstated_metrics_cannot_raise_the_reported_execution_count() -> None:
    decisions = Decisions()
    run = _run(decisions.run(_hold(20)))
    (only,) = CalculatePaperTradingMetricsUseCase().execute(run)
    overstated = dataclasses.replace(
        only,
        execution_count=1,
        buy_execution_count=1,
        hold_no_intent_count=0,
    )

    report = PaperTradingReport(run=run, metrics=(overstated,))

    assert report.metrics[0].execution_count == 1
    assert report.execution_count == 0


def test_a_built_report_agrees_with_its_own_metrics() -> None:
    """The builder's metrics are truthful, so both readings coincide."""
    decisions = Decisions()
    report = _report(
        _run(
            decisions.run(_buy(20), quantity="10"),
            decisions.run(_hold(21)),
            decisions.run(_sell(22), quantity="4"),
        )
    )

    assert report.execution_count == sum(entry.execution_count for entry in report.metrics)
