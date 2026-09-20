"""Tests for structured forward research report assembly."""

from __future__ import annotations

import dataclasses

import pytest
from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import (
    Currency,
    ExchangeCode,
    Percentage,
    PointInTime,
    Price,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.market_data import HistoricalOHLCVBar
from northstar_core.strategy import (
    AssetAnalysisGenerator,
    MarketObservationContext,
    ResearchHorizon,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services import (
    AnalyzeAssetResult,
    AnalyzeMarketObservationContextService,
    BuildForwardResearchReportUseCase,
    CalculateForwardResearchMetricsUseCase,
    ForwardResearchRecord,
    ForwardResearchReport,
    ForwardResearchRun,
    MeasureForwardResearchRecordUseCase,
    MeasureRecommendationOutcomeUseCase,
    RunForwardResearchUseCase,
)
from northstar_application.ports import (
    ForwardResearchRecordQuery,
    ForwardResearchRecordRepository,
    HistoricalMarketDataQuery,
    HistoricalMarketDataRepository,
)

_USD = Currency("USD")
_DAILY = Timeframe("1d")
_SYMBOL = Symbol("AAPL")
_EXCHANGE = ExchangeCode("NASDAQ")
_AVAILABLE_THROUGH = PointInTime("2026-03-01T16:00:00Z")
_ALPHA = StrategyIdentity("alpha")
_ZETA = StrategyIdentity("zeta")


def _instant(day: int) -> PointInTime:
    return PointInTime(f"2026-01-{day:02d}T16:00:00Z")


def _bar(day: int, close: str) -> HistoricalOHLCVBar:
    return HistoricalOHLCVBar(
        _SYMBOL,
        _EXCHANGE,
        _instant(day),
        _DAILY,
        Price("50", _USD),
        Price("900", _USD),
        Price("1", _USD),
        Price(close, _USD),
        Quantity("1000"),
    )


class StubMarketData(HistoricalMarketDataRepository):
    def __init__(self, observations: tuple[HistoricalOHLCVBar, ...] = ()) -> None:
        self.observations = observations

    def get_history(self, query: HistoricalMarketDataQuery) -> tuple[HistoricalOHLCVBar, ...]:
        return tuple(
            bar
            for bar in self.observations
            if bar.symbol == query.symbol
            and bar.exchange_code == query.exchange_code
            and bar.timeframe == query.timeframe
            and query.start.compare(bar.point_in_time) <= 0
            and query.end.compare(bar.point_in_time) >= 0
        )


class StubRecordRepository(ForwardResearchRecordRepository):
    def __init__(self, records: tuple[ForwardResearchRecord, ...] = ()) -> None:
        self.records = records

    def get_records(self, query: ForwardResearchRecordQuery) -> tuple[ForwardResearchRecord, ...]:
        return self.records


class CountingCalculator(CalculateForwardResearchMetricsUseCase):
    """Records how often metrics were calculated."""

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, run: ForwardResearchRun) -> tuple:
        self.calls += 1
        return super().execute(run)


def _result(day: int, *, strategy: str = "alpha", latest_close: str = "100") -> AnalyzeAssetResult:
    context = MarketObservationContext(
        ListingReference(_SYMBOL, _EXCHANGE),
        _instant(day),
        Price(latest_close, _USD),
        Price("99", _USD),
        Quantity("1000"),
        Price("900", _USD),
        Price("1", _USD),
        tuple(Price("100", _USD) for _ in range(20)),
        tuple(Quantity("1000") for _ in range(20)),
    )
    return AnalyzeMarketObservationContextService(
        strategy=Strategy(StrategyIdentity(strategy)),
        analysis_generator=AssetAnalysisGenerator(),
    ).execute(context)


def _record(day: int, **kwargs: str) -> ForwardResearchRecord:
    return ForwardResearchRecord(result=_result(day, **kwargs), timeframe=_DAILY)


def _run(
    records: tuple[ForwardResearchRecord, ...] = (),
    horizons: tuple[ResearchHorizon, ...] = (ResearchHorizon(1),),
    observations: tuple[HistoricalOHLCVBar, ...] = (),
) -> ForwardResearchRun:
    use_case = RunForwardResearchUseCase(
        StubRecordRepository(records),
        MeasureForwardResearchRecordUseCase(
            MeasureRecommendationOutcomeUseCase(StubMarketData(observations))
        ),
    )
    return use_case.execute(
        ForwardResearchRecordQuery(_SYMBOL, _EXCHANGE, _DAILY),
        horizons,
        _AVAILABLE_THROUGH,
    )


def _two_strategy_run(
    horizons: tuple[ResearchHorizon, ...] = (ResearchHorizon(1), ResearchHorizon(2)),
) -> ForwardResearchRun:
    records = (
        _record(20, strategy="alpha", latest_close="100"),
        _record(22, strategy="zeta", latest_close="200"),
        _record(24, strategy="alpha", latest_close="400"),
        _record(24, strategy="zeta", latest_close="400"),
    )
    observations = (
        _bar(20, "100"),
        _bar(21, "110"),
        _bar(22, "200"),
        _bar(23, "260"),
        _bar(24, "400"),
        _bar(25, "380"),
    )
    return _run(records, horizons, observations)


def _report(run: ForwardResearchRun) -> ForwardResearchReport:
    return BuildForwardResearchReportUseCase().execute(run)


# ---------------------------------------------------------------------------
# Dependency and input validation
# ---------------------------------------------------------------------------


def test_builder_rejects_an_invalid_calculator() -> None:
    with pytest.raises(TypeError, match="must be a CalculateForwardResearchMetricsUseCase"):
        BuildForwardResearchReportUseCase("calculator")


def test_builder_defaults_to_its_own_calculator() -> None:
    report = BuildForwardResearchReportUseCase().execute(_two_strategy_run())

    assert len(report.metrics) == 4


def test_builder_uses_an_injected_calculator() -> None:
    calculator = CountingCalculator()

    report = BuildForwardResearchReportUseCase(calculator).execute(_two_strategy_run())

    assert calculator.calls == 1
    assert len(report.metrics) == 4


def test_builder_requires_a_forward_research_run() -> None:
    use_case = BuildForwardResearchReportUseCase()

    with pytest.raises(TypeError, match="run cannot be None"):
        use_case.execute(None)
    with pytest.raises(TypeError, match="must be a ForwardResearchRun"):
        use_case.execute("run")


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def test_a_report_pairs_one_metrics_entry_per_strategy_and_horizon() -> None:
    run = _two_strategy_run()

    report = _report(run)

    assert report.run is run
    assert [(entry.strategy_identity, entry.horizon) for entry in report.metrics] == [
        (_ALPHA, ResearchHorizon(1)),
        (_ALPHA, ResearchHorizon(2)),
        (_ZETA, ResearchHorizon(1)),
        (_ZETA, ResearchHorizon(2)),
    ]


def test_a_report_keeps_strategy_summaries_separate() -> None:
    report = _report(_two_strategy_run((ResearchHorizon(1),)))

    alpha, zeta = report.metrics
    assert alpha.average_forward_return == Percentage("2.5")
    assert zeta.average_forward_return == Percentage("12.5")


def test_repeated_builds_of_one_run_are_equal() -> None:
    run = _two_strategy_run()

    assert _report(run) == _report(run)


def test_repeated_builds_by_different_builders_are_equal() -> None:
    run = _two_strategy_run()

    assert BuildForwardResearchReportUseCase().execute(run) == BuildForwardResearchReportUseCase(
        CountingCalculator()
    ).execute(run)


# ---------------------------------------------------------------------------
# Empty run
# ---------------------------------------------------------------------------


def test_an_empty_run_builds_an_empty_report() -> None:
    report = _report(_run())

    assert report.metrics == ()
    assert report.record_count == 0
    assert report.pending_count == 0
    assert report.strategy_identities == ()
    assert report.first_decision_instant is None
    assert report.last_decision_instant is None


def test_an_empty_report_still_identifies_the_monitored_series() -> None:
    report = _report(_run())

    assert report.listing_reference == ListingReference(_SYMBOL, _EXCHANGE)
    assert report.timeframe == _DAILY
    assert report.available_through == _AVAILABLE_THROUGH


# ---------------------------------------------------------------------------
# Derived properties
# ---------------------------------------------------------------------------


def test_listing_and_timeframe_come_from_the_run_query() -> None:
    report = _report(_two_strategy_run())

    assert report.listing_reference == ListingReference(_SYMBOL, _EXCHANGE)
    assert report.listing_reference.symbol == report.run.query.symbol
    assert report.listing_reference.exchange_code == report.run.query.exchange_code
    assert report.timeframe == report.run.query.timeframe


def test_available_through_comes_from_the_run() -> None:
    report = _report(_two_strategy_run())

    assert report.available_through == _AVAILABLE_THROUGH


def test_strategy_identities_are_distinct_and_ascending() -> None:
    report = _two_strategy_run()

    assert _report(report).strategy_identities == (_ALPHA, _ZETA)


def test_record_and_pending_counts_describe_the_run() -> None:
    report = _report(_two_strategy_run())

    assert report.record_count == 4
    assert report.pending_count == 2


def test_pending_count_agrees_with_the_metrics_entries() -> None:
    report = _report(_two_strategy_run())

    assert report.pending_count == sum(entry.pending_count for entry in report.metrics)


def test_decision_instants_span_the_frozen_decisions() -> None:
    report = _report(_two_strategy_run())

    assert report.first_decision_instant == _instant(20)
    assert report.last_decision_instant == _instant(24)


def test_a_report_exposes_no_single_strategy_identity() -> None:
    """A forward run may monitor several strategies, so no one strategy owns it."""
    assert not hasattr(_report(_two_strategy_run()), "strategy_identity")


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------


def test_report_rejects_invalid_member_types() -> None:
    run = _two_strategy_run()
    metrics = CalculateForwardResearchMetricsUseCase().execute(run)

    with pytest.raises(TypeError, match="run must be a ForwardResearchRun"):
        ForwardResearchReport(run="run", metrics=metrics)
    with pytest.raises(TypeError, match="metrics must be a tuple"):
        ForwardResearchReport(run=run, metrics=list(metrics))
    with pytest.raises(TypeError, match="must contain ForwardResearchStrategyHorizonMetrics"):
        ForwardResearchReport(run=run, metrics=("metrics",))


def test_report_rejects_a_missing_metrics_entry() -> None:
    run = _two_strategy_run()
    metrics = CalculateForwardResearchMetricsUseCase().execute(run)

    with pytest.raises(ValueError, match="one metrics entry for every run strategy and horizon"):
        ForwardResearchReport(run=run, metrics=metrics[:-1])


def test_report_rejects_a_surplus_metrics_entry() -> None:
    run = _two_strategy_run()
    metrics = CalculateForwardResearchMetricsUseCase().execute(run)

    with pytest.raises(ValueError, match="one metrics entry for every run strategy and horizon"):
        ForwardResearchReport(run=run, metrics=metrics + (metrics[0],))


def test_report_rejects_metrics_for_a_run_without_strategies() -> None:
    populated = _two_strategy_run()
    metrics = CalculateForwardResearchMetricsUseCase().execute(populated)

    with pytest.raises(ValueError, match="one metrics entry for every run strategy and horizon"):
        ForwardResearchReport(run=_run(), metrics=metrics)


def test_report_rejects_strategies_out_of_ascending_order() -> None:
    run = _two_strategy_run((ResearchHorizon(1),))
    alpha, zeta = CalculateForwardResearchMetricsUseCase().execute(run)

    with pytest.raises(ValueError, match="ascending strategy order"):
        ForwardResearchReport(run=run, metrics=(zeta, alpha))


def test_report_rejects_horizons_reordered_within_a_strategy() -> None:
    run = _two_strategy_run()
    first, second, third, fourth = CalculateForwardResearchMetricsUseCase().execute(run)

    with pytest.raises(ValueError, match="horizon order within each strategy"):
        ForwardResearchReport(run=run, metrics=(second, first, third, fourth))


def test_report_rejects_metrics_interleaved_across_strategies() -> None:
    run = _two_strategy_run()
    first, second, third, fourth = CalculateForwardResearchMetricsUseCase().execute(run)

    with pytest.raises(ValueError, match="ascending strategy order"):
        ForwardResearchReport(run=run, metrics=(first, third, second, fourth))


def test_report_rejects_a_strategy_absent_from_the_run() -> None:
    run = _two_strategy_run((ResearchHorizon(1),))
    alpha, zeta = CalculateForwardResearchMetricsUseCase().execute(run)
    impostor = dataclasses.replace(alpha, strategy_identity=StrategyIdentity("omega"))

    with pytest.raises(ValueError, match="ascending strategy order"):
        ForwardResearchReport(run=run, metrics=(impostor, zeta))


def test_report_rejects_a_horizon_absent_from_the_run() -> None:
    run = _two_strategy_run((ResearchHorizon(1),))
    alpha, zeta = CalculateForwardResearchMetricsUseCase().execute(run)
    wrong_horizon = dataclasses.replace(alpha, horizon=ResearchHorizon(9))

    with pytest.raises(ValueError, match="horizon order within each strategy"):
        ForwardResearchReport(run=run, metrics=(wrong_horizon, zeta))


def test_report_is_immutable() -> None:
    report = _report(_two_strategy_run())

    with pytest.raises(AttributeError):
        report.metrics = ()
