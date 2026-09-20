"""Tests for structured historical research report assembly."""

from __future__ import annotations

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
    AnalyzeMarketObservationContextService,
    BuildHistoricalResearchReportUseCase,
    CalculateHistoricalResearchMetricsUseCase,
    HistoricalResearchEvaluation,
    HistoricalResearchReport,
    HistoricalResearchRun,
    MeasureRecommendationOutcomeUseCase,
    RunHistoricalResearchUseCase,
)
from northstar_application.ports import (
    HistoricalMarketDataQuery,
    HistoricalMarketDataRepository,
)

_USD = Currency("USD")
_EUR = Currency("EUR")
_DAILY = Timeframe("1d")
_SYMBOL = Symbol("AAPL")
_EXCHANGE = ExchangeCode("NASDAQ")
_AVAILABLE_THROUGH = PointInTime("2026-03-01T16:00:00Z")


def _instant(day: int) -> PointInTime:
    return PointInTime(f"2026-01-{day:02d}T16:00:00Z")


def _bar(day: int, close: str, *, currency: Currency = _USD) -> HistoricalOHLCVBar:
    return HistoricalOHLCVBar(
        _SYMBOL,
        _EXCHANGE,
        _instant(day),
        _DAILY,
        Price("50", currency),
        Price("900", currency),
        Price("1", currency),
        Price(close, currency),
        Quantity("1000"),
    )


class StubRepository(HistoricalMarketDataRepository):
    """Returns bounded observations exactly as a conforming repository would."""

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


def _evaluation(day: int, *, strategy: str = "report-test") -> HistoricalResearchEvaluation:
    context = MarketObservationContext(
        ListingReference(_SYMBOL, _EXCHANGE),
        _instant(day),
        Price("100", _USD),
        Price("99", _USD),
        Quantity("1000"),
        Price("900", _USD),
        Price("1", _USD),
        tuple(Price("100", _USD) for _ in range(20)),
        tuple(Quantity("1000") for _ in range(20)),
    )
    result = AnalyzeMarketObservationContextService(
        strategy=Strategy(StrategyIdentity(strategy)),
        analysis_generator=AssetAnalysisGenerator(),
    ).execute(context)
    return HistoricalResearchEvaluation(replay_instant=_instant(day), result=result)


def _series() -> tuple[HistoricalOHLCVBar, ...]:
    return tuple(_bar(20 + offset, str(100 + offset * 10)) for offset in range(6))


def _run(
    evaluations: tuple[HistoricalResearchEvaluation, ...],
    horizons: tuple[ResearchHorizon, ...],
    observations: tuple[HistoricalOHLCVBar, ...] = (),
    *,
    available_through: PointInTime = _AVAILABLE_THROUGH,
) -> HistoricalResearchRun:
    use_case = RunHistoricalResearchUseCase(
        MeasureRecommendationOutcomeUseCase(StubRepository(observations))
    )
    return use_case.execute(evaluations, horizons, _DAILY, available_through)


def _report(run: HistoricalResearchRun) -> HistoricalResearchReport:
    return BuildHistoricalResearchReportUseCase().execute(run)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_execute_rejects_none_run() -> None:
    with pytest.raises(TypeError, match="run cannot be None"):
        BuildHistoricalResearchReportUseCase().execute(None)


def test_execute_rejects_wrong_run_type() -> None:
    with pytest.raises(TypeError, match="must be a HistoricalResearchRun"):
        BuildHistoricalResearchReportUseCase().execute("run")


def test_use_case_rejects_a_wrong_calculator_dependency() -> None:
    with pytest.raises(TypeError, match="must be a CalculateHistoricalResearchMetricsUseCase"):
        BuildHistoricalResearchReportUseCase("calculator")


# ---------------------------------------------------------------------------
# Non-empty report
# ---------------------------------------------------------------------------


def test_non_empty_report_pairs_the_run_with_its_metrics() -> None:
    run = _run((_evaluation(20), _evaluation(21)), (ResearchHorizon(1),), _series())

    report = _report(run)

    assert isinstance(report, HistoricalResearchReport)
    assert report.run is run
    assert len(report.metrics) == 1
    assert report.metrics[0].horizon == ResearchHorizon(1)
    assert report.metrics[0].total_count == 2


def test_report_metrics_match_the_existing_calculator_exactly() -> None:
    evaluations = (_evaluation(20), _evaluation(21))
    horizons = (ResearchHorizon(1), ResearchHorizon(2))
    run = _run(evaluations, horizons, _series())

    report = _report(run)
    directly = CalculateHistoricalResearchMetricsUseCase().execute(run)

    assert report.metrics == directly


def test_report_metrics_preserve_horizon_order() -> None:
    horizons = (ResearchHorizon(3), ResearchHorizon(1), ResearchHorizon(2))
    run = _run((_evaluation(20),), horizons, _series())

    report = _report(run)

    assert [entry.horizon for entry in report.metrics] == list(horizons)
    assert [entry.horizon for entry in report.metrics] == list(report.run.horizons)


# ---------------------------------------------------------------------------
# Empty-run report
# ---------------------------------------------------------------------------


def test_empty_run_produces_a_valid_report() -> None:
    run = _run((), (ResearchHorizon(1), ResearchHorizon(5)))

    report = _report(run)

    assert report.evaluation_count == 0
    assert len(report.metrics) == 2
    assert all(entry.total_count == 0 for entry in report.metrics)
    assert all(entry.average_forward_return is None for entry in report.metrics)


def test_empty_run_metadata_is_none_where_undefined() -> None:
    report = _report(_run((), (ResearchHorizon(1),)))

    assert report.listing_reference is None
    assert report.strategy_identity is None
    assert report.first_replay_instant is None
    assert report.last_replay_instant is None


def test_empty_run_still_reports_run_configuration() -> None:
    report = _report(_run((), (ResearchHorizon(1),)))

    assert report.timeframe == _DAILY
    assert report.available_through == _AVAILABLE_THROUGH


# ---------------------------------------------------------------------------
# Derived metadata
# ---------------------------------------------------------------------------


def test_listing_reference_is_derived_from_the_first_evaluation() -> None:
    report = _report(_run((_evaluation(20), _evaluation(21)), (ResearchHorizon(1),), _series()))

    assert report.listing_reference == ListingReference(_SYMBOL, _EXCHANGE)


def test_strategy_identity_is_derived_from_the_first_evaluation() -> None:
    run = _run((_evaluation(20, strategy="momentum"),), (ResearchHorizon(1),), _series())

    assert _report(run).strategy_identity == StrategyIdentity("momentum")


def test_timeframe_and_available_through_are_surfaced_from_the_run() -> None:
    available_through = PointInTime("2026-02-10T16:00:00Z")
    run = _run(
        (_evaluation(20),),
        (ResearchHorizon(1),),
        _series(),
        available_through=available_through,
    )

    report = _report(run)

    assert report.timeframe is run.timeframe
    assert report.available_through is run.available_through
    assert report.available_through == available_through


def test_replay_span_and_evaluation_count_are_derived() -> None:
    evaluations = (_evaluation(20), _evaluation(21), _evaluation(22))
    report = _report(_run(evaluations, (ResearchHorizon(1),), _series()))

    assert report.first_replay_instant == _instant(20)
    assert report.last_replay_instant == _instant(22)
    assert report.evaluation_count == 3


def test_single_evaluation_span_starts_and_ends_at_the_same_instant() -> None:
    report = _report(_run((_evaluation(20),), (ResearchHorizon(1),), _series()))

    assert report.first_replay_instant == report.last_replay_instant == _instant(20)


def test_report_does_not_duplicate_run_metadata_as_stored_fields() -> None:
    report = _report(_run((_evaluation(20),), (ResearchHorizon(1),), _series()))

    assert HistoricalResearchReport.__slots__ == ("run", "metrics")
    assert not hasattr(report, "__dict__")


# ---------------------------------------------------------------------------
# Report coherence
# ---------------------------------------------------------------------------


def test_report_rejects_a_wrong_metrics_count() -> None:
    run = _run((_evaluation(20),), (ResearchHorizon(1), ResearchHorizon(2)), _series())
    metrics = CalculateHistoricalResearchMetricsUseCase().execute(run)

    with pytest.raises(ValueError, match="one metrics entry for every run horizon"):
        HistoricalResearchReport(run=run, metrics=metrics[:1])


def test_report_rejects_missing_metrics() -> None:
    run = _run((_evaluation(20),), (ResearchHorizon(1),), _series())

    with pytest.raises(ValueError, match="one metrics entry for every run horizon"):
        HistoricalResearchReport(run=run, metrics=())


def test_report_rejects_duplicate_horizon_metrics() -> None:
    run = _run((_evaluation(20),), (ResearchHorizon(1), ResearchHorizon(2)), _series())
    metrics = CalculateHistoricalResearchMetricsUseCase().execute(run)

    with pytest.raises(ValueError, match="must follow the run horizon order"):
        HistoricalResearchReport(run=run, metrics=(metrics[0], metrics[0]))


def test_report_rejects_reordered_horizon_metrics() -> None:
    run = _run((_evaluation(20),), (ResearchHorizon(1), ResearchHorizon(2)), _series())
    metrics = CalculateHistoricalResearchMetricsUseCase().execute(run)

    with pytest.raises(ValueError, match="must follow the run horizon order"):
        HistoricalResearchReport(run=run, metrics=(metrics[1], metrics[0]))


def test_report_rejects_non_tuple_metrics() -> None:
    run = _run((_evaluation(20),), (ResearchHorizon(1),), _series())
    metrics = CalculateHistoricalResearchMetricsUseCase().execute(run)

    with pytest.raises(TypeError, match="metrics must be a tuple"):
        HistoricalResearchReport(run=run, metrics=list(metrics))


def test_report_rejects_wrong_metrics_element_type() -> None:
    run = _run((_evaluation(20),), (ResearchHorizon(1),), _series())

    with pytest.raises(TypeError, match="must contain HistoricalResearchHorizonMetrics values"):
        HistoricalResearchReport(run=run, metrics=("metrics",))


def test_report_rejects_wrong_run_type() -> None:
    with pytest.raises(TypeError, match="run must be a HistoricalResearchRun"):
        HistoricalResearchReport(run="run", metrics=())


# ---------------------------------------------------------------------------
# Preserved measurement semantics
# ---------------------------------------------------------------------------


def test_unavailable_counts_are_preserved_exactly() -> None:
    observations = (
        _bar(20, "100"),
        _bar(21, "110"),
        _bar(22, "100"),
        _bar(23, "120", currency=_EUR),
    )
    evaluations = (_evaluation(20), _evaluation(22), _evaluation(23))
    run = _run(evaluations, (ResearchHorizon(1),), observations)

    only = _report(run).metrics[0]

    assert only.total_count == 3
    assert only.measured_count == 1
    assert only.currency_mismatch_count == 1
    assert only.insufficient_future_observations_count == 1


def test_return_statistics_are_preserved_exactly() -> None:
    run = _run((_evaluation(20),), (ResearchHorizon(1),), _series())

    only = _report(run).metrics[0]

    assert only.average_forward_return == Percentage(10)
    assert only.median_forward_return == Percentage(10)
    assert only.minimum_forward_return == Percentage(10)
    assert only.maximum_forward_return == Percentage(10)


def test_none_statistics_are_preserved_for_unmeasurable_horizons() -> None:
    run = _run((_evaluation(20),), (ResearchHorizon(50),), _series())

    only = _report(run).metrics[0]

    assert only.measured_count == 0
    assert only.insufficient_future_observations_count == 1
    assert only.average_forward_return is None
    assert only.minimum_forward_return is None


def test_report_applies_no_success_interpretation_or_trading_semantics() -> None:
    report = _report(_run((_evaluation(20),), (ResearchHorizon(1),), _series()))

    for forbidden in (
        "accuracy",
        "win_rate",
        "is_successful",
        "success_count",
        "verdict",
        "pnl",
        "profit",
        "trades",
        "positions",
        "equity_curve",
        "markdown",
        "to_json",
        "render",
    ):
        assert not hasattr(report, forbidden)


# ---------------------------------------------------------------------------
# Determinism and value semantics
# ---------------------------------------------------------------------------


def test_repeated_builds_are_deterministic() -> None:
    evaluations = (_evaluation(20), _evaluation(21))
    horizons = (ResearchHorizon(1), ResearchHorizon(2))
    run = _run(evaluations, horizons, _series())

    assert _report(run) == _report(run)


def test_reports_from_equal_runs_compare_equal() -> None:
    evaluations = (_evaluation(20),)
    horizons = (ResearchHorizon(1),)

    left = _report(_run(evaluations, horizons, _series()))
    right = _report(_run(evaluations, horizons, _series()))

    assert left == right


def test_report_is_immutable() -> None:
    report = _report(_run((_evaluation(20),), (ResearchHorizon(1),), _series()))

    with pytest.raises(AttributeError):
        report.metrics = ()


def test_report_uses_an_injected_calculator_when_supplied() -> None:
    class RecordingCalculator(CalculateHistoricalResearchMetricsUseCase):
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, run: HistoricalResearchRun):
            self.calls += 1
            return super().execute(run)

    calculator = RecordingCalculator()
    run = _run((_evaluation(20),), (ResearchHorizon(1),), _series())

    BuildHistoricalResearchReportUseCase(calculator).execute(run)

    assert calculator.calls == 1
