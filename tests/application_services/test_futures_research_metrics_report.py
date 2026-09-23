"""Tests for futures historical research metrics and the report built from them.

Most runs here are assembled directly from hand-built analysis results and
outcomes, which gives exact control over which outcomes are measured,
insufficient or of undefined basis. The run value validates them, so every
fixture is a run the research pipeline could legitimately have produced. One
end-to-end test proves the same through the real replay-analyse-measure run.
"""

from __future__ import annotations

import ast
import dataclasses
from datetime import date, timedelta
from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_HALF_UP, Decimal, localcontext
from pathlib import Path

import pytest
from northstar_core.derivatives import ExpirationDate, QuoteValue
from northstar_core.foundation.value_objects import (
    ExchangeCode,
    Percentage,
    PointInTime,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.futures import FuturesContract, FuturesOHLCVBar, FuturesProductReference
from northstar_core.strategy import (
    FuturesAssetAnalysis,
    FuturesAssetAnalysisGenerator,
    FuturesMarketObservationContext,
    FuturesRecommendationOutcome,
    ResearchHorizon,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services import (
    BuildFuturesHistoricalResearchReportUseCase,
    CalculateFuturesHistoricalResearchMetricsUseCase,
    FuturesAnalysisResult,
    FuturesHistoricalResearchReport,
    FuturesHistoricalResearchRun,
    FuturesHistoricalResearchStrategyHorizonMetrics,
    RunFuturesHistoricalResearchUseCase,
)
from northstar_application.ports import FuturesHistoricalMarketDataRepository

_ES_DEC = FuturesContract(
    FuturesProductReference(Symbol("ES"), ExchangeCode("CME")), ExpirationDate("2026-12-18")
)
_DAILY = Timeframe("1d")
_STRATEGY = Strategy(StrategyIdentity("futures-directional"))
_H1, _H2, _H3 = ResearchHorizon(1), ResearchHorizon(2), ResearchHorizon(3)


def _instant(session: int) -> PointInTime:
    return PointInTime(f"{date(2026, 6, 1) + timedelta(days=session - 1)}T21:00:00Z")


def _quote(value: str) -> QuoteValue:
    return QuoteValue(Decimal(value))


def _percent(value: str) -> Percentage:
    return Percentage(Decimal(value))


def _result(
    session: int,
    decision_quote: str = "100",
    *,
    signal: str = "neutral trend",
    strategy: Strategy = _STRATEGY,
) -> FuturesAnalysisResult:
    instant = _instant(session)
    quote = _quote(decision_quote)
    context = FuturesMarketObservationContext(
        contract=_ES_DEC,
        timeframe=_DAILY,
        observed_at=instant,
        latest_quote=quote,
        previous_close=quote,
        latest_volume=Quantity(Decimal("1000")),
        session_high=quote,
        session_low=quote,
        recent_closes=(quote,) * 20,
        recent_volumes=(Quantity(Decimal("1000")),) * 20,
    )
    recommendation = strategy.evaluate_futures(FuturesAssetAnalysis(_ES_DEC, instant, (signal,)))
    return FuturesAnalysisResult(recommendation, context)


def _outcome(
    result: FuturesAnalysisResult, horizon: ResearchHorizon, evaluation: str | None
) -> FuturesRecommendationOutcome:
    """Measure ``result`` at ``horizon``; ``None`` means the horizon was not reached."""
    decision_quote = result.market_observation_context.latest_quote
    if evaluation is None:
        return FuturesRecommendationOutcome(result.recommendation, horizon, decision_quote)
    decision_session = (
        date.fromisoformat(result.recommendation.point_in_time.value[:10]) - date(2026, 6, 1)
    ).days + 1
    return FuturesRecommendationOutcome(
        result.recommendation,
        horizon,
        decision_quote,
        _instant(decision_session + horizon.observations),
        _quote(evaluation),
    )


def _run(
    decisions: list[tuple[int, str]],
    horizons: tuple[ResearchHorizon, ...],
    evaluations: dict[tuple[int, ResearchHorizon], str | None],
    *,
    through: int = 40,
    signal: str = "neutral trend",
    strategy: Strategy = _STRATEGY,
) -> FuturesHistoricalResearchRun:
    """A run of ``(session, decision quote)`` decisions and their chosen evaluations.

    ``evaluations`` maps ``(decision index, horizon)`` to an evaluation quote,
    or to None for a horizon not reached. Missing pairs are not reached.
    """
    results = tuple(
        _result(session, quote, signal=signal, strategy=strategy) for session, quote in decisions
    )
    outcomes = tuple(
        _outcome(result, horizon, evaluations.get((index, horizon)))
        for index, result in enumerate(results)
        for horizon in horizons
    )
    return FuturesHistoricalResearchRun(
        contract=_ES_DEC,
        timeframe=_DAILY,
        strategy_identity=strategy.strategy_identity,
        available_through=_instant(through),
        horizons=horizons,
        analysis_results=results,
        outcomes=outcomes,
    )


def _metrics(run: FuturesHistoricalResearchRun) -> tuple:
    return CalculateFuturesHistoricalResearchMetricsUseCase().execute(run)


def _single(run: FuturesHistoricalResearchRun) -> FuturesHistoricalResearchStrategyHorizonMetrics:
    (entry,) = _metrics(run)
    return entry


def _counts(entry: FuturesHistoricalResearchStrategyHorizonMetrics) -> tuple[int, int, int, int]:
    return (
        entry.total_count,
        entry.measured_count,
        entry.insufficient_future_observations_count,
        entry.undefined_return_basis_count,
    )


def _statistics(entry: FuturesHistoricalResearchStrategyHorizonMetrics) -> tuple:
    return (
        entry.average_forward_return,
        entry.median_forward_return,
        entry.minimum_forward_return,
        entry.maximum_forward_return,
    )


# ---------------------------------------------------------------------------
# Empty runs
# ---------------------------------------------------------------------------


def test_an_empty_run_reports_every_requested_horizon_with_nothing_in_it() -> None:
    run = _run([], (_H3, _H1, _H2), {})

    metrics = _metrics(run)

    assert [entry.horizon for entry in metrics] == [_H3, _H1, _H2]
    for entry in metrics:
        assert entry.strategy_identity == _STRATEGY.strategy_identity
        assert _counts(entry) == (0, 0, 0, 0)
        assert _statistics(entry) == (None, None, None, None)


def test_an_empty_run_report_still_names_what_it_covered() -> None:
    run = _run([], (_H1,), {}, through=19)

    report = BuildFuturesHistoricalResearchReportUseCase().execute(run)

    assert report.contract == _ES_DEC
    assert report.strategy_identity == _STRATEGY.strategy_identity
    assert report.timeframe == _DAILY
    assert report.available_through == _instant(19)
    assert report.horizons == (_H1,)
    assert report.decision_count == 0
    assert report.first_decision_instant is None
    assert report.last_decision_instant is None
    assert _counts(report.metrics[0]) == (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# Single-state runs
# ---------------------------------------------------------------------------


def test_an_all_insufficient_run_counts_every_outcome_as_insufficient() -> None:
    run = _run([(20, "100"), (21, "105"), (22, "110")], (_H3,), {}, through=22)

    entry = _single(run)

    assert _counts(entry) == (3, 0, 3, 0)
    assert _statistics(entry) == (None, None, None, None)


def test_an_all_undefined_basis_run_counts_every_outcome_as_undefined() -> None:
    decisions = [(20, "0"), (21, "-10"), (22, "-37.63")]
    evaluations = {(0, _H1): "10", (1, _H1): "-5", (2, _H1): "20"}

    entry = _single(_run(decisions, (_H1,), evaluations))

    assert _counts(entry) == (3, 0, 0, 3)
    assert _statistics(entry) == (None, None, None, None)


def test_positive_and_negative_returns_are_summarised_without_clipping() -> None:
    decisions = [(20, "100"), (21, "100"), (22, "10"), (23, "10")]
    evaluations = {(0, _H1): "110", (1, _H1): "90", (2, _H1): "0", (3, _H1): "-5"}

    entry = _single(_run(decisions, (_H1,), evaluations))

    # Returns: +10, -10, -100, -150.
    assert _counts(entry) == (4, 4, 0, 0)
    assert entry.average_forward_return == _percent("-62.5")
    assert entry.median_forward_return == _percent("-55")
    assert entry.minimum_forward_return == _percent("-150")
    assert entry.maximum_forward_return == _percent("10")


# ---------------------------------------------------------------------------
# Mixed states
# ---------------------------------------------------------------------------


def _mixed_run() -> FuturesHistoricalResearchRun:
    """Horizon 1: two measured (+10, -150), one undefined basis, two insufficient."""
    decisions = [(20, "100"), (21, "10"), (22, "0"), (23, "100"), (24, "100")]
    evaluations = {(0, _H1): "110", (1, _H1): "-5", (2, _H1): "50"}
    return _run(decisions, (_H1,), evaluations, through=24)


def test_a_mixed_horizon_counts_each_state_exactly() -> None:
    entry = _single(_mixed_run())

    assert _counts(entry) == (5, 2, 2, 1)
    assert entry.total_count == (
        entry.measured_count
        + entry.insufficient_future_observations_count
        + entry.undefined_return_basis_count
    )


def test_a_mixed_horizon_summarises_measured_returns_only() -> None:
    entry = _single(_mixed_run())

    assert entry.average_forward_return == _percent("-70")
    assert entry.median_forward_return == _percent("-70")
    assert entry.minimum_forward_return == _percent("-150")
    assert entry.maximum_forward_return == _percent("10")


def test_a_fabricated_zero_for_the_unmeasured_would_have_changed_every_statistic() -> None:
    """Guard: the mixed-state test is load-bearing only if this holds."""
    measured = [Decimal("10"), Decimal("-150")]
    with_zeros = sorted([*measured, Decimal(0), Decimal(0), Decimal(0)])

    assert sum(with_zeros) / len(with_zeros) != Decimal("-70")
    assert with_zeros[len(with_zeros) // 2] != Decimal("-70")
    assert max(with_zeros) == Decimal("10")  # max alone would not have caught it


def test_the_partition_holds_for_every_horizon_of_a_multi_horizon_run() -> None:
    decisions = [(20, "100"), (21, "0"), (22, "100")]
    evaluations = {(0, _H1): "101", (0, _H2): "102", (1, _H1): "5", (2, _H1): "99"}

    for entry in _metrics(_run(decisions, (_H1, _H2, _H3), evaluations, through=23)):
        assert entry.total_count == 3
        assert entry.total_count == sum(_counts(entry)[1:])


# ---------------------------------------------------------------------------
# Horizon grouping and ordering
# ---------------------------------------------------------------------------


def test_metrics_follow_the_callers_horizon_order() -> None:
    horizons = (_H3, _H1, _H2)
    evaluations = {(0, _H1): "110", (0, _H2): "120", (0, _H3): "130"}

    metrics = _metrics(_run([(20, "100")], horizons, evaluations))

    assert [entry.horizon for entry in metrics] == list(horizons)
    assert [entry.average_forward_return for entry in metrics] == [
        _percent("30"),
        _percent("10"),
        _percent("20"),
    ]


def test_distinct_horizons_are_never_merged() -> None:
    evaluations = {(0, _H1): "110", (0, _H2): "90"}

    one, two = _metrics(_run([(20, "100")], (_H1, _H2), evaluations))

    assert one.average_forward_return == _percent("10")
    assert two.average_forward_return == _percent("-10")
    assert one.total_count == two.total_count == 1


def test_a_rebuilt_equal_horizon_groups_as_the_same_horizon() -> None:
    rebuilt = ResearchHorizon(1)
    assert rebuilt is not _H1
    run = _run([(20, "100"), (21, "100")], (rebuilt,), {(0, _H1): "110", (1, _H1): "90"})

    entry = _single(run)

    assert entry.horizon == _H1
    assert _counts(entry) == (2, 2, 0, 0)


def test_every_outcome_is_counted_under_its_own_horizon_exactly_once() -> None:
    run = _mixed_run()
    multi = _run(
        [(20, "100"), (21, "100")], (_H2, _H1), {(0, _H1): "110", (1, _H2): "120"}, through=23
    )

    for candidate in (run, multi):
        metrics = _metrics(candidate)
        assert sum(entry.total_count for entry in metrics) == len(candidate.outcomes)
        for entry in metrics:
            assert entry.total_count == sum(
                1 for outcome in candidate.outcomes if outcome.horizon == entry.horizon
            )


def test_metrics_are_labelled_with_the_run_strategy() -> None:
    strategy = Strategy(StrategyIdentity("another-strategy"))

    entry = _single(_run([(20, "100")], (_H1,), {(0, _H1): "110"}, strategy=strategy))

    assert entry.strategy_identity == StrategyIdentity("another-strategy")


# ---------------------------------------------------------------------------
# Action independence
# ---------------------------------------------------------------------------


def test_buy_hold_and_sell_runs_summarise_identically() -> None:
    decisions = [(20, "100"), (21, "10"), (22, "0")]
    evaluations = {(0, _H1): "90", (1, _H1): "-5", (2, _H1): "5"}

    runs = [
        _run(decisions, (_H1,), evaluations, signal=signal)
        for signal in ("strong bullish", "neutral trend", "strong bearish")
    ]
    assert {run.analysis_results[0].recommendation.action.value for run in runs} == {
        "BUY",
        "HOLD",
        "SELL",
    }

    summaries = [_metrics(run) for run in runs]

    assert summaries[0] == summaries[1] == summaries[2]
    assert summaries[0][0].average_forward_return == _percent("-80")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def _repeating_run() -> FuturesHistoricalResearchRun:
    """Returns of 33.3..%, 28.57..% and 66.6..%: every average and median repeats."""
    decisions = [(20, "3"), (21, "7"), (22, "3")]
    evaluations = {(0, _H1): "4", (1, _H1): "9", (2, _H1): "5"}
    return _run(decisions, (_H1,), evaluations)


@pytest.mark.parametrize("rounding", [ROUND_DOWN, ROUND_CEILING, ROUND_HALF_UP])
@pytest.mark.parametrize("precision", [6, 28, 50])
def test_caller_decimal_context_does_not_alter_the_statistics(
    precision: int, rounding: str
) -> None:
    expected = _metrics(_repeating_run())

    with localcontext() as caller:
        caller.prec = precision
        caller.rounding = rounding
        produced = _metrics(_repeating_run())

    assert produced == expected


def test_low_ambient_precision_would_have_changed_the_average() -> None:
    """Guard: the determinism test is load-bearing only if this holds."""
    entry = _single(_repeating_run())
    returns = sorted(
        outcome.forward_return.value
        for outcome in _repeating_run().outcomes
        if outcome.forward_return is not None
    )

    with localcontext() as caller:
        caller.prec = 6
        ambient = sum(returns, Decimal()) / len(returns)

    assert Percentage(ambient) != entry.average_forward_return


def test_repeated_generation_is_identical() -> None:
    run = _mixed_run()
    builder = BuildFuturesHistoricalResearchReportUseCase()

    first = builder.execute(run)

    assert builder.execute(run) == first
    assert BuildFuturesHistoricalResearchReportUseCase().execute(_mixed_run()) == first
    assert repr(builder.execute(run)) == repr(first)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def test_the_report_pairs_the_run_with_its_metrics_and_derives_its_facts() -> None:
    run = _mixed_run()

    report = BuildFuturesHistoricalResearchReportUseCase().execute(run)

    assert report.run is run
    assert report.metrics == _metrics(run)
    assert report.contract == run.contract
    assert report.strategy_identity == run.strategy_identity
    assert report.available_through == run.available_through == _instant(24)
    assert report.decision_count == 5
    assert report.first_decision_instant == _instant(20)
    assert report.last_decision_instant == _instant(24)
    assert set(FuturesHistoricalResearchReport.__slots__) == {"run", "metrics"}


def test_the_report_cutoff_is_the_runs_not_the_last_decision_or_evaluation() -> None:
    """Decisions end at session 21 and evaluations at 22; the run looked through 30."""
    run = _run([(20, "100"), (21, "100")], (_H1,), {(0, _H1): "110", (1, _H1): "120"}, through=30)

    report = BuildFuturesHistoricalResearchReportUseCase().execute(run)

    assert report.available_through == _instant(30)
    assert report.last_decision_instant == _instant(21)
    assert report.available_through != report.last_decision_instant
    assert report.available_through != run.outcomes[-1].evaluation_instant


def test_the_report_keeps_insufficient_outcomes_insufficient_at_the_run_cutoff() -> None:
    report = BuildFuturesHistoricalResearchReportUseCase().execute(_mixed_run())

    assert report.metrics[0].insufficient_future_observations_count == 2
    assert report.available_through == _instant(24)


def test_the_report_metrics_follow_the_callers_horizon_order() -> None:
    run = _run([(20, "100")], (_H3, _H1, _H2), {})

    report = BuildFuturesHistoricalResearchReportUseCase().execute(run)

    assert [entry.horizon for entry in report.metrics] == [_H3, _H1, _H2]
    assert report.horizons == (_H3, _H1, _H2)


def test_the_builder_uses_an_injected_calculator() -> None:
    class RecordingCalculator(CalculateFuturesHistoricalResearchMetricsUseCase):
        def __init__(self) -> None:
            self.runs: list[FuturesHistoricalResearchRun] = []

        def execute(self, run):
            self.runs.append(run)
            return super().execute(run)

    calculator = RecordingCalculator()
    run = _mixed_run()

    BuildFuturesHistoricalResearchReportUseCase(calculator).execute(run)

    assert calculator.runs == [run]


def _entry(**overrides: object) -> FuturesHistoricalResearchStrategyHorizonMetrics:
    return dataclasses.replace(_single(_mixed_run()), **overrides)


@pytest.mark.parametrize(
    ("metrics_for", "message"),
    [
        pytest.param(lambda run, m: m[:-1], "one metrics entry", id="missing-entry"),
        pytest.param(lambda run, m: tuple(reversed(m)), "horizon order", id="reordered"),
        pytest.param(
            lambda run, m: (
                dataclasses.replace(m[0], strategy_identity=StrategyIdentity("impostor")),
                *m[1:],
            ),
            "run strategy",
            id="another-strategy",
        ),
        pytest.param(
            lambda run, m: (
                dataclasses.replace(
                    m[0],
                    total_count=m[0].total_count + 1,
                    insufficient_future_observations_count=(
                        m[0].insufficient_future_observations_count + 1
                    ),
                ),
                *m[1:],
            ),
            "every run outcome",
            id="miscounted",
        ),
    ],
)
def test_an_incoherent_report_is_rejected(metrics_for, message: str) -> None:
    run = _run([(20, "100"), (21, "100")], (_H1, _H2), {(0, _H1): "110"}, through=23)
    metrics = _metrics(run)

    with pytest.raises(ValueError, match=message):
        FuturesHistoricalResearchReport(run, metrics_for(run, metrics))


@pytest.mark.parametrize(("run", "metrics"), [("run", ()), (None, ()), ("mixed", [])])
def test_the_report_type_checks_its_fields(run: object, metrics: object) -> None:
    actual_run = _mixed_run() if run == "mixed" else run

    with pytest.raises(TypeError):
        FuturesHistoricalResearchReport(actual_run, metrics)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Metrics value invariants
# ---------------------------------------------------------------------------


def test_counts_that_do_not_partition_the_total_are_rejected() -> None:
    with pytest.raises(ValueError, match="must equal the total count"):
        _entry(total_count=6)


def test_statistics_without_a_measured_outcome_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be None when no outcome was measured"):
        _entry(measured_count=0, insufficient_future_observations_count=4)


def test_a_measured_outcome_without_statistics_is_rejected() -> None:
    with pytest.raises(ValueError, match="required when an outcome was measured"):
        _entry(median_forward_return=None)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("strategy_identity", "futures-directional", TypeError),
        ("horizon", 1, TypeError),
        ("total_count", True, TypeError),
        ("measured_count", 2.0, TypeError),
        ("undefined_return_basis_count", -1, ValueError),
        ("average_forward_return", Decimal("-70"), TypeError),
    ],
)
def test_metrics_fields_are_validated(field: str, value: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        _entry(**{field: value})


def test_the_metrics_have_no_currency_or_profit_concept() -> None:
    names = {
        field.name for field in dataclasses.fields(FuturesHistoricalResearchStrategyHorizonMetrics)
    }

    for absent in ("currency_mismatch_count", "currency", "pnl", "win_rate", "multiplier"):
        assert absent not in names


# ---------------------------------------------------------------------------
# End to end through the real research run
# ---------------------------------------------------------------------------


class ConformingRepository(FuturesHistoricalMarketDataRepository):
    def __init__(self, bars: tuple[FuturesOHLCVBar, ...]) -> None:
        self.bars = bars

    def get_bars(self, query):
        return tuple(bar for bar in self.bars if query.covers(bar.point_in_time))


def test_a_real_run_reports_measured_and_insufficient_outcomes_consistently() -> None:
    def bar(session: int, close: int) -> FuturesOHLCVBar:
        quote = Decimal(close)
        return FuturesOHLCVBar(
            contract=_ES_DEC,
            point_in_time=_instant(session),
            timeframe=_DAILY,
            open=QuoteValue(quote),
            high=QuoteValue(quote + 1),
            low=QuoteValue(quote - 1),
            close=QuoteValue(quote),
            volume=Quantity(Decimal("1000")),
        )

    history = tuple(bar(session, 100 + session % 7) for session in range(1, 26))
    run = RunFuturesHistoricalResearchUseCase(
        ConformingRepository(history), FuturesAssetAnalysisGenerator()
    ).execute(_ES_DEC, _DAILY, _STRATEGY, (_H3, _H1), _instant(25))

    report = BuildFuturesHistoricalResearchReportUseCase().execute(run)

    three, one = report.metrics
    assert (three.horizon, one.horizon) == (_H3, _H1)
    assert _counts(three) == (6, 3, 3, 0)  # decisions 20..25; 23..25 cannot reach +3
    assert _counts(one) == (6, 5, 1, 0)  # only decision 25 cannot reach +1
    assert report.decision_count == 6
    assert report.available_through == _instant(25)


# ---------------------------------------------------------------------------
# Boundaries and exports
# ---------------------------------------------------------------------------


def _tree(module_name: str) -> ast.Module:
    import importlib

    module = importlib.import_module(f"northstar_application.application_services.{module_name}")
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


_MODULES = ("calculate_futures_research_metrics", "build_futures_research_report")


@pytest.mark.parametrize("module_name", _MODULES)
def test_metrics_and_report_derive_from_the_run_alone(module_name: str) -> None:
    tree = _tree(module_name)
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    for forbidden in (
        "FuturesHistoricalMarketDataRepository",
        "FuturesHistoricalMarketDataQuery",
        "FuturesHistoricalMarketDataSource",
        "FuturesHistoricalMarketDataStore",
        "ReplayFuturesHistoricalMarketDataUseCase",
        "MeasureFuturesRecommendationOutcomeUseCase",
        "RunFuturesHistoricalResearchUseCase",
        "AcquireFuturesDailyHistoryUseCase",
    ):
        assert forbidden not in names
    for module in modules:
        assert module.split(".")[0] not in {
            "databento",
            "exchange_calendars",
            "sqlite3",
            "northstar_infrastructure",
            "time",
            "datetime",
            "random",
            "decimal",
        }
        assert not module.startswith("northstar_application.ports")
        assert "paper_trading" not in module
        assert "execution" not in module


@pytest.mark.parametrize("module_name", _MODULES)
def test_metrics_and_report_never_read_the_action_or_a_clock(module_name: str) -> None:
    for node in ast.walk(_tree(module_name)):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"action", "now", "today", "utcnow"}
        if isinstance(node, ast.Name):
            assert node.id != "RecommendationAction"


def test_the_statistics_mathematics_is_the_shared_helper() -> None:
    import northstar_application.application_services._return_statistics as statistics
    import northstar_application.application_services.calculate_futures_research_metrics as m

    assert m.average is statistics.average
    assert m.median is statistics.median
    for node in ast.walk(_tree("calculate_futures_research_metrics")):
        assert not (isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Mult)))


def test_the_metrics_and_report_are_exported_and_helpers_are_not() -> None:
    import northstar_application.application_services as services

    for name in (
        "FuturesHistoricalResearchStrategyHorizonMetrics",
        "CalculateFuturesHistoricalResearchMetricsUseCase",
        "FuturesHistoricalResearchReport",
        "BuildFuturesHistoricalResearchReportUseCase",
    ):
        assert name in services.__all__
    for private in ("average", "median", "_INSUFFICIENT", "_UNDEFINED_BASIS"):
        assert private not in services.__all__


@pytest.mark.parametrize(
    "call",
    [
        lambda: CalculateFuturesHistoricalResearchMetricsUseCase().execute("run"),
        lambda: BuildFuturesHistoricalResearchReportUseCase().execute(None),
        lambda: BuildFuturesHistoricalResearchReportUseCase("calculator"),
    ],
    ids=["calculator-run", "builder-run", "builder-dependency"],
)
def test_inputs_are_type_checked(call) -> None:
    with pytest.raises(TypeError):
        call()
