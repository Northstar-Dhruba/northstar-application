"""Tests for futures forward research metrics and the report built from them.

Most runs here are assembled directly from hand-built frozen records and
measurements, which gives exact control over which measurements are MEASURED,
PENDING or UNAVAILABLE. The run value validates them, so every fixture is a run
the forward pipeline could legitimately have produced. One end-to-end test
shows the same frozen decisions reporting differently at two cutoffs.
"""

from __future__ import annotations

import ast
import dataclasses
import importlib
from datetime import date, timedelta
from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_HALF_UP, Decimal, localcontext
from functools import cmp_to_key
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
    FuturesMarketObservationContext,
    FuturesRecommendationOutcome,
    ResearchHorizon,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services import (
    BuildFuturesForwardResearchReportUseCase,
    CalculateFuturesForwardResearchMetricsUseCase,
    FuturesAnalysisResult,
    FuturesForwardResearchMeasurement,
    FuturesForwardResearchRecord,
    FuturesForwardResearchReport,
    FuturesForwardResearchRun,
    FuturesForwardResearchStrategyHorizonMetrics,
    RunFuturesForwardResearchUseCase,
)
from northstar_application.ports import (
    FuturesForwardResearchRecordQuery,
    FuturesForwardResearchRecordRepository,
    FuturesHistoricalMarketDataRepository,
)

_ES_DEC = FuturesContract(
    FuturesProductReference(Symbol("ES"), ExchangeCode("CME")), ExpirationDate("2026-12-18")
)
_DAILY = Timeframe("1d")
_QUERY = FuturesForwardResearchRecordQuery(_ES_DEC, _DAILY)
_H1, _H2, _H3 = ResearchHorizon(1), ResearchHorizon(2), ResearchHorizon(3)


def _instant(session: int) -> PointInTime:
    return PointInTime(f"{date(2026, 9, 1) + timedelta(days=session - 1)}T21:00:00Z")


def _percent(value: str) -> Percentage:
    return Percentage(Decimal(value))


def _record(
    session: int,
    decision_quote: str = "100",
    *,
    strategy: str = "alpha",
    signal: str = "strong bullish",
) -> FuturesForwardResearchRecord:
    instant = _instant(session)
    quote = QuoteValue(Decimal(decision_quote))
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
    recommendation = Strategy(StrategyIdentity(strategy)).evaluate_futures(
        FuturesAssetAnalysis(_ES_DEC, instant, (signal,))
    )
    return FuturesForwardResearchRecord(FuturesAnalysisResult(recommendation, context))


def _session(record: FuturesForwardResearchRecord) -> int:
    return (date.fromisoformat(record.decision_instant.value[:10]) - date(2026, 9, 1)).days + 1


def _measurement(
    record: FuturesForwardResearchRecord, horizon: ResearchHorizon, evaluation: str | None
) -> FuturesForwardResearchMeasurement:
    quote = record.result.market_observation_context.latest_quote
    if evaluation is None:
        outcome = FuturesRecommendationOutcome(record.result.recommendation, horizon, quote)
    else:
        outcome = FuturesRecommendationOutcome(
            record.result.recommendation,
            horizon,
            quote,
            _instant(_session(record) + horizon.observations),
            QuoteValue(Decimal(evaluation)),
        )
    return FuturesForwardResearchMeasurement(record, horizon, outcome)


def _canonical(left: FuturesForwardResearchRecord, right: FuturesForwardResearchRecord) -> int:
    instant = left.decision_instant.compare(right.decision_instant)
    if instant:
        return instant
    a, b = left.strategy_identity.identity, right.strategy_identity.identity
    return (a > b) - (a < b)


def _run(
    decisions: list[FuturesForwardResearchRecord],
    horizons: tuple[ResearchHorizon, ...],
    evaluations: dict[tuple[str, int, ResearchHorizon], str | None],
    *,
    through: int = 40,
) -> FuturesForwardResearchRun:
    """A run whose evaluations are keyed by (strategy, decision session, horizon).

    A missing key is a horizon not yet reached, so the measurement is pending.
    """
    records = tuple(sorted(decisions, key=cmp_to_key(_canonical)))
    measurements = tuple(
        _measurement(
            record,
            horizon,
            evaluations.get((record.strategy_identity.identity, _session(record), horizon)),
        )
        for record in records
        for horizon in horizons
    )
    return FuturesForwardResearchRun(_QUERY, horizons, _instant(through), records, measurements)


def _metrics(
    run: FuturesForwardResearchRun,
) -> tuple[FuturesForwardResearchStrategyHorizonMetrics, ...]:
    return CalculateFuturesForwardResearchMetricsUseCase().execute(run)


def _single(run: FuturesForwardResearchRun) -> FuturesForwardResearchStrategyHorizonMetrics:
    (entry,) = _metrics(run)
    return entry


def _counts(entry: FuturesForwardResearchStrategyHorizonMetrics) -> tuple[int, int, int, int]:
    return (
        entry.total_count,
        entry.measured_count,
        entry.pending_count,
        entry.undefined_return_basis_count,
    )


def _statistics(entry: FuturesForwardResearchStrategyHorizonMetrics) -> tuple:
    return (
        entry.average_forward_return,
        entry.median_forward_return,
        entry.minimum_forward_return,
        entry.maximum_forward_return,
    )


# ---------------------------------------------------------------------------
# Grouping and ordering
# ---------------------------------------------------------------------------


def test_one_strategy_at_one_horizon() -> None:
    entry = _single(_run([_record(10)], (_H1,), {("alpha", 10, _H1): "110"}))

    assert entry.strategy_identity == StrategyIdentity("alpha")
    assert entry.horizon == _H1
    assert _counts(entry) == (1, 1, 0, 0)
    assert entry.average_forward_return == _percent("10")


def test_horizons_keep_the_callers_order_within_a_strategy() -> None:
    evaluations = {("alpha", 10, _H1): "110", ("alpha", 10, _H2): "120", ("alpha", 10, _H3): "130"}

    metrics = _metrics(_run([_record(10)], (_H3, _H1, _H2), evaluations))

    assert [entry.horizon for entry in metrics] == [_H3, _H1, _H2]
    assert [entry.average_forward_return for entry in metrics] == [
        _percent("30"),
        _percent("10"),
        _percent("20"),
    ]


def test_strategies_are_grouped_separately_and_ordered_ascending_by_identity() -> None:
    decisions = [
        _record(10, strategy="zeta"),
        _record(11, strategy="alpha"),
        _record(12, strategy="mu"),
    ]
    evaluations = {("zeta", 10, _H1): "90", ("alpha", 11, _H1): "110", ("mu", 12, _H1): "105"}

    metrics = _metrics(_run(decisions, (_H3, _H1), evaluations))

    assert [(entry.strategy_identity.identity, entry.horizon) for entry in metrics] == [
        ("alpha", _H3),
        ("alpha", _H1),
        ("mu", _H3),
        ("mu", _H1),
        ("zeta", _H3),
        ("zeta", _H1),
    ]
    by_key = {(entry.strategy_identity.identity, entry.horizon): entry for entry in metrics}
    assert by_key[("alpha", _H1)].average_forward_return == _percent("10")
    assert by_key[("zeta", _H1)].average_forward_return == _percent("-10")
    assert by_key[("mu", _H1)].average_forward_return == _percent("5")


def test_two_strategies_at_one_instant_are_never_pooled() -> None:
    decisions = [_record(10, strategy="beta"), _record(10, strategy="alpha")]
    evaluations = {("alpha", 10, _H1): "110", ("beta", 10, _H1): "80"}

    alpha, beta = _metrics(_run(decisions, (_H1,), evaluations))

    assert (alpha.strategy_identity.identity, beta.strategy_identity.identity) == ("alpha", "beta")
    assert alpha.average_forward_return == _percent("10")
    assert beta.average_forward_return == _percent("-20")
    assert alpha.total_count == beta.total_count == 1


def test_an_empty_run_yields_no_metrics_and_invents_no_strategy() -> None:
    assert _metrics(_run([], (_H3, _H1), {})) == ()


def test_a_rebuilt_equal_horizon_and_strategy_group_by_value() -> None:
    rebuilt = ResearchHorizon(1)
    run = _run(
        [_record(10), _record(11)],
        (rebuilt,),
        {("alpha", 10, _H1): "110", ("alpha", 11, _H1): "90"},
    )

    entry = _single(run)

    assert entry.horizon == _H1
    assert entry.strategy_identity == StrategyIdentity("alpha")
    assert _counts(entry) == (2, 2, 0, 0)


# ---------------------------------------------------------------------------
# States, counts and statistics
# ---------------------------------------------------------------------------


def test_an_all_pending_group_has_no_statistics() -> None:
    entry = _single(_run([_record(10), _record(11), _record(12)], (_H3,), {}))

    assert _counts(entry) == (3, 0, 3, 0)
    assert _statistics(entry) == (None, None, None, None)


def test_an_all_undefined_basis_group_has_no_statistics() -> None:
    decisions = [_record(10, "0"), _record(11, "-10"), _record(12, "-37.63")]
    evaluations = {("alpha", 10, _H1): "10", ("alpha", 11, _H1): "-5", ("alpha", 12, _H1): "20"}

    entry = _single(_run(decisions, (_H1,), evaluations))

    assert _counts(entry) == (3, 0, 0, 3)
    assert _statistics(entry) == (None, None, None, None)


def _mixed_run() -> FuturesForwardResearchRun:
    """One group: +10% and -150% measured, one pending, one undefined basis."""
    decisions = [_record(10, "100"), _record(11, "10"), _record(12, "100"), _record(13, "0")]
    evaluations = {("alpha", 10, _H1): "110", ("alpha", 11, _H1): "-5", ("alpha", 13, _H1): "50"}
    return _run(decisions, (_H1,), evaluations)


def test_a_mixed_group_counts_each_state_exactly() -> None:
    entry = _single(_mixed_run())

    assert _counts(entry) == (4, 2, 1, 1)
    assert entry.total_count == (
        entry.measured_count + entry.pending_count + entry.undefined_return_basis_count
    )


def test_a_mixed_group_summarises_measured_returns_only() -> None:
    entry = _single(_mixed_run())

    assert entry.average_forward_return == _percent("-70")
    assert entry.median_forward_return == _percent("-70")
    assert entry.minimum_forward_return == _percent("-150")
    assert entry.maximum_forward_return == _percent("10")


def test_zeros_for_the_unmeasured_would_have_changed_the_statistics() -> None:
    """Guard: the mixed test is load-bearing only if fake zeros would be visible."""
    padded = sorted([Decimal("10"), Decimal("-150"), Decimal(0), Decimal(0)])

    assert sum(padded) / len(padded) != Decimal("-70")
    assert (padded[1] + padded[2]) / 2 != Decimal("-70")


def test_returns_below_minus_one_hundred_percent_are_summarised_unclipped() -> None:
    decisions = [_record(10, "100"), _record(11, "100"), _record(12, "10"), _record(13, "10")]
    evaluations = {
        ("alpha", 10, _H1): "110",
        ("alpha", 11, _H1): "90",
        ("alpha", 12, _H1): "0",
        ("alpha", 13, _H1): "-5",
    }

    entry = _single(_run(decisions, (_H1,), evaluations))

    assert entry.average_forward_return == _percent("-62.5")
    assert entry.median_forward_return == _percent("-55")
    assert entry.minimum_forward_return == _percent("-150")
    assert entry.maximum_forward_return == _percent("10")


def test_buy_hold_and_sell_summarise_identically() -> None:
    runs = [
        _run(
            [_record(10, "100", signal=signal)],
            (_H1,),
            {("alpha", 10, _H1): "90"},
        )
        for signal in ("strong bullish", "neutral trend", "strong bearish")
    ]
    assert [run.records[0].result.recommendation.action.value for run in runs] == [
        "BUY",
        "HOLD",
        "SELL",
    ]

    summaries = [_metrics(run) for run in runs]

    assert summaries[0] == summaries[1] == summaries[2]
    assert summaries[0][0].average_forward_return == _percent("-10")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def _repeating_run() -> FuturesForwardResearchRun:
    """33.3..%, 28.57..% and 66.6..%: every average and median repeats."""
    decisions = [_record(10, "3"), _record(11, "7"), _record(12, "3")]
    evaluations = {("alpha", 10, _H1): "4", ("alpha", 11, _H1): "9", ("alpha", 12, _H1): "5"}
    return _run(decisions, (_H1,), evaluations)


@pytest.mark.parametrize("rounding", [ROUND_DOWN, ROUND_CEILING, ROUND_HALF_UP])
@pytest.mark.parametrize("precision", [6, 28, 50])
def test_caller_decimal_context_does_not_alter_the_metrics(precision: int, rounding: str) -> None:
    expected = _metrics(_repeating_run())

    with localcontext() as caller:
        caller.prec = precision
        caller.rounding = rounding
        produced = _metrics(_repeating_run())

    assert produced == expected


def test_low_ambient_precision_would_have_changed_the_average() -> None:
    """Guard: the determinism test is load-bearing only if this holds."""
    returns = [m.outcome.forward_return.value for m in _repeating_run().measurements]

    with localcontext() as caller:
        caller.prec = 6
        ambient = sum(returns, Decimal()) / len(returns)

    assert Percentage(ambient) != _single(_repeating_run()).average_forward_return


def test_equal_runs_give_equal_metrics_and_reports() -> None:
    first = BuildFuturesForwardResearchReportUseCase().execute(_mixed_run())
    second = BuildFuturesForwardResearchReportUseCase().execute(_mixed_run())

    assert first == second
    assert first.metrics == CalculateFuturesForwardResearchMetricsUseCase().execute(_mixed_run())
    assert repr(first) == repr(second)


# ---------------------------------------------------------------------------
# Metrics value invariants
# ---------------------------------------------------------------------------


def _entry(**overrides: object) -> FuturesForwardResearchStrategyHorizonMetrics:
    return dataclasses.replace(_single(_mixed_run()), **overrides)


def test_counts_that_do_not_partition_the_total_are_rejected() -> None:
    with pytest.raises(ValueError, match="must equal the total count"):
        _entry(total_count=5)


def test_statistics_without_a_measured_outcome_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be None when no outcome was measured"):
        _entry(measured_count=0, pending_count=3)


def test_a_measured_outcome_without_statistics_is_rejected() -> None:
    with pytest.raises(ValueError, match="required when an outcome was measured"):
        _entry(median_forward_return=None)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("strategy_identity", "alpha", TypeError),
        ("horizon", 1, TypeError),
        ("total_count", True, TypeError),
        ("pending_count", 1.0, TypeError),
        ("undefined_return_basis_count", -1, ValueError),
        ("average_forward_return", Decimal("-70"), TypeError),
    ],
)
def test_metrics_fields_are_validated(field: str, value: object, error: type[Exception]) -> None:
    with pytest.raises(error):
        _entry(**{field: value})


def test_the_metrics_have_no_currency_profit_or_win_concept() -> None:
    names = {
        field.name for field in dataclasses.fields(FuturesForwardResearchStrategyHorizonMetrics)
    }

    for absent in ("currency_mismatch_count", "unavailable_count", "pnl", "win_rate", "multiplier"):
        assert absent not in names


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _multi_run() -> FuturesForwardResearchRun:
    decisions = [
        _record(10, strategy="beta"),
        _record(10, strategy="alpha"),
        _record(12, "0", strategy="alpha"),
    ]
    evaluations = {
        ("alpha", 10, _H1): "110",
        ("beta", 10, _H1): "95",
        ("alpha", 12, _H1): "5",
    }
    return _run(decisions, (_H3, _H1), evaluations, through=30)


def test_the_report_pairs_the_run_with_its_metrics_and_derives_its_facts() -> None:
    run = _multi_run()

    report = BuildFuturesForwardResearchReportUseCase().execute(run)

    assert report.run is run
    assert report.metrics == _metrics(run)
    assert report.contract == _ES_DEC
    assert report.timeframe == _DAILY
    assert report.horizons == (_H3, _H1)
    assert report.strategy_identities == (StrategyIdentity("alpha"), StrategyIdentity("beta"))
    assert report.record_count == 3
    assert report.pending_count == 3  # every H3 is pending
    assert report.first_decision_instant == _instant(10)
    assert report.last_decision_instant == _instant(12)
    assert set(FuturesForwardResearchReport.__slots__) == {"run", "metrics"}


def test_the_report_cutoff_is_the_runs_not_the_last_decision_or_evaluation() -> None:
    report = BuildFuturesForwardResearchReportUseCase().execute(_multi_run())

    assert report.available_through == _instant(30)
    assert report.available_through != report.last_decision_instant
    evaluations = [
        m.outcome.evaluation_instant
        for m in report.run.measurements
        if m.outcome.evaluation_instant
    ]
    assert all(report.available_through != instant for instant in evaluations)


def test_an_empty_run_gives_a_valid_report_with_no_metrics() -> None:
    run = _run([], (_H3, _H1), {}, through=20)

    report = BuildFuturesForwardResearchReportUseCase().execute(run)

    assert report.metrics == ()
    assert report.strategy_identities == ()
    assert report.record_count == 0
    assert report.pending_count == 0
    assert report.first_decision_instant is None
    assert report.last_decision_instant is None
    assert report.contract == _ES_DEC
    assert report.horizons == (_H3, _H1)
    assert report.available_through == _instant(20)


def _swap(metrics: tuple, left: int, right: int) -> tuple:
    entries = list(metrics)
    entries[left], entries[right] = entries[right], entries[left]
    return tuple(entries)


@pytest.mark.parametrize(
    ("mangle", "message"),
    [
        pytest.param(lambda m: m[:-1], "one metrics entry for every", id="missing-entry"),
        pytest.param(lambda m: (*m, m[-1]), "one metrics entry for every", id="extra-entry"),
        pytest.param(lambda m: (m[0], m[0], *m[2:]), "horizon order", id="duplicate-key"),
        pytest.param(lambda m: _swap(m, 0, 2), "ascending strategy order", id="strategy-order"),
        pytest.param(lambda m: _swap(m, 0, 1), "horizon order", id="horizon-order"),
        pytest.param(
            lambda m: (
                dataclasses.replace(m[0], strategy_identity=StrategyIdentity("gamma")),
                *m[1:],
            ),
            "ascending strategy order",
            id="foreign-strategy",
        ),
        pytest.param(
            lambda m: (dataclasses.replace(m[0], horizon=_H2), *m[1:]),
            "horizon order",
            id="foreign-horizon",
        ),
        pytest.param(
            lambda m: (
                dataclasses.replace(
                    m[0], total_count=m[0].total_count + 1, pending_count=m[0].pending_count + 1
                ),
                *m[1:],
            ),
            "the run holds",
            id="count-mismatch",
        ),
    ],
)
def test_an_incoherent_report_is_rejected(mangle, message: str) -> None:
    run = _multi_run()

    with pytest.raises(ValueError, match=message):
        FuturesForwardResearchReport(run, mangle(_metrics(run)))


@pytest.mark.parametrize(("run", "metrics"), [("run", ()), ("multi", [])])
def test_the_report_type_checks_its_fields(run: object, metrics: object) -> None:
    actual = _multi_run() if run == "multi" else run

    with pytest.raises(TypeError):
        FuturesForwardResearchReport(actual, metrics)  # type: ignore[arg-type]


def test_the_builder_uses_an_injected_calculator() -> None:
    class RecordingCalculator(CalculateFuturesForwardResearchMetricsUseCase):
        def __init__(self) -> None:
            self.runs: list[FuturesForwardResearchRun] = []

        def execute(self, run):
            self.runs.append(run)
            return super().execute(run)

    calculator = RecordingCalculator()
    run = _multi_run()

    BuildFuturesForwardResearchReportUseCase(calculator).execute(run)

    assert calculator.runs == [run]


@pytest.mark.parametrize(
    "call",
    [
        lambda: CalculateFuturesForwardResearchMetricsUseCase().execute("run"),
        lambda: BuildFuturesForwardResearchReportUseCase().execute(None),
        lambda: BuildFuturesForwardResearchReportUseCase("calculator"),
    ],
    ids=["calculator-run", "builder-run", "builder-dependency"],
)
def test_inputs_are_type_checked(call) -> None:
    with pytest.raises(TypeError):
        call()


# ---------------------------------------------------------------------------
# A derived view: the same frozen decisions at two cutoffs
# ---------------------------------------------------------------------------


class _Records(FuturesForwardResearchRecordRepository):
    def __init__(self, records: tuple[FuturesForwardResearchRecord, ...]) -> None:
        self.records = records

    def get_records(self, query):
        return tuple(sorted(self.records, key=cmp_to_key(_canonical)))


class _Bars(FuturesHistoricalMarketDataRepository):
    def __init__(self, bars: tuple[FuturesOHLCVBar, ...]) -> None:
        self.bars = bars

    def get_bars(self, query):
        return tuple(
            sorted(
                (bar for bar in self.bars if query.covers(bar.point_in_time)),
                key=cmp_to_key(lambda a, b: a.point_in_time.compare(b.point_in_time)),
            )
        )


def test_the_same_frozen_decisions_report_differently_at_a_later_cutoff() -> None:
    def bar(session: int, close: str) -> FuturesOHLCVBar:
        quote = Decimal(close)
        return FuturesOHLCVBar(
            _ES_DEC,
            _instant(session),
            _DAILY,
            QuoteValue(quote),
            QuoteValue(quote + 1),
            QuoteValue(quote - 1),
            QuoteValue(quote),
            Quantity(Decimal("1000")),
        )

    records = _Records((_record(10, "100"), _record(10, "0", strategy="beta")))
    bars = _Bars((bar(11, "101"), bar(12, "102"), bar(13, "130")))
    use_case = RunFuturesForwardResearchUseCase(records, bars)
    builder = BuildFuturesForwardResearchReportUseCase()

    early = builder.execute(use_case.execute(_QUERY, (_H3,), _instant(12)))
    later = builder.execute(use_case.execute(_QUERY, (_H3,), _instant(13)))

    assert [_counts(entry) for entry in early.metrics] == [(1, 0, 1, 0), (1, 0, 1, 0)]
    assert [_counts(entry) for entry in later.metrics] == [(1, 1, 0, 0), (1, 0, 0, 1)]
    assert later.metrics[0].average_forward_return == _percent("30")
    assert early.run.records == later.run.records


# ---------------------------------------------------------------------------
# Boundaries and exports
# ---------------------------------------------------------------------------

_MODULES = ("calculate_futures_forward_metrics", "build_futures_forward_report")


def _tree(name: str) -> ast.Module:
    module = importlib.import_module(f"northstar_application.application_services.{name}")
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", _MODULES)
def test_metrics_and_report_derive_from_the_run_alone(name: str) -> None:
    tree = _tree(name)
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
        "FuturesForwardResearchRecordRepository",
        "FuturesForwardResearchRecordStore",
        "FuturesHistoricalMarketDataRepository",
        "MeasureFuturesForwardResearchRecordUseCase",
        "MeasureFuturesRecommendationOutcomeUseCase",
        "RunFuturesForwardResearchUseCase",
        "FreezeFuturesForwardResearchDecisionUseCase",
        "FuturesHistoricalMarketDataSource",
        "AcquireFuturesDailyHistoryUseCase",
    ):
        assert forbidden not in names
    for module_path in modules:
        assert module_path.split(".")[0] not in {
            "databento",
            "northstar_infrastructure",
            "sqlite3",
            "requests",
            "socket",
            "time",
            "datetime",
            "random",
            "decimal",
        }
        assert not module_path.startswith("northstar_application.ports")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"action", "now", "today", "utcnow"}


def test_the_statistics_mathematics_is_the_shared_helper() -> None:
    import northstar_application.application_services._return_statistics as statistics
    import northstar_application.application_services.calculate_futures_forward_metrics as m

    assert m.average is statistics.average
    assert m.median is statistics.median
    for node in ast.walk(_tree("calculate_futures_forward_metrics")):
        assert not (isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.Mult)))


def test_the_metrics_and_report_are_exported_and_helpers_are_not() -> None:
    import northstar_application.application_services as services

    for name in (
        "FuturesForwardResearchStrategyHorizonMetrics",
        "CalculateFuturesForwardResearchMetricsUseCase",
        "FuturesForwardResearchReport",
        "BuildFuturesForwardResearchReportUseCase",
    ):
        assert name in services.__all__
    for private in ("_distinct_strategies", "_state", "average", "median"):
        assert private not in services.__all__
