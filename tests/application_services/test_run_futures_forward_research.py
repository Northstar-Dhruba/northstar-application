"""Tests for the futures forward research run.

Both repositories behave as conforming ones must: the forward-record repository
returns every record for the queried contract in canonical order, and the
market repository answers each query within its window in semantic order. Both
can be extended between runs, to model decisions being frozen and sessions
being persisted as forward research matures.
"""

from __future__ import annotations

import ast
import dataclasses
from datetime import date, timedelta
from decimal import Decimal
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
    ResearchHorizon,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services import (
    ForwardResearchContractViolationError,
    ForwardResearchMeasurementState,
    FuturesAnalysisResult,
    FuturesForwardResearchRecord,
    FuturesForwardResearchRun,
    RunFuturesForwardResearchUseCase,
)
from northstar_application.ports import (
    FuturesForwardResearchRecordQuery,
    FuturesForwardResearchRecordRepository,
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
    InvalidFuturesForwardResearchRecordQueryError,
)

_ES_DEC = FuturesContract(
    FuturesProductReference(Symbol("ES"), ExchangeCode("CME")), ExpirationDate("2026-12-18")
)
_MES_DEC = FuturesContract(
    FuturesProductReference(Symbol("MES"), ExchangeCode("CME")), ExpirationDate("2026-12-18")
)
_DAILY = Timeframe("1d")
_QUERY = FuturesForwardResearchRecordQuery(_ES_DEC, _DAILY)
_H1, _H2, _H3 = ResearchHorizon(1), ResearchHorizon(2), ResearchHorizon(3)

_PENDING = ForwardResearchMeasurementState.PENDING
_MEASURED = ForwardResearchMeasurementState.MEASURED
_UNAVAILABLE = ForwardResearchMeasurementState.UNAVAILABLE


def _instant(session: int) -> PointInTime:
    return PointInTime(f"{date(2026, 9, 1) + timedelta(days=session - 1)}T21:00:00Z")


def _record(
    session: int | None = None,
    decision_quote: str = "100",
    *,
    strategy: str = "alpha",
    signal: str = "strong bullish",
    contract: FuturesContract = _ES_DEC,
    instant: PointInTime | None = None,
) -> FuturesForwardResearchRecord:
    observed_at = instant if instant is not None else _instant(session or 10)
    quote = QuoteValue(Decimal(decision_quote))
    context = FuturesMarketObservationContext(
        contract=contract,
        timeframe=_DAILY,
        observed_at=observed_at,
        latest_quote=quote,
        previous_close=QuoteValue(quote.value - 1),
        latest_volume=Quantity(Decimal("1000")),
        session_high=QuoteValue(quote.value + 2),
        session_low=QuoteValue(quote.value - 2),
        recent_closes=(QuoteValue(Decimal("12345")),) * 19 + (quote,),
        recent_volumes=(Quantity(Decimal("1000")),) * 20,
    )
    recommendation = Strategy(StrategyIdentity(strategy)).evaluate_futures(
        FuturesAssetAnalysis(contract, observed_at, (signal,))
    )
    return FuturesForwardResearchRecord(FuturesAnalysisResult(recommendation, context))


def _bar(session: int, close: str, *, instant: PointInTime | None = None) -> FuturesOHLCVBar:
    quote = Decimal(close)
    return FuturesOHLCVBar(
        contract=_ES_DEC,
        point_in_time=instant if instant is not None else _instant(session),
        timeframe=_DAILY,
        open=QuoteValue(quote),
        high=QuoteValue(quote + 1),
        low=QuoteValue(quote - 1),
        close=QuoteValue(quote),
        volume=Quantity(Decimal("1000")),
    )


def _canonical(left: FuturesForwardResearchRecord, right: FuturesForwardResearchRecord) -> int:
    instant = left.decision_instant.compare(right.decision_instant)
    if instant:
        return instant
    a, b = left.strategy_identity.identity, right.strategy_identity.identity
    return (a > b) - (a < b)


class ForwardRepository(FuturesForwardResearchRecordRepository):
    """Every frozen record for the queried contract, in canonical order."""

    def __init__(self, records: tuple[FuturesForwardResearchRecord, ...] = ()) -> None:
        self.records = records
        self.queries: list[FuturesForwardResearchRecordQuery] = []

    def get_records(self, query):
        self.queries.append(query)
        matching = [record for record in self.records if record.contract == query.contract]
        return tuple(sorted(matching, key=cmp_to_key(_canonical)))


class StubForwardRepository(FuturesForwardResearchRecordRepository):
    """Returns exactly what it is given, conforming or not."""

    def __init__(self, records: object) -> None:
        self.records = records

    def get_records(self, query):
        return self.records


class MarketRepository(FuturesHistoricalMarketDataRepository):
    def __init__(self, bars: tuple[FuturesOHLCVBar, ...] = ()) -> None:
        self.bars = bars
        self.queries: list[FuturesHistoricalMarketDataQuery] = []

    def get_bars(self, query):
        self.queries.append(query)
        matching = [
            bar
            for bar in self.bars
            if bar.contract == query.contract and query.covers(bar.point_in_time)
        ]
        return tuple(
            sorted(matching, key=cmp_to_key(lambda a, b: a.point_in_time.compare(b.point_in_time)))
        )


def _run(
    forward: FuturesForwardResearchRecordRepository,
    market: FuturesHistoricalMarketDataRepository,
    horizons: tuple[ResearchHorizon, ...],
    through: PointInTime,
) -> FuturesForwardResearchRun:
    return RunFuturesForwardResearchUseCase(forward, market).execute(_QUERY, horizons, through)


def _states(run: FuturesForwardResearchRun) -> list[ForwardResearchMeasurementState]:
    return [measurement.state for measurement in run.measurements]


# ---------------------------------------------------------------------------
# Empty runs
# ---------------------------------------------------------------------------


def test_no_frozen_records_is_a_valid_empty_run_that_keeps_its_configuration() -> None:
    run = _run(ForwardRepository(), MarketRepository(), (_H3, _H1), _instant(20))

    assert run.records == ()
    assert run.measurements == ()
    assert run.query == _QUERY
    assert run.horizons == (_H3, _H1)
    assert run.available_through == _instant(20)


def test_records_decided_only_after_the_cutoff_give_an_empty_run() -> None:
    forward = ForwardRepository((_record(15), _record(16)))

    run = _run(forward, MarketRepository(), (_H1,), _instant(14))

    assert run.records == ()
    assert run.measurements == ()


# ---------------------------------------------------------------------------
# Records, strategies and ordering
# ---------------------------------------------------------------------------


def test_one_record_at_one_horizon() -> None:
    forward = ForwardRepository((_record(10),))
    market = MarketRepository((_bar(11, "110"),))

    run = _run(forward, market, (_H1,), _instant(11))

    assert run.records == (_record(10),)
    (measurement,) = run.measurements
    assert measurement.state is _MEASURED
    assert measurement.outcome.forward_return == Percentage(Decimal("10"))


def test_horizons_keep_the_callers_order() -> None:
    forward = ForwardRepository((_record(10),))
    market = MarketRepository(tuple(_bar(session, str(100 + session)) for session in range(11, 14)))

    run = _run(forward, market, (_H3, _H1, _H2), _instant(13))

    assert run.horizons == (_H3, _H1, _H2)
    assert [measurement.horizon for measurement in run.measurements] == [_H3, _H1, _H2]


def test_several_strategies_and_instants_are_ordered_canonically_record_major() -> None:
    beta_10 = _record(10, strategy="beta")
    alpha_10 = _record(10, strategy="alpha")
    alpha_12 = _record(12, strategy="alpha")
    forward = ForwardRepository((alpha_12, beta_10, alpha_10))  # stored out of order
    market = MarketRepository(tuple(_bar(session, str(100 + session)) for session in range(11, 16)))

    run = _run(forward, market, (_H3, _H1), _instant(15))

    assert run.records == (alpha_10, beta_10, alpha_12)
    assert [(m.record, m.horizon) for m in run.measurements] == [
        (alpha_10, _H3),
        (alpha_10, _H1),
        (beta_10, _H3),
        (beta_10, _H1),
        (alpha_12, _H3),
        (alpha_12, _H1),
    ]
    assert {record.strategy_identity.identity for record in run.records} == {"alpha", "beta"}


def test_the_forward_repository_is_queried_once_with_the_run_query() -> None:
    forward = ForwardRepository((_record(10), _record(11)))

    _run(forward, MarketRepository(), (_H1, _H2), _instant(12))

    assert forward.queries == [_QUERY]


def test_every_measurement_uses_the_run_cutoff() -> None:
    forward = ForwardRepository(
        (_record(10, strategy="alpha"), _record(10, strategy="beta"), _record(12))
    )
    market = MarketRepository(tuple(_bar(session, "101") for session in range(11, 20)))
    cutoff = _instant(14)

    _run(forward, market, (_H1, _H3), cutoff)

    assert len(market.queries) == 3 * 2
    assert {query.end for query in market.queries} == {cutoff}


# ---------------------------------------------------------------------------
# Evolution across runs
# ---------------------------------------------------------------------------


def test_a_pending_decision_becomes_measured_at_a_later_cutoff() -> None:
    record = _record(10, "100")
    forward = ForwardRepository((record,))
    market = MarketRepository((_bar(11, "101"), _bar(12, "102")))

    early = _run(forward, market, (_H3,), _instant(12))
    market.bars = (*market.bars, _bar(13, "130"))
    later = _run(forward, market, (_H3,), _instant(13))

    assert _states(early) == [_PENDING]
    assert _states(later) == [_MEASURED]
    assert later.records == early.records == (_record(10, "100"),)
    assert later.measurements[0].outcome.forward_return == Percentage(Decimal("30"))


def test_a_non_positive_decision_becomes_unavailable_at_a_later_cutoff() -> None:
    forward = ForwardRepository((_record(10, "-10"),))
    market = MarketRepository((_bar(11, "-9"), _bar(12, "-8")))

    early = _run(forward, market, (_H3,), _instant(12))
    market.bars = (*market.bars, _bar(13, "-5"))
    later = _run(forward, market, (_H3,), _instant(13))

    assert _states(early) == [_PENDING]
    assert _states(later) == [_UNAVAILABLE]
    assert later.records == early.records


def test_market_bars_after_the_cutoff_cannot_change_the_run() -> None:
    """Load-bearing: the appended bars resolve and reverse measurements at a later cutoff."""
    forward = ForwardRepository((_record(10, "100"), _record(12, "0")))
    market = MarketRepository((_bar(11, "101"), _bar(12, "102")))
    first = _run(forward, market, (_H1, _H3), _instant(12))

    market.bars = (
        *market.bars,
        _bar(13, "-99999"),
        _bar(14, "99999"),
        _bar(15, "0"),
    )
    second = _run(forward, market, (_H1, _H3), _instant(12))

    assert second == first


def test_a_later_cutoff_admits_those_bars() -> None:
    """Negative control for the reproducibility test."""
    forward = ForwardRepository((_record(10, "100"), _record(12, "0")))
    market = MarketRepository(
        (_bar(11, "101"), _bar(12, "102"), _bar(13, "-99999"), _bar(14, "99999"), _bar(15, "0"))
    )

    at_12 = _run(forward, market, (_H1, _H3), _instant(12))
    at_15 = _run(forward, market, (_H1, _H3), _instant(15))

    assert _states(at_12) == [_MEASURED, _PENDING, _PENDING, _PENDING]
    assert _states(at_15) == [_MEASURED, _MEASURED, _UNAVAILABLE, _UNAVAILABLE]
    assert at_15.records == at_12.records
    assert at_15.measurements[1].outcome.evaluation_quote == QuoteValue(Decimal("-99999"))


def test_a_decision_frozen_after_the_cutoff_does_not_enter_the_run() -> None:
    forward = ForwardRepository((_record(10),))
    market = MarketRepository((_bar(11, "101"), _bar(12, "102")))
    first = _run(forward, market, (_H1,), _instant(12))

    forward.records = (*forward.records, _record(13, strategy="later"))
    again = _run(forward, market, (_H1,), _instant(12))
    later = _run(forward, market, (_H1,), _instant(13))

    assert again == first
    assert _record(13, strategy="later") in later.records


def test_a_decision_frozen_later_at_or_before_the_cutoff_joins_a_re_run() -> None:
    """Accepted limitation: a forward run is re-derived from the records frozen now."""
    forward = ForwardRepository((_record(10),))
    market = MarketRepository((_bar(11, "101"), _bar(12, "102")))
    first = _run(forward, market, (_H1,), _instant(12))

    forward.records = (*forward.records, _record(11, strategy="late-freeze"))
    again = _run(forward, market, (_H1,), _instant(12))

    assert again != first
    assert again.records == (_record(10), _record(11, strategy="late-freeze"))


# ---------------------------------------------------------------------------
# Semantic timestamps and determinism
# ---------------------------------------------------------------------------


def test_an_offset_spelled_cutoff_gives_the_same_run() -> None:
    forward = ForwardRepository((_record(10),))
    market = MarketRepository((_bar(11, "101"),))

    z = _run(forward, market, (_H1,), _instant(11))
    offset = _run(forward, market, (_H1,), PointInTime("2026-09-11T16:00:00-05:00"))

    assert offset == z


def test_a_decision_exactly_at_the_cutoff_is_included_and_pending() -> None:
    run = _run(ForwardRepository((_record(10),)), MarketRepository(), (_H1,), _instant(10))

    assert run.records == (_record(10),)
    assert _states(run) == [_PENDING]


def test_a_decision_half_a_second_after_the_cutoff_is_excluded() -> None:
    late = _record(instant=PointInTime("2026-09-10T21:00:00.5Z"))

    run = _run(ForwardRepository((late,)), MarketRepository(), (_H1,), _instant(10))

    assert run.records == ()


def test_records_straddling_the_text_order_trap_are_ordered_semantically() -> None:
    whole = _record(instant=_instant(10), strategy="zeta")
    fractional = _record(instant=PointInTime("2026-09-10T21:00:00.5Z"), strategy="alpha")
    assert fractional.decision_instant.value < whole.decision_instant.value

    run = _run(ForwardRepository((fractional, whole)), MarketRepository(), (_H1,), _instant(11))

    assert run.records == (whole, fractional)


def test_repeated_runs_are_equal() -> None:
    records = (_record(10, strategy="alpha"), _record(10, strategy="beta"), _record(12))
    bars = tuple(_bar(session, str(100 + session)) for session in range(11, 16))

    first = _run(ForwardRepository(records), MarketRepository(bars), (_H1, _H3), _instant(14))
    second = _run(ForwardRepository(records), MarketRepository(bars), (_H1, _H3), _instant(14))

    assert first == second
    assert repr(first) == repr(second)


def test_buy_hold_and_sell_decisions_are_measured_identically() -> None:
    records = tuple(
        _record(10, strategy=name, signal=signal)
        for name, signal in (
            ("a", "strong bullish"),
            ("b", "neutral trend"),
            ("c", "strong bearish"),
        )
    )

    run = _run(
        ForwardRepository(records), MarketRepository((_bar(11, "90"),)), (_H1,), _instant(11)
    )

    assert [record.result.recommendation.action.value for record in run.records] == [
        "BUY",
        "HOLD",
        "SELL",
    ]
    assert {measurement.outcome.forward_return for measurement in run.measurements} == {
        Percentage(Decimal("-10"))
    }


# ---------------------------------------------------------------------------
# Repository output validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("records", "message"),
    [
        pytest.param([_record(10)], "must return a tuple", id="list"),
        pytest.param((_record(10), "record"), "record 1 must be", id="foreign-member"),
        pytest.param((_record(10, contract=_MES_DEC),), "not the queried", id="other-contract"),
        pytest.param((_record(10), _record(10)), "repeats a natural key", id="duplicate"),
        pytest.param((_record(12), _record(10)), "out of order", id="instant-order"),
        pytest.param(
            (_record(10, strategy="beta"), _record(10, strategy="alpha")),
            "out of order",
            id="strategy-order",
        ),
        pytest.param(
            (
                _record(instant=PointInTime("2026-09-10T21:00:00.5Z")),
                _record(instant=_instant(10)),
            ),
            "out of order",
            id="text-ordered-instants",
        ),
    ],
)
def test_malformed_repository_output_is_refused_not_repaired(records: object, message: str) -> None:
    with pytest.raises(ForwardResearchContractViolationError, match=message):
        _run(StubForwardRepository(records), MarketRepository(), (_H1,), _instant(20))


def test_records_after_the_cutoff_are_valid_output_not_a_violation() -> None:
    run = _run(
        StubForwardRepository((_record(10), _record(30))), MarketRepository(), (_H1,), _instant(20)
    )

    assert run.records == (_record(10),)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("horizons", "error", "message"),
    [
        pytest.param((), ValueError, "at least one horizon", id="empty"),
        pytest.param((_H1, _H1), ValueError, "duplicates", id="duplicate"),
        pytest.param((_H1, ResearchHorizon(1)), ValueError, "duplicates", id="rebuilt-duplicate"),
        pytest.param([_H1], TypeError, "must be a tuple", id="list"),
        pytest.param((_H1, 2), TypeError, "ResearchHorizon values", id="non-horizon"),
    ],
)
def test_invalid_horizons_are_refused_before_any_read(
    horizons: object, error: type[Exception], message: str
) -> None:
    forward = ForwardRepository((_record(10),))

    with pytest.raises(error, match=message):
        _run(forward, MarketRepository(), horizons, _instant(20))  # type: ignore[arg-type]

    assert forward.queries == []


@pytest.mark.parametrize("position", [0, 2])
def test_the_query_and_cutoff_are_type_checked_before_any_read(position: int) -> None:
    arguments: list[object] = [_QUERY, (_H1,), _instant(20)]
    arguments[position] = "not an input"
    forward = ForwardRepository((_record(10),))

    with pytest.raises(TypeError):
        RunFuturesForwardResearchUseCase(forward, MarketRepository()).execute(*arguments)  # type: ignore[arg-type]

    assert forward.queries == []


@pytest.mark.parametrize("position", [0, 1])
def test_both_dependencies_are_type_checked(position: int) -> None:
    dependencies: list[object] = [ForwardRepository(), MarketRepository()]
    dependencies[position] = "not a dependency"

    with pytest.raises(TypeError):
        RunFuturesForwardResearchUseCase(*dependencies)  # type: ignore[arg-type]


def test_an_intraday_forward_run_cannot_even_be_queried() -> None:
    with pytest.raises(InvalidFuturesForwardResearchRecordQueryError, match="session-daily"):
        FuturesForwardResearchRecordQuery(_ES_DEC, Timeframe("1m"))


# ---------------------------------------------------------------------------
# FuturesForwardResearchRun invariants
# ---------------------------------------------------------------------------


def _valid_run() -> FuturesForwardResearchRun:
    records = (_record(10, strategy="alpha"), _record(10, strategy="beta"), _record(12))
    bars = tuple(_bar(session, str(100 + session)) for session in range(11, 16))
    return _run(ForwardRepository(records), MarketRepository(bars), (_H3, _H1), _instant(14))


def test_a_run_for_another_contract_rejects_its_records() -> None:
    with pytest.raises(ValueError, match="query contract"):
        dataclasses.replace(_valid_run(), query=FuturesForwardResearchRecordQuery(_MES_DEC, _DAILY))


def test_a_record_after_the_cutoff_is_rejected() -> None:
    """Pending measurements only, so the record check is the one that must fire."""
    run = _run(
        ForwardRepository((_record(10), _record(12))), MarketRepository(), (_H1,), _instant(12)
    )
    assert all(m.outcome.evaluation_instant is None for m in run.measurements)

    with pytest.raises(ValueError, match="record decided at .* after the run cutoff"):
        dataclasses.replace(run, available_through=_instant(11))


def test_duplicate_records_are_rejected() -> None:
    run = _valid_run()

    with pytest.raises(ValueError, match="share a natural key"):
        dataclasses.replace(
            run,
            records=(run.records[0], run.records[0]),
            measurements=run.measurements[:2] * 2,
        )


def test_out_of_order_records_are_rejected_not_sorted() -> None:
    run = _valid_run()
    reversed_records = tuple(reversed(run.records))
    matching = tuple(
        m for record in reversed_records for m in run.measurements if m.record == record
    )

    with pytest.raises(ValueError, match="ordered by decision instant"):
        dataclasses.replace(run, records=reversed_records, measurements=matching)


def test_the_measurement_count_must_be_exact() -> None:
    run = _valid_run()

    with pytest.raises(ValueError, match="every record and horizon pair"):
        dataclasses.replace(run, measurements=run.measurements[:-1])


def test_measurements_out_of_record_order_are_rejected() -> None:
    run = _valid_run()
    measurements = list(run.measurements)
    measurements[0], measurements[2] = measurements[2], measurements[0]

    with pytest.raises(ValueError, match="follow record order"):
        dataclasses.replace(run, measurements=tuple(measurements))


def test_measurements_out_of_horizon_order_are_rejected() -> None:
    with pytest.raises(ValueError, match="horizon order"):
        dataclasses.replace(_valid_run(), horizons=(_H1, _H3))


def test_an_evaluation_after_the_cutoff_is_rejected() -> None:
    """Drop the session-12 record so the cutoff can fall before a session-13 evaluation."""
    run = _run(
        ForwardRepository((_record(10),)),
        MarketRepository(tuple(_bar(session, str(100 + session)) for session in range(11, 16))),
        (_H3,),
        _instant(14),
    )
    assert run.measurements[0].outcome.evaluation_instant == _instant(13)

    with pytest.raises(ValueError, match="evaluated at .* after the run cutoff"):
        dataclasses.replace(run, available_through=_instant(12))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("query", "query"),
        ("horizons", [_H1]),
        ("available_through", "2026-09-14"),
        ("records", []),
        ("measurements", []),
    ],
)
def test_every_run_field_is_type_checked(field: str, value: object) -> None:
    with pytest.raises(TypeError):
        dataclasses.replace(_valid_run(), **{field: value})


def test_the_run_stores_no_single_strategy_or_market_data() -> None:
    fields = {field.name for field in dataclasses.fields(FuturesForwardResearchRun)}

    assert fields == {"query", "horizons", "available_through", "records", "measurements"}


# ---------------------------------------------------------------------------
# Boundaries and exports
# ---------------------------------------------------------------------------


def _module_tree() -> ast.Module:
    import northstar_application.application_services.run_futures_forward_research as module

    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


def test_the_run_evaluates_frozen_decisions_and_creates_none() -> None:
    tree = _module_tree()
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
        "FuturesForwardResearchRecordStore",
        "FreezeFuturesForwardResearchDecisionUseCase",
        "FuturesHistoricalMarketDataSource",
        "AcquireFuturesDailyHistoryUseCase",
        "FuturesTradingSessionResolver",
    ):
        assert forbidden not in names
    for module_path in modules:
        assert module_path.split(".")[0] not in {
            "databento",
            "exchange_calendars",
            "northstar_infrastructure",
            "sqlite3",
            "requests",
            "socket",
            "time",
            "datetime",
            "random",
        }
        assert "paper_trading" not in module_path
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"action", "now", "today", "utcnow"}


def test_the_run_and_use_case_are_exported() -> None:
    import northstar_application.application_services as services

    for name in ("FuturesForwardResearchRun", "RunFuturesForwardResearchUseCase"):
        assert name in services.__all__
    for private in ("_canonical_order", "_validate_horizons"):
        assert not hasattr(services, private)
