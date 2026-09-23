"""Tests for deterministic futures historical replay.

The repository is a stub that returns whatever it is given, deliberately
including output a conforming repository would never produce. That is the
point: the replay must refuse contaminated, unordered or out-of-window history
rather than tidy it up, so the tests hand it exactly that.
"""

from __future__ import annotations

import ast
from decimal import Decimal
from pathlib import Path

import pytest
from northstar_core.derivatives import ExpirationDate, QuoteValue
from northstar_core.foundation.value_objects import (
    ExchangeCode,
    PointInTime,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.futures import (
    FuturesContract,
    FuturesOHLCVBar,
    FuturesProductReference,
    FuturesReplaySnapshot,
)

from northstar_application.application_services import (
    FuturesHistoricalDataContractViolationError,
    ReplayFuturesHistoricalMarketDataUseCase,
    UnsupportedFuturesReplayTimeframeError,
)
from northstar_application.ports import (
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
)

_CME = ExchangeCode("CME")
_ES = FuturesProductReference(Symbol("ES"), _CME)
_MES = FuturesProductReference(Symbol("MES"), _CME)

_ES_DEC = FuturesContract(_ES, ExpirationDate("2026-12-18"))
_ES_MAR = FuturesContract(_ES, ExpirationDate("2027-03-19"))
_MES_DEC = FuturesContract(_MES, ExpirationDate("2026-12-18"))

_DAILY = Timeframe("1d")
_MINUTE = Timeframe("1m")
_HOURLY = Timeframe("1h")


def _bar(
    completion: str,
    *,
    contract: FuturesContract = _ES_DEC,
    timeframe: Timeframe = _DAILY,
    close: str = "7663",
    high: str = "7700",
    low: str = "7500",
) -> FuturesOHLCVBar:
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=PointInTime(completion),
        timeframe=timeframe,
        open=QuoteValue(Decimal("7660")),
        high=QuoteValue(Decimal(high)),
        low=QuoteValue(Decimal(low)),
        close=QuoteValue(Decimal(close)),
        volume=Quantity(Decimal("1000")),
    )


def _query(
    contract: FuturesContract = _ES_DEC,
    timeframe: Timeframe = _DAILY,
    start: str | None = None,
    end: str | None = None,
) -> FuturesHistoricalMarketDataQuery:
    return FuturesHistoricalMarketDataQuery(
        contract,
        timeframe,
        PointInTime(start) if start is not None else None,
        PointInTime(end) if end is not None else None,
    )


# Five consecutive CME session completions.
_B1 = _bar("2026-09-14T21:00:00Z")
_B2 = _bar("2026-09-15T21:00:00Z")
_B3 = _bar("2026-09-16T21:00:00Z")
_B4 = _bar("2026-09-17T21:00:00Z")
_B5 = _bar("2026-09-18T21:00:00Z")
_HISTORY = (_B1, _B2, _B3, _B4, _B5)


class StubRepository(FuturesHistoricalMarketDataRepository):
    def __init__(self, bars: object = ()) -> None:
        self.bars = bars
        self.queries: list[FuturesHistoricalMarketDataQuery] = []

    def get_bars(self, query):
        self.queries.append(query)
        return self.bars


class FailingRepository(FuturesHistoricalMarketDataRepository):
    def get_bars(self, query):
        raise RuntimeError("storage unavailable")


def _replay(
    bars: object, query: FuturesHistoricalMarketDataQuery | None = None
) -> tuple[FuturesReplaySnapshot, ...]:
    return ReplayFuturesHistoricalMarketDataUseCase(StubRepository(bars)).execute(
        query if query is not None else _query()
    )


# ---------------------------------------------------------------------------
# Repository query
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        _query(),
        _query(start="2026-09-15T21:00:00Z"),
        _query(end="2026-09-16T21:00:00Z"),
        _query(start="2026-09-15T21:00:00Z", end="2026-09-16T21:00:00Z"),
        _query(_MES_DEC),
    ],
)
def test_the_query_reaches_the_repository_unchanged_and_exactly_once(
    query: FuturesHistoricalMarketDataQuery,
) -> None:
    repository = StubRepository()

    ReplayFuturesHistoricalMarketDataUseCase(repository).execute(query)

    assert repository.queries == [query]
    assert repository.queries[0] is query


def test_replay_does_not_widen_the_start_for_warm_up() -> None:
    """Warm-up belongs to the context builder; replay asks for what it was asked for."""
    repository = StubRepository((_B3, _B4))
    query = _query(start="2026-09-16T21:00:00Z")

    snapshots = ReplayFuturesHistoricalMarketDataUseCase(repository).execute(query)

    assert repository.queries[0].start == PointInTime("2026-09-16T21:00:00Z")
    assert snapshots[0].observations == (_B3,)


# ---------------------------------------------------------------------------
# Session-daily scope
# ---------------------------------------------------------------------------


def test_the_session_daily_timeframe_is_accepted() -> None:
    repository = StubRepository((_B1, _B2))

    snapshots = ReplayFuturesHistoricalMarketDataUseCase(repository).execute(_query())

    assert len(repository.queries) == 1
    assert [s.timeframe for s in snapshots] == [Timeframe("1d"), Timeframe("1d")]


@pytest.mark.parametrize("timeframe", [_MINUTE, _HOURLY], ids=["1m", "1h"])
def test_an_intraday_timeframe_is_rejected_before_the_repository_is_read(
    timeframe: Timeframe,
) -> None:
    """The stub holds matching bars, so only the gate can stop the replay."""
    repository = StubRepository(
        (
            _bar("2026-09-14T21:00:00Z", timeframe=timeframe),
            _bar("2026-09-14T21:01:00Z", timeframe=timeframe),
        )
    )

    with pytest.raises(UnsupportedFuturesReplayTimeframeError, match="session-daily") as raised:
        ReplayFuturesHistoricalMarketDataUseCase(repository).execute(_query(timeframe=timeframe))

    assert repository.queries == []
    assert str(timeframe) in str(raised.value)


def test_an_unsupported_timeframe_is_a_caller_error_not_a_repository_violation() -> None:
    assert issubclass(UnsupportedFuturesReplayTimeframeError, ValueError)
    assert not issubclass(
        UnsupportedFuturesReplayTimeframeError, FuturesHistoricalDataContractViolationError
    )
    assert not issubclass(
        FuturesHistoricalDataContractViolationError, UnsupportedFuturesReplayTimeframeError
    )


# ---------------------------------------------------------------------------
# Snapshot construction
# ---------------------------------------------------------------------------


def test_no_stored_bars_produce_no_snapshots() -> None:
    assert _replay(()) == ()


def test_one_bar_produces_one_single_observation_snapshot() -> None:
    (snapshot,) = _replay((_B1,))

    assert snapshot.observations == (_B1,)
    assert snapshot.replay_instant == _B1.point_in_time


def test_each_snapshot_holds_every_bar_up_to_and_including_its_own() -> None:
    snapshots = _replay((_B1, _B2, _B3))

    assert [s.observations for s in snapshots] == [
        (_B1,),
        (_B1, _B2),
        (_B1, _B2, _B3),
    ]
    assert [s.replay_instant for s in snapshots] == [
        _B1.point_in_time,
        _B2.point_in_time,
        _B3.point_in_time,
    ]


def test_exactly_one_snapshot_is_produced_per_stored_bar() -> None:
    snapshots = _replay(_HISTORY)

    assert len(snapshots) == len(_HISTORY)
    for snapshot, bar in zip(snapshots, _HISTORY, strict=True):
        assert snapshot.latest_observation == bar


def test_every_snapshot_is_bound_to_the_queried_contract_and_timeframe() -> None:
    for snapshot in _replay(_HISTORY):
        assert type(snapshot) is FuturesReplaySnapshot
        assert snapshot.contract == _ES_DEC
        assert snapshot.timeframe == _DAILY


def test_each_snapshot_extends_its_predecessor_in_the_same_order() -> None:
    snapshots = _replay(_HISTORY)

    for earlier, later in zip(snapshots, snapshots[1:], strict=False):
        assert later.observations[: len(earlier.observations)] == earlier.observations
        assert len(later.observations) == len(earlier.observations) + 1


def test_the_result_is_an_immutable_tuple_of_immutable_snapshots() -> None:
    snapshots = _replay(_HISTORY)

    assert isinstance(snapshots, tuple)
    with pytest.raises(AttributeError):
        snapshots[0].observations = _HISTORY  # type: ignore[misc]
    assert snapshots[0].observations == (_B1,)


# ---------------------------------------------------------------------------
# Look-ahead protection
# ---------------------------------------------------------------------------


def test_a_snapshot_cannot_see_bars_that_complete_after_it() -> None:
    """B4 and B5 are extreme on purpose; if they leaked into S3 it would show."""
    crash = _bar("2026-09-17T21:00:00Z", close="-37.63", high="7700", low="-40")
    spike = _bar("2026-09-18T21:00:00Z", close="99999", high="99999", low="7500")

    with_future = _replay((_B1, _B2, _B3, crash, spike))
    without_future = _replay((_B1, _B2, _B3))

    third = with_future[2]
    assert third.observations == (_B1, _B2, _B3)
    assert third.replay_instant == _B3.point_in_time
    assert crash not in third.observations
    assert spike not in third.observations
    assert third == without_future[2]
    assert third == FuturesReplaySnapshot(_ES_DEC, _DAILY, _B3.point_in_time, (_B1, _B2, _B3))


def test_snapshots_are_not_all_the_full_history() -> None:
    """Fails if one tuple of every bar were reused for each snapshot."""
    snapshots = _replay(_HISTORY)

    assert len(set(snapshots)) == len(_HISTORY)
    assert [len(s.observations) for s in snapshots] == [1, 2, 3, 4, 5]
    for position, snapshot in enumerate(snapshots):
        for later in _HISTORY[position + 1 :]:
            assert later not in snapshot.observations
        for observation in snapshot.observations:
            assert observation.point_in_time.compare(snapshot.replay_instant) <= 0


# ---------------------------------------------------------------------------
# Sparse history and contract lifecycle
# ---------------------------------------------------------------------------


def test_sparse_sessions_produce_one_snapshot_per_stored_bar_and_none_between() -> None:
    stored = (
        _bar("2026-07-01T21:00:00Z"),
        _bar("2026-07-02T21:00:00Z"),
        _bar("2026-07-06T21:00:00Z"),
        _bar("2026-07-09T21:00:00Z"),
    )

    snapshots = _replay(stored, _query(start="2026-07-01T00:00:00Z", end="2026-07-10T00:00:00Z"))

    assert [s.replay_instant.value for s in snapshots] == [
        "2026-07-01T21:00:00Z",
        "2026-07-02T21:00:00Z",
        "2026-07-06T21:00:00Z",
        "2026-07-09T21:00:00Z",
    ]
    assert snapshots[-1].observations == stored


def test_a_window_before_the_first_stored_bar_replays_nothing() -> None:
    assert _replay((), _query(end="2026-01-01T00:00:00Z")) == ()


def test_a_window_past_the_last_stored_bar_stops_at_that_bar() -> None:
    snapshots = _replay(
        (_B1, _B2), _query(start="2026-09-01T00:00:00Z", end="2027-06-01T00:00:00Z")
    )

    assert len(snapshots) == 2
    assert snapshots[-1].replay_instant == _B2.point_in_time


def test_bars_through_expiration_replay_normally_and_nothing_follows() -> None:
    near_expiry = (
        _bar("2026-12-16T22:00:00Z"),
        _bar("2026-12-17T22:00:00Z"),
        _bar("2026-12-18T14:30:00Z"),
    )

    snapshots = _replay(near_expiry, _query(end="2027-03-31T00:00:00Z"))

    assert [s.latest_observation for s in snapshots] == list(near_expiry)
    assert all(s.contract == _ES_DEC for s in snapshots)
    assert all(o.contract == _ES_DEC for s in snapshots for o in s.observations)


# ---------------------------------------------------------------------------
# Inclusive bounds
# ---------------------------------------------------------------------------


def test_bars_exactly_on_both_inclusive_bounds_are_replayed() -> None:
    query = _query(start=_B2.point_in_time.value, end=_B4.point_in_time.value)

    snapshots = _replay((_B2, _B3, _B4), query)

    assert [s.latest_observation for s in snapshots] == [_B2, _B3, _B4]


def test_equal_bounds_replay_the_single_instant() -> None:
    query = _query(start=_B3.point_in_time.value, end=_B3.point_in_time.value)

    (snapshot,) = _replay((_B3,), query)

    assert snapshot.observations == (_B3,)


@pytest.mark.parametrize(
    ("bars", "instant"),
    [
        ((_B1, _B2, _B3), "2026-09-14T21:00:00Z"),
        ((_B2, _B3, _B4), "2026-09-17T21:00:00Z"),
    ],
)
def test_a_bar_outside_the_queried_window_is_rejected_not_filtered(
    bars: tuple[FuturesOHLCVBar, ...], instant: str
) -> None:
    query = _query(start=_B2.point_in_time.value, end=_B3.point_in_time.value)

    with pytest.raises(FuturesHistoricalDataContractViolationError, match="outside") as raised:
        _replay(bars, query)

    assert instant in str(raised.value)


def test_a_sub_second_bar_just_past_the_end_bound_is_rejected() -> None:
    """Lexically ``...00.5Z`` < ``...00Z``; a text bound check would admit it."""
    query = _query(end="2026-09-14T21:00:00Z")

    with pytest.raises(FuturesHistoricalDataContractViolationError, match="outside"):
        _replay((_bar("2026-09-14T21:00:00.5Z"),), query)


# ---------------------------------------------------------------------------
# Repository-output validation
# ---------------------------------------------------------------------------


def test_a_non_tuple_result_is_rejected() -> None:
    with pytest.raises(FuturesHistoricalDataContractViolationError, match="must return a tuple"):
        _replay([_B1, _B2])


@pytest.mark.parametrize("member", [None, "bar", object()])
def test_a_non_bar_member_is_rejected(member: object) -> None:
    with pytest.raises(FuturesHistoricalDataContractViolationError, match="observation 1"):
        _replay((_B1, member))


def test_a_bar_at_another_timeframe_is_rejected() -> None:
    with pytest.raises(FuturesHistoricalDataContractViolationError, match="timeframe"):
        _replay((_B1, _bar("2026-09-15T21:00:00Z", timeframe=_MINUTE)))


@pytest.mark.parametrize(
    "intruder",
    [
        pytest.param(_ES_MAR, id="same-product-and-exchange-later-expiry"),
        pytest.param(_MES_DEC, id="micro-product-same-exchange-and-expiry"),
        pytest.param(
            FuturesContract(
                FuturesProductReference(Symbol("ES"), ExchangeCode("CBOT")),
                ExpirationDate("2026-12-18"),
            ),
            id="same-product-and-expiry-other-exchange",
        ),
        pytest.param(
            FuturesContract(_ES, ExpirationDate("2026-12-17")),
            id="same-product-and-exchange-adjacent-expiry",
        ),
    ],
)
def test_a_bar_for_another_contract_is_rejected(intruder: FuturesContract) -> None:
    contaminated = (_B1, _B2, _bar("2026-09-16T21:00:00Z", contract=intruder))

    with pytest.raises(
        FuturesHistoricalDataContractViolationError, match="not the queried contract"
    ) as raised:
        _replay(contaminated)

    assert "observation 2" in str(raised.value)


def test_a_rebuilt_equal_contract_is_accepted_by_value() -> None:
    rebuilt = FuturesContract(
        FuturesProductReference(Symbol("ES"), ExchangeCode("CME")),
        ExpirationDate("2026-12-18"),
    )
    assert rebuilt is not _ES_DEC

    snapshots = _replay((_B1, _bar("2026-09-15T21:00:00Z", contract=rebuilt)), _query(rebuilt))

    assert len(snapshots) == 2
    assert snapshots[-1].contract == _ES_DEC


def test_a_duplicate_instant_is_rejected() -> None:
    with pytest.raises(FuturesHistoricalDataContractViolationError, match="share the instant"):
        _replay((_B1, _B2, _B2))


def test_two_spellings_of_one_instant_are_a_duplicate() -> None:
    offset_spelling = _bar("2026-09-15T16:00:00-05:00")
    assert offset_spelling.point_in_time.compare(_B2.point_in_time) == 0

    with pytest.raises(FuturesHistoricalDataContractViolationError, match="share the instant"):
        _replay((_B1, _B2, offset_spelling))


def test_out_of_order_history_is_rejected_rather_than_sorted() -> None:
    with pytest.raises(FuturesHistoricalDataContractViolationError, match="oldest to newest"):
        _replay((_B1, _B3, _B2))


def test_the_first_violation_is_reported_before_any_snapshot_is_built() -> None:
    """Validation precedes construction, so the error names the repository."""
    with pytest.raises(FuturesHistoricalDataContractViolationError, match="Repository"):
        _replay((_B2, _B1))


def test_repository_failures_propagate_unchanged() -> None:
    with pytest.raises(RuntimeError, match="storage unavailable"):
        ReplayFuturesHistoricalMarketDataUseCase(FailingRepository()).execute(_query())


def test_the_violation_error_is_the_shared_futures_contract_error() -> None:
    import northstar_application.application_services.acquire_futures_daily_history as acquire

    assert FuturesHistoricalDataContractViolationError is (
        acquire.FuturesHistoricalDataContractViolationError
    )
    assert issubclass(FuturesHistoricalDataContractViolationError, ValueError)


# ---------------------------------------------------------------------------
# Semantic timestamps
# ---------------------------------------------------------------------------


def test_whole_then_sub_second_order_is_accepted_despite_reversed_text() -> None:
    whole = _bar("2026-09-14T21:00:00Z")
    fractional = _bar("2026-09-14T21:00:00.5Z")
    assert fractional.point_in_time.value < whole.point_in_time.value

    snapshots = _replay((whole, fractional), _query())

    assert [s.latest_observation for s in snapshots] == [whole, fractional]


def test_sub_second_then_whole_order_is_rejected_despite_ascending_text() -> None:
    fractional = _bar("2026-09-14T21:00:00.5Z")
    whole = _bar("2026-09-14T21:00:00Z")
    assert fractional.point_in_time.value < whole.point_in_time.value

    with pytest.raises(FuturesHistoricalDataContractViolationError, match="oldest to newest"):
        _replay((fractional, whole), _query())


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_the_same_history_and_query_replay_identically() -> None:
    repository = StubRepository(_HISTORY)
    use_case = ReplayFuturesHistoricalMarketDataUseCase(repository)

    first = use_case.execute(_query())
    second = use_case.execute(_query())
    fresh = ReplayFuturesHistoricalMarketDataUseCase(StubRepository(_HISTORY)).execute(_query())

    assert first == second == fresh
    assert repr(first) == repr(fresh)


# ---------------------------------------------------------------------------
# Construction, purity and exports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("repository", [None, "repository", object()])
def test_the_repository_is_type_checked(repository: object) -> None:
    with pytest.raises(TypeError, match="FuturesHistoricalMarketDataRepository"):
        ReplayFuturesHistoricalMarketDataUseCase(repository)  # type: ignore[arg-type]


@pytest.mark.parametrize("query", [None, "ES", _ES_DEC])
def test_the_query_is_type_checked(query: object) -> None:
    use_case = ReplayFuturesHistoricalMarketDataUseCase(StubRepository())

    with pytest.raises(TypeError, match="FuturesHistoricalMarketDataQuery"):
        use_case.execute(query)  # type: ignore[arg-type]


def _module_tree() -> ast.Module:
    import northstar_application.application_services.replay_futures_historical_market_data as m

    return ast.parse(Path(m.__file__).read_text(encoding="utf-8"))


def test_replay_reads_no_clock_and_reaches_no_provider_or_acquisition() -> None:
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
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    for forbidden in (
        "databento",
        "exchange_calendars",
        "sqlite3",
        "requests",
        "urllib",
        "socket",
        "random",
        "time",
        "datetime",
        "uuid",
        "os",
    ):
        assert not any(m.split(".")[0] == forbidden for m in modules)

    for forbidden_name in (
        "FuturesHistoricalMarketDataSource",
        "FuturesTradingSessionResolver",
        "AcquireFuturesDailyHistoryUseCase",
        "AggregateFuturesDailySessionBarUseCase",
        "FuturesHistoricalMarketDataStore",
    ):
        assert forbidden_name not in imported_names

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"now", "utcnow", "today", "monotonic", "time"}


def test_replay_never_sorts_repository_output() -> None:
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.Call):
            func = node.func
            assert not (isinstance(func, ast.Name) and func.id in {"sorted", "reversed", "set"})
            assert not (isinstance(func, ast.Attribute) and func.attr in {"sort", "reverse"})


def test_the_use_case_is_exported() -> None:
    import northstar_application.application_services as services
    from northstar_application.application_services.replay_futures_historical_market_data import (
        ReplayFuturesHistoricalMarketDataUseCase as ViaModule,
    )

    assert "ReplayFuturesHistoricalMarketDataUseCase" in services.__all__
    assert services.ReplayFuturesHistoricalMarketDataUseCase is ViaModule


def test_the_unsupported_timeframe_error_is_exported() -> None:
    import northstar_application.application_services as services
    from northstar_application.application_services.replay_futures_historical_market_data import (
        UnsupportedFuturesReplayTimeframeError as ViaModule,
    )

    assert "UnsupportedFuturesReplayTimeframeError" in services.__all__
    assert services.UnsupportedFuturesReplayTimeframeError is ViaModule


def test_no_deferred_futures_research_concept_is_exported_yet() -> None:
    """Outcome measurement landed in 9.7d2; the research run has not."""
    import northstar_application.application_services as services

    for deferred in (
        "BuildFuturesMarketObservationContextUseCase",
        "RunFuturesHistoricalResearchUseCase",
        "FuturesHistoricalResearchRun",
        "FuturesHistoricalSnapshotEvaluator",
    ):
        assert deferred not in services.__all__
