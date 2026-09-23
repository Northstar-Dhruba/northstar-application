"""Tests for the futures daily historical acquisition workflow.

Every collaborator is a fake, and the fake store mirrors the real store's
contract exactly: a new key is accepted, an identical value under an existing
key is accepted idempotently, and a differing value under an existing key
raises. That last behaviour is what makes the retry and conflict tests mean
something -- a fake that simply counted calls would agree with any
implementation.

The sessions below are the real CME shapes confirmed by the Epic 9.6a live
probe, including the abutting boundary where one session's close is the next
session's open.
"""

from __future__ import annotations

import ast
import inspect
from datetime import date, datetime
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
from northstar_core.futures import FuturesContract, FuturesOHLCVBar, FuturesProductReference

from northstar_application.application_services import (
    AcquireFuturesDailyHistoryUseCase,
    AggregateFuturesDailySessionBarUseCase,
    FuturesDailyAcquisitionResult,
    FuturesHistoricalDataContractViolationError,
    InvalidFuturesSessionAggregationError,
)
from northstar_application.ports import (
    FuturesDailyHistoricalAcquisitionQuery,
    FuturesHistoricalMarketDataConflictError,
    FuturesHistoricalMarketDataSource,
    FuturesHistoricalMarketDataStore,
    FuturesSessionResolutionError,
    FuturesTradingSession,
    FuturesTradingSessionResolver,
    InvalidFuturesDailyHistoricalAcquisitionQueryError,
)

_CME = ExchangeCode("CME")
_ES = FuturesProductReference(Symbol("ES"), _CME)
_MES = FuturesProductReference(Symbol("MES"), _CME)

_ES_DEC = FuturesContract(_ES, ExpirationDate("2026-12-18"))
_MES_DEC = FuturesContract(_MES, ExpirationDate("2026-12-18"))

_MINUTE = Timeframe("1m")
_DAILY = Timeframe("1d")

# Consecutive CME sessions. Note session B opens exactly when session A closes.
_SESSION_A = FuturesTradingSession(
    date(2026, 9, 15), PointInTime("2026-09-14T22:00:00Z"), PointInTime("2026-09-15T22:00:00Z")
)
_SESSION_B = FuturesTradingSession(
    date(2026, 9, 16), PointInTime("2026-09-15T22:00:00Z"), PointInTime("2026-09-16T22:00:00Z")
)
_SESSION_C = FuturesTradingSession(
    date(2026, 9, 17), PointInTime("2026-09-16T22:00:00Z"), PointInTime("2026-09-17T22:00:00Z")
)
_EARLY = FuturesTradingSession(
    date(2026, 7, 3), PointInTime("2026-07-02T22:00:00Z"), PointInTime("2026-07-03T17:00:00Z")
)


def _bar(
    completion: str,
    *,
    contract: FuturesContract = _ES_DEC,
    timeframe: Timeframe = _MINUTE,
    open_quote: str = "7660",
    high: str = "7700",
    low: str = "7500",
    close: str = "7663",
    volume: str = "100",
) -> FuturesOHLCVBar:
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=PointInTime(completion),
        timeframe=timeframe,
        open=QuoteValue(Decimal(open_quote)),
        high=QuoteValue(Decimal(high)),
        low=QuoteValue(Decimal(low)),
        close=QuoteValue(Decimal(close)),
        volume=Quantity(Decimal(volume)),
    )


def _query(
    contract: FuturesContract = _ES_DEC,
    start: date = date(2026, 9, 15),
    end: date = date(2026, 9, 17),
) -> FuturesDailyHistoricalAcquisitionQuery:
    return FuturesDailyHistoricalAcquisitionQuery(contract, start, end)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSessionResolver(FuturesTradingSessionResolver):
    def __init__(self, sessions: tuple[FuturesTradingSession, ...] = ()) -> None:
        self.sessions = sessions
        self.calls: list[tuple[object, date, date]] = []
        self.error: Exception | None = None

    def resolve(self, product, trading_date):  # pragma: no cover - unused here
        raise NotImplementedError

    def sessions_in_range(self, product, start_date, end_date):
        self.calls.append((product, start_date, end_date))
        if self.error is not None:
            raise self.error
        return self.sessions


class FakeSource(FuturesHistoricalMarketDataSource):
    """Returns per-session bars, or raises for a nominated session."""

    def __init__(self, by_session: dict[date, object] | None = None) -> None:
        self.by_session = by_session or {}
        self.calls: list[tuple[FuturesContract, FuturesTradingSession]] = []
        self.failures: dict[date, Exception] = {}

    def fetch_session_bars(self, contract, session):
        self.calls.append((contract, session))
        failure = self.failures.get(session.trading_date)
        if failure is not None:
            raise failure
        return self.by_session.get(session.trading_date, ())


class ProviderUnavailableError(RuntimeError):
    """Stands in for an Infrastructure-defined provider failure."""


class FakeStore(FuturesHistoricalMarketDataStore):
    """Mirrors the real store: idempotent on equal values, conflict on differing."""

    def __init__(self) -> None:
        self.by_key: dict[tuple, FuturesOHLCVBar] = {}
        self.calls: list[tuple[FuturesOHLCVBar, ...]] = []
        self.inserts = 0
        self.return_override: int | None = None

    def store(self, bars: tuple[FuturesOHLCVBar, ...]) -> int:
        self.calls.append(bars)
        if not bars:
            return 0
        for bar in bars:
            stored = self.by_key.get(bar.natural_key)
            if stored is not None and stored != bar:
                raise FuturesHistoricalMarketDataConflictError(
                    "A different futures bar is already stored under this natural key."
                )
        for bar in bars:
            if bar.natural_key not in self.by_key:
                self.by_key[bar.natural_key] = bar
                self.inserts += 1
        return self.return_override if self.return_override is not None else len(bars)


def _use_case(
    resolver: FakeSessionResolver, source: FakeSource, store: FakeStore
) -> AcquireFuturesDailyHistoryUseCase:
    return AcquireFuturesDailyHistoryUseCase(
        resolver, source, AggregateFuturesDailySessionBarUseCase(), store
    )


def _traded(session: FuturesTradingSession, close: str = "7663") -> tuple[FuturesOHLCVBar, ...]:
    """Two in-window minute bars for a session."""
    opens = session.opens_at.value
    day = session.trading_date.isoformat()
    return (
        _bar(opens.replace("22:00:00Z", "22:01:00Z"), close="7650", volume="10"),
        _bar(f"{day}T20:59:00Z", close=close, volume="25"),
    )


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def test_a_single_date_range_is_valid() -> None:
    query = _query(start=date(2026, 9, 15), end=date(2026, 9, 15))

    assert query.start_trading_date == query.end_trading_date == date(2026, 9, 15)


def test_an_inclusive_range_is_valid() -> None:
    query = _query(start=date(2026, 9, 15), end=date(2026, 9, 17))

    assert query.contract == _ES_DEC
    assert query.start_trading_date == date(2026, 9, 15)
    assert query.end_trading_date == date(2026, 9, 17)


def test_the_contract_must_be_a_futures_contract() -> None:
    with pytest.raises(
        InvalidFuturesDailyHistoricalAcquisitionQueryError, match="must be a FuturesContract"
    ):
        FuturesDailyHistoricalAcquisitionQuery(_ES, date(2026, 9, 15), date(2026, 9, 17))


@pytest.mark.parametrize("position", ["start", "end"])
def test_a_datetime_bound_is_rejected(position: str) -> None:
    bounds = {"start": date(2026, 9, 15), "end": date(2026, 9, 17)}
    bounds[position] = datetime(2026, 9, 15, 22, 0)

    with pytest.raises(InvalidFuturesDailyHistoricalAcquisitionQueryError, match="not a datetime"):
        FuturesDailyHistoricalAcquisitionQuery(_ES_DEC, bounds["start"], bounds["end"])


@pytest.mark.parametrize("value", ["2026-09-15", 20260915, None])
def test_a_non_date_bound_is_rejected(value: object) -> None:
    with pytest.raises(InvalidFuturesDailyHistoricalAcquisitionQueryError):
        FuturesDailyHistoricalAcquisitionQuery(_ES_DEC, value, date(2026, 9, 17))


def test_an_inverted_range_is_rejected() -> None:
    with pytest.raises(
        InvalidFuturesDailyHistoricalAcquisitionQueryError, match="must not be after"
    ):
        FuturesDailyHistoricalAcquisitionQuery(_ES_DEC, date(2026, 9, 17), date(2026, 9, 15))


def test_the_query_is_immutable_and_value_typed() -> None:
    query = _query()

    with pytest.raises(AttributeError):
        query.contract = _MES_DEC
    assert query == _query()
    assert hash(query) == hash(_query())
    assert set(FuturesDailyHistoricalAcquisitionQuery.__slots__) == {
        "contract",
        "start_trading_date",
        "end_trading_date",
    }


def test_the_query_carries_no_timeframe_or_instant() -> None:
    query = _query()

    for absent in ("timeframe", "start", "end", "opens_at", "closes_at"):
        assert not hasattr(query, absent)


def test_the_query_error_is_a_value_error() -> None:
    assert issubclass(InvalidFuturesDailyHistoricalAcquisitionQueryError, ValueError)


# ---------------------------------------------------------------------------
# Source port shape
# ---------------------------------------------------------------------------


def test_the_source_port_signature_is_pinned() -> None:
    signature = inspect.signature(FuturesHistoricalMarketDataSource.fetch_session_bars)

    assert list(signature.parameters) == ["self", "contract", "session"]
    assert signature.parameters["contract"].annotation == "FuturesContract"
    assert signature.parameters["session"].annotation == "FuturesTradingSession"
    assert signature.return_annotation == "tuple[FuturesOHLCVBar, ...]"


def test_the_source_port_takes_no_timeframe() -> None:
    signature = inspect.signature(FuturesHistoricalMarketDataSource.fetch_session_bars)

    assert "timeframe" not in signature.parameters
    assert FuturesHistoricalMarketDataSource.__abstractmethods__ == frozenset(
        {"fetch_session_bars"}
    )


def test_the_source_port_is_provider_neutral() -> None:
    import northstar_application.ports.futures_historical_market_data_source as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
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

    for forbidden in ("databento", "exchange_calendars", "sqlite3", "pandas", "requests"):
        assert not any(m.split(".")[0] == forbidden for m in modules)

    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    for forbidden in ("raw_symbol", "instrument_id", "ListingReference"):
        assert forbidden not in names


def test_the_acquisition_query_is_not_the_repository_query() -> None:
    from northstar_application.ports import FuturesHistoricalMarketDataQuery

    assert FuturesDailyHistoricalAcquisitionQuery is not FuturesHistoricalMarketDataQuery
    assert not issubclass(FuturesDailyHistoricalAcquisitionQuery, FuturesHistoricalMarketDataQuery)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_sessions_are_requested_using_the_contract_product() -> None:
    resolver = FakeSessionResolver()
    source, store = FakeSource(), FakeStore()

    _use_case(resolver, source, store).execute(_query())

    assert resolver.calls == [(_ES, date(2026, 9, 15), date(2026, 9, 17))]
    assert resolver.calls[0][0] is _ES_DEC.product


def test_sessions_are_processed_in_resolver_order() -> None:
    resolver = FakeSessionResolver((_SESSION_A, _SESSION_B, _SESSION_C))
    source = FakeSource(
        {
            _SESSION_A.trading_date: _traded(_SESSION_A),
            _SESSION_B.trading_date: _traded(_SESSION_B),
            _SESSION_C.trading_date: _traded(_SESSION_C),
        }
    )
    store = FakeStore()

    _use_case(resolver, source, store).execute(_query())

    assert [session.trading_date for _, session in source.calls] == [
        date(2026, 9, 15),
        date(2026, 9, 16),
        date(2026, 9, 17),
    ]
    assert [bars[0].point_in_time for bars in store.calls] == [
        _SESSION_A.closes_at,
        _SESSION_B.closes_at,
        _SESSION_C.closes_at,
    ]


def test_weekend_and_holiday_omission_shows_in_session_count() -> None:
    """The resolver omits non-sessions; the count is how many it found."""
    resolver = FakeSessionResolver((_SESSION_A, _SESSION_B))
    source = FakeSource(
        {
            _SESSION_A.trading_date: _traded(_SESSION_A),
            _SESSION_B.trading_date: _traded(_SESSION_B),
        }
    )

    result = _use_case(resolver, source, FakeStore()).execute(
        _query(start=date(2026, 9, 12), end=date(2026, 9, 16))
    )

    assert result.session_count == 2
    assert result.daily_bar_count == 2


def test_a_no_trade_session_stores_nothing() -> None:
    resolver = FakeSessionResolver((_SESSION_A,))
    store = FakeStore()

    result = _use_case(resolver, FakeSource(), store).execute(_query())

    assert store.calls == []
    assert result.session_count == 1
    assert result.daily_bar_count == 0


def test_mixed_traded_and_no_trade_sessions() -> None:
    resolver = FakeSessionResolver((_SESSION_A, _SESSION_B, _SESSION_C))
    source = FakeSource(
        {
            _SESSION_A.trading_date: _traded(_SESSION_A),
            _SESSION_C.trading_date: _traded(_SESSION_C),
        }
    )
    store = FakeStore()

    result = _use_case(resolver, source, store).execute(_query())

    assert result.session_count == 3
    assert result.daily_bar_count == 2
    assert len(store.calls) == 2


def test_each_store_call_contains_exactly_one_daily_bar() -> None:
    resolver = FakeSessionResolver((_SESSION_A, _SESSION_B))
    source = FakeSource(
        {
            _SESSION_A.trading_date: _traded(_SESSION_A),
            _SESSION_B.trading_date: _traded(_SESSION_B),
        }
    )
    store = FakeStore()

    _use_case(resolver, source, store).execute(_query())

    assert len(store.calls) == 2
    for call in store.calls:
        assert isinstance(call, tuple)
        assert len(call) == 1
        assert call[0].timeframe == _DAILY
        assert call[0].contract == _ES_DEC


def test_an_empty_session_range_returns_zero_counts() -> None:
    result = _use_case(FakeSessionResolver(), FakeSource(), FakeStore()).execute(_query())

    assert result.session_count == 0
    assert result.daily_bar_count == 0


def test_the_daily_bar_folds_the_sessions_minutes() -> None:
    resolver = FakeSessionResolver((_SESSION_A,))
    source = FakeSource({_SESSION_A.trading_date: _traded(_SESSION_A, close="7671")})
    store = FakeStore()

    _use_case(resolver, source, store).execute(_query())

    (daily,) = store.calls[0]
    assert daily.point_in_time == _SESSION_A.closes_at
    assert daily.close == QuoteValue(Decimal("7671"))
    assert daily.volume == Quantity(Decimal("35"))


# ---------------------------------------------------------------------------
# Source contract violations
# ---------------------------------------------------------------------------


def _violating(bars: object) -> AcquireFuturesDailyHistoryUseCase:
    resolver = FakeSessionResolver((_SESSION_A,))
    source = FakeSource({_SESSION_A.trading_date: bars})
    return _use_case(resolver, source, FakeStore())


@pytest.mark.parametrize(
    ("label", "bars"),
    [
        ("list_not_tuple", [_bar("2026-09-15T14:30:00Z")]),
        ("wrong_member_type", ("not a bar",)),
        ("wrong_contract", (_bar("2026-09-15T14:30:00Z", contract=_MES_DEC),)),
        ("wrong_timeframe", (_bar("2026-09-15T14:30:00Z", timeframe=Timeframe("1h")),)),
        (
            "reversed",
            (_bar("2026-09-15T14:31:00Z"), _bar("2026-09-15T14:30:00Z")),
        ),
        (
            "duplicate_instant",
            (_bar("2026-09-15T14:30:00Z"), _bar("2026-09-15T14:30:00Z", close="7664")),
        ),
        ("at_opens_at", (_bar("2026-09-14T22:00:00Z"),)),
        ("after_closes_at", (_bar("2026-09-15T22:01:00Z"),)),
        ("fractional_volume", (_bar("2026-09-15T14:30:00Z", volume="1.5"),)),
    ],
)
def test_a_source_contract_violation_is_reported_as_such(label: str, bars: object) -> None:
    with pytest.raises(FuturesHistoricalDataContractViolationError) as raised:
        _violating(bars).execute(_query())

    assert isinstance(raised.value.__cause__, InvalidFuturesSessionAggregationError)


def test_the_violation_message_names_the_contract_and_session() -> None:
    with pytest.raises(FuturesHistoricalDataContractViolationError) as raised:
        _violating((_bar("2026-09-15T22:01:00Z"),)).execute(_query())

    message = str(raised.value)
    assert "2026-09-15" in message
    assert "ES@CME" in message


def test_a_violation_in_a_later_session_leaves_earlier_sessions_persisted() -> None:
    resolver = FakeSessionResolver((_SESSION_A, _SESSION_B))
    source = FakeSource(
        {
            _SESSION_A.trading_date: _traded(_SESSION_A),
            _SESSION_B.trading_date: (_bar("2026-09-16T22:01:00Z"),),
        }
    )
    store = FakeStore()

    with pytest.raises(FuturesHistoricalDataContractViolationError):
        _use_case(resolver, source, store).execute(_query())

    assert store.inserts == 1
    assert len(store.by_key) == 1


def test_the_violation_error_is_a_value_error() -> None:
    assert issubclass(FuturesHistoricalDataContractViolationError, ValueError)


# ---------------------------------------------------------------------------
# Session boundary
# ---------------------------------------------------------------------------


def test_a_bar_completing_at_the_session_open_is_refused() -> None:
    """Its interval opened before the session; it is the previous session's."""
    with pytest.raises(FuturesHistoricalDataContractViolationError):
        _violating((_bar("2026-09-14T22:00:00Z"),)).execute(_query())


def test_the_first_completion_after_the_open_is_accepted() -> None:
    resolver = FakeSessionResolver((_SESSION_A,))
    source = FakeSource({_SESSION_A.trading_date: (_bar("2026-09-14T22:01:00Z"),)})
    store = FakeStore()

    assert _use_case(resolver, source, store).execute(_query()).daily_bar_count == 1


def test_a_completion_exactly_at_the_close_is_accepted() -> None:
    resolver = FakeSessionResolver((_SESSION_A,))
    source = FakeSource({_SESSION_A.trading_date: (_bar("2026-09-15T22:00:00Z"),)})
    store = FakeStore()

    assert _use_case(resolver, source, store).execute(_query()).daily_bar_count == 1


def test_the_previous_sessions_final_bar_cannot_leak_into_the_next_session() -> None:
    """The exact instant an inclusive instant-range query would have admitted."""
    boundary = _bar("2026-09-15T22:00:00Z")

    assert _SESSION_A.closes_at == _SESSION_B.opens_at
    assert _SESSION_A.closes_at == boundary.point_in_time

    resolver = FakeSessionResolver((_SESSION_B,))
    source = FakeSource({_SESSION_B.trading_date: (boundary,)})

    with pytest.raises(FuturesHistoricalDataContractViolationError):
        _use_case(resolver, source, FakeStore()).execute(_query())


def test_the_early_close_final_minute_is_acquired() -> None:
    resolver = FakeSessionResolver((_EARLY,))
    source = FakeSource({_EARLY.trading_date: (_bar("2026-07-03T17:00:00Z", close="7623.75"),)})
    store = FakeStore()

    result = _use_case(resolver, source, store).execute(
        _query(start=date(2026, 7, 3), end=date(2026, 7, 3))
    )

    assert result.daily_bar_count == 1
    assert store.calls[0][0].point_in_time == PointInTime("2026-07-03T17:00:00Z")


# ---------------------------------------------------------------------------
# Persistence and retry
# ---------------------------------------------------------------------------


def test_a_store_accepting_one_bar_is_counted() -> None:
    resolver = FakeSessionResolver((_SESSION_A,))
    source = FakeSource({_SESSION_A.trading_date: _traded(_SESSION_A)})
    store = FakeStore()

    assert _use_case(resolver, source, store).execute(_query()).daily_bar_count == 1
    assert store.inserts == 1


def test_an_identical_rerun_is_idempotent_and_reports_the_same_counts() -> None:
    """daily_bar_count is sessions persisted, not rows inserted."""
    resolver = FakeSessionResolver((_SESSION_A, _SESSION_B))
    source = FakeSource(
        {
            _SESSION_A.trading_date: _traded(_SESSION_A),
            _SESSION_B.trading_date: _traded(_SESSION_B),
        }
    )
    store = FakeStore()
    use_case = _use_case(resolver, source, store)

    first = use_case.execute(_query())
    second = use_case.execute(_query())

    assert first == second
    assert first.daily_bar_count == 2
    assert store.inserts == 2  # no new rows on the rerun
    assert len(store.calls) == 4


def test_a_store_conflict_propagates_unwrapped() -> None:
    """A conflict is the one signal requiring an explicit reconciliation."""
    resolver = FakeSessionResolver((_SESSION_A,))
    store = FakeStore()

    first_source = FakeSource({_SESSION_A.trading_date: _traded(_SESSION_A, close="7663")})
    _use_case(resolver, first_source, store).execute(_query())

    second_source = FakeSource({_SESSION_A.trading_date: _traded(_SESSION_A, close="7690")})

    with pytest.raises(FuturesHistoricalMarketDataConflictError):
        _use_case(resolver, second_source, store).execute(_query())


def test_a_later_store_conflict_leaves_earlier_sessions_persisted() -> None:
    resolver = FakeSessionResolver((_SESSION_A, _SESSION_B))
    store = FakeStore()

    store.by_key[(_ES_DEC, _SESSION_B.closes_at, _DAILY)] = _bar(
        _SESSION_B.closes_at.value, timeframe=_DAILY, close="9999", high="9999", low="0"
    )

    source = FakeSource(
        {
            _SESSION_A.trading_date: _traded(_SESSION_A),
            _SESSION_B.trading_date: _traded(_SESSION_B),
        }
    )

    with pytest.raises(FuturesHistoricalMarketDataConflictError):
        _use_case(resolver, source, store).execute(_query())

    assert (_ES_DEC, _SESSION_A.closes_at, _DAILY) in store.by_key


def test_a_later_provider_failure_leaves_earlier_sessions_persisted() -> None:
    resolver = FakeSessionResolver((_SESSION_A, _SESSION_B))
    source = FakeSource({_SESSION_A.trading_date: _traded(_SESSION_A)})
    source.failures[_SESSION_B.trading_date] = ProviderUnavailableError("provider down")
    store = FakeStore()

    with pytest.raises(ProviderUnavailableError):
        _use_case(resolver, source, store).execute(_query())

    assert store.inserts == 1
    assert (_ES_DEC, _SESSION_A.closes_at, _DAILY) in store.by_key


def test_no_result_is_returned_on_partial_failure() -> None:
    resolver = FakeSessionResolver((_SESSION_A, _SESSION_B))
    source = FakeSource({_SESSION_A.trading_date: _traded(_SESSION_A)})
    source.failures[_SESSION_B.trading_date] = ProviderUnavailableError("provider down")

    with pytest.raises(ProviderUnavailableError):
        _use_case(resolver, source, FakeStore()).execute(_query())


@pytest.mark.parametrize("returned", [0, 2, -1])
def test_an_unexpected_store_count_is_detected(returned: int) -> None:
    resolver = FakeSessionResolver((_SESSION_A,))
    source = FakeSource({_SESSION_A.trading_date: _traded(_SESSION_A)})
    store = FakeStore()
    store.return_override = returned

    with pytest.raises(FuturesHistoricalDataContractViolationError, match="returned count"):
        _use_case(resolver, source, store).execute(_query())


def test_a_non_integer_store_count_is_detected() -> None:
    resolver = FakeSessionResolver((_SESSION_A,))
    source = FakeSource({_SESSION_A.trading_date: _traded(_SESSION_A)})
    store = FakeStore()
    store.return_override = "1"  # type: ignore[assignment]

    with pytest.raises(FuturesHistoricalDataContractViolationError, match="must return an integer"):
        _use_case(resolver, source, store).execute(_query())


def test_a_session_resolution_error_propagates_unwrapped() -> None:
    resolver = FakeSessionResolver()
    resolver.error = FuturesSessionResolutionError("Unsupported futures venue: XXXX.")

    with pytest.raises(FuturesSessionResolutionError):
        _use_case(resolver, FakeSource(), FakeStore()).execute(_query())


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


def test_the_result_carries_the_query_and_both_counts() -> None:
    resolver = FakeSessionResolver((_SESSION_A, _SESSION_B))
    source = FakeSource({_SESSION_A.trading_date: _traded(_SESSION_A)})

    result = _use_case(resolver, source, FakeStore()).execute(_query())

    assert result.query == _query()
    assert result.session_count == 2
    assert result.daily_bar_count == 1


def test_the_result_is_immutable() -> None:
    result = FuturesDailyAcquisitionResult(_query(), 2, 1)

    with pytest.raises(AttributeError):
        result.daily_bar_count = 5


def test_daily_bar_count_cannot_exceed_session_count() -> None:
    with pytest.raises(ValueError, match="cannot exceed session_count"):
        FuturesDailyAcquisitionResult(_query(), 1, 2)


@pytest.mark.parametrize(("sessions", "bars"), [(-1, 0), (0, -1)])
def test_negative_counts_are_rejected(sessions: int, bars: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        FuturesDailyAcquisitionResult(_query(), sessions, bars)


@pytest.mark.parametrize("value", [1.0, "1", True])
def test_non_integer_counts_are_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        FuturesDailyAcquisitionResult(_query(), value, 0)


def test_the_result_exposes_no_minute_bar_count() -> None:
    result = FuturesDailyAcquisitionResult(_query(), 2, 1)

    for absent in ("minute_bar_count", "acquired_count", "stored_bar_count", "stored_count"):
        assert not hasattr(result, absent)


# ---------------------------------------------------------------------------
# Construction and purity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_every_dependency_is_type_checked(position: int) -> None:
    dependencies: list[object] = [
        FakeSessionResolver(),
        FakeSource(),
        AggregateFuturesDailySessionBarUseCase(),
        FakeStore(),
    ]
    dependencies[position] = "not a dependency"

    with pytest.raises(TypeError):
        AcquireFuturesDailyHistoryUseCase(*dependencies)  # type: ignore[arg-type]


def test_a_foreign_query_is_rejected() -> None:
    with pytest.raises(TypeError, match="FuturesDailyHistoricalAcquisitionQuery"):
        _use_case(FakeSessionResolver(), FakeSource(), FakeStore()).execute("2026-09-15")


def test_the_use_case_reads_no_clock_and_touches_no_provider() -> None:
    import northstar_application.application_services.acquire_futures_daily_history as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
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

    for forbidden in (
        "databento",
        "exchange_calendars",
        "sqlite3",
        "requests",
        "urllib",
        "random",
        "time",
        "os",
    ):
        assert not any(m.split(".")[0] == forbidden for m in modules)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"now", "utcnow", "today", "monotonic"}


def test_the_use_case_never_sorts_source_output() -> None:
    """Malformed order is refused; reordering would hide the defect."""
    import northstar_application.application_services.acquire_futures_daily_history as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                assert node.func.id not in {"sorted", "sort"}
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr != "sort"
