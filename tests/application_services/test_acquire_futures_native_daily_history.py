"""Tests for the provider-native daily futures acquisition workflow.

Every collaborator is a fake. The session resolver reads a fixed window table,
so which dates are sessions is decided only by that table -- exactly the
authority the use case must defer to. The fake store mirrors the real store's
contract: a new key is accepted, an identical value under an existing key is
accepted idempotently, and a differing value under an existing key raises.

The sessions are NSE-shaped (09:15-15:30 IST, i.e. 03:45Z-10:00Z) around the
2026-10-02 holiday and the following weekend. Nothing NSE-specific is assumed by
the use case; the table could equally be any venue's.
"""

from __future__ import annotations

import ast
import inspect
import random
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
    AcquireFuturesNativeDailyHistoryUseCase,
    FuturesDailyAcquisitionResult,
    FuturesDailySessionCoverageError,
    FuturesHistoricalDataContractViolationError,
)
from northstar_application.ports import (
    FuturesDailyHistoricalAcquisitionQuery,
    FuturesHistoricalMarketDataConflictError,
    FuturesHistoricalMarketDataSource,
    FuturesHistoricalMarketDataStore,
    FuturesNativeDailyMarketDataSource,
    FuturesNativeDailyObservation,
    FuturesSessionResolutionError,
    FuturesTradingSession,
    FuturesTradingSessionResolver,
    InvalidFuturesNativeDailyObservationError,
)

_NSE = ExchangeCode("NSE")
_NIFTY = FuturesProductReference(Symbol("NIFTY"), _NSE)
_BANKNIFTY = FuturesProductReference(Symbol("BANKNIFTY"), _NSE)

_NIFTY_OCT = FuturesContract(_NIFTY, ExpirationDate("2026-10-27"))
_NIFTY_NOV = FuturesContract(_NIFTY, ExpirationDate("2026-11-24"))
_BANKNIFTY_OCT = FuturesContract(_BANKNIFTY, ExpirationDate("2026-10-27"))

_DAILY = Timeframe("1d")

_TUE = date(2026, 9, 29)
_WED = date(2026, 9, 30)
_THU = date(2026, 10, 1)
_HOLIDAY = date(2026, 10, 2)
_SAT = date(2026, 10, 3)
_SUN = date(2026, 10, 4)
_MON = date(2026, 10, 5)


def _window(label: date) -> tuple[str, str]:
    day = label.isoformat()
    return (f"{day}T03:45:00Z", f"{day}T10:00:00Z")


# The calendar. 2026-10-02 and the weekend are absent, so they are not sessions.
_WINDOWS: dict[date, tuple[str, str]] = {
    label: _window(label) for label in (_TUE, _WED, _THU, _MON)
}


def _session(label: date) -> FuturesTradingSession:
    opens, closes = _WINDOWS[label]
    return FuturesTradingSession(label, PointInTime(opens), PointInTime(closes))


def _obs(
    trading_date: date,
    *,
    contract: FuturesContract = _NIFTY_OCT,
    open_quote: str = "25100",
    high: str = "25250",
    low: str = "25020",
    close: str = "25180.5",
    volume: str = "4200",
) -> FuturesNativeDailyObservation:
    return FuturesNativeDailyObservation(
        contract=contract,
        trading_date=trading_date,
        open=QuoteValue(Decimal(open_quote)),
        high=QuoteValue(Decimal(high)),
        low=QuoteValue(Decimal(low)),
        close=QuoteValue(Decimal(close)),
        volume=Quantity(Decimal(volume)),
    )


def _query(
    contract: FuturesContract = _NIFTY_OCT,
    start: date = _TUE,
    end: date = _MON,
) -> FuturesDailyHistoricalAcquisitionQuery:
    return FuturesDailyHistoricalAcquisitionQuery(contract, start, end)


def _week() -> tuple[FuturesNativeDailyObservation, ...]:
    """One distinct candle per resolved session in the default range."""
    return (
        _obs(_TUE, close="25180.5", volume="4200"),
        _obs(_WED, close="25200", volume="3900"),
        _obs(_THU, close="25150", volume="5100"),
        _obs(_MON, close="25230", volume="4700"),
    )


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class TableSessionResolver(FuturesTradingSessionResolver):
    """Answers only from a window table; anything absent is not a session."""

    def __init__(self, windows: dict[date, tuple[str, str]] | None = None) -> None:
        self.windows = _WINDOWS if windows is None else windows
        self.calls: list[tuple[FuturesProductReference, date, date]] = []
        self.error: Exception | None = None

    def resolve(self, product, trading_date):  # pragma: no cover - unused here
        raise NotImplementedError

    def sessions_in_range(self, product, start_date, end_date):
        self.calls.append((product, start_date, end_date))
        if self.error is not None:
            raise self.error
        return tuple(
            FuturesTradingSession(label, PointInTime(opens), PointInTime(closes))
            for label, (opens, closes) in sorted(self.windows.items())
            if start_date <= label <= end_date
        )


class FakeSource(FuturesNativeDailyMarketDataSource):
    def __init__(self, observations: object = ()) -> None:
        self.observations = observations
        self.calls: list[tuple[FuturesContract, date, date]] = []
        self.error: Exception | None = None

    def fetch_daily_observations(self, contract, start_trading_date, end_trading_date):
        self.calls.append((contract, start_trading_date, end_trading_date))
        if self.error is not None:
            raise self.error
        return self.observations


class ProviderUnavailableError(RuntimeError):
    """Stands in for an Infrastructure-defined provider failure."""


class FakeStore(FuturesHistoricalMarketDataStore):
    """Mirrors the real store: atomic batch, idempotent on equal, conflict on differing."""

    def __init__(self) -> None:
        self.by_key: dict[tuple, FuturesOHLCVBar] = {}
        self.calls: list[tuple[FuturesOHLCVBar, ...]] = []
        self.inserts = 0
        self.return_override: object | None = None

    def store(self, bars: tuple[FuturesOHLCVBar, ...]) -> int:
        self.calls.append(bars)
        keys = [bar.natural_key for bar in bars]
        assert len(keys) == len(set(keys)), "batch must not repeat a natural key"
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
    resolver: TableSessionResolver | None = None,
    source: FakeSource | None = None,
    store: FakeStore | None = None,
) -> AcquireFuturesNativeDailyHistoryUseCase:
    return AcquireFuturesNativeDailyHistoryUseCase(
        resolver or TableSessionResolver(), source or FakeSource(), store or FakeStore()
    )


def _stored(store: FakeStore) -> tuple[FuturesOHLCVBar, ...]:
    return tuple(bar for batch in store.calls for bar in batch)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


def test_an_observation_carries_contract_label_and_values() -> None:
    observation = _obs(_TUE)

    assert observation.contract == _NIFTY_OCT
    assert observation.trading_date == _TUE
    assert observation.close == QuoteValue(Decimal("25180.5"))
    assert observation.volume == Quantity(Decimal("4200"))


def test_an_observation_is_immutable_and_value_typed() -> None:
    with pytest.raises(AttributeError):
        _obs(_TUE).trading_date = _WED
    assert _obs(_TUE) == _obs(_TUE)
    assert hash(_obs(_TUE)) == hash(_obs(_TUE))


def test_an_observation_carries_no_instant_or_provider_identity() -> None:
    assert set(FuturesNativeDailyObservation.__slots__) == {
        "contract",
        "trading_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    for absent in ("point_in_time", "timestamp", "instrument_key", "timeframe", "open_interest"):
        assert not hasattr(_obs(_TUE), absent)


def test_an_observation_requires_a_futures_contract() -> None:
    with pytest.raises(InvalidFuturesNativeDailyObservationError, match="FuturesContract"):
        _obs(_TUE, contract=_NIFTY)  # type: ignore[arg-type]


def test_an_observation_rejects_a_datetime_label() -> None:
    with pytest.raises(InvalidFuturesNativeDailyObservationError, match="not a datetime"):
        _obs(datetime(2026, 9, 29, 0, 0))


@pytest.mark.parametrize("value", ["2026-09-29", 20260929, None])
def test_an_observation_rejects_a_non_date_label(value: object) -> None:
    with pytest.raises(InvalidFuturesNativeDailyObservationError):
        _obs(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["open", "high", "low", "close"])
def test_an_observation_requires_quote_values(field: str) -> None:
    values = {
        "contract": _NIFTY_OCT,
        "trading_date": _TUE,
        "open": QuoteValue(Decimal("1")),
        "high": QuoteValue(Decimal("1")),
        "low": QuoteValue(Decimal("1")),
        "close": QuoteValue(Decimal("1")),
        "volume": Quantity(Decimal("1")),
    }
    values[field] = Decimal("1")

    with pytest.raises(InvalidFuturesNativeDailyObservationError, match=field):
        FuturesNativeDailyObservation(**values)


def test_an_observation_requires_a_quantity_volume() -> None:
    with pytest.raises(InvalidFuturesNativeDailyObservationError, match="Quantity"):
        FuturesNativeDailyObservation(
            _NIFTY_OCT,
            _TUE,
            QuoteValue(Decimal("1")),
            QuoteValue(Decimal("1")),
            QuoteValue(Decimal("1")),
            QuoteValue(Decimal("1")),
            Decimal("1"),  # type: ignore[arg-type]
        )


def test_an_observation_refuses_a_fractional_volume() -> None:
    with pytest.raises(InvalidFuturesNativeDailyObservationError, match="integral"):
        _obs(_TUE, volume="10.5")


def test_the_observation_error_is_a_value_error() -> None:
    assert issubclass(InvalidFuturesNativeDailyObservationError, ValueError)


# ---------------------------------------------------------------------------
# Source port shape
# ---------------------------------------------------------------------------


def test_the_source_port_signature_is_pinned() -> None:
    signature = inspect.signature(FuturesNativeDailyMarketDataSource.fetch_daily_observations)

    assert list(signature.parameters) == [
        "self",
        "contract",
        "start_trading_date",
        "end_trading_date",
    ]
    assert signature.parameters["contract"].annotation == "FuturesContract"
    assert signature.parameters["start_trading_date"].annotation == "date"
    assert signature.parameters["end_trading_date"].annotation == "date"
    assert signature.return_annotation == "tuple[FuturesNativeDailyObservation, ...]"
    assert FuturesNativeDailyMarketDataSource.__abstractmethods__ == frozenset(
        {"fetch_daily_observations"}
    )


def test_the_native_port_is_separate_from_the_minute_port() -> None:
    assert not issubclass(FuturesNativeDailyMarketDataSource, FuturesHistoricalMarketDataSource)
    assert not issubclass(FuturesHistoricalMarketDataSource, FuturesNativeDailyMarketDataSource)


def test_the_source_port_is_provider_neutral() -> None:
    import northstar_application.ports.futures_native_daily_market_data_source as module

    text = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
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

    for forbidden in ("upstox", "databento", "exchange_calendars", "requests", "httpx", "urllib"):
        assert not any(m.split(".")[0] == forbidden for m in modules)
    for forbidden in ("upstox", "instrument_key", "NSE_FO", "access_token"):
        assert forbidden.lower() not in text.lower()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_the_exact_contract_and_range_are_passed_to_the_source() -> None:
    source = FakeSource(_week())

    _use_case(source=source).execute(_query(_NIFTY_OCT, _TUE, _MON))

    assert source.calls == [(_NIFTY_OCT, _TUE, _MON)]
    assert source.calls[0][0] is _NIFTY_OCT


def test_sessions_are_resolved_for_the_contract_product() -> None:
    resolver = TableSessionResolver()

    _use_case(resolver=resolver, source=FakeSource(_week())).execute(_query())

    assert resolver.calls == [(_NIFTY, _TUE, _MON)]


def test_multiple_observations_become_canonical_daily_bars() -> None:
    store = FakeStore()

    _use_case(source=FakeSource(_week()), store=store).execute(_query())

    bars = _stored(store)
    assert len(bars) == 4
    for bar, observation in zip(bars, _week(), strict=True):
        assert isinstance(bar, FuturesOHLCVBar)
        assert bar.timeframe == _DAILY
        assert bar.open == observation.open
        assert bar.high == observation.high
        assert bar.low == observation.low
        assert bar.close == observation.close
        assert bar.volume == observation.volume


def test_each_bar_is_stamped_at_its_resolved_session_close() -> None:
    store = FakeStore()

    _use_case(source=FakeSource(_week()), store=store).execute(_query())

    assert [bar.point_in_time for bar in _stored(store)] == [
        _session(label).closes_at for label in (_TUE, _WED, _THU, _MON)
    ]
    assert _stored(store)[0].point_in_time == PointInTime("2026-09-29T10:00:00Z")


def test_the_stamp_follows_the_calendar_not_a_fixed_hour() -> None:
    """A shortened session moves the stamp; nothing assumes a standard close."""
    windows = dict(_WINDOWS)
    windows[_WED] = ("2026-09-30T03:45:00Z", "2026-09-30T07:30:00Z")
    store = FakeStore()

    _use_case(TableSessionResolver(windows), FakeSource(_week()), store).execute(_query())

    assert _stored(store)[1].point_in_time == PointInTime("2026-09-30T07:30:00Z")


def test_the_stamp_is_never_the_session_open_or_a_midnight() -> None:
    store = FakeStore()

    _use_case(source=FakeSource(_week()), store=store).execute(_query())

    for bar in _stored(store):
        assert not bar.point_in_time.value.endswith("T00:00:00Z")
        assert not bar.point_in_time.value.endswith("T03:45:00Z")


def test_weekends_and_holidays_are_governed_by_the_resolver() -> None:
    """The range names seven days; the calendar says four are sessions."""
    store = FakeStore()

    result = _use_case(source=FakeSource(_week()), store=store).execute(_query())

    assert result.session_count == 4
    stored_labels = {bar.point_in_time.value[:10] for bar in _stored(store)}
    assert stored_labels == {"2026-09-29", "2026-09-30", "2026-10-01", "2026-10-05"}
    for absent in (_HOLIDAY, _SAT, _SUN):
        assert absent.isoformat() not in stored_labels


def test_a_session_the_calendar_adds_on_a_weekend_is_expected() -> None:
    """Only the calendar decides; a special weekend session must have a candle."""
    windows = dict(_WINDOWS)
    windows[_SAT] = _window(_SAT)
    observations = (*_week(), _obs(_SAT))
    store = FakeStore()

    result = _use_case(TableSessionResolver(windows), FakeSource(observations), store).execute(
        _query()
    )

    assert result.session_count == result.daily_bar_count == 5
    assert _stored(store)[3].point_in_time == PointInTime("2026-10-03T10:00:00Z")


def test_the_whole_range_is_stored_in_one_batch() -> None:
    store = FakeStore()

    _use_case(source=FakeSource(_week()), store=store).execute(_query())

    assert len(store.calls) == 1
    assert len(store.calls[0]) == 4


def test_the_result_counts_every_resolved_session() -> None:
    result = _use_case(source=FakeSource(_week())).execute(_query())

    assert isinstance(result, FuturesDailyAcquisitionResult)
    assert result.query == _query()
    assert result.session_count == 4
    assert result.daily_bar_count == 4


def test_a_range_with_no_sessions_stores_nothing() -> None:
    store = FakeStore()

    result = _use_case(store=store).execute(_query(start=_HOLIDAY, end=_SUN))

    assert result.session_count == result.daily_bar_count == 0
    assert store.calls == []


# ---------------------------------------------------------------------------
# Coverage: the calendar and the source must agree exactly
# ---------------------------------------------------------------------------


def test_a_missing_expected_session_fails_explicitly() -> None:
    observations = tuple(obs for obs in _week() if obs.trading_date != _WED)
    store = FakeStore()

    with pytest.raises(FuturesDailySessionCoverageError, match="2026-09-30") as excinfo:
        _use_case(source=FakeSource(observations), store=store).execute(_query())

    assert excinfo.value.missing_trading_dates == (_WED,)
    assert excinfo.value.unexpected_trading_dates == ()
    assert excinfo.value.contract == _NIFTY_OCT
    assert store.calls == []


def test_a_missing_final_session_fails_rather_than_shortening_the_range() -> None:
    """A not-yet-published latest candle is a gap, never a shorter success."""
    observations = tuple(obs for obs in _week() if obs.trading_date != _MON)
    store = FakeStore()

    with pytest.raises(FuturesDailySessionCoverageError) as excinfo:
        _use_case(source=FakeSource(observations), store=store).execute(_query())

    assert excinfo.value.missing_trading_dates == (_MON,)
    assert store.calls == []


def test_an_empty_source_for_a_range_with_sessions_fails() -> None:
    with pytest.raises(FuturesDailySessionCoverageError) as excinfo:
        _use_case(source=FakeSource(())).execute(_query())

    assert excinfo.value.missing_trading_dates == (_TUE, _WED, _THU, _MON)


@pytest.mark.parametrize("label", [_HOLIDAY, _SAT, _SUN])
def test_a_candle_for_a_non_trading_date_fails(label: date) -> None:
    store = FakeStore()

    with pytest.raises(FuturesDailySessionCoverageError, match=label.isoformat()) as excinfo:
        _use_case(source=FakeSource((*_week(), _obs(label))), store=store).execute(_query())

    assert excinfo.value.unexpected_trading_dates == (label,)
    assert excinfo.value.missing_trading_dates == ()
    assert store.calls == []


def test_a_candle_outside_the_requested_range_fails() -> None:
    with pytest.raises(FuturesDailySessionCoverageError) as excinfo:
        _use_case(source=FakeSource((*_week(), _obs(date(2026, 10, 6))))).execute(_query())

    assert excinfo.value.unexpected_trading_dates == (date(2026, 10, 6),)


def test_missing_and_unexpected_dates_are_both_reported() -> None:
    observations = (_obs(_TUE), _obs(_WED), _obs(_HOLIDAY), _obs(_MON))

    with pytest.raises(FuturesDailySessionCoverageError) as excinfo:
        _use_case(source=FakeSource(observations)).execute(_query())

    assert excinfo.value.missing_trading_dates == (_THU,)
    assert excinfo.value.unexpected_trading_dates == (_HOLIDAY,)


def test_the_coverage_error_is_not_a_source_contract_violation() -> None:
    assert issubclass(FuturesDailySessionCoverageError, ValueError)
    assert not issubclass(
        FuturesDailySessionCoverageError, FuturesHistoricalDataContractViolationError
    )


# ---------------------------------------------------------------------------
# Source contract violations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "observations",
    [
        (_obs(_TUE), _obs(_TUE, close="25190")),
        (_obs(_TUE), _obs(_TUE)),
    ],
    ids=["differing", "identical"],
)
def test_duplicate_source_dates_fail(observations: tuple) -> None:
    store = FakeStore()

    with pytest.raises(FuturesHistoricalDataContractViolationError, match="more than one candle"):
        _use_case(source=FakeSource(observations), store=store).execute(_query(end=_TUE))

    assert store.calls == []


@pytest.mark.parametrize("contract", [_NIFTY_NOV, _BANKNIFTY_OCT], ids=["expiry", "product"])
def test_a_candle_for_another_contract_fails(contract: FuturesContract) -> None:
    observations = (_obs(_TUE), _obs(_WED, contract=contract), _obs(_THU), _obs(_MON))
    store = FakeStore()

    with pytest.raises(FuturesHistoricalDataContractViolationError, match="not the requested"):
        _use_case(source=FakeSource(observations), store=store).execute(_query())

    assert store.calls == []


@pytest.mark.parametrize(
    "observations",
    [[_obs(_TUE)], None, "candles", (_obs(_TUE), "not an observation")],
    ids=["list", "none", "string", "foreign-member"],
)
def test_malformed_source_output_fails(observations: object) -> None:
    with pytest.raises(FuturesHistoricalDataContractViolationError):
        _use_case(source=FakeSource(observations)).execute(_query(end=_TUE))


def test_an_incoherent_candle_fails_as_a_source_violation() -> None:
    store = FakeStore()
    observations = (_obs(_TUE, high="25000", low="25020"),)

    with pytest.raises(FuturesHistoricalDataContractViolationError, match="2026-09-29"):
        _use_case(source=FakeSource(observations), store=store).execute(_query(end=_TUE))

    assert store.calls == []


def test_a_provider_failure_propagates_unwrapped_and_stores_nothing() -> None:
    source = FakeSource()
    source.error = ProviderUnavailableError("provider down")
    store = FakeStore()

    with pytest.raises(ProviderUnavailableError):
        _use_case(source=source, store=store).execute(_query())

    assert store.calls == []


def test_a_session_resolution_error_propagates_unwrapped() -> None:
    resolver = TableSessionResolver()
    resolver.error = FuturesSessionResolutionError("calendar unavailable")
    source = FakeSource(_week())

    with pytest.raises(FuturesSessionResolutionError):
        _use_case(resolver=resolver, source=source).execute(_query())


# ---------------------------------------------------------------------------
# Determinism and identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_source_order_does_not_affect_persisted_order(seed: int) -> None:
    shuffled = list(_week())
    random.Random(seed).shuffle(shuffled)
    expected_store, shuffled_store = FakeStore(), FakeStore()

    _use_case(source=FakeSource(_week()), store=expected_store).execute(_query())
    _use_case(source=FakeSource(tuple(shuffled)), store=shuffled_store).execute(_query())

    assert shuffled_store.calls == expected_store.calls


def test_a_reversed_source_is_persisted_oldest_to_newest() -> None:
    store = FakeStore()

    _use_case(source=FakeSource(tuple(reversed(_week()))), store=store).execute(_query())

    instants = [bar.point_in_time for bar in _stored(store)]
    for earlier, later in zip(instants, instants[1:], strict=False):
        assert earlier.compare(later) < 0


def test_exact_contract_identity_is_preserved() -> None:
    store = FakeStore()
    observations = tuple(_obs(label, contract=_NIFTY_NOV) for label in (_TUE, _WED, _THU, _MON))

    _use_case(source=FakeSource(observations), store=store).execute(_query(_NIFTY_NOV))

    for bar in _stored(store):
        assert bar.contract == _NIFTY_NOV
        assert bar.contract != _NIFTY_OCT
        assert bar.contract.expiration_date == ExpirationDate("2026-11-24")
        assert bar.natural_key == (_NIFTY_NOV, bar.point_in_time, _DAILY)


# ---------------------------------------------------------------------------
# Idempotency and persistence
# ---------------------------------------------------------------------------


def test_an_identical_rerun_is_idempotent() -> None:
    store = FakeStore()
    use_case = _use_case(source=FakeSource(_week()), store=store)

    first = use_case.execute(_query())
    second = use_case.execute(_query())

    assert first == second
    assert store.inserts == 4
    assert len(store.by_key) == 4


def test_a_store_conflict_propagates_unwrapped() -> None:
    store = FakeStore()
    _use_case(source=FakeSource(_week()), store=store).execute(_query())
    revised = (_obs(_TUE, close="25199"), *_week()[1:])

    with pytest.raises(FuturesHistoricalMarketDataConflictError):
        _use_case(source=FakeSource(revised), store=store).execute(_query())


@pytest.mark.parametrize("returned", [0, 3, 5])
def test_an_unexpected_store_count_is_detected(returned: int) -> None:
    store = FakeStore()
    store.return_override = returned

    with pytest.raises(FuturesHistoricalDataContractViolationError, match="expected 4"):
        _use_case(source=FakeSource(_week()), store=store).execute(_query())


@pytest.mark.parametrize("returned", [4.0, "4", True])
def test_a_non_integer_store_count_is_detected(returned: object) -> None:
    store = FakeStore()
    store.return_override = returned

    with pytest.raises(FuturesHistoricalDataContractViolationError, match="integer count"):
        _use_case(source=FakeSource(_week()), store=store).execute(_query())


# ---------------------------------------------------------------------------
# Construction, finality and purity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position", [0, 1, 2])
def test_every_dependency_is_type_checked(position: int) -> None:
    dependencies: list[object] = [TableSessionResolver(), FakeSource(), FakeStore()]
    dependencies[position] = "not a dependency"

    with pytest.raises(TypeError):
        AcquireFuturesNativeDailyHistoryUseCase(*dependencies)  # type: ignore[arg-type]


def test_a_minute_source_is_not_accepted() -> None:
    class MinuteSource(FuturesHistoricalMarketDataSource):
        def fetch_session_bars(self, contract, session):  # pragma: no cover
            return ()

    with pytest.raises(TypeError, match="FuturesNativeDailyMarketDataSource"):
        AcquireFuturesNativeDailyHistoryUseCase(
            TableSessionResolver(),
            MinuteSource(),  # type: ignore[arg-type]
            FakeStore(),
        )


def test_a_foreign_query_is_rejected() -> None:
    with pytest.raises(TypeError, match="FuturesDailyHistoricalAcquisitionQuery"):
        _use_case().execute("2026-09-29")  # type: ignore[arg-type]


def test_a_session_after_today_is_acquired_without_finality_gating() -> None:
    """The caller's range is authoritative; no wall-clock cutoff is applied."""
    future = date(2099, 1, 5)
    windows = {future: _window(future)}
    store = FakeStore()

    result = _use_case(TableSessionResolver(windows), FakeSource((_obs(future),)), store).execute(
        _query(start=future, end=future)
    )

    assert result.daily_bar_count == 1
    assert _stored(store)[0].point_in_time == PointInTime("2099-01-05T10:00:00Z")


def test_the_use_case_reads_no_clock_and_touches_no_provider() -> None:
    import northstar_application.application_services.acquire_futures_native_daily_history as m

    text = Path(m.__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
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
        "upstox",
        "databento",
        "exchange_calendars",
        "sqlite3",
        "requests",
        "urllib",
        "random",
        "time",
        "os",
        "zoneinfo",
    ):
        assert not any(mod.split(".")[0] == forbidden for mod in modules)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"now", "utcnow", "today", "monotonic", "time"}

    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "datetime" not in imported
    assert "NSE" not in text
