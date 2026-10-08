"""Tests for acquiring and persisting native daily option bars."""

from __future__ import annotations

import ast
from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest
from northstar_core.derivatives import ExpirationDate
from northstar_core.foundation.value_objects import (
    ExchangeCode,
    PointInTime,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.options import (
    OptionContract,
    OptionOHLCVBar,
    OptionPremium,
    OptionProductReference,
    OptionRight,
    OptionStrike,
)

import northstar_application.application_services as services
import northstar_application.application_services.acquire_option_native_daily_history as module
from northstar_application.application_services import (
    AcquireOptionNativeDailyHistoryUseCase,
    OptionDailyAcquisitionResult,
    OptionDailySessionCoverageError,
    OptionHistoricalDataContractViolationError,
)
from northstar_application.ports import (
    OptionDailyAcquisitionQuery,
    OptionHistoricalMarketDataConflictError,
    OptionHistoricalMarketDataStore,
    OptionNativeDailyMarketDataSource,
    OptionNativeDailyObservation,
    OptionTradingSession,
    OptionTradingSessionResolutionError,
    OptionTradingSessionResolver,
)

_NIFTY = OptionProductReference(Symbol("NIFTY"), ExchangeCode("NSE"))


def _contract(strike: str = "25000", right: OptionRight = OptionRight.CALL) -> OptionContract:
    return OptionContract(
        _NIFTY, ExpirationDate("2026-10-27"), OptionStrike(Decimal(strike)), right
    )


_CONTRACT = _contract()
_DAYS = (date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7))


def _session(day: date) -> OptionTradingSession:
    return OptionTradingSession(
        day,
        PointInTime(f"{day.isoformat()}T09:15:00+05:30"),
        PointInTime(f"{day.isoformat()}T15:40:00+05:30"),
    )


def _observation(day: date, contract: OptionContract = _CONTRACT, close: str = "186.1"):
    return OptionNativeDailyObservation(
        contract=contract,
        trading_date=day,
        open=OptionPremium(Decimal("182.35")),
        high=OptionPremium(Decimal("190")),
        low=OptionPremium(Decimal("175.5")),
        close=OptionPremium(Decimal(close)),
        volume=Quantity(Decimal("1200")),
    )


class Resolver(OptionTradingSessionResolver):
    def __init__(self, sessions=None, error: Exception | None = None) -> None:
        self.sessions = tuple(_session(day) for day in _DAYS) if sessions is None else sessions
        self.error = error
        self.calls: list[tuple] = []

    def sessions_in_range(self, product, start_date, end_date):
        self.calls.append((product, start_date, end_date))
        if self.error is not None:
            raise self.error
        return self.sessions


class Source(OptionNativeDailyMarketDataSource):
    def __init__(self, observations=None) -> None:
        self.observations = (
            tuple(_observation(day) for day in _DAYS) if observations is None else observations
        )
        self.calls: list[tuple] = []

    def fetch_daily_observations(self, contract, start_trading_date, end_trading_date):
        self.calls.append((contract, start_trading_date, end_trading_date))
        return self.observations


class Store(OptionHistoricalMarketDataStore):
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result, self.error = result, error
        self.batches: list[tuple[OptionOHLCVBar, ...]] = []

    def store(self, bars):
        self.batches.append(bars)
        if self.error is not None:
            raise self.error
        return len(bars) if self.result is None else self.result


def _query(start: date = _DAYS[0], end: date = _DAYS[-1]) -> OptionDailyAcquisitionQuery:
    return OptionDailyAcquisitionQuery(_CONTRACT, start, end)


def _run(resolver=None, source=None, store=None, query=None):
    resolver, source, store = resolver or Resolver(), source or Source(), store or Store()
    result = AcquireOptionNativeDailyHistoryUseCase(resolver, source, store).execute(
        query or _query()
    )
    return result, resolver, source, store


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_one_observation_per_session_is_stored_as_one_batch() -> None:
    result, resolver, source, store = _run()

    assert result == OptionDailyAcquisitionResult(_query(), session_count=3, daily_bar_count=3)
    assert resolver.calls == [(_NIFTY, _DAYS[0], _DAYS[-1])]
    assert source.calls == [(_CONTRACT, _DAYS[0], _DAYS[-1])]
    assert len(store.batches) == 1
    assert len(store.batches[0]) == 3


def test_each_bar_is_stamped_at_its_resolved_session_close() -> None:
    _, _, _, store = _run()

    bars = store.batches[0]
    assert [bar.point_in_time for bar in bars] == [_session(day).closes_at for day in _DAYS]
    assert bars[0].point_in_time == PointInTime("2026-10-05T10:10:00Z")
    assert {bar.timeframe for bar in bars} == {Timeframe("1d")}
    assert {bar.contract for bar in bars} == {_CONTRACT}
    assert bars[0].close == OptionPremium(Decimal("186.1"))
    assert bars[0].volume == Quantity(Decimal("1200"))


def test_bars_follow_calendar_order_whatever_the_source_order() -> None:
    source = Source(tuple(_observation(day) for day in reversed(_DAYS)))

    _, _, _, store = _run(source=source)

    assert [bar.point_in_time for bar in store.batches[0]] == [
        _session(day).closes_at for day in _DAYS
    ]


def test_a_range_without_sessions_stores_nothing() -> None:
    result, _, source, store = _run(resolver=Resolver(()), source=Source(()))

    assert (result.session_count, result.daily_bar_count) == (0, 0)
    assert store.batches == []
    assert len(source.calls) == 1


# ---------------------------------------------------------------------------
# Strict coverage
# ---------------------------------------------------------------------------


def test_a_session_without_an_observation_fails_closed_and_stores_nothing() -> None:
    store = Store()

    with pytest.raises(OptionDailySessionCoverageError) as raised:
        _run(source=Source((_observation(_DAYS[0]), _observation(_DAYS[2]))), store=store)

    assert raised.value.missing_trading_dates == (_DAYS[1],)
    assert raised.value.unexpected_trading_dates == ()
    assert raised.value.contract == _CONTRACT
    assert "2026-10-06" in str(raised.value)
    assert store.batches == []


def test_an_observation_outside_the_resolved_sessions_fails_closed() -> None:
    sessions = (_session(_DAYS[0]), _session(_DAYS[2]))
    store = Store()

    with pytest.raises(OptionDailySessionCoverageError) as raised:
        _run(resolver=Resolver(sessions), store=store)

    assert raised.value.unexpected_trading_dates == (_DAYS[1],)
    assert store.batches == []


def test_an_observation_outside_the_requested_range_is_a_contract_violation() -> None:
    source = Source((*(_observation(day) for day in _DAYS), _observation(date(2026, 10, 8))))

    with pytest.raises(OptionHistoricalDataContractViolationError, match="outside the requested"):
        _run(source=source)


def test_a_duplicate_observation_is_a_contract_violation() -> None:
    source = Source((*(_observation(day) for day in _DAYS), _observation(_DAYS[0])))

    with pytest.raises(OptionHistoricalDataContractViolationError, match="more than one candle"):
        _run(source=source)


@pytest.mark.parametrize(
    "other", [_contract(strike="25050"), _contract(right=OptionRight.PUT)], ids=["strike", "right"]
)
def test_an_observation_for_another_contract_is_a_contract_violation(
    other: OptionContract,
) -> None:
    source = Source((_observation(_DAYS[0]), _observation(_DAYS[1], other),
                     _observation(_DAYS[2])))  # fmt: skip

    with pytest.raises(OptionHistoricalDataContractViolationError, match="not the requested"):
        _run(source=source)


@pytest.mark.parametrize("observations", [[], (object(),), None])
def test_a_malformed_source_answer_is_a_contract_violation(observations) -> None:
    source = Source(())
    source.observations = observations

    with pytest.raises(OptionHistoricalDataContractViolationError):
        _run(source=source)


@pytest.mark.parametrize(
    "sessions",
    [
        [_session(_DAYS[0])],
        (_session(_DAYS[1]), _session(_DAYS[0])),
        (_session(_DAYS[0]), _session(_DAYS[0])),
        (_session(date(2026, 10, 12)),),
    ],
    ids=["list", "descending", "duplicate", "outside-range"],
)
def test_a_malformed_resolver_answer_is_a_contract_violation(sessions) -> None:
    with pytest.raises(OptionHistoricalDataContractViolationError):
        _run(resolver=Resolver(sessions))


def test_a_resolver_failure_propagates_before_any_fetch() -> None:
    source = Source()

    with pytest.raises(OptionTradingSessionResolutionError, match="not loaded"):
        _run(resolver=Resolver(error=OptionTradingSessionResolutionError("2027 not loaded")),
             source=source)  # fmt: skip
    assert source.calls == []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_a_store_conflict_propagates_unwrapped() -> None:
    conflict = OptionHistoricalMarketDataConflictError("already stored differently")

    with pytest.raises(OptionHistoricalMarketDataConflictError) as raised:
        _run(store=Store(error=conflict))

    assert raised.value is conflict


@pytest.mark.parametrize("result", [2, True, "3", None.__class__])
def test_a_store_misreporting_its_count_is_a_contract_violation(result) -> None:
    store = Store(result=result)

    with pytest.raises(OptionHistoricalDataContractViolationError, match="store"):
        _run(store=store)


# ---------------------------------------------------------------------------
# Construction, result and boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position", [0, 1, 2])
def test_the_collaborators_must_be_the_option_ports(position: int) -> None:
    arguments = [Resolver(), Source(), Store()]
    arguments[position] = object()

    with pytest.raises(TypeError, match="must be an Option"):
        AcquireOptionNativeDailyHistoryUseCase(*arguments)


def test_the_query_must_be_an_option_daily_acquisition_query() -> None:
    use_case = AcquireOptionNativeDailyHistoryUseCase(Resolver(), Source(), Store())

    with pytest.raises(TypeError, match="OptionDailyAcquisitionQuery"):
        use_case.execute((_CONTRACT, _DAYS[0], _DAYS[-1]))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("sessions", "bars"), [(2, 3), (-1, 0), (True, 0)], ids=["more-bars", "negative", "bool"]
)
def test_an_inconsistent_result_is_rejected(sessions, bars) -> None:
    with pytest.raises((TypeError, ValueError)):
        OptionDailyAcquisitionResult(_query(), sessions, bars)


def test_the_use_case_and_errors_are_exported() -> None:
    for name in (
        "AcquireOptionNativeDailyHistoryUseCase",
        "OptionDailyAcquisitionResult",
        "OptionDailySessionCoverageError",
        "OptionHistoricalDataContractViolationError",
    ):
        assert name in services.__all__


def test_the_use_case_depends_on_no_futures_provider_or_clock() -> None:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}

    for name in imported:
        assert not name.startswith(("northstar_core.futures", "northstar_infrastructure"))
        assert name.split(".")[0] not in {"time", "datetime_now", "sqlite3", "socket", "requests"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"now", "utcnow", "today", "time", "monotonic"}
