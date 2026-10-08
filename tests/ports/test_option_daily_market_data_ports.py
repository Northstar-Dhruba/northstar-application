"""Contract tests for the option session, native-daily and historical market-data ports."""

from __future__ import annotations

import ast
import importlib
import inspect
from dataclasses import FrozenInstanceError, fields
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import get_type_hints

import pytest
from northstar_core.derivatives import ExpirationDate, QuoteValue
from northstar_core.foundation.value_objects import (
    ExchangeCode,
    PointInTime,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.futures import FuturesContract, FuturesProductReference
from northstar_core.options import (
    OptionContract,
    OptionOHLCVBar,
    OptionPremium,
    OptionProductReference,
    OptionRight,
    OptionStrike,
)

import northstar_application.ports as ports
from northstar_application.ports import (
    FuturesTradingSession,
    InvalidOptionDailyAcquisitionQueryError,
    InvalidOptionHistoricalMarketDataQueryError,
    InvalidOptionNativeDailyObservationError,
    InvalidOptionTradingSessionError,
    OptionDailyAcquisitionQuery,
    OptionHistoricalMarketDataConflictError,
    OptionHistoricalMarketDataQuery,
    OptionHistoricalMarketDataRepository,
    OptionHistoricalMarketDataStore,
    OptionNativeDailyMarketDataSource,
    OptionNativeDailyObservation,
    OptionTradingSession,
    OptionTradingSessionResolutionError,
    OptionTradingSessionResolver,
)

_NIFTY = OptionProductReference(Symbol("NIFTY"), ExchangeCode("NSE"))
_CONTRACT = OptionContract(
    _NIFTY, ExpirationDate("2026-10-27"), OptionStrike(Decimal("25000")), OptionRight.CALL
)
_OPEN = PointInTime("2026-10-08T03:45:00Z")
_CLOSE = PointInTime("2026-10-08T10:10:00Z")


def _p(value: str) -> OptionPremium:
    return OptionPremium(Decimal(value))


# ---------------------------------------------------------------------------
# OptionTradingSession
# ---------------------------------------------------------------------------


def test_a_session_preserves_its_members() -> None:
    session = OptionTradingSession(date(2026, 10, 8), _OPEN, _CLOSE)

    assert session.trading_date == date(2026, 10, 8)
    assert session.opens_at == _OPEN
    assert session.closes_at == _CLOSE
    assert [field.name for field in fields(OptionTradingSession)] == [
        "trading_date",
        "opens_at",
        "closes_at",
    ]


@pytest.mark.parametrize(
    ("trading_date", "opens_at", "closes_at", "message"),
    [
        (None, _OPEN, _CLOSE, "trading date cannot be None"),
        (datetime(2026, 10, 8), _OPEN, _CLOSE, "plain date"),
        ("2026-10-08", _OPEN, _CLOSE, "plain date"),
        (date(2026, 10, 8), None, _CLOSE, "opens_at cannot be None"),
        (date(2026, 10, 8), "2026-10-08T03:45:00Z", _CLOSE, "opens_at must be a PointInTime"),
        (date(2026, 10, 8), _OPEN, None, "closes_at cannot be None"),
        (date(2026, 10, 8), _CLOSE, _OPEN, "opens_at must be before closes_at"),
        (date(2026, 10, 8), _CLOSE, _CLOSE, "opens_at must be before closes_at"),
    ],
)
def test_invalid_sessions_are_rejected(trading_date, opens_at, closes_at, message: str) -> None:
    with pytest.raises(InvalidOptionTradingSessionError, match=message):
        OptionTradingSession(trading_date, opens_at, closes_at)


def test_a_session_is_immutable_and_hashable() -> None:
    session = OptionTradingSession(date(2026, 10, 8), _OPEN, _CLOSE)

    with pytest.raises(FrozenInstanceError):
        session.closes_at = _OPEN  # type: ignore[misc]
    assert hash(session) == hash(OptionTradingSession(date(2026, 10, 8), _OPEN, _CLOSE))
    assert str(session) == "2026-10-08 [2026-10-08T03:45:00Z .. 2026-10-08T10:10:00Z]"


def test_an_option_session_is_not_a_futures_session() -> None:
    session = OptionTradingSession(date(2026, 10, 8), _OPEN, _CLOSE)

    assert not isinstance(session, FuturesTradingSession)
    assert session != FuturesTradingSession(date(2026, 10, 8), _OPEN, _CLOSE)


def test_the_resolver_port_has_exactly_sessions_in_range() -> None:
    assert inspect.isabstract(OptionTradingSessionResolver)
    assert OptionTradingSessionResolver.__abstractmethods__ == frozenset({"sessions_in_range"})
    hints = get_type_hints(OptionTradingSessionResolver.sessions_in_range)
    assert hints == {
        "product": OptionProductReference,
        "start_date": date,
        "end_date": date,
        "return": tuple[OptionTradingSession, ...],
    }


def test_the_resolution_error_is_distinct() -> None:
    assert issubclass(OptionTradingSessionResolutionError, RuntimeError)
    assert not issubclass(OptionTradingSessionResolutionError, ports.FuturesSessionResolutionError)


# ---------------------------------------------------------------------------
# OptionNativeDailyObservation
# ---------------------------------------------------------------------------


def _observation(**override) -> OptionNativeDailyObservation:
    values = {
        "contract": _CONTRACT,
        "trading_date": date(2026, 10, 8),
        "open": _p("182.35"),
        "high": _p("190"),
        "low": _p("175.5"),
        "close": _p("186.1"),
        "volume": Quantity(Decimal("1200")),
    }
    values.update(override)
    return OptionNativeDailyObservation(**values)


def test_the_observation_field_shape_is_exact() -> None:
    assert [field.name for field in fields(OptionNativeDailyObservation)] == [
        "contract",
        "trading_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"contract": "NIFTY@NSE 2026-10-27 25000 CALL"}, "must be an OptionContract"),
        (
            {"contract": FuturesContract(FuturesProductReference(Symbol("NIFTY"),
                                         ExchangeCode("NSE")), ExpirationDate("2026-10-27"))},
            "must be an OptionContract",
        ),
        ({"trading_date": datetime(2026, 10, 8)}, "plain date"),
        ({"trading_date": None}, "cannot be None"),
        ({"open": QuoteValue(Decimal("182.35"))}, "open must be an OptionPremium"),
        ({"close": Decimal("186.1")}, "close must be an OptionPremium"),
        ({"volume": Decimal("1200")}, "volume must be a Quantity"),
        ({"volume": Quantity(Decimal("12.5"))}, "whole number of option contracts"),
        ({"high": _p("180")}, "high must be greater than or equal"),
        ({"low": _p("183")}, "low must be less than or equal"),
    ],
)  # fmt: skip
def test_invalid_observations_are_rejected(override: dict, message: str) -> None:
    with pytest.raises(InvalidOptionNativeDailyObservationError, match=message):
        _observation(**override)


def test_the_observation_carries_no_instant_provider_or_open_interest() -> None:
    observation = _observation()

    for absent in ("point_in_time", "instant", "open_interest", "provider", "instrument_key"):
        assert not hasattr(observation, absent)


# ---------------------------------------------------------------------------
# OptionDailyAcquisitionQuery
# ---------------------------------------------------------------------------


def test_a_query_preserves_an_explicit_inclusive_range() -> None:
    query = OptionDailyAcquisitionQuery(_CONTRACT, date(2026, 10, 5), date(2026, 10, 9))

    assert (query.start_trading_date, query.end_trading_date) == (date(2026, 10, 5),
                                                                 date(2026, 10, 9))  # fmt: skip
    assert str(query) == "NIFTY@NSE 2026-10-27 25000 CALL [2026-10-05 .. 2026-10-09]"


@pytest.mark.parametrize(
    ("contract", "start", "end", "message"),
    [
        ("NIFTY", date(2026, 10, 5), date(2026, 10, 9), "must be an OptionContract"),
        (_CONTRACT, None, date(2026, 10, 9), "cannot be None"),
        (_CONTRACT, date(2026, 10, 5), datetime(2026, 10, 9), "plain date"),
        (_CONTRACT, date(2026, 10, 9), date(2026, 10, 5), "must not be after"),
    ],
)
def test_invalid_queries_are_rejected(contract, start, end, message: str) -> None:
    with pytest.raises(InvalidOptionDailyAcquisitionQueryError, match=message):
        OptionDailyAcquisitionQuery(contract, start, end)


def test_a_query_has_no_default_range() -> None:
    signature = inspect.signature(OptionDailyAcquisitionQuery)
    assert all(p.default is inspect.Parameter.empty for p in signature.parameters.values())


# ---------------------------------------------------------------------------
# Source, store and repository
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("port", "methods"),
    [
        (OptionNativeDailyMarketDataSource, {"fetch_daily_observations"}),
        (OptionHistoricalMarketDataStore, {"store"}),
        (OptionHistoricalMarketDataRepository, {"get_bars"}),
    ],
)
def test_the_ports_are_abstract(port: type, methods: set[str]) -> None:
    assert inspect.isabstract(port)
    assert port.__abstractmethods__ == frozenset(methods)


def test_the_port_signatures_are_option_typed() -> None:
    source = get_type_hints(OptionNativeDailyMarketDataSource.fetch_daily_observations)
    store = get_type_hints(OptionHistoricalMarketDataStore.store)
    repository = get_type_hints(OptionHistoricalMarketDataRepository.get_bars)

    assert source == {
        "contract": OptionContract,
        "start_trading_date": date,
        "end_trading_date": date,
        "return": tuple[OptionNativeDailyObservation, ...],
    }
    assert store == {"bars": tuple[OptionOHLCVBar, ...], "return": int}
    assert repository == {
        "query": OptionHistoricalMarketDataQuery,
        "return": tuple[OptionOHLCVBar, ...],
    }


def test_the_conflict_error_is_a_value_error_of_its_own() -> None:
    assert issubclass(OptionHistoricalMarketDataConflictError, ValueError)
    assert not issubclass(
        OptionHistoricalMarketDataConflictError, ports.FuturesHistoricalMarketDataConflictError
    )


def test_a_repository_query_covers_its_inclusive_window() -> None:
    query = OptionHistoricalMarketDataQuery(
        _CONTRACT, Timeframe("1d"), PointInTime("2026-10-08T10:10:00Z"),
        PointInTime("2026-10-09T10:10:00Z"),
    )  # fmt: skip

    assert query.covers(PointInTime("2026-10-08T15:40:00+05:30"))
    assert query.covers(PointInTime("2026-10-09T10:10:00Z"))
    assert not query.covers(PointInTime("2026-10-07T10:10:00Z"))
    assert OptionHistoricalMarketDataQuery(_CONTRACT, Timeframe("1d")).covers(_CLOSE)


@pytest.mark.parametrize(
    ("contract", "timeframe", "start", "end", "message"),
    [
        ("NIFTY", Timeframe("1d"), None, None, "must be an OptionContract"),
        (_CONTRACT, "1d", None, None, "must be a Timeframe"),
        (_CONTRACT, Timeframe("1d"), "2026-10-08", None, "start must be a PointInTime"),
        (_CONTRACT, Timeframe("1d"), _CLOSE, _OPEN, "before or equal to end"),
    ],
)
def test_invalid_repository_queries_are_rejected(contract, timeframe, start, end, message) -> None:
    with pytest.raises(InvalidOptionHistoricalMarketDataQueryError, match=message):
        OptionHistoricalMarketDataQuery(contract, timeframe, start, end)


def test_every_new_name_is_exported() -> None:
    for name in (
        "InvalidOptionDailyAcquisitionQueryError",
        "InvalidOptionHistoricalMarketDataQueryError",
        "InvalidOptionNativeDailyObservationError",
        "InvalidOptionTradingSessionError",
        "OptionDailyAcquisitionQuery",
        "OptionHistoricalMarketDataConflictError",
        "OptionHistoricalMarketDataQuery",
        "OptionHistoricalMarketDataRepository",
        "OptionHistoricalMarketDataStore",
        "OptionNativeDailyMarketDataSource",
        "OptionNativeDailyObservation",
        "OptionTradingSession",
        "OptionTradingSessionResolutionError",
        "OptionTradingSessionResolver",
    ):
        assert name in ports.__all__
        assert getattr(ports, name) is not None


@pytest.mark.parametrize(
    "module",
    [
        "northstar_application.ports.option_trading_session_resolver",
        "northstar_application.ports.option_native_daily_market_data_source",
        "northstar_application.ports.option_historical_market_data_store",
        "northstar_application.ports.option_historical_market_data_repository",
    ],
)
def test_the_option_ports_depend_on_no_futures_provider_or_clock(module: str) -> None:
    tree = ast.parse(Path(importlib.import_module(module).__file__).read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}

    for name in imported:
        assert not name.startswith(("northstar_core.futures", "northstar_infrastructure"))
        assert name.split(".")[0] not in {"time", "zoneinfo", "sqlite3", "socket", "requests"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"now", "utcnow", "today", "time", "monotonic"}
