"""Tests for building a point-in-time option chain snapshot."""

from __future__ import annotations

import ast
from datetime import date, datetime
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
    OptionChainEntry,
    OptionChainSnapshot,
    OptionContract,
    OptionOHLCVBar,
    OptionPremium,
    OptionProductReference,
    OptionRight,
    OptionStrike,
)

import northstar_application.application_services as services
import northstar_application.application_services.build_option_chain_snapshot as module
from northstar_application.application_services import (
    BuildOptionChainSnapshotUseCase,
    InvalidOptionChainSnapshotQueryError,
    OptionChainContractViolationError,
    OptionChainListingNotKnownError,
    OptionChainSessionNotFoundError,
    OptionChainSnapshotQuery,
)
from northstar_application.ports import (
    OptionChainDailyBarQuery,
    OptionChainDailyBarRepository,
    OptionListedContractQuery,
    OptionListedContractRepository,
    OptionTradingSession,
    OptionTradingSessionResolutionError,
    OptionTradingSessionResolver,
)

_NIFTY = OptionProductReference(Symbol("NIFTY"), ExchangeCode("NSE"))
_BANKNIFTY = OptionProductReference(Symbol("BANKNIFTY"), ExchangeCode("NSE"))
_EXPIRY = ExpirationDate("2026-10-27")
_DAY = date(2026, 10, 8)
_CLOSE = PointInTime("2026-10-08T10:10:00Z")
_CALL, _PUT = OptionRight.CALL, OptionRight.PUT


def _contract(
    strike: str = "22600",
    right: OptionRight = _CALL,
    expiration: str = "2026-10-27",
    product: OptionProductReference = _NIFTY,
) -> OptionContract:
    return OptionContract(product, ExpirationDate(expiration), OptionStrike(Decimal(strike)), right)


def _bar(contract: OptionContract, instant: str = "2026-10-08T10:10:00Z", timeframe="1d"):
    return OptionOHLCVBar(
        contract=contract,
        point_in_time=PointInTime(instant),
        timeframe=Timeframe(timeframe),
        open=OptionPremium(Decimal("191.8")),
        high=OptionPremium(Decimal("226.05")),
        low=OptionPremium(Decimal("122.45")),
        close=OptionPremium(Decimal("132.6")),
        volume=Quantity(Decimal("40")),
    )


def _session(day: date = _DAY) -> OptionTradingSession:
    return OptionTradingSession(
        day,
        PointInTime(f"{day.isoformat()}T03:45:00Z"),
        PointInTime(f"{day.isoformat()}T10:10:00Z"),
    )


_C22550, _P22550 = _contract("22550"), _contract("22550", _PUT)
_C22600, _P22600 = _contract("22600"), _contract("22600", _PUT)
_C22650 = _contract("22650")


class Resolver(OptionTradingSessionResolver):
    def __init__(self, sessions=None, error: Exception | None = None) -> None:
        self.sessions = (_session(),) if sessions is None else sessions
        self.error = error
        self.calls: list[tuple] = []

    def sessions_in_range(self, product, start_date, end_date):
        self.calls.append((product, start_date, end_date))
        if self.error is not None:
            raise self.error
        return self.sessions


class Listings(OptionListedContractRepository):
    """Applies the port contract to (contract, first observed at) pairs."""

    def __init__(self, known=None, answer=None) -> None:
        stamp = PointInTime("2026-10-08T06:59:12.429903Z")
        self.known = (
            [(c, stamp) for c in (_C22550, _P22550, _C22600, _P22600)] if known is None else known
        )
        self.answer = answer
        self.queries: list[OptionListedContractQuery] = []

    def listed_contracts(self, query):
        self.queries.append(query)
        if self.answer is not None:
            return self.answer
        return tuple(
            contract
            for contract, established in self.known
            if contract.product == query.product
            and contract.expiration_date == query.expiration_date
            and established.compare(query.known_by) <= 0
        )


class Bars(OptionChainDailyBarRepository):
    def __init__(self, bars=None) -> None:
        self.bars = (_bar(_C22550), _bar(_P22600)) if bars is None else bars
        self.queries: list[OptionChainDailyBarQuery] = []

    def daily_bars_at(self, query):
        self.queries.append(query)
        return self.bars


def _query(expiration: str = "2026-10-27", day: date = _DAY) -> OptionChainSnapshotQuery:
    return OptionChainSnapshotQuery(_NIFTY, ExpirationDate(expiration), day)


def _build(resolver=None, listings=None, bars=None, query=None):
    resolver, listings, bars = resolver or Resolver(), listings or Listings(), bars or Bars()
    snapshot = BuildOptionChainSnapshotUseCase(resolver, listings, bars).execute(query or _query())
    return snapshot, resolver, listings, bars


# ---------------------------------------------------------------------------
# The chain
# ---------------------------------------------------------------------------


def test_every_known_listing_becomes_an_entry_with_its_bar_or_none() -> None:
    snapshot, _, _, _ = _build()

    assert snapshot == OptionChainSnapshot(
        _NIFTY,
        _EXPIRY,
        _CLOSE,
        (
            OptionChainEntry(_C22550, _bar(_C22550)),
            OptionChainEntry(_P22550, None),
            OptionChainEntry(_C22600, None),
            OptionChainEntry(_P22600, _bar(_P22600)),
        ),
    )


def test_one_session_is_resolved_and_as_of_is_its_exact_close() -> None:
    snapshot, resolver, listings, bars = _build()

    assert resolver.calls == [(_NIFTY, _DAY, _DAY)]
    assert snapshot.as_of == _CLOSE
    assert listings.queries == [OptionListedContractQuery(_NIFTY, _EXPIRY, _CLOSE)]
    assert bars.queries == [OptionChainDailyBarQuery(_NIFTY, _EXPIRY, _CLOSE)]


def test_listings_first_observed_at_or_before_as_of_are_included_later_ones_excluded() -> None:
    known = [
        (_C22550, PointInTime("2026-10-07T06:59:12.429903Z")),
        (_P22550, PointInTime("2026-10-08T10:10:00Z")),
        (_C22600, PointInTime("2026-10-08T10:10:00.000001Z")),
        (_P22600, PointInTime("2026-10-08T14:26:54Z")),
    ]

    snapshot, _, _, _ = _build(listings=Listings(known), bars=Bars(()))

    assert [entry.contract for entry in snapshot.entries] == [_C22550, _P22550]


def test_a_hindsight_bar_never_establishes_listing_membership() -> None:
    # 22650 CALL has an exactly stamped bar, but its listing was first observed after as_of.
    known = [
        (_C22600, PointInTime("2026-10-08T06:59:12Z")),
        (_C22650, PointInTime("2026-10-08T14:26:54Z")),
    ]

    snapshot, _, _, _ = _build(listings=Listings(known), bars=Bars((_bar(_C22600), _bar(_C22650))))

    assert snapshot.entries == (OptionChainEntry(_C22600, _bar(_C22600)),)


def test_a_date_before_any_known_listing_has_no_chain_even_with_bars() -> None:
    known = [(_C22600, PointInTime("2026-10-08T06:59:12Z"))]
    resolver = Resolver((_session(date(2026, 10, 7)),))

    with pytest.raises(OptionChainListingNotKnownError) as raised:
        _build(
            resolver=resolver,
            listings=Listings(known),
            bars=Bars((_bar(_C22600, "2026-10-07T10:10:00Z"),)),
            query=_query(day=date(2026, 10, 7)),
        )

    assert raised.value.as_of == PointInTime("2026-10-07T10:10:00Z")
    assert raised.value.expiration_date == _EXPIRY
    assert "known by 2026-10-07T10:10:00Z" in str(raised.value)


def test_no_known_listing_at_all_fails() -> None:
    bars = Bars()

    with pytest.raises(OptionChainListingNotKnownError):
        _build(listings=Listings([]), bars=bars)
    assert bars.queries == []


def test_an_entry_without_a_bar_is_none_never_a_substituted_bar() -> None:
    snapshot, _, _, _ = _build(bars=Bars(()))

    assert [entry.daily_bar for entry in snapshot.entries] == [None, None, None, None]


@pytest.mark.parametrize("instant", ["2026-10-07T10:10:00Z", "2026-10-09T10:10:00Z"])
def test_a_bar_from_another_session_is_a_contract_violation(instant: str) -> None:
    with pytest.raises(OptionChainContractViolationError, match="stamped"):
        _build(bars=Bars((_bar(_C22550, instant),)))


def test_the_expiration_day_itself_has_a_chain() -> None:
    known = [(_contract("22600", expiration="2026-10-08"), PointInTime("2026-10-01T06:59:12Z"))]

    snapshot, _, _, _ = _build(
        listings=Listings(known), bars=Bars(()), query=_query(expiration="2026-10-08")
    )

    assert snapshot.expiration_date == ExpirationDate("2026-10-08")


def test_entries_follow_canonical_order_whatever_the_bar_order() -> None:
    snapshot, _, _, _ = _build()

    assert [(e.contract.strike.value, e.contract.right) for e in snapshot.entries] == [
        (Decimal("22550"), _CALL),
        (Decimal("22550"), _PUT),
        (Decimal("22600"), _CALL),
        (Decimal("22600"), _PUT),
    ]


def test_a_rebuild_is_identical_and_a_later_bar_only_fills_an_entry() -> None:
    first, _, _, _ = _build()
    again, _, _, _ = _build()
    enriched, _, _, _ = _build(bars=Bars((_bar(_C22550), _bar(_C22600), _bar(_P22600))))

    assert again == first
    assert [e.contract for e in enriched.entries] == [e.contract for e in first.entries]
    assert enriched.entries[2].daily_bar == _bar(_C22600)
    assert first.entries[2].daily_bar is None


# ---------------------------------------------------------------------------
# Sessions and the query
# ---------------------------------------------------------------------------


def test_a_weekend_or_holiday_has_no_session() -> None:
    listings = Listings()

    with pytest.raises(OptionChainSessionNotFoundError) as raised:
        _build(resolver=Resolver(()), listings=listings, query=_query(day=date(2026, 10, 10)))

    assert raised.value.trading_date == date(2026, 10, 10)
    assert "not a NIFTY@NSE option trading session" in str(raised.value)
    assert listings.queries == []


def test_a_resolution_failure_propagates_unwrapped() -> None:
    error = OptionTradingSessionResolutionError("2027 not loaded")
    listings = Listings()

    with pytest.raises(OptionTradingSessionResolutionError) as raised:
        _build(resolver=Resolver(error=error), listings=listings)

    assert raised.value is error
    assert listings.queries == []


@pytest.mark.parametrize(
    "sessions",
    [
        [_session()],
        (_session(), _session(date(2026, 10, 9))),
        (_session(date(2026, 10, 9)),),
        ("session",),
    ],
    ids=["list", "two", "other-date", "foreign"],
)
def test_a_malformed_resolver_answer_is_a_contract_violation(sessions) -> None:
    with pytest.raises(OptionChainContractViolationError, match="OptionTradingSessionResolver"):
        _build(resolver=Resolver(sessions))


def test_an_expiration_before_the_trading_date_is_an_invalid_query() -> None:
    with pytest.raises(InvalidOptionChainSnapshotQueryError, match="have expired"):
        _query(expiration="2026-10-07")


@pytest.mark.parametrize(
    ("product", "expiration", "day"),
    [
        ("NIFTY@NSE", _EXPIRY, _DAY),
        (_NIFTY, "2026-10-27", _DAY),
        (_NIFTY, _EXPIRY, "2026-10-08"),
        (_NIFTY, _EXPIRY, datetime(2026, 10, 8)),
    ],
    ids=["product", "expiration", "text-date", "datetime"],
)
def test_the_query_needs_core_values_and_a_plain_date(product, expiration, day) -> None:
    with pytest.raises(InvalidOptionChainSnapshotQueryError):
        OptionChainSnapshotQuery(product, expiration, day)


def test_the_query_must_be_a_chain_snapshot_query() -> None:
    use_case = BuildOptionChainSnapshotUseCase(Resolver(), Listings(), Bars())

    with pytest.raises(TypeError, match="OptionChainSnapshotQuery"):
        use_case.execute((_NIFTY, _EXPIRY, _DAY))  # type: ignore[arg-type]


@pytest.mark.parametrize("position", [0, 1, 2])
def test_the_collaborators_must_be_the_ports(position: int) -> None:
    arguments: list[object] = [Resolver(), Listings(), Bars()]
    arguments[position] = object()

    with pytest.raises(TypeError, match="must be an Option"):
        BuildOptionChainSnapshotUseCase(*arguments)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Repository contracts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        ([_C22550], "tuple of OptionContract"),
        ((_C22550, "22600 CALL"), "tuple of OptionContract"),
        ((_C22550, _contract("22600", product=_BANKNIFTY)), "returned"),
        ((_C22550, _contract("22600", expiration="2026-11-24")), "returned"),
        ((_C22550, _C22550), "more than once"),
        ((_P22550, _C22550), "strike ascending"),
        ((_C22600, _C22550), "strike ascending"),
    ],
    ids=["list", "foreign", "product", "expiration", "duplicate", "right-order", "strike-order"],
)
def test_a_malformed_listing_answer_is_a_contract_violation(answer, message) -> None:
    with pytest.raises(OptionChainContractViolationError, match=message):
        _build(listings=Listings(answer=answer))


@pytest.mark.parametrize(
    ("bars", "message"),
    [
        ([_bar(_C22550)], "tuple of OptionOHLCVBar"),
        ((_bar(_C22550), "bar"), "tuple of OptionOHLCVBar"),
        ((_bar(_contract("22550", product=_BANKNIFTY)),), "returned a bar"),
        ((_bar(_contract("22550", expiration="2026-11-24")),), "returned a bar"),
        ((_bar(_C22550, timeframe="1h"),), "not 1d"),
        ((_bar(_C22550), _bar(_C22550)), "more than one bar"),
        ((_bar(_P22600), _bar(_C22550)), "strike ascending"),
    ],
    ids=["list", "foreign", "product", "expiration", "timeframe", "duplicate", "order"],
)
def test_a_malformed_bar_answer_is_a_contract_violation(bars, message) -> None:
    with pytest.raises(OptionChainContractViolationError, match=message):
        _build(bars=Bars(bars))


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------


def test_the_use_case_query_and_errors_are_exported() -> None:
    for name in (
        "BuildOptionChainSnapshotUseCase",
        "InvalidOptionChainSnapshotQueryError",
        "OptionChainContractViolationError",
        "OptionChainListingNotKnownError",
        "OptionChainSessionNotFoundError",
        "OptionChainSnapshotQuery",
    ):
        assert name in services.__all__


def test_the_use_case_reads_no_clock_network_storage_or_provider() -> None:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}

    for name in imported:
        assert not name.startswith(("northstar_core.futures", "northstar_infrastructure"))
        assert name.split(".")[0] not in {"time", "sqlite3", "socket", "urllib", "requests"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in {"now", "utcnow", "today", "time", "monotonic"}
        if isinstance(node, ast.Name):
            assert node.id not in {"OptionExpirySeries", "open_interest", "underlying_value"}
