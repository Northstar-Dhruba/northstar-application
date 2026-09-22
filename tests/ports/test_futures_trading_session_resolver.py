"""Contract tests for FuturesTradingSession and its Application resolver port.

The port cannot enforce its own semantics, so the obligations are pinned here
against a conforming in-memory reference implementation. No calendar library is
imported: this port's contract is about what a session window *means*, not how
a venue's schedule is computed.

The session data below mirrors the real CME shape, including the properties
that are easy to assume away -- sessions opening on the previous civil day, an
early close, and a holiday gap where one session's open is nowhere near the
previous session's close.
"""

from __future__ import annotations

import ast
import inspect
from datetime import date, datetime
from pathlib import Path

import pytest
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol
from northstar_core.futures import FuturesContract, FuturesProductReference

from northstar_application.ports import (
    FuturesDailyBarCompletionResolver,
    FuturesSessionResolutionError,
    FuturesTradingSession,
    FuturesTradingSessionResolver,
    InvalidFuturesTradingSessionError,
    TradingSessionResolutionError,
    TradingSessionResolver,
)

_CME = ExchangeCode("CME")
_NYMEX = ExchangeCode("NYMEX")

_ES = FuturesProductReference(Symbol("ES"), _CME)
_MES = FuturesProductReference(Symbol("MES"), _CME)
_CL = FuturesProductReference(Symbol("CL"), _NYMEX)
_UNSUPPORTED = FuturesProductReference(Symbol("ES"), ExchangeCode("NASDAQ"))

_SUPPORTED_VENUES = frozenset({"CME", "NYMEX"})

# Real CME windows. Note 2026-07-03 is an early close and 2026-07-06 opens on
# the Sunday, more than two days after Friday's close.
_WINDOWS: dict[date, tuple[str, str]] = {
    date(2026, 7, 1): ("2026-06-30T22:00:00Z", "2026-07-01T22:00:00Z"),
    date(2026, 7, 2): ("2026-07-01T22:00:00Z", "2026-07-02T22:00:00Z"),
    date(2026, 7, 3): ("2026-07-02T22:00:00Z", "2026-07-03T17:00:00Z"),
    date(2026, 7, 6): ("2026-07-05T22:00:00Z", "2026-07-06T22:00:00Z"),
    date(2026, 9, 15): ("2026-09-14T22:00:00Z", "2026-09-15T22:00:00Z"),
    date(2026, 9, 16): ("2026-09-15T22:00:00Z", "2026-09-16T22:00:00Z"),
}
_NON_SESSIONS = (date(2026, 7, 4), date(2026, 7, 5), date(2026, 12, 25))


def _session(trading_date: date) -> FuturesTradingSession:
    opens, closes = _WINDOWS[trading_date]
    return FuturesTradingSession(trading_date, PointInTime(opens), PointInTime(closes))


class ReferenceFuturesTradingSessionResolver(FuturesTradingSessionResolver):
    """A minimal resolver honouring every documented obligation."""

    def __init__(self) -> None:
        self.calls: list[tuple[FuturesProductReference, object]] = []

    def _check(self, product: FuturesProductReference, *dates: object) -> None:
        if not isinstance(product, FuturesProductReference):
            raise TypeError("product must be a FuturesProductReference.")
        for value in dates:
            if isinstance(value, datetime) or not isinstance(value, date):
                raise TypeError("trading dates must be plain datetime.date values.")
        if product.exchange_code.value not in _SUPPORTED_VENUES:
            raise FuturesSessionResolutionError(
                f"Unsupported futures venue: {product.exchange_code.value}."
            )

    def resolve(
        self, product: FuturesProductReference, trading_date: date
    ) -> FuturesTradingSession | None:
        self._check(product, trading_date)
        self.calls.append((product, trading_date))
        return _session(trading_date) if trading_date in _WINDOWS else None

    def sessions_in_range(
        self, product: FuturesProductReference, start_date: date, end_date: date
    ) -> tuple[FuturesTradingSession, ...]:
        self._check(product, start_date, end_date)
        if start_date > end_date:
            raise FuturesSessionResolutionError(
                "sessions_in_range start_date must not be after end_date."
            )
        self.calls.append((product, (start_date, end_date)))
        return tuple(
            _session(label) for label in sorted(_WINDOWS) if start_date <= label <= end_date
        )


@pytest.fixture
def resolver() -> ReferenceFuturesTradingSessionResolver:
    return ReferenceFuturesTradingSessionResolver()


# ---------------------------------------------------------------------------
# FuturesTradingSession: construction and invariants
# ---------------------------------------------------------------------------


def test_a_valid_session_preserves_its_members() -> None:
    session = _session(date(2026, 9, 15))

    assert session.trading_date == date(2026, 9, 15)
    assert session.opens_at == PointInTime("2026-09-14T22:00:00Z")
    assert session.closes_at == PointInTime("2026-09-15T22:00:00Z")


def test_a_session_may_open_on_the_previous_civil_day() -> None:
    """For CME this is every session, not an edge case."""
    session = _session(date(2026, 9, 15))

    assert session.opens_at.value.startswith("2026-09-14")
    assert session.closes_at.value.startswith("2026-09-15")


def test_a_session_may_be_shorter_than_a_full_day() -> None:
    early = _session(date(2026, 7, 3))

    assert early.opens_at == PointInTime("2026-07-02T22:00:00Z")
    assert early.closes_at == PointInTime("2026-07-03T17:00:00Z")


def test_a_session_open_need_not_equal_the_previous_session_close() -> None:
    """The holiday gap that makes a close-only chain wrong."""
    friday = _session(date(2026, 7, 3))
    monday = _session(date(2026, 7, 6))

    assert friday.closes_at == PointInTime("2026-07-03T17:00:00Z")
    assert monday.opens_at == PointInTime("2026-07-05T22:00:00Z")
    assert monday.opens_at != friday.closes_at
    assert friday.closes_at.compare(monday.opens_at) < 0


def test_consecutive_weekday_sessions_do_abut() -> None:
    """Within a week they meet, which is exactly why the gap is easy to miss."""
    assert _session(date(2026, 9, 15)).closes_at == _session(date(2026, 9, 16)).opens_at


# ---------------------------------------------------------------------------
# FuturesTradingSession: validation
# ---------------------------------------------------------------------------


def test_none_members_are_rejected() -> None:
    opens, closes = PointInTime("2026-09-14T22:00:00Z"), PointInTime("2026-09-15T22:00:00Z")

    with pytest.raises(InvalidFuturesTradingSessionError, match="trading date cannot be None"):
        FuturesTradingSession(None, opens, closes)
    with pytest.raises(InvalidFuturesTradingSessionError, match="opens_at cannot be None"):
        FuturesTradingSession(date(2026, 9, 15), None, closes)
    with pytest.raises(InvalidFuturesTradingSessionError, match="closes_at cannot be None"):
        FuturesTradingSession(date(2026, 9, 15), opens, None)


def test_a_datetime_is_rejected_as_the_trading_date() -> None:
    """A datetime carries a clock; a session label has none."""
    with pytest.raises(InvalidFuturesTradingSessionError, match="not a datetime"):
        FuturesTradingSession(
            datetime(2026, 9, 15, 22, 0),
            PointInTime("2026-09-14T22:00:00Z"),
            PointInTime("2026-09-15T22:00:00Z"),
        )


@pytest.mark.parametrize("value", ["2026-09-15", 20260915, None])
def test_a_non_date_trading_date_is_rejected(value: object) -> None:
    with pytest.raises(InvalidFuturesTradingSessionError):
        FuturesTradingSession(
            value,
            PointInTime("2026-09-14T22:00:00Z"),
            PointInTime("2026-09-15T22:00:00Z"),
        )


@pytest.mark.parametrize("field_name", ["opens_at", "closes_at"])
def test_a_non_point_in_time_boundary_is_rejected(field_name: str) -> None:
    members = {
        "trading_date": date(2026, 9, 15),
        "opens_at": PointInTime("2026-09-14T22:00:00Z"),
        "closes_at": PointInTime("2026-09-15T22:00:00Z"),
    }
    members[field_name] = "2026-09-15T22:00:00Z"

    with pytest.raises(InvalidFuturesTradingSessionError, match="must be a PointInTime value"):
        FuturesTradingSession(**members)


def test_open_before_close_is_accepted() -> None:
    assert (
        FuturesTradingSession(
            date(2026, 9, 15),
            PointInTime("2026-09-14T22:00:00Z"),
            PointInTime("2026-09-15T22:00:00Z"),
        ).closes_at
        is not None
    )


def test_equal_bounds_are_rejected() -> None:
    instant = PointInTime("2026-09-15T22:00:00Z")

    with pytest.raises(InvalidFuturesTradingSessionError, match="must be before closes_at"):
        FuturesTradingSession(date(2026, 9, 15), instant, instant)


def test_reversed_bounds_are_rejected() -> None:
    with pytest.raises(InvalidFuturesTradingSessionError, match="must be before closes_at"):
        FuturesTradingSession(
            date(2026, 9, 15),
            PointInTime("2026-09-15T22:00:00Z"),
            PointInTime("2026-09-14T22:00:00Z"),
        )


def test_offset_equivalent_bounds_are_equal_and_therefore_rejected() -> None:
    """Semantic comparison: two spellings of one instant are one instant."""
    utc = PointInTime("2026-09-15T22:00:00Z")
    offset = PointInTime("2026-09-16T03:30:00+05:30")

    assert utc.compare(offset) == 0
    with pytest.raises(InvalidFuturesTradingSessionError, match="must be before closes_at"):
        FuturesTradingSession(date(2026, 9, 15), utc, offset)


def test_bounds_are_compared_semantically_not_textually() -> None:
    """A canonical instant drops zero fractions, so text ordering misleads."""
    opens = PointInTime("2026-09-15T22:00:00.5Z")
    closes = PointInTime("2026-09-15T22:00:00Z")

    # The text trap: '.' (0x2E) sorts before 'Z' (0x5A), so a lexical check
    # would call this window validly ordered. It is reversed in fact.
    assert opens.value < closes.value
    assert opens.compare(closes) > 0
    with pytest.raises(InvalidFuturesTradingSessionError, match="must be before closes_at"):
        FuturesTradingSession(date(2026, 9, 15), opens, closes)


def test_a_sub_second_window_is_valid_when_genuinely_ordered() -> None:
    session = FuturesTradingSession(
        date(2026, 9, 15),
        PointInTime("2026-09-15T22:00:00Z"),
        PointInTime("2026-09-15T22:00:00.5Z"),
    )

    assert session.opens_at.compare(session.closes_at) < 0


def test_the_error_is_a_value_error() -> None:
    assert issubclass(InvalidFuturesTradingSessionError, ValueError)


# ---------------------------------------------------------------------------
# FuturesTradingSession: value semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_name", ["trading_date", "opens_at", "closes_at"])
def test_the_session_is_immutable(field_name: str) -> None:
    with pytest.raises(AttributeError):
        setattr(_session(date(2026, 9, 15)), field_name, None)


def test_equality_and_hashing_follow_value() -> None:
    assert _session(date(2026, 9, 15)) == _session(date(2026, 9, 15))
    assert hash(_session(date(2026, 9, 15))) == hash(_session(date(2026, 9, 15)))
    assert len({_session(date(2026, 9, 15)), _session(date(2026, 9, 15))}) == 1
    assert _session(date(2026, 9, 15)) != _session(date(2026, 9, 16))


def test_only_the_approved_fields_are_stored() -> None:
    assert set(FuturesTradingSession.__slots__) == {"trading_date", "opens_at", "closes_at"}


def test_the_session_carries_no_instrument_identity() -> None:
    """Calendar-derived Application data, not a Core value."""
    session = _session(date(2026, 9, 15))

    for absent in ("contract", "product", "symbol", "exchange_code", "expiration_date"):
        assert not hasattr(session, absent)


def test_string_and_repr_forms() -> None:
    session = _session(date(2026, 9, 15))

    assert str(session) == "2026-09-15 [2026-09-14T22:00:00Z .. 2026-09-15T22:00:00Z]"
    assert repr(session).startswith("FuturesTradingSession(trading_date=")


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


def test_a_session_date_resolves_to_a_window(
    resolver: ReferenceFuturesTradingSessionResolver,
) -> None:
    session = resolver.resolve(_ES, date(2026, 9, 15))

    assert session == _session(date(2026, 9, 15))
    assert session is not None
    assert session.trading_date == date(2026, 9, 15)


@pytest.mark.parametrize("trading_date", _NON_SESSIONS)
def test_a_non_session_date_returns_none(
    resolver: ReferenceFuturesTradingSessionResolver, trading_date: date
) -> None:
    assert resolver.resolve(_ES, trading_date) is None


def test_an_unsupported_venue_raises_rather_than_returning_none(
    resolver: ReferenceFuturesTradingSessionResolver,
) -> None:
    with pytest.raises(FuturesSessionResolutionError, match="Unsupported futures venue"):
        resolver.resolve(_UNSUPPORTED, date(2026, 9, 15))


def test_none_and_an_error_remain_different_facts(
    resolver: ReferenceFuturesTradingSessionResolver,
) -> None:
    assert resolver.resolve(_ES, date(2026, 12, 25)) is None

    with pytest.raises(FuturesSessionResolutionError):
        resolver.resolve(_UNSUPPORTED, date(2026, 12, 25))


def test_the_returned_label_matches_the_request(
    resolver: ReferenceFuturesTradingSessionResolver,
) -> None:
    for label in (date(2026, 7, 3), date(2026, 9, 15)):
        session = resolver.resolve(_ES, label)
        assert session is not None
        assert session.trading_date == label


# ---------------------------------------------------------------------------
# sessions_in_range
# ---------------------------------------------------------------------------


def test_the_range_is_inclusive_at_both_ends(
    resolver: ReferenceFuturesTradingSessionResolver,
) -> None:
    sessions = resolver.sessions_in_range(_ES, date(2026, 7, 1), date(2026, 7, 6))

    assert [s.trading_date for s in sessions] == [
        date(2026, 7, 1),
        date(2026, 7, 2),
        date(2026, 7, 3),
        date(2026, 7, 6),
    ]


def test_weekends_and_holidays_are_absent_not_none(
    resolver: ReferenceFuturesTradingSessionResolver,
) -> None:
    sessions = resolver.sessions_in_range(_ES, date(2026, 7, 3), date(2026, 7, 6))

    assert [s.trading_date for s in sessions] == [date(2026, 7, 3), date(2026, 7, 6)]
    assert all(s is not None for s in sessions)


def test_a_single_day_range_returns_that_session(
    resolver: ReferenceFuturesTradingSessionResolver,
) -> None:
    sessions = resolver.sessions_in_range(_ES, date(2026, 9, 15), date(2026, 9, 15))

    assert len(sessions) == 1
    assert sessions[0].trading_date == date(2026, 9, 15)


def test_a_range_containing_no_session_returns_an_empty_tuple(
    resolver: ReferenceFuturesTradingSessionResolver,
) -> None:
    assert resolver.sessions_in_range(_ES, date(2026, 7, 4), date(2026, 7, 5)) == ()


def test_results_are_strictly_ascending_by_trading_date(
    resolver: ReferenceFuturesTradingSessionResolver,
) -> None:
    sessions = resolver.sessions_in_range(_ES, date(2026, 7, 1), date(2026, 9, 16))
    labels = [s.trading_date for s in sessions]

    assert labels == sorted(labels)
    assert len(labels) == len(set(labels))


def test_every_returned_session_falls_inside_the_requested_range(
    resolver: ReferenceFuturesTradingSessionResolver,
) -> None:
    start, end = date(2026, 7, 2), date(2026, 7, 6)

    for session in resolver.sessions_in_range(_ES, start, end):
        assert start <= session.trading_date <= end


def test_a_session_may_open_before_the_requested_range_starts(
    resolver: ReferenceFuturesTradingSessionResolver,
) -> None:
    """The range bounds labels, not instants; the first window reaches back."""
    (first,) = resolver.sessions_in_range(_ES, date(2026, 9, 15), date(2026, 9, 15))

    assert first.trading_date == date(2026, 9, 15)
    assert first.opens_at.value.startswith("2026-09-14")


def test_an_inverted_range_raises(
    resolver: ReferenceFuturesTradingSessionResolver,
) -> None:
    with pytest.raises(FuturesSessionResolutionError, match="must not be after end_date"):
        resolver.sessions_in_range(_ES, date(2026, 9, 16), date(2026, 9, 15))


def test_an_unsupported_venue_raises_for_a_range(
    resolver: ReferenceFuturesTradingSessionResolver,
) -> None:
    with pytest.raises(FuturesSessionResolutionError, match="Unsupported futures venue"):
        resolver.sessions_in_range(_UNSUPPORTED, date(2026, 9, 15), date(2026, 9, 16))


@pytest.mark.parametrize("position", ["start", "end"])
def test_a_datetime_bound_is_rejected(
    resolver: ReferenceFuturesTradingSessionResolver, position: str
) -> None:
    bounds = {"start_date": date(2026, 9, 15), "end_date": date(2026, 9, 16)}
    bounds[f"{position}_date"] = datetime(2026, 9, 15, 22, 0)

    with pytest.raises(TypeError, match="plain datetime.date"):
        resolver.sessions_in_range(_ES, **bounds)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def test_a_datetime_is_rejected_by_resolve(
    resolver: ReferenceFuturesTradingSessionResolver,
) -> None:
    with pytest.raises(TypeError, match="plain datetime.date"):
        resolver.resolve(_ES, datetime(2026, 9, 15, 22, 0))


def test_a_bare_exchange_code_is_not_accepted(
    resolver: ReferenceFuturesTradingSessionResolver,
) -> None:
    with pytest.raises(TypeError, match="FuturesProductReference"):
        resolver.resolve(_CME, date(2026, 9, 15))  # type: ignore[arg-type]


def test_a_futures_contract_is_not_accepted(
    resolver: ReferenceFuturesTradingSessionResolver,
) -> None:
    from northstar_core.derivatives import ExpirationDate

    with pytest.raises(TypeError, match="FuturesProductReference"):
        resolver.resolve(  # type: ignore[arg-type]
            FuturesContract(_ES, ExpirationDate("2026-12-18")), date(2026, 9, 15)
        )


def test_the_product_reaches_the_implementation_intact(
    resolver: ReferenceFuturesTradingSessionResolver,
) -> None:
    resolver.resolve(_CL, date(2026, 9, 15))
    resolver.sessions_in_range(_CL, date(2026, 9, 15), date(2026, 9, 16))

    assert [call[0] for call in resolver.calls] == [_CL, _CL]
    assert resolver.calls[0][0].product_code == Symbol("CL")


def test_products_sharing_a_venue_resolve_alike(
    resolver: ReferenceFuturesTradingSessionResolver,
) -> None:
    assert resolver.resolve(_ES, date(2026, 9, 15)) == resolver.resolve(_MES, date(2026, 9, 15))


# ---------------------------------------------------------------------------
# Port shape
# ---------------------------------------------------------------------------


def test_the_port_is_abstract() -> None:
    with pytest.raises(TypeError):
        FuturesTradingSessionResolver()  # type: ignore[abstract]


def test_the_port_exposes_exactly_two_methods() -> None:
    assert FuturesTradingSessionResolver.__abstractmethods__ == frozenset(
        {"resolve", "sessions_in_range"}
    )


def test_resolve_has_the_expected_signature() -> None:
    signature = inspect.signature(FuturesTradingSessionResolver.resolve)

    assert list(signature.parameters) == ["self", "product", "trading_date"]
    assert signature.parameters["product"].annotation == "FuturesProductReference"
    assert signature.parameters["trading_date"].annotation == "date"
    assert signature.return_annotation == "FuturesTradingSession | None"


def test_sessions_in_range_has_the_expected_signature() -> None:
    signature = inspect.signature(FuturesTradingSessionResolver.sessions_in_range)

    assert list(signature.parameters) == ["self", "product", "start_date", "end_date"]
    assert signature.parameters["start_date"].annotation == "date"
    assert signature.parameters["end_date"].annotation == "date"
    assert signature.return_annotation == "tuple[FuturesTradingSession, ...]"


# ---------------------------------------------------------------------------
# Boundaries and migration compatibility
# ---------------------------------------------------------------------------


def _module_imports(module_name: str) -> set[str]:
    import importlib

    module = importlib.import_module(module_name)
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    names |= {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    return names


def test_the_new_port_does_not_depend_on_the_equity_resolver() -> None:
    assert not issubclass(FuturesTradingSessionResolver, TradingSessionResolver)
    assert not issubclass(TradingSessionResolver, FuturesTradingSessionResolver)
    assert not hasattr(FuturesTradingSessionResolver, "resolve_session_close")

    modules = _module_imports("northstar_application.ports.futures_trading_session_resolver")
    assert not any("trading_session_resolver" in m for m in modules)


def test_the_new_port_imports_no_calendar_or_provider_library() -> None:
    modules = _module_imports("northstar_application.ports.futures_trading_session_resolver")

    for forbidden in ("exchange_calendars", "databento", "pandas", "zoneinfo", "pytz"):
        assert not any(m.split(".")[0] == forbidden for m in modules), f"port imports {forbidden}"


def test_the_two_futures_ports_are_unrelated_by_inheritance() -> None:
    """Neither supersedes the other by subclassing; the old one is just retained."""
    assert not issubclass(FuturesTradingSessionResolver, FuturesDailyBarCompletionResolver)
    assert not issubclass(FuturesDailyBarCompletionResolver, FuturesTradingSessionResolver)


def test_the_superseded_port_remains_available_for_infrastructure() -> None:
    """Temporary: northstar-infrastructure still implements this port."""
    assert FuturesDailyBarCompletionResolver.__abstractmethods__ == frozenset(
        {"resolve_completion"}
    )
    signature = inspect.signature(FuturesDailyBarCompletionResolver.resolve_completion)
    assert list(signature.parameters) == ["self", "product", "trading_date"]
    assert signature.return_annotation == "PointInTime | None"


def test_one_session_resolution_error_serves_both_ports() -> None:
    """A second equivalent error was deliberately not created."""
    from northstar_application.ports.futures_daily_bar_completion_resolver import (
        FuturesSessionResolutionError as ViaOldModule,
    )
    from northstar_application.ports.futures_trading_session_resolver import (
        FuturesSessionResolutionError as ViaNewModule,
    )

    assert ViaOldModule is ViaNewModule is FuturesSessionResolutionError
    assert issubclass(FuturesSessionResolutionError, RuntimeError)
    assert not issubclass(FuturesSessionResolutionError, TradingSessionResolutionError)


def test_the_public_surface_is_pinned() -> None:
    import northstar_application.ports as ports

    for name in (
        "FuturesTradingSession",
        "FuturesTradingSessionResolver",
        "InvalidFuturesTradingSessionError",
        "FuturesSessionResolutionError",
        "FuturesDailyBarCompletionResolver",
    ):
        assert name in ports.__all__
        assert getattr(ports, name) is not None


def test_no_deferred_acquisition_concept_is_exported_yet() -> None:
    import northstar_application.ports as ports

    for deferred in (
        "FuturesHistoricalMarketDataSource",
        "FuturesDailyHistoricalAcquisitionQuery",
        "FuturesSettlementScheduleResolver",
        "ProviderSymbolResolver",
    ):
        assert deferred not in ports.__all__
