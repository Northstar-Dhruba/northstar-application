"""Contract tests for EquityTradingSession, its phases and its Application resolver port.

The port cannot enforce its own semantics, so the obligations are pinned here
against a conforming in-memory implementation. Every date, venue state and
phase below is invented for the test: none is an exchange calendar fact.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime

from northstar_application import ports
from northstar_application.ports import (
    EquitySessionKind,
    EquitySessionResolutionError,
    EquityTradingPhase,
    EquityTradingSession,
    EquityTradingSessionResolver,
    InvalidEquityTradingSessionError,
)

_VENUE = ExchangeCode("NSE")
_REFERENCE = "synthetic-reference port-contract"
_LABEL = date(2031, 3, 4)


def _phase(opens: str, closes: str) -> EquityTradingPhase:
    return EquityTradingPhase(opens_at=PointInTime(opens), closes_at=PointInTime(closes))


_MORNING = _phase("2031-03-04T04:30:00Z", "2031-03-04T06:30:00Z")
_AFTERNOON = _phase("2031-03-04T07:30:00Z", "2031-03-04T09:00:00Z")


def _session(
    phases: tuple[EquityTradingPhase, ...] = (_MORNING, _AFTERNOON),
    **overrides: object,
) -> EquityTradingSession:
    values: dict[str, object] = {
        "exchange_code": _VENUE,
        "trading_date": _LABEL,
        "kind": EquitySessionKind.NORMAL,
        "phases": phases,
        "reference": _REFERENCE,
    }
    values.update(overrides)
    return EquityTradingSession(**values)  # type: ignore[arg-type]


class _InMemoryResolver(EquityTradingSessionResolver):
    """Conforming implementation: stated sessions, stated closures, otherwise raise."""

    def __init__(self, sessions: dict[date, EquityTradingSession], closed: frozenset[date]) -> None:
        self._sessions = sessions
        self._closed = closed

    def resolve(
        self, exchange_code: ExchangeCode, trading_date: date
    ) -> EquityTradingSession | None:
        if exchange_code != _VENUE:
            raise EquitySessionResolutionError(f"Unsupported equity venue: {exchange_code}.")
        if trading_date in self._closed:
            return None
        session = self._sessions.get(trading_date)
        if session is None:
            raise EquitySessionResolutionError(f"Equity session on {trading_date} is not stated.")
        return session


# ----------------------------------------------------------------------
# EquityTradingPhase
# ----------------------------------------------------------------------


def test_phase_keeps_canonical_utc_boundaries() -> None:
    phase = _phase("2031-03-04T10:00:00+05:30", "2031-03-04T12:00:00+05:30")

    assert phase.opens_at == PointInTime("2031-03-04T04:30:00Z")
    assert phase.closes_at == PointInTime("2031-03-04T06:30:00Z")


@pytest.mark.parametrize(
    ("opens", "closes"),
    [
        ("2031-03-04T06:30:00Z", "2031-03-04T04:30:00Z"),
        ("2031-03-04T04:30:00Z", "2031-03-04T04:30:00Z"),
    ],
)
def test_phase_refuses_empty_or_reversed_boundaries(opens: str, closes: str) -> None:
    with pytest.raises(InvalidEquityTradingSessionError, match="before closes_at"):
        _phase(opens, closes)


@pytest.mark.parametrize("field", ["opens_at", "closes_at"])
@pytest.mark.parametrize("bad", [None, "2031-03-04T04:30:00Z", datetime(2031, 3, 4, 4, 30)])
def test_phase_refuses_non_point_in_time_boundaries(field: str, bad: object) -> None:
    values: dict[str, object] = {
        "opens_at": PointInTime("2031-03-04T04:30:00Z"),
        "closes_at": PointInTime("2031-03-04T06:30:00Z"),
    }
    values[field] = bad

    with pytest.raises(InvalidEquityTradingSessionError, match=field):
        EquityTradingPhase(**values)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# EquityTradingSession
# ----------------------------------------------------------------------


def test_session_exposes_first_open_and_last_close() -> None:
    session = _session()

    assert session.opens_at == _MORNING.opens_at
    assert session.closes_at == _AFTERNOON.closes_at
    assert session.phases == (_MORNING, _AFTERNOON)
    assert session.reference == _REFERENCE


def test_session_refuses_overlapping_phases() -> None:
    overlapping = _phase("2031-03-04T06:00:00Z", "2031-03-04T07:00:00Z")

    with pytest.raises(InvalidEquityTradingSessionError, match="separated by a break"):
        _session((_MORNING, overlapping))


def test_session_refuses_touching_phases_without_a_break() -> None:
    touching = _phase("2031-03-04T06:30:00Z", "2031-03-04T07:00:00Z")

    with pytest.raises(InvalidEquityTradingSessionError, match="separated by a break"):
        _session((_MORNING, touching))


def test_session_refuses_descending_phases() -> None:
    with pytest.raises(InvalidEquityTradingSessionError, match="ascending"):
        _session((_AFTERNOON, _MORNING))


@pytest.mark.parametrize("phases", [(), [_MORNING], None])
def test_session_refuses_missing_phases(phases: object) -> None:
    with pytest.raises(InvalidEquityTradingSessionError, match="non-empty tuple"):
        _session(phases)  # type: ignore[arg-type]


def test_session_refuses_a_phase_of_the_wrong_type() -> None:
    with pytest.raises(InvalidEquityTradingSessionError, match="EquityTradingPhase values"):
        _session((_MORNING, ("2031-03-04T07:30:00Z", "2031-03-04T09:00:00Z")))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("exchange_code", "NSE", "ExchangeCode"),
        ("trading_date", datetime(2031, 3, 4), "plain date"),
        ("trading_date", "2031-03-04", "plain date"),
        ("kind", "NORMAL", "EquitySessionKind"),
        ("reference", "", "reference"),
        ("reference", "   ", "reference"),
        ("reference", None, "reference"),
    ],
)
def test_session_refuses_invalid_identity_fields(field: str, bad: object, message: str) -> None:
    with pytest.raises(InvalidEquityTradingSessionError, match=message):
        _session(**{field: bad})


def test_session_is_immutable_and_value_equal() -> None:
    session = _session()

    assert session == _session()
    with pytest.raises(AttributeError):
        session.trading_date = date(2031, 3, 5)  # type: ignore[misc]


# ----------------------------------------------------------------------
# Resolver contract
# ----------------------------------------------------------------------


def test_resolver_distinguishes_session_closed_and_unstated_dates() -> None:
    closed = date(2031, 3, 5)
    unstated = date(2031, 3, 6)
    resolver = _InMemoryResolver({_LABEL: _session()}, frozenset({closed}))

    session = resolver.resolve(_VENUE, _LABEL)
    assert session is not None
    assert session.trading_date == _LABEL
    assert session.exchange_code == _VENUE
    assert resolver.resolve(_VENUE, closed) is None
    with pytest.raises(EquitySessionResolutionError, match="not stated"):
        resolver.resolve(_VENUE, unstated)
    with pytest.raises(EquitySessionResolutionError, match="Unsupported"):
        resolver.resolve(ExchangeCode("BSE"), _LABEL)


def test_resolver_port_is_abstract() -> None:
    with pytest.raises(TypeError):
        EquityTradingSessionResolver()  # type: ignore[abstract]


def test_ports_package_exports_the_equity_session_contract() -> None:
    for name in (
        "EquitySessionKind",
        "EquitySessionResolutionError",
        "EquityTradingPhase",
        "EquityTradingSession",
        "EquityTradingSessionResolver",
        "InvalidEquityTradingSessionError",
    ):
        assert name in ports.__all__


def test_session_kind_vocabulary_is_exactly_normal_and_special() -> None:
    assert {kind.value for kind in EquitySessionKind} == {"NORMAL", "SPECIAL"}
