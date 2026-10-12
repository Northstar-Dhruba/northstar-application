"""Tests for resolving a recommendation's validity window in trading sessions.

The calendar is SYNTHETIC, not NSE reference data: weekday sessions 03:45-10:00Z,
holidays 2026-10-02, 2026-10-20 and 2026-11-24, one special session on Saturday
2026-10-17 (03:30-05:00Z), and only 2026 loaded -- any other year raises, as an
unloaded calendar year does. Contract cutoffs are synthetic verified fixtures.

Research decides at the 2026-10-07 (Wed) close, 10:00Z. With M = 5 and issuance
at 12:00Z that day, the eligible sessions are Thu 10-08, Fri 10-09, Mon 10-12,
Tue 10-13 and Wed 10-14.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from northstar_core.derivatives import ExpirationDate
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol
from northstar_core.futures import FuturesContract, FuturesProductReference
from northstar_core.recommendations.futures import (
    FuturesRecommendationSessionWindow,
    RecommendationPolicyVersion,
)

from northstar_application.application_services import (
    FuturesRecommendationSessionWindowResolver,
    FuturesRecommendationWindowResolutionError,
    FuturesTradeRecommendationGenerationProfile,
)
from northstar_application.ports import (
    FuturesContractCutoff,
    FuturesSessionResolutionError,
    FuturesTradingSession,
    FuturesTradingSessionResolver,
)

_NIFTY = FuturesProductReference(Symbol("NIFTY"), ExchangeCode("NSE"))
_OCT = FuturesContract(_NIFTY, ExpirationDate("2026-10-27"))
_NOV = FuturesContract(_NIFTY, ExpirationDate("2026-11-24"))
_DEC = FuturesContract(_NIFTY, ExpirationDate("2026-12-29"))
_POLICY = RecommendationPolicyVersion("f4-synthetic-fixture-v1")
_DECIDED = PointInTime("2026-10-07T10:00:00Z")

_HOLIDAYS = frozenset({date(2026, 10, 2), date(2026, 10, 20), date(2026, 11, 24)})
_SPECIAL = {date(2026, 10, 17): ("03:30:00", "05:00:00")}


class SyntheticCalendar(FuturesTradingSessionResolver):
    """Table-driven synthetic calendar with only 2026 loaded."""

    def __init__(self, missing: frozenset[date] = frozenset()) -> None:
        self._missing = missing

    def resolve(
        self, product: FuturesProductReference, trading_date: date
    ) -> FuturesTradingSession | None:
        if trading_date.year != 2026:
            raise FuturesSessionResolutionError(f"Calendar year {trading_date.year} is not loaded.")
        if trading_date in self._missing:
            return None
        if trading_date in _SPECIAL:
            opens, closes = _SPECIAL[trading_date]
        elif trading_date.weekday() >= 5 or trading_date in _HOLIDAYS:
            return None
        else:
            opens, closes = "03:45:00", "10:00:00"
        day = trading_date.isoformat()
        return FuturesTradingSession(
            trading_date, PointInTime(f"{day}T{opens}Z"), PointInTime(f"{day}T{closes}Z")
        )

    def sessions_in_range(
        self, product: FuturesProductReference, start_date: date, end_date: date
    ) -> tuple[FuturesTradingSession, ...]:
        sessions = []
        day = start_date
        while day <= end_date:
            session = self.resolve(product, day)
            if session is not None:
                sessions.append(session)
            day += timedelta(days=1)
        return tuple(sessions)


_CALENDAR = SyntheticCalendar()


def _session(day: str) -> FuturesTradingSession:
    session = _CALENDAR.resolve(_NIFTY, date.fromisoformat(day))
    assert session is not None
    return session


def _close(day: str) -> PointInTime:
    return _session(day).closes_at


def _profile(sessions: int = 5) -> FuturesTradeRecommendationGenerationProfile:
    return FuturesTradeRecommendationGenerationProfile(
        "f5a-synthetic-profile-v1", sessions, _POLICY
    )


def _cutoff(contract: FuturesContract = _OCT, last: str = "2026-10-27") -> FuturesContractCutoff:
    return FuturesContractCutoff(contract, _session(last), "synthetic verified contract reference")


def _resolve(
    issued_at: str = "2026-10-07T12:00:00Z",
    *,
    sessions: int = 5,
    contract: FuturesContract = _OCT,
    cutoff: FuturesContractCutoff | None = None,
    decided: PointInTime = _DECIDED,
    calendar: FuturesTradingSessionResolver = _CALENDAR,
) -> FuturesRecommendationSessionWindow:
    return FuturesRecommendationSessionWindowResolver(calendar).resolve(
        contract, decided, PointInTime(issued_at), _profile(sessions), cutoff or _cutoff(contract)
    )


# -- eligibility ---------------------------------------------------------------------


def test_the_normal_window_starts_at_the_next_session() -> None:
    window = _resolve()

    assert window.eligible_from == _close("2026-10-08")
    assert window.valid_through == _close("2026-10-14")
    assert window.contract_cutoff == _close("2026-10-27")
    assert window.fits_contract


@pytest.mark.parametrize(
    "issued_at, first_eligible",
    [
        ("2026-10-08T03:44:59Z", "2026-10-08"),
        ("2026-10-08T03:45:00Z", "2026-10-09"),
        ("2026-10-08T06:00:00Z", "2026-10-09"),
        ("2026-10-08T10:00:00Z", "2026-10-09"),
        ("2026-10-08T12:00:00Z", "2026-10-09"),
    ],
    ids=[
        "just-before-open",
        "exactly-at-open",
        "during-session",
        "exactly-at-close",
        "after-close",
    ],
)
def test_a_session_is_eligible_only_if_it_opens_strictly_after_issuance(
    issued_at: str, first_eligible: str
) -> None:
    assert _resolve(issued_at, sessions=1).eligible_from == _close(first_eligible)


def test_late_issuance_moves_the_window_later_not_earlier() -> None:
    window = _resolve("2026-10-13T12:00:00Z", sessions=2)

    assert (window.eligible_from, window.valid_through) == (
        _close("2026-10-14"),
        _close("2026-10-15"),
    )


def test_weekends_are_skipped() -> None:
    window = _resolve("2026-10-09T12:00:00Z", sessions=1)

    assert window.eligible_from == _close("2026-10-12")


def test_holidays_are_skipped_and_special_sessions_count() -> None:
    """From 10-15 12:00Z: Fri 10-16, special Sat 10-17, Mon 10-19, (holiday 10-20), Wed 10-21."""
    window = _resolve("2026-10-15T12:00:00Z", sessions=4)

    assert window.eligible_from == _close("2026-10-16")
    assert window.valid_through == _close("2026-10-21")


def test_issuance_before_a_holiday_skips_it() -> None:
    assert _resolve(
        "2026-10-01T12:00:00Z", sessions=1, decided=PointInTime("2026-10-01T10:00:00Z")
    ).eligible_from == _close("2026-10-05")


def test_m_equal_to_one_is_a_single_session_window() -> None:
    window = _resolve(sessions=1)

    assert window.eligible_from == window.valid_through == _close("2026-10-08")


# -- the verified cutoff -------------------------------------------------------------


def test_a_horizon_ending_exactly_at_the_cutoff_fits() -> None:
    window = _resolve("2026-10-21T12:00:00Z", sessions=4)

    assert window.valid_through == window.contract_cutoff == _close("2026-10-27")
    assert window.fits_contract


def test_a_horizon_beyond_the_cutoff_is_kept_whole_for_f4_to_reject() -> None:
    window = _resolve("2026-10-21T12:00:00Z", sessions=5)

    assert window.valid_through == _close("2026-10-28")
    assert not window.fits_contract
    assert not window.expires_before_eligibility


def test_a_cutoff_before_any_eligible_session_is_kept_for_f4_to_reject() -> None:
    window = _resolve(
        "2026-10-27T12:00:00Z", sessions=1, decided=PointInTime("2026-10-27T10:00:00Z")
    )

    assert window.expires_before_eligibility


def test_an_expiration_date_that_is_not_a_session_uses_the_verified_last_session() -> None:
    """NOV expires on 2026-11-24, a holiday here; its verified last session is 11-23."""
    window = _resolve(
        "2026-11-18T12:00:00Z",
        sessions=3,
        contract=_NOV,
        cutoff=_cutoff(_NOV, "2026-11-23"),
        decided=PointInTime("2026-11-18T10:00:00Z"),
    )

    assert window.contract_cutoff == _close("2026-11-23")
    assert window.valid_through == _close("2026-11-23")


def test_the_cutoff_is_never_taken_from_the_expiration_date() -> None:
    """A cutoff dated on the holiday expiration cannot even be built from the calendar."""
    assert _CALENDAR.resolve(_NIFTY, date(2026, 11, 24)) is None


@pytest.mark.parametrize(
    "cutoff",
    [
        FuturesContractCutoff(_NOV, _session("2026-10-27"), "synthetic"),
        FuturesContractCutoff(
            _OCT,
            FuturesTradingSession(
                date(2026, 10, 27),
                PointInTime("2026-10-27T03:45:00Z"),
                PointInTime("2026-10-27T11:00:00Z"),
            ),
            "synthetic",
        ),
        FuturesContractCutoff(_OCT, _session("2026-10-28"), "synthetic"),
        FuturesContractCutoff(
            _OCT,
            FuturesTradingSession(
                date(2026, 10, 24),
                PointInTime("2026-10-24T03:45:00Z"),
                PointInTime("2026-10-24T10:00:00Z"),
            ),
            "synthetic",
        ),
    ],
    ids=["other-contract", "close-disagrees-with-calendar", "after-expiration", "not-a-session"],
)
def test_contradictory_cutoff_evidence_refuses_resolution(cutoff: FuturesContractCutoff) -> None:
    with pytest.raises(FuturesRecommendationWindowResolutionError):
        _resolve(cutoff=cutoff)


# -- fail closed -----------------------------------------------------------------------


def test_issuance_before_the_research_decision_is_refused() -> None:
    with pytest.raises(FuturesRecommendationWindowResolutionError, match="before its research"):
        _resolve("2026-10-07T09:59:59Z")


def test_an_unloaded_calendar_year_propagates_its_resolution_error() -> None:
    """From 12-28 a five-session window needs 2027, which this calendar does not hold."""
    with pytest.raises(FuturesSessionResolutionError):
        _resolve(
            "2026-12-28T12:00:00Z",
            contract=_DEC,
            cutoff=_cutoff(_DEC, "2026-12-29"),
            decided=PointInTime("2026-12-28T10:00:00Z"),
        )


def test_a_calendar_that_yields_too_few_sessions_refuses_resolution() -> None:
    """Every session from 08-04 is missing except the cutoff's; the span stays inside 2026."""
    gap = frozenset(date(2026, 8, 4) + timedelta(days=offset) for offset in range(120))
    with pytest.raises(FuturesRecommendationWindowResolutionError, match="resolves only 1 of 5"):
        _resolve(
            "2026-08-03T12:00:00Z",
            decided=PointInTime("2026-08-03T10:00:00Z"),
            calendar=SyntheticCalendar(missing=gap - {date(2026, 10, 27)}),
        )


@pytest.mark.parametrize("argument", ["contract", "decided", "issued", "profile", "cutoff"])
def test_wrong_input_types_are_refused(argument: str) -> None:
    values: dict[str, object] = {
        "contract": _OCT,
        "decided": _DECIDED,
        "issued": PointInTime("2026-10-07T12:00:00Z"),
        "profile": _profile(),
        "cutoff": _cutoff(),
    }
    values[argument] = None
    with pytest.raises(TypeError):
        FuturesRecommendationSessionWindowResolver(_CALENDAR).resolve(
            values["contract"],  # type: ignore[arg-type]
            values["decided"],  # type: ignore[arg-type]
            values["issued"],  # type: ignore[arg-type]
            values["profile"],  # type: ignore[arg-type]
            values["cutoff"],  # type: ignore[arg-type]
        )


def test_the_resolver_requires_a_session_resolver() -> None:
    with pytest.raises(TypeError):
        FuturesRecommendationSessionWindowResolver(object())  # type: ignore[arg-type]


def test_resolution_is_deterministic() -> None:
    assert _resolve() == _resolve()
