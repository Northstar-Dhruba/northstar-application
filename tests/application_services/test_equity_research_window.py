"""EI-1C-3: research-window eligibility, with in-test port doubles.

The venue and applicability answers here are invented test doubles on a 2031
label. They are shaped like the NSE common boundary (09:15 to 15:15 IST) only
to exercise the policy arithmetic; they assert no exchange fact. Accepted NSE
evidence is exercised by the Infrastructure acceptance tests.

An ended interval is a time fact. Nothing here asserts that a candle exists,
is complete or is final, or that any order, fill or profit occurred.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from datetime import date, datetime, time, timedelta

import pytest
from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol, Timeframe

from northstar_application import application_services
from northstar_application.application_services import (
    NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW,
    EquityResearchWindowRejection,
    EvaluateEquityResearchWindowUseCase,
    InvalidEquityResearchWindowPolicyError,
)
from northstar_application.application_services import equity_research_window as window_module
from northstar_application.ports import (
    EquityListingApplicability,
    EquityListingApplicabilityClassification,
    EquityListingApplicabilitySource,
    EquitySessionKind,
    EquityVenueDayClassification,
    EquityVenueDayReference,
    EquityVenueDayReferenceSource,
    InvalidEquityResearchWindowReferenceError,
)
from northstar_application.ports import equity_research_window_reference as port_module

_NSE = ExchangeCode("NSE")
_DAY = date(2031, 3, 4)
_IST = timedelta(hours=5, minutes=30)
_LISTING = ListingReference(Symbol("TESTCO"), _NSE)
_OTHER_LISTING = ListingReference(Symbol("OTHERCO"), _NSE)
_REJECT = EquityResearchWindowRejection
_ELIGIBLE = EquityListingApplicabilityClassification.ELIGIBLE_FOR_NORMAL_RESEARCH_WINDOW
_LATE = PointInTime("2031-03-05T00:00:00Z")


def _venue(**overrides: object) -> EquityVenueDayReference:
    values: dict[str, object] = {
        "exchange_code": _NSE,
        "trading_date": _DAY,
        "classification": EquityVenueDayClassification.NORMAL_SESSION,
        "common_opens_at": PointInTime("2031-03-04T03:45:00Z"),
        "common_closes_at": PointInTime("2031-03-04T09:45:00Z"),
        "utc_offset": _IST,
        "reference": "test-venue-reference/1",
        "reason": "test normal session",
    }
    values.update(overrides)
    return EquityVenueDayReference(**values)  # type: ignore[arg-type]


def _unsupported(status: EquityVenueDayClassification) -> EquityVenueDayReference:
    return _venue(
        classification=status, common_opens_at=None, common_closes_at=None, utc_offset=None,
        reason=f"test {status}",
    )  # fmt: skip


class _Venue(EquityVenueDayReferenceSource):
    def __init__(self, answer: EquityVenueDayReference) -> None:
        self.answer = answer
        self.calls: list[tuple[ExchangeCode, date]] = []

    def venue_day(self, exchange_code: ExchangeCode, trading_date: date) -> EquityVenueDayReference:
        self.calls.append((exchange_code, trading_date))
        return self.answer


class _Applicability(EquityListingApplicabilitySource):
    def __init__(
        self, status: EquityListingApplicabilityClassification = _ELIGIBLE, **overrides: object
    ):
        self.status = status
        self.overrides = overrides
        self.calls = 0

    def applicability(
        self, listing: ListingReference, trading_date: date
    ) -> EquityListingApplicability:
        self.calls += 1
        values: dict[str, object] = {
            "listing": listing,
            "trading_date": trading_date,
            "classification": self.status,
            "evidence_reference": "test-applicability/1",
            "reason": f"test {self.status}",
        }
        values.update(self.overrides)
        return EquityListingApplicability(**values)  # type: ignore[arg-type]


def _evaluate(
    venue: EquityVenueDayReference | None = None,
    applicability: _Applicability | None = None,
    as_of: PointInTime = _LATE,
    listing: ListingReference = _LISTING,
) -> window_module.EquityResearchWindowEvaluation:
    use_case = EvaluateEquityResearchWindowUseCase(
        _Venue(venue or _venue()), applicability or _Applicability()
    )
    return use_case.execute(listing, _DAY, as_of)


def _ist(instant: PointInTime) -> time:
    utc = datetime.fromisoformat(instant.value.replace("Z", "+00:00"))
    return (utc + _IST).time()


# ----------------------------------------------------------------------
# A. Versioned policy
# ----------------------------------------------------------------------


def test_v1_policy_is_the_approved_versioned_window() -> None:
    policy = NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW
    assert policy.version == "nse-equity-intraday-v1-research-window/1"
    assert policy.exchange_code == _NSE
    assert policy.session_kind is EquitySessionKind.NORMAL
    assert (policy.starts, policy.ends) == (time(9, 15), time(15, 0))
    assert policy.timeframe == Timeframe("5m")
    assert "CE-D17" in policy.decision_ids
    assert EvaluateEquityResearchWindowUseCase(_Venue(_venue()), _Applicability()).policy is policy


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"version": " "}, "version"),
        ({"session_kind": EquitySessionKind.SPECIAL}, "NORMAL"),
        ({"timeframe": Timeframe("1m")}, "5m"),
        ({"starts": time(9, 17)}, "5-minute"),
        ({"ends": time(15, 0, 30)}, "5-minute"),
        ({"starts": time(15, 0), "ends": time(9, 15)}, "before"),
        ({"decision_ids": ()}, "decision_ids"),
        ({"exchange_code": "NSE"}, "ExchangeCode"),
    ],
)
def test_policy_refuses_invalid_values(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(InvalidEquityResearchWindowPolicyError, match=message):
        dataclasses.replace(NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW, **overrides)


# ----------------------------------------------------------------------
# B. Eligible window and intervals
# ----------------------------------------------------------------------


def test_eligible_window_has_exactly_69_intervals_from_0920_to_1500_ist() -> None:
    result = _evaluate()
    assert result.is_eligible and result.rejection is None
    assert len(result.intervals) == 69
    assert result.intervals[0].closes_at == PointInTime("2031-03-04T03:50:00Z")
    assert result.intervals[-1].closes_at == PointInTime("2031-03-04T09:30:00Z")
    assert (_ist(result.intervals[0].closes_at), _ist(result.intervals[-1].closes_at)) == (
        time(9, 20),
        time(15, 0),
    )
    assert result.window_opens_at == PointInTime("2031-03-04T03:45:00Z")
    assert result.window_closes_at == PointInTime("2031-03-04T09:30:00Z")
    for earlier, later in zip(result.intervals, result.intervals[1:], strict=False):
        assert earlier.closes_at == later.opens_at
    for interval in result.intervals:
        assert interval.closes_at.value.endswith("Z")
        assert interval.point_in_time == interval.closes_at


def test_window_intervals_never_reach_beyond_the_window_into_the_common_boundary() -> None:
    result = _evaluate()
    end = PointInTime("2031-03-04T09:30:00Z")
    assert all(interval.closes_at.compare(end) <= 0 for interval in result.intervals)
    assert all(_ist(i.opens_at) >= time(9, 15) for i in result.intervals)


def test_eligible_result_records_its_evidence_and_what_it_does_not_assess() -> None:
    result = _evaluate()
    assert result.venue_reference == "test-venue-reference/1"
    assert result.applicability_reference == "test-applicability/1"
    assert result.policy_version == NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW.version
    assert result.unscheduled_interruptions_assessed is False
    assert result.candles_assessed is False
    assert "not assessed" in result.detail


def test_cas_status_is_never_requested_and_a_15_15_common_boundary_suffices() -> None:
    applicability = _Applicability()
    result = _evaluate(applicability=applicability)
    assert result.is_eligible
    assert applicability.calls == 1
    status_names = {status.name for status in EquityListingApplicabilityClassification}
    assert not any("CAS" in name for name in status_names)


def test_a_wider_common_boundary_gives_the_same_window() -> None:
    wide = _venue(common_closes_at=PointInTime("2031-03-04T10:00:00Z"))
    assert _evaluate(wide).intervals == _evaluate().intervals


# ----------------------------------------------------------------------
# C. Point in time
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("as_of", "ended"),
    [
        ("2031-03-04T03:49:59Z", 0),
        ("2031-03-04T03:50:00Z", 1),
        ("2031-03-04T03:54:59.999999Z", 1),
        ("2031-03-04T03:55:00Z", 2),
        ("2031-03-04T06:02:00Z", 27),
        ("2031-03-04T09:29:59.999999Z", 68),
        ("2031-03-04T09:30:00Z", 69),
        ("2031-03-04T12:00:00Z", 69),
    ],
)
def test_ended_intervals_follow_the_supplied_as_of(as_of: str, ended: int) -> None:
    result = _evaluate(as_of=PointInTime(as_of))
    assert len(result.ended_intervals) == ended
    assert result.ended_intervals == result.intervals[:ended]
    assert result.as_of == PointInTime(as_of)


def test_identical_inputs_produce_identical_results() -> None:
    as_of = PointInTime("2031-03-04T07:00:00Z")
    assert _evaluate(as_of=as_of) == _evaluate(as_of=as_of)


# ----------------------------------------------------------------------
# D. Venue rejections
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "rejection"),
    [
        (EquityVenueDayClassification.CLOSED, _REJECT.VENUE_CLOSED),
        (EquityVenueDayClassification.SPECIAL_SESSION, _REJECT.SPECIAL_SESSION_UNSUPPORTED),
        (EquityVenueDayClassification.NOT_STATED, _REJECT.VENUE_DAY_NOT_STATED),
        (EquityVenueDayClassification.OUTSIDE_COVERAGE, _REJECT.OUTSIDE_REFERENCE_COVERAGE),
        (
            EquityVenueDayClassification.NORMAL_SESSION_COMMON_BOUNDARY_UNSUPPORTED,
            _REJECT.COMMON_BOUNDARY_UNSUPPORTED,
        ),
    ],
)
def test_unsupported_venue_days_fail_closed_before_applicability_is_asked(
    status: EquityVenueDayClassification, rejection: EquityResearchWindowRejection
) -> None:
    applicability = _Applicability()
    result = _evaluate(_unsupported(status), applicability)
    assert result.rejection is rejection
    assert result.intervals == () and result.ended_intervals == ()
    assert result.window_opens_at is None
    assert result.detail == f"test {status}"
    assert result.venue_reference == "test-venue-reference/1"
    assert applicability.calls == 0


@pytest.mark.parametrize(
    ("opens", "closes"),
    [
        ("2031-03-04T04:00:00Z", "2031-03-04T09:45:00Z"),  # opens 09:30 IST
        ("2031-03-04T03:45:00Z", "2031-03-04T09:00:00Z"),  # closes 14:30 IST
    ],
)
def test_window_outside_the_common_boundary_is_refused(opens: str, closes: str) -> None:
    venue = _venue(common_opens_at=PointInTime(opens), common_closes_at=PointInTime(closes))
    assert _evaluate(venue).rejection is _REJECT.WINDOW_OUTSIDE_COMMON_BOUNDARY


def test_misaligned_common_boundary_is_refused_not_rounded() -> None:
    venue = _venue(common_opens_at=PointInTime("2031-03-04T03:44:00Z"))
    assert _evaluate(venue).rejection is _REJECT.WINDOW_NOT_ALIGNED


def test_an_offset_that_breaks_alignment_is_refused() -> None:
    venue = _venue(
        utc_offset=timedelta(hours=5, minutes=33),
        common_opens_at=PointInTime("2031-03-04T03:40:00Z"),
    )
    assert _evaluate(venue).rejection is _REJECT.WINDOW_NOT_ALIGNED


@pytest.mark.parametrize(
    "overrides", [{"trading_date": date(2031, 3, 5)}, {"exchange_code": ExchangeCode("BSE")}]
)
def test_a_venue_answer_for_another_date_or_venue_is_inconsistent(
    overrides: dict[str, object],
) -> None:
    assert _evaluate(_venue(**overrides)).rejection is _REJECT.VENUE_REFERENCE_INCONSISTENT


def test_other_exchange_listings_are_not_covered_by_the_nse_policy() -> None:
    venue = _Venue(_venue())
    use_case = EvaluateEquityResearchWindowUseCase(venue, _Applicability())
    result = use_case.execute(ListingReference(Symbol("TESTCO"), ExchangeCode("BSE")), _DAY, _LATE)
    assert result.rejection is _REJECT.POLICY_NOT_APPLICABLE
    assert venue.calls == []


# ----------------------------------------------------------------------
# E. Listing applicability
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "rejection"),
    [
        (EquityListingApplicabilityClassification.UNKNOWN, _REJECT.LISTING_APPLICABILITY_UNKNOWN),
        (
            EquityListingApplicabilityClassification.SPECIAL_PRE_OPEN,
            _REJECT.LISTING_SPECIAL_PRE_OPEN,
        ),
        (
            EquityListingApplicabilityClassification.SUSPENDED_OR_NONCONTINUOUS,
            _REJECT.LISTING_SUSPENDED_OR_NONCONTINUOUS,
        ),
        (
            EquityListingApplicabilityClassification.OTHER_UNSUPPORTED_EXCEPTION,
            _REJECT.LISTING_OTHER_EXCEPTION,
        ),
    ],
)
def test_anything_but_explicit_eligibility_fails_closed(
    status: EquityListingApplicabilityClassification, rejection: EquityResearchWindowRejection
) -> None:
    result = _evaluate(applicability=_Applicability(status))
    assert result.rejection is rejection
    assert result.intervals == ()
    assert result.applicability_reference == "test-applicability/1"


@pytest.mark.parametrize(
    "overrides", [{"listing": _OTHER_LISTING}, {"trading_date": date(2031, 3, 5)}]
)
def test_an_applicability_answer_for_another_listing_or_date_is_inconsistent(
    overrides: dict[str, object],
) -> None:
    result = _evaluate(applicability=_Applicability(**overrides))
    assert result.rejection is _REJECT.APPLICABILITY_INCONSISTENT


# ----------------------------------------------------------------------
# F. Port values
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"common_closes_at": None},
        {"utc_offset": None},
        {"utc_offset": timedelta(seconds=30)},
        {"utc_offset": timedelta(days=1)},
        {"common_opens_at": PointInTime("2031-03-04T09:45:00Z")},
        {"reference": ""},
        {"trading_date": datetime(2031, 3, 4)},
    ],
)
def test_a_normal_session_answer_needs_complete_consistent_boundaries(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(InvalidEquityResearchWindowReferenceError):
        _venue(**overrides)


def test_a_non_normal_answer_cannot_carry_boundaries() -> None:
    with pytest.raises(InvalidEquityResearchWindowReferenceError, match="must not carry"):
        _venue(classification=EquityVenueDayClassification.CLOSED)


def test_applicability_answers_need_identity_evidence_and_reason() -> None:
    for overrides in (
        {"listing": "TESTCO@NSE"},
        {"evidence_reference": " "},
        {"reason": ""},
        {"classification": "ELIGIBLE"},
    ):
        values: dict[str, object] = {
            "listing": _LISTING,
            "trading_date": _DAY,
            "classification": _ELIGIBLE,
            "evidence_reference": "test",
            "reason": "test",
        }
        values.update(overrides)
        with pytest.raises(InvalidEquityResearchWindowReferenceError):
            EquityListingApplicability(**values)  # type: ignore[arg-type]


def test_use_case_refuses_wrong_inputs_and_wrong_port_answers() -> None:
    use_case = EvaluateEquityResearchWindowUseCase(_Venue(_venue()), _Applicability())
    with pytest.raises(TypeError):
        use_case.execute("TESTCO@NSE", _DAY, _LATE)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        use_case.execute(_LISTING, datetime(2031, 3, 4), _LATE)
    with pytest.raises(TypeError):
        use_case.execute(_LISTING, _DAY, "2031-03-04T09:30:00Z")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        EvaluateEquityResearchWindowUseCase(object(), _Applicability())  # type: ignore[arg-type]

    class _Broken(EquityVenueDayReferenceSource):
        def venue_day(self, exchange_code: ExchangeCode, trading_date: date) -> object:  # type: ignore[override]
            return {"classification": "NORMAL_SESSION"}

    with pytest.raises(TypeError):
        EvaluateEquityResearchWindowUseCase(_Broken(), _Applicability()).execute(
            _LISTING, _DAY, _LATE
        )


# ----------------------------------------------------------------------
# G. Scope
# ----------------------------------------------------------------------


@pytest.mark.parametrize("module", [window_module, port_module])
def test_modules_read_no_clock_and_import_no_infrastructure(module: object) -> None:
    source = inspect.getsource(module)  # type: ignore[arg-type]
    tree = ast.parse(source)
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not called & {"now", "today", "utcnow", "monotonic", "localtime"}
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    for name in imported:
        assert not name.startswith("northstar_infrastructure")
        assert "zoneinfo" not in name and "socket" not in name
    for word in ("order", "fill", "position", "pnl", "broker"):
        assert not any(word == n.lower() for n in imported)


def test_window_module_returns_no_session_and_implements_no_strategy() -> None:
    fields = {f.name for f in dataclasses.fields(window_module.EquityResearchWindowEvaluation)}
    assert "session" not in fields
    tree = ast.parse(inspect.getsource(window_module))
    identifiers = {node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)}
    identifiers |= {node.attr.lower() for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    for word in ("signal", "entry", "exit", "cutoff", "order", "fill", "position", "pnl"):
        assert not any(word in identifier for identifier in identifiers), word
    assert "time(14, 25)" not in inspect.getsource(window_module)
    assert "time(14, 30)" not in inspect.getsource(window_module)


def test_packages_export_the_research_window_contracts() -> None:
    for name in (
        "NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW",
        "EquityResearchWindowPolicy",
        "EquityResearchWindowRejection",
        "EvaluateEquityResearchWindowUseCase",
    ):
        assert name in application_services.__all__
