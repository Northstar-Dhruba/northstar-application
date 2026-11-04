"""Equity Intraday V1 research-window eligibility (EI-1C-3).

A research window is a Northstar policy, not an exchange session. The approved
V1 window (owner decision CE-D17) is 09:15 to 15:00 IST on normal NSE
cash-equity sessions. This module answers, for one listing on one date and at
one explicitly supplied observation instant:

A. Does the venue reference state a scheduled normal session with a common
   continuous-trading boundary? (``EquityVenueDayReferenceSource``)
B. Does the window lie inside that common boundary?
C. Is there explicit evidence that the listing is eligible for the window on
   that date? Unknown fails closed (CE-D18). (``EquityListingApplicabilitySource``)
D. Which 5-minute intervals of the window had ended at the observation instant?
E. If not eligible, exactly why.

Why a common boundary
---------------------
From the CAS go-live, an NSE security's continuous close depends on whether it
is CAS-eligible, and no historical eligibility is reconstructed (CE-D14). The
window therefore never relies on a security's full session. It relies only on
the boundary common to every normal-market security -- the latest opening and
the earliest continuous close -- so no CAS status is assumed. Where any group's
close is not established, there is no common boundary and the date fails
closed.

How intervals are produced
--------------------------
The EI-1A ``EquitySessionIntervalGrid`` is built over the common continuous
boundary, and only the intervals wholly inside the window are kept. The
session value fed to the grid is internal and never returned: the window is
never presented as an exchange session, and no universal full session is
manufactured.

What an eligible answer does not mean
-------------------------------------
- It is not evidence that the listing traded, that the market ran without an
  unscheduled interruption (owner decision CE-D8 is pending), or that any
  candle exists, is complete or is final.
- An interval that has ended is a time fact only. The interval ending at
  15:00 may have ended at 15:00; that says nothing about its candle, and no
  order, fill, position or profit is implied.
- It is not an approval of any EI-0 signal, entry or exit rule.

Nothing reads a clock, provider or network: the observation instant is always
supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timezone
from enum import StrEnum

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Timeframe

from northstar_application.application_services.equity_session_interval_grid import (
    EquitySessionInterval,
    EquitySessionIntervalGrid,
    InvalidEquitySessionIntervalGridError,
)
from northstar_application.ports.equity_research_window_reference import (
    EquityListingApplicability,
    EquityListingApplicabilityClassification,
    EquityListingApplicabilitySource,
    EquityVenueDayClassification,
    EquityVenueDayReference,
    EquityVenueDayReferenceSource,
)
from northstar_application.ports.equity_trading_session_resolver import (
    EquitySessionKind,
    EquityTradingPhase,
    EquityTradingSession,
)

_FIVE_MINUTES = Timeframe("5m")


class InvalidEquityResearchWindowPolicyError(ValueError):
    """Raised when a research-window policy value is invalid."""


def _policy_fail(message: str) -> None:
    raise InvalidEquityResearchWindowPolicyError(message)


@dataclass(frozen=True, slots=True)
class EquityResearchWindowPolicy:
    """One versioned research window: local wall-clock bounds on one venue's normal sessions.

    It is a Northstar research policy, never the exchange's trading hours.
    ``decision_ids`` name the owner decisions that approved it.
    """

    version: str
    exchange_code: ExchangeCode
    session_kind: EquitySessionKind
    starts: time
    ends: time
    timeframe: Timeframe
    decision_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        subject = "EquityResearchWindowPolicy"
        if not isinstance(self.version, str) or not self.version.strip():
            _policy_fail(f"{subject} version must be a non-empty string.")
        if not isinstance(self.exchange_code, ExchangeCode):
            _policy_fail(f"{subject} exchange_code must be an ExchangeCode.")
        if self.session_kind is not EquitySessionKind.NORMAL:
            _policy_fail(f"{subject} applies to NORMAL sessions only.")
        if self.timeframe != _FIVE_MINUTES:
            _policy_fail(f"{subject} supports only the 5m timeframe of the EI-1A grid.")
        for name in ("starts", "ends"):
            value = getattr(self, name)
            if not isinstance(value, time) or value.tzinfo is not None:
                _policy_fail(f"{subject} {name} must be a naive local wall-clock time.")
            if value.second or value.microsecond or value.minute % 5:
                _policy_fail(f"{subject} {name} must lie on a 5-minute boundary.")
        if self.starts >= self.ends:
            _policy_fail(f"{subject} starts must be before ends.")
        if (
            not isinstance(self.decision_ids, tuple)
            or not self.decision_ids
            or not all(isinstance(d, str) and d.strip() for d in self.decision_ids)
        ):
            _policy_fail(f"{subject} decision_ids must be a non-empty tuple of strings.")


NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW = EquityResearchWindowPolicy(
    version="nse-equity-intraday-v1-research-window/1",
    exchange_code=ExchangeCode("NSE"),
    session_kind=EquitySessionKind.NORMAL,
    starts=time(9, 15),
    ends=time(15, 0),
    timeframe=_FIVE_MINUTES,
    decision_ids=("CE-D17", "CE-D14", "CE-D18"),
)


class EquityResearchWindowRejection(StrEnum):
    """Why a listing and date may not use the research window."""

    POLICY_NOT_APPLICABLE = "POLICY_NOT_APPLICABLE"
    VENUE_REFERENCE_INCONSISTENT = "VENUE_REFERENCE_INCONSISTENT"
    OUTSIDE_REFERENCE_COVERAGE = "OUTSIDE_REFERENCE_COVERAGE"
    VENUE_DAY_NOT_STATED = "VENUE_DAY_NOT_STATED"
    VENUE_CLOSED = "VENUE_CLOSED"
    SPECIAL_SESSION_UNSUPPORTED = "SPECIAL_SESSION_UNSUPPORTED"
    COMMON_BOUNDARY_UNSUPPORTED = "COMMON_BOUNDARY_UNSUPPORTED"
    WINDOW_OUTSIDE_COMMON_BOUNDARY = "WINDOW_OUTSIDE_COMMON_BOUNDARY"
    WINDOW_NOT_ALIGNED = "WINDOW_NOT_ALIGNED"
    APPLICABILITY_INCONSISTENT = "APPLICABILITY_INCONSISTENT"
    LISTING_APPLICABILITY_UNKNOWN = "LISTING_APPLICABILITY_UNKNOWN"
    LISTING_SPECIAL_PRE_OPEN = "LISTING_SPECIAL_PRE_OPEN"
    LISTING_SUSPENDED_OR_NONCONTINUOUS = "LISTING_SUSPENDED_OR_NONCONTINUOUS"
    LISTING_OTHER_EXCEPTION = "LISTING_OTHER_EXCEPTION"


_DAY = EquityVenueDayClassification
_VENUE_REJECTIONS: dict[EquityVenueDayClassification, EquityResearchWindowRejection] = {
    _DAY.OUTSIDE_COVERAGE: EquityResearchWindowRejection.OUTSIDE_REFERENCE_COVERAGE,
    _DAY.NOT_STATED: EquityResearchWindowRejection.VENUE_DAY_NOT_STATED,
    _DAY.CLOSED: EquityResearchWindowRejection.VENUE_CLOSED,
    _DAY.SPECIAL_SESSION: EquityResearchWindowRejection.SPECIAL_SESSION_UNSUPPORTED,
    _DAY.NORMAL_SESSION_COMMON_BOUNDARY_UNSUPPORTED: (
        EquityResearchWindowRejection.COMMON_BOUNDARY_UNSUPPORTED
    ),
}

_LISTING_REJECTIONS: dict[
    EquityListingApplicabilityClassification, EquityResearchWindowRejection
] = {
    EquityListingApplicabilityClassification.UNKNOWN: (
        EquityResearchWindowRejection.LISTING_APPLICABILITY_UNKNOWN
    ),
    EquityListingApplicabilityClassification.SPECIAL_PRE_OPEN: (
        EquityResearchWindowRejection.LISTING_SPECIAL_PRE_OPEN
    ),
    EquityListingApplicabilityClassification.SUSPENDED_OR_NONCONTINUOUS: (
        EquityResearchWindowRejection.LISTING_SUSPENDED_OR_NONCONTINUOUS
    ),
    EquityListingApplicabilityClassification.OTHER_UNSUPPORTED_EXCEPTION: (
        EquityResearchWindowRejection.LISTING_OTHER_EXCEPTION
    ),
}


@dataclass(frozen=True, slots=True)
class EquityResearchWindowEvaluation:
    """The research-window answer for one listing, date and observation instant.

    ``rejection`` is None exactly when the window may be used. Then
    ``intervals`` are the window's expected 5-minute intervals, ascending and
    stamped at their UTC end, and ``ended_intervals`` those whose end is at or
    before ``as_of``. A rejected answer has no window and no intervals.
    """

    listing: ListingReference
    trading_date: date
    as_of: PointInTime
    policy_version: str
    rejection: EquityResearchWindowRejection | None
    detail: str
    venue_reference: str | None
    applicability_reference: str | None
    window_opens_at: PointInTime | None
    window_closes_at: PointInTime | None
    intervals: tuple[EquitySessionInterval, ...]
    ended_intervals: tuple[EquitySessionInterval, ...]

    @property
    def is_eligible(self) -> bool:
        """Return whether the scheduled window may be used for this listing and date."""
        return self.rejection is None

    @property
    def unscheduled_interruptions_assessed(self) -> bool:
        """Return whether unscheduled halts or outages were ruled out; never (CE-D8)."""
        return False

    @property
    def candles_assessed(self) -> bool:
        """Return whether any candle's existence, completeness or finality was checked; never."""
        return False


def _to_datetime(instant: PointInTime) -> datetime:
    return datetime.fromisoformat(instant.value.replace("Z", "+00:00")).astimezone(UTC)


def _to_point_in_time(value: datetime) -> PointInTime:
    return PointInTime(value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))


class EvaluateEquityResearchWindowUseCase:
    """Decides whether one listing and date may use a research window, and which intervals ended."""

    def __init__(
        self,
        venue_days: EquityVenueDayReferenceSource,
        applicability: EquityListingApplicabilitySource,
        policy: EquityResearchWindowPolicy = NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW,
    ) -> None:
        if not isinstance(venue_days, EquityVenueDayReferenceSource):
            raise TypeError("venue_days must be an EquityVenueDayReferenceSource.")
        if not isinstance(applicability, EquityListingApplicabilitySource):
            raise TypeError("applicability must be an EquityListingApplicabilitySource.")
        if not isinstance(policy, EquityResearchWindowPolicy):
            raise TypeError("policy must be an EquityResearchWindowPolicy.")
        self._venue_days = venue_days
        self._applicability = applicability
        self._policy = policy

    @property
    def policy(self) -> EquityResearchWindowPolicy:
        """Return the research-window policy this use case applies."""
        return self._policy

    def execute(
        self, listing: ListingReference, trading_date: date, as_of: PointInTime
    ) -> EquityResearchWindowEvaluation:
        """Evaluate the window for ``listing`` on ``trading_date`` as observed at ``as_of``."""
        if not isinstance(listing, ListingReference):
            raise TypeError("listing must be a ListingReference.")
        if isinstance(trading_date, datetime) or not isinstance(trading_date, date):
            raise TypeError("trading_date must be a plain date.")
        if not isinstance(as_of, PointInTime):
            raise TypeError("as_of must be a PointInTime.")
        policy = self._policy

        def reject(
            reason: EquityResearchWindowRejection,
            detail: str,
            venue: str | None = None,
            applicability: str | None = None,
        ) -> EquityResearchWindowEvaluation:
            return EquityResearchWindowEvaluation(
                listing=listing,
                trading_date=trading_date,
                as_of=as_of,
                policy_version=policy.version,
                rejection=reason,
                detail=detail,
                venue_reference=venue,
                applicability_reference=applicability,
                window_opens_at=None,
                window_closes_at=None,
                intervals=(),
                ended_intervals=(),
            )

        if listing.exchange_code != policy.exchange_code:
            return reject(
                EquityResearchWindowRejection.POLICY_NOT_APPLICABLE,
                f"Policy {policy.version} applies to {policy.exchange_code}, "
                f"not {listing.exchange_code}.",
            )

        venue = self._venue_days.venue_day(policy.exchange_code, trading_date)
        if not isinstance(venue, EquityVenueDayReference):
            raise TypeError("The venue-day source must return an EquityVenueDayReference.")
        if venue.exchange_code != policy.exchange_code or venue.trading_date != trading_date:
            return reject(
                EquityResearchWindowRejection.VENUE_REFERENCE_INCONSISTENT,
                f"The venue reference answered for {venue.exchange_code} "
                f"{venue.trading_date.isoformat()}, not the requested date.",
            )
        if venue.classification is not EquityVenueDayClassification.NORMAL_SESSION:
            return reject(_VENUE_REJECTIONS[venue.classification], venue.reason, venue.reference)

        offset = timezone(venue.utc_offset)  # type: ignore[arg-type]
        window_start = datetime.combine(trading_date, policy.starts, tzinfo=offset).astimezone(UTC)
        window_end = datetime.combine(trading_date, policy.ends, tzinfo=offset).astimezone(UTC)
        common_open = _to_datetime(venue.common_opens_at)  # type: ignore[arg-type]
        common_close = _to_datetime(venue.common_closes_at)  # type: ignore[arg-type]
        if window_start < common_open or window_end > common_close:
            return reject(
                EquityResearchWindowRejection.WINDOW_OUTSIDE_COMMON_BOUNDARY,
                f"Window {policy.starts}-{policy.ends} does not lie inside the common continuous "
                f"boundary {venue.common_opens_at}-{venue.common_closes_at}.",
                venue.reference,
            )

        found = self._applicability.applicability(listing, trading_date)
        if not isinstance(found, EquityListingApplicability):
            raise TypeError("The applicability source must return an EquityListingApplicability.")
        if found.listing != listing or found.trading_date != trading_date:
            return reject(
                EquityResearchWindowRejection.APPLICABILITY_INCONSISTENT,
                f"The applicability source answered for {found.listing} "
                f"{found.trading_date.isoformat()}, not the requested listing and date.",
                venue.reference,
                found.evidence_reference,
            )
        if (
            found.classification
            is not EquityListingApplicabilityClassification.ELIGIBLE_FOR_NORMAL_RESEARCH_WINDOW
        ):
            return reject(
                _LISTING_REJECTIONS[found.classification],
                found.reason,
                venue.reference,
                found.evidence_reference,
            )

        # Internal grid input over the common continuous boundary; never returned.
        common = EquityTradingSession(
            exchange_code=policy.exchange_code,
            trading_date=trading_date,
            kind=EquitySessionKind.NORMAL,
            phases=(
                EquityTradingPhase(
                    opens_at=venue.common_opens_at,  # type: ignore[arg-type]
                    closes_at=venue.common_closes_at,  # type: ignore[arg-type]
                ),
            ),
            reference=f"common continuous boundary of {venue.reference}",
        )
        try:
            grid = EquitySessionIntervalGrid(common, policy.timeframe)
        except InvalidEquitySessionIntervalGridError as exc:
            return reject(
                EquityResearchWindowRejection.WINDOW_NOT_ALIGNED,
                str(exc),
                venue.reference,
                found.evidence_reference,
            )
        opens_at = _to_point_in_time(window_start)
        closes_at = _to_point_in_time(window_end)
        intervals = tuple(
            interval
            for interval in grid.intervals
            if interval.opens_at.compare(opens_at) >= 0
            and interval.closes_at.compare(closes_at) <= 0
        )
        tiled = (
            bool(intervals)
            and intervals[0].opens_at == opens_at
            and intervals[-1].closes_at == closes_at
            and all(
                a.closes_at == b.opens_at for a, b in zip(intervals, intervals[1:], strict=False)
            )
        )
        if not tiled:
            return reject(
                EquityResearchWindowRejection.WINDOW_NOT_ALIGNED,
                f"Window {opens_at}-{closes_at} is not tiled exactly by {policy.timeframe} "
                "intervals of the common continuous boundary.",
                venue.reference,
                found.evidence_reference,
            )
        return EquityResearchWindowEvaluation(
            listing=listing,
            trading_date=trading_date,
            as_of=as_of,
            policy_version=policy.version,
            rejection=None,
            detail=(
                "Scheduled research window eligible; unscheduled interruptions and candle "
                "availability are not assessed."
            ),
            venue_reference=venue.reference,
            applicability_reference=found.evidence_reference,
            window_opens_at=opens_at,
            window_closes_at=closes_at,
            intervals=intervals,
            ended_intervals=tuple(i for i in intervals if i.closes_at.compare(as_of) <= 0),
        )
