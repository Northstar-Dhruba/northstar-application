"""Application ports for the reference facts an equity research window depends on.

Two questions are kept apart, and neither is an exchange trading session:

1. ``EquityVenueDayReferenceSource``: for a venue and date, is there a
   scheduled normal session, and what is its *common* continuous-trading
   boundary -- the latest opening and the earliest continuous close that apply
   to every normal-market security, whatever its instrument-level group (for
   NSE, CAS-eligible or not)? A research window may only lie inside it. When a
   group's close is not established, there is no common boundary, and the
   answer says so rather than guessing.

2. ``EquityListingApplicabilitySource``: for one listing on one date, is there
   explicit evidence that the listing is eligible for the normal research
   window, or a known reason it is not? Unknown is a first-class answer, and
   it is never read as eligible. A missing exception record is not evidence
   that no exception occurred.

Neither port resolves a complete instrument session, and neither replaces
``EquityTradingSessionResolver``: a venue whose full session depends on an
instrument's group still fails closed there. Nothing here says that any
security traded, that the market ran without interruption, or that any candle
exists or is final.

The venue answer carries the venue's UTC offset on that date, derived by the
adapter from its named timezone, so Application can place wall-clock policy
times without a timezone database and without a hard-coded offset.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime


class InvalidEquityResearchWindowReferenceError(ValueError):
    """Raised when a venue-day or listing-applicability answer is malformed."""


class EquityVenueDayClassification(StrEnum):
    """What the venue reference states about one date.

    NORMAL_SESSION means a scheduled normal venue session with an accepted
    common continuous boundary. It does not prove actual trading,
    uninterrupted availability or complete market data.
    """

    NORMAL_SESSION = "NORMAL_SESSION"
    NORMAL_SESSION_COMMON_BOUNDARY_UNSUPPORTED = "NORMAL_SESSION_COMMON_BOUNDARY_UNSUPPORTED"
    CLOSED = "CLOSED"
    SPECIAL_SESSION = "SPECIAL_SESSION"
    NOT_STATED = "NOT_STATED"
    OUTSIDE_COVERAGE = "OUTSIDE_COVERAGE"


class EquityListingApplicabilityClassification(StrEnum):
    """What explicit evidence says about one listing on one date."""

    ELIGIBLE_FOR_NORMAL_RESEARCH_WINDOW = "ELIGIBLE_FOR_NORMAL_RESEARCH_WINDOW"
    SPECIAL_PRE_OPEN = "SPECIAL_PRE_OPEN"
    SUSPENDED_OR_NONCONTINUOUS = "SUSPENDED_OR_NONCONTINUOUS"
    OTHER_UNSUPPORTED_EXCEPTION = "OTHER_UNSUPPORTED_EXCEPTION"
    UNKNOWN = "UNKNOWN"


def _fail(message: str) -> None:
    raise InvalidEquityResearchWindowReferenceError(message)


def _require_plain_date(value: object, subject: str) -> None:
    if isinstance(value, datetime) or not isinstance(value, date):
        _fail(f"{subject} trading_date must be a plain date.")


def _require_text(value: object, subject: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{subject} {name} must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class EquityVenueDayReference:
    """The venue reference's answer for one date.

    ``common_opens_at`` and ``common_closes_at`` are set exactly when the
    classification is NORMAL_SESSION: they bound the continuous trading common to
    every normal-market security, in canonical UTC. ``utc_offset`` is the
    venue's offset from UTC on the date, set with them. ``reference`` names
    the evidence the answer rests on and its limitations; ``reason`` explains
    any other classification.
    """

    exchange_code: ExchangeCode
    trading_date: date
    classification: EquityVenueDayClassification
    common_opens_at: PointInTime | None
    common_closes_at: PointInTime | None
    utc_offset: timedelta | None
    reference: str
    reason: str

    def __post_init__(self) -> None:
        subject = "EquityVenueDayReference"
        if not isinstance(self.exchange_code, ExchangeCode):
            _fail(f"{subject} exchange_code must be an ExchangeCode.")
        _require_plain_date(self.trading_date, subject)
        if not isinstance(self.classification, EquityVenueDayClassification):
            _fail(f"{subject} classification must be an EquityVenueDayClassification.")
        _require_text(self.reference, subject, "reference")
        _require_text(self.reason, subject, "reason")
        bounds = (self.common_opens_at, self.common_closes_at, self.utc_offset)
        if self.classification is EquityVenueDayClassification.NORMAL_SESSION:
            if not isinstance(self.common_opens_at, PointInTime) or not isinstance(
                self.common_closes_at, PointInTime
            ):
                _fail(f"{subject} NORMAL_SESSION needs both common boundaries.")
            if self.common_opens_at.compare(self.common_closes_at) >= 0:
                _fail(f"{subject} common boundary must open before it closes.")
            offset = self.utc_offset
            if (
                not isinstance(offset, timedelta)
                or offset % timedelta(minutes=1)
                or abs(offset) >= timedelta(days=1)
            ):
                _fail(f"{subject} NORMAL_SESSION needs a whole-minute utc_offset under a day.")
        elif any(value is not None for value in bounds):
            _fail(f"{subject} {self.classification} must not carry boundaries or an offset.")


@dataclass(frozen=True, slots=True)
class EquityListingApplicability:
    """Explicit evidence about one listing on one date, or the lack of it.

    ``evidence_reference`` names the source of the answer. For UNKNOWN it names
    the source that had no answer. Every other classification needs a reason.
    """

    listing: ListingReference
    trading_date: date
    classification: EquityListingApplicabilityClassification
    evidence_reference: str
    reason: str

    def __post_init__(self) -> None:
        subject = "EquityListingApplicability"
        if not isinstance(self.listing, ListingReference):
            _fail(f"{subject} listing must be a ListingReference.")
        _require_plain_date(self.trading_date, subject)
        if not isinstance(self.classification, EquityListingApplicabilityClassification):
            _fail(f"{subject} classification must be an EquityListingApplicabilityClassification.")
        _require_text(self.evidence_reference, subject, "evidence_reference")
        _require_text(self.reason, subject, "reason")


class EquityVenueDayReferenceSource(ABC):
    """Answers venue-day questions for a research window from accepted reference data.

    Implementations return an answer for every supported venue and date, and
    use a non-NORMAL_SESSION classification, never an exception, for a date they cannot
    support. They read no clock, provider or network.
    """

    @abstractmethod
    def venue_day(self, exchange_code: ExchangeCode, trading_date: date) -> EquityVenueDayReference:
        """Return what the reference states about ``trading_date`` on ``exchange_code``."""


class EquityListingApplicabilitySource(ABC):
    """Answers whether one listing on one date may use the normal research window.

    Implementations return UNKNOWN when they hold no explicit evidence. They
    never infer eligibility from a symbol existing, from market data, from
    derivatives membership or from the absence of an exception record.
    """

    @abstractmethod
    def applicability(
        self, listing: ListingReference, trading_date: date
    ) -> EquityListingApplicability:
        """Return the explicit applicability of ``listing`` on ``trading_date``."""
