"""Application ports for synthetic equity recommendation eligibility inputs (EI-4C).

EI-4C turns the first authoritative EI-3C research CANDIDATE into a synthetic,
non-authoritative recommendation proposal. Everything it needs beyond the
research timeline comes through the three ports below, under the owner
decisions recorded on 2026-11-07 in the EI-4 Decision Register (northstar-docs,
product/Equity-Intraday-V1-EI4-Decision-Register.md): EI4-D4 (versioned
synthetic inputs), EI4-D7 (synthetic eligibility references) and EI4-D8
(conservative open state).

1. ``EquitySyntheticRecommendationReferenceSource`` returns the versioned
   synthetic generation policy, risk policy (with model capital), cost
   schedule, price increment and entry-timing policy. The first four are the
   merged EI-4B Core types, so Core validates them; the timing policy is the
   Application value below, because Core defines no cutoff (EI0-D7).
2. ``EquitySyntheticListingEligibilitySource`` returns a versioned synthetic
   listing and universe eligibility reference.
3. ``EquitySessionRecommendationHistorySource`` returns the authoritative
   record of recommendations already issued in a venue session.

Shared rules
------------
- Every version begins ``synthetic-`` (EquitySyntheticVersion). No value is
  an exchange fact, broker tariff, production listing authority or approved
  trading parameter, and no port here may present one as such.
- None means the source holds no such reference: it is never a default, and
  the caller fails closed. A source that cannot answer reliably raises its
  unavailable error instead. "Not held" and "cannot answer" are different
  facts, and neither is ever read as eligible or empty.
- An implementation reads no clock, provider, broker or network. These
  contracts document obligations they cannot enforce at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime
from northstar_core.recommendations.equity import (
    EquityIssuanceKind,
    EquityPriceIncrement,
    EquitySyntheticCostSchedule,
    EquitySyntheticGenerationPolicy,
    EquitySyntheticRiskPolicy,
    EquitySyntheticVersion,
)
from northstar_core.strategy import StrategyIdentity

_FIVE_MINUTES = timedelta(minutes=5)


class InvalidEquityRecommendationReferenceError(ValueError):
    """Raised when a synthetic reference, eligibility or state value is malformed."""


class EquitySyntheticReferenceUnavailableError(RuntimeError):
    """Raised when a synthetic reference source cannot answer reliably."""


class EquityRecommendationHistoryUnavailableError(RuntimeError):
    """Raised when authoritative same-session recommendation state cannot be read."""


def _fail(message: str) -> None:
    raise InvalidEquityRecommendationReferenceError(message)


def _plain_date(value: object, subject: str) -> None:
    if isinstance(value, datetime) or not isinstance(value, date):
        _fail(f"{subject} trading_date must be a plain date.")


def _text(value: object, subject: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{subject} {name} must be non-empty text.")


def _version(value: object, subject: str, name: str) -> None:
    if not isinstance(value, EquitySyntheticVersion):
        _fail(f"{subject} {name} must be an EquitySyntheticVersion.")


# ---------------------------------------------------------------------------
# Entry timing policy (EI4-D1, EI0-D7 structure)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquitySyntheticEntryTimingPolicy:
    """Versioned synthetic entry timing for one venue's normal sessions (EI0-D7 structure).

    Times are naive local wall-clock times on five-minute boundaries, placed
    on a date with the venue reference's UTC offset, like the research window.

    - ``latest_signal_bar_end``: no candidate whose signal bar ends later is
      eligible.
    - ``latest_entry_bar_opening``: the latest permissible opening of a full
      modeled entry bar. Entry eligibility never extends beyond it.
    - ``modeled_exit_boundary``: the mandatory modeled exit boundary.
    - ``minimum_remaining_session``: the least time from the earliest
      permissible entry-bar opening to the modeled exit boundary.
    - ``research_window_policy_version``: the research window whose grid the
      entry bars come from; a candidate under another window is not covered.

    Only normal sessions are covered: a special or shortened session is
    refused, never re-timed. No value here is an approved production cutoff.
    """

    version: EquitySyntheticVersion
    exchange_code: ExchangeCode
    research_window_policy_version: str
    latest_signal_bar_end: time
    latest_entry_bar_opening: time
    modeled_exit_boundary: time
    minimum_remaining_session: timedelta

    def __post_init__(self) -> None:
        subject = "EquitySyntheticEntryTimingPolicy"
        _version(self.version, subject, "version")
        if not isinstance(self.exchange_code, ExchangeCode):
            _fail(f"{subject} exchange_code must be an ExchangeCode.")
        _text(self.research_window_policy_version, subject, "research_window_policy_version")
        for name in ("latest_signal_bar_end", "latest_entry_bar_opening", "modeled_exit_boundary"):
            value = getattr(self, name)
            if not isinstance(value, time) or value.tzinfo is not None:
                _fail(f"{subject} {name} must be a naive local wall-clock time.")
            if value.second or value.microsecond or value.minute % 5:
                _fail(f"{subject} {name} must lie on a five-minute boundary.")
        if not self.latest_signal_bar_end <= self.latest_entry_bar_opening:
            _fail(f"{subject} entry bars cannot open before the latest signal bar ends.")
        if not self.latest_entry_bar_opening < self.modeled_exit_boundary:
            _fail(f"{subject} the latest entry bar must open before the modeled exit boundary.")
        remaining = self.minimum_remaining_session
        if not isinstance(remaining, timedelta) or remaining < _FIVE_MINUTES:
            _fail(f"{subject} minimum_remaining_session must be at least one five-minute bar.")
        if remaining % timedelta(minutes=1):
            _fail(f"{subject} minimum_remaining_session must be whole minutes.")

    @property
    def is_synthetic(self) -> bool:
        """Return True: no value here is an approved cutoff (EI0-D7)."""
        return True

    def document(self) -> dict[str, object]:
        """Return the canonical content."""
        return {
            "version": self.version.value,
            "exchange_code": self.exchange_code.value,
            "research_window_policy_version": self.research_window_policy_version,
            "latest_signal_bar_end": self.latest_signal_bar_end.isoformat(),
            "latest_entry_bar_opening": self.latest_entry_bar_opening.isoformat(),
            "modeled_exit_boundary": self.modeled_exit_boundary.isoformat(),
            "minimum_remaining_session_minutes": self.minimum_remaining_session
            // timedelta(minutes=1),
        }


# ---------------------------------------------------------------------------
# Synthetic listing eligibility (EI4-D7)
# ---------------------------------------------------------------------------


class EquitySyntheticEligibilityCriterion(StrEnum):
    """A listing-eligibility criterion every synthetic reference must answer."""

    UNIVERSE_MEMBERSHIP = "UNIVERSE_MEMBERSHIP"
    LISTING_STATUS = "LISTING_STATUS"
    SERIES = "SERIES"
    PRICE = "PRICE"
    LIQUIDITY = "LIQUIDITY"


class EquitySyntheticEligibilityVerdict(StrEnum):
    """What a synthetic reference states for one criterion. UNKNOWN is never eligible."""

    ELIGIBLE = "ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class EquitySyntheticEligibilityFinding:
    """One criterion's synthetic verdict, with what it rests on."""

    criterion: EquitySyntheticEligibilityCriterion
    verdict: EquitySyntheticEligibilityVerdict
    basis: str

    def __post_init__(self) -> None:
        subject = "EquitySyntheticEligibilityFinding"
        if not isinstance(self.criterion, EquitySyntheticEligibilityCriterion):
            _fail(f"{subject} criterion must be an EquitySyntheticEligibilityCriterion.")
        if not isinstance(self.verdict, EquitySyntheticEligibilityVerdict):
            _fail(f"{subject} verdict must be an EquitySyntheticEligibilityVerdict.")
        _text(self.basis, subject, "basis")

    def document(self) -> dict[str, object]:
        """Return the canonical content."""
        return {
            "criterion": self.criterion.value,
            "verdict": self.verdict.value,
            "basis": self.basis,
        }


@dataclass(frozen=True, slots=True)
class EquitySyntheticListingEligibility:
    """A versioned synthetic eligibility reference for one listing on one date.

    ``findings`` answer every EquitySyntheticEligibilityCriterion exactly once,
    in declaration order. ``provenance`` names the synthetic source. Thresholds
    behind PRICE and LIQUIDITY belong to the reference version and are
    synthetic; none is an approved real threshold (EI0-D15). Corporate actions
    and exchange surveillance are not criteria: no source exists (EI0-D17,
    EI0-D18), and they are never inferred.
    """

    version: EquitySyntheticVersion
    universe_version: EquitySyntheticVersion
    listing: ListingReference
    trading_date: date
    findings: tuple[EquitySyntheticEligibilityFinding, ...]
    provenance: str

    def __post_init__(self) -> None:
        subject = "EquitySyntheticListingEligibility"
        _version(self.version, subject, "version")
        _version(self.universe_version, subject, "universe_version")
        if not isinstance(self.listing, ListingReference):
            _fail(f"{subject} listing must be a ListingReference.")
        _plain_date(self.trading_date, subject)
        if not isinstance(self.findings, tuple) or not all(
            isinstance(f, EquitySyntheticEligibilityFinding) for f in self.findings
        ):
            _fail(f"{subject} findings must be a tuple of EquitySyntheticEligibilityFinding.")
        if tuple(f.criterion for f in self.findings) != tuple(EquitySyntheticEligibilityCriterion):
            _fail(f"{subject} must answer every criterion exactly once, in declaration order.")
        _text(self.provenance, subject, "provenance")

    @property
    def is_production_authority(self) -> bool:
        """Return False: synthetic eligibility is never production authority (EI4-D7)."""
        return False

    def with_verdict(
        self, verdict: EquitySyntheticEligibilityVerdict
    ) -> tuple[EquitySyntheticEligibilityCriterion, ...]:
        """Return the criteria with ``verdict``, in declaration order."""
        return tuple(f.criterion for f in self.findings if f.verdict is verdict)

    def document(self) -> dict[str, object]:
        """Return the canonical content."""
        return {
            "version": self.version.value,
            "universe_version": self.universe_version.value,
            "listing": {
                "symbol": self.listing.symbol.value,
                "exchange_code": self.listing.exchange_code.value,
            },
            "trading_date": self.trading_date.isoformat(),
            "findings": [f.document() for f in self.findings],
            "provenance": self.provenance,
        }


# ---------------------------------------------------------------------------
# Same-session recommendation state (EI4-D8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquityPriorSessionRecommendation:
    """One recommendation the authoritative record states was issued in the session.

    Only an authoritatively recorded issuance -- actual, or an explicitly
    modeled issuance that was recorded -- is such a record. A transient EI-4C
    proposal is never one. ``issued_at`` is the recorded issuance instant.
    It carries no outcome: until EI-5 provides lifecycle state, a record is
    treated as open from ``issued_at`` (EI4-D8), whatever its levels might
    have touched.
    """

    recommendation_key: str
    strategy: StrategyIdentity
    listing: ListingReference
    trading_date: date
    issuance_kind: EquityIssuanceKind
    issued_at: PointInTime

    def __post_init__(self) -> None:
        subject = "EquityPriorSessionRecommendation"
        _text(self.recommendation_key, subject, "recommendation_key")
        if not isinstance(self.strategy, StrategyIdentity):
            _fail(f"{subject} strategy must be a StrategyIdentity.")
        if not isinstance(self.listing, ListingReference):
            _fail(f"{subject} listing must be a ListingReference.")
        _plain_date(self.trading_date, subject)
        if not isinstance(self.issuance_kind, EquityIssuanceKind):
            _fail(f"{subject} issuance_kind must be an EquityIssuanceKind.")
        if not isinstance(self.issued_at, PointInTime):
            _fail(f"{subject} issued_at must be a PointInTime.")

    def document(self) -> dict[str, object]:
        """Return the canonical content."""
        return {
            "recommendation_key": self.recommendation_key,
            "strategy": self.strategy.identity,
            "listing": {
                "symbol": self.listing.symbol.value,
                "exchange_code": self.listing.exchange_code.value,
            },
            "trading_date": self.trading_date.isoformat(),
            "issuance_kind": self.issuance_kind.value,
            "issued_at": self.issued_at.value,
        }


def _issued_order(record: EquityPriorSessionRecommendation) -> tuple[datetime, str]:
    # Instants are compared as times, never as text; the key breaks ties.
    instant = datetime.fromisoformat(record.issued_at.value.replace("Z", "+00:00"))
    return (instant, record.recommendation_key)


@dataclass(frozen=True, slots=True)
class EquitySessionRecommendationHistory:
    """The authoritative record of recommendations issued in one venue session.

    ``complete_through`` is the instant through which the record is complete:
    every recommendation issued in the session at or before it is listed, and
    none is listed after it. Records are in ascending (issued_at, key) order,
    so records sharing an instant have one deterministic order.

    ``issued_by(instant)`` is the point-in-time view: the records issued at or
    before ``instant``, which needs ``complete_through`` at or after it. A
    record issued later never belongs to an earlier view.

    An empty ``recommendations`` tuple is an explicitly established empty
    session through ``complete_through``, distinct from a source that holds no
    state (None) or cannot answer (EquityRecommendationHistoryUnavailableError).
    ``reference`` names the authoritative record and its version.
    """

    exchange_code: ExchangeCode
    trading_date: date
    recommendations: tuple[EquityPriorSessionRecommendation, ...]
    complete_through: PointInTime
    reference: str

    def __post_init__(self) -> None:
        subject = "EquitySessionRecommendationHistory"
        if not isinstance(self.exchange_code, ExchangeCode):
            _fail(f"{subject} exchange_code must be an ExchangeCode.")
        _plain_date(self.trading_date, subject)
        if not isinstance(self.recommendations, tuple) or not all(
            isinstance(r, EquityPriorSessionRecommendation) for r in self.recommendations
        ):
            _fail(f"{subject} recommendations must be a tuple of EquityPriorSessionRecommendation.")
        if not isinstance(self.complete_through, PointInTime):
            _fail(f"{subject} complete_through must be a PointInTime.")
        for item in self.recommendations:
            if item.trading_date != self.trading_date or (
                item.listing.exchange_code != self.exchange_code
            ):
                _fail(f"{subject} holds a recommendation from another session.")
            if item.issued_at.compare(self.complete_through) > 0:
                _fail(f"{subject} lists a recommendation issued after it is complete.")
        keys = [r.recommendation_key for r in self.recommendations]
        if len(set(keys)) != len(keys):
            _fail(f"{subject} recommendation keys must be distinct.")
        if list(self.recommendations) != sorted(self.recommendations, key=_issued_order):
            _fail(f"{subject} recommendations must be in ascending (issued_at, key) order.")
        _text(self.reference, subject, "reference")

    def covers(self, instant: PointInTime) -> bool:
        """Return whether the record is complete through ``instant``."""
        return self.complete_through.compare(instant) >= 0

    def issued_by(self, instant: PointInTime) -> tuple[EquityPriorSessionRecommendation, ...]:
        """Return the records issued at or before ``instant``, which the record must cover."""
        if not self.covers(instant):
            raise InvalidEquityRecommendationReferenceError(
                "The recommendation history is not complete through the requested instant."
            )
        return tuple(r for r in self.recommendations if r.issued_at.compare(instant) <= 0)


# ---------------------------------------------------------------------------
# Ports
# ---------------------------------------------------------------------------


class EquitySyntheticRecommendationReferenceSource(ABC):
    """Supplies versioned synthetic recommendation inputs (EI4-D4).

    Each method returns the input stored under exactly ``version``, or None
    when the source holds none. It raises EquitySyntheticReferenceUnavailableError
    when it cannot answer reliably. It never substitutes another version.
    """

    @abstractmethod
    def generation_policy(
        self, version: EquitySyntheticVersion
    ) -> EquitySyntheticGenerationPolicy | None:
        """Return the synthetic generation policy ``version``."""

    @abstractmethod
    def risk_policy(self, version: EquitySyntheticVersion) -> EquitySyntheticRiskPolicy | None:
        """Return the synthetic risk policy ``version``, with its model capital."""

    @abstractmethod
    def cost_schedule(self, version: EquitySyntheticVersion) -> EquitySyntheticCostSchedule | None:
        """Return the synthetic cost schedule ``version``."""

    @abstractmethod
    def price_increment(
        self, version: EquitySyntheticVersion, listing: ListingReference, trading_date: date
    ) -> EquityPriceIncrement | None:
        """Return the synthetic increment ``version`` for ``listing`` on ``trading_date``.

        None when the reference states no increment for that listing and date.
        """

    @abstractmethod
    def timing_policy(
        self, version: EquitySyntheticVersion
    ) -> EquitySyntheticEntryTimingPolicy | None:
        """Return the synthetic entry-timing policy ``version``."""


class EquitySyntheticListingEligibilitySource(ABC):
    """Supplies versioned synthetic listing eligibility (EI4-D7).

    ``eligibility`` returns the reference ``version``'s answer for exactly
    ``listing`` and ``trading_date``, or None when it holds none, and raises
    EquitySyntheticReferenceUnavailableError when it cannot answer reliably.
    A criterion it cannot decide is UNKNOWN, never ELIGIBLE.
    """

    @abstractmethod
    def eligibility(
        self, version: EquitySyntheticVersion, listing: ListingReference, trading_date: date
    ) -> EquitySyntheticListingEligibility | None:
        """Return the synthetic eligibility of ``listing`` on ``trading_date``."""


class EquitySessionRecommendationHistorySource(ABC):
    """Supplies the authoritative record of a session's issued recommendations (EI4-D8).

    ``session_history`` returns the session's record across listings and
    strategies, complete at least through ``as_of``, or None when no
    authoritative record is established for it. It raises
    EquityRecommendationHistoryUnavailableError when the record cannot be read
    or verified. It never returns an assumed-empty session, and it never
    states completeness it has not established: a record whose
    ``complete_through`` is before ``as_of`` is refused by the caller as
    incomplete.
    """

    @abstractmethod
    def session_history(
        self, exchange_code: ExchangeCode, trading_date: date, as_of: PointInTime
    ) -> EquitySessionRecommendationHistory | None:
        """Return the session's authoritative recommendation record as of ``as_of``."""
