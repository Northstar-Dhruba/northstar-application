"""Synthetic equity recommendation eligibility contracts and proposal identity (EI-4C).

The request, outcome vocabulary, proposal and eligibility audit of
GenerateEquityTradeRecommendationUseCase (generate_equity_trade_recommendation),
under the owner decisions recorded on 2026-11-07 in the EI-4 Decision Register
(northstar-docs, product/Equity-Intraday-V1-EI4-Decision-Register.md) and in
ADR-020. Everything here is synthetic and non-authoritative: nothing is issued,
persisted, actionable, an order, a fill or a position.

Outcomes
--------
Every reason has one fixed category, and an attempt's status is derived from
the reasons it recorded, in this precedence:

1. RESEARCH_INTEGRITY_FAILURE -- authoritative research history or its
   evidence failed verification. It is never reported as NO_SIGNAL or as an
   ordinary rejection.
2. DEPENDENCY_UNAVAILABLE -- a required input is missing, unknown,
   inconsistent or its source cannot answer. Generation fails closed.
3. NO_CANDIDATE -- the verified history holds no CANDIDATE (EI4-D2).
4. NOT_ELIGIBLE -- the candidate was evaluated and a confirmed check refused
   it, Core formation rejections included, with Core's exact reasons kept.
5. EVALUATION_UNRESOLVED -- no confirmed check refused it, but a required
   result could not be established: an open recommendation whose effect
   depends on an exclusivity or exposure policy that is not approved, or
   Core's exact sizing search reaching its bound. It fails closed, and it is
   never presented as a violation, an ineligible setup or an outage.
6. PROPOSED -- every implemented check passed and Core formed a definition.

A confirmed refusal outranks an unresolved one: if the candidate fails a
confirmed check, no policy decision could make it eligible.

Open recommendations (EI4-D8)
-----------------------------
Recommendations are read as of the modeled issuance instant from an
authoritative record complete through it: those issued at or before it are
open, whatever their levels might have touched; those issued later never
change the result. EI4-D8 approves counting them as open. It approves no
exclusivity rule and no exposure limit: one open recommendation per listing
(product specification, section 9.5) and any simultaneous-open or aggregate
exposure limit (EI0-D11) are not approved. So an open recommendation on the
same listing is EXCLUSIVITY_POLICY_UNRESOLVED, one on another listing is
EXPOSURE_POLICY_UNRESOLVED, and neither is reported as a breach. With none
open the check passes, while SAME_LISTING_EXCLUSIVITY and PORTFOLIO_EXPOSURE
stay NOT_ASSESSED and block actionable issuance.

Reasons are reported distinct and in declaration order. Checks are reported
in declaration order with one status each: PASSED, FAILED, NOT_EVALUATED
(an earlier step failed) or NOT_ASSESSED (the control does not exist yet). A
NOT_ASSESSED control is never PASSED, and every proposal names it as blocking
actionable issuance (EI-4 Decision Register, section 4).

Proposal key
------------
A provisional key, derived the way Futures recommendation identities are: a
length-prefixed SHA-256 over, in this order,

    northstar.equity-trade-recommendation-proposal.v1
    strategy identity
    research configuration version
    symbol, exchange code
    trading date                       (YYYY-MM-DD)
    first CANDIDATE entry fingerprint  (eirt1)
    generation-policy version

Each part is its UTF-8 byte length, a colon, then the bytes, so no value can
make two keys collide. The candidate entry fingerprint binds the timeline's
full configuration, the decision and its evidence snapshot. Issuance and
eligibility instants, levels, other policy versions, clock readings, random
values and row ids are excluded, so one candidate under one generation-policy
version always reaches one key (EI4-D2: zero or one recommendation per
version), and the other versions are content: an equal key with different
content is a contradiction a store must refuse. The key identifies; it
enforces nothing and proves no issuance. Equal keys never prove equal
proposals: two proposals under one key may differ in risk, cost, increment,
timing or eligibility inputs, and in modeled instants. A future store must
therefore compare the full canonical proposal document under a key and fail
closed -- record nothing and report the conflict -- when it differs, never
keeping either silently. Which inputs the authoritative identity binds, and
how recording is sequenced, is EI4-D9, which is still pending; this key is
provisional until then.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import PointInTime
from northstar_core.recommendations.equity import (
    EquityIssuanceKind,
    EquityPriceIncrement,
    EquityRecommendationDefinition,
    EquityRecommendationRejection,
    EquityRecommendationRejectionReason,
    EquityRecommendationTiming,
    EquityResearchCandidateReference,
    EquitySyntheticCostSchedule,
    EquitySyntheticGenerationPolicy,
    EquitySyntheticRiskPolicy,
    EquitySyntheticVersion,
)

from northstar_application.application_services.equity_candle_evidence_journal import (
    canonical_equity_candle_json,
)
from northstar_application.application_services.evaluate_equity_intraday_research import (
    EquityOpeningRangeBreakoutConfiguration,
)
from northstar_application.ports.equity_recommendation_references import (
    EquityPriorSessionRecommendation,
    EquitySyntheticEntryTimingPolicy,
    EquitySyntheticListingEligibility,
)

EQUITY_TRADE_RECOMMENDATION_PROPOSAL_NAMESPACE = "northstar.equity-trade-recommendation-proposal.v1"
_AUDIT_FORMAT = "northstar.equity-recommendation-eligibility-audit/1"


class InvalidEquityTradeRecommendationRequestError(ValueError):
    """Raised when a generation request is malformed."""


class InvalidEquityRecommendationEligibilityError(ValueError):
    """Raised when a proposal, check or audit is internally inconsistent."""


def _invalid(message: str) -> None:
    raise InvalidEquityRecommendationEligibilityError(message)


def _listing(listing: ListingReference) -> dict[str, str]:
    return {"symbol": listing.symbol.value, "exchange_code": listing.exchange_code.value}


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquityTradeRecommendationRequest:
    """What one generation attempt is asked to evaluate. Nothing is inferred or defaulted.

    ``configuration`` names the strategy and its full synthetic research
    configuration; every reference is named by its exact synthetic version.
    ``modeled_issued_at`` is an explicitly modeled issuance instant for
    synthetic analysis. It is never an actual issuance, and it must not
    precede the research decision.
    """

    configuration: EquityOpeningRangeBreakoutConfiguration
    listing: ListingReference
    trading_date: date
    generation_policy_version: EquitySyntheticVersion
    risk_policy_version: EquitySyntheticVersion
    cost_schedule_version: EquitySyntheticVersion
    price_increment_version: EquitySyntheticVersion
    timing_policy_version: EquitySyntheticVersion
    listing_eligibility_version: EquitySyntheticVersion
    modeled_issued_at: PointInTime

    def __post_init__(self) -> None:
        error = InvalidEquityTradeRecommendationRequestError
        if not isinstance(self.configuration, EquityOpeningRangeBreakoutConfiguration):
            raise error("configuration must be an EquityOpeningRangeBreakoutConfiguration.")
        if not isinstance(self.listing, ListingReference):
            raise error("listing must be a ListingReference.")
        if isinstance(self.trading_date, datetime) or not isinstance(self.trading_date, date):
            raise error("trading_date must be a plain date.")
        for name in _VERSION_FIELDS:
            if not isinstance(getattr(self, name), EquitySyntheticVersion):
                raise error(f"{name} must be an EquitySyntheticVersion.")
        if not isinstance(self.modeled_issued_at, PointInTime):
            raise error("modeled_issued_at must be a PointInTime.")

    def document(self) -> dict[str, Any]:
        """Return the canonical content."""
        return {
            "strategy": self.configuration.strategy.identity,
            "research_configuration": self.configuration.document(),
            "listing": _listing(self.listing),
            "trading_date": self.trading_date.isoformat(),
            **{name: getattr(self, name).value for name in _VERSION_FIELDS},
            "modeled_issued_at": self.modeled_issued_at.value,
        }


_VERSION_FIELDS = (
    "generation_policy_version",
    "risk_policy_version",
    "cost_schedule_version",
    "price_increment_version",
    "timing_policy_version",
    "listing_eligibility_version",
)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class EquityRecommendationGenerationStatus(StrEnum):
    """How one attempt ended. Only PROPOSED carries a proposal, and it is not issued."""

    PROPOSED = "PROPOSED"
    NO_CANDIDATE = "NO_CANDIDATE"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    EVALUATION_UNRESOLVED = "EVALUATION_UNRESOLVED"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"
    RESEARCH_INTEGRITY_FAILURE = "RESEARCH_INTEGRITY_FAILURE"


class EquityRecommendationEligibilityReason(StrEnum):
    """Why an attempt produced no proposal, in report order."""

    # Authoritative research state (EI3-D11 to EI3-D13, EI4-D2)
    RESEARCH_TIMELINE_UNAVAILABLE = "RESEARCH_TIMELINE_UNAVAILABLE"
    EVIDENCE_SNAPSHOT_UNAVAILABLE = "EVIDENCE_SNAPSHOT_UNAVAILABLE"
    RESEARCH_TIMELINE_INVALID = "RESEARCH_TIMELINE_INVALID"
    RESEARCH_DECISION_NOT_REPRODUCED = "RESEARCH_DECISION_NOT_REPRODUCED"
    HISTORICAL_EVALUATION_UNAVAILABLE = "HISTORICAL_EVALUATION_UNAVAILABLE"
    CANDIDATE_EVIDENCE_INVALID = "CANDIDATE_EVIDENCE_INVALID"
    RESEARCH_CONFIGURATION_MISMATCH = "RESEARCH_CONFIGURATION_MISMATCH"
    NO_AUTHORITATIVE_CANDIDATE = "NO_AUTHORITATIVE_CANDIDATE"
    CANDIDATE_EVIDENCE_NOT_SYNTHETIC = "CANDIDATE_EVIDENCE_NOT_SYNTHETIC"
    # Synthetic references (EI4-D4, EI4-D7, EI4-D8)
    TIMING_POLICY_UNAVAILABLE = "TIMING_POLICY_UNAVAILABLE"
    GENERATION_POLICY_UNAVAILABLE = "GENERATION_POLICY_UNAVAILABLE"
    RISK_POLICY_UNAVAILABLE = "RISK_POLICY_UNAVAILABLE"
    COST_SCHEDULE_UNAVAILABLE = "COST_SCHEDULE_UNAVAILABLE"
    PRICE_INCREMENT_UNAVAILABLE = "PRICE_INCREMENT_UNAVAILABLE"
    POLICY_REFERENCE_MISMATCH = "POLICY_REFERENCE_MISMATCH"
    SESSION_REFERENCE_UNAVAILABLE = "SESSION_REFERENCE_UNAVAILABLE"
    SESSION_REFERENCE_MISMATCH = "SESSION_REFERENCE_MISMATCH"
    LISTING_REFERENCE_UNAVAILABLE = "LISTING_REFERENCE_UNAVAILABLE"
    LISTING_REFERENCE_MISMATCH = "LISTING_REFERENCE_MISMATCH"
    LISTING_ELIGIBILITY_UNKNOWN = "LISTING_ELIGIBILITY_UNKNOWN"
    OPEN_RECOMMENDATION_STATE_UNAVAILABLE = "OPEN_RECOMMENDATION_STATE_UNAVAILABLE"
    RECOMMENDATION_HISTORY_INCOMPLETE = "RECOMMENDATION_HISTORY_INCOMPLETE"
    # Timing (EI4-D1, EI0-D7 structure)
    SESSION_NOT_SUPPORTED = "SESSION_NOT_SUPPORTED"
    TIMING_POLICY_NOT_APPLICABLE = "TIMING_POLICY_NOT_APPLICABLE"
    MODELED_ISSUANCE_BEFORE_DECISION = "MODELED_ISSUANCE_BEFORE_DECISION"
    AFTER_SIGNAL_CUTOFF = "AFTER_SIGNAL_CUTOFF"
    POST_WINDOW_DECISION = "POST_WINDOW_DECISION"
    NO_PERMISSIBLE_ENTRY_BAR = "NO_PERMISSIBLE_ENTRY_BAR"
    INSUFFICIENT_SESSION_TIME = "INSUFFICIENT_SESSION_TIME"
    # Listing eligibility (EI4-D7)
    LISTING_NOT_ELIGIBLE = "LISTING_NOT_ELIGIBLE"
    # Open recommendations (EI4-D8): the policy that would decide is not approved
    EXCLUSIVITY_POLICY_UNRESOLVED = "EXCLUSIVITY_POLICY_UNRESOLVED"
    EXPOSURE_POLICY_UNRESOLVED = "EXPOSURE_POLICY_UNRESOLVED"
    # Core formation (EI-4B); Core's own reasons are kept with it
    CORE_FORMATION_REJECTED = "CORE_FORMATION_REJECTED"
    CORE_FORMATION_UNRESOLVED = "CORE_FORMATION_UNRESOLVED"


_R = EquityRecommendationEligibilityReason
_STATUS = EquityRecommendationGenerationStatus
_INTEGRITY = frozenset(
    {
        _R.RESEARCH_TIMELINE_INVALID,
        _R.RESEARCH_DECISION_NOT_REPRODUCED,
        _R.HISTORICAL_EVALUATION_UNAVAILABLE,
        _R.CANDIDATE_EVIDENCE_INVALID,
    }
)
_DEPENDENCY = frozenset(
    {
        _R.RESEARCH_TIMELINE_UNAVAILABLE,
        _R.EVIDENCE_SNAPSHOT_UNAVAILABLE,
        _R.TIMING_POLICY_UNAVAILABLE,
        _R.GENERATION_POLICY_UNAVAILABLE,
        _R.RISK_POLICY_UNAVAILABLE,
        _R.COST_SCHEDULE_UNAVAILABLE,
        _R.PRICE_INCREMENT_UNAVAILABLE,
        _R.POLICY_REFERENCE_MISMATCH,
        _R.SESSION_REFERENCE_UNAVAILABLE,
        _R.SESSION_REFERENCE_MISMATCH,
        _R.LISTING_REFERENCE_UNAVAILABLE,
        _R.LISTING_REFERENCE_MISMATCH,
        _R.LISTING_ELIGIBILITY_UNKNOWN,
        _R.OPEN_RECOMMENDATION_STATE_UNAVAILABLE,
        _R.RECOMMENDATION_HISTORY_INCOMPLETE,
    }
)
_UNRESOLVED = frozenset(
    {
        _R.EXCLUSIVITY_POLICY_UNRESOLVED,
        _R.EXPOSURE_POLICY_UNRESOLVED,
        _R.CORE_FORMATION_UNRESOLVED,
    }
)
# Core reasons that leave a result undetermined rather than refused.
EQUITY_CORE_UNRESOLVED_REASONS = frozenset(
    {EquityRecommendationRejectionReason.SIZING_SEARCH_LIMIT_EXCEEDED}
)


def equity_recommendation_status(
    reasons: tuple[EquityRecommendationEligibilityReason, ...],
) -> EquityRecommendationGenerationStatus:
    """Return the status the reasons establish, in the documented precedence."""
    found = set(reasons)
    if found & _INTEGRITY:
        return _STATUS.RESEARCH_INTEGRITY_FAILURE
    if found & _DEPENDENCY:
        return _STATUS.DEPENDENCY_UNAVAILABLE
    if _R.NO_AUTHORITATIVE_CANDIDATE in found:
        return _STATUS.NO_CANDIDATE
    if found - _UNRESOLVED:
        return _STATUS.NOT_ELIGIBLE
    return _STATUS.EVALUATION_UNRESOLVED if found else _STATUS.PROPOSED


class EquityRecommendationCheck(StrEnum):
    """A control an attempt reports, in evaluation order. The last seven do not exist yet."""

    AUTHORITATIVE_CANDIDATE = "AUTHORITATIVE_CANDIDATE"
    SYNTHETIC_REFERENCES = "SYNTHETIC_REFERENCES"
    SESSION_TIMING = "SESSION_TIMING"
    LISTING_ELIGIBILITY = "LISTING_ELIGIBILITY"
    OPEN_RECOMMENDATIONS = "OPEN_RECOMMENDATIONS"
    CORE_FORMATION = "CORE_FORMATION"
    SAME_LISTING_EXCLUSIVITY = "SAME_LISTING_EXCLUSIVITY"
    PORTFOLIO_EXPOSURE = "PORTFOLIO_EXPOSURE"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    PRE_ENTRY_INVALIDATION = "PRE_ENTRY_INVALIDATION"
    CORPORATE_ACTIONS = "CORPORATE_ACTIONS"
    EXCHANGE_SURVEILLANCE = "EXCHANGE_SURVEILLANCE"
    CANDLE_REVISION_MATERIALITY = "CANDLE_REVISION_MATERIALITY"


class EquityRecommendationCheckStatus(StrEnum):
    """One control's result. NOT_ASSESSED is never a pass."""

    PASSED = "PASSED"
    FAILED = "FAILED"
    NOT_EVALUATED = "NOT_EVALUATED"
    NOT_ASSESSED = "NOT_ASSESSED"


# The controls with no approved policy or input yet, and the decision each awaits.
EQUITY_RECOMMENDATION_UNASSESSED_CONTROLS: tuple[tuple[EquityRecommendationCheck, str], ...] = (
    (
        EquityRecommendationCheck.SAME_LISTING_EXCLUSIVITY,
        "No exclusivity rule is approved: one open recommendation per listing (product "
        "specification, section 9.5) is proposed only. EI4-D8 approves only counting earlier "
        "recommendations as open.",
    ),
    (
        EquityRecommendationCheck.PORTFOLIO_EXPOSURE,
        "No aggregate exposure or simultaneous-open limit is approved (EI0-D11); open "
        "recommendations are counted, never assumed closed (EI4-D8).",
    ),
    (
        EquityRecommendationCheck.DAILY_LOSS_LIMIT,
        "The daily modeled loss limit is pending (EI0-D19) and needs EI-5 outcomes.",
    ),
    (
        EquityRecommendationCheck.PRE_ENTRY_INVALIDATION,
        "Pre-entry invalidation is deferred (EI4-D5).",
    ),
    (
        EquityRecommendationCheck.CORPORATE_ACTIONS,
        "No corporate-action source exists (EI0-D17); none is inferred.",
    ),
    (
        EquityRecommendationCheck.EXCHANGE_SURVEILLANCE,
        "No surveillance source exists (EI0-D18); none is inferred.",
    ),
    (
        EquityRecommendationCheck.CANDLE_REVISION_MATERIALITY,
        "Candle-revision materiality is pending (EI0-D21); corrections never rewrite the "
        "candidate.",
    ),
)
_UNASSESSED = tuple(check for check, _ in EQUITY_RECOMMENDATION_UNASSESSED_CONTROLS)


@dataclass(frozen=True, slots=True)
class EquityRecommendationCheckResult:
    """One control's status, its reasons in report order and a deterministic detail."""

    check: EquityRecommendationCheck
    status: EquityRecommendationCheckStatus
    reasons: tuple[EquityRecommendationEligibilityReason, ...]
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.check, EquityRecommendationCheck):
            _invalid("check must be an EquityRecommendationCheck.")
        if not isinstance(self.status, EquityRecommendationCheckStatus):
            _invalid("status must be an EquityRecommendationCheckStatus.")
        reasons = self.reasons
        if not isinstance(reasons, tuple) or not all(
            isinstance(r, EquityRecommendationEligibilityReason) for r in reasons
        ):
            _invalid("reasons must be a tuple of EquityRecommendationEligibilityReason.")
        if reasons != ordered_equity_recommendation_reasons(reasons):
            _invalid("reasons must be distinct and in declaration order.")
        if (self.status is EquityRecommendationCheckStatus.FAILED) != bool(reasons):
            _invalid("Exactly a FAILED check has reasons.")
        if (self.status is EquityRecommendationCheckStatus.NOT_ASSESSED) != (
            self.check in _UNASSESSED
        ):
            _invalid("Exactly the controls that do not exist yet are NOT_ASSESSED.")
        if not isinstance(self.detail, str) or not self.detail:
            _invalid("detail must be non-empty text.")

    def document(self) -> dict[str, Any]:
        """Return the canonical content."""
        return {
            "check": self.check.value,
            "status": self.status.value,
            "reasons": [reason.value for reason in self.reasons],
            "detail": self.detail,
        }


def ordered_equity_recommendation_reasons(
    reasons: tuple[EquityRecommendationEligibilityReason, ...] | set[Any],
) -> tuple[EquityRecommendationEligibilityReason, ...]:
    """Return ``reasons`` distinct and in declaration order."""
    found = set(reasons)
    return tuple(reason for reason in EquityRecommendationEligibilityReason if reason in found)


# ---------------------------------------------------------------------------
# Proposal key
# ---------------------------------------------------------------------------


def _length_prefixed(parts: tuple[str, ...]) -> bytes:
    encoded = bytearray()
    for part in parts:
        raw = part.encode("utf-8")
        encoded += str(len(raw)).encode("ascii") + b":" + raw
    return bytes(encoded)


def equity_trade_recommendation_proposal_key(
    source: EquityResearchCandidateReference, generation_policy_version: EquitySyntheticVersion
) -> str:
    """Return the provisional key of ``source``'s proposal under ``generation_policy_version``."""
    if not isinstance(source, EquityResearchCandidateReference):
        raise TypeError("source must be an EquityResearchCandidateReference.")
    if not isinstance(generation_policy_version, EquitySyntheticVersion):
        raise TypeError("generation_policy_version must be an EquitySyntheticVersion.")
    parts = (
        EQUITY_TRADE_RECOMMENDATION_PROPOSAL_NAMESPACE,
        source.strategy.identity,
        source.research_configuration_version.value,
        source.listing.symbol.value,
        source.listing.exchange_code.value,
        source.trading_date.isoformat(),
        source.candidate_entry_fingerprint,
        generation_policy_version.value,
    )
    return hashlib.sha256(_length_prefixed(parts)).hexdigest()


# ---------------------------------------------------------------------------
# Candidate, entry window and proposal
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquityRecommendationCandidateProvenance:
    """The authoritative candidate an attempt evaluated, and what verified it.

    ``source`` is the Core reference; the rest pins the evidence and history it
    was read from: the snapshot's manifest fingerprint, the candidate's
    position and the verified timeline head at evaluation.
    """

    source: EquityResearchCandidateReference
    manifest_fingerprint: str
    candidate_sequence: int
    timeline_head_fingerprint: str
    signal_interval_end: PointInTime

    def __post_init__(self) -> None:
        if not isinstance(self.source, EquityResearchCandidateReference):
            _invalid("source must be an EquityResearchCandidateReference.")
        for name in ("manifest_fingerprint", "timeline_head_fingerprint"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                _invalid(f"{name} must be non-empty text.")
        sequence = self.candidate_sequence
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            _invalid("candidate_sequence must be an integer of at least 1.")
        if not isinstance(self.signal_interval_end, PointInTime):
            _invalid("signal_interval_end must be a PointInTime.")

    def document(self) -> dict[str, Any]:
        """Return the canonical content."""
        return {
            "source": self.source.document(),
            "manifest_fingerprint": self.manifest_fingerprint,
            "candidate_sequence": self.candidate_sequence,
            "timeline_head_fingerprint": self.timeline_head_fingerprint,
            "signal_interval_end": self.signal_interval_end.value,
        }


@dataclass(frozen=True, slots=True)
class EquityModeledEntryWindow:
    """The modeled instants of a proposal and the bounds of its entry eligibility.

    ``timing`` is Core's EI4-D1 timing with a MODELED issuance; its
    ``eligible_from`` is the earliest full entry bar opening at or after the
    modeled issuance. No bar opening after ``latest_entry_bar_opens_at`` may
    evidence an entry, and ``modeled_exit_at`` is the mandatory modeled exit
    boundary. Nothing here scans bars or touches levels.
    """

    timing: EquityRecommendationTiming
    latest_entry_bar_opens_at: PointInTime
    modeled_exit_at: PointInTime
    timing_policy_version: EquitySyntheticVersion

    def __post_init__(self) -> None:
        if not isinstance(self.timing, EquityRecommendationTiming):
            _invalid("timing must be an EquityRecommendationTiming.")
        if self.timing.issuance_kind is not EquityIssuanceKind.MODELED:
            _invalid("EI-4C models issuance only; it never records an actual issuance.")
        for name in ("latest_entry_bar_opens_at", "modeled_exit_at"):
            if not isinstance(getattr(self, name), PointInTime):
                _invalid(f"{name} must be a PointInTime.")
        if self.timing.eligible_from.compare(self.latest_entry_bar_opens_at) > 0:
            _invalid("Entry eligibility cannot begin after the latest entry bar opening.")
        if self.latest_entry_bar_opens_at.compare(self.modeled_exit_at) >= 0:
            _invalid("The latest entry bar must open before the modeled exit boundary.")
        if not isinstance(self.timing_policy_version, EquitySyntheticVersion):
            _invalid("timing_policy_version must be an EquitySyntheticVersion.")

    def document(self) -> dict[str, Any]:
        """Return the canonical content."""
        return {
            "timing": self.timing.document(),
            "latest_entry_bar_opens_at": self.latest_entry_bar_opens_at.value,
            "modeled_exit_at": self.modeled_exit_at.value,
            "timing_policy_version": self.timing_policy_version.value,
        }


@dataclass(frozen=True, slots=True)
class EquityTradeRecommendationProposal:
    """A synthetic, modeled, non-authoritative recommendation proposal.

    It is preparation for future persistence and issuance (EI-4D), never
    execution permission: it is not issued, not recorded, not actionable,
    and every control in ``not_assessed`` still blocks actionable issuance.

    ``proposal_key`` identifies the candidate under the generation-policy
    version only. It does not identify this content: compare ``document()``
    to tell two proposals under one key apart.
    """

    proposal_key: str
    definition: EquityRecommendationDefinition
    entry_window: EquityModeledEntryWindow
    listing_eligibility_version: EquitySyntheticVersion
    universe_version: EquitySyntheticVersion
    not_assessed: tuple[EquityRecommendationCheck, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.definition, EquityRecommendationDefinition):
            _invalid("definition must be an EquityRecommendationDefinition.")
        if not isinstance(self.entry_window, EquityModeledEntryWindow):
            _invalid("entry_window must be an EquityModeledEntryWindow.")
        source = self.definition.source
        if self.proposal_key != equity_trade_recommendation_proposal_key(
            source, self.definition.generation_policy_version
        ):
            _invalid("proposal_key must be derived from the candidate and generation policy.")
        if self.entry_window.timing.decision_as_of != source.decision_as_of:
            _invalid("The modeled timing must start at the candidate's decision instant.")
        for name in ("listing_eligibility_version", "universe_version"):
            if not isinstance(getattr(self, name), EquitySyntheticVersion):
                _invalid(f"{name} must be an EquitySyntheticVersion.")
        if self.not_assessed != _UNASSESSED:
            _invalid("A proposal names every control that is not assessed yet.")

    @property
    def is_synthetic(self) -> bool:
        """Return True: every input is synthetic (EI4-D4)."""
        return True

    @property
    def is_authoritative(self) -> bool:
        """Return False: nothing is recorded before EI-4D."""
        return False

    @property
    def is_issued(self) -> bool:
        """Return False: issuance is modeled, never actual."""
        return False

    @property
    def is_actionable(self) -> bool:
        """Return False: unassessed controls still block actionable issuance."""
        return False

    def document(self) -> dict[str, Any]:
        """Return the canonical content."""
        return {
            "proposal_key": self.proposal_key,
            "synthetic": True,
            "authoritative": False,
            "issued": False,
            "actionable": False,
            "definition": self.definition.document(),
            "entry_window": self.entry_window.document(),
            "listing_eligibility_version": self.listing_eligibility_version.value,
            "universe_version": self.universe_version.value,
            "not_assessed": [check.value for check in self.not_assessed],
        }


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquityRecommendationInputs:
    """Every synthetic input an attempt applied, and the history it read, as found.

    The full documents are kept, not only their versions, so the audit shows
    exactly what was applied. ``open_recommendations`` is the point-in-time
    view of the session record: the recommendations issued at or before the
    modeled issuance, in (issued_at, key) order, each treated as open
    (EI4-D8). ``history_complete_through`` is the instant through which that
    record was complete.
    """

    timing_policy: EquitySyntheticEntryTimingPolicy
    generation_policy: EquitySyntheticGenerationPolicy
    risk_policy: EquitySyntheticRiskPolicy
    cost_schedule: EquitySyntheticCostSchedule
    price_increment: EquityPriceIncrement
    listing_eligibility: EquitySyntheticListingEligibility
    venue_reference: str
    history_reference: str
    history_complete_through: PointInTime
    open_recommendations: tuple[EquityPriorSessionRecommendation, ...]

    def __post_init__(self) -> None:
        for name, kind in (
            ("timing_policy", EquitySyntheticEntryTimingPolicy),
            ("generation_policy", EquitySyntheticGenerationPolicy),
            ("risk_policy", EquitySyntheticRiskPolicy),
            ("cost_schedule", EquitySyntheticCostSchedule),
            ("price_increment", EquityPriceIncrement),
            ("listing_eligibility", EquitySyntheticListingEligibility),
        ):
            if not isinstance(getattr(self, name), kind):
                _invalid(f"{name} must be an {kind.__name__}.")
        for name in ("venue_reference", "history_reference"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                _invalid(f"{name} must be non-empty text.")
        if not isinstance(self.history_complete_through, PointInTime):
            _invalid("history_complete_through must be a PointInTime.")
        if not isinstance(self.open_recommendations, tuple) or not all(
            isinstance(r, EquityPriorSessionRecommendation) for r in self.open_recommendations
        ):
            _invalid("open_recommendations must be a tuple of EquityPriorSessionRecommendation.")

    def versions(self) -> dict[str, EquitySyntheticVersion]:
        """Return each applied input's version, keyed like the request's version fields."""
        return {
            "generation_policy_version": self.generation_policy.version,
            "risk_policy_version": self.risk_policy.version,
            "cost_schedule_version": self.cost_schedule.version,
            "price_increment_version": self.price_increment.version,
            "timing_policy_version": self.timing_policy.version,
            "listing_eligibility_version": self.listing_eligibility.version,
        }

    def document(self) -> dict[str, Any]:
        """Return the canonical content."""
        return {
            "timing_policy": self.timing_policy.document(),
            "generation_policy": self.generation_policy.document(),
            "risk_policy": self.risk_policy.document(),
            "cost_schedule": self.cost_schedule.document(),
            "price_increment": self.price_increment.document(),
            "listing_eligibility": self.listing_eligibility.document(),
            "venue_reference": self.venue_reference,
            "history_reference": self.history_reference,
            "history_complete_through": self.history_complete_through.value,
            "open_recommendations": [r.document() for r in self.open_recommendations],
        }


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquityRecommendationEligibilityAudit:
    """The immutable eligibility audit of one generation attempt.

    It records the request, every check in declaration order, the candidate
    evaluated (when one was), every input applied and the point-in-time
    history read (when all loaded), the modeled entry window (when timing
    passed), Core's formation outcome (when Core ran) and the proposal (when
    PROPOSED). ``status`` and ``reasons`` are derived from the checks. It is
    suitable for later EI-4D persistence but is not persisted, and it is not
    authoritative.
    """

    request: EquityTradeRecommendationRequest
    checks: tuple[EquityRecommendationCheckResult, ...]
    candidate: EquityRecommendationCandidateProvenance | None
    inputs: EquityRecommendationInputs | None
    entry_window: EquityModeledEntryWindow | None
    core_outcome: EquityRecommendationDefinition | EquityRecommendationRejection | None
    proposal: EquityTradeRecommendationProposal | None

    def __post_init__(self) -> None:
        if not isinstance(self.request, EquityTradeRecommendationRequest):
            _invalid("request must be an EquityTradeRecommendationRequest.")
        checks = self.checks
        if not isinstance(checks, tuple) or tuple(
            c.check for c in checks if isinstance(c, EquityRecommendationCheckResult)
        ) != tuple(EquityRecommendationCheck):
            _invalid("checks must report every EquityRecommendationCheck once, in order.")
        if self.candidate is not None and not isinstance(
            self.candidate, EquityRecommendationCandidateProvenance
        ):
            _invalid("candidate must be an EquityRecommendationCandidateProvenance or None.")
        if self.entry_window is not None and not isinstance(
            self.entry_window, EquityModeledEntryWindow
        ):
            _invalid("entry_window must be an EquityModeledEntryWindow or None.")
        core = self.core_outcome
        if core is not None and not isinstance(
            core, EquityRecommendationDefinition | EquityRecommendationRejection
        ):
            _invalid("core_outcome must be a Core definition, rejection or None.")
        proposed = self.status is EquityRecommendationGenerationStatus.PROPOSED
        if proposed != isinstance(self.proposal, EquityTradeRecommendationProposal):
            _invalid("Exactly a PROPOSED audit carries a proposal.")
        if self.proposal is not None and (
            self.proposal.definition != core or self.proposal.entry_window != self.entry_window
        ):
            _invalid("The proposal must be the audited Core definition and entry window.")
        reported = {_R.CORE_FORMATION_REJECTED, _R.CORE_FORMATION_UNRESOLVED} & set(self.reasons)
        if isinstance(core, EquityRecommendationRejection):
            unresolved = bool(EQUITY_CORE_UNRESOLVED_REASONS & set(core.reasons))
            expected = _R.CORE_FORMATION_UNRESOLVED if unresolved else _R.CORE_FORMATION_REJECTED
            if reported != {expected}:
                _invalid(f"A Core rejection with reasons {core.reasons} is reported as {expected}.")
        elif reported:
            _invalid("A Core formation reason needs a Core rejection.")
        if core is not None and (self.candidate is None or core.source != self.candidate.source):
            _invalid("Core can only have run on the audited candidate.")
        inputs = self.inputs
        if inputs is not None:
            if not isinstance(inputs, EquityRecommendationInputs):
                _invalid("inputs must be EquityRecommendationInputs or None.")
            if self.candidate is None:
                _invalid("Inputs are loaded only for an audited candidate.")
            if any(
                getattr(self.request, name) != version
                for name, version in inputs.versions().items()
            ):
                _invalid("Every applied input must be the requested version.")
        elif self.entry_window is not None or core is not None:
            _invalid("Timing and Core formation run only on loaded inputs.")

    @property
    def reasons(self) -> tuple[EquityRecommendationEligibilityReason, ...]:
        """Return every check's reasons, distinct and in declaration order."""
        return ordered_equity_recommendation_reasons(
            {reason for check in self.checks for reason in check.reasons}
        )

    @property
    def status(self) -> EquityRecommendationGenerationStatus:
        """Return the status the reasons establish."""
        return equity_recommendation_status(self.reasons)

    @property
    def core_reasons(self) -> tuple[str, ...]:
        """Return Core's exact reasons, when Core rejected or could not resolve the candidate."""
        core = self.core_outcome
        if isinstance(core, EquityRecommendationRejection):
            return tuple(reason.value for reason in core.reasons)
        return ()

    @property
    def is_authoritative(self) -> bool:
        """Return False: no audit is recorded before EI-4D."""
        return False

    def document(self) -> dict[str, Any]:
        """Return the canonical content."""
        core = self.core_outcome
        return {
            "format": _AUDIT_FORMAT,
            "synthetic": True,
            "authoritative": False,
            "request": self.request.document(),
            "status": self.status.value,
            "reasons": [reason.value for reason in self.reasons],
            "checks": [check.document() for check in self.checks],
            "candidate": None if self.candidate is None else self.candidate.document(),
            "inputs": None if self.inputs is None else self.inputs.document(),
            "entry_window": None if self.entry_window is None else self.entry_window.document(),
            "core_outcome": None
            if core is None
            else {
                "kind": "DEFINITION"
                if isinstance(core, EquityRecommendationDefinition)
                else "REJECTION",
                **core.document(),
            },
            "proposal": None if self.proposal is None else self.proposal.document(),
        }

    def canonical_json(self) -> str:
        """Return the canonical JSON text of ``document()``."""
        return canonical_equity_candle_json(self.document())
