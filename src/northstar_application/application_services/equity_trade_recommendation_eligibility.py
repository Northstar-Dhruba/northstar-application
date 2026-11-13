"""Synthetic equity recommendation eligibility contracts (EI-4C, EI4-D9 compatible).

The request, outcome vocabulary, proposal and eligibility audit of
GenerateEquityTradeRecommendationUseCase (generate_equity_trade_recommendation),
under the owner decisions recorded on 2026-11-07 in the EI-4 Decision Register
(northstar-docs, product/Equity-Intraday-V1-EI4-Decision-Register.md), ADR-020
and ADR-021. Everything here is synthetic and non-authoritative: nothing is
issued, persisted, actionable, an order, a fill or a position.

Outcomes
--------
Every reason has one fixed category, and an attempt's status is derived from
the reasons it recorded, in this precedence:

1. RESEARCH_INTEGRITY_FAILURE -- authoritative research history or its
   evidence failed verification. It is never reported as NO_SIGNAL or as an
   ordinary rejection.
2. DEPENDENCY_UNAVAILABLE -- a required input is missing, unknown,
   inconsistent or its source cannot answer. Generation fails closed. It
   includes the EI4-D10 policy-catalog reasons (ADR-022 section 8):
   POLICY_REFERENCE_CONFLICT, a published label bound to other content -- a
   permanent integrity conflict, never a transient outage, never retried --
   and POLICY_SET_UNAVAILABLE, a composite set that is not published or
   cannot be read. Callers distinguish a conflict from an outage by these
   reasons, never by the status.
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

Identity-bearing content and diagnostics (EI4-D9-C)
---------------------------------------------------
``assessment_document()`` is the identity-bearing assessment: the request,
the generation key, the status and reasons, every check's structured result,
the candidate evidence, the complete frozen inputs, the modeled entry window,
Core's outcome and the proposal. Its ``eira1`` fingerprint is
``assessment_fingerprint``. Check details and the timeline head observed while
loading are diagnostics: ``diagnostics_document()`` keeps them for inspection,
and they never move a fingerprint. A later timeline head -- after a
correction, for example -- therefore never changes a frozen assessment.

Generation identity (EI4-D9-B)
------------------------------
A proposal and an assessment are identified by the ``eirk1`` generation key
of the candidate and the resolved policy set
(equity_recommendation_generation_evidence). The earlier provisional key,
``equity_trade_recommendation_proposal_key`` under the namespace
``northstar.equity-trade-recommendation-proposal.v1``, binds only the
generation-policy version. It keeps exactly that meaning, is not the
generation identity and is never reinterpreted.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from typing import Any

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import PointInTime
from northstar_core.recommendations.equity import (
    EquityIssuanceKind,
    EquityRecommendationDefinition,
    EquityRecommendationRejection,
    EquityRecommendationRejectionReason,
    EquityRecommendationTiming,
    EquityResearchCandidateReference,
    EquitySyntheticVersion,
)

from northstar_application.application_services.equity_recommendation_generation_evidence import (
    EQUITY_RECOMMENDATION_CORE_OUTCOME_FORMAT,
    EQUITY_RECOMMENDATION_EVALUATOR_V1,
    EQUITY_RECOMMENDATION_EVALUATOR_V2,
    EQUITY_RECOMMENDATION_EVALUATORS,
    EQUITY_RECOMMENDATION_PROPOSAL_FORMAT,
    EquityRecommendationCandidateProvenance,
    EquityRecommendationInputs,
    EquitySyntheticPolicySetReference,
    canonical_equity_recommendation_json,
    equity_recommendation_fingerprint,
    is_equity_recommendation_generation_key,
)
from northstar_application.application_services.evaluate_equity_intraday_research import (
    EquityOpeningRangeBreakoutConfiguration,
)

EQUITY_TRADE_RECOMMENDATION_PROPOSAL_NAMESPACE = "northstar.equity-trade-recommendation-proposal.v1"
_AUDIT_FORMAT = "northstar.equity-recommendation-eligibility-audit/2"


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
    configuration. ``policy_set`` names the composite synthetic policy set
    (EI4-D9-B) and ``listing_eligibility_version`` the synthetic eligibility
    reference. There is no issuance instant: modeled issuance is always derived
    from the candidate's decision instant and the policy set's versioned
    latency (EI4-D9-C), and a caller can never supply or override it.
    """

    configuration: EquityOpeningRangeBreakoutConfiguration
    listing: ListingReference
    trading_date: date
    policy_set: EquitySyntheticPolicySetReference
    listing_eligibility_version: EquitySyntheticVersion

    def __post_init__(self) -> None:
        error = InvalidEquityTradeRecommendationRequestError
        if not isinstance(self.configuration, EquityOpeningRangeBreakoutConfiguration):
            raise error("configuration must be an EquityOpeningRangeBreakoutConfiguration.")
        if not isinstance(self.listing, ListingReference):
            raise error("listing must be a ListingReference.")
        if isinstance(self.trading_date, datetime) or not isinstance(self.trading_date, date):
            raise error("trading_date must be a plain date.")
        if not isinstance(self.policy_set, EquitySyntheticPolicySetReference):
            raise error("policy_set must be an EquitySyntheticPolicySetReference.")
        if not isinstance(self.listing_eligibility_version, EquitySyntheticVersion):
            raise error("listing_eligibility_version must be an EquitySyntheticVersion.")

    def document(self) -> dict[str, Any]:
        """Return the canonical content."""
        return {
            "strategy": self.configuration.strategy.identity,
            "research_configuration": self.configuration.document(),
            "listing": _listing(self.listing),
            "trading_date": self.trading_date.isoformat(),
            "policy_set": self.policy_set.document(),
            "listing_eligibility_version": self.listing_eligibility_version.value,
        }


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
    # Policy catalog (EI4-D10, ADR-022 section 8): a permanent integrity conflict,
    # and an unpublished composite set. Both are DEPENDENCY_UNAVAILABLE.
    POLICY_REFERENCE_CONFLICT = "POLICY_REFERENCE_CONFLICT"
    POLICY_SET_UNAVAILABLE = "POLICY_SET_UNAVAILABLE"
    SESSION_REFERENCE_UNAVAILABLE = "SESSION_REFERENCE_UNAVAILABLE"
    SESSION_REFERENCE_MISMATCH = "SESSION_REFERENCE_MISMATCH"
    LISTING_REFERENCE_UNAVAILABLE = "LISTING_REFERENCE_UNAVAILABLE"
    LISTING_REFERENCE_MISMATCH = "LISTING_REFERENCE_MISMATCH"
    LISTING_ELIGIBILITY_UNKNOWN = "LISTING_ELIGIBILITY_UNKNOWN"
    OPEN_RECOMMENDATION_STATE_UNAVAILABLE = "OPEN_RECOMMENDATION_STATE_UNAVAILABLE"
    RECOMMENDATION_HISTORY_INCOMPLETE = "RECOMMENDATION_HISTORY_INCOMPLETE"
    # N10 case C (owner approved 2026-11-13): evaluator version 2 only. The history is
    # declared complete beyond b = t - 1 us, so the modeled issuance at t can never be
    # issued; permanent, never retried into success, never INCOMPLETE.
    RECOMMENDATION_HISTORY_BOUNDARY_EXCEEDED = "RECOMMENDATION_HISTORY_BOUNDARY_EXCEEDED"
    # Timing (EI4-D1, EI0-D7 structure, EI4-D9-C)
    SESSION_NOT_SUPPORTED = "SESSION_NOT_SUPPORTED"
    TIMING_POLICY_NOT_APPLICABLE = "TIMING_POLICY_NOT_APPLICABLE"
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
        _R.POLICY_REFERENCE_CONFLICT,
        _R.POLICY_SET_UNAVAILABLE,
        _R.SESSION_REFERENCE_UNAVAILABLE,
        _R.SESSION_REFERENCE_MISMATCH,
        _R.LISTING_REFERENCE_UNAVAILABLE,
        _R.LISTING_REFERENCE_MISMATCH,
        _R.LISTING_ELIGIBILITY_UNKNOWN,
        _R.OPEN_RECOMMENDATION_STATE_UNAVAILABLE,
        _R.RECOMMENDATION_HISTORY_INCOMPLETE,
        _R.RECOMMENDATION_HISTORY_BOUNDARY_EXCEEDED,
    }
)
# Reasons only a version-2 evaluation can produce (ADR-025, N10 case C).
EQUITY_RECOMMENDATION_VERSION_2_REASONS = frozenset({_R.RECOMMENDATION_HISTORY_BOUNDARY_EXCEEDED})
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
    """One control's status, its reasons in report order and a diagnostic detail.

    ``check``, ``status`` and ``reasons`` are identity-bearing. ``detail`` is
    human-readable diagnostics and never part of a fingerprint.
    """

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

    def structured_document(self) -> dict[str, Any]:
        """Return the identity-bearing content: no diagnostic prose."""
        return {
            "check": self.check.value,
            "status": self.status.value,
            "reasons": [reason.value for reason in self.reasons],
        }

    def document(self) -> dict[str, Any]:
        """Return the full content, diagnostic detail included."""
        return {**self.structured_document(), "detail": self.detail}


def ordered_equity_recommendation_reasons(
    reasons: tuple[EquityRecommendationEligibilityReason, ...] | set[Any],
) -> tuple[EquityRecommendationEligibilityReason, ...]:
    """Return ``reasons`` distinct and in declaration order."""
    found = set(reasons)
    return tuple(reason for reason in EquityRecommendationEligibilityReason if reason in found)


# ---------------------------------------------------------------------------
# The earlier provisional key, unchanged
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
    """Return the provisional EI-4C key of ``source`` under ``generation_policy_version``.

    It binds only the generation-policy version and keeps exactly the meaning
    it had before EI4-D9. It is not the generation identity: that is the
    ``eirk1`` key of equity_recommendation_generation_key.
    """
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
# Entry window and proposal
# ---------------------------------------------------------------------------


def equity_core_outcome_document(
    core: EquityRecommendationDefinition | EquityRecommendationRejection,
) -> dict[str, Any]:
    """Return Core's formation outcome in a versioned Application envelope."""
    if not isinstance(core, EquityRecommendationDefinition | EquityRecommendationRejection):
        raise TypeError("core must be a Core definition or rejection.")
    return {
        "format": EQUITY_RECOMMENDATION_CORE_OUTCOME_FORMAT,
        "kind": "DEFINITION" if isinstance(core, EquityRecommendationDefinition) else "REJECTION",
        "outcome": core.document(),
    }


@dataclass(frozen=True, slots=True)
class EquityModeledEntryWindow:
    """The modeled instants of a proposal and the bounds of its entry eligibility.

    ``timing`` is Core's EI4-D1 timing with a MODELED issuance derived from
    the policy set's latency; its ``eligible_from`` is the earliest full entry
    bar opening at or after that issuance. No bar opening after
    ``latest_entry_bar_opens_at`` may evidence an entry, and
    ``modeled_exit_at`` is the mandatory modeled exit boundary. Nothing here
    scans bars or touches levels.
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

    ``generation_key`` identifies the candidate under the policy set
    (EI4-D9-B). It does not identify this content: ``fingerprint``, over the
    format-versioned ``document()``, does. Equal keys never prove equal
    proposals.
    """

    generation_key: str
    definition: EquityRecommendationDefinition
    entry_window: EquityModeledEntryWindow
    listing_eligibility_version: EquitySyntheticVersion
    universe_version: EquitySyntheticVersion
    not_assessed: tuple[EquityRecommendationCheck, ...]
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not is_equity_recommendation_generation_key(self.generation_key):
            _invalid("generation_key must be an eirk1 generation key.")
        if not isinstance(self.definition, EquityRecommendationDefinition):
            _invalid("definition must be an EquityRecommendationDefinition.")
        if not isinstance(self.entry_window, EquityModeledEntryWindow):
            _invalid("entry_window must be an EquityModeledEntryWindow.")
        if self.entry_window.timing.decision_as_of != self.definition.source.decision_as_of:
            _invalid("The modeled timing must start at the candidate's decision instant.")
        for name in ("listing_eligibility_version", "universe_version"):
            if not isinstance(getattr(self, name), EquitySyntheticVersion):
                _invalid(f"{name} must be an EquitySyntheticVersion.")
        if self.not_assessed != _UNASSESSED:
            _invalid("A proposal names every control that is not assessed yet.")
        object.__setattr__(self, "fingerprint", equity_recommendation_fingerprint(self.document()))

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
        """Return the canonical, format-versioned content."""
        return {
            "format": EQUITY_RECOMMENDATION_PROPOSAL_FORMAT,
            "generation_key": self.generation_key,
            "synthetic": True,
            "authoritative": False,
            "issued": False,
            "actionable": False,
            "definition": equity_core_outcome_document(self.definition),
            "entry_window": self.entry_window.document(),
            "listing_eligibility_version": self.listing_eligibility_version.value,
            "universe_version": self.universe_version.value,
            "not_assessed": [check.value for check in self.not_assessed],
        }


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquityRecommendationEligibilityAudit:
    """The immutable eligibility audit of one generation attempt.

    It records the request, every check in declaration order, the candidate
    evaluated (when one was), the complete frozen inputs (when all loaded),
    the modeled entry window (when timing passed), Core's formation outcome
    (when Core ran) and the proposal (when PROPOSED). ``status`` and
    ``reasons`` are derived from the checks. ``observed_timeline_head`` is the
    timeline head seen while loading: diagnostics only, never identity.
    ``evaluator_version`` is the governed evaluator that produced it (ADR-025):
    it chooses the assessment and attempt formats, and so their identities,
    even when no inputs were frozen; it is never a field of any document.

    It is suitable for EI-4D persistence but is not persisted, and it is not
    authoritative.
    """

    request: EquityTradeRecommendationRequest
    checks: tuple[EquityRecommendationCheckResult, ...]
    candidate: EquityRecommendationCandidateProvenance | None
    inputs: EquityRecommendationInputs | None
    entry_window: EquityModeledEntryWindow | None
    core_outcome: EquityRecommendationDefinition | EquityRecommendationRejection | None
    proposal: EquityTradeRecommendationProposal | None
    observed_timeline_head: str | None = None
    evaluator_version: str = EQUITY_RECOMMENDATION_EVALUATOR_V1

    def __post_init__(self) -> None:
        if not isinstance(self.request, EquityTradeRecommendationRequest):
            _invalid("request must be an EquityTradeRecommendationRequest.")
        if self.evaluator_version not in EQUITY_RECOMMENDATION_EVALUATORS:
            _invalid(f"{self.evaluator_version!r} is not a governed evaluator version.")
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
        if self.observed_timeline_head is not None and not (
            isinstance(self.observed_timeline_head, str) and self.observed_timeline_head
        ):
            _invalid("observed_timeline_head must be non-empty text or None.")
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
        if self.evaluator_version != EQUITY_RECOMMENDATION_EVALUATOR_V2 and (
            EQUITY_RECOMMENDATION_VERSION_2_REASONS & set(self.reasons)
        ):
            _invalid("Only a version-2 evaluation can report a version-2 reason.")
        self._check_inputs()

    def _check_inputs(self) -> None:
        inputs = self.inputs
        if inputs is None:
            if self.entry_window is not None or self.core_outcome is not None:
                _invalid("Timing and Core formation run only on loaded inputs.")
            return
        if not isinstance(inputs, EquityRecommendationInputs):
            _invalid("inputs must be EquityRecommendationInputs or None.")
        if inputs.evaluator_version != self.evaluator_version:
            _invalid("The inputs were frozen for another evaluator version.")
        if inputs.candidate != self.candidate:
            _invalid("The inputs must hold the audited candidate.")
        if inputs.policy_set.reference != self.request.policy_set:
            _invalid("The applied policy set must be the requested one.")
        if inputs.listing_eligibility.version != self.request.listing_eligibility_version:
            _invalid("The applied listing eligibility must be the requested version.")
        window = self.entry_window
        if window is not None and (
            window.timing.issued_at != inputs.modeled_issued_at
            or window.timing_policy_version != inputs.policy_set.timing_policy.version
        ):
            _invalid("The entry window must use the inputs' derived modeled issuance.")
        if self.proposal is not None and self.proposal.generation_key != inputs.generation_key:
            _invalid("The proposal must carry the inputs' generation key.")

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
    def generation_key(self) -> str | None:
        """Return the generation key, when the candidate and policy set were established."""
        return None if self.inputs is None else self.inputs.generation_key

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

    def assessment_document(self) -> dict[str, Any]:
        """Return the identity-bearing, format-versioned assessment (no diagnostics)."""
        core = self.core_outcome
        proposal = self.proposal
        return {
            "format": EQUITY_RECOMMENDATION_EVALUATORS[self.evaluator_version].assessment_format,
            "synthetic": True,
            "authoritative": False,
            "request": self.request.document(),
            "generation_key": self.generation_key,
            "status": self.status.value,
            "reasons": [reason.value for reason in self.reasons],
            "checks": [check.structured_document() for check in self.checks],
            "candidate": None if self.candidate is None else self.candidate.document(),
            "inputs": None if self.inputs is None else self.inputs.document(),
            "inputs_fingerprint": None if self.inputs is None else self.inputs.fingerprint,
            "entry_window": None if self.entry_window is None else self.entry_window.document(),
            "core_outcome": None if core is None else equity_core_outcome_document(core),
            "proposal": None if proposal is None else proposal.document(),
            "proposal_fingerprint": None if proposal is None else proposal.fingerprint,
        }

    @property
    def assessment_fingerprint(self) -> str:
        """Return the ``eira1`` fingerprint of the identity-bearing assessment."""
        return equity_recommendation_fingerprint(self.assessment_document())

    def diagnostics_document(self) -> dict[str, Any]:
        """Return the diagnostics: check details and the observed timeline head."""
        return {
            "observed_timeline_head": self.observed_timeline_head,
            "details": {check.check.value: check.detail for check in self.checks},
        }

    def document(self) -> dict[str, Any]:
        """Return the full audit: the assessment, its fingerprint and the diagnostics."""
        return {
            "format": _AUDIT_FORMAT,
            "assessment": self.assessment_document(),
            "assessment_fingerprint": self.assessment_fingerprint,
            "diagnostics": self.diagnostics_document(),
        }

    def canonical_json(self) -> str:
        """Return the canonical JSON text of ``document()``."""
        return canonical_equity_recommendation_json(self.document())
