"""Immutable generation disposition and logical-attempt records (EI-4C, EI-4D-A).

These are the records EI-4D persists (EI-4 Decision Register, section 9;
ADR-021, section 1). Building one verifies it completely: its documents are
decoded through equity_recommendation_generation_decoding, every fingerprint
is re-derived, and every identity relation is checked. A record that does
not verify can never be constructed, so a stored row that decodes is
internally sound -- but not yet shown to rest on existing evidence or to
reproduce; that is EquityRecommendationGenerationVerifier's job
(verify_equity_recommendation_generation).

Dispositions
------------
A generation disposition exists only for an established generation identity:
a verified CANDIDATE under a resolved synthetic policy set, so an ``eirk1``
key. It has exactly one kind:

    EI-4C status              disposition kind
    PROPOSED                  PROPOSED       (carries the proposal)
    NOT_ELIGIBLE              NOT_ELIGIBLE
    EVALUATION_UNRESOLVED     UNRESOLVED

It holds the canonical assessment document and its ``eira1`` fingerprint, the
evaluator version that produced it and, for PROPOSED only, the canonical
proposal document and its ``eirp1`` fingerprint. The assessment carries the
exact frozen inputs (``eiri1``), the policy set (``eirs1``), the original
candidate and its evidence references and the ordered reasons; the record
exposes them. A NOT_ELIGIBLE or UNRESOLVED disposition never carries a
proposal, and nothing turns one kind into another: a different disposition
needs a different approved policy identity, so a different key.

Attempt-only results
--------------------
RESEARCH_INTEGRITY_FAILURE, DEPENDENCY_UNAVAILABLE and NO_CANDIDATE never have
a disposition. Neither does a NOT_ELIGIBLE result reached before a generation
identity could be established -- another research configuration, or
non-synthetic candidate evidence. Every attempt, these included, has a
logical-attempt record.

Logical attempts
----------------
An attempt's ``eirat1`` identity is the fingerprint of its structured
document: the request, the status, the reasons, every check's structured
result and, when established, the generation key, disposition kind and
assessment fingerprint. It holds no instant, counter or prose, so identical
invocations are one logical attempt (EI4-D9 clarification B), and two outages
that differ only in their exception messages are the same attempt.

Evaluator versions (ADR-025)
----------------------------
A disposition names its evaluator label; it must be one of the two governed
labels (an unknown one is Incompatible, a missing one Invalid) and its
assessment -- with its embedded inputs -- must be in that evaluator's formats,
or construction fails Invalid (N6). An attempt names no label: its format is
its evaluator's (``generation-attempt/1`` or ``/2``, N4), so attempts of the
two versions never share an identity, and only a version-2 attempt may state a
version-2 reason. Every load by a store adapter constructs through here, so
an unpaired record is never returned, verified or authorized.

Content equality
----------------
For one generation key, equal assessment fingerprints mean equal content and
different fingerprints a conflict (EI4-D9-D). Equal keys never prove equal
content. None of these records is an issued recommendation or a lifecycle
state (EI4-D9-H, EI4-D9-J).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from northstar_application.application_services.equity_recommendation_generation_decoding import (
    EquityDecodedAssessment,
    InvalidEquityRecommendationRecordError,
    decode_equity_recommendation_assessment,
    decode_equity_trade_recommendation_proposal,
    decode_equity_trade_recommendation_request,
    decode_structured_checks,
    equity_recommendation_document_evaluator,
    require_paired_equity_recommendation_evaluator,
    require_supported_equity_recommendation_evaluator,
    verified_equity_recommendation_document,
)
from northstar_application.application_services.equity_recommendation_generation_evidence import (
    EQUITY_RECOMMENDATION_EVALUATOR_V2,
    EQUITY_RECOMMENDATION_EVALUATORS,
    EQUITY_RECOMMENDATION_PROPOSAL_FORMAT,
    EquityRecommendationCandidateProvenance,
    EquityRecommendationInputs,
    canonical_equity_recommendation_json,
    equity_recommendation_fingerprint,
    is_equity_recommendation_generation_key,
)
from northstar_application.application_services.equity_trade_recommendation_eligibility import (
    EQUITY_RECOMMENDATION_VERSION_2_REASONS,
    EquityRecommendationEligibilityAudit,
    EquityRecommendationEligibilityReason,
    EquityRecommendationGenerationStatus,
    EquityTradeRecommendationProposal,
    EquityTradeRecommendationRequest,
    equity_recommendation_status,
    ordered_equity_recommendation_reasons,
)

_STATUS = EquityRecommendationGenerationStatus


class EquityRecommendationDispositionKind(StrEnum):
    """The kind of a generation disposition. None of them is a lifecycle state."""

    PROPOSED = "PROPOSED"
    NOT_ELIGIBLE = "NOT_ELIGIBLE"
    UNRESOLVED = "UNRESOLVED"


_KIND_OF_STATUS = {
    _STATUS.PROPOSED: EquityRecommendationDispositionKind.PROPOSED,
    _STATUS.NOT_ELIGIBLE: EquityRecommendationDispositionKind.NOT_ELIGIBLE,
    _STATUS.EVALUATION_UNRESOLVED: EquityRecommendationDispositionKind.UNRESOLVED,
}


def _invalid(message: str) -> None:
    raise InvalidEquityRecommendationRecordError(message)


def equity_recommendation_disposition_kind(
    audit: EquityRecommendationEligibilityAudit,
) -> EquityRecommendationDispositionKind | None:
    """Return the disposition kind ``audit`` establishes, or None for an attempt-only result."""
    if not isinstance(audit, EquityRecommendationEligibilityAudit):
        raise TypeError("audit must be an EquityRecommendationEligibilityAudit.")
    if audit.generation_key is None:
        return None
    return _KIND_OF_STATUS.get(audit.status)


# ---------------------------------------------------------------------------
# Dispositions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquityRecommendationGenerationDisposition:
    """One immutable generation disposition, as EI-4D persists it (EI4-D9-A).

    Construction decodes and verifies everything it holds; ``assessment`` is
    the decoded result. It is a generation record: never issued, never
    actionable and never a lifecycle state.
    """

    generation_key: str
    kind: EquityRecommendationDispositionKind
    evaluator_version: str
    assessment_fingerprint: str
    assessment_document: str
    proposal_fingerprint: str | None
    proposal_document: str | None
    assessment: EquityDecodedAssessment = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not is_equity_recommendation_generation_key(self.generation_key):
            _invalid("A disposition needs an eirk1 generation key.")
        if not isinstance(self.kind, EquityRecommendationDispositionKind):
            _invalid("kind must be an EquityRecommendationDispositionKind.")
        # An unknown label is Incompatible and a missing one Invalid, before anything is
        # decoded; then the documents must be that evaluator's (N6): otherwise Invalid.
        require_supported_equity_recommendation_evaluator(self.evaluator_version)
        assessment = decode_equity_recommendation_assessment(
            self.assessment_document, self.assessment_fingerprint
        )
        require_paired_equity_recommendation_evaluator(
            self.evaluator_version, assessment.evaluator_version, "assessment"
        )
        if assessment.inputs is None or assessment.generation_key != self.generation_key:
            _invalid("A disposition's assessment must establish this generation key.")
        if _KIND_OF_STATUS.get(assessment.status) is not self.kind:
            _invalid("The assessment's status is not this disposition's kind.")
        proposed = self.kind is EquityRecommendationDispositionKind.PROPOSED
        if proposed != (self.proposal_document is not None) or proposed != (
            self.proposal_fingerprint is not None
        ):
            _invalid("Exactly a PROPOSED disposition carries a proposal.")
        if assessment.document["proposal_fingerprint"] != self.proposal_fingerprint:
            _invalid("The assessment names another proposal.")
        if proposed:
            document = verified_equity_recommendation_document(
                self.proposal_document,  # type: ignore[arg-type]
                self.proposal_fingerprint,  # type: ignore[arg-type]
                EQUITY_RECOMMENDATION_PROPOSAL_FORMAT,
                "proposal",
            )
            proposal = decode_equity_trade_recommendation_proposal(document)
            if proposal != assessment.proposal or document != assessment.document["proposal"]:
                _invalid("The proposal is not the one the assessment holds.")
        object.__setattr__(self, "assessment", assessment)

    # -- what the record states, read from its verified content -------------

    @property
    def request(self) -> EquityTradeRecommendationRequest:
        """Return the generation request the assessment answered."""
        return self.assessment.request

    @property
    def inputs(self) -> EquityRecommendationInputs:
        """Return the exact frozen inputs of the assessment."""
        assert self.assessment.inputs is not None
        return self.assessment.inputs

    @property
    def candidate(self) -> EquityRecommendationCandidateProvenance:
        """Return the original authoritative CANDIDATE and its evidence references."""
        return self.inputs.candidate

    @property
    def inputs_fingerprint(self) -> str:
        """Return the ``eiri1`` fingerprint of the frozen inputs."""
        return self.inputs.fingerprint

    @property
    def inputs_document(self) -> str:
        """Return the frozen inputs as canonical JSON text."""
        return canonical_equity_recommendation_json(self.inputs.document())

    @property
    def policy_set_fingerprint(self) -> str:
        """Return the ``eirs1`` fingerprint of the policy set."""
        return self.inputs.policy_set.fingerprint

    @property
    def reasons(self) -> tuple[EquityRecommendationEligibilityReason, ...]:
        """Return the ordered reason codes."""
        return self.assessment.reasons

    @property
    def proposal(self) -> EquityTradeRecommendationProposal | None:
        """Return the decoded proposal, for a PROPOSED disposition only."""
        return self.assessment.proposal

    @property
    def is_issued(self) -> bool:
        """Return False: a disposition is a generation record, never an issued
        recommendation or a state of one (EI4-D9-H, EI4-D9-J)."""
        return False

    @classmethod
    def from_audit(
        cls, audit: EquityRecommendationEligibilityAudit
    ) -> EquityRecommendationGenerationDisposition:
        """Return ``audit``'s disposition; refuse an attempt-only result."""
        kind = equity_recommendation_disposition_kind(audit)
        if kind is None:
            _invalid(f"A {audit.status.value} attempt has no generation disposition.")
        proposal = audit.proposal
        return cls(
            generation_key=audit.generation_key,  # type: ignore[arg-type]
            kind=kind,  # type: ignore[arg-type]
            evaluator_version=audit.evaluator_version,  # the evaluator that produced it
            assessment_fingerprint=audit.assessment_fingerprint,
            assessment_document=canonical_equity_recommendation_json(audit.assessment_document()),
            proposal_fingerprint=None if proposal is None else proposal.fingerprint,
            proposal_document=None
            if proposal is None
            else canonical_equity_recommendation_json(proposal.document()),
        )


def equity_recommendation_disposition(
    audit: EquityRecommendationEligibilityAudit,
) -> EquityRecommendationGenerationDisposition | None:
    """Return ``audit``'s disposition, or None when it is attempt-only."""
    if equity_recommendation_disposition_kind(audit) is None:
        return None
    return EquityRecommendationGenerationDisposition.from_audit(audit)


# ---------------------------------------------------------------------------
# Logical attempts
# ---------------------------------------------------------------------------

_ATTEMPT_FIELDS = (
    "format",
    "request",
    "status",
    "reasons",
    "checks",
    "generation_key",
    "disposition_kind",
    "assessment_fingerprint",
)


def _attempt_document(audit: EquityRecommendationEligibilityAudit) -> dict[str, Any]:
    kind = equity_recommendation_disposition_kind(audit)
    keyed = audit.generation_key is not None
    return {
        "format": EQUITY_RECOMMENDATION_EVALUATORS[audit.evaluator_version].attempt_format,
        "request": audit.request.document(),
        "status": audit.status.value,
        "reasons": [reason.value for reason in audit.reasons],
        "checks": [check.structured_document() for check in audit.checks],
        "generation_key": audit.generation_key,
        "disposition_kind": None if kind is None else kind.value,
        "assessment_fingerprint": audit.assessment_fingerprint if keyed else None,
    }


@dataclass(frozen=True, slots=True)
class EquityRecommendationGenerationAttempt:
    """One logical generation attempt, as EI-4D appends it (EI4-D9-A).

    ``attempt_identity`` is the ``eirat1`` fingerprint of ``attempt_document``;
    construction decodes the document and verifies them against each other.
    ``evaluator_version`` is the evaluator its format belongs to (N4): an attempt
    pairs with the evaluator of its own audit, and names no label of its own.
    """

    attempt_identity: str
    attempt_document: str
    evaluator_version: str = field(init=False)
    status: EquityRecommendationGenerationStatus = field(init=False)
    reasons: tuple[EquityRecommendationEligibilityReason, ...] = field(init=False)
    generation_key: str | None = field(init=False)
    disposition_kind: EquityRecommendationDispositionKind | None = field(init=False)
    assessment_fingerprint: str | None = field(init=False)

    def __post_init__(self) -> None:
        document = verified_equity_recommendation_document(
            self.attempt_document,
            self.attempt_identity,
            tuple(row.attempt_format for row in EQUITY_RECOMMENDATION_EVALUATORS.values()),
            "attempt",
        )
        evaluator = equity_recommendation_document_evaluator("attempt", document["format"])
        try:
            if set(document) != set(_ATTEMPT_FIELDS):
                _invalid("An attempt document has missing or unexpected fields.")
            decode_equity_trade_recommendation_request(document["request"])
            status = _STATUS(document["status"])
            raw = document["reasons"]
            if not isinstance(raw, list):
                _invalid("An attempt's reasons must be a list.")
            reasons = tuple(EquityRecommendationEligibilityReason(r) for r in raw)
            if reasons != ordered_equity_recommendation_reasons(reasons):
                _invalid("An attempt's reasons must be distinct and in declaration order.")
            if equity_recommendation_status(reasons) is not status:
                _invalid("An attempt's status does not follow from its reasons.")
            if evaluator != EQUITY_RECOMMENDATION_EVALUATOR_V2 and (
                EQUITY_RECOMMENDATION_VERSION_2_REASONS & set(reasons)
            ):
                _invalid("Only a version-2 attempt can state a version-2 reason.")
            checks = decode_structured_checks(document["checks"])
            if set(reasons) != {r for _, _, found in checks for r in found}:
                _invalid("An attempt's reasons are not its checks' reasons.")
            key = document["generation_key"]
            if key is not None and not is_equity_recommendation_generation_key(key):
                _invalid("An attempt's generation key must be an eirk1 key.")
            expected = None if key is None else _KIND_OF_STATUS.get(status)
            stated = document["disposition_kind"]
            kind = None if stated is None else EquityRecommendationDispositionKind(stated)
            if kind is not expected:
                _invalid("An attempt's disposition kind must follow its status and key.")
            fingerprint = document["assessment_fingerprint"]
            if (key is None) != (fingerprint is None) or not (
                fingerprint is None
                or (isinstance(fingerprint, str) and fingerprint.startswith("eira1-sha256:"))
            ):
                _invalid("Exactly a keyed attempt names its eira1 assessment fingerprint.")
        except (ValueError, TypeError) as exc:
            if isinstance(exc, InvalidEquityRecommendationRecordError):
                raise
            raise InvalidEquityRecommendationRecordError(f"Attempt does not decode: {exc}") from exc
        object.__setattr__(self, "evaluator_version", evaluator)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "generation_key", key)
        object.__setattr__(self, "disposition_kind", expected)
        object.__setattr__(self, "assessment_fingerprint", fingerprint)

    @classmethod
    def from_audit(
        cls, audit: EquityRecommendationEligibilityAudit
    ) -> EquityRecommendationGenerationAttempt:
        """Return the logical attempt ``audit`` represents."""
        if not isinstance(audit, EquityRecommendationEligibilityAudit):
            raise TypeError("audit must be an EquityRecommendationEligibilityAudit.")
        document = _attempt_document(audit)
        return cls(
            attempt_identity=equity_recommendation_fingerprint(document),
            attempt_document=canonical_equity_recommendation_json(document),
        )


def equity_recommendation_attempt(
    audit: EquityRecommendationEligibilityAudit,
) -> EquityRecommendationGenerationAttempt:
    """Return the logical attempt ``audit`` represents."""
    return EquityRecommendationGenerationAttempt.from_audit(audit)
