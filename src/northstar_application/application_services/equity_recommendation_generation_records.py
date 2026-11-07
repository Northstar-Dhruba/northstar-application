"""Pure generation disposition and logical-attempt contracts (EI-4C, EI4-D9-A).

These are the immutable records EI-4D will persist (EI-4 Decision Register,
section 9; ADR-021, section 1). This module builds and verifies them; it
stores nothing. Recording, conflict arbitration and reconciliation belong to
EI-4D-A and EI-4D-B.

Dispositions
------------
A generation disposition exists only for an established generation identity:
a verified CANDIDATE under a resolved synthetic policy set, so an ``eirk1``
key. It has exactly one kind:

    EI-4C status              disposition kind
    PROPOSED                  PROPOSED       (carries the proposal)
    NOT_ELIGIBLE              NOT_ELIGIBLE
    EVALUATION_UNRESOLVED     UNRESOLVED

It holds the canonical assessment document and its ``eira1`` fingerprint and,
for PROPOSED only, the canonical proposal document and its ``eirp1``
fingerprint. Construction verifies every document and fingerprint. A
NOT_ELIGIBLE or UNRESOLVED disposition never carries a proposal, and nothing
here can turn one kind into another.

Attempt-only results
--------------------
RESEARCH_INTEGRITY_FAILURE, DEPENDENCY_UNAVAILABLE and NO_CANDIDATE never have
a disposition. Neither does a NOT_ELIGIBLE result reached before a generation
identity could be established -- another research configuration, or
non-synthetic candidate evidence -- because no verified candidate and policy
key exists for it. Every attempt, these included, has a logical-attempt
record.

Logical attempts
----------------
An attempt's ``eirat1`` identity is the fingerprint of its structured
document: the request, the status, the reasons, every check's structured
result and, when established, the generation key, disposition kind and
assessment fingerprint. It holds no instant, counter or prose, so identical
invocations are one logical attempt (EI4-D9 clarification B), and two outages
that differ only in their exception messages are the same attempt.

Content equality
----------------
For one generation key, equal assessment fingerprints mean equal content and
different fingerprints a conflict (EI4-D9-D). Equal keys never prove equal
content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from northstar_application.application_services.equity_recommendation_generation_evidence import (
    EQUITY_RECOMMENDATION_ASSESSMENT_FORMAT,
    EQUITY_RECOMMENDATION_ATTEMPT_FORMAT,
    EQUITY_RECOMMENDATION_PROPOSAL_FORMAT,
    InvalidEquityRecommendationDocumentError,
    canonical_equity_recommendation_json,
    equity_recommendation_fingerprint,
    is_equity_recommendation_generation_key,
    verify_equity_recommendation_document,
)
from northstar_application.application_services.equity_trade_recommendation_eligibility import (
    EquityRecommendationEligibilityAudit,
    EquityRecommendationGenerationStatus,
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


class InvalidEquityRecommendationRecordError(ValueError):
    """Raised when a disposition or attempt record is inconsistent."""


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


def _verified(text: str, fingerprint: str, expected_format: str) -> dict[str, Any]:
    try:
        document = verify_equity_recommendation_document(text, fingerprint)
    except InvalidEquityRecommendationDocumentError as exc:
        raise InvalidEquityRecommendationRecordError(str(exc)) from exc
    if document["format"] != expected_format:
        _invalid(f"Expected a {expected_format} document.")
    return document


@dataclass(frozen=True, slots=True)
class EquityRecommendationGenerationDisposition:
    """One immutable generation disposition, as EI-4D will persist it (EI4-D9-A)."""

    generation_key: str
    kind: EquityRecommendationDispositionKind
    assessment_fingerprint: str
    assessment_document: str
    proposal_fingerprint: str | None
    proposal_document: str | None

    def __post_init__(self) -> None:
        if not is_equity_recommendation_generation_key(self.generation_key):
            _invalid("A disposition needs an eirk1 generation key.")
        if not isinstance(self.kind, EquityRecommendationDispositionKind):
            _invalid("kind must be an EquityRecommendationDispositionKind.")
        assessment = _verified(
            self.assessment_document,
            self.assessment_fingerprint,
            EQUITY_RECOMMENDATION_ASSESSMENT_FORMAT,
        )
        if assessment["generation_key"] != self.generation_key:
            _invalid("The assessment is for another generation key.")
        if _KIND_OF_STATUS.get(_STATUS(assessment["status"])) is not self.kind:
            _invalid("The assessment's status is not this disposition's kind.")
        proposed = self.kind is EquityRecommendationDispositionKind.PROPOSED
        if proposed != (self.proposal_document is not None) or proposed != (
            self.proposal_fingerprint is not None
        ):
            _invalid("Exactly a PROPOSED disposition carries a proposal.")
        if assessment["proposal_fingerprint"] != self.proposal_fingerprint:
            _invalid("The assessment names another proposal.")
        if proposed:
            proposal = _verified(
                self.proposal_document,  # type: ignore[arg-type]
                self.proposal_fingerprint,  # type: ignore[arg-type]
                EQUITY_RECOMMENDATION_PROPOSAL_FORMAT,
            )
            if proposal["generation_key"] != self.generation_key:
                _invalid("The proposal is for another generation key.")
            if proposal != assessment["proposal"]:
                _invalid("The proposal is not the one the assessment holds.")

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


def _attempt_document(audit: EquityRecommendationEligibilityAudit) -> dict[str, Any]:
    kind = equity_recommendation_disposition_kind(audit)
    keyed = audit.generation_key is not None
    return {
        "format": EQUITY_RECOMMENDATION_ATTEMPT_FORMAT,
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
    """One logical generation attempt, as EI-4D will append it (EI4-D9-A).

    ``attempt_identity`` is the ``eirat1`` fingerprint of ``attempt_document``;
    construction verifies them against each other.
    """

    attempt_identity: str
    attempt_document: str
    status: EquityRecommendationGenerationStatus = field(init=False)
    generation_key: str | None = field(init=False)
    disposition_kind: EquityRecommendationDispositionKind | None = field(init=False)

    def __post_init__(self) -> None:
        document = _verified(
            self.attempt_document, self.attempt_identity, EQUITY_RECOMMENDATION_ATTEMPT_FORMAT
        )
        status = _STATUS(document["status"])
        key = document["generation_key"]
        if key is not None and not is_equity_recommendation_generation_key(key):
            _invalid("An attempt's generation key must be an eirk1 key.")
        expected = None if key is None else _KIND_OF_STATUS.get(status)
        stated = document["disposition_kind"]
        if (
            None if stated is None else EquityRecommendationDispositionKind(stated)
        ) is not expected:
            _invalid("An attempt's disposition kind must follow its status and key.")
        if (key is None) != (document["assessment_fingerprint"] is None):
            _invalid("Exactly a keyed attempt names its assessment fingerprint.")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "generation_key", key)
        object.__setattr__(self, "disposition_kind", expected)

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
