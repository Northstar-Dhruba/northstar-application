"""Verified-write authorizations for EI-4D generation records (EI4-D11, ADR-023).

EI4-D11-A makes the verified Application recorder
(RecordEquityRecommendationGenerationUseCase) the required operational path
for committing generation records. ADR-023 enforces it in layers; this module
is the Application-owned authorization layer:

- VerifiedEquityRecommendationGenerationWrite authorizes one exact generation
  write: a disposition (with its proposal when it has one), its logical
  attempt, the verifier's EquityVerifiedRecommendationGeneration (levels A to
  C: documents, EI-3C source and EI-2B snapshot, frozen replay) and the
  EI4-D10 gate's verified EquitySyntheticPolicySetVerification;
- VerifiedEquityRecommendationAttemptWrite authorizes one exact attempt-only
  write. It needs no policy evidence, assessment, proposal or disposition: an
  attempt-only result may arise before any policy-bearing input exists.

Issuance
--------
Only the verified recorder issues an authorization, after its checks pass,
through this module's private issuing functions. Issuance registers the
object in a module-private registry keyed by identity, and the validators
accept only a registered object. No token or secret is exposed: an
authorization constructed through the public constructor, copied, or made by
``dataclasses.replace`` is simply not registered, and is refused.

Exact content binding (ADR-023, section 4; clarification B)
-----------------------------------------------------------
At construction an authorization captures the exact persisted content it
covers, from existing canonical values only -- no new fingerprint or
identity scheme:

- generation: the generation key, disposition kind, evaluator version, the
  ``eira1`` assessment fingerprint and canonical assessment document, the
  ``eirp1`` proposal fingerprint and canonical proposal document (or none),
  and the ``eirat1`` attempt identity and canonical attempt document;
- attempt-only: the ``eirat1`` identity and canonical attempt document.

``require_verified_equity_recommendation_generation_write`` and
``require_verified_equity_recommendation_attempt_write`` -- the public
validators a store calls before writing -- re-derive that content from the
contained records and refuse any difference. They also refuse a decoded
assessment whose mutable nested document no longer encodes to its canonical
text, a verifier result for another disposition, a policy verification that
is not verified for the disposition's set, an attempt that is not the
disposition's attempt, and an attempt-only authorization holding a keyed
attempt. ``eirk1`` alone, policy labels or a "verified" flag never suffice.

Reuse
-----
The recorder issues one authorization per recording operation and reuses it
only for the identical operation during its bounded reconciliation
(EI4-D9-G). An authorization is never single-use: it covers fixed content,
and writing it again is EQUIVALENT under first-writer immutability.

Trust boundary (EI4D11-DQ1)
---------------------------
This stops accidental and ordinary unsupported writes within Northstar's
supported Application and Infrastructure interfaces. It is not
cryptographically unforgeable: privileged Python that reaches into this
module's private registry, direct SQL or file access can bypass it (ADR-022
DQ9). Nothing here is persisted, and nothing reads a clock or calls out.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any, NoReturn

from northstar_application.application_services.equity_recommendation_generation_evidence import (
    canonical_equity_recommendation_json,
)
from northstar_application.application_services.equity_recommendation_generation_records import (
    EquityRecommendationGenerationAttempt,
    EquityRecommendationGenerationDisposition,
)
from northstar_application.application_services.equity_synthetic_policy_bindings import (
    EquitySyntheticPolicyType,
)
from northstar_application.application_services.verify_equity_recommendation_generation import (
    EquityVerifiedRecommendationGeneration,
)
from northstar_application.application_services.verify_equity_synthetic_policy_bindings import (
    EquitySyntheticPolicySetVerification,
)
from northstar_application.ports.equity_recommendation_generation_store import (
    UnverifiedEquityRecommendationGenerationWriteError,
)

# Authorizations issued by the verified recorder in this process, keyed by
# identity. Weak: an authorization no one holds can no longer be written.
_ISSUED: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()


def _refuse(message: str) -> NoReturn:
    raise UnverifiedEquityRecommendationGenerationWriteError(message)


def _generation_content(
    disposition: EquityRecommendationGenerationDisposition,
    attempt: EquityRecommendationGenerationAttempt,
) -> tuple[str | None, ...]:
    """The exact persisted content of one generation write, from existing canonical values."""
    return (
        disposition.generation_key,
        disposition.kind.value,
        disposition.evaluator_version,
        disposition.assessment_fingerprint,
        disposition.assessment_document,
        disposition.proposal_fingerprint,
        disposition.proposal_document,
        attempt.attempt_identity,
        attempt.attempt_document,
    )


def _attempt_content(attempt: EquityRecommendationGenerationAttempt) -> tuple[str, str]:
    return attempt.attempt_identity, attempt.attempt_document


def _check_generation(
    verified: object, policy: object, attempt: object
) -> EquityRecommendationGenerationDisposition:
    """Refuse a generation write whose parts do not form one verified operation."""
    if not isinstance(verified, EquityVerifiedRecommendationGeneration):
        _refuse("A generation write needs the verifier's EquityVerifiedRecommendationGeneration.")
    if not isinstance(policy, EquitySyntheticPolicySetVerification):
        _refuse("A generation write needs the gate's EquitySyntheticPolicySetVerification.")
    if not isinstance(attempt, EquityRecommendationGenerationAttempt):
        _refuse("A generation write needs an EquityRecommendationGenerationAttempt.")
    disposition = verified.disposition
    if not isinstance(disposition, EquityRecommendationGenerationDisposition):
        _refuse("The verified result holds no generation disposition.")
    # The decoded assessment is what a store reads structured columns from; its
    # nested document must still be exactly the canonical text being written.
    if (
        canonical_equity_recommendation_json(disposition.assessment.document)
        != disposition.assessment_document
    ):
        _refuse("The disposition's decoded assessment no longer matches its canonical document.")
    if verified.replayed.assessment_fingerprint != disposition.assessment_fingerprint:
        _refuse("The verified replay is not this disposition's assessment.")
    if not policy.is_verified:
        _refuse("The policy-binding verification did not verify.")
    policy_set = policy.checks[-1]
    if (
        policy_set.identity.policy_type is not EquitySyntheticPolicyType.POLICY_SET
        or policy_set.identity.version != disposition.inputs.policy_set.reference.version
        or policy_set.offered_fingerprint != disposition.policy_set_fingerprint
    ):
        _refuse("The policy-binding verification is not this disposition's policy set.")
    if (attempt.generation_key, attempt.assessment_fingerprint, attempt.disposition_kind) != (
        disposition.generation_key,
        disposition.assessment_fingerprint,
        disposition.kind,
    ):
        _refuse("The attempt is not the attempt of this disposition.")
    return disposition


def _check_attempt(attempt: object) -> EquityRecommendationGenerationAttempt:
    if not isinstance(attempt, EquityRecommendationGenerationAttempt):
        _refuse("An attempt-only write needs an EquityRecommendationGenerationAttempt.")
    if attempt.disposition_kind is not None:
        _refuse("An attempt with a disposition kind is written with its disposition.")
    return attempt


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class VerifiedEquityRecommendationGenerationWrite:
    """The recorder's authorization to write exactly one verified disposition and attempt.

    ``covered`` is the exact persisted content captured at construction.
    Equality is identity: two authorizations are never interchangeable.
    """

    verified: EquityVerifiedRecommendationGeneration
    policy: EquitySyntheticPolicySetVerification
    attempt: EquityRecommendationGenerationAttempt
    covered: tuple[str | None, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        disposition = _check_generation(self.verified, self.policy, self.attempt)
        object.__setattr__(self, "covered", _generation_content(disposition, self.attempt))

    @property
    def disposition(self) -> EquityRecommendationGenerationDisposition:
        """Return the verified disposition this authorization covers."""
        return self.verified.disposition


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class VerifiedEquityRecommendationAttemptWrite:
    """The recorder's authorization to write exactly one attempt-only result."""

    attempt: EquityRecommendationGenerationAttempt
    covered: tuple[str, str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        attempt = _check_attempt(self.attempt)
        object.__setattr__(self, "covered", _attempt_content(attempt))


def _issue_generation_write(
    verified: EquityVerifiedRecommendationGeneration,
    policy: EquitySyntheticPolicySetVerification,
    attempt: EquityRecommendationGenerationAttempt,
) -> VerifiedEquityRecommendationGenerationWrite:
    """Private to the verified recorder: issue after its gate and verification passed."""
    authorization = VerifiedEquityRecommendationGenerationWrite(verified, policy, attempt)
    _ISSUED[id(authorization)] = authorization
    return authorization


def _issue_attempt_write(
    attempt: EquityRecommendationGenerationAttempt,
) -> VerifiedEquityRecommendationAttemptWrite:
    """Private to the verified recorder: issue for the attempt-only result it records."""
    authorization = VerifiedEquityRecommendationAttemptWrite(attempt)
    _ISSUED[id(authorization)] = authorization
    return authorization


def _require_issued(authorization: object, kind: type) -> None:
    if type(authorization) is not kind:
        _refuse(
            f"A {kind.__name__} issued by the verified recorder is required, not "
            f"{type(authorization).__name__}."
        )
    if _ISSUED.get(id(authorization)) is not authorization:
        _refuse(f"This {kind.__name__} was not issued by the verified recorder.")


def require_verified_equity_recommendation_generation_write(
    authorization: object,
) -> VerifiedEquityRecommendationGenerationWrite:
    """Return ``authorization`` when it authorizes exactly its content; refuse otherwise.

    A store calls this before writing anything. It raises
    UnverifiedEquityRecommendationGenerationWriteError for a missing, foreign
    or unissued authorization, or one whose content no longer matches.
    """
    _require_issued(authorization, VerifiedEquityRecommendationGenerationWrite)
    assert isinstance(authorization, VerifiedEquityRecommendationGenerationWrite)
    disposition = _check_generation(
        authorization.verified, authorization.policy, authorization.attempt
    )
    if _generation_content(disposition, authorization.attempt) != authorization.covered:
        _refuse("The write's content differs from the content this authorization covers.")
    return authorization


def require_verified_equity_recommendation_attempt_write(
    authorization: object,
) -> VerifiedEquityRecommendationAttemptWrite:
    """Return ``authorization`` when it authorizes exactly its attempt; refuse otherwise."""
    _require_issued(authorization, VerifiedEquityRecommendationAttemptWrite)
    assert isinstance(authorization, VerifiedEquityRecommendationAttemptWrite)
    attempt = _check_attempt(authorization.attempt)
    if _attempt_content(attempt) != authorization.covered:
        _refuse("The write's attempt differs from the attempt this authorization covers.")
    return authorization
