"""Recording verified recommendation-generation outcomes (EI-4D-A).

RecordEquityRecommendationGenerationUseCase turns one EI-4C generation audit
into its durable EI-4D records through the Application-owned
EquityRecommendationGenerationStore, under EI4-D9-A to EI4-D9-J (EI-4 Decision
Register, section 9, approved 2026-11-07) and ADR-021. Generation and
recording stay separate: the EI-4C generator never touches storage, and a
generation that completed in memory is never reported as recorded.

Recording a disposition
-----------------------
1. The audit's deterministic logical attempt and -- when a generation identity
   was established -- its disposition are built; both verify their own
   documents on construction.
2. The EI4-D10 recording gate (ADR-022 section 7, layer E) checks the
   disposition's frozen policy set against the immutable policy catalog,
   independently of how the disposition was built: its five component
   bindings and its POLICY_SET binding must each be published with exactly
   the frozen canonical content and fingerprint
   (verify_equity_synthetic_policy_bindings). Otherwise nothing is written
   and the attempt is reported not recorded (False):

   - POLICY_REFERENCE_CONFLICT: a label is published with other content, or
     its published binding does not verify or is unsupported -- permanent,
     never retried;
   - POLICY_REFERENCE_NOT_ESTABLISHED: a label is not published. Recording
     never publishes it;
   - POLICY_REFERENCE_UNAVAILABLE: the catalog cannot answer. Fail closed.

   ``policy_verification`` on the result keeps the typed verdict of each of
   the six identities. The gate applies to every disposition, including one
   equal to a disposition already stored: recording is a write request and
   never certifies a record's policy publication. Historical replay and
   read-only verification never consult the catalog. The gate does not stop
   a direct write through a storage adapter; that boundary is EI4-D11's.
3. The disposition is verified before anything is written: its original
   CANDIDATE and EI-2B snapshot through EI-3C, and its assessment by replay
   from its frozen inputs (EquityRecommendationGenerationVerifier).
   Only after the gate and this verification pass does the recorder issue a
   VerifiedEquityRecommendationGenerationWrite (EI4-D11, ADR-023): an
   authorization bound to the exact disposition, proposal and attempt, the
   verifier's actual result and the verified policy check
   (equity_recommendation_generation_write_authorization). It issues one per
   operation and reuses it, unchanged, for the bounded retries below. An
   attempt-only result gets a VerifiedEquityRecommendationAttemptWrite for
   exactly that attempt, with no policy evidence.
4. ``write_generation(authorization)`` writes the disposition, its proposal and the attempt
   as ONE atomic port operation, and its committed outcome is interpreted:

   - COMMITTED: this write recorded the disposition;
   - EQUIVALENT: the same content is already stored -- idempotent;
   - CONFLICT: other content is stored under the key. It stays authoritative;
     it is verified, never overwritten, and the conflicting attempt is
     recorded with it (EI4-D9-D). An UNRESOLVED disposition therefore never
     becomes PROPOSED under the same key.

Commit uncertainty (EI4-D9-G)
-----------------------------
An unconfirmed COMMIT is never treated as a rollback. The key is reloaded
once, authoritatively:

- the same record is stored -- the same fingerprints and the same evaluator
  version: committed, or an equivalent committed;
- other content is stored: a content conflict (the competing writer won);
- the same content under another evaluator version is stored:
  EVALUATOR_VERSION_CONFLICT, never equivalence -- no fingerprint covers the
  evaluator, so it is compared explicitly;
- nothing is stored, after a successful reload: the write is retried with the
  same records under the same authorization -- never regenerated from fresh
  references, never re-authorized -- at most
  ``max_writes`` writes in all, then RETRY_LIMIT_REACHED;
- the reload fails, or returns a record that does not decode:
  COMMIT_RECONCILIATION_UNRESOLVED. Nothing is claimed either way.

Attempt-only results -- integrity failures, outages, no candidate, and early
refusals with no generation identity -- are recorded with this recorder's
``record_attempt``, which writes through the port's ``write_attempt``, and
reconciled the same way, by reloading their attempt identity. The
absence of a disposition is never evidence about an attempt. A stored record
that does not construct is never authoritative: met during a write -- of a
disposition or of an attempt alone -- it is STORED_RECORD_INVALID, or
STORED_RECORD_INCOMPATIBLE for an unsupported format, with
``attempt_recorded`` unknown (None); met during a reload it leaves the commit
unresolved. It is never equivalent, retried, repaired or migrated.

Every outcome is typed (EquityRecommendationRecordingOutcome). Storage
outcomes are deliberately distinct from generation outcomes: a
COMMIT_RECONCILIATION_UNRESOLVED write has nothing to do with an
EVALUATION_UNRESOLVED assessment. When storage fails, the attempt is reported
as not recorded, never as audited. Nothing here sleeps, reads a clock, spawns
background work, writes the issued-recommendation history or issues anything.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from northstar_application.application_services.equity_recommendation_generation_decoding import (
    IncompatibleEquityRecommendationRecordError,
    InvalidEquityRecommendationRecordError,
)
from northstar_application.application_services.equity_recommendation_generation_records import (
    EquityRecommendationGenerationAttempt,
    EquityRecommendationGenerationDisposition,
    equity_recommendation_attempt,
    equity_recommendation_disposition,
)
from northstar_application.application_services.equity_recommendation_generation_write_authorization import (  # noqa: E501
    _issue_attempt_write,
    _issue_generation_write,
)
from northstar_application.application_services.equity_trade_recommendation_eligibility import (
    EquityRecommendationEligibilityAudit,
)
from northstar_application.application_services.verify_equity_recommendation_generation import (
    EquityRecommendationGenerationVerifier,
    EquityRecommendationReplayMismatchError,
    EquityRecommendationSourceUnavailableError,
    EquityRecommendationSourceVerificationError,
    EquityVerifiedRecommendationGeneration,
)
from northstar_application.application_services.verify_equity_synthetic_policy_bindings import (
    EquitySyntheticPolicyBindingVerdict,
    EquitySyntheticPolicySetVerification,
    verify_equity_synthetic_policy_set,
)
from northstar_application.ports.equity_recommendation_generation_store import (
    EquityRecommendationGenerationCommitUnconfirmedError,
    EquityRecommendationGenerationNotCommittedError,
    EquityRecommendationGenerationStore,
    EquityRecommendationGenerationStoreUnavailableError,
    EquityRecommendationStoreOutcome,
    EquityRecommendationStoreWrite,
)
from northstar_application.ports.equity_synthetic_policy_catalog_store import (
    EquitySyntheticPolicyCatalogStore,
)

EQUITY_RECOMMENDATION_RECORD_WRITES = 3


class EquityRecommendationRecordingOutcome(StrEnum):
    """How one recording ended. Only the first two mean the record is durably stored."""

    COMMITTED = "COMMITTED"
    EQUIVALENT_COMMITTED = "EQUIVALENT_COMMITTED"
    CONTENT_CONFLICT = "CONTENT_CONFLICT"
    EVALUATOR_VERSION_CONFLICT = "EVALUATOR_VERSION_CONFLICT"
    NOT_COMMITTED = "NOT_COMMITTED"
    PERSISTENCE_UNAVAILABLE = "PERSISTENCE_UNAVAILABLE"
    COMMIT_RECONCILIATION_UNRESOLVED = "COMMIT_RECONCILIATION_UNRESOLVED"
    RETRY_LIMIT_REACHED = "RETRY_LIMIT_REACHED"
    STORED_RECORD_INVALID = "STORED_RECORD_INVALID"
    STORED_RECORD_INCOMPATIBLE = "STORED_RECORD_INCOMPATIBLE"
    SOURCE_EVIDENCE_INVALID = "SOURCE_EVIDENCE_INVALID"
    SOURCE_EVIDENCE_UNAVAILABLE = "SOURCE_EVIDENCE_UNAVAILABLE"
    REPLAY_MISMATCH = "REPLAY_MISMATCH"
    # The EI4-D10 recording gate (ADR-022 sections 7 and 8): nothing is written.
    POLICY_REFERENCE_CONFLICT = "POLICY_REFERENCE_CONFLICT"
    POLICY_REFERENCE_NOT_ESTABLISHED = "POLICY_REFERENCE_NOT_ESTABLISHED"
    POLICY_REFERENCE_UNAVAILABLE = "POLICY_REFERENCE_UNAVAILABLE"


_O = EquityRecommendationRecordingOutcome
_V = EquitySyntheticPolicyBindingVerdict
_STORED = frozenset({_O.COMMITTED, _O.EQUIVALENT_COMMITTED})
# The recording-gate outcome of a policy-set verification verdict. A binding
# that does not verify or is unsupported is the permanent conflict at the gate
# (ADR-022 section 8, case D); the result's policy_verification keeps which.
_GATE = {
    _V.CONFLICT: _O.POLICY_REFERENCE_CONFLICT,
    _V.INVALID: _O.POLICY_REFERENCE_CONFLICT,
    _V.INCOMPATIBLE: _O.POLICY_REFERENCE_CONFLICT,
    _V.UNAVAILABLE: _O.POLICY_REFERENCE_UNAVAILABLE,
    _V.NOT_PUBLISHED: _O.POLICY_REFERENCE_NOT_ESTABLISHED,
}


@dataclass(frozen=True, slots=True)
class EquityRecommendationRecordingResult:
    """The outcome of recording one generation audit.

    ``disposition`` is the disposition this recording offered, or None for an
    attempt-only result. ``authoritative`` is the verified stored disposition
    for the key -- this one once committed, the other content on a conflict --
    or None when none is established. ``attempt_recorded`` is True when the
    logical attempt is durably stored (newly or already), False when it is
    known not to be, and None when that is unknown. ``writes`` counts the write
    calls made; ``reconciled`` says whether an unconfirmed commit was
    reconciled by reload. ``policy_verification`` is the recording gate's
    catalog verification of the disposition's frozen policy set: present
    whenever the gate ran, and the machine-readable reason for a
    POLICY_REFERENCE_* refusal -- including whether a conflicting binding
    was a valid other definition, did not verify, or was unsupported.
    """

    outcome: EquityRecommendationRecordingOutcome
    attempt: EquityRecommendationGenerationAttempt
    disposition: EquityRecommendationGenerationDisposition | None
    authoritative: EquityRecommendationGenerationDisposition | None
    attempt_recorded: bool | None
    reconciled: bool
    writes: int
    detail: str
    policy_verification: EquitySyntheticPolicySetVerification | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, EquityRecommendationRecordingOutcome):
            raise ValueError("outcome must be an EquityRecommendationRecordingOutcome.")
        if self.policy_verification is not None and not isinstance(
            self.policy_verification, EquitySyntheticPolicySetVerification
        ):
            raise ValueError("policy_verification must be an EquitySyntheticPolicySetVerification.")
        if self.outcome in _GATE.values() and (
            self.policy_verification is None
            or self.policy_verification.is_verified
            or self.writes != 0
        ):
            raise ValueError("A policy-reference refusal is a failed verification with no write.")
        if self.outcome in _STORED and self.attempt_recorded is False:
            raise ValueError("A stored outcome cannot have an unrecorded attempt.")
        if (
            self.disposition is not None
            and self.outcome in _STORED
            and (self.authoritative != self.disposition)
        ):
            raise ValueError("A stored disposition outcome names the offered disposition.")

    @property
    def is_stored(self) -> bool:
        """Return whether the offered record is durably stored, by this write or an equal one."""
        return self.outcome in _STORED


_Result = Callable[..., EquityRecommendationRecordingResult]


class RecordEquityRecommendationGenerationUseCase:
    """Records a generation audit's verified disposition and logical attempt.

    ``policy_catalog`` is the authoritative published-policy catalog that the
    recording gate reads (EI4-D10). It is required: no disposition can be
    recorded without it, and recording never publishes.
    """

    def __init__(
        self,
        store: EquityRecommendationGenerationStore,
        verifier: EquityRecommendationGenerationVerifier,
        policy_catalog: EquitySyntheticPolicyCatalogStore,
        max_writes: int = EQUITY_RECOMMENDATION_RECORD_WRITES,
    ) -> None:
        if not isinstance(store, EquityRecommendationGenerationStore):
            raise TypeError("store must be an EquityRecommendationGenerationStore.")
        if not isinstance(verifier, EquityRecommendationGenerationVerifier):
            raise TypeError("verifier must be an EquityRecommendationGenerationVerifier.")
        if not isinstance(policy_catalog, EquitySyntheticPolicyCatalogStore):
            raise TypeError("policy_catalog must be an EquitySyntheticPolicyCatalogStore.")
        if isinstance(max_writes, bool) or not isinstance(max_writes, int) or max_writes < 1:
            raise ValueError("max_writes must be an integer of at least 1.")
        self._store = store
        self._verifier = verifier
        self._catalog = policy_catalog
        self._max_writes = max_writes

    def execute(
        self, audit: EquityRecommendationEligibilityAudit
    ) -> EquityRecommendationRecordingResult:
        """Record ``audit``: its disposition with its attempt, or its attempt alone."""
        if not isinstance(audit, EquityRecommendationEligibilityAudit):
            raise TypeError("audit must be an EquityRecommendationEligibilityAudit.")
        attempt = equity_recommendation_attempt(audit)
        disposition = equity_recommendation_disposition(audit)
        if disposition is None:
            return self.record_attempt(attempt)
        return self.record(disposition, attempt)

    # ------------------------------------------------------------------
    # A disposition with its attempt
    # ------------------------------------------------------------------

    def record(
        self,
        disposition: EquityRecommendationGenerationDisposition,
        attempt: EquityRecommendationGenerationAttempt,
    ) -> EquityRecommendationRecordingResult:
        """Verify ``disposition`` and record it atomically with ``attempt``."""
        if not isinstance(disposition, EquityRecommendationGenerationDisposition):
            raise TypeError("disposition must be an EquityRecommendationGenerationDisposition.")
        if not isinstance(attempt, EquityRecommendationGenerationAttempt):
            raise TypeError("attempt must be an EquityRecommendationGenerationAttempt.")
        if (
            attempt.generation_key,
            attempt.assessment_fingerprint,
            attempt.disposition_kind,
            attempt.evaluator_version,
        ) != (
            disposition.generation_key,
            disposition.assessment_fingerprint,
            disposition.kind,
            disposition.evaluator_version,  # an attempt pairs with its own evaluator (N6)
        ):
            raise ValueError("The attempt is not the attempt of this disposition.")

        def result(
            outcome: EquityRecommendationRecordingOutcome,
            detail: str,
            *,
            authoritative: EquityRecommendationGenerationDisposition | None = None,
            attempt_recorded: bool | None = False,
            reconciled: bool = False,
            writes: int = 0,
        ) -> EquityRecommendationRecordingResult:
            return EquityRecommendationRecordingResult(
                outcome,
                attempt,
                disposition,
                authoritative,
                attempt_recorded,
                reconciled,
                writes,
                detail,
                policy,
            )

        # The EI4-D10 recording gate, before anything else and before any write:
        # the frozen policy content must be exactly what the catalog publishes.
        policy = verify_equity_synthetic_policy_set(self._catalog, disposition.inputs.policy_set)
        if not policy.is_verified:
            return result(
                _GATE[policy.verdict],
                f"Not written: the frozen policy set is not established by the policy "
                f"catalog ({policy.verdict.value}). {policy.describe()}",
            )

        verification = self._verification(disposition)
        if not isinstance(verification, EquityVerifiedRecommendationGeneration):
            outcome, detail = verification
            return result(outcome, f"Not written: {detail}")

        # EI4-D11 (ADR-023): every check passed, so -- and only now -- the
        # recorder authorizes this exact write. The one authorization is reused,
        # unchanged, for the bounded retries of this same operation.
        authorization = _issue_generation_write(verification, policy, attempt)
        key = disposition.generation_key
        for write in range(1, self._max_writes + 1):
            try:
                stored = self._store.write_generation(authorization)
            except EquityRecommendationGenerationStoreUnavailableError as exc:
                return result(
                    _O.PERSISTENCE_UNAVAILABLE, f"Storage unavailable: {exc}", writes=write
                )
            except EquityRecommendationGenerationNotCommittedError as exc:
                return result(_O.NOT_COMMITTED, f"Rolled back: {exc}", writes=write)
            # A stored row -- the disposition or the attempt -- does not construct.
            # This write is not confirmed, and whether the attempt is durably
            # recorded cannot be verified: unknown (None), neither True nor False.
            except IncompatibleEquityRecommendationRecordError as exc:
                return result(
                    _O.STORED_RECORD_INCOMPATIBLE, str(exc), attempt_recorded=None, writes=write
                )
            except InvalidEquityRecommendationRecordError as exc:
                return result(
                    _O.STORED_RECORD_INVALID, str(exc), attempt_recorded=None, writes=write
                )
            except EquityRecommendationGenerationCommitUnconfirmedError as exc:
                try:
                    found = self._store.load_disposition(key)
                except (
                    EquityRecommendationGenerationStoreUnavailableError,
                    InvalidEquityRecommendationRecordError,
                ) as reload_error:
                    return result(
                        _O.COMMIT_RECONCILIATION_UNRESOLVED,
                        f"The commit is unconfirmed ({exc}) and the reload failed "
                        f"({reload_error}); nothing is claimed.",
                        attempt_recorded=None,
                        reconciled=True,
                        writes=write,
                    )
                if found is None:
                    continue
                return self._settled(
                    disposition,
                    attempt,
                    found,
                    result,
                    reconciled=True,
                    writes=write,
                    attempt_recorded=self._attempt_stored(attempt),
                )
            return self._interpret(disposition, attempt, stored, result, writes=write)
        return result(
            _O.RETRY_LIMIT_REACHED,
            f"No write of {self._max_writes} was confirmed; each reload found nothing stored.",
            reconciled=True,
            writes=self._max_writes,
        )

    def _verification(
        self, disposition: EquityRecommendationGenerationDisposition
    ) -> EquityVerifiedRecommendationGeneration | tuple[EquityRecommendationRecordingOutcome, str]:
        """Return the verifier's actual result, or the refusal it establishes."""
        try:
            return self._verifier.verify(disposition)
        except EquityRecommendationSourceUnavailableError as exc:
            return _O.SOURCE_EVIDENCE_UNAVAILABLE, f"Source evidence unavailable: {exc}"
        except EquityRecommendationSourceVerificationError as exc:
            return _O.SOURCE_EVIDENCE_INVALID, str(exc)
        except EquityRecommendationReplayMismatchError as exc:
            return _O.REPLAY_MISMATCH, str(exc)
        except IncompatibleEquityRecommendationRecordError as exc:
            return _O.STORED_RECORD_INCOMPATIBLE, str(exc)

    def _verified(
        self, disposition: EquityRecommendationGenerationDisposition
    ) -> tuple[EquityRecommendationRecordingOutcome, str] | None:
        verification = self._verification(disposition)
        if isinstance(verification, EquityVerifiedRecommendationGeneration):
            return None
        return verification

    def _interpret(
        self,
        disposition: EquityRecommendationGenerationDisposition,
        attempt: EquityRecommendationGenerationAttempt,
        stored: EquityRecommendationStoreWrite,
        result: _Result,
        *,
        writes: int,
    ) -> EquityRecommendationRecordingResult:
        if not isinstance(stored, EquityRecommendationStoreWrite):
            raise TypeError("The store must return an EquityRecommendationStoreWrite.")
        if stored.outcome is EquityRecommendationStoreOutcome.COMMITTED:
            return result(
                _O.COMMITTED,
                "Recorded the disposition, its proposal when it has one, and the attempt.",
                authoritative=disposition,
                attempt_recorded=True,
                writes=writes,
            )
        if stored.stored is None:
            return result(
                _O.STORED_RECORD_INVALID,
                f"The store reported {stored.outcome.value} without the stored disposition.",
                attempt_recorded=None,
                writes=writes,
            )
        expected_equal = stored.outcome is EquityRecommendationStoreOutcome.EQUIVALENT
        if expected_equal != (stored.stored == disposition):
            return result(
                _O.STORED_RECORD_INVALID,
                f"The store reported {stored.outcome.value}, which the stored content contradicts.",
                attempt_recorded=None,
                writes=writes,
            )
        return self._settled(
            disposition,
            attempt,
            stored.stored,
            result,
            reconciled=False,
            writes=writes,
            attempt_recorded=True,
        )

    def _settled(
        self,
        disposition: EquityRecommendationGenerationDisposition,
        attempt: EquityRecommendationGenerationAttempt,
        found: EquityRecommendationGenerationDisposition,
        result: _Result,
        *,
        reconciled: bool,
        writes: int,
        attempt_recorded: bool | None,
    ) -> EquityRecommendationRecordingResult:
        if found.generation_key != disposition.generation_key:
            return result(
                _O.STORED_RECORD_INVALID,
                "The store returned a disposition for another generation key.",
                attempt_recorded=None,
                reconciled=reconciled,
                writes=writes,
            )
        if found == disposition:
            return result(
                _O.EQUIVALENT_COMMITTED,
                "The same content is stored under the key"
                + (" (this commit or an equivalent one)." if reconciled else "."),
                authoritative=disposition,
                attempt_recorded=attempt_recorded,
                reconciled=reconciled,
                writes=writes,
            )
        if found.evaluator_version != disposition.evaluator_version:
            # Not equivalent, whatever the fingerprints say: no fingerprint covers
            # the evaluator. The stored record is never replayed by an evaluator
            # it does not name, so it is reported, not verified or adopted.
            return result(
                _O.EVALUATOR_VERSION_CONFLICT,
                f"A disposition from evaluator {found.evaluator_version} is stored under the "
                f"key; this record is from {disposition.evaluator_version}. "
                + (
                    "The assessment content is identical, but the records are not equivalent."
                    if found.assessment_fingerprint == disposition.assessment_fingerprint
                    else "Their assessment content differs too."
                ),
                attempt_recorded=attempt_recorded,
                reconciled=reconciled,
                writes=writes,
            )
        refused = self._verified(found)
        if refused is not None:
            outcome, detail = refused
            # The stored record is never treated as authoritative unverified.
            if outcome is not _O.SOURCE_EVIDENCE_UNAVAILABLE:
                outcome = _O.STORED_RECORD_INVALID
            return result(
                outcome,
                f"Other content is stored under the key, and it does not verify: {detail}",
                attempt_recorded=attempt_recorded,
                reconciled=reconciled,
                writes=writes,
            )
        return result(
            _O.CONTENT_CONFLICT,
            f"Other content ({found.kind.value}, {found.assessment_fingerprint}) is stored under "
            "the key. It stays authoritative and is never overwritten (EI4-D9-D).",
            authoritative=found,
            attempt_recorded=attempt_recorded,
            reconciled=reconciled,
            writes=writes,
        )

    def _attempt_stored(self, attempt: EquityRecommendationGenerationAttempt) -> bool | None:
        try:
            found = self._store.load_attempt(attempt.attempt_identity)
        except (
            EquityRecommendationGenerationStoreUnavailableError,
            InvalidEquityRecommendationRecordError,
        ):
            return None
        return found == attempt

    # ------------------------------------------------------------------
    # An attempt alone
    # ------------------------------------------------------------------

    def record_attempt(
        self, attempt: EquityRecommendationGenerationAttempt
    ) -> EquityRecommendationRecordingResult:
        """Record an attempt-only result: never a disposition."""
        if not isinstance(attempt, EquityRecommendationGenerationAttempt):
            raise TypeError("attempt must be an EquityRecommendationGenerationAttempt.")
        if attempt.disposition_kind is not None:
            raise ValueError("An attempt with a generation disposition is recorded with it.")

        def result(
            outcome: EquityRecommendationRecordingOutcome,
            detail: str,
            *,
            attempt_recorded: bool | None,
            reconciled: bool = False,
            writes: int,
        ) -> EquityRecommendationRecordingResult:
            return EquityRecommendationRecordingResult(
                outcome, attempt, None, None, attempt_recorded, reconciled, writes, detail
            )

        # EI4-D11 (ADR-023): authorize exactly this attempt-only write. No policy
        # evidence, assessment or disposition is needed or fabricated; the one
        # authorization is reused for the bounded retries of this operation.
        authorization = _issue_attempt_write(attempt)
        for write in range(1, self._max_writes + 1):
            try:
                stored = self._store.write_attempt(authorization)
            except EquityRecommendationGenerationStoreUnavailableError as exc:
                return result(
                    _O.PERSISTENCE_UNAVAILABLE,
                    f"Storage unavailable; the attempt is not audited: {exc}",
                    attempt_recorded=False,
                    writes=write,
                )
            except EquityRecommendationGenerationNotCommittedError as exc:
                return result(
                    _O.NOT_COMMITTED,
                    f"Rolled back; the attempt is not audited: {exc}",
                    attempt_recorded=False,
                    writes=write,
                )
            except IncompatibleEquityRecommendationRecordError as exc:
                # A row under this identity does not construct, so whether the
                # attempt is durably recorded cannot be verified: unknown, never True.
                return result(
                    _O.STORED_RECORD_INCOMPATIBLE,
                    "The attempt stored under this identity has an unsupported format; it is "
                    f"not migrated, and this attempt is not shown to be recorded: {exc}",
                    attempt_recorded=None,
                    writes=write,
                )
            except InvalidEquityRecommendationRecordError as exc:
                return result(
                    _O.STORED_RECORD_INVALID,
                    "The attempt stored under this identity does not decode; it is not "
                    f"repaired, and this attempt is not shown to be recorded: {exc}",
                    attempt_recorded=None,
                    writes=write,
                )
            except EquityRecommendationGenerationCommitUnconfirmedError as exc:
                try:
                    found = self._store.load_attempt(attempt.attempt_identity)
                except (
                    EquityRecommendationGenerationStoreUnavailableError,
                    InvalidEquityRecommendationRecordError,
                ) as reload_error:
                    return result(
                        _O.COMMIT_RECONCILIATION_UNRESOLVED,
                        f"The commit is unconfirmed ({exc}) and the reload failed "
                        f"({reload_error}); nothing is claimed.",
                        attempt_recorded=None,
                        reconciled=True,
                        writes=write,
                    )
                if found is None:
                    continue
                if found != attempt:
                    return result(
                        _O.STORED_RECORD_INVALID,
                        "A different attempt is stored under this attempt identity.",
                        attempt_recorded=None,
                        reconciled=True,
                        writes=write,
                    )
                return result(
                    _O.EQUIVALENT_COMMITTED,
                    "The attempt is stored (this commit or an equivalent one).",
                    attempt_recorded=True,
                    reconciled=True,
                    writes=write,
                )
            if not isinstance(stored, EquityRecommendationStoreWrite) or stored.outcome not in (
                EquityRecommendationStoreOutcome.COMMITTED,
                EquityRecommendationStoreOutcome.EQUIVALENT,
            ):
                return result(
                    _O.STORED_RECORD_INVALID,
                    "The store answered an attempt-only write outside its contract.",
                    attempt_recorded=None,
                    writes=write,
                )
            committed = stored.outcome is EquityRecommendationStoreOutcome.COMMITTED
            return result(
                _O.COMMITTED if committed else _O.EQUIVALENT_COMMITTED,
                "Recorded the attempt." if committed else "The attempt was already recorded.",
                attempt_recorded=True,
                writes=write,
            )
        return result(
            _O.RETRY_LIMIT_REACHED,
            f"No write of {self._max_writes} was confirmed; each reload found nothing stored.",
            attempt_recorded=False,
            reconciled=True,
            writes=self._max_writes,
        )
