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
2. The disposition is verified before anything is written: its original
   CANDIDATE and EI-2B snapshot through EI-3C, and its assessment by replay
   from its frozen inputs (EquityRecommendationGenerationVerifier).
3. ``record_generation`` writes the disposition, its proposal and the attempt
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
  same records -- never regenerated from fresh references -- at most
  ``max_writes`` writes in all, then RETRY_LIMIT_REACHED;
- the reload fails, or returns a record that does not decode:
  COMMIT_RECONCILIATION_UNRESOLVED. Nothing is claimed either way.

Attempt-only results -- integrity failures, outages, no candidate, and early
refusals with no generation identity -- are recorded with ``record_attempt``
and reconciled the same way, by reloading their attempt identity. The
absence of a disposition is never evidence about an attempt.

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
from northstar_application.application_services.equity_trade_recommendation_eligibility import (
    EquityRecommendationEligibilityAudit,
)
from northstar_application.application_services.verify_equity_recommendation_generation import (
    EquityRecommendationGenerationVerifier,
    EquityRecommendationReplayMismatchError,
    EquityRecommendationSourceUnavailableError,
    EquityRecommendationSourceVerificationError,
)
from northstar_application.ports.equity_recommendation_generation_store import (
    EquityRecommendationGenerationCommitUnconfirmedError,
    EquityRecommendationGenerationNotCommittedError,
    EquityRecommendationGenerationStore,
    EquityRecommendationGenerationStoreUnavailableError,
    EquityRecommendationStoreOutcome,
    EquityRecommendationStoreWrite,
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


_O = EquityRecommendationRecordingOutcome
_STORED = frozenset({_O.COMMITTED, _O.EQUIVALENT_COMMITTED})


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
    reconciled by reload.
    """

    outcome: EquityRecommendationRecordingOutcome
    attempt: EquityRecommendationGenerationAttempt
    disposition: EquityRecommendationGenerationDisposition | None
    authoritative: EquityRecommendationGenerationDisposition | None
    attempt_recorded: bool | None
    reconciled: bool
    writes: int
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, EquityRecommendationRecordingOutcome):
            raise ValueError("outcome must be an EquityRecommendationRecordingOutcome.")
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
    """Records a generation audit's verified disposition and logical attempt."""

    def __init__(
        self,
        store: EquityRecommendationGenerationStore,
        verifier: EquityRecommendationGenerationVerifier,
        max_writes: int = EQUITY_RECOMMENDATION_RECORD_WRITES,
    ) -> None:
        if not isinstance(store, EquityRecommendationGenerationStore):
            raise TypeError("store must be an EquityRecommendationGenerationStore.")
        if not isinstance(verifier, EquityRecommendationGenerationVerifier):
            raise TypeError("verifier must be an EquityRecommendationGenerationVerifier.")
        if isinstance(max_writes, bool) or not isinstance(max_writes, int) or max_writes < 1:
            raise ValueError("max_writes must be an integer of at least 1.")
        self._store = store
        self._verifier = verifier
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
        if (attempt.generation_key, attempt.assessment_fingerprint, attempt.disposition_kind) != (
            disposition.generation_key,
            disposition.assessment_fingerprint,
            disposition.kind,
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
            )

        refused = self._verified(disposition)
        if refused is not None:
            outcome, detail = refused
            return result(outcome, f"Not written: {detail}")

        key = disposition.generation_key
        for write in range(1, self._max_writes + 1):
            try:
                stored = self._store.record_generation(disposition, attempt)
            except EquityRecommendationGenerationStoreUnavailableError as exc:
                return result(
                    _O.PERSISTENCE_UNAVAILABLE, f"Storage unavailable: {exc}", writes=write
                )
            except EquityRecommendationGenerationNotCommittedError as exc:
                return result(_O.NOT_COMMITTED, f"Rolled back: {exc}", writes=write)
            except IncompatibleEquityRecommendationRecordError as exc:
                return result(_O.STORED_RECORD_INCOMPATIBLE, str(exc), writes=write)
            except InvalidEquityRecommendationRecordError as exc:
                return result(_O.STORED_RECORD_INVALID, str(exc), writes=write)
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

    def _verified(
        self, disposition: EquityRecommendationGenerationDisposition
    ) -> tuple[EquityRecommendationRecordingOutcome, str] | None:
        try:
            self._verifier.verify(disposition)
        except EquityRecommendationSourceUnavailableError as exc:
            return _O.SOURCE_EVIDENCE_UNAVAILABLE, f"Source evidence unavailable: {exc}"
        except EquityRecommendationSourceVerificationError as exc:
            return _O.SOURCE_EVIDENCE_INVALID, str(exc)
        except EquityRecommendationReplayMismatchError as exc:
            return _O.REPLAY_MISMATCH, str(exc)
        except IncompatibleEquityRecommendationRecordError as exc:
            return _O.STORED_RECORD_INCOMPATIBLE, str(exc)
        return None

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

        for write in range(1, self._max_writes + 1):
            try:
                stored = self._store.record_attempt(attempt)
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
