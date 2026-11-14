"""Recording the lifecycle of an authoritative MODELED issuance (EI-6B design, section 7).

``RecordEquityRecommendationLifecycleUseCase.execute(issuance_id, snapshot_ids)``
is the only issuer of lifecycle-append authorizations. It acts on explicitly
supplied snapshot identities only, and on SYNTHETIC evidence only (D6B-1):
being non-synthetic in name grants no permission to use real data.

1. Load the authoritative issuance through the issuance reader (EI-6A).
2. Re-verify its EI-4D disposition at levels A to C, by generation key.
3. Derive the lifecycle window from them (``equity_issued_lifecycle_window``).
4. Load the stored lifecycle; its parent and window must be the ones derived.
5. Evaluate each snapshot under the EI-5C rules, shared rather than copied:
   SYNTHETIC evidence; the issuance's listing, trading date and venue
   reference; its research window, timeframe and currency; ``as_of`` at or
   after the modeled issuance and the last recorded event; within one call,
   strictly increasing ``as_of`` and a non-regressing watermark. Core
   ``advance`` decides every transition, and corrections to cited evidence are
   recorded as ``SOURCE_REVISION_RECORDED`` exactly where Core permits (EI5-D8).
6. No new event: ``NO_NEW_EVENTS``; nothing is written or fabricated.
7. Otherwise issue a content-bound authorization over the stored prior history
   and the new events, and append once.
8. Commit uncertainty is reconciled by reloading the lifecycle, at most
   ``max_writes`` writes: a stored history beginning with exactly prior plus
   the new events -- the events are confirmed committed, even if later events
   follow -- is EQUIVALENT (reconciled); one equal to prior is retried;
   anything else is CONFLICT. An unreadable reload, or the bound exhausted, raises
   EquityLifecycleRecordingUnresolvedError: nothing is claimed. Corruption met
   on reload is the permanent integrity error, raised unchanged.

Quiet evaluations (EI-6B design, section 7)
-------------------------------------------
There is no persisted quiet-evaluation cursor. A snapshot that changes nothing
leaves no event, so across calls only the last *recorded event* bounds
chronology: a snapshot older than one evaluated quietly in an earlier call is
still accepted when it is not older than the last event. Re-evaluating any
snapshot is deterministic and safe -- Core re-reads quiet bars from the cursor
and revision recording is idempotent -- and no exactly-once processing is
claimed. Which snapshots are evaluated, and when, is EI-6C's (O-2).

O-3 (option (b)): late evidence filling the gap of a terminal
``INCOMPLETE_EVIDENCE`` produces no lifecycle event; Core refuses a revision
citing ``EVIDENCE_INCOMPLETE``, and nothing here works around it.

It never selects snapshots, polls, schedules, keeps a cursor, issues, or
creates an order, fill or position. Nothing reads a clock or calls out.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from northstar_core.recommendations.equity.lifecycle import (
    EquityLifecycleEvent,
    EquityLifecycleState,
    EquityLifecycleWindow,
    EquityRecommendationLifecycle,
)

from northstar_application.application_services.equity_candle_evidence_snapshot import (
    EquityCandleEvidenceSnapshot,
)
from northstar_application.application_services.equity_recommendation_generation_decoding import (
    IncompatibleEquityRecommendationRecordError,
    InvalidEquityRecommendationRecordError,
)
from northstar_application.application_services.equity_recommendation_issuance_records import (
    EquityRecordedIssuance,
)
from northstar_application.application_services.equity_recommendation_lifecycle_authorization import (  # noqa: E501
    VerifiedEquityRecommendationLifecycleAppend,
    _issue_lifecycle_append,
)
from northstar_application.application_services.equity_recommendation_lifecycle_events import (
    equity_lifecycle_event_documents,
)
from northstar_application.application_services.equity_recommendation_lifecycle_window import (
    InvalidEquityLifecycleWindowProvenanceError,
    equity_issued_lifecycle_window,
)
from northstar_application.application_services.evaluate_synthetic_equity_recommendation_lifecycle import (  # noqa: E501
    EquitySyntheticLifecycleRefusedError,
    EquitySyntheticLifecycleSnapshotPosition,
    EvaluateSyntheticEquityRecommendationLifecycleUseCase,
    _observation,
    _record_revisions,
)
from northstar_application.application_services.verify_equity_recommendation_generation import (
    EquityRecommendationReplayMismatchError,
    EquityRecommendationSourceUnavailableError,
    EquityRecommendationSourceVerificationError,
    EquityVerifiedRecommendationGeneration,
    VerifyStoredEquityRecommendationGenerationUseCase,
)
from northstar_application.ports.equity_candle_evidence_journal import (
    EquityCandleEvidenceSnapshotStore,
)
from northstar_application.ports.equity_recommendation_generation_store import (
    EquityRecommendationGenerationStoreUnavailableError,
)
from northstar_application.ports.equity_recommendation_issuance import (
    EquityRecommendationIssuanceReader,
    EquityRecommendationIssuanceUnavailableError,
    InvalidStoredEquityRecommendationIssuanceAuthorityError,
)
from northstar_application.ports.equity_recommendation_lifecycle_store import (
    EquityRecommendationLifecycleCommitUnconfirmedError,
    EquityRecommendationLifecycleStore,
    EquityRecommendationLifecycleStoreUnavailableError,
    EquityRecommendationLifecycleWrite,
    EquityRecommendationLifecycleWriteOutcome,
    EquityRecordedLifecycle,
    InvalidEquityRecommendationLifecycleWriteError,
    InvalidStoredEquityRecommendationLifecycleError,
)

EQUITY_LIFECYCLE_RECORDING_WRITES = 3

_W = EquityRecommendationLifecycleWriteOutcome
_Synthetic = EvaluateSyntheticEquityRecommendationLifecycleUseCase


class EquityLifecycleRecordingOutcome(StrEnum):
    """What one recording request established."""

    APPENDED = "APPENDED"
    EQUIVALENT = "EQUIVALENT"
    CONFLICT = "CONFLICT"
    NO_NEW_EVENTS = "NO_NEW_EVENTS"


class EquityLifecycleRecordingRefusal(StrEnum):
    """Why a recording request was refused before any write. Nothing was written.

    The ``SNAPSHOT_*`` reasons are EI-5C's, applied by the same rules.
    """

    ISSUANCE_NOT_FOUND = "ISSUANCE_NOT_FOUND"
    AUTHORITY_UNAVAILABLE = "AUTHORITY_UNAVAILABLE"
    STORED_AUTHORITY_INVALID = "STORED_AUTHORITY_INVALID"
    GENERATION_NOT_FOUND = "GENERATION_NOT_FOUND"
    GENERATION_STORE_UNAVAILABLE = "GENERATION_STORE_UNAVAILABLE"
    STORED_GENERATION_INVALID = "STORED_GENERATION_INVALID"
    STORED_GENERATION_INCOMPATIBLE = "STORED_GENERATION_INCOMPATIBLE"
    SOURCE_EVIDENCE_UNAVAILABLE = "SOURCE_EVIDENCE_UNAVAILABLE"
    SOURCE_EVIDENCE_INVALID = "SOURCE_EVIDENCE_INVALID"
    REPLAY_MISMATCH = "REPLAY_MISMATCH"
    WINDOW_PROVENANCE_INVALID = "WINDOW_PROVENANCE_INVALID"
    STORE_UNAVAILABLE = "STORE_UNAVAILABLE"
    STORED_LIFECYCLE_INVALID = "STORED_LIFECYCLE_INVALID"
    SNAPSHOT_NOT_FOUND = "SNAPSHOT_NOT_FOUND"
    SNAPSHOT_IDENTITY_MISMATCH = "SNAPSHOT_IDENTITY_MISMATCH"
    SNAPSHOT_NOT_SYNTHETIC = "SNAPSHOT_NOT_SYNTHETIC"
    SNAPSHOT_LISTING_MISMATCH = "SNAPSHOT_LISTING_MISMATCH"
    SNAPSHOT_SESSION_MISMATCH = "SNAPSHOT_SESSION_MISMATCH"
    SNAPSHOT_WINDOW_MISMATCH = "SNAPSHOT_WINDOW_MISMATCH"
    SNAPSHOT_BEFORE_ISSUANCE = "SNAPSHOT_BEFORE_ISSUANCE"
    SNAPSHOT_OUT_OF_ORDER = "SNAPSHOT_OUT_OF_ORDER"
    SNAPSHOT_WATERMARK_REGRESSION = "SNAPSHOT_WATERMARK_REGRESSION"
    SNAPSHOT_EVIDENCE_INCONSISTENT = "SNAPSHOT_EVIDENCE_INCONSISTENT"
    SNAPSHOT_CONTRADICTS_CITED_EVIDENCE = "SNAPSHOT_CONTRADICTS_CITED_EVIDENCE"


_RETRYABLE = frozenset(
    {
        EquityLifecycleRecordingRefusal.AUTHORITY_UNAVAILABLE,
        EquityLifecycleRecordingRefusal.GENERATION_STORE_UNAVAILABLE,
        EquityLifecycleRecordingRefusal.SOURCE_EVIDENCE_UNAVAILABLE,
        EquityLifecycleRecordingRefusal.STORE_UNAVAILABLE,
    }
)
_R = EquityLifecycleRecordingRefusal


class EquityLifecycleRecordingRefusedError(ValueError):
    """A recording request was refused before any write; ``reason`` says why.

    ``retryable`` is True only for an outage of the issuance authority, the
    generation store, the source evidence or the lifecycle store.
    """

    def __init__(self, reason: EquityLifecycleRecordingRefusal, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason

    @property
    def retryable(self) -> bool:
        """Return whether the same request may succeed once an outage clears."""
        return self.reason in _RETRYABLE


class EquityLifecycleRecordingUnresolvedError(RuntimeError):
    """Raised when an unconfirmed append could not be settled; nothing is claimed."""


def _refuse(
    reason: EquityLifecycleRecordingRefusal, message: str
) -> EquityLifecycleRecordingRefusedError:
    return EquityLifecycleRecordingRefusedError(reason, message)


@dataclass(frozen=True, slots=True)
class EquityLifecycleRecordingResult:
    """The outcome of one recording request.

    ``recorded`` is the authoritative history the outcome concerns: after the
    append (APPENDED), the equal history (EQUIVALENT), the stored history that
    refused it (CONFLICT), or the unchanged stored history (NO_NEW_EVENTS).
    ``new_events`` are the events this request derived; they are recorded only
    for APPENDED and EQUIVALENT. ``writes`` counts store appends; ``reconciled``
    says whether an unconfirmed commit was settled by reload.
    """

    outcome: EquityLifecycleRecordingOutcome
    recorded: EquityRecordedLifecycle
    new_events: tuple[EquityLifecycleEvent, ...]
    writes: int
    reconciled: bool

    @property
    def state(self) -> EquityLifecycleState:
        """Return the authoritative lifecycle state after this request."""
        return self.recorded.lifecycle.state


class RecordEquityRecommendationLifecycleUseCase:
    """Records Core lifecycle events of an authoritative issuance from explicit snapshots.

    See the module docstring. ``issuances`` is the EI-6A reader; ``generations``
    re-verifies the disposition; ``snapshots`` is the read-only EI-2B store;
    ``lifecycles`` is the lifecycle store.
    """

    def __init__(
        self,
        issuances: EquityRecommendationIssuanceReader,
        generations: VerifyStoredEquityRecommendationGenerationUseCase,
        snapshots: EquityCandleEvidenceSnapshotStore,
        lifecycles: EquityRecommendationLifecycleStore,
        max_writes: int = EQUITY_LIFECYCLE_RECORDING_WRITES,
    ) -> None:
        if not isinstance(issuances, EquityRecommendationIssuanceReader):
            raise TypeError("issuances must be an EquityRecommendationIssuanceReader.")
        if not isinstance(generations, VerifyStoredEquityRecommendationGenerationUseCase):
            raise TypeError(
                "generations must be a VerifyStoredEquityRecommendationGenerationUseCase."
            )
        if not isinstance(snapshots, EquityCandleEvidenceSnapshotStore):
            raise TypeError("snapshots must be an EquityCandleEvidenceSnapshotStore.")
        if not isinstance(lifecycles, EquityRecommendationLifecycleStore):
            raise TypeError("lifecycles must be an EquityRecommendationLifecycleStore.")
        if isinstance(max_writes, bool) or not isinstance(max_writes, int) or max_writes < 1:
            raise ValueError("max_writes must be an integer of at least 1.")
        self._issuances = issuances
        self._generations = generations
        self._snapshots = snapshots
        self._lifecycles = lifecycles
        self._max_writes = max_writes

    def execute(
        self, issuance_id: str, snapshot_ids: tuple[str, ...]
    ) -> EquityLifecycleRecordingResult:
        """Record the lifecycle events ``snapshot_ids`` produce, or raise a typed refusal."""
        if not isinstance(issuance_id, str) or not issuance_id:
            raise TypeError("issuance_id must be non-empty text.")
        if (
            not isinstance(snapshot_ids, tuple)
            or not snapshot_ids
            or not all(isinstance(item, str) and item for item in snapshot_ids)
        ):
            raise TypeError("snapshot_ids must be a non-empty tuple of non-empty text.")
        issuance = self._issuance(issuance_id)
        verified = self._verified(issuance.recommendation.generation_key)
        try:
            window = equity_issued_lifecycle_window(issuance.recommendation, verified.disposition)
        except InvalidEquityLifecycleWindowProvenanceError as exc:
            raise _refuse(_R.WINDOW_PROVENANCE_INVALID, str(exc)) from exc
        stored = self._stored(issuance, window)
        lifecycle = self._evaluate(verified, stored.lifecycle, snapshot_ids)
        new_events = lifecycle.events[stored.position :]
        if not new_events:
            return EquityLifecycleRecordingResult(
                EquityLifecycleRecordingOutcome.NO_NEW_EVENTS, stored, (), 0, False
            )
        authorization = _issue_lifecycle_append(
            issuance, window, stored.lifecycle.events, new_events
        )
        return self._write(issuance_id, authorization, new_events)

    # ------------------------------------------------------------------
    # Authoritative sources
    # ------------------------------------------------------------------

    def _issuance(self, issuance_id: str) -> EquityRecordedIssuance:
        try:
            issuance = self._issuances.load_issuance(issuance_id)
        except EquityRecommendationIssuanceUnavailableError as exc:
            raise _refuse(_R.AUTHORITY_UNAVAILABLE, str(exc)) from exc
        except InvalidStoredEquityRecommendationIssuanceAuthorityError as exc:
            raise _refuse(_R.STORED_AUTHORITY_INVALID, str(exc)) from exc
        if issuance is None:
            raise _refuse(_R.ISSUANCE_NOT_FOUND, f"No issuance is recorded under {issuance_id}.")
        return issuance

    def _verified(self, generation_key: str) -> EquityVerifiedRecommendationGeneration:
        try:
            verified = self._generations.execute(generation_key)
        except EquityRecommendationGenerationStoreUnavailableError as exc:
            raise _refuse(_R.GENERATION_STORE_UNAVAILABLE, str(exc)) from exc
        except IncompatibleEquityRecommendationRecordError as exc:
            raise _refuse(_R.STORED_GENERATION_INCOMPATIBLE, str(exc)) from exc
        except InvalidEquityRecommendationRecordError as exc:
            raise _refuse(_R.STORED_GENERATION_INVALID, str(exc)) from exc
        except EquityRecommendationSourceUnavailableError as exc:
            raise _refuse(_R.SOURCE_EVIDENCE_UNAVAILABLE, str(exc)) from exc
        except EquityRecommendationSourceVerificationError as exc:
            raise _refuse(_R.SOURCE_EVIDENCE_INVALID, str(exc)) from exc
        except EquityRecommendationReplayMismatchError as exc:
            raise _refuse(_R.REPLAY_MISMATCH, str(exc)) from exc
        if verified is None:
            raise _refuse(
                _R.GENERATION_NOT_FOUND, f"No stored disposition exists for {generation_key}."
            )
        return verified

    def _stored(
        self, issuance: EquityRecordedIssuance, window: EquityLifecycleWindow
    ) -> EquityRecordedLifecycle:
        issuance_id = issuance.recommendation.issuance_id
        try:
            stored = self._lifecycles.load_lifecycle(issuance_id)
        except EquityRecommendationLifecycleStoreUnavailableError as exc:
            raise _refuse(_R.STORE_UNAVAILABLE, str(exc)) from exc
        except InvalidStoredEquityRecommendationLifecycleError as exc:
            raise _refuse(_R.STORED_LIFECYCLE_INVALID, str(exc)) from exc
        except InvalidStoredEquityRecommendationIssuanceAuthorityError as exc:
            raise _refuse(_R.STORED_AUTHORITY_INVALID, str(exc)) from exc
        if stored is None:
            raise _refuse(
                _R.ISSUANCE_NOT_FOUND, f"The lifecycle store holds no issuance {issuance_id}."
            )
        if stored.issuance != issuance:
            raise _refuse(
                _R.STORED_AUTHORITY_INVALID,
                "The lifecycle store's parent issuance is not the authoritative one.",
            )
        if stored.lifecycle.window != window:
            raise _refuse(
                _R.WINDOW_PROVENANCE_INVALID,
                "The lifecycle store's window is not the authoritatively derived one.",
            )
        return stored

    # ------------------------------------------------------------------
    # Evaluation: the EI-5C rules, shared, and Core
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        verified: EquityVerifiedRecommendationGeneration,
        lifecycle: EquityRecommendationLifecycle,
        snapshot_ids: tuple[str, ...],
    ) -> EquityRecommendationLifecycle:
        disposition = verified.disposition
        proposal, candidate = disposition.proposal, disposition.candidate
        assert proposal is not None
        position: EquitySyntheticLifecycleSnapshotPosition | None = None
        try:
            for snapshot_id in snapshot_ids:
                snapshot = self._load(snapshot_id)
                _Synthetic._require_session(snapshot, proposal, candidate)
                if position is not None and snapshot_id == position.snapshot_id:
                    continue  # repeated within the call: idempotent, as in EI-5C
                _Synthetic._require_order(snapshot, position, lifecycle)
                start = lifecycle.cursor or lifecycle.window.eligible_from
                advance = lifecycle.advance(_observation(snapshot, lifecycle.window, start))
                lifecycle, _ = _record_revisions(advance.lifecycle, snapshot)
                position = EquitySyntheticLifecycleSnapshotPosition.of(snapshot)
        except EquitySyntheticLifecycleRefusedError as exc:
            raise _refuse(_R(exc.reason.value), str(exc)) from exc
        return lifecycle

    def _load(self, snapshot_id: str) -> EquityCandleEvidenceSnapshot:
        snapshot = self._snapshots.get_snapshot(snapshot_id)
        if snapshot is None:
            raise _refuse(
                _R.SNAPSHOT_NOT_FOUND, f"No EI-2B snapshot is stored under {snapshot_id}."
            )
        if snapshot.snapshot_id != snapshot_id:
            raise _refuse(
                _R.SNAPSHOT_IDENTITY_MISMATCH,
                f"The store answered {snapshot.snapshot_id} for {snapshot_id}.",
            )
        return snapshot

    # ------------------------------------------------------------------
    # The authorized append and commit reconciliation
    # ------------------------------------------------------------------

    def _write(
        self,
        issuance_id: str,
        authorization: VerifiedEquityRecommendationLifecycleAppend,
        new_events: tuple[EquityLifecycleEvent, ...],
    ) -> EquityLifecycleRecordingResult:
        prior = authorization.prior
        for write in range(1, self._max_writes + 1):
            try:
                written = self._lifecycles.append(authorization)
            except EquityRecommendationLifecycleStoreUnavailableError as exc:
                raise _refuse(
                    _R.STORE_UNAVAILABLE,
                    f"The lifecycle store is unavailable; nothing was written: {exc}",
                ) from exc
            except EquityRecommendationLifecycleCommitUnconfirmedError as exc:
                try:
                    found = self._lifecycles.load_lifecycle(issuance_id)
                except EquityRecommendationLifecycleStoreUnavailableError as reload_error:
                    raise EquityLifecycleRecordingUnresolvedError(
                        f"The commit is unconfirmed ({exc}) and the reload failed "
                        f"({reload_error}); nothing is claimed."
                    ) from reload_error
                if found is None:
                    raise EquityLifecycleRecordingUnresolvedError(
                        f"The commit is unconfirmed ({exc}) and the reload finds no issuance "
                        f"{issuance_id}; nothing is claimed."
                    ) from exc
                stored = equity_lifecycle_event_documents(found.lifecycle.events)
                expected = equity_lifecycle_event_documents(prior)
                if stored == expected:
                    continue  # not committed: write again, within the bound
                ours = expected + equity_lifecycle_event_documents(new_events)
                # Committed exactly after the same prior history -- possibly with later
                # events appended on top -- is confirmed; anything else never adopted it.
                outcome = (
                    EquityLifecycleRecordingOutcome.EQUIVALENT
                    if stored[: len(ours)] == ours
                    else EquityLifecycleRecordingOutcome.CONFLICT
                )
                return EquityLifecycleRecordingResult(outcome, found, new_events, write, True)
            return self._interpret(written, new_events, prior, write)
        raise EquityLifecycleRecordingUnresolvedError(
            f"No append of {self._max_writes} was confirmed; each reload found it not recorded."
        )

    @staticmethod
    def _interpret(
        written: EquityRecommendationLifecycleWrite,
        new_events: tuple[EquityLifecycleEvent, ...],
        prior: tuple[EquityLifecycleEvent, ...],
        writes: int,
    ) -> EquityLifecycleRecordingResult:
        if not isinstance(written, EquityRecommendationLifecycleWrite):
            raise InvalidEquityRecommendationLifecycleWriteError(
                "The lifecycle store must return an EquityRecommendationLifecycleWrite."
            )
        events = written.recorded.lifecycle.events
        if written.outcome in (_W.APPENDED, _W.EQUIVALENT) and events != prior + new_events:
            raise InvalidEquityRecommendationLifecycleWriteError(
                f"The store reported {written.outcome.value}, which its recorded history "
                "contradicts."
            )
        if written.outcome is _W.CONFLICT and events == prior + new_events:
            raise InvalidEquityRecommendationLifecycleWriteError(
                "The store reported CONFLICT for a history that holds exactly these events."
            )
        return EquityLifecycleRecordingResult(
            EquityLifecycleRecordingOutcome(written.outcome.value),
            written.recorded,
            new_events,
            writes,
            False,
        )
