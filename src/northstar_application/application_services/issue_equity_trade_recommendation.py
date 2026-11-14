"""MODELED recommendation issuance: the only operational issuer (EI-5D, ADR-024).

``IssueEquityTradeRecommendationUseCase.execute(generation_key, issuance_kind)``
turns one verified, version-2, PROPOSED generation disposition into one MODELED
issued recommendation (EI5-D1). It takes a generation key and nothing the
proposal already fixes: every issued field is copied from the re-verified
stored proposal (ADR-024, section 2).

The order of checks (EI-5 design, section 7.1)
----------------------------------------------
1. **Only MODELED.** ``ACTUAL`` is refused before any read (ADR-024, 3).
2. **Re-verify.** ``VerifyStoredEquityRecommendationGenerationUseCase``
   reloads the disposition and verifies it at levels A to C. No stored
   disposition, and invalid, incompatible, source-invalid or replay-mismatch
   records, are permanent refusals; the stored record is preserved and
   nothing is written (EI4-D11-B). An unavailable store or unavailable source
   evidence is a retryable refusal.
3. **PROPOSED only.** NOT_ELIGIBLE and UNRESOLVED dispositions are refused.
4. **Evaluator version 2 only.** A version-1 disposition stays verifiable and
   is never issuable; it is refused and preserved (ADR-025, N7; C-2).
5. **Authoritative history.** The frozen issued-history evidence must name
   the issuance authority (``northstar.equity-recommendation-issuance-
   authority/1``, D5D-3) and reproduce exactly from the authority's view at
   its recorded watermark (``session_view_at``). Fixture histories are
   therefore never issuable.
6. **Completeness at the boundary.** The session's greatest declared
   ``complete_through`` must be at least ``b = t - 1 us``, where ``t`` is the
   modeled issuance. Without that authoritative declaration the request is
   ineligible (D5D-4), never treated as an empty history.
7. **Derive, authorize, write.** The issued record and its ``eiris1``
   identity are derived; a content-bound authorization is issued; the writer
   is called.

Exactly ``b``, atomically
-------------------------
Version-2 generation already required ``complete_through == b`` (N10), and
the frozen evidence reproduces it. A declaration beyond ``b`` made since then
is at or after ``t`` -- ``PointInTime`` has microsecond resolution, so no
instant lies strictly between ``b`` and ``t`` -- and the writer refuses the new
issuance as BACKDATED_ISSUANCE inside its serialized write (EI5-D4). That
check, not this use case's reads, is authoritative: Application pre-checks
are never authoritative (ADR-024, section 7). An identical retry after a
later declaration is EQUIVALENT, because the writer matches the stored
identity before it checks backdating.

Outcomes and failures
---------------------
The result's outcome is the writer's (``EquityRecommendationIssuanceOutcome``):
ISSUED, EQUIVALENT, GENERATION_ALREADY_ISSUED, SESSION_STRATEGY_ALREADY_ISSUED,
BACKDATED_ISSUANCE or STORED_ISSUANCE_INVALID. A refusal before any write
raises EquityRecommendationIssuanceRefusedError, whose ``reason`` and
``retryable`` keep permanent and retryable refusals apart. An unconfirmed
commit is reconciled by an authoritative reload by identity, with bounded
retries (the EI4-D9-G pattern); when that cannot settle it,
EquityRecommendationIssuanceUnresolvedError says nothing is claimed.

Nothing here reads a clock, writes the EI-4D records, creates an order, a
fill, a position or a lifecycle event, or calls a provider, broker or network.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from northstar_core.recommendations.equity import EquityIssuanceKind

from northstar_application.application_services.equity_recommendation_generation_decoding import (
    IncompatibleEquityRecommendationRecordError,
    InvalidEquityRecommendationRecordError,
)
from northstar_application.application_services.equity_recommendation_generation_evidence import (
    EQUITY_RECOMMENDATION_EVALUATOR_V2,
    EQUITY_RECOMMENDATION_STRICTLY_BEFORE_ISSUANCE,
    EquityIssuedHistoryEvidence,
    InvalidEquityRecommendationEvidenceError,
    equity_history_boundary,
)
from northstar_application.application_services.equity_recommendation_generation_records import (
    EquityRecommendationDispositionKind,
)
from northstar_application.application_services.equity_recommendation_issuance_authorization import (  # noqa: E501
    _issue_issuance_write,
)
from northstar_application.application_services.equity_recommendation_issuance_records import (
    EQUITY_RECOMMENDATION_ISSUANCE_AUTHORITY_REFERENCE,
    EquityIssuedRecommendation,
    EquityRecordedIssuance,
    equity_issued_recommendation,
)
from northstar_application.application_services.verify_equity_recommendation_generation import (
    EquityRecommendationReplayMismatchError,
    EquityRecommendationSourceUnavailableError,
    EquityRecommendationSourceVerificationError,
    EquityVerifiedRecommendationGeneration,
    VerifyStoredEquityRecommendationGenerationUseCase,
)
from northstar_application.ports.equity_recommendation_generation_store import (
    EquityRecommendationGenerationStoreUnavailableError,
)
from northstar_application.ports.equity_recommendation_issuance import (
    EquityIssuanceWatermarkBeyondAuthorityError,
    EquityRecommendationIssuanceCommitUnconfirmedError,
    EquityRecommendationIssuanceOutcome,
    EquityRecommendationIssuanceReader,
    EquityRecommendationIssuanceUnavailableError,
    EquityRecommendationIssuanceWrite,
    EquityRecommendationIssuanceWriter,
    InvalidEquityRecommendationIssuanceWriteError,
)

EQUITY_RECOMMENDATION_ISSUANCE_WRITES = 3

_O = EquityRecommendationIssuanceOutcome
_ISSUED_OUTCOMES = frozenset({_O.ISSUED, _O.EQUIVALENT})
_CONFLICTS = frozenset({_O.GENERATION_ALREADY_ISSUED, _O.SESSION_STRATEGY_ALREADY_ISSUED})


class EquityRecommendationIssuanceRefusal(StrEnum):
    """Why an issuance request was refused before any write. Nothing was written."""

    ACTUAL_ISSUANCE_NOT_AUTHORIZED = "ACTUAL_ISSUANCE_NOT_AUTHORIZED"
    GENERATION_NOT_FOUND = "GENERATION_NOT_FOUND"
    GENERATION_STORE_UNAVAILABLE = "GENERATION_STORE_UNAVAILABLE"
    STORED_GENERATION_INVALID = "STORED_GENERATION_INVALID"
    STORED_GENERATION_INCOMPATIBLE = "STORED_GENERATION_INCOMPATIBLE"
    SOURCE_EVIDENCE_INVALID = "SOURCE_EVIDENCE_INVALID"
    SOURCE_EVIDENCE_UNAVAILABLE = "SOURCE_EVIDENCE_UNAVAILABLE"
    REPLAY_MISMATCH = "REPLAY_MISMATCH"
    DISPOSITION_NOT_PROPOSED = "DISPOSITION_NOT_PROPOSED"
    EVALUATOR_NOT_ISSUABLE = "EVALUATOR_NOT_ISSUABLE"
    HISTORY_NOT_AUTHORITATIVE = "HISTORY_NOT_AUTHORITATIVE"
    HISTORY_NOT_REPRODUCED = "HISTORY_NOT_REPRODUCED"
    AUTHORITATIVE_COMPLETENESS_NOT_DECLARED = "AUTHORITATIVE_COMPLETENESS_NOT_DECLARED"
    AUTHORITY_UNAVAILABLE = "AUTHORITY_UNAVAILABLE"


_RETRYABLE = frozenset(
    {
        EquityRecommendationIssuanceRefusal.GENERATION_STORE_UNAVAILABLE,
        EquityRecommendationIssuanceRefusal.SOURCE_EVIDENCE_UNAVAILABLE,
        EquityRecommendationIssuanceRefusal.AUTHORITY_UNAVAILABLE,
    }
)


class EquityRecommendationIssuanceRefusedError(ValueError):
    """An issuance request was refused before any write; ``reason`` says why.

    ``retryable`` is True only for an outage -- the generation store, the
    source evidence or the issuance authority could not answer. Every other
    refusal is permanent for that request.
    """

    def __init__(self, reason: EquityRecommendationIssuanceRefusal, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason

    @property
    def retryable(self) -> bool:
        """Return whether the same request may succeed once an outage clears."""
        return self.reason in _RETRYABLE


class EquityRecommendationIssuanceUnresolvedError(RuntimeError):
    """Raised when an unconfirmed issuance commit could not be settled; nothing is claimed."""


def _refuse(
    reason: EquityRecommendationIssuanceRefusal, message: str
) -> EquityRecommendationIssuanceRefusedError:
    return EquityRecommendationIssuanceRefusedError(reason, message)


_R = EquityRecommendationIssuanceRefusal


@dataclass(frozen=True, slots=True)
class EquityRecommendationIssuanceResult:
    """The outcome of one issuance request that reached the writer.

    ``issuance`` is the issued recommendation this request derived and
    offered. ``authoritative`` is the recorded issuance the outcome concerns:
    this one at its recording sequence for ISSUED and EQUIVALENT, the stored
    conflicting one for the two ``*_ALREADY_ISSUED`` outcomes, and None for
    BACKDATED_ISSUANCE and STORED_ISSUANCE_INVALID. ``writes`` counts writer
    calls; ``reconciled`` says whether an unconfirmed commit was settled by
    reload.
    """

    outcome: EquityRecommendationIssuanceOutcome
    issuance: EquityIssuedRecommendation
    authoritative: EquityRecordedIssuance | None
    writes: int
    reconciled: bool
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, EquityRecommendationIssuanceOutcome):
            raise ValueError("outcome must be an EquityRecommendationIssuanceOutcome.")
        if self.outcome in _ISSUED_OUTCOMES and (
            self.authoritative is None or self.authoritative.recommendation != self.issuance
        ):
            raise ValueError("An issued outcome names this issuance as recorded.")

    @property
    def is_issued(self) -> bool:
        """Return whether the offered issuance is recorded, by this write or an equal one."""
        return self.outcome in _ISSUED_OUTCOMES

    @property
    def recording_sequence(self) -> int | None:
        """Return the recording sequence of the recorded issuance, when it is recorded."""
        return self.authoritative.recording_sequence if self.is_issued else None  # type: ignore[union-attr]


class IssueEquityTradeRecommendationUseCase:
    """Issues one verified version-2 PROPOSED disposition as a MODELED recommendation.

    ``generations`` re-verifies the stored disposition (levels A to C).
    ``reader`` answers the authority's views; ``writer`` records the issuance.
    They may be one object, but this use case depends on each port only for
    what it does. See the module docstring.
    """

    def __init__(
        self,
        generations: VerifyStoredEquityRecommendationGenerationUseCase,
        reader: EquityRecommendationIssuanceReader,
        writer: EquityRecommendationIssuanceWriter,
        max_writes: int = EQUITY_RECOMMENDATION_ISSUANCE_WRITES,
    ) -> None:
        if not isinstance(generations, VerifyStoredEquityRecommendationGenerationUseCase):
            raise TypeError(
                "generations must be a VerifyStoredEquityRecommendationGenerationUseCase."
            )
        if not isinstance(reader, EquityRecommendationIssuanceReader):
            raise TypeError("reader must be an EquityRecommendationIssuanceReader.")
        if not isinstance(writer, EquityRecommendationIssuanceWriter):
            raise TypeError("writer must be an EquityRecommendationIssuanceWriter.")
        if isinstance(max_writes, bool) or not isinstance(max_writes, int) or max_writes < 1:
            raise ValueError("max_writes must be an integer of at least 1.")
        self._generations = generations
        self._reader = reader
        self._writer = writer
        self._max_writes = max_writes

    def execute(
        self,
        generation_key: str,
        issuance_kind: EquityIssuanceKind = EquityIssuanceKind.MODELED,
    ) -> EquityRecommendationIssuanceResult:
        """Issue the stored disposition under ``generation_key``, or raise a typed refusal."""
        if not isinstance(issuance_kind, EquityIssuanceKind):
            raise TypeError("issuance_kind must be an EquityIssuanceKind.")
        if issuance_kind is not EquityIssuanceKind.MODELED:
            raise _refuse(
                _R.ACTUAL_ISSUANCE_NOT_AUTHORIZED,
                "Only MODELED issuance is authorized; ACTUAL issuance is refused before any read.",
            )
        if not isinstance(generation_key, str) or not generation_key:
            raise TypeError("generation_key must be non-empty text.")
        verified = self._verified(generation_key)
        disposition = verified.disposition
        if disposition.kind is not EquityRecommendationDispositionKind.PROPOSED:
            raise _refuse(
                _R.DISPOSITION_NOT_PROPOSED,
                f"The disposition is {disposition.kind}: only PROPOSED is issuable (EI5-D1).",
            )
        if disposition.evaluator_version != EQUITY_RECOMMENDATION_EVALUATOR_V2:
            raise _refuse(
                _R.EVALUATOR_NOT_ISSUABLE,
                f"The disposition is from {disposition.evaluator_version}; only "
                f"{EQUITY_RECOMMENDATION_EVALUATOR_V2} records are issuable. It stays "
                "verifiable and is preserved.",
            )
        self._require_authoritative_history(verified)
        issuance = equity_issued_recommendation(disposition)
        authorization = _issue_issuance_write(verified, issuance)
        return self._write(issuance, authorization)

    # ------------------------------------------------------------------
    # Re-verification (EI5-D1)
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Authoritative history at t - 1 us (C-2, N10, D5D-3, D5D-4)
    # ------------------------------------------------------------------

    def _require_authoritative_history(
        self, verified: EquityVerifiedRecommendationGeneration
    ) -> None:
        inputs = verified.disposition.inputs
        frozen = inputs.issued_history
        boundary = equity_history_boundary(inputs.modeled_issued_at)
        if frozen.reference != EQUITY_RECOMMENDATION_ISSUANCE_AUTHORITY_REFERENCE:
            raise _refuse(
                _R.HISTORY_NOT_AUTHORITATIVE,
                f"The frozen issued history names {frozen.reference!r}, not the issuance "
                f"authority {EQUITY_RECOMMENDATION_ISSUANCE_AUTHORITY_REFERENCE!r}.",
            )
        exchange, day = frozen.exchange_code, frozen.trading_date
        try:
            then = self._reader.session_view_at(exchange, day, frozen.recording_watermark)
            now = self._reader.session_view(exchange, day)
        except EquityRecommendationIssuanceUnavailableError as exc:
            raise _refuse(_R.AUTHORITY_UNAVAILABLE, str(exc)) from exc
        except EquityIssuanceWatermarkBeyondAuthorityError as exc:
            raise _refuse(
                _R.HISTORY_NOT_REPRODUCED,
                f"The frozen evidence's watermark is beyond the authority's position: {exc}",
            ) from exc
        if then is None:
            raise _refuse(
                _R.HISTORY_NOT_REPRODUCED,
                f"The authority holds no declaration for the session at watermark "
                f"{frozen.recording_watermark}.",
            )
        try:
            reproduced = EquityIssuedHistoryEvidence.as_of_view(
                then, boundary, history_boundary=EQUITY_RECOMMENDATION_STRICTLY_BEFORE_ISSUANCE
            )
        except InvalidEquityRecommendationEvidenceError as exc:
            raise _refuse(
                _R.HISTORY_NOT_REPRODUCED,
                f"The authority's view at watermark {frozen.recording_watermark} does not "
                f"reproduce the frozen evidence: {exc}",
            ) from exc
        if reproduced != frozen:
            raise _refuse(
                _R.HISTORY_NOT_REPRODUCED,
                f"The authority's view at watermark {frozen.recording_watermark} differs from "
                "the frozen issued-history evidence.",
            )
        if now is None or now.complete_through.compare(boundary) < 0:
            raise _refuse(
                _R.AUTHORITATIVE_COMPLETENESS_NOT_DECLARED,
                f"The session is not authoritatively declared complete through {boundary} "
                "(t - 1 us); a missing declaration is never an empty history (D5D-4).",
            )
        if now.reference != EQUITY_RECOMMENDATION_ISSUANCE_AUTHORITY_REFERENCE:
            raise _refuse(
                _R.HISTORY_NOT_AUTHORITATIVE,
                f"The authority's current view names {now.reference!r}.",
            )

    # ------------------------------------------------------------------
    # The authorized write and commit reconciliation (EI4-D9-G pattern)
    # ------------------------------------------------------------------

    def _write(
        self, issuance: EquityIssuedRecommendation, authorization: object
    ) -> EquityRecommendationIssuanceResult:
        for write in range(1, self._max_writes + 1):
            try:
                written = self._writer.write_issuance(authorization)  # type: ignore[arg-type]
            except EquityRecommendationIssuanceUnavailableError as exc:
                raise _refuse(
                    _R.AUTHORITY_UNAVAILABLE,
                    f"The authority is unavailable; nothing was written: {exc}",
                ) from exc
            except EquityRecommendationIssuanceCommitUnconfirmedError as exc:
                try:
                    found = self._reader.load_issuance(issuance.issuance_id)
                except EquityRecommendationIssuanceUnavailableError as reload_error:
                    raise EquityRecommendationIssuanceUnresolvedError(
                        f"The commit is unconfirmed ({exc}) and the reload failed "
                        f"({reload_error}); nothing is claimed."
                    ) from reload_error
                if found is None:
                    continue
                if found.recommendation != issuance:
                    return EquityRecommendationIssuanceResult(
                        _O.STORED_ISSUANCE_INVALID,
                        issuance,
                        None,
                        write,
                        True,
                        "The reload found another record under this issuance identity; it is "
                        "preserved and never treated as equivalent.",
                    )
                return EquityRecommendationIssuanceResult(
                    _O.EQUIVALENT,
                    issuance,
                    found,
                    write,
                    True,
                    "The issuance is recorded (this commit or an equivalent one).",
                )
            return self._interpret(issuance, written, write)
        raise EquityRecommendationIssuanceUnresolvedError(
            f"No write of {self._max_writes} was confirmed; each reload found nothing recorded."
        )

    def _interpret(
        self,
        issuance: EquityIssuedRecommendation,
        written: EquityRecommendationIssuanceWrite,
        writes: int,
    ) -> EquityRecommendationIssuanceResult:
        if not isinstance(written, EquityRecommendationIssuanceWrite):
            raise InvalidEquityRecommendationIssuanceWriteError(
                "The writer must return an EquityRecommendationIssuanceWrite."
            )
        outcome, stored = written.outcome, written.stored
        if outcome in _ISSUED_OUTCOMES:
            assert stored is not None
            if stored.recommendation != issuance:
                # The writer claims this issuance but holds other content: never adopted.
                return EquityRecommendationIssuanceResult(
                    _O.STORED_ISSUANCE_INVALID,
                    issuance,
                    None,
                    writes,
                    False,
                    f"The writer reported {outcome.value}, which the recorded content contradicts.",
                )
            detail = (
                f"Recorded at sequence {stored.recording_sequence}."
                if outcome is _O.ISSUED
                else "The same issuance was already recorded; nothing was written."
            )
            return EquityRecommendationIssuanceResult(
                outcome, issuance, stored, writes, False, detail
            )
        if outcome in _CONFLICTS:
            assert stored is not None
            if stored.recommendation.issuance_id == issuance.issuance_id:
                raise InvalidEquityRecommendationIssuanceWriteError(
                    f"The writer reported {outcome.value} for this issuance's own identity."
                )
            return EquityRecommendationIssuanceResult(
                outcome,
                issuance,
                stored,
                writes,
                False,
                f"{stored.recommendation.issuance_id} is already recorded and stays "
                "authoritative; nothing was written (EI5-D1).",
            )
        detail = {
            _O.BACKDATED_ISSUANCE: (
                f"issued_at {issuance.issued_at} is at or before the session's declared "
                "completeness; nothing was written (EI5-D4)."
            ),
            _O.STORED_ISSUANCE_INVALID: (
                "Another record is stored under this issuance identity; it does not re-derive, "
                "is preserved and is never treated as equivalent."
            ),
        }[outcome]
        return EquityRecommendationIssuanceResult(outcome, issuance, None, writes, False, detail)
