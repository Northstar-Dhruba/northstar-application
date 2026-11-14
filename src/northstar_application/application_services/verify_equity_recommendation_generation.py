"""Verification of stored recommendation-generation records (EI-4D-A, EI4-D9-I).

A generation disposition is accepted -- before it is recorded, and whenever a
stored one is reloaded -- only when it passes three distinct levels:

A. **Document integrity.** Its canonical documents decode exactly and every
   fingerprint re-derives. EquityRecommendationGenerationDisposition cannot be
   constructed otherwise (equity_recommendation_generation_decoding), and an
   unsupported format or evaluator version fails closed.
B. **Source referential integrity.** The record's original CANDIDATE is
   re-established through the accepted EI-3C path -- the committed timeline,
   rehydrated whole (EI3-D13), its single CANDIDATE, and that candidate's own
   EI-2B snapshot -- by ``establish_equity_recommendation_candidate``, the
   same function generation uses. The re-established candidate evidence must
   equal the stored evidence exactly: strategy, configuration version, listing,
   date, decision instant and fingerprints, entry fingerprint and sequence,
   snapshot identity and manifest, signal bar, signal close, opening-range low,
   research window and venue reference. A later timeline entry, a correction
   included, never changes it, and nothing is relinked to newer evidence.
   A fingerprint verifying is never taken as proof that the source exists.
C. **Historical evaluation reproducibility.** Verification dispatches on the
   record's evaluator label (ADR-025, N1): it must be one of the closed
   table's two -- any other is IncompatibleEquityRecommendationRecordError --
   and its documents must be that evaluator's (N6), so a record is never
   "verified" by replaying it with an evaluator it does not name, and a
   version-1 record is never reinterpreted under version-2 rules. The pure
   ``assess_equity_recommendation_inputs`` -- and through it EI-4B Core -- is
   then re-run on the record's own frozen inputs, under their own version's
   history policy (as of ``t`` for version 1, strictly before it for version
   2): the stored policy set, listing
   eligibility, venue-day evidence and as-of issued-history evidence with its
   completeness and recording watermark. No reference source, venue source or
   history source is consulted. The replayed assessment document and
   fingerprint, and the proposal fingerprint, must equal the stored ones.

Passing A to C does not establish actual issuance (level D): a disposition is
never an issued recommendation (EI4-D9-H).

Failures are typed and never conflated: EI-3C or EI-2B storage that cannot
answer is EquityRecommendationSourceUnavailableError; missing, refused or
mismatched source evidence is EquityRecommendationSourceVerificationError; a
replay that differs is EquityRecommendationReplayMismatchError.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from northstar_application.application_services.equity_recommendation_generation_decoding import (
    require_paired_equity_recommendation_evaluator,
    require_supported_equity_recommendation_evaluator,
)
from northstar_application.application_services.equity_recommendation_generation_records import (
    EquityRecommendationGenerationDisposition,
)
from northstar_application.application_services.equity_trade_recommendation_eligibility import (
    EquityRecommendationEligibilityAudit,
    EquityRecommendationEligibilityReason,
)
from northstar_application.application_services.generate_equity_trade_recommendation import (
    EquityCandidateRefusal,
    assess_equity_recommendation_inputs,
    establish_equity_recommendation_candidate,
)
from northstar_application.ports.equity_candle_evidence_journal import (
    EquityCandleEvidenceSnapshotStore,
)
from northstar_application.ports.equity_intraday_research_timeline_store import (
    EquityIntradayResearchTimelineStore,
)
from northstar_application.ports.equity_recommendation_generation_store import (
    EquityRecommendationGenerationRecordReader,
)

_SOURCE_OUTAGES = frozenset(
    {
        EquityRecommendationEligibilityReason.RESEARCH_TIMELINE_UNAVAILABLE,
        EquityRecommendationEligibilityReason.EVIDENCE_SNAPSHOT_UNAVAILABLE,
    }
)


class EquityRecommendationSourceUnavailableError(RuntimeError):
    """Raised when EI-3C or EI-2B storage cannot answer; nothing is verified either way."""


class EquityRecommendationSourceVerificationError(ValueError):
    """Raised when a record's original candidate or snapshot is missing, refused or different."""

    def __init__(self, message: str, reason: EquityRecommendationEligibilityReason | None) -> None:
        super().__init__(message)
        self.reason = reason


class EquityRecommendationReplayMismatchError(ValueError):
    """Raised when re-running the assessment on the frozen inputs does not reproduce it."""


@dataclass(frozen=True, slots=True)
class EquityVerifiedRecommendationGeneration:
    """A disposition that passed document, source and reproducibility verification.

    ``replayed`` is the assessment re-run from the record's frozen inputs.
    ``observed_timeline_head`` is the head seen while re-establishing the
    candidate: diagnostics only. Verification never establishes issuance.
    """

    disposition: EquityRecommendationGenerationDisposition
    replayed: EquityRecommendationEligibilityAudit
    observed_timeline_head: str

    @property
    def document_integrity_verified(self) -> bool:
        """Return True: the documents decoded exactly and every fingerprint re-derived."""
        return True

    @property
    def source_integrity_verified(self) -> bool:
        """Return True: the original CANDIDATE and snapshot were re-established and match."""
        return True

    @property
    def evaluation_reproduced(self) -> bool:
        """Return True: the frozen inputs reproduce the stored assessment exactly."""
        return True

    @property
    def issuance_established(self) -> bool:
        """Return False: no verification here establishes an actual issuance."""
        return False


class EquityRecommendationGenerationVerifier:
    """Verifies a disposition's source evidence and historical reproducibility."""

    def __init__(
        self,
        timelines: EquityIntradayResearchTimelineStore,
        snapshots: EquityCandleEvidenceSnapshotStore,
    ) -> None:
        if not isinstance(timelines, EquityIntradayResearchTimelineStore):
            raise TypeError("timelines must be an EquityIntradayResearchTimelineStore.")
        if not isinstance(snapshots, EquityCandleEvidenceSnapshotStore):
            raise TypeError("snapshots must be an EquityCandleEvidenceSnapshotStore.")
        self._timelines = timelines
        self._snapshots = snapshots

    def verify(
        self, disposition: EquityRecommendationGenerationDisposition
    ) -> EquityVerifiedRecommendationGeneration:
        """Verify ``disposition`` at levels A to C, or raise the typed failure."""
        if not isinstance(disposition, EquityRecommendationGenerationDisposition):
            raise TypeError("disposition must be an EquityRecommendationGenerationDisposition.")
        # Dispatch on the recorded label: governed (else Incompatible) and paired with
        # the frozen inputs' version (else Invalid). Construction already enforced
        # both; they are re-checked here because replay depends on them.
        label = require_supported_equity_recommendation_evaluator(disposition.evaluator_version)
        request, inputs = disposition.request, disposition.inputs
        require_paired_equity_recommendation_evaluator(
            label, inputs.evaluator_version, "generation inputs"
        )
        found = establish_equity_recommendation_candidate(request, self._timelines, self._snapshots)
        if isinstance(found, EquityCandidateRefusal):
            if found.reason in _SOURCE_OUTAGES:
                raise EquityRecommendationSourceUnavailableError(found.detail)
            raise EquityRecommendationSourceVerificationError(
                f"The record's source does not re-establish ({found.reason.value}): {found.detail}",
                found.reason,
            )
        if found.provenance != inputs.candidate:
            differing = [
                f.name
                for f in fields(inputs.candidate)
                if getattr(found.provenance, f.name) != getattr(inputs.candidate, f.name)
            ]
            raise EquityRecommendationSourceVerificationError(
                "The stored candidate evidence is not the authoritative CANDIDATE's: "
                + ", ".join(differing)
                + " differ.",
                None,
            )
        replayed = assess_equity_recommendation_inputs(
            request, inputs, observed_timeline_head=found.observed_timeline_head
        )
        if replayed.evaluator_version != label:
            raise EquityRecommendationReplayMismatchError(
                f"The replay ran under {replayed.evaluator_version}, not the recorded {label}."
            )
        if (
            replayed.assessment_document() != disposition.assessment.document
            or replayed.assessment_fingerprint != disposition.assessment_fingerprint
        ):
            raise EquityRecommendationReplayMismatchError(
                "Re-running the assessment on the stored frozen inputs does not reproduce "
                f"{disposition.assessment_fingerprint}."
            )
        replayed_proposal = None if replayed.proposal is None else replayed.proposal.fingerprint
        if replayed_proposal != disposition.proposal_fingerprint:
            raise EquityRecommendationReplayMismatchError(
                "Re-running the assessment does not reproduce the stored proposal."
            )
        return EquityVerifiedRecommendationGeneration(
            disposition, replayed, found.observed_timeline_head
        )


class VerifyStoredEquityRecommendationGenerationUseCase:
    """Reloads a stored disposition and verifies it before anything treats it as authoritative.

    Store failures propagate typed: an unavailable store, and a stored row
    that does not decode -- InvalidEquityRecommendationRecordError, or its
    Incompatible subclass for an unsupported format or evaluator -- raise.
    Verification failures raise the verifier's typed errors.
    """

    def __init__(
        self,
        store: EquityRecommendationGenerationRecordReader,
        verifier: EquityRecommendationGenerationVerifier,
    ) -> None:
        if not isinstance(store, EquityRecommendationGenerationRecordReader):
            raise TypeError("store must be an EquityRecommendationGenerationRecordReader.")
        if not isinstance(verifier, EquityRecommendationGenerationVerifier):
            raise TypeError("verifier must be an EquityRecommendationGenerationVerifier.")
        self._store = store
        self._verifier = verifier

    def execute(self, generation_key: str) -> EquityVerifiedRecommendationGeneration | None:
        """Return the verified stored disposition for ``generation_key``, or None if none."""
        stored = self._store.load_disposition(generation_key)
        if stored is None:
            return None
        if stored.generation_key != generation_key:
            raise EquityRecommendationSourceVerificationError(
                "The store returned a disposition for another generation key.", None
            )
        return self._verifier.verify(stored)
