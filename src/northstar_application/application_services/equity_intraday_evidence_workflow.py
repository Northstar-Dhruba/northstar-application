"""Synthetic intraday evidence ingestion and research-evidence issuance (EI-2C).

Two small use cases join EI-1C-3, EI-2A and EI-2B; neither adds storage,
validation or snapshot logic of its own.

Ingestion
---------
``IngestSyntheticEquityCandleEventsUseCase`` takes a finite batch of EI-2B
journal records -- ``EquityCandleObservationRecord`` and
``EquityCandleFinalityRecord`` carry every field a synthetic event needs, so
there is no separate event type -- and, in submitted order:

1. refuses an event that is not an EI-2B record, is not explicitly synthetic,
   is not from one of the configured synthetic sources, or (an observation)
   is not of the configured timeframe: REJECTED_INVALID. Malformed records
   cannot be built at all: the EI-2A/EI-2B constructors reject them;
2. appends it through the EI-2B journal port, whose rules decide the rest:
   APPENDED_OBSERVATION / APPENDED_FINALITY, IDEMPOTENT_RETRY for the same
   record under the same ingestion id (nothing written),
   REJECTED_INGESTION_CONFLICT for other content under that id, and
   REJECTED_SOURCE_IDENTITY_CONFLICT for a record its source identity's
   earlier entries contradict. Different OHLCV values under one identity are
   appended and their earlier entries named; nothing is chosen;
3. stops at a storage failure (EquityCandleEvidenceUnavailableError):
   FAILED_PERSISTENCE, then NOT_ATTEMPTED for every later event.

Partial progress, not batch atomicity
-------------------------------------
Each event is committed in its own journal transaction. A batch is not
atomic and nothing committed is ever rolled back. A rejected event writes
nothing and the batch continues -- rejection is a fact about that event. A
storage failure stops the batch and the report's ``resume_from`` names the
failing event, whose own outcome is unknown: it may have committed before the
failure was reported. Resubmitting the whole batch, or only the events from
``resume_from``, resolves it: committed events, that one included, come back
as IDEMPOTENT_RETRY and the rest are appended. Submitted with the same first-
time order, the journal ends identical -- same records, same sequences --
whatever the batch boundaries or interruptions.

Ordering
--------
Source publication, Northstar receipt, batch position, journal sequence and
snapshot issuance are different facts. Only the journal sequence orders
evidence, and no order chooses a revision. A finality statement may be
journaled before its candle (EI-2B): it is appended as APPENDED_FINALITY,
composes onto a candle only when both are visible to a snapshot, and never
enters a snapshot without one. No observation is fabricated.

Issuance
--------
``IssueEquityResearchEvidenceUseCase`` evaluates EI-1C-3 eligibility for a
listing, date and observation instant and, only for an eligible window,
issues through the EI-2B ``IssueEquityCandleEvidenceSnapshotUseCase`` --
whose latest-at-issuance check, bounded retry and provenance rules apply
unchanged. It reports ISSUED, WINDOW_INELIGIBLE, PROVENANCE_UNKNOWN or
STALE_AFTER_RETRIES. Storage failures and caller errors propagate. An issued
snapshot may be incomplete, unfinalized or conflicted; the result names
those intervals and never calls such a snapshot complete.

Nothing reads a clock, provider or network, and nothing here is a decision,
signal, recommendation or order.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import PointInTime, Timeframe

from northstar_application.application_services.equity_candle_evidence_journal import (
    EquityCandleAppendOutcome,
    EquityCandleFinalityRecord,
    EquityCandleObservationRecord,
)
from northstar_application.application_services.equity_candle_evidence_snapshot import (
    EquityCandleEvidenceSnapshot,
    IssueEquityCandleEvidenceSnapshotUseCase,
)
from northstar_application.application_services.equity_intraday_candle_validation import (
    EquityCandleCoverage,
    EquityCandleIntervalOutcome,
)
from northstar_application.application_services.equity_research_window import (
    EquityResearchWindowEvaluation,
    EvaluateEquityResearchWindowUseCase,
)
from northstar_application.ports.equity_candle_evidence_journal import (
    EquityCandleEvidenceJournal,
    EquityCandleEvidenceUnavailableError,
    EquityCandleIngestionConflictError,
    EquityCandleSnapshotStaleError,
    EquityCandleSourceIdentityConflictError,
)
from northstar_application.ports.equity_research_reference_provenance import (
    UnknownEquityReferenceProvenanceError,
)

_FIVE_MINUTES = Timeframe("5m")


class EquityCandleIngestionOutcome(StrEnum):
    """What ingestion did with one submitted event."""

    APPENDED_OBSERVATION = "APPENDED_OBSERVATION"
    APPENDED_FINALITY = "APPENDED_FINALITY"
    IDEMPOTENT_RETRY = "IDEMPOTENT_RETRY"
    REJECTED_INVALID = "REJECTED_INVALID"
    REJECTED_INGESTION_CONFLICT = "REJECTED_INGESTION_CONFLICT"
    REJECTED_SOURCE_IDENTITY_CONFLICT = "REJECTED_SOURCE_IDENTITY_CONFLICT"
    FAILED_PERSISTENCE = "FAILED_PERSISTENCE"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


_ING = EquityCandleIngestionOutcome
_JOURNALED = frozenset({_ING.APPENDED_OBSERVATION, _ING.APPENDED_FINALITY, _ING.IDEMPOTENT_RETRY})
_REJECTED = frozenset(
    {
        _ING.REJECTED_INVALID,
        _ING.REJECTED_INGESTION_CONFLICT,
        _ING.REJECTED_SOURCE_IDENTITY_CONFLICT,
    }
)


@dataclass(frozen=True, slots=True)
class EquityCandleIngestionResult:
    """The outcome for the event at ``position`` in the submitted batch.

    FAILED_PERSISTENCE means the append was not confirmed, not that it did not
    commit; resubmitting it resolves which.

    ``sequence`` is the journal entry holding the event -- the new one, or the
    original one for a retry -- and None when nothing is journaled under it.
    ``conflicting_values`` names earlier entries of the same source identity
    with other OHLCV values; both are kept.
    """

    position: int
    ingestion_id: str | None
    outcome: EquityCandleIngestionOutcome
    sequence: int | None
    conflicting_values: tuple[int, ...]
    detail: str

    @property
    def is_journaled(self) -> bool:
        """Return whether the event is durably in the journal."""
        return self.outcome in _JOURNALED

    @property
    def is_rejected(self) -> bool:
        """Return whether the event was refused and nothing was written for it."""
        return self.outcome in _REJECTED


@dataclass(frozen=True, slots=True)
class EquityCandleIngestionReport:
    """Every submitted event's outcome, in submitted order."""

    results: tuple[EquityCandleIngestionResult, ...]

    @property
    def interrupted(self) -> bool:
        """Return whether a storage failure stopped the batch."""
        return self.resume_from is not None

    @property
    def resume_from(self) -> int | None:
        """Return the position of the event a storage failure stopped at, if any."""
        for result in self.results:
            if result.outcome is _ING.FAILED_PERSISTENCE:
                return result.position
        return None

    def count(self, outcome: EquityCandleIngestionOutcome) -> int:
        """Return how many events had ``outcome``."""
        return sum(1 for result in self.results if result.outcome is outcome)

    def with_outcome(
        self, *outcomes: EquityCandleIngestionOutcome
    ) -> tuple[EquityCandleIngestionResult, ...]:
        """Return the results with any of ``outcomes``."""
        return tuple(result for result in self.results if result.outcome in outcomes)


class IngestSyntheticEquityCandleEventsUseCase:
    """Appends a finite batch of synthetic candle and finality events to the EI-2B journal."""

    def __init__(
        self,
        journal: EquityCandleEvidenceJournal,
        synthetic_source_ids: frozenset[str],
        timeframe: Timeframe = _FIVE_MINUTES,
    ) -> None:
        if not isinstance(journal, EquityCandleEvidenceJournal):
            raise TypeError("journal must be an EquityCandleEvidenceJournal.")
        if (
            not isinstance(synthetic_source_ids, frozenset)
            or not synthetic_source_ids
            or not all(isinstance(s, str) and s.strip() for s in synthetic_source_ids)
        ):
            raise ValueError("synthetic_source_ids must be a non-empty frozenset of source ids.")
        if not isinstance(timeframe, Timeframe):
            raise TypeError("timeframe must be a Timeframe.")
        self._journal = journal
        self._sources = synthetic_source_ids
        self._timeframe = timeframe

    def execute(self, events: Iterable[object]) -> EquityCandleIngestionReport:
        """Ingest ``events`` in order; see the module documentation for every outcome."""
        results: list[EquityCandleIngestionResult] = []
        stopped = False
        for position, event in enumerate(tuple(events)):
            ingestion_id = getattr(event, "ingestion_id", None)
            ingestion_id = ingestion_id if isinstance(ingestion_id, str) else None

            def result(
                outcome: EquityCandleIngestionOutcome,
                detail: str,
                sequence: int | None = None,
                conflicts: tuple[int, ...] = (),
                position: int = position,
                ingestion_id: str | None = ingestion_id,
            ) -> EquityCandleIngestionResult:
                return EquityCandleIngestionResult(
                    position, ingestion_id, outcome, sequence, conflicts, detail
                )

            if stopped:
                results.append(result(_ING.NOT_ATTEMPTED, "An earlier storage failure stopped."))
                continue
            problem = self._invalid(event)
            if problem is not None:
                results.append(result(_ING.REJECTED_INVALID, problem))
                continue
            try:
                appended = self._journal.append(event)  # type: ignore[arg-type]
            except EquityCandleIngestionConflictError as exc:
                results.append(result(_ING.REJECTED_INGESTION_CONFLICT, str(exc)))
            except EquityCandleSourceIdentityConflictError as exc:
                results.append(result(_ING.REJECTED_SOURCE_IDENTITY_CONFLICT, str(exc)))
            except EquityCandleEvidenceUnavailableError as exc:
                results.append(result(_ING.FAILED_PERSISTENCE, str(exc)))
                stopped = True
            else:
                entry = appended.entry
                if appended.outcome is EquityCandleAppendOutcome.ALREADY_APPENDED:
                    outcome = _ING.IDEMPOTENT_RETRY
                    detail = f"Already journaled as entry {entry.sequence}; nothing was written."
                elif isinstance(event, EquityCandleObservationRecord):
                    outcome = _ING.APPENDED_OBSERVATION
                    detail = f"Journaled as entry {entry.sequence}."
                else:
                    outcome = _ING.APPENDED_FINALITY
                    detail = (
                        f"Journaled as entry {entry.sequence}; it composes onto its candle only "
                        "when both are visible to a snapshot."
                    )
                results.append(result(outcome, detail, entry.sequence, appended.conflicting_values))
        return EquityCandleIngestionReport(tuple(results))

    def _invalid(self, event: object) -> str | None:
        if not isinstance(event, EquityCandleObservationRecord | EquityCandleFinalityRecord):
            return "Not an EI-2B candle observation or finality record."
        if not event.is_synthetic:
            return "Not explicitly synthetic; this workflow ingests synthetic evidence only."
        source_id = event.source_identity[0]
        if source_id not in self._sources:
            return f"Source {source_id!r} is not a configured synthetic source."
        if (
            isinstance(event, EquityCandleObservationRecord)
            and event.observation.bar.timeframe != self._timeframe
        ):
            return (
                f"Timeframe {event.observation.bar.timeframe} is not the configured "
                f"{self._timeframe}."
            )
        return None


class EquityResearchEvidenceOutcome(StrEnum):
    """What issuing research evidence for a listing, date and instant established."""

    ISSUED = "ISSUED"
    WINDOW_INELIGIBLE = "WINDOW_INELIGIBLE"
    PROVENANCE_UNKNOWN = "PROVENANCE_UNKNOWN"
    STALE_AFTER_RETRIES = "STALE_AFTER_RETRIES"


@dataclass(frozen=True, slots=True)
class EquityResearchEvidenceIssue:
    """The window evaluation and, when ISSUED, the immutable snapshot.

    An issued snapshot records what was known, complete or not; the interval
    properties name what keeps it from being complete.
    """

    outcome: EquityResearchEvidenceOutcome
    window: EquityResearchWindowEvaluation
    snapshot: EquityCandleEvidenceSnapshot | None
    detail: str

    def _ends(self, *outcomes: EquityCandleIntervalOutcome) -> tuple[PointInTime, ...]:
        if self.snapshot is None:
            return ()
        return tuple(i.closes_at for i in self.snapshot.intervals if i.outcome in outcomes)

    @property
    def coverage(self) -> EquityCandleCoverage | None:
        """Return the snapshot's coverage, or None when nothing was issued."""
        return None if self.snapshot is None else self.snapshot.coverage

    @property
    def is_complete(self) -> bool:
        """Return whether a snapshot was issued and every ended interval is acceptable."""
        return self.snapshot is not None and self.snapshot.is_complete

    @property
    def acceptable(self) -> tuple[PointInTime, ...]:
        """Return the interval ends whose candle is acceptable."""
        return self._ends(EquityCandleIntervalOutcome.ACCEPTABLE)

    @property
    def missing(self) -> tuple[PointInTime, ...]:
        """Return the ended interval ends with no candle visible."""
        return self._ends(EquityCandleIntervalOutcome.MISSING)

    @property
    def unfinalized(self) -> tuple[PointInTime, ...]:
        """Return the ended interval ends whose candle has no acceptable finality."""
        return self._ends(EquityCandleIntervalOutcome.UNFINALIZED)

    @property
    def conflicted(self) -> tuple[PointInTime, ...]:
        """Return the interval ends with conflicting values or unresolved revisions."""
        return self._ends(
            EquityCandleIntervalOutcome.CONFLICTING_VALUES,
            EquityCandleIntervalOutcome.UNRESOLVED_REVISIONS,
        )

    @property
    def not_ended(self) -> tuple[PointInTime, ...]:
        """Return the interval ends that had not ended at the observation instant."""
        return self._ends(EquityCandleIntervalOutcome.NOT_ENDED)


class IssueEquityResearchEvidenceUseCase:
    """Evaluates EI-1C-3 eligibility and, for an eligible window, issues through EI-2B."""

    def __init__(
        self,
        evaluate_window: EvaluateEquityResearchWindowUseCase,
        issue_snapshot: IssueEquityCandleEvidenceSnapshotUseCase,
    ) -> None:
        if not isinstance(evaluate_window, EvaluateEquityResearchWindowUseCase):
            raise TypeError("evaluate_window must be an EvaluateEquityResearchWindowUseCase.")
        if not isinstance(issue_snapshot, IssueEquityCandleEvidenceSnapshotUseCase):
            raise TypeError("issue_snapshot must be an IssueEquityCandleEvidenceSnapshotUseCase.")
        self._evaluate = evaluate_window
        self._issue = issue_snapshot

    def execute(
        self,
        listing: ListingReference,
        trading_date: date,
        as_of: PointInTime,
        issued_at: PointInTime,
    ) -> EquityResearchEvidenceIssue:
        """Issue the research evidence of ``listing`` on ``trading_date`` known at ``as_of``."""
        window = self._evaluate.execute(listing, trading_date, as_of)

        def outcome(
            kind: EquityResearchEvidenceOutcome,
            detail: str,
            snapshot: EquityCandleEvidenceSnapshot | None = None,
        ) -> EquityResearchEvidenceIssue:
            return EquityResearchEvidenceIssue(kind, window, snapshot, detail)

        if not window.is_eligible:
            return outcome(
                EquityResearchEvidenceOutcome.WINDOW_INELIGIBLE,
                f"{window.rejection}: {window.detail}",
            )
        try:
            snapshot = self._issue.execute(window, as_of, issued_at)
        except UnknownEquityReferenceProvenanceError as exc:
            return outcome(EquityResearchEvidenceOutcome.PROVENANCE_UNKNOWN, str(exc))
        except EquityCandleSnapshotStaleError as exc:
            return outcome(EquityResearchEvidenceOutcome.STALE_AFTER_RETRIES, str(exc))
        return outcome(
            EquityResearchEvidenceOutcome.ISSUED,
            f"Issued {snapshot.snapshot_id} at watermark {snapshot.watermark}: "
            f"{snapshot.coverage.value}.",
            snapshot,
        )
