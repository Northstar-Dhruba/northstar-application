"""Immutable as-issued equity candle evidence snapshots (EI-2B).

A snapshot freezes exactly which journaled evidence one EI-2A validation used
for one listing, trading date and observation instant, and what EI-2A found.
It is a record of what was known when it was issued; it is not a research
decision, signal, recommendation or order, and an incomplete snapshot is as
legitimate as a complete one.

Dual visibility
---------------
An observation contributes to a snapshot only when both hold:

1. it is in the journal at the snapshot's watermark -- its sequence is at or
   below it; and
2. EI-2A accepts it at the observation instant: received by then, its interval
   ended, finality from its own source, event and revision confirmed by then
   and approved by the recorded finality policy, for the right listing,
   timeframe, currency and research window.

A finality statement is composed onto its candle only when it, too, is within
the watermark and was received by the observation instant. So evidence
appended after issuance -- however early its source, receipt or confirmation
times -- is above the watermark and can never enter the snapshot, and reading
the journal again at the same watermark returns the same evidence.

What is recorded
----------------
The listing, trading date, timeframe and currency; the research window with
its policy version and venue and applicability references; the observation
instant and journal watermark; the complete EI-2A finality policy; every
in-window observation and finality entry at the watermark, record for record,
with its fingerprint; each observation's EI-2A finding and composed finality
entries; every interval's outcome with its observations and acceptable ones;
and the coverage.

Provenance
----------
The provenance of the window's venue and applicability references is
classified by an EquityResearchReferenceProvenanceSource from the sources that
produced them, never from how a reference is spelled, and is recorded in the
manifest. A reference whose provenance is unknown refuses issuance.
``evidence_class`` is derived, never supplied: it is PRODUCTION only when both
references are ACCEPTED_PRODUCTION, the finality policy does not permit
synthetic evidence and no journaled record is synthetic; anything else is
SYNTHETIC, so synthetic evidence cannot be promoted. With no accepted
production listing-applicability source today, no eligible window is
production-backed.

Fingerprints
------------
``ecm1-sha256:`` is SHA-256 over the canonical JSON manifest above (no
issuance time). ``ecs1-sha256:`` -- the snapshot identity -- binds the
manifest fingerprint and ``issued_at``. Both are recomputed whenever a
snapshot is built or decoded, so stored data that disagrees with its
fingerprint fails closed.

Issuance and replay
-------------------
IssueEquityCandleEvidenceSnapshotUseCase refuses an ineligible research
window or unknown provenance, reads the journal at one watermark, runs EI-2A,
and hands the snapshot to the store. In its write transaction the store
requires the journal at that watermark to hold exactly the listed evidence,
and the window's evidence at the journal's current watermark to be the same
-- so a snapshot is the latest committed evidence when it is recorded. If the
window gained evidence after the watermark was read, the store refuses it as
stale and the use case starts again: it reads a new watermark, re-reads and
re-validates the evidence and rebuilds the snapshot, at most
``EQUITY_CANDLE_SNAPSHOT_ISSUE_ATTEMPTS`` times in all, then raises
EquityCandleSnapshotStaleError. Nothing waits or polls.

ReplayEquityCandleEvidenceSnapshotUseCase reads a
stored snapshot and re-derives it from the journal at its watermark under
supplied references; when a reference or policy is not the recorded one it
reports REFERENCE_UNAVAILABLE rather than substituting another. Nothing reads
a clock: every instant is supplied.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import (
    Currency,
    ExchangeCode,
    PointInTime,
    Symbol,
    Timeframe,
)

from northstar_application.application_services.equity_candle_evidence_journal import (
    EquityCandleFinalityRecord,
    EquityCandleJournalWindow,
    EquityCandleObservationRecord,
    canonical_equity_candle_json,
    compose_equity_candle_observations,
    equity_candle_in_window,
    equity_candle_record_document,
    equity_candle_record_fingerprint,
    equity_candle_record_from_document,
)
from northstar_application.application_services.equity_intraday_candle_validation import (
    NO_APPROVED_EQUITY_CANDLE_FINALITY,
    EquityCandleCoverage,
    EquityCandleFinalityPolicy,
    EquityCandleFinding,
    EquityCandleIntervalOutcome,
    ValidateEquityIntradayCandlesUseCase,
)
from northstar_application.application_services.equity_research_window import (
    NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW,
    EquityResearchWindowEvaluation,
    EquityResearchWindowPolicy,
)
from northstar_application.ports.equity_candle_evidence_journal import (
    EquityCandleEvidenceJournal,
    EquityCandleEvidenceSnapshotStore,
    EquityCandleSnapshotEvidenceError,
    EquityCandleSnapshotStaleError,
)
from northstar_application.ports.equity_research_reference_provenance import (
    EquityReferenceProvenanceKind,
    EquityResearchReferenceProvenanceSource,
    EquityResearchWindowProvenance,
    UnknownEquityReferenceProvenanceError,
)

EQUITY_CANDLE_MANIFEST_FINGERPRINT_PREFIX = "ecm1-sha256:"
EQUITY_CANDLE_SNAPSHOT_ID_PREFIX = "ecs1-sha256:"
EQUITY_CANDLE_SNAPSHOT_ISSUE_ATTEMPTS = 3
_FORMAT = "northstar.equity-candle-evidence-snapshot/1"


class InvalidEquityCandleEvidenceSnapshotError(ValueError):
    """Raised when a snapshot is malformed or internally inconsistent."""


class EquityCandleSnapshotRefusedError(ValueError):
    """Raised when no snapshot may be issued, such as for an ineligible research window."""


def _fail(message: str) -> None:
    raise InvalidEquityCandleEvidenceSnapshotError(message)


def _sequences(value: object, subject: str, watermark: int | None = None) -> tuple[int, ...]:
    if not isinstance(value, tuple) or not all(
        isinstance(s, int) and not isinstance(s, bool) and s >= 1 for s in value
    ):
        _fail(f"{subject} must be a tuple of journal sequences.")
    sequences: tuple[int, ...] = value  # type: ignore[assignment]
    if list(sequences) != sorted(set(sequences)):
        _fail(f"{subject} must be ascending without repeats.")
    if watermark is not None and any(s > watermark for s in sequences):
        _fail(f"{subject} must be at or below the watermark.")
    return sequences


class EquityCandleEvidenceClass(StrEnum):
    """Whether a snapshot rests on any synthetic input."""

    SYNTHETIC = "SYNTHETIC"
    PRODUCTION = "PRODUCTION"


@dataclass(frozen=True, slots=True)
class EquityCandleSnapshotObservation:
    """One journaled observation in a snapshot, with its EI-2A finding."""

    sequence: int
    record: EquityCandleObservationRecord
    finding: EquityCandleFinding
    finality_sequences: tuple[int, ...]

    def __post_init__(self) -> None:
        _sequences((self.sequence,), "Snapshot observation sequence")
        if not isinstance(self.record, EquityCandleObservationRecord):
            _fail("Snapshot observation record must be an EquityCandleObservationRecord.")
        if not isinstance(self.finding, EquityCandleFinding):
            _fail("Snapshot observation finding must be an EquityCandleFinding.")
        _sequences(self.finality_sequences, "Snapshot observation finality_sequences")


@dataclass(frozen=True, slots=True)
class EquityCandleSnapshotFinality:
    """One journaled finality statement in a snapshot."""

    sequence: int
    record: EquityCandleFinalityRecord

    def __post_init__(self) -> None:
        _sequences((self.sequence,), "Snapshot finality sequence")
        if not isinstance(self.record, EquityCandleFinalityRecord):
            _fail("Snapshot finality record must be an EquityCandleFinalityRecord.")


@dataclass(frozen=True, slots=True)
class EquityCandleSnapshotInterval:
    """One expected interval's EI-2A outcome, naming its observations by sequence."""

    opens_at: PointInTime
    closes_at: PointInTime
    outcome: EquityCandleIntervalOutcome
    observation_sequences: tuple[int, ...]
    acceptable_sequences: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.opens_at, PointInTime) or not isinstance(
            self.closes_at, PointInTime
        ):
            _fail("Snapshot interval bounds must be PointInTime values.")
        if self.opens_at.compare(self.closes_at) >= 0:
            _fail("Snapshot interval must open before it closes.")
        if not isinstance(self.outcome, EquityCandleIntervalOutcome):
            _fail("Snapshot interval outcome must be an EquityCandleIntervalOutcome.")
        observed = _sequences(self.observation_sequences, "Snapshot interval observations")
        acceptable = _sequences(self.acceptable_sequences, "Snapshot interval acceptable")
        if not set(acceptable) <= set(observed):
            _fail("Snapshot interval acceptable observations must be among its observations.")
        if bool(acceptable) != (self.outcome is EquityCandleIntervalOutcome.ACCEPTABLE):
            _fail("Only an ACCEPTABLE interval names acceptable observations, and it must.")


def _coverage(intervals: tuple[EquityCandleSnapshotInterval, ...]) -> EquityCandleCoverage:
    """Return EI-2A's coverage of these interval outcomes."""
    ended = [i for i in intervals if i.outcome is not EquityCandleIntervalOutcome.NOT_ENDED]
    if not ended:
        return EquityCandleCoverage.NO_INTERVALS_ENDED
    if any(i.outcome is not EquityCandleIntervalOutcome.ACCEPTABLE for i in ended):
        return EquityCandleCoverage.INCOMPLETE_AS_OF
    if len(ended) == len(intervals):
        return EquityCandleCoverage.WINDOW_COMPLETE
    return EquityCandleCoverage.COMPLETE_AS_OF


def _digest(prefix: str, document: dict[str, Any]) -> str:
    encoded = canonical_equity_candle_json(document).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class EquityCandleEvidenceSnapshot:
    """The immutable evidence one EI-2A validation used, and its outcomes, as issued.

    ``evidence_class``, ``manifest_fingerprint`` and ``snapshot_id`` are
    derived from the content and cannot be supplied.
    """

    listing: ListingReference
    trading_date: date
    timeframe: Timeframe
    currency: Currency
    window_policy_version: str
    window_opens_at: PointInTime
    window_closes_at: PointInTime
    venue_reference: str
    applicability_reference: str
    venue_provenance: EquityReferenceProvenanceKind
    applicability_provenance: EquityReferenceProvenanceKind
    as_of: PointInTime
    watermark: int
    finality_policy: EquityCandleFinalityPolicy
    observations: tuple[EquityCandleSnapshotObservation, ...]
    finalities: tuple[EquityCandleSnapshotFinality, ...]
    intervals: tuple[EquityCandleSnapshotInterval, ...]
    unmatched_sequences: tuple[int, ...]
    coverage: EquityCandleCoverage
    issued_at: PointInTime
    evidence_class: EquityCandleEvidenceClass = field(init=False)
    manifest_fingerprint: str = field(init=False)
    snapshot_id: str = field(init=False)

    def __post_init__(self) -> None:
        self._check_scalars()
        self._check_evidence()
        self._check_outcomes()
        production = (
            self.venue_provenance is EquityReferenceProvenanceKind.ACCEPTED_PRODUCTION
            and self.applicability_provenance is EquityReferenceProvenanceKind.ACCEPTED_PRODUCTION
            and not self.finality_policy.permits_synthetic
            and not any(o.record.is_synthetic for o in self.observations)
            and not any(f.record.is_synthetic for f in self.finalities)
        )
        object.__setattr__(
            self,
            "evidence_class",
            EquityCandleEvidenceClass.PRODUCTION
            if production
            else EquityCandleEvidenceClass.SYNTHETIC,
        )
        fingerprint = _digest(EQUITY_CANDLE_MANIFEST_FINGERPRINT_PREFIX, self.manifest())
        object.__setattr__(self, "manifest_fingerprint", fingerprint)
        object.__setattr__(
            self,
            "snapshot_id",
            _digest(
                EQUITY_CANDLE_SNAPSHOT_ID_PREFIX,
                {
                    "format": _FORMAT,
                    "manifest_fingerprint": fingerprint,
                    "issued_at": self.issued_at.value,
                },
            ),
        )

    def _check_scalars(self) -> None:
        if not isinstance(self.listing, ListingReference):
            _fail("Snapshot listing must be a ListingReference.")
        if not isinstance(self.trading_date, date) or type(self.trading_date) is not date:
            _fail("Snapshot trading_date must be a plain date.")
        if not isinstance(self.timeframe, Timeframe):
            _fail("Snapshot timeframe must be a Timeframe.")
        if not isinstance(self.currency, Currency):
            _fail("Snapshot currency must be a Currency.")
        for name in ("window_policy_version", "venue_reference", "applicability_reference"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                _fail(f"Snapshot {name} must be a non-empty string.")
        for name in ("venue_provenance", "applicability_provenance"):
            if not isinstance(getattr(self, name), EquityReferenceProvenanceKind):
                _fail(f"Snapshot {name} must be an EquityReferenceProvenanceKind.")
        for name in ("window_opens_at", "window_closes_at", "as_of", "issued_at"):
            if not isinstance(getattr(self, name), PointInTime):
                _fail(f"Snapshot {name} must be a PointInTime.")
        if self.window_opens_at.compare(self.window_closes_at) >= 0:
            _fail("Snapshot research window must open before it closes.")
        if self.issued_at.compare(self.as_of) < 0:
            _fail("A snapshot cannot be issued before its observation instant.")
        if not isinstance(self.watermark, int) or isinstance(self.watermark, bool):
            _fail("Snapshot watermark must be an integer.")
        if self.watermark < 0:
            _fail("Snapshot watermark cannot be negative.")
        if not isinstance(self.finality_policy, EquityCandleFinalityPolicy):
            _fail("Snapshot finality_policy must be an EquityCandleFinalityPolicy.")
        if not isinstance(self.coverage, EquityCandleCoverage):
            _fail("Snapshot coverage must be an EquityCandleCoverage.")

    def _check_evidence(self) -> None:
        for name, kind in (
            ("observations", EquityCandleSnapshotObservation),
            ("finalities", EquityCandleSnapshotFinality),
        ):
            entries = getattr(self, name)
            if not isinstance(entries, tuple) or not all(isinstance(e, kind) for e in entries):
                _fail(f"Snapshot {name} must be a tuple of {kind.__name__}.")
            _sequences(tuple(e.sequence for e in entries), f"Snapshot {name}", self.watermark)
        for observation in self.observations:
            if not equity_candle_in_window(
                observation.record,
                self.listing,
                self.timeframe,
                self.window_opens_at,
                self.window_closes_at,
            ):
                _fail(f"Snapshot observation {observation.sequence} is not in the window.")
        identities = {o.record.source_identity for o in self.observations}
        for finality in self.finalities:
            if finality.record.source_identity not in identities:
                _fail(f"Snapshot finality {finality.sequence} confirms no listed observation.")
        for observation in self.observations:
            visible = tuple(
                f.sequence
                for f in self.finalities
                if f.record.source_identity == observation.record.source_identity
                and f.record.received_at.compare(self.as_of) <= 0
            )
            if observation.finality_sequences != visible:
                _fail(
                    f"Snapshot observation {observation.sequence} must compose exactly the "
                    "finality statements visible at the observation instant."
                )

    def _check_outcomes(self) -> None:
        if not isinstance(self.intervals, tuple) or not self.intervals:
            _fail("Snapshot intervals must be a non-empty tuple.")
        if not all(isinstance(i, EquityCandleSnapshotInterval) for i in self.intervals):
            _fail("Snapshot intervals must be EquityCandleSnapshotInterval values.")
        ends = [i.closes_at for i in self.intervals]
        for earlier, later in zip(ends, ends[1:], strict=False):
            if earlier.compare(later) >= 0:
                _fail("Snapshot intervals must be in ascending order.")
        by_sequence = {o.sequence: o for o in self.observations}
        placed: list[int] = list(_sequences(self.unmatched_sequences, "Snapshot unmatched"))
        for interval in self.intervals:
            for sequence in interval.observation_sequences:
                observation = by_sequence.get(sequence)
                if observation is None or (
                    observation.record.observation.bar.point_in_time != interval.closes_at
                ):
                    _fail(f"Snapshot interval {interval.closes_at} names entry {sequence} wrongly.")
            placed.extend(interval.observation_sequences)
        if sorted(placed) != sorted(by_sequence):
            _fail("Every snapshot observation must be in exactly one interval or unmatched.")
        if self.coverage is not _coverage(self.intervals):
            _fail("Snapshot coverage disagrees with its interval outcomes.")

    @property
    def is_complete(self) -> bool:
        """Return whether every interval ended by the observation instant is acceptable."""
        return self.coverage in (
            EquityCandleCoverage.COMPLETE_AS_OF,
            EquityCandleCoverage.WINDOW_COMPLETE,
        )

    @property
    def observation_sequences(self) -> tuple[int, ...]:
        """Return the journal sequences of every listed observation."""
        return tuple(o.sequence for o in self.observations)

    @property
    def finality_sequences(self) -> tuple[int, ...]:
        """Return the journal sequences of every listed finality statement."""
        return tuple(f.sequence for f in self.finalities)

    def manifest(self) -> dict[str, Any]:
        """Return the canonical manifest the fingerprint is computed over."""
        policy = self.finality_policy
        return {
            "format": _FORMAT,
            "listing": {
                "symbol": self.listing.symbol.value,
                "exchange_code": self.listing.exchange_code.value,
            },
            "trading_date": self.trading_date.isoformat(),
            "timeframe": self.timeframe.value,
            "currency": self.currency.value,
            "window": {
                "policy_version": self.window_policy_version,
                "opens_after": self.window_opens_at.value,
                "closes_at": self.window_closes_at.value,
                "venue_reference": self.venue_reference,
                "applicability_reference": self.applicability_reference,
                "venue_provenance": self.venue_provenance.value,
                "applicability_provenance": self.applicability_provenance.value,
            },
            "as_of": self.as_of.value,
            "watermark": self.watermark,
            "finality_policy": {
                "version": policy.version,
                "approved_source_ids": sorted(policy.approved_source_ids),
                "permits_synthetic": policy.permits_synthetic,
            },
            "evidence_class": self.evidence_class.value,
            "observations": [
                {
                    "sequence": o.sequence,
                    "record": equity_candle_record_document(o.record),
                    "record_fingerprint": equity_candle_record_fingerprint(o.record),
                    "finding": o.finding.value,
                    "finality_sequences": list(o.finality_sequences),
                }
                for o in self.observations
            ],
            "finalities": [
                {
                    "sequence": f.sequence,
                    "record": equity_candle_record_document(f.record),
                    "record_fingerprint": equity_candle_record_fingerprint(f.record),
                }
                for f in self.finalities
            ],
            "intervals": [
                {
                    "opens_at": i.opens_at.value,
                    "closes_at": i.closes_at.value,
                    "outcome": i.outcome.value,
                    "observation_sequences": list(i.observation_sequences),
                    "acceptable_sequences": list(i.acceptable_sequences),
                }
                for i in self.intervals
            ],
            "unmatched_sequences": list(self.unmatched_sequences),
            "coverage": self.coverage.value,
        }

    def document(self) -> dict[str, Any]:
        """Return the canonical stored document: identity, fingerprint, issuance and manifest."""
        return {
            "snapshot_id": self.snapshot_id,
            "manifest_fingerprint": self.manifest_fingerprint,
            "issued_at": self.issued_at.value,
            "manifest": self.manifest(),
        }


def equity_candle_snapshot_json(snapshot: EquityCandleEvidenceSnapshot) -> str:
    """Return ``snapshot``'s stored document as canonical JSON."""
    return canonical_equity_candle_json(snapshot.document())


def _text(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("Encoded snapshot values must be text.")
    return value


def _ints(values: Any) -> tuple[int, ...]:
    if not isinstance(values, list) or not all(
        isinstance(v, int) and not isinstance(v, bool) for v in values
    ):
        raise TypeError("Encoded snapshot sequences must be a list of integers.")
    return tuple(values)


def _observation_record(document: Any) -> EquityCandleObservationRecord:
    record = equity_candle_record_from_document(document)
    if not isinstance(record, EquityCandleObservationRecord):
        raise ValueError("Snapshot observation entry is not an observation record.")
    return record


def _finality_record(document: Any) -> EquityCandleFinalityRecord:
    record = equity_candle_record_from_document(document)
    if not isinstance(record, EquityCandleFinalityRecord):
        raise ValueError("Snapshot finality entry is not a finality record.")
    return record


def equity_candle_snapshot_from_document(document: Any) -> EquityCandleEvidenceSnapshot:
    """Rebuild a snapshot from its stored document, re-verifying every derived fact.

    The document must re-encode exactly, so a changed value, fingerprint or
    identity raises ValueError, TypeError or KeyError; nothing is repaired.
    """
    if not isinstance(document, dict):
        raise TypeError("A snapshot document must be an object.")
    manifest = document["manifest"]
    if not isinstance(manifest, dict) or manifest.get("format") != _FORMAT:
        raise ValueError("Not an equity candle evidence snapshot manifest.")
    window, policy, listing = manifest["window"], manifest["finality_policy"], manifest["listing"]
    approved = policy["approved_source_ids"]
    if not isinstance(approved, list):
        raise TypeError("Encoded approved sources must be a list.")
    permits = policy["permits_synthetic"]
    watermark = manifest["watermark"]
    if not isinstance(permits, bool) or not isinstance(watermark, int):
        raise TypeError("Encoded snapshot policy flag or watermark has the wrong type.")
    snapshot = EquityCandleEvidenceSnapshot(
        listing=ListingReference(
            Symbol(_text(listing["symbol"])), ExchangeCode(_text(listing["exchange_code"]))
        ),
        trading_date=date.fromisoformat(_text(manifest["trading_date"])),
        timeframe=Timeframe(_text(manifest["timeframe"])),
        currency=Currency(_text(manifest["currency"])),
        window_policy_version=_text(window["policy_version"]),
        window_opens_at=PointInTime(_text(window["opens_after"])),
        window_closes_at=PointInTime(_text(window["closes_at"])),
        venue_reference=_text(window["venue_reference"]),
        applicability_reference=_text(window["applicability_reference"]),
        venue_provenance=EquityReferenceProvenanceKind(_text(window["venue_provenance"])),
        applicability_provenance=EquityReferenceProvenanceKind(
            _text(window["applicability_provenance"])
        ),
        as_of=PointInTime(_text(manifest["as_of"])),
        watermark=watermark,
        finality_policy=EquityCandleFinalityPolicy(
            _text(policy["version"]), frozenset(_text(s) for s in approved), permits
        ),
        observations=tuple(
            EquityCandleSnapshotObservation(
                item["sequence"],
                _observation_record(item["record"]),
                EquityCandleFinding(_text(item["finding"])),
                _ints(item["finality_sequences"]),
            )
            for item in manifest["observations"]
        ),
        finalities=tuple(
            EquityCandleSnapshotFinality(item["sequence"], _finality_record(item["record"]))
            for item in manifest["finalities"]
        ),
        intervals=tuple(
            EquityCandleSnapshotInterval(
                PointInTime(_text(item["opens_at"])),
                PointInTime(_text(item["closes_at"])),
                EquityCandleIntervalOutcome(_text(item["outcome"])),
                _ints(item["observation_sequences"]),
                _ints(item["acceptable_sequences"]),
            )
            for item in manifest["intervals"]
        ),
        unmatched_sequences=_ints(manifest["unmatched_sequences"]),
        coverage=EquityCandleCoverage(_text(manifest["coverage"])),
        issued_at=PointInTime(_text(document["issued_at"])),
    )
    if snapshot.document() != document:
        raise ValueError(
            "Snapshot document does not match its re-derived fingerprint, identity or "
            "canonical content."
        )
    return snapshot


def require_equity_candle_snapshot_evidence(
    snapshot: EquityCandleEvidenceSnapshot, evidence: EquityCandleJournalWindow
) -> None:
    """Require ``evidence`` -- the journal read at the snapshot's watermark -- to be exactly
    the snapshot's, record for record.

    Every snapshot store calls this in the transaction that issues a snapshot,
    and again whenever it reads one back, so a manifest can neither omit
    journaled evidence nor gain any. Raises EquityCandleSnapshotEvidenceError.
    """
    expected = (
        snapshot.listing,
        snapshot.timeframe,
        snapshot.window_opens_at,
        snapshot.window_closes_at,
        snapshot.watermark,
    )
    actual = (
        evidence.listing,
        evidence.timeframe,
        evidence.opens_after,
        evidence.closes_at,
        evidence.watermark,
    )
    if actual != expected:
        raise EquityCandleSnapshotEvidenceError(
            f"Snapshot {snapshot.snapshot_id} was compared with another window or watermark."
        )

    if not _lists(snapshot, evidence):
        raise EquityCandleSnapshotEvidenceError(
            f"Snapshot {snapshot.snapshot_id} does not list exactly the journal's evidence at "
            f"watermark {snapshot.watermark}."
        )


def _listed(entries: tuple[Any, ...]) -> list[tuple[int, str]]:
    return [(e.sequence, equity_candle_record_fingerprint(e.record)) for e in entries]


def _lists(snapshot: EquityCandleEvidenceSnapshot, evidence: EquityCandleJournalWindow) -> bool:
    return _listed(snapshot.observations) == _listed(evidence.observations) and _listed(
        snapshot.finalities
    ) == _listed(evidence.finalities)


def require_equity_candle_snapshot_current(
    snapshot: EquityCandleEvidenceSnapshot, current: EquityCandleJournalWindow
) -> None:
    """Require ``current`` -- the window read at the journal's watermark inside the issuing
    transaction -- to hold exactly the snapshot's evidence.

    Every snapshot store calls this in the transaction that issues a snapshot,
    after require_equity_candle_snapshot_evidence. Evidence for the window
    committed after the snapshot's watermark raises
    EquityCandleSnapshotStaleError, so no snapshot is recorded while omitting
    evidence already committed. Entries outside the window do not count.
    """
    scope = (
        snapshot.listing,
        snapshot.timeframe,
        snapshot.window_opens_at,
        snapshot.window_closes_at,
    )
    current_scope = (current.listing, current.timeframe, current.opens_after, current.closes_at)
    if current_scope != scope or current.watermark < snapshot.watermark:
        raise EquityCandleSnapshotEvidenceError(
            f"Snapshot {snapshot.snapshot_id} was compared with another window or an earlier "
            "watermark."
        )
    if not _lists(snapshot, current):
        raise EquityCandleSnapshotStaleError(
            f"Snapshot {snapshot.snapshot_id} was built at watermark {snapshot.watermark}, but "
            f"the window gained evidence by watermark {current.watermark}; nothing was written."
        )


def classify_equity_research_window_provenance(
    source: EquityResearchReferenceProvenanceSource, window: EquityResearchWindowEvaluation
) -> EquityResearchWindowProvenance:
    """Return ``source``'s provenance of ``window``'s references, failing closed.

    The answer must classify exactly the window's two references. A missing
    answer, a missing reference or an answer for other references raises
    UnknownEquityReferenceProvenanceError.
    """
    if window.venue_reference is None or window.applicability_reference is None:
        raise UnknownEquityReferenceProvenanceError(
            "The research window names no venue or applicability reference."
        )
    provenance = source.classify(window)
    if not isinstance(provenance, EquityResearchWindowProvenance):
        raise UnknownEquityReferenceProvenanceError("No reference provenance was supplied.")
    if (provenance.venue_reference, provenance.applicability_reference) != (
        window.venue_reference,
        window.applicability_reference,
    ):
        raise UnknownEquityReferenceProvenanceError(
            "The reference provenance classifies other references than the window's."
        )
    return provenance


class _SnapshotBuilder:
    """Builds a snapshot from one window evaluation and the journal at one watermark."""

    def __init__(
        self,
        currency: Currency,
        finality_policy: EquityCandleFinalityPolicy,
        window_policy: EquityResearchWindowPolicy,
    ) -> None:
        self._currency = currency
        self._finality = finality_policy
        self._window = window_policy
        self._validator = ValidateEquityIntradayCandlesUseCase(
            currency, finality_policy, window_policy
        )

    def read(
        self,
        journal: EquityCandleEvidenceJournal,
        window: EquityResearchWindowEvaluation,
        watermark: int,
    ) -> EquityCandleJournalWindow:
        """Read the window's evidence at ``watermark``; refuse an answer for anything else."""
        opens, closes = window.window_opens_at, window.window_closes_at
        if opens is None or closes is None:
            raise EquityCandleSnapshotRefusedError("An eligible window has bounds.")
        evidence = journal.read_window(
            window.listing, self._window.timeframe, opens, closes, watermark
        )
        if not isinstance(evidence, EquityCandleJournalWindow) or (
            evidence.listing,
            evidence.timeframe,
            evidence.opens_after,
            evidence.closes_at,
            evidence.watermark,
        ) != (window.listing, self._window.timeframe, opens, closes, watermark):
            raise InvalidEquityCandleEvidenceSnapshotError(
                "The journal answered for another window or watermark."
            )
        return evidence

    def build(
        self,
        window: EquityResearchWindowEvaluation,
        evidence: EquityCandleJournalWindow,
        issued_at: PointInTime,
        provenance: EquityResearchWindowProvenance,
    ) -> EquityCandleEvidenceSnapshot:
        """Run EI-2A over the composed evidence and freeze the result."""
        as_of = window.as_of
        composed = compose_equity_candle_observations(evidence, as_of)
        by_object = {id(c.observation): c.entry.sequence for c in composed}
        result = self._validator.execute(window, [c.observation for c in composed], as_of)
        findings: dict[int, EquityCandleFinding] = {}
        for assessment in result.unmatched:
            findings[by_object[id(assessment.observation)]] = assessment.finding
        for interval in result.intervals:
            for assessment in interval.observations:
                findings[by_object[id(assessment.observation)]] = assessment.finding
        return EquityCandleEvidenceSnapshot(
            listing=window.listing,
            trading_date=window.trading_date,
            timeframe=self._window.timeframe,
            currency=self._currency,
            window_policy_version=window.policy_version,
            window_opens_at=evidence.opens_after,
            window_closes_at=evidence.closes_at,
            venue_reference=provenance.venue_reference,
            applicability_reference=provenance.applicability_reference,
            venue_provenance=provenance.venue,
            applicability_provenance=provenance.applicability,
            as_of=as_of,
            watermark=evidence.watermark,
            finality_policy=self._finality,
            observations=tuple(
                EquityCandleSnapshotObservation(
                    c.entry.sequence,
                    c.entry.record,  # type: ignore[arg-type]
                    findings[c.entry.sequence],
                    c.finality_sequences,
                )
                for c in composed
            ),
            finalities=tuple(
                EquityCandleSnapshotFinality(e.sequence, e.record)  # type: ignore[arg-type]
                for e in evidence.finalities
            ),
            intervals=tuple(
                EquityCandleSnapshotInterval(
                    a.interval.opens_at,
                    a.interval.closes_at,
                    a.outcome,
                    tuple(sorted(by_object[id(x.observation)] for x in a.observations)),
                    tuple(sorted(by_object[id(o)] for o in a.acceptable)),
                )
                for a in result.intervals
            ),
            unmatched_sequences=tuple(
                sorted(by_object[id(a.observation)] for a in result.unmatched)
            ),
            coverage=result.coverage,
            issued_at=issued_at,
        )


def _require_types(
    journal: object,
    store: object,
    currency: object,
    finality_policy: object,
    window_policy: object,
    reference_provenance: object,
) -> None:
    if not isinstance(reference_provenance, EquityResearchReferenceProvenanceSource):
        raise TypeError(
            "reference_provenance must be an EquityResearchReferenceProvenanceSource; "
            "provenance is never inferred."
        )
    if not isinstance(journal, EquityCandleEvidenceJournal):
        raise TypeError("journal must be an EquityCandleEvidenceJournal.")
    if not isinstance(store, EquityCandleEvidenceSnapshotStore):
        raise TypeError("store must be an EquityCandleEvidenceSnapshotStore.")
    if not isinstance(currency, Currency):
        raise TypeError("currency must be a Currency.")
    if not isinstance(finality_policy, EquityCandleFinalityPolicy):
        raise TypeError("finality_policy must be an EquityCandleFinalityPolicy.")
    if not isinstance(window_policy, EquityResearchWindowPolicy):
        raise TypeError("window_policy must be an EquityResearchWindowPolicy.")


class IssueEquityCandleEvidenceSnapshotUseCase:
    """Issues an immutable as-issued evidence snapshot for one eligible research window."""

    def __init__(
        self,
        journal: EquityCandleEvidenceJournal,
        store: EquityCandleEvidenceSnapshotStore,
        currency: Currency,
        finality_policy: EquityCandleFinalityPolicy = NO_APPROVED_EQUITY_CANDLE_FINALITY,
        window_policy: EquityResearchWindowPolicy = NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW,
        *,
        reference_provenance: EquityResearchReferenceProvenanceSource,
    ) -> None:
        _require_types(
            journal, store, currency, finality_policy, window_policy, reference_provenance
        )
        self._journal = journal
        self._store = store
        self._provenance = reference_provenance
        self._builder = _SnapshotBuilder(currency, finality_policy, window_policy)

    def execute(
        self,
        window: EquityResearchWindowEvaluation,
        as_of: PointInTime,
        issued_at: PointInTime,
    ) -> EquityCandleEvidenceSnapshot:
        """Freeze the journal's evidence for ``window`` at ``as_of`` and record it."""
        if not isinstance(window, EquityResearchWindowEvaluation):
            raise TypeError("window must be an EquityResearchWindowEvaluation.")
        if not isinstance(as_of, PointInTime) or not isinstance(issued_at, PointInTime):
            raise TypeError("as_of and issued_at must be PointInTime values.")
        if not window.is_eligible:
            raise EquityCandleSnapshotRefusedError(
                f"A rejected research window ({window.rejection}) cannot become an evidence "
                "snapshot."
            )
        if window.as_of != as_of:
            raise ValueError("as_of must equal the research-window evaluation's as_of.")
        if issued_at.compare(as_of) < 0:
            raise EquityCandleSnapshotRefusedError(
                "A snapshot cannot be issued before its observation instant."
            )
        provenance = classify_equity_research_window_provenance(self._provenance, window)
        stale: EquityCandleSnapshotStaleError | None = None
        for _ in range(EQUITY_CANDLE_SNAPSHOT_ISSUE_ATTEMPTS):
            evidence = self._builder.read(self._journal, window, self._journal.watermark())
            snapshot = self._builder.build(window, evidence, issued_at, provenance)
            try:
                self._store.issue(snapshot)
            except EquityCandleSnapshotStaleError as exc:
                stale = exc
                continue
            return snapshot
        raise EquityCandleSnapshotStaleError(
            f"The window kept gaining evidence during {EQUITY_CANDLE_SNAPSHOT_ISSUE_ATTEMPTS} "
            "issuance attempts; no snapshot was recorded."
        ) from stale


class EquityCandleSnapshotReplayOutcome(StrEnum):
    """What re-deriving a stored snapshot established."""

    REPRODUCED = "REPRODUCED"
    DIVERGED = "DIVERGED"
    REFERENCE_UNAVAILABLE = "REFERENCE_UNAVAILABLE"
    NOT_FOUND = "NOT_FOUND"


@dataclass(frozen=True, slots=True)
class EquityCandleSnapshotReplay:
    """A stored snapshot, and the snapshot re-derived from the journal, if any was."""

    snapshot_id: str
    outcome: EquityCandleSnapshotReplayOutcome
    detail: str
    snapshot: EquityCandleEvidenceSnapshot | None
    replayed: EquityCandleEvidenceSnapshot | None


class ReplayEquityCandleEvidenceSnapshotUseCase:
    """Re-derives a stored snapshot from the journal at its watermark, under supplied references.

    The references -- the window evaluation, its reference provenance, the
    finality policy, window policy and currency -- must be exactly the
    recorded ones. Anything else is
    REFERENCE_UNAVAILABLE: no other reference is substituted, and the stored
    snapshot is returned unchanged.
    """

    def __init__(
        self,
        journal: EquityCandleEvidenceJournal,
        store: EquityCandleEvidenceSnapshotStore,
        currency: Currency,
        finality_policy: EquityCandleFinalityPolicy = NO_APPROVED_EQUITY_CANDLE_FINALITY,
        window_policy: EquityResearchWindowPolicy = NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW,
        *,
        reference_provenance: EquityResearchReferenceProvenanceSource,
    ) -> None:
        _require_types(
            journal, store, currency, finality_policy, window_policy, reference_provenance
        )
        self._journal = journal
        self._store = store
        self._provenance = reference_provenance
        self._currency = currency
        self._finality = finality_policy
        self._window = window_policy
        self._builder = _SnapshotBuilder(currency, finality_policy, window_policy)

    def execute(
        self, snapshot_id: str, window: EquityResearchWindowEvaluation
    ) -> EquityCandleSnapshotReplay:
        """Re-derive snapshot ``snapshot_id`` and compare it with the stored one."""
        if not isinstance(snapshot_id, str):
            raise TypeError("snapshot_id must be text.")
        if not isinstance(window, EquityResearchWindowEvaluation):
            raise TypeError("window must be an EquityResearchWindowEvaluation.")
        stored = self._store.get_snapshot(snapshot_id)
        if stored is None:
            return EquityCandleSnapshotReplay(
                snapshot_id,
                EquityCandleSnapshotReplayOutcome.NOT_FOUND,
                "No snapshot is issued under this identity.",
                None,
                None,
            )
        missing = self._unavailable(stored, window)
        provenance: EquityResearchWindowProvenance | None = None
        try:
            provenance = classify_equity_research_window_provenance(self._provenance, window)
        except UnknownEquityReferenceProvenanceError as exc:
            missing.append(f"reference provenance is unknown ({exc})")
        if provenance is not None and (provenance.venue, provenance.applicability) != (
            stored.venue_provenance,
            stored.applicability_provenance,
        ):
            missing.append("reference provenance is not the recorded provenance")
        if missing or provenance is None:
            return EquityCandleSnapshotReplay(
                snapshot_id,
                EquityCandleSnapshotReplayOutcome.REFERENCE_UNAVAILABLE,
                "The recorded references are not available: " + "; ".join(missing) + ".",
                stored,
                None,
            )
        evidence = self._builder.read(self._journal, window, stored.watermark)
        replayed = self._builder.build(window, evidence, stored.issued_at, provenance)
        if replayed != stored:
            return EquityCandleSnapshotReplay(
                snapshot_id,
                EquityCandleSnapshotReplayOutcome.DIVERGED,
                f"Re-derivation produced {replayed.manifest_fingerprint}, not "
                f"{stored.manifest_fingerprint}.",
                stored,
                replayed,
            )
        return EquityCandleSnapshotReplay(
            snapshot_id,
            EquityCandleSnapshotReplayOutcome.REPRODUCED,
            f"Re-derived exactly as {stored.manifest_fingerprint}.",
            stored,
            replayed,
        )

    def _unavailable(
        self, stored: EquityCandleEvidenceSnapshot, window: EquityResearchWindowEvaluation
    ) -> list[str]:
        checks = (
            ("finality policy", self._finality, stored.finality_policy),
            ("research-window policy", self._window.version, stored.window_policy_version),
            ("timeframe", self._window.timeframe, stored.timeframe),
            ("currency", self._currency, stored.currency),
            ("window eligibility", window.is_eligible, True),
            ("window listing", window.listing, stored.listing),
            ("window trading date", window.trading_date, stored.trading_date),
            ("window observation instant", window.as_of, stored.as_of),
            ("window evaluation policy", window.policy_version, stored.window_policy_version),
            ("window opening", window.window_opens_at, stored.window_opens_at),
            ("window close", window.window_closes_at, stored.window_closes_at),
            ("venue reference", window.venue_reference, stored.venue_reference),
            (
                "applicability reference",
                window.applicability_reference,
                stored.applicability_reference,
            ),
        )
        return [
            f"{name} {supplied!s} is not the recorded {recorded!s}"
            for name, supplied, recorded in checks
            if supplied != recorded
        ]
