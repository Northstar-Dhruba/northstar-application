"""EI-2B: append-only equity candle evidence and immutable as-issued snapshots.

Everything here is invented. The research window comes from the real EI-1C-3
use case over in-test port doubles on a 2031 label; every candle, source
identity and finality statement is synthetic test evidence, not NSE or
provider data. The journal and snapshot store are in-memory doubles that apply
the shared Application append and snapshot-evidence rules; the SQLite adapter
is tested in Infrastructure.

A snapshot is a record of what was known; it is not a decision, signal,
recommendation or order.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal

import pytest
from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import (
    Currency,
    ExchangeCode,
    PointInTime,
    Price,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.market_data import HistoricalOHLCVBar

from northstar_application import application_services, ports
from northstar_application.application_services import (
    EquityCandleAppendOutcome,
    EquityCandleAppendResult,
    EquityCandleCoverage,
    EquityCandleEvidenceClass,
    EquityCandleEvidenceSnapshot,
    EquityCandleFinalityEvidence,
    EquityCandleFinalityPolicy,
    EquityCandleFinalityRecord,
    EquityCandleFinding,
    EquityCandleIntervalOutcome,
    EquityCandleJournalEntry,
    EquityCandleJournalEntryKind,
    EquityCandleJournalRecord,
    EquityCandleJournalWindow,
    EquityCandleObservation,
    EquityCandleObservationRecord,
    EquityCandleSnapshotRefusedError,
    EquityCandleSnapshotReplayOutcome,
    EquityResearchWindowEvaluation,
    EvaluateEquityResearchWindowUseCase,
    InvalidEquityCandleEvidenceSnapshotError,
    InvalidEquityCandleJournalRecordError,
    IssueEquityCandleEvidenceSnapshotUseCase,
    ReplayEquityCandleEvidenceSnapshotUseCase,
    equity_candle_identity_conflicts,
    equity_candle_in_window,
    equity_candle_payload_fingerprint,
    equity_candle_record_document,
    equity_candle_record_fingerprint,
    equity_candle_record_from_document,
    equity_candle_retry,
    equity_candle_snapshot_from_document,
    equity_candle_snapshot_json,
    require_equity_candle_snapshot_current,
    require_equity_candle_snapshot_evidence,
)
from northstar_application.application_services import (
    equity_candle_evidence_journal as journal_module,
)
from northstar_application.application_services import (
    equity_candle_evidence_snapshot as snapshot_module,
)
from northstar_application.ports import (
    EquityCandleEvidenceJournal,
    EquityCandleEvidenceSnapshotStore,
    EquityCandleIngestionConflictError,
    EquityCandleSnapshotConflictError,
    EquityCandleSnapshotEvidenceError,
    EquityCandleSnapshotStaleError,
    EquityCandleSourceIdentityConflictError,
    EquityListingApplicability,
    EquityListingApplicabilityClassification,
    EquityListingApplicabilitySource,
    EquityReferenceProvenanceKind,
    EquityResearchReferenceProvenanceSource,
    EquityResearchWindowProvenance,
    EquityVenueDayClassification,
    EquityVenueDayReference,
    EquityVenueDayReferenceSource,
    InvalidEquityCandleJournalQueryError,
    UnknownEquityReferenceProvenanceError,
)
from northstar_application.ports import equity_candle_evidence_journal as port_module
from northstar_application.ports import equity_research_reference_provenance as provenance_port

_NSE = ExchangeCode("NSE")
_INR = Currency("INR")
_DAY = date(2031, 3, 4)
_M_AND_M = ListingReference(Symbol("M&M"), _NSE)
_OTHER = ListingReference(Symbol("TESTCO"), _NSE)
_FIRST_END = datetime(2031, 3, 4, 3, 50, tzinfo=UTC)
_LATE = datetime(2031, 3, 4, 10, 0, tzinfo=UTC)
_SOURCE = "synthetic-source"
_SYNTHETIC = EquityCandleFinalityPolicy(
    "synthetic-ei2b-test/1", frozenset({_SOURCE}), permits_synthetic=True
)
_PRODUCTION_LIKE = EquityCandleFinalityPolicy(
    "test-approved-source/1", frozenset({"approved-source"}), permits_synthetic=False
)
_OUTCOME = EquityCandleIntervalOutcome
_FINDING = EquityCandleFinding
_COVERAGE = EquityCandleCoverage


# ----------------------------------------------------------------------
# Research-window doubles (as in EI-2A)
# ----------------------------------------------------------------------


class _Venue(EquityVenueDayReferenceSource):
    def __init__(self, reference: str = "synthetic-reference test-venue/1") -> None:
        self.reference = reference

    def venue_day(self, exchange_code: ExchangeCode, trading_date: date) -> EquityVenueDayReference:
        return EquityVenueDayReference(
            exchange_code=exchange_code,
            trading_date=trading_date,
            classification=EquityVenueDayClassification.NORMAL_SESSION,
            common_opens_at=PointInTime("2031-03-04T03:45:00Z"),
            common_closes_at=PointInTime("2031-03-04T09:45:00Z"),
            utc_offset=timedelta(hours=5, minutes=30),
            reference=self.reference,
            reason="test normal session",
        )


class _Applicability(EquityListingApplicabilitySource):
    def __init__(
        self, eligible: bool = True, reference: str = "synthetic-reference test/1"
    ) -> None:
        self.eligible = eligible
        self.reference = reference

    def applicability(
        self, listing: ListingReference, trading_date: date
    ) -> EquityListingApplicability:
        classification = (
            EquityListingApplicabilityClassification.ELIGIBLE_FOR_NORMAL_RESEARCH_WINDOW
            if self.eligible
            else EquityListingApplicabilityClassification.UNKNOWN
        )
        return EquityListingApplicability(
            listing, trading_date, classification, self.reference, "t"
        )


def _pit(value: datetime) -> PointInTime:
    return PointInTime(value.isoformat().replace("+00:00", "Z"))


def _end(index: int) -> datetime:
    return _FIRST_END + timedelta(minutes=5 * (index - 1))


_VENUE_REF = "test-venue/1"
_APPLICABILITY_REF = "test-applicability/1"
_SYN = EquityReferenceProvenanceKind.SYNTHETIC
_PROD = EquityReferenceProvenanceKind.ACCEPTED_PRODUCTION


def _window(
    as_of: datetime,
    listing: ListingReference = _M_AND_M,
    *,
    eligible: bool = True,
    venue: str = _VENUE_REF,
    applicability: str = _APPLICABILITY_REF,
) -> EquityResearchWindowEvaluation:
    use_case = EvaluateEquityResearchWindowUseCase(
        _Venue(venue), _Applicability(eligible, applicability)
    )
    return use_case.execute(listing, _DAY, _pit(as_of))


class _Provenance(EquityResearchReferenceProvenanceSource):
    """Classifies only the exact references in its table; anything else is unknown.

    The test references are deliberately not spelled "synthetic": provenance
    comes from this explicit table, never from a reference's name.
    """

    def __init__(self, table: dict[str, EquityReferenceProvenanceKind] | None = None) -> None:
        self.table = table if table is not None else {_VENUE_REF: _SYN, _APPLICABILITY_REF: _SYN}

    def classify(self, window: EquityResearchWindowEvaluation) -> EquityResearchWindowProvenance:
        venue, applicability = window.venue_reference, window.applicability_reference
        for reference in (venue, applicability):
            if reference not in self.table:
                raise UnknownEquityReferenceProvenanceError(f"Unknown reference {reference!r}.")
        assert venue is not None and applicability is not None
        return EquityResearchWindowProvenance(
            venue, self.table[venue], applicability, self.table[applicability]
        )


_PRODUCTION_REFERENCES = _Provenance({_VENUE_REF: _PROD, _APPLICABILITY_REF: _PROD})


# ----------------------------------------------------------------------
# Evidence builders
# ----------------------------------------------------------------------


def _bar(
    index: int,
    *,
    listing: ListingReference = _M_AND_M,
    prices: tuple[str, str, str, str] = ("100", "101", "99", "100.5"),
    volume: str = "1000",
    timeframe: str = "5m",
    currency: Currency = _INR,
) -> HistoricalOHLCVBar:
    o, h, low, c = (Price(Decimal(p), currency) for p in prices)
    return HistoricalOHLCVBar(
        symbol=listing.symbol,
        exchange_code=listing.exchange_code,
        point_in_time=_pit(_end(index)),
        timeframe=Timeframe(timeframe),
        open=o,
        high=h,
        low=low,
        close=c,
        volume=Quantity(Decimal(volume)),
    )


def _observation(
    index: int,
    *,
    ingestion: str | None = None,
    source: str = _SOURCE,
    event: str | None = None,
    revision: str = "r1",
    received: datetime | None = None,
    corrects: str | None = None,
    synthetic: bool = True,
    published: datetime | None = None,
    **bar: object,
) -> EquityCandleObservationRecord:
    return EquityCandleObservationRecord(
        ingestion_id=ingestion or f"obs-{index}-{source}-{event}-{revision}",
        observation=EquityCandleObservation(
            bar=_bar(index, **bar),  # type: ignore[arg-type]
            source_id=source,
            source_event_id=event or f"evt-{index}",
            source_revision=revision,
            received_at=_pit(received or _end(index) + timedelta(seconds=5)),
            finality=None,
            corrects_revision=corrects,
        ),
        is_synthetic=synthetic,
        source_published_at=None if published is None else _pit(published),
    )


def _finality(
    index: int,
    *,
    ingestion: str | None = None,
    source: str = _SOURCE,
    event: str | None = None,
    revision: str = "r1",
    confirmed: datetime | None = None,
    received: datetime | None = None,
    synthetic: bool = True,
    reference: str = "synthetic finality statement",
) -> EquityCandleFinalityRecord:
    confirmed = confirmed or _end(index) + timedelta(seconds=10)
    return EquityCandleFinalityRecord(
        ingestion_id=ingestion or f"fin-{index}-{source}-{event}-{revision}",
        evidence=EquityCandleFinalityEvidence(
            source_id=source,
            source_event_id=event or f"evt-{index}",
            source_revision=revision,
            confirmed_at=_pit(confirmed),
            evidence_reference=reference,
            is_synthetic=synthetic,
        ),
        received_at=_pit(received or confirmed + timedelta(seconds=1)),
    )


# ----------------------------------------------------------------------
# In-memory port doubles applying the shared Application rules
# ----------------------------------------------------------------------


class _Journal(EquityCandleEvidenceJournal):
    def __init__(self) -> None:
        self.entries: list[EquityCandleJournalEntry] = []

    def append(self, record: EquityCandleJournalRecord) -> EquityCandleAppendResult:
        same = next((e for e in self.entries if e.record.ingestion_id == record.ingestion_id), None)
        retry = equity_candle_retry(record, same)
        if retry is not None:
            return EquityCandleAppendResult(EquityCandleAppendOutcome.ALREADY_APPENDED, retry)
        conflicts = equity_candle_identity_conflicts(
            record,
            tuple(e for e in self.entries if e.record.source_identity == record.source_identity),
        )
        sequence = len(self.entries) + 1
        entry = EquityCandleJournalEntry(
            sequence, _pit(_LATE + timedelta(days=1, seconds=sequence)), record
        )
        self.entries.append(entry)
        return EquityCandleAppendResult(EquityCandleAppendOutcome.APPENDED, entry, conflicts)

    def watermark(self) -> int:
        return len(self.entries)

    def read_window(
        self,
        listing: ListingReference,
        timeframe: Timeframe,
        opens_after: PointInTime,
        closes_at: PointInTime,
        watermark: int,
    ) -> EquityCandleJournalWindow:
        if watermark > len(self.entries):
            raise InvalidEquityCandleJournalQueryError("beyond the journal")
        visible = self.entries[:watermark]
        observations = tuple(
            e
            for e in visible
            if equity_candle_in_window(e.record, listing, timeframe, opens_after, closes_at)
        )
        identities = {e.record.source_identity for e in observations}
        finalities = tuple(
            e
            for e in visible
            if e.record.kind is EquityCandleJournalEntryKind.FINALITY
            and e.record.source_identity in identities
        )
        return EquityCandleJournalWindow(
            listing, timeframe, opens_after, closes_at, watermark, observations, finalities
        )


class _Store(EquityCandleEvidenceSnapshotStore):
    def __init__(self, journal: _Journal) -> None:
        self.journal = journal
        self.documents: dict[str, str] = {}

    def _verify(self, snapshot: EquityCandleEvidenceSnapshot) -> None:
        require_equity_candle_snapshot_evidence(
            snapshot,
            self.journal.read_window(
                snapshot.listing,
                snapshot.timeframe,
                snapshot.window_opens_at,
                snapshot.window_closes_at,
                snapshot.watermark,
            ),
        )

    def issue(self, snapshot: EquityCandleEvidenceSnapshot) -> int:
        text = equity_candle_snapshot_json(snapshot)
        existing = self.documents.get(snapshot.snapshot_id)
        if existing is not None:
            if existing != text:
                raise EquityCandleSnapshotConflictError("other content")
            return 0
        self._verify(snapshot)
        require_equity_candle_snapshot_current(
            snapshot,
            self.journal.read_window(
                snapshot.listing,
                snapshot.timeframe,
                snapshot.window_opens_at,
                snapshot.window_closes_at,
                self.journal.watermark(),
            ),
        )
        self.documents[snapshot.snapshot_id] = text
        return 1

    def get_snapshot(self, snapshot_id: str) -> EquityCandleEvidenceSnapshot | None:
        text = self.documents.get(snapshot_id)
        if text is None:
            return None
        snapshot = equity_candle_snapshot_from_document(json.loads(text))
        self._verify(snapshot)
        return snapshot


def _ports() -> tuple[_Journal, _Store]:
    journal = _Journal()
    return journal, _Store(journal)


def _issue(  # type: ignore[no-untyped-def]
    journal: EquityCandleEvidenceJournal,
    store: EquityCandleEvidenceSnapshotStore,
    as_of: datetime,
    *,
    issued: datetime | None = None,
    policy: EquityCandleFinalityPolicy = _SYNTHETIC,
    listing: ListingReference = _M_AND_M,
    provenance: EquityResearchReferenceProvenanceSource | None = None,
    venue: str = _VENUE_REF,
    applicability: str = _APPLICABILITY_REF,
):
    window = _window(as_of, listing, venue=venue, applicability=applicability)
    use_case = IssueEquityCandleEvidenceSnapshotUseCase(
        journal, store, _INR, policy, reference_provenance=provenance or _Provenance()
    )
    return use_case.execute(window, _pit(as_of), _pit(issued or as_of + timedelta(minutes=1)))


def _replay(  # type: ignore[no-untyped-def]
    journal: _Journal,
    store: _Store,
    snapshot: EquityCandleEvidenceSnapshot,
    *,
    policy: EquityCandleFinalityPolicy = _SYNTHETIC,
    as_of: datetime | None = None,
    provenance: EquityResearchReferenceProvenanceSource | None = None,
):
    window = _window(as_of or datetime.fromisoformat(snapshot.as_of.value.replace("Z", "+00:00")))
    use_case = ReplayEquityCandleEvidenceSnapshotUseCase(
        journal, store, _INR, policy, reference_provenance=provenance or _Provenance()
    )
    return use_case.execute(snapshot.snapshot_id, window)


def _final_candle(journal: _Journal, index: int, **observation: object) -> None:
    journal.append(_observation(index, **observation))  # type: ignore[arg-type]
    journal.append(_finality(index))


def _interval(snapshot: EquityCandleEvidenceSnapshot, index: int):  # type: ignore[no-untyped-def]
    return snapshot.intervals[index - 1]


# ----------------------------------------------------------------------
# A. Journal records and canonical encoding
# ----------------------------------------------------------------------


def test_an_observation_record_carries_no_finality() -> None:
    observation = dataclasses.replace(
        _observation(1).observation,
        finality=_finality(1).evidence,
    )
    with pytest.raises(InvalidEquityCandleJournalRecordError, match="its own"):
        EquityCandleObservationRecord("obs-1", observation, is_synthetic=True)


def test_records_refuse_impossible_times() -> None:
    with pytest.raises(InvalidEquityCandleJournalRecordError, match="published"):
        _observation(1, received=_end(1), published=_end(1) + timedelta(seconds=1))
    with pytest.raises(InvalidEquityCandleJournalRecordError, match="confirmed after"):
        _finality(1, confirmed=_end(1) + timedelta(minutes=1), received=_end(1))


@pytest.mark.parametrize(
    "build",
    [
        lambda: EquityCandleObservationRecord(" ", _observation(1).observation, True),
        lambda: EquityCandleObservationRecord("x", _observation(1).observation, 1),  # type: ignore[arg-type]
        lambda: EquityCandleObservationRecord("x", _bar(1), True),  # type: ignore[arg-type]
        lambda: EquityCandleFinalityRecord("x", _finality(1).evidence, "2031-03-04T04:00:00Z"),  # type: ignore[arg-type]
        lambda: EquityCandleJournalEntry(0, _pit(_LATE), _observation(1)),
        lambda: EquityCandleJournalEntry(True, _pit(_LATE), _observation(1)),  # type: ignore[arg-type]
    ],
)
def test_malformed_records_and_entries_are_refused(build) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(InvalidEquityCandleJournalRecordError):
        build()


def test_missing_source_identity_is_refused_by_the_reused_ei2a_contracts() -> None:
    with pytest.raises(ValueError, match="source_id"):
        _observation(1, source=" ")
    with pytest.raises(ValueError, match="source_event_id"):
        _finality(1, event=" ")


def test_documents_keep_decimals_as_exact_text_and_round_trip() -> None:
    record = _observation(1, prices=("100.50", "101.25", "99.05", "100.5"), volume="1000.000")
    document = equity_candle_record_document(record)
    payload = document["payload"]
    assert (payload["open"], payload["high"], payload["low"], payload["volume"]) == (
        "100.5",
        "101.25",
        "99.05",
        "1000",
    )
    assert not any(isinstance(value, float) for value in payload.values())
    assert equity_candle_record_from_document(json.loads(json.dumps(document))) == record
    finality = _finality(1)
    assert equity_candle_record_from_document(equity_candle_record_document(finality)) == finality


def test_fingerprints_are_stable_and_separate_payload_from_receipt() -> None:
    record = _observation(1)
    same = _observation(1, prices=("100.000", "101", "99.0", "100.50"))
    retried_elsewhere = _observation(1, ingestion="another-receipt", received=_end(2))
    assert equity_candle_record_fingerprint(record) == equity_candle_record_fingerprint(same)
    assert equity_candle_record_fingerprint(record).startswith("ecr1-sha256:")
    assert equity_candle_payload_fingerprint(record).startswith("ecp1-sha256:")
    assert equity_candle_payload_fingerprint(record) == equity_candle_payload_fingerprint(
        retried_elsewhere
    )
    assert equity_candle_record_fingerprint(record) != equity_candle_record_fingerprint(
        retried_elsewhere
    )
    other_revision = _observation(1, revision="r2")
    assert equity_candle_payload_fingerprint(record) != equity_candle_payload_fingerprint(
        other_revision
    )


@pytest.mark.parametrize(
    "tamper",
    [
        lambda d: d["payload"].__setitem__("open", "100.50"),
        lambda d: d["payload"].__setitem__("open", 100.5),
        lambda d: d["payload"].__setitem__("is_synthetic", 1),
        lambda d: d["payload"].__setitem__("kind", "TRADE"),
        lambda d: d["payload"].__setitem__("unexpected", "x"),
        lambda d: d.__setitem__("format", "other/1"),
        lambda d: d.__setitem__("received_at", "2031-03-04T09:20:05+05:30"),
    ],
)
def test_non_canonical_documents_are_refused(tamper) -> None:  # type: ignore[no-untyped-def]
    document = json.loads(json.dumps(equity_candle_record_document(_observation(1))))
    tamper(document)
    with pytest.raises((ValueError, TypeError, KeyError)):
        equity_candle_record_from_document(document)


def test_a_journal_window_refuses_inconsistent_entries() -> None:
    journal = _Journal()
    journal.append(_observation(1))
    journal.append(_observation(2, listing=_OTHER))
    window = _window(_LATE)
    good = journal.read_window(
        _M_AND_M, Timeframe("5m"), window.window_opens_at, window.window_closes_at, 2
    )
    assert [e.sequence for e in good.observations] == [1]
    with pytest.raises(InvalidEquityCandleJournalRecordError):
        dataclasses.replace(good, watermark=0)
    with pytest.raises(InvalidEquityCandleJournalRecordError):
        dataclasses.replace(good, observations=(journal.entries[1],))
    with pytest.raises(InvalidEquityCandleJournalRecordError):
        dataclasses.replace(good, finalities=good.observations)
    with pytest.raises(InvalidEquityCandleJournalQueryError):
        journal.read_window(_M_AND_M, Timeframe("5m"), good.opens_after, good.closes_at, 3)


# ----------------------------------------------------------------------
# B. Append-only rules: retries, identities and conflicts
# ----------------------------------------------------------------------


def test_a_retry_returns_the_original_entry_and_writes_nothing() -> None:
    journal = _Journal()
    first = journal.append(_observation(1))
    again = journal.append(_observation(1))
    assert first.outcome is EquityCandleAppendOutcome.APPENDED
    assert again.outcome is EquityCandleAppendOutcome.ALREADY_APPENDED
    assert again.entry == first.entry
    assert journal.watermark() == 1


@pytest.mark.parametrize(
    "changed",
    [
        {"prices": ("100", "102", "99", "100.5")},
        {"received": _end(1) + timedelta(seconds=6)},
        {"revision": "r2"},
    ],
)
def test_reusing_an_ingestion_identity_for_other_content_is_refused(changed) -> None:  # type: ignore[no-untyped-def]
    journal = _Journal()
    journal.append(_observation(1, ingestion="receipt-1"))
    with pytest.raises(EquityCandleIngestionConflictError, match="nothing was written"):
        journal.append(_observation(1, ingestion="receipt-1", **changed))
    assert journal.watermark() == 1
    assert journal.entries[0].record == _observation(1, ingestion="receipt-1")


def test_distinct_receipts_of_one_identity_are_kept_as_exact_repeats() -> None:
    journal, store = _ports()
    journal.append(_observation(1, ingestion="receipt-1"))
    repeat = journal.append(
        _observation(1, ingestion="receipt-2", received=_end(1) + timedelta(seconds=30))
    )
    journal.append(_finality(1))
    assert repeat.outcome is EquityCandleAppendOutcome.APPENDED
    assert repeat.conflicting_values == ()
    interval = _interval(_issue(journal, store, _LATE), 1)
    assert interval.outcome is _OUTCOME.ACCEPTABLE
    assert interval.acceptable_sequences == (1, 2)


def test_conflicting_values_under_one_identity_are_kept_and_reported() -> None:
    journal, store = _ports()
    journal.append(_observation(1, ingestion="receipt-1"))
    other = journal.append(
        _observation(1, ingestion="receipt-2", prices=("100", "101", "99", "100.75"))
    )
    journal.append(_finality(1))
    assert other.outcome is EquityCandleAppendOutcome.APPENDED
    assert other.conflicting_values == (1,)
    snapshot = _issue(journal, store, _LATE)
    interval = _interval(snapshot, 1)
    assert interval.outcome is _OUTCOME.CONFLICTING_VALUES
    assert interval.observation_sequences == (1, 2)
    assert interval.acceptable_sequences == ()


@pytest.mark.parametrize(
    "changed",
    [
        {"index": 2},
        {"listing": _OTHER},
        {"timeframe": "1m"},
        {"corrects": "r0"},
        {"published": _end(1)},
        {"synthetic": False},
    ],
)
def test_one_source_identity_cannot_be_bound_to_other_facts(changed) -> None:  # type: ignore[no-untyped-def]
    journal = _Journal()
    journal.append(_observation(1, event="evt-x"))
    index = changed.pop("index", 1)
    with pytest.raises(EquityCandleSourceIdentityConflictError, match="nothing was written"):
        journal.append(_observation(index, event="evt-x", ingestion="receipt-2", **changed))
    assert journal.watermark() == 1


def test_finality_statements_for_one_identity_must_agree() -> None:
    journal = _Journal()
    journal.append(_observation(1))
    journal.append(_finality(1))
    repeat = journal.append(_finality(1, ingestion="fin-repeat"))
    assert repeat.outcome is EquityCandleAppendOutcome.APPENDED
    with pytest.raises(EquityCandleSourceIdentityConflictError, match="finality evidence"):
        journal.append(_finality(1, ingestion="fin-other", confirmed=_end(1) + timedelta(1)))
    with pytest.raises(EquityCandleSourceIdentityConflictError, match="synthetic"):
        journal.append(_finality(2, ingestion="fin-2", event="evt-1", synthetic=False))
    assert journal.watermark() == 3


def test_a_finality_received_before_its_candle_classification_still_binds_it() -> None:
    journal = _Journal()
    journal.append(_finality(1, synthetic=False, reference="statement"))
    with pytest.raises(EquityCandleSourceIdentityConflictError, match="synthetic"):
        journal.append(_observation(1, synthetic=True))


def test_distinct_revisions_with_identical_prices_are_unresolved() -> None:
    journal, store = _ports()
    _final_candle(journal, 1)
    journal.append(_observation(1, revision="r2"))
    journal.append(_finality(1, revision="r2"))
    interval = _interval(_issue(journal, store, _LATE), 1)
    assert interval.outcome is _OUTCOME.UNRESOLVED_REVISIONS
    assert interval.observation_sequences == (1, 3)
    assert interval.acceptable_sequences == ()


def test_distinct_revisions_with_conflicting_prices_are_unresolved() -> None:
    journal, store = _ports()
    _final_candle(journal, 1)
    journal.append(_observation(1, revision="r2", prices=("100", "101", "98", "99")))
    journal.append(_finality(1, revision="r2"))
    assert _interval(_issue(journal, store, _LATE), 1).outcome is _OUTCOME.UNRESOLVED_REVISIONS


def test_distinct_source_events_with_identical_prices_are_never_merged() -> None:
    journal, store = _ports()
    _final_candle(journal, 1)
    journal.append(_observation(1, event="evt-other"))
    journal.append(_finality(1, event="evt-other"))
    assert _interval(_issue(journal, store, _LATE), 1).outcome is _OUTCOME.UNRESOLVED_REVISIONS


def test_an_explicit_correction_is_kept_as_data_and_unresolved() -> None:
    journal, store = _ports()
    _final_candle(journal, 1)
    journal.append(_observation(1, revision="r2", corrects="r1", prices=("100", "101", "99", "99")))
    journal.append(_finality(1, revision="r2"))
    snapshot = _issue(journal, store, _LATE)
    interval = _interval(snapshot, 1)
    assert interval.outcome is _OUTCOME.UNRESOLVED_REVISIONS
    assert interval.acceptable_sequences == ()
    original = snapshot.observations[0]
    assert original.record == _observation(1)
    assert original.finding is _FINDING.ACCEPTABLE
    assert snapshot.observations[1].record.observation.corrects_revision == "r1"


def test_a_finality_may_arrive_before_its_candle() -> None:
    journal, store = _ports()
    journal.append(_finality(1))
    journal.append(_observation(1, received=_end(1) + timedelta(minutes=2)))
    snapshot = _issue(journal, store, _LATE)
    assert _interval(snapshot, 1).outcome is _OUTCOME.ACCEPTABLE
    assert snapshot.observations[0].finality_sequences == (1,)


def test_a_missing_or_mismatched_finality_leaves_the_candle_unfinalized() -> None:
    journal, store = _ports()
    journal.append(_observation(1))
    journal.append(_observation(2))
    journal.append(_finality(2, revision="r9"))
    snapshot = _issue(journal, store, _LATE)
    assert [o.finding for o in snapshot.observations] == [
        _FINDING.FINALITY_MISSING,
        _FINDING.FINALITY_MISSING,
    ]
    assert snapshot.finalities == ()
    assert _interval(snapshot, 2).outcome is _OUTCOME.UNFINALIZED


# ----------------------------------------------------------------------
# C. Scenarios A-E
# ----------------------------------------------------------------------


def test_scenario_a_and_b_an_original_observation_is_frozen_and_reopens_identically() -> None:
    journal, store = _ports()
    appended = journal.append(_observation(1))
    journal.append(_finality(1))
    assert appended.entry.sequence == 1
    as_of = _end(1) + timedelta(minutes=1)
    snapshot = _issue(journal, store, as_of)
    assert snapshot.evidence_class is EquityCandleEvidenceClass.SYNTHETIC
    assert snapshot.coverage is _COVERAGE.COMPLETE_AS_OF
    assert snapshot.is_complete
    assert snapshot.watermark == 2
    assert snapshot.observation_sequences == (1,)
    assert snapshot.finality_sequences == (2,)
    assert snapshot.manifest_fingerprint.startswith("ecm1-sha256:")
    assert snapshot.snapshot_id.startswith("ecs1-sha256:")
    reopened = store.get_snapshot(snapshot.snapshot_id)
    assert reopened == snapshot
    assert reopened is not None
    assert reopened.manifest_fingerprint == snapshot.manifest_fingerprint
    assert reopened.observations[0].record.observation.bar == _bar(1)
    assert _replay(journal, store, snapshot).outcome is EquityCandleSnapshotReplayOutcome.REPRODUCED
    rebuilt_journal, rebuilt_store = _ports()
    rebuilt_journal.append(_observation(1))
    rebuilt_journal.append(_finality(1))
    assert _issue(rebuilt_journal, rebuilt_store, as_of) == snapshot


def test_scenario_c_a_later_correction_never_changes_an_issued_snapshot() -> None:
    journal, store = _ports()
    _final_candle(journal, 1)
    as_of = _end(1) + timedelta(minutes=1)
    original = _issue(journal, store, as_of)
    journal.append(
        _observation(
            1,
            revision="r2",
            corrects="r1",
            prices=("100", "102", "99", "101"),
            received=_end(1) + timedelta(seconds=20),
        )
    )
    journal.append(_finality(1, revision="r2", confirmed=_end(1) + timedelta(seconds=21)))
    assert len(journal.entries) == 4
    assert journal.entries[0].record == _observation(1)
    assert store.get_snapshot(original.snapshot_id) == original
    assert _replay(journal, store, original).outcome is EquityCandleSnapshotReplayOutcome.REPRODUCED
    later = _issue(journal, store, as_of, issued=_LATE)
    assert later.snapshot_id != original.snapshot_id
    assert _interval(later, 1).outcome is _OUTCOME.UNRESOLVED_REVISIONS
    assert _interval(later, 1).acceptable_sequences == ()
    assert later.observation_sequences == (1, 3)
    assert len(store.documents) == 2


def test_scenario_d_backdated_ingestion_never_enters_an_earlier_snapshot() -> None:
    journal, store = _ports()
    _final_candle(journal, 1)
    as_of = _end(2) + timedelta(minutes=1)
    earlier = _issue(journal, store, as_of)
    assert _interval(earlier, 2).outcome is _OUTCOME.MISSING
    journal.append(_observation(2, received=_end(2) + timedelta(seconds=1), published=_end(2)))
    journal.append(_finality(2, confirmed=_end(2) + timedelta(seconds=2)))
    reopened = store.get_snapshot(earlier.snapshot_id)
    assert reopened == earlier
    assert reopened is not None
    assert _interval(reopened, 2).outcome is _OUTCOME.MISSING
    assert _replay(journal, store, earlier).outcome is EquityCandleSnapshotReplayOutcome.REPRODUCED
    later = _issue(journal, store, as_of, issued=as_of + timedelta(minutes=5))
    assert later.watermark == 4 and earlier.watermark == 2
    assert _interval(later, 2).outcome is _OUTCOME.ACCEPTABLE


def test_finality_received_after_the_observation_instant_is_never_composed() -> None:
    journal, store = _ports()
    journal.append(_observation(1))
    as_of = _end(1) + timedelta(minutes=1)
    journal.append(
        _finality(1, confirmed=_end(1) + timedelta(seconds=10), received=as_of + timedelta(1))
    )
    snapshot = _issue(journal, store, as_of)
    assert snapshot.finality_sequences == (2,)
    assert snapshot.observations[0].finality_sequences == ()
    assert snapshot.observations[0].finding is _FINDING.FINALITY_MISSING
    assert _interval(snapshot, 1).outcome is _OUTCOME.UNFINALIZED


def test_finality_appended_after_issuance_does_not_alter_the_snapshot() -> None:
    journal, store = _ports()
    journal.append(_observation(1))
    as_of = _end(1) + timedelta(minutes=1)
    snapshot = _issue(journal, store, as_of)
    journal.append(_finality(1))
    assert store.get_snapshot(snapshot.snapshot_id) == snapshot
    assert _interval(snapshot, 1).outcome is _OUTCOME.UNFINALIZED
    assert _replay(journal, store, snapshot).outcome is EquityCandleSnapshotReplayOutcome.REPRODUCED


def test_scenario_e_a_complete_synthetic_window_of_69_intervals() -> None:
    journal, store = _ports()
    for index in range(69, 0, -1):
        _final_candle(journal, index)
    partial = _issue(journal, store, _end(10) + timedelta(minutes=1))
    complete = _issue(journal, store, _LATE)
    assert len(complete.intervals) == 69
    assert all(i.outcome is _OUTCOME.ACCEPTABLE for i in complete.intervals)
    assert complete.coverage is _COVERAGE.WINDOW_COMPLETE
    assert complete.evidence_class is EquityCandleEvidenceClass.SYNTHETIC
    assert partial.coverage is _COVERAGE.COMPLETE_AS_OF
    assert len(complete.observations) == 69 and len(complete.finalities) == 69
    assert store.get_snapshot(complete.snapshot_id) == complete
    assert _replay(journal, store, complete).outcome is EquityCandleSnapshotReplayOutcome.REPRODUCED
    again_journal, again_store = _ports()
    for index in range(69, 0, -1):
        _final_candle(again_journal, index)
    assert _issue(again_journal, again_store, _LATE).manifest_fingerprint == (
        complete.manifest_fingerprint
    )


def test_incomplete_snapshots_are_legitimate_records() -> None:
    journal, store = _ports()
    _final_candle(journal, 1)
    journal.append(_observation(3))
    _final_candle(journal, 5)
    snapshot = _issue(journal, store, _end(5) + timedelta(minutes=1))
    outcomes = [i.outcome for i in snapshot.intervals[:6]]
    assert outcomes == [
        _OUTCOME.ACCEPTABLE,
        _OUTCOME.MISSING,
        _OUTCOME.UNFINALIZED,
        _OUTCOME.MISSING,
        _OUTCOME.ACCEPTABLE,
        _OUTCOME.NOT_ENDED,
    ]
    assert snapshot.coverage is _COVERAGE.INCOMPLETE_AS_OF
    assert not snapshot.is_complete
    assert store.get_snapshot(snapshot.snapshot_id) == snapshot


def test_a_candle_whose_interval_has_not_ended_is_recorded_as_not_ended() -> None:
    journal, store = _ports()
    journal.append(_observation(3, received=_end(1)))
    snapshot = _issue(journal, store, _end(1) + timedelta(minutes=1))
    assert snapshot.observations[0].finding is _FINDING.INTERVAL_NOT_ENDED
    assert _interval(snapshot, 3).outcome is _OUTCOME.NOT_ENDED
    assert snapshot.coverage is _COVERAGE.INCOMPLETE_AS_OF


def test_a_rejected_research_window_is_refused_and_nothing_is_issued() -> None:
    journal, store = _ports()
    _final_candle(journal, 1)
    window = _window(_LATE, eligible=False)
    use_case = IssueEquityCandleEvidenceSnapshotUseCase(
        journal, store, _INR, _SYNTHETIC, reference_provenance=_Provenance()
    )
    with pytest.raises(EquityCandleSnapshotRefusedError, match="rejected research window"):
        use_case.execute(window, _pit(_LATE), _pit(_LATE))
    assert store.documents == {}


def test_issuance_instants_are_checked() -> None:
    journal, store = _ports()
    use_case = IssueEquityCandleEvidenceSnapshotUseCase(
        journal, store, _INR, _SYNTHETIC, reference_provenance=_Provenance()
    )
    window = _window(_LATE)
    with pytest.raises(EquityCandleSnapshotRefusedError, match="before its observation"):
        use_case.execute(window, _pit(_LATE), _pit(_LATE - timedelta(seconds=1)))
    with pytest.raises(ValueError, match="as_of"):
        use_case.execute(window, _pit(_end(3)), _pit(_LATE))
    with pytest.raises(TypeError):
        use_case.execute(window, _LATE, _pit(_LATE))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        IssueEquityCandleEvidenceSnapshotUseCase(
            journal,
            store,
            "INR",  # type: ignore[arg-type]
            _SYNTHETIC,
            reference_provenance=_Provenance(),
        )
    assert store.documents == {}


def test_other_listings_timeframes_and_dates_stay_out_of_a_snapshot() -> None:
    journal, store = _ports()
    _final_candle(journal, 1, listing=_OTHER)
    _final_candle(journal, 2, timeframe="1m")
    _final_candle(journal, 3, currency=Currency("USD"))
    snapshot = _issue(journal, store, _LATE)
    assert snapshot.observation_sequences == (5,)
    assert snapshot.observations[0].finding is _FINDING.WRONG_CURRENCY
    assert snapshot.unmatched_sequences == (5,)
    other = _issue(journal, store, _LATE, listing=_OTHER)
    assert other.observation_sequences == (1,)


def test_evidence_class_comes_from_explicit_provenance_never_from_reference_names() -> None:
    journal, store = _ports()
    journal.append(_observation(1, source="approved-source", synthetic=False))
    journal.append(_finality(1, source="approved-source", synthetic=False, reference="statement"))
    production = _issue(
        journal, store, _LATE, policy=_PRODUCTION_LIKE, provenance=_PRODUCTION_REFERENCES
    )
    assert production.evidence_class is EquityCandleEvidenceClass.PRODUCTION
    assert (production.venue_provenance, production.applicability_provenance) == (_PROD, _PROD)
    assert _interval(production, 1).outcome is _OUTCOME.ACCEPTABLE
    # The same journal under synthetic references is synthetic, whatever the references are called.
    synthetic = _issue(journal, store, _LATE, policy=_PRODUCTION_LIKE)
    assert synthetic.evidence_class is EquityCandleEvidenceClass.SYNTHETIC
    production_named = "nse-accepted-equity-session-evidence renamed/1"
    renamed = _issue(
        journal,
        store,
        _LATE,
        policy=_PRODUCTION_LIKE,
        venue=production_named,
        provenance=_Provenance({production_named: _SYN, _APPLICABILITY_REF: _SYN}),
    )
    assert renamed.evidence_class is EquityCandleEvidenceClass.SYNTHETIC
    assert renamed.venue_reference == production_named
    with pytest.raises(TypeError):
        dataclasses.replace(production, evidence_class=EquityCandleEvidenceClass.PRODUCTION)


@pytest.mark.parametrize(("venue", "applicability"), [(_PROD, _SYN), (_SYN, _PROD), (_SYN, _SYN)])
def test_any_synthetic_reference_makes_the_snapshot_synthetic(venue, applicability) -> None:  # type: ignore[no-untyped-def]
    journal, store = _ports()
    journal.append(_observation(1, source="approved-source", synthetic=False))
    journal.append(_finality(1, source="approved-source", synthetic=False, reference="statement"))
    mixed = _issue(
        journal,
        store,
        _LATE,
        policy=_PRODUCTION_LIKE,
        provenance=_Provenance({_VENUE_REF: venue, _APPLICABILITY_REF: applicability}),
    )
    assert mixed.evidence_class is EquityCandleEvidenceClass.SYNTHETIC


def test_synthetic_records_or_policy_keep_production_references_synthetic() -> None:
    journal, store = _ports()
    _final_candle(journal, 1)
    with_synthetic_records = _issue(
        journal, store, _LATE, policy=_PRODUCTION_LIKE, provenance=_PRODUCTION_REFERENCES
    )
    assert with_synthetic_records.evidence_class is EquityCandleEvidenceClass.SYNTHETIC
    clean, clean_store = _ports()
    clean.append(_observation(1, source="approved-source", synthetic=False))
    synthetic_policy = _issue(clean, clean_store, _LATE, provenance=_PRODUCTION_REFERENCES)
    assert synthetic_policy.evidence_class is EquityCandleEvidenceClass.SYNTHETIC


def test_accepted_production_references_under_the_no_approved_finality_policy() -> None:
    journal, store = _ports()
    journal.append(_observation(1, source="approved-source", synthetic=False))
    journal.append(_finality(1, source="approved-source", synthetic=False, reference="statement"))
    snapshot = IssueEquityCandleEvidenceSnapshotUseCase(
        journal, store, _INR, reference_provenance=_PRODUCTION_REFERENCES
    ).execute(_window(_LATE), _pit(_LATE), _pit(_LATE))
    assert snapshot.finality_policy.version == "no-approved-equity-candle-finality/1"
    assert snapshot.evidence_class is EquityCandleEvidenceClass.PRODUCTION
    assert snapshot.observations[0].finding is _FINDING.FINALITY_SOURCE_NOT_APPROVED
    assert _interval(snapshot, 1).outcome is _OUTCOME.UNFINALIZED
    assert not snapshot.is_complete


def test_the_production_default_policy_approves_no_synthetic_finality() -> None:
    journal, store = _ports()
    _final_candle(journal, 1)
    snapshot = IssueEquityCandleEvidenceSnapshotUseCase(
        journal, store, _INR, reference_provenance=_Provenance()
    ).execute(_window(_LATE), _pit(_LATE), _pit(_LATE))
    assert snapshot.observations[0].finding is _FINDING.SYNTHETIC_FINALITY_NOT_PERMITTED
    assert _interval(snapshot, 1).outcome is _OUTCOME.UNFINALIZED
    assert snapshot.evidence_class is EquityCandleEvidenceClass.SYNTHETIC


class _Answering(EquityResearchReferenceProvenanceSource):
    def __init__(self, answer: object) -> None:
        self.answer = answer

    def classify(self, window: EquityResearchWindowEvaluation) -> EquityResearchWindowProvenance:
        return self.answer  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("provenance", "venue"),
    [
        (_Provenance(), "renamed-synthetic-venue/1"),
        (_Provenance({}), _VENUE_REF),
        (_Answering(None), _VENUE_REF),
        (
            _Answering(EquityResearchWindowProvenance("other/1", _SYN, _APPLICABILITY_REF, _SYN)),
            _VENUE_REF,
        ),
    ],
)
def test_unknown_or_missing_provenance_fails_closed(provenance, venue) -> None:  # type: ignore[no-untyped-def]
    journal, store = _ports()
    _final_candle(journal, 1)
    with pytest.raises(UnknownEquityReferenceProvenanceError):
        _issue(journal, store, _LATE, provenance=provenance, venue=venue)
    assert store.documents == {}


def test_provenance_must_be_supplied_and_well_formed() -> None:
    journal, store = _ports()
    with pytest.raises(TypeError):
        IssueEquityCandleEvidenceSnapshotUseCase(journal, store, _INR)  # type: ignore[call-arg]
    with pytest.raises(TypeError, match="never inferred"):
        IssueEquityCandleEvidenceSnapshotUseCase(
            journal,
            store,
            _INR,
            reference_provenance=None,  # type: ignore[arg-type]
        )
    with pytest.raises(UnknownEquityReferenceProvenanceError):
        EquityResearchWindowProvenance(_VENUE_REF, "ACCEPTED_PRODUCTION", _APPLICABILITY_REF, _SYN)  # type: ignore[arg-type]
    with pytest.raises(UnknownEquityReferenceProvenanceError):
        EquityResearchWindowProvenance(" ", _SYN, _APPLICABILITY_REF, _SYN)


def test_replay_requires_the_recorded_provenance() -> None:
    journal, store = _ports()
    _final_candle(journal, 1)
    snapshot = _issue(journal, store, _LATE)
    assert _replay(journal, store, snapshot).outcome is EquityCandleSnapshotReplayOutcome.REPRODUCED
    reclassified = _replay(journal, store, snapshot, provenance=_PRODUCTION_REFERENCES)
    assert reclassified.outcome is EquityCandleSnapshotReplayOutcome.REFERENCE_UNAVAILABLE
    assert "provenance" in reclassified.detail
    unknown = _replay(journal, store, snapshot, provenance=_Provenance({}))
    assert unknown.outcome is EquityCandleSnapshotReplayOutcome.REFERENCE_UNAVAILABLE


# ----------------------------------------------------------------------
# D. Manifest integrity, identity and replay
# ----------------------------------------------------------------------


def test_snapshot_identity_binds_the_manifest_and_its_issuance() -> None:
    journal, store = _ports()
    _final_candle(journal, 1)
    first = _issue(journal, store, _LATE, issued=_LATE)
    again = _issue(journal, store, _LATE, issued=_LATE)
    later = _issue(journal, store, _LATE, issued=_LATE + timedelta(hours=1))
    assert again == first
    assert later.manifest_fingerprint == first.manifest_fingerprint
    assert later.snapshot_id != first.snapshot_id
    assert len(store.documents) == 2
    assert store.issue(first) == 0


@pytest.mark.parametrize(
    "tamper",
    [
        lambda d: d["manifest"]["observations"][0]["record"]["payload"].__setitem__("close", "1"),
        lambda d: d["manifest"].__setitem__("coverage", "WINDOW_COMPLETE"),
        lambda d: d["manifest"]["intervals"][0].__setitem__("outcome", "MISSING"),
        lambda d: d["manifest"].__setitem__("watermark", 1),
        lambda d: d["manifest"].__setitem__("evidence_class", "PRODUCTION"),
        lambda d: d["manifest"]["finality_policy"].__setitem__("version", "synthetic-other/1"),
        lambda d: d.__setitem__("manifest_fingerprint", "ecm1-sha256:" + "0" * 64),
        lambda d: d.__setitem__("issued_at", "2031-03-04T11:00:00Z"),
        lambda d: d["manifest"]["observations"][0].__setitem__("finality_sequences", []),
        lambda d: d["manifest"]["window"].__setitem__("venue_provenance", "ACCEPTED_PRODUCTION"),
        lambda d: d["manifest"]["window"].__setitem__("applicability_provenance", "UNKNOWN"),
        lambda d: d["manifest"]["window"].pop("venue_provenance"),
    ],
)
def test_a_document_inconsistent_with_its_fingerprint_fails_closed(tamper) -> None:  # type: ignore[no-untyped-def]
    journal, store = _ports()
    _final_candle(journal, 1)
    snapshot = _issue(journal, store, _LATE)
    document = json.loads(equity_candle_snapshot_json(snapshot))
    tamper(document)
    with pytest.raises((ValueError, TypeError, KeyError)):
        equity_candle_snapshot_from_document(document)


def test_a_snapshot_cannot_claim_another_evidence_set() -> None:
    journal, store = _ports()
    _final_candle(journal, 1)
    _final_candle(journal, 2)

    class _Omitting(_Journal):
        def read_window(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            full = journal.read_window(*args, **kwargs)
            return dataclasses.replace(full, observations=full.observations[:1], finalities=())

    lying = _Omitting()
    lying.entries = journal.entries
    with pytest.raises(EquityCandleSnapshotEvidenceError, match="exactly"):
        _issue(lying, store, _LATE)
    assert store.documents == {}


class _Racing(_Journal):
    """Commits ``racing`` records right after the first ``rounds`` watermark reads."""

    def __init__(self, base: _Journal, racing, rounds: int) -> None:  # type: ignore[no-untyped-def]
        super().__init__()
        self.entries = base.entries
        self.racing = racing
        self.rounds = rounds
        self.reads = 0

    def watermark(self) -> int:
        mark = super().watermark()
        self.reads += 1
        if self.reads <= self.rounds:
            for record in self.racing(self.reads):
                self.append(record)
        return mark


def test_a_relevant_commit_after_the_watermark_is_never_omitted() -> None:
    journal, _ = _ports()
    _final_candle(journal, 1)
    racing = _Racing(
        journal,
        lambda _: (_observation(1, revision="r2", corrects="r1"), _finality(1, revision="r2")),
        rounds=1,
    )
    store = _Store(journal)
    snapshot = _issue(racing, store, _LATE)
    assert racing.reads == 2
    assert snapshot.watermark == 4
    assert snapshot.observation_sequences == (1, 3)
    assert _interval(snapshot, 1).outcome is _OUTCOME.UNRESOLVED_REVISIONS
    assert len(store.documents) == 1
    assert store.get_snapshot(snapshot.snapshot_id) == snapshot


def test_an_unrelated_commit_after_the_watermark_does_not_block_issuance() -> None:
    journal, _ = _ports()
    _final_candle(journal, 1)
    racing = _Racing(
        journal,
        lambda _: (_observation(1, listing=_OTHER, ingestion="other", event="evt-other"),),
        rounds=1,
    )
    store = _Store(journal)
    snapshot = _issue(racing, store, _LATE)
    assert racing.reads == 1
    assert snapshot.watermark == 2 and racing.watermark() == 3
    assert _interval(snapshot, 1).outcome is _OUTCOME.ACCEPTABLE


def test_issuance_gives_up_after_a_bounded_number_of_attempts() -> None:
    journal, _ = _ports()
    _final_candle(journal, 1)
    racing = _Racing(journal, lambda n: (_observation(1, revision=f"r{n + 1}"),), rounds=10**6)
    store = _Store(journal)
    with pytest.raises(EquityCandleSnapshotStaleError, match="3 issuance attempts"):
        _issue(racing, store, _LATE)
    assert racing.reads == 3
    assert store.documents == {}


def test_the_store_refuses_a_snapshot_the_journal_has_moved_past() -> None:
    journal, store = _ports()
    _final_candle(journal, 1)
    snapshot = _issue(journal, store, _LATE)
    journal.append(_observation(1, revision="r2"))
    fresh = _Store(journal)
    with pytest.raises(EquityCandleSnapshotStaleError, match="gained evidence"):
        fresh.issue(snapshot)
    assert fresh.documents == {}
    assert store.issue(snapshot) == 0
    assert store.get_snapshot(snapshot.snapshot_id) == snapshot


def test_replay_reports_unavailable_references_rather_than_substituting() -> None:
    journal, store = _ports()
    _final_candle(journal, 1)
    snapshot = _issue(journal, store, _LATE)
    other_policy = EquityCandleFinalityPolicy("synthetic-other/1", frozenset({_SOURCE}), True)
    unavailable = _replay(journal, store, snapshot, policy=other_policy)
    assert unavailable.outcome is EquityCandleSnapshotReplayOutcome.REFERENCE_UNAVAILABLE
    assert "finality policy" in unavailable.detail
    assert unavailable.snapshot == snapshot and unavailable.replayed is None
    moved = _replay(journal, store, snapshot, as_of=_LATE + timedelta(minutes=1))
    assert moved.outcome is EquityCandleSnapshotReplayOutcome.REFERENCE_UNAVAILABLE
    use_case = ReplayEquityCandleEvidenceSnapshotUseCase(
        journal, store, _INR, _SYNTHETIC, reference_provenance=_Provenance()
    )
    missing = use_case.execute("ecs1-sha256:" + "0" * 64, _window(_LATE))
    assert missing.outcome is EquityCandleSnapshotReplayOutcome.NOT_FOUND


def test_replay_reports_divergence_when_the_journal_disagrees() -> None:
    journal, store = _ports()
    _final_candle(journal, 1)
    snapshot = _issue(journal, store, _LATE)

    class _Unchecked(_Store):
        def get_snapshot(self, snapshot_id: str) -> EquityCandleEvidenceSnapshot | None:
            return snapshot

    altered = _Journal()
    altered.append(_observation(1, prices=("100", "101", "99", "100.75")))
    altered.append(_finality(1))
    result = ReplayEquityCandleEvidenceSnapshotUseCase(
        altered, _Unchecked(altered), _INR, _SYNTHETIC, reference_provenance=_Provenance()
    ).execute(snapshot.snapshot_id, _window(_LATE))
    assert result.outcome is EquityCandleSnapshotReplayOutcome.DIVERGED
    assert result.replayed is not None
    assert result.replayed.manifest_fingerprint != snapshot.manifest_fingerprint


def test_snapshot_construction_refuses_inconsistent_content() -> None:
    journal, store = _ports()
    _final_candle(journal, 1)
    snapshot = _issue(journal, store, _LATE)
    with pytest.raises(InvalidEquityCandleEvidenceSnapshotError, match="watermark"):
        dataclasses.replace(snapshot, watermark=1)
    with pytest.raises(InvalidEquityCandleEvidenceSnapshotError, match="coverage"):
        dataclasses.replace(snapshot, coverage=_COVERAGE.WINDOW_COMPLETE)
    with pytest.raises(InvalidEquityCandleEvidenceSnapshotError, match="exactly one interval"):
        dataclasses.replace(snapshot, unmatched_sequences=(1,))
    with pytest.raises(InvalidEquityCandleEvidenceSnapshotError, match="compose"):
        dataclasses.replace(
            snapshot,
            observations=(dataclasses.replace(snapshot.observations[0], finality_sequences=()),),
        )


def test_repeated_issuance_over_the_same_journal_is_deterministic() -> None:
    journal, store = _ports()
    for index in (3, 1, 2):
        _final_candle(journal, index)
    snapshots = {_issue(journal, store, _LATE, issued=_LATE).snapshot_id for _ in range(3)}
    assert len(snapshots) == 1


# ----------------------------------------------------------------------
# E. Scope
# ----------------------------------------------------------------------


@pytest.mark.parametrize("module", [journal_module, snapshot_module, port_module, provenance_port])
def test_modules_read_no_clock_and_import_no_infrastructure(module) -> None:  # type: ignore[no-untyped-def]
    tree = ast.parse(inspect.getsource(module))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not called & {"now", "today", "utcnow", "monotonic", "time", "sleep"}
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    for name in imported:
        assert name is not None
        assert not name.startswith("northstar_infrastructure"), name
        for forbidden in ("sqlite3", "socket", "urllib", "requests", "recommendation", "broker"):
            assert forbidden not in name, name


def test_packages_export_the_ei2b_contracts() -> None:
    for name in (
        "EquityCandleObservationRecord",
        "EquityCandleFinalityRecord",
        "EquityCandleEvidenceSnapshot",
        "IssueEquityCandleEvidenceSnapshotUseCase",
        "ReplayEquityCandleEvidenceSnapshotUseCase",
        "require_equity_candle_snapshot_evidence",
    ):
        assert name in application_services.__all__
    assert "require_equity_candle_snapshot_current" in application_services.__all__
    for name in (
        "EquityCandleEvidenceJournal",
        "EquityCandleEvidenceSnapshotStore",
        "EquityCandleSnapshotStaleError",
        "EquityResearchReferenceProvenanceSource",
    ):
        assert name in ports.__all__
    for forbidden in ("Status", "State", "Pending"):
        assert not [name for name in ports.__all__ if forbidden in name]
