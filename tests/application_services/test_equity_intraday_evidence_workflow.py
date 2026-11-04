"""EI-2C: synthetic intraday ingestion and research-evidence issuance (Application).

Everything is invented: a 2031 window over in-test reference doubles, and
synthetic candles and finality from one synthetic source. The journal and
snapshot store are in-memory doubles applying the shared EI-2B rules; SQLite
acceptance is in Infrastructure. Nothing here is a decision, signal or order.
"""

from __future__ import annotations

import ast
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

from northstar_application import application_services
from northstar_application.application_services import (
    EquityCandleAppendOutcome,
    EquityCandleAppendResult,
    EquityCandleCoverage,
    EquityCandleEvidenceSnapshot,
    EquityCandleFinalityEvidence,
    EquityCandleFinalityPolicy,
    EquityCandleFinalityRecord,
    EquityCandleIngestionOutcome,
    EquityCandleJournalEntry,
    EquityCandleJournalEntryKind,
    EquityCandleJournalRecord,
    EquityCandleJournalWindow,
    EquityCandleObservation,
    EquityCandleObservationRecord,
    EquityResearchEvidenceOutcome,
    EquityResearchWindowEvaluation,
    EvaluateEquityResearchWindowUseCase,
    IngestSyntheticEquityCandleEventsUseCase,
    IssueEquityCandleEvidenceSnapshotUseCase,
    IssueEquityResearchEvidenceUseCase,
    equity_candle_identity_conflicts,
    equity_candle_in_window,
    equity_candle_retry,
    equity_candle_snapshot_from_document,
    equity_candle_snapshot_json,
    require_equity_candle_snapshot_current,
    require_equity_candle_snapshot_evidence,
)
from northstar_application.application_services import (
    equity_intraday_evidence_workflow as workflow_module,
)
from northstar_application.ports import (
    EquityCandleEvidenceJournal,
    EquityCandleEvidenceSnapshotStore,
    EquityCandleEvidenceUnavailableError,
    EquityListingApplicability,
    EquityListingApplicabilityClassification,
    EquityListingApplicabilitySource,
    EquityReferenceProvenanceKind,
    EquityResearchReferenceProvenanceSource,
    EquityResearchWindowProvenance,
    EquityVenueDayClassification,
    EquityVenueDayReference,
    EquityVenueDayReferenceSource,
    UnknownEquityReferenceProvenanceError,
)

_NSE = ExchangeCode("NSE")
_INR = Currency("INR")
_DAY = date(2031, 3, 4)
_LISTING = ListingReference(Symbol("M&M"), _NSE)
_FIRST_END = datetime(2031, 3, 4, 3, 50, tzinfo=UTC)
_LATE = datetime(2031, 3, 4, 10, 0, tzinfo=UTC)
_SOURCE = "synthetic-ei2c-source"
_POLICY = EquityCandleFinalityPolicy("synthetic-ei2c/1", frozenset({_SOURCE}), True)
_ING = EquityCandleIngestionOutcome
_SYN = EquityReferenceProvenanceKind.SYNTHETIC


def _pit(value: datetime) -> PointInTime:
    return PointInTime(value.isoformat().replace("+00:00", "Z"))


def _end(index: int) -> datetime:
    return _FIRST_END + timedelta(minutes=5 * (index - 1))


class _Venue(EquityVenueDayReferenceSource):
    def venue_day(self, exchange_code: ExchangeCode, trading_date: date) -> EquityVenueDayReference:
        return EquityVenueDayReference(
            exchange_code=exchange_code,
            trading_date=trading_date,
            classification=EquityVenueDayClassification.NORMAL_SESSION,
            common_opens_at=PointInTime("2031-03-04T03:45:00Z"),
            common_closes_at=PointInTime("2031-03-04T09:45:00Z"),
            utc_offset=timedelta(hours=5, minutes=30),
            reference="test-venue/1",
            reason="test",
        )


class _Applicability(EquityListingApplicabilitySource):
    def applicability(
        self, listing: ListingReference, trading_date: date
    ) -> EquityListingApplicability:
        eligible = listing == _LISTING and trading_date == _DAY
        return EquityListingApplicability(
            listing,
            trading_date,
            EquityListingApplicabilityClassification.ELIGIBLE_FOR_NORMAL_RESEARCH_WINDOW
            if eligible
            else EquityListingApplicabilityClassification.UNKNOWN,
            "test-applicability/1",
            "test",
        )


class _Provenance(EquityResearchReferenceProvenanceSource):
    def __init__(self, known: bool = True) -> None:
        self.known = known

    def classify(self, window: EquityResearchWindowEvaluation) -> EquityResearchWindowProvenance:
        if not self.known:
            raise UnknownEquityReferenceProvenanceError("unknown test reference")
        assert window.venue_reference and window.applicability_reference
        return EquityResearchWindowProvenance(
            window.venue_reference, _SYN, window.applicability_reference, _SYN
        )


class _Journal(EquityCandleEvidenceJournal):
    def __init__(self, fail_at: int | None = None) -> None:
        self.entries: list[EquityCandleJournalEntry] = []
        self.fail_at = fail_at
        self.calls = 0

    def append(self, record: EquityCandleJournalRecord) -> EquityCandleAppendResult:
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise EquityCandleEvidenceUnavailableError("simulated storage failure")
        same = next((e for e in self.entries if e.record.ingestion_id == record.ingestion_id), None)
        retry = equity_candle_retry(record, same)
        if retry is not None:
            return EquityCandleAppendResult(EquityCandleAppendOutcome.ALREADY_APPENDED, retry)
        conflicts = equity_candle_identity_conflicts(
            record,
            tuple(e for e in self.entries if e.record.source_identity == record.source_identity),
        )
        entry = EquityCandleJournalEntry(len(self.entries) + 1, _pit(_LATE), record)
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

    def _read(self, snapshot: EquityCandleEvidenceSnapshot, watermark: int):  # type: ignore[no-untyped-def]
        return self.journal.read_window(
            snapshot.listing,
            snapshot.timeframe,
            snapshot.window_opens_at,
            snapshot.window_closes_at,
            watermark,
        )

    def issue(self, snapshot: EquityCandleEvidenceSnapshot) -> int:
        if snapshot.snapshot_id in self.documents:
            return 0
        require_equity_candle_snapshot_evidence(snapshot, self._read(snapshot, snapshot.watermark))
        require_equity_candle_snapshot_current(
            snapshot, self._read(snapshot, self.journal.watermark())
        )
        self.documents[snapshot.snapshot_id] = equity_candle_snapshot_json(snapshot)
        return 1

    def get_snapshot(self, snapshot_id: str) -> EquityCandleEvidenceSnapshot | None:
        text = self.documents.get(snapshot_id)
        return None if text is None else equity_candle_snapshot_from_document(json.loads(text))


def _observation(
    index: int,
    *,
    revision: str = "r1",
    ingestion: str | None = None,
    source: str = _SOURCE,
    synthetic: bool = True,
    timeframe: str = "5m",
    close: str = "100.5",
    corrects: str | None = None,
    end_index: int | None = None,
) -> EquityCandleObservationRecord:
    end = _end(end_index or index)
    price = lambda p: Price(Decimal(p), _INR)  # noqa: E731
    bar = HistoricalOHLCVBar(
        symbol=_LISTING.symbol,
        exchange_code=_NSE,
        point_in_time=_pit(end),
        timeframe=Timeframe(timeframe),
        open=price("100"),
        high=price("101"),
        low=price("99"),
        close=price(close),
        volume=Quantity(Decimal("1000")),
    )
    return EquityCandleObservationRecord(
        ingestion or f"obs-{index}-{revision}",
        EquityCandleObservation(
            bar, source, f"evt-{index}", revision, _pit(end + timedelta(seconds=5)), None, corrects
        ),
        synthetic,
    )


def _finality(index: int, *, revision: str = "r1", received: datetime | None = None):  # type: ignore[no-untyped-def]
    confirmed = _end(index) + timedelta(seconds=10)
    return EquityCandleFinalityRecord(
        f"fin-{index}-{revision}",
        EquityCandleFinalityEvidence(
            _SOURCE, f"evt-{index}", revision, _pit(confirmed), "synthetic statement", True
        ),
        _pit(received or confirmed + timedelta(seconds=1)),
    )


def _events(count: int, skip_finality: int | None = None) -> list[EquityCandleJournalRecord]:
    events: list[EquityCandleJournalRecord] = []
    for index in range(1, count + 1):
        events.append(_observation(index))
        if index != skip_finality:
            events.append(_finality(index))
    return events


def _ingest(journal: _Journal):  # type: ignore[no-untyped-def]
    return IngestSyntheticEquityCandleEventsUseCase(journal, frozenset({_SOURCE}))


def _issuer(journal: _Journal, store: _Store, provenance: bool = True):  # type: ignore[no-untyped-def]
    return IssueEquityResearchEvidenceUseCase(
        EvaluateEquityResearchWindowUseCase(_Venue(), _Applicability()),
        IssueEquityCandleEvidenceSnapshotUseCase(
            journal, store, _INR, _POLICY, reference_provenance=_Provenance(provenance)
        ),
    )


def _issue(journal: _Journal, store: _Store, as_of: datetime, **kwargs):  # type: ignore[no-untyped-def]
    listing = kwargs.pop("listing", _LISTING)
    day = kwargs.pop("day", _DAY)
    return _issuer(journal, store, **kwargs).execute(
        listing, day, _pit(as_of), _pit(as_of + timedelta(minutes=1))
    )


# ----------------------------------------------------------------------
# Ingestion outcomes
# ----------------------------------------------------------------------


def test_events_are_appended_in_submitted_order_with_their_sequences() -> None:
    journal = _Journal()
    report = _ingest(journal).execute(_events(3))
    assert [r.outcome for r in report.results] == [
        _ING.APPENDED_OBSERVATION,
        _ING.APPENDED_FINALITY,
    ] * 3
    assert [r.sequence for r in report.results] == [1, 2, 3, 4, 5, 6]
    assert [r.position for r in report.results] == list(range(6))
    assert not report.interrupted and report.resume_from is None
    assert all(r.is_journaled for r in report.results)


def test_a_resubmitted_batch_is_entirely_idempotent() -> None:
    journal = _Journal()
    first = _ingest(journal).execute(_events(3))
    again = _ingest(journal).execute(_events(3))
    assert again.count(_ING.IDEMPOTENT_RETRY) == 6
    assert [r.sequence for r in again.results] == [r.sequence for r in first.results]
    assert journal.watermark() == 6


def test_invalid_events_are_rejected_explicitly_and_the_batch_continues() -> None:
    journal = _Journal()
    events = [
        {"ingestion_id": "raw-dict"},
        _observation(1, synthetic=False),
        _observation(2, source="unconfigured-source"),
        _observation(3, timeframe="1m"),
        _observation(4),
    ]
    report = _ingest(journal).execute(events)
    assert [r.outcome for r in report.results] == [_ING.REJECTED_INVALID] * 4 + [
        _ING.APPENDED_OBSERVATION
    ]
    assert report.results[0].ingestion_id is None
    assert "synthetic" in report.results[1].detail
    assert "configured synthetic source" in report.results[2].detail
    assert "Timeframe" in report.results[3].detail
    assert all(r.sequence is None for r in report.results[:4])
    assert journal.watermark() == 1


def test_conflicts_are_rejected_without_overwriting() -> None:
    journal = _Journal()
    _ingest(journal).execute([_observation(1)])
    report = _ingest(journal).execute(
        [
            _observation(1, close="100.75"),
            _observation(1, ingestion="moved", end_index=2),
            _observation(1, ingestion="other-values", close="100.75"),
        ]
    )
    assert [r.outcome for r in report.results] == [
        _ING.REJECTED_INGESTION_CONFLICT,
        _ING.REJECTED_SOURCE_IDENTITY_CONFLICT,
        _ING.APPENDED_OBSERVATION,
    ]
    assert report.results[2].conflicting_values == (1,)
    assert journal.entries[0].record == _observation(1)
    assert journal.watermark() == 2


def test_a_storage_failure_stops_the_batch_and_resumption_completes_it() -> None:
    events = _events(4)
    journal = _Journal(fail_at=5)
    report = _ingest(journal).execute(events)
    assert report.interrupted and report.resume_from == 4
    assert [r.outcome for r in report.results[4:]] == [_ING.FAILED_PERSISTENCE] + [
        _ING.NOT_ATTEMPTED
    ] * 3
    assert journal.watermark() == 4
    journal.fail_at = None
    resumed = _ingest(journal).execute(events)
    assert resumed.count(_ING.IDEMPOTENT_RETRY) == 4
    assert resumed.count(_ING.APPENDED_OBSERVATION) + resumed.count(_ING.APPENDED_FINALITY) == 4
    clean = _Journal()
    _ingest(clean).execute(events)
    assert [e.record for e in journal.entries] == [e.record for e in clean.entries]


def test_batch_boundaries_do_not_change_the_journal() -> None:
    events = _events(5)
    whole, split = _Journal(), _Journal()
    _ingest(whole).execute(events)
    _ingest(split).execute(events[:3])
    _ingest(split).execute(events[2:7])
    _ingest(split).execute(events[6:])
    assert [e.record for e in split.entries] == [e.record for e in whole.entries]


def test_finality_before_its_observation_is_journaled_and_composes_later() -> None:
    journal, store = _Journal(), None
    report = _ingest(journal).execute([_finality(1), _observation(1)])
    assert [r.outcome for r in report.results] == [
        _ING.APPENDED_FINALITY,
        _ING.APPENDED_OBSERVATION,
    ]
    store = _Store(journal)
    issued = _issue(journal, store, _end(1) + timedelta(minutes=1))
    assert issued.acceptable == (_pit(_end(1)),)


def test_the_use_case_checks_its_configuration() -> None:
    with pytest.raises(TypeError):
        IngestSyntheticEquityCandleEventsUseCase(object(), frozenset({_SOURCE}))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="synthetic_source_ids"):
        IngestSyntheticEquityCandleEventsUseCase(_Journal(), frozenset())
    with pytest.raises(ValueError, match="synthetic_source_ids"):
        IngestSyntheticEquityCandleEventsUseCase(_Journal(), {_SOURCE})  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Issuance outcomes
# ----------------------------------------------------------------------


def test_a_partial_session_is_complete_as_of_its_instant() -> None:
    journal = _Journal()
    store = _Store(journal)
    _ingest(journal).execute(_events(10))
    issued = _issue(journal, store, _end(10) + timedelta(minutes=1))
    assert issued.outcome is EquityResearchEvidenceOutcome.ISSUED
    assert issued.coverage is EquityCandleCoverage.COMPLETE_AS_OF
    assert len(issued.acceptable) == 10 and len(issued.not_ended) == 59
    assert issued.is_complete


def test_missing_finality_is_reported_as_unfinalized() -> None:
    journal = _Journal()
    store = _Store(journal)
    _ingest(journal).execute(_events(69, skip_finality=7))
    issued = _issue(journal, store, _LATE)
    assert issued.unfinalized == (_pit(_end(7)),)
    assert issued.coverage is EquityCandleCoverage.INCOMPLETE_AS_OF
    assert not issued.is_complete


def test_conflicts_are_reported_and_never_resolved() -> None:
    journal = _Journal()
    store = _Store(journal)
    _ingest(journal).execute(_events(69))
    _ingest(journal).execute(
        [_observation(5, revision="r2", corrects="r1", close="100.75"), _finality(5, revision="r2")]
    )
    issued = _issue(journal, store, _LATE)
    assert issued.conflicted == (_pit(_end(5)),)
    assert issued.coverage is EquityCandleCoverage.INCOMPLETE_AS_OF


def test_an_ineligible_window_issues_nothing() -> None:
    journal = _Journal()
    store = _Store(journal)
    _ingest(journal).execute(_events(3))
    other = ListingReference(Symbol("TESTCO"), _NSE)
    for listing, day in ((other, _DAY), (_LISTING, date(2031, 3, 5))):
        issued = _issue(journal, store, _LATE, listing=listing, day=day)
        assert issued.outcome is EquityResearchEvidenceOutcome.WINDOW_INELIGIBLE
        assert issued.snapshot is None and issued.coverage is None and not issued.is_complete
        assert issued.window.rejection is not None
        assert str(issued.window.rejection) in issued.detail
    assert store.documents == {}


def test_unknown_provenance_issues_nothing() -> None:
    journal = _Journal()
    store = _Store(journal)
    _ingest(journal).execute(_events(1))
    issued = _issue(journal, store, _LATE, provenance=False)
    assert issued.outcome is EquityResearchEvidenceOutcome.PROVENANCE_UNKNOWN
    assert store.documents == {}


def test_a_window_that_never_settles_is_reported_stale() -> None:
    class _Churning(_Journal):
        def watermark(self) -> int:
            mark = super().watermark()
            self.append(_observation(1, revision=f"r{mark}", ingestion=f"churn-{mark}"))
            return mark

    journal = _Churning()
    store = _Store(journal)
    _ingest(journal).execute(_events(1))
    issued = _issue(journal, store, _LATE)
    assert issued.outcome is EquityResearchEvidenceOutcome.STALE_AFTER_RETRIES
    assert issued.snapshot is None and store.documents == {}


def test_the_coordinator_checks_its_collaborators() -> None:
    with pytest.raises(TypeError):
        IssueEquityResearchEvidenceUseCase(object(), object())  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Scope
# ----------------------------------------------------------------------


def test_the_workflow_reads_no_clock_and_imports_no_infrastructure() -> None:
    tree = ast.parse(inspect.getsource(workflow_module))
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
        for forbidden in ("northstar_infrastructure", "sqlite", "socket", "broker", "recommend"):
            assert forbidden not in name, name
    # No catch-all: only named EI-2B errors are caught.
    handlers = [node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]
    assert handlers and all(h.type is not None for h in handlers)
    caught = {ast.unparse(h.type) for h in handlers}  # type: ignore[arg-type]
    assert not caught & {"Exception", "BaseException", "ValueError", "RuntimeError"}


def test_the_workflow_contracts_are_exported() -> None:
    for name in (
        "IngestSyntheticEquityCandleEventsUseCase",
        "IssueEquityResearchEvidenceUseCase",
        "EquityCandleIngestionOutcome",
        "EquityResearchEvidenceOutcome",
    ):
        assert name in application_services.__all__
