"""EI-3C-A: authoritative synthetic equity intraday research timeline advancement.

Everything is invented: an M&M@NSE session on 2026-08-12 over in-test
reference doubles, synthetic candles and finality, and a synthetic
calibration that is an example, never an approved parameter. Snapshots are
issued through the real EI-2B use case over in-memory doubles, as in the
EI-3A and EI-3B tests, and every timeline is produced by the real EI-3B
scheduler. The timeline store is an in-memory compare-and-append double whose
failures -- a competing writer, a lost acknowledgment, an unavailable read --
are injected explicitly, with no threads, sleeps or clocks. Stored records
altered after the fact are test-only fixtures and say so.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta, timezone
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
from northstar_core.strategy import StrategyIdentity

from northstar_application import application_services, ports
from northstar_application.application_services import (
    EQUITY_INTRADAY_RESEARCH_TIMELINE_COMMIT_ATTEMPTS,
    NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW,
    AdvanceEquityIntradayResearchTimelineUseCase,
    EquityCandleAppendOutcome,
    EquityCandleAppendResult,
    EquityCandleEvidenceSnapshot,
    EquityCandleFinalityEvidence,
    EquityCandleFinalityPolicy,
    EquityCandleFinalityRecord,
    EquityCandleJournalEntry,
    EquityCandleJournalEntryKind,
    EquityCandleJournalRecord,
    EquityCandleJournalWindow,
    EquityCandleObservation,
    EquityCandleObservationRecord,
    EquityIntradayResearchScheduleOutcome,
    EquityIntradayResearchScheduleRefusal,
    EquityIntradayResearchTimeline,
    EquityIntradayResearchTimelineAdvance,
    EquityIntradayResearchTimelineAdvanceStatus,
    EquityIntradayResearchTimelineEntry,
    EquityIntradayResearchTimelineIdentity,
    EquityIntradayResearchTimelineRole,
    EquityOpeningRangeBreakoutConfiguration,
    EquityOpeningRangeBreakoutEvaluator,
    EquityResearchWindowEvaluation,
    EvaluateEquityIntradayResearchUseCase,
    EvaluateEquityResearchWindowUseCase,
    IssueEquityCandleEvidenceSnapshotUseCase,
    StoredEquityIntradayResearchTimeline,
    StoredEquityIntradayResearchTimelineEntry,
    canonical_equity_candle_json,
    equity_candle_identity_conflicts,
    equity_candle_in_window,
    equity_candle_retry,
    equity_candle_snapshot_from_document,
    equity_candle_snapshot_json,
    rehydrate_equity_intraday_research_timeline,
    stored_equity_intraday_research_timeline,
)
from northstar_application.application_services import (
    advance_equity_intraday_research_timeline as advance_module,
)
from northstar_application.ports import (
    EquityCandleEvidenceJournal,
    EquityCandleEvidenceSnapshotStore,
    EquityCandleEvidenceUnavailableError,
    EquityIntradayResearchDecisionNotReproducedError,
    EquityIntradayResearchHistoricalEvaluationUnavailableError,
    EquityIntradayResearchTimelineCommitUnconfirmedError,
    EquityIntradayResearchTimelineCommitUnresolvedError,
    EquityIntradayResearchTimelineConfigurationConflictError,
    EquityIntradayResearchTimelineStaleError,
    EquityIntradayResearchTimelineStore,
    EquityIntradayResearchTimelineUnavailableError,
    EquityListingApplicability,
    EquityListingApplicabilityClassification,
    EquityListingApplicabilitySource,
    EquityReferenceProvenanceKind,
    EquityResearchReferenceProvenanceSource,
    EquityResearchWindowProvenance,
    EquityVenueDayClassification,
    EquityVenueDayReference,
    EquityVenueDayReferenceSource,
    InvalidStoredEquityIntradayResearchTimelineError,
)
from northstar_application.ports import (
    equity_intraday_research_timeline_store as port_module,
)

_NSE = ExchangeCode("NSE")
_INR = Currency("INR")
_DAY = date(2026, 8, 12)
_NEXT_DAY = date(2026, 8, 13)
_LISTING = ListingReference(Symbol("M&M"), _NSE)
_OTHER = ListingReference(Symbol("RELIANCE"), _NSE)
_IST = timezone(timedelta(hours=5, minutes=30))
_SOURCE = "synthetic-ei3c-source"
_FINALITY = EquityCandleFinalityPolicy("synthetic-ei3c/1", frozenset({_SOURCE}), True)
_CONFIG = EquityOpeningRangeBreakoutConfiguration(
    strategy=StrategyIdentity("ei3c-orb-vwap-volume"),
    version="synthetic-ei3c-test/1",
    opening_range_bars=3,
    volume_multiplier=Decimal("1.5"),
    minimum_baseline_bars=3,
)
_STATUS = EquityIntradayResearchTimelineAdvanceStatus
_ROLE = EquityIntradayResearchTimelineRole
_DONE = EquityIntradayResearchScheduleOutcome
_WHY = EquityIntradayResearchScheduleRefusal

Candle = tuple[str, str, str, str, str]

# Bars 1-3 form the opening range (high 102); bar 4 (09:35 IST) breaks out and
# bars 5 and 6 (09:40, 09:45 IST) qualify again.
_MORNING: dict[int, Candle] = {
    1: ("100", "101", "99", "100.5", "1000"),
    2: ("100.5", "102", "100", "101.5", "1000"),
    3: ("101.5", "102", "100.5", "101", "1000"),
    4: ("101", "103", "101", "102.5", "2000"),
    5: ("102.5", "104", "102", "103.5", "3000"),
    6: ("103.5", "105", "103", "104.5", "4500"),
}


def _ist(hour: int, minute: int, second: int = 0, day: date = _DAY) -> datetime:
    local = datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=_IST)
    return local.astimezone(UTC)


def _end(index: int, day: date = _DAY) -> datetime:
    return _ist(9, 15, day=day) + timedelta(minutes=5 * index)


def _pit(value: datetime) -> PointInTime:
    return PointInTime(value.isoformat().replace("+00:00", "Z"))


# ----------------------------------------------------------------------
# Evidence doubles (shared EI-2B rules)
# ----------------------------------------------------------------------


class _Venue(EquityVenueDayReferenceSource):
    def venue_day(self, exchange_code: ExchangeCode, trading_date: date) -> EquityVenueDayReference:
        return EquityVenueDayReference(
            exchange_code=exchange_code,
            trading_date=trading_date,
            classification=EquityVenueDayClassification.NORMAL_SESSION,
            common_opens_at=_pit(_ist(9, 15, day=trading_date)),
            common_closes_at=_pit(_ist(15, 30, day=trading_date)),
            utc_offset=timedelta(hours=5, minutes=30),
            reference="test-venue/1",
            reason="test",
        )


class _Applicability(EquityListingApplicabilitySource):
    def applicability(
        self, listing: ListingReference, trading_date: date
    ) -> EquityListingApplicability:
        return EquityListingApplicability(
            listing,
            trading_date,
            EquityListingApplicabilityClassification.ELIGIBLE_FOR_NORMAL_RESEARCH_WINDOW,
            "test-applicability/1",
            "test",
        )


class _Provenance(EquityResearchReferenceProvenanceSource):
    def classify(self, window: EquityResearchWindowEvaluation) -> EquityResearchWindowProvenance:
        assert window.venue_reference and window.applicability_reference
        synthetic = EquityReferenceProvenanceKind.SYNTHETIC
        return EquityResearchWindowProvenance(
            window.venue_reference, synthetic, window.applicability_reference, synthetic
        )


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
        entry = EquityCandleJournalEntry(len(self.entries) + 1, _pit(_ist(23, 0)), record)
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


class _Snapshots(EquityCandleEvidenceSnapshotStore):
    def __init__(self) -> None:
        self.documents: dict[str, str] = {}

    def issue(self, snapshot: EquityCandleEvidenceSnapshot) -> int:
        self.documents.setdefault(snapshot.snapshot_id, equity_candle_snapshot_json(snapshot))
        return 1

    def get_snapshot(self, snapshot_id: str) -> EquityCandleEvidenceSnapshot | None:
        text = self.documents.get(snapshot_id)
        return None if text is None else equity_candle_snapshot_from_document(json.loads(text))


class _UnavailableSnapshots(_Snapshots):
    def get_snapshot(self, snapshot_id: str) -> EquityCandleEvidenceSnapshot | None:
        raise EquityCandleEvidenceUnavailableError("test: snapshot storage is unavailable")


def _event(index: int, listing: ListingReference, day: date) -> str:
    return f"evt-{listing.symbol.value}-{day.isoformat()}-{index}"


def _observation(
    index: int, candle: Candle, listing: ListingReference, day: date
) -> EquityCandleObservationRecord:
    o, h, low, c, v = candle
    end = _end(index, day)
    bar = HistoricalOHLCVBar(
        symbol=listing.symbol,
        exchange_code=_NSE,
        point_in_time=_pit(end),
        timeframe=Timeframe("5m"),
        open=Price(Decimal(o), _INR),
        high=Price(Decimal(h), _INR),
        low=Price(Decimal(low), _INR),
        close=Price(Decimal(c), _INR),
        volume=Quantity(Decimal(v)),
    )
    event = _event(index, listing, day)
    return EquityCandleObservationRecord(
        f"obs-{event}",
        EquityCandleObservation(bar, _SOURCE, event, "r1", _pit(end + timedelta(seconds=5)), None),
        True,
    )


def _finality(index: int, listing: ListingReference, day: date) -> EquityCandleFinalityRecord:
    end = _end(index, day)
    event = _event(index, listing, day)
    return EquityCandleFinalityRecord(
        f"fin-{event}",
        EquityCandleFinalityEvidence(
            _SOURCE, event, "r1", _pit(end + timedelta(seconds=10)), "synthetic", True
        ),
        _pit(end + timedelta(seconds=11)),
    )


class _Session:
    """A journal, a snapshot store and the EI-2B issue use case."""

    def __init__(self) -> None:
        self.journal, self.snapshots = _Journal(), _Snapshots()
        self.issuer = IssueEquityCandleEvidenceSnapshotUseCase(
            self.journal,
            self.snapshots,
            _INR,
            _FINALITY,
            NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW,
            reference_provenance=_Provenance(),
        )

    def add(
        self,
        candles: dict[int, Candle],
        listing: ListingReference = _LISTING,
        day: date = _DAY,
    ) -> None:
        for index, candle in sorted(candles.items()):
            self.journal.append(_observation(index, candle, listing, day))
            self.journal.append(_finality(index, listing, day))

    def issue(
        self,
        as_of: datetime,
        issued: datetime | None = None,
        listing: ListingReference = _LISTING,
        day: date = _DAY,
    ) -> EquityCandleEvidenceSnapshot:
        window = EvaluateEquityResearchWindowUseCase(
            _Venue(), _Applicability(), NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW
        ).execute(listing, day, _pit(as_of))
        return self.issuer.execute(window, _pit(as_of), _pit(issued or as_of))

    def at(self, *minutes: int) -> list[EquityCandleEvidenceSnapshot]:
        """Issue one snapshot at each 09:MM IST instant."""
        return [self.issue(_ist(9, minute)) for minute in minutes]

    def unrelated_entry(self) -> None:
        """Journal a candle outside M&M's window, so the next snapshot has a higher watermark."""
        self.journal.append(
            _observation(len(self.journal.entries) + 1, _MORNING[1], _OTHER, _NEXT_DAY)
        )


def _morning() -> _Session:
    session = _Session()
    session.add(_MORNING)
    return session


# ----------------------------------------------------------------------
# The timeline store double
# ----------------------------------------------------------------------

_Key = tuple[StrategyIdentity, ListingReference, date]


class _TimelineStore(EquityIntradayResearchTimelineStore):
    """Test-only compare-and-append store with explicitly injected failures.

    ``before_append`` and ``after_commit`` hooks run once each, in order, for
    the outer writer's calls -- never for a writer they themselves run.
    ``drop_commits`` raises an unconfirmed commit without committing;
    ``lose_acks`` commits, then raises one.
    """

    def __init__(self) -> None:
        self.records: dict[_Key, StoredEquityIntradayResearchTimeline] = {}
        self.calls: list[tuple[StoredEquityIntradayResearchTimeline, str | None]] = []
        self.commits = 0
        self.before_append: list[Callable[[], object]] = []
        self.after_commit: list[Callable[[], object]] = []
        self.drop_commits = 0
        self.lose_acks = 0
        self.unavailable = False
        self._nested = False

    def load(
        self, strategy: StrategyIdentity, listing: ListingReference, trading_date: date
    ) -> StoredEquityIntradayResearchTimeline | None:
        if self.unavailable:
            raise EquityIntradayResearchTimelineUnavailableError("test: storage is unavailable")
        return self.records.get((strategy, listing, trading_date))

    def append(
        self,
        timeline: StoredEquityIntradayResearchTimeline,
        expected_head_fingerprint: str | None,
    ) -> None:
        self.calls.append((timeline, expected_head_fingerprint))
        outer = not self._nested
        if outer:
            self._run(self.before_append)
        if outer and self.drop_commits:
            self.drop_commits -= 1
            raise EquityIntradayResearchTimelineCommitUnconfirmedError("test: lost before commit")
        key = (timeline.strategy, timeline.listing, timeline.trading_date)
        stored = self.records.get(key)
        if stored is not None and stored.configuration_document != timeline.configuration_document:
            raise EquityIntradayResearchTimelineConfigurationConflictError("test: configuration")
        head = None if stored is None else stored.head_fingerprint
        if head != expected_head_fingerprint or (
            stored is not None and timeline.entries[: len(stored.entries)] != stored.entries
        ):
            raise EquityIntradayResearchTimelineStaleError("test: the head moved")
        self.records[key] = timeline
        self.commits += 1
        if outer:
            self._run(self.after_commit)
        if outer and self.lose_acks:
            self.lose_acks -= 1
            raise EquityIntradayResearchTimelineCommitUnconfirmedError("test: acknowledgment lost")

    def _run(self, hooks: list[Callable[[], object]]) -> None:
        if hooks:
            hook = hooks.pop(0)
            self._nested = True
            try:
                hook()
            finally:
                self._nested = False

    def record(
        self,
        listing: ListingReference = _LISTING,
        day: date = _DAY,
        strategy: StrategyIdentity | None = None,
    ) -> StoredEquityIntradayResearchTimeline:
        return self.records[(strategy or _CONFIG.strategy, listing, day)]


def _writer(
    session: _Session,
    store: _TimelineStore,
    configuration: EquityOpeningRangeBreakoutConfiguration = _CONFIG,
) -> AdvanceEquityIntradayResearchTimelineUseCase:
    return AdvanceEquityIntradayResearchTimelineUseCase(store, session.snapshots, configuration)


def _advance(
    session: _Session,
    store: _TimelineStore,
    snapshots: list[EquityCandleEvidenceSnapshot],
    configuration: EquityOpeningRangeBreakoutConfiguration = _CONFIG,
    listing: ListingReference = _LISTING,
    day: date = _DAY,
) -> EquityIntradayResearchTimelineAdvance:
    return _writer(session, store, configuration).execute(
        listing, day, [s.snapshot_id for s in snapshots]
    )


def _later(
    session: _Session, store: _TimelineStore, snapshots: list[EquityCandleEvidenceSnapshot]
) -> Callable[[], object]:
    """A competing writer that commits ``snapshots`` when the hook runs."""
    return lambda: _advance(session, store, snapshots)


def _roles(entries: tuple[object, ...]) -> list[EquityIntradayResearchTimelineRole]:
    return [entry.role for entry in entries]  # type: ignore[attr-defined]


def _results(advance: EquityIntradayResearchTimelineAdvance) -> list[tuple[object, object]]:
    return [(item.outcome, item.refusal) for item in advance.items]


def _identity(
    listing: ListingReference = _LISTING, day: date = _DAY
) -> EquityIntradayResearchTimelineIdentity:
    return EquityIntradayResearchTimelineIdentity(listing, day, _CONFIG)


def _conflicting(session: _Session) -> tuple[EquityCandleEvidenceSnapshot, ...]:
    """Two different manifests for 09:36 IST: the journal grows in between."""
    a = session.issue(_ist(9, 36))
    session.unrelated_entry()
    b = session.issue(_ist(9, 36), issued=_ist(9, 37))
    assert a.as_of == b.as_of and a.manifest_fingerprint != b.manifest_fingerprint
    return a, b


# ----------------------------------------------------------------------
# A, B -- establishment (EI3-D11)
# ----------------------------------------------------------------------


def test_a_the_first_committed_entry_establishes_the_timeline() -> None:
    session, store = _morning(), _TimelineStore()
    (s36,) = session.at(36)
    result = _advance(session, store, [s36])
    assert result.status is _STATUS.COMMITTED
    assert (result.commit_confirmed, result.reconciled, result.attempts) == (True, False, 1)
    assert len(store.calls) == 1 and store.calls[0][1] is None
    record = store.record()
    assert (record.strategy, record.listing, record.trading_date) == (
        _CONFIG.strategy,
        _LISTING,
        _DAY,
    )
    assert record.configuration_document == canonical_equity_candle_json(_CONFIG.document())
    timeline = result.timeline
    assert timeline is not None and timeline == result.schedule.timeline  # type: ignore[union-attr]
    assert _roles(timeline.entries) == [_ROLE.CANDIDATE]
    assert result.appended == timeline.entries
    assert record == stored_equity_intraday_research_timeline(timeline)
    entry, stored = timeline.entries[0], record.entries[0]
    decision = EvaluateEquityIntradayResearchUseCase(session.snapshots, _CONFIG).execute(
        _LISTING, _DAY, s36.snapshot_id
    )
    assert entry.decision == decision
    assert stored.decision_document == canonical_equity_candle_json(decision.document())
    digest = hashlib.sha256(stored.decision_document.encode()).hexdigest()
    assert stored.decision_fingerprint == "eir1-sha256:" + digest == decision.fingerprint
    assert stored.entry_fingerprint == entry.fingerprint and stored.previous_fingerprint is None
    assert record.head_fingerprint == timeline.head_fingerprint


def test_b_an_advance_that_appends_nothing_writes_nothing() -> None:
    session, store = _morning(), _TimelineStore()
    session.add(_MORNING, listing=_OTHER)
    other = session.issue(_ist(9, 36), listing=_OTHER)
    result = _writer(session, store).execute(
        _LISTING, _DAY, ["ecs1-sha256:" + "0" * 64, other.snapshot_id]
    )
    assert result.status is _STATUS.NOTHING_TO_COMMIT
    assert _results(result) == [
        (_DONE.REFUSED, _WHY.SNAPSHOT_UNAVAILABLE),
        (_DONE.REFUSED, _WHY.OTHER_LISTING_OR_DATE),
    ]
    assert result.timeline is None and not result.commit_confirmed and result.appended == ()
    assert store.records == {} and store.calls == []


def test_a_committed_timeline_rehydrates_identically_after_restart() -> None:
    session, store = _morning(), _TimelineStore()
    committed = _advance(session, store, session.at(31, 36, 41)).timeline
    assert committed is not None
    reloaded = rehydrate_equity_intraday_research_timeline(
        _identity(), store.record(), session.snapshots
    )
    assert reloaded == committed and reloaded.document() == committed.document()
    assert [e.fingerprint for e in reloaded.entries] == [e.fingerprint for e in committed.entries]
    assert reloaded.candidate is not None and reloaded.candidate == committed.candidate


# ----------------------------------------------------------------------
# C, D, E -- existing history, retries and equivalent evidence
# ----------------------------------------------------------------------


def test_c_existing_history_is_reloaded_and_extended_without_a_second_candidate() -> None:
    session, store = _morning(), _TimelineStore()
    s31, s36, s41 = session.at(31, 36, 41)
    first = _advance(session, store, [s31, s36])
    before = store.record()
    later = _advance(session, store, [s41])
    assert later.status is _STATUS.COMMITTED
    assert _roles(later.timeline.entries) == [  # type: ignore[union-attr]
        _ROLE.NO_SIGNAL,
        _ROLE.CANDIDATE,
        _ROLE.REPEAT_QUALIFYING,
    ]
    assert store.calls[-1][1] == before.head_fingerprint
    assert store.record().entries[:2] == before.entries
    assert later.timeline.entries[:2] == first.timeline.entries  # type: ignore[union-attr]


def test_d_an_exact_retry_is_a_duplicate_and_writes_nothing() -> None:
    session, store = _morning(), _TimelineStore()
    (s36,) = session.at(36)
    first = _advance(session, store, [s36])
    record = store.record()
    again = _advance(session, store, [s36])
    assert again.status is _STATUS.NOTHING_TO_COMMIT
    assert _results(again) == [(_DONE.DUPLICATE, None)]
    assert len(store.calls) == 1 and store.record() is record
    assert again.timeline is not None
    assert again.timeline.document() == first.timeline.document()  # type: ignore[union-attr]


def test_e_a_reissued_equivalent_manifest_keeps_the_committed_representative() -> None:
    session, store = _morning(), _TimelineStore()
    x = session.issue(_ist(9, 36))
    y = session.issue(_ist(9, 36), issued=_ist(9, 40))
    smaller, larger = sorted((x, y), key=lambda s: s.snapshot_id)
    assert smaller.manifest_fingerprint == larger.manifest_fingerprint
    _advance(session, store, [larger])
    record = store.record()
    again = _advance(session, store, [smaller])
    assert _results(again) == [(_DONE.DUPLICATE, None)]
    assert again.items[0].entry is not None
    assert again.items[0].entry.snapshot_id == larger.snapshot_id
    assert store.record() is record and record.entries[0].snapshot_id == larger.snapshot_id
    assert len(store.calls) == 1


# ----------------------------------------------------------------------
# F -- configuration (EI3-D11, EI3-D7)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes",
    [
        {"version": "synthetic-ei3c-test/2", "volume_multiplier": Decimal("3")},
        {"volume_multiplier": Decimal("1.50")},
    ],
)
def test_f_a_stored_timeline_refuses_another_configuration(changes) -> None:  # type: ignore[no-untyped-def]
    session, store = _morning(), _TimelineStore()
    s36, s41 = session.at(36, 41)
    _advance(session, store, [s36])
    record = store.record()
    changed = dataclasses.replace(_CONFIG, **changes)
    result = _advance(session, store, [s41], configuration=changed)
    assert result.status is _STATUS.CONFIGURATION_CONFLICT
    assert _results(result) == [(_DONE.REFUSED, _WHY.CONFIGURATION_MISMATCH)]
    assert result.timeline is None and not result.commit_confirmed
    assert len(store.calls) == 1 and store.record() is record


def test_f_a_competing_first_commit_under_another_configuration_wins_nothing_silently() -> None:
    session, store = _morning(), _TimelineStore()
    s36, s41 = session.at(36, 41)
    other = dataclasses.replace(_CONFIG, version="synthetic-ei3c-test/2")
    store.before_append.append(lambda: _advance(session, store, [s36], configuration=other))
    result = _advance(session, store, [s41])
    assert result.status is _STATUS.CONFIGURATION_CONFLICT
    assert store.record().configuration_document == canonical_equity_candle_json(other.document())
    assert len(store.record().entries) == 1


# ----------------------------------------------------------------------
# G, H, I -- stale heads and concurrent writers (EI3-D12)
# ----------------------------------------------------------------------


def test_g_a_stale_head_reloads_and_reevaluates_the_whole_batch() -> None:
    session, store = _morning(), _TimelineStore()
    s36, s41 = session.at(36, 41)
    store.before_append.append(_later(session, store, [s36]))
    result = _advance(session, store, [s41])
    assert result.status is _STATUS.COMMITTED and result.attempts == 2
    timeline = result.timeline
    assert timeline is not None
    assert _roles(timeline.entries) == [_ROLE.CANDIDATE, _ROLE.REPEAT_QUALIFYING]
    assert timeline.entries[0].snapshot_id == s36.snapshot_id
    assert result.items[0].entry is timeline.entries[1]
    assert timeline.entries[1].previous_fingerprint == timeline.entries[0].fingerprint
    # The stale attempt expected an empty timeline; the retry expected s36's entry.
    assert [expected for _, expected in store.calls] == [
        None,
        None,
        timeline.entries[0].fingerprint,
    ]
    assert store.record() == stored_equity_intraday_research_timeline(timeline)


def test_h_stale_heads_on_every_attempt_report_nothing_committed() -> None:
    session, store = _morning(), _TimelineStore()
    s31, s36, s41, s46 = session.at(31, 36, 41, 46)
    for competing in ([s31], [s36], [s41]):
        store.before_append.append(_later(session, store, competing))
    result = _advance(session, store, [s46])
    assert result.status is _STATUS.STALE_AFTER_RETRIES
    assert result.attempts == EQUITY_INTRADAY_RESEARCH_TIMELINE_COMMIT_ATTEMPTS == 3
    assert result.timeline is None and result.schedule is None and result.items == ()
    assert not result.commit_confirmed and result.appended == ()
    record = store.record()
    assert [e.snapshot_id for e in record.entries] == [
        s31.snapshot_id,
        s36.snapshot_id,
        s41.snapshot_id,
    ]


@pytest.mark.parametrize("case", ["later", "earlier", "conflicting"])
def test_i_concurrent_writers_never_produce_two_candidates(case: str) -> None:
    session, store = _morning(), _TimelineStore()
    s36, s41 = session.at(36, 41)
    a, b = _conflicting(session)
    winner, loser = {"later": (s36, s41), "earlier": (s41, s36), "conflicting": (a, b)}[case]
    store.before_append.append(_later(session, store, [winner]))
    result = _advance(session, store, [loser])
    record = store.record()
    assert sum(e.role is _ROLE.CANDIDATE for e in record.entries) == 1
    assert record.entries[0].snapshot_id == winner.snapshot_id
    if case == "later":
        assert result.status is _STATUS.COMMITTED
        assert _roles(record.entries) == [_ROLE.CANDIDATE, _ROLE.REPEAT_QUALIFYING]
    else:
        refusal = (
            _WHY.BEFORE_TIMELINE_HEAD
            if case == "earlier"
            else _WHY.CONFLICTING_EVIDENCE_FOR_INSTANT
        )
        assert result.status is _STATUS.NOTHING_TO_COMMIT and result.attempts == 2
        assert _results(result) == [(_DONE.REFUSED, refusal)]
        assert len(record.entries) == 1


# ----------------------------------------------------------------------
# J, K, L -- unconfirmed commits
# ----------------------------------------------------------------------


def test_j_a_lost_acknowledgment_is_confirmed_by_reload_without_a_second_write() -> None:
    session, store = _morning(), _TimelineStore()
    store.lose_acks = 1
    result = _advance(session, store, session.at(31, 36))
    assert result.status is _STATUS.COMMITTED
    assert (result.commit_confirmed, result.reconciled, result.attempts) == (True, True, 1)
    assert store.commits == 1 and len(store.calls) == 1
    assert result.timeline is not None
    assert store.record() == stored_equity_intraday_research_timeline(result.timeline)
    assert result.appended == result.timeline.entries


def test_k_a_later_writer_does_not_hide_a_confirmed_uncertain_commit() -> None:
    session, store = _morning(), _TimelineStore()
    s36, s41 = session.at(36, 41)
    store.lose_acks = 1
    store.after_commit.append(_later(session, store, [s41]))
    result = _advance(session, store, [s36])
    assert result.status is _STATUS.COMMITTED and result.reconciled
    timeline = result.timeline
    assert timeline is not None and result.schedule is not None
    assert _roles(timeline.entries) == [_ROLE.CANDIDATE, _ROLE.REPEAT_QUALIFYING]
    assert timeline.entries[0] == result.schedule.timeline.entries[0]
    assert timeline.head_fingerprint != result.schedule.timeline.head_fingerprint
    assert store.commits == 2 and [t.entries[-1].snapshot_id for t, _ in store.calls] == [
        s36.snapshot_id,
        s41.snapshot_id,
    ]


def test_k_an_absent_uncertain_commit_is_reevaluated_against_the_new_history() -> None:
    session, store = _morning(), _TimelineStore()
    s31, s36 = session.at(31, 36)
    store.before_append.append(_later(session, store, [s31]))
    store.drop_commits = 1
    result = _advance(session, store, [s36])
    assert result.status is _STATUS.COMMITTED
    assert (result.attempts, result.reconciled) == (2, False)
    assert _roles(store.record().entries) == [_ROLE.NO_SIGNAL, _ROLE.CANDIDATE]
    assert store.commits == 2


def test_l_an_unresolvable_commit_outcome_is_raised_not_guessed() -> None:
    session, store = _morning(), _TimelineStore()
    store.lose_acks = 1
    store.after_commit.append(lambda: setattr(store, "unavailable", True))
    with pytest.raises(EquityIntradayResearchTimelineCommitUnresolvedError) as raised:
        _advance(session, store, session.at(36))
    assert raised.value.expected_head is None
    store.unavailable = False
    # The entry did commit; the use case claimed neither success nor rollback.
    assert raised.value.attempted_head == store.record().head_fingerprint
    assert store.commits == 1


def test_l_history_that_no_longer_extends_the_loaded_history_is_unresolved() -> None:
    session, store = _morning(), _TimelineStore()
    s31, s36, s41 = session.at(31, 36, 41)
    _advance(session, store, [s31])
    rewritten = _TimelineStore()
    _advance(session, rewritten, [s36])
    store.lose_acks = 1
    # Test-only: the stored history is replaced after the commit.
    store.after_commit.append(lambda: store.records.update(rewritten.records))
    with pytest.raises(EquityIntradayResearchTimelineCommitUnresolvedError, match="extends"):
        _advance(session, store, [s41])


# ----------------------------------------------------------------------
# M, N, O, P -- rehydration fails closed (EI3-D13)
# ----------------------------------------------------------------------


def _replace_entry(store: _TimelineStore, index: int, **changes: object) -> None:
    """Test-only: alter one stored entry in place of the committed record."""
    record = store.record()
    entries = list(record.entries)
    entries[index] = dataclasses.replace(entries[index], **changes)  # type: ignore[arg-type]
    store.records[(_CONFIG.strategy, _LISTING, _DAY)] = dataclasses.replace(
        record, entries=tuple(entries)
    )


def _document(text: str, **values: object) -> dict:  # type: ignore[type-arg]
    document = json.loads(text)
    document["values"].update(values)
    return document  # type: ignore[no-any-return]


def _fingerprint(text: str) -> str:
    return "eir1-sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _committed() -> tuple[_Session, _TimelineStore, EquityCandleEvidenceSnapshot]:
    session, store = _morning(), _TimelineStore()
    s31, s36, s41 = session.at(31, 36, 41)
    _advance(session, store, [s31, s36])
    return session, store, s41


def _refused(
    session: _Session, store: _TimelineStore, s41: EquityCandleEvidenceSnapshot, error: type
) -> pytest.ExceptionInfo:  # type: ignore[type-arg]
    calls = len(store.calls)
    with pytest.raises(error) as raised:
        _advance(session, store, [s41])
    assert len(store.calls) == calls
    return raised


def test_m_a_tampered_decision_document_fails_closed() -> None:
    session, store, s41 = _committed()
    original = store.record().entries[1].decision_document
    altered = canonical_equity_candle_json(_document(original, ended_bars=5))
    _replace_entry(store, 1, decision_document=altered)
    raised = _refused(session, store, s41, InvalidStoredEquityIntradayResearchTimelineError)
    assert "decision fingerprint" in str(raised.value)
    _replace_entry(store, 1, decision_fingerprint=_fingerprint(altered))
    _refused(session, store, s41, EquityIntradayResearchDecisionNotReproducedError)


def test_m_a_non_canonical_decision_document_fails_closed() -> None:
    session, store, s41 = _committed()
    spaced = json.dumps(json.loads(store.record().entries[0].decision_document), indent=1)
    _replace_entry(store, 0, decision_document=spaced, decision_fingerprint=_fingerprint(spaced))
    raised = _refused(session, store, s41, InvalidStoredEquityIntradayResearchTimelineError)
    assert "canonical" in str(raised.value)


def test_n_a_missing_source_snapshot_fails_closed() -> None:
    session, store, s41 = _committed()
    del session.snapshots.documents[store.record().entries[0].snapshot_id]  # test-only
    raised = _refused(session, store, s41, InvalidStoredEquityIntradayResearchTimelineError)
    assert "unavailable" in str(raised.value)


def test_o_another_decision_format_has_no_compatible_evaluator() -> None:
    session, store, s41 = _committed()
    document = json.loads(store.record().entries[0].decision_document)
    document["format"] = "northstar.equity-intraday-research-decision/2"
    text = canonical_equity_candle_json(document)
    _replace_entry(store, 0, decision_document=text, decision_fingerprint=_fingerprint(text))
    _refused(session, store, s41, EquityIntradayResearchHistoricalEvaluationUnavailableError)


def test_o_changed_evaluation_behaviour_is_never_silently_migrated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, store, s41 = _committed()
    original = EquityOpeningRangeBreakoutEvaluator.evaluate
    drifted = EquityOpeningRangeBreakoutEvaluator(
        dataclasses.replace(_CONFIG, minimum_baseline_bars=4)
    )

    def evaluate(self, listing, day, snapshot, requested=None):  # type: ignore[no-untyped-def]
        return original(drifted, listing, day, snapshot, requested)

    # Test-only: the evaluator now behaves differently from when history was committed.
    monkeypatch.setattr(EquityOpeningRangeBreakoutEvaluator, "evaluate", evaluate)
    _refused(session, store, s41, EquityIntradayResearchDecisionNotReproducedError)


def _record(
    entry: EquityIntradayResearchTimelineEntry, role: EquityIntradayResearchTimelineRole
) -> StoredEquityIntradayResearchTimelineEntry:
    return StoredEquityIntradayResearchTimelineEntry(
        sequence=entry.sequence,
        role=role,
        decision_outcome=entry.decision.outcome,
        snapshot_id=entry.snapshot_id,
        manifest_fingerprint=entry.manifest_fingerprint,
        decision_instant=entry.decision_instant,
        watermark=entry.watermark,
        research_window_closes_at=entry.research_window_closes_at,
        decision_document=canonical_equity_candle_json(entry.decision.document()),
        decision_fingerprint=entry.decision.fingerprint,
        previous_fingerprint=entry.previous_fingerprint,
        entry_fingerprint=entry.fingerprint,
    )


@pytest.mark.parametrize(
    ("index", "changes"),
    [
        (1, {"entry_fingerprint": "eirt1-sha256:" + "0" * 64}),
        (1, {"previous_fingerprint": "eirt1-sha256:" + "1" * 64}),
        (1, {"sequence": 3}),
        (1, {"role": _ROLE.REPEAT_QUALIFYING}),
        (0, {"role": _ROLE.CANDIDATE}),
        (0, {"watermark": 1}),
        (0, {"decision_instant": PointInTime("2026-08-12T04:02:00Z")}),
    ],
)
def test_p_an_invalid_stored_entry_rejects_the_whole_history(index, changes) -> None:  # type: ignore[no-untyped-def]
    session, store, s41 = _committed()
    _replace_entry(store, index, **changes)
    _refused(session, store, s41, InvalidStoredEquityIntradayResearchTimelineError)


def test_p_two_consistently_fingerprinted_candidates_are_still_refused() -> None:
    session, store = _morning(), _TimelineStore()
    s36, s41, s46 = session.at(36, 41, 46)
    evaluator = EquityOpeningRangeBreakoutEvaluator(_CONFIG)
    first = EquityIntradayResearchTimelineEntry(
        _identity(),
        1,
        _ROLE.CANDIDATE,
        evaluator.evaluate(_LISTING, _DAY, s36),
        s36.window_closes_at,
        None,
    )
    second = EquityIntradayResearchTimelineEntry(
        _identity(),
        2,
        _ROLE.CANDIDATE,
        evaluator.evaluate(_LISTING, _DAY, s41),
        s41.window_closes_at,
        first.fingerprint,
    )
    # Test-only: a stored record whose every fingerprint is consistent, with two CANDIDATEs.
    store.records[(_CONFIG.strategy, _LISTING, _DAY)] = StoredEquityIntradayResearchTimeline(
        _CONFIG.strategy,
        _LISTING,
        _DAY,
        canonical_equity_candle_json(_CONFIG.document()),
        (_record(first, _ROLE.CANDIDATE), _record(second, _ROLE.CANDIDATE)),
    )
    raised = _refused(session, store, s46, InvalidStoredEquityIntradayResearchTimelineError)
    assert "REPEAT_QUALIFYING" in str(raised.value)


def test_p_a_record_for_another_timeline_is_refused() -> None:
    session, store, s41 = _committed()
    record = store.record()
    elsewhere = dataclasses.replace(record, trading_date=_NEXT_DAY)
    with pytest.raises(InvalidStoredEquityIntradayResearchTimelineError, match="another"):
        rehydrate_equity_intraday_research_timeline(_identity(), elsewhere, session.snapshots)
    with pytest.raises(EquityIntradayResearchTimelineConfigurationConflictError):
        rehydrate_equity_intraday_research_timeline(
            EquityIntradayResearchTimelineIdentity(
                _LISTING, _DAY, dataclasses.replace(_CONFIG, version="synthetic-ei3c-test/2")
            ),
            record,
            session.snapshots,
        )


# ----------------------------------------------------------------------
# Q -- EI-3B refusals are preserved
# ----------------------------------------------------------------------


def test_q_regression_conflicts_and_late_history_are_refused_and_never_written() -> None:
    session, store = _morning(), _TimelineStore()
    s46 = session.issue(_ist(9, 46), issued=_ist(9, 47))
    session.unrelated_entry()
    s36, s41 = session.issue(_ist(9, 36)), session.issue(_ist(9, 41), issued=_ist(9, 48))
    assert s41.watermark > s46.watermark
    a, b = _conflicting(session)
    together = _advance(session, store, [a, b])
    assert together.status is _STATUS.NOTHING_TO_COMMIT and store.records == {}
    assert _results(together) == [(_DONE.REFUSED, _WHY.CONFLICTING_EVIDENCE_FOR_INSTANT)] * 2
    _advance(session, store, [s41])
    record = store.record()
    for late, refusal in (
        (s46, _WHY.EVIDENCE_REGRESSION),
        (s36, _WHY.BEFORE_TIMELINE_HEAD),
    ):
        result = _advance(session, store, [late])
        assert result.status is _STATUS.NOTHING_TO_COMMIT
        assert _results(result) == [(_DONE.REFUSED, refusal)]
    assert store.record() is record and len(store.calls) == 1


# ----------------------------------------------------------------------
# R, S -- availability and isolation
# ----------------------------------------------------------------------


def test_r_an_unavailable_store_or_snapshot_store_propagates_and_writes_nothing() -> None:
    session, store = _morning(), _TimelineStore()
    (s36,) = session.at(36)
    store.unavailable = True
    with pytest.raises(EquityIntradayResearchTimelineUnavailableError):
        _advance(session, store, [s36])
    store.unavailable = False
    assert store.calls == [] and store.records == {}
    _advance(session, store, [s36])
    writer = AdvanceEquityIntradayResearchTimelineUseCase(store, _UnavailableSnapshots(), _CONFIG)
    with pytest.raises(EquityCandleEvidenceUnavailableError):
        writer.execute(_LISTING, _DAY, [s36.snapshot_id])
    assert len(store.calls) == 1


def test_s_listings_dates_and_strategies_have_independent_timelines() -> None:
    session, store = _morning(), _TimelineStore()
    session.add(_MORNING, listing=_OTHER)
    session.add(_MORNING, day=_NEXT_DAY)
    other_strategy = dataclasses.replace(_CONFIG, strategy=StrategyIdentity("ei3c-other"))
    runs = [
        (_LISTING, _DAY, _CONFIG, session.issue(_ist(9, 36))),
        (_OTHER, _DAY, _CONFIG, session.issue(_ist(9, 36), listing=_OTHER)),
        (_LISTING, _NEXT_DAY, _CONFIG, session.issue(_ist(9, 36, day=_NEXT_DAY), day=_NEXT_DAY)),
        (_LISTING, _DAY, other_strategy, session.issue(_ist(9, 41))),
    ]
    for listing, day, configuration, snapshot in runs:
        result = _advance(
            session, store, [snapshot], configuration=configuration, listing=listing, day=day
        )
        assert result.status is _STATUS.COMMITTED
        assert _roles(result.timeline.entries) == [_ROLE.CANDIDATE]  # type: ignore[union-attr]
    assert len(store.records) == 4
    for (strategy, listing, day), record in store.records.items():
        assert (record.strategy, record.listing, record.trading_date) == (strategy, listing, day)
        assert len(record.entries) == 1 and record.entries[0].role is _ROLE.CANDIDATE


# ----------------------------------------------------------------------
# Contracts
# ----------------------------------------------------------------------


def test_inputs_records_and_results_are_checked() -> None:
    session, store = _morning(), _TimelineStore()
    writer = _writer(session, store)
    with pytest.raises(TypeError):
        writer.execute(_LISTING, _DAY, "ecs1-sha256:abc")
    with pytest.raises(TypeError):
        writer.execute(_LISTING, _DAY, [""])
    with pytest.raises(TypeError):
        writer.execute(_LISTING, datetime(2026, 8, 12), [])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        writer.execute("M&M", _DAY, [])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AdvanceEquityIntradayResearchTimelineUseCase(object(), session.snapshots, _CONFIG)  # type: ignore[arg-type]
    assert store.calls == []
    empty = EquityIntradayResearchTimeline.start(_LISTING, _DAY, _CONFIG)
    with pytest.raises(ValueError, match="EI3-D11"):
        stored_equity_intraday_research_timeline(empty)
    with pytest.raises(InvalidStoredEquityIntradayResearchTimelineError, match="EI3-D11"):
        StoredEquityIntradayResearchTimeline(_CONFIG.strategy, _LISTING, _DAY, "{}", ())
    _advance(session, store, session.at(36))
    with pytest.raises(InvalidStoredEquityIntradayResearchTimelineError):
        dataclasses.replace(store.record().entries[0], sequence=0)
    with pytest.raises(ValueError, match="confirmed"):
        EquityIntradayResearchTimelineAdvance(
            _STATUS.COMMITTED, None, None, (), 1, False, False, "x"
        )
    with pytest.raises(ValueError, match="nothing"):
        EquityIntradayResearchTimelineAdvance(
            _STATUS.STALE_AFTER_RETRIES, empty, None, (), 3, False, False, "x"
        )


# ----------------------------------------------------------------------
# T -- architecture
# ----------------------------------------------------------------------


@pytest.mark.parametrize("module", [advance_module, port_module])
def test_t_the_modules_stay_application_only(module) -> None:  # type: ignore[no-untyped-def]
    tree = ast.parse(inspect.getsource(module))
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not called & {
        "now",
        "today",
        "utcnow",
        "monotonic",
        "time",
        "sleep",
        "random",
        "uuid4",
        "connect",
        "cursor",
        "commit",
        "rollback",
        "float",
    }
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    for name in imported:
        assert name is not None
        assert name.split(".")[0] not in {"time", "random", "uuid", "secrets", "threading"}
        for forbidden in (
            "northstar_infrastructure",
            "sqlite",
            "socket",
            "http",
            "urllib",
            "futures",
            "paper",
            "broker",
            "historical_market_data",
        ):
            assert forbidden not in name, name
    for node in tree.body:
        if isinstance(node, ast.Assign | ast.AnnAssign):
            assert not isinstance(node.value, ast.Dict | ast.List | ast.Set)
    identifiers = {
        name.lower()
        for node in ast.walk(tree)
        for name in (
            getattr(node, "id", None),
            getattr(node, "attr", None),
            getattr(node, "arg", None),
            node.name if isinstance(node, ast.FunctionDef | ast.ClassDef) else None,
        )
        if isinstance(name, str)
    }
    for identifier in identifiers:
        for forbidden in (
            "stop",
            "target",
            "exit",
            "fill",
            "order",
            "broker",
            "paper",
            "pnl",
            "profit",
            "quantity",
            "short",
            "buy",
            "sell",
            "recommend",
        ):
            assert forbidden not in identifier, identifier


def test_t_the_contracts_are_exported() -> None:
    for name in (
        "EQUITY_INTRADAY_RESEARCH_TIMELINE_COMMIT_ATTEMPTS",
        "AdvanceEquityIntradayResearchTimelineUseCase",
        "EquityIntradayResearchTimelineAdvance",
        "EquityIntradayResearchTimelineAdvanceStatus",
        "StoredEquityIntradayResearchTimeline",
        "StoredEquityIntradayResearchTimelineEntry",
        "rehydrate_equity_intraday_research_timeline",
        "stored_equity_intraday_research_timeline",
    ):
        assert name in application_services.__all__
    for name in (
        "EquityIntradayResearchTimelineStore",
        "EquityIntradayResearchTimelineStaleError",
        "EquityIntradayResearchTimelineConfigurationConflictError",
        "EquityIntradayResearchTimelineUnavailableError",
        "EquityIntradayResearchTimelineCommitUnconfirmedError",
        "EquityIntradayResearchTimelineCommitUnresolvedError",
        "InvalidStoredEquityIntradayResearchTimelineError",
        "EquityIntradayResearchDecisionNotReproducedError",
        "EquityIntradayResearchHistoricalEvaluationUnavailableError",
    ):
        assert name in ports.__all__
