"""EI-3B: deterministic synthetic equity intraday research timeline and scheduling.

Everything is invented: an M&M@NSE session on 2026-08-12 over in-test
reference doubles, synthetic candles and finality from one synthetic source,
and a synthetic calibration (opening range 3 bars, volume multiplier 1.5) that
is an example, never an approved parameter. Snapshots are issued through the
real EI-2B use case over in-memory doubles applying the shared EI-2B rules,
as in the EI-3A tests. Fixtures that alter a snapshot or a timeline after the
fact are test-only and say so; nothing here stands for a real EI-2B store.
A CANDIDATE is a research label, never a recommendation, order or fill.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import itertools
import json
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

from northstar_application import application_services
from northstar_application.application_services import (
    NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW,
    EquityCandleAppendOutcome,
    EquityCandleAppendResult,
    EquityCandleEvidenceClass,
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
    EquityIntradayNoSignalReason,
    EquityIntradayResearchDecision,
    EquityIntradayResearchOutcome,
    EquityIntradayResearchRejection,
    EquityIntradayResearchScheduleOutcome,
    EquityIntradayResearchScheduleRefusal,
    EquityIntradayResearchScheduleResult,
    EquityIntradayResearchTimeline,
    EquityIntradayResearchTimelineEntry,
    EquityIntradayResearchTimelineIdentity,
    EquityIntradayResearchTimelineRole,
    EquityOpeningRangeBreakoutConfiguration,
    EquityOpeningRangeBreakoutEvaluator,
    EquityResearchWindowEvaluation,
    EvaluateEquityIntradayResearchUseCase,
    EvaluateEquityResearchWindowUseCase,
    InvalidEquityIntradayResearchTimelineError,
    IssueEquityCandleEvidenceSnapshotUseCase,
    ScheduleEquityIntradayResearchUseCase,
    equity_candle_identity_conflicts,
    equity_candle_in_window,
    equity_candle_retry,
    equity_candle_snapshot_from_document,
    equity_candle_snapshot_json,
)
from northstar_application.application_services import (
    schedule_equity_intraday_research as schedule_module,
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
)

_NSE = ExchangeCode("NSE")
_INR = Currency("INR")
_DAY = date(2026, 8, 12)
_NEXT_DAY = date(2026, 8, 13)
_LISTING = ListingReference(Symbol("M&M"), _NSE)
_OTHER = ListingReference(Symbol("RELIANCE"), _NSE)
_IST = timezone(timedelta(hours=5, minutes=30))
_SOURCE = "synthetic-ei3b-source"
_SYNTHETIC_FINALITY = EquityCandleFinalityPolicy("synthetic-ei3b/1", frozenset({_SOURCE}), True)
_PRODUCTION_LIKE = EquityCandleFinalityPolicy(
    "test-approved-source/1", frozenset({"approved-source"}), False
)
_CONFIG = EquityOpeningRangeBreakoutConfiguration(
    strategy=StrategyIdentity("ei3b-orb-vwap-volume"),
    version="synthetic-ei3b-test/1",
    opening_range_bars=3,
    volume_multiplier=Decimal("1.5"),
    minimum_baseline_bars=3,
)
_ROLE = EquityIntradayResearchTimelineRole
_DONE = EquityIntradayResearchScheduleOutcome
_WHY = EquityIntradayResearchScheduleRefusal
_OUT = EquityIntradayResearchOutcome
_REJ = EquityIntradayResearchRejection
_NO = EquityIntradayNoSignalReason
_SYN = EquityReferenceProvenanceKind.SYNTHETIC
_PROD = EquityReferenceProvenanceKind.ACCEPTED_PRODUCTION

Candle = tuple[str, str, str, str, str]

# Bars 1-3 form the opening range (high 102, low 99). Bar 4 (ends 09:35 IST)
# breaks out; bars 5 and 6 (09:40, 09:45 IST) qualify again.
_BASE: dict[int, Candle] = {
    1: ("100", "101", "99", "100.5", "1000"),
    2: ("100.5", "102", "100", "101.5", "1000"),
    3: ("101.5", "102", "100.5", "101", "1000"),
}
_BREAKOUT: Candle = ("101", "103", "101", "102.5", "2000")
_MORNING: dict[int, Candle] = {
    **_BASE,
    4: _BREAKOUT,
    5: ("102.5", "104", "102", "103.5", "3000"),
    6: ("103.5", "105", "103", "104.5", "4500"),
}
_FLAT: Candle = ("101", "101.5", "100.5", "101", "1000")


def _ist(hour: int, minute: int, second: int = 0, day: date = _DAY) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=_IST).astimezone(UTC)


def _end(index: int, day: date = _DAY) -> datetime:
    """Return the end of five-minute bar ``index`` (bar 1 ends 09:20 IST)."""
    return _ist(9, 15, day=day) + timedelta(minutes=5 * index)


def _pit(value: datetime) -> PointInTime:
    return PointInTime(value.isoformat().replace("+00:00", "Z"))


def _whole_session() -> dict[int, Candle]:
    """Bars 1-69 (to 15:00 IST): a first breakout at 14:30, qualifying again at 14:55 and 15:00."""
    candles = {**_BASE, **{index: _FLAT for index in range(4, 70)}}
    candles[63] = ("101", "103", "101", "102.5", "2000")
    candles[68] = ("101", "104", "101", "103.5", "3000")
    candles[69] = ("103.5", "105", "103", "104.5", "4000")
    return candles


# ----------------------------------------------------------------------
# Reference, journal and store doubles (shared EI-2B rules)
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
    def __init__(self, kind: EquityReferenceProvenanceKind) -> None:
        self.kind = kind

    def classify(self, window: EquityResearchWindowEvaluation) -> EquityResearchWindowProvenance:
        assert window.venue_reference and window.applicability_reference
        return EquityResearchWindowProvenance(
            window.venue_reference, self.kind, window.applicability_reference, self.kind
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


class _Store(EquityCandleEvidenceSnapshotStore):
    def __init__(self) -> None:
        self.documents: dict[str, str] = {}

    def issue(self, snapshot: EquityCandleEvidenceSnapshot) -> int:
        self.documents.setdefault(snapshot.snapshot_id, equity_candle_snapshot_json(snapshot))
        return 1

    def get_snapshot(self, snapshot_id: str) -> EquityCandleEvidenceSnapshot | None:
        text = self.documents.get(snapshot_id)
        return None if text is None else equity_candle_snapshot_from_document(json.loads(text))


# ----------------------------------------------------------------------
# Evidence builders
# ----------------------------------------------------------------------


def _event(index: int, listing: ListingReference, day: date) -> str:
    return f"evt-{listing.symbol.value}-{day.isoformat()}-{index}"


def _observation(
    index: int,
    candle: Candle,
    *,
    listing: ListingReference = _LISTING,
    day: date = _DAY,
    revision: str = "r1",
    source: str = _SOURCE,
    synthetic: bool = True,
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
        f"obs-{event}-{revision}-{source}",
        EquityCandleObservation(
            bar, source, event, revision, _pit(end + timedelta(seconds=5)), None
        ),
        synthetic,
    )


def _finality(
    index: int,
    *,
    listing: ListingReference = _LISTING,
    day: date = _DAY,
    revision: str = "r1",
    latency: timedelta = timedelta(seconds=11),
    source: str = _SOURCE,
    synthetic: bool = True,
) -> EquityCandleFinalityRecord:
    """Return finality confirmed 10s after the bar ends and received ``latency`` after it."""
    end = _end(index, day)
    event = _event(index, listing, day)
    return EquityCandleFinalityRecord(
        f"fin-{event}-{revision}-{source}",
        EquityCandleFinalityEvidence(
            source,
            event,
            revision,
            _pit(end + timedelta(seconds=10)),
            "synthetic statement",
            synthetic,
        ),
        _pit(end + latency),
    )


class _Session:
    """A journal, a store and the EI-2B issue use case for synthetic sessions."""

    def __init__(
        self,
        provenance: EquityReferenceProvenanceKind = _SYN,
        finality: EquityCandleFinalityPolicy = _SYNTHETIC_FINALITY,
    ) -> None:
        self.journal, self.store = _Journal(), _Store()
        self.issuer = IssueEquityCandleEvidenceSnapshotUseCase(
            self.journal,
            self.store,
            _INR,
            finality,
            NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW,
            reference_provenance=_Provenance(provenance),
        )

    def add(
        self,
        candles: dict[int, Candle],
        *,
        listing: ListingReference = _LISTING,
        day: date = _DAY,
        latency: timedelta = timedelta(seconds=11),
        skip_finality: tuple[int, ...] = (),
        source: str = _SOURCE,
        synthetic: bool = True,
    ) -> None:
        record = {"listing": listing, "day": day, "source": source, "synthetic": synthetic}
        for index, candle in sorted(candles.items()):
            self.journal.append(_observation(index, candle, **record))  # type: ignore[arg-type]
            if index not in skip_finality:
                self.journal.append(_finality(index, latency=latency, **record))  # type: ignore[arg-type]

    def issue(
        self,
        as_of: datetime,
        issued: datetime | None = None,
        *,
        listing: ListingReference = _LISTING,
        day: date = _DAY,
    ) -> EquityCandleEvidenceSnapshot:
        window = EvaluateEquityResearchWindowUseCase(
            _Venue(), _Applicability(), NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW
        ).execute(listing, day, _pit(as_of))
        return self.issuer.execute(window, _pit(as_of), _pit(issued or as_of))

    def scheduler(
        self, configuration: EquityOpeningRangeBreakoutConfiguration = _CONFIG
    ) -> ScheduleEquityIntradayResearchUseCase:
        return ScheduleEquityIntradayResearchUseCase(self.store, configuration)

    def run(
        self,
        snapshots: list[EquityCandleEvidenceSnapshot],
        timeline: EquityIntradayResearchTimeline | None = None,
    ) -> EquityIntradayResearchScheduleResult:
        scheduler = self.scheduler()
        start = scheduler.start(_LISTING, _DAY) if timeline is None else timeline
        return scheduler.advance(start, [s.snapshot_id for s in snapshots])

    def research(self, snapshot: EquityCandleEvidenceSnapshot) -> EquityIntradayResearchDecision:
        """Return the stand-alone EI-3A decision for ``snapshot``."""
        return EvaluateEquityIntradayResearchUseCase(self.store, _CONFIG).execute(
            _LISTING, _DAY, snapshot.snapshot_id
        )


def _roles(timeline: EquityIntradayResearchTimeline) -> list[EquityIntradayResearchTimelineRole]:
    return [entry.role for entry in timeline.entries]


def _results(result: EquityIntradayResearchScheduleResult) -> list[tuple[object, object]]:
    return [(item.outcome, item.refusal) for item in result.items]


def _morning() -> _Session:
    session = _Session()
    session.add(_MORNING)
    return session


# ----------------------------------------------------------------------
# A, B -- candidate frequency (EI3-D7)
# ----------------------------------------------------------------------


def test_a_no_signal_then_one_candidate() -> None:
    session = _morning()
    s31, s36 = session.issue(_ist(9, 31)), session.issue(_ist(9, 36))
    result = session.run([s31, s36])
    timeline = result.timeline
    assert _results(result) == [(_DONE.APPENDED, None), (_DONE.APPENDED, None)]
    assert _roles(timeline) == [_ROLE.NO_SIGNAL, _ROLE.CANDIDATE]
    assert timeline.entries[0].decision.reasons == (_NO.NO_POST_OPENING_RANGE_INTERVAL,)
    assert timeline.candidate is timeline.entries[1]
    assert [e.decision_instant for e in timeline.entries] == [s31.as_of, s36.as_of]
    assert [e.sequence for e in timeline.entries] == [1, 2]
    assert result.appended == timeline.entries
    # The entries carry the unchanged EI-3A decisions.
    assert timeline.entries[1].decision == session.research(s36)
    assert [i.entry for i in result.items] == list(timeline.entries)


def test_b_repeated_qualifying_signals_create_one_candidate() -> None:
    session = _morning()
    snapshots = [session.issue(_ist(9, minute)) for minute in (36, 41, 46)]
    timeline = session.run(snapshots).timeline
    assert _roles(timeline) == [_ROLE.CANDIDATE, _ROLE.REPEAT_QUALIFYING, _ROLE.REPEAT_QUALIFYING]
    assert all(e.decision.outcome is _OUT.LONG_CANDIDATE for e in timeline.entries)
    assert [e.decision.values.signal_interval_end for e in timeline.entries] == [  # type: ignore[union-attr]
        _pit(_end(4)),
        _pit(_end(5)),
        _pit(_end(6)),
    ]
    for entry, snapshot in zip(timeline.entries, snapshots, strict=True):
        assert entry.decision == session.research(snapshot)
    assert timeline.candidate is timeline.entries[0]


def test_candidate_selection_uses_the_established_timeline_not_only_the_batch() -> None:
    session = _morning()
    first = session.run([session.issue(_ist(9, 36))]).timeline
    later = session.run([session.issue(_ist(9, 41)), session.issue(_ist(9, 46))], first)
    assert _roles(later.timeline) == [
        _ROLE.CANDIDATE,
        _ROLE.REPEAT_QUALIFYING,
        _ROLE.REPEAT_QUALIFYING,
    ]
    assert [e.role for e in later.appended] == [_ROLE.REPEAT_QUALIFYING] * 2


# ----------------------------------------------------------------------
# C, D -- duplicates
# ----------------------------------------------------------------------


def test_c_an_exact_retry_is_a_duplicate_and_changes_nothing() -> None:
    session = _morning()
    s36 = session.issue(_ist(9, 36))
    first = session.run([s36])
    again = session.run([s36], first.timeline)
    assert _results(first) == [(_DONE.APPENDED, None)]
    assert _results(again) == [(_DONE.DUPLICATE, None)]
    assert again.items[0].entry is first.timeline.entries[0]
    assert again.timeline == first.timeline
    assert again.timeline.document() == first.timeline.document()
    assert again.timeline.head_fingerprint == first.timeline.head_fingerprint
    assert again.appended == ()
    together = session.run([s36, s36])
    assert _results(together) == [(_DONE.APPENDED, None), (_DONE.DUPLICATE, None)]
    assert together.timeline.document() == first.timeline.document()


def test_d_a_reissued_identical_manifest_is_one_entry() -> None:
    session = _morning()
    a = session.issue(_ist(9, 36))
    b = session.issue(_ist(9, 36), issued=_ist(9, 40))
    assert a.manifest_fingerprint == b.manifest_fingerprint and a.snapshot_id != b.snapshot_id
    # EI-3A decision fingerprints include the snapshot ID, so they differ.
    assert session.research(a).fingerprint != session.research(b).fingerprint
    result = session.run([b, a])
    timeline = result.timeline
    assert len(timeline.entries) == 1 and _roles(timeline) == [_ROLE.CANDIDATE]
    representative = min(a.snapshot_id, b.snapshot_id)
    assert timeline.entries[0].snapshot_id == representative
    by_id = {item.snapshot_id: item for item in result.items}
    assert by_id[representative].outcome is _DONE.APPENDED
    other = b.snapshot_id if representative == a.snapshot_id else a.snapshot_id
    assert by_id[other].outcome is _DONE.DUPLICATE
    assert by_id[other].entry is timeline.entries[0]
    retried = session.scheduler().advance(timeline, [other])
    assert _results(retried) == [(_DONE.DUPLICATE, None)]
    assert retried.timeline.document() == timeline.document()


def test_d_batch_partitions_keep_the_research_but_may_change_equivalent_identity() -> None:
    """The guarantee is batch-order independence, not batch-partition independence.

    Research is identical under every partition. The snapshot representing an
    equivalent instant, and so every fingerprint from that entry on, depends on
    which equivalent snapshot was established first: history is never rewritten.
    """
    session = _morning()
    x = session.issue(_ist(9, 36))
    y = session.issue(_ist(9, 36), issued=_ist(9, 40))
    a, b = sorted((x, y), key=lambda s: s.snapshot_id)
    s41 = session.issue(_ist(9, 41))
    assert a.manifest_fingerprint == b.manifest_fingerprint and a.snapshot_id < b.snapshot_id

    def run(*batches: list[EquityCandleEvidenceSnapshot]) -> tuple[list[str], object]:
        scheduler = session.scheduler()
        timeline = scheduler.start(_LISTING, _DAY)
        outcomes: list[str] = []
        for batch in batches:
            result = scheduler.advance(timeline, [s.snapshot_id for s in batch])
            outcomes += [f"{i.snapshot_id}:{i.outcome.value}" for i in result.items]
            timeline = result.timeline
        return outcomes, timeline

    cases = {
        "together": run([a, b, s41]),
        "together, reversed": run([b, a, s41]),
        "A first": run([a], [b, s41]),
        "A first, reversed": run([a], [s41, b]),
        "B first": run([b], [a, s41]),
        "B first, reversed": run([b], [s41, a]),
    }
    timelines: dict[str, EquityIntradayResearchTimeline] = {k: v[1] for k, v in cases.items()}  # type: ignore[misc]
    reference = timelines["together"]
    for name, timeline in timelines.items():
        assert _roles(timeline) == [_ROLE.CANDIDATE, _ROLE.REPEAT_QUALIFYING], name
        assert sum(e.role is _ROLE.CANDIDATE for e in timeline.entries) == 1, name
        for entry, expected in zip(timeline.entries, reference.entries, strict=True):
            assert entry.manifest_fingerprint == expected.manifest_fingerprint, name
            assert entry.decision.outcome is expected.decision.outcome, name
            assert entry.decision.values == expected.decision.values, name
        outcomes = sorted(cases[name][0])
        appended = [o for o in outcomes if o.endswith(":APPENDED")]
        duplicates = [o for o in outcomes if o.endswith(":DUPLICATE")]
        assert len(appended) == 2 and len(duplicates) == 1, name
    # A single batch, or A established first, represents 09:36 by A.
    for name in ("together", "together, reversed", "A first", "A first, reversed"):
        assert timelines[name].document() == reference.document(), name
        assert timelines[name].entries[0].snapshot_id == a.snapshot_id
    # B established first stays the entry; A later is its DUPLICATE.
    for name in ("B first", "B first, reversed"):
        timeline = timelines[name]
        assert timeline.entries[0].snapshot_id == b.snapshot_id
        assert f"{a.snapshot_id}:DUPLICATE" in cases[name][0]
        assert timeline.entries[0].decision.fingerprint != reference.entries[0].decision.fingerprint
        assert timeline.entries[0].fingerprint != reference.entries[0].fingerprint
        assert timeline.entries[1].decision == reference.entries[1].decision
        assert timeline.entries[1].fingerprint != reference.entries[1].fingerprint
        assert timeline.head_fingerprint != reference.head_fingerprint
        assert timeline.document() == timelines["B first"].document()


def test_matching_research_on_a_different_manifest_is_not_a_duplicate() -> None:
    session = _morning()
    s36 = session.issue(_ist(9, 36))
    s37 = session.issue(_ist(9, 37))
    assert session.research(s36).values == session.research(s37).values
    result = session.run([s36, s37])
    assert _results(result) == [(_DONE.APPENDED, None), (_DONE.APPENDED, None)]
    assert _roles(result.timeline) == [_ROLE.CANDIDATE, _ROLE.REPEAT_QUALIFYING]


# ----------------------------------------------------------------------
# E -- conflicting manifests at one instant (EI3-D9)
# ----------------------------------------------------------------------


def _conflicting() -> tuple[_Session, EquityCandleEvidenceSnapshot, EquityCandleEvidenceSnapshot]:
    """Two manifests for 09:36: the journal gains an entry outside M&M's window in between."""
    session = _morning()
    a = session.issue(_ist(9, 36))
    session.journal.append(_observation(1, _BASE[1], listing=_OTHER))
    b = session.issue(_ist(9, 36), issued=_ist(9, 37))
    assert a.as_of == b.as_of and a.manifest_fingerprint != b.manifest_fingerprint
    assert a.watermark < b.watermark
    return session, a, b


def test_e_conflicting_manifests_arriving_together_are_all_refused() -> None:
    session, a, b = _conflicting()
    s41 = session.issue(_ist(9, 41))
    result = session.run([b, s41, a])
    assert _results(result) == [
        (_DONE.REFUSED, _WHY.CONFLICTING_EVIDENCE_FOR_INSTANT),
        (_DONE.APPENDED, None),
        (_DONE.REFUSED, _WHY.CONFLICTING_EVIDENCE_FOR_INSTANT),
    ]
    assert [e.snapshot_id for e in result.timeline.entries] == [s41.snapshot_id]
    assert a.manifest_fingerprint in result.items[0].detail
    assert b.manifest_fingerprint in result.items[0].detail
    alone = session.run([a, b])
    assert alone.timeline.entries == () and alone.timeline.candidate is None


def test_e_an_established_entry_is_kept_against_a_conflicting_manifest() -> None:
    session, a, b = _conflicting()
    established = session.run([a]).timeline
    refused = session.run([b], established)
    assert _results(refused) == [(_DONE.REFUSED, _WHY.CONFLICTING_EVIDENCE_FOR_INSTANT)]
    assert refused.timeline == established
    mixed = session.run([b, a], established)
    assert _results(mixed) == [
        (_DONE.REFUSED, _WHY.CONFLICTING_EVIDENCE_FOR_INSTANT),
        (_DONE.DUPLICATE, None),
    ]
    assert mixed.timeline.document() == established.document()


# ----------------------------------------------------------------------
# F -- input order
# ----------------------------------------------------------------------


def test_f_the_canonical_timeline_does_not_depend_on_input_order() -> None:
    session = _morning()
    s31, s36, s41, s46 = (session.issue(_ist(9, m)) for m in (31, 36, 41, 46))
    s36_reissued = session.issue(_ist(9, 36), issued=_ist(9, 39))
    batch = [s31, s36, s36_reissued, s41, s46, s41]
    reference = session.run(batch)
    assert _roles(reference.timeline) == [
        _ROLE.NO_SIGNAL,
        _ROLE.CANDIDATE,
        _ROLE.REPEAT_QUALIFYING,
        _ROLE.REPEAT_QUALIFYING,
    ]
    expected = sorted((i.snapshot_id, i.outcome) for i in reference.items)
    for order in itertools.islice(itertools.permutations(batch), 0, None, 61):
        result = session.run(list(order))
        assert result.timeline.document() == reference.timeline.document()
        assert [i.snapshot_id for i in result.items] == [s.snapshot_id for s in order]
        assert [i.position for i in result.items] == list(range(len(order)))
        assert sorted((i.snapshot_id, i.outcome) for i in result.items) == expected
    # The equivalent 09:36 pair is represented by its smaller snapshot ID.
    representative = min((s36, s36_reissued), key=lambda s: s.snapshot_id)
    assert reference.timeline.entries[1].snapshot_id == representative.snapshot_id
    stepwise = session.run([s46, s41], session.run([representative, s31]).timeline)
    assert stepwise.timeline.document() == reference.timeline.document()


# ----------------------------------------------------------------------
# G, H -- regression and late history (EI3-D9)
# ----------------------------------------------------------------------


def test_g_a_later_instant_with_a_lower_watermark_is_refused() -> None:
    session = _morning()
    s46 = session.issue(_ist(9, 46), issued=_ist(9, 47))
    session.journal.append(_observation(1, _BASE[1], listing=_OTHER))
    s41 = session.issue(_ist(9, 41), issued=_ist(9, 48))
    assert s41.watermark > s46.watermark
    together = session.run([s46, s41])
    assert _results(together) == [
        (_DONE.REFUSED, _WHY.EVIDENCE_REGRESSION),
        (_DONE.APPENDED, None),
    ]
    established = session.run([s41]).timeline
    refused = session.run([s46], established)
    assert _results(refused) == [(_DONE.REFUSED, _WHY.EVIDENCE_REGRESSION)]
    assert refused.timeline == established


def test_h_unseen_history_before_the_head_never_becomes_a_retroactive_candidate() -> None:
    session = _Session()
    session.add({**_BASE, 4: _BREAKOUT, 5: _MORNING[5]}, skip_finality=(5,))
    s31, s41 = session.issue(_ist(9, 31)), session.issue(_ist(9, 41))
    timeline = session.run([s31, s41]).timeline
    assert _roles(timeline) == [_ROLE.NO_SIGNAL, _ROLE.REJECTED]
    s36 = session.issue(_ist(9, 36), issued=_ist(9, 50))
    assert session.research(s36).outcome is _OUT.LONG_CANDIDATE
    result = session.run([s36], timeline)
    assert _results(result) == [(_DONE.REFUSED, _WHY.BEFORE_TIMELINE_HEAD)]
    assert result.timeline == timeline and result.timeline.candidate is None


# ----------------------------------------------------------------------
# I, J -- corrections and late finality
# ----------------------------------------------------------------------


def test_i_a_later_correction_never_withdraws_an_earlier_candidate() -> None:
    session = _Session()
    session.add({**_BASE, 4: _BREAKOUT})
    s36 = session.issue(_ist(9, 36))
    established = session.run([s36]).timeline
    candidate = established.entries[0]
    fingerprints = (candidate.fingerprint, candidate.decision.fingerprint)
    session.journal.append(_observation(2, _BASE[2], revision="r2"))
    session.journal.append(_finality(2, revision="r2"))
    session.add({5: _MORNING[5]})
    s41 = session.issue(_ist(9, 41))
    result = session.run([s36, s41], established)
    assert _results(result) == [(_DONE.DUPLICATE, None), (_DONE.APPENDED, None)]
    timeline = result.timeline
    assert _roles(timeline) == [_ROLE.CANDIDATE, _ROLE.REJECTED]
    assert timeline.entries[0] is candidate and timeline.candidate is candidate
    assert (candidate.fingerprint, candidate.decision.fingerprint) == fingerprints
    assert candidate.snapshot_id == s36.snapshot_id
    rejected = timeline.entries[1]
    assert rejected.decision.reasons == (_REJ.UNRESOLVED_SOURCE_REVISION,)
    assert rejected.snapshot_id == s41.snapshot_id
    assert rejected.previous_fingerprint == candidate.fingerprint
    timeline.verify()
    assert session.research(s36) == candidate.decision


def test_j_late_finality_rejects_first_and_qualifies_later() -> None:
    session = _Session()
    session.add({**_BASE, 4: _BREAKOUT}, skip_finality=(4,))
    s36 = session.issue(_ist(9, 36))
    session.journal.append(_finality(4, latency=timedelta(minutes=2, seconds=30)))
    s38 = session.issue(_ist(9, 38))
    timeline = session.run([s38, s36]).timeline
    assert _roles(timeline) == [_ROLE.REJECTED, _ROLE.CANDIDATE]
    assert timeline.entries[0].decision.reasons == (_REJ.UNFINALIZED_CANDLE,)
    assert timeline.candidate is timeline.entries[1]
    assert timeline.entries[1].decision_instant == s38.as_of


# ----------------------------------------------------------------------
# K, L -- scope and configuration
# ----------------------------------------------------------------------


def test_k_another_listing_or_date_is_refused_without_contamination() -> None:
    session = _morning()
    session.add(_MORNING, listing=_OTHER)
    session.add(_MORNING, day=_NEXT_DAY)
    own = session.issue(_ist(9, 36))
    other = session.issue(_ist(9, 36), listing=_OTHER)
    next_day = session.issue(_ist(9, 36, day=_NEXT_DAY), day=_NEXT_DAY)
    result = session.run([other, own, next_day])
    assert _results(result) == [
        (_DONE.REFUSED, _WHY.OTHER_LISTING_OR_DATE),
        (_DONE.APPENDED, None),
        (_DONE.REFUSED, _WHY.OTHER_LISTING_OR_DATE),
    ]
    assert [e.snapshot_id for e in result.timeline.entries] == [own.snapshot_id]
    # Another listing at the same instant is not a same-instant conflict.
    assert other.as_of == own.as_of


def test_l_a_changed_configuration_is_refused_and_never_a_second_candidate() -> None:
    session = _morning()
    established = session.run([session.issue(_ist(9, 36))]).timeline
    s41 = session.issue(_ist(9, 41))
    for changed in (
        dataclasses.replace(
            _CONFIG, version="synthetic-ei3b-test/2", volume_multiplier=Decimal("3")
        ),
        dataclasses.replace(_CONFIG, volume_multiplier=Decimal("1.50")),
        dataclasses.replace(_CONFIG, strategy=StrategyIdentity("ei3b-other-strategy")),
    ):
        result = session.scheduler(changed).advance(established, [s41.snapshot_id])
        assert _results(result) == [(_DONE.REFUSED, _WHY.CONFIGURATION_MISMATCH)]
        assert result.timeline is established
    other_decision = EquityOpeningRangeBreakoutEvaluator(
        dataclasses.replace(_CONFIG, version="synthetic-ei3b-test/2")
    ).evaluate(_LISTING, _DAY, s41)
    with pytest.raises(InvalidEquityIntradayResearchTimelineError, match="configuration"):
        EquityIntradayResearchTimelineEntry(
            established.identity,
            2,
            _ROLE.REPEAT_QUALIFYING,
            other_decision,
            _pit(_ist(15, 0)),
            established.head_fingerprint,
        )


# ----------------------------------------------------------------------
# M, N -- window boundary and finality latency (EI3-D6; EI3-D10 pending)
# ----------------------------------------------------------------------


def test_m_window_boundary_has_no_1425_cutoff_and_admits_post_window_research() -> None:
    session = _Session()
    session.add(_whole_session())
    session.add({70: ("104.5", "110", "104", "109", "90000")})  # ends 15:05 IST, outside
    instants = [_ist(14, 26), _ist(14, 31), _ist(14, 56), _ist(15, 1), _ist(15, 20)]
    timeline = session.run([session.issue(i) for i in instants]).timeline
    assert _roles(timeline) == [
        _ROLE.NO_SIGNAL,
        _ROLE.CANDIDATE,
        _ROLE.REPEAT_QUALIFYING,
        _ROLE.REPEAT_QUALIFYING,
        _ROLE.REPEAT_QUALIFYING,
    ]
    signals = [e.decision.values.signal_interval_end for e in timeline.entries]  # type: ignore[union-attr]
    close = _pit(_ist(15, 0))
    assert signals == [
        _pit(_ist(14, 25)),
        _pit(_ist(14, 30)),
        _pit(_ist(14, 55)),
        close,
        close,
    ]
    assert all(e.research_window_closes_at == close for e in timeline.entries)
    assert [e.decided_after_research_window for e in timeline.entries] == [
        False,
        False,
        False,
        True,
        True,
    ]
    text = json.dumps(timeline.document())
    for forbidden in ("SHORT", "BUY", "SELL", "stop", "target", "fill", "exit", "quantity"):
        assert forbidden not in text


def test_n_finality_under_five_minutes_allows_evaluation_and_strictness_is_kept() -> None:
    session = _Session()
    session.add({**_BASE, 4: _BREAKOUT}, latency=timedelta(minutes=2))
    early, valid = session.issue(_ist(9, 36)), session.issue(_ist(9, 38))
    timeline = session.run([early, valid]).timeline
    assert _roles(timeline) == [_ROLE.REJECTED, _ROLE.CANDIDATE]
    assert timeline.entries[0].decision.reasons == (_REJ.UNFINALIZED_CANDLE,)


def test_n_five_minute_latency_prevents_in_session_evaluation_but_not_post_window() -> None:
    session = _Session()
    session.add(_whole_session(), latency=timedelta(minutes=5))
    instants = [_ist(9, 20, 30) + timedelta(minutes=7 * k) for k in range(49)]
    instants += [_ist(14, 59, 30), _ist(15, 4, 59)]
    assert all(i < _ist(15, 5) for i in instants)
    post_window = _ist(15, 5)
    snapshots = [session.issue(i) for i in [*instants, post_window]]
    timeline = session.run(snapshots).timeline
    in_session, last = timeline.entries[:-1], timeline.entries[-1]
    assert len(in_session) == len(instants)
    assert all(e.role is _ROLE.REJECTED for e in in_session)
    assert all(e.decision.reasons == (_REJ.UNFINALIZED_CANDLE,) for e in in_session)
    assert last.role is _ROLE.CANDIDATE and last.decision_instant == _pit(post_window)
    assert last.decided_after_research_window
    assert last.decision.values.signal_interval_end == _pit(_ist(15, 0))  # type: ignore[union-attr]


# ----------------------------------------------------------------------
# Refused versus rejected
# ----------------------------------------------------------------------


def test_a_rejected_entry_belongs_to_the_timeline_but_a_refused_snapshot_does_not() -> None:
    session = _Session()
    session.add({**_BASE, 4: _BREAKOUT}, skip_finality=(4,))
    session.add({**_BASE, 4: _BREAKOUT}, listing=_OTHER)
    unfinalized = session.issue(_ist(9, 36))
    other = session.issue(_ist(9, 37), listing=_OTHER)
    result = session.run([unfinalized, other])
    assert _results(result) == [
        (_DONE.APPENDED, None),
        (_DONE.REFUSED, _WHY.OTHER_LISTING_OR_DATE),
    ]
    assert _roles(result.timeline) == [_ROLE.REJECTED]
    assert result.items[1].entry is None


class _AlteringStore(_Store):
    """Test-only: returns snapshots altered after issue, as a corrupted store might."""

    def get_snapshot(self, snapshot_id: str) -> EquityCandleEvidenceSnapshot | None:
        snapshot = super().get_snapshot(snapshot_id)
        if snapshot is not None:
            object.__setattr__(snapshot, "watermark", snapshot.watermark + 1)
        return snapshot


def test_missing_or_untrustworthy_snapshots_are_refused_not_rejected() -> None:
    session = _morning()
    real = session.issue(_ist(9, 36))
    forged = "ecs1-sha256:" + "f" * 64
    session.store.documents[forged] = session.store.documents[real.snapshot_id]  # test-only
    missing = "ecs1-sha256:" + "0" * 64
    result = session.scheduler().advance(
        session.scheduler().start(_LISTING, _DAY), [missing, forged]
    )
    assert _results(result) == [
        (_DONE.REFUSED, _WHY.SNAPSHOT_UNAVAILABLE),
        (_DONE.REFUSED, _WHY.UNTRUSTED_SNAPSHOT),
    ]
    assert result.timeline.entries == ()
    altering = _AlteringStore()
    altering.documents = session.store.documents
    scheduler = ScheduleEquityIntradayResearchUseCase(altering, _CONFIG)
    altered = scheduler.advance(scheduler.start(_LISTING, _DAY), [real.snapshot_id])
    assert _results(altered) == [(_DONE.REFUSED, _WHY.UNTRUSTED_SNAPSHOT)]


def test_production_evidence_and_another_research_context_are_refused() -> None:
    production = _Session(_PROD, _PRODUCTION_LIKE)
    production.add(_MORNING, source="approved-source", synthetic=False)
    snapshot = production.issue(_ist(9, 36))
    assert snapshot.evidence_class is EquityCandleEvidenceClass.PRODUCTION
    result = production.run([snapshot])
    assert _results(result) == [(_DONE.REFUSED, _WHY.EVIDENCE_NOT_SYNTHETIC)]
    session = _morning()
    # Test-only: a snapshot under another window policy, placed straight in the store.
    other_policy = dataclasses.replace(
        session.issue(_ist(9, 36)), window_policy_version="other-research-window/1"
    )
    session.store.issue(other_policy)
    result = session.run([other_policy])
    assert _results(result) == [(_DONE.REFUSED, _WHY.RESEARCH_CONTEXT_MISMATCH)]
    assert result.timeline.entries == ()


# ----------------------------------------------------------------------
# O -- timeline integrity
# ----------------------------------------------------------------------


def _three_entries() -> tuple[_Session, EquityIntradayResearchTimeline]:
    session = _morning()
    snapshots = [session.issue(_ist(9, m)) for m in (31, 36, 41)]
    return session, session.run(snapshots).timeline


def test_o_entries_and_timelines_are_immutable_and_reproducible() -> None:
    _, timeline = _three_entries()
    _, again = _three_entries()
    assert timeline.document() == again.document()
    assert [e.fingerprint for e in timeline.entries] == [e.fingerprint for e in again.entries]
    assert all(e.fingerprint.startswith("eirt1-sha256:") for e in timeline.entries)
    assert isinstance(timeline.entries, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        timeline.entries[0].role = _ROLE.CANDIDATE  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        timeline.entries = ()  # type: ignore[misc]
    chain = [None, *[e.fingerprint for e in timeline.entries[:-1]]]
    assert [e.previous_fingerprint for e in timeline.entries] == chain


def test_o_the_same_decision_at_another_position_has_another_fingerprint() -> None:
    session, timeline = _three_entries()
    alone = session.run([session.issue(_ist(9, 36))]).timeline
    assert alone.entries[0].decision == timeline.entries[1].decision
    assert alone.entries[0].fingerprint != timeline.entries[1].fingerprint


def test_o_a_rebuilt_earlier_entry_breaks_the_chain() -> None:
    _, timeline = _three_entries()
    first = timeline.entries[0]
    rebuilt = EquityIntradayResearchTimelineEntry(
        first.identity,
        1,
        first.role,
        first.decision,
        _pit(_ist(15, 5)),
        None,
    )
    assert rebuilt.fingerprint != first.fingerprint
    with pytest.raises(InvalidEquityIntradayResearchTimelineError, match="chain"):
        EquityIntradayResearchTimeline(timeline.identity, (rebuilt, *timeline.entries[1:]))


def test_o_an_altered_timeline_is_refused_whole() -> None:
    session, timeline = _three_entries()
    # Test-only: alter a chained decision's evidence in place.
    evidence = timeline.entries[0].decision.evidence
    object.__setattr__(evidence, "watermark", 999)
    with pytest.raises(InvalidEquityIntradayResearchTimelineError):
        timeline.verify()
    s46 = session.issue(_ist(9, 46))
    result = session.scheduler().advance(timeline, [s46.snapshot_id])
    assert _results(result) == [(_DONE.REFUSED, _WHY.INVALID_TIMELINE)]
    assert result.timeline is timeline


def test_o_an_altered_entry_fingerprint_is_detected() -> None:
    _, timeline = _three_entries()
    object.__setattr__(timeline.entries[2], "fingerprint", "eirt1-sha256:" + "0" * 64)  # test-only
    with pytest.raises(InvalidEquityIntradayResearchTimelineError, match="fingerprint"):
        timeline.verify()


def _chain(
    identity: EquityIntradayResearchTimelineIdentity,
    parts: list[tuple[EquityIntradayResearchTimelineRole, EquityIntradayResearchDecision]],
) -> tuple[EquityIntradayResearchTimelineEntry, ...]:
    entries: list[EquityIntradayResearchTimelineEntry] = []
    for role, decision in parts:
        entries.append(
            EquityIntradayResearchTimelineEntry(
                identity,
                len(entries) + 1,
                role,
                decision,
                _pit(_ist(15, 0)),
                entries[-1].fingerprint if entries else None,
            )
        )
    return tuple(entries)


def test_o_timeline_invariants_are_enforced() -> None:
    session = _morning()
    s36, s41 = session.issue(_ist(9, 36)), session.issue(_ist(9, 41))
    d36, d41 = session.research(s36), session.research(s41)
    identity = EquityIntradayResearchTimelineIdentity(_LISTING, _DAY, _CONFIG)
    timeline = EquityIntradayResearchTimeline
    with pytest.raises(InvalidEquityIntradayResearchTimelineError, match="REPEAT_QUALIFYING"):
        timeline(identity, _chain(identity, [(_ROLE.CANDIDATE, d36), (_ROLE.CANDIDATE, d41)]))
    with pytest.raises(InvalidEquityIntradayResearchTimelineError, match="CANDIDATE"):
        timeline(identity, _chain(identity, [(_ROLE.REPEAT_QUALIFYING, d36)]))
    with pytest.raises(InvalidEquityIntradayResearchTimelineError, match="instant"):
        timeline(
            identity,
            _chain(identity, [(_ROLE.CANDIDATE, d41), (_ROLE.REPEAT_QUALIFYING, d36)]),
        )
    with pytest.raises(InvalidEquityIntradayResearchTimelineError, match="tuple"):
        timeline(identity, list(_chain(identity, [(_ROLE.CANDIDATE, d36)])))  # type: ignore[arg-type]
    session.journal.append(_observation(1, _BASE[1], listing=_OTHER))
    higher = session.research(session.issue(_ist(9, 31)))
    with pytest.raises(InvalidEquityIntradayResearchTimelineError, match="watermark"):
        timeline(identity, _chain(identity, [(_ROLE.NO_SIGNAL, higher), (_ROLE.CANDIDATE, d36)]))
    other = EquityIntradayResearchTimelineIdentity(_LISTING, _NEXT_DAY, _CONFIG)
    with pytest.raises(InvalidEquityIntradayResearchTimelineError, match="another"):
        _chain(other, [(_ROLE.CANDIDATE, d36)])
    with pytest.raises(InvalidEquityIntradayResearchTimelineError, match="another timeline"):
        timeline(other, _chain(identity, [(_ROLE.CANDIDATE, d36)]))


@pytest.mark.parametrize(
    ("role", "previous", "match"),
    [
        (_ROLE.NO_SIGNAL, None, "role"),
        (_ROLE.CANDIDATE, "eirt1-sha256:" + "a" * 64, "first entry"),
    ],
)
def test_o_entries_check_role_and_chain_start(role, previous, match) -> None:  # type: ignore[no-untyped-def]
    session = _morning()
    decision = session.research(session.issue(_ist(9, 36)))
    identity = EquityIntradayResearchTimelineIdentity(_LISTING, _DAY, _CONFIG)
    with pytest.raises(InvalidEquityIntradayResearchTimelineError, match=match):
        EquityIntradayResearchTimelineEntry(
            identity, 1, role, decision, _pit(_ist(15, 0)), previous
        )


def test_o_refused_or_untrusted_decisions_can_never_be_entries() -> None:
    session = _morning()
    snapshot = session.issue(_ist(9, 36))
    identity = EquityIntradayResearchTimelineIdentity(_LISTING, _DAY, _CONFIG)
    evaluator = EquityOpeningRangeBreakoutEvaluator(_CONFIG)
    untrusted = evaluator.evaluate(_LISTING, _DAY, None, "ecs1-sha256:" + "0" * 64)
    with pytest.raises(InvalidEquityIntradayResearchTimelineError, match="untrusted"):
        _chain(identity, [(_ROLE.REJECTED, untrusted)])
    elsewhere = evaluator.evaluate(_LISTING, _NEXT_DAY, snapshot)
    assert elsewhere.reasons == (_REJ.TRADING_DATE_MISMATCH,)
    other = EquityIntradayResearchTimelineIdentity(_LISTING, _NEXT_DAY, _CONFIG)
    with pytest.raises(InvalidEquityIntradayResearchTimelineError, match="scope"):
        _chain(other, [(_ROLE.REJECTED, elsewhere)])


# ----------------------------------------------------------------------
# Inputs and operational failure
# ----------------------------------------------------------------------


def test_start_creates_an_empty_timeline_and_inputs_are_checked() -> None:
    session = _morning()
    scheduler = session.scheduler()
    timeline = scheduler.start(_LISTING, _DAY)
    assert timeline.entries == () and timeline.head is None and timeline.head_fingerprint is None
    assert timeline == EquityIntradayResearchTimeline.start(_LISTING, _DAY, _CONFIG)
    assert (timeline.listing, timeline.trading_date, timeline.configuration) == (
        _LISTING,
        _DAY,
        _CONFIG,
    )
    assert timeline.identity.strategy == _CONFIG.strategy
    empty = scheduler.advance(timeline, [])
    assert empty.items == () and empty.timeline == timeline
    with pytest.raises(TypeError):
        scheduler.advance(timeline, "ecs1-sha256:abc")
    with pytest.raises(TypeError):
        scheduler.advance(timeline, [""])
    with pytest.raises(TypeError):
        scheduler.advance(None, [])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ScheduleEquityIntradayResearchUseCase(object(), _CONFIG)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ScheduleEquityIntradayResearchUseCase(session.store, object())  # type: ignore[arg-type]
    with pytest.raises(InvalidEquityIntradayResearchTimelineError):
        EquityIntradayResearchTimeline.start(_LISTING, datetime(2026, 8, 12), _CONFIG)  # type: ignore[arg-type]


class _FailingStore(_Store):
    def get_snapshot(self, snapshot_id: str) -> EquityCandleEvidenceSnapshot | None:
        raise EquityCandleEvidenceUnavailableError("storage is unavailable")


def test_a_storage_failure_propagates_and_appends_nothing() -> None:
    scheduler = ScheduleEquityIntradayResearchUseCase(_FailingStore(), _CONFIG)
    timeline = scheduler.start(_LISTING, _DAY)
    with pytest.raises(EquityCandleEvidenceUnavailableError):
        scheduler.advance(timeline, ["ecs1-sha256:" + "0" * 64])
    assert timeline.entries == ()


# ----------------------------------------------------------------------
# P -- scope of the module
# ----------------------------------------------------------------------


def test_p_the_module_reads_no_clock_and_imports_nothing_out_of_scope() -> None:
    tree = ast.parse(inspect.getsource(schedule_module))
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
        "uuid1",
        "uuid4",
        "token_hex",
        "urandom",
        "connect",
        "execute",
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


def test_p_the_module_has_no_mutable_globals_and_no_execution_vocabulary() -> None:
    tree = ast.parse(inspect.getsource(schedule_module))
    assert not any(isinstance(node, ast.Global | ast.Nonlocal) for node in ast.walk(tree))
    for node in tree.body:
        if isinstance(node, ast.Assign | ast.AnnAssign):
            value = node.value
            assert not isinstance(
                value, ast.Dict | ast.List | ast.Set | ast.DictComp | ast.ListComp | ast.SetComp
            )
        if isinstance(node, ast.ClassDef):
            for statement in node.body:
                if isinstance(statement, ast.Assign):
                    assert not isinstance(statement.value, ast.Dict | ast.List | ast.Set)
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


def test_p_the_scheduling_contracts_are_exported() -> None:
    for name in (
        "EQUITY_INTRADAY_RESEARCH_TIMELINE_ENTRY_PREFIX",
        "EquityIntradayResearchScheduleItem",
        "EquityIntradayResearchScheduleOutcome",
        "EquityIntradayResearchScheduleRefusal",
        "EquityIntradayResearchScheduleResult",
        "EquityIntradayResearchTimeline",
        "EquityIntradayResearchTimelineEntry",
        "EquityIntradayResearchTimelineIdentity",
        "EquityIntradayResearchTimelineRole",
        "InvalidEquityIntradayResearchTimelineError",
        "ScheduleEquityIntradayResearchUseCase",
    ):
        assert name in application_services.__all__
