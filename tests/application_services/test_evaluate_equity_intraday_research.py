"""EI-3A: synthetic equity intraday research evaluation from immutable evidence snapshots.

Everything is invented: a 2031 window over in-test reference doubles, synthetic
candles and finality from one synthetic source, and a synthetic calibration
(opening range 3 bars, volume multiplier 1.5) that is an example, never an
approved parameter. Snapshots are issued through the real EI-2B use case over
in-memory doubles applying the shared EI-2B rules. A LONG_CANDIDATE is a
research result, not a recommendation, order or fill.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction

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
    EquityOpeningRangeBreakoutConfiguration,
    EquityOpeningRangeBreakoutEvaluator,
    EquityResearchWindowEvaluation,
    EvaluateEquityIntradayResearchUseCase,
    EvaluateEquityResearchWindowUseCase,
    InvalidEquityIntradayResearchConfigurationError,
    IssueEquityCandleEvidenceSnapshotUseCase,
    equity_candle_identity_conflicts,
    equity_candle_in_window,
    equity_candle_retry,
    equity_candle_snapshot_from_document,
    equity_candle_snapshot_json,
)
from northstar_application.application_services import (
    evaluate_equity_intraday_research as research_module,
)
from northstar_application.application_services.evaluate_equity_intraday_research import (
    exact_text,
)
from northstar_application.ports import (
    EquityCandleEvidenceJournal,
    EquityCandleEvidenceSnapshotStore,
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
_DAY = date(2031, 3, 4)
_LISTING = ListingReference(Symbol("M&M"), _NSE)
_FIRST_END = datetime(2031, 3, 4, 3, 50, tzinfo=UTC)
_SOURCE = "synthetic-ei3a-source"
_SYNTHETIC_FINALITY = EquityCandleFinalityPolicy("synthetic-ei3a/1", frozenset({_SOURCE}), True)
_PRODUCTION_LIKE = EquityCandleFinalityPolicy(
    "test-approved-source/1", frozenset({"approved-source"}), False
)
_CONFIG = EquityOpeningRangeBreakoutConfiguration(
    strategy=StrategyIdentity("ei3a-orb-vwap-volume"),
    version="synthetic-ei3a-test/1",
    opening_range_bars=3,
    volume_multiplier=Decimal("1.5"),
    minimum_baseline_bars=3,
)
_OUT = EquityIntradayResearchOutcome
_REJ = EquityIntradayResearchRejection
_NO = EquityIntradayNoSignalReason
_SYN = EquityReferenceProvenanceKind.SYNTHETIC
_PROD = EquityReferenceProvenanceKind.ACCEPTED_PRODUCTION

# Bars 1-3 form the opening range (high 102, low 99); bar 4 breaks out.
_BASE = {
    1: ("100", "101", "99", "100.5", "1000"),
    2: ("100.5", "102", "100", "101.5", "1000"),
    3: ("101.5", "102", "100.5", "101", "1000"),
}
_BREAKOUT = ("101", "103", "101", "102.5", "2000")


def _pit(value: datetime) -> PointInTime:
    return PointInTime(value.isoformat().replace("+00:00", "Z"))


def _end(index: int) -> datetime:
    return _FIRST_END + timedelta(minutes=5 * (index - 1))


def _after(index: int, minutes: int = 1) -> datetime:
    return _end(index) + timedelta(minutes=minutes)


# ----------------------------------------------------------------------
# Reference, journal and store doubles (shared EI-2B rules)
# ----------------------------------------------------------------------


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
        return EquityListingApplicability(
            listing,
            trading_date,
            EquityListingApplicabilityClassification.ELIGIBLE_FOR_NORMAL_RESEARCH_WINDOW,
            "test-applicability/1",
            "test",
        )


class _Provenance(EquityResearchReferenceProvenanceSource):
    def __init__(self, kind: EquityReferenceProvenanceKind = _SYN) -> None:
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
        entry = EquityCandleJournalEntry(len(self.entries) + 1, _pit(_end(80)), record)
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


def _observation(
    index: int,
    candle: tuple[str, str, str, str, str],
    *,
    revision: str = "r1",
    ingestion: str | None = None,
    source: str = _SOURCE,
    synthetic: bool = True,
    currency: Currency = _INR,
    day_offset: int = 0,
) -> EquityCandleObservationRecord:
    o, h, low, c, v = candle
    end = _end(index) + timedelta(days=day_offset)
    bar = HistoricalOHLCVBar(
        symbol=_LISTING.symbol,
        exchange_code=_NSE,
        point_in_time=_pit(end),
        timeframe=Timeframe("5m"),
        open=Price(Decimal(o), currency),
        high=Price(Decimal(h), currency),
        low=Price(Decimal(low), currency),
        close=Price(Decimal(c), currency),
        volume=Quantity(Decimal(v)),
    )
    return EquityCandleObservationRecord(
        ingestion or f"obs-{index}-{revision}-{day_offset}-{source}",
        EquityCandleObservation(
            bar,
            source,
            f"evt-{index}-{day_offset}",
            revision,
            _pit(end + timedelta(seconds=5)),
            None,
        ),
        synthetic,
    )


def _finality(
    index: int,
    *,
    revision: str = "r1",
    received: datetime | None = None,
    source: str = _SOURCE,
    synthetic: bool = True,
    day_offset: int = 0,
) -> EquityCandleFinalityRecord:
    confirmed = _end(index) + timedelta(days=day_offset, seconds=10)
    return EquityCandleFinalityRecord(
        f"fin-{index}-{revision}-{day_offset}-{source}",
        EquityCandleFinalityEvidence(
            source,
            f"evt-{index}-{day_offset}",
            revision,
            _pit(confirmed),
            "synthetic statement",
            synthetic,
        ),
        _pit(received or confirmed + timedelta(seconds=1)),
    )


def _candles(**overrides: tuple[str, str, str, str, str] | None) -> dict[int, tuple]:  # type: ignore[type-arg]
    candles: dict[int, tuple] = {**_BASE, 4: _BREAKOUT}  # type: ignore[type-arg]
    for key, value in overrides.items():
        index = int(key.removeprefix("bar"))
        if value is None:
            candles.pop(index, None)
        else:
            candles[index] = value
    return candles


class _Session:
    """A journal, a store and the EI-2B issue use case for one synthetic session."""

    def __init__(
        self,
        provenance: EquityReferenceProvenanceKind = _SYN,
        finality: EquityCandleFinalityPolicy = _SYNTHETIC_FINALITY,
        window_policy=NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW,  # type: ignore[no-untyped-def]
    ) -> None:
        self.journal, self.store = _Journal(), _Store()
        self.window_policy = window_policy
        self.issuer = IssueEquityCandleEvidenceSnapshotUseCase(
            self.journal,
            self.store,
            _INR,
            finality,
            window_policy,
            reference_provenance=_Provenance(provenance),
        )

    def add(self, candles: dict[int, tuple], *, skip_finality: tuple[int, ...] = ()) -> None:  # type: ignore[type-arg]
        for index, candle in sorted(candles.items()):
            self.journal.append(_observation(index, candle))
            if index not in skip_finality:
                self.journal.append(_finality(index))

    def issue(
        self, as_of: datetime, issued: datetime | None = None
    ) -> EquityCandleEvidenceSnapshot:
        window = EvaluateEquityResearchWindowUseCase(
            _Venue(), _Applicability(), self.window_policy
        ).execute(_LISTING, _DAY, _pit(as_of))
        return self.issuer.execute(window, _pit(as_of), _pit(issued or as_of))


def _evaluate(
    snapshot: EquityCandleEvidenceSnapshot,
    configuration: EquityOpeningRangeBreakoutConfiguration = _CONFIG,
    listing: ListingReference = _LISTING,
    day: date = _DAY,
) -> EquityIntradayResearchDecision:
    return EquityOpeningRangeBreakoutEvaluator(configuration).evaluate(listing, day, snapshot)


def _decide(candles: dict[int, tuple], as_of: datetime, **kwargs) -> EquityIntradayResearchDecision:  # type: ignore[no-untyped-def,type-arg]
    session = _Session()
    session.add(candles, **kwargs)
    return _evaluate(session.issue(as_of))


def _vwap(candles: list[tuple[str, str, str, str, str]]) -> Fraction:
    weighted = sum(
        (Fraction(Decimal(h)) + Fraction(Decimal(low)) + Fraction(Decimal(c)))
        / 3
        * Fraction(Decimal(v))
        for _, h, low, c, v in candles
    )
    return weighted / sum(Fraction(Decimal(c[4])) for c in candles)


# ----------------------------------------------------------------------
# LONG_CANDIDATE
# ----------------------------------------------------------------------


def test_a_valid_synthetic_breakout_is_a_long_candidate_with_its_evidence() -> None:
    session = _Session()
    session.add(_candles())
    as_of = _after(4, minutes=2)
    snapshot = session.issue(as_of)
    decision = _evaluate(snapshot)
    assert decision.outcome is _OUT.LONG_CANDIDATE and decision.reasons == ()
    values = decision.values
    assert values is not None
    assert (values.opening_range_high, values.opening_range_low) == (Decimal(102), Decimal(99))
    assert values.signal_interval_end == PointInTime("2031-03-04T04:05:00Z")  # 09:35 IST
    assert (values.signal_close, values.signal_volume) == (Decimal("102.5"), Decimal(2000))
    assert values.vwap_approximation == _vwap([*_BASE.values(), _BREAKOUT])
    assert values.cumulative_volume == Decimal(5000)
    assert (values.baseline_bars, values.baseline_mean_volume) == (3, Fraction(1000))
    assert values.volume_threshold == Fraction(1500)
    evidence = decision.evidence
    assert evidence is not None
    assert decision.decision_instant == snapshot.as_of == _pit(as_of)
    assert decision.decision_instant != values.signal_interval_end
    assert (evidence.snapshot_id, evidence.manifest_fingerprint, evidence.watermark) == (
        snapshot.snapshot_id,
        snapshot.manifest_fingerprint,
        snapshot.watermark,
    )
    assert evidence.evidence_class is EquityCandleEvidenceClass.SYNTHETIC
    assert evidence.finality_policy_version == "synthetic-ei3a/1"
    assert evidence.window_policy_version == NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW.version
    assert (evidence.venue_provenance, evidence.applicability_provenance) == (_SYN, _SYN)
    assert [e.value for e in evidence.required_interval_ends] == [
        "2031-03-04T03:50:00Z",
        "2031-03-04T03:55:00Z",
        "2031-03-04T04:00:00Z",
        "2031-03-04T04:05:00Z",
    ]
    assert evidence.cited_sequences == (1, 3, 5, 7)
    assert decision.configuration == _CONFIG and decision.is_synthetic
    assert decision.fingerprint.startswith("eir1-sha256:")
    document = json.dumps(decision.document())
    for forbidden in ("SHORT", "entry", "stop", "target", "quantity", "BUY", "SELL"):
        assert forbidden not in document


def test_the_opening_range_spans_exactly_the_configured_bars() -> None:
    wide = _candles(bar3=("101.5", "104", "98", "101", "1000"))
    decision = _decide(wide, _after(4))
    assert decision.values is not None
    assert (decision.values.opening_range_high, decision.values.opening_range_low) == (
        Decimal(104),
        Decimal(98),
    )
    assert decision.reasons == (_NO.CLOSE_NOT_ABOVE_OPENING_RANGE_HIGH,)
    two_bar = dataclasses.replace(_CONFIG, opening_range_bars=2, version="synthetic-ei3a-2bar/1")
    session = _Session()
    session.add(wide)
    shorter = _evaluate(session.issue(_after(4)), two_bar)
    assert shorter.values is not None and shorter.values.opening_range_high == Decimal(102)


# ----------------------------------------------------------------------
# NO_SIGNAL
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("as_of", "reason", "ended"),
    [
        (datetime(2031, 3, 4, 3, 49, tzinfo=UTC), _NO.OPENING_RANGE_INCOMPLETE, 0),
        (_after(2), _NO.OPENING_RANGE_INCOMPLETE, 2),
        (_after(3), _NO.NO_POST_OPENING_RANGE_INTERVAL, 3),
    ],
)
def test_no_signal_before_the_first_possible_breakout(as_of, reason, ended) -> None:  # type: ignore[no-untyped-def]
    decision = _decide(_candles(), as_of)
    assert decision.outcome is _OUT.NO_SIGNAL and decision.reasons == (reason,)
    assert decision.values is not None and decision.values.ended_bars == ended
    assert decision.values.signal_interval_end is None


def test_the_first_possible_signal_is_the_interval_ending_09_35_ist() -> None:
    decision = _decide(_candles(), _after(4))
    assert decision.values is not None
    assert decision.values.signal_interval_end == PointInTime("2031-03-04T04:05:00Z")


@pytest.mark.parametrize(
    ("close", "reasons"),
    [
        ("102", (_NO.CLOSE_NOT_ABOVE_OPENING_RANGE_HIGH,)),
        ("101.75", (_NO.CLOSE_NOT_ABOVE_OPENING_RANGE_HIGH,)),
        (
            "98.5",
            (_NO.CLOSE_NOT_ABOVE_OPENING_RANGE_HIGH, _NO.BEARISH_RESEARCH_NOT_SUPPORTED),
        ),
    ],
)
def test_close_must_be_strictly_above_the_opening_range_high(close, reasons) -> None:  # type: ignore[no-untyped-def]
    low = min(Decimal(close), Decimal("98"))
    decision = _decide(_candles(bar4=("101", "102", str(low), close, "2000")), _after(4))
    assert decision.outcome is _OUT.NO_SIGNAL
    assert decision.reasons[: len(reasons)] == reasons
    assert "SHORT" not in json.dumps(decision.document())


def test_a_breakout_below_the_vwap_approximation_is_no_signal() -> None:
    candles = _candles(
        bar4=("104", "106", "103", "105", "5000"), bar5=("102", "103", "101.5", "102.5", "9000")
    )
    decision = _decide(candles, _after(5))
    assert decision.reasons == (_NO.CLOSE_NOT_ABOVE_VWAP_APPROXIMATION,)
    assert decision.values is not None
    assert decision.values.vwap_approximation == _vwap([candles[i] for i in range(1, 6)])
    assert decision.values.vwap_approximation > Fraction(Decimal("102.5"))


def test_the_vwap_includes_the_signal_bar_and_never_a_later_one() -> None:
    session = _Session()
    session.add(_candles(bar5=("500", "900", "400", "800", "1000000")))
    decision = _evaluate(session.issue(_after(4, minutes=4)))
    assert decision.outcome is _OUT.LONG_CANDIDATE
    assert decision.values is not None
    assert decision.values.vwap_approximation == _vwap([*_BASE.values(), _BREAKOUT])
    assert decision.values.vwap_approximation != _vwap(list(_BASE.values()))


@pytest.mark.parametrize(
    ("volume", "outcome"),
    [("1500", _OUT.LONG_CANDIDATE), ("1499.99", _OUT.NO_SIGNAL), ("1000", _OUT.NO_SIGNAL)],
)
def test_volume_confirmation_boundary_and_baseline_exclusion(volume, outcome) -> None:  # type: ignore[no-untyped-def]
    decision = _decide(_candles(bar4=("101", "103", "101", "102.5", volume)), _after(4))
    assert decision.outcome is outcome
    assert decision.values is not None
    # The baseline excludes the signal bar: including it would raise the mean above 1000.
    assert decision.values.baseline_mean_volume == Fraction(1000)
    if outcome is _OUT.NO_SIGNAL:
        assert decision.reasons == (_NO.VOLUME_NOT_CONFIRMED,)


def test_zero_volume_never_divides_or_confirms() -> None:
    silent = {i: (*c[:4], "0") for i, c in _candles().items()}
    decision = _decide(silent, _after(4))
    assert decision.reasons == (
        _NO.VWAP_APPROXIMATION_UNAVAILABLE,
        _NO.ZERO_VOLUME_BASELINE,
    )
    assert decision.values is not None and decision.values.vwap_approximation is None
    quiet_baseline = {**{i: (*c[:4], "0") for i, c in _BASE.items()}, 4: _BREAKOUT}
    baseline_only = _decide(quiet_baseline, _after(4))
    assert baseline_only.reasons == (_NO.ZERO_VOLUME_BASELINE,)
    assert baseline_only.values is not None
    assert baseline_only.values.vwap_approximation == _vwap([_BREAKOUT])


def test_an_insufficient_baseline_is_no_signal() -> None:
    longer = dataclasses.replace(_CONFIG, minimum_baseline_bars=5, version="synthetic-ei3a-b5/1")
    session = _Session()
    session.add(_candles())
    decision = _evaluate(session.issue(_after(4)), longer)
    assert decision.reasons == (_NO.INSUFFICIENT_VOLUME_BASELINE,)
    assert decision.values is not None and decision.values.baseline_mean_volume is None


def test_the_latest_interval_is_the_signal_and_older_breakouts_are_not_revived() -> None:
    quiet = ("102", "102.25", "101.5", "101.75", "3000")
    decision = _decide(_candles(bar5=quiet), _after(5))
    assert decision.values is not None
    assert decision.values.signal_interval_end == PointInTime("2031-03-04T04:10:00Z")
    assert decision.reasons == (_NO.CLOSE_NOT_ABOVE_OPENING_RANGE_HIGH,)


# ----------------------------------------------------------------------
# Exactness and determinism
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (Fraction(1, 4), "0.25"),
        (Fraction(1, 3), "1/3"),
        (Fraction(-5, 2), "-2.5"),
        (Fraction(10), "10"),
        (Fraction(1, 1000), "0.001"),
        (Fraction(304, 3), "304/3"),
    ],
)
def test_exact_text_never_rounds(value, text) -> None:  # type: ignore[no-untyped-def]
    assert exact_text(value) == text


def test_decisions_are_exact_and_repeatable() -> None:
    session = _Session()
    session.add(_candles())
    snapshot = session.issue(_after(4))
    first, second = _evaluate(snapshot), _evaluate(snapshot)
    assert first == second and first.fingerprint == second.fingerprint
    assert "/" in first.document()["values"]["vwap_approximation"]
    other = dataclasses.replace(_CONFIG, volume_multiplier=Decimal("1.25"), version="synthetic-x/2")
    assert _evaluate(snapshot, other).fingerprint != first.fingerprint


# ----------------------------------------------------------------------
# Rejections
# ----------------------------------------------------------------------


def test_a_missing_required_interval_is_rejected() -> None:
    decision = _decide(_candles(bar2=None), _after(4))
    assert decision.outcome is _OUT.REJECTED and decision.reasons == (_REJ.MISSING_CANDLE,)
    assert decision.values is None and decision.evidence is not None


def test_an_unfinalized_required_interval_is_rejected() -> None:
    decision = _decide(_candles(), _after(4), skip_finality=(2,))
    assert decision.reasons == (_REJ.UNFINALIZED_CANDLE,)


def test_finality_received_after_the_decision_instant_is_not_usable() -> None:
    session = _Session()
    for index, candle in sorted(_candles().items()):
        session.journal.append(_observation(index, candle))
        late = _after(4, minutes=30) if index == 3 else None
        session.journal.append(_finality(index, received=late))
    decision = _evaluate(session.issue(_after(4)))
    assert decision.reasons == (_REJ.UNFINALIZED_CANDLE,)


def test_finality_from_an_unapproved_source_is_rejected() -> None:
    session = _Session()
    session.add(_candles())
    session.journal.append(_observation(2, _BASE[2], source="other-source"))
    session.journal.append(_finality(2, source="other-source"))
    assert _evaluate(session.issue(_after(4))).reasons == (_REJ.UNRESOLVED_SOURCE_REVISION,)
    unapproved = _Session(
        finality=EquityCandleFinalityPolicy("synthetic-other/1", frozenset(), True)
    )
    unapproved.add(_candles())
    assert _evaluate(unapproved.issue(_after(4))).reasons == (_REJ.UNFINALIZED_CANDLE,)


def test_conflicting_values_under_one_identity_are_rejected() -> None:
    session = _Session()
    session.add(_candles())
    session.journal.append(
        _observation(2, ("100.5", "102", "100", "101.75", "1000"), ingestion="second-receipt")
    )
    assert _evaluate(session.issue(_after(4))).reasons == (_REJ.CONFLICTING_CANDLE_VALUES,)


def test_distinct_revisions_are_rejected_and_no_winner_is_chosen() -> None:
    session = _Session()
    session.add(_candles())
    session.journal.append(_observation(2, _BASE[2], revision="r2"))
    session.journal.append(_finality(2, revision="r2"))
    decision = _evaluate(session.issue(_after(4)))
    assert decision.reasons == (_REJ.UNRESOLVED_SOURCE_REVISION,)


def test_several_evidence_failures_are_all_reported_in_precedence_order() -> None:
    session = _Session()
    session.add(_candles(bar1=None))
    session.journal.append(_observation(2, _BASE[2], revision="r2"))
    session.journal.append(
        _observation(3, ("101.5", "102", "100.5", "101.25", "1000"), ingestion="x")
    )
    decision = _evaluate(session.issue(_after(4)))
    assert decision.reasons == (
        _REJ.UNRESOLVED_SOURCE_REVISION,
        _REJ.CONFLICTING_CANDLE_VALUES,
        _REJ.MISSING_CANDLE,
    )


def test_unplaced_session_evidence_is_rejected() -> None:
    session = _Session()
    session.add(_candles())
    session.journal.append(_observation(2, _BASE[2], ingestion="usd", currency=Currency("USD")))
    decision = _evaluate(session.issue(_after(4)))
    assert decision.reasons == (_REJ.UNMATCHED_SESSION_EVIDENCE,)


def test_a_snapshot_for_another_listing_or_date_is_rejected() -> None:
    session = _Session()
    session.add(_candles())
    snapshot = session.issue(_after(4))
    other = ListingReference(Symbol("M&M"), ExchangeCode("BSE"))
    assert _evaluate(snapshot, listing=other).reasons == (_REJ.LISTING_MISMATCH,)
    assert _evaluate(snapshot, day=date(2031, 3, 5)).reasons == (_REJ.TRADING_DATE_MISMATCH,)
    assert _evaluate(snapshot, listing=other).evidence is not None


def test_an_unsupported_timeframe_or_window_policy_is_rejected() -> None:
    # EI-1C-3 refuses any non-5m window policy, so a 15m snapshot can only be a
    # genuine but empty EI-2B snapshot; replace() re-runs every snapshot check.
    empty = _Session().issue(_after(4))
    fifteen = dataclasses.replace(empty, timeframe=Timeframe("15m"))
    assert fifteen.timeframe == Timeframe("15m")
    assert _evaluate(fifteen).reasons == (_REJ.UNSUPPORTED_TIMEFRAME,)
    other = dataclasses.replace(NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW, version="test-other/1")
    renamed = _Session(window_policy=other)
    renamed.add(_candles())
    assert _evaluate(renamed.issue(_after(4))).reasons == (_REJ.RESEARCH_WINDOW_POLICY_MISMATCH,)
    expects_other = dataclasses.replace(_CONFIG, window_policy_version="test-other/1")
    plain = _Session()
    plain.add(_candles())
    assert _evaluate(plain.issue(_after(4)), expects_other).reasons == (
        _REJ.RESEARCH_WINDOW_POLICY_MISMATCH,
    )


def test_a_grid_that_does_not_cover_the_window_is_rejected() -> None:
    empty = _Session().issue(_after(4))
    short = dataclasses.replace(empty, intervals=empty.intervals[:-1])
    assert _evaluate(short).reasons == (_REJ.INVALID_INTERVAL_GRID,)
    gapped = dataclasses.replace(empty, intervals=empty.intervals[:5] + empty.intervals[6:])
    assert _evaluate(gapped).reasons == (_REJ.INVALID_INTERVAL_GRID,)


def test_production_evidence_is_refused_by_the_synthetic_evaluator() -> None:
    session = _Session(provenance=_PROD, finality=_PRODUCTION_LIKE)
    for index, candle in sorted(_candles().items()):
        session.journal.append(
            _observation(index, candle, source="approved-source", synthetic=False)
        )
        session.journal.append(_finality(index, source="approved-source", synthetic=False))
    snapshot = session.issue(_after(4))
    assert snapshot.evidence_class is EquityCandleEvidenceClass.PRODUCTION
    decision = _evaluate(snapshot)
    assert decision.reasons == (_REJ.EVIDENCE_NOT_SYNTHETIC,)
    assert decision.values is None


def test_an_unavailable_or_mismatched_snapshot_reference_is_rejected() -> None:
    session = _Session()
    session.add(_candles())
    snapshot = session.issue(_after(4))
    use_case = EvaluateEquityIntradayResearchUseCase(session.store, _CONFIG)
    missing = use_case.execute(_LISTING, _DAY, "ecs1-sha256:" + "0" * 64)
    assert missing.reasons == (_REJ.SNAPSHOT_UNAVAILABLE,)
    assert missing.evidence is None and missing.decision_instant is None

    class _Wrong(_Store):
        def get_snapshot(self, snapshot_id: str) -> EquityCandleEvidenceSnapshot | None:
            return snapshot

    wrong = EvaluateEquityIntradayResearchUseCase(_Wrong(), _CONFIG)
    assert wrong.execute(_LISTING, _DAY, "ecs1-sha256:" + "1" * 64).reasons == (
        _REJ.SNAPSHOT_REFERENCE_MISMATCH,
    )
    found = use_case.execute(_LISTING, _DAY, snapshot.snapshot_id)
    assert found.outcome is _OUT.LONG_CANDIDATE
    assert found.requested_snapshot_id == snapshot.snapshot_id


def test_unknown_provenance_and_tampered_snapshots_are_rejected() -> None:
    session = _Session()
    session.add(_candles())
    unknown = session.issue(_after(4))
    object.__setattr__(unknown, "venue_provenance", "UNKNOWN")
    assert _evaluate(unknown).reasons == (_REJ.PROVENANCE_UNKNOWN,)
    assert _evaluate(unknown).evidence is None
    tampered = session.issue(_after(4))
    object.__setattr__(tampered, "watermark", tampered.watermark + 1)
    assert _evaluate(tampered).reasons == (_REJ.SNAPSHOT_INTEGRITY_FAILURE,)


@pytest.mark.parametrize(
    "changes",
    [
        {"version": "ei3a-production/1"},
        {"version": "synthetic-"},
        {"opening_range_bars": 0},
        {"opening_range_bars": True},
        {"minimum_baseline_bars": 0},
        {"volume_multiplier": Decimal(0)},
        {"volume_multiplier": Decimal("-1.5")},
        {"volume_multiplier": Decimal("NaN")},
        {"volume_multiplier": 1.5},
        {"timeframe": Timeframe("15m")},
        {"strategy": "ei3a-orb"},
    ],
)
def test_invalid_strategy_parameters_are_refused(changes) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(InvalidEquityIntradayResearchConfigurationError):
        dataclasses.replace(_CONFIG, **changes)


def test_the_evaluator_and_use_case_check_their_inputs() -> None:
    with pytest.raises(TypeError):
        EquityOpeningRangeBreakoutEvaluator(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        EvaluateEquityIntradayResearchUseCase(object(), _CONFIG)  # type: ignore[arg-type]
    evaluator = EquityOpeningRangeBreakoutEvaluator(_CONFIG)
    with pytest.raises(ValueError, match="requested snapshot"):
        evaluator.evaluate(_LISTING, _DAY, None)
    with pytest.raises(TypeError):
        evaluator.evaluate(_LISTING, datetime(2031, 3, 4), None, "x")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Point-in-time safeguards
# ----------------------------------------------------------------------


def test_later_candles_and_corrections_never_change_an_earlier_decision() -> None:
    session = _Session()
    session.add(_candles())
    earlier = session.issue(_after(4))
    use_case = EvaluateEquityIntradayResearchUseCase(session.store, _CONFIG)
    before = use_case.execute(_LISTING, _DAY, earlier.snapshot_id)
    session.add({5: ("103", "110", "102", "109", "50000")})
    session.journal.append(_observation(2, _BASE[2], revision="r2"))
    session.journal.append(_finality(2, revision="r2"))
    again = use_case.execute(_LISTING, _DAY, earlier.snapshot_id)
    assert again == before and again.fingerprint == before.fingerprint
    later = _evaluate(session.issue(_after(5)))
    assert later.reasons == (_REJ.UNRESOLVED_SOURCE_REVISION,)
    assert later.evidence is not None and before.evidence is not None
    assert later.evidence.snapshot_id != before.evidence.snapshot_id


def test_a_decision_made_long_after_the_signal_bar_uses_only_ended_intervals() -> None:
    session = _Session()
    session.add(_candles())
    session.journal.append(_observation(6, ("90", "91", "80", "81", "90000")))
    session.journal.append(_finality(6))
    decision = _evaluate(session.issue(_after(4, minutes=4)))
    assert decision.outcome is _OUT.LONG_CANDIDATE
    assert decision.evidence is not None and len(decision.evidence.required_interval_ends) == 4


def test_candles_from_another_session_never_enter_a_decision() -> None:
    session = _Session()
    session.add(_candles())
    for index in range(1, 5):
        session.journal.append(
            _observation(index, ("1", "900", "1", "800", "99999"), day_offset=-1)
        )
    decision = _evaluate(session.issue(_after(4)))
    assert decision.outcome is _OUT.LONG_CANDIDATE
    assert decision.values is not None and decision.values.cumulative_volume == Decimal(5000)


def test_journal_order_changes_evidence_references_but_not_the_research() -> None:
    forward, backward = _Session(), _Session()
    forward.add(_candles())
    for index, candle in sorted(_candles().items(), reverse=True):
        backward.journal.append(_finality(index))
        backward.journal.append(_observation(index, candle))
    first, second = _evaluate(forward.issue(_after(4))), _evaluate(backward.issue(_after(4)))
    assert (first.outcome, first.reasons, first.values) == (
        second.outcome,
        second.reasons,
        second.values,
    )
    assert first.evidence is not None and second.evidence is not None
    assert first.evidence.cited_sequences != second.evidence.cited_sequences


# ----------------------------------------------------------------------
# Scope
# ----------------------------------------------------------------------


def test_the_module_reads_no_clock_uses_no_float_and_imports_no_infrastructure() -> None:
    source = inspect.getsource(research_module)
    tree = ast.parse(source)
    called = {
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert not called & {"now", "today", "utcnow", "monotonic", "time", "sleep", "float", "random"}
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    for name in imported:
        assert name is not None
        for forbidden in (
            "northstar_infrastructure",
            "sqlite",
            "socket",
            "futures",
            "paper",
            "broker",
            "historical_market_data",
        ):
            assert forbidden not in name, name


def test_the_research_contracts_are_exported() -> None:
    for name in (
        "EquityOpeningRangeBreakoutConfiguration",
        "EquityOpeningRangeBreakoutEvaluator",
        "EvaluateEquityIntradayResearchUseCase",
        "EquityIntradayResearchDecision",
        "EquityIntradayResearchOutcome",
        "EquityIntradayResearchRejection",
        "EquityIntradayNoSignalReason",
    ):
        assert name in application_services.__all__
