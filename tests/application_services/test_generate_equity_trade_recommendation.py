"""EI-4C: synthetic equity intraday recommendation eligibility and generation.

Everything is invented: an M&M@NSE session on 2026-08-12 over in-test
reference doubles, synthetic candles and finality, a synthetic research
calibration, and synthetic timing, generation, risk, cost, increment,
eligibility and same-session state references. None of it is an approved
value, exchange fact or broker tariff. Evidence is issued through the real
EI-2B use case and every timeline is committed by the real EI-3C advancement
over an in-memory store double, so the use case reads only authoritative,
verified history. Records altered after the fact are test-only fixtures and
say so. No clock, network, provider, broker or database is used.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import json
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Any

import pytest
from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import (
    Currency,
    ExchangeCode,
    Money,
    PointInTime,
    Price,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.market_data import HistoricalOHLCVBar
from northstar_core.recommendations.equity import (
    EquityCostSide,
    EquityIssuanceKind,
    EquityPriceIncrement,
    EquityRecommendationDefinition,
    EquityRecommendationRejection,
    EquityRecommendationRejectionReason,
    EquityRecommendationTiming,
    EquitySyntheticCostComponent,
    EquitySyntheticCostSchedule,
    EquitySyntheticGenerationPolicy,
    EquitySyntheticRiskPolicy,
    EquitySyntheticVersion,
    form_equity_recommendation,
)
from northstar_core.recommendations.equity import sizing as core_sizing
from northstar_core.strategy import StrategyIdentity

from northstar_application import application_services, ports
from northstar_application.application_services import (
    EQUITY_RECOMMENDATION_GENERATION_KEY_NAMESPACE,
    EQUITY_RECOMMENDATION_GENERATION_KEY_PREFIX,
    EQUITY_TRADE_RECOMMENDATION_PROPOSAL_NAMESPACE,
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
    EquityIntradayResearchTimelineAdvanceStatus,
    EquityIntradayResearchTimelineRole,
    EquityIssuedHistoryEvidence,
    EquityModeledEntryWindow,
    EquityOpeningRangeBreakoutConfiguration,
    EquityRecommendationCheck,
    EquityRecommendationCheckStatus,
    EquityRecommendationDispositionKind,
    EquityRecommendationEligibilityAudit,
    EquityRecommendationEligibilityReason,
    EquityRecommendationGenerationAttempt,
    EquityRecommendationGenerationDisposition,
    EquityRecommendationGenerationStatus,
    EquityResearchWindowEvaluation,
    EquityResolvedSyntheticPolicySet,
    EquitySyntheticPolicyBindingIdentity,
    EquitySyntheticPolicyBundle,
    EquitySyntheticPolicyDefinitionBinding,
    EquitySyntheticPolicyPublicationResult,
    EquitySyntheticPolicySetBinding,
    EquitySyntheticPolicySetReference,
    EquityTradeRecommendationRequest,
    EquityVenueDayEvidence,
    EvaluateEquityResearchWindowUseCase,
    GenerateEquityTradeRecommendationUseCase,
    IncompatibleEquitySyntheticPolicyBindingError,
    InvalidEquityRecommendationDocumentError,
    InvalidEquityRecommendationEligibilityError,
    InvalidEquityRecommendationEvidenceError,
    InvalidEquityRecommendationRecordError,
    InvalidEquitySyntheticPolicyBindingError,
    InvalidEquityTradeRecommendationRequestError,
    IssueEquityCandleEvidenceSnapshotUseCase,
    PublishEquitySyntheticPoliciesUseCase,
    StoredEquityIntradayResearchTimeline,
    assess_equity_recommendation_inputs,
    canonical_equity_candle_json,
    canonical_equity_recommendation_json,
    equity_candle_identity_conflicts,
    equity_candle_in_window,
    equity_candle_retry,
    equity_candle_snapshot_from_document,
    equity_candle_snapshot_json,
    equity_modeled_issuance,
    equity_recommendation_attempt,
    equity_recommendation_disposition,
    equity_recommendation_fingerprint,
    equity_recommendation_generation_key,
    equity_recommendation_status,
    equity_trade_recommendation_proposal_key,
    verify_equity_recommendation_document,
)
from northstar_application.application_services import (
    equity_recommendation_generation_evidence as evidence_module,
)
from northstar_application.application_services import (
    equity_recommendation_generation_records as records_module,
)
from northstar_application.application_services import (
    equity_trade_recommendation_eligibility as eligibility_module,
)
from northstar_application.application_services import (
    generate_equity_trade_recommendation as generate_module,
)
from northstar_application.ports import (
    EquityCandleEvidenceJournal,
    EquityCandleEvidenceSnapshotStore,
    EquityCandleEvidenceUnavailableError,
    EquityIntradayResearchTimelineStaleError,
    EquityIntradayResearchTimelineStore,
    EquityIntradayResearchTimelineUnavailableError,
    EquityListingApplicability,
    EquityListingApplicabilityClassification,
    EquityListingApplicabilitySource,
    EquityPriorSessionRecommendation,
    EquityRecommendationHistoryUnavailableError,
    EquityReferenceProvenanceKind,
    EquityResearchReferenceProvenanceSource,
    EquityResearchWindowProvenance,
    EquitySessionRecommendationHistory,
    EquitySessionRecommendationHistorySource,
    EquitySyntheticEligibilityCriterion,
    EquitySyntheticEligibilityFinding,
    EquitySyntheticEligibilityVerdict,
    EquitySyntheticEntryTimingPolicy,
    EquitySyntheticListingEligibility,
    EquitySyntheticListingEligibilitySource,
    EquitySyntheticPolicyCatalogCommitUnconfirmedError,
    EquitySyntheticPolicyCatalogNotCommittedError,
    EquitySyntheticPolicyCatalogOutcome,
    EquitySyntheticPolicyCatalogStore,
    EquitySyntheticPolicyCatalogUnavailableError,
    EquitySyntheticPolicyCatalogWrite,
    EquitySyntheticRecommendationReferenceSource,
    EquitySyntheticReferenceUnavailableError,
    EquityVenueDayClassification,
    EquityVenueDayReference,
    EquityVenueDayReferenceSource,
    InvalidEquityRecommendationReferenceError,
)
from northstar_application.ports import (
    equity_recommendation_references as references_module,
)

_NSE = ExchangeCode("NSE")
_INR = Currency("INR")
_DAY = date(2026, 8, 12)
_LISTING = ListingReference(Symbol("M&M"), _NSE)
_OTHER = ListingReference(Symbol("RELIANCE"), _NSE)
_IST = timezone(timedelta(hours=5, minutes=30))
_SOURCE = "synthetic-ei4c-source"
_FINALITY = EquityCandleFinalityPolicy("synthetic-ei4c/1", frozenset({_SOURCE}), True)
_CONFIG = EquityOpeningRangeBreakoutConfiguration(
    strategy=StrategyIdentity("ei4c-orb-vwap-volume"),
    version="synthetic-ei4c-research/1",
    opening_range_bars=3,
    volume_multiplier=Decimal("1.5"),
    minimum_baseline_bars=3,
)
_R = EquityRecommendationEligibilityReason
_S = EquityRecommendationGenerationStatus
_C = EquityRecommendationCheck
_CS = EquityRecommendationCheckStatus
_CORE = EquityRecommendationRejectionReason
_ROLE = EquityIntradayResearchTimelineRole
_CRITERIA = tuple(EquitySyntheticEligibilityCriterion)
_ELIGIBLE = EquitySyntheticEligibilityVerdict.ELIGIBLE

Candle = tuple[str, str, str, str, str]

# Bars 1-3 form the opening range (high 102, low 99); bar 4 (ending 09:35 IST)
# breaks out with close 102.5 and bars 5 and 6 qualify again.
_MORNING: dict[int, Candle] = {
    1: ("100", "101", "99", "100.5", "1000"),
    2: ("100.5", "102", "100", "101.5", "1000"),
    3: ("101.5", "102", "100.5", "101", "1000"),
    4: ("101", "103", "101", "102.5", "2000"),
    5: ("102.5", "104", "102", "103.5", "3000"),
    6: ("103.5", "105", "103", "104.5", "4500"),
}


def _v(name: str) -> EquitySyntheticVersion:
    return EquitySyntheticVersion(f"synthetic-ei4c-{name}")


def _ist(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 8, 12, hour, minute, second, tzinfo=_IST).astimezone(UTC)


def _end(index: int) -> datetime:
    return _ist(9, 15) + timedelta(minutes=5 * index)


def _pit(value: datetime) -> PointInTime:
    return PointInTime(value.isoformat().replace("+00:00", "Z"))


def _at(hour: int, minute: int, second: int = 0) -> PointInTime:
    return _pit(_ist(hour, minute, second))


# ----------------------------------------------------------------------
# EI-2B evidence doubles
# ----------------------------------------------------------------------


class _Venue(EquityVenueDayReferenceSource):
    """A configurable venue reference: NORMAL 09:15-15:30 IST unless changed by a test."""

    def __init__(self) -> None:
        self.classification = EquityVenueDayClassification.NORMAL_SESSION
        self.closes = time(15, 30)
        self.reference = "test-venue/1"
        self.answer_date: date | None = None

    def venue_day(self, exchange_code: ExchangeCode, trading_date: date) -> EquityVenueDayReference:
        normal = self.classification is EquityVenueDayClassification.NORMAL_SESSION
        day = self.answer_date or trading_date
        return EquityVenueDayReference(
            exchange_code=exchange_code,
            trading_date=day,
            classification=self.classification,
            common_opens_at=_pit(_ist(9, 15)) if normal else None,
            common_closes_at=_pit(_ist(self.closes.hour, self.closes.minute)) if normal else None,
            utc_offset=timedelta(hours=5, minutes=30) if normal else None,
            reference=self.reference,
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
        self.unavailable = False

    def issue(self, snapshot: EquityCandleEvidenceSnapshot) -> int:
        self.documents.setdefault(snapshot.snapshot_id, equity_candle_snapshot_json(snapshot))
        return 1

    def get_snapshot(self, snapshot_id: str) -> EquityCandleEvidenceSnapshot | None:
        if self.unavailable:
            raise EquityCandleEvidenceUnavailableError("test: snapshot storage is unavailable")
        text = self.documents.get(snapshot_id)
        return None if text is None else equity_candle_snapshot_from_document(json.loads(text))


def _observation(index: int, candle: Candle, revision: str = "r1") -> EquityCandleObservationRecord:
    o, h, low, c, v = candle
    end = _end(index)
    bar = HistoricalOHLCVBar(
        symbol=_LISTING.symbol,
        exchange_code=_NSE,
        point_in_time=_pit(end),
        timeframe=Timeframe("5m"),
        open=Price(Decimal(o), _INR),
        high=Price(Decimal(h), _INR),
        low=Price(Decimal(low), _INR),
        close=Price(Decimal(c), _INR),
        volume=Quantity(Decimal(v)),
    )
    event = f"evt-{index}"
    return EquityCandleObservationRecord(
        f"obs-{event}-{revision}",
        EquityCandleObservation(
            bar, _SOURCE, event, revision, _pit(end + timedelta(seconds=5)), None
        ),
        True,
    )


def _finality(index: int) -> EquityCandleFinalityRecord:
    end = _end(index)
    return EquityCandleFinalityRecord(
        f"fin-evt-{index}",
        EquityCandleFinalityEvidence(
            _SOURCE, f"evt-{index}", "r1", _pit(end + timedelta(seconds=10)), "synthetic", True
        ),
        _pit(end + timedelta(seconds=11)),
    )


class _TimelineStore(EquityIntradayResearchTimelineStore):
    """A test-only compare-and-append store; ``unavailable`` injects an outage."""

    def __init__(self) -> None:
        self.records: dict[tuple[Any, ...], StoredEquityIntradayResearchTimeline] = {}
        self.unavailable = False
        self.loads = 0

    def load(
        self, strategy: StrategyIdentity, listing: ListingReference, trading_date: date
    ) -> StoredEquityIntradayResearchTimeline | None:
        self.loads += 1
        if self.unavailable:
            raise EquityIntradayResearchTimelineUnavailableError("test: storage is unavailable")
        return self.records.get((strategy, listing, trading_date))

    def append(
        self, timeline: StoredEquityIntradayResearchTimeline, expected_head_fingerprint: str | None
    ) -> None:
        key = (timeline.strategy, timeline.listing, timeline.trading_date)
        stored = self.records.get(key)
        if (None if stored is None else stored.head_fingerprint) != expected_head_fingerprint:
            raise EquityIntradayResearchTimelineStaleError("test: the head moved")
        self.records[key] = timeline

    def only(self) -> StoredEquityIntradayResearchTimeline:
        (record,) = self.records.values()
        return record


class _World:
    """A journal, snapshot store, venue double and authoritative timeline store."""

    def __init__(self, candles: dict[int, Candle] | None = None) -> None:
        self.journal, self.snapshots, self.venue = _Journal(), _Snapshots(), _Venue()
        self.timelines = _TimelineStore()
        self.issuer = IssueEquityCandleEvidenceSnapshotUseCase(
            self.journal,
            self.snapshots,
            _INR,
            _FINALITY,
            NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW,
            reference_provenance=_Provenance(),
        )
        self.add(_MORNING if candles is None else candles)

    def add(self, candles: dict[int, Candle], revision: str = "r1") -> None:
        for index, candle in sorted(candles.items()):
            self.journal.append(_observation(index, candle, revision))
            if revision == "r1":
                self.journal.append(_finality(index))

    def issue(self, as_of: datetime) -> EquityCandleEvidenceSnapshot:
        window = EvaluateEquityResearchWindowUseCase(
            self.venue, _Applicability(), NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW
        ).execute(_LISTING, _DAY, _pit(as_of))
        return self.issuer.execute(window, _pit(as_of), _pit(as_of))

    def commit(self, *as_ofs: datetime) -> None:
        ids = [self.issue(as_of).snapshot_id for as_of in as_ofs]
        advance = AdvanceEquityIntradayResearchTimelineUseCase(
            self.timelines, self.snapshots, _CONFIG
        ).execute(_LISTING, _DAY, ids)
        assert advance.status is EquityIntradayResearchTimelineAdvanceStatus.COMMITTED


def _committed(*as_ofs: datetime) -> _World:
    world = _World()
    world.commit(*(as_ofs or (_ist(9, 31), _ist(9, 36), _ist(9, 41))))
    return world


# ----------------------------------------------------------------------
# Synthetic reference doubles
# ----------------------------------------------------------------------


def _timing_policy(
    version: str = "timing/1",
    signal: time = time(10, 30),
    entry: time = time(10, 45),
    exit_: time = time(14, 0),
    remaining: int = 60,
    latency: timedelta = timedelta(0),
) -> EquitySyntheticEntryTimingPolicy:
    return EquitySyntheticEntryTimingPolicy(
        version=_v(version),
        exchange_code=_NSE,
        research_window_policy_version=NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW.version,
        latest_signal_bar_end=signal,
        latest_entry_bar_opening=entry,
        modeled_exit_boundary=exit_,
        minimum_remaining_session=timedelta(minutes=remaining),
        modeled_issuance_latency=latency,
    )


def _generation(
    version: str = "generation/1", minimum: str = "1.5"
) -> EquitySyntheticGenerationPolicy:
    return EquitySyntheticGenerationPolicy(
        _v(version), Decimal("0.01"), Decimal("0"), Decimal("2"), Decimal(minimum)
    )


def _risk(
    version: str = "risk/1",
    capital: str = "100000",
    risk: str = "0.01",
    exposure: str = "0.2",
    currency: Currency = _INR,
) -> EquitySyntheticRiskPolicy:
    return EquitySyntheticRiskPolicy(
        _v(version), Money(Decimal(capital), currency), Decimal(risk), Decimal(exposure)
    )


def _costs(
    version: str = "costs/1",
    entry: str = "0.0003",
    exit_: str = "0.0005",
    slippage: str = "0.0002",
    rounding: str = "0.01",
) -> EquitySyntheticCostSchedule:
    def side(name: str, which: EquityCostSide, rate: str) -> EquitySyntheticCostComponent:
        return EquitySyntheticCostComponent(name, which, Decimal(rate), Decimal(0), Decimal(0))

    return EquitySyntheticCostSchedule(
        _v(version),
        _INR,
        (
            side("synthetic-entry", EquityCostSide.ENTRY, entry),
            side("synthetic-exit", EquityCostSide.EXIT, exit_),
        ),
        Decimal(slippage),
        Decimal(rounding),
    )


def _increment(version: str = "increment/1") -> EquityPriceIncrement:
    return EquityPriceIncrement(Decimal("0.05"), _v(version))


class _References(EquitySyntheticRecommendationReferenceSource):
    """Holds synthetic inputs by version; nothing is defaulted."""

    def __init__(self) -> None:
        self.held: dict[str, dict[str, object]] = {
            "timing": {},
            "generation": {},
            "risk": {},
            "costs": {},
            "increment": {},
        }
        for kind, value in (
            ("timing", _timing_policy()),
            ("generation", _generation()),
            ("risk", _risk()),
            ("costs", _costs()),
            ("increment", _increment()),
        ):
            self.hold(kind, value)
        self.unavailable: set[str] = set()
        self.answers: dict[str, object] = {}

    def hold(self, kind: str, value: Any) -> None:
        self.held[kind][value.version.value] = value

    def _get(self, kind: str, version: EquitySyntheticVersion) -> Any:
        if kind in self.unavailable:
            raise EquitySyntheticReferenceUnavailableError(f"test: {kind} source is down")
        if kind in self.answers:
            return self.answers[kind]
        return self.held[kind].get(version.value)

    def timing_policy(self, version: EquitySyntheticVersion) -> Any:
        return self._get("timing", version)

    def generation_policy(self, version: EquitySyntheticVersion) -> Any:
        return self._get("generation", version)

    def risk_policy(self, version: EquitySyntheticVersion) -> Any:
        return self._get("risk", version)

    def cost_schedule(self, version: EquitySyntheticVersion) -> Any:
        return self._get("costs", version)

    def price_increment(
        self, version: EquitySyntheticVersion, listing: ListingReference, trading_date: date
    ) -> Any:
        return self._get("increment", version)


def _eligibility(
    version: str = "eligibility/1",
    statuses: dict[EquitySyntheticEligibilityCriterion, EquitySyntheticEligibilityVerdict]
    | None = None,
    listing: ListingReference = _LISTING,
) -> EquitySyntheticListingEligibility:
    statuses = statuses or {}
    return EquitySyntheticListingEligibility(
        version=_v(version),
        universe_version=_v("universe/1"),
        listing=listing,
        trading_date=_DAY,
        findings=tuple(
            EquitySyntheticEligibilityFinding(c, statuses.get(c, _ELIGIBLE), "synthetic fixture")
            for c in _CRITERIA
        ),
        provenance="test-synthetic-eligibility",
    )


class _Eligibility(EquitySyntheticListingEligibilitySource):
    def __init__(self) -> None:
        self.answer: EquitySyntheticListingEligibility | None = _eligibility()
        self.unavailable = False

    def eligibility(
        self, version: EquitySyntheticVersion, listing: ListingReference, trading_date: date
    ) -> EquitySyntheticListingEligibility | None:
        if self.unavailable:
            raise EquitySyntheticReferenceUnavailableError("test: eligibility source is down")
        answer = self.answer
        if answer is not None and answer.version == version:
            return answer
        if answer is not None and version == _v("eligibility/2"):
            return dataclasses.replace(answer, version=version)
        return answer if answer is not None and version.value.endswith("/wrong") else None


def _prior(
    listing: ListingReference, key: str = "prior-1", sequence: int = 1
) -> EquityPriorSessionRecommendation:
    return EquityPriorSessionRecommendation(
        key,
        StrategyIdentity("ei4c-other"),
        listing,
        _DAY,
        EquityIssuanceKind.MODELED,
        _at(9, 20),
        sequence,
    )


def _history(
    *records: EquityPriorSessionRecommendation,
    through: PointInTime | None = None,
    day: date = _DAY,
    watermark: int | None = None,
    reference: str = "test-history/1",
) -> EquitySessionRecommendationHistory:
    """A labelled synthetic test record, complete through 15:30 IST unless stated.

    It is a test fixture establishing a complete history, never production
    authority. Its recording watermark is the highest recording sequence it
    lists unless stated.
    """
    if watermark is None:
        watermark = max((r.recording_sequence for r in records), default=0)
    return EquitySessionRecommendationHistory(
        _NSE, day, records, through or _at(15, 30), watermark, reference
    )


class _State(EquitySessionRecommendationHistorySource):
    """Answers one fixed record; ``asked`` records each requested as-of instant."""

    def __init__(self) -> None:
        self.answer: EquitySessionRecommendationHistory | None = _history()
        self.unavailable = False
        self.asked: list[PointInTime] = []

    def session_history(
        self, exchange_code: ExchangeCode, trading_date: date, as_of: PointInTime
    ) -> EquitySessionRecommendationHistory | None:
        self.asked.append(as_of)
        if self.unavailable:
            raise EquityRecommendationHistoryUnavailableError("test: record is unreadable")
        return self.answer


_CatalogBinding = EquitySyntheticPolicyDefinitionBinding | EquitySyntheticPolicySetBinding


class _Catalog(EquitySyntheticPolicyCatalogStore):
    """Deterministic test double of the EI4-D10 catalog port; stores canonical text only.

    It rebuilds every binding with ``from_stored`` on load, as an adapter
    must, so a corrupted entry fails exactly as a stored one would. Its
    scripted failures exercise the Application's semantics; nothing here
    establishes SQLite durability, atomicity or cross-process correctness.
    """

    def __init__(self) -> None:
        self.definitions: dict[EquitySyntheticPolicyBindingIdentity, tuple[str, str]] = {}
        self.sets: dict[EquitySyntheticPolicyBindingIdentity, tuple[str, str]] = {}
        self.script: list[str] = []
        self.load_failure: str | None = None
        self.before_load: Callable[[], None] | None = None
        self.publishes = 0
        self.loads = 0

    def seed(self, bundle: EquitySyntheticPolicyBundle, *, with_set: bool = True) -> None:
        """Test setup: store a bundle's bindings directly."""
        for c in bundle.components:
            self.definitions.setdefault(c.identity, (c.canonical_json(), c.fingerprint))
        if with_set:
            s = bundle.policy_set
            self.sets.setdefault(s.identity, (s.reference_json(), s.policy_set_fingerprint))

    def _definition(
        self, identity: EquitySyntheticPolicyBindingIdentity
    ) -> EquitySyntheticPolicyDefinitionBinding | None:
        raw = self.definitions.get(identity)
        return None if raw is None else EquitySyntheticPolicyDefinitionBinding.from_stored(*raw)

    def _set(
        self, identity: EquitySyntheticPolicyBindingIdentity
    ) -> EquitySyntheticPolicySetBinding | None:
        raw = self.sets.get(identity)
        return (
            None
            if raw is None
            else EquitySyntheticPolicySetBinding.from_stored(identity.namespace, *raw)
        )

    def _before_load(self) -> None:
        if self.before_load is not None:
            hook, self.before_load = self.before_load, None
            hook()
        self.loads += 1
        if self.load_failure == "unavailable":
            raise EquitySyntheticPolicyCatalogUnavailableError("test: catalog down")
        if self.load_failure == "invalid":
            raise InvalidEquitySyntheticPolicyBindingError("test: a stored row is corrupt")

    def load_definition(self, identity):  # type: ignore[no-untyped-def]
        self._before_load()
        return self._definition(identity)

    def load_policy_set(self, identity):  # type: ignore[no-untyped-def]
        self._before_load()
        return self._set(identity)

    def publish(self, bundle: EquitySyntheticPolicyBundle) -> EquitySyntheticPolicyCatalogWrite:
        self.publishes += 1
        mode = self.script.pop(0) if self.script else "ok"
        failures = {
            "unavailable": EquitySyntheticPolicyCatalogUnavailableError("test: catalog down"),
            "rollback": EquitySyntheticPolicyCatalogNotCommittedError("test: rolled back"),
            "invalid": InvalidEquitySyntheticPolicyBindingError("test: stored row corrupt"),
            "incompatible": IncompatibleEquitySyntheticPolicyBindingError("test: format /2"),
            "lost_before": EquitySyntheticPolicyCatalogCommitUnconfirmedError(
                "test: COMMIT failed"
            ),
        }
        if mode in failures:
            raise failures[mode]
        outcome_type = EquitySyntheticPolicyCatalogOutcome
        if mode == "contradicting":
            return EquitySyntheticPolicyCatalogWrite(outcome_type.CONFLICT, (bundle.policy_set,))
        conflicts: list[_CatalogBinding] = []
        present = 0
        for offered in (*bundle.components, bundle.policy_set):
            stored = (
                self._set(offered.identity)
                if isinstance(offered, EquitySyntheticPolicySetBinding)
                else self._definition(offered.identity)
            )
            if stored is not None:
                present += 1
                if stored != offered:
                    conflicts.append(stored)
        if conflicts:
            return EquitySyntheticPolicyCatalogWrite(outcome_type.CONFLICT, tuple(conflicts))
        outcome = outcome_type.EQUIVALENT if present == 6 else outcome_type.COMMITTED
        self.seed(bundle)
        if mode == "lost_after":
            raise EquitySyntheticPolicyCatalogCommitUnconfirmedError("test: ack lost after COMMIT")
        return EquitySyntheticPolicyCatalogWrite(outcome, ())

    def rows(self) -> tuple[dict[Any, Any], dict[Any, Any]]:
        return dict(self.definitions), dict(self.sets)


_HELD = (
    ("generation", "generation_policy_version", "generation_policy"),
    ("risk", "risk_policy_version", "risk_policy"),
    ("costs", "cost_schedule_version", "cost_schedule"),
    ("increment", "price_increment_version", "price_increment"),
    ("timing", "timing_policy_version", "timing_policy"),
)


class _Ports:
    """The generator's ports, and the published-policy catalog (EI4-D10).

    ``catalog`` is the catalog generation reads, and recording too. As test
    setup, ``run`` first publishes -- explicitly, through the real S1
    PublishEquitySyntheticPoliciesUseCase -- the policy set the request names,
    from the definitions ``references`` holds. A label later held with other
    content is then a conflicting reuse: publication refuses it, the first
    binding stays, and generation reports POLICY_REFERENCE_CONFLICT. Tests of
    unpublished or failing catalogs set ``publish`` False.
    """

    def __init__(self, world: _World) -> None:
        self.world = world
        self.references = _References()
        self.eligibility = _Eligibility()
        self.state = _State()
        self.catalog = _Catalog()
        self.publish = True
        self.evaluator_version: str | None = None  # None: the use case's default (/1)

    def use_case(self) -> GenerateEquityTradeRecommendationUseCase:
        selection = (
            {} if self.evaluator_version is None else {"evaluator_version": self.evaluator_version}
        )
        return GenerateEquityTradeRecommendationUseCase(
            self.world.timelines,
            self.world.snapshots,
            self.world.venue,
            self.references,
            self.eligibility,
            self.state,
            self.catalog,
            **selection,
        )

    def publish_policies(
        self, reference: EquitySyntheticPolicySetReference
    ) -> EquitySyntheticPolicyPublicationResult | None:
        """Test setup: publish ``reference`` and the held components it names, if all held."""
        if not isinstance(self.references, _References):
            return None
        components = {
            field: self.references.held[kind].get(getattr(reference, label).value)
            for kind, label, field in _HELD
        }
        if any(component is None for component in components.values()):
            return None
        try:
            bundle = EquitySyntheticPolicyBundle.of(
                EquityResolvedSyntheticPolicySet(reference=reference, **components)
            )
        except (InvalidEquityRecommendationEvidenceError, InvalidEquitySyntheticPolicyBindingError):
            return None
        return PublishEquitySyntheticPoliciesUseCase(self.catalog).execute(bundle)

    def run(self, **changes: Any) -> EquityRecommendationEligibilityAudit:
        request = _request(**changes)
        if self.publish:
            self.publish_policies(request.policy_set)
        return self.use_case().execute(request)


_POLICY_FIELDS = (
    "generation_policy_version",
    "risk_policy_version",
    "cost_schedule_version",
    "price_increment_version",
    "timing_policy_version",
)


def _policy_set(set_version: str = "set/1", **versions: Any) -> EquitySyntheticPolicySetReference:
    components = {
        "generation_policy_version": _v("generation/1"),
        "risk_policy_version": _v("risk/1"),
        "cost_schedule_version": _v("costs/1"),
        "price_increment_version": _v("increment/1"),
        "timing_policy_version": _v("timing/1"),
    }
    components.update(versions)
    return EquitySyntheticPolicySetReference(_v(set_version), **components)


def _set_label(components: dict[str, Any]) -> str:
    """The set label a request names unless it states one (EI4-D10).

    A published set label binds exactly its five component labels, so a set
    naming other components is another set: "set/1" plus the changed labels.
    The default set keeps "set/1", and with it its pinned identities.
    """
    defaults = _policy_set()
    changed = [
        version.value.removeprefix("synthetic-ei4c-")
        for name, version in components.items()
        if isinstance(version, EquitySyntheticVersion) and version != getattr(defaults, name)
    ]
    return "+".join(["set/1", *changed])


def _request(**changes: Any) -> EquityTradeRecommendationRequest:
    """Build a request; component version keywords build its policy set."""
    components = {name: changes.pop(name) for name in _POLICY_FIELDS if name in changes}
    set_version = changes.pop("policy_set_version", None) or _set_label(components)
    values: dict[str, Any] = {
        "configuration": _CONFIG,
        "listing": _LISTING,
        "trading_date": _DAY,
        "policy_set": _policy_set(set_version, **components),
        "listing_eligibility_version": _v("eligibility/1"),
    }
    values.update(changes)
    return EquityTradeRecommendationRequest(**values)


def _ports(*as_ofs: datetime) -> _Ports:
    return _Ports(_committed(*as_ofs))


def _check(audit: EquityRecommendationEligibilityAudit, check: EquityRecommendationCheck) -> Any:
    return next(result for result in audit.checks if result.check is check)


def _refused(
    audit: EquityRecommendationEligibilityAudit,
    status: EquityRecommendationGenerationStatus,
    *reasons: EquityRecommendationEligibilityReason,
) -> None:
    assert audit.status is status, (audit.status, audit.reasons)
    assert audit.reasons == reasons
    assert audit.proposal is None


# ----------------------------------------------------------------------
# A-E -- authoritative candidate selection
# ----------------------------------------------------------------------


def test_a_the_first_authoritative_candidate_generates_the_proposal() -> None:
    ports_ = _ports()
    audit = ports_.run()
    assert audit.status is _S.PROPOSED and audit.reasons == ()
    timeline = ports_.world.timelines.only()
    roles = [entry.role for entry in timeline.entries]
    assert roles == [_ROLE.NO_SIGNAL, _ROLE.CANDIDATE, _ROLE.REPEAT_QUALIFYING]
    candidate = timeline.entries[1]
    proposal = audit.proposal
    assert proposal is not None
    source = proposal.definition.source
    assert source.candidate_entry_fingerprint == candidate.entry_fingerprint
    assert source.research_decision_fingerprint == candidate.decision_fingerprint
    assert source.evidence_snapshot_id == candidate.snapshot_id
    assert source.decision_as_of == candidate.decision_instant == _at(9, 36)
    assert source.research_configuration_version.value == _CONFIG.version
    assert audit.candidate is not None
    assert audit.candidate.candidate_sequence == 2
    assert audit.candidate.manifest_fingerprint == candidate.manifest_fingerprint
    # The head seen while loading is diagnostics only (EI4-D9-C).
    assert audit.observed_timeline_head == timeline.head_fingerprint
    assert "timeline_head" not in json.dumps(audit.assessment_document())


def test_b_no_candidate_is_an_explicit_non_generation_result() -> None:
    never = _Ports(_World())
    audit = never.run()
    _refused(audit, _S.NO_CANDIDATE, _R.NO_AUTHORITATIVE_CANDIDATE)
    assert "EI3-D11" in _check(audit, _C.AUTHORITATIVE_CANDIDATE).detail
    assert audit.candidate is None and audit.core_outcome is None
    no_signal = _ports(_ist(9, 31))
    audit = no_signal.run()
    _refused(audit, _S.NO_CANDIDATE, _R.NO_AUTHORITATIVE_CANDIDATE)
    assert "holds no CANDIDATE" in _check(audit, _C.AUTHORITATIVE_CANDIDATE).detail
    for check in (_C.SYNTHETIC_REFERENCES, _C.SESSION_TIMING, _C.CORE_FORMATION):
        assert _check(audit, check).status is _CS.NOT_EVALUATED


def test_c_repeat_qualifying_never_creates_another_proposal() -> None:
    ports_ = _ports()
    first = ports_.run()
    ports_.world.commit(_ist(9, 46))
    roles = [entry.role for entry in ports_.world.timelines.only().entries]
    assert roles.count(_ROLE.CANDIDATE) == 1 and roles.count(_ROLE.REPEAT_QUALIFYING) == 2
    later = ports_.run()
    assert later.proposal == first.proposal
    assert later.proposal is not None
    assert later.proposal.definition.source.decision_as_of == _at(9, 36)


def test_d_no_signal_and_rejected_entries_never_generate() -> None:
    gap = _World({k: v for k, v in _MORNING.items() if k != 2})  # bar 2 is missing
    gap.commit(_ist(9, 31), _ist(9, 36), _ist(9, 41))
    roles = {entry.role for entry in gap.timelines.only().entries}
    assert roles == {_ROLE.REJECTED}
    _refused(_Ports(gap).run(), _S.NO_CANDIDATE, _R.NO_AUTHORITATIVE_CANDIDATE)
    quiet = _ports(_ist(9, 31))
    assert {e.role for e in quiet.world.timelines.only().entries} == {_ROLE.NO_SIGNAL}
    _refused(quiet.run(), _S.NO_CANDIDATE, _R.NO_AUTHORITATIVE_CANDIDATE)


def test_e_generation_reads_only_verified_rehydrated_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ports_ = _ports()
    seen: list[StoredEquityIntradayResearchTimeline] = []
    real = generate_module.rehydrate_equity_intraday_research_timeline

    def spy(identity, stored, snapshots):  # type: ignore[no-untyped-def]
        seen.append(stored)
        return real(identity, stored, snapshots)

    monkeypatch.setattr(generate_module, "rehydrate_equity_intraday_research_timeline", spy)
    loads = ports_.world.timelines.loads
    assert ports_.run().status is _S.PROPOSED
    assert seen == [ports_.world.timelines.only()]
    assert ports_.world.timelines.loads == loads + 1


# ----------------------------------------------------------------------
# F-H -- research integrity and configuration
# ----------------------------------------------------------------------


def _replace_entry(world: _World, index: int, **changes: Any) -> None:
    """Test-only: rewrite a stored entry after the fact."""
    record = world.timelines.only()
    entries = list(record.entries)
    entries[index] = dataclasses.replace(entries[index], **changes)
    key = (record.strategy, record.listing, record.trading_date)
    world.timelines.records[key] = dataclasses.replace(record, entries=tuple(entries))


def test_f_an_invalid_research_fingerprint_is_an_integrity_failure() -> None:
    ports_ = _ports()
    _replace_entry(ports_.world, 1, decision_fingerprint="eir1-sha256:" + "0" * 64)
    audit = ports_.run()
    _refused(audit, _S.RESEARCH_INTEGRITY_FAILURE, _R.RESEARCH_TIMELINE_INVALID)
    assert "decision fingerprint" in _check(audit, _C.AUTHORITATIVE_CANDIDATE).detail
    ports_ = _ports()
    _replace_entry(ports_.world, 1, entry_fingerprint="eirt1-sha256:" + "0" * 64)
    _refused(ports_.run(), _S.RESEARCH_INTEGRITY_FAILURE, _R.RESEARCH_TIMELINE_INVALID)


def test_f_a_decision_that_does_not_reproduce_is_never_masked_as_no_signal() -> None:
    ports_ = _ports()
    original = ports_.world.timelines.only().entries[1].decision_document
    document = json.loads(original)
    document["values"]["ended_bars"] = 5
    altered = canonical_equity_candle_json(document)
    fingerprint = "eir1-sha256:" + hashlib.sha256(altered.encode()).hexdigest()
    _replace_entry(ports_.world, 1, decision_document=altered, decision_fingerprint=fingerprint)
    _refused(ports_.run(), _S.RESEARCH_INTEGRITY_FAILURE, _R.RESEARCH_DECISION_NOT_REPRODUCED)


def test_g_a_missing_source_snapshot_fails_closed() -> None:
    ports_ = _ports()
    snapshot_id = ports_.world.timelines.only().entries[1].snapshot_id
    del ports_.world.snapshots.documents[snapshot_id]  # test-only
    audit = ports_.run()
    _refused(audit, _S.RESEARCH_INTEGRITY_FAILURE, _R.RESEARCH_TIMELINE_INVALID)
    assert "unavailable" in _check(audit, _C.AUTHORITATIVE_CANDIDATE).detail


def test_g_unavailable_storage_is_a_dependency_outage_not_a_rejection() -> None:
    ports_ = _ports()
    ports_.world.snapshots.unavailable = True
    _refused(ports_.run(), _S.DEPENDENCY_UNAVAILABLE, _R.EVIDENCE_SNAPSHOT_UNAVAILABLE)
    ports_ = _ports()
    ports_.world.timelines.unavailable = True
    _refused(ports_.run(), _S.DEPENDENCY_UNAVAILABLE, _R.RESEARCH_TIMELINE_UNAVAILABLE)


def test_h_another_research_configuration_is_refused() -> None:
    ports_ = _ports()
    other = dataclasses.replace(_CONFIG, version="synthetic-ei4c-research/2")
    audit = ports_.run(configuration=other)
    _refused(audit, _S.NOT_ELIGIBLE, _R.RESEARCH_CONFIGURATION_MISMATCH)
    assert audit.candidate is None


def test_h_a_stored_timeline_for_another_listing_fails_closed() -> None:
    ports_ = _ports()
    record = ports_.world.timelines.only()
    key = (record.strategy, record.listing, record.trading_date)
    ports_.world.timelines.records[key] = dataclasses.replace(record, listing=_OTHER)  # test-only
    _refused(ports_.run(), _S.RESEARCH_INTEGRITY_FAILURE, _R.RESEARCH_TIMELINE_INVALID)


# ----------------------------------------------------------------------
# I -- the original EI-3A research values
# ----------------------------------------------------------------------


def test_i_core_receives_the_original_ei3a_signal_close_and_opening_range_low(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ports_ = _ports()
    calls: list[tuple[Any, ...]] = []
    real = generate_module.form_equity_recommendation

    def spy(*args):  # type: ignore[no-untyped-def]
        calls.append(args)
        return real(*args)

    monkeypatch.setattr(generate_module, "form_equity_recommendation", spy)
    audit = ports_.run()
    ((source, close, low, generation, risk, increment, costs),) = calls
    assert (close, low) == (Price(Decimal("102.5"), _INR), Price(Decimal("99"), _INR))
    assert (generation, risk, costs, increment) == (_generation(), _risk(), _costs(), _increment())
    assert audit.proposal is not None
    levels = audit.proposal.definition.levels
    assert (levels.entry_zone.low, levels.entry_zone.high, levels.stop, levels.target) == (
        Price(Decimal("101.5"), _INR),
        Price(Decimal("102.5"), _INR),
        Price(Decimal("99"), _INR),
        Price(Decimal("109.5"), _INR),
    )
    assert source == audit.proposal.definition.source


# ----------------------------------------------------------------------
# J-N -- timing and look-ahead
# ----------------------------------------------------------------------


def _window(audit: EquityRecommendationEligibilityAudit) -> EquityModeledEntryWindow:
    assert audit.entry_window is not None
    return audit.entry_window


def test_j_eligibility_starts_at_the_first_grid_bar_opening_at_or_after_issuance() -> None:
    ports_ = _ports()
    window = _window(ports_.run())
    assert window.timing == EquityRecommendationTiming(
        _at(9, 36), EquityIssuanceKind.MODELED, _at(9, 36), _at(9, 40)
    )
    assert window.latest_entry_bar_opens_at == _at(10, 45)
    assert window.modeled_exit_at == _at(14, 0)
    # Modeled issuance is the decision plus the policy's latency (EI4-D9-C).
    ports_.references.hold("timing", _timing_policy("timing/4m", latency=timedelta(minutes=4)))
    exact = _window(ports_.run(timing_policy_version=_v("timing/4m")))
    assert (exact.timing.issued_at, exact.timing.eligible_from) == (_at(9, 40), _at(9, 40))
    ports_.references.hold(
        "timing", _timing_policy("timing/4m1s", latency=timedelta(minutes=4, seconds=1))
    )
    late = _window(ports_.run(timing_policy_version=_v("timing/4m1s")))
    assert (late.timing.issued_at, late.timing.eligible_from) == (_at(9, 40, 1), _at(9, 45))
    assert late.timing.decision_as_of == _at(9, 36)


def test_k_positive_finality_latency_never_uses_the_bar_already_open() -> None:
    # The 09:35 signal bar is final at 09:35:11; the decision is available at
    # 09:35:30. The 09:35-09:40 bar opened before that, so it can never
    # evidence a full-bar entry: eligibility starts at the 09:40 opening.
    ports_ = _ports(_ist(9, 31), _ist(9, 35, 30))
    audit = ports_.run()
    assert audit.status is _S.PROPOSED
    assert audit.candidate is not None
    assert audit.candidate.signal_interval_end == _at(9, 35)
    assert _window(audit).timing.decision_as_of == _at(9, 35, 30)
    assert _window(audit).timing.issued_at == _at(9, 35, 30)
    assert _window(audit).timing.eligible_from == _at(9, 40)


def test_l_a_signal_after_the_synthetic_cutoff_is_refused() -> None:
    ports_ = _ports()
    ports_.references.hold("timing", _timing_policy("timing/late", signal=time(9, 30)))
    audit = ports_.run(timing_policy_version=_v("timing/late"))
    _refused(audit, _S.NOT_ELIGIBLE, _R.AFTER_SIGNAL_CUTOFF)
    assert audit.core_outcome is None
    assert _check(audit, _C.CORE_FORMATION).status is _CS.NOT_EVALUATED


def test_l_no_permissible_entry_bar_and_insufficient_session_time() -> None:
    ports_ = _ports()
    ports_.references.hold(
        "timing", _timing_policy("timing/tight", signal=time(9, 35), entry=time(9, 35))
    )
    audit = ports_.run(timing_policy_version=_v("timing/tight"))
    _refused(audit, _S.NOT_ELIGIBLE, _R.NO_PERMISSIBLE_ENTRY_BAR)
    # Eligible from 09:40, leaving 50 minutes before a 10:30 exit; 60 are required.
    ports_.references.hold(
        "timing",
        _timing_policy("timing/short", signal=time(10, 0), entry=time(10, 0), exit_=time(10, 30)),
    )
    audit = ports_.run(timing_policy_version=_v("timing/short"))
    _refused(audit, _S.NOT_ELIGIBLE, _R.INSUFFICIENT_SESSION_TIME)
    ports_.references.hold(
        "timing", _timing_policy("timing/just", exit_=time(10, 50), remaining=70)
    )
    assert ports_.run(timing_policy_version=_v("timing/just")).status is _S.PROPOSED


def _whole_day() -> dict[int, Candle]:
    """Bars 1-3 form the range; bars 4-68 stay inside it; bar 69 (15:00) breaks out."""
    candles = {index: _MORNING[index] for index in (1, 2, 3)}
    for index in range(4, 69):
        candles[index] = ("101", "101.2", "100.8", "101", "1000")
    candles[69] = ("101", "103.5", "101", "103", "5000")
    return candles


def test_m_a_post_window_historical_decision_is_never_actionable() -> None:
    world = _World(_whole_day())
    world.commit(_ist(15, 10))
    (entry,) = world.timelines.only().entries
    assert entry.role is _ROLE.CANDIDATE
    ports_ = _Ports(world)
    audit = ports_.run()
    _refused(
        audit,
        _S.NOT_ELIGIBLE,
        _R.AFTER_SIGNAL_CUTOFF,
        _R.POST_WINDOW_DECISION,
        _R.NO_PERMISSIBLE_ENTRY_BAR,
    )
    assert audit.entry_window is None


@pytest.mark.parametrize(
    "classification",
    [
        EquityVenueDayClassification.SPECIAL_SESSION,
        EquityVenueDayClassification.CLOSED,
        EquityVenueDayClassification.NORMAL_SESSION_COMMON_BOUNDARY_UNSUPPORTED,
    ],
)
def test_n_special_or_unsupported_sessions_are_refused_never_re_timed(classification) -> None:  # type: ignore[no-untyped-def]
    ports_ = _ports()
    ports_.world.venue.classification = classification
    audit = ports_.run()
    _refused(audit, _S.NOT_ELIGIBLE, _R.SESSION_NOT_SUPPORTED)


def test_n_a_session_closing_before_the_modeled_exit_is_refused() -> None:
    ports_ = _ports()
    ports_.world.venue.closes = time(13, 30)
    audit = ports_.run()
    _refused(audit, _S.NOT_ELIGIBLE, _R.SESSION_NOT_SUPPORTED)
    assert "never re-timed" in _check(audit, _C.SESSION_TIMING).detail


def test_n_an_unresolvable_or_changed_session_reference_fails_closed() -> None:
    ports_ = _ports()
    ports_.world.venue.classification = EquityVenueDayClassification.NOT_STATED
    _refused(ports_.run(), _S.DEPENDENCY_UNAVAILABLE, _R.SESSION_REFERENCE_UNAVAILABLE)
    ports_ = _ports()
    ports_.world.venue.reference = "test-venue/2"
    _refused(ports_.run(), _S.DEPENDENCY_UNAVAILABLE, _R.SESSION_REFERENCE_MISMATCH)
    ports_ = _ports()
    ports_.world.venue.answer_date = date(2026, 8, 13)
    _refused(ports_.run(), _S.DEPENDENCY_UNAVAILABLE, _R.SESSION_REFERENCE_MISMATCH)


def test_n_a_timing_policy_off_the_research_window_grid_does_not_apply() -> None:
    ports_ = _ports()
    ports_.references.hold("timing", _timing_policy("timing/beyond", exit_=time(15, 10)))
    audit = ports_.run(timing_policy_version=_v("timing/beyond"))
    _refused(audit, _S.NOT_ELIGIBLE, _R.TIMING_POLICY_NOT_APPLICABLE)
    other = dataclasses.replace(
        _timing_policy("timing/other"), research_window_policy_version="w/2"
    )
    ports_.references.hold("timing", other)
    audit = ports_.run(timing_policy_version=_v("timing/other"))
    _refused(audit, _S.NOT_ELIGIBLE, _R.TIMING_POLICY_NOT_APPLICABLE)


# ----------------------------------------------------------------------
# O-Q -- synthetic references
# ----------------------------------------------------------------------


def test_o_versioned_synthetic_listing_eligibility_is_applied_and_recorded() -> None:
    ports_ = _ports()
    audit = ports_.run()
    assert audit.proposal is not None
    assert audit.proposal.listing_eligibility_version == _v("eligibility/1")
    assert audit.proposal.universe_version == _v("universe/1")
    assert not _eligibility().is_production_authority
    second = ports_.run(listing_eligibility_version=_v("eligibility/2"))
    assert second.proposal is not None
    assert second.proposal.listing_eligibility_version == _v("eligibility/2")
    for criterion in _CRITERIA:
        ports_.eligibility.answer = _eligibility(
            statuses={criterion: EquitySyntheticEligibilityVerdict.INELIGIBLE}
        )
        audit = ports_.run()
        _refused(audit, _S.NOT_ELIGIBLE, _R.LISTING_NOT_ELIGIBLE)
        assert criterion.value in _check(audit, _C.LISTING_ELIGIBILITY).detail


def test_o_an_eligibility_answer_for_another_version_or_listing_is_refused() -> None:
    ports_ = _ports()
    ports_.eligibility.answer = _eligibility(listing=_OTHER)
    _refused(ports_.run(), _S.DEPENDENCY_UNAVAILABLE, _R.LISTING_REFERENCE_MISMATCH)
    ports_.eligibility.answer = _eligibility()
    audit = ports_.run(listing_eligibility_version=_v("eligibility/wrong"))
    _refused(audit, _S.DEPENDENCY_UNAVAILABLE, _R.LISTING_REFERENCE_MISMATCH)


def test_p_unknown_or_unavailable_eligibility_is_refused_never_eligible() -> None:
    ports_ = _ports()
    ports_.eligibility.answer = _eligibility(
        statuses={
            EquitySyntheticEligibilityCriterion.LIQUIDITY: EquitySyntheticEligibilityVerdict.UNKNOWN
        }
    )
    audit = ports_.run()
    _refused(audit, _S.DEPENDENCY_UNAVAILABLE, _R.LISTING_ELIGIBILITY_UNKNOWN)
    assert "LIQUIDITY" in _check(audit, _C.SYNTHETIC_REFERENCES).detail
    ports_.eligibility.answer = None
    _refused(ports_.run(), _S.DEPENDENCY_UNAVAILABLE, _R.LISTING_REFERENCE_UNAVAILABLE)
    ports_.eligibility.answer = _eligibility()
    ports_.eligibility.unavailable = True
    _refused(ports_.run(), _S.DEPENDENCY_UNAVAILABLE, _R.LISTING_REFERENCE_UNAVAILABLE)


def test_p_an_eligibility_reference_must_answer_every_criterion() -> None:
    with pytest.raises(InvalidEquityRecommendationReferenceError, match="every criterion"):
        dataclasses.replace(_eligibility(), findings=_eligibility().findings[:-1])
    with pytest.raises(InvalidEquityRecommendationReferenceError):
        dataclasses.replace(_eligibility(), universe_version="universe/1")


def test_q_missing_increment_and_costs_are_reported_together_and_core_never_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ports_ = _ports()
    monkeypatch.setattr(generate_module, "form_equity_recommendation", None)
    audit = ports_.run(
        price_increment_version=_v("increment/missing"), cost_schedule_version=_v("costs/missing")
    )
    _refused(
        audit,
        _S.DEPENDENCY_UNAVAILABLE,
        _R.COST_SCHEDULE_UNAVAILABLE,
        _R.PRICE_INCREMENT_UNAVAILABLE,
    )
    assert audit.core_outcome is None
    for kind, reason in (
        ("timing", _R.TIMING_POLICY_UNAVAILABLE),
        ("generation", _R.GENERATION_POLICY_UNAVAILABLE),
        ("risk", _R.RISK_POLICY_UNAVAILABLE),
        ("costs", _R.COST_SCHEDULE_UNAVAILABLE),
        ("increment", _R.PRICE_INCREMENT_UNAVAILABLE),
    ):
        ports_.references.unavailable = {kind}
        audit = ports_.run()
        _refused(audit, _S.DEPENDENCY_UNAVAILABLE, reason)
        assert "unavailable" in _check(audit, _C.SYNTHETIC_REFERENCES).detail


def test_q_a_reference_answering_another_version_is_never_substituted() -> None:
    ports_ = _ports()
    ports_.references.answers["risk"] = _risk("risk/other")
    _refused(ports_.run(), _S.DEPENDENCY_UNAVAILABLE, _R.POLICY_REFERENCE_MISMATCH)


# ----------------------------------------------------------------------
# R-T -- Core formation
# ----------------------------------------------------------------------


def test_r_core_rejections_are_propagated_exactly() -> None:
    ports_ = _ports()
    ports_.references.hold("generation", _generation("generation/strict", minimum="100"))
    audit = ports_.run(generation_policy_version=_v("generation/strict"))
    _refused(audit, _S.NOT_ELIGIBLE, _R.CORE_FORMATION_REJECTED)
    assert isinstance(audit.core_outcome, EquityRecommendationRejection)
    assert audit.core_reasons == ("NET_REWARD_TO_RISK_BELOW_MINIMUM",)
    ports_.references.hold("risk", _risk("risk/usd", currency=Currency("USD")))
    audit = ports_.run(risk_policy_version=_v("risk/usd"))
    assert audit.core_reasons == ("INVALID_SYNTHETIC_POLICY",)
    ports_.references.hold("risk", _risk("risk/tiny", capital="100"))
    audit = ports_.run(risk_policy_version=_v("risk/tiny"))
    assert audit.core_reasons == (
        "NO_WHOLE_SHARE_QUANTITY",
        "INSUFFICIENT_RISK_BUDGET",
        "INSUFFICIENT_EXPOSURE_BUDGET",
    )
    core = json.loads(audit.canonical_json())["assessment"]["core_outcome"]
    assert core["format"] == "northstar.equity-recommendation-core-outcome/1"
    assert core["kind"] == "REJECTION"
    assert core["outcome"]["reasons"] == list(audit.core_reasons)


def test_s_core_costs_and_sizing_are_preserved_exactly() -> None:
    ports_ = _ports()
    audit = ports_.run()
    assert audit.proposal is not None and audit.candidate is not None
    definition = audit.proposal.definition
    direct = form_equity_recommendation(
        audit.candidate.source,
        Price(Decimal("102.5"), _INR),
        Price(Decimal("99"), _INR),
        _generation(),
        _risk(),
        _increment(),
        _costs(),
    )
    assert isinstance(direct, EquityRecommendationDefinition)
    assert definition == direct == audit.core_outcome
    sizing = definition.sizing
    assert sizing.quantity == 195
    assert (
        sizing.entry_notional,
        sizing.entry_cost,
        sizing.stop_exit_cost,
        sizing.target_exit_cost,
        sizing.net_risk,
        sizing.net_reward,
        sizing.risk_budget,
        sizing.exposure_budget,
    ) == tuple(
        Decimal(x)
        for x in ("19987.5", "10", "13.52", "14.95", "706.02", "1340.05", "1000", "20000")
    )
    assert definition.cost_schedule_version == _v("costs/1")
    assert definition.risk_policy_version == _v("risk/1")


def _search_ports() -> _Ports:
    ports_ = _ports()
    ports_.references.hold("generation", _generation("generation/near", minimum="1.95"))
    ports_.references.hold(
        "costs", _costs("costs/rupee", entry="0.0005", exit_="0", slippage="0", rounding="1")
    )
    ports_.references.hold("risk", _risk("risk/small", capital="2050", risk="0.1", exposure="1"))
    return ports_


_SEARCH = {
    "generation_policy_version": _v("generation/near"),
    "cost_schedule_version": _v("costs/rupee"),
    "risk_policy_version": _v("risk/small"),
}


def test_t_cores_exact_search_bound_is_propagated_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ports_ = _search_ports()
    # With rupee cost rounding the largest in-budget quantity (20) fails the
    # minimum; Core's exact search finds 19.
    sized = ports_.run(**_SEARCH)
    assert sized.proposal is not None and sized.proposal.definition.sizing.quantity == 19
    # Test-only: shrink Core's bound so the same search exceeds it.
    monkeypatch.setattr(core_sizing, "_SEARCH_LIMIT", 0)
    audit = ports_.run(**_SEARCH)
    # Indeterminate, not a failed reward-to-risk test and not an outage.
    _refused(audit, _S.EVALUATION_UNRESOLVED, _R.CORE_FORMATION_UNRESOLVED)
    assert audit.core_reasons == (_CORE.SIZING_SEARCH_LIMIT_EXCEEDED.value,)
    detail = _check(audit, _C.CORE_FORMATION).detail
    assert "SIZING_SEARCH_LIMIT_EXCEEDED" in detail and "reached its bound" in detail
    assert _R.CORE_FORMATION_REJECTED not in audit.reasons


def test_d_a_confirmed_core_rejection_stays_distinct_from_an_unresolved_one() -> None:
    ports_ = _search_ports()
    ports_.references.hold("generation", _generation("generation/high", minimum="2.5"))
    confirmed = ports_.run(**{**_SEARCH, "generation_policy_version": _v("generation/high")})
    # A minimum above the gross reward-to-risk of 2 fails at every quantity.
    _refused(confirmed, _S.NOT_ELIGIBLE, _R.CORE_FORMATION_REJECTED)
    assert confirmed.core_reasons == (_CORE.NET_REWARD_TO_RISK_BELOW_MINIMUM.value,)
    assert ports_.run(**_SEARCH).status is _S.PROPOSED
    with pytest.raises(InvalidEquityRecommendationEligibilityError, match="reported as"):
        dataclasses.replace(
            confirmed,
            checks=tuple(
                dataclasses.replace(c, reasons=(_R.CORE_FORMATION_UNRESOLVED,))
                if c.check is _C.CORE_FORMATION
                else c
                for c in confirmed.checks
            ),
        )


def test_d_reason_precedence_is_deterministic() -> None:
    assert equity_recommendation_status(()) is _S.PROPOSED
    assert equity_recommendation_status((_R.CORE_FORMATION_UNRESOLVED,)) is _S.EVALUATION_UNRESOLVED
    assert (
        equity_recommendation_status((_R.AFTER_SIGNAL_CUTOFF, _R.EXCLUSIVITY_POLICY_UNRESOLVED))
        is _S.NOT_ELIGIBLE
    )
    assert (
        equity_recommendation_status(
            (_R.OPEN_RECOMMENDATION_STATE_UNAVAILABLE, _R.EXPOSURE_POLICY_UNRESOLVED)
        )
        is _S.DEPENDENCY_UNAVAILABLE
    )
    assert (
        equity_recommendation_status((_R.RESEARCH_TIMELINE_INVALID, _R.NO_AUTHORITATIVE_CANDIDATE))
        is _S.RESEARCH_INTEGRITY_FAILURE
    )
    # End to end: a confirmed refusal outranks an unresolved one, and both are reported.
    ports_ = _ports()
    ports_.state.answer = _history(_prior(_LISTING))
    ports_.references.hold("timing", _timing_policy("timing/late", signal=time(9, 30)))
    audit = ports_.run(timing_policy_version=_v("timing/late"))
    _refused(audit, _S.NOT_ELIGIBLE, _R.AFTER_SIGNAL_CUTOFF, _R.EXCLUSIVITY_POLICY_UNRESOLVED)


# ----------------------------------------------------------------------
# U-W -- point-in-time open recommendations (EI4-D8)
# ----------------------------------------------------------------------


def _issued(
    key: str,
    listing: ListingReference,
    at: PointInTime,
    kind: EquityIssuanceKind = EquityIssuanceKind.MODELED,
    sequence: int = 1,
) -> EquityPriorSessionRecommendation:
    return EquityPriorSessionRecommendation(
        key, StrategyIdentity("ei4c-other"), listing, _DAY, kind, at, sequence
    )


@pytest.mark.parametrize("kind", list(EquityIssuanceKind))
def test_u_a_recommendation_issued_before_the_modeled_instant_is_open(kind) -> None:  # type: ignore[no-untyped-def]
    ports_ = _ports()
    ports_.state.answer = _history(_issued("prior-1", _LISTING, _at(9, 20), kind))
    audit = ports_.run()
    # Open under EI4-D8, but the deciding exclusivity rule is not approved:
    # unresolved and fail-closed, never reported as a breach.
    _refused(audit, _S.EVALUATION_UNRESOLVED, _R.EXCLUSIVITY_POLICY_UNRESOLVED)
    detail = _check(audit, _C.OPEN_RECOMMENDATIONS).detail
    assert "prior-1" in detail and "not a breach" in detail
    assert ports_.state.asked == [_at(9, 36)]
    assert audit.inputs is not None
    frozen = audit.inputs.issued_history
    assert [r.recommendation_key for r in frozen.recommendations] == ["prior-1"]
    assert (frozen.as_of, frozen.recording_watermark) == (_at(9, 36), 1)


def test_u_an_open_recommendation_on_another_listing_leaves_exposure_unresolved() -> None:
    ports_ = _ports()
    ports_.state.answer = _history(_issued("other-1", _OTHER, _at(9, 20)))
    audit = ports_.run()
    _refused(audit, _S.EVALUATION_UNRESOLVED, _R.EXPOSURE_POLICY_UNRESOLVED)
    assert _check(audit, _C.PORTFOLIO_EXPOSURE).status is _CS.NOT_ASSESSED
    assert _check(audit, _C.SAME_LISTING_EXCLUSIVITY).status is _CS.NOT_ASSESSED


def test_u_a_later_issuance_never_changes_an_earlier_assessment() -> None:
    # Counterexample: before the correction, a same-listing recommendation
    # issued at 10:00 refused the 09:36 assessment retroactively.
    ports_ = _ports()
    before = ports_.run()
    ports_.state.answer = _history(
        _issued("later-same", _LISTING, _at(10, 0), sequence=1),
        _issued("later-other", _OTHER, _at(10, 5), sequence=2),
    )
    after = ports_.run()
    assert after.status is _S.PROPOSED
    assert after.proposal == before.proposal
    assert after.inputs is not None and after.inputs.issued_history.recommendations == ()
    assert "lists no recommendation issued by" in _check(after, _C.OPEN_RECOMMENDATIONS).detail
    # Under a policy whose latency models issuance at 10:00, the 10:00 issuance is open.
    ports_.references.hold("timing", _timing_policy("timing/24m", latency=timedelta(minutes=24)))
    late = ports_.run(timing_policy_version=_v("timing/24m"))
    _refused(late, _S.EVALUATION_UNRESOLVED, _R.EXCLUSIVITY_POLICY_UNRESOLVED)


def test_u_a_transient_proposal_is_never_an_issued_recommendation() -> None:
    ports_ = _ports()
    first = ports_.run()
    assert first.proposal is not None
    # Generating records nothing: the authoritative record is unchanged, so a
    # repeat is not blocked by its own earlier proposal.
    assert ports_.state.answer == _history()
    assert ports_.run() == first
    with pytest.raises(InvalidEquityRecommendationReferenceError):
        EquitySessionRecommendationHistory(
            _NSE,
            _DAY,
            (first.proposal,),
            _at(15, 30),
            0,
            "test-history/1",  # type: ignore[arg-type]
        )


def test_v_an_established_empty_record_is_empty_but_none_is_missing() -> None:
    ports_ = _ports()
    audit = ports_.run()
    assert audit.status is _S.PROPOSED
    assert audit.inputs is not None
    frozen = audit.inputs.issued_history
    assert (frozen.reference, frozen.complete_through) == ("test-history/1", _at(15, 30))
    assert (frozen.recording_watermark, frozen.recommendations) == (0, ())
    ports_.state.answer = None
    missing = ports_.run()
    _refused(missing, _S.DEPENDENCY_UNAVAILABLE, _R.OPEN_RECOMMENDATION_STATE_UNAVAILABLE)
    assert "never assumed" in _check(missing, _C.SYNTHETIC_REFERENCES).detail


def test_w_an_incomplete_unreadable_or_wrong_session_record_fails_closed() -> None:
    ports_ = _ports()
    ports_.state.answer = _history(through=_at(9, 35, 59))
    audit = ports_.run()
    _refused(audit, _S.DEPENDENCY_UNAVAILABLE, _R.RECOMMENDATION_HISTORY_INCOMPLETE)
    assert "complete only through" in _check(audit, _C.SYNTHETIC_REFERENCES).detail
    ports_.state.answer = _history(through=_at(9, 36))
    assert ports_.run().status is _S.PROPOSED  # complete exactly through the instant
    ports_.state.answer = _history(day=date(2026, 8, 13), through=_at(15, 30))
    _refused(ports_.run(), _S.DEPENDENCY_UNAVAILABLE, _R.OPEN_RECOMMENDATION_STATE_UNAVAILABLE)
    ports_.state.unavailable = True
    _refused(ports_.run(), _S.DEPENDENCY_UNAVAILABLE, _R.OPEN_RECOMMENDATION_STATE_UNAVAILABLE)


def test_w_malformed_or_contradictory_records_cannot_be_constructed() -> None:
    with pytest.raises(InvalidEquityRecommendationReferenceError, match="another session"):
        _history(_issued("x", _LISTING, _at(9, 20)), day=date(2026, 8, 13))
    with pytest.raises(InvalidEquityRecommendationReferenceError, match="another session"):
        _history(_issued("x", ListingReference(Symbol("M&M"), ExchangeCode("BSE")), _at(9, 20)))
    with pytest.raises(InvalidEquityRecommendationReferenceError, match="after it is complete"):
        _history(_issued("x", _LISTING, _at(9, 40)), through=_at(9, 36))
    with pytest.raises(InvalidEquityRecommendationReferenceError, match="distinct"):
        _history(_issued("x", _LISTING, _at(9, 20)), _issued("x", _OTHER, _at(9, 21)))
    with pytest.raises(InvalidEquityRecommendationReferenceError, match="order"):
        _history(
            _issued("b", _LISTING, _at(9, 21), sequence=1),
            _issued("a", _OTHER, _at(9, 20), sequence=2),
        )
    with pytest.raises(InvalidEquityRecommendationReferenceError, match="not complete"):
        _history(through=_at(9, 30)).issued_by(_at(9, 36))
    # Recording provenance (EI4-D9-H): sequences are distinct and within the watermark.
    with pytest.raises(InvalidEquityRecommendationReferenceError, match="recording watermark"):
        _history(_issued("x", _LISTING, _at(9, 20), sequence=3), watermark=2)
    with pytest.raises(InvalidEquityRecommendationReferenceError, match="recording sequences"):
        _history(_issued("x", _LISTING, _at(9, 20)), _issued("y", _OTHER, _at(9, 21)))
    for watermark in (-1, True, "1"):
        with pytest.raises(InvalidEquityRecommendationReferenceError, match="watermark"):
            _history(watermark=watermark)  # type: ignore[arg-type]
    with pytest.raises(InvalidEquityRecommendationReferenceError, match="recording_sequence"):
        _issued("x", _LISTING, _at(9, 20), sequence=0)


def test_w_records_at_one_instant_have_one_deterministic_order() -> None:
    a = _issued("a-key", _OTHER, _at(9, 36), sequence=2)
    b = _issued("b-key", _LISTING, _at(9, 36), sequence=1)
    with pytest.raises(InvalidEquityRecommendationReferenceError, match="order"):
        _history(b, a)
    record = _history(a, b)
    # Issued exactly at the modeled instant: conservatively open.
    assert record.issued_by(_at(9, 36)) == (a, b)
    assert record.issued_by(_at(9, 35, 59)) == ()
    ports_ = _ports()
    ports_.state.answer = record
    audit = ports_.run()
    _refused(
        audit,
        _S.EVALUATION_UNRESOLVED,
        _R.EXCLUSIVITY_POLICY_UNRESOLVED,
        _R.EXPOSURE_POLICY_UNRESOLVED,
    )
    assert audit.inputs is not None and audit.inputs.issued_history.recommendations == (a, b)
    # Repeated evaluation gives the same canonical result.
    assert ports_.run().canonical_json() == audit.canonical_json()


def test_v_unassessed_controls_block_actionability_even_with_nothing_open() -> None:
    audit = _ports().run()
    assert audit.status is _S.PROPOSED
    assert "no recommendation issued by" in _check(audit, _C.OPEN_RECOMMENDATIONS).detail
    for check in (_C.SAME_LISTING_EXCLUSIVITY, _C.PORTFOLIO_EXPOSURE):
        assert _check(audit, check).status is _CS.NOT_ASSESSED
    assert "section 9.5" in _check(audit, _C.SAME_LISTING_EXCLUSIVITY).detail
    assert audit.proposal is not None and not audit.proposal.is_actionable


_UNASSESSED = (
    _C.SAME_LISTING_EXCLUSIVITY,
    _C.PORTFOLIO_EXPOSURE,
    _C.DAILY_LOSS_LIMIT,
    _C.PRE_ENTRY_INVALIDATION,
    _C.CORPORATE_ACTIONS,
    _C.EXCHANGE_SURVEILLANCE,
    _C.CANDLE_REVISION_MATERIALITY,
)


def test_x_the_daily_loss_limit_and_other_missing_controls_are_never_satisfied() -> None:
    audit = _ports().run()
    assert audit.proposal is not None
    for check in _UNASSESSED:
        assert _check(audit, check).status is _CS.NOT_ASSESSED
    assert "EI0-D19" in _check(audit, _C.DAILY_LOSS_LIMIT).detail
    assert audit.proposal.not_assessed == _UNASSESSED
    assert not audit.proposal.is_actionable
    passed = {r.check for r in audit.checks if r.status is _CS.PASSED}
    assert passed == {
        _C.AUTHORITATIVE_CANDIDATE,
        _C.SYNTHETIC_REFERENCES,
        _C.SESSION_TIMING,
        _C.LISTING_ELIGIBILITY,
        _C.OPEN_RECOMMENDATIONS,
        _C.CORE_FORMATION,
    }
    with pytest.raises(InvalidEquityRecommendationEligibilityError, match="not assessed"):
        dataclasses.replace(audit.proposal, not_assessed=_UNASSESSED[:-1])


# ----------------------------------------------------------------------
# Y-AD -- identity, audit, idempotency, corrections and issuance
# ----------------------------------------------------------------------


def _length_prefixed_sha256(*parts: str) -> str:
    encoded = b"".join(str(len(p.encode())).encode() + b":" + p.encode() for p in parts)
    return hashlib.sha256(encoded).hexdigest()


def test_y_the_generation_key_is_deterministic_and_canonical() -> None:
    ports_ = _ports()
    first, second = ports_.run(), _Ports(ports_.world).run()
    assert first.proposal is not None and second.proposal is not None and first.inputs
    key = first.proposal.generation_key
    assert key == second.proposal.generation_key == first.generation_key
    candidate = ports_.world.timelines.only().entries[1]
    policy_set = first.inputs.policy_set
    assert key == EQUITY_RECOMMENDATION_GENERATION_KEY_PREFIX + _length_prefixed_sha256(
        EQUITY_RECOMMENDATION_GENERATION_KEY_NAMESPACE,
        "ei4c-orb-vwap-volume",
        "synthetic-ei4c-research/1",
        "M&M",
        "NSE",
        "2026-08-12",
        candidate.entry_fingerprint,
        "synthetic-ei4c-set/1",
        policy_set.fingerprint,
    )
    assert policy_set.fingerprint.startswith("eirs1-sha256:")
    with pytest.raises(InvalidEquityRecommendationEligibilityError, match="eirk1"):
        dataclasses.replace(first.proposal, generation_key="0" * 64)


def test_z_policy_set_components_are_identity_eligibility_and_history_are_content() -> None:
    ports_ = _ports()
    ports_.references.hold("generation", _generation("generation/2"))
    ports_.references.hold("risk", _risk("risk/2", capital="50000"))
    base = ports_.run()
    other = ports_.run(generation_policy_version=_v("generation/2"))
    risk = ports_.run(risk_policy_version=_v("risk/2"))
    assert base.proposal and other.proposal and risk.proposal
    # EI4-D9-B: a different policy-set component is a different identity.
    assert len({base.generation_key, other.generation_key, risk.generation_key}) == 3
    assert other.proposal.definition.levels == base.proposal.definition.levels
    # Eligibility evidence is content: the same key, different content.
    eligibility = ports_.run(listing_eligibility_version=_v("eligibility/2"))
    assert eligibility.generation_key == base.generation_key
    assert eligibility.assessment_fingerprint != base.assessment_fingerprint
    assert eligibility.proposal is not None
    assert eligibility.proposal.fingerprint != base.proposal.fingerprint
    # History evidence is content too: an other-listing record issued later
    # leaves the as-of view empty but changes what was observed.
    ports_.state.answer = _history(_issued("later", _OTHER, _at(10, 0)))
    history = ports_.run()
    assert history.status is _S.PROPOSED
    assert history.generation_key == base.generation_key
    assert history.assessment_fingerprint != base.assessment_fingerprint
    assert history.proposal == base.proposal


def test_z_the_audit_records_every_applied_input_in_full() -> None:
    audit = _ports().run()
    inputs = audit.assessment_document()["inputs"]
    assert inputs["format"] == "northstar.equity-recommendation-generation-inputs/1"
    policy_set = inputs["policy_set"]
    assert policy_set["cost_schedule"] == _costs().document()
    assert policy_set["risk_policy"] == _risk().document()
    assert policy_set["timing_policy"] == _timing_policy().document()
    assert policy_set["price_increment"] == _increment().document()
    assert policy_set["generation_policy"] == _generation().document()
    assert policy_set["reference"] == _policy_set().document()
    assert inputs["listing_eligibility"] == _eligibility().document()
    assert inputs["venue_day"]["reference"] == "test-venue/1"
    assert inputs["issued_history"]["reference"] == "test-history/1"
    assert inputs["modeled_issued_at"] == _at(9, 36).value
    assert audit.inputs is not None
    expected = {k: v for k, v in _policy_set().document().items() if k != "version"}
    expected["listing_eligibility_version"] = _v("eligibility/1").value
    assert {name: v.value for name, v in audit.inputs.versions().items()} == expected


def test_aa_the_eligibility_audit_document_is_canonical_and_deterministic() -> None:
    ports_ = _ports()
    audit = ports_.run()
    text = audit.canonical_json()
    assert text == canonical_equity_candle_json(audit.document())
    assert text == _Ports(ports_.world).run().canonical_json()

    def no_floats(value: str) -> None:
        raise AssertionError(f"float {value}")

    document = json.loads(text, parse_float=no_floats)
    assert document["format"] == "northstar.equity-recommendation-eligibility-audit/2"
    assert document["assessment_fingerprint"] == audit.assessment_fingerprint
    assessment = document["assessment"]
    assert assessment["format"] == "northstar.equity-recommendation-assessment/1"
    assert (assessment["synthetic"], assessment["authoritative"]) == (True, False)
    assert assessment["status"] == "PROPOSED" and assessment["reasons"] == []
    assert [c["check"] for c in assessment["checks"]] == [c.value for c in _C]
    assert all(set(c) == {"check", "status", "reasons"} for c in assessment["checks"])
    assert assessment["proposal"]["issued"] is False
    assert assessment["proposal"]["actionable"] is False
    assert assessment["entry_window"]["timing"]["issuance_kind"] == "MODELED"
    assert assessment["candidate"]["source"]["candidate_entry_fingerprint"].startswith("eirt1-")
    assert assessment["request"]["research_configuration"] == _CONFIG.document()
    assert set(document["diagnostics"]["details"]) == {c.value for c in _C}
    refused = ports_.run(timing_policy_version=_v("timing/missing"))
    refused_document = json.loads(refused.canonical_json())["assessment"]
    assert refused_document["status"] == "DEPENDENCY_UNAVAILABLE"
    assert refused_document["reasons"] == ["TIMING_POLICY_UNAVAILABLE"]
    assert refused_document["proposal"] is None and refused_document["generation_key"] is None


def test_ab_identical_inputs_give_identical_outputs() -> None:
    first, second = _ports(), _ports()
    a, b = first.run(), second.run()
    assert a == b
    assert a.document() == b.document()
    assert first.run() == a
    refusals = [_ports(), _ports()]
    for ports_ in refusals:
        ports_.state.answer = None
    assert refusals[0].run() == refusals[1].run()


def test_ac_a_later_correction_never_rewrites_the_candidate_or_its_proposal() -> None:
    ports_ = _ports()
    before = ports_.run()
    # A corrected revision of the signal bar arrives after the candidate.
    corrected = ("101", "103", "101", "101.8", "2000")
    ports_.world.add({4: corrected}, revision="r2")
    ports_.world.commit(_ist(9, 46))
    entries = ports_.world.timelines.only().entries
    assert entries[-1].role is _ROLE.REJECTED
    assert entries[1].role is _ROLE.CANDIDATE
    after = ports_.run()
    assert after.proposal == before.proposal
    assert after.candidate is not None and before.candidate is not None
    assert after.candidate.source == before.candidate.source
    # The frozen assessment is unchanged; only the diagnostic head moved, and
    # the candidate is never relinked to the corrected snapshot.
    assert after.assessment_fingerprint == before.assessment_fingerprint
    assert after.observed_timeline_head != before.observed_timeline_head
    assert after.candidate.source.evidence_snapshot_id == entries[1].snapshot_id
    assert entries[-1].snapshot_id != entries[1].snapshot_id


def test_ad_a_proposal_is_modeled_synthetic_and_never_an_issuance() -> None:
    audit = _ports().run()
    proposal = audit.proposal
    assert proposal is not None
    assert proposal.entry_window.timing.issuance_kind is EquityIssuanceKind.MODELED
    assert not proposal.entry_window.timing.is_actual_issuance
    assert proposal.is_synthetic
    assert not (proposal.is_issued or proposal.is_authoritative or proposal.is_actionable)
    assert not audit.is_authoritative
    actual = dataclasses.replace(
        proposal.entry_window.timing, issuance_kind=EquityIssuanceKind.ACTUAL
    )
    with pytest.raises(InvalidEquityRecommendationEligibilityError, match="never records"):
        dataclasses.replace(proposal.entry_window, timing=actual)


# ----------------------------------------------------------------------
# Contract validation
# ----------------------------------------------------------------------


def test_requests_policies_and_audits_are_validated() -> None:
    with pytest.raises(InvalidEquityTradeRecommendationRequestError):
        _request(listing_eligibility_version="synthetic-ei4c-eligibility/1")
    with pytest.raises(InvalidEquityTradeRecommendationRequestError):
        _request(policy_set=_policy_set().document())
    with pytest.raises(InvalidEquityRecommendationEvidenceError, match="synthetic"):
        _request(risk_policy_version="synthetic-ei4c-risk/1")
    with pytest.raises(InvalidEquityTradeRecommendationRequestError):
        _request(trading_date=datetime(2026, 8, 12, tzinfo=UTC))
    with pytest.raises(InvalidEquityRecommendationReferenceError, match="five-minute"):
        _timing_policy(signal=time(10, 32))
    with pytest.raises(InvalidEquityRecommendationReferenceError, match="before the modeled exit"):
        _timing_policy(entry=time(14, 0))
    with pytest.raises(InvalidEquityRecommendationReferenceError):
        _timing_policy(remaining=0)
    with pytest.raises(TypeError):
        _ports().use_case().execute("request")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        GenerateEquityTradeRecommendationUseCase(None, None, None, None, None, None)  # type: ignore[arg-type]
    audit = _ports().run()
    with pytest.raises(InvalidEquityRecommendationEligibilityError):
        dataclasses.replace(audit, checks=audit.checks[1:])
    with pytest.raises(InvalidEquityRecommendationEligibilityError):
        dataclasses.replace(audit, proposal=None)
    assert audit.proposal is not None
    legacy = equity_trade_recommendation_proposal_key(
        audit.proposal.definition.source, _v("generation/1")
    )
    # The provisional v1 key keeps its meaning and never collides with eirk1 keys.
    assert len(legacy) == 64 and not legacy.startswith(EQUITY_RECOMMENDATION_GENERATION_KEY_PREFIX)
    assert legacy != audit.proposal.generation_key
    assert legacy == _length_prefixed_sha256(
        EQUITY_TRADE_RECOMMENDATION_PROPOSAL_NAMESPACE,
        "ei4c-orb-vwap-volume",
        "synthetic-ei4c-research/1",
        "M&M",
        "NSE",
        "2026-08-12",
        audit.proposal.definition.source.candidate_entry_fingerprint,
        "synthetic-ei4c-generation/1",
    )


# ----------------------------------------------------------------------
# AE-AF -- architecture
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    [generate_module, eligibility_module, evidence_module, records_module, references_module],
)
def test_ae_af_the_modules_stay_application_only_and_model_no_execution(module) -> None:  # type: ignore[no-untyped-def]
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
        "sleep",
        "random",
        "uuid4",
        "connect",
        "cursor",
        "commit",
        "rollback",
        "float",
        "hash",
        "open",
        "getenv",
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
        assert name.split(".")[0] not in {"time", "random", "uuid", "secrets", "threading", "os"}
        for forbidden in (
            "northstar_infrastructure",
            "sqlite",
            "socket",
            "http",
            "urllib",
            "requests",
            "futures",
            "paper",
            "broker",
            "derivatives",
            "options",
        ):
            assert forbidden not in name, name
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
        # "disposition" is the approved generation-record term (EI4-D9-A), not a position.
        name = identifier.replace("disposition", "")
        for forbidden in ("fill", "broker", "paper", "pnl", "profit", "position", "lifecycle"):
            assert forbidden not in name, identifier


def test_the_contracts_are_exported() -> None:
    for name in (
        "GenerateEquityTradeRecommendationUseCase",
        "EquityTradeRecommendationRequest",
        "EquityTradeRecommendationProposal",
        "EquityRecommendationEligibilityAudit",
        "EquityRecommendationEligibilityReason",
        "EquityRecommendationGenerationStatus",
        "EquityRecommendationCheck",
        "EquityRecommendationCheckStatus",
        "EquityModeledEntryWindow",
        "equity_trade_recommendation_proposal_key",
    ):
        assert name in application_services.__all__
    for name in (
        "EquitySyntheticRecommendationReferenceSource",
        "EquitySyntheticListingEligibilitySource",
        "EquitySessionRecommendationHistorySource",
        "EquitySyntheticEntryTimingPolicy",
        "EquitySyntheticListingEligibility",
        "EquitySessionRecommendationHistory",
        "EquitySyntheticReferenceUnavailableError",
        "EquityRecommendationHistoryUnavailableError",
    ):
        assert name in ports.__all__


# ======================================================================
# EI-4C compatibility with EI4-D9 and ADR-021 (owner approval 2026-11-07)
# ======================================================================


def _proposed() -> tuple[_Ports, EquityRecommendationEligibilityAudit]:
    ports_ = _ports()
    audit = ports_.run()
    assert audit.status is _S.PROPOSED and audit.inputs is not None
    return ports_, audit


# ADR-025 version 2: t = 09:36 IST, so b = t - 1 us. A version-2 PROPOSED needs the
# issued history declared complete exactly through b (N10, case A).
_V2 = "northstar.equity-recommendation-evaluator/2"
_B = PointInTime("2026-08-12T04:05:59.999999Z")


def _proposed_v2(
    ports_: _Ports | None = None,
) -> tuple[_Ports, EquityRecommendationEligibilityAudit]:
    """The default scenario under evaluator version 2, history complete exactly through b."""
    ports_ = ports_ or _ports()
    ports_.evaluator_version = _V2
    ports_.state.answer = _history(through=_B)
    audit = ports_.run()
    assert audit.status is _S.PROPOSED and audit.inputs is not None, audit.reasons
    return ports_, audit


# -- A-D: composite policy-set identity ---------------------------------


def test_compat_a_the_resolved_policy_set_binds_every_component_version() -> None:
    _, audit = _proposed()
    assert audit.inputs is not None
    policy_set = audit.inputs.policy_set
    assert policy_set.reference == _policy_set()
    assert policy_set.document()["format"] == "northstar.equity-recommendation-policy-set/1"
    assert policy_set.fingerprint == equity_recommendation_fingerprint(policy_set.document())
    assert policy_set.timing_policy.modeled_issuance_latency == timedelta(0)
    with pytest.raises(InvalidEquityRecommendationEvidenceError, match="version the policy set"):
        dataclasses.replace(policy_set, risk_policy=_risk("risk/other"))


def test_compat_b_identity_ignores_key_order_and_rebuilt_equal_documents() -> None:
    document = {"format": "northstar.equity-recommendation-policy-set/1", "a": "1", "b": ["2"]}
    reordered = {"b": ["2"], "a": "1", "format": "northstar.equity-recommendation-policy-set/1"}
    assert equity_recommendation_fingerprint(document) == equity_recommendation_fingerprint(
        reordered
    )
    _, audit = _proposed()
    assert audit.inputs is not None
    rebuilt = EquityResolvedSyntheticPolicySet(
        _policy_set(), _generation(), _risk(), _costs(), _increment(), _timing_policy()
    )
    assert rebuilt.fingerprint == audit.inputs.policy_set.fingerprint
    source = audit.inputs.candidate.source
    assert equity_recommendation_generation_key(source, rebuilt) == audit.generation_key


@pytest.mark.parametrize(
    ("kind", "value", "field"),
    [
        ("generation", _generation("generation/c"), "generation_policy_version"),
        ("risk", _risk("risk/c", capital="90000"), "risk_policy_version"),
        ("costs", _costs("costs/c", entry="0.0004"), "cost_schedule_version"),
        (
            "increment",
            EquityPriceIncrement(Decimal("0.1"), _v("increment/c")),
            "price_increment_version",
        ),
        (
            "timing",
            _timing_policy("timing/c", latency=timedelta(seconds=30)),
            "timing_policy_version",
        ),
    ],
)
def test_compat_c_every_policy_set_component_changes_the_identity(kind, value, field) -> None:  # type: ignore[no-untyped-def]
    ports_, base = _proposed()
    ports_.references.hold(kind, value)
    changed = ports_.run(**{field: value.version})
    assert changed.generation_key is not None
    assert changed.generation_key != base.generation_key
    renamed = ports_.run(policy_set_version="set/renamed")
    assert renamed.generation_key not in {base.generation_key, changed.generation_key}


def test_compat_c_same_set_version_with_different_content_is_a_policy_reference_conflict() -> None:
    # Before EI4-D10-S3 a reused set and component label with other content
    # produced a second generation key (EI-4E AA-1). The catalog now publishes
    # costs/1 as its first content, so other content under the label is a
    # permanent conflict: no key, no inputs, nothing to record but the attempt.
    ports_, base = _proposed()
    ports_.references.answers["costs"] = dataclasses.replace(
        _costs(), slippage_rate_per_side=Decimal("0.0003")
    )
    other = ports_.run()
    _refused(other, _S.DEPENDENCY_UNAVAILABLE, _R.POLICY_REFERENCE_CONFLICT)
    assert other.generation_key is None and other.inputs is None
    assert base.generation_key is not None


def test_compat_d_the_v1_proposal_key_is_kept_and_never_the_generation_key() -> None:
    _, audit = _proposed()
    assert audit.proposal is not None and audit.inputs is not None
    v1 = equity_trade_recommendation_proposal_key(audit.inputs.candidate.source, _v("generation/1"))
    assert audit.proposal.generation_key.startswith("eirk1-sha256:")
    assert not v1.startswith("eirk1-") and v1 not in audit.proposal.generation_key
    assert EQUITY_TRADE_RECOMMENDATION_PROPOSAL_NAMESPACE.endswith(".v1")
    assert (
        EQUITY_RECOMMENDATION_GENERATION_KEY_NAMESPACE
        != EQUITY_TRADE_RECOMMENDATION_PROPOSAL_NAMESPACE
    )


# -- E-J: canonical documents and fingerprints --------------------------


def test_compat_e_f_g_assessment_and_proposal_documents_verify() -> None:
    _, audit = _proposed()
    assert audit.proposal is not None
    assessment = canonical_equity_recommendation_json(audit.assessment_document())
    assert audit.assessment_fingerprint.startswith("eira1-sha256:")
    decoded = verify_equity_recommendation_document(assessment, audit.assessment_fingerprint)
    assert decoded["format"] == "northstar.equity-recommendation-assessment/1"
    proposal = canonical_equity_recommendation_json(audit.proposal.document())
    assert audit.proposal.fingerprint.startswith("eirp1-sha256:")
    decoded = verify_equity_recommendation_document(proposal, audit.proposal.fingerprint)
    assert decoded["format"] == "northstar.equity-recommendation-proposal/1"
    assert decoded["definition"]["format"] == "northstar.equity-recommendation-core-outcome/1"
    assert decoded["definition"]["outcome"] == audit.proposal.definition.document()
    assert audit.inputs is not None and audit.inputs.fingerprint.startswith("eiri1-sha256:")
    # Tampering, re-encoding or a float never verifies.
    with pytest.raises(InvalidEquityRecommendationDocumentError, match="fingerprint"):
        verify_equity_recommendation_document(
            assessment.replace('"PROPOSED"', '"NOT_ELIGIBLE"', 1), audit.assessment_fingerprint
        )
    with pytest.raises(InvalidEquityRecommendationDocumentError, match="canonical"):
        verify_equity_recommendation_document(
            json.dumps(json.loads(assessment), indent=1), audit.assessment_fingerprint
        )
    with pytest.raises(InvalidEquityRecommendationDocumentError, match="float"):
        verify_equity_recommendation_document(
            '{"format":"x","v":1.5}', audit.assessment_fingerprint
        )
    with pytest.raises(InvalidEquityRecommendationDocumentError, match="float"):
        equity_recommendation_fingerprint(
            {"format": "northstar.equity-recommendation-assessment/1", "v": 1.5}
        )


def test_compat_h_diagnostic_prose_never_moves_a_fingerprint() -> None:
    _, audit = _proposed()
    reworded = dataclasses.replace(
        audit,
        checks=tuple(dataclasses.replace(c, detail="reworded diagnostics") for c in audit.checks),
        observed_timeline_head="eirt1-sha256:" + "f" * 64,
    )
    assert reworded.assessment_fingerprint == audit.assessment_fingerprint
    assert reworded.document() != audit.document()
    # Two outages that differ only in their exception messages are one logical attempt.
    first, second = _ports(), _ports()
    first.references.unavailable = {"risk"}
    second.references.unavailable = {"risk"}
    a, b = first.run(), second.run()
    assert a.status is _S.DEPENDENCY_UNAVAILABLE
    second_detail = dataclasses.replace(
        b,
        checks=tuple(dataclasses.replace(c, detail="another exception message") for c in b.checks),
    )
    assert (
        equity_recommendation_attempt(a).attempt_identity
        == equity_recommendation_attempt(second_detail).attempt_identity
    )


def test_compat_i_stable_reason_codes_are_identity_bearing() -> None:
    ports_ = _ports()
    ports_.state.answer = _history(_prior(_LISTING))
    audit = ports_.run()
    document = audit.assessment_document()
    assert document["status"] == "EVALUATION_UNRESOLVED"
    assert document["reasons"] == ["EXCLUSIVITY_POLICY_UNRESOLVED"]
    open_check = next(c for c in document["checks"] if c["check"] == "OPEN_RECOMMENDATIONS")
    assert open_check == {
        "check": "OPEN_RECOMMENDATIONS",
        "status": "FAILED",
        "reasons": ["EXCLUSIVITY_POLICY_UNRESOLVED"],
    }
    attempt = json.loads(equity_recommendation_attempt(audit).attempt_document)
    assert attempt["reasons"] == ["EXCLUSIVITY_POLICY_UNRESOLVED"]


@pytest.mark.parametrize(
    "document",
    [
        {"status": "PROPOSED"},
        # /9: an unapproved version (ADR-025 approves assessment/2).
        {"format": "northstar.equity-recommendation-assessment/9", "status": "PROPOSED"},
        {"format": "northstar.equity-recommendation-eligibility-audit/2"},
    ],
)
def test_compat_j_a_missing_or_unsupported_format_is_refused(document) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(InvalidEquityRecommendationDocumentError, match="format"):
        equity_recommendation_fingerprint(document)
    text = canonical_equity_recommendation_json(document)
    with pytest.raises(InvalidEquityRecommendationDocumentError, match="format"):
        verify_equity_recommendation_document(text, "eira1-sha256:" + "0" * 64)


# -- K-P: derived modeled issuance --------------------------------------


def test_compat_k_n_modeled_issuance_is_derived_exactly_from_the_latency() -> None:
    assert equity_modeled_issuance(_at(9, 36), timedelta(0)) == _at(9, 36)
    exact = equity_modeled_issuance(_at(9, 36), timedelta(seconds=2, microseconds=123456))
    assert exact == PointInTime("2026-08-12T04:06:02.123456Z")
    assert equity_modeled_issuance(exact, timedelta(microseconds=876544)) == PointInTime(
        "2026-08-12T04:06:03Z"
    )
    ports_ = _ports()
    ports_.references.hold("timing", _timing_policy("timing/us", latency=timedelta(microseconds=1)))
    audit = ports_.run(timing_policy_version=_v("timing/us"))
    assert audit.inputs is not None
    assert audit.inputs.modeled_issued_at == PointInTime("2026-08-12T04:06:00.000001Z")
    assert ports_.state.asked == [audit.inputs.modeled_issued_at]
    window = _window(audit)
    assert window.timing.issuance_kind is EquityIssuanceKind.MODELED
    assert window.timing.issued_at == audit.inputs.modeled_issued_at
    assert window.timing.eligible_from == _at(9, 40)
    assert (
        audit.inputs.policy_set.timing_policy.document()["modeled_issuance_latency_microseconds"]
        == 1
    )


def test_compat_l_the_latency_is_a_non_negative_versioned_policy_value() -> None:
    with pytest.raises(InvalidEquityRecommendationReferenceError, match="latency"):
        _timing_policy(latency=timedelta(microseconds=-1))
    with pytest.raises(InvalidEquityRecommendationReferenceError, match="latency"):
        _timing_policy(latency=60)  # type: ignore[arg-type]
    with pytest.raises(InvalidEquityRecommendationEvidenceError, match="non-negative"):
        equity_modeled_issuance(_at(9, 36), timedelta(seconds=-1))


def test_compat_m_a_caller_can_never_supply_or_override_modeled_issuance() -> None:
    with pytest.raises(TypeError):
        _request(modeled_issued_at=_at(9, 41))
    assert "modeled_issued_at" not in _request().document()
    _, audit = _proposed()
    assert audit.inputs is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        audit.inputs.modeled_issued_at = _at(9, 41)  # type: ignore[misc]
    frozen = audit.inputs.issued_history
    with pytest.raises(InvalidEquityRecommendationEvidenceError, match="derived modeled issuance"):
        dataclasses.replace(
            audit.inputs, issued_history=dataclasses.replace(frozen, as_of=_at(9, 41))
        )


def test_compat_o_the_instants_stay_in_order() -> None:
    ports_ = _ports()
    ports_.references.hold("timing", _timing_policy("timing/90s", latency=timedelta(seconds=90)))
    timing = _window(ports_.run(timing_policy_version=_v("timing/90s"))).timing
    assert timing.decision_as_of.compare(timing.issued_at) < 0
    assert timing.issued_at.compare(timing.eligible_from) < 0
    assert (timing.issued_at, timing.eligible_from) == (_at(9, 37, 30), _at(9, 40))


def test_compat_p_late_modeled_issuance_leaves_no_permissible_entry() -> None:
    ports_ = _ports()
    # Issuance modeled at 15:36 needs history complete through 15:36; the
    # default fixture is complete only through 15:30 and fails closed.
    ports_.references.hold("timing", _timing_policy("timing/6h", latency=timedelta(hours=6)))
    incomplete = ports_.run(timing_policy_version=_v("timing/6h"))
    _refused(incomplete, _S.DEPENDENCY_UNAVAILABLE, _R.RECOMMENDATION_HISTORY_INCOMPLETE)
    ports_.state.answer = _history(through=_at(18, 0))
    for name, latency in (
        ("timing/late-entry", timedelta(hours=1, minutes=10)),
        ("timing/after-window", timedelta(hours=6)),
    ):
        ports_.references.hold("timing", _timing_policy(name, latency=latency))
        audit = ports_.run(timing_policy_version=_v(name))
        _refused(audit, _S.NOT_ELIGIBLE, _R.NO_PERMISSIBLE_ENTRY_BAR)
        assert audit.entry_window is None and audit.generation_key is not None


# -- Q-S: venue-day evidence --------------------------------------------


def test_compat_q_the_venue_day_answer_is_frozen_without_its_prose() -> None:
    ports_, audit = _proposed()
    assert audit.inputs is not None
    venue = audit.inputs.venue_day
    reference = ports_.world.venue.venue_day(_NSE, _DAY)
    assert venue == EquityVenueDayEvidence.from_reference(reference)
    assert venue.document() == {
        "exchange_code": "NSE",
        "trading_date": "2026-08-12",
        "classification": "NORMAL_SESSION",
        "common_opens_at": _at(9, 15).value,
        "common_closes_at": _at(15, 30).value,
        "utc_offset_minutes": 330,
        "reference": "test-venue/1",
    }
    assert EquityVenueDayEvidence.from_document(venue.document()) == venue
    window = audit.inputs.candidate.document()["research_window"]
    assert window == {
        "policy_version": NSE_EQUITY_INTRADAY_V1_RESEARCH_WINDOW.version,
        "opens_at": _at(9, 15).value,
        "closes_at": _at(15, 0).value,
        "timeframe": "5m",
    }


def test_compat_r_replay_uses_the_frozen_venue_answer_not_todays() -> None:
    ports_, original = _proposed()
    assert original.inputs is not None
    ports_.world.venue.closes = time(15, 15)  # the reference later answers differently
    today = ports_.run()
    assert today.generation_key == original.generation_key
    assert today.assessment_fingerprint != original.assessment_fingerprint
    replayed = assess_equity_recommendation_inputs(_request(), original.inputs)
    assert replayed.assessment_fingerprint == original.assessment_fingerprint
    assert replayed.inputs is not None
    assert replayed.inputs.venue_day.common_closes_at == _at(15, 30)


def test_compat_s_missing_or_inconsistent_venue_attributes_fail_closed() -> None:
    _, audit = _proposed()
    assert audit.inputs is not None
    document = audit.inputs.venue_day.document()
    for key in document:
        partial = {k: v for k, v in document.items() if k != key}
        with pytest.raises(InvalidEquityRecommendationEvidenceError):
            EquityVenueDayEvidence.from_document(partial)
    with pytest.raises(InvalidEquityRecommendationEvidenceError, match="UTC offset"):
        EquityVenueDayEvidence.from_document({**document, "utc_offset_minutes": None})
    with pytest.raises(InvalidEquityRecommendationEvidenceError, match="boundaries"):
        EquityVenueDayEvidence.from_document({**document, "common_closes_at": None})
    special = {
        **document,
        "classification": "SPECIAL_SESSION",
        "common_opens_at": None,
        "common_closes_at": None,
        "utc_offset_minutes": None,
    }
    assert EquityVenueDayEvidence.from_document(special).classification is (
        EquityVenueDayClassification.SPECIAL_SESSION
    )
    with pytest.raises(InvalidEquityRecommendationEvidenceError, match="no boundaries"):
        EquityVenueDayEvidence.from_document({**special, "utc_offset_minutes": 330})
    with pytest.raises(InvalidEquityRecommendationEvidenceError, match="snapshot used"):
        dataclasses.replace(
            audit.inputs,
            venue_day=dataclasses.replace(audit.inputs.venue_day, reference="test-venue/2"),
        )


# -- T-W: point-in-time issued history ----------------------------------


def test_compat_t_u_the_frozen_view_keeps_completeness_and_recording_watermark() -> None:
    record = _history(
        _issued("open-1", _OTHER, _at(9, 30), sequence=4),
        _issued("later-1", _LISTING, _at(10, 0), sequence=2),
        through=_at(15, 30),
        watermark=7,
    )
    view = EquityIssuedHistoryEvidence.as_of_view(record, _at(9, 36))
    assert [r.recommendation_key for r in view.recommendations] == ["open-1"]
    assert (view.as_of, view.complete_through, view.recording_watermark) == (
        _at(9, 36),
        _at(15, 30),
        7,
    )
    assert view.document()["recommendations"][0]["recording_sequence"] == 4
    ports_, base = _proposed()
    ports_.state.answer = _history(watermark=5)
    rewatermarked = ports_.run()
    assert rewatermarked.generation_key == base.generation_key
    assert rewatermarked.assessment_fingerprint != base.assessment_fingerprint
    assert rewatermarked.inputs is not None
    assert rewatermarked.inputs.issued_history.recording_watermark == 5


def test_compat_v_contradictory_frozen_history_cannot_be_built() -> None:
    record = _history(_issued("open-1", _OTHER, _at(9, 30)))
    view = EquityIssuedHistoryEvidence.as_of_view(record, _at(9, 36))
    with pytest.raises(InvalidEquityRecommendationEvidenceError, match="complete through"):
        dataclasses.replace(view, complete_through=_at(9, 35))
    with pytest.raises(InvalidEquityRecommendationEvidenceError, match="after its as-of"):
        dataclasses.replace(view, as_of=_at(9, 25), complete_through=_at(9, 25))
    with pytest.raises(InvalidEquityRecommendationEvidenceError, match="watermark"):
        dataclasses.replace(view, recording_watermark=0)
    with pytest.raises(InvalidEquityRecommendationEvidenceError, match="not complete"):
        EquityIssuedHistoryEvidence.as_of_view(_history(through=_at(9, 30)), _at(9, 36))


def test_compat_w_no_application_source_fabricates_an_issued_history() -> None:
    import northstar_application
    from northstar_application.application_services import (
        AuthoritativeEquitySessionRecommendationHistorySource as Authoritative,
    )

    sources = {
        obj
        for module_name, module in list(__import__("sys").modules.items())
        if module_name.startswith("northstar_application")
        for obj in vars(module).values()
        if isinstance(obj, type)
        and issubclass(obj, EquitySessionRecommendationHistorySource)
        and obj is not EquitySessionRecommendationHistorySource
    }
    # EI-5D (ADR-024, sections 6 and 12) adds exactly one implementation, over the
    # issuance authority's reader only: it never fabricates or assumes an empty history.
    assert sources <= {Authoritative}, northstar_application.__name__
    assert list(inspect.signature(Authoritative).parameters) == ["reader"]
    ports_ = _ports()
    ports_.state.answer = None
    _refused(ports_.run(), _S.DEPENDENCY_UNAVAILABLE, _R.OPEN_RECOMMENDATION_STATE_UNAVAILABLE)


# -- X-Z: replay and correction stability ------------------------------


def test_compat_x_replaying_frozen_inputs_reproduces_the_assessment() -> None:
    ports_, original = _proposed()
    assert original.inputs is not None
    replayed = assess_equity_recommendation_inputs(_request(), original.inputs)
    assert replayed.assessment_document() == original.assessment_document()
    assert replayed.assessment_fingerprint == original.assessment_fingerprint
    assert replayed.proposal == original.proposal
    # The sources can change or disappear: the frozen inputs need none of them.
    ports_.references.unavailable = {"risk", "costs", "timing", "generation", "increment"}
    ports_.state.unavailable = True
    assert ports_.run().status is _S.DEPENDENCY_UNAVAILABLE
    again = assess_equity_recommendation_inputs(_request(), original.inputs)
    assert again.assessment_fingerprint == original.assessment_fingerprint
    with pytest.raises(InvalidEquityRecommendationEligibilityError, match="policy set"):
        assess_equity_recommendation_inputs(_request(policy_set_version="set/2"), original.inputs)


def test_compat_y_a_later_correction_never_changes_the_frozen_candidate() -> None:
    ports_, before = _proposed()
    ports_.world.add({4: ("101", "103", "101", "101.8", "2000")}, revision="r2")
    ports_.world.commit(_ist(9, 46), _ist(9, 51))
    after = ports_.run()
    assert after.inputs is not None and before.inputs is not None
    assert after.inputs.candidate == before.inputs.candidate
    assert after.inputs.fingerprint == before.inputs.fingerprint
    assert after.assessment_fingerprint == before.assessment_fingerprint
    assert after.generation_key == before.generation_key
    assert after.observed_timeline_head != before.observed_timeline_head


# -- AA-AB: dispositions and attempts ----------------------------------


def test_compat_aa_dispositions_map_from_the_three_assessed_outcomes() -> None:
    _, proposed = _proposed()
    disposition = equity_recommendation_disposition(proposed)
    assert disposition is not None
    assert disposition.kind is EquityRecommendationDispositionKind.PROPOSED
    assert disposition.generation_key == proposed.generation_key
    assert disposition.assessment_fingerprint == proposed.assessment_fingerprint
    assert proposed.proposal is not None
    assert disposition.proposal_fingerprint == proposed.proposal.fingerprint
    assert not disposition.is_issued
    ports_ = _ports()
    ports_.references.hold("generation", _generation("generation/strict", minimum="100"))
    refused = ports_.run(generation_policy_version=_v("generation/strict"))
    ports_ = _ports()
    ports_.state.answer = _history(_prior(_LISTING))
    unresolved = ports_.run()
    for audit, kind in (
        (refused, EquityRecommendationDispositionKind.NOT_ELIGIBLE),
        (unresolved, EquityRecommendationDispositionKind.UNRESOLVED),
    ):
        record = equity_recommendation_disposition(audit)
        assert record is not None and record.kind is kind
        assert record.proposal_document is None and record.proposal_fingerprint is None
    assert {kind.value for kind in EquityRecommendationDispositionKind} == {
        "PROPOSED",
        "NOT_ELIGIBLE",
        "UNRESOLVED",
    }
    # A recorded kind can never be changed: an UNRESOLVED disposition never becomes PROPOSED.
    record = equity_recommendation_disposition(unresolved)
    assert record is not None
    with pytest.raises(InvalidEquityRecommendationRecordError):
        dataclasses.replace(record, kind=EquityRecommendationDispositionKind.PROPOSED)
    with pytest.raises(InvalidEquityRecommendationRecordError):
        dataclasses.replace(
            disposition,
            proposal_document=None,
            proposal_fingerprint=None,
        )
    with pytest.raises(InvalidEquityRecommendationRecordError):
        dataclasses.replace(
            disposition,
            assessment_document=disposition.assessment_document.replace("PROPOSED", "X", 1),
        )


def test_compat_ab_failures_without_an_identity_are_attempt_only() -> None:
    integrity = _ports()
    _replace_entry(integrity.world, 1, decision_fingerprint="eir1-sha256:" + "0" * 64)
    outage = _ports()
    outage.references.unavailable = {"costs"}
    mismatch = _ports()
    for audit, status in (
        (_Ports(_World()).run(), _S.NO_CANDIDATE),
        (integrity.run(), _S.RESEARCH_INTEGRITY_FAILURE),
        (outage.run(), _S.DEPENDENCY_UNAVAILABLE),
        (
            mismatch.run(configuration=dataclasses.replace(_CONFIG, version="synthetic-x/2")),
            _S.NOT_ELIGIBLE,
        ),
    ):
        assert audit.status is status
        assert audit.generation_key is None
        assert equity_recommendation_disposition(audit) is None
        with pytest.raises(InvalidEquityRecommendationRecordError, match="no generation"):
            EquityRecommendationGenerationDisposition.from_audit(audit)
        attempt = equity_recommendation_attempt(audit)
        assert (attempt.status, attempt.generation_key, attempt.disposition_kind) == (
            status,
            None,
            None,
        )
        assert attempt.attempt_identity.startswith("eirat1-sha256:")
    # Identical invocations are one logical attempt; a proposal's attempt names its disposition.
    _, proposed = _proposed()
    one, two = equity_recommendation_attempt(proposed), equity_recommendation_attempt(proposed)
    assert one == two
    assert one.disposition_kind is EquityRecommendationDispositionKind.PROPOSED
    assert json.loads(one.attempt_document)["assessment_fingerprint"] == (
        proposed.assessment_fingerprint
    )
    with pytest.raises(InvalidEquityRecommendationRecordError):
        EquityRecommendationGenerationAttempt("eirat1-sha256:" + "0" * 64, one.attempt_document)


# -- AC-AF: boundaries --------------------------------------------------


def test_compat_ac_ad_no_storage_issuance_or_lifecycle_contract_is_added() -> None:
    for module in (evidence_module, records_module, eligibility_module, generate_module):
        names = [
            node.name
            for node in ast.walk(ast.parse(inspect.getsource(module)))
            if isinstance(node, ast.ClassDef)
        ]
        for name in names:
            assert not name.endswith(("Store", "Repository", "Adapter", "Port")), name
            assert "Issued" not in name or name == "EquityIssuedHistoryEvidence", name
        source = inspect.getsource(module)
        for word in ("sqlite", "INSERT", "COMMIT", "RECORDED", "ACTIVE", "CLOSED_"):
            assert word not in source.replace("RECORDED`", ""), (module.__name__, word)
    _, audit = _proposed()
    assert audit.proposal is not None and not audit.proposal.is_issued
    assert audit.entry_window is not None
    assert audit.entry_window.timing.issuance_kind is EquityIssuanceKind.MODELED


def test_compat_ae_ei3c_and_ei4b_integration_is_intact() -> None:
    ports_, audit = _proposed()
    assert audit.inputs is not None and audit.proposal is not None
    entry = ports_.world.timelines.only().entries[1]
    source = audit.inputs.candidate.source
    assert (source.candidate_entry_fingerprint, source.evidence_snapshot_id) == (
        entry.entry_fingerprint,
        entry.snapshot_id,
    )
    assert audit.inputs.candidate.manifest_fingerprint == entry.manifest_fingerprint
    policy = audit.inputs.policy_set
    direct = form_equity_recommendation(
        source,
        audit.inputs.candidate.signal_close,
        audit.inputs.candidate.opening_range_low,
        policy.generation_policy,
        policy.risk_policy,
        policy.price_increment,
        policy.cost_schedule,
    )
    assert direct == audit.proposal.definition
    assert direct.sizing.quantity == 195  # type: ignore[union-attr]


def test_compat_af_the_compatibility_contracts_are_exported() -> None:
    for name in (
        "EquitySyntheticPolicySetReference",
        "EquityResolvedSyntheticPolicySet",
        "EquityVenueDayEvidence",
        "EquityIssuedHistoryEvidence",
        "EquityRecommendationInputs",
        "EquityRecommendationGenerationDisposition",
        "EquityRecommendationGenerationAttempt",
        "EquityRecommendationDispositionKind",
        "assess_equity_recommendation_inputs",
        "equity_recommendation_generation_key",
        "equity_recommendation_fingerprint",
        "verify_equity_recommendation_document",
        "equity_modeled_issuance",
    ):
        assert name in application_services.__all__
