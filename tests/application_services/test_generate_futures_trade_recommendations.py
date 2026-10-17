"""Tests for generating internal futures trade recommendations (F-5A-1).

Everything here is SYNTHETIC: the calendar (weekday sessions 03:45-10:00Z,
holidays 2026-10-02 / 10-20 / 11-24, only 2026 loaded), the bars, the research,
the candle approvals, the contract cutoffs and every policy and profile value.
None is NSE reference data, a provider candle or an approved configuration.
"Recorded" means accepted by an in-memory fake store; it is not durable.

The operated contract is NIFTY@NSE 2026-10-27. Its latest persisted bar is the
2026-10-07 session (close 10:00Z); research decides there and issuance is
captured at 12:00Z, so with M = 5 the window runs from the 10-08 close through
the 10-14 close. The last twenty closes are 24000, 24020, ..., 24360, 24383 for
BUY (reversed for SELL), so F-4 yields entry 24378-24388, stop 24347.75 and
target 24448.45 for LONG.
"""

from __future__ import annotations

import ast
import dataclasses
from datetime import date, timedelta
from decimal import Decimal
from itertools import permutations
from pathlib import Path

import pytest
from northstar_core.derivatives import ExpirationDate, QuoteValue
from northstar_core.foundation.value_objects import (
    ExchangeCode,
    PointInTime,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.futures import FuturesContract, FuturesOHLCVBar, FuturesProductReference
from northstar_core.recommendations.futures import (
    CandidateRejectionReason,
    EntryZone,
    FuturesRecommendationGenerationPolicy,
    FuturesResearchDecisionReference,
    FuturesTradeRecommendation,
    FuturesTradeRecommendationIdentity,
    FuturesTradeRecommendationLifecycle,
    GenerationRejectionReason,
    PriceTarget,
    RecommendationPolicyVersion,
    RecommendationStatus,
    RecommendationValidity,
    StopLoss,
    TradeDirection,
)
from northstar_core.strategy import (
    ExplanationReason,
    FuturesAssetAnalysis,
    FuturesMarketObservationContext,
    FuturesRecommendation,
    RecommendationAction,
    StrategyIdentity,
)

import northstar_application.application_services as services
from northstar_application.application_services import (
    DisabledFuturesDailyBarFinalityPolicy,
    FuturesAnalysisResult,
    FuturesCandleApproval,
    FuturesCandleApprovalMismatchError,
    FuturesCandleApprovalSet,
    FuturesCandleReviewAssessment,
    FuturesForwardResearchRecord,
    FuturesRecommendationGenerationAudit,
    FuturesRecommendationGenerationAuditOutcome,
    FuturesRecommendationWindowResolutionError,
    FuturesResearchHistoryCoverage,
    FuturesResearchHistoryDeficiency,
    FuturesReviewedCandleVerification,
    FuturesTradeRecommendationBlockReason,
    FuturesTradeRecommendationContractViolationError,
    FuturesTradeRecommendationGenerationProfile,
    FuturesTradeRecommendationGenerationResult,
    FuturesTradeRecommendationGenerationStatus,
    FuturesTradeRecommendationIdentityService,
    FuturesTradeRecommendationSourceError,
    GenerateFuturesTradeRecommendationsUseCase,
    InvalidFuturesTradeRecommendationGenerationError,
    OperatorApprovedFuturesDailyBarFinalityPolicy,
    futures_daily_candle_fingerprint,
)
from northstar_application.application_services import (
    generate_futures_trade_recommendations as generation_module,
)
from northstar_application.ports import (
    FuturesContractCutoff,
    FuturesContractCutoffSource,
    FuturesDailyBarFinality,
    FuturesDailyBarFinalityOutcome,
    FuturesForwardResearchRecordQuery,
    FuturesForwardResearchRecordRepository,
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
    FuturesRecommendationGenerationAuditConflictError,
    FuturesRecommendationGenerationAuditStore,
    FuturesSessionResolutionError,
    FuturesTradeRecommendationConfigurationConflictError,
    FuturesTradeRecommendationConflictError,
    FuturesTradeRecommendationExclusivityError,
    FuturesTradeRecommendationQuery,
    FuturesTradeRecommendationRepository,
    FuturesTradeRecommendationStore,
    FuturesTradingSession,
    FuturesTradingSessionResolver,
    InvalidFuturesRecommendationGenerationAuditRecordingError,
    InvalidFuturesTradeRecommendationRecordingError,
    validate_futures_trade_recommendation_recording,
    validate_standalone_futures_recommendation_generation_audit,
)

Block = FuturesTradeRecommendationBlockReason
Status = FuturesTradeRecommendationGenerationStatus
LONG, SHORT = TradeDirection.LONG, TradeDirection.SHORT

_NIFTY = FuturesProductReference(Symbol("NIFTY"), ExchangeCode("NSE"))
_BANKNIFTY = FuturesProductReference(Symbol("BANKNIFTY"), ExchangeCode("NSE"))
_OCT = FuturesContract(_NIFTY, ExpirationDate("2026-10-27"))
_NOV = FuturesContract(_NIFTY, ExpirationDate("2026-11-24"))
_BANK_OCT = FuturesContract(_BANKNIFTY, ExpirationDate("2026-10-27"))
_DAILY = Timeframe("1d")
_STRATEGY = StrategyIdentity("synthetic-directional-v1")
_OTHER = StrategyIdentity("synthetic-other-v1")
_SESSION_DATE = date(2026, 10, 7)
_ISSUED = PointInTime("2026-10-07T12:00:00Z")

_RISING = tuple(Decimal(24000 + 20 * step) for step in range(19)) + (Decimal("24383"),)
_FALLING = tuple(reversed(_RISING))
_WARMUP = (Decimal("23900"), Decimal("23950"), Decimal("23925"), Decimal("23975"), Decimal("23990"))


class SyntheticCalendar(FuturesTradingSessionResolver):
    """Table-driven synthetic calendar with only 2026 loaded."""

    _HOLIDAYS = frozenset({date(2026, 10, 2), date(2026, 10, 20), date(2026, 11, 24)})

    def resolve(
        self, product: FuturesProductReference, trading_date: date
    ) -> FuturesTradingSession | None:
        if trading_date.year != 2026:
            raise FuturesSessionResolutionError(f"Calendar year {trading_date.year} is not loaded.")
        if trading_date.weekday() >= 5 or trading_date in self._HOLIDAYS:
            return None
        day = trading_date.isoformat()
        return FuturesTradingSession(
            trading_date, PointInTime(f"{day}T03:45:00Z"), PointInTime(f"{day}T10:00:00Z")
        )

    def sessions_in_range(
        self, product: FuturesProductReference, start_date: date, end_date: date
    ) -> tuple[FuturesTradingSession, ...]:
        found, day = [], start_date
        while day <= end_date:
            session = self.resolve(product, day)
            if session is not None:
                found.append(session)
            day += timedelta(days=1)
        return tuple(found)


_CALENDAR = SyntheticCalendar()


def _session(day: date) -> FuturesTradingSession:
    session = _CALENDAR.resolve(_NIFTY, day)
    assert session is not None
    return session


def _sessions_through(last: date, count: int, *, skip: frozenset[date] = frozenset()) -> tuple:
    sessions = [
        s
        for s in _CALENDAR.sessions_in_range(_NIFTY, last - timedelta(days=60), last)
        if s.trading_date not in skip
    ]
    return tuple(sessions[-count:])


def _q(value: Decimal | str) -> QuoteValue:
    return QuoteValue(Decimal(value))


def _bar(
    contract: FuturesContract, session: FuturesTradingSession, close: Decimal
) -> FuturesOHLCVBar:
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=session.closes_at,
        timeframe=_DAILY,
        open=_q(close),
        high=_q(close + 5),
        low=_q(close - 5),
        close=_q(close),
        volume=Quantity(Decimal("1000")),
    )


def _bars(
    contract: FuturesContract = _OCT,
    closes: tuple[Decimal, ...] = _WARMUP + _RISING,
    last: date = _SESSION_DATE,
    skip: frozenset[date] = frozenset(),
) -> tuple[FuturesOHLCVBar, ...]:
    sessions = _sessions_through(last, len(closes), skip=skip)
    return tuple(_bar(contract, s, c) for s, c in zip(sessions, closes, strict=True))


def _record(
    bars: tuple[FuturesOHLCVBar, ...],
    action: str = "BUY",
    strategy: StrategyIdentity = _STRATEGY,
    *,
    latest: Decimal | None = None,
) -> FuturesForwardResearchRecord:
    window = bars[-20:]
    last = window[-1]
    instant = last.point_in_time
    context = FuturesMarketObservationContext(
        contract=last.contract,
        timeframe=_DAILY,
        observed_at=instant,
        latest_quote=_q(latest) if latest is not None else last.close,
        previous_close=window[-2].close,
        latest_volume=last.volume,
        session_high=last.high,
        session_low=last.low,
        recent_closes=tuple(bar.close for bar in window),
        recent_volumes=tuple(bar.volume for bar in window),
    )
    signal = {"BUY": "strong bullish", "SELL": "strong bearish"}.get(action, "neutral trend")
    recommendation = FuturesRecommendation(
        RecommendationAction(action),
        FuturesAssetAnalysis(last.contract, instant, (signal,)),
        strategy,
        instant,
    )
    return FuturesForwardResearchRecord(FuturesAnalysisResult(recommendation, context))


def _approvals(bars: tuple[FuturesOHLCVBar, ...], product: FuturesProductReference = _NIFTY):
    return FuturesReviewedCandleVerification(
        FuturesCandleApprovalSet(
            product,
            tuple(
                FuturesCandleApproval(
                    bar.contract,
                    _trading_date(bar),
                    futures_daily_candle_fingerprint(bar.contract, _trading_date(bar), bar),
                )
                for bar in bars
            ),
        )
    )


def _trading_date(bar: FuturesOHLCVBar) -> date:
    return date.fromisoformat(bar.point_in_time.value[:10])


_POLICY = FuturesRecommendationGenerationPolicy(
    version=RecommendationPolicyVersion("f4-synthetic-fixture-v1"),
    approved_strategies=(_STRATEGY,),
    lookback_changes=19,
    entry_multiplier=Decimal("0.25"),
    stop_multiplier=Decimal("1.5"),
    target_multiplier=Decimal("3"),
    price_increment=Decimal("0.05"),
    min_reward_to_risk=Decimal("1.5"),
    short_enabled=True,
)
_PROFILE = FuturesTradeRecommendationGenerationProfile(
    "f5a-synthetic-profile-v1", 5, _POLICY.version
)


# -- fakes -------------------------------------------------------------------------------


class Research(FuturesForwardResearchRecordRepository):
    def __init__(self, records: tuple[FuturesForwardResearchRecord, ...] = ()) -> None:
        self.records = records

    def get_records(self, query: FuturesForwardResearchRecordQuery):
        return tuple(r for r in self.records if r.contract == query.contract)


class Market(FuturesHistoricalMarketDataRepository):
    def __init__(self, bars: tuple[FuturesOHLCVBar, ...] = ()) -> None:
        self.bars = bars

    def get_bars(self, query: FuturesHistoricalMarketDataQuery):
        return tuple(
            b for b in self.bars if b.contract == query.contract and query.covers(b.point_in_time)
        )


class Cutoffs(FuturesContractCutoffSource):
    def __init__(self, known: dict[FuturesContract, FuturesContractCutoff]) -> None:
        self.known = known

    def resolve(self, contract: FuturesContract) -> FuturesContractCutoff | None:
        return self.known.get(contract)


_CUTOFFS = {
    _OCT: FuturesContractCutoff(_OCT, _session(date(2026, 10, 27)), "synthetic verified reference"),
    _NOV: FuturesContractCutoff(_NOV, _session(date(2026, 11, 23)), "synthetic verified reference"),
}


class Recommendations(
    FuturesTradeRecommendationRepository,
    FuturesTradeRecommendationStore,
    FuturesRecommendationGenerationAuditStore,
):
    """In-memory recording and audits with the documented semantics; not durable."""

    def __init__(self, extra: tuple[FuturesTradeRecommendationLifecycle, ...] = ()) -> None:
        self.recorded: dict[FuturesTradeRecommendationIdentity, FuturesTradeRecommendation] = {}
        self.extra = extra
        self.writes: list[tuple[FuturesTradeRecommendation, ...]] = []
        self.received: list[tuple[object, object]] = []
        self.answer: int | None = None
        # A test-only stand-in for the durable version registry of F-5B-1B.
        self.policies: dict[RecommendationPolicyVersion, FuturesRecommendationGenerationPolicy] = {}
        self.profiles: dict[str, FuturesTradeRecommendationGenerationProfile] = {}
        self.audits: dict[str, FuturesRecommendationGenerationAudit] = {}

    def get_recommendation(self, identity):
        if identity in self.recorded:
            return self.recorded[identity]
        return next(
            (lc.recommendation for lc in self.extra if lc.recommendation.identity == identity),
            None,
        )

    def get_lifecycles(self, query: FuturesTradeRecommendationQuery):
        lifecycles = tuple(FuturesTradeRecommendationLifecycle(r) for r in self.recorded.values())
        return tuple(
            lc
            for lc in lifecycles + self.extra
            if lc.recommendation.contract.product == query.product
        )

    def record(self, recommendations, *, policy, profile, audit):
        validate_futures_trade_recommendation_recording(
            recommendations, policy=policy, profile=profile, audit=audit
        )
        self.writes.append(recommendations)
        self.received.append((policy, profile))
        if self.policies.get(policy.version, policy) != policy or (
            self.profiles.get(profile.version, profile) != profile
        ):
            raise FuturesTradeRecommendationConfigurationConflictError(profile.version)
        known = [self.get_recommendation(r.identity) for r in recommendations]
        if any(item is not None for item in known):
            if known == list(recommendations) and self.audits.get(audit.identity) == audit:
                return len(recommendations) if self.answer is None else self.answer
            raise FuturesTradeRecommendationConflictError("already recorded by another recording")
        if self.audits.get(audit.identity, audit) != audit:
            raise FuturesRecommendationGenerationAuditConflictError(audit.identity)
        batch_products: set[FuturesProductReference] = set()
        for recommendation in recommendations:
            product = recommendation.contract.product
            if product in batch_products or any(
                not lc.is_terminal and lc.recommendation.identity != recommendation.identity
                for lc in self.get_lifecycles(FuturesTradeRecommendationQuery(product))
            ):
                raise FuturesTradeRecommendationExclusivityError(str(recommendation.contract))
            batch_products.add(product)
        self.policies[policy.version] = policy
        self.profiles[profile.version] = profile
        for recommendation in recommendations:
            self.recorded[recommendation.identity] = recommendation
        self.audits[audit.identity] = audit
        return len(recommendations) if self.answer is None else self.answer

    def record_audit(self, audit):
        validate_standalone_futures_recommendation_generation_audit(audit)
        if any(self._known(identity) is None for identity in audit.existing):
            raise InvalidFuturesRecommendationGenerationAuditRecordingError("unknown existing")
        known = self.audits.get(audit.identity)
        if known is not None:
            if known != audit:
                raise FuturesRecommendationGenerationAuditConflictError(audit.identity)
            return 0
        self.audits[audit.identity] = audit
        return 1

    def _known(self, identity):
        return self.get_recommendation(identity)

    def outcomes(self) -> list[str]:
        return sorted(audit.outcome.value for audit in self.audits.values())


@dataclasses.dataclass
class World:
    bars: tuple[FuturesOHLCVBar, ...]
    records: tuple[FuturesForwardResearchRecord, ...]
    recommendations: Recommendations = dataclasses.field(default_factory=Recommendations)
    cutoffs: dict = dataclasses.field(default_factory=lambda: dict(_CUTOFFS))
    final_through: date | None = _SESSION_DATE
    calendar: FuturesTradingSessionResolver = _CALENDAR

    def use_case(self) -> GenerateFuturesTradeRecommendationsUseCase:
        finality = (
            DisabledFuturesDailyBarFinalityPolicy()
            if self.final_through is None
            else OperatorApprovedFuturesDailyBarFinalityPolicy(self.final_through)
        )
        return GenerateFuturesTradeRecommendationsUseCase(
            Research(self.records),
            Market(self.bars),
            self.calendar,
            finality,
            Cutoffs(self.cutoffs),
            self.recommendations,
            self.recommendations,
            self.recommendations,
        )


def _world(action: str = "BUY", **overrides) -> World:
    closes = _WARMUP + (_FALLING if action == "SELL" else _RISING)
    bars = overrides.pop("bars") if "bars" in overrides else _bars(closes=closes)
    records = overrides.pop("records", None)
    if records is None:
        records = (_record(bars, action),)
    return World(bars=bars, records=records, **overrides)


def _run(
    world: World,
    *,
    issued_at: PointInTime = _ISSUED,
    profile: FuturesTradeRecommendationGenerationProfile = _PROFILE,
    policy: FuturesRecommendationGenerationPolicy = _POLICY,
    reviewed: FuturesReviewedCandleVerification | None | str = "all",
    strategy: StrategyIdentity = _STRATEGY,
) -> FuturesTradeRecommendationGenerationResult:
    if reviewed == "all":
        reviewed = _approvals(world.bars)
    return world.use_case().execute(
        contract=_OCT,
        strategy=strategy,
        policy=policy,
        profile=profile,
        issued_at=issued_at,
        reviewed_candles=reviewed,  # type: ignore[arg-type]
    )


def _identity(world: World, profile=_PROFILE) -> FuturesTradeRecommendationIdentity:
    source = world.records[0]
    return FuturesTradeRecommendationIdentityService().identity(
        FuturesResearchDecisionReference(
            source.contract, source.timeframe, source.decision_instant, source.strategy_identity
        ),
        profile,
    )


# ---------------------------------------------------------------------------
# Happy paths: frozen research -> F-3 -> F-4 -> identity -> F-2 -> recorded
# ---------------------------------------------------------------------------


def test_buy_research_becomes_a_recorded_long_recommendation() -> None:
    world = _world("BUY")

    result = _run(world)

    assert result.status is Status.COMPLETED and result.block_reason is None
    (recommendation,) = result.recorded
    assert result.existing == ()
    assert recommendation.direction is LONG
    assert recommendation.identity == _identity(world)
    assert recommendation.entry == EntryZone(_q("24378"), _q("24388"))
    assert recommendation.stop == StopLoss(_q("24347.75"))
    assert recommendation.target == PriceTarget(_q("24448.45"))
    assert recommendation.issued_at == _ISSUED
    assert recommendation.validity == RecommendationValidity(
        _session(date(2026, 10, 8)).closes_at,
        _session(date(2026, 10, 14)).closes_at,
        _session(date(2026, 10, 27)).closes_at,
    )
    assert world.recommendations.recorded == {recommendation.identity: recommendation}


def test_the_result_preserves_research_provenance_and_audit() -> None:
    world = _world("BUY")
    result = _run(world)

    (candidate,) = result.selection.candidates
    assert candidate.research_action == RecommendationAction("BUY")
    assert result.recorded[0].source == candidate.source
    assert result.session == _session(_SESSION_DATE)
    assert result.finality.outcome is FuturesDailyBarFinalityOutcome.FINAL
    assert result.history_coverage is FuturesResearchHistoryCoverage.VERIFIED
    assert result.generation.proposals[0].source == candidate.source
    assert result.profile == _PROFILE


def test_sell_research_becomes_a_recorded_short_recommendation() -> None:
    result = _run(_world("SELL"))
    (recommendation,) = result.recorded

    assert result.history_coverage is FuturesResearchHistoryCoverage.VERIFIED

    assert recommendation.direction is SHORT
    assert recommendation.entry == EntryZone(_q("23995"), _q("24005"))
    assert recommendation.stop == StopLoss(_q("24035.25"))
    assert recommendation.target == PriceTarget(_q("23934.55"))


def test_a_recorded_recommendation_starts_its_lifecycle_waiting() -> None:
    (recommendation,) = _run(_world()).recorded

    assert FuturesTradeRecommendationLifecycle(recommendation).status is (
        RecommendationStatus.WAITING_FOR_ENTRY
    )


# ---------------------------------------------------------------------------
# F-3 and F-4 rejections are results, not errors
# ---------------------------------------------------------------------------


def test_hold_research_is_an_f3_rejection_and_nothing_is_generated() -> None:
    world = _world("HOLD")

    result = _run(world)

    assert result.status is Status.COMPLETED
    assert [r.reason for r in result.selection.rejections] == [
        CandidateRejectionReason.NO_DIRECTIONAL_VIEW
    ]
    assert result.generation is None and result.recorded == ()
    assert world.recommendations.writes == []


def test_another_strategys_research_is_rejected_beside_the_approved_one() -> None:
    bars = _bars()
    world = _world(records=(_record(bars), _record(bars, "SELL", _OTHER)), bars=bars)

    result = _run(world)

    assert [r.reason for r in result.selection.rejections] == [
        CandidateRejectionReason.UNAPPROVED_STRATEGY
    ]
    assert result.recorded[0].direction is LONG


def test_inadequate_contract_life_is_f4s_rejection_and_nothing_is_recorded() -> None:
    world = _world(cutoffs={_OCT: FuturesContractCutoff(_OCT, _session(date(2026, 10, 12)), "s")})

    result = _run(world)

    assert [r.reason for r in result.generation.rejections] == [
        GenerationRejectionReason.INSUFFICIENT_CONTRACT_LIFE
    ]
    assert result.recorded == () and world.recommendations.writes == []


# ---------------------------------------------------------------------------
# The session gate: final, operator-reviewed, latest persisted session
# ---------------------------------------------------------------------------


def test_no_persisted_bar_blocks() -> None:
    result = _run(_world(bars=(), records=()), reviewed=None)

    assert (result.status, result.block_reason) == (Status.BLOCKED, Block.NO_PERSISTED_SESSION)


def test_a_session_that_is_not_yet_final_blocks() -> None:
    world = _world(final_through=date(2026, 10, 6))

    result = _run(world)

    assert result.block_reason is Block.SESSION_NOT_FINAL
    assert result.finality.outcome is FuturesDailyBarFinalityOutcome.NOT_YET_FINAL
    assert world.recommendations.writes == []


def test_disabled_finality_is_unknown_and_blocks() -> None:
    result = _run(_world(final_through=None))

    assert result.block_reason is Block.SESSION_NOT_FINAL
    assert result.finality.outcome is FuturesDailyBarFinalityOutcome.UNKNOWN


def test_final_without_operator_review_blocks() -> None:
    assert _run(_world(), reviewed=None).block_reason is Block.SESSION_NOT_OPERATOR_REVIEWED


def test_final_with_approvals_that_omit_the_session_blocks() -> None:
    world = _world()

    result = _run(world, reviewed=_approvals(world.bars[:-1]))

    assert result.block_reason is Block.SESSION_NOT_OPERATOR_REVIEWED


def test_a_stored_bar_that_differs_from_its_approved_candle_raises() -> None:
    world = _world()
    altered = dataclasses.replace(world.bars[-1], volume=Quantity(Decimal("999")))

    with pytest.raises(FuturesCandleApprovalMismatchError):
        _run(world, reviewed=_approvals(world.bars[:-1] + (altered,)))


def test_stale_research_is_not_promoted_when_the_latest_session_has_none() -> None:
    bars = _bars()
    world = _world(bars=bars, records=(_record(bars[:-1]),))

    result = _run(world)

    assert result.block_reason is Block.NO_RESEARCH_FOR_SESSION
    assert result.selection is None and world.recommendations.writes == []


def test_research_only_by_another_strategy_blocks() -> None:
    bars = _bars()

    assert _run(_world(bars=bars, records=(_record(bars, "BUY", _OTHER),))).block_reason is (
        Block.NO_RESEARCH_FOR_SESSION
    )


def test_research_after_the_latest_persisted_session_is_never_used() -> None:
    """A record dated after the latest stored bar is not the session's research."""
    bars = _bars()
    future_bars = _bars(last=date(2026, 10, 8))
    world = _world(bars=bars, records=(_record(future_bars),))

    assert _run(world).block_reason is Block.NO_RESEARCH_FOR_SESSION


def test_a_missing_contract_cutoff_blocks_after_selection() -> None:
    world = _world(cutoffs={})

    result = _run(world)

    assert result.block_reason is Block.CONTRACT_CUTOFF_UNAVAILABLE
    assert len(result.selection.candidates) == 1
    assert world.recommendations.writes == []


def test_a_contradictory_contract_cutoff_raises() -> None:
    wrong = FuturesContractCutoff(_OCT, _session(date(2026, 10, 28)), "synthetic")

    with pytest.raises(FuturesRecommendationWindowResolutionError):
        _run(_world(cutoffs={_OCT: wrong}))


# ---------------------------------------------------------------------------
# Research history: a NEW recommendation requires VERIFIED history
# ---------------------------------------------------------------------------

History = FuturesResearchHistoryDeficiency
_REQUIRED = _sessions_through(_SESSION_DATE, 20)  # the 20 expected research sessions


def _without(bars: tuple[FuturesOHLCVBar, ...], *days: date) -> tuple[FuturesOHLCVBar, ...]:
    return tuple(bar for bar in bars if _trading_date(bar) not in days)


def _assert_blocked_on_history(
    world: World,
    result: FuturesTradeRecommendationGenerationResult,
    deficiency: FuturesResearchHistoryDeficiency,
    sessions: tuple[date, ...],
) -> None:
    assert result.status is Status.BLOCKED
    assert result.block_reason is Block.RESEARCH_HISTORY_NOT_VERIFIED
    assert result.history_coverage is FuturesResearchHistoryCoverage.UNVERIFIED
    assert result.history_deficiency is deficiency
    assert result.history_deficient_sessions == sessions
    assert result.generation is None and result.recorded == ()
    assert world.recommendations.writes == []


def test_the_required_history_spans_weekends_and_a_holiday_and_is_verified() -> None:
    """09-09 .. 10-07 holds four weekends and the 10-02 holiday; none is a gap."""
    assert _REQUIRED[0].trading_date == date(2026, 9, 9)
    assert date(2026, 10, 2) not in {session.trading_date for session in _REQUIRED}

    for action in ("BUY", "SELL"):
        result = _run(_world(action))
        assert result.history_coverage is FuturesResearchHistoryCoverage.VERIFIED
        assert result.history_deficiency is None and len(result.recorded) == 1


def test_a_missing_intermediate_bar_blocks_even_when_research_skipped_it() -> None:
    """Research frozen over the stored, gappy series still has twenty closes."""
    bars = _bars(skip=frozenset({date(2026, 9, 30)}))
    world = _world(bars=bars, records=(_record(bars),))

    result = _run(world)

    assert len(world.records[0].result.market_observation_context.recent_closes) == 20
    _assert_blocked_on_history(world, result, History.MISSING_BAR, (date(2026, 9, 30),))


def test_a_bar_missing_from_storage_is_missing_evidence_not_a_contradiction() -> None:
    world = _world()
    world.bars = _without(world.bars, date(2026, 9, 30))

    _assert_blocked_on_history(world, _run(world), History.MISSING_BAR, (date(2026, 9, 30),))


def test_a_missing_first_required_bar_blocks() -> None:
    world = _world()
    world.bars = _without(world.bars, _REQUIRED[0].trading_date)

    _assert_blocked_on_history(
        world, _run(world), History.MISSING_BAR, (_REQUIRED[0].trading_date,)
    )


def test_only_nineteen_stored_bars_block_despite_twenty_frozen_closes() -> None:
    world = _world()
    world.bars = world.bars[-19:]

    _assert_blocked_on_history(
        world, _run(world), History.MISSING_BAR, (_REQUIRED[0].trading_date,)
    )


def test_a_missing_decision_session_bar_never_generates_on_the_older_session() -> None:
    """The latest stored session becomes 10-06, which has no research of its own."""
    world = _world()
    world.bars = world.bars[:-1]

    result = _run(world)

    assert result.block_reason is Block.NO_RESEARCH_FOR_SESSION
    assert world.recommendations.writes == []


def test_a_calendar_that_cannot_supply_the_sessions_blocks() -> None:
    class ShortCalendar(SyntheticCalendar):
        def resolve(self, product, trading_date):
            return (
                None if trading_date < date(2026, 9, 25) else super().resolve(product, trading_date)
            )

    world = _world(calendar=ShortCalendar())
    world.bars = world.bars[-9:]

    _assert_blocked_on_history(world, _run(world), History.INCOMPLETE_CALENDAR, ())


def test_an_unloaded_history_calendar_year_propagates_and_never_generates() -> None:
    class Unloaded(SyntheticCalendar):
        def resolve(self, product, trading_date):
            if trading_date < date(2026, 9, 20):
                raise FuturesSessionResolutionError("history year not loaded")
            return super().resolve(product, trading_date)

    world = _world(calendar=Unloaded())

    with pytest.raises(FuturesSessionResolutionError):
        _run(world)
    assert world.recommendations.writes == []


@pytest.mark.parametrize(
    "unapproved",
    [(date(2026, 9, 30),), (date(2026, 9, 9),)],
    ids=["intermediate", "first-required"],
)
def test_an_unapproved_required_candle_blocks(unapproved: tuple[date, ...]) -> None:
    world = _world()

    result = _run(world, reviewed=_approvals(_without(world.bars, *unapproved)))

    _assert_blocked_on_history(world, result, History.UNAPPROVED_CANDLE, unapproved)


def test_approving_only_the_latest_candle_is_not_approving_the_history() -> None:
    world = _world()

    result = _run(world, reviewed=_approvals(world.bars[-1:]))

    _assert_blocked_on_history(
        world,
        result,
        History.UNAPPROVED_CANDLE,
        tuple(session.trading_date for session in _REQUIRED[:-1]),
    )


def test_a_history_candle_that_differs_from_its_approved_version_raises() -> None:
    world = _world()
    tampered = tuple(
        dataclasses.replace(bar, volume=Quantity(Decimal("1")))
        if _trading_date(bar) == date(2026, 9, 30)
        else bar
        for bar in world.bars
    )

    with pytest.raises(FuturesCandleApprovalMismatchError):
        _run(world, reviewed=_approvals(tampered))
    assert world.recommendations.writes == []


def test_research_that_disagrees_with_the_reviewed_candle_raises() -> None:
    bars = _bars()
    world = _world(bars=bars, records=(_record(bars, latest=Decimal("24380")),))

    with pytest.raises(FuturesTradeRecommendationSourceError):
        _run(world)


def test_a_stored_close_that_differs_from_the_frozen_close_raises() -> None:
    bars = _bars()
    other = _bars(closes=_WARMUP + tuple(c + 1 for c in _RISING[:-1]) + (_RISING[-1],))
    world = _world(bars=bars, records=(_record(other),))

    with pytest.raises(FuturesTradeRecommendationSourceError):
        _run(world)
    assert world.recommendations.writes == []


def test_a_stored_bar_on_no_expected_session_raises() -> None:
    world = _world()
    saturday = FuturesTradingSession(
        date(2026, 10, 3),
        PointInTime("2026-10-03T03:45:00Z"),
        PointInTime("2026-10-03T10:00:00Z"),
    )
    stray = _bar(_OCT, saturday, Decimal("24300"))
    world.bars = tuple(sorted(world.bars + (stray,), key=lambda bar: bar.point_in_time.value))

    with pytest.raises(FuturesTradeRecommendationSourceError, match="no expected"):
        _run(world, reviewed=_approvals(world.bars))


@pytest.mark.parametrize(
    "foreign",
    [
        _bar(_NOV, _session(date(2026, 9, 30)), Decimal("24100")),
        dataclasses.replace(
            _bar(_OCT, _session(date(2026, 9, 30)), Decimal("24100")), timeframe=Timeframe("1h")
        ),
    ],
    ids=["wrong-contract", "wrong-timeframe"],
)
def test_history_for_another_contract_or_timeframe_is_a_contract_violation(
    foreign: FuturesOHLCVBar,
) -> None:
    class Leaky(Market):
        def get_bars(self, query):
            return tuple(sorted(self.bars + (foreign,), key=lambda bar: bar.point_in_time.value))

    world = _world()
    use_case = world.use_case()
    use_case._market = Leaky(world.bars)

    with pytest.raises(FuturesTradeRecommendationContractViolationError):
        use_case.execute(
            contract=_OCT,
            strategy=_STRATEGY,
            policy=_POLICY,
            profile=_PROFILE,
            issued_at=_ISSUED,
            reviewed_candles=_approvals(world.bars),
        )


def test_duplicate_conflicting_bars_are_a_contract_violation() -> None:
    world = _world()
    clean = world.bars
    twin = _bar(_OCT, _session(_trading_date(clean[-3])), clean[-3].close.value + 1)
    world.bars = clean[:-2] + (twin,) + clean[-2:]

    with pytest.raises(FuturesTradeRecommendationContractViolationError):
        _run(world, reviewed=_approvals(clean))
    assert world.recommendations.writes == []


def test_a_later_stored_bar_moves_the_session_and_stale_research_is_not_used() -> None:
    world = _world()
    world.bars = world.bars + (_bar(_OCT, _session(date(2026, 10, 8)), Decimal("24390")),)

    result = _run(world, reviewed=_approvals(world.bars))

    assert result.block_reason is Block.SESSION_NOT_FINAL
    assert world.recommendations.writes == []


def test_unverified_history_never_reaches_f4(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("F-4 must not run on unverified history")

    monkeypatch.setattr(generation_module.FuturesRecommendationGeneration, "generate", forbidden)
    world = _world()

    result = _run(world, reviewed=_approvals(_without(world.bars, date(2026, 9, 30))))

    assert result.block_reason is Block.RESEARCH_HISTORY_NOT_VERIFIED
    assert len(result.selection.candidates) == 1


def test_a_blocked_attempt_succeeds_once_the_history_is_approved() -> None:
    world = _world()
    blocked = _run(world, reviewed=_approvals(_without(world.bars, date(2026, 9, 30))))
    assert blocked.block_reason is Block.RESEARCH_HISTORY_NOT_VERIFIED

    result = _run(world)

    assert result.status is Status.COMPLETED and len(result.recorded) == 1
    assert result.recorded[0].identity == _identity(world)


def test_a_result_cannot_claim_a_recording_from_unverified_history() -> None:
    world = _world()
    (recorded,) = _run(world).recorded

    with pytest.raises(InvalidFuturesTradeRecommendationGenerationError):
        FuturesTradeRecommendationGenerationResult(
            contract=_OCT,
            strategy=_STRATEGY,
            profile=_PROFILE,
            issued_at=_ISSUED,
            status=Status.COMPLETED,
            history_coverage=FuturesResearchHistoryCoverage.UNVERIFIED,
            history_deficiency=History.MISSING_BAR,
            recorded=(recorded,),
        )


def test_hold_research_is_not_blocked_by_history_and_is_not_generated() -> None:
    """HOLD is a research outcome, not a candidate: no history gate, no trade."""
    world = _world("HOLD")
    world.bars = _without(world.bars, date(2026, 9, 30))
    world.records = (_record(world.bars, "HOLD"),)

    result = _run(world)

    assert result.status is Status.COMPLETED and result.history_coverage is None
    assert [r.reason for r in result.selection.rejections] == [
        CandidateRejectionReason.NO_DIRECTIONAL_VIEW
    ]
    assert world.recommendations.writes == []


def test_a_contradictory_hold_record_is_not_disguised_as_a_hold_result() -> None:
    bars = _bars()
    world = _world(bars=bars, records=(_record(bars, "HOLD", latest=Decimal("24380")),))

    with pytest.raises(FuturesTradeRecommendationSourceError):
        _run(world)


# ---------------------------------------------------------------------------
# Identical retries return the recorded recommendation
# ---------------------------------------------------------------------------


def test_an_identical_retry_returns_the_recorded_recommendation_unchanged() -> None:
    world = _world()
    (first,) = _run(world).recorded

    retry = _run(world, issued_at=PointInTime("2026-10-08T06:00:00Z"))

    assert retry.recorded == () and retry.existing == (first,)
    assert retry.existing[0].issued_at == _ISSUED
    assert retry.existing[0].validity == first.validity
    assert retry.existing[0].thesis == first.thesis
    assert retry.generation is None
    assert len(world.recommendations.writes) == 1


def test_a_restarted_use_case_returns_the_same_recorded_recommendation() -> None:
    world = _world()
    (first,) = _run(world).recorded

    restarted = world.use_case().execute(
        contract=_OCT,
        strategy=_STRATEGY,
        policy=_POLICY,
        profile=_PROFILE,
        issued_at=PointInTime("2026-10-07T13:00:00Z"),
        reviewed_candles=_approvals(world.bars),
    )

    assert restarted.existing == (first,)


@pytest.mark.parametrize(
    "degrade",
    ["unapproved-history", "missing-history-bar"],
)
def test_a_retry_returns_the_recorded_recommendation_without_reverifying_history(
    degrade: str,
) -> None:
    world = _world()
    (first,) = _run(world).recorded
    if degrade == "unapproved-history":
        reviewed = _approvals(world.bars[-1:])
    else:
        world.bars = _without(world.bars, date(2026, 9, 30))
        reviewed = _approvals(world.bars)

    retry = _run(world, issued_at=PointInTime("2026-10-08T06:00:00Z"), reviewed=reviewed)

    assert retry.status is Status.COMPLETED and retry.existing == (first,)
    assert retry.history_coverage is None
    assert retry.generation is None and retry.recorded == ()
    assert len(world.recommendations.writes) == 1


def test_a_recorded_identity_with_different_content_is_a_conflict() -> None:
    world = _world()
    (first,) = _run(world).recorded
    world.recommendations.recorded[first.identity] = dataclasses.replace(
        first, policy_version=RecommendationPolicyVersion("tampered")
    )

    with pytest.raises(FuturesTradeRecommendationConflictError):
        _run(world)


@pytest.mark.parametrize("change", ["profile", "policy"])
def test_a_new_configuration_identity_requires_verified_history(change: str) -> None:
    world = _world()
    (first,) = _run(world).recorded
    if change == "profile":
        policy = _POLICY
        profile = FuturesTradeRecommendationGenerationProfile(
            "f5a-synthetic-profile-v2", 5, _POLICY.version
        )
    else:
        policy = dataclasses.replace(
            _POLICY, version=RecommendationPolicyVersion("f4-synthetic-fixture-v2")
        )
        profile = FuturesTradeRecommendationGenerationProfile(
            "f5a-synthetic-profile-v1", 5, policy.version
        )

    unverified = _run(world, policy=policy, profile=profile, reviewed=_approvals(world.bars[-1:]))
    verified = _run(world, policy=policy, profile=profile)

    assert _identity(world, profile) != first.identity
    assert unverified.block_reason is Block.RESEARCH_HISTORY_NOT_VERIFIED
    assert [r.reason for r in verified.generation.rejections] == [
        GenerationRejectionReason.DUPLICATE_OPEN_RECOMMENDATION
    ]
    assert len(world.recommendations.writes) == 1


def test_a_new_profile_version_cannot_bypass_the_open_recommendation() -> None:
    world = _world()
    _run(world)
    newer = FuturesTradeRecommendationGenerationProfile(
        "f5a-synthetic-profile-v2", 5, _POLICY.version
    )

    result = _run(world, profile=newer)

    assert _identity(world, newer) != _identity(world)
    assert [r.reason for r in result.generation.rejections] == [
        GenerationRejectionReason.DUPLICATE_OPEN_RECOMMENDATION
    ]
    assert result.recorded == () and len(world.recommendations.writes) == 1


# ---------------------------------------------------------------------------
# One open recommendation per product, across expiries (P-1)
# ---------------------------------------------------------------------------


def _open(
    direction: TradeDirection,
    contract: FuturesContract = _NOV,
    identity: str = "open-1",
    *,
    active: bool = False,
    terminal: bool = False,
) -> FuturesTradeRecommendationLifecycle:
    long = direction is LONG
    eligible = _session(date(2026, 10, 7)).closes_at
    recommendation = FuturesTradeRecommendation(
        identity=FuturesTradeRecommendationIdentity(identity),
        contract=contract,
        direction=direction,
        entry=EntryZone(_q("24300"), _q("24310")),
        stop=StopLoss(_q("24250" if long else "24360")),
        target=PriceTarget(_q("24400" if long else "24200")),
        validity=RecommendationValidity(
            eligible, _session(date(2026, 10, 14)).closes_at, _session(date(2026, 10, 27)).closes_at
        ),
        source=_source_of(contract),
        policy_version=_POLICY.version,
        issued_at=PointInTime("2026-10-06T12:00:00Z"),
        thesis=(ExplanationReason("Synthetic open recommendation"),),
    )
    lifecycle = FuturesTradeRecommendationLifecycle(recommendation)
    if active or terminal:
        low = "24240" if terminal and long else "24295"
        high = "24370" if terminal and not long else "24320"
        lifecycle = lifecycle.advance(
            FuturesOHLCVBar(
                contract, eligible, _DAILY, _q(low), _q(high), _q(low), _q(high), Quantity(1)
            )
        )
    return lifecycle


def _source_of(contract: FuturesContract) -> FuturesResearchDecisionReference:
    return FuturesResearchDecisionReference(
        contract, _DAILY, _session(date(2026, 10, 6)).closes_at, _STRATEGY
    )


@pytest.mark.parametrize(
    "existing, action, expected",
    [
        (_open(LONG), "BUY", GenerationRejectionReason.DUPLICATE_OPEN_RECOMMENDATION),
        (_open(SHORT), "BUY", GenerationRejectionReason.CONTRADICTS_OPEN_RECOMMENDATION),
        (_open(LONG, active=True), "BUY", GenerationRejectionReason.DUPLICATE_OPEN_RECOMMENDATION),
        (
            _open(LONG, active=True),
            "SELL",
            GenerationRejectionReason.CONTRADICTS_OPEN_RECOMMENDATION,
        ),
    ],
    ids=["waiting-same", "waiting-opposite", "active-same", "active-opposite"],
)
def test_an_open_recommendation_on_another_expiry_blocks_the_product(
    existing: FuturesTradeRecommendationLifecycle, action: str, expected: GenerationRejectionReason
) -> None:
    world = _world(action, recommendations=Recommendations((existing,)))

    result = _run(world)

    assert [r.reason for r in result.generation.rejections] == [expected]
    assert result.recorded == () and world.recommendations.writes == []


def test_a_terminal_recommendation_does_not_block() -> None:
    finished = _open(LONG, terminal=True)
    assert finished.status is RecommendationStatus.INDETERMINATE_ENTRY

    result = _run(_world(recommendations=Recommendations((finished,))))

    assert len(result.recorded) == 1


def test_an_open_recommendation_on_another_product_does_not_block() -> None:
    result = _run(_world(recommendations=Recommendations((_open(SHORT, _BANK_OCT),))))

    assert len(result.recorded) == 1


def test_several_open_recommendations_are_rejected_and_the_breach_is_surfaced() -> None:
    opened = (_open(LONG, _NOV, "a"), _open(SHORT, _NOV, "b"))

    results = {
        _run(_world(recommendations=Recommendations(ordering))).generation
        for ordering in permutations(opened)
    }

    (generation,) = results
    assert [r.reason for r in generation.rejections] == [
        GenerationRejectionReason.CONTRADICTS_OPEN_RECOMMENDATION
    ]
    assert generation.exclusivity_breaches == (_NIFTY,)


# ---------------------------------------------------------------------------
# Configuration, ports and failure propagation
# ---------------------------------------------------------------------------


def test_a_profile_for_another_policy_version_is_refused() -> None:
    profile = FuturesTradeRecommendationGenerationProfile(
        "p", 5, RecommendationPolicyVersion("f4-synthetic-fixture-v9")
    )

    with pytest.raises(InvalidFuturesTradeRecommendationGenerationError):
        _run(_world(), profile=profile)


def test_a_policy_for_another_strategy_is_refused() -> None:
    with pytest.raises(InvalidFuturesTradeRecommendationGenerationError):
        _run(_world(), strategy=_OTHER)


def test_approvals_for_another_product_are_refused() -> None:
    world = _world()
    with pytest.raises(InvalidFuturesTradeRecommendationGenerationError):
        _run(
            world,
            reviewed=FuturesReviewedCandleVerification(FuturesCandleApprovalSet(_BANKNIFTY, ())),
        )


def test_issuance_before_the_research_session_closes_is_refused() -> None:
    with pytest.raises(InvalidFuturesTradeRecommendationGenerationError, match="backdated"):
        _run(_world(), issued_at=PointInTime("2026-10-07T09:00:00Z"))


def test_research_for_another_contract_is_a_port_contract_violation() -> None:
    class WrongResearch(Research):
        def get_records(self, query):
            return tuple(self.records)

    world = _world()
    use_case = world.use_case()
    use_case._research = WrongResearch((_record(_bars(contract=_NOV)),))

    with pytest.raises(FuturesTradeRecommendationContractViolationError):
        use_case.execute(
            contract=_OCT,
            strategy=_STRATEGY,
            policy=_POLICY,
            profile=_PROFILE,
            issued_at=_ISSUED,
            reviewed_candles=_approvals(world.bars),
        )


# ---------------------------------------------------------------------------
# The recording contract: exact policy and profile, store semantics
# ---------------------------------------------------------------------------

_T1 = PointInTime("2026-10-07T11:00:00Z")
_T2 = PointInTime("2026-10-07T12:00:00Z")


def _recorded(issued_at: PointInTime = _T1) -> FuturesTradeRecommendation:
    """The recommendation one attempt records, from a fresh synthetic world."""
    (recommendation,) = _run(_world(), issued_at=issued_at).recorded
    return recommendation


def _audit_for(
    *recommendations: FuturesTradeRecommendation,
    profile: FuturesTradeRecommendationGenerationProfile = _PROFILE,
) -> FuturesRecommendationGenerationAudit:
    """A RECORDED audit naming exactly ``recommendations``, as an attempt would."""
    first = recommendations[0]
    session = _session(_SESSION_DATE)
    identities = tuple(recommendation.identity for recommendation in recommendations)
    return FuturesRecommendationGenerationAudit(
        contract=first.contract,
        strategy=first.source.strategy_identity,
        policy_version=first.policy_version,
        profile_version=profile.version,
        attempted_at=first.issued_at,
        status=Status.COMPLETED,
        session=session,
        finality=FuturesDailyBarFinality(
            first.contract, session, FuturesDailyBarFinalityOutcome.FINAL, "synthetic"
        ),
        history_coverage=FuturesResearchHistoryCoverage.VERIFIED,
        proposed=identities,
        recorded=identities,
    )


def _derived(
    base: FuturesTradeRecommendation,
    *,
    contract: FuturesContract | None = None,
    strategy: StrategyIdentity | None = None,
    instant: PointInTime | None = None,
    profile: FuturesTradeRecommendationGenerationProfile = _PROFILE,
) -> FuturesTradeRecommendation:
    """A variant of ``base`` whose identity is correctly derived from its new source."""
    source = FuturesResearchDecisionReference(
        contract or base.contract,
        base.source.timeframe,
        instant or base.source.decision_instant,
        strategy or base.source.strategy_identity,
    )
    return dataclasses.replace(
        base,
        contract=source.contract,
        source=source,
        identity=FuturesTradeRecommendationIdentityService().identity(source, profile),
    )


def test_the_store_receives_the_exact_policy_and_profile() -> None:
    world = _world()

    _run(world)

    ((policy, profile),) = world.recommendations.received
    assert policy is _POLICY and profile is _PROFILE
    assert profile.policy_version == policy.version
    assert world.recommendations.writes[0][0].policy_version == policy.version


def _store_record(store, *recommendations, profile=_PROFILE, audit=None):
    return store.record(
        recommendations,
        policy=_POLICY,
        profile=profile,
        audit=audit or _audit_for(*recommendations, profile=profile),
    )


def test_replaying_the_same_committed_recording_is_idempotent() -> None:
    store, recommendation = Recommendations(), _recorded()
    _store_record(store, recommendation)

    assert _store_record(store, recommendation) == 1
    assert store.recorded == {recommendation.identity: recommendation}
    assert store.outcomes() == ["RECORDED"]


def test_an_equal_definition_recorded_by_another_attempt_is_a_conflict() -> None:
    store, recommendation = Recommendations(), _recorded()
    _store_record(store, recommendation)
    other_audit = dataclasses.replace(
        _audit_for(recommendation), candidates=(recommendation.source,)
    )

    with pytest.raises(FuturesTradeRecommendationConflictError):
        _store_record(store, recommendation, audit=other_audit)
    assert list(store.audits.values()) == [_audit_for(recommendation)]


def test_a_different_definition_under_one_identity_is_a_conflict() -> None:
    store = Recommendations()
    _store_record(store, _recorded(_T1))

    with pytest.raises(FuturesTradeRecommendationConflictError):
        _store_record(store, _recorded(_T2))
    assert next(iter(store.recorded.values())).issued_at == _T1
    assert len(store.audits) == 1


def test_another_open_recommendation_on_the_product_is_exclusive() -> None:
    store, first = Recommendations(), _recorded()
    _store_record(store, first)

    with pytest.raises(FuturesTradeRecommendationExclusivityError):
        _store_record(store, _derived(first, contract=_NOV))


def test_a_batch_is_all_or_nothing() -> None:
    store, first = Recommendations(), _recorded()
    sibling = _derived(first, instant=_session(date(2026, 10, 6)).closes_at)

    with pytest.raises(FuturesTradeRecommendationExclusivityError):
        _store_record(store, first, sibling)
    assert store.recorded == {} and store.audits == {}


def test_a_profile_version_reused_with_other_contents_is_a_configuration_conflict() -> None:
    store = Recommendations()
    _store_record(store, _recorded())
    reused = FuturesTradeRecommendationGenerationProfile(_PROFILE.version, 7, _POLICY.version)

    with pytest.raises(FuturesTradeRecommendationConfigurationConflictError):
        _store_record(store, _recorded(_T2), profile=reused)


def test_an_invalid_recording_request_is_refused_before_anything_is_written() -> None:
    store = Recommendations()
    other_profile = FuturesTradeRecommendationGenerationProfile(
        "f5a-synthetic-profile-v1", 5, RecommendationPolicyVersion("f4-synthetic-fixture-v9")
    )

    with pytest.raises(InvalidFuturesTradeRecommendationRecordingError):
        store.record(
            (_recorded(),), policy=_POLICY, profile=other_profile, audit=_audit_for(_recorded())
        )
    assert store.writes == [] and store.recorded == {}


# ---------------------------------------------------------------------------
# Same-identity races: the first recording wins and is returned unchanged
# ---------------------------------------------------------------------------


class Racing(Recommendations):
    """Another attempt records ``winner`` after this one's pre-read and before its write.

    The pre-read sees nothing; the write then meets the winner and is refused
    with a same-identity conflict, exactly as a durable store would refuse it.
    """

    def __init__(
        self,
        winner: object,
        *,
        refuse_with: type[Exception] = FuturesTradeRecommendationConflictError,
    ) -> None:
        super().__init__()
        self.winner = winner
        self.refuse_with = refuse_with
        self.raced = False
        self.reads: list[FuturesTradeRecommendationIdentity] = []

    def get_recommendation(self, identity):
        self.reads.append(identity)
        if not self.raced:
            return None
        return self.winner

    def _known(self, identity):
        return self.winner if self.raced else None

    def record(self, recommendations, *, policy, profile, audit):
        validate_futures_trade_recommendation_recording(
            recommendations, policy=policy, profile=profile, audit=audit
        )
        self.writes.append(recommendations)
        self.received.append((policy, profile))
        self.raced = True
        raise self.refuse_with("synthetic: the identity is already recorded")


def _count_f4(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls: list[int] = []
    original = generation_module.FuturesRecommendationGeneration.generate

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(generation_module.FuturesRecommendationGeneration, "generate", counted)
    return calls


def test_a_losing_attempt_returns_the_recorded_winner_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    winner = _recorded(_T1)
    world = _world(recommendations=Racing(winner))
    f4_calls = _count_f4(monkeypatch)

    result = _run(world, issued_at=_T2)

    assert result.status is Status.COMPLETED
    assert result.recorded == () and result.existing == (winner,)
    assert result.existing[0].issued_at == _T1
    assert (result.existing[0].entry, result.existing[0].stop, result.existing[0].target) == (
        winner.entry,
        winner.stop,
        winner.target,
    )
    assert result.existing[0].validity == winner.validity
    assert result.existing[0].thesis == winner.thesis
    assert result.existing[0].identity == _identity(world)
    assert len(world.recommendations.writes) == 1 and f4_calls == [1]
    assert world.recommendations.reads == [_identity(world), _identity(world)]


def test_the_losing_definition_is_discarded_not_stored() -> None:
    winner = _recorded(_T1)
    world = _world(recommendations=Racing(winner))

    result = _run(world, issued_at=_T2)

    (losing,) = world.recommendations.writes[0]
    assert losing.issued_at == _T2 and losing != winner
    assert world.recommendations.recorded == {}
    assert result.generation.proposals[0].issued_at == _T2


@pytest.mark.parametrize(
    "tamper",
    [
        lambda w: dataclasses.replace(
            w,
            source=dataclasses.replace(
                w.source, decision_instant=_session(date(2026, 10, 6)).closes_at
            ),
        ),
        lambda w: dataclasses.replace(
            w, source=dataclasses.replace(w.source, strategy_identity=_OTHER)
        ),
        lambda w: dataclasses.replace(
            w, contract=_NOV, source=dataclasses.replace(w.source, contract=_NOV)
        ),
        lambda w: dataclasses.replace(w, policy_version=RecommendationPolicyVersion("tampered")),
    ],
    ids=["decision-instant", "strategy", "contract", "policy-version"],
)
def test_a_winner_for_another_source_or_configuration_is_a_conflict(tamper) -> None:
    world = _world(recommendations=Racing(tamper(_recorded(_T1))))

    with pytest.raises(FuturesTradeRecommendationConflictError):
        _run(world, issued_at=_T2)


@pytest.mark.parametrize(
    "stored",
    ["not a recommendation", None],
    ids=["malformed", "missing"],
)
def test_a_missing_or_malformed_winner_never_succeeds(stored: object) -> None:
    world = _world(recommendations=Racing(stored))

    with pytest.raises(
        (FuturesTradeRecommendationConflictError, FuturesTradeRecommendationContractViolationError)
    ):
        _run(world, issued_at=_T2)


def test_a_winner_answered_under_another_identity_is_a_contract_violation() -> None:
    winner = _derived(_recorded(_T1), strategy=_OTHER)
    world = _world(recommendations=Racing(winner))

    with pytest.raises(FuturesTradeRecommendationContractViolationError):
        _run(world, issued_at=_T2)


@pytest.mark.parametrize(
    "error",
    [
        FuturesTradeRecommendationConfigurationConflictError,
        FuturesTradeRecommendationExclusivityError,
    ],
)
def test_configuration_and_exclusivity_conflicts_are_never_reconciled(error: type) -> None:
    racing = Racing(_recorded(_T1), refuse_with=error)
    world = _world(recommendations=racing)

    with pytest.raises(error):
        _run(world, issued_at=_T2)
    assert racing.reads == [_identity(world)]


def test_a_batch_race_is_reconciled_only_when_every_winner_verifies() -> None:
    winner = _recorded(_T1)
    sibling = _derived(winner, strategy=_OTHER)
    identities = {winner.source: winner.identity, sibling.source: sibling.identity}

    class Partial(Racing):
        def get_recommendation(self, identity):
            return {winner.identity: winner}.get(identity) if self.raced else None

    class Complete(Racing):
        def get_recommendation(self, identity):
            stored = {winner.identity: winner, sibling.identity: sibling}
            return stored.get(identity) if self.raced else None

    def reconcile(store: Racing):
        use_case = _world(recommendations=store).use_case()
        store.raced = True
        use_case._store = _Refusing()
        return use_case._record(
            (winner, sibling), identities, _POLICY, _PROFILE, _audit_for(winner)
        )

    with pytest.raises(FuturesTradeRecommendationConflictError):
        reconcile(Partial(winner))
    assert reconcile(Complete(winner)) == (winner, sibling)


class _Refusing(FuturesTradeRecommendationStore):
    def record(self, recommendations, *, policy, profile, audit):
        raise FuturesTradeRecommendationConflictError("synthetic batch race")


def test_a_store_conflict_propagates_and_nothing_is_recorded() -> None:
    world = _world()

    class Conflicting(Recommendations):
        def record(self, recommendations, *, policy, profile, audit):
            raise FuturesTradeRecommendationConflictError("synthetic conflict")

    world.recommendations = Conflicting()
    with pytest.raises(FuturesTradeRecommendationConflictError):
        _run(world)
    assert world.recommendations.recorded == {}


def test_a_store_that_accepts_a_partial_batch_is_a_contract_violation() -> None:
    world = _world()
    world.recommendations.answer = 0

    with pytest.raises(FuturesTradeRecommendationContractViolationError):
        _run(world)


def test_a_calendar_that_cannot_cover_the_horizon_propagates_its_error() -> None:
    profile = FuturesTradeRecommendationGenerationProfile("p", 70, _POLICY.version)

    with pytest.raises(FuturesSessionResolutionError):
        _run(_world(), profile=profile)


# ---------------------------------------------------------------------------
# Determinism, purity and boundaries
# ---------------------------------------------------------------------------


def test_the_same_inputs_give_the_same_result() -> None:
    assert _run(_world()) == _run(_world())


def test_inputs_are_not_mutated() -> None:
    world = _world()
    before = (repr(world.bars), repr(world.records))

    _run(world)

    assert (repr(world.bars), repr(world.records)) == before


_NEW_MODULES = (
    "futures_recommendation_generation_audit",
    "futures_recommendation_generation_outcome",
    "futures_recommendation_identities",
    "futures_recommendation_profile",
    "futures_recommendation_session_windows",
    "generate_futures_trade_recommendations",
)


@pytest.mark.parametrize("module", _NEW_MODULES)
def test_new_modules_have_no_outward_execution_or_clock_dependency(module: str) -> None:
    path = Path(services.__file__).parent / f"{module}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    for name in imported:
        assert not name.startswith(
            (
                "northstar_core.paper_trading",
                "northstar_core.broker_execution",
                "northstar_infrastructure",
                "northstar_api",
                "sqlite3",
                "upstox",
                "requests",
                "httpx",
                "random",
                "uuid",
                "time",
            )
        ), f"{module} imports {name}"
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not calls & {"now", "utcnow", "today", "time", "uuid4", "random"}


def test_public_exports() -> None:
    for name in (
        "GenerateFuturesTradeRecommendationsUseCase",
        "FuturesTradeRecommendationGenerationResult",
        "FuturesTradeRecommendationGenerationStatus",
        "FuturesTradeRecommendationBlockReason",
        "FuturesResearchHistoryCoverage",
        "FuturesTradeRecommendationContractViolationError",
        "FuturesTradeRecommendationSourceError",
        "InvalidFuturesTradeRecommendationGenerationError",
        "FuturesRecommendationSessionWindowResolver",
        "FuturesRecommendationWindowResolutionError",
    ):
        assert name in services.__all__


# ---------------------------------------------------------------------------
# Generation audits (F-5B-2B)
# ---------------------------------------------------------------------------

Outcome = FuturesRecommendationGenerationAuditOutcome


def _only_audit(world: World) -> FuturesRecommendationGenerationAudit:
    (audit,) = world.recommendations.audits.values()
    return audit


@pytest.mark.parametrize("action, direction", [("BUY", LONG), ("SELL", SHORT)])
def test_a_new_recommendation_commits_with_the_audit_of_exactly_its_result(
    action: str, direction: TradeDirection
) -> None:
    world = _world(action)

    result = _run(world)

    audit = _only_audit(world)
    (recommendation,) = result.recorded
    assert recommendation.direction is direction
    assert audit == FuturesRecommendationGenerationAudit.of(result)
    assert audit.outcome is Outcome.RECORDED
    assert audit.proposed == audit.recorded == (recommendation.identity,)
    assert audit.existing == ()
    assert audit.candidates == (recommendation.source,)
    assert audit.history_coverage is FuturesResearchHistoryCoverage.VERIFIED
    assert audit.operator_review is FuturesCandleReviewAssessment.REVIEWED
    assert audit.finality == result.finality and audit.finality.is_final
    assert audit.session == result.session
    assert (audit.contract, audit.strategy, audit.attempted_at) == (_OCT, _STRATEGY, _ISSUED)
    assert (audit.policy_version, audit.profile_version) == (_POLICY.version, _PROFILE.version)


def test_hold_is_audited_as_an_f3_rejection() -> None:
    world = _world("HOLD")

    _run(world)

    audit = _only_audit(world)
    assert audit.outcome is Outcome.REJECTED
    (rejection,) = audit.candidate_rejections
    assert rejection.reason is CandidateRejectionReason.NO_DIRECTIONAL_VIEW
    assert rejection.research_action == RecommendationAction("HOLD")
    assert audit.candidates == audit.proposed == audit.recorded == audit.existing == ()
    assert world.recommendations.writes == []


def test_an_f4_rejection_is_audited_and_nothing_is_recorded() -> None:
    world = _world(cutoffs={_OCT: FuturesContractCutoff(_OCT, _session(date(2026, 10, 12)), "s")})

    result = _run(world)

    audit = _only_audit(world)
    assert audit.outcome is Outcome.REJECTED
    assert [r.reason for r in audit.generation_rejections] == [
        r.reason for r in result.generation.rejections
    ]
    assert audit.proposed == audit.recorded == ()


def test_an_open_recommendation_on_another_expiry_is_audited_with_its_breach() -> None:
    world = _world(
        recommendations=Recommendations((_open(LONG, _NOV, "a"), _open(SHORT, _NOV, "b")))
    )

    result = _run(world)

    audit = _only_audit(world)
    assert audit.outcome is Outcome.REJECTED
    assert audit.exclusivity_breaches == result.generation.exclusivity_breaches == (_NIFTY,)
    assert audit.generation_rejections


@pytest.mark.parametrize(
    "world, reviewed, reason, review",
    [
        (lambda: _world(bars=(), records=()), None, Block.NO_PERSISTED_SESSION, "NOT_ASSESSED"),
        (
            lambda: _world(final_through=date(2026, 10, 6)),
            "all",
            Block.SESSION_NOT_FINAL,
            "NOT_ASSESSED",
        ),
        (lambda: _world(), None, Block.SESSION_NOT_OPERATOR_REVIEWED, "NOT_REVIEWED"),
        (lambda: _world(records=()), "all", Block.NO_RESEARCH_FOR_SESSION, "REVIEWED"),
        (lambda: _world(cutoffs={}), "all", Block.CONTRACT_CUTOFF_UNAVAILABLE, "REVIEWED"),
    ],
    ids=["no-session", "not-final", "not-reviewed", "no-research", "no-cutoff"],
)
def test_a_blocked_attempt_is_audited_with_its_reason(
    world, reviewed: object, reason: Block, review: str
) -> None:
    world = world()

    result = _run(world, reviewed=reviewed)  # type: ignore[arg-type]

    audit = _only_audit(world)
    assert result.block_reason is reason
    assert audit == FuturesRecommendationGenerationAudit.of(result)
    assert audit.outcome is Outcome.BLOCKED and audit.block_reason is reason
    assert audit.operator_review.value == review
    assert audit.session == result.session and audit.finality == result.finality
    assert audit.proposed == audit.recorded == ()
    assert world.recommendations.writes == []


def test_unverified_history_is_audited_with_its_deficiency_and_sessions() -> None:
    world = _world()
    world.bars = _without(world.bars, date(2026, 10, 1))

    result = _run(world, reviewed=_approvals(_world().bars))

    audit = _only_audit(world)
    assert audit.outcome is Outcome.BLOCKED
    assert audit.block_reason is Block.RESEARCH_HISTORY_NOT_VERIFIED
    assert audit.history_coverage is FuturesResearchHistoryCoverage.UNVERIFIED
    assert audit.history_deficiency is FuturesResearchHistoryDeficiency.MISSING_BAR
    assert audit.history_deficient_sessions == result.history_deficient_sessions
    assert audit.history_deficient_sessions == (date(2026, 10, 1),)


@pytest.mark.parametrize("retry_at", [_ISSUED, PointInTime("2026-10-07T13:00:00Z")])
def test_a_retry_is_audited_as_existing_and_never_as_recorded(retry_at: PointInTime) -> None:
    world = _world()
    (recorded,) = _run(world).recorded
    first = _only_audit(world)

    retry = _run(world, issued_at=retry_at)

    assert retry.existing == (recorded,) and retry.recorded == ()
    audits = world.recommendations.audits
    assert sorted(a.outcome.value for a in audits.values()) == ["EXISTING_RETURNED", "RECORDED"]
    assert audits[first.identity] == first
    (again,) = (a for a in audits.values() if a.outcome is Outcome.EXISTING_RETURNED)
    assert again.identity != first.identity
    assert again.attempted_at == retry_at
    assert again.proposed == again.recorded == () and again.existing == (recorded.identity,)
    assert world.recommendations.recorded == {recorded.identity: recorded}


def test_a_lost_race_is_audited_as_existing_with_its_unrecorded_proposal() -> None:
    winner = _recorded(_T1)
    world = _world(recommendations=Racing(winner))

    result = _run(world, issued_at=_T2)

    audit = _only_audit(world)
    assert audit == FuturesRecommendationGenerationAudit.of(result)
    assert audit.outcome is Outcome.EXISTING_RETURNED
    assert audit.proposed == (winner.identity,) and audit.recorded == ()
    assert audit.existing == (winner.identity,)
    assert audit.attempted_at == _T2


def test_replaying_a_blocked_attempt_records_its_audit_once() -> None:
    world = _world()

    _run(world, reviewed=None)
    _run(world, reviewed=None)

    assert world.recommendations.outcomes() == ["BLOCKED"]


def test_one_attempt_instant_reaching_different_content_is_an_audit_conflict() -> None:
    world = _world()
    approvals = _approvals(_world().bars)
    world.bars = _without(world.bars, date(2026, 10, 1))
    _run(world, reviewed=approvals)
    world.bars = _without(world.bars, date(2026, 9, 30))

    with pytest.raises(FuturesRecommendationGenerationAuditConflictError):
        _run(world, reviewed=approvals)
    _run(world, reviewed=approvals, issued_at=PointInTime("2026-10-07T12:30:00Z"))
    assert len(world.recommendations.audits) == 2


def test_a_failed_recording_records_no_audit() -> None:
    world = _world()

    class Failing(Recommendations):
        def record(self, recommendations, *, policy, profile, audit):
            raise RuntimeError("synthetic storage failure")

    world.recommendations = Failing()
    with pytest.raises(RuntimeError):
        _run(world)
    assert world.recommendations.audits == {} and world.recommendations.recorded == {}


def test_an_unreconciled_store_conflict_records_no_audit() -> None:
    world = _world()

    class Conflicting(Recommendations):
        def record(self, recommendations, *, policy, profile, audit):
            raise FuturesTradeRecommendationExclusivityError("synthetic concurrent writer")

    world.recommendations = Conflicting()
    with pytest.raises(FuturesTradeRecommendationExclusivityError):
        _run(world)
    assert world.recommendations.audits == {}


def test_contradictory_evidence_raises_and_is_never_audited_as_a_rejection() -> None:
    bars = _bars()
    world = _world(bars=bars, records=(_record(bars, latest=Decimal("24380")),))

    with pytest.raises(FuturesTradeRecommendationSourceError):
        _run(world)
    assert world.recommendations.audits == {}


def test_a_failing_audit_store_propagates_and_records_nothing() -> None:
    world = _world()

    class AuditFailing(Recommendations):
        def record_audit(self, audit):
            raise RuntimeError("synthetic audit failure")

    world.recommendations = AuditFailing()
    with pytest.raises(RuntimeError):
        _run(world, reviewed=None)
    assert world.recommendations.recorded == {}


def test_the_use_case_requires_an_audit_store() -> None:
    world = _world()
    store = world.recommendations

    with pytest.raises(TypeError, match="audits"):
        GenerateFuturesTradeRecommendationsUseCase(
            Research(world.records),
            Market(world.bars),
            world.calendar,
            DisabledFuturesDailyBarFinalityPolicy(),
            Cutoffs(world.cutoffs),
            store,
            store,
            object(),  # type: ignore[arg-type]
        )
