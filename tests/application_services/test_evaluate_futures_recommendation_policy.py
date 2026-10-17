"""Tests for the pure historical futures recommendation evaluator (R-0).

Everything here is SYNTHETIC: the calendar (weekday sessions 03:45-10:00Z,
holidays 2026-10-02 and 10-20, only 2026 loaded), the bars, the research
actions (a scripted analysis generator), the cutoffs, the policy and the
profile. None is market data, and nothing here says anything about real
profitability.

The default world: NIFTY@NSE 2026-10-27, twenty-five closes ending at the
2026-10-07 session (23900 ... 24383, the generation fixture's), on which F-4
yields LONG entry 24378-24388, stop 24347.75, target 24448.45 from a BUY.
Issued at that close with M = 5, the window is the 10-08 .. 10-14 closes.
"""

from __future__ import annotations

import ast
import dataclasses
from datetime import date, timedelta
from decimal import Decimal
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
    FuturesRecommendationGenerationPolicy,
    FuturesResearchDecisionReference,
    GenerationRejectionReason,
    IntrabarOrdering,
    RecommendationPolicyVersion,
    RecommendationStatus,
    TradeDirection,
)
from northstar_core.strategy import (
    FuturesAssetAnalysis,
    FuturesAssetAnalysisGenerator,
    FuturesMarketObservationContext,
    Strategy,
    StrategyIdentity,
)

import northstar_application.application_services as services
from northstar_application.application_services import (
    EvaluateFuturesRecommendationPolicyUseCase,
    FuturesEvaluatedLifecycleState,
    FuturesEvaluationEvidence,
    FuturesEvaluationEvidenceClassification,
    FuturesEvaluationSessionBlock,
    FuturesRecommendationEvaluationEvidenceError,
    FuturesRecommendationEvaluationRequest,
    FuturesRecommendationSessionWindowResolver,
    FuturesTradeRecommendationContractViolationError,
    FuturesTradeRecommendationGenerationProfile,
    FuturesTradeRecommendationIdentityService,
    InvalidFuturesRecommendationEvaluationError,
)
from northstar_application.ports import (
    FuturesContractCutoff,
    FuturesContractCutoffSource,
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
    FuturesSessionResolutionError,
    FuturesTradingSession,
    FuturesTradingSessionResolver,
)

State = FuturesEvaluatedLifecycleState
Status = RecommendationStatus
LONG, SHORT = TradeDirection.LONG, TradeDirection.SHORT

_NIFTY = FuturesProductReference(Symbol("NIFTY"), ExchangeCode("NSE"))
_OCT = FuturesContract(_NIFTY, ExpirationDate("2026-10-27"))
_NOV = FuturesContract(_NIFTY, ExpirationDate("2026-11-24"))
_STRATEGY = StrategyIdentity("synthetic-directional-v1")
_DAILY = Timeframe("1d")
_D = date(2026, 10, 7)
_WINDOW = (date(2026, 10, 8), date(2026, 10, 9), date(2026, 10, 12), date(2026, 10, 13),
           date(2026, 10, 14))  # fmt: skip

_RISING = tuple(Decimal(24000 + 20 * step) for step in range(19)) + (Decimal("24383"),)
_WARMUP = (Decimal("23900"), Decimal("23950"), Decimal("23925"), Decimal("23975"), Decimal("23990"))

_QUIET = ("24400", "24420")
_ENTRY = ("24380", "24400")
_STOP = ("24340", "24390")
_TARGET = ("24400", "24450")
_BOTH = ("24340", "24450")


class Calendar(FuturesTradingSessionResolver):
    def __init__(self, years: frozenset[int] = frozenset({2026})) -> None:
        self.years = years

    def resolve(self, product, trading_date):
        if trading_date.year not in self.years:
            raise FuturesSessionResolutionError(f"Calendar year {trading_date.year} is not loaded.")
        if trading_date.weekday() >= 5 or trading_date in {date(2026, 10, 2), date(2026, 10, 20)}:
            return None
        day = trading_date.isoformat()
        return FuturesTradingSession(
            trading_date, PointInTime(f"{day}T03:45:00Z"), PointInTime(f"{day}T10:00:00Z")
        )

    def sessions_in_range(self, product, start_date, end_date):
        found, day = [], start_date
        while day <= end_date:
            if (session := self.resolve(product, day)) is not None:
                found.append(session)
            day += timedelta(days=1)
        return tuple(found)


_CALENDAR = Calendar()


def _session(day: date) -> FuturesTradingSession:
    session = _CALENDAR.resolve(_NIFTY, day)
    assert session is not None
    return session


def _close(day: date) -> PointInTime:
    return _session(day).closes_at


def _q(value) -> QuoteValue:
    return QuoteValue(Decimal(value))


def _bar(contract: FuturesContract, day: date, low, high, close=None) -> FuturesOHLCVBar:
    close = Decimal(close) if close is not None else (Decimal(low) + Decimal(high)) / 2
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=_close(day),
        timeframe=_DAILY,
        open=_q(close),
        high=_q(high),
        low=_q(low),
        close=_q(close),
        volume=Quantity(Decimal("1000")),
    )


def _history(contract: FuturesContract = _OCT, closes=_WARMUP + _RISING, last: date = _D):
    sessions = _CALENDAR.sessions_in_range(_NIFTY, last - timedelta(days=60), last)[-len(closes) :]
    return [
        _bar(contract, s.trading_date, c - 5, c + 5, c)
        for s, c in zip(sessions, closes, strict=True)
    ]


class Market(FuturesHistoricalMarketDataRepository):
    def __init__(self, bars) -> None:
        self.bars = tuple(sorted(bars, key=lambda b: (str(b.contract), b.point_in_time.value)))
        self.queries: list[FuturesHistoricalMarketDataQuery] = []

    def get_bars(self, query):
        self.queries.append(query)
        return tuple(
            bar
            for bar in self.bars
            if bar.contract == query.contract
            and bar.timeframe == query.timeframe
            and query.covers(bar.point_in_time)
        )


class Cutoffs(FuturesContractCutoffSource):
    def __init__(self, known: dict) -> None:
        self.known = known

    def resolve(self, contract):
        return self.known.get(contract)


_CUTOFFS = {
    _OCT: FuturesContractCutoff(_OCT, _session(date(2026, 10, 27)), "synthetic reference"),
    _NOV: FuturesContractCutoff(_NOV, _session(date(2026, 11, 23)), "synthetic reference"),
}


class Scripted(FuturesAssetAnalysisGenerator):
    """BUY or SELL exactly where scripted, HOLD everywhere else; records every context."""

    def __init__(self, actions: dict) -> None:
        self.actions = actions
        self.contexts: list[FuturesMarketObservationContext] = []

    def generate(self, context):
        self.contexts.append(context)
        day = date.fromisoformat(context.observed_at.value[:10])
        action = self.actions.get((context.contract, day))
        signal = {"BUY": "strong bullish", "SELL": "strong bearish"}.get(action, "neutral trend")
        return FuturesAssetAnalysis(context.contract, context.observed_at, (signal,))


def _policy(**overrides) -> FuturesRecommendationGenerationPolicy:
    values = {
        "version": RecommendationPolicyVersion("f4-synthetic-evaluation-v1"),
        "approved_strategies": (_STRATEGY,),
        "lookback_changes": 19,
        "entry_multiplier": Decimal("0.25"),
        "stop_multiplier": Decimal("1.5"),
        "target_multiplier": Decimal("3"),
        "price_increment": Decimal("0.05"),
        "min_reward_to_risk": Decimal("1.5"),
        "short_enabled": True,
    }
    values.update(overrides)
    return FuturesRecommendationGenerationPolicy(**values)  # type: ignore[arg-type]


_POLICY = _policy()
_PROFILE = FuturesTradeRecommendationGenerationProfile(
    "r0-synthetic-profile-v1", 5, _POLICY.version
)
_EVIDENCE = FuturesEvaluationEvidence(
    FuturesEvaluationEvidenceClassification.SYNTHETIC, "r0-synthetic-fixture-v1"
)


@dataclasses.dataclass
class World:
    bars: list
    actions: dict
    contracts: tuple = (_OCT,)
    cutoffs: dict = dataclasses.field(default_factory=lambda: dict(_CUTOFFS))
    calendar: FuturesTradingSessionResolver = _CALENDAR
    policy: FuturesRecommendationGenerationPolicy = _POLICY
    profile: FuturesTradeRecommendationGenerationProfile = _PROFILE

    def run(self, start: date = _D, end: date = date(2026, 10, 16)):
        self.market = Market(self.bars)
        self.generator = Scripted(self.actions)
        return EvaluateFuturesRecommendationPolicyUseCase(
            self.market, self.calendar, Cutoffs(self.cutoffs), self.generator
        ).execute(
            FuturesRecommendationEvaluationRequest(
                product=_NIFTY,
                contracts=self.contracts,
                strategy=Strategy(_STRATEGY),
                policy=self.policy,
                profile=self.profile,
                start=_close(start),
                end=_close(end),
                evidence=_EVIDENCE,
            )
        )


def _world(*after, action="BUY", contract=_OCT, **overrides) -> World:
    """History to 10-07, a scripted action there, then ``after`` (low, high) per window day."""
    bars = _history(contract) + [
        _bar(contract, day, *prices) for day, prices in zip(_WINDOW, after, strict=False) if prices
    ]
    actions = {(contract, _D): action} if action else {}
    return World(bars=bars, actions=actions, contracts=(contract,), **overrides)


def _only(result):
    (record,) = result.recommendations
    return record


# ---------------------------------------------------------------------------
# A. F-3 and F-4 reuse
# ---------------------------------------------------------------------------


def test_buy_research_becomes_the_production_long_recommendation() -> None:
    result = _world(_QUIET).run(end=_WINDOW[0])

    record = _only(result)
    recommendation = record.recommendation
    assert recommendation.direction is LONG
    assert (recommendation.entry.low, recommendation.entry.high) == (_q("24378"), _q("24388"))
    assert (recommendation.stop.price, recommendation.target.price) == (
        _q("24347.75"),
        _q("24448.45"),
    )
    source = FuturesResearchDecisionReference(_OCT, _DAILY, _close(_D), _STRATEGY)
    assert recommendation.source == source
    assert recommendation.identity == FuturesTradeRecommendationIdentityService().identity(
        source, _PROFILE
    )
    assert recommendation.issued_at == _close(_D)
    assert result.sessions[0].recommended == (recommendation.identity,)
    assert result.generated_count == 1


def test_sell_research_becomes_a_short_when_enabled() -> None:
    bars = _history(closes=_WARMUP + tuple(reversed(_RISING)))
    world = World(bars=bars, actions={(_OCT, _D): "SELL"})

    assert _only(world.run(end=_D)).recommendation.direction is SHORT


def test_short_not_enabled_is_preserved() -> None:
    bars = _history(closes=_WARMUP + tuple(reversed(_RISING)))
    world = World(bars=bars, actions={(_OCT, _D): "SELL"}, policy=_policy(short_enabled=False))
    world.profile = _PROFILE

    result = world.run(end=_D)

    assert result.recommendations == ()
    assert result.generation_rejections == ((GenerationRejectionReason.SHORT_NOT_ENABLED, 1),)


def test_hold_is_an_f3_rejection_and_generates_nothing() -> None:
    result = _world(action=None).run(end=_D)

    assert result.recommendations == ()
    assert result.candidate_rejections == ((CandidateRejectionReason.NO_DIRECTIONAL_VIEW, 1),)
    assert result.sessions[0].generation is None


@pytest.mark.parametrize(
    "overrides, reason",
    [
        ({"entry_multiplier": Decimal("0"), "price_increment": Decimal("10")},
         GenerationRejectionReason.DEGENERATE_ENTRY_ZONE),
        ({"min_reward_to_risk": Decimal("100")},
         GenerationRejectionReason.INSUFFICIENT_REWARD_TO_RISK),
    ],
    ids=["entry-zone", "reward-to-risk"],
)  # fmt: skip
def test_f4_rejections_are_preserved(overrides, reason) -> None:
    policy = _policy(**overrides)
    world = _world(policy=policy)

    result = world.run(end=_D)

    assert result.recommendations == ()
    assert result.generation_rejections == ((reason, 1),)


def test_the_nearest_eligible_contract_is_selected() -> None:
    bars = _history(_OCT) + _history(_NOV)
    world = World(bars=bars, actions={(_OCT, _D): "BUY", (_NOV, _D): "BUY"}, contracts=(_NOV, _OCT))

    result = world.run(end=_D)

    assert _only(result).recommendation.contract == _OCT
    assert result.generation_rejections == (
        (GenerationRejectionReason.NOT_NEAREST_ELIGIBLE_CONTRACT, 1),
    )


def test_cross_expiry_contradiction_is_preserved() -> None:
    bars = _history(_OCT) + _history(_NOV)
    world = World(
        bars=bars, actions={(_OCT, _D): "BUY", (_NOV, _D): "SELL"}, contracts=(_OCT, _NOV)
    )

    result = world.run(end=_D)

    assert result.recommendations == ()
    assert result.generation_rejections == (
        (GenerationRejectionReason.CONTRADICTORY_PRODUCT_DIRECTION, 2),
    )


# ---------------------------------------------------------------------------
# B. Timing
# ---------------------------------------------------------------------------


def test_the_decision_session_never_evaluates_its_own_recommendation() -> None:
    # The decision bar (10-07: 24378 .. 24388) touches the entry zone itself.
    record = _only(_world().run(end=_D))

    assert record.observations == 0 and record.status is Status.WAITING_FOR_ENTRY
    assert record.state is State.OPEN_HORIZON_ENDED


def test_the_window_is_the_resolvers_and_starts_after_issuance() -> None:
    record = _only(_world(_QUIET).run(end=_WINDOW[0]))
    window = FuturesRecommendationSessionWindowResolver(_CALENDAR).resolve(
        _OCT, _close(_D), _close(_D), _PROFILE, _CUTOFFS[_OCT]
    )

    validity = record.recommendation.validity
    assert (validity.eligible_from, validity.valid_through) == (
        window.eligible_from,
        window.valid_through,
    )
    assert validity.eligible_from == _close(_WINDOW[0])
    assert validity.valid_through == _close(_WINDOW[-1])
    assert record.recommendation.issued_at.compare(validity.eligible_from) < 0


def test_insufficient_contract_life_rejects_generation() -> None:
    cutoffs = {_OCT: FuturesContractCutoff(_OCT, _session(date(2026, 10, 12)), "synthetic")}

    result = _world(cutoffs=cutoffs).run(end=_D)

    assert result.generation_rejections == (
        (GenerationRejectionReason.INSUFFICIENT_CONTRACT_LIFE, 1),
    )


def test_an_unloaded_calendar_year_fails_closed() -> None:
    world = _world(calendar=Calendar(years=frozenset({2025})))

    with pytest.raises(FuturesSessionResolutionError):
        world.run(end=_D)


# ---------------------------------------------------------------------------
# C. Lifecycle replay through Core's advance
# ---------------------------------------------------------------------------


_END = date(2026, 10, 16)


@pytest.mark.parametrize(
    "after, end, status, ordering, observations",
    [
        ((_ENTRY,), _WINDOW[0], Status.ACTIVE, None, 1),
        ((_ENTRY, _STOP), _END, Status.CLOSED_STOP, IntrabarOrdering.NOT_REQUIRED, 2),
        ((_ENTRY, _TARGET), _END, Status.CLOSED_TARGET, IntrabarOrdering.NOT_REQUIRED, 2),
        ((_ENTRY, _BOTH), _END, Status.CLOSED_STOP, IntrabarOrdering.MODELED_STOP_FIRST, 2),
        ((_STOP,), _END, Status.INDETERMINATE_ENTRY, IntrabarOrdering.UNKNOWN, 1),
        ((_QUIET,) * 5, _END, Status.EXPIRED_BEFORE_ENTRY, IntrabarOrdering.NOT_REQUIRED, 5),
        ((_QUIET, _QUIET, _QUIET, _ENTRY, _QUIET), _END, Status.EXPIRED_AFTER_ACTIVATION, None, 5),
    ],
    ids=["active", "stop", "target", "stop-first", "indeterminate", "no-entry", "after-entry"],
)
def test_lifecycle_outcomes_are_cores(after, end, status, ordering, observations) -> None:
    record = _only(_world(*after).run(end=end))

    assert record.status is status and record.observations == observations
    if ordering is not None:
        assert record.lifecycle.events[-1].ordering is ordering
    expected = State.TERMINAL if status.is_terminal else State.OPEN_HORIZON_ENDED
    assert record.state is expected
    # Exactly Core's own transitions over the same bars, in the same order.
    lifecycle = type(record.lifecycle)(record.recommendation)
    for day, prices in zip(_WINDOW, after, strict=False):
        if not lifecycle.is_terminal:
            lifecycle = lifecycle.advance(_bar(_OCT, day, *prices))
    assert lifecycle.events == record.lifecycle.events


def test_quiet_observations_produce_no_events() -> None:
    record = _only(_world(_QUIET, _QUIET).run(end=_WINDOW[1]))

    assert record.observations == 2 and record.lifecycle.events == ()
    assert record.lifecycle.last_observation.point_in_time == _close(_WINDOW[1])
    assert record.state is State.OPEN_HORIZON_ENDED


def test_a_gap_stops_the_lifecycle_and_is_reported() -> None:
    result = _world(_QUIET, None, _ENTRY, _TARGET).run()

    record = _only(result)
    assert record.state is State.OPEN_EVIDENCE_GAP
    assert record.gap_session == _session(_WINDOW[1])
    assert record.observations == 1 and record.status is Status.WAITING_FOR_ENTRY
    assert result.state_distribution == ((State.OPEN_EVIDENCE_GAP, 1),)
    assert result.status_distribution == ((Status.WAITING_FOR_ENTRY, 1),)
    assert result.open_recommendations == (record,)


# ---------------------------------------------------------------------------
# D. P-1 over historical time
# ---------------------------------------------------------------------------


def _later_buy(world: World, contract=_OCT, day=_WINDOW[1], action="BUY") -> World:
    world.actions[(contract, day)] = action
    return world


def test_an_open_recommendation_blocks_a_later_same_contract_candidate() -> None:
    world = _later_buy(_world(_ENTRY, _QUIET))

    result = world.run(end=_WINDOW[1])

    assert result.generated_count == 1
    assert (GenerationRejectionReason.DUPLICATE_OPEN_RECOMMENDATION, 1) in (
        result.generation_rejections
    )


def test_another_expiry_cannot_bypass_the_open_recommendation() -> None:
    world = _world(_ENTRY, _QUIET)
    world.bars += _history(_NOV, last=_WINDOW[1])
    world.contracts = (_OCT, _NOV)
    world.actions[(_NOV, _WINDOW[1])] = "BUY"

    result = world.run(end=_WINDOW[1])

    assert result.generated_count == 1
    assert result.generation_rejections == (
        (GenerationRejectionReason.DUPLICATE_OPEN_RECOMMENDATION, 1),
    )


def test_the_opposite_direction_cannot_bypass_it_either() -> None:
    world = _later_buy(_world(_ENTRY, _QUIET), action="SELL")

    result = world.run(end=_WINDOW[1])

    assert result.generation_rejections == (
        (GenerationRejectionReason.CONTRADICTS_OPEN_RECOMMENDATION, 1),
    )


def test_a_terminal_lifecycle_releases_the_product_monitor_before_generate() -> None:
    # The stop is taken at 10-09; the BUY decided at that same close is then free.
    world = _later_buy(_world(_ENTRY, _STOP))

    result = world.run(end=_WINDOW[1])

    first, second = sorted(result.recommendations, key=lambda r: r.recommendation.issued_at.value)
    assert first.status is Status.CLOSED_STOP
    assert second.recommendation.source.decision_instant == _close(_WINDOW[1])
    assert result.sessions[2].recommended == (second.identity,)


def test_a_gap_never_releases_the_product() -> None:
    world = _world(_QUIET, None)
    world.bars += _history(_NOV, last=_WINDOW[2])
    world.contracts = (_OCT, _NOV)
    world.actions[(_NOV, _WINDOW[2])] = "BUY"

    result = world.run(end=_WINDOW[2])

    assert result.generated_count == 1
    assert (GenerationRejectionReason.DUPLICATE_OPEN_RECOMMENDATION, 1) in (
        result.generation_rejections
    )


# ---------------------------------------------------------------------------
# E. Look-ahead and replay boundaries
# ---------------------------------------------------------------------------


def test_every_decision_sees_only_bars_up_to_its_own_close() -> None:
    world = _world(_ENTRY, _TARGET, _QUIET)
    world.run()

    assert world.generator.contexts
    for context in world.generator.contexts:
        visible = sorted(
            (b for b in world.bars if b.point_in_time.compare(context.observed_at) <= 0),
            key=lambda b: b.point_in_time.value,
        )
        assert context.recent_closes == tuple(b.close for b in visible[-20:])
        assert context.latest_quote == visible[-1].close


def test_later_bars_never_change_the_recommendation_issued_earlier() -> None:
    early = _only(_world().run(end=_D)).recommendation
    late = _only(_world(_ENTRY, _TARGET, _QUIET, _QUIET).run(end=date(2026, 10, 16)))

    assert late.recommendation == early


def test_later_bars_may_advance_the_lifecycle() -> None:
    world = _world(_ENTRY, _STOP)

    assert _only(world.run(end=_WINDOW[0])).status is Status.ACTIVE
    assert _only(world.run(end=_WINDOW[1])).status is Status.CLOSED_STOP


def test_no_bar_after_the_end_of_evidence_is_read() -> None:
    world = _world(_ENTRY, _STOP)

    record = _only(world.run(end=_WINDOW[0]))

    assert record.status is Status.ACTIVE
    assert all(
        query.end is not None and query.end.compare(_close(_WINDOW[0])) <= 0
        for query in world.market.queries
    )


def test_a_listed_contract_without_stored_research_is_never_a_candidate() -> None:
    world = _world()
    world.contracts = (_OCT, _NOV)  # NOV is in the universe, but has no bar at all

    result = world.run(end=_D)

    assert _only(result).recommendation.contract == _OCT
    assert result.research_decision_count == 1


def test_missing_cutoff_evidence_fails_closed() -> None:
    world = _world(cutoffs={})

    with pytest.raises(FuturesRecommendationEvaluationEvidenceError, match="expiration"):
        world.run(end=_D)


def test_a_cutoff_for_another_contract_is_a_violation() -> None:
    world = _world(cutoffs={_OCT: _CUTOFFS[_NOV]})

    with pytest.raises(FuturesTradeRecommendationContractViolationError):
        world.run(end=_D)


def test_incomplete_research_history_blocks_generation() -> None:
    world = _world()
    missing = world.bars[10].point_in_time
    world.bars = [b for b in world.bars if b.point_in_time != missing]

    result = world.run(end=_D)

    (session,) = [s for s in result.sessions if s.session.closes_at == _close(_D)]
    assert session.block is FuturesEvaluationSessionBlock.HISTORY_INCOMPLETE
    assert session.history_deficient_sessions == (date.fromisoformat(missing.value[:10]),)
    assert result.recommendations == () and result.blocked_session_count == 1


def test_a_bar_on_no_calendar_session_is_contradictory() -> None:
    world = _world()
    saturday = date(2026, 10, 3)
    world.bars.append(
        dataclasses.replace(
            world.bars[-1], point_in_time=PointInTime(f"{saturday.isoformat()}T10:00:00Z")
        )
    )

    with pytest.raises(FuturesRecommendationEvaluationEvidenceError):
        world.run(end=_D)


# ---------------------------------------------------------------------------
# F. Determinism
# ---------------------------------------------------------------------------


def test_identical_inputs_give_identical_results() -> None:
    assert _world(_ENTRY, _TARGET).run() == _world(_ENTRY, _TARGET).run()


def test_contract_and_bar_order_never_matter() -> None:
    def world(contracts):
        w = World(
            bars=_history(_OCT) + _history(_NOV),
            actions={(_OCT, _D): "BUY", (_NOV, _D): "BUY"},
            contracts=contracts,
        )
        return w.run(end=_D)

    assert world((_OCT, _NOV)) == world((_NOV, _OCT))


def test_distributions_follow_enum_order() -> None:
    result = _world(_ENTRY, _TARGET).run()

    assert result.status_distribution == ((Status.CLOSED_TARGET, 1),)
    assert result.state_distribution == ((State.TERMINAL, 1),)


def test_the_result_is_immutable() -> None:
    result = _world().run(end=_D)

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.recommendations = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# G. Request validation and safety
# ---------------------------------------------------------------------------


def _request(**overrides) -> FuturesRecommendationEvaluationRequest:
    values = {
        "product": _NIFTY,
        "contracts": (_OCT,),
        "strategy": Strategy(_STRATEGY),
        "policy": _POLICY,
        "profile": _PROFILE,
        "start": _close(_D),
        "end": _close(_WINDOW[-1]),
        "evidence": _EVIDENCE,
    }
    values.update(overrides)
    return FuturesRecommendationEvaluationRequest(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides",
    [
        {"contracts": ()},
        {"contracts": (_OCT, _OCT)},
        {"contracts": [_OCT]},
        {"contracts": (FuturesContract(FuturesProductReference(Symbol("BANKNIFTY"),
                                                                ExchangeCode("NSE")),
                                        ExpirationDate("2026-10-27")),)},
        {"start": _close(_WINDOW[-1]), "end": _close(_D)},
        {"profile": FuturesTradeRecommendationGenerationProfile(
            "p", 5, RecommendationPolicyVersion("other"))},
        {"strategy": Strategy(StrategyIdentity("synthetic-other-v1"))},
        {"evidence": "synthetic"},
    ],
    ids=["no-contracts", "duplicate", "list", "other-product", "inverted-window",
         "profile-for-another-policy", "unapproved-strategy", "untyped-evidence"],
)  # fmt: skip
def test_an_invalid_request_is_refused(overrides) -> None:
    with pytest.raises(InvalidFuturesRecommendationEvaluationError):
        _request(**overrides)


@pytest.mark.parametrize(
    "classification, reference", [("SYNTHETIC", "x"), ("MARKET", "x"), (None, "x")]
)
def test_only_the_typed_synthetic_classification_is_accepted(classification, reference) -> None:
    with pytest.raises(InvalidFuturesRecommendationEvaluationError):
        FuturesEvaluationEvidence(classification, reference)  # type: ignore[arg-type]


def test_synthetic_evidence_is_never_market_evidence_and_must_be_named() -> None:
    assert _EVIDENCE.is_market_evidence is False
    with pytest.raises(InvalidFuturesRecommendationEvaluationError):
        FuturesEvaluationEvidence(FuturesEvaluationEvidenceClassification.SYNTHETIC, " ")


def test_the_evaluator_has_no_outward_dependency_or_clock() -> None:
    path = Path(services.__file__).parent / "evaluate_futures_recommendation_policy.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    } | {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    for name in imported:
        assert not name.startswith(
            ("sqlite3", "northstar_infrastructure", "northstar_api", "upstox", "requests", "httpx",
             "socket", "urllib", "time", "random", "uuid", "northstar_core.paper_trading",
             "northstar_core.broker_execution")
        ), name  # fmt: skip
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not calls & {"now", "utcnow", "today", "time", "uuid4", "random", "expire"}


def test_the_result_carries_no_fill_or_profit_figure() -> None:
    names = {
        field.name.lower()
        for cls in (
            services.FuturesRecommendationEvaluationResult,
            services.FuturesRecommendationEvaluationRecord,
            services.FuturesEvaluationSessionRecord,
        )
        for field in dataclasses.fields(cls)
    } | {
        name.lower()
        for name in dir(services.FuturesRecommendationEvaluationResult)
        if not name.startswith("_")
    }
    for fragment in ("pnl", "profit", "fill", "expectancy", "sharpe", "drawdown", "cost", "price"):
        assert not any(fragment in name for name in names), fragment


def test_public_exports() -> None:
    for name in (
        "EvaluateFuturesRecommendationPolicyUseCase",
        "FuturesRecommendationEvaluationRequest",
        "FuturesRecommendationEvaluationResult",
        "FuturesRecommendationEvaluationRecord",
        "FuturesEvaluationSessionRecord",
        "FuturesEvaluationEvidence",
        "FuturesEvaluationEvidenceClassification",
        "FuturesEvaluatedLifecycleState",
        "FuturesEvaluationSessionBlock",
        "FuturesRecommendationEvaluationEvidenceError",
        "InvalidFuturesRecommendationEvaluationError",
    ):
        assert name in services.__all__
