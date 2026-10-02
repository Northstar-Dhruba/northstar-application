"""Tests for the pre-expiry flatten guard and its paper-trading integration.

The calendar is a table-driven FuturesTradingSessionResolver mirroring the
NSEFuturesTradingSessionResolver reference data for the dates used here.
Application cannot import Infrastructure, so the real resolver is not used
directly; the table reproduces its answers, including the facts that make the
countdown non-trivial:

    2026-10-20 (Tue)   holiday, inside the NIFTY October countdown
    2026-11-24 (Tue)   holiday, so a 24 Nov expiration is not a session
    2026-01-26 (Mon)   holiday
    2026-02-01 (Sun)   special session (Union Budget), 03:30-10:00Z
    2026-11-08 (Sun)   Muhurat special session without notified timings: raises

NIFTY October 2026 expires on Tuesday 2026-10-27 (E). Its countdown runs:

    E-1 Mon 10-26   E-2 Fri 10-23   E-3 Thu 10-22   E-4 Wed 10-21
    E-5 Mon 10-19   (10-20 holiday) E-6 Fri 10-16   E-7 Thu 10-15   E-8 Wed 10-14

With K = 5 the flatten is decided at the E-6 close (Fri 10-16) and fills at the
E-5 open (Mon 10-19), across a weekend. Subtracting five calendar days from the
expiry would have pointed at 10-22 instead.
"""

from __future__ import annotations

import ast
import dataclasses
from datetime import date, timedelta
from decimal import Decimal
from functools import cmp_to_key
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
from northstar_core.paper_trading import (
    FuturesContractCount,
    FuturesPaperPortfolio,
    FuturesPosition,
    OrderSide,
    PaperPortfolioIdentity,
)
from northstar_core.strategy import (
    FuturesAssetAnalysis,
    FuturesMarketObservationContext,
    FuturesRecommendation,
    RecommendationAction,
    StrategyIdentity,
)

from northstar_application.application_services import (
    CalculateFuturesPaperTradingMetricsUseCase,
    CreateFuturesExecutionIntentUseCase,
    FuturesAnalysisResult,
    FuturesExecutionIntentDecision,
    FuturesExecutionIntentNoIntentReason,
    FuturesExpiryFlattenGuard,
    FuturesExpiryFlattenPolicy,
    FuturesExpiryWindowAssessment,
    FuturesExpiryWindowError,
    FuturesForwardResearchRecord,
    FuturesPaperExecutionIdentityService,
    FuturesPaperTradingDecisionResult,
    FuturesPaperTradingStrategyContractMetrics,
    InvalidFuturesExpiryFlattenPolicyError,
    RunFuturesPaperTradingDecisionUseCase,
    RunFuturesPaperTradingSessionUseCase,
    RunFuturesPaperTradingUseCase,
)
from northstar_application.ports import (
    FuturesForwardResearchRecordQuery,
    FuturesForwardResearchRecordRepository,
    FuturesForwardResearchRecordStore,
    FuturesHistoricalMarketDataRepository,
    FuturesPaperFillRepository,
    FuturesPaperFillStore,
    FuturesPaperOrderConflictError,
    FuturesPaperOrderRepository,
    FuturesPaperOrderStore,
    FuturesSessionResolutionError,
    FuturesTradingSession,
    FuturesTradingSessionResolver,
)

_NSE = ExchangeCode("NSE")
_NIFTY = FuturesProductReference(Symbol("NIFTY"), _NSE)
_BANKNIFTY = FuturesProductReference(Symbol("BANKNIFTY"), _NSE)

_OCT = FuturesContract(_NIFTY, ExpirationDate("2026-10-27"))
_NOV = FuturesContract(_NIFTY, ExpirationDate("2026-11-23"))
_NOV_HOLIDAY = FuturesContract(_NIFTY, ExpirationDate("2026-11-24"))
_BANK_OCT = FuturesContract(_BANKNIFTY, ExpirationDate("2026-10-27"))
# A hypothetical early-February expiry, only to count across the Budget session.
_FEB = FuturesContract(_NIFTY, ExpirationDate("2026-02-03"))

_E = date(2026, 10, 27)
_E1 = date(2026, 10, 26)
_E2 = date(2026, 10, 23)
_E3 = date(2026, 10, 22)
_E4 = date(2026, 10, 21)
_E5 = date(2026, 10, 19)
_E6 = date(2026, 10, 16)
_E7 = date(2026, 10, 15)
_E8 = date(2026, 10, 14)
_COUNTDOWN = (_E8, _E7, _E6, _E5, _E4, _E3, _E2, _E1, _E)

_DAILY = Timeframe("1d")
_STRATEGY = "futures-forward"
_PORTFOLIO = "nifty-paper-1"
_WINDOW = FuturesExecutionIntentNoIntentReason.EXPIRY_FLATTEN_WINDOW


# ---------------------------------------------------------------------------
# NSE calendar table (mirrors NSEFuturesTradingSessionResolver reference data)
# ---------------------------------------------------------------------------

_COVERED = ((date(2026, 1, 19), date(2026, 2, 6)), (date(2026, 9, 28), date(2026, 11, 30)))
_HOLIDAYS = frozenset(
    {
        date(2026, 1, 26),
        date(2026, 10, 2),
        date(2026, 10, 20),
        date(2026, 11, 10),
        date(2026, 11, 24),
    }
)
_SPECIAL = frozenset({date(2026, 2, 1)})
_UNNOTIFIED = frozenset({date(2026, 11, 8)})


def _window(day: date) -> tuple[str, str]:
    close = "10:00" if day.month <= 2 else "10:10"
    return f"{day.isoformat()}T03:30:00Z", f"{day.isoformat()}T{close}:00Z"


class NseCalendar(FuturesTradingSessionResolver):
    """Answers only from the table; uncovered or un-notified dates raise."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, date, date]] = []

    def _session(self, day: date) -> FuturesTradingSession | None:
        if not any(start <= day <= end for start, end in _COVERED):
            raise FuturesSessionResolutionError(f"{day} is not covered by reference data.")
        if day in _UNNOTIFIED:
            raise FuturesSessionResolutionError(f"Special session {day} has no notified timings.")
        if day in _SPECIAL or (day.weekday() < 5 and day not in _HOLIDAYS):
            opens, closes = _window(day)
            return FuturesTradingSession(day, PointInTime(opens), PointInTime(closes))
        return None

    def resolve(self, product, trading_date):
        self.calls.append(("resolve", trading_date, trading_date))
        return self._session(trading_date)

    def sessions_in_range(self, product, start_date, end_date):
        self.calls.append(("range", start_date, end_date))
        sessions = []
        for ordinal in range(start_date.toordinal(), end_date.toordinal() + 1):
            session = self._session(date.fromordinal(ordinal))
            if session is not None:
                sessions.append(session)
        return tuple(sessions)


def _close(day: date) -> PointInTime:
    return PointInTime(_window(day)[1])


def _guard(k: int = 5, resolver: FuturesTradingSessionResolver | None = None):
    return FuturesExpiryFlattenGuard(resolver or NseCalendar(), FuturesExpiryFlattenPolicy(k))


def _assess(day: date, contract: FuturesContract = _OCT, k: int = 5):
    return _guard(k).assess(contract, _close(day))


# ---------------------------------------------------------------------------
# Domain fixtures
# ---------------------------------------------------------------------------


def _quote(value: str) -> QuoteValue:
    return QuoteValue(Decimal(value))


def _record(
    action: str, day: date, contract: FuturesContract = _OCT
) -> FuturesForwardResearchRecord:
    observed_at = _close(day)
    context = FuturesMarketObservationContext(
        contract=contract,
        timeframe=_DAILY,
        observed_at=observed_at,
        latest_quote=_quote("25180"),
        previous_close=_quote("25150"),
        latest_volume=Quantity(Decimal("4200")),
        session_high=_quote("25250"),
        session_low=_quote("25020"),
        recent_closes=tuple(_quote(str(25000 + index)) for index in range(20)),
        recent_volumes=tuple(Quantity(Decimal(4000 + index)) for index in range(20)),
    )
    recommendation = FuturesRecommendation(
        action=RecommendationAction(action),
        asset_analysis=FuturesAssetAnalysis(contract, observed_at, ("signal",)),
        strategy_identity=StrategyIdentity(_STRATEGY),
        point_in_time=observed_at,
    )
    return FuturesForwardResearchRecord(FuturesAnalysisResult(recommendation, context))


def _bar(day: date, open_: int, contract: FuturesContract = _OCT) -> FuturesOHLCVBar:
    base = Decimal(open_)
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=_close(day),
        timeframe=_DAILY,
        open=QuoteValue(base),
        high=QuoteValue(base + 60),
        low=QuoteValue(base - 60),
        close=QuoteValue(base + 10),
        volume=Quantity(Decimal("5000")),
    )


def _portfolio(*holdings: tuple[FuturesContract, int], day: date = _E6) -> FuturesPaperPortfolio:
    positions = tuple(
        FuturesPosition(contract, net, _quote("25000"))
        for contract, net in sorted(holdings, key=lambda item: item[0].natural_key)
    )
    return FuturesPaperPortfolio(
        identity=PaperPortfolioIdentity(_PORTFOLIO),
        strategy_identity=StrategyIdentity(_STRATEGY),
        positions=positions,
        as_of=_close(day),
    )


def _policy(action: str, *holdings, day: date = _E6, target: int = 2, window: bool = True):
    record = _record(action, day)
    return CreateFuturesExecutionIntentUseCase().execute(
        record,
        _portfolio(*holdings, day=day),
        FuturesContractCount(target),
        expiry_window=_assess(day) if window else None,
    )


# ---------------------------------------------------------------------------
# 1. Policy validation
# ---------------------------------------------------------------------------


def test_k_five_is_a_valid_policy() -> None:
    assert FuturesExpiryFlattenPolicy(5).sessions_before_expiry == 5


@pytest.mark.parametrize("value", [0, -1, -5])
def test_k_below_one_is_rejected(value: int) -> None:
    with pytest.raises(InvalidFuturesExpiryFlattenPolicyError, match="at least 1"):
        FuturesExpiryFlattenPolicy(value)


@pytest.mark.parametrize("value", [True, False, 5.0, "5", None, Decimal("5")])
def test_a_non_integer_k_is_rejected(value: object) -> None:
    with pytest.raises(InvalidFuturesExpiryFlattenPolicyError, match="must be an integer"):
        FuturesExpiryFlattenPolicy(value)  # type: ignore[arg-type]


def test_the_policy_error_is_a_value_error_and_the_policy_is_immutable() -> None:
    assert issubclass(InvalidFuturesExpiryFlattenPolicyError, ValueError)
    with pytest.raises(dataclasses.FrozenInstanceError):
        FuturesExpiryFlattenPolicy(5).sessions_before_expiry = 3  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Session counting through the resolver only
# ---------------------------------------------------------------------------


def test_counting_skips_the_weekend_and_the_holiday() -> None:
    """E-8 (Wed 10-14) to Tue 10-27 is 13 calendar days but 8 sessions."""
    assessment = _assess(_E8)

    assert assessment.sessions_after_decision_through_expiry == 8
    assert assessment.next_trading_date == _E7


def test_the_next_session_after_a_friday_is_monday() -> None:
    assessment = _assess(_E6)

    assert assessment.next_trading_date == _E5 == date(2026, 10, 19)


def test_the_holiday_is_not_counted() -> None:
    """From Mon 10-19 the next session is Wed 10-21: Tue 10-20 is a holiday."""
    assessment = _assess(_E5)

    assert assessment.next_trading_date == _E4
    assert assessment.sessions_after_decision_through_expiry == 5


def test_a_special_weekend_session_is_counted() -> None:
    """From Fri 01-30 the next session is the Sunday Budget session on 02-01."""
    assessment = _assess(date(2026, 1, 30), _FEB)

    assert assessment.next_trading_date == date(2026, 2, 1)
    assert assessment.sessions_after_decision_through_expiry == 3


def test_counting_crosses_a_weekend_and_a_holiday_together() -> None:
    """Fri 01-23: Sat, Sun and the Mon 01-26 holiday are all skipped."""
    assessment = _assess(date(2026, 1, 23), _FEB)

    assert assessment.next_trading_date == date(2026, 1, 27)
    assert assessment.sessions_after_decision_through_expiry == 7


def test_the_assessment_records_its_audit_facts() -> None:
    assessment = _assess(_E6)

    assert assessment.contract == _OCT
    assert assessment.decision_instant == _close(_E6)
    assert assessment.decision_trading_date == _E6
    assert assessment.expiry_trading_date == _E
    assert assessment.sessions_before_expiry == 5


# ---------------------------------------------------------------------------
# 3-4. K = 5 boundary semantics: flat by the E-5 open
# ---------------------------------------------------------------------------


def test_a_decision_at_the_e7_close_does_not_flatten_yet() -> None:
    assessment = _assess(_E7)

    assert assessment.sessions_after_decision_through_expiry == 7
    assert assessment.next_trading_date == _E6
    assert not assessment.flatten_required


def test_a_decision_at_the_e6_close_requires_the_flatten() -> None:
    assessment = _assess(_E6)

    assert assessment.sessions_after_decision_through_expiry == 6
    assert assessment.flatten_required


def test_the_e6_flatten_fills_at_the_e5_open() -> None:
    """Pinned semantics: flat by the OPEN of E-5, not decided at E-5."""
    assert _assess(_E6).next_trading_date == _E5


@pytest.mark.parametrize("day", [_E5, _E4, _E3, _E2, _E1, _E])
def test_every_session_from_e5_through_expiry_is_protected(day: date) -> None:
    assert _assess(day).flatten_required


def test_the_expiry_session_counts_no_further_sessions() -> None:
    assessment = _assess(_E)

    assert assessment.sessions_after_decision_through_expiry == 0
    assert assessment.next_trading_date is None
    assert assessment.flatten_required


def test_a_decision_after_expiry_is_still_protected() -> None:
    assessment = _assess(date(2026, 10, 28))

    assert assessment.sessions_after_decision_through_expiry == 0
    assert assessment.flatten_required


@pytest.mark.parametrize("k", [1, 2, 3, 5, 6])
def test_the_threshold_is_k_plus_one_sessions_for_every_k(k: int) -> None:
    sessions = _COUNTDOWN[::-1]  # E, E-1, ...
    first_flatten = sessions[k + 1]
    last_unguarded = sessions[k + 2]

    assert _assess(first_flatten, k=k).flatten_required
    assert not _assess(last_unguarded, k=k).flatten_required


def test_k_counts_trading_sessions_not_calendar_days() -> None:
    """The K=5 flatten point is Fri 10-16, eleven calendar days before expiry."""
    assert (_E - _E6).days == 11
    assert _assess(_E6).flatten_required
    assert not _assess(_E7).flatten_required


# ---------------------------------------------------------------------------
# Session membership and explicit failures
# ---------------------------------------------------------------------------


def test_an_offset_spelling_of_the_close_belongs_to_the_same_session() -> None:
    assessment = _guard().assess(_OCT, PointInTime("2026-10-16T15:40:00+05:30"))

    assert assessment.decision_trading_date == _E6


def test_an_intraday_instant_belongs_to_its_session() -> None:
    assert _guard().assess(_OCT, PointInTime("2026-10-16T06:00:00Z")).decision_trading_date == _E6


@pytest.mark.parametrize(
    "instant",
    [
        "2026-10-16T03:30:00Z",  # exactly at the open: the window is open on the left
        "2026-10-16T12:00:00Z",  # after the close
        "2026-10-17T06:00:00Z",  # a Saturday
        "2026-10-20T06:00:00Z",  # the holiday
    ],
)
def test_a_decision_instant_inside_no_session_fails(instant: str) -> None:
    with pytest.raises(FuturesExpiryWindowError, match="inside no trading session"):
        _guard().assess(_OCT, PointInTime(instant))


def test_an_expiration_that_is_not_a_session_fails() -> None:
    with pytest.raises(FuturesExpiryWindowError, match="2026-11-24 .* is not a trading session"):
        _guard().assess(_NOV_HOLIDAY, _close(date(2026, 11, 16)))


def test_a_resolver_failure_propagates_unchanged() -> None:
    """The NIFTY November countdown from 11-02 crosses the un-notified Muhurat session."""
    with pytest.raises(FuturesSessionResolutionError, match="Muhurat|2026-11-08"):
        _guard().assess(_NOV, _close(date(2026, 11, 2)))


def test_an_uncovered_calendar_year_propagates_unchanged() -> None:
    contract = FuturesContract(_NIFTY, ExpirationDate("2027-01-26"))

    with pytest.raises(FuturesSessionResolutionError, match="not covered"):
        _guard().assess(contract, _close(date(2027, 1, 19)))


def test_the_guard_uses_only_the_port_and_never_the_expiry_minus_days() -> None:
    calendar = NseCalendar()

    _guard(resolver=calendar).assess(_OCT, _close(_E6))

    assert ("resolve", _E, _E) in calendar.calls
    assert ("range", _E6, _E) in calendar.calls
    for kind, start, end in calendar.calls:
        assert start <= end
        if kind == "range" and end == _E:
            assert start == _E6


def test_the_guard_type_checks_its_inputs() -> None:
    with pytest.raises(TypeError, match="FuturesTradingSessionResolver"):
        FuturesExpiryFlattenGuard(object(), FuturesExpiryFlattenPolicy(5))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="FuturesExpiryFlattenPolicy"):
        FuturesExpiryFlattenGuard(NseCalendar(), 5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="FuturesContract"):
        _guard().assess(_NIFTY, _close(_E6))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="PointInTime"):
        _guard().assess(_OCT, "2026-10-16T10:10:00Z")  # type: ignore[arg-type]


def _assessment_kwargs(**overrides: object) -> dict:
    values: dict[str, object] = {
        "contract": _OCT,
        "decision_instant": _close(_E6),
        "decision_trading_date": _E6,
        "expiry_trading_date": _E,
        "sessions_before_expiry": 5,
        "sessions_after_decision_through_expiry": 6,
        "next_trading_date": _E5,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    "overrides",
    [
        {"expiry_trading_date": date(2026, 10, 28)},
        {"sessions_after_decision_through_expiry": -1},
        {"sessions_after_decision_through_expiry": True},
        {"sessions_after_decision_through_expiry": 0},
        {"next_trading_date": None},
        {"next_trading_date": _E6},
        {"sessions_before_expiry": 0},
    ],
    ids=[
        "expiry-not-contract",
        "negative-count",
        "bool-count",
        "zero-count-with-next",
        "count-without-next",
        "next-not-after-decision",
        "bad-k",
    ],
)
def test_an_incoherent_assessment_is_rejected(overrides: dict) -> None:
    with pytest.raises((TypeError, ValueError)):
        FuturesExpiryWindowAssessment(**_assessment_kwargs(**overrides))


# ---------------------------------------------------------------------------
# 5-10. Policy inside and outside the window
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("action", ["BUY", "SELL", "HOLD"])
def test_a_long_is_sold_in_full_whatever_the_action(action: str) -> None:
    decision = _policy(action, (_OCT, 2))

    assert decision.expiry_flatten
    assert decision.intent.side is OrderSide.SELL
    assert decision.intent.contracts == FuturesContractCount(2)
    assert decision.intent.contract == _OCT


@pytest.mark.parametrize("action", ["BUY", "SELL", "HOLD"])
def test_a_short_is_bought_back_in_full_whatever_the_action(action: str) -> None:
    decision = _policy(action, (_OCT, -3))

    assert decision.expiry_flatten
    assert decision.intent.side is OrderSide.BUY
    assert decision.intent.contracts == FuturesContractCount(3)


@pytest.mark.parametrize("action", ["BUY", "SELL", "HOLD"])
def test_an_already_flat_portfolio_gets_no_order(action: str) -> None:
    decision = _policy(action)

    assert decision.intent is None
    assert decision.no_intent_reason is _WINDOW
    assert decision.expiry_flatten


@pytest.mark.parametrize("day", [_E5, _E4, _E1, _E])
@pytest.mark.parametrize("action", ["BUY", "SELL"])
def test_no_directional_signal_reopens_inside_the_window(action: str, day: date) -> None:
    decision = _policy(action, day=day)

    assert decision.intent is None
    assert decision.no_intent_reason is _WINDOW


@pytest.mark.parametrize(("net", "side"), [(5, OrderSide.SELL), (-7, OrderSide.BUY)])
def test_an_over_target_position_flattens_completely(net: int, side: OrderSide) -> None:
    decision = _policy("BUY", (_OCT, net), target=2)

    assert decision.intent.side is side
    assert decision.intent.contracts == FuturesContractCount(abs(net))


def test_a_flatten_never_reverses_through_zero() -> None:
    decision = _policy("SELL", (_OCT, 2), target=2)

    # Outside the window this SELL would trade 4 to reach -2.
    assert decision.intent.side is OrderSide.SELL
    assert decision.intent.contracts == FuturesContractCount(2)


def test_other_contracts_and_expiries_are_untouched() -> None:
    decision = _policy("BUY", (_OCT, 2), (_NOV, 3), (_BANK_OCT, -1))

    assert decision.intent.contract == _OCT
    assert decision.intent.contracts == FuturesContractCount(2)


def test_another_expiry_alone_does_not_trigger_a_flatten_order() -> None:
    decision = _policy("BUY", (_NOV, 3))

    assert decision.intent is None
    assert decision.no_intent_reason is _WINDOW


@pytest.mark.parametrize("action", ["BUY", "SELL", "HOLD"])
@pytest.mark.parametrize("holdings", [(), ((_OCT, 2),), ((_OCT, -2),), ((_OCT, 5),)])
def test_outside_the_window_the_policy_is_exactly_unchanged(action, holdings) -> None:
    with_window = _policy(action, *holdings, day=_E7, window=True)
    without = _policy(action, *holdings, day=_E7, window=False)

    assert with_window == without
    assert not with_window.expiry_flatten


def test_hold_outside_the_window_keeps_the_exact_old_behaviour() -> None:
    decision = _policy("HOLD", (_OCT, 2), day=_E7)

    assert decision == FuturesExecutionIntentDecision(
        no_intent_reason=FuturesExecutionIntentNoIntentReason.HOLD
    )


def test_an_assessment_of_another_contract_is_rejected() -> None:
    with pytest.raises(ValueError, match="not the decided contract"):
        CreateFuturesExecutionIntentUseCase().execute(
            _record("BUY", _E6),
            _portfolio(),
            FuturesContractCount(2),
            expiry_window=_guard().assess(_BANK_OCT, _close(_E6)),
        )


def test_an_assessment_of_another_instant_is_rejected() -> None:
    with pytest.raises(ValueError, match="another decision instant"):
        CreateFuturesExecutionIntentUseCase().execute(
            _record("BUY", _E6), _portfolio(), FuturesContractCount(2), expiry_window=_assess(_E5)
        )


def test_a_foreign_assessment_is_rejected() -> None:
    with pytest.raises(TypeError, match="FuturesExpiryWindowAssessment"):
        CreateFuturesExecutionIntentUseCase().execute(
            _record("BUY", _E6), _portfolio(), FuturesContractCount(2), expiry_window=True
        )


def test_the_expiry_reason_requires_the_expiry_marker() -> None:
    with pytest.raises(ValueError, match="only when the expiry guard governs"):
        FuturesExecutionIntentDecision(no_intent_reason=_WINDOW)
    with pytest.raises(ValueError, match="only decline as EXPIRY_FLATTEN_WINDOW"):
        FuturesExecutionIntentDecision(
            no_intent_reason=FuturesExecutionIntentNoIntentReason.HOLD, expiry_flatten=True
        )
    with pytest.raises(TypeError, match="must be a bool"):
        FuturesExecutionIntentDecision(no_intent_reason=_WINDOW, expiry_flatten=1)


def test_the_marker_defaults_to_an_ordinary_decision() -> None:
    decision = FuturesExecutionIntentDecision(
        no_intent_reason=FuturesExecutionIntentNoIntentReason.HOLD
    )

    assert decision.expiry_flatten is False


# ---------------------------------------------------------------------------
# 11-17. Orchestration: decision, run, replay, fills and metrics
# ---------------------------------------------------------------------------


def _compare_text(left: str, right: str) -> int:
    return (left > right) - (left < right)


class World:
    def __init__(self) -> None:
        self.records: list[FuturesForwardResearchRecord] = []
        self.bars: list[FuturesOHLCVBar] = []
        self.orders: dict[str, object] = {}
        self.fills: dict[str, object] = {}
        self.writes: list[tuple[str, int]] = []


class Forward(FuturesForwardResearchRecordRepository):
    def __init__(self, world: World) -> None:
        self.world = world

    def get_records(self, query):
        matching = [r for r in self.world.records if r.contract == query.contract]
        return tuple(
            sorted(
                matching,
                key=cmp_to_key(lambda a, b: a.decision_instant.compare(b.decision_instant)),
            )
        )


class Market(FuturesHistoricalMarketDataRepository):
    def __init__(self, world: World) -> None:
        self.world = world

    def get_bars(self, query):
        matching = [
            bar
            for bar in self.world.bars
            if bar.contract == query.contract
            and bar.timeframe == query.timeframe
            and query.covers(bar.point_in_time)
        ]
        return tuple(
            sorted(matching, key=cmp_to_key(lambda a, b: a.point_in_time.compare(b.point_in_time)))
        )


class Orders(FuturesPaperOrderStore, FuturesPaperOrderRepository):
    def __init__(self, world: World) -> None:
        self.world = world

    def store(self, orders):
        self.world.writes.append(("orders", len(orders)))
        for order in orders:
            existing = self.world.orders.get(order.identity.identity)
            if existing is not None and existing != order:
                raise FuturesPaperOrderConflictError("different order under this identity")
            self.world.orders[order.identity.identity] = order
        return len(orders)

    def get_orders(self, query):
        def compare(left, right) -> int:
            instant = left.intent.decided_at.compare(right.intent.decided_at)
            return instant or _compare_text(left.identity.identity, right.identity.identity)

        return tuple(sorted(self.world.orders.values(), key=cmp_to_key(compare)))


class Fills(FuturesPaperFillStore, FuturesPaperFillRepository):
    def __init__(self, world: World) -> None:
        self.world = world

    def store(self, fills):
        self.world.writes.append(("fills", len(fills)))
        for fill in fills:
            self.world.fills[fill.identity.identity] = fill
        return len(fills)

    def get_fills(self, query):
        def compare(left, right) -> int:
            instant = left.filled_at.compare(right.filled_at)
            return instant or _compare_text(
                left.order_identity.identity, right.order_identity.identity
            )

        return tuple(sorted(self.world.fills.values(), key=cmp_to_key(compare)))


def _ports(world: World) -> dict[str, object]:
    orders, fills = Orders(world), Fills(world)
    return {
        "forward_repository": Forward(world),
        "market_repository": Market(world),
        "order_store": orders,
        "order_repository": orders,
        "fill_store": fills,
        "fill_repository": fills,
    }


_SESSIONS = (_E8, _E7, _E6, _E5, _E4, _E3, _E2, _E1, _E)
_OPENS = {day: 25000 + 10 * index for index, day in enumerate(_SESSIONS)}


def _world(*decisions: tuple[str, date]) -> World:
    world = World()
    world.records.extend(_record(action, day) for action, day in decisions)
    world.bars.extend(_bar(day, _OPENS[day]) for day in _SESSIONS)
    return world


_LONG_SCENARIO = (
    ("BUY", _E8),  # opens +2, filled at the E-7 open
    ("BUY", _E7),  # target already met; not yet protected
    ("HOLD", _E6),  # expiry flatten: SELL 2, filled at the E-5 open
    ("BUY", _E5),  # protected and flat: no reopening
    ("SELL", _E4),
    ("HOLD", _E1),
    ("BUY", _E),
)


def _run(world: World, guard: FuturesExpiryFlattenGuard | None, through: date = _E):
    return RunFuturesPaperTradingUseCase(**_ports(world), expiry_guard=guard).execute(
        FuturesForwardResearchRecordQuery(_OCT, _DAILY),
        PaperPortfolioIdentity(_PORTFOLIO),
        StrategyIdentity(_STRATEGY),
        FuturesContractCount(2),
        _close(through),
    )


def _by_day(run) -> dict[date, FuturesPaperTradingDecisionResult]:
    return {
        date.fromisoformat(result.record.decision_instant.value[:10]): result
        for result in run.results
    }


def test_the_guarded_run_flattens_before_expiry_and_never_reopens() -> None:
    run = _run(_world(*_LONG_SCENARIO), _guard())
    results = _by_day(run)

    assert results[_E8].order.intent.side is OrderSide.BUY
    assert results[_E7].decision.no_intent_reason is (
        FuturesExecutionIntentNoIntentReason.TARGET_ALREADY_MET
    )
    flatten = results[_E6]
    assert flatten.decision.expiry_flatten
    assert flatten.order.intent.side is OrderSide.SELL
    assert flatten.order.intent.contracts == FuturesContractCount(2)
    for day in (_E5, _E4, _E1, _E):
        assert results[day].order is None
        assert results[day].decision.no_intent_reason is _WINDOW
    assert run.portfolio.get_position(_OCT) is None


def test_hold_inside_the_window_creates_the_flatten_order() -> None:
    results = _by_day(_run(_world(*_LONG_SCENARIO), _guard()))

    flatten = results[_E6]
    assert flatten.record.result.recommendation.action.value == "HOLD"
    assert flatten.order is not None
    assert flatten.expiry_window.flatten_required


def test_the_flatten_fills_at_the_e5_open() -> None:
    results = _by_day(_run(_world(*_LONG_SCENARIO), _guard()))

    fill = results[_E6].fill
    assert fill.filled_at == _close(_E5)
    assert fill.fill_quote == QuoteValue(Decimal(_OPENS[_E5]))


def test_the_flatten_order_identity_is_the_ordinary_one() -> None:
    world = _world(*_LONG_SCENARIO)
    results = _by_day(_run(world, _guard()))

    expected = FuturesPaperExecutionIdentityService().order_identity(
        _record("HOLD", _E6), PaperPortfolioIdentity(_PORTFOLIO)
    )
    assert results[_E6].order.identity == expected


def test_a_short_is_flattened_by_a_buy_not_reversed() -> None:
    world = _world(("SELL", _E8), ("BUY", _E6), ("BUY", _E5))
    results = _by_day(_run(world, _guard()))

    assert results[_E6].order.intent.side is OrderSide.BUY
    assert results[_E6].order.intent.contracts == FuturesContractCount(2)
    assert results[_E5].decision.no_intent_reason is _WINDOW


def test_a_guarded_replay_is_idempotent() -> None:
    world = _world(*_LONG_SCENARIO)
    first = _run(world, _guard())
    orders, fills = dict(world.orders), dict(world.fills)

    second = _run(world, _guard())

    assert second.results == first.results
    assert world.orders == orders
    assert world.fills == fills


def test_without_a_guard_the_old_behaviour_is_reproduced_exactly() -> None:
    results = _by_day(_run(_world(*_LONG_SCENARIO), None))

    assert results[_E6].decision == FuturesExecutionIntentDecision(
        no_intent_reason=FuturesExecutionIntentNoIntentReason.HOLD
    )
    assert results[_E5].decision.no_intent_reason is (
        FuturesExecutionIntentNoIntentReason.TARGET_ALREADY_MET
    )
    assert results[_E4].order.intent.side is OrderSide.SELL
    assert results[_E4].order.intent.contracts == FuturesContractCount(4)
    assert all(result.expiry_window is None for result in results.values())
    assert all(not result.decision.expiry_flatten for result in results.values())


def test_metrics_reconcile_expiry_window_decisions() -> None:
    run = _run(_world(*_LONG_SCENARIO), _guard())

    [metrics] = CalculateFuturesPaperTradingMetricsUseCase().execute(run)

    assert metrics.decision_count == 7
    assert metrics.hold_count == 0
    assert metrics.target_already_met_count == 1
    assert metrics.expiry_window_count == 4
    assert metrics.order_count == 2
    assert metrics.contracts_bought == 2 and metrics.contracts_sold == 2
    assert metrics.current_net_contracts == 0


def test_the_metrics_invariant_counts_expiry_window_decisions() -> None:
    values = {
        "contract": _OCT,
        "strategy_identity": StrategyIdentity(_STRATEGY),
        "decision_count": 3,
        "hold_count": 1,
        "target_already_met_count": 1,
        "order_count": 0,
        "filled_order_count": 0,
        "pending_order_count": 0,
        "contracts_bought": 0,
        "contracts_sold": 0,
        "current_net_contracts": 0,
    }
    with pytest.raises(ValueError, match="expiry-window non-actions"):
        FuturesPaperTradingStrategyContractMetrics(**values)

    assert FuturesPaperTradingStrategyContractMetrics(**values, expiry_window_count=1)


def test_a_calendar_failure_writes_nothing() -> None:
    record = _record("HOLD", date(2026, 11, 2), _NOV)
    world = World()
    world.records.append(record)

    with pytest.raises(FuturesSessionResolutionError):
        RunFuturesPaperTradingDecisionUseCase(**_ports(world), expiry_guard=_guard()).execute(
            record,
            PaperPortfolioIdentity(_PORTFOLIO),
            FuturesContractCount(2),
            record.decision_instant,
        )

    assert world.writes == []


# ---------------------------------------------------------------------------
# Result invariants: HOLD is relaxed only for an expiry-governed flatten
# ---------------------------------------------------------------------------


def _flatten_result() -> FuturesPaperTradingDecisionResult:
    return _by_day(_run(_world(*_LONG_SCENARIO), _guard()))[_E6]


def test_a_hold_order_without_an_expiry_window_is_still_rejected() -> None:
    result = _flatten_result()

    with pytest.raises(ValueError, match="expiry-governed exactly when"):
        dataclasses.replace(result, expiry_window=None)


def test_a_hold_order_without_the_marker_is_still_rejected() -> None:
    result = _flatten_result()
    unmarked = FuturesExecutionIntentDecision(intent=result.decision.intent)

    with pytest.raises(ValueError, match="expiry-governed exactly when"):
        dataclasses.replace(result, decision=unmarked)


def test_an_expiry_window_that_does_not_require_a_flatten_cannot_govern() -> None:
    result = _flatten_result()
    not_yet = dataclasses.replace(result.expiry_window, sessions_after_decision_through_expiry=7)

    with pytest.raises(ValueError, match="expiry-governed exactly when"):
        dataclasses.replace(result, expiry_window=not_yet)


def test_a_partial_flatten_is_rejected() -> None:
    result = _flatten_result()
    partial = dataclasses.replace(result.decision.intent, contracts=FuturesContractCount(1))

    with pytest.raises(ValueError):
        dataclasses.replace(
            result,
            decision=FuturesExecutionIntentDecision(intent=partial, expiry_flatten=True),
            order=dataclasses.replace(result.order, intent=partial),
            fill=None,
        )


def test_an_expiry_window_for_another_decision_is_rejected() -> None:
    result = _flatten_result()

    with pytest.raises(ValueError, match="must assess the record's decision"):
        dataclasses.replace(result, expiry_window=_assess(_E5))


def test_a_required_window_with_an_ordinary_decision_is_rejected() -> None:
    results = _by_day(_run(_world(*_LONG_SCENARIO), None))
    ordinary = results[_E5]  # TARGET_ALREADY_MET, judged without a guard

    with pytest.raises(ValueError, match="expiry-governed exactly when"):
        dataclasses.replace(ordinary, expiry_window=_assess(_E5))


# ---------------------------------------------------------------------------
# Construction, wiring and purity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [RunFuturesPaperTradingDecisionUseCase, RunFuturesPaperTradingUseCase],
)
def test_orchestrators_type_check_the_guard(factory) -> None:
    with pytest.raises(TypeError, match="expiry_guard"):
        factory(**_ports(World()), expiry_guard=NseCalendar())


class _ForwardStore(FuturesForwardResearchRecordStore):
    def store(self, records):  # pragma: no cover - not exercised
        return len(records)


def _session_ports(world: World) -> dict[str, object]:
    ports = _ports(world)
    return {**ports, "forward_store": _ForwardStore()}


def test_the_session_passes_the_guard_through_to_every_decision() -> None:
    guard = _guard()
    session = RunFuturesPaperTradingSessionUseCase(**_session_ports(World()), expiry_guard=guard)

    assert session._run._decide._expiry_guard is guard


def test_the_session_defaults_to_no_guard_and_type_checks_it() -> None:
    session = RunFuturesPaperTradingSessionUseCase(**_session_ports(World()))
    assert session._run._decide._expiry_guard is None

    with pytest.raises(TypeError, match="expiry_guard"):
        RunFuturesPaperTradingSessionUseCase(**_session_ports(World()), expiry_guard="K=5")


_GUARD_MODULE = Path(
    __import__(
        "northstar_application.application_services.futures_expiry_flatten_guard",
        fromlist=["_"],
    ).__file__
)


def _guard_tree() -> ast.Module:
    return ast.parse(_GUARD_MODULE.read_text(encoding="utf-8"))


def test_the_guard_reads_no_clock() -> None:
    tree = _guard_tree()
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}

    assert not any(str(name).split(".")[0] in {"time", "os", "random"} for name in imported)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"now", "utcnow", "today", "monotonic"}


def test_no_timedelta_is_ever_applied_to_the_expiration() -> None:
    """The only timedelta widens the membership search around the decision instant."""
    tree = _guard_tree()
    margin_uses = []
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            text = ast.unparse(node)
            if "_MEMBERSHIP_SEARCH_MARGIN" in text or "timedelta" in text:
                margin_uses.append(text)
                assert "expir" not in text.lower()
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "timedelta":
            assert ast.unparse(node) == "timedelta(days=2)"
    assert margin_uses == [
        "around - _MEMBERSHIP_SEARCH_MARGIN",
        "around + _MEMBERSHIP_SEARCH_MARGIN",
    ]


def test_the_guard_depends_only_on_the_resolver_port() -> None:
    tree = _guard_tree()
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert not any(str(module).startswith("northstar_infrastructure") for module in modules)
    assert "FuturesTradingSessionResolver" in names
    assert not {"NSEFuturesTradingSessionResolver", "Upstox", "Databento"} & names
    assert "timedelta(days=5)" not in _GUARD_MODULE.read_text(encoding="utf-8")


def test_the_public_surface_is_exported() -> None:
    import northstar_application.application_services as services

    for name in (
        "FuturesExpiryFlattenGuard",
        "FuturesExpiryFlattenPolicy",
        "FuturesExpiryWindowAssessment",
        "FuturesExpiryWindowError",
        "InvalidFuturesExpiryFlattenPolicyError",
    ):
        assert name in services.__all__
    assert not hasattr(services, "_MEMBERSHIP_SEARCH_MARGIN")


def test_the_calendar_table_has_the_expected_nse_shape() -> None:
    """Guards the fixture itself against drifting from the dates documented above."""
    calendar = NseCalendar()

    assert calendar.resolve(_NIFTY, date(2026, 10, 20)) is None
    assert calendar.resolve(_NIFTY, date(2026, 11, 24)) is None
    assert calendar.resolve(_NIFTY, date(2026, 2, 1)) is not None
    labels = [s.trading_date for s in calendar.sessions_in_range(_NIFTY, _E8, _E)]
    assert labels == list(_SESSIONS)
    assert timedelta(days=0) == _E - labels[-1]
