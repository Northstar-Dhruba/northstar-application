"""Tests for monitoring futures trade recommendation lifecycles (F-5C).

Everything here is SYNTHETIC: the calendar (weekday sessions 03:45-10:00Z,
loaded years configurable), the bars, the candle approvals and the
recommendations. None is NSE reference data, a provider candle or an approved
configuration. Storage is an in-memory fake with the lifecycle store's
documented semantics; the SQLite path is covered in northstar-infrastructure.

The default recommendation is LONG on NIFTY@NSE 2026-10-27: entry 24378-24388,
stop 24347.75, target 24448.45, issued 2026-10-07T12:00Z, so its window is the
five sessions 10-08, 10-09, 10-12, 10-13 and 10-14 (eligible_from is the 10-08
close and valid_through the 10-14 close).
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
    CoverageDeficiency,
    EntryZone,
    FuturesResearchDecisionReference,
    FuturesTradeRecommendation,
    FuturesTradeRecommendationIdentity,
    FuturesTradeRecommendationLifecycle,
    IntrabarOrdering,
    ObservationCoverage,
    PriceTarget,
    RecommendationEventKind,
    RecommendationPolicyVersion,
    RecommendationStatus,
    RecommendationValidity,
    StopLoss,
    TradeDirection,
)
from northstar_core.strategy import ExplanationReason, StrategyIdentity

import northstar_application.application_services as services
from northstar_application.application_services import (
    FUTURES_INCOMPLETE_COVERAGE_ATTESTATION_FORMAT,
    DisabledFuturesDailyBarFinalityPolicy,
    FuturesCandleApproval,
    FuturesCandleApprovalMismatchError,
    FuturesCandleApprovalSet,
    FuturesIncompleteCoverageDeclaration,
    FuturesRecommendationEvidenceGap,
    FuturesRecommendationMonitoringOutcome,
    FuturesRecommendationMonitoringSourceError,
    FuturesRecommendationMonitoringStopReason,
    FuturesReviewedCandleVerification,
    FuturesTradeRecommendationContractViolationError,
    InvalidFuturesRecommendationMonitoringError,
    MonitorFuturesTradeRecommendationsUseCase,
    OperatorApprovedFuturesDailyBarFinalityPolicy,
    futures_daily_candle_fingerprint,
)
from northstar_application.ports import (
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
    FuturesSessionResolutionError,
    FuturesTradeRecommendationLifecycleConflictError,
    FuturesTradeRecommendationLifecycleEvidenceError,
    FuturesTradeRecommendationLifecycleStore,
    FuturesTradeRecommendationObservationStep,
    FuturesTradeRecommendationQuery,
    FuturesTradeRecommendationRepository,
    FuturesTradeRecommendationTimeExpiryStep,
    FuturesTradingSession,
    FuturesTradingSessionResolver,
    apply_futures_trade_recommendation_lifecycle_step,
    validate_futures_trade_recommendation_lifecycle_progress,
)

Stop = FuturesRecommendationMonitoringStopReason
Gap = FuturesRecommendationEvidenceGap
Status = RecommendationStatus

_NIFTY = FuturesProductReference(Symbol("NIFTY"), ExchangeCode("NSE"))
_BANKNIFTY = FuturesProductReference(Symbol("BANKNIFTY"), ExchangeCode("NSE"))
_OCT = FuturesContract(_NIFTY, ExpirationDate("2026-10-27"))
_NOV = FuturesContract(_NIFTY, ExpirationDate("2026-11-24"))
_JAN = FuturesContract(_NIFTY, ExpirationDate("2027-01-28"))
_STRATEGY = StrategyIdentity("synthetic-directional-v1")
_DAILY = Timeframe("1d")

_QUIET = ("24400", "24420")
_ENTRY = ("24380", "24400")
_STOP = ("24340", "24390")
_TARGET = ("24400", "24450")
_BOTH = ("24340", "24450")

_WINDOW_DAYS = (
    date(2026, 10, 8),
    date(2026, 10, 9),
    date(2026, 10, 12),
    date(2026, 10, 13),
    date(2026, 10, 14),
)
_AFTER_VALIDITY = PointInTime("2026-10-15T00:00:00Z")


class Calendar(FuturesTradingSessionResolver):
    """Weekday sessions 03:45-10:00Z; only the loaded years resolve."""

    def __init__(
        self, holidays: frozenset[date] = frozenset(), years: frozenset[int] = frozenset({2026})
    ) -> None:
        self.holidays, self.years = holidays, years

    def resolve(self, product, trading_date):
        if trading_date.year not in self.years:
            raise FuturesSessionResolutionError(f"Calendar year {trading_date.year} is not loaded.")
        if trading_date.weekday() >= 5 or trading_date in self.holidays:
            return None
        day = trading_date.isoformat()
        return FuturesTradingSession(
            trading_date, PointInTime(f"{day}T03:45:00Z"), PointInTime(f"{day}T10:00:00Z")
        )

    def sessions_in_range(self, product, start_date, end_date):
        found, day = [], start_date
        while day <= end_date:
            session = self.resolve(product, day)
            if session is not None:
                found.append(session)
            day += timedelta(days=1)
        return tuple(found)


def _q(value: str | Decimal) -> QuoteValue:
    return QuoteValue(Decimal(value))


def _close(day: date) -> PointInTime:
    return PointInTime(f"{day.isoformat()}T10:00:00Z")


def _recommendation(
    contract: FuturesContract = _OCT,
    *,
    identity: str = "a" * 64,
    eligible: date = date(2026, 10, 8),
    through: date = date(2026, 10, 14),
    issued: str = "2026-10-07T12:00:00Z",
    cutoff: date = date(2026, 10, 27),
) -> FuturesTradeRecommendation:
    return FuturesTradeRecommendation(
        identity=FuturesTradeRecommendationIdentity(identity),
        contract=contract,
        direction=TradeDirection.LONG,
        entry=EntryZone(_q("24378"), _q("24388")),
        stop=StopLoss(_q("24347.75")),
        target=PriceTarget(_q("24448.45")),
        validity=RecommendationValidity(_close(eligible), _close(through), _close(cutoff)),
        source=FuturesResearchDecisionReference(
            contract, _DAILY, PointInTime(issued[:10] + "T10:00:00Z"), _STRATEGY
        ),
        policy_version=RecommendationPolicyVersion("f4-synthetic-fixture-v1"),
        issued_at=PointInTime(issued),
        thesis=(ExplanationReason("Synthetic thesis"),),
    )


def _bar(
    day: date,
    prices: tuple[str, str] = _QUIET,
    contract: FuturesContract = _OCT,
    timeframe: Timeframe = _DAILY,
) -> FuturesOHLCVBar:
    low, high = prices
    middle = (Decimal(low) + Decimal(high)) / 2
    return FuturesOHLCVBar(
        contract=contract,
        point_in_time=_close(day),
        timeframe=timeframe,
        open=_q(middle),
        high=_q(high),
        low=_q(low),
        close=_q(middle),
        volume=Quantity(Decimal("1250")),
    )


def _approvals(*bars: FuturesOHLCVBar, product=_NIFTY) -> FuturesReviewedCandleVerification:
    return FuturesReviewedCandleVerification(
        FuturesCandleApprovalSet(
            product,
            tuple(
                FuturesCandleApproval(
                    bar.contract,
                    date.fromisoformat(bar.point_in_time.value[:10]),
                    futures_daily_candle_fingerprint(
                        bar.contract, date.fromisoformat(bar.point_in_time.value[:10]), bar
                    ),
                )
                for bar in bars
            ),
        )
    )


class Market(FuturesHistoricalMarketDataRepository):
    def __init__(self, bars: tuple[FuturesOHLCVBar, ...] = ()) -> None:
        self.bars = bars

    def get_bars(self, query: FuturesHistoricalMarketDataQuery):
        return tuple(
            bar
            for bar in sorted(self.bars, key=lambda b: b.point_in_time.value)
            if bar.contract == query.contract and query.covers(bar.point_in_time)
        )


class Store(FuturesTradeRecommendationRepository, FuturesTradeRecommendationLifecycleStore):
    """In-memory lifecycles with the lifecycle store's documented semantics."""

    def __init__(self, *lifecycles: FuturesTradeRecommendationLifecycle) -> None:
        self.stored = {lc.recommendation.identity: lc for lc in lifecycles}
        self.recorded: list[tuple] = []
        self.before_write = None  # a concurrent writer, run just before the compare

    def get_recommendation(self, identity):
        lifecycle = self.stored.get(identity)
        return None if lifecycle is None else lifecycle.recommendation

    def get_lifecycles(self, query: FuturesTradeRecommendationQuery):
        return tuple(
            lc for lc in self.stored.values() if lc.recommendation.contract.product == query.product
        )

    def record_progress(self, previous, current, *, steps):
        validate_futures_trade_recommendation_lifecycle_progress(previous, current, steps)
        if self.before_write is not None:
            self.before_write(self)
        stored = self.stored[previous.recommendation.identity]
        if stored == previous:
            self.stored[previous.recommendation.identity] = current
            self.recorded.append(steps)
            return len(steps)
        if stored == current:
            return 0
        raise FuturesTradeRecommendationLifecycleConflictError("synthetic concurrent progress")


@dataclasses.dataclass
class World:
    bars: tuple[FuturesOHLCVBar, ...]
    store: Store
    approved: tuple[FuturesOHLCVBar, ...] | None = None
    final_through: date | None = date(2026, 12, 31)
    calendar: FuturesTradingSessionResolver = dataclasses.field(default_factory=Calendar)

    def run(self, as_of: PointInTime = _AFTER_VALIDITY, declarations=(), product=_NIFTY):
        finality = (
            DisabledFuturesDailyBarFinalityPolicy()
            if self.final_through is None
            else OperatorApprovedFuturesDailyBarFinalityPolicy(self.final_through)
        )
        approved = self.bars if self.approved is None else self.approved
        return MonitorFuturesTradeRecommendationsUseCase(
            self.store, self.store, Market(self.bars), self.calendar, finality
        ).execute(
            product=product,
            as_of=as_of,
            reviewed_candles=_approvals(*approved, product=product),
            coverage_declarations=tuple(declarations),
        )

    def lifecycle(self, identity: str = "a" * 64) -> FuturesTradeRecommendationLifecycle:
        return self.store.stored[FuturesTradeRecommendationIdentity(identity)]

    def steps(self) -> list:
        return [step for call in self.store.recorded for step in call]


def _world(*prices: tuple[str, str] | None, recommendation=None, **overrides) -> World:
    """A world whose stored bars follow ``prices`` over the window; None leaves a gap."""
    bars = tuple(
        _bar(day, price) for day, price in zip(_WINDOW_DAYS, prices, strict=False) if price
    )
    lifecycle = FuturesTradeRecommendationLifecycle(recommendation or _recommendation())
    return World(bars=bars, store=Store(lifecycle), **overrides)


def _declaration(identity: str = "a" * 64, **overrides) -> FuturesIncompleteCoverageDeclaration:
    values = {
        "recommendation_identity": FuturesTradeRecommendationIdentity(identity),
        "operator_identity": "synthetic-operator",
        "case_reference": "CASE-0001",
        "monitor_policy_version": "f5c-synthetic-v1",
    }
    values.update(overrides)
    return FuturesIncompleteCoverageDeclaration(**values)  # type: ignore[arg-type]


def _only(result) -> FuturesRecommendationMonitoringOutcome:
    (outcome,) = result.outcomes
    return outcome


# ---------------------------------------------------------------------------
# A. Lifecycle transitions through Core's advance
# ---------------------------------------------------------------------------


def test_an_entry_activates_and_monitoring_waits_for_the_next_session() -> None:
    world = _world(_ENTRY)

    outcome = _only(world.run(as_of=PointInTime("2026-10-09T12:00:00Z")))

    assert (outcome.status_before, outcome.status_after) == (
        Status.WAITING_FOR_ENTRY,
        Status.ACTIVE,
    )
    assert outcome.stop_reason is Stop.MISSING_BAR
    assert outcome.steps_recorded == 1
    assert world.lifecycle().status is Status.ACTIVE


@pytest.mark.parametrize(
    "exit_prices, status, ordering",
    [
        (_STOP, Status.CLOSED_STOP, IntrabarOrdering.NOT_REQUIRED),
        (_TARGET, Status.CLOSED_TARGET, IntrabarOrdering.NOT_REQUIRED),
        (_BOTH, Status.CLOSED_STOP, IntrabarOrdering.MODELED_STOP_FIRST),
    ],
    ids=["stop", "target", "stop-first"],
)
def test_an_active_recommendation_closes_and_nothing_later_is_fed(
    exit_prices, status: Status, ordering: IntrabarOrdering
) -> None:
    world = _world(_ENTRY, exit_prices, _TARGET, _TARGET, _TARGET)

    outcome = _only(world.run())

    assert outcome.stop_reason is Stop.TERMINAL and outcome.status_after is status
    assert outcome.steps_recorded == 2
    assert world.lifecycle().events[-1].ordering is ordering
    assert world.lifecycle().last_observation == _bar(date(2026, 10, 9), exit_prices)


def test_an_ambiguous_entry_bar_is_indeterminate() -> None:
    world = _world(_STOP)

    outcome = _only(world.run())

    assert outcome.status_after is Status.INDETERMINATE_ENTRY
    assert world.lifecycle().events[0].ordering is IntrabarOrdering.UNKNOWN


def test_quiet_bars_are_recorded_as_steps_and_advance_the_cursor() -> None:
    world = _world(_QUIET, _QUIET, _QUIET)

    outcome = _only(world.run(as_of=PointInTime("2026-10-13T00:00:00Z")))

    assert outcome.stop_reason is Stop.UP_TO_DATE and outcome.steps_recorded == 3
    assert world.lifecycle().events == ()
    assert world.lifecycle().last_observation == _bar(date(2026, 10, 12))
    assert all(isinstance(s, FuturesTradeRecommendationObservationStep) for s in world.steps())
    assert [s.trading_date for s in world.steps()] == list(_WINDOW_DAYS[:3])


def test_the_validity_end_bar_expires_through_advance_with_no_time_expiry() -> None:
    world = _world(_QUIET, _QUIET, _QUIET, _QUIET, _QUIET)

    outcome = _only(world.run())

    assert outcome.stop_reason is Stop.TERMINAL
    assert outcome.status_after is Status.EXPIRED_BEFORE_ENTRY
    assert world.lifecycle().events[-1].kind is RecommendationEventKind.VALIDITY_EXPIRED
    assert not any(isinstance(s, FuturesTradeRecommendationTimeExpiryStep) for s in world.steps())


def test_an_entry_on_the_validity_end_bar_activates_and_expires_in_one_step() -> None:
    world = _world(_QUIET, _QUIET, _QUIET, _QUIET, _ENTRY)

    outcome = _only(world.run())

    assert outcome.status_after is Status.EXPIRED_AFTER_ACTIVATION
    assert [e.kind for e in world.lifecycle().events] == [
        RecommendationEventKind.ENTRY_ACTIVATED,
        RecommendationEventKind.VALIDITY_EXPIRED,
    ]
    assert outcome.steps_recorded == 5


def test_a_terminal_bar_is_never_followed_by_a_second_expiry() -> None:
    world = _world(_ENTRY, _STOP)
    world.run()
    recorded = list(world.store.recorded)

    second = world.run(declarations=())

    assert second.outcomes == () and world.store.recorded == recorded


# ---------------------------------------------------------------------------
# B. Cursor, issuance timing and as_of
# ---------------------------------------------------------------------------


def test_monitoring_starts_exactly_at_eligible_from_and_never_before() -> None:
    world = _world(_QUIET)
    world.bars = (_bar(date(2026, 10, 7), _ENTRY), *world.bars)  # the issuance session

    _only(world.run(as_of=PointInTime("2026-10-08T12:00:00Z")))

    assert [s.bar.point_in_time for s in world.steps()] == [_close(date(2026, 10, 8))]
    assert world.lifecycle().status is Status.WAITING_FOR_ENTRY


def test_a_session_closing_at_as_of_is_not_yet_evaluated() -> None:
    world = _world(_ENTRY)

    outcome = _only(world.run(as_of=_close(date(2026, 10, 8))))

    assert outcome.stop_reason is Stop.UP_TO_DATE and outcome.steps_recorded == 0
    assert world.store.recorded == []


def test_a_stored_cursor_resumes_at_the_next_session_and_repeats_nothing() -> None:
    world = _world(_QUIET, _QUIET, _ENTRY)
    observed = FuturesTradeRecommendationLifecycle(_recommendation()).advance(_bar(_WINDOW_DAYS[0]))
    world.store = Store(observed)

    _only(world.run(as_of=PointInTime("2026-10-12T12:00:00Z")))

    assert [s.trading_date for s in world.steps()] == [_WINDOW_DAYS[1], _WINDOW_DAYS[2]]
    assert world.lifecycle().status is Status.ACTIVE


def test_bars_after_valid_through_are_never_evaluated() -> None:
    world = _world(_QUIET, _QUIET, _QUIET, _QUIET, None)
    world.bars = (*world.bars, _bar(date(2026, 10, 15), _ENTRY))

    outcome = _only(world.run(as_of=PointInTime("2026-10-16T00:00:00Z")))

    assert outcome.stop_reason is Stop.AWAITING_EVIDENCE_PAST_VALIDITY
    assert outcome.evidence_gap is Gap.MISSING_BAR
    assert world.lifecycle().last_observation == _bar(_WINDOW_DAYS[3])


def test_the_use_case_reads_no_clock() -> None:
    path = Path(services.__file__).parent / "monitor_futures_trade_recommendations.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert not calls & {"now", "utcnow", "today", "time", "monotonic", "uuid4", "random"}


# ---------------------------------------------------------------------------
# C. Session gaps
# ---------------------------------------------------------------------------


def test_a_missing_intermediate_bar_stops_and_keeps_earlier_progress() -> None:
    world = _world(_QUIET, None, _ENTRY, _TARGET, _TARGET)

    outcome = _only(world.run(as_of=PointInTime("2026-10-13T12:00:00Z")))

    assert outcome.stop_reason is Stop.MISSING_BAR and outcome.evidence_gap is Gap.MISSING_BAR
    assert outcome.blocking_session.trading_date == date(2026, 10, 9)
    assert outcome.finality.is_final
    assert outcome.steps_recorded == 1
    assert world.lifecycle().status is Status.WAITING_FOR_ENTRY
    assert world.lifecycle().last_observation == _bar(_WINDOW_DAYS[0])


def test_a_session_that_is_not_yet_final_stops_even_with_a_stored_bar() -> None:
    world = _world(_QUIET, _ENTRY, final_through=date(2026, 10, 8))

    outcome = _only(world.run(as_of=PointInTime("2026-10-09T12:00:00Z")))

    assert outcome.stop_reason is Stop.SESSION_NOT_FINAL
    assert outcome.finality is not None and not outcome.finality.is_final
    assert world.lifecycle().status is Status.WAITING_FOR_ENTRY and outcome.steps_recorded == 1


def test_unknown_finality_never_counts_as_final() -> None:
    world = _world(_ENTRY, final_through=None)

    outcome = _only(world.run(as_of=PointInTime("2026-10-09T00:00:00Z")))

    assert outcome.stop_reason is Stop.FINALITY_UNKNOWN and outcome.steps_recorded == 0
    assert world.store.recorded == []


def test_an_unapproved_candle_stops_monitoring() -> None:
    world = _world(_QUIET, _ENTRY)
    world.approved = world.bars[:1]

    outcome = _only(world.run(as_of=PointInTime("2026-10-09T12:00:00Z")))

    assert outcome.stop_reason is Stop.UNAPPROVED_CANDLE
    assert outcome.blocking_session.trading_date == date(2026, 10, 9)
    assert world.lifecycle().status is Status.WAITING_FOR_ENTRY


def test_a_candle_differing_from_its_approved_version_raises_and_records_nothing() -> None:
    world = _world(_QUIET, _ENTRY)
    world.approved = (world.bars[0], _bar(date(2026, 10, 9), _QUIET))

    with pytest.raises(FuturesCandleApprovalMismatchError):
        world.run(declarations=(_declaration(),))
    assert world.store.recorded == []


@pytest.mark.parametrize("day", [date(2026, 10, 10), date(2026, 10, 11)], ids=["sat", "sun"])
def test_a_stored_bar_on_no_calendar_session_raises(day: date) -> None:
    world = _world(_QUIET, _QUIET)
    world.bars = (*world.bars, _bar(day))

    with pytest.raises(FuturesRecommendationMonitoringSourceError):
        world.run()
    assert world.store.recorded == []


@pytest.mark.parametrize(
    "bar",
    [
        _bar(date(2026, 10, 8), contract=_NOV),
        _bar(date(2026, 10, 8), timeframe=Timeframe("1h")),
    ],
    ids=["other-contract", "other-timeframe"],
)
def test_a_market_answer_outside_its_contract_is_a_violation(bar: FuturesOHLCVBar) -> None:
    world = _world()

    class Wrong(Market):
        def get_bars(self, query):
            return (bar,)

    use_case = MonitorFuturesTradeRecommendationsUseCase(
        world.store, world.store, Wrong(), world.calendar, DisabledFuturesDailyBarFinalityPolicy()
    )
    with pytest.raises(FuturesTradeRecommendationContractViolationError):
        use_case.execute(product=_NIFTY, as_of=_AFTER_VALIDITY, reviewed_candles=_approvals())


def test_later_candles_never_bypass_the_first_gap() -> None:
    world = _world(_QUIET, _QUIET, None, _ENTRY, _TARGET)

    outcome = _only(world.run(declarations=()))

    assert outcome.stop_reason is Stop.AWAITING_EVIDENCE_PAST_VALIDITY
    assert all(s.trading_date < date(2026, 10, 12) for s in world.steps())
    assert world.lifecycle().status is Status.WAITING_FOR_ENTRY


def test_a_missing_validity_end_bar_holds_the_lifecycle_open() -> None:
    world = _world(_QUIET, _QUIET, _QUIET, _QUIET, None)

    outcome = _only(world.run())

    assert outcome.stop_reason is Stop.AWAITING_EVIDENCE_PAST_VALIDITY
    assert outcome.blocking_session.trading_date == date(2026, 10, 14)
    assert world.lifecycle().status is Status.WAITING_FOR_ENTRY


# ---------------------------------------------------------------------------
# D. Incomplete-coverage declarations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "configure, gap, deficiency",
    [
        (lambda w: None, Gap.MISSING_BAR, CoverageDeficiency.MISSING_OBSERVATIONS),
        (
            lambda w: setattr(w, "approved", w.bars[:2]),
            Gap.UNAPPROVED_CANDLE,
            CoverageDeficiency.UNAPPROVED_OBSERVATIONS,
        ),
        (
            lambda w: setattr(w, "final_through", date(2026, 10, 9)),
            Gap.SESSION_NOT_FINAL,
            CoverageDeficiency.UNVERIFIED_OBSERVATIONS,
        ),
    ],
    ids=["missing", "unapproved", "not-final"],
)
def test_a_declaration_expires_the_held_lifecycle_with_the_gaps_deficiency(
    configure, gap: Gap, deficiency: CoverageDeficiency
) -> None:
    prices = (_QUIET, _ENTRY, None if gap is Gap.MISSING_BAR else _QUIET, _QUIET, _QUIET)
    world = _world(*prices)
    configure(world)
    declaration = _declaration()

    outcome = _only(world.run(declarations=(declaration,)))

    assert outcome.stop_reason is Stop.EXPIRED_INCOMPLETE_BY_DECLARATION
    assert outcome.evidence_gap is gap and outcome.declaration == declaration
    assert outcome.status_after is Status.EXPIRED_WITH_INCOMPLETE_COVERAGE
    *observations, expiry = world.steps()
    assert len(observations) == 2 and outcome.steps_recorded == 3
    evidence = expiry.evidence
    assert evidence.coverage is ObservationCoverage.INCOMPLETE_OR_UNVERIFIED_COVERAGE
    assert evidence.deficiency is deficiency
    assert evidence.evaluated_through == _close(date(2026, 10, 9))
    assert evidence.as_of == _AFTER_VALIDITY
    assert evidence.attested_by == declaration.attested_by


def test_a_never_observed_recommendation_expires_with_no_evaluated_bar() -> None:
    world = _world(final_through=None)

    outcome = _only(world.run(declarations=(_declaration(),)))

    assert outcome.status_after is Status.EXPIRED_WITH_INCOMPLETE_COVERAGE
    assert outcome.evidence_gap is Gap.FINALITY_UNKNOWN
    (expiry,) = world.steps()
    assert expiry.evidence.evaluated_through is None
    assert expiry.evidence.deficiency is CoverageDeficiency.UNVERIFIED_OBSERVATIONS


def test_the_attestation_is_the_documented_canonical_encoding() -> None:
    assert _declaration().attested_by == (
        "northstar.futures-incomplete-coverage/1;monitor_policy=f5c-synthetic-v1;"
        f"operator=synthetic-operator;case=CASE-0001;recommendation={'a' * 64}"
    )
    assert _declaration().attested_by.startswith(FUTURES_INCOMPLETE_COVERAGE_ATTESTATION_FORMAT)
    assert (
        _declaration(operator_identity="Jane Doe").attested_by
        == _declaration(operator_identity="Jane Doe").attested_by
    )


def test_without_a_declaration_a_lifecycle_past_validity_stays_open() -> None:
    world = _world(_QUIET, None)

    outcome = _only(world.run())

    assert outcome.stop_reason is Stop.AWAITING_EVIDENCE_PAST_VALIDITY
    assert outcome.status_after is Status.WAITING_FOR_ENTRY
    assert outcome.declaration is None and outcome.evidence_gap is Gap.MISSING_BAR


def test_a_declaration_never_expires_before_validity_ends() -> None:
    world = _world(_QUIET, None)

    outcome = _only(
        world.run(as_of=PointInTime("2026-10-13T12:00:00Z"), declarations=(_declaration(),))
    )

    assert outcome.stop_reason is Stop.MISSING_BAR and outcome.declaration is None
    assert world.lifecycle().status is Status.WAITING_FOR_ENTRY


def test_a_declaration_is_unused_when_evidence_completes_the_window() -> None:
    world = _world(_QUIET, _QUIET, _QUIET, _QUIET, _QUIET)

    outcome = _only(world.run(declarations=(_declaration(),)))

    assert outcome.stop_reason is Stop.TERMINAL and outcome.declaration is None
    assert outcome.status_after is Status.EXPIRED_BEFORE_ENTRY


def test_no_complete_coverage_time_expiry_is_ever_made() -> None:
    source = (
        Path(services.__file__).parent / "monitor_futures_trade_recommendations.py"
    ).read_text(encoding="utf-8")
    code = ast.unparse(ast.parse(source))

    assert "COMPLETE_VERIFIED_COVERAGE" not in code


@pytest.mark.parametrize(
    "field, value",
    [
        ("operator_identity", ""),
        ("operator_identity", " synthetic-operator"),
        ("operator_identity", "op;case=x"),
        ("operator_identity", "op=1"),
        ("operator_identity", "opérateur"),
        ("operator_identity", "line\nbreak"),
        ("case_reference", ""),
        ("case_reference", "   "),
        ("monitor_policy_version", ""),
        ("monitor_policy_version", None),
        ("recommendation_identity", "a" * 64),
    ],
)
def test_an_invalid_declaration_is_refused(field: str, value: object) -> None:
    with pytest.raises(InvalidFuturesRecommendationMonitoringError):
        _declaration(**{field: value})


def test_a_declaration_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _declaration().case_reference = "CASE-2"  # type: ignore[misc]


def test_two_declarations_for_one_recommendation_are_refused() -> None:
    world = _world(_QUIET, None)

    with pytest.raises(InvalidFuturesRecommendationMonitoringError, match="more than once"):
        world.run(declarations=(_declaration(), _declaration(case_reference="CASE-0002")))
    assert world.store.recorded == []


def test_a_declaration_for_an_unknown_recommendation_is_refused() -> None:
    world = _world(_QUIET, None)

    with pytest.raises(InvalidFuturesRecommendationMonitoringError, match="not an open"):
        world.run(declarations=(_declaration("b" * 64),))
    assert world.store.recorded == []


def test_a_declaration_cannot_touch_a_terminal_recommendation() -> None:
    closed = (
        FuturesTradeRecommendationLifecycle(_recommendation())
        .advance(_bar(_WINDOW_DAYS[0], _ENTRY))
        .advance(_bar(_WINDOW_DAYS[1], _STOP))
    )
    world = _world()
    world.store = Store(closed)

    with pytest.raises(InvalidFuturesRecommendationMonitoringError):
        world.run(declarations=(_declaration(),))
    assert world.lifecycle() == closed


def test_a_declaration_cannot_override_contradictory_market_evidence() -> None:
    world = _world(_QUIET, None)
    world.bars = (*world.bars, _bar(date(2026, 10, 10)))

    with pytest.raises(FuturesRecommendationMonitoringSourceError):
        world.run(declarations=(_declaration(),))
    assert world.store.recorded == []


# ---------------------------------------------------------------------------
# E. Idempotency, interruption and concurrency
# ---------------------------------------------------------------------------


def test_an_identical_rerun_records_nothing() -> None:
    world = _world(_QUIET, _ENTRY)
    first = _only(world.run(as_of=PointInTime("2026-10-09T12:00:00Z")))

    second = _only(world.run(as_of=PointInTime("2026-10-09T12:00:00Z")))

    assert first.steps_recorded == 2 and second.steps_recorded == 0
    assert len(world.store.recorded) == 1
    assert second.status_before is second.status_after is Status.ACTIVE


def test_an_interrupted_run_resumes_from_the_stored_cursor_without_losing_quiet_steps() -> None:
    world = _world(_QUIET, _QUIET, _ENTRY, _TARGET)
    _only(world.run(as_of=PointInTime("2026-10-09T12:00:00Z")))

    outcome = _only(world.run(as_of=PointInTime("2026-10-13T12:00:00Z")))

    assert [s.trading_date for s in world.steps()] == list(_WINDOW_DAYS[:4])
    assert outcome.status_after is Status.CLOSED_TARGET
    assert world.lifecycle().last_observation == _bar(_WINDOW_DAYS[3], _TARGET)


def test_concurrent_progress_is_reported_without_claiming_the_computed_state() -> None:
    world = _world(_QUIET, _ENTRY, _TARGET)

    def other_writer(store: Store) -> None:
        key = FuturesTradeRecommendationIdentity("a" * 64)
        store.stored[key] = store.stored[key].advance(_bar(_WINDOW_DAYS[0]))
        store.before_write = None

    world.store.before_write = other_writer
    outcome = _only(world.run(as_of=PointInTime("2026-10-12T12:00:00Z")))

    assert outcome.stop_reason is Stop.CONCURRENT_PROGRESS
    assert outcome.status_after is outcome.status_before is Status.WAITING_FOR_ENTRY
    assert outcome.steps_recorded == 0 and world.store.recorded == []

    resumed = _only(world.run(as_of=PointInTime("2026-10-12T12:00:00Z")))
    assert resumed.status_after is Status.CLOSED_TARGET
    assert [s.trading_date for s in world.steps()] == [_WINDOW_DAYS[1], _WINDOW_DAYS[2]]


def test_progress_already_recorded_by_another_writer_is_accepted_as_durable() -> None:
    world = _world(_QUIET, _ENTRY)

    def same_writer(store: Store) -> None:
        key = FuturesTradeRecommendationIdentity("a" * 64)
        lifecycle = (
            store.stored[key].advance(_bar(_WINDOW_DAYS[0])).advance(_bar(_WINDOW_DAYS[1], _ENTRY))
        )
        store.stored[key] = lifecycle

    world.store.before_write = same_writer
    outcome = _only(world.run(as_of=PointInTime("2026-10-09T12:00:00Z")))

    assert outcome.steps_recorded == 0 and outcome.status_after is Status.ACTIVE
    assert outcome.stop_reason is Stop.UP_TO_DATE


@pytest.mark.parametrize(
    "error",
    [
        FuturesTradeRecommendationLifecycleEvidenceError("synthetic candle changed"),
        RuntimeError("synthetic database is locked"),
    ],
    ids=["evidence", "storage"],
)
def test_evidence_and_storage_failures_propagate(error: Exception) -> None:
    world = _world(_QUIET)

    def failing(store: Store) -> None:
        raise error

    world.store.before_write = failing
    with pytest.raises(type(error)):
        world.run(as_of=PointInTime("2026-10-08T12:00:00Z"))
    assert world.lifecycle().last_observation is None


def test_a_store_recording_part_of_the_steps_is_a_contract_violation() -> None:
    world = _world(_QUIET, _QUIET)

    class Partial(Store):
        def record_progress(self, previous, current, *, steps):
            return 1

    world.store = Partial(world.lifecycle())
    with pytest.raises(FuturesTradeRecommendationContractViolationError):
        world.run(as_of=PointInTime("2026-10-09T12:00:00Z"))


# ---------------------------------------------------------------------------
# F. Multiple recommendations
# ---------------------------------------------------------------------------


def test_terminal_histories_are_not_monitored() -> None:
    closed = FuturesTradeRecommendationLifecycle(_recommendation(identity="b" * 64)).advance(
        _bar(_WINDOW_DAYS[0], _STOP)
    )
    world = _world(_QUIET)
    world.store.stored[closed.recommendation.identity] = closed

    result = world.run(as_of=PointInTime("2026-10-08T12:00:00Z"))

    assert [o.recommendation.identity for o in result.outcomes] == ["a" * 64]
    assert result.open_anomaly is False
    assert world.lifecycle("b" * 64) == closed


def test_several_open_recommendations_are_each_monitored_and_flagged() -> None:
    oct_rec = _recommendation(identity="c" * 64)
    nov_rec = _recommendation(_NOV, identity="b" * 64)
    world = World(
        bars=(
            _bar(_WINDOW_DAYS[0], _ENTRY),
            _bar(_WINDOW_DAYS[0], _QUIET, contract=_NOV),
        ),
        store=Store(
            FuturesTradeRecommendationLifecycle(oct_rec),
            FuturesTradeRecommendationLifecycle(nov_rec),
        ),
    )

    result = world.run(as_of=PointInTime("2026-10-08T12:00:00Z"))

    assert result.open_anomaly is True
    assert [o.recommendation.identity for o in result.outcomes] == ["b" * 64, "c" * 64]
    assert [o.status_after for o in result.outcomes] == [Status.WAITING_FOR_ENTRY, Status.ACTIVE]
    assert all(o.steps_recorded == 1 for o in result.outcomes)


def test_the_anomaly_is_reported_even_when_monitoring_closes_one_of_them() -> None:
    world = World(
        bars=(_bar(_WINDOW_DAYS[0], _STOP), _bar(_WINDOW_DAYS[0], _QUIET, contract=_NOV)),
        store=Store(
            FuturesTradeRecommendationLifecycle(_recommendation(identity="c" * 64)),
            FuturesTradeRecommendationLifecycle(_recommendation(_NOV, identity="b" * 64)),
        ),
    )

    result = world.run(as_of=PointInTime("2026-10-08T12:00:00Z"))

    assert result.open_anomaly is True
    assert sorted(o.status_after for o in result.outcomes) == sorted(
        [Status.INDETERMINATE_ENTRY, Status.WAITING_FOR_ENTRY]
    )


def test_a_repository_answer_for_another_product_is_a_violation() -> None:
    world = _world()

    class Wrong(Store):
        def get_lifecycles(self, query):
            return tuple(self.stored.values())

    store = Wrong(world.lifecycle())
    use_case = MonitorFuturesTradeRecommendationsUseCase(
        store, store, Market(), world.calendar, DisabledFuturesDailyBarFinalityPolicy()
    )
    with pytest.raises(FuturesTradeRecommendationContractViolationError):
        use_case.execute(
            product=_BANKNIFTY,
            as_of=_AFTER_VALIDITY,
            reviewed_candles=_approvals(product=_BANKNIFTY),
        )


def test_approvals_for_another_product_are_refused() -> None:
    world = _world(_QUIET)

    with pytest.raises(InvalidFuturesRecommendationMonitoringError):
        MonitorFuturesTradeRecommendationsUseCase(
            world.store,
            world.store,
            Market(),
            world.calendar,
            DisabledFuturesDailyBarFinalityPolicy(),
        ).execute(
            product=_NIFTY, as_of=_AFTER_VALIDITY, reviewed_candles=_approvals(product=_BANKNIFTY)
        )


@pytest.mark.parametrize("missing", ["reviewed", "product", "as_of"])
def test_required_inputs_are_enforced(missing: str) -> None:
    world = _world()
    use_case = MonitorFuturesTradeRecommendationsUseCase(
        world.store, world.store, Market(), world.calendar, DisabledFuturesDailyBarFinalityPolicy()
    )
    arguments = {"product": _NIFTY, "as_of": _AFTER_VALIDITY, "reviewed_candles": _approvals()}
    arguments[{"reviewed": "reviewed_candles"}.get(missing, missing)] = None

    with pytest.raises(TypeError):
        use_case.execute(**arguments)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# G. Calendar coverage
# ---------------------------------------------------------------------------


def test_an_unloaded_calendar_year_fails_closed() -> None:
    world = _world(_QUIET, calendar=Calendar(years=frozenset({2025})))

    with pytest.raises(FuturesSessionResolutionError):
        world.run()
    assert world.store.recorded == []


def test_a_window_whose_end_is_no_longer_a_session_fails_closed() -> None:
    world = _world(_QUIET, calendar=Calendar(holidays=frozenset({date(2026, 10, 14)})))

    with pytest.raises(FuturesRecommendationMonitoringSourceError):
        world.run()


def test_a_calendar_session_removed_inside_the_window_is_never_treated_as_quiet() -> None:
    world = _world(_QUIET, _QUIET, _QUIET, _QUIET, _QUIET)
    world.calendar = Calendar(holidays=frozenset({date(2026, 10, 12)}))

    with pytest.raises(FuturesRecommendationMonitoringSourceError):
        world.run()  # the stored 10-12 bar is now on no session


def test_a_window_across_the_year_boundary_is_resolved_in_order() -> None:
    days = (date(2026, 12, 30), date(2026, 12, 31), date(2027, 1, 4), date(2027, 1, 5))
    recommendation = _recommendation(
        _JAN,
        eligible=days[0],
        through=days[-1],
        issued="2026-12-29T12:00:00Z",
        cutoff=date(2027, 1, 28),
    )
    calendar = Calendar(holidays=frozenset({date(2027, 1, 1)}), years=frozenset({2026, 2027}))
    world = World(
        bars=tuple(_bar(day, contract=_JAN) for day in days),
        store=Store(FuturesTradeRecommendationLifecycle(recommendation)),
        final_through=date(2027, 1, 5),
        calendar=calendar,
    )

    outcome = _only(world.run(as_of=PointInTime("2027-01-06T00:00:00Z")))

    assert outcome.status_after is Status.EXPIRED_BEFORE_ENTRY
    assert [s.trading_date for s in world.steps()] == list(days)


# ---------------------------------------------------------------------------
# H. Boundaries and exports
# ---------------------------------------------------------------------------


def test_the_module_has_no_outward_dependency() -> None:
    path = Path(services.__file__).parent / "monitor_futures_trade_recommendations.py"
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
            (
                "sqlite3",
                "northstar_infrastructure",
                "northstar_api",
                "northstar_core.paper_trading",
                "northstar_core.broker_execution",
                "upstox",
                "requests",
                "httpx",
                "time",
                "random",
                "uuid",
            )
        ), name


def test_public_exports() -> None:
    for name in (
        "MonitorFuturesTradeRecommendationsUseCase",
        "FuturesIncompleteCoverageDeclaration",
        "FuturesRecommendationMonitoringResult",
        "FuturesRecommendationMonitoringOutcome",
        "FuturesRecommendationMonitoringStopReason",
        "FuturesRecommendationEvidenceGap",
        "FuturesRecommendationMonitoringSourceError",
        "InvalidFuturesRecommendationMonitoringError",
        "FUTURES_INCOMPLETE_COVERAGE_ATTESTATION_FORMAT",
    ):
        assert name in services.__all__


def test_an_outcome_cannot_claim_contradictory_facts() -> None:
    with pytest.raises(InvalidFuturesRecommendationMonitoringError):
        FuturesRecommendationMonitoringOutcome(
            recommendation=FuturesTradeRecommendationIdentity("a" * 64),
            contract=_OCT,
            status_before=Status.WAITING_FOR_ENTRY,
            status_after=Status.WAITING_FOR_ENTRY,
            steps_recorded=0,
            stop_reason=Stop.TERMINAL,
        )
    with pytest.raises(InvalidFuturesRecommendationMonitoringError):
        FuturesRecommendationMonitoringOutcome(
            recommendation=FuturesTradeRecommendationIdentity("a" * 64),
            contract=_OCT,
            status_before=Status.WAITING_FOR_ENTRY,
            status_after=Status.WAITING_FOR_ENTRY,
            steps_recorded=0,
            stop_reason=Stop.MISSING_BAR,
        )


def test_apply_step_is_the_only_lifecycle_transition_used() -> None:
    """The monitor delegates every transition to Core through the port's helper."""
    source = (
        Path(services.__file__).parent / "monitor_futures_trade_recommendations.py"
    ).read_text(encoding="utf-8")
    assert apply_futures_trade_recommendation_lifecycle_step.__name__ in source
    for forbidden in (
        "_touches_entry",
        "_touches_stop",
        "_touches_target",
        ".entry.low",
        ".stop.price",
    ):
        assert forbidden not in source
