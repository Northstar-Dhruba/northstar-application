"""Application monitoring of open futures trade recommendations' lifecycles (F-5C).

``MonitorFuturesTradeRecommendationsUseCase.execute(...)`` advances every open
recommendation of one futures product, on every expiry, through the trading
sessions of its validity window, using only stored, final, operator-reviewed
daily candles, and records the evaluations through the lifecycle store
(F-5B-2A). It acquires nothing, reads no clock, and publishes nothing.

The window
----------
A recommendation's sessions are the calendar's sessions whose close lies from
``eligible_from`` through ``valid_through`` inclusive. Both ends must be actual
resolved session closes -- they were resolved from the same calendar when the
recommendation was generated -- so a session issued mid-session is never
evaluated (F-2 B1) and nothing after ``valid_through`` is ever fed. Sessions at
or before the lifecycle's ``last_observation`` were already evaluated and are
skipped; that cursor must itself be a session close of the window. Resolver
errors -- an unloaded calendar year -- propagate: no session is invented.

Each session, strictly in order
-------------------------------
Four guarantees are checked, and none stands in for another:

1. the session closed strictly before the caller's fixed ``as_of``; otherwise
   monitoring is UP_TO_DATE;
2. the finality policy says its daily bar is FINAL; NOT_YET_FINAL and UNKNOWN
   stop, as SESSION_NOT_FINAL and FINALITY_UNKNOWN;
3. a bar is stored at exactly the session's close; otherwise MISSING_BAR;
4. that bar is the operator-approved version of the session's candle, by its
   ``fdc1`` fingerprint; no approval stops as UNAPPROVED_CANDLE, and an approval
   for a different version raises FuturesCandleApprovalMismatchError.

Only then is the bar applied with Core's ``advance`` -- through
apply_futures_trade_recommendation_lifecycle_step, so no price rule is
repeated here -- and recorded as an observation step even when it produced no
event. Monitoring stops when the lifecycle becomes terminal. The first session
that fails 2-4 is a gap: no later session is ever fed, because a missing
session is never evidence that a level was not touched (F-2 section 9).

The stored fingerprint, re-checked by the lifecycle store, proves which bytes
were evaluated. It does not prove that the bar was final or operator-reviewed
when evaluated; that provenance is not persisted (F-5C D4).

Validity end and declarations
-----------------------------
A gap holding a lifecycle open once ``as_of`` is at or after ``valid_through``
is AWAITING_EVIDENCE_PAST_VALIDITY. The lifecycle stays open and keeps
blocking its product (P-1): nothing here decides that evidence is lost for
good. Only an explicit FuturesIncompleteCoverageDeclaration for that exact
recommendation lets the monitor apply Core's ``expire`` with
INCOMPLETE_OR_UNVERIFIED_COVERAGE, the deficiency of the gap that stopped it
and the declaration's structured attestation (F-5C D1, D3). A declaration
never applies before validity has ended, never to a lifecycle that is not held
open by a gap, and never overrides an error.

No complete-coverage time expiry is ever made (F-5C D2): when every session
of the window has been evaluated, the bar at ``valid_through`` has already
expired the lifecycle through ``advance``.

Persistence, retries and concurrency
------------------------------------
The steps of one recommendation are recorded in one ``record_progress`` call,
from the lifecycle loaded at the start of the run. A rerun over the same
evidence finds nothing new and writes nothing. If another writer recorded
progress first, the store's lifecycle conflict is reported as
CONCURRENT_PROGRESS with the loaded status: nothing computed here is claimed as
recorded, and the next run reloads and resumes (F-5C D6). Every other failure
propagates. Nothing here is exactly-once; a durable single writer is F-5D's.

Several open recommendations on one product are each monitored, in identity
order, and reported as ``open_anomaly`` (F-5C D5); monitoring never clears it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum

from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.futures import FuturesContract, FuturesOHLCVBar, FuturesProductReference
from northstar_core.recommendations.futures import (
    CoverageDeficiency,
    FuturesTradeRecommendationIdentity,
    FuturesTradeRecommendationLifecycle,
    ObservationCoverage,
    RecommendationStatus,
    TimeExpiryEvidence,
)

from northstar_application.application_services.acquire_futures_native_daily_history import (
    FuturesReviewedCandleVerification,
)
from northstar_application.application_services.futures_candle_approvals import (
    FuturesCandleApprovalMissingError,
)
from northstar_application.application_services.generate_futures_trade_recommendations import (
    FuturesTradeRecommendationContractViolationError,
)
from northstar_application.ports.futures_daily_bar_finality_policy import (
    FuturesDailyBarFinality,
    FuturesDailyBarFinalityOutcome,
    FuturesDailyBarFinalityPolicy,
)
from northstar_application.ports.futures_historical_market_data_repository import (
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
)
from northstar_application.ports.futures_trade_recommendation_lifecycle_store import (
    FuturesTradeRecommendationLifecycleConflictError,
    FuturesTradeRecommendationLifecycleStep,
    FuturesTradeRecommendationLifecycleStore,
    FuturesTradeRecommendationObservationStep,
    FuturesTradeRecommendationTimeExpiryStep,
    apply_futures_trade_recommendation_lifecycle_step,
)
from northstar_application.ports.futures_trade_recommendation_repository import (
    FuturesTradeRecommendationQuery,
    FuturesTradeRecommendationRepository,
)
from northstar_application.ports.futures_trading_session_resolver import (
    FuturesTradingSession,
    FuturesTradingSessionResolver,
)

_DAILY = Timeframe("1d")
# A session label may differ from its close's UTC date by a day.
_SESSION_MARGIN = timedelta(days=1)

FUTURES_INCOMPLETE_COVERAGE_ATTESTATION_FORMAT = "northstar.futures-incomplete-coverage/1"


class InvalidFuturesRecommendationMonitoringError(ValueError):
    """Raised when a monitoring request or declaration is invalid."""


class FuturesRecommendationMonitoringSourceError(ValueError):
    """Raised when the calendar, stored bars or a lifecycle contradict one another."""


class FuturesRecommendationEvidenceGap(StrEnum):
    """Why the first unevaluated session could not be evaluated."""

    SESSION_NOT_FINAL = "SESSION_NOT_FINAL"
    FINALITY_UNKNOWN = "FINALITY_UNKNOWN"
    MISSING_BAR = "MISSING_BAR"
    UNAPPROVED_CANDLE = "UNAPPROVED_CANDLE"

    @property
    def deficiency(self) -> CoverageDeficiency:
        """Return the F-2 coverage deficiency this gap is, if expiry is declared."""
        if self is FuturesRecommendationEvidenceGap.MISSING_BAR:
            return CoverageDeficiency.MISSING_OBSERVATIONS
        if self is FuturesRecommendationEvidenceGap.UNAPPROVED_CANDLE:
            return CoverageDeficiency.UNAPPROVED_OBSERVATIONS
        return CoverageDeficiency.UNVERIFIED_OBSERVATIONS


class FuturesRecommendationMonitoringStopReason(StrEnum):
    """Why monitoring of one recommendation stopped in this run."""

    TERMINAL = "TERMINAL"
    UP_TO_DATE = "UP_TO_DATE"
    SESSION_NOT_FINAL = "SESSION_NOT_FINAL"
    FINALITY_UNKNOWN = "FINALITY_UNKNOWN"
    MISSING_BAR = "MISSING_BAR"
    UNAPPROVED_CANDLE = "UNAPPROVED_CANDLE"
    AWAITING_EVIDENCE_PAST_VALIDITY = "AWAITING_EVIDENCE_PAST_VALIDITY"
    EXPIRED_INCOMPLETE_BY_DECLARATION = "EXPIRED_INCOMPLETE_BY_DECLARATION"
    CONCURRENT_PROGRESS = "CONCURRENT_PROGRESS"


_Stop = FuturesRecommendationMonitoringStopReason
_Gap = FuturesRecommendationEvidenceGap
_GAP_STOPS = {gap: _Stop(gap.value) for gap in _Gap}


def _declaration_text(value: object, name: str) -> str:
    """Require printable ASCII text with no ';' or '=', so the attestation stays unambiguous."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise InvalidFuturesRecommendationMonitoringError(
            f"FuturesIncompleteCoverageDeclaration {name} must be non-empty text without "
            "surrounding whitespace."
        )
    if any(not " " <= character <= "~" or character in ";=" for character in value):
        raise InvalidFuturesRecommendationMonitoringError(
            f"FuturesIncompleteCoverageDeclaration {name} must be printable ASCII without "
            "';' or '='."
        )
    return value


@dataclass(frozen=True, slots=True)
class FuturesIncompleteCoverageDeclaration:
    """An operator's accountable declaration that one recommendation may expire incomplete.

    It authorises the monitor to expire one exact recommendation with
    INCOMPLETE_OR_UNVERIFIED_COVERAGE once its validity has ended while a gap
    still holds it open. It is an operator assertion, not proof that evidence
    is permanently unavailable, and it changes no F-2 rule.

    ``attested_by`` is its one canonical encoding, recorded as the time
    evidence's attestation::

        northstar.futures-incomplete-coverage/1;monitor_policy=<v>;operator=<o>;case=<c>;recommendation=<id>

    Every part is printable ASCII without ';' or '=', so the encoding is
    unambiguous; it contains no timestamp or generated value.
    """

    recommendation_identity: FuturesTradeRecommendationIdentity
    operator_identity: str
    case_reference: str
    monitor_policy_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.recommendation_identity, FuturesTradeRecommendationIdentity):
            raise InvalidFuturesRecommendationMonitoringError(
                "FuturesIncompleteCoverageDeclaration recommendation identity must be a "
                "FuturesTradeRecommendationIdentity."
            )
        _declaration_text(self.operator_identity, "operator identity")
        _declaration_text(self.case_reference, "case reference")
        _declaration_text(self.monitor_policy_version, "monitor policy version")

    @property
    def attested_by(self) -> str:
        """Return the declaration's canonical attestation text."""
        return ";".join(
            (
                FUTURES_INCOMPLETE_COVERAGE_ATTESTATION_FORMAT,
                f"monitor_policy={self.monitor_policy_version}",
                f"operator={self.operator_identity}",
                f"case={self.case_reference}",
                f"recommendation={self.recommendation_identity.identity}",
            )
        )


@dataclass(frozen=True, slots=True)
class FuturesRecommendationMonitoringOutcome:
    """What monitoring did to one open recommendation in one run.

    - ``status_before`` is the status loaded at the start of the run;
      ``status_after`` is the durable status afterwards: the recorded one, or,
      after CONCURRENT_PROGRESS, the loaded one, since nothing computed here
      was recorded.
    - ``steps_recorded`` is what the lifecycle store reported inserting; 0
      when nothing was new, when another writer recorded the same steps, or
      on CONCURRENT_PROGRESS.
    - ``blocking_session``, ``finality`` and ``evidence_gap`` name the first
      session that could not be evaluated, its finality assessment and why;
      ``evidence_gap`` is kept under AWAITING_EVIDENCE_PAST_VALIDITY and
      EXPIRED_INCOMPLETE_BY_DECLARATION so the cause is never hidden.
    - ``declaration`` is the declaration applied, only under
      EXPIRED_INCOMPLETE_BY_DECLARATION.
    """

    recommendation: FuturesTradeRecommendationIdentity
    contract: FuturesContract
    status_before: RecommendationStatus
    status_after: RecommendationStatus
    steps_recorded: int
    stop_reason: FuturesRecommendationMonitoringStopReason
    blocking_session: FuturesTradingSession | None = None
    finality: FuturesDailyBarFinality | None = None
    evidence_gap: FuturesRecommendationEvidenceGap | None = None
    declaration: FuturesIncompleteCoverageDeclaration | None = None

    def __post_init__(self) -> None:
        gap_stops = set(_GAP_STOPS.values()) | {
            _Stop.AWAITING_EVIDENCE_PAST_VALIDITY,
            _Stop.EXPIRED_INCOMPLETE_BY_DECLARATION,
        }
        has_gap = self.stop_reason in gap_stops
        if has_gap != (self.evidence_gap is not None) or has_gap != (
            self.blocking_session is not None
        ):
            raise InvalidFuturesRecommendationMonitoringError(
                "A monitoring outcome names a blocking session and gap exactly when a gap "
                "stopped it."
            )
        if self.stop_reason in _GAP_STOPS.values() and (
            _GAP_STOPS[self.evidence_gap] is not self.stop_reason  # type: ignore[index]
        ):
            raise InvalidFuturesRecommendationMonitoringError(
                "A monitoring outcome's stop reason must be its evidence gap."
            )
        expired = self.stop_reason is _Stop.EXPIRED_INCOMPLETE_BY_DECLARATION
        if expired != (self.declaration is not None):
            raise InvalidFuturesRecommendationMonitoringError(
                "A monitoring outcome names a declaration exactly when one expired it."
            )
        if expired and (
            self.status_after is not RecommendationStatus.EXPIRED_WITH_INCOMPLETE_COVERAGE
        ):
            raise InvalidFuturesRecommendationMonitoringError(
                "A declared incomplete expiry must end EXPIRED_WITH_INCOMPLETE_COVERAGE."
            )
        ends_terminal = self.stop_reason in (
            _Stop.TERMINAL,
            _Stop.EXPIRED_INCOMPLETE_BY_DECLARATION,
        )
        if ends_terminal != self.status_after.is_terminal:
            raise InvalidFuturesRecommendationMonitoringError(
                "A monitoring outcome ends terminal exactly when its stop reason says so."
            )
        if (
            not isinstance(self.steps_recorded, int)
            or isinstance(self.steps_recorded, bool)
            or self.steps_recorded < 0
        ):
            raise InvalidFuturesRecommendationMonitoringError(
                "A monitoring outcome's recorded step count must be a non-negative integer."
            )


@dataclass(frozen=True, slots=True)
class FuturesRecommendationMonitoringResult:
    """Immutable outcome of monitoring one product's open recommendations."""

    product: FuturesProductReference
    as_of: PointInTime
    outcomes: tuple[FuturesRecommendationMonitoringOutcome, ...]
    open_anomaly: bool


def _utc_date(instant: PointInTime) -> date:
    return datetime.fromisoformat(instant.value.replace("Z", "+00:00")).astimezone(UTC).date()


@dataclass(frozen=True, slots=True)
class _Evaluation:
    lifecycle: FuturesTradeRecommendationLifecycle
    steps: tuple[FuturesTradeRecommendationLifecycleStep, ...]
    stop_reason: FuturesRecommendationMonitoringStopReason
    blocking_session: FuturesTradingSession | None = None
    finality: FuturesDailyBarFinality | None = None
    evidence_gap: FuturesRecommendationEvidenceGap | None = None
    declaration: FuturesIncompleteCoverageDeclaration | None = None


class MonitorFuturesTradeRecommendationsUseCase:
    """Advances one product's open recommendations through verified stored candles."""

    def __init__(
        self,
        recommendations: FuturesTradeRecommendationRepository,
        lifecycles: FuturesTradeRecommendationLifecycleStore,
        market: FuturesHistoricalMarketDataRepository,
        sessions: FuturesTradingSessionResolver,
        finality: FuturesDailyBarFinalityPolicy,
    ) -> None:
        for value, expected, name in (
            (recommendations, FuturesTradeRecommendationRepository, "recommendations"),
            (lifecycles, FuturesTradeRecommendationLifecycleStore, "lifecycles"),
            (market, FuturesHistoricalMarketDataRepository, "market"),
            (sessions, FuturesTradingSessionResolver, "sessions"),
            (finality, FuturesDailyBarFinalityPolicy, "finality"),
        ):
            if not isinstance(value, expected):
                raise TypeError(
                    f"MonitorFuturesTradeRecommendationsUseCase {name} must be a "
                    f"{expected.__name__}."
                )
        self._recommendations = recommendations
        self._lifecycles = lifecycles
        self._market = market
        self._sessions = sessions
        self._finality = finality

    def execute(
        self,
        *,
        product: FuturesProductReference,
        as_of: PointInTime,
        reviewed_candles: FuturesReviewedCandleVerification,
        coverage_declarations: tuple[FuturesIncompleteCoverageDeclaration, ...] = (),
    ) -> FuturesRecommendationMonitoringResult:
        """Monitor every open recommendation of ``product`` as of the fixed ``as_of``."""
        declarations = self._validate_request(
            product, as_of, reviewed_candles, coverage_declarations
        )
        lifecycles = self._load(product)
        open_lifecycles = sorted(
            (lifecycle for lifecycle in lifecycles if not lifecycle.is_terminal),
            key=lambda lifecycle: lifecycle.recommendation.identity.identity,
        )
        open_identities = {lifecycle.recommendation.identity for lifecycle in open_lifecycles}
        for identity in declarations:
            if identity not in open_identities:
                raise InvalidFuturesRecommendationMonitoringError(
                    f"A coverage declaration names {identity}, which is not an open "
                    f"recommendation of {product}."
                )

        outcomes = tuple(
            self._monitor(
                lifecycle,
                as_of,
                reviewed_candles,
                declarations.get(lifecycle.recommendation.identity),
            )
            for lifecycle in open_lifecycles
        )
        return FuturesRecommendationMonitoringResult(
            product=product,
            as_of=as_of,
            outcomes=outcomes,
            open_anomaly=len(open_lifecycles) > 1,
        )

    # -- request ------------------------------------------------------------------------

    @staticmethod
    def _validate_request(
        product: object,
        as_of: object,
        reviewed_candles: object,
        coverage_declarations: object,
    ) -> dict[FuturesTradeRecommendationIdentity, FuturesIncompleteCoverageDeclaration]:
        if not isinstance(product, FuturesProductReference):
            raise TypeError("Recommendation monitoring product must be a FuturesProductReference.")
        if not isinstance(as_of, PointInTime):
            raise TypeError("Recommendation monitoring as of must be a PointInTime.")
        if not isinstance(reviewed_candles, FuturesReviewedCandleVerification):
            raise TypeError(
                "Recommendation monitoring requires a FuturesReviewedCandleVerification."
            )
        if reviewed_candles.approvals.product != product:
            raise InvalidFuturesRecommendationMonitoringError(
                f"The candle approvals are for {reviewed_candles.approvals.product}, not {product}."
            )
        if not isinstance(coverage_declarations, tuple) or not all(
            isinstance(item, FuturesIncompleteCoverageDeclaration) for item in coverage_declarations
        ):
            raise InvalidFuturesRecommendationMonitoringError(
                "Coverage declarations must be a tuple of FuturesIncompleteCoverageDeclaration."
            )
        declarations: dict[
            FuturesTradeRecommendationIdentity, FuturesIncompleteCoverageDeclaration
        ] = {}
        for declaration in coverage_declarations:
            if declaration.recommendation_identity in declarations:
                raise InvalidFuturesRecommendationMonitoringError(
                    f"Recommendation {declaration.recommendation_identity} is declared more "
                    "than once."
                )
            declarations[declaration.recommendation_identity] = declaration
        return declarations

    def _load(
        self, product: FuturesProductReference
    ) -> tuple[FuturesTradeRecommendationLifecycle, ...]:
        lifecycles = self._recommendations.get_lifecycles(FuturesTradeRecommendationQuery(product))
        if not isinstance(lifecycles, tuple) or not all(
            isinstance(lifecycle, FuturesTradeRecommendationLifecycle)
            and lifecycle.recommendation.contract.product == product
            for lifecycle in lifecycles
        ):
            raise FuturesTradeRecommendationContractViolationError(
                f"The recommendation repository must return {product} lifecycles."
            )
        identities = [lifecycle.recommendation.identity for lifecycle in lifecycles]
        if len(set(identities)) != len(identities):
            raise FuturesTradeRecommendationContractViolationError(
                "The recommendation repository returned one recommendation more than once."
            )
        return lifecycles

    # -- one recommendation -------------------------------------------------------------

    def _monitor(
        self,
        loaded: FuturesTradeRecommendationLifecycle,
        as_of: PointInTime,
        reviewed_candles: FuturesReviewedCandleVerification,
        declaration: FuturesIncompleteCoverageDeclaration | None,
    ) -> FuturesRecommendationMonitoringOutcome:
        evaluation = self._evaluate(loaded, as_of, reviewed_candles, declaration)
        recommendation = loaded.recommendation
        recorded = 0
        if evaluation.steps:
            try:
                recorded = self._lifecycles.record_progress(
                    loaded, evaluation.lifecycle, steps=evaluation.steps
                )
            except FuturesTradeRecommendationLifecycleConflictError:
                return FuturesRecommendationMonitoringOutcome(
                    recommendation=recommendation.identity,
                    contract=recommendation.contract,
                    status_before=loaded.status,
                    status_after=loaded.status,
                    steps_recorded=0,
                    stop_reason=_Stop.CONCURRENT_PROGRESS,
                )
            if (
                not isinstance(recorded, int)
                or isinstance(recorded, bool)
                or recorded not in (0, len(evaluation.steps))
            ):
                raise FuturesTradeRecommendationContractViolationError(
                    f"The lifecycle store recorded {recorded!r} of {len(evaluation.steps)} "
                    "steps; progress must be recorded all or nothing."
                )
        return FuturesRecommendationMonitoringOutcome(
            recommendation=recommendation.identity,
            contract=recommendation.contract,
            status_before=loaded.status,
            status_after=evaluation.lifecycle.status,
            steps_recorded=recorded,
            stop_reason=evaluation.stop_reason,
            blocking_session=evaluation.blocking_session,
            finality=evaluation.finality,
            evidence_gap=evaluation.evidence_gap,
            declaration=evaluation.declaration,
        )

    def _evaluate(
        self,
        loaded: FuturesTradeRecommendationLifecycle,
        as_of: PointInTime,
        reviewed_candles: FuturesReviewedCandleVerification,
        declaration: FuturesIncompleteCoverageDeclaration | None,
    ) -> _Evaluation:
        recommendation = loaded.recommendation
        contract = recommendation.contract
        window = self._window(loaded)
        bars = self._bars(contract, window)
        cursor = loaded.last_observation
        pending = tuple(
            session
            for session in window
            if cursor is None or session.closes_at.compare(cursor.point_in_time) > 0
        )

        lifecycle = loaded
        steps: list[FuturesTradeRecommendationLifecycleStep] = []
        for session in pending:
            if as_of.compare(session.closes_at) <= 0:
                return _Evaluation(lifecycle, tuple(steps), _Stop.UP_TO_DATE)
            assessment = self._assess(contract, session)
            gap: FuturesRecommendationEvidenceGap | None = None
            if assessment.outcome is FuturesDailyBarFinalityOutcome.NOT_YET_FINAL:
                gap = _Gap.SESSION_NOT_FINAL
            elif assessment.outcome is not FuturesDailyBarFinalityOutcome.FINAL:
                gap = _Gap.FINALITY_UNKNOWN
            elif (bar := bars.get(session.closes_at)) is None:
                gap = _Gap.MISSING_BAR
            else:
                try:
                    reviewed_candles.approvals.verify_bar(contract, session.trading_date, bar)
                except FuturesCandleApprovalMissingError:
                    gap = _Gap.UNAPPROVED_CANDLE
            if gap is not None:
                return self._held(
                    lifecycle, tuple(steps), as_of, session, assessment, gap, declaration
                )
            step = FuturesTradeRecommendationObservationStep(bar, session.trading_date)
            lifecycle = apply_futures_trade_recommendation_lifecycle_step(lifecycle, step)
            steps.append(step)
            if lifecycle.is_terminal:
                return _Evaluation(lifecycle, tuple(steps), _Stop.TERMINAL)
        # Every session of the window has been evaluated, so the bar at
        # valid_through has expired the lifecycle; an open one here is impossible.
        if not lifecycle.is_terminal:
            raise FuturesRecommendationMonitoringSourceError(
                f"Recommendation {recommendation.identity} is still open after its whole "
                "validity window was evaluated."
            )
        return _Evaluation(lifecycle, tuple(steps), _Stop.TERMINAL)

    @staticmethod
    def _held(
        lifecycle: FuturesTradeRecommendationLifecycle,
        steps: tuple[FuturesTradeRecommendationLifecycleStep, ...],
        as_of: PointInTime,
        session: FuturesTradingSession,
        assessment: FuturesDailyBarFinality,
        gap: FuturesRecommendationEvidenceGap,
        declaration: FuturesIncompleteCoverageDeclaration | None,
    ) -> _Evaluation:
        """Return the evaluation held open at a gap, or expired by an explicit declaration."""
        facts = {"blocking_session": session, "finality": assessment, "evidence_gap": gap}
        valid_through = lifecycle.recommendation.validity.valid_through
        if as_of.compare(valid_through) < 0:
            return _Evaluation(lifecycle, steps, _GAP_STOPS[gap], **facts)
        if declaration is None:
            return _Evaluation(lifecycle, steps, _Stop.AWAITING_EVIDENCE_PAST_VALIDITY, **facts)
        last = lifecycle.last_observation
        step = FuturesTradeRecommendationTimeExpiryStep(
            TimeExpiryEvidence(
                as_of=as_of,
                evaluated_through=None if last is None else last.point_in_time,
                coverage=ObservationCoverage.INCOMPLETE_OR_UNVERIFIED_COVERAGE,
                attested_by=declaration.attested_by,
                deficiency=gap.deficiency,
            )
        )
        expired = apply_futures_trade_recommendation_lifecycle_step(lifecycle, step)
        return _Evaluation(
            expired,
            (*steps, step),
            _Stop.EXPIRED_INCOMPLETE_BY_DECLARATION,
            declaration=declaration,
            **facts,
        )

    # -- calendar, finality and bars ----------------------------------------------------

    def _window(
        self, lifecycle: FuturesTradeRecommendationLifecycle
    ) -> tuple[FuturesTradingSession, ...]:
        """Return the resolved sessions closing from eligible_from through valid_through."""
        recommendation = lifecycle.recommendation
        validity = recommendation.validity
        product = recommendation.contract.product
        resolved = self._sessions.sessions_in_range(
            product,
            _utc_date(validity.eligible_from) - _SESSION_MARGIN,
            _utc_date(validity.valid_through) + _SESSION_MARGIN,
        )
        if not isinstance(resolved, tuple) or not all(
            isinstance(session, FuturesTradingSession) for session in resolved
        ):
            raise FuturesTradeRecommendationContractViolationError(
                "The trading calendar must return a tuple of FuturesTradingSession values."
            )
        window = tuple(
            session
            for session in resolved
            if session.closes_at.compare(validity.eligible_from) >= 0
            and session.closes_at.compare(validity.valid_through) <= 0
        )
        for earlier, later in zip(window, window[1:], strict=False):
            if earlier.closes_at.compare(later.closes_at) >= 0:
                raise FuturesTradeRecommendationContractViolationError(
                    "The trading calendar must return sessions strictly in order."
                )
        if (
            not window
            or window[0].closes_at.compare(validity.eligible_from) != 0
            or window[-1].closes_at.compare(validity.valid_through) != 0
        ):
            raise FuturesRecommendationMonitoringSourceError(
                f"Recommendation {recommendation.identity}'s validity does not begin and end at "
                "resolved session closes of its venue's calendar."
            )
        cursor = lifecycle.last_observation
        if cursor is not None and not any(
            session.closes_at.compare(cursor.point_in_time) == 0 for session in window
        ):
            raise FuturesRecommendationMonitoringSourceError(
                f"Recommendation {recommendation.identity}'s last observation at "
                f"{cursor.point_in_time} is on no session of its validity window."
            )
        return window

    def _assess(
        self, contract: FuturesContract, session: FuturesTradingSession
    ) -> FuturesDailyBarFinality:
        assessment = self._finality.assess(contract, session)
        if not isinstance(assessment, FuturesDailyBarFinality) or (
            assessment.contract != contract or assessment.session != session
        ):
            raise FuturesTradeRecommendationContractViolationError(
                "The finality policy must assess exactly the contract and session asked about."
            )
        return assessment

    def _bars(
        self, contract: FuturesContract, window: tuple[FuturesTradingSession, ...]
    ) -> dict[PointInTime, FuturesOHLCVBar]:
        """Return the stored daily bars of the window by close; anything off-session raises."""
        bars = self._market.get_bars(
            FuturesHistoricalMarketDataQuery(
                contract, _DAILY, start=window[0].closes_at, end=window[-1].closes_at
            )
        )
        if not isinstance(bars, tuple) or not all(
            isinstance(bar, FuturesOHLCVBar)
            and bar.contract == contract
            and bar.timeframe == _DAILY
            for bar in bars
        ):
            raise FuturesTradeRecommendationContractViolationError(
                f"The market repository must return a tuple of {contract} daily bars."
            )
        for earlier, later in zip(bars, bars[1:], strict=False):
            if earlier.point_in_time.compare(later.point_in_time) >= 0:
                raise FuturesTradeRecommendationContractViolationError(
                    "The market repository must return bars strictly oldest to newest."
                )
        closes = {session.closes_at for session in window}
        by_close: dict[PointInTime, FuturesOHLCVBar] = {}
        for bar in bars:
            if bar.point_in_time.compare(window[0].closes_at) < 0 or (
                bar.point_in_time.compare(window[-1].closes_at) > 0
            ):
                continue
            if bar.point_in_time not in closes:
                raise FuturesRecommendationMonitoringSourceError(
                    f"The stored bar of {contract} at {bar.point_in_time} is on no trading "
                    "session of the recommendation's validity window."
                )
            by_close[bar.point_in_time] = bar
        return by_close
