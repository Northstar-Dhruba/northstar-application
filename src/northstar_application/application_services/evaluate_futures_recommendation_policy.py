"""Pure historical evaluation of one futures recommendation policy (R-0).

``EvaluateFuturesRecommendationPolicyUseCase.execute(request)`` replays one
product's stored daily history and produces exactly the recommendations, and
the F-2 lifecycle outcomes, that Northstar's existing rules permit. It is an
evaluation harness, not a strategy engine: every decision belongs to an
existing rule.

- research: the built-in strategy is replayed point-in-time with
  ReplayFuturesHistoricalMarketDataUseCase and AnalyzeFuturesReplaySnapshotService,
  exactly as historical research does: the decision at a session sees only the
  bars up to that session's close;
- F-3 FuturesCandidateSelection, F-4 FuturesRecommendationGeneration and the
  F-2 lifecycle's ``advance`` decide everything else, with the Application's
  session windows (B1) and deterministic recommendation identities.

It writes nothing, contacts no provider, reads no clock and approves nothing.

Three boundaries
----------------
A. Decision cutoff -- each decision session's close. Research, F-3 and F-4 at
   that session read nothing later. Issuance is that close: the earliest
   instant the decision exists. By B1 the first eligible observation is then
   the next session, as for any issuance before the next session opens.
B. Outcome horizon -- later stored bars, used only to advance recommendations
   already issued, in strict session order.
C. End of evidence -- ``request.end``. Nothing after it is read. A lifecycle
   still open there is reported as OPEN_HORIZON_ENDED, never as finished.

Adding bars after a decision cutoff can never change what was recommended at
it; it can legitimately advance that recommendation's later lifecycle.

Evidence
--------
The request names its evidence classification. Only SYNTHETIC is supported: it
validates the evaluator's mechanics and is never market evidence, never an
operator-reviewed candle and never a basis for a profitability claim.
Operational candle approvals are not used or bypassed here; the production
generation and monitoring use cases are untouched.

Each candidate's research history must be structurally complete: the N + 1
consecutive calendar sessions ending at the decision each have their stored
bar, and no stored bar in that span is off-session. Otherwise the session's
generation is BLOCKED with HISTORY_INCOMPLETE, as production blocks on
unverified history. A contract is a candidate at a session only if it has
stored research there, so a caller's contract universe is never taken as proof
that a contract traded. Every contract needs an explicit cutoff from the
injected FuturesContractCutoffSource, cross-checked by the window resolver;
a missing one fails the whole evaluation. None is derived from an expiration
date.

Lifecycle replay
----------------
Sessions are processed in calendar order. At each session open recommendations
are first advanced (monitor before generate), then the session's research is
selected and generated, so a recommendation issued at a session is first
observed at the next. A recommendation's session without a stored bar is a gap:
the recommendation stops there for the rest of the evaluation, reported as
OPEN_EVIDENCE_GAP, and -- being open -- keeps blocking its product under P-1,
as it would in operation. No time expiry is attested and no gap is quiet.

What is not evaluated
---------------------
F-2 statuses are lifecycle classifications, not fills. No entry or exit price,
fill, slippage, brokerage, charge, profit, expectancy, Sharpe ratio or drawdown
is computed; those need valuation conventions the owner has not approved. The
result carries only descriptive counts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum

from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.futures import FuturesContract, FuturesOHLCVBar, FuturesProductReference
from northstar_core.recommendations.futures import (
    CandidateRejectionReason,
    FuturesCandidateEvidence,
    FuturesCandidateSelection,
    FuturesCandidateSelectionCriteria,
    FuturesRecommendationGeneration,
    FuturesRecommendationGenerationPolicy,
    FuturesResearchDecision,
    FuturesResearchDecisionReference,
    FuturesTradeRecommendation,
    FuturesTradeRecommendationIdentity,
    FuturesTradeRecommendationLifecycle,
    GenerationRejectionReason,
    RecommendationStatus,
)
from northstar_core.strategy import FuturesAssetAnalysisGenerator, Strategy

from northstar_application.application_services.analyze_futures_replay_snapshot import (
    AnalyzeFuturesReplaySnapshotService,
)
from northstar_application.application_services.futures_analysis_result import (
    FuturesAnalysisResult,
)
from northstar_application.application_services.futures_recommendation_identities import (
    FuturesTradeRecommendationIdentityService,
)
from northstar_application.application_services.futures_recommendation_profile import (
    FuturesTradeRecommendationGenerationProfile,
)
from northstar_application.application_services.futures_recommendation_session_windows import (
    FuturesRecommendationSessionWindowResolver,
)
from northstar_application.application_services.generate_futures_trade_recommendations import (
    FuturesTradeRecommendationContractViolationError,
)
from northstar_application.application_services.replay_futures_historical_market_data import (
    ReplayFuturesHistoricalMarketDataUseCase,
)
from northstar_application.ports.futures_contract_cutoff_source import (
    FuturesContractCutoff,
    FuturesContractCutoffSource,
)
from northstar_application.ports.futures_historical_market_data_repository import (
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
)
from northstar_application.ports.futures_trading_session_resolver import (
    FuturesTradingSession,
    FuturesTradingSessionResolver,
)

_DAILY = Timeframe("1d")
_SESSION_MARGIN = timedelta(days=1)
_HISTORY_CHUNK = timedelta(days=31)


class InvalidFuturesRecommendationEvaluationError(ValueError):
    """Raised when an evaluation request is invalid."""


class FuturesRecommendationEvaluationEvidenceError(ValueError):
    """Raised when required historical evidence is missing or contradictory."""


class FuturesEvaluationEvidenceClassification(StrEnum):
    """What an evaluation's data is. Only SYNTHETIC is supported in R-0."""

    SYNTHETIC = "SYNTHETIC"


@dataclass(frozen=True, slots=True)
class FuturesEvaluationEvidence:
    """The classification and named dataset an evaluation is run on."""

    classification: FuturesEvaluationEvidenceClassification
    dataset_reference: str

    def __post_init__(self) -> None:
        if not isinstance(self.classification, FuturesEvaluationEvidenceClassification):
            raise InvalidFuturesRecommendationEvaluationError(
                "Evaluation evidence classification must be a supported "
                "FuturesEvaluationEvidenceClassification."
            )
        if not isinstance(self.dataset_reference, str) or not self.dataset_reference.strip():
            raise InvalidFuturesRecommendationEvaluationError(
                "Evaluation evidence must name its dataset."
            )

    @property
    def is_market_evidence(self) -> bool:
        """Return whether conclusions about real markets may be drawn; never for SYNTHETIC."""
        return False


@dataclass(frozen=True, slots=True)
class FuturesRecommendationEvaluationRequest:
    """One product, contract universe, policy, profile, window and evidence classification.

    ``start`` and ``end`` bound the decision sessions by their close,
    inclusive; ``end`` is also the end of evidence: no later bar is read.
    """

    product: FuturesProductReference
    contracts: tuple[FuturesContract, ...]
    strategy: Strategy
    policy: FuturesRecommendationGenerationPolicy
    profile: FuturesTradeRecommendationGenerationProfile
    start: PointInTime
    end: PointInTime
    evidence: FuturesEvaluationEvidence

    def __post_init__(self) -> None:
        def require(value: object, expected: type, name: str) -> None:
            if not isinstance(value, expected):
                raise InvalidFuturesRecommendationEvaluationError(
                    f"Evaluation request {name} must be a {expected.__name__}."
                )

        require(self.product, FuturesProductReference, "product")
        require(self.strategy, Strategy, "strategy")
        require(self.policy, FuturesRecommendationGenerationPolicy, "policy")
        require(self.profile, FuturesTradeRecommendationGenerationProfile, "profile")
        require(self.start, PointInTime, "start")
        require(self.end, PointInTime, "end")
        require(self.evidence, FuturesEvaluationEvidence, "evidence")
        if (
            not isinstance(self.contracts, tuple)
            or not self.contracts
            or not all(isinstance(c, FuturesContract) for c in self.contracts)
        ):
            raise InvalidFuturesRecommendationEvaluationError(
                "Evaluation request contracts must be a non-empty tuple of FuturesContract values."
            )
        if len(set(self.contracts)) != len(self.contracts):
            raise InvalidFuturesRecommendationEvaluationError(
                "Evaluation request contracts name one contract more than once."
            )
        if any(contract.product != self.product for contract in self.contracts):
            raise InvalidFuturesRecommendationEvaluationError(
                f"Every evaluated contract must be of {self.product}."
            )
        if self.start.compare(self.end) > 0:
            raise InvalidFuturesRecommendationEvaluationError(
                "Evaluation start cannot be after its end."
            )
        if self.profile.policy_version != self.policy.version:
            raise InvalidFuturesRecommendationEvaluationError(
                f"Profile {self.profile.version} applies to {self.profile.policy_version}, "
                f"not {self.policy.version}."
            )
        if self.strategy.strategy_identity not in self.policy.approved_strategies:
            raise InvalidFuturesRecommendationEvaluationError(
                f"Policy {self.policy.version} does not approve strategy "
                f"{self.strategy.strategy_identity}."
            )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class FuturesEvaluatedLifecycleState(StrEnum):
    """Where an evaluated recommendation's lifecycle stands at the end of evidence."""

    TERMINAL = "TERMINAL"
    OPEN_EVIDENCE_GAP = "OPEN_EVIDENCE_GAP"
    OPEN_HORIZON_ENDED = "OPEN_HORIZON_ENDED"


class FuturesEvaluationSessionBlock(StrEnum):
    """Why a session's generation did not run."""

    HISTORY_INCOMPLETE = "HISTORY_INCOMPLETE"


@dataclass(frozen=True, slots=True)
class FuturesRecommendationEvaluationRecord:
    """One generated recommendation and its evaluated F-2 lifecycle.

    ``lifecycle`` is Core's own lifecycle after every evaluated observation;
    its status is a lifecycle classification, never a fill or a trade result.
    ``gap_session`` is the first session of its window without a stored bar,
    where evaluation of this recommendation stopped.
    """

    lifecycle: FuturesTradeRecommendationLifecycle
    observations: int
    state: FuturesEvaluatedLifecycleState
    gap_session: FuturesTradingSession | None = None

    @property
    def recommendation(self) -> FuturesTradeRecommendation:
        return self.lifecycle.recommendation

    @property
    def identity(self) -> FuturesTradeRecommendationIdentity:
        return self.lifecycle.recommendation.identity

    @property
    def status(self) -> RecommendationStatus:
        return self.lifecycle.status


@dataclass(frozen=True, slots=True)
class FuturesEvaluationSessionRecord:
    """What one decision session's research, F-3 and F-4 produced.

    ``selection`` is None when no contract had research at the session;
    ``generation`` is None when nothing was selected or the session was
    blocked. ``recommended`` lists the identities issued at the session.
    """

    session: FuturesTradingSession
    selection: FuturesCandidateSelection | None
    generation: FuturesRecommendationGeneration | None
    recommended: tuple[FuturesTradeRecommendationIdentity, ...] = ()
    block: FuturesEvaluationSessionBlock | None = None
    history_deficient_sessions: tuple[date, ...] = ()


@dataclass(frozen=True, slots=True)
class FuturesRecommendationEvaluationResult:
    """The complete, reproducible evaluation of one policy over one window."""

    product: FuturesProductReference
    policy_version: str
    profile_version: str
    start: PointInTime
    end: PointInTime
    evidence: FuturesEvaluationEvidence
    sessions: tuple[FuturesEvaluationSessionRecord, ...]
    recommendations: tuple[FuturesRecommendationEvaluationRecord, ...]

    @property
    def research_decision_count(self) -> int:
        """Return how many research decisions the sessions held, selected or not."""
        return sum(
            len(s.selection.candidates) + len(s.selection.rejections)
            for s in self.sessions
            if s.selection is not None
        )

    @property
    def candidate_rejections(self) -> tuple[tuple[CandidateRejectionReason, int], ...]:
        counts = Counter(
            rejection.reason
            for s in self.sessions
            if s.selection is not None
            for rejection in s.selection.rejections
        )
        return tuple(
            (reason, counts[reason]) for reason in CandidateRejectionReason if counts[reason]
        )

    @property
    def generation_rejections(self) -> tuple[tuple[GenerationRejectionReason, int], ...]:
        counts = Counter(
            rejection.reason
            for s in self.sessions
            if s.generation is not None
            for rejection in s.generation.rejections
        )
        return tuple(
            (reason, counts[reason]) for reason in GenerationRejectionReason if counts[reason]
        )

    @property
    def blocked_session_count(self) -> int:
        return sum(1 for s in self.sessions if s.block is not None)

    @property
    def generated_count(self) -> int:
        return len(self.recommendations)

    @property
    def status_distribution(self) -> tuple[tuple[RecommendationStatus, int], ...]:
        """Return every final F-2 status with its count, open ones included."""
        counts = Counter(record.status for record in self.recommendations)
        return tuple((status, counts[status]) for status in RecommendationStatus if counts[status])

    @property
    def state_distribution(self) -> tuple[tuple[FuturesEvaluatedLifecycleState, int], ...]:
        counts = Counter(record.state for record in self.recommendations)
        return tuple(
            (state, counts[state]) for state in FuturesEvaluatedLifecycleState if counts[state]
        )

    @property
    def open_recommendations(self) -> tuple[FuturesRecommendationEvaluationRecord, ...]:
        return tuple(
            r
            for r in self.recommendations
            if r.state is not FuturesEvaluatedLifecycleState.TERMINAL
        )


# ---------------------------------------------------------------------------
# The use case
# ---------------------------------------------------------------------------


def _utc_date(instant: PointInTime) -> date:
    return datetime.fromisoformat(instant.value.replace("Z", "+00:00")).astimezone(UTC).date()


@dataclass
class _Open:
    lifecycle: FuturesTradeRecommendationLifecycle
    observations: int = 0
    gap_session: FuturesTradingSession | None = None


class EvaluateFuturesRecommendationPolicyUseCase:
    """Replays one product's history through research, F-3, F-4 and F-2; writes nothing."""

    def __init__(
        self,
        market: FuturesHistoricalMarketDataRepository,
        sessions: FuturesTradingSessionResolver,
        cutoffs: FuturesContractCutoffSource,
        analysis_generator: FuturesAssetAnalysisGenerator,
    ) -> None:
        for value, expected, name in (
            (market, FuturesHistoricalMarketDataRepository, "market"),
            (sessions, FuturesTradingSessionResolver, "sessions"),
            (cutoffs, FuturesContractCutoffSource, "cutoffs"),
            (analysis_generator, FuturesAssetAnalysisGenerator, "analysis_generator"),
        ):
            if not isinstance(value, expected):
                raise TypeError(
                    f"EvaluateFuturesRecommendationPolicyUseCase {name} must be a "
                    f"{expected.__name__}."
                )
        self._market = market
        self._sessions = sessions
        self._cutoffs = cutoffs
        self._generator = analysis_generator
        self._replay = ReplayFuturesHistoricalMarketDataUseCase(market)
        self._windows = FuturesRecommendationSessionWindowResolver(sessions)
        self._identities = FuturesTradeRecommendationIdentityService()

    def execute(
        self, request: FuturesRecommendationEvaluationRequest
    ) -> FuturesRecommendationEvaluationResult:
        """Return the evaluation of ``request``, using no evidence after ``request.end``."""
        if not isinstance(request, FuturesRecommendationEvaluationRequest):
            raise TypeError(
                "The evaluation request must be a FuturesRecommendationEvaluationRequest."
            )
        contracts = tuple(sorted(request.contracts, key=lambda c: c.expiration_date.value))
        cutoffs = {contract: self._cutoff(contract) for contract in contracts}
        sessions = self._evaluation_sessions(request)
        bars = {contract: self._bars(contract, request.end) for contract in contracts}
        research = {
            contract: self._research(contract, request.strategy, request.end)
            for contract in contracts
        }

        open_: dict[FuturesTradeRecommendationIdentity, _Open] = {}
        records: list[FuturesEvaluationSessionRecord] = []
        for session in sessions:
            # Monitor before generate: earlier recommendations see this session first.
            for identity in sorted(open_, key=lambda i: i.identity):
                self._observe(open_[identity], session, bars)
            records.append(
                self._generate(request, session, contracts, cutoffs, bars, research, open_)
            )

        evaluated = tuple(
            self._record(state, request.end)
            for _, state in sorted(open_.items(), key=lambda item: item[0].identity)
        )
        return FuturesRecommendationEvaluationResult(
            product=request.product,
            policy_version=request.policy.version.version,
            profile_version=request.profile.version,
            start=request.start,
            end=request.end,
            evidence=request.evidence,
            sessions=tuple(records),
            recommendations=evaluated,
        )

    # -- evidence -------------------------------------------------------------------------

    def _cutoff(self, contract: FuturesContract) -> FuturesContractCutoff:
        cutoff = self._cutoffs.resolve(contract)
        if cutoff is None:
            raise FuturesRecommendationEvaluationEvidenceError(
                f"No historical cutoff evidence is known for {contract}; none is derived "
                "from its expiration date."
            )
        if not isinstance(cutoff, FuturesContractCutoff) or cutoff.contract != contract:
            raise FuturesTradeRecommendationContractViolationError(
                f"The cutoff source answered {contract} with another contract's cutoff."
            )
        return cutoff

    def _evaluation_sessions(
        self, request: FuturesRecommendationEvaluationRequest
    ) -> tuple[FuturesTradingSession, ...]:
        resolved = self._sessions.sessions_in_range(
            request.product,
            _utc_date(request.start) - _SESSION_MARGIN,
            _utc_date(request.end) + _SESSION_MARGIN,
        )
        return tuple(
            s
            for s in resolved
            if s.closes_at.compare(request.start) >= 0 and s.closes_at.compare(request.end) <= 0
        )

    def _bars(
        self, contract: FuturesContract, end: PointInTime
    ) -> dict[PointInTime, FuturesOHLCVBar]:
        stored = self._market.get_bars(FuturesHistoricalMarketDataQuery(contract, _DAILY, end=end))
        if not isinstance(stored, tuple) or not all(
            isinstance(bar, FuturesOHLCVBar)
            and bar.contract == contract
            and bar.timeframe == _DAILY
            for bar in stored
        ):
            raise FuturesTradeRecommendationContractViolationError(
                f"The market repository must return a tuple of {contract} daily bars."
            )
        if any(bar.point_in_time.compare(end) > 0 for bar in stored):
            raise FuturesTradeRecommendationContractViolationError(
                "The market repository returned a bar after the end of evidence."
            )
        return {bar.point_in_time: bar for bar in stored}

    def _research(
        self, contract: FuturesContract, strategy: Strategy, end: PointInTime
    ) -> dict[PointInTime, FuturesAnalysisResult]:
        """Replay the strategy point-in-time: each result sees only bars up to its own close."""
        analyzer = AnalyzeFuturesReplaySnapshotService(strategy, self._generator)
        results = (
            analyzer.execute(snapshot)
            for snapshot in self._replay.execute(
                FuturesHistoricalMarketDataQuery(contract, _DAILY, end=end)
            )
        )
        return {
            result.market_observation_context.observed_at: result
            for result in results
            if result is not None
        }

    # -- one session ------------------------------------------------------------------------

    def _observe(
        self,
        state: _Open,
        session: FuturesTradingSession,
        bars: dict[FuturesContract, dict[PointInTime, FuturesOHLCVBar]],
    ) -> None:
        """Advance one open recommendation by this session's bar, or stop it at a gap."""
        lifecycle = state.lifecycle
        validity = lifecycle.recommendation.validity
        if lifecycle.is_terminal or state.gap_session is not None:
            return
        if session.closes_at.compare(validity.eligible_from) < 0:
            return
        if session.closes_at.compare(validity.valid_through) > 0:
            return
        bar = bars[lifecycle.recommendation.contract].get(session.closes_at)
        if bar is None:
            state.gap_session = session
            return
        state.lifecycle = lifecycle.advance(bar)
        state.observations += 1

    def _generate(
        self,
        request: FuturesRecommendationEvaluationRequest,
        session: FuturesTradingSession,
        contracts: tuple[FuturesContract, ...],
        cutoffs: dict[FuturesContract, FuturesContractCutoff],
        bars: dict[FuturesContract, dict[PointInTime, FuturesOHLCVBar]],
        research: dict[FuturesContract, dict[PointInTime, FuturesAnalysisResult]],
        open_: dict[FuturesTradeRecommendationIdentity, _Open],
    ) -> FuturesEvaluationSessionRecord:
        at_session = {
            contract: research[contract][session.closes_at]
            for contract in contracts
            if session.closes_at in research[contract]
        }
        if not at_session:
            return FuturesEvaluationSessionRecord(session, None, None)
        sources = {
            contract: FuturesResearchDecisionReference(
                contract, _DAILY, session.closes_at, request.strategy.strategy_identity
            )
            for contract in at_session
        }
        criteria = FuturesCandidateSelectionCriteria(
            session.closes_at, (request.product,), (request.strategy.strategy_identity,)
        )
        selection = FuturesCandidateSelection.select(
            criteria,
            tuple(
                FuturesResearchDecision(sources[c], at_session[c].recommendation)
                for c in at_session
            ),
        )
        if not selection.candidates:
            return FuturesEvaluationSessionRecord(session, selection, None)

        deficient: list[date] = []
        for candidate in selection.candidates:
            deficient += self._history_gaps(
                candidate.contract, session, bars[candidate.contract], request.policy
            )
        if deficient:
            return FuturesEvaluationSessionRecord(
                session,
                selection,
                None,
                block=FuturesEvaluationSessionBlock.HISTORY_INCOMPLETE,
                history_deficient_sessions=tuple(sorted(set(deficient))),
            )

        issued_at = session.closes_at
        windows = tuple(
            self._windows.resolve(
                candidate.contract, session.closes_at, issued_at, request.profile,
                cutoffs[candidate.contract],
            )
            for candidate in selection.candidates
        )  # fmt: skip
        generation = FuturesRecommendationGeneration.generate(
            request.policy,
            selection,
            tuple(
                FuturesCandidateEvidence(
                    candidate.source, at_session[candidate.contract].market_observation_context
                )
                for candidate in selection.candidates
            ),
            windows,
            tuple(state.lifecycle for state in open_.values()),
            issued_at,
        )
        recommended = []
        for proposal in generation.proposals:
            identity = self._identities.identity(proposal.source, request.profile)
            open_[identity] = _Open(
                FuturesTradeRecommendationLifecycle(proposal.to_recommendation(identity))
            )
            recommended.append(identity)
        return FuturesEvaluationSessionRecord(session, selection, generation, tuple(recommended))

    def _history_gaps(
        self,
        contract: FuturesContract,
        session: FuturesTradingSession,
        bars: dict[PointInTime, FuturesOHLCVBar],
        policy: FuturesRecommendationGenerationPolicy,
    ) -> list[date]:
        """Return the sessions of the N + 1 consecutive history lacking a stored bar.

        A stored bar inside that span on no expected session contradicts the
        calendar and raises. Resolver errors -- an unloaded year -- propagate.
        """
        required = policy.lookback_changes + 1
        limit = session.trading_date - timedelta(days=7 * required + 62)
        found: list[FuturesTradingSession] = []
        chunk_end = session.trading_date
        while len(found) < required and chunk_end >= limit:
            chunk_start = max(chunk_end - _HISTORY_CHUNK, limit)
            found = [
                s
                for s in self._sessions.sessions_in_range(contract.product, chunk_start, chunk_end)
                if s.closes_at.compare(session.closes_at) <= 0
            ] + found
            chunk_end = chunk_start - timedelta(days=1)
        expected = found[-required:]
        if len(expected) < required:
            raise FuturesRecommendationEvaluationEvidenceError(
                f"The trading calendar does not supply {required} sessions ending at {session}."
            )
        closes = {s.closes_at for s in expected}
        first, last = expected[0].closes_at, expected[-1].closes_at
        for instant in bars:
            if instant.compare(first) >= 0 and instant.compare(last) <= 0 and instant not in closes:
                raise FuturesRecommendationEvaluationEvidenceError(
                    f"The stored bar of {contract} at {instant} is on no trading session."
                )
        return [s.trading_date for s in expected if s.closes_at not in bars]

    @staticmethod
    def _record(state: _Open, end: PointInTime) -> FuturesRecommendationEvaluationRecord:
        if state.lifecycle.is_terminal:
            kind = FuturesEvaluatedLifecycleState.TERMINAL
        elif state.gap_session is not None:
            kind = FuturesEvaluatedLifecycleState.OPEN_EVIDENCE_GAP
        else:
            kind = FuturesEvaluatedLifecycleState.OPEN_HORIZON_ENDED
        return FuturesRecommendationEvaluationRecord(
            state.lifecycle, state.observations, kind, state.gap_session
        )
