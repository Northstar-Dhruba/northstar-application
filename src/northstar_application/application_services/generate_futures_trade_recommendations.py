"""Application generation of internal futures trade recommendations for one contract.

``GenerateFuturesTradeRecommendationsUseCase.execute(...)`` connects the frozen
futures research of one explicitly configured contract and strategy to Core's
candidate selection (F-3) and recommendation generation (F-4), assigns each
proposal its deterministic identity, and records the resulting F-2
FuturesTradeRecommendations through an Application port. Recording is internal:
nothing is published, delivered, notified or executed.

The session
-----------
The selection session is the session of the contract's latest persisted daily
bar -- persisted evidence, never the clock. It must be FINAL under the finality
policy *and* its stored bar must be the operator-reviewed version of that
session's candle, verified against the approval set. Trading-session finality,
operator review of a candle, completeness of the research history and the data
provider's contractual permission are four different guarantees; this use case
requires the first three for a new recommendation and asserts nothing about the
fourth. A later session that is not yet final, or not reviewed, blocks
generation: older research is never promoted in its place.

Research history
----------------
No new recommendation is generated unless its research history is VERIFIED.
F-4 measures price movement over the trailing N + 1 closes of the frozen
context (N is the policy's lookback; 20 closes in Futures V1). Those closes are
VERIFIED only when the injected trading calendar resolves N + 1 consecutive
sessions ending at the decision session, each session has its stored bar, each
stored close equals the frozen close in the same position, and every one of
those candles is operator-approved. Weekends and holidays are not gaps, because
the expected sequence is the calendar's sessions, never calendar days. The
number of closes alone never establishes coverage.

A missing bar, a missing approval or a calendar that cannot supply the sessions
blocks with RESEARCH_HISTORY_NOT_VERIFIED, naming the deficiency and sessions.
Evidence that contradicts itself -- a stored close that differs from the frozen
one over complete history, a stored bar on no expected session, a candle that
differs from its approved version -- raises. An unloaded calendar year
propagates its resolution error. Neither ever generates.

This gate concerns the evidence a recommendation is *created* from. Observations
found missing or unverified later, while an existing recommendation is
monitored, are a lifecycle matter (F-2 time-expiry coverage, F-5C) and are not
reclassified here.

Outcomes
--------
An expected operational condition -- no persisted bar, a session that is not
final or not reviewed, no frozen research for the session, research history that
is not verified, no verified contract cutoff -- returns a BLOCKED result naming
the reason, and nothing is generated or recorded.
Malformed or contradictory input -- a port answering outside its contract,
research that disagrees with the reviewed candle, a contradictory cutoff, an
identity recorded with different content -- raises. F-3 and F-4 rejections are
ordinary results and are returned unchanged for audit.

Retries
-------
The order is fixed: the session gates (persisted, FINAL, operator-reviewed, its
research present and agreeing with the reviewed candle); F-3 selection, whose
rejections -- HOLD, an unapproved strategy -- are ordinary results; the retry
lookup of each candidate's identity; the research-history gate for candidates
not yet recorded; then cutoff, window, F-4 and recording.

A recommendation already recorded under a candidate's identity is returned as
it was recorded -- its issuance time, levels, validity and thesis unchanged --
and that candidate is neither generated again nor re-verified: its history is
not re-assessed, so no fresh VERIFIED claim is made for it, and history that has
since become unverifiable does not hide it. The session gates still apply,
because they establish which research is the session's. A new profile or policy
version is a new identity, needs VERIFIED history, and remains subject to F-4's
one open recommendation per product.

Two attempts can race: both find the identity absent, both generate, and the
first to record wins. The loser's store refuses its definition -- it differs,
at least in issuance time -- with FuturesTradeRecommendationConflictError. Only
that error is reconciled: every identity of the batch is re-read, and each
stored recommendation must be exactly the expected one -- same identity, same
contract and research source (timeframe, decision instant, strategy), same
policy version, and the identity derived from that source under this profile.
Only then are the winners returned, unchanged, as ``existing``; the loser's
own definitions are discarded and nothing is recorded twice. A missing or
mismatching winner raises. A configuration-version conflict and a product
exclusivity conflict are never reconciled: they propagate.

The F-2 definition carries no profile contents, so a winner's configuration
equality rests on the identity's binding to the profile version and on the
store's durable version registry (F-5B-1B), which refuses a version reused with
different contents.

The issuance instant is captured once by the caller and used for the whole
operation. This use case reads no clock.

Audits
------
Every attempt that returns a result is audited with a
FuturesRecommendationGenerationAudit built from exactly that result:

- a result that newly records recommendations is the audit's source *before*
  the write, and the audit travels inside ``store.record``: the store commits
  the definitions and that RECORDED audit in one transaction or neither (I-2).
  The result is returned only once that write succeeded, so the audit always
  describes what was committed;
- every other result -- BLOCKED, a completed attempt that recorded nothing, a
  retry that returned existing recommendations, an attempt that lost the race
  and returned the winner -- is audited after the fact through the standalone
  audit store, which refuses a RECORDED audit. A loser's audit is therefore
  EXISTING_RETURNED: it shows its own proposal, and the winner as existing.

An attempt that raises -- contradictory evidence, a port outside its contract,
a configuration or exclusivity conflict, a storage failure -- returns no result
and records no audit; none of them is disguised as a rejection. If the
standalone audit cannot be recorded, its error propagates, and since nothing
was recorded the attempt may simply be retried.

Limits
------
Reads and the final write are separate port calls. Nothing here is a
transaction: a concurrent writer could record between the open-set read and the
write. The store's own semantics -- exclusivity checked and the rows and audit
inserted in one transaction -- are what make that safe, and only a durable
implementation can provide them (F-5B).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta

from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.futures import FuturesContract, FuturesOHLCVBar
from northstar_core.recommendations.futures import (
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
    InvalidFuturesCandidateEvidenceError,
    InvalidFuturesResearchDecisionError,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services.acquire_futures_native_daily_history import (
    FuturesReviewedCandleVerification,
)
from northstar_application.application_services.futures_candle_approvals import (
    FuturesCandleApprovalMissingError,
    FuturesCandleApprovalSet,
)
from northstar_application.application_services.futures_forward_research_record import (
    FuturesForwardResearchRecord,
)
from northstar_application.application_services.futures_recommendation_generation_audit import (
    FuturesRecommendationGenerationAudit,
)
from northstar_application.application_services.futures_recommendation_generation_outcome import (
    FuturesResearchHistoryCoverage,
    FuturesResearchHistoryDeficiency,
    FuturesTradeRecommendationBlockReason,
    FuturesTradeRecommendationGenerationStatus,
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
from northstar_application.ports.futures_contract_cutoff_source import (
    FuturesContractCutoffSource,
)
from northstar_application.ports.futures_daily_bar_finality_policy import (
    FuturesDailyBarFinality,
    FuturesDailyBarFinalityPolicy,
)
from northstar_application.ports.futures_forward_research_record_repository import (
    FuturesForwardResearchRecordQuery,
    FuturesForwardResearchRecordRepository,
)
from northstar_application.ports.futures_historical_market_data_repository import (
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
)
from northstar_application.ports.futures_recommendation_generation_audit_store import (
    FuturesRecommendationGenerationAuditStore,
)
from northstar_application.ports.futures_trade_recommendation_repository import (
    FuturesTradeRecommendationQuery,
    FuturesTradeRecommendationRepository,
)
from northstar_application.ports.futures_trade_recommendation_store import (
    FuturesTradeRecommendationConflictError,
    FuturesTradeRecommendationStore,
)
from northstar_application.ports.futures_trading_session_resolver import (
    FuturesTradingSession,
    FuturesTradingSessionResolver,
)

_DAILY = Timeframe("1d")
# A daily bar is stamped at its session's close; the session label may differ
# from the bar's UTC date by a day, so the lookup searches a day either side.
_BAR_SESSION_MARGIN = timedelta(days=1)
_HISTORY_CHUNK = timedelta(days=31)


class InvalidFuturesTradeRecommendationGenerationError(ValueError):
    """Raised when the generation's own configuration or inputs are invalid."""


class FuturesTradeRecommendationContractViolationError(RuntimeError):
    """Raised when a port answers outside its documented contract."""


class FuturesTradeRecommendationSourceError(ValueError):
    """Raised when frozen research contradicts the reviewed market evidence."""


@dataclass(frozen=True, slots=True)
class FuturesTradeRecommendationGenerationResult:
    """Immutable outcome of one generation for one contract and strategy.

    - ``session`` and ``finality`` describe the selection session, once known.
    - ``selection`` is F-3's complete outcome, with its rejections.
    - ``generation`` is F-4's outcome for the candidates not already recorded,
      with its rejections and exclusivity breaches; None when every candidate
      was already recorded, or there was none, so F-4 did not run.
    - ``recorded`` holds recommendations recorded by this run; ``existing``
      those already recorded under the same identity and returned unchanged.
    - ``history_coverage`` is the assessment of the research history behind the
      candidates to generate; None when none needed generating, so none was
      assessed. When UNVERIFIED, ``history_deficiency`` and
      ``history_deficient_sessions`` say why and for which sessions.

    ``recorded`` means accepted by the injected store, which may or may not be
    durable. Nothing here is published or executed.
    """

    contract: FuturesContract
    strategy: StrategyIdentity
    profile: FuturesTradeRecommendationGenerationProfile
    issued_at: PointInTime
    status: FuturesTradeRecommendationGenerationStatus
    block_reason: FuturesTradeRecommendationBlockReason | None = None
    session: FuturesTradingSession | None = None
    finality: FuturesDailyBarFinality | None = None
    history_coverage: FuturesResearchHistoryCoverage | None = None
    history_deficiency: FuturesResearchHistoryDeficiency | None = None
    history_deficient_sessions: tuple[date, ...] = ()
    selection: FuturesCandidateSelection | None = None
    generation: FuturesRecommendationGeneration | None = None
    recorded: tuple[FuturesTradeRecommendation, ...] = ()
    existing: tuple[FuturesTradeRecommendation, ...] = ()

    def __post_init__(self) -> None:
        blocked = self.status is FuturesTradeRecommendationGenerationStatus.BLOCKED
        if blocked != (self.block_reason is not None):
            raise InvalidFuturesTradeRecommendationGenerationError(
                "A generation result names a block reason exactly when it is BLOCKED."
            )
        if blocked and (self.recorded or self.generation is not None):
            raise InvalidFuturesTradeRecommendationGenerationError(
                "A BLOCKED generation result cannot carry generated or recorded recommendations."
            )
        verified = self.history_coverage is FuturesResearchHistoryCoverage.VERIFIED
        if self.recorded and not verified:
            raise InvalidFuturesTradeRecommendationGenerationError(
                "A recommendation can only be recorded from VERIFIED research history."
            )
        unverified = self.history_coverage is FuturesResearchHistoryCoverage.UNVERIFIED
        if unverified != (self.history_deficiency is not None):
            raise InvalidFuturesTradeRecommendationGenerationError(
                "A generation result names a history deficiency exactly when it is UNVERIFIED."
            )


def _utc(instant: PointInTime) -> datetime:
    return datetime.fromisoformat(instant.value.replace("Z", "+00:00")).astimezone(UTC)


def _source(record: FuturesForwardResearchRecord) -> FuturesResearchDecisionReference:
    return FuturesResearchDecisionReference(
        record.contract, record.timeframe, record.decision_instant, record.strategy_identity
    )


class GenerateFuturesTradeRecommendationsUseCase:
    """Generates and records internal trade recommendations for one contract."""

    def __init__(
        self,
        research: FuturesForwardResearchRecordRepository,
        market: FuturesHistoricalMarketDataRepository,
        sessions: FuturesTradingSessionResolver,
        finality: FuturesDailyBarFinalityPolicy,
        cutoffs: FuturesContractCutoffSource,
        recommendations: FuturesTradeRecommendationRepository,
        store: FuturesTradeRecommendationStore,
        audits: FuturesRecommendationGenerationAuditStore,
    ) -> None:
        for value, expected, name in (
            (research, FuturesForwardResearchRecordRepository, "research"),
            (market, FuturesHistoricalMarketDataRepository, "market"),
            (sessions, FuturesTradingSessionResolver, "sessions"),
            (finality, FuturesDailyBarFinalityPolicy, "finality"),
            (cutoffs, FuturesContractCutoffSource, "cutoffs"),
            (recommendations, FuturesTradeRecommendationRepository, "recommendations"),
            (store, FuturesTradeRecommendationStore, "store"),
            (audits, FuturesRecommendationGenerationAuditStore, "audits"),
        ):
            if not isinstance(value, expected):
                raise TypeError(
                    f"GenerateFuturesTradeRecommendationsUseCase {name} must be a "
                    f"{expected.__name__}."
                )
        self._research = research
        self._market = market
        self._sessions = sessions
        self._finality = finality
        self._cutoffs = cutoffs
        self._recommendations = recommendations
        self._store = store
        self._audits = audits
        self._identities = FuturesTradeRecommendationIdentityService()
        self._windows = FuturesRecommendationSessionWindowResolver(sessions)

    def execute(
        self,
        *,
        contract: FuturesContract,
        strategy: StrategyIdentity,
        policy: FuturesRecommendationGenerationPolicy,
        profile: FuturesTradeRecommendationGenerationProfile,
        issued_at: PointInTime,
        reviewed_candles: FuturesReviewedCandleVerification | None,
    ) -> FuturesTradeRecommendationGenerationResult:
        """Generate, identify and record recommendations for ``contract``'s latest session."""
        self._validate_request(contract, strategy, policy, profile, issued_at, reviewed_candles)

        def result(
            status: FuturesTradeRecommendationGenerationStatus, **facts: object
        ) -> FuturesTradeRecommendationGenerationResult:
            return FuturesTradeRecommendationGenerationResult(
                contract=contract,
                strategy=strategy,
                profile=profile,
                issued_at=issued_at,
                status=status,
                **facts,  # type: ignore[arg-type]
            )

        def audited(
            outcome: FuturesTradeRecommendationGenerationResult,
        ) -> FuturesTradeRecommendationGenerationResult:
            self._audits.record_audit(FuturesRecommendationGenerationAudit.of(outcome))
            return outcome

        def blocked(
            reason: FuturesTradeRecommendationBlockReason, **facts: object
        ) -> FuturesTradeRecommendationGenerationResult:
            return audited(
                result(
                    FuturesTradeRecommendationGenerationStatus.BLOCKED, block_reason=reason, **facts
                )
            )

        bars = self._bars(contract)
        if not bars:
            return blocked(FuturesTradeRecommendationBlockReason.NO_PERSISTED_SESSION)
        latest = bars[-1]
        session = self._session_of(contract, latest)

        finality = self._finality.assess(contract, session)
        if not isinstance(finality, FuturesDailyBarFinality) or (
            finality.contract != contract or finality.session != session
        ):
            raise FuturesTradeRecommendationContractViolationError(
                "The finality policy must assess exactly the contract and session asked about."
            )
        if not finality.is_final:
            return blocked(
                FuturesTradeRecommendationBlockReason.SESSION_NOT_FINAL,
                session=session,
                finality=finality,
            )
        if reviewed_candles is None or not self._is_reviewed(
            reviewed_candles.approvals, contract, session, latest
        ):
            return blocked(
                FuturesTradeRecommendationBlockReason.SESSION_NOT_OPERATOR_REVIEWED,
                session=session,
                finality=finality,
            )
        if issued_at.compare(session.closes_at) < 0:
            raise InvalidFuturesTradeRecommendationGenerationError(
                f"Issuance at {issued_at} cannot precede the close of the research session "
                f"{session}; a recommendation is never backdated."
            )

        at_session = self._research_at(contract, session)
        if not any(record.strategy_identity == strategy for record in at_session):
            return blocked(
                FuturesTradeRecommendationBlockReason.NO_RESEARCH_FOR_SESSION,
                session=session,
                finality=finality,
            )
        for record in at_session:
            self._require_agrees_with_bar(record, latest)

        criteria = FuturesCandidateSelectionCriteria(
            session.closes_at, (contract.product,), (strategy,)
        )
        decisions = {_source(record): record for record in at_session}
        selection = FuturesCandidateSelection.select(
            criteria, tuple(self._decision(record) for record in at_session)
        )

        identities = {
            candidate.source: self._identities.identity(candidate.source, profile)
            for candidate in selection.candidates
        }
        existing = self._existing(identities, policy, profile)
        already_recorded = {recommendation.source for recommendation in existing}
        to_generate = FuturesCandidateSelection.select(
            criteria,
            tuple(
                self._decision(record)
                for source, record in decisions.items()
                if source not in already_recorded
            ),
        )

        generation: FuturesRecommendationGeneration | None = None
        recorded: tuple[FuturesTradeRecommendation, ...] = ()
        coverage: FuturesResearchHistoryCoverage | None = None
        if to_generate.candidates:
            # The gate for a NEW recommendation: its research history must be
            # VERIFIED before F-4 runs. Recorded candidates never reach here.
            coverage, deficiency, deficient = self._assess_history(
                contract,
                session,
                tuple(decisions[c.source] for c in to_generate.candidates),
                bars,
                reviewed_candles.approvals,
                policy.lookback_changes + 1,
            )
            if coverage is not FuturesResearchHistoryCoverage.VERIFIED:
                return blocked(
                    FuturesTradeRecommendationBlockReason.RESEARCH_HISTORY_NOT_VERIFIED,
                    session=session,
                    finality=finality,
                    history_coverage=coverage,
                    history_deficiency=deficiency,
                    history_deficient_sessions=deficient,
                    selection=selection,
                    existing=existing,
                )
            cutoff = self._cutoffs.resolve(contract)
            if cutoff is None:
                return blocked(
                    FuturesTradeRecommendationBlockReason.CONTRACT_CUTOFF_UNAVAILABLE,
                    session=session,
                    finality=finality,
                    history_coverage=coverage,
                    selection=selection,
                    existing=existing,
                )
            window = self._windows.resolve(contract, session.closes_at, issued_at, profile, cutoff)
            generation = FuturesRecommendationGeneration.generate(
                policy,
                to_generate,
                tuple(
                    self._evidence(candidate.source, decisions[candidate.source])
                    for candidate in to_generate.candidates
                ),
                (window,),
                self._open_lifecycles(contract),
                issued_at,
            )
            recorded = tuple(
                proposal.to_recommendation(identities[proposal.source])
                for proposal in generation.proposals
            )

        def completed(
            recorded: tuple[FuturesTradeRecommendation, ...],
            existing: tuple[FuturesTradeRecommendation, ...],
        ) -> FuturesTradeRecommendationGenerationResult:
            return result(
                FuturesTradeRecommendationGenerationStatus.COMPLETED,
                session=session,
                finality=finality,
                history_coverage=coverage,
                selection=selection,
                generation=generation,
                recorded=recorded,
                existing=existing,
            )

        if recorded:
            # The audit of exactly this result commits with its definitions, or
            # neither does; the result is returned only after that commit.
            committed = completed(recorded, existing)
            winners = self._record(
                recorded,
                identities,
                policy,
                profile,
                FuturesRecommendationGenerationAudit.of(committed),
            )
            if not winners:
                return committed
            return audited(completed((), existing + winners))
        return audited(completed((), existing))

    # -- request ------------------------------------------------------------------------

    @staticmethod
    def _validate_request(
        contract: object,
        strategy: object,
        policy: object,
        profile: object,
        issued_at: object,
        reviewed_candles: object,
    ) -> None:
        if not isinstance(contract, FuturesContract):
            raise TypeError("Recommendation generation contract must be a FuturesContract.")
        if not isinstance(strategy, StrategyIdentity):
            raise TypeError("Recommendation generation strategy must be a StrategyIdentity.")
        if not isinstance(policy, FuturesRecommendationGenerationPolicy):
            raise TypeError(
                "Recommendation generation policy must be a FuturesRecommendationGenerationPolicy."
            )
        if not isinstance(profile, FuturesTradeRecommendationGenerationProfile):
            raise TypeError(
                "Recommendation generation profile must be a "
                "FuturesTradeRecommendationGenerationProfile."
            )
        if not isinstance(issued_at, PointInTime):
            raise TypeError("Recommendation generation issued at must be a PointInTime.")
        if reviewed_candles is not None and not isinstance(
            reviewed_candles, FuturesReviewedCandleVerification
        ):
            raise TypeError(
                "Recommendation generation reviewed candles must be a "
                "FuturesReviewedCandleVerification or None."
            )
        if profile.policy_version != policy.version:
            raise InvalidFuturesTradeRecommendationGenerationError(
                f"Profile {profile.version} applies to policy {profile.policy_version}, "
                f"not {policy.version}."
            )
        if policy.strategy != strategy:
            raise InvalidFuturesTradeRecommendationGenerationError(
                f"Policy {policy.version} approves strategy {policy.strategy}, not {strategy}."
            )
        if reviewed_candles is not None and reviewed_candles.approvals.product != contract.product:
            raise InvalidFuturesTradeRecommendationGenerationError(
                f"The candle approvals are for {reviewed_candles.approvals.product}, "
                f"not {contract.product}."
            )

    # -- market evidence and the session --------------------------------------------------

    def _bars(self, contract: FuturesContract) -> tuple[FuturesOHLCVBar, ...]:
        bars = self._market.get_bars(FuturesHistoricalMarketDataQuery(contract, _DAILY))
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
        return bars

    def _session_of(self, contract: FuturesContract, bar: FuturesOHLCVBar) -> FuturesTradingSession:
        day = _utc(bar.point_in_time).date()
        matching = [
            session
            for session in self._sessions.sessions_in_range(
                contract.product, day - _BAR_SESSION_MARGIN, day + _BAR_SESSION_MARGIN
            )
            if session.closes_at.compare(bar.point_in_time) == 0
        ]
        if len(matching) != 1:
            raise FuturesTradeRecommendationContractViolationError(
                f"The latest persisted bar of {contract} at {bar.point_in_time} closes "
                f"{'no' if not matching else len(matching)} resolved trading session(s)."
            )
        return matching[0]

    @staticmethod
    def _is_reviewed(
        approvals: FuturesCandleApprovalSet,
        contract: FuturesContract,
        session: FuturesTradingSession,
        bar: FuturesOHLCVBar,
    ) -> bool:
        """Return whether the stored bar is the approved candle; a mismatch raises."""
        try:
            approvals.verify_bar(contract, session.trading_date, bar)
        except FuturesCandleApprovalMissingError:
            return False
        return True

    def _assess_history(
        self,
        contract: FuturesContract,
        session: FuturesTradingSession,
        records: tuple[FuturesForwardResearchRecord, ...],
        bars: tuple[FuturesOHLCVBar, ...],
        approvals: FuturesCandleApprovalSet,
        required: int,
    ) -> tuple[
        FuturesResearchHistoryCoverage,
        FuturesResearchHistoryDeficiency | None,
        tuple[date, ...],
    ]:
        """Assess the trailing ``required`` closes of each record against the evidence.

        The expected history is the calendar's ``required`` consecutive sessions
        ending at the decision session. Checked in this order, so that missing
        evidence is never mistaken for contradictory evidence:

        1. a stored bar inside the span that is on no expected session raises;
        2. a session without its stored bar blocks (MISSING_BAR);
        3. a stored close that differs from the frozen close in its position
           raises -- the history is complete, so the two genuinely disagree;
        4. an unapproved candle blocks (UNAPPROVED_CANDLE); a candle that
           differs from its approved version raises.
        """
        expected = self._expected_sessions(contract, session, required)
        if len(expected) < required:
            return (
                FuturesResearchHistoryCoverage.UNVERIFIED,
                FuturesResearchHistoryDeficiency.INCOMPLETE_CALENDAR,
                (),
            )
        first, last = expected[0].closes_at, expected[-1].closes_at
        on_session = {expected_session.closes_at for expected_session in expected}
        for bar in bars:
            inside = bar.point_in_time.compare(first) >= 0 and bar.point_in_time.compare(last) <= 0
            if inside and bar.point_in_time not in on_session:
                raise FuturesTradeRecommendationSourceError(
                    f"The stored bar of {contract} at {bar.point_in_time} is on no expected "
                    "trading session of the research history."
                )
        by_instant = {bar.point_in_time: bar for bar in bars}
        missing = tuple(
            expected_session.trading_date
            for expected_session in expected
            if expected_session.closes_at not in by_instant
        )
        if missing:
            return (
                FuturesResearchHistoryCoverage.UNVERIFIED,
                FuturesResearchHistoryDeficiency.MISSING_BAR,
                missing,
            )
        history = tuple(by_instant[expected_session.closes_at] for expected_session in expected)
        for record in records:
            closes = record.result.market_observation_context.recent_closes[-required:]
            if len(closes) != required or tuple(bar.close for bar in history) != closes:
                raise FuturesTradeRecommendationSourceError(
                    f"Frozen research {record.natural_key} decided on closes that differ from "
                    "the stored bars of its consecutive research sessions."
                )
        unapproved = tuple(
            expected_session.trading_date
            for expected_session, bar in zip(expected, history, strict=True)
            if not self._is_reviewed(approvals, contract, expected_session, bar)
        )
        if unapproved:
            return (
                FuturesResearchHistoryCoverage.UNVERIFIED,
                FuturesResearchHistoryDeficiency.UNAPPROVED_CANDLE,
                unapproved,
            )
        return FuturesResearchHistoryCoverage.VERIFIED, None, ()

    def _expected_sessions(
        self, contract: FuturesContract, session: FuturesTradingSession, required: int
    ) -> tuple[FuturesTradingSession, ...]:
        """Return up to ``required`` consecutive resolved sessions ending at ``session``.

        Searched backwards in non-overlapping chunks within a bounded span; never
        computed from calendar days. Resolver errors -- an unloaded year -- propagate.
        """
        limit = session.trading_date - timedelta(days=7 * required + 62)
        found: list[FuturesTradingSession] = []
        chunk_end = session.trading_date
        while len(found) < required and chunk_end >= limit:
            chunk_start = max(chunk_end - _HISTORY_CHUNK, limit)
            resolved = tuple(
                candidate
                for candidate in self._sessions.sessions_in_range(
                    contract.product, chunk_start, chunk_end
                )
                if candidate.closes_at.compare(session.closes_at) <= 0
            )
            found = list(resolved) + found
            chunk_end = chunk_start - timedelta(days=1)
        if not found or found[-1] != session:
            raise FuturesTradeRecommendationContractViolationError(
                f"The trading calendar does not resolve the decision session {session}."
            )
        return tuple(found[-required:])

    # -- research -------------------------------------------------------------------------

    def _research_at(
        self, contract: FuturesContract, session: FuturesTradingSession
    ) -> tuple[FuturesForwardResearchRecord, ...]:
        records = self._research.get_records(FuturesForwardResearchRecordQuery(contract, _DAILY))
        if not isinstance(records, tuple) or not all(
            isinstance(record, FuturesForwardResearchRecord)
            and record.contract == contract
            and record.timeframe == _DAILY
            for record in records
        ):
            raise FuturesTradeRecommendationContractViolationError(
                f"The research repository must return a tuple of {contract} daily records."
            )
        keys = [record.natural_key for record in records]
        if len(set(keys)) != len(keys):
            raise FuturesTradeRecommendationContractViolationError(
                "The research repository returned one natural key more than once."
            )
        return tuple(
            record for record in records if record.decision_instant.compare(session.closes_at) == 0
        )

    @staticmethod
    def _require_agrees_with_bar(
        record: FuturesForwardResearchRecord, bar: FuturesOHLCVBar
    ) -> None:
        if record.result.market_observation_context.latest_quote != bar.close:
            raise FuturesTradeRecommendationSourceError(
                f"Frozen research {record.natural_key} decided on a close that differs from "
                "the reviewed candle of its session."
            )

    @staticmethod
    def _decision(record: FuturesForwardResearchRecord) -> FuturesResearchDecision:
        try:
            return FuturesResearchDecision(_source(record), record.result.recommendation)
        except InvalidFuturesResearchDecisionError as exc:
            raise FuturesTradeRecommendationSourceError(
                f"Frozen research {record.natural_key} is not a valid research decision."
            ) from exc

    @staticmethod
    def _evidence(
        source: FuturesResearchDecisionReference, record: FuturesForwardResearchRecord
    ) -> FuturesCandidateEvidence:
        try:
            return FuturesCandidateEvidence(source, record.result.market_observation_context)
        except InvalidFuturesCandidateEvidenceError as exc:
            raise FuturesTradeRecommendationSourceError(
                f"Frozen research {record.natural_key} is not valid recommendation evidence."
            ) from exc

    # -- recorded recommendations ---------------------------------------------------------

    def _record(
        self,
        recommendations: tuple[FuturesTradeRecommendation, ...],
        identities: dict[FuturesResearchDecisionReference, FuturesTradeRecommendationIdentity],
        policy: FuturesRecommendationGenerationPolicy,
        profile: FuturesTradeRecommendationGenerationProfile,
        audit: FuturesRecommendationGenerationAudit,
    ) -> tuple[FuturesTradeRecommendation, ...]:
        """Record the batch with its audit; on a same-identity race, return the winners.

        Returns () when this batch and its audit were recorded. When the store reports that an
        identity is already recorded differently, every identity of the batch
        is re-read and must be the expected recommendation; the stored winners
        are returned in the batch's order and nothing of this batch is kept.
        """
        try:
            accepted = self._store.record(
                recommendations, policy=policy, profile=profile, audit=audit
            )
        except FuturesTradeRecommendationConflictError as conflict:
            winners: list[FuturesTradeRecommendation] = []
            for recommendation in recommendations:
                stored = self._recommendations.get_recommendation(recommendation.identity)
                if stored is None:
                    raise FuturesTradeRecommendationConflictError(
                        f"Recording {recommendation.identity} conflicted, but no recommendation "
                        "is recorded under it."
                    ) from conflict
                winners.append(
                    self._verify_recorded(
                        stored,
                        identities[recommendation.source],
                        recommendation.source,
                        policy,
                        profile,
                    )
                )
            return tuple(winners)
        if (
            not isinstance(accepted, int)
            or isinstance(accepted, bool)
            or (accepted != len(recommendations))
        ):
            raise FuturesTradeRecommendationContractViolationError(
                f"The recommendation store accepted {accepted!r} of {len(recommendations)} "
                "recommendations; recording must be all-or-nothing."
            )
        return ()

    def _verify_recorded(
        self,
        stored: object,
        identity: FuturesTradeRecommendationIdentity,
        source: FuturesResearchDecisionReference,
        policy: FuturesRecommendationGenerationPolicy,
        profile: FuturesTradeRecommendationGenerationProfile,
    ) -> FuturesTradeRecommendation:
        """Return ``stored`` only if it is the recommendation ``identity`` must name.

        Issuance time, levels, validity and thesis are first-write-wins facts
        and are not compared; everything that determines the identity is.
        """
        if not isinstance(stored, FuturesTradeRecommendation) or stored.identity != identity:
            raise FuturesTradeRecommendationContractViolationError(
                f"The recommendation repository answered {identity} with another value."
            )
        if (
            stored.contract != source.contract
            or stored.source != source
            or stored.policy_version != policy.version
            or self._identities.identity(stored.source, profile) != identity
        ):
            raise FuturesTradeRecommendationConflictError(
                f"Recommendation {identity} is recorded for {stored.source} under "
                f"{stored.policy_version}, not {source} under {policy.version} and profile "
                f"{profile.version}."
            )
        return stored

    def _existing(
        self,
        identities: dict[FuturesResearchDecisionReference, FuturesTradeRecommendationIdentity],
        policy: FuturesRecommendationGenerationPolicy,
        profile: FuturesTradeRecommendationGenerationProfile,
    ) -> tuple[FuturesTradeRecommendation, ...]:
        found: list[FuturesTradeRecommendation] = []
        for source, identity in identities.items():
            recorded = self._recommendations.get_recommendation(identity)
            if recorded is None:
                continue
            found.append(self._verify_recorded(recorded, identity, source, policy, profile))
        return tuple(found)

    def _open_lifecycles(
        self, contract: FuturesContract
    ) -> tuple[FuturesTradeRecommendationLifecycle, ...]:
        lifecycles = self._recommendations.get_lifecycles(
            FuturesTradeRecommendationQuery(contract.product)
        )
        if not isinstance(lifecycles, tuple) or not all(
            isinstance(lifecycle, FuturesTradeRecommendationLifecycle)
            and lifecycle.recommendation.contract.product == contract.product
            for lifecycle in lifecycles
        ):
            raise FuturesTradeRecommendationContractViolationError(
                f"The recommendation repository must return {contract.product} lifecycles."
            )
        return lifecycles
