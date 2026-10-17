"""The immutable audit of one futures trade recommendation generation attempt.

A FuturesRecommendationGenerationAudit records what one execution of
GenerateFuturesTradeRecommendationsUseCase established and decided, using only
facts that execution already holds. It stores no price, candle or provider
payload: research decisions are named by their F-2 source reference, and
recommendations by their identity.

Four outcomes, never interchangeable
------------------------------------
``outcome`` is derived, not supplied:

- RECORDED: this attempt's recommendations were newly recorded. Such an audit
  is committed only by the recommendation store, in the transaction that
  inserts exactly ``recorded`` (I-2); it can never be recorded on its own.
- EXISTING_RETURNED: nothing new was recorded, and recommendations already
  recorded under the candidates' identities were returned -- a retry, or an
  attempt that lost a first-write race. ``proposed`` still names what this
  attempt generated, so a lost race shows a proposal that was not recorded.
- REJECTED: the attempt ran to completion and nothing was recorded or returned:
  F-3 rejected every decision (HOLD included) or F-4 rejected every candidate,
  exclusivity breaches included.
- BLOCKED: an expected operational condition stopped the attempt before
  generation; ``block_reason`` names it.

Unexpected failures -- contradictory research, a port outside its contract,
storage or configuration conflicts -- raise in the use case and produce no
audit; they are never disguised as a rejection.

Identity
--------
Length-prefixed SHA-256, as for recommendation identities, under its own
namespace ``northstar.futures-recommendation-generation-audit.v1``, over, in
this order:

    product code, exchange code, expiration date (YYYY-MM-DD),
    strategy identity, F-4 policy version, Application profile version,
    attempted at                (canonical UTC PointInTime text),
    session label               (YYYY-MM-DD, or empty when none was found),
    outcome, block reason       (empty when not blocked)

``attempted_at`` is the issuance instant the caller captured once for the
attempt; nothing here reads a clock or draws a random value. The same attempt
replayed reaches the same identity, so recording its audit again is a no-op.
Distinct attempts differ in that instant; a recording and a later retry that
returns it differ in outcome; different sessions, configurations and block
reasons differ in their own parts. Equal identity with different content is a
contradiction the stores refuse.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from northstar_core.foundation.value_objects import PointInTime
from northstar_core.futures import FuturesContract, FuturesProductReference
from northstar_core.recommendations.futures import (
    CandidateRejectionReason,
    FuturesResearchDecisionReference,
    FuturesTradeRecommendationIdentity,
    GenerationRejectionReason,
    RecommendationPolicyVersion,
)
from northstar_core.strategy import RecommendationAction, StrategyIdentity

from northstar_application.application_services.futures_recommendation_generation_outcome import (
    FuturesResearchHistoryCoverage,
    FuturesResearchHistoryDeficiency,
    FuturesTradeRecommendationBlockReason,
    FuturesTradeRecommendationGenerationStatus,
)
from northstar_application.application_services.futures_recommendation_identities import (
    FuturesTradeRecommendationIdentityService,
)
from northstar_application.ports.futures_daily_bar_finality_policy import (
    FuturesDailyBarFinality,
)
from northstar_application.ports.futures_trading_session_resolver import FuturesTradingSession

if TYPE_CHECKING:
    from northstar_application.application_services.generate_futures_trade_recommendations import (
        FuturesTradeRecommendationGenerationResult,
    )

FUTURES_RECOMMENDATION_GENERATION_AUDIT_VERSION = 1
FUTURES_RECOMMENDATION_GENERATION_AUDIT_IDENTITY_NAMESPACE = (
    "northstar.futures-recommendation-generation-audit.v1"
)

_Status = FuturesTradeRecommendationGenerationStatus
_Reason = FuturesTradeRecommendationBlockReason
_Coverage = FuturesResearchHistoryCoverage


class InvalidFuturesRecommendationGenerationAuditError(ValueError):
    """Raised when a generation audit contradicts itself or its attempt."""


class FuturesRecommendationGenerationAuditOutcome(StrEnum):
    """What one generation attempt produced; see the module documentation."""

    RECORDED = "RECORDED"
    EXISTING_RETURNED = "EXISTING_RETURNED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"


class FuturesCandleReviewAssessment(StrEnum):
    """Whether the session's stored candle was found to be the operator-reviewed one."""

    NOT_ASSESSED = "NOT_ASSESSED"
    NOT_REVIEWED = "NOT_REVIEWED"
    REVIEWED = "REVIEWED"


def _error(message: str) -> InvalidFuturesRecommendationGenerationAuditError:
    return InvalidFuturesRecommendationGenerationAuditError(
        f"FuturesRecommendationGenerationAudit {message}"
    )


def _require(value: object, expected: type | tuple[type, ...], name: str) -> None:
    if not isinstance(value, expected):
        raise _error(f"{name} has the wrong type.")


def _require_tuple_of(value: object, expected: type, name: str) -> None:
    if not isinstance(value, tuple) or not all(isinstance(item, expected) for item in value):
        raise _error(f"{name} must be a tuple of {expected.__name__} values.")
    if len(set(value)) != len(value):
        raise _error(f"{name} names one value more than once.")


@dataclass(frozen=True, slots=True)
class FuturesCandidateRejectionAudit:
    """One research decision F-3 did not select, with its action and reason."""

    source: FuturesResearchDecisionReference
    research_action: RecommendationAction
    reason: CandidateRejectionReason

    def __post_init__(self) -> None:
        _require(self.source, FuturesResearchDecisionReference, "candidate rejection source")
        _require(self.research_action, RecommendationAction, "candidate rejection action")
        _require(self.reason, CandidateRejectionReason, "candidate rejection reason")


@dataclass(frozen=True, slots=True)
class FuturesGenerationRejectionAudit:
    """One candidate F-4 produced no proposal for, with its reason."""

    source: FuturesResearchDecisionReference
    reason: GenerationRejectionReason

    def __post_init__(self) -> None:
        _require(self.source, FuturesResearchDecisionReference, "generation rejection source")
        _require(self.reason, GenerationRejectionReason, "generation rejection reason")


def _canonical(parts: tuple[str, ...]) -> bytes:
    encoded = bytearray()
    for part in parts:
        raw = part.encode("utf-8")
        encoded += str(len(raw)).encode("ascii") + b":" + raw
    return bytes(encoded)


@dataclass(frozen=True, slots=True)
class FuturesRecommendationGenerationAudit:
    """Immutable audit of one generation attempt for one contract and strategy."""

    contract: FuturesContract
    strategy: StrategyIdentity
    policy_version: RecommendationPolicyVersion
    profile_version: str
    attempted_at: PointInTime
    status: FuturesTradeRecommendationGenerationStatus
    block_reason: FuturesTradeRecommendationBlockReason | None = None
    session: FuturesTradingSession | None = None
    finality: FuturesDailyBarFinality | None = None
    history_coverage: FuturesResearchHistoryCoverage | None = None
    history_deficiency: FuturesResearchHistoryDeficiency | None = None
    history_deficient_sessions: tuple[date, ...] = ()
    candidates: tuple[FuturesResearchDecisionReference, ...] = ()
    candidate_rejections: tuple[FuturesCandidateRejectionAudit, ...] = ()
    generation_rejections: tuple[FuturesGenerationRejectionAudit, ...] = ()
    exclusivity_breaches: tuple[FuturesProductReference, ...] = ()
    proposed: tuple[FuturesTradeRecommendationIdentity, ...] = ()
    recorded: tuple[FuturesTradeRecommendationIdentity, ...] = ()
    existing: tuple[FuturesTradeRecommendationIdentity, ...] = ()
    version: int = FUTURES_RECOMMENDATION_GENERATION_AUDIT_VERSION

    def __post_init__(self) -> None:
        self._validate_types()
        self._validate_session()
        self._validate_history()
        self._validate_outcome()
        self._validate_references()

    # -- validation -----------------------------------------------------------------------

    def _validate_types(self) -> None:
        if self.version != FUTURES_RECOMMENDATION_GENERATION_AUDIT_VERSION or isinstance(
            self.version, bool
        ):
            raise _error(f"version {self.version!r} is not supported.")
        _require(self.contract, FuturesContract, "contract")
        _require(self.strategy, StrategyIdentity, "strategy")
        _require(self.policy_version, RecommendationPolicyVersion, "policy version")
        if not isinstance(self.profile_version, str) or not self.profile_version.strip():
            raise _error("profile version must be non-empty text.")
        if self.profile_version != self.profile_version.strip():
            raise _error("profile version must not have surrounding whitespace.")
        _require(self.attempted_at, PointInTime, "attempted at")
        _require(self.status, FuturesTradeRecommendationGenerationStatus, "status")
        if self.block_reason is not None:
            _require(self.block_reason, FuturesTradeRecommendationBlockReason, "block reason")
        if self.session is not None:
            _require(self.session, FuturesTradingSession, "session")
        if self.finality is not None:
            _require(self.finality, FuturesDailyBarFinality, "finality")
        if self.history_coverage is not None:
            _require(self.history_coverage, FuturesResearchHistoryCoverage, "history coverage")
        if self.history_deficiency is not None:
            _require(
                self.history_deficiency, FuturesResearchHistoryDeficiency, "history deficiency"
            )
        _require_tuple_of(self.history_deficient_sessions, date, "history deficient sessions")
        if any(isinstance(day, datetime) for day in self.history_deficient_sessions):
            raise _error("history deficient sessions must be plain dates.")
        _require_tuple_of(self.candidates, FuturesResearchDecisionReference, "candidates")
        _require_tuple_of(
            self.candidate_rejections, FuturesCandidateRejectionAudit, "candidate rejections"
        )
        _require_tuple_of(
            self.generation_rejections, FuturesGenerationRejectionAudit, "generation rejections"
        )
        _require_tuple_of(self.exclusivity_breaches, FuturesProductReference, "breaches")
        for name in ("proposed", "recorded", "existing"):
            _require_tuple_of(getattr(self, name), FuturesTradeRecommendationIdentity, name)

    def _validate_session(self) -> None:
        blocked = self.status is _Status.BLOCKED
        if blocked != (self.block_reason is not None):
            raise _error("names a block reason exactly when it is BLOCKED.")
        no_session = self.block_reason is _Reason.NO_PERSISTED_SESSION
        if no_session != (self.session is None):
            raise _error("names a session exactly when one was persisted.")
        if (self.session is None) != (self.finality is None):
            raise _error("names a finality assessment exactly when it names a session.")
        if self.finality is not None:
            if self.finality.contract != self.contract or self.finality.session != self.session:
                raise _error("finality must assess the audited contract and session.")
            not_final = self.block_reason is _Reason.SESSION_NOT_FINAL
            if self.finality.is_final == not_final:
                raise _error("finality must be FINAL exactly when the session was not refused.")

    def _validate_history(self) -> None:
        unverified = self.history_coverage is _Coverage.UNVERIFIED
        if unverified != (self.history_deficiency is not None):
            raise _error("names a history deficiency exactly when history is UNVERIFIED.")
        if self.history_deficient_sessions and not unverified:
            raise _error("names deficient sessions only for UNVERIFIED history.")
        days = self.history_deficient_sessions
        if any(earlier >= later for earlier, later in zip(days, days[1:], strict=False)):
            raise _error("history deficient sessions must be strictly ascending.")
        if unverified != (self.block_reason is _Reason.RESEARCH_HISTORY_NOT_VERIFIED):
            raise _error("is blocked on research history exactly when it is UNVERIFIED.")

    def _validate_outcome(self) -> None:
        f4_output = self.proposed or self.generation_rejections or self.exclusivity_breaches
        if self.status is _Status.BLOCKED and (f4_output or self.recorded):
            raise _error("BLOCKED cannot carry F-4 output or recorded recommendations.")
        if self.recorded and self.history_coverage is not _Coverage.VERIFIED:
            raise _error("can record recommendations only from VERIFIED research history.")
        if not set(self.recorded) <= set(self.proposed):
            raise _error("can record only recommendations this attempt proposed.")
        if set(self.recorded) & set(self.existing):
            raise _error("cannot name one recommendation as both recorded and existing.")

    def _validate_references(self) -> None:
        sources = (
            *self.candidates,
            *(rejection.source for rejection in self.candidate_rejections),
            *(rejection.source for rejection in self.generation_rejections),
        )
        if any(source.contract != self.contract for source in sources):
            raise _error("references research of another contract.")
        if any(product != self.contract.product for product in self.exclusivity_breaches):
            raise _error("names an exclusivity breach on another product.")

    # -- derived facts --------------------------------------------------------------------

    @property
    def outcome(self) -> FuturesRecommendationGenerationAuditOutcome:
        """Return what the attempt produced, derived from its status and identities."""
        if self.status is _Status.BLOCKED:
            return FuturesRecommendationGenerationAuditOutcome.BLOCKED
        if self.recorded:
            return FuturesRecommendationGenerationAuditOutcome.RECORDED
        if self.existing:
            return FuturesRecommendationGenerationAuditOutcome.EXISTING_RETURNED
        return FuturesRecommendationGenerationAuditOutcome.REJECTED

    @property
    def operator_review(self) -> FuturesCandleReviewAssessment:
        """Return whether the session's candle was assessed and found operator-reviewed."""
        if self.block_reason in (_Reason.NO_PERSISTED_SESSION, _Reason.SESSION_NOT_FINAL):
            return FuturesCandleReviewAssessment.NOT_ASSESSED
        if self.block_reason is _Reason.SESSION_NOT_OPERATOR_REVIEWED:
            return FuturesCandleReviewAssessment.NOT_REVIEWED
        return FuturesCandleReviewAssessment.REVIEWED

    @property
    def identity(self) -> str:
        """Return the deterministic audit identity; see the module documentation."""
        product = self.contract.product
        key = (
            FUTURES_RECOMMENDATION_GENERATION_AUDIT_IDENTITY_NAMESPACE,
            product.product_code.value,
            product.exchange_code.value,
            self.contract.expiration_date.value,
            self.strategy.identity,
            self.policy_version.version,
            self.profile_version,
            self.attempted_at.value,
            "" if self.session is None else self.session.trading_date.isoformat(),
            self.outcome.value,
            "" if self.block_reason is None else self.block_reason.value,
        )
        return hashlib.sha256(_canonical(key)).hexdigest()

    # -- construction ---------------------------------------------------------------------

    @classmethod
    def of(
        cls, result: FuturesTradeRecommendationGenerationResult
    ) -> FuturesRecommendationGenerationAudit:
        """Return the audit of exactly what ``result`` reports, inventing nothing."""
        from northstar_application.application_services.generate_futures_trade_recommendations import (  # noqa: E501
            FuturesTradeRecommendationGenerationResult,
        )

        if not isinstance(result, FuturesTradeRecommendationGenerationResult):
            raise TypeError("A generation audit is built from a generation result.")
        identities = FuturesTradeRecommendationIdentityService()
        selection, generation = result.selection, result.generation
        return cls(
            contract=result.contract,
            strategy=result.strategy,
            policy_version=result.profile.policy_version,
            profile_version=result.profile.version,
            attempted_at=result.issued_at,
            status=result.status,
            block_reason=result.block_reason,
            session=result.session,
            finality=result.finality,
            history_coverage=result.history_coverage,
            history_deficiency=result.history_deficiency,
            history_deficient_sessions=result.history_deficient_sessions,
            candidates=(() if selection is None else tuple(c.source for c in selection.candidates)),
            candidate_rejections=(
                ()
                if selection is None
                else tuple(
                    FuturesCandidateRejectionAudit(
                        rejection.decision.reference,
                        rejection.decision.research_action,
                        rejection.reason,
                    )
                    for rejection in selection.rejections
                )
            ),
            generation_rejections=(
                ()
                if generation is None
                else tuple(
                    FuturesGenerationRejectionAudit(rejection.source, rejection.reason)
                    for rejection in generation.rejections
                )
            ),
            exclusivity_breaches=(() if generation is None else generation.exclusivity_breaches),
            proposed=(
                ()
                if generation is None
                else tuple(
                    identities.identity(proposal.source, result.profile)
                    for proposal in generation.proposals
                )
            ),
            recorded=tuple(recommendation.identity for recommendation in result.recorded),
            existing=tuple(recommendation.identity for recommendation in result.existing),
        )
