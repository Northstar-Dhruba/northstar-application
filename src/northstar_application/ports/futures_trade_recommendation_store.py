"""Application port for internally recording futures trade recommendations.

Recording is internal: a recorded recommendation is not published, delivered,
notified or executed, and this port models none of those states.

Every recording names the effective generation configuration it was produced
under -- the F-4 policy and the Application profile -- because an F-2
recommendation carries its policy version but not its profile version, and the
profile version is part of its identity. A store cannot re-derive an identity,
or register the configuration, from the recommendation alone.

Every new recommendation is recorded with its generation audit (I-2)
--------------------------------------------------------------------
``record`` takes the RECORDED FuturesRecommendationGenerationAudit of the
attempt that produced the batch, and a durable store commits the definitions
and that audit in ONE transaction: no audit without its definitions, no
definition without its audit. The audit must name exactly the batch as
``recorded`` -- same identities, same order -- and the batch must come from the
audited attempt: its contract, strategy, configuration and issuance instant.
So a caller cannot claim as newly recorded an identity the transaction did not
insert, and the store never inserts a recommendation no audit accounts for.

A batch whose identities are already recorded is an idempotent success only
when it is the same committed recording replayed: every definition equal and
the same audit already stored with equal content. Otherwise an identity
already recorded -- even with an equal definition, by another attempt -- is a
FuturesTradeRecommendationConflictError, and the caller reconciles it as an
existing recommendation. That keeps a lost race from storing a RECORDED audit.

Four distinct conflicts
-----------------------
- FuturesTradeRecommendationConflictError: an identity of the batch is already
  recorded, by another recording. Under one source and one configuration this
  is the expected outcome of two attempts racing -- they differ in issuance
  time, so in levels and validity too -- and the first recording wins. A caller
  may reconcile it by re-reading the identity.
- FuturesTradeRecommendationConfigurationConflictError: a policy version or a
  profile version is already registered with different contents. That is a
  version reused for a different configuration, never a retry, and it is
  deliberately not a FuturesTradeRecommendationConflictError so that no
  reconciliation path can catch it.
- FuturesTradeRecommendationExclusivityError: another open recommendation is
  already on the product, on any expiry (ADR-018, P-1).
- FuturesRecommendationGenerationAuditConflictError (audit store port): the
  audit identity is already stored with different content.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from northstar_core.recommendations.futures import (
    FuturesRecommendationGenerationPolicy,
    FuturesTradeRecommendation,
)

if TYPE_CHECKING:
    from northstar_application.application_services.futures_recommendation_generation_audit import (  # noqa: E501
        FuturesRecommendationGenerationAudit,
    )
    from northstar_application.application_services.futures_recommendation_profile import (
        FuturesTradeRecommendationGenerationProfile,
    )


class FuturesTradeRecommendationConflictError(ValueError):
    """Raised when an identity of a batch is already recorded by another recording."""


class FuturesTradeRecommendationConfigurationConflictError(ValueError):
    """Raised when a policy or profile version is already registered differently."""


class FuturesTradeRecommendationExclusivityError(ValueError):
    """Raised when recording would leave two open recommendations on one product."""


class InvalidFuturesTradeRecommendationRecordingError(ValueError):
    """Raised when a recording request contradicts its own configuration."""


def validate_futures_trade_recommendation_recording(
    recommendations: tuple[FuturesTradeRecommendation, ...],
    *,
    policy: FuturesRecommendationGenerationPolicy,
    profile: FuturesTradeRecommendationGenerationProfile,
    audit: FuturesRecommendationGenerationAudit,
) -> None:
    """Refuse a recording request whose parts do not describe one audited recording.

    Every store implementation calls this before touching storage, so all of
    them refuse the same requests:

    - ``recommendations`` is a tuple of FuturesTradeRecommendation values;
    - ``policy`` is a FuturesRecommendationGenerationPolicy and ``profile`` a
      FuturesTradeRecommendationGenerationProfile naming that policy's version;
    - every recommendation carries the policy's version and comes from the
      policy's approved strategy;
    - every identity is exactly the one derived from the recommendation's
      source under the profile;
    - ``audit`` is a RECORDED FuturesRecommendationGenerationAudit naming the
      batch's identities, in order, as ``recorded``, under the same policy and
      profile versions; every recommendation is of its contract and strategy
      and was issued at its attempt instant. An empty batch is refused: there
      is nothing an audit could truthfully report as recorded.

    It cannot check that a version was never registered with other contents;
    that needs the store's durable registry.
    """
    from northstar_application.application_services.futures_recommendation_generation_audit import (  # noqa: E501
        FuturesRecommendationGenerationAudit,
        FuturesRecommendationGenerationAuditOutcome,
    )
    from northstar_application.application_services.futures_recommendation_identities import (
        FuturesTradeRecommendationIdentityService,
    )
    from northstar_application.application_services.futures_recommendation_profile import (
        FuturesTradeRecommendationGenerationProfile,
    )

    if not isinstance(recommendations, tuple) or not all(
        isinstance(recommendation, FuturesTradeRecommendation) for recommendation in recommendations
    ):
        raise InvalidFuturesTradeRecommendationRecordingError(
            "Recommendations to record must be a tuple of FuturesTradeRecommendation values."
        )
    if not isinstance(policy, FuturesRecommendationGenerationPolicy):
        raise InvalidFuturesTradeRecommendationRecordingError(
            "A recording's policy must be a FuturesRecommendationGenerationPolicy."
        )
    if not isinstance(profile, FuturesTradeRecommendationGenerationProfile):
        raise InvalidFuturesTradeRecommendationRecordingError(
            "A recording's profile must be a FuturesTradeRecommendationGenerationProfile."
        )
    if profile.policy_version != policy.version:
        raise InvalidFuturesTradeRecommendationRecordingError(
            f"Profile {profile.version} applies to policy {profile.policy_version}, "
            f"not {policy.version}."
        )
    identities = FuturesTradeRecommendationIdentityService()
    for recommendation in recommendations:
        if recommendation.policy_version != policy.version:
            raise InvalidFuturesTradeRecommendationRecordingError(
                f"Recommendation {recommendation.identity} was generated under "
                f"{recommendation.policy_version}, not {policy.version}."
            )
        if recommendation.strategy_identity not in policy.approved_strategies:
            raise InvalidFuturesTradeRecommendationRecordingError(
                f"Recommendation {recommendation.identity} comes from strategy "
                f"{recommendation.strategy_identity}, which policy {policy.version} does not "
                "approve."
            )
        if identities.identity(recommendation.source, profile) != recommendation.identity:
            raise InvalidFuturesTradeRecommendationRecordingError(
                f"Recommendation {recommendation.identity} is not the identity derived from "
                f"its source under profile {profile.version}."
            )

    if not isinstance(audit, FuturesRecommendationGenerationAudit):
        raise InvalidFuturesTradeRecommendationRecordingError(
            "A recording's audit must be a FuturesRecommendationGenerationAudit."
        )
    if audit.outcome is not FuturesRecommendationGenerationAuditOutcome.RECORDED:
        raise InvalidFuturesTradeRecommendationRecordingError(
            f"A recording's audit must report RECORDED, not {audit.outcome}."
        )
    if audit.recorded != tuple(recommendation.identity for recommendation in recommendations):
        raise InvalidFuturesTradeRecommendationRecordingError(
            "A recording's audit must name exactly the batch, in order, as recorded."
        )
    if audit.policy_version != policy.version or audit.profile_version != profile.version:
        raise InvalidFuturesTradeRecommendationRecordingError(
            f"The audit is of policy {audit.policy_version} and profile "
            f"{audit.profile_version}, not {policy.version} and {profile.version}."
        )
    for recommendation in recommendations:
        if (
            recommendation.contract != audit.contract
            or recommendation.source.strategy_identity != audit.strategy
            or recommendation.issued_at != audit.attempted_at
        ):
            raise InvalidFuturesTradeRecommendationRecordingError(
                f"Recommendation {recommendation.identity} is not of the audited attempt on "
                f"{audit.contract} by {audit.strategy} at {audit.attempted_at}."
            )


class FuturesTradeRecommendationStore(ABC):
    """Records futures trade recommendations as immutable history.

    Implementations must call validate_futures_trade_recommendation_recording
    first, then honour these semantics (the module documentation explains the
    audit):

    - the policy and profile versions are registered with their contents on
      first use; a version already registered with different contents raises
      FuturesTradeRecommendationConfigurationConflictError.
    - identities absent, and no open recommendation on their product: recorded,
      with the audit.
    - the same recording already committed -- equal definitions, equal audit:
      an idempotent success.
    - any other identity already present: raise
      FuturesTradeRecommendationConflictError; the record is unchanged.
    - the audit identity already stored with other content: raise
      FuturesRecommendationGenerationAuditConflictError.
    - another open (non-terminal) recommendation already on the product, on any
      expiry: raise FuturesTradeRecommendationExclusivityError (ADR-018, P-1).

    A batch is all-or-nothing -- registration and audit included -- and returns
    the number of recommendations accepted. A durable implementation must
    register, check and insert definitions and audit in one transaction, so
    that concurrent writers cannot both record an open recommendation for one
    product and no audit outlives a rolled-back definition; an in-memory
    implementation cannot prove that.

    The abstract contract documents these obligations but cannot enforce them
    at runtime.
    """

    @abstractmethod
    def record(
        self,
        recommendations: tuple[FuturesTradeRecommendation, ...],
        *,
        policy: FuturesRecommendationGenerationPolicy,
        profile: FuturesTradeRecommendationGenerationProfile,
        audit: FuturesRecommendationGenerationAudit,
    ) -> int:
        """Record a batch and its RECORDED audit in one transaction; return the count."""
