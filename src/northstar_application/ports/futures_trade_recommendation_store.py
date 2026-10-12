"""Application port for internally recording futures trade recommendations.

Recording is internal: a recorded recommendation is not published, delivered,
notified or executed, and this port models none of those states.

Every recording names the effective generation configuration it was produced
under -- the F-4 policy and the Application profile -- because an F-2
recommendation carries its policy version but not its profile version, and the
profile version is part of its identity. A store cannot re-derive an identity,
or register the configuration, from the recommendation alone.

Three distinct conflicts
------------------------
- FuturesTradeRecommendationConflictError: the identity is already recorded
  with a different definition. Under one source and one configuration this is
  the expected outcome of two attempts racing -- they differ in issuance time,
  so in levels and validity too -- and the first recording wins. A caller may
  reconcile it by re-reading the identity.
- FuturesTradeRecommendationConfigurationConflictError: a policy version or a
  profile version is already registered with different contents. That is a
  version reused for a different configuration, never a retry, and it is
  deliberately not a FuturesTradeRecommendationConflictError so that no
  reconciliation path can catch it.
- FuturesTradeRecommendationExclusivityError: another open recommendation is
  already on the product, on any expiry (ADR-018, P-1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from northstar_core.recommendations.futures import (
    FuturesRecommendationGenerationPolicy,
    FuturesTradeRecommendation,
)

if TYPE_CHECKING:
    from northstar_application.application_services.futures_recommendation_profile import (
        FuturesTradeRecommendationGenerationProfile,
    )


class FuturesTradeRecommendationConflictError(ValueError):
    """Raised when a recorded recommendation would be replaced by a different one."""


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
) -> None:
    """Refuse a recording request whose parts do not describe one configuration.

    Every store implementation calls this before touching storage, so all of
    them refuse the same requests:

    - ``recommendations`` is a tuple of FuturesTradeRecommendation values;
    - ``policy`` is a FuturesRecommendationGenerationPolicy and ``profile`` a
      FuturesTradeRecommendationGenerationProfile naming that policy's version;
    - every recommendation carries the policy's version and comes from the
      policy's approved strategy;
    - every identity is exactly the one derived from the recommendation's
      source under the profile.

    It cannot check that a version was never registered with other contents;
    that needs the store's durable registry.
    """
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


class FuturesTradeRecommendationStore(ABC):
    """Records futures trade recommendations as immutable history.

    Implementations must call validate_futures_trade_recommendation_recording
    first, then honour these semantics:

    - the policy and profile versions are registered with their contents on
      first use; a version already registered with different contents raises
      FuturesTradeRecommendationConfigurationConflictError.
    - identity absent, and no open recommendation on its product: recorded.
    - identity present with an equal recommendation: an idempotent success.
    - identity present with a different recommendation: raise
      FuturesTradeRecommendationConflictError; the record is unchanged.
    - another open (non-terminal) recommendation already on the product, on any
      expiry: raise FuturesTradeRecommendationExclusivityError (ADR-018, P-1).

    A batch is all-or-nothing -- registration included -- and returns the
    number of recommendations accepted. A durable implementation must register,
    check and insert in one transaction, so that concurrent writers cannot both
    record an open recommendation for one product; an in-memory implementation
    cannot prove that.

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
    ) -> int:
        """Record a batch generated under ``policy`` and ``profile``; return the count."""
