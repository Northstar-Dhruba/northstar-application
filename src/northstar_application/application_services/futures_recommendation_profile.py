"""The versioned Application configuration of futures recommendation generation.

Core's FuturesRecommendationGenerationPolicy fixes the price-level mathematics.
What the policy cannot know is how long a recommendation should stay valid in
trading sessions, because sessions are calendar facts Core never resolves. The
profile carries that horizon, M sessions, under its own version, and names the
F-4 policy version it is used with. The effective generation configuration is
the policy plus the profile.

Every material value is explicit; there is no default and no production value.
A changed horizon or a different policy needs a new profile version. Nothing in
one process can detect a version reused elsewhere with different values: a
durable registry that refuses a known version with different contents is a
persistence responsibility (F-5B).
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.recommendations.futures import RecommendationPolicyVersion


class InvalidFuturesTradeRecommendationGenerationProfileError(ValueError):
    """Raised when a FuturesTradeRecommendationGenerationProfile value is invalid."""


@dataclass(frozen=True, slots=True)
class FuturesTradeRecommendationGenerationProfile:
    """Immutable, versioned validity horizon and the F-4 policy version it applies to.

    - ``version`` names this profile; it is part of every recommendation identity.
    - ``validity_sessions`` is M: a recommendation is valid through the close of
      the M-th trading session that opens after issuance.
    - ``policy_version`` is the F-4 policy version this profile is used with; a
      generation must be run with exactly that policy.
    """

    version: str
    validity_sessions: int
    policy_version: RecommendationPolicyVersion

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise InvalidFuturesTradeRecommendationGenerationProfileError(
                "FuturesTradeRecommendationGenerationProfile version must be non-empty text."
            )
        if isinstance(self.validity_sessions, bool) or not isinstance(self.validity_sessions, int):
            raise InvalidFuturesTradeRecommendationGenerationProfileError(
                "FuturesTradeRecommendationGenerationProfile validity sessions must be an integer."
            )
        if self.validity_sessions < 1:
            raise InvalidFuturesTradeRecommendationGenerationProfileError(
                "FuturesTradeRecommendationGenerationProfile validity sessions must be positive."
            )
        if not isinstance(self.policy_version, RecommendationPolicyVersion):
            raise InvalidFuturesTradeRecommendationGenerationProfileError(
                "FuturesTradeRecommendationGenerationProfile policy version must be a "
                "RecommendationPolicyVersion."
            )
        object.__setattr__(self, "version", self.version.strip())

    def __str__(self) -> str:
        return (
            f"profile {self.version}: {self.validity_sessions} session(s) "
            f"with policy {self.policy_version}"
        )
