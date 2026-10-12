"""Application derivation of deterministic futures trade recommendation identities.

Core never generates a recommendation identity (F-2). Application derives one
from the frozen research decision a recommendation comes from and the effective
generation configuration, with the length-prefixed SHA-256 scheme of the paper
execution identities under its own namespace:

    northstar.futures-trade-recommendation.v1

Canonical key, in this exact order:

    product code
    exchange code
    expiration date            (YYYY-MM-DD)
    timeframe
    decision instant           (canonical UTC PointInTime text)
    strategy identity
    F-4 policy version
    Application profile version

The first six are the frozen research decision's natural key; the last two are
the effective configuration. Each part is encoded as its UTF-8 byte length, a
colon, then the bytes, so no delimiter inside a value can make two keys collide.
The instant is the canonical UTC spelling, so one instant written at different
offsets gives one identity.

Deliberately excluded: issuance time, price levels, validity, any clock reading,
random value or storage row id. The same source decision under the same
configuration therefore always arrives at the same identity, and a retry finds
what was recorded; a different configuration arrives at a different one.

A digest identifies; it does not enforce. Business uniqueness needs the durable
store's constraints (F-5B). The identities are opaque: nothing may parse them.
"""

from __future__ import annotations

import hashlib

from northstar_core.recommendations.futures import (
    FuturesResearchDecisionReference,
    FuturesTradeRecommendationIdentity,
)

from northstar_application.application_services.futures_recommendation_profile import (
    FuturesTradeRecommendationGenerationProfile,
)

FUTURES_TRADE_RECOMMENDATION_IDENTITY_NAMESPACE = "northstar.futures-trade-recommendation.v1"


def _canonical(parts: tuple[str, ...]) -> bytes:
    """Return an injective byte encoding of an ordered string tuple."""
    encoded = bytearray()
    for part in parts:
        raw = part.encode("utf-8")
        encoded += str(len(raw)).encode("ascii")
        encoded += b":"
        encoded += raw
    return bytes(encoded)


class FuturesTradeRecommendationIdentityService:
    """Derives the identity of the recommendation for one source decision and profile."""

    def identity(
        self,
        source: FuturesResearchDecisionReference,
        profile: FuturesTradeRecommendationGenerationProfile,
    ) -> FuturesTradeRecommendationIdentity:
        """Return the deterministic identity for ``source`` under ``profile``."""
        if not isinstance(source, FuturesResearchDecisionReference):
            raise TypeError(
                "FuturesTradeRecommendationIdentityService source must be a "
                "FuturesResearchDecisionReference."
            )
        if not isinstance(profile, FuturesTradeRecommendationGenerationProfile):
            raise TypeError(
                "FuturesTradeRecommendationIdentityService profile must be a "
                "FuturesTradeRecommendationGenerationProfile."
            )
        contract = source.contract
        key = (
            FUTURES_TRADE_RECOMMENDATION_IDENTITY_NAMESPACE,
            contract.product.product_code.value,
            contract.product.exchange_code.value,
            contract.expiration_date.value,
            source.timeframe.value,
            source.decision_instant.value,
            source.strategy_identity.identity,
            profile.policy_version.version,
            profile.version,
        )
        return FuturesTradeRecommendationIdentity(hashlib.sha256(_canonical(key)).hexdigest())
