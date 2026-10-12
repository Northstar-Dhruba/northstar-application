"""Tests for deterministic futures trade recommendation identities and the profile.

Synthetic values only. The profile and policy versions below are non-production
fixtures; nothing here is an approved configuration.

The golden digest was computed independently of the service, from the documented
key -- namespace, NIFTY, NSE, 2026-10-27, 1d, 2026-10-07T10:00:00Z,
synthetic-directional-v1, f4-synthetic-fixture-v1, f5a-synthetic-profile-v1 --
each part encoded as its UTF-8 byte length, a colon and the bytes.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
from pathlib import Path

import pytest
from northstar_core.derivatives import ExpirationDate
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol, Timeframe
from northstar_core.futures import FuturesContract, FuturesProductReference
from northstar_core.recommendations.futures import (
    FuturesResearchDecisionReference,
    FuturesTradeRecommendationIdentity,
    RecommendationPolicyVersion,
)
from northstar_core.strategy import StrategyIdentity

import northstar_application.application_services as services
from northstar_application.application_services import (
    FUTURES_TRADE_RECOMMENDATION_IDENTITY_NAMESPACE,
    FuturesTradeRecommendationGenerationProfile,
    FuturesTradeRecommendationIdentityService,
    InvalidFuturesTradeRecommendationGenerationProfileError,
)
from northstar_application.application_services import (
    futures_recommendation_identities as identities_module,
)

_NIFTY = FuturesProductReference(Symbol("NIFTY"), ExchangeCode("NSE"))
_OCT = FuturesContract(_NIFTY, ExpirationDate("2026-10-27"))
_NOV = FuturesContract(_NIFTY, ExpirationDate("2026-11-24"))
_DAILY = Timeframe("1d")
_DECIDED = PointInTime("2026-10-07T10:00:00Z")
_STRATEGY = StrategyIdentity("synthetic-directional-v1")
_POLICY = RecommendationPolicyVersion("f4-synthetic-fixture-v1")
_PROFILE = FuturesTradeRecommendationGenerationProfile("f5a-synthetic-profile-v1", 5, _POLICY)

_GOLDEN = "baf040ab56f622c38dd47d279c91ccf4508969f375ce3567a51ba028aed8b43f"


def _source(
    contract: FuturesContract = _OCT,
    timeframe: Timeframe = _DAILY,
    instant: PointInTime = _DECIDED,
    strategy: StrategyIdentity = _STRATEGY,
) -> FuturesResearchDecisionReference:
    return FuturesResearchDecisionReference(contract, timeframe, instant, strategy)


def _identity(
    source: FuturesResearchDecisionReference | None = None,
    profile: FuturesTradeRecommendationGenerationProfile = _PROFILE,
) -> FuturesTradeRecommendationIdentity:
    return FuturesTradeRecommendationIdentityService().identity(source or _source(), profile)


# -- identity -----------------------------------------------------------------------


def test_the_golden_vector() -> None:
    assert _identity() == FuturesTradeRecommendationIdentity(_GOLDEN)


def test_the_namespace_is_the_documented_one() -> None:
    assert FUTURES_TRADE_RECOMMENDATION_IDENTITY_NAMESPACE == (
        "northstar.futures-trade-recommendation.v1"
    )


def test_the_same_inputs_give_the_same_identity_from_separate_services() -> None:
    first = FuturesTradeRecommendationIdentityService().identity(_source(), _PROFILE)
    second = FuturesTradeRecommendationIdentityService().identity(_source(), _PROFILE)

    assert first == second


def test_one_instant_at_another_utc_offset_is_one_identity() -> None:
    assert _identity(_source(instant=PointInTime("2026-10-07T15:30:00+05:30"))) == _identity()


@pytest.mark.parametrize(
    "changed",
    [
        _source(contract=_NOV),
        _source(
            contract=FuturesContract(
                FuturesProductReference(Symbol("BANKNIFTY"), ExchangeCode("NSE")),
                ExpirationDate("2026-10-27"),
            )
        ),
        _source(
            contract=FuturesContract(
                FuturesProductReference(Symbol("NIFTY"), ExchangeCode("SGX")),
                ExpirationDate("2026-10-27"),
            )
        ),
        _source(timeframe=Timeframe("1h")),
        _source(instant=PointInTime("2026-10-08T10:00:00Z")),
        _source(strategy=StrategyIdentity("synthetic-other-v1")),
    ],
    ids=["expiry", "product", "exchange", "timeframe", "instant", "strategy"],
)
def test_any_changed_source_field_changes_the_identity(
    changed: FuturesResearchDecisionReference,
) -> None:
    assert _identity(changed) != _identity()


def test_a_different_policy_version_changes_the_identity() -> None:
    profile = FuturesTradeRecommendationGenerationProfile(
        "f5a-synthetic-profile-v1", 5, RecommendationPolicyVersion("f4-synthetic-fixture-v2")
    )

    assert _identity(profile=profile) != _identity()


def test_a_different_profile_version_changes_the_identity() -> None:
    profile = FuturesTradeRecommendationGenerationProfile("f5a-synthetic-profile-v2", 5, _POLICY)

    assert _identity(profile=profile) != _identity()


def test_the_horizon_is_named_by_the_profile_version_not_hashed_separately() -> None:
    """A changed horizon must come with a new version; the version is what identifies it."""
    same_version_other_horizon = FuturesTradeRecommendationGenerationProfile(
        "f5a-synthetic-profile-v1", 7, _POLICY
    )

    assert _identity(profile=same_version_other_horizon) == _identity()


def test_delimiters_inside_values_cannot_make_two_keys_collide() -> None:
    """Without length prefixes ('a:b', 'c') and ('a', 'b:c') would encode alike."""
    one = FuturesTradeRecommendationGenerationProfile("b:c", 5, RecommendationPolicyVersion("x:a"))
    two = FuturesTradeRecommendationGenerationProfile("c", 5, RecommendationPolicyVersion("x:a:b"))

    assert _identity(profile=one) != _identity(profile=two)


def test_the_digest_is_sha256_of_the_length_prefixed_key() -> None:
    parts = (
        FUTURES_TRADE_RECOMMENDATION_IDENTITY_NAMESPACE,
        "NIFTY",
        "NSE",
        "2026-10-27",
        "1d",
        "2026-10-07T10:00:00Z",
        "synthetic-directional-v1",
        "f4-synthetic-fixture-v1",
        "f5a-synthetic-profile-v1",
    )
    encoded = b"".join(str(len(part.encode())).encode() + b":" + part.encode() for part in parts)

    assert _identity().identity == hashlib.sha256(encoded).hexdigest()


def test_it_cannot_overlap_the_paper_order_namespace() -> None:
    assert "futures-paper-order" not in FUTURES_TRADE_RECOMMENDATION_IDENTITY_NAMESPACE


@pytest.mark.parametrize(
    "source, profile",
    [(None, _PROFILE), ("source", _PROFILE), (_source(), None), (_source(), "profile")],
)
def test_wrong_input_types_are_refused(source: object, profile: object) -> None:
    with pytest.raises(TypeError):
        FuturesTradeRecommendationIdentityService().identity(source, profile)  # type: ignore[arg-type]


def test_the_identity_module_reads_no_clock_randomness_or_state() -> None:
    tree = ast.parse(Path(identities_module.__file__).read_text(encoding="utf-8"))
    modules = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0
    }

    assert modules <= {"__future__", "hashlib", "northstar_core", "northstar_application"}
    assert not any(isinstance(node, ast.Global) for node in ast.walk(tree))


# -- profile ------------------------------------------------------------------------


def test_the_profile_keeps_its_values() -> None:
    assert (_PROFILE.version, _PROFILE.validity_sessions, _PROFILE.policy_version) == (
        "f5a-synthetic-profile-v1",
        5,
        _POLICY,
    )


@pytest.mark.parametrize(
    "version, sessions, policy",
    [
        ("", 5, _POLICY),
        ("   ", 5, _POLICY),
        (None, 5, _POLICY),
        ("v", 0, _POLICY),
        ("v", -1, _POLICY),
        ("v", True, _POLICY),
        ("v", 5.0, _POLICY),
        ("v", "5", _POLICY),
        ("v", 5, "f4-synthetic-fixture-v1"),
    ],
)
def test_an_invalid_profile_is_refused(version: object, sessions: object, policy: object) -> None:
    with pytest.raises(InvalidFuturesTradeRecommendationGenerationProfileError):
        FuturesTradeRecommendationGenerationProfile(version, sessions, policy)  # type: ignore[arg-type]


def test_the_profile_has_no_default_horizon() -> None:
    with pytest.raises(TypeError):
        FuturesTradeRecommendationGenerationProfile("v", policy_version=_POLICY)  # type: ignore[call-arg]


def test_the_profile_is_immutable_and_compares_by_value() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        _PROFILE.validity_sessions = 6  # type: ignore[misc]
    assert (
        FuturesTradeRecommendationGenerationProfile(" f5a-synthetic-profile-v1 ", 5, _POLICY)
        == _PROFILE
    )


def test_public_exports() -> None:
    for name in (
        "FUTURES_TRADE_RECOMMENDATION_IDENTITY_NAMESPACE",
        "FuturesTradeRecommendationIdentityService",
        "FuturesTradeRecommendationGenerationProfile",
        "InvalidFuturesTradeRecommendationGenerationProfileError",
    ):
        assert name in services.__all__
