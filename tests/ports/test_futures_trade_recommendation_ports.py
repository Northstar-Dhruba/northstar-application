"""Contract tests for the futures trade recommendation and contract cutoff ports."""

from __future__ import annotations

import dataclasses
import inspect
from datetime import date
from decimal import Decimal

import pytest
from northstar_core.derivatives import ExpirationDate, QuoteValue
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol, Timeframe
from northstar_core.futures import FuturesContract, FuturesProductReference
from northstar_core.recommendations.futures import (
    EntryZone,
    FuturesRecommendationGenerationPolicy,
    FuturesResearchDecisionReference,
    FuturesTradeRecommendation,
    FuturesTradeRecommendationIdentity,
    PriceTarget,
    RecommendationPolicyVersion,
    RecommendationValidity,
    StopLoss,
    TradeDirection,
)
from northstar_core.strategy import ExplanationReason, StrategyIdentity

import northstar_application.ports as ports
from northstar_application.application_services import (
    FuturesTradeRecommendationGenerationProfile,
    FuturesTradeRecommendationIdentityService,
)
from northstar_application.ports import (
    FuturesContractCutoff,
    FuturesContractCutoffSource,
    FuturesTradeRecommendationConfigurationConflictError,
    FuturesTradeRecommendationConflictError,
    FuturesTradeRecommendationExclusivityError,
    FuturesTradeRecommendationQuery,
    FuturesTradeRecommendationRepository,
    FuturesTradeRecommendationStore,
    FuturesTradingSession,
    InvalidFuturesContractCutoffError,
    InvalidFuturesTradeRecommendationQueryError,
    InvalidFuturesTradeRecommendationRecordingError,
    validate_futures_trade_recommendation_recording,
)

_NIFTY = FuturesProductReference(Symbol("NIFTY"), ExchangeCode("NSE"))
_OCT = FuturesContract(_NIFTY, ExpirationDate("2026-10-27"))
_LAST = FuturesTradingSession(
    date(2026, 10, 27), PointInTime("2026-10-27T03:45:00Z"), PointInTime("2026-10-27T10:00:00Z")
)


def test_the_query_is_by_product_only() -> None:
    query = FuturesTradeRecommendationQuery(_NIFTY)

    assert query.product == _NIFTY
    assert {field.name for field in dataclasses.fields(query)} == {"product"}


@pytest.mark.parametrize("value", [None, "NIFTY@NSE", _OCT])
def test_the_query_refuses_anything_but_a_product(value: object) -> None:
    with pytest.raises(InvalidFuturesTradeRecommendationQueryError):
        FuturesTradeRecommendationQuery(value)  # type: ignore[arg-type]


def test_a_cutoff_names_its_contract_session_and_authority() -> None:
    cutoff = FuturesContractCutoff(_OCT, _LAST, "  synthetic verified reference  ")

    assert (cutoff.contract, cutoff.last_session, cutoff.authority) == (
        _OCT,
        _LAST,
        "synthetic verified reference",
    )


@pytest.mark.parametrize(
    "contract, session, authority",
    [
        (None, _LAST, "a"),
        (_OCT, None, "a"),
        (_OCT, PointInTime("2026-10-27T10:00:00Z"), "a"),
        (_OCT, _LAST, ""),
        (_OCT, _LAST, "   "),
        (_OCT, _LAST, None),
    ],
)
def test_an_invalid_cutoff_is_refused(contract: object, session: object, authority: object) -> None:
    with pytest.raises(InvalidFuturesContractCutoffError):
        FuturesContractCutoff(contract, session, authority)  # type: ignore[arg-type]


def test_the_cutoff_cannot_be_built_from_an_expiration_date_alone() -> None:
    with pytest.raises(InvalidFuturesContractCutoffError):
        FuturesContractCutoff(_OCT, _OCT.expiration_date, "expiration")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "port",
    [
        FuturesContractCutoffSource,
        FuturesTradeRecommendationRepository,
        FuturesTradeRecommendationStore,
    ],
)
def test_the_ports_are_abstract(port: type) -> None:
    with pytest.raises(TypeError):
        port()


def test_the_store_errors_are_distinct_value_errors() -> None:
    assert issubclass(FuturesTradeRecommendationConflictError, ValueError)
    assert issubclass(FuturesTradeRecommendationExclusivityError, ValueError)
    assert not issubclass(
        FuturesTradeRecommendationConflictError, FuturesTradeRecommendationExclusivityError
    )


def test_no_publication_or_delivery_state_is_modelled() -> None:
    names = {name.lower() for name in ports.__all__}
    for fragment in ("publish", "deliver", "notif", "subscriber", "executed"):
        assert not any(fragment in name for name in names), fragment


def test_public_exports() -> None:
    for name in (
        "FuturesContractCutoff",
        "FuturesContractCutoffSource",
        "InvalidFuturesContractCutoffError",
        "FuturesTradeRecommendationQuery",
        "FuturesTradeRecommendationRepository",
        "InvalidFuturesTradeRecommendationQueryError",
        "FuturesTradeRecommendationConflictError",
        "FuturesTradeRecommendationExclusivityError",
        "FuturesTradeRecommendationStore",
    ):
        assert name in ports.__all__


# -- the recording contract ------------------------------------------------------------

_STRATEGY = StrategyIdentity("synthetic-directional-v1")
_POLICY = FuturesRecommendationGenerationPolicy(
    version=RecommendationPolicyVersion("f4-synthetic-fixture-v1"),
    approved_strategies=(_STRATEGY,),
    lookback_changes=19,
    entry_multiplier=Decimal("0.25"),
    stop_multiplier=Decimal("1.5"),
    target_multiplier=Decimal("3"),
    price_increment=Decimal("0.05"),
    min_reward_to_risk=Decimal("1.5"),
    short_enabled=True,
)
_PROFILE = FuturesTradeRecommendationGenerationProfile(
    "f5a-synthetic-profile-v1", 5, _POLICY.version
)


def _recommendation(
    *,
    strategy: StrategyIdentity = _STRATEGY,
    policy_version: RecommendationPolicyVersion = _POLICY.version,
    identity: str | None = None,
) -> FuturesTradeRecommendation:
    source = FuturesResearchDecisionReference(
        _OCT, Timeframe("1d"), PointInTime("2026-10-07T10:00:00Z"), strategy
    )
    derived = FuturesTradeRecommendationIdentityService().identity(source, _PROFILE)
    return FuturesTradeRecommendation(
        identity=FuturesTradeRecommendationIdentity(identity) if identity else derived,
        contract=_OCT,
        direction=TradeDirection.LONG,
        entry=EntryZone(QuoteValue(Decimal("24378")), QuoteValue(Decimal("24388"))),
        stop=StopLoss(QuoteValue(Decimal("24347.75"))),
        target=PriceTarget(QuoteValue(Decimal("24448.45"))),
        validity=RecommendationValidity(
            PointInTime("2026-10-08T10:00:00Z"),
            PointInTime("2026-10-14T10:00:00Z"),
            PointInTime("2026-10-27T10:00:00Z"),
        ),
        source=source,
        policy_version=policy_version,
        issued_at=PointInTime("2026-10-07T12:00:00Z"),
        thesis=(ExplanationReason("Synthetic thesis"),),
    )


def test_record_requires_policy_and_profile_by_keyword() -> None:
    parameters = inspect.signature(FuturesTradeRecommendationStore.record).parameters

    for name in ("policy", "profile"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty
    assert list(parameters) == ["self", "recommendations", "policy", "profile"]


def test_a_coherent_recording_request_is_accepted() -> None:
    validate_futures_trade_recommendation_recording(
        (_recommendation(),), policy=_POLICY, profile=_PROFILE
    )
    validate_futures_trade_recommendation_recording((), policy=_POLICY, profile=_PROFILE)


@pytest.mark.parametrize(
    "recommendations, policy, profile",
    [
        ([_recommendation()], _POLICY, _PROFILE),
        (("recommendation",), _POLICY, _PROFILE),
        ((), "policy", _PROFILE),
        ((), _POLICY, "profile"),
        (
            (),
            _POLICY,
            FuturesTradeRecommendationGenerationProfile(
                "p", 5, RecommendationPolicyVersion("f4-synthetic-fixture-v9")
            ),
        ),
        (
            (_recommendation(policy_version=RecommendationPolicyVersion("other")),),
            _POLICY,
            _PROFILE,
        ),
        ((_recommendation(strategy=StrategyIdentity("synthetic-other-v1")),), _POLICY, _PROFILE),
        ((_recommendation(identity="not-derived"),), _POLICY, _PROFILE),
    ],
    ids=[
        "list",
        "not-a-recommendation",
        "policy-type",
        "profile-type",
        "profile-for-another-policy",
        "recommendation-under-another-policy",
        "unapproved-strategy",
        "identity-not-derived",
    ],
)
def test_an_incoherent_recording_request_is_refused(
    recommendations: object, policy: object, profile: object
) -> None:
    with pytest.raises(InvalidFuturesTradeRecommendationRecordingError):
        validate_futures_trade_recommendation_recording(
            recommendations,  # type: ignore[arg-type]
            policy=policy,  # type: ignore[arg-type]
            profile=profile,  # type: ignore[arg-type]
        )


def test_an_identity_derived_under_another_profile_is_refused() -> None:
    other = FuturesTradeRecommendationGenerationProfile(
        "f5a-synthetic-profile-v2", 5, _POLICY.version
    )

    with pytest.raises(InvalidFuturesTradeRecommendationRecordingError):
        validate_futures_trade_recommendation_recording(
            (_recommendation(),), policy=_POLICY, profile=other
        )


def test_a_configuration_conflict_is_never_a_same_identity_conflict() -> None:
    """The race reconciliation catches only the latter; the former must escape it."""
    assert not issubclass(
        FuturesTradeRecommendationConfigurationConflictError,
        FuturesTradeRecommendationConflictError,
    )
    assert not issubclass(
        FuturesTradeRecommendationConflictError,
        FuturesTradeRecommendationConfigurationConflictError,
    )
    assert not issubclass(
        FuturesTradeRecommendationExclusivityError, FuturesTradeRecommendationConflictError
    )


def test_recording_contract_exports() -> None:
    for name in (
        "FuturesTradeRecommendationConfigurationConflictError",
        "InvalidFuturesTradeRecommendationRecordingError",
        "validate_futures_trade_recommendation_recording",
    ):
        assert name in ports.__all__
