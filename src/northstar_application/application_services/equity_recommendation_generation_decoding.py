"""Strict decoding of stored equity recommendation-generation documents (EI-4D-A).

EI-4D records hold canonical documents (EI4-D9-C, ADR-021). Rehydrating one
must rebuild exactly what was recorded, or fail closed (EI4-D9-I). Every
decoder here:

- requires a JSON object with exactly its expected fields: a missing or an
  extra field is refused;
- requires every value to have its exact JSON type -- text for decimals,
  instants, dates and versions; integers for counts; never a float;
- rebuilds the value through the real Core and Application constructors, so
  every existing invariant runs again, and duplicates no recommendation
  mathematics;
- re-encodes the rebuilt value and requires the result to equal the stored
  document exactly, so a non-canonical spelling -- ``"1.50"`` for ``"1.5"``,
  an instant written at an offset -- is refused rather than coerced.

A document whose ``format`` belongs to a known family but names another
version raises IncompatibleEquityRecommendationRecordError: it may be valid
for another evaluator, and nothing here migrates it. Every other defect
raises InvalidEquityRecommendationRecordError. Nothing is reconstructed from
today's reference sources.

``EQUITY_RECOMMENDATION_EVALUATOR_VERSION`` names the evaluation behaviour --
the EI-4C assessment and the EI-4B Core formation -- that produced a record.
As with EI3-D13, there is no historical evaluator registry: only the current
version is supported, and any other fails closed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, time, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, NoReturn

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import (
    Currency,
    ExchangeCode,
    Money,
    PointInTime,
    Price,
    Symbol,
    Timeframe,
)
from northstar_core.recommendations.equity import (
    EquityCostSide,
    EquityEntryZone,
    EquityIssuanceKind,
    EquityPriceIncrement,
    EquityRecommendationDefinition,
    EquityRecommendationDirection,
    EquityRecommendationLevels,
    EquityRecommendationRejection,
    EquityRecommendationRejectionReason,
    EquityRecommendationTiming,
    EquityResearchCandidateReference,
    EquitySyntheticCostComponent,
    EquitySyntheticCostSchedule,
    EquitySyntheticGenerationPolicy,
    EquitySyntheticRiskPolicy,
    EquitySyntheticSizing,
    EquitySyntheticVersion,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services.equity_recommendation_generation_evidence import (
    EQUITY_RECOMMENDATION_ASSESSMENT_FORMAT,
    EQUITY_RECOMMENDATION_CORE_OUTCOME_FORMAT,
    EQUITY_RECOMMENDATION_INPUTS_FORMAT,
    EQUITY_RECOMMENDATION_POLICY_SET_FORMAT,
    EQUITY_RECOMMENDATION_PROPOSAL_FORMAT,
    EquityIssuedHistoryEvidence,
    EquityRecommendationCandidateProvenance,
    EquityRecommendationInputs,
    EquityResolvedSyntheticPolicySet,
    EquitySyntheticPolicySetReference,
    EquityVenueDayEvidence,
    InvalidEquityRecommendationDocumentError,
    verify_equity_recommendation_document,
)
from northstar_application.application_services.equity_trade_recommendation_eligibility import (
    EquityModeledEntryWindow,
    EquityRecommendationCheck,
    EquityRecommendationCheckStatus,
    EquityRecommendationEligibilityReason,
    EquityRecommendationGenerationStatus,
    EquityTradeRecommendationProposal,
    EquityTradeRecommendationRequest,
    equity_core_outcome_document,
    equity_recommendation_status,
    ordered_equity_recommendation_reasons,
)
from northstar_application.application_services.evaluate_equity_intraday_research import (
    EquityOpeningRangeBreakoutConfiguration,
)
from northstar_application.ports.equity_recommendation_references import (
    EquityPriorSessionRecommendation,
    EquitySyntheticEligibilityCriterion,
    EquitySyntheticEligibilityFinding,
    EquitySyntheticEligibilityVerdict,
    EquitySyntheticEntryTimingPolicy,
    EquitySyntheticListingEligibility,
)
from northstar_application.ports.equity_research_reference_provenance import (
    EquityReferenceProvenanceKind,
)

EQUITY_RECOMMENDATION_EVALUATOR_VERSION = "northstar.equity-recommendation-evaluator/1"
SUPPORTED_EQUITY_RECOMMENDATION_EVALUATOR_VERSIONS = frozenset(
    {EQUITY_RECOMMENDATION_EVALUATOR_VERSION}
)


class InvalidEquityRecommendationRecordError(ValueError):
    """Raised when a stored document or record is malformed, inconsistent or does not verify."""


class IncompatibleEquityRecommendationRecordError(InvalidEquityRecommendationRecordError):
    """Raised when a record uses a format or evaluator version this code does not support."""


def _bad(message: str) -> NoReturn:
    raise InvalidEquityRecommendationRecordError(message)


# ---------------------------------------------------------------------------
# Primitive readers
# ---------------------------------------------------------------------------


def _object(value: object, keys: tuple[str, ...], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _bad(f"{where} must be a JSON object.")
    if set(value) != set(keys):
        missing = sorted(set(keys) - set(value))
        extra = sorted(set(value) - set(keys))
        _bad(f"{where} has missing fields {missing} or unexpected fields {extra}.")
    return value


def _text(value: object, where: str) -> str:
    if not isinstance(value, str):
        _bad(f"{where} must be text.")
    return value


def _integer(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _bad(f"{where} must be an integer.")
    return value


def _flag(value: object, expected: bool, where: str) -> None:
    if value is not expected:
        _bad(f"{where} must be {str(expected).lower()}.")


def _decimal(value: object, where: str) -> Decimal:
    text = _text(value, where)
    try:
        number = Decimal(text)
    except InvalidOperation:
        _bad(f"{where} is not a decimal.")
    if not number.is_finite():
        _bad(f"{where} must be finite.")
    return number


def _instant(value: object, where: str) -> PointInTime:
    text = _text(value, where)
    instant = PointInTime(text)
    if instant.value != text:
        _bad(f"{where} is not a canonical UTC instant.")
    return instant


def _day(value: object, where: str) -> date:
    text = _text(value, where)
    day = date.fromisoformat(text)
    if day.isoformat() != text:
        _bad(f"{where} is not a canonical date.")
    return day


def _clock(value: object, where: str) -> time:
    text = _text(value, where)
    clock = time.fromisoformat(text)
    if clock.isoformat() != text:
        _bad(f"{where} is not a canonical wall-clock time.")
    return clock


def _version(value: object, where: str) -> EquitySyntheticVersion:
    return EquitySyntheticVersion(_text(value, where))


def _listing(value: object, where: str) -> ListingReference:
    document = _object(value, ("symbol", "exchange_code"), where)
    return ListingReference(
        Symbol(_text(document["symbol"], f"{where}.symbol")),
        ExchangeCode(_text(document["exchange_code"], f"{where}.exchange_code")),
    )


def _require_format(document: dict[str, Any], expected: str, where: str) -> None:
    stated = document.get("format")
    if stated == expected:
        return
    family = expected.rsplit("/", 1)[0] + "/"
    if isinstance(stated, str) and stated.startswith(family):
        raise IncompatibleEquityRecommendationRecordError(
            f"{where} is {stated}; only {expected} is supported. Nothing is migrated."
        )
    _bad(f"{where} must be a {expected} document, not {stated!r}.")


def _round_trip(value: Any, document: dict[str, Any], where: str) -> None:
    if value.document() != document:
        _bad(f"{where} is not in canonical form.")


def _decoding(where: str, decode: Any) -> Any:
    """Run ``decode``; turn any constructor refusal into a record error."""
    try:
        return decode()
    except InvalidEquityRecommendationRecordError:
        raise
    except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
        raise InvalidEquityRecommendationRecordError(f"{where} does not decode: {exc}") from exc


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

_CONFIGURATION_FIELDS = (
    "strategy",
    "version",
    "opening_range_bars",
    "volume_multiplier",
    "minimum_baseline_bars",
    "timeframe",
    "window_policy_version",
)
_POLICY_FIELDS = (
    "generation_policy_version",
    "risk_policy_version",
    "cost_schedule_version",
    "price_increment_version",
    "timing_policy_version",
)


def decode_equity_research_configuration(value: object) -> EquityOpeningRangeBreakoutConfiguration:
    """Rebuild a stored EI-3A research configuration exactly."""

    def decode() -> EquityOpeningRangeBreakoutConfiguration:
        d = _object(value, _CONFIGURATION_FIELDS, "research configuration")
        configuration = EquityOpeningRangeBreakoutConfiguration(
            strategy=StrategyIdentity(_text(d["strategy"], "strategy")),
            version=_text(d["version"], "version"),
            opening_range_bars=_integer(d["opening_range_bars"], "opening_range_bars"),
            volume_multiplier=_decimal(d["volume_multiplier"], "volume_multiplier"),
            minimum_baseline_bars=_integer(d["minimum_baseline_bars"], "minimum_baseline_bars"),
            timeframe=Timeframe(_text(d["timeframe"], "timeframe")),
            window_policy_version=_text(d["window_policy_version"], "window_policy_version"),
        )
        _round_trip(configuration, d, "research configuration")
        return configuration

    return _decoding("research configuration", decode)  # type: ignore[no-any-return]


def decode_equity_policy_set_reference(value: object) -> EquitySyntheticPolicySetReference:
    """Rebuild a stored policy-set reference exactly."""

    def decode() -> EquitySyntheticPolicySetReference:
        d = _object(value, ("version", *_POLICY_FIELDS), "policy-set reference")
        reference = EquitySyntheticPolicySetReference(
            *(_version(d[name], name) for name in ("version", *_POLICY_FIELDS))
        )
        if reference.document() != d:
            _bad("policy-set reference is not in canonical form.")
        return reference

    return _decoding("policy-set reference", decode)  # type: ignore[no-any-return]


def decode_equity_trade_recommendation_request(value: object) -> EquityTradeRecommendationRequest:
    """Rebuild a stored generation request exactly."""

    def decode() -> EquityTradeRecommendationRequest:
        d = _object(
            value,
            (
                "strategy",
                "research_configuration",
                "listing",
                "trading_date",
                "policy_set",
                "listing_eligibility_version",
            ),
            "request",
        )
        request = EquityTradeRecommendationRequest(
            configuration=decode_equity_research_configuration(d["research_configuration"]),
            listing=_listing(d["listing"], "request.listing"),
            trading_date=_day(d["trading_date"], "request.trading_date"),
            policy_set=decode_equity_policy_set_reference(d["policy_set"]),
            listing_eligibility_version=_version(
                d["listing_eligibility_version"], "listing_eligibility_version"
            ),
        )
        _round_trip(request, d, "request")
        return request

    return _decoding("request", decode)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Synthetic policies
# ---------------------------------------------------------------------------


def _generation_policy(value: object) -> EquitySyntheticGenerationPolicy:
    d = _object(
        value,
        (
            "version",
            "entry_offset_below",
            "entry_offset_above",
            "target_risk_multiple",
            "minimum_net_reward_to_risk",
        ),
        "generation policy",
    )
    policy = EquitySyntheticGenerationPolicy(
        _version(d["version"], "version"),
        _decimal(d["entry_offset_below"], "entry_offset_below"),
        _decimal(d["entry_offset_above"], "entry_offset_above"),
        _decimal(d["target_risk_multiple"], "target_risk_multiple"),
        _decimal(d["minimum_net_reward_to_risk"], "minimum_net_reward_to_risk"),
    )
    _round_trip(policy, d, "generation policy")
    return policy


def _risk_policy(value: object) -> EquitySyntheticRiskPolicy:
    d = _object(
        value,
        ("version", "model_capital", "currency", "risk_fraction", "exposure_fraction"),
        "risk policy",
    )
    policy = EquitySyntheticRiskPolicy(
        _version(d["version"], "version"),
        Money(_decimal(d["model_capital"], "model_capital"), Currency(_text(d["currency"], "c"))),
        _decimal(d["risk_fraction"], "risk_fraction"),
        _decimal(d["exposure_fraction"], "exposure_fraction"),
    )
    _round_trip(policy, d, "risk policy")
    return policy


def _cost_schedule(value: object) -> EquitySyntheticCostSchedule:
    d = _object(
        value,
        ("version", "currency", "components", "slippage_rate_per_side", "rounding_increment"),
        "cost schedule",
    )
    components = d["components"]
    if not isinstance(components, list):
        _bad("cost schedule components must be a list.")
    rebuilt = []
    for index, item in enumerate(components):
        c = _object(
            item,
            ("label", "side", "proportional_rate", "per_share", "per_order"),
            f"cost component {index}",
        )
        rebuilt.append(
            EquitySyntheticCostComponent(
                _text(c["label"], "label"),
                EquityCostSide(_text(c["side"], "side")),
                _decimal(c["proportional_rate"], "proportional_rate"),
                _decimal(c["per_share"], "per_share"),
                _decimal(c["per_order"], "per_order"),
            )
        )
    schedule = EquitySyntheticCostSchedule(
        _version(d["version"], "version"),
        Currency(_text(d["currency"], "currency")),
        tuple(rebuilt),
        _decimal(d["slippage_rate_per_side"], "slippage_rate_per_side"),
        _decimal(d["rounding_increment"], "rounding_increment"),
    )
    _round_trip(schedule, d, "cost schedule")
    return schedule


def _price_increment(value: object) -> EquityPriceIncrement:
    d = _object(value, ("version", "amount"), "price increment")
    increment = EquityPriceIncrement(_decimal(d["amount"], "amount"), _version(d["version"], "v"))
    _round_trip(increment, d, "price increment")
    return increment


def _timing_policy(value: object) -> EquitySyntheticEntryTimingPolicy:
    d = _object(
        value,
        (
            "version",
            "exchange_code",
            "research_window_policy_version",
            "latest_signal_bar_end",
            "latest_entry_bar_opening",
            "modeled_exit_boundary",
            "minimum_remaining_session_minutes",
            "modeled_issuance_latency_microseconds",
        ),
        "timing policy",
    )
    policy = EquitySyntheticEntryTimingPolicy(
        version=_version(d["version"], "version"),
        exchange_code=ExchangeCode(_text(d["exchange_code"], "exchange_code")),
        research_window_policy_version=_text(
            d["research_window_policy_version"], "research_window_policy_version"
        ),
        latest_signal_bar_end=_clock(d["latest_signal_bar_end"], "latest_signal_bar_end"),
        latest_entry_bar_opening=_clock(d["latest_entry_bar_opening"], "latest_entry_bar_opening"),
        modeled_exit_boundary=_clock(d["modeled_exit_boundary"], "modeled_exit_boundary"),
        minimum_remaining_session=timedelta(
            minutes=_integer(d["minimum_remaining_session_minutes"], "minimum_remaining")
        ),
        modeled_issuance_latency=timedelta(
            microseconds=_integer(d["modeled_issuance_latency_microseconds"], "latency")
        ),
    )
    _round_trip(policy, d, "timing policy")
    return policy


_SYNTHETIC_POLICY_DECODERS = {
    "generation_policy": _generation_policy,
    "risk_policy": _risk_policy,
    "cost_schedule": _cost_schedule,
    "price_increment": _price_increment,
    "timing_policy": _timing_policy,
}


def decode_equity_synthetic_policy_component(kind: str, value: object) -> Any:
    """Rebuild one stored synthetic policy component exactly (EI4-D10).

    ``kind`` is the component's key in the policy-set document --
    ``generation_policy``, ``risk_policy``, ``cost_schedule``,
    ``price_increment`` or ``timing_policy`` -- and the same strict decoders
    as ``decode_equity_policy_set`` apply.
    """
    decoder = _SYNTHETIC_POLICY_DECODERS.get(kind)
    if decoder is None:
        _bad(f"{kind!r} is not a synthetic policy component.")
    return _decoding(kind.replace("_", " "), lambda: decoder(value))


def decode_equity_policy_set(value: object) -> EquityResolvedSyntheticPolicySet:
    """Rebuild a stored resolved policy set, every component included, exactly."""

    def decode() -> EquityResolvedSyntheticPolicySet:
        d = _object(
            value,
            (
                "format",
                "reference",
                "generation_policy",
                "risk_policy",
                "cost_schedule",
                "price_increment",
                "timing_policy",
            ),
            "policy set",
        )
        _require_format(d, EQUITY_RECOMMENDATION_POLICY_SET_FORMAT, "policy set")
        policy_set = EquityResolvedSyntheticPolicySet(
            reference=decode_equity_policy_set_reference(d["reference"]),
            generation_policy=_generation_policy(d["generation_policy"]),
            risk_policy=_risk_policy(d["risk_policy"]),
            cost_schedule=_cost_schedule(d["cost_schedule"]),
            price_increment=_price_increment(d["price_increment"]),
            timing_policy=_timing_policy(d["timing_policy"]),
        )
        _round_trip(policy_set, d, "policy set")
        return policy_set

    return _decoding("policy set", decode)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Frozen evidence
# ---------------------------------------------------------------------------


def _research_reference(value: object) -> EquityResearchCandidateReference:
    d = _object(
        value,
        (
            "strategy",
            "research_configuration_version",
            "listing",
            "trading_date",
            "decision_as_of",
            "research_decision_fingerprint",
            "candidate_entry_fingerprint",
            "evidence_snapshot_id",
        ),
        "candidate reference",
    )
    reference = EquityResearchCandidateReference(
        strategy=StrategyIdentity(_text(d["strategy"], "strategy")),
        research_configuration_version=_version(
            d["research_configuration_version"], "research_configuration_version"
        ),
        listing=_listing(d["listing"], "candidate reference.listing"),
        trading_date=_day(d["trading_date"], "trading_date"),
        decision_as_of=_instant(d["decision_as_of"], "decision_as_of"),
        research_decision_fingerprint=_text(d["research_decision_fingerprint"], "eir1"),
        candidate_entry_fingerprint=_text(d["candidate_entry_fingerprint"], "eirt1"),
        evidence_snapshot_id=_text(d["evidence_snapshot_id"], "ecs1"),
    )
    _round_trip(reference, d, "candidate reference")
    return reference


def _candidate(value: object) -> EquityRecommendationCandidateProvenance:
    d = _object(
        value,
        (
            "source",
            "manifest_fingerprint",
            "candidate_sequence",
            "signal_interval_end",
            "signal_close",
            "opening_range_low",
            "currency",
            "research_window",
            "venue_reference",
            "venue_provenance",
        ),
        "candidate evidence",
    )
    window = _object(
        d["research_window"], ("policy_version", "opens_at", "closes_at", "timeframe"), "window"
    )
    if window["timeframe"] != "5m":
        _bad("The research window must be the five-minute grid.")
    currency = Currency(_text(d["currency"], "currency"))
    candidate = EquityRecommendationCandidateProvenance(
        source=_research_reference(d["source"]),
        manifest_fingerprint=_text(d["manifest_fingerprint"], "manifest_fingerprint"),
        candidate_sequence=_integer(d["candidate_sequence"], "candidate_sequence"),
        signal_interval_end=_instant(d["signal_interval_end"], "signal_interval_end"),
        signal_close=Price(_decimal(d["signal_close"], "signal_close"), currency),
        opening_range_low=Price(_decimal(d["opening_range_low"], "opening_range_low"), currency),
        research_window_policy_version=_text(window["policy_version"], "window.policy_version"),
        research_window_opens_at=_instant(window["opens_at"], "window.opens_at"),
        research_window_closes_at=_instant(window["closes_at"], "window.closes_at"),
        venue_reference=_text(d["venue_reference"], "venue_reference"),
        venue_provenance=EquityReferenceProvenanceKind(
            _text(d["venue_provenance"], "venue_provenance")
        ),
    )
    _round_trip(candidate, d, "candidate evidence")
    return candidate


def _listing_eligibility(value: object) -> EquitySyntheticListingEligibility:
    d = _object(
        value,
        ("version", "universe_version", "listing", "trading_date", "findings", "provenance"),
        "listing eligibility",
    )
    findings = d["findings"]
    if not isinstance(findings, list):
        _bad("listing eligibility findings must be a list.")
    rebuilt = []
    for index, item in enumerate(findings):
        f = _object(item, ("criterion", "verdict", "basis"), f"eligibility finding {index}")
        rebuilt.append(
            EquitySyntheticEligibilityFinding(
                EquitySyntheticEligibilityCriterion(_text(f["criterion"], "criterion")),
                EquitySyntheticEligibilityVerdict(_text(f["verdict"], "verdict")),
                _text(f["basis"], "basis"),
            )
        )
    eligibility = EquitySyntheticListingEligibility(
        version=_version(d["version"], "version"),
        universe_version=_version(d["universe_version"], "universe_version"),
        listing=_listing(d["listing"], "listing eligibility.listing"),
        trading_date=_day(d["trading_date"], "trading_date"),
        findings=tuple(rebuilt),
        provenance=_text(d["provenance"], "provenance"),
    )
    _round_trip(eligibility, d, "listing eligibility")
    return eligibility


def _issued_record(value: object, index: int) -> EquityPriorSessionRecommendation:
    d = _object(
        value,
        (
            "recommendation_key",
            "strategy",
            "listing",
            "trading_date",
            "issuance_kind",
            "issued_at",
            "recording_sequence",
        ),
        f"issued record {index}",
    )
    record = EquityPriorSessionRecommendation(
        recommendation_key=_text(d["recommendation_key"], "recommendation_key"),
        strategy=StrategyIdentity(_text(d["strategy"], "strategy")),
        listing=_listing(d["listing"], "issued record.listing"),
        trading_date=_day(d["trading_date"], "trading_date"),
        issuance_kind=EquityIssuanceKind(_text(d["issuance_kind"], "issuance_kind")),
        issued_at=_instant(d["issued_at"], "issued_at"),
        recording_sequence=_integer(d["recording_sequence"], "recording_sequence"),
    )
    _round_trip(record, d, f"issued record {index}")
    return record


def _issued_history(value: object) -> EquityIssuedHistoryEvidence:
    d = _object(
        value,
        (
            "exchange_code",
            "trading_date",
            "reference",
            "as_of",
            "complete_through",
            "recording_watermark",
            "recommendations",
        ),
        "issued history",
    )
    records = d["recommendations"]
    if not isinstance(records, list):
        _bad("issued history recommendations must be a list.")
    history = EquityIssuedHistoryEvidence(
        exchange_code=ExchangeCode(_text(d["exchange_code"], "exchange_code")),
        trading_date=_day(d["trading_date"], "trading_date"),
        reference=_text(d["reference"], "reference"),
        as_of=_instant(d["as_of"], "as_of"),
        complete_through=_instant(d["complete_through"], "complete_through"),
        recording_watermark=_integer(d["recording_watermark"], "recording_watermark"),
        recommendations=tuple(_issued_record(item, i) for i, item in enumerate(records)),
    )
    _round_trip(history, d, "issued history")
    return history


def decode_equity_recommendation_inputs(value: object) -> EquityRecommendationInputs:
    """Rebuild a stored frozen input envelope exactly, every fingerprint re-derived."""

    def decode() -> EquityRecommendationInputs:
        d = _object(
            value,
            (
                "format",
                "candidate",
                "policy_set",
                "policy_set_fingerprint",
                "listing_eligibility",
                "venue_day",
                "issued_history",
                "modeled_issued_at",
            ),
            "generation inputs",
        )
        _require_format(d, EQUITY_RECOMMENDATION_INPUTS_FORMAT, "generation inputs")
        inputs = EquityRecommendationInputs(
            candidate=_candidate(d["candidate"]),
            policy_set=decode_equity_policy_set(d["policy_set"]),
            listing_eligibility=_listing_eligibility(d["listing_eligibility"]),
            venue_day=EquityVenueDayEvidence.from_document(d["venue_day"]),
            issued_history=_issued_history(d["issued_history"]),
        )
        _round_trip(inputs, d, "generation inputs")
        return inputs

    return _decoding("generation inputs", decode)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Core formation outcome
# ---------------------------------------------------------------------------

_SIZING_AMOUNTS = (
    "entry_notional",
    "entry_cost",
    "stop_exit_cost",
    "target_exit_cost",
    "gross_risk",
    "net_risk",
    "gross_reward",
    "net_reward",
    "risk_budget",
    "exposure_budget",
)


def _definition(value: object) -> EquityRecommendationDefinition:
    d = _object(
        value,
        (
            "source",
            "direction",
            "levels",
            "sizing",
            "price_increment",
            "generation_policy_version",
            "risk_policy_version",
            "cost_schedule_version",
        ),
        "Core definition",
    )
    levels = _object(d["levels"], ("currency", "entry_zone", "stop", "target"), "levels")
    zone = _object(levels["entry_zone"], ("low", "high"), "entry zone")
    currency = Currency(_text(levels["currency"], "levels.currency"))

    def price(raw: object, where: str) -> Price:
        return Price(_decimal(raw, where), currency)

    sizing = _object(d["sizing"], ("currency", "quantity", *_SIZING_AMOUNTS), "sizing")
    definition = EquityRecommendationDefinition(
        source=_research_reference(d["source"]),
        direction=EquityRecommendationDirection(_text(d["direction"], "direction")),
        levels=EquityRecommendationLevels(
            EquityEntryZone(price(zone["low"], "zone.low"), price(zone["high"], "zone.high")),
            price(levels["stop"], "stop"),
            price(levels["target"], "target"),
        ),
        sizing=EquitySyntheticSizing(
            Currency(_text(sizing["currency"], "sizing.currency")),
            _integer(sizing["quantity"], "quantity"),
            *(_decimal(sizing[name], name) for name in _SIZING_AMOUNTS),
        ),
        price_increment=_price_increment(d["price_increment"]),
        generation_policy_version=_version(d["generation_policy_version"], "generation"),
        risk_policy_version=_version(d["risk_policy_version"], "risk"),
        cost_schedule_version=_version(d["cost_schedule_version"], "cost"),
    )
    _round_trip(definition, d, "Core definition")
    return definition


def _rejection(value: object) -> EquityRecommendationRejection:
    d = _object(value, ("source", "generation_policy_version", "reasons"), "Core rejection")
    reasons = d["reasons"]
    if not isinstance(reasons, list):
        _bad("Core rejection reasons must be a list.")
    rejection = EquityRecommendationRejection(
        _research_reference(d["source"]),
        _version(d["generation_policy_version"], "generation_policy_version"),
        tuple(EquityRecommendationRejectionReason(_text(r, "reason")) for r in reasons),
    )
    _round_trip(rejection, d, "Core rejection")
    return rejection


def decode_equity_core_outcome(
    value: object,
) -> EquityRecommendationDefinition | EquityRecommendationRejection:
    """Rebuild Core's stored formation outcome through Core's own constructors."""

    def decode() -> EquityRecommendationDefinition | EquityRecommendationRejection:
        d = _object(value, ("format", "kind", "outcome"), "Core outcome")
        _require_format(d, EQUITY_RECOMMENDATION_CORE_OUTCOME_FORMAT, "Core outcome")
        kind = d["kind"]
        if kind == "DEFINITION":
            outcome: Any = _definition(d["outcome"])
        elif kind == "REJECTION":
            outcome = _rejection(d["outcome"])
        else:
            _bad(f"Core outcome kind {kind!r} is unknown.")
        if equity_core_outcome_document(outcome) != d:
            _bad("Core outcome is not in canonical form.")
        return outcome  # type: ignore[no-any-return]

    return _decoding("Core outcome", decode)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Entry window and proposal
# ---------------------------------------------------------------------------


def _entry_window(value: object) -> EquityModeledEntryWindow:
    d = _object(
        value,
        ("timing", "latest_entry_bar_opens_at", "modeled_exit_at", "timing_policy_version"),
        "entry window",
    )
    timing = _object(
        d["timing"], ("decision_as_of", "issuance_kind", "issued_at", "eligible_from"), "timing"
    )
    window = EquityModeledEntryWindow(
        timing=EquityRecommendationTiming(
            _instant(timing["decision_as_of"], "decision_as_of"),
            EquityIssuanceKind(_text(timing["issuance_kind"], "issuance_kind")),
            _instant(timing["issued_at"], "issued_at"),
            _instant(timing["eligible_from"], "eligible_from"),
        ),
        latest_entry_bar_opens_at=_instant(d["latest_entry_bar_opens_at"], "latest_entry"),
        modeled_exit_at=_instant(d["modeled_exit_at"], "modeled_exit_at"),
        timing_policy_version=_version(d["timing_policy_version"], "timing_policy_version"),
    )
    _round_trip(window, d, "entry window")
    return window


def decode_equity_trade_recommendation_proposal(value: object) -> EquityTradeRecommendationProposal:
    """Rebuild a stored synthetic proposal exactly."""

    def decode() -> EquityTradeRecommendationProposal:
        d = _object(
            value,
            (
                "format",
                "generation_key",
                "synthetic",
                "authoritative",
                "issued",
                "actionable",
                "definition",
                "entry_window",
                "listing_eligibility_version",
                "universe_version",
                "not_assessed",
            ),
            "proposal",
        )
        _require_format(d, EQUITY_RECOMMENDATION_PROPOSAL_FORMAT, "proposal")
        definition = decode_equity_core_outcome(d["definition"])
        if not isinstance(definition, EquityRecommendationDefinition):
            _bad("A proposal holds a Core definition, never a rejection.")
        not_assessed = d["not_assessed"]
        if not isinstance(not_assessed, list):
            _bad("proposal not_assessed must be a list.")
        proposal = EquityTradeRecommendationProposal(
            generation_key=_text(d["generation_key"], "generation_key"),
            definition=definition,
            entry_window=_entry_window(d["entry_window"]),
            listing_eligibility_version=_version(d["listing_eligibility_version"], "eligibility"),
            universe_version=_version(d["universe_version"], "universe_version"),
            not_assessed=tuple(EquityRecommendationCheck(_text(c, "check")) for c in not_assessed),
        )
        _round_trip(proposal, d, "proposal")
        return proposal

    return _decoding("proposal", decode)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------

_ASSESSMENT_FIELDS = (
    "format",
    "synthetic",
    "authoritative",
    "request",
    "generation_key",
    "status",
    "reasons",
    "checks",
    "candidate",
    "inputs",
    "inputs_fingerprint",
    "entry_window",
    "core_outcome",
    "proposal",
    "proposal_fingerprint",
)


@dataclass(frozen=True, slots=True)
class EquityDecodedAssessment:
    """A stored assessment, verified and decoded into its structured parts.

    The diagnostics of the original audit are not part of an assessment and
    are not recovered. ``inputs`` is None exactly when no generation identity
    was established.
    """

    document: dict[str, Any]
    fingerprint: str
    request: EquityTradeRecommendationRequest
    generation_key: str | None
    status: EquityRecommendationGenerationStatus
    reasons: tuple[EquityRecommendationEligibilityReason, ...]
    checks: tuple[
        tuple[
            EquityRecommendationCheck,
            EquityRecommendationCheckStatus,
            tuple[EquityRecommendationEligibilityReason, ...],
        ],
        ...,
    ]
    candidate: EquityRecommendationCandidateProvenance | None
    inputs: EquityRecommendationInputs | None
    entry_window: EquityModeledEntryWindow | None
    core_outcome: EquityRecommendationDefinition | EquityRecommendationRejection | None
    proposal: EquityTradeRecommendationProposal | None


def _reasons(value: object, where: str) -> tuple[EquityRecommendationEligibilityReason, ...]:
    if not isinstance(value, list):
        _bad(f"{where} must be a list.")
    reasons = tuple(EquityRecommendationEligibilityReason(_text(r, where)) for r in value)
    if reasons != ordered_equity_recommendation_reasons(reasons):
        _bad(f"{where} must be distinct and in declaration order.")
    return reasons


def decode_structured_checks(
    value: object,
) -> tuple[
    tuple[
        EquityRecommendationCheck,
        EquityRecommendationCheckStatus,
        tuple[EquityRecommendationEligibilityReason, ...],
    ],
    ...,
]:
    """Decode an assessment's structured checks: every check once, in order."""
    if not isinstance(value, list):
        _bad("checks must be a list.")
    checks = []
    for index, item in enumerate(value):
        c = _object(item, ("check", "status", "reasons"), f"check {index}")
        status = EquityRecommendationCheckStatus(_text(c["status"], "status"))
        reasons = _reasons(c["reasons"], f"check {index} reasons")
        if (status is EquityRecommendationCheckStatus.FAILED) != bool(reasons):
            _bad(f"check {index}: exactly a FAILED check has reasons.")
        checks.append((EquityRecommendationCheck(_text(c["check"], "check")), status, reasons))
    if tuple(c for c, _, _ in checks) != tuple(EquityRecommendationCheck):
        _bad("checks must report every check once, in declaration order.")
    return tuple(checks)


def decode_equity_recommendation_assessment(text: str, fingerprint: str) -> EquityDecodedAssessment:
    """Verify a stored assessment against its fingerprint and decode it, or fail closed."""

    def decode() -> EquityDecodedAssessment:
        d = verified_equity_recommendation_document(
            text, fingerprint, EQUITY_RECOMMENDATION_ASSESSMENT_FORMAT, "assessment"
        )
        _object(d, _ASSESSMENT_FIELDS, "assessment")
        _flag(d["synthetic"], True, "assessment.synthetic")
        _flag(d["authoritative"], False, "assessment.authoritative")
        request = decode_equity_trade_recommendation_request(d["request"])
        status = EquityRecommendationGenerationStatus(_text(d["status"], "status"))
        reasons = _reasons(d["reasons"], "reasons")
        checks = decode_structured_checks(d["checks"])
        if equity_recommendation_status(reasons) is not status:
            _bad("The assessment's status does not follow from its reasons.")
        if set(reasons) != {r for _, _, found in checks for r in found}:
            _bad("The assessment's reasons are not its checks' reasons.")
        inputs = None if d["inputs"] is None else decode_equity_recommendation_inputs(d["inputs"])
        candidate = (
            None
            if d["candidate"] is None
            else _decoding("candidate", lambda: _candidate(d["candidate"]))
        )
        key = d["generation_key"]
        if inputs is None:
            if key is not None or d["inputs_fingerprint"] is not None:
                _bad("Without inputs there is no generation key or inputs fingerprint.")
        else:
            if inputs.candidate != candidate:
                _bad("The inputs hold another candidate than the assessment.")
            if (key, d["inputs_fingerprint"]) != (inputs.generation_key, inputs.fingerprint):
                _bad("The generation key or inputs fingerprint does not re-derive.")
            if inputs.policy_set.reference != request.policy_set or (
                inputs.listing_eligibility.version != request.listing_eligibility_version
            ):
                _bad("The inputs are not the requested policy set and eligibility.")
        window = (
            None
            if d["entry_window"] is None
            else _decoding("entry window", lambda: _entry_window(d["entry_window"]))
        )
        core = None if d["core_outcome"] is None else decode_equity_core_outcome(d["core_outcome"])
        proposal = (
            None
            if d["proposal"] is None
            else decode_equity_trade_recommendation_proposal(d["proposal"])
        )
        proposed = status is EquityRecommendationGenerationStatus.PROPOSED
        if proposed != (proposal is not None):
            _bad("Exactly a PROPOSED assessment holds a proposal.")
        if (None if proposal is None else proposal.fingerprint) != d["proposal_fingerprint"]:
            _bad("The proposal fingerprint does not re-derive.")
        if proposal is not None and (
            proposal.generation_key != key
            or proposal.definition != core
            or proposal.entry_window != window
        ):
            _bad("The proposal is not the assessment's key, Core definition and entry window.")
        if (window is not None or core is not None) and inputs is None:
            _bad("Timing and Core formation exist only with inputs.")
        if core is not None and inputs is not None and core.source != inputs.candidate.source:
            _bad("Core's outcome names another candidate than the assessment's.")
        if (
            window is not None
            and inputs is not None
            and (
                window.timing.decision_as_of != inputs.candidate.source.decision_as_of
                or window.timing.issued_at != inputs.modeled_issued_at
            )
        ):
            _bad("The entry window is not timed from the frozen candidate and issuance.")
        return EquityDecodedAssessment(
            document=d,
            fingerprint=fingerprint,
            request=request,
            generation_key=key,
            status=status,
            reasons=reasons,
            checks=checks,
            candidate=candidate,
            inputs=inputs,
            entry_window=window,
            core_outcome=core,
            proposal=proposal,
        )

    return _decoding("assessment", decode)  # type: ignore[no-any-return]


def verified_equity_recommendation_document(
    text: str, fingerprint: str, expected_format: str, where: str
) -> dict[str, Any]:
    """Decode ``text``; require ``expected_format`` and the fingerprint it states.

    Another version of the same format family is incompatible, not corrupt.
    """
    if not isinstance(text, str) or not isinstance(fingerprint, str):
        _bad(f"The stored {where} and its fingerprint must be text.")
    try:
        loose = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InvalidEquityRecommendationRecordError(f"The stored {where} is not JSON.") from exc
    if not isinstance(loose, dict):
        _bad(f"The stored {where} is not a JSON object.")
    _require_format(loose, expected_format, where)
    try:
        return verify_equity_recommendation_document(text, fingerprint)
    except InvalidEquityRecommendationDocumentError as exc:
        raise InvalidEquityRecommendationRecordError(f"The stored {where}: {exc}") from exc


def require_supported_equity_recommendation_evaluator(version: object) -> str:
    """Return ``version`` if it is a supported evaluator version; otherwise fail closed."""
    if not isinstance(version, str) or not version:
        _bad("A record names its evaluator version.")
    if version not in SUPPORTED_EQUITY_RECOMMENDATION_EVALUATOR_VERSIONS:
        raise IncompatibleEquityRecommendationRecordError(
            f"Evaluator {version} is not supported; only "
            f"{EQUITY_RECOMMENDATION_EVALUATOR_VERSION} is. Nothing is migrated."
        )
    return version
