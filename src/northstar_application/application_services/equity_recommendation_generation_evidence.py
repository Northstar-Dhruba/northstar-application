"""Persistence-ready identity, canonical documents and frozen generation evidence (EI-4C).

The EI-4C compatibility contracts required by EI4-D9 (EI-4 Decision Register,
section 9, approved by the owner on 2026-11-07) and ADR-021. Everything is
synthetic and pure: nothing here reads a clock, persists, issues or trades.

Canonical documents and fingerprints (EI4-D9-C)
-----------------------------------------------
Every identity-bearing document is a JSON object with a ``format`` version.
It is encoded canonically -- sorted keys, no whitespace, no NaN -- and holds
only text, integers, booleans, null, lists and objects: exact decimals and
instants are canonical text, and a float anywhere is refused. Its fingerprint
is SHA-256 over that encoding, prefixed by its format's namespace:

    northstar.equity-recommendation-policy-set/1          eirs1-sha256:
    northstar.equity-recommendation-generation-inputs/1   eiri1-sha256:
    northstar.equity-recommendation-assessment/1          eira1-sha256:
    northstar.equity-recommendation-proposal/1            eirp1-sha256:
    northstar.equity-recommendation-generation-attempt/1  eirat1-sha256:

Free-text diagnostics -- check details, port exception messages, a venue
reference's reason -- are never part of these documents, so they never move a
fingerprint. ``verify_equity_recommendation_document`` decodes a stored
document, refuses any float, a non-canonical encoding, a missing or
unsupported format and a fingerprint that does not re-derive. SHA-256 is
verified here, by the Application; SQLite does not verify it.

Generation identity (EI4-D9-B)
------------------------------
A composite synthetic policy set names one generation policy, one risk policy
with its model capital, one cost schedule, one price-increment reference and
one timing policy with its modeled-issuance latency. Resolved, it is the
canonical document of all five, under the set's version; its ``eirs1``
fingerprint binds their full content, not only their names.

The generation key is ``eirk1-sha256:`` followed by SHA-256 over the
length-prefixed parts (UTF-8 byte length, a colon, the bytes), in this order:

    northstar.equity-recommendation-generation-key/1
    strategy identity
    research configuration version
    symbol, exchange code
    trading date                        (YYYY-MM-DD)
    first CANDIDATE entry fingerprint   (eirt1)
    policy-set version
    policy-set fingerprint              (eirs1)

So two policy sets that differ in any component, or in their version, never
share a key. The prefix keeps it distinguishable from the provisional EI-4C
key ``northstar.equity-trade-recommendation-proposal.v1``, which keeps its own
meaning and is not the generation identity. Listing eligibility, venue-day
evidence and issued-history evidence are content, never key parts: a change
to them under one key changes the assessment fingerprint, and a store must
treat that as a conflict, never as a new disposition (EI4-D9-D). Equal keys
never prove equal content.

Frozen evidence (EI4-D9-H, EI4-D9-I)
-----------------------------------
``EquityRecommendationInputs`` is the complete, immutable input envelope of
one assessment: the original candidate and its research values and window,
the resolved policy set, the listing eligibility, the venue-day evidence used
for timing and the as-of issued-history evidence with its completeness and
recording watermark. An assessment re-run from it needs no reference source,
snapshot store or history source: today's answers are never substituted.

The candidate evidence is anchored to the first CANDIDATE entry and its own
snapshot. A later timeline head is never part of it, so later timeline entries
-- corrections included -- never change a frozen assessment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from types import MappingProxyType
from typing import Any, NoReturn

from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Price
from northstar_core.recommendations.equity import (
    EquityPriceIncrement,
    EquityResearchCandidateReference,
    EquitySyntheticCostSchedule,
    EquitySyntheticGenerationPolicy,
    EquitySyntheticRiskPolicy,
    EquitySyntheticVersion,
)

from northstar_application.application_services.equity_candle_evidence_journal import (
    canonical_equity_candle_json,
)
from northstar_application.ports.equity_recommendation_references import (
    EquityPriorSessionRecommendation,
    EquitySessionRecommendationHistory,
    EquitySyntheticEntryTimingPolicy,
    EquitySyntheticListingEligibility,
)
from northstar_application.ports.equity_research_reference_provenance import (
    EquityReferenceProvenanceKind,
)
from northstar_application.ports.equity_research_window_reference import (
    EquityVenueDayClassification,
    EquityVenueDayReference,
)

EQUITY_RECOMMENDATION_POLICY_SET_FORMAT = "northstar.equity-recommendation-policy-set/1"
EQUITY_RECOMMENDATION_INPUTS_FORMAT = "northstar.equity-recommendation-generation-inputs/1"
EQUITY_RECOMMENDATION_ASSESSMENT_FORMAT = "northstar.equity-recommendation-assessment/1"
EQUITY_RECOMMENDATION_PROPOSAL_FORMAT = "northstar.equity-recommendation-proposal/1"
EQUITY_RECOMMENDATION_ATTEMPT_FORMAT = "northstar.equity-recommendation-generation-attempt/1"
EQUITY_RECOMMENDATION_CORE_OUTCOME_FORMAT = "northstar.equity-recommendation-core-outcome/1"
EQUITY_RECOMMENDATION_GENERATION_KEY_NAMESPACE = "northstar.equity-recommendation-generation-key/1"
EQUITY_RECOMMENDATION_GENERATION_KEY_PREFIX = "eirk1-sha256:"

# The fingerprint namespace of every identity-bearing document format.
EQUITY_RECOMMENDATION_DOCUMENT_PREFIXES = MappingProxyType(
    {
        EQUITY_RECOMMENDATION_POLICY_SET_FORMAT: "eirs1-sha256:",
        EQUITY_RECOMMENDATION_INPUTS_FORMAT: "eiri1-sha256:",
        EQUITY_RECOMMENDATION_ASSESSMENT_FORMAT: "eira1-sha256:",
        EQUITY_RECOMMENDATION_PROPOSAL_FORMAT: "eirp1-sha256:",
        EQUITY_RECOMMENDATION_ATTEMPT_FORMAT: "eirat1-sha256:",
    }
)

_FIVE_MINUTES = timedelta(minutes=5)
_ONE_MINUTE = timedelta(minutes=1)
_HEX = frozenset("0123456789abcdef")


class InvalidEquityRecommendationDocumentError(ValueError):
    """Raised when a canonical document is malformed, unsupported or does not verify."""


class InvalidEquityRecommendationEvidenceError(ValueError):
    """Raised when frozen generation evidence is missing, malformed or inconsistent."""


def _evidence(message: str) -> NoReturn:
    raise InvalidEquityRecommendationEvidenceError(message)


def _document_error(message: str) -> NoReturn:
    raise InvalidEquityRecommendationDocumentError(message)


# ---------------------------------------------------------------------------
# Canonical encoding and fingerprints
# ---------------------------------------------------------------------------


def _require_plain(value: object, where: str) -> None:
    if value is None or isinstance(value, bool | int | str):
        return
    if isinstance(value, float):
        _document_error(f"{where} holds a float; canonical documents hold exact text.")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_plain(item, f"{where}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                _document_error(f"{where} has a non-text key.")
            _require_plain(item, f"{where}.{key}")
        return
    _document_error(f"{where} holds an unsupported {type(value).__name__}.")


def canonical_equity_recommendation_json(document: dict[str, Any]) -> str:
    """Return ``document`` as canonical JSON, refusing floats and unsupported values."""
    if not isinstance(document, dict):
        _document_error("A canonical document is a JSON object.")
    _require_plain(document, "document")
    return canonical_equity_candle_json(document)


def equity_recommendation_fingerprint(document: dict[str, Any]) -> str:
    """Return the namespaced SHA-256 fingerprint of a format-versioned ``document``."""
    if not isinstance(document, dict):
        _document_error("A canonical document is a JSON object.")
    prefix = EQUITY_RECOMMENDATION_DOCUMENT_PREFIXES.get(document.get("format"))  # type: ignore[arg-type]
    if prefix is None:
        _document_error(f"Unsupported or missing document format {document.get('format')!r}.")
    encoded = canonical_equity_recommendation_json(document).encode("utf-8")
    return prefix + hashlib.sha256(encoded).hexdigest()


def _refuse_float(value: str) -> NoReturn:
    _document_error(f"Stored documents hold no floating-point numbers ({value}).")


def verify_equity_recommendation_document(text: str, fingerprint: str) -> dict[str, Any]:
    """Decode ``text`` and verify it is the canonical document ``fingerprint`` names.

    Refuses a float, a non-canonical encoding, a missing or unsupported format
    and a fingerprint that does not re-derive. Returns the decoded document.
    """
    if not isinstance(text, str) or not isinstance(fingerprint, str):
        _document_error("A stored document and its fingerprint are text.")
    try:
        document = json.loads(text, parse_float=_refuse_float, parse_constant=_refuse_float)
    except json.JSONDecodeError as exc:
        _document_error(f"The stored document is not JSON: {exc}")
    if not isinstance(document, dict):
        _document_error("A canonical document is a JSON object.")
    if canonical_equity_recommendation_json(document) != text:
        _document_error("The stored document is not canonically encoded.")
    if equity_recommendation_fingerprint(document) != fingerprint:
        _document_error("The stored document does not match its fingerprint.")
    return document


def _length_prefixed(parts: tuple[str, ...]) -> bytes:
    encoded = bytearray()
    for part in parts:
        raw = part.encode("utf-8")
        encoded += str(len(raw)).encode("ascii") + b":" + raw
    return bytes(encoded)


def is_equity_recommendation_generation_key(value: object) -> bool:
    """Return whether ``value`` has the shape of a generation key."""
    prefix = EQUITY_RECOMMENDATION_GENERATION_KEY_PREFIX
    if not isinstance(value, str) or not value.startswith(prefix):
        return False
    digest = value[len(prefix) :]
    return len(digest) == 64 and set(digest) <= _HEX


# ---------------------------------------------------------------------------
# Instants
# ---------------------------------------------------------------------------


def _datetime(instant: PointInTime) -> datetime:
    return datetime.fromisoformat(instant.value.replace("Z", "+00:00")).astimezone(UTC)


def _exact_point(value: datetime) -> PointInTime:
    # Microseconds are written out in full; PointInTime canonicalizes them.
    return PointInTime(value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))


def equity_modeled_issuance(decision_as_of: PointInTime, latency: timedelta) -> PointInTime:
    """Return the modeled issuance: ``decision_as_of`` plus ``latency``, exactly (EI4-D9-C).

    Integer microsecond arithmetic only; no clock and no float.
    """
    if not isinstance(decision_as_of, PointInTime):
        raise TypeError("decision_as_of must be a PointInTime.")
    if not isinstance(latency, timedelta) or latency < timedelta(0):
        _evidence("The modeled issuance latency must be a non-negative timedelta.")
    return _exact_point(_datetime(decision_as_of) + latency)


# ---------------------------------------------------------------------------
# Policy set and generation key (EI4-D9-B)
# ---------------------------------------------------------------------------

_POLICY_FIELDS = (
    "generation_policy_version",
    "risk_policy_version",
    "cost_schedule_version",
    "price_increment_version",
    "timing_policy_version",
)


@dataclass(frozen=True, slots=True)
class EquitySyntheticPolicySetReference:
    """A versioned synthetic policy set, named by the exact version of each component."""

    version: EquitySyntheticVersion
    generation_policy_version: EquitySyntheticVersion
    risk_policy_version: EquitySyntheticVersion
    cost_schedule_version: EquitySyntheticVersion
    price_increment_version: EquitySyntheticVersion
    timing_policy_version: EquitySyntheticVersion

    def __post_init__(self) -> None:
        for name in ("version", *_POLICY_FIELDS):
            if not isinstance(getattr(self, name), EquitySyntheticVersion):
                _evidence(f"EquitySyntheticPolicySetReference {name} must be synthetic.")

    def document(self) -> dict[str, Any]:
        """Return the canonical content."""
        return {name: getattr(self, name).value for name in ("version", *_POLICY_FIELDS)}


@dataclass(frozen=True, slots=True)
class EquityResolvedSyntheticPolicySet:
    """A policy set with the full content of every component, as applied.

    Each component must be the version its reference names. ``fingerprint``
    binds the set's version and every component's full canonical content.
    """

    reference: EquitySyntheticPolicySetReference
    generation_policy: EquitySyntheticGenerationPolicy
    risk_policy: EquitySyntheticRiskPolicy
    cost_schedule: EquitySyntheticCostSchedule
    price_increment: EquityPriceIncrement
    timing_policy: EquitySyntheticEntryTimingPolicy
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.reference, EquitySyntheticPolicySetReference):
            _evidence("reference must be an EquitySyntheticPolicySetReference.")
        for name, kind, version_field in (
            ("generation_policy", EquitySyntheticGenerationPolicy, "generation_policy_version"),
            ("risk_policy", EquitySyntheticRiskPolicy, "risk_policy_version"),
            ("cost_schedule", EquitySyntheticCostSchedule, "cost_schedule_version"),
            ("price_increment", EquityPriceIncrement, "price_increment_version"),
            ("timing_policy", EquitySyntheticEntryTimingPolicy, "timing_policy_version"),
        ):
            component = getattr(self, name)
            if not isinstance(component, kind):
                _evidence(f"{name} must be an {kind.__name__}.")
            if component.version != getattr(self.reference, version_field):
                _evidence(f"{name} is not the version the policy set names.")
        object.__setattr__(self, "fingerprint", equity_recommendation_fingerprint(self.document()))

    def document(self) -> dict[str, Any]:
        """Return the canonical content."""
        return {
            "format": EQUITY_RECOMMENDATION_POLICY_SET_FORMAT,
            "reference": self.reference.document(),
            "generation_policy": self.generation_policy.document(),
            "risk_policy": self.risk_policy.document(),
            "cost_schedule": self.cost_schedule.document(),
            "price_increment": self.price_increment.document(),
            "timing_policy": self.timing_policy.document(),
        }


def equity_recommendation_generation_key(
    source: EquityResearchCandidateReference, policy_set: EquityResolvedSyntheticPolicySet
) -> str:
    """Return the generation key of ``source`` under ``policy_set`` (EI4-D9-B)."""
    if not isinstance(source, EquityResearchCandidateReference):
        raise TypeError("source must be an EquityResearchCandidateReference.")
    if not isinstance(policy_set, EquityResolvedSyntheticPolicySet):
        raise TypeError("policy_set must be an EquityResolvedSyntheticPolicySet.")
    parts = (
        EQUITY_RECOMMENDATION_GENERATION_KEY_NAMESPACE,
        source.strategy.identity,
        source.research_configuration_version.value,
        source.listing.symbol.value,
        source.listing.exchange_code.value,
        source.trading_date.isoformat(),
        source.candidate_entry_fingerprint,
        policy_set.reference.version.value,
        policy_set.fingerprint,
    )
    digest = hashlib.sha256(_length_prefixed(parts)).hexdigest()
    return EQUITY_RECOMMENDATION_GENERATION_KEY_PREFIX + digest


# ---------------------------------------------------------------------------
# Candidate evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquityRecommendationCandidateProvenance:
    """The authoritative first CANDIDATE and the research evidence it rests on, frozen.

    ``source`` is the Core reference: strategy, research configuration
    version, listing, date, decision instant and the ``eir1``, ``eirt1`` and
    ``ecs1`` references. The rest is read from that candidate's own decision
    and snapshot: the manifest, the candidate's sequence, the signal bar's
    end, the EI-3A signal close and opening-range low in the snapshot's
    currency, the research window whose five-minute grid evidences entry, and
    the venue reference the snapshot used with its provenance. None of it
    depends on later timeline entries.
    """

    source: EquityResearchCandidateReference
    manifest_fingerprint: str
    candidate_sequence: int
    signal_interval_end: PointInTime
    signal_close: Price
    opening_range_low: Price
    research_window_policy_version: str
    research_window_opens_at: PointInTime
    research_window_closes_at: PointInTime
    venue_reference: str
    venue_provenance: EquityReferenceProvenanceKind

    def __post_init__(self) -> None:
        if not isinstance(self.source, EquityResearchCandidateReference):
            _evidence("source must be an EquityResearchCandidateReference.")
        for name in ("manifest_fingerprint", "research_window_policy_version", "venue_reference"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                _evidence(f"{name} must be non-empty text.")
        sequence = self.candidate_sequence
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
            _evidence("candidate_sequence must be an integer of at least 1.")
        for name in (
            "signal_interval_end",
            "research_window_opens_at",
            "research_window_closes_at",
        ):
            if not isinstance(getattr(self, name), PointInTime):
                _evidence(f"{name} must be a PointInTime.")
        if not isinstance(self.signal_close, Price) or not isinstance(
            self.opening_range_low, Price
        ):
            _evidence("The research values must be Prices.")
        if self.signal_close.currency != self.opening_range_low.currency:
            _evidence("The research values must share one currency.")
        if not isinstance(self.venue_provenance, EquityReferenceProvenanceKind):
            _evidence("venue_provenance must be an EquityReferenceProvenanceKind.")
        opens = _datetime(self.research_window_opens_at)
        closes = _datetime(self.research_window_closes_at)
        if opens >= closes or (closes - opens) % _FIVE_MINUTES:
            _evidence("The research window must be a whole number of five-minute bars.")
        end = _datetime(self.signal_interval_end)
        if not opens < end <= closes or (end - opens) % _FIVE_MINUTES:
            _evidence("The signal bar must end on the research window's grid.")
        if self.signal_interval_end.compare(self.source.decision_as_of) >= 0:
            _evidence("The decision must follow the end of its signal bar.")

    def entry_bar_openings(self) -> tuple[PointInTime, ...]:
        """Return the opening of every five-minute bar in the research window, ascending."""
        opens = _datetime(self.research_window_opens_at)
        closes = _datetime(self.research_window_closes_at)
        count = (closes - opens) // _FIVE_MINUTES
        return tuple(_exact_point(opens + index * _FIVE_MINUTES) for index in range(count))

    def bar_ends(self) -> frozenset[PointInTime]:
        """Return the end of every five-minute bar in the research window."""
        return frozenset(
            _exact_point(_datetime(opening) + _FIVE_MINUTES)
            for opening in self.entry_bar_openings()
        )

    def document(self) -> dict[str, Any]:
        """Return the canonical content."""
        return {
            "source": self.source.document(),
            "manifest_fingerprint": self.manifest_fingerprint,
            "candidate_sequence": self.candidate_sequence,
            "signal_interval_end": self.signal_interval_end.value,
            "signal_close": format(self.signal_close.amount, "f"),
            "opening_range_low": format(self.opening_range_low.amount, "f"),
            "currency": self.signal_close.currency.value,
            "research_window": {
                "policy_version": self.research_window_policy_version,
                "opens_at": self.research_window_opens_at.value,
                "closes_at": self.research_window_closes_at.value,
                "timeframe": "5m",
            },
            "venue_reference": self.venue_reference,
            "venue_provenance": self.venue_provenance.value,
        }


# ---------------------------------------------------------------------------
# Venue-day evidence
# ---------------------------------------------------------------------------

_NORMAL = EquityVenueDayClassification.NORMAL_SESSION


@dataclass(frozen=True, slots=True)
class EquityVenueDayEvidence:
    """The venue-day answer an assessment's timing used, frozen without its prose.

    It is the existing EquityVenueDayReference's structured content: the
    venue and date, the classification -- NORMAL_SESSION, a special or
    shortened session, closed, or not stated -- and, for a normal session, the
    common continuous boundary and the UTC offset that placed the timing
    policy's wall-clock times. ``reference`` names the evidence. The
    reference's human-readable reason is not part of it.
    """

    exchange_code: ExchangeCode
    trading_date: date
    classification: EquityVenueDayClassification
    common_opens_at: PointInTime | None
    common_closes_at: PointInTime | None
    utc_offset: timedelta | None
    reference: str

    def __post_init__(self) -> None:
        if not isinstance(self.exchange_code, ExchangeCode):
            _evidence("Venue-day evidence needs an ExchangeCode.")
        if isinstance(self.trading_date, datetime) or not isinstance(self.trading_date, date):
            _evidence("Venue-day evidence needs a plain trading date.")
        if not isinstance(self.classification, EquityVenueDayClassification):
            _evidence("Venue-day evidence needs an EquityVenueDayClassification.")
        if not isinstance(self.reference, str) or not self.reference.strip():
            _evidence("Venue-day evidence needs its reference.")
        bounds = (self.common_opens_at, self.common_closes_at, self.utc_offset)
        if self.classification is _NORMAL:
            if not isinstance(self.common_opens_at, PointInTime) or not isinstance(
                self.common_closes_at, PointInTime
            ):
                _evidence("A normal session's evidence needs both common boundaries.")
            if self.common_opens_at.compare(self.common_closes_at) >= 0:
                _evidence("A normal session's common boundary must open before it closes.")
            offset = self.utc_offset
            if (
                not isinstance(offset, timedelta)
                or offset % _ONE_MINUTE
                or abs(offset) >= timedelta(days=1)
            ):
                _evidence("A normal session's evidence needs a whole-minute UTC offset.")
        elif any(value is not None for value in bounds):
            _evidence(f"{self.classification} evidence carries no boundaries or offset.")

    @classmethod
    def from_reference(cls, reference: EquityVenueDayReference) -> EquityVenueDayEvidence:
        """Freeze the structured content of a venue-day reference."""
        if not isinstance(reference, EquityVenueDayReference):
            raise TypeError("reference must be an EquityVenueDayReference.")
        return cls(
            exchange_code=reference.exchange_code,
            trading_date=reference.trading_date,
            classification=reference.classification,
            common_opens_at=reference.common_opens_at,
            common_closes_at=reference.common_closes_at,
            utc_offset=reference.utc_offset,
            reference=reference.reference,
        )

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> EquityVenueDayEvidence:
        """Rebuild frozen evidence from its document; anything missing or extra fails closed."""
        expected = {
            "exchange_code",
            "trading_date",
            "classification",
            "common_opens_at",
            "common_closes_at",
            "utc_offset_minutes",
            "reference",
        }
        if not isinstance(document, dict) or set(document) != expected:
            _evidence("Venue-day evidence must state exactly its required attributes.")
        try:
            offset = document["utc_offset_minutes"]
            if offset is not None and (isinstance(offset, bool) or not isinstance(offset, int)):
                _evidence("utc_offset_minutes must be an integer or null.")

            def instant(value: object) -> PointInTime | None:
                return None if value is None else PointInTime(value)  # type: ignore[arg-type]

            evidence = cls(
                exchange_code=ExchangeCode(document["exchange_code"]),
                trading_date=date.fromisoformat(document["trading_date"]),
                classification=EquityVenueDayClassification(document["classification"]),
                common_opens_at=instant(document["common_opens_at"]),
                common_closes_at=instant(document["common_closes_at"]),
                utc_offset=None if offset is None else timedelta(minutes=offset),
                reference=document["reference"],
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, InvalidEquityRecommendationEvidenceError):
                raise
            _evidence(f"Venue-day evidence does not decode: {exc}")
        if evidence.document() != document:
            _evidence("Venue-day evidence is not in canonical form.")
        return evidence

    def document(self) -> dict[str, Any]:
        """Return the canonical content."""
        return {
            "exchange_code": self.exchange_code.value,
            "trading_date": self.trading_date.isoformat(),
            "classification": self.classification.value,
            "common_opens_at": None if self.common_opens_at is None else self.common_opens_at.value,
            "common_closes_at": (
                None if self.common_closes_at is None else self.common_closes_at.value
            ),
            "utc_offset_minutes": None
            if self.utc_offset is None
            else self.utc_offset // _ONE_MINUTE,
            "reference": self.reference,
        }


# ---------------------------------------------------------------------------
# Issued-history evidence
# ---------------------------------------------------------------------------


def _issued_order(record: EquityPriorSessionRecommendation) -> tuple[datetime, str]:
    return (_datetime(record.issued_at), record.recommendation_key)


@dataclass(frozen=True, slots=True)
class EquityIssuedHistoryEvidence:
    """The as-of issued-history evidence of one assessment, frozen (EI4-D9-H).

    ``recommendations`` are exactly the records issued at or before ``as_of``
    -- the modeled issuance -- in (issued_at, key) order, each open under
    EI4-D8. ``complete_through`` and ``recording_watermark`` are the observed
    record's completeness instant and recording position, so the same view can
    be requested and verified again. Records issued after ``as_of`` are not
    part of the evidence.
    """

    exchange_code: ExchangeCode
    trading_date: date
    reference: str
    as_of: PointInTime
    complete_through: PointInTime
    recording_watermark: int
    recommendations: tuple[EquityPriorSessionRecommendation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.exchange_code, ExchangeCode):
            _evidence("Issued-history evidence needs an ExchangeCode.")
        if isinstance(self.trading_date, datetime) or not isinstance(self.trading_date, date):
            _evidence("Issued-history evidence needs a plain trading date.")
        if not isinstance(self.reference, str) or not self.reference.strip():
            _evidence("Issued-history evidence needs its reference.")
        for name in ("as_of", "complete_through"):
            if not isinstance(getattr(self, name), PointInTime):
                _evidence(f"Issued-history evidence {name} must be a PointInTime.")
        if self.complete_through.compare(self.as_of) < 0:
            _evidence("Issued-history evidence must be complete through its as-of instant.")
        watermark = self.recording_watermark
        if isinstance(watermark, bool) or not isinstance(watermark, int) or watermark < 0:
            _evidence("Issued-history evidence needs a non-negative recording watermark.")
        records = self.recommendations
        if not isinstance(records, tuple) or not all(
            isinstance(r, EquityPriorSessionRecommendation) for r in records
        ):
            _evidence("Issued-history evidence holds EquityPriorSessionRecommendation records.")
        for record in records:
            if (record.trading_date, record.listing.exchange_code) != (
                self.trading_date,
                self.exchange_code,
            ):
                _evidence("Issued-history evidence holds a record from another session.")
            if record.issued_at.compare(self.as_of) > 0:
                _evidence("Issued-history evidence holds a record issued after its as-of instant.")
            if record.recording_sequence > watermark:
                _evidence("Issued-history evidence holds a record above its recording watermark.")
        if len({r.recommendation_key for r in records}) != len(records) or len(
            {r.recording_sequence for r in records}
        ) != len(records):
            _evidence("Issued-history evidence keys and recording sequences are distinct.")
        if list(records) != sorted(records, key=_issued_order):
            _evidence("Issued-history evidence is in ascending (issued_at, key) order.")

    @classmethod
    def as_of_view(
        cls, history: EquitySessionRecommendationHistory, as_of: PointInTime
    ) -> EquityIssuedHistoryEvidence:
        """Freeze ``history``'s point-in-time view at ``as_of``, which it must cover."""
        if not isinstance(history, EquitySessionRecommendationHistory):
            raise TypeError("history must be an EquitySessionRecommendationHistory.")
        if not history.covers(as_of):
            _evidence("The issued history is not complete through the as-of instant.")
        return cls(
            exchange_code=history.exchange_code,
            trading_date=history.trading_date,
            reference=history.reference,
            as_of=as_of,
            complete_through=history.complete_through,
            recording_watermark=history.recording_watermark,
            recommendations=history.issued_by(as_of),
        )

    def document(self) -> dict[str, Any]:
        """Return the canonical content."""
        return {
            "exchange_code": self.exchange_code.value,
            "trading_date": self.trading_date.isoformat(),
            "reference": self.reference,
            "as_of": self.as_of.value,
            "complete_through": self.complete_through.value,
            "recording_watermark": self.recording_watermark,
            "recommendations": [record.document() for record in self.recommendations],
        }


# ---------------------------------------------------------------------------
# The complete input envelope
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EquityRecommendationInputs:
    """Every input one assessment applied, frozen, and nothing else (EI4-D9-I).

    The envelope is consistent: one listing, date and venue throughout; the
    venue-day evidence is the venue reference the candidate's snapshot used;
    and the issued-history evidence is taken as of the modeled issuance the
    policy set's timing policy derives from the candidate's decision instant.
    """

    candidate: EquityRecommendationCandidateProvenance
    policy_set: EquityResolvedSyntheticPolicySet
    listing_eligibility: EquitySyntheticListingEligibility
    venue_day: EquityVenueDayEvidence
    issued_history: EquityIssuedHistoryEvidence
    modeled_issued_at: PointInTime = field(init=False)
    generation_key: str = field(init=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        for name, kind in (
            ("candidate", EquityRecommendationCandidateProvenance),
            ("policy_set", EquityResolvedSyntheticPolicySet),
            ("listing_eligibility", EquitySyntheticListingEligibility),
            ("venue_day", EquityVenueDayEvidence),
            ("issued_history", EquityIssuedHistoryEvidence),
        ):
            if not isinstance(getattr(self, name), kind):
                _evidence(f"{name} must be an {kind.__name__}.")
        source = self.candidate.source
        listing, day = source.listing, source.trading_date
        if (self.listing_eligibility.listing, self.listing_eligibility.trading_date) != (
            listing,
            day,
        ):
            _evidence("The listing eligibility is for another listing or date.")
        for name in ("venue_day", "issued_history"):
            evidence = getattr(self, name)
            if (evidence.exchange_code, evidence.trading_date) != (listing.exchange_code, day):
                _evidence(f"The {name} evidence is for another venue or date.")
        if self.venue_day.reference != self.candidate.venue_reference:
            _evidence("The venue-day evidence is not the venue reference the snapshot used.")
        issued = equity_modeled_issuance(
            source.decision_as_of, self.policy_set.timing_policy.modeled_issuance_latency
        )
        if self.issued_history.as_of != issued:
            _evidence("The issued history must be taken as of the derived modeled issuance.")
        object.__setattr__(self, "modeled_issued_at", issued)
        object.__setattr__(
            self, "generation_key", equity_recommendation_generation_key(source, self.policy_set)
        )
        object.__setattr__(self, "fingerprint", equity_recommendation_fingerprint(self.document()))

    def versions(self) -> dict[str, EquitySyntheticVersion]:
        """Return the policy-set component versions and the listing eligibility version."""
        reference = self.policy_set.reference
        return {
            **{name: getattr(reference, name) for name in _POLICY_FIELDS},
            "listing_eligibility_version": self.listing_eligibility.version,
        }

    def document(self) -> dict[str, Any]:
        """Return the canonical content."""
        return {
            "format": EQUITY_RECOMMENDATION_INPUTS_FORMAT,
            "candidate": self.candidate.document(),
            "policy_set": self.policy_set.document(),
            "policy_set_fingerprint": self.policy_set.fingerprint,
            "listing_eligibility": self.listing_eligibility.document(),
            "venue_day": self.venue_day.document(),
            "issued_history": self.issued_history.document(),
            "modeled_issued_at": self.modeled_issued_at.value,
        }
