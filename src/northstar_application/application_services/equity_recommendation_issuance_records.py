"""MODELED recommendation issuance records and identity (EI-5D, ADR-024, EI5-D1 to EI5-D4).

An issued recommendation is a separate record from the EI-4D generation
disposition it comes from (ADR-024, section 2). Generating a proposal never
issues it, and issuing it never changes it. Every field of
``EquityIssuedRecommendation`` is copied from the re-verified stored proposal
or derived from it: nothing is supplied by a caller, so no caller can move
``issued_at`` or ``eligible_from``.

Identity (EI5-D3, C-1)
----------------------
``eiris1-sha256:`` followed by the SHA-256 digest of the length-prefixed
parts, in order -- exactly the encoding of the ``eirk1`` generation key:

1. ``northstar.equity-recommendation-issuance-key/1``;
2. the generation key text (``eirk1-sha256:...``);
3. the proposal fingerprint text (``eirp1-sha256:...``);
4. the issuance kind value (``MODELED``).

No clock, sequence, path or process value enters it, so it is stable across
processes and restarts. It is distinct from ``eirk1`` by construction and from
the frozen-inputs fingerprint ``eiri1``. No content fingerprint is added: the
record is a pure function of the verified proposal and the identity's parts,
so a stored record that does not re-derive is invalid, never equivalent.

The authority reference (D5D-3)
-------------------------------
``EQUITY_RECOMMENDATION_ISSUANCE_AUTHORITY_REFERENCE`` is the fixed, versioned
``reference`` of every authoritative issued-history view. Version-2 frozen
inputs embed it, so it is identity-bearing (``eiri1``, ``eira1``); changing it
needs a new version.

Not actionable
--------------
An issued recommendation is MODELED and synthetic. It carries no actionability
flag that could be set, no quantity committed to any account, and no order,
fill, position, P&L or exit price. Every control EI-4C reported as not
assessed stays listed as not assessed. ``recording_sequence`` is assigned by
the authority at commit and is not part of the record or its identity.

Nothing here reads a clock, writes anything or calls out.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime
from northstar_core.recommendations.equity import (
    EquityIssuanceKind,
    EquityRecommendationDirection,
    EquityRecommendationLevels,
    EquityRecommendationTiming,
    EquitySyntheticVersion,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services.equity_recommendation_generation_evidence import (
    _length_prefixed,
    canonical_equity_recommendation_json,
    is_equity_recommendation_generation_key,
)
from northstar_application.application_services.equity_recommendation_generation_records import (
    EquityRecommendationDispositionKind,
    EquityRecommendationGenerationDisposition,
)
from northstar_application.application_services.equity_trade_recommendation_eligibility import (
    EquityRecommendationCheck,
)
from northstar_application.ports.equity_recommendation_references import (
    EquityPriorSessionRecommendation,
)

EQUITY_RECOMMENDATION_ISSUANCE_KEY_NAMESPACE = "northstar.equity-recommendation-issuance-key/1"
EQUITY_RECOMMENDATION_ISSUANCE_PREFIX = "eiris1-sha256:"
EQUITY_RECOMMENDATION_ISSUANCE_AUTHORITY_REFERENCE = (
    "northstar.equity-recommendation-issuance-authority/1"
)

_PROPOSAL_PREFIX = "eirp1-sha256:"
_ASSESSMENT_PREFIX = "eira1-sha256:"
_HEX = frozenset("0123456789abcdef")


class InvalidEquityIssuedRecommendationError(ValueError):
    """Raised when an issued recommendation, identity or declaration is malformed."""


def _invalid(message: str) -> InvalidEquityIssuedRecommendationError:
    return InvalidEquityIssuedRecommendationError(message)


def _is_fingerprint(value: object, prefix: str) -> bool:
    if not isinstance(value, str) or not value.startswith(prefix):
        return False
    digest = value[len(prefix) :]
    return len(digest) == 64 and set(digest) <= _HEX


def is_equity_recommendation_issuance_identity(value: object) -> bool:
    """Return whether ``value`` has the shape of an ``eiris1`` issuance identity."""
    return _is_fingerprint(value, EQUITY_RECOMMENDATION_ISSUANCE_PREFIX)


def _identity_from_parts(parts: tuple[str, ...]) -> str:
    digest = hashlib.sha256(_length_prefixed(parts)).hexdigest()
    return EQUITY_RECOMMENDATION_ISSUANCE_PREFIX + digest


def equity_recommendation_issuance_identity(
    generation_key: str, proposal_fingerprint: str, issuance_kind: EquityIssuanceKind
) -> str:
    """Return the ``eiris1`` issuance identity of these parts (EI5-D3, C-1)."""
    if not is_equity_recommendation_generation_key(generation_key):
        raise _invalid("An issuance identity needs an eirk1 generation key.")
    if not _is_fingerprint(proposal_fingerprint, _PROPOSAL_PREFIX):
        raise _invalid("An issuance identity needs an eirp1 proposal fingerprint.")
    if not isinstance(issuance_kind, EquityIssuanceKind):
        raise _invalid("An issuance identity needs an EquityIssuanceKind.")
    return _identity_from_parts(
        (
            EQUITY_RECOMMENDATION_ISSUANCE_KEY_NAMESPACE,
            generation_key,
            proposal_fingerprint,
            issuance_kind.value,
        )
    )


def _plain_date(value: object, subject: str) -> None:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise _invalid(f"{subject} trading_date must be a plain date.")


@dataclass(frozen=True, slots=True)
class EquityIssuedRecommendation:
    """One MODELED issued recommendation, derived from one verified stored proposal.

    See the module docstring. ``recording_sequence`` is not a field: the
    authority assigns it at commit (``EquityRecordedIssuance``).
    """

    issuance_id: str
    generation_key: str
    proposal_fingerprint: str
    assessment_fingerprint: str
    issuance_kind: EquityIssuanceKind
    strategy: StrategyIdentity
    listing: ListingReference
    trading_date: date
    decision_as_of: PointInTime
    issued_at: PointInTime
    eligible_from: PointInTime
    latest_entry_bar_opens_at: PointInTime
    modeled_exit_at: PointInTime
    timing_policy_version: EquitySyntheticVersion
    direction: EquityRecommendationDirection
    levels: EquityRecommendationLevels
    not_assessed: tuple[EquityRecommendationCheck, ...]

    def __post_init__(self) -> None:
        subject = "EquityIssuedRecommendation"
        if self.issuance_kind is not EquityIssuanceKind.MODELED:
            raise _invalid(f"{subject} is MODELED only; ACTUAL issuance is not authorized.")
        if not is_equity_recommendation_generation_key(self.generation_key):
            raise _invalid(f"{subject} needs an eirk1 generation key.")
        if not _is_fingerprint(self.proposal_fingerprint, _PROPOSAL_PREFIX):
            raise _invalid(f"{subject} needs an eirp1 proposal fingerprint.")
        if not _is_fingerprint(self.assessment_fingerprint, _ASSESSMENT_PREFIX):
            raise _invalid(f"{subject} needs an eira1 assessment fingerprint.")
        if not isinstance(self.strategy, StrategyIdentity):
            raise _invalid(f"{subject} strategy must be a StrategyIdentity.")
        if not isinstance(self.listing, ListingReference):
            raise _invalid(f"{subject} listing must be a ListingReference.")
        _plain_date(self.trading_date, subject)
        for name in (
            "decision_as_of",
            "issued_at",
            "eligible_from",
            "latest_entry_bar_opens_at",
            "modeled_exit_at",
        ):
            if not isinstance(getattr(self, name), PointInTime):
                raise _invalid(f"{subject} {name} must be a PointInTime.")
        # The timing order is Core's EI4-D1 rule: decision <= issuance <= eligibility.
        try:
            EquityRecommendationTiming(
                self.decision_as_of, self.issuance_kind, self.issued_at, self.eligible_from
            )
        except ValueError as exc:
            raise _invalid(f"{subject} timing is out of order: {exc}") from exc
        if self.eligible_from.compare(self.latest_entry_bar_opens_at) > 0:
            raise _invalid(f"{subject} eligibility begins after the latest entry bar opening.")
        if self.latest_entry_bar_opens_at.compare(self.modeled_exit_at) >= 0:
            raise _invalid(f"{subject} latest entry bar must open before the modeled exit.")
        if not isinstance(self.timing_policy_version, EquitySyntheticVersion):
            raise _invalid(f"{subject} timing_policy_version must be an EquitySyntheticVersion.")
        if not isinstance(self.direction, EquityRecommendationDirection):
            raise _invalid(f"{subject} direction must be an EquityRecommendationDirection.")
        if not isinstance(self.levels, EquityRecommendationLevels):
            raise _invalid(f"{subject} levels must be EquityRecommendationLevels.")
        if (
            not isinstance(self.not_assessed, tuple)
            or not self.not_assessed
            or not all(isinstance(c, EquityRecommendationCheck) for c in self.not_assessed)
        ):
            raise _invalid(f"{subject} lists every control that is not assessed.")
        expected = equity_recommendation_issuance_identity(
            self.generation_key, self.proposal_fingerprint, self.issuance_kind
        )
        if self.issuance_id != expected:
            raise _invalid(f"{subject} issuance_id does not re-derive from its parts.")

    @property
    def exchange_code(self) -> ExchangeCode:
        """Return the venue of the issued recommendation's session."""
        return self.listing.exchange_code

    @property
    def is_synthetic(self) -> bool:
        """Return True: every input is synthetic (EI4-D4)."""
        return True

    @property
    def is_actionable(self) -> bool:
        """Return False: a MODELED issued recommendation is never actionable (ADR-024, 3)."""
        return False

    def history_record(self, recording_sequence: int) -> EquityPriorSessionRecommendation:
        """Return this issuance's projection into the existing history contract."""
        return EquityPriorSessionRecommendation(
            self.issuance_id,
            self.strategy,
            self.listing,
            self.trading_date,
            self.issuance_kind,
            self.issued_at,
            recording_sequence,
        )

    def content(self) -> str:
        """Return the record's exact content as canonical text, for content binding.

        It is not a fingerprint and is never hashed into an identity.
        """
        document: dict[str, Any] = {
            "issuance_id": self.issuance_id,
            "generation_key": self.generation_key,
            "proposal_fingerprint": self.proposal_fingerprint,
            "assessment_fingerprint": self.assessment_fingerprint,
            "issuance_kind": self.issuance_kind.value,
            "strategy": self.strategy.identity,
            "listing": {
                "symbol": self.listing.symbol.value,
                "exchange_code": self.listing.exchange_code.value,
            },
            "trading_date": self.trading_date.isoformat(),
            "decision_as_of": self.decision_as_of.value,
            "issued_at": self.issued_at.value,
            "eligible_from": self.eligible_from.value,
            "latest_entry_bar_opens_at": self.latest_entry_bar_opens_at.value,
            "modeled_exit_at": self.modeled_exit_at.value,
            "timing_policy_version": self.timing_policy_version.value,
            "direction": self.direction.value,
            "levels": self.levels.document(),
            "not_assessed": [check.value for check in self.not_assessed],
        }
        return canonical_equity_recommendation_json(document)


def equity_issued_recommendation(
    disposition: EquityRecommendationGenerationDisposition,
) -> EquityIssuedRecommendation:
    """Derive the MODELED issued recommendation of a PROPOSED disposition.

    Pure: every field is copied from the disposition's own verified proposal
    and candidate. The caller re-verifies the disposition first; this
    function never establishes that it is issuable.
    """
    if not isinstance(disposition, EquityRecommendationGenerationDisposition):
        raise TypeError("disposition must be an EquityRecommendationGenerationDisposition.")
    proposal = disposition.proposal
    if disposition.kind is not EquityRecommendationDispositionKind.PROPOSED or proposal is None:
        raise _invalid("Only a PROPOSED disposition has a proposal to issue.")
    source = disposition.candidate.source
    window = proposal.entry_window
    timing = window.timing
    kind = EquityIssuanceKind.MODELED
    assert disposition.proposal_fingerprint is not None
    return EquityIssuedRecommendation(
        issuance_id=equity_recommendation_issuance_identity(
            disposition.generation_key, disposition.proposal_fingerprint, kind
        ),
        generation_key=disposition.generation_key,
        proposal_fingerprint=disposition.proposal_fingerprint,
        assessment_fingerprint=disposition.assessment_fingerprint,
        issuance_kind=kind,
        strategy=source.strategy,
        listing=source.listing,
        trading_date=source.trading_date,
        decision_as_of=timing.decision_as_of,
        issued_at=timing.issued_at,
        eligible_from=timing.eligible_from,
        latest_entry_bar_opens_at=window.latest_entry_bar_opens_at,
        modeled_exit_at=window.modeled_exit_at,
        timing_policy_version=window.timing_policy_version,
        direction=proposal.definition.direction,
        levels=proposal.definition.levels,
        not_assessed=proposal.not_assessed,
    )


def _sequence(value: object, subject: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _invalid(f"{subject} recording_sequence must be an integer of at least 1.")


@dataclass(frozen=True, slots=True)
class EquityRecordedIssuance:
    """An issued recommendation as the authority recorded it, at its recording sequence."""

    recommendation: EquityIssuedRecommendation
    recording_sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.recommendation, EquityIssuedRecommendation):
            raise _invalid("A recorded issuance holds an EquityIssuedRecommendation.")
        _sequence(self.recording_sequence, "EquityRecordedIssuance")

    def history_record(self) -> EquityPriorSessionRecommendation:
        """Return the existing history contract's record of this issuance."""
        return self.recommendation.history_record(self.recording_sequence)


@dataclass(frozen=True, slots=True)
class EquityIssuanceCompletenessDeclaration:
    """ "Exchange ``X``, trading date ``D`` is complete through instant ``T``" (EI5-D4).

    An append-only fact the authority records with a recording sequence. The
    caller never supplies the sequence or a watermark.
    """

    exchange_code: ExchangeCode
    trading_date: date
    complete_through: PointInTime

    def __post_init__(self) -> None:
        subject = "EquityIssuanceCompletenessDeclaration"
        if not isinstance(self.exchange_code, ExchangeCode):
            raise _invalid(f"{subject} exchange_code must be an ExchangeCode.")
        _plain_date(self.trading_date, subject)
        if not isinstance(self.complete_through, PointInTime):
            raise _invalid(f"{subject} complete_through must be a PointInTime.")

    def content(self) -> str:
        """Return the declaration's exact content as canonical text, for content binding."""
        return canonical_equity_recommendation_json(
            {
                "exchange_code": self.exchange_code.value,
                "trading_date": self.trading_date.isoformat(),
                "complete_through": self.complete_through.value,
            }
        )


@dataclass(frozen=True, slots=True)
class EquityRecordedCompletenessDeclaration:
    """A completeness declaration as the authority recorded it."""

    declaration: EquityIssuanceCompletenessDeclaration
    recording_sequence: int = field()

    def __post_init__(self) -> None:
        if not isinstance(self.declaration, EquityIssuanceCompletenessDeclaration):
            raise _invalid("A recorded declaration holds an EquityIssuanceCompletenessDeclaration.")
        _sequence(self.recording_sequence, "EquityRecordedCompletenessDeclaration")
