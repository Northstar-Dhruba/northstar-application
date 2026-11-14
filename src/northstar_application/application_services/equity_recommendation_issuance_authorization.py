"""Content-bound authorizations for the issuance authority (EI-5D, ADR-024 section 4).

ADR-024 applies the EI4-D11 verified-write pattern (ADR-023) to the issuance
authority from the start. This module is its Application-owned authorization
layer:

- ``VerifiedEquityRecommendationIssuance`` authorizes one exact MODELED issued
  recommendation, derived from one disposition that the stored-generation
  verifier re-verified at levels A to C;
- ``VerifiedEquityIssuanceCompletenessDeclaration`` authorizes one exact
  session completeness declaration, checked against the session's venue day.

Issuance
--------
Only ``IssueEquityTradeRecommendationUseCase`` and
``DeclareEquitySessionIssuanceCompletenessUseCase`` issue them, after their
checks pass, through this module's private issuing functions. Issuance
registers the object in a module-private registry keyed by identity; the
validators accept only a registered object. An authorization built through
the public constructor, copied, or made by ``dataclasses.replace`` is not
registered and is refused.

Exact content binding
---------------------
At construction an authorization captures the exact content it covers as
canonical text (``content()``). The public validators -- which a writer calls
before opening anything -- re-derive that content and refuse any difference:

- issuance: the issued record must re-derive exactly from the verified
  disposition, which must be a PROPOSED evaluator-``/2`` disposition, and must
  be MODELED; the identity must be the record's own;
- declaration: the declaration must be for the venue day's own exchange and
  date, on a NORMAL_SESSION, with ``complete_through`` inside the venue's
  local calendar day.

The generation key, an issuance identity or a "verified" flag alone never
suffice. An authorization is never single-use: it covers fixed content, and
writing it again is EQUIVALENT (or DECLARATION_EQUIVALENT) under first-writer
immutability.

Trust boundary (ADR-024, section 14)
------------------------------------
This stops accidental and ordinary unsupported writes within Northstar's
supported interfaces. It is not cryptographically unforgeable: privileged
Python reaching into the registry, direct SQL or file access can bypass it.
Nothing here is persisted, and nothing reads a clock or calls out.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from typing import Any, NoReturn

from northstar_core.recommendations.equity import EquityIssuanceKind

from northstar_application.application_services.equity_recommendation_generation_evidence import (
    EQUITY_RECOMMENDATION_EVALUATOR_V2,
)
from northstar_application.application_services.equity_recommendation_generation_records import (
    EquityRecommendationDispositionKind,
)
from northstar_application.application_services.equity_recommendation_issuance_records import (
    EquityIssuanceCompletenessDeclaration,
    EquityIssuedRecommendation,
    InvalidEquityIssuedRecommendationError,
    equity_issued_recommendation,
)
from northstar_application.application_services.verify_equity_recommendation_generation import (
    EquityVerifiedRecommendationGeneration,
)
from northstar_application.ports.equity_recommendation_issuance import (
    UnverifiedEquityRecommendationIssuanceWriteError,
)
from northstar_application.ports.equity_research_window_reference import (
    EquityVenueDayClassification,
    EquityVenueDayReference,
)

# Authorizations issued by the two use cases in this process, keyed by
# identity. Weak: an authorization no one holds can no longer be written.
_ISSUED_ISSUANCE_WRITES: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()


class InvalidEquityIssuanceCompletenessDeclarationError(ValueError):
    """Raised when a completeness declaration is outside its session's venue day."""


def _refuse(message: str) -> NoReturn:
    raise UnverifiedEquityRecommendationIssuanceWriteError(message)


def _instant(value: Any) -> datetime:
    return datetime.fromisoformat(value.value.replace("Z", "+00:00"))


def equity_declaration_venue_day_problem(
    declaration: EquityIssuanceCompletenessDeclaration, venue_day: EquityVenueDayReference
) -> str | None:
    """Return why ``declaration`` does not lie within ``venue_day``, or None when it does.

    The venue day is the session's own exchange and trading date, on a
    NORMAL_SESSION, and ``complete_through`` must fall inside that date's
    local calendar day at the venue's UTC offset (ADR-024, section 7).
    """
    if not isinstance(declaration, EquityIssuanceCompletenessDeclaration):
        return "the declaration is not an EquityIssuanceCompletenessDeclaration"
    if not isinstance(venue_day, EquityVenueDayReference):
        return "the venue day is not an EquityVenueDayReference"
    if (venue_day.exchange_code, venue_day.trading_date) != (
        declaration.exchange_code,
        declaration.trading_date,
    ):
        return "the venue-day evidence is for another exchange or date"
    if venue_day.classification is not EquityVenueDayClassification.NORMAL_SESSION:
        return f"the venue day is {venue_day.classification}, not a normal session"
    offset = venue_day.utc_offset
    if not isinstance(offset, timedelta):
        return "the venue day states no UTC offset"
    zone = timezone(offset)
    starts = datetime.combine(declaration.trading_date, time(0), tzinfo=zone)
    ends = starts + timedelta(days=1)
    through = _instant(declaration.complete_through)
    if not starts <= through < ends:
        return (
            f"complete_through {declaration.complete_through} is outside the venue's "
            f"local day {declaration.trading_date.isoformat()}"
        )
    return None


def _check_issuance(verified: object, issuance: object) -> EquityIssuedRecommendation:
    """Refuse an issuance whose record is not exactly the verified disposition's."""
    if not isinstance(verified, EquityVerifiedRecommendationGeneration):
        _refuse("An issuance needs the verifier's EquityVerifiedRecommendationGeneration.")
    if not isinstance(issuance, EquityIssuedRecommendation):
        _refuse("An issuance needs an EquityIssuedRecommendation.")
    disposition = verified.disposition
    if disposition.kind is not EquityRecommendationDispositionKind.PROPOSED:
        _refuse("Only a PROPOSED disposition is issuable.")
    if disposition.evaluator_version != EQUITY_RECOMMENDATION_EVALUATOR_V2:
        _refuse("Only an evaluator-/2 disposition is issuable (ADR-025, N7).")
    if verified.replayed.assessment_fingerprint != disposition.assessment_fingerprint:
        _refuse("The verified replay is not this disposition's assessment.")
    if issuance.issuance_kind is not EquityIssuanceKind.MODELED:
        _refuse("Only MODELED issuance is authorized.")
    try:
        derived = equity_issued_recommendation(disposition)
    except InvalidEquityIssuedRecommendationError as exc:
        _refuse(f"The disposition does not derive an issued recommendation: {exc}")
    if derived != issuance:
        _refuse("The issued recommendation is not the one the verified disposition derives.")
    return issuance


def _check_declaration(
    declaration: object, venue_day: object
) -> EquityIssuanceCompletenessDeclaration:
    if not isinstance(declaration, EquityIssuanceCompletenessDeclaration):
        _refuse("A declaration write needs an EquityIssuanceCompletenessDeclaration.")
    if not isinstance(venue_day, EquityVenueDayReference):
        _refuse("A declaration write needs the session's EquityVenueDayReference.")
    problem = equity_declaration_venue_day_problem(declaration, venue_day)
    if problem is not None:
        _refuse(f"The declaration is not within its venue day: {problem}.")
    return declaration


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class VerifiedEquityRecommendationIssuance:
    """The issuer's authorization to record exactly one MODELED issued recommendation.

    ``covered`` is the exact content captured at construction. Equality is
    identity: two authorizations are never interchangeable.
    """

    verified: EquityVerifiedRecommendationGeneration
    issuance: EquityIssuedRecommendation
    covered: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        issuance = _check_issuance(self.verified, self.issuance)
        object.__setattr__(self, "covered", issuance.content())


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class VerifiedEquityIssuanceCompletenessDeclaration:
    """The declarer's authorization to record exactly one completeness declaration."""

    declaration: EquityIssuanceCompletenessDeclaration
    venue_day: EquityVenueDayReference
    covered: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        declaration = _check_declaration(self.declaration, self.venue_day)
        object.__setattr__(self, "covered", declaration.content())


def _issue_issuance_write(
    verified: EquityVerifiedRecommendationGeneration, issuance: EquityIssuedRecommendation
) -> VerifiedEquityRecommendationIssuance:
    """Private to the issuance use case: issue after its checks passed."""
    authorization = VerifiedEquityRecommendationIssuance(verified, issuance)
    _ISSUED_ISSUANCE_WRITES[id(authorization)] = authorization
    return authorization


def _issue_declaration_write(
    declaration: EquityIssuanceCompletenessDeclaration, venue_day: EquityVenueDayReference
) -> VerifiedEquityIssuanceCompletenessDeclaration:
    """Private to the declaration use case: issue after its checks passed."""
    authorization = VerifiedEquityIssuanceCompletenessDeclaration(declaration, venue_day)
    _ISSUED_ISSUANCE_WRITES[id(authorization)] = authorization
    return authorization


def _require_issued(authorization: object, kind: type) -> None:
    if type(authorization) is not kind:
        _refuse(
            f"A {kind.__name__} issued by its use case is required, not "
            f"{type(authorization).__name__}."
        )
    if _ISSUED_ISSUANCE_WRITES.get(id(authorization)) is not authorization:
        _refuse(f"This {kind.__name__} was not issued by its use case.")


def require_verified_equity_recommendation_issuance(
    authorization: object,
) -> VerifiedEquityRecommendationIssuance:
    """Return ``authorization`` when it authorizes exactly its issuance; refuse otherwise.

    A writer calls this before opening anything. It raises
    UnverifiedEquityRecommendationIssuanceWriteError for a missing, foreign or
    unissued authorization, or one whose content no longer matches.
    """
    _require_issued(authorization, VerifiedEquityRecommendationIssuance)
    assert isinstance(authorization, VerifiedEquityRecommendationIssuance)
    issuance = _check_issuance(authorization.verified, authorization.issuance)
    if issuance.content() != authorization.covered:
        _refuse("The issuance's content differs from the content this authorization covers.")
    return authorization


def require_verified_equity_issuance_completeness_declaration(
    authorization: object,
) -> VerifiedEquityIssuanceCompletenessDeclaration:
    """Return ``authorization`` when it authorizes exactly its declaration; refuse otherwise."""
    _require_issued(authorization, VerifiedEquityIssuanceCompletenessDeclaration)
    assert isinstance(authorization, VerifiedEquityIssuanceCompletenessDeclaration)
    declaration = _check_declaration(authorization.declaration, authorization.venue_day)
    if declaration.content() != authorization.covered:
        _refuse("The declaration differs from the declaration this authorization covers.")
    return authorization
