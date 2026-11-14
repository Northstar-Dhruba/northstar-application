"""The issuance authority's invariants as pure functions (EI-5D, ADR-024 sections 6 and 7).

The EI-5 design (section 8.2) fixes what an issuance authority must enforce.
These functions state it once, over records already loaded, so every writer
applies the same rules: EI-5D's labelled in-memory test authority, and EI-6's
SQLite adapter inside its own serialized transaction. They decide; they never
store, lock or read a clock. Atomicity is the writer's: the decision and the
insert must happen in one serialized write (EI-6: ``BEGIN IMMEDIATE``).

1. One authority-wide ``recording_sequence``, shared by issuances and
   declarations; assigned at commit; unique, monotone, never reused.
2. One issuance per issuance identity, per generation key and per
   (strategy, listing, trading date) (EI5-D1).
3. No backdating: an issuance whose ``issued_at`` is at or before the
   session's greatest declared ``complete_through`` is refused (EI5-D4).
4. A declaration below the session's current maximum is a regression; an
   equal one is idempotent; a greater one is recorded.
5. The view at watermark ``W`` lists the issuances recorded at or below ``W``
   whose ``issued_at`` is at or before ``CT(W)`` -- the greatest
   ``complete_through`` declared at or below ``W`` -- in (issued_at, key)
   order.
6. Stability: for ``W1 < W2``, the view at ``W2`` restricted to
   ``issued_at <= CT(W1)`` equals the view at ``W1``; it follows from 3.
7. No declaration at or below ``W`` means unknown (None), never an empty
   tuple; a declaration with no issuance is an explicit empty tuple.
8. Records are append-only.

Rule 2 is checked in this order: the same identity first (EQUIVALENT, or
STORED_ISSUANCE_INVALID when the stored record does not re-derive), then the
generation key, then the session. An identical retry is therefore EQUIVALENT
even after the session has been declared further, while any new issuance
into a declared-complete interval is refused (rule 3).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime

from northstar_core.foundation.value_objects import ExchangeCode, PointInTime

from northstar_application.application_services.equity_recommendation_issuance_records import (
    EQUITY_RECOMMENDATION_ISSUANCE_AUTHORITY_REFERENCE,
    EquityIssuanceCompletenessDeclaration,
    EquityIssuedRecommendation,
    EquityRecordedCompletenessDeclaration,
    EquityRecordedIssuance,
)
from northstar_application.ports.equity_recommendation_issuance import (
    EquityIssuanceDeclarationOutcome,
    EquityRecommendationIssuanceOutcome,
)
from northstar_application.ports.equity_recommendation_references import (
    EquityPriorSessionRecommendation,
    EquitySessionRecommendationHistory,
)

_I = EquityRecommendationIssuanceOutcome
_D = EquityIssuanceDeclarationOutcome


def _instant(value: PointInTime) -> datetime:
    return datetime.fromisoformat(value.value.replace("Z", "+00:00"))


def _in_session(
    declaration: EquityRecordedCompletenessDeclaration, exchange_code: ExchangeCode, day: date
) -> bool:
    found = declaration.declaration
    return (found.exchange_code, found.trading_date) == (exchange_code, day)


def equity_declared_complete_through(
    declarations: Iterable[EquityRecordedCompletenessDeclaration],
    exchange_code: ExchangeCode,
    trading_date: date,
    watermark: int | None = None,
) -> PointInTime | None:
    """Return the session's greatest declared ``complete_through`` at or below ``watermark``.

    ``watermark`` None means every recorded declaration. None when the session
    has no declaration there: its completeness is unknown.
    """
    greatest: PointInTime | None = None
    for recorded in declarations:
        if not _in_session(recorded, exchange_code, trading_date):
            continue
        if watermark is not None and recorded.recording_sequence > watermark:
            continue
        through = recorded.declaration.complete_through
        if greatest is None or through.compare(greatest) > 0:
            greatest = through
    return greatest


def equity_issuance_session_view(
    exchange_code: ExchangeCode,
    trading_date: date,
    watermark: int,
    issuances: Iterable[EquityPriorSessionRecommendation],
    declarations: Iterable[EquityRecordedCompletenessDeclaration],
    reference: str = EQUITY_RECOMMENDATION_ISSUANCE_AUTHORITY_REFERENCE,
) -> EquitySessionRecommendationHistory | None:
    """Return the session's issued-history view at ``watermark`` (rules 5 and 7)."""
    if isinstance(watermark, bool) or not isinstance(watermark, int) or watermark < 0:
        raise ValueError("watermark must be a non-negative integer.")
    complete_through = equity_declared_complete_through(
        declarations, exchange_code, trading_date, watermark
    )
    if complete_through is None:
        return None
    listed = sorted(
        (
            record
            for record in issuances
            if record.trading_date == trading_date
            and record.listing.exchange_code == exchange_code
            and record.recording_sequence <= watermark
            and record.issued_at.compare(complete_through) <= 0
        ),
        key=lambda record: (_instant(record.issued_at), record.recommendation_key),
    )
    return EquitySessionRecommendationHistory(
        exchange_code, trading_date, tuple(listed), complete_through, watermark, reference
    )


def equity_issuance_is_backdated(
    issued_at: PointInTime, declared_complete_through: PointInTime | None
) -> bool:
    """Return whether an issuance at ``issued_at`` would be backdated (rule 3)."""
    return declared_complete_through is not None and (
        issued_at.compare(declared_complete_through) <= 0
    )


def equity_issuance_write_decision(
    offered: EquityIssuedRecommendation,
    issuances: Iterable[EquityRecordedIssuance],
    declarations: Iterable[EquityRecordedCompletenessDeclaration],
) -> tuple[EquityRecommendationIssuanceOutcome, EquityRecordedIssuance | None]:
    """Decide one issuance write against the records already held (rules 2 and 3).

    Returns the outcome and the stored record it concerns: the stored equal
    record for EQUIVALENT, the stored conflicting record for the two
    ``*_ALREADY_ISSUED`` outcomes, and None for ISSUED (the writer then
    records ``offered``), BACKDATED_ISSUANCE and STORED_ISSUANCE_INVALID.
    """
    held = tuple(issuances)
    for recorded in held:
        if recorded.recommendation.issuance_id == offered.issuance_id:
            if recorded.recommendation == offered:
                return _I.EQUIVALENT, recorded
            return _I.STORED_ISSUANCE_INVALID, None
    for recorded in held:
        if recorded.recommendation.generation_key == offered.generation_key:
            return _I.GENERATION_ALREADY_ISSUED, recorded
    session = (offered.strategy, offered.listing, offered.trading_date)
    for recorded in held:
        stored = recorded.recommendation
        if (stored.strategy, stored.listing, stored.trading_date) == session:
            return _I.SESSION_STRATEGY_ALREADY_ISSUED, recorded
    declared = equity_declared_complete_through(
        declarations, offered.exchange_code, offered.trading_date
    )
    if equity_issuance_is_backdated(offered.issued_at, declared):
        return _I.BACKDATED_ISSUANCE, None
    return _I.ISSUED, None


def equity_declaration_write_decision(
    offered: EquityIssuanceCompletenessDeclaration,
    declarations: Iterable[EquityRecordedCompletenessDeclaration],
) -> tuple[EquityIssuanceDeclarationOutcome, EquityRecordedCompletenessDeclaration | None]:
    """Decide one declaration write (rule 4).

    Returns DECLARED with None (the writer records ``offered``),
    DECLARATION_EQUIVALENT with the stored equal declaration, or
    COMPLETENESS_REGRESSION with the stored declaration of the current maximum.
    """
    current: EquityRecordedCompletenessDeclaration | None = None
    for recorded in declarations:
        if not _in_session(recorded, offered.exchange_code, offered.trading_date):
            continue
        through = recorded.declaration.complete_through
        if current is None or through.compare(current.declaration.complete_through) > 0:
            current = recorded
    if current is None:
        return _D.DECLARED, None
    relation = offered.complete_through.compare(current.declaration.complete_through)
    if relation < 0:
        return _D.COMPLETENESS_REGRESSION, current
    if relation == 0:
        return _D.DECLARATION_EQUIVALENT, current
    return _D.DECLARED, None
