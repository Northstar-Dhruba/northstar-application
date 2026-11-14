"""Canonical lifecycle-event documents and verified rehydration (EI-6B design, sections 3 and 8.6).

Format ``northstar.equity-recommendation-lifecycle-event/1`` (D6B-4), owned by
the Application. One document is exactly one Core ``EquityLifecycleEvent``: a
JSON object with exactly 14 keys -- ``format`` and the event's 13 fields --
encoded by ``canonical_equity_recommendation_json`` (sorted keys, no
whitespace, no floats). Evidence objects have exactly ``reference``,
``opens_at``, ``closes_at`` and ``observations``; deficiency objects exactly
``opens_at``, ``closes_at`` and ``kind``. Instants are canonical
``PointInTime`` text. The issuance identity is not in the document: Core events
carry none, and the binding is the store record and the append authorization.

Strict decoding refuses malformed JSON, another format (unsupported), a
missing or extra key, a wrong JSON type (no boolean for an integer), an
unknown enum value, non-canonical instant text, an event shape Core refuses,
and any document that does not re-encode to exactly its own text. Nothing is
normalized. Order, continuity, window and terminality across events are Core
replay's, not decoding's.

There is no content fingerprint (D6B-4): integrity rests on strict canonical
validation, Core construction and replay, the store's constraints, parent
provenance and the single-host trust model. None of it is cryptographic.

``equity_recorded_lifecycle`` is the one rehydration every store uses:
documents are decoded strictly (``EVENT``) and the history replayed by Core
with the authoritative window (``HISTORY``).

Nothing here reads a clock, stores anything or calls out.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

from northstar_core.foundation.value_objects import PointInTime
from northstar_core.recommendations.equity.lifecycle import (
    EquityEvidenceDeficiency,
    EquityEvidenceDeficiencyKind,
    EquityIntrabarOrdering,
    EquityLifecycleCancellationReason,
    EquityLifecycleEvent,
    EquityLifecycleEventKind,
    EquityLifecycleEvidence,
    EquityLifecycleLevel,
    EquityLifecycleState,
    EquityLifecycleWindow,
    EquityRecommendationLifecycle,
    EquityRevisionMateriality,
    InvalidEquityRecommendationLifecycleError,
)

from northstar_application.application_services.equity_recommendation_generation_evidence import (
    canonical_equity_recommendation_json,
)
from northstar_application.application_services.equity_recommendation_issuance_records import (
    EquityRecordedIssuance,
)
from northstar_application.ports.equity_recommendation_lifecycle_store import (
    EquityRecordedLifecycle,
    InvalidStoredEquityRecommendationLifecycleError,
)

EQUITY_RECOMMENDATION_LIFECYCLE_EVENT_FORMAT = "northstar.equity-recommendation-lifecycle-event/1"

_EVENT_KEYS = frozenset(
    {
        "format",
        "sequence",
        "kind",
        "from_state",
        "to_state",
        "evaluated_at",
        "evidence",
        "touched",
        "ordering",
        "reason",
        "deficiencies",
        "cited_sequence",
        "superseded_evidence",
        "materiality",
    }
)
_EVIDENCE_KEYS = frozenset({"reference", "opens_at", "closes_at", "observations"})
_DEFICIENCY_KEYS = frozenset({"opens_at", "closes_at", "kind"})
_DECODE_ERRORS = (KeyError, TypeError, ValueError, AttributeError)


class InvalidEquityLifecycleEventDocumentError(ValueError):
    """Raised when a lifecycle-event document is malformed, non-canonical or not a Core event."""


class IncompatibleEquityLifecycleEventDocumentError(InvalidEquityLifecycleEventDocumentError):
    """Raised when a lifecycle-event document uses a format this code does not support."""


def _bad(message: str) -> InvalidEquityLifecycleEventDocumentError:
    return InvalidEquityLifecycleEventDocumentError(message)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _evidence_document(evidence: EquityLifecycleEvidence) -> dict[str, Any]:
    return {
        "reference": evidence.reference,
        "opens_at": evidence.opens_at.value,
        "closes_at": evidence.closes_at.value,
        "observations": list(evidence.observations),
    }


def equity_lifecycle_event_document(event: EquityLifecycleEvent) -> str:
    """Return the canonical ``/1`` document of a Core lifecycle event."""
    if not isinstance(event, EquityLifecycleEvent):
        raise TypeError("event must be an EquityLifecycleEvent.")
    document: dict[str, Any] = {
        "format": EQUITY_RECOMMENDATION_LIFECYCLE_EVENT_FORMAT,
        "sequence": event.sequence,
        "kind": event.kind.value,
        "from_state": event.from_state.value,
        "to_state": event.to_state.value,
        "evaluated_at": event.evaluated_at.value,
        "evidence": _evidence_document(event.evidence),
        "touched": [level.value for level in event.touched],
        "ordering": event.ordering.value,
        "reason": None if event.reason is None else event.reason.value,
        "deficiencies": [
            {"opens_at": d.opens_at.value, "closes_at": d.closes_at.value, "kind": d.kind.value}
            for d in event.deficiencies
        ],
        "cited_sequence": event.cited_sequence,
        "superseded_evidence": (
            None
            if event.superseded_evidence is None
            else _evidence_document(event.superseded_evidence)
        ),
        "materiality": None if event.materiality is None else event.materiality.value,
    }
    return canonical_equity_recommendation_json(document)


def equity_lifecycle_event_documents(events: Iterable[EquityLifecycleEvent]) -> tuple[str, ...]:
    """Return the canonical documents of ``events``, in order."""
    return tuple(equity_lifecycle_event_document(event) for event in events)


# ---------------------------------------------------------------------------
# Strict decoding
# ---------------------------------------------------------------------------


def _object(value: object, keys: frozenset[str], subject: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _bad(f"{subject} must be a JSON object.")
    if set(value) != keys:
        missing, extra = sorted(keys - set(value)), sorted(set(value) - keys)
        raise _bad(f"{subject} keys differ (missing {missing}, extra {extra}).")
    return value


def _integer(value: object, subject: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _bad(f"{subject} must be a JSON integer.")
    return value


def _text(value: object, subject: str) -> str:
    if not isinstance(value, str):
        raise _bad(f"{subject} must be a JSON string.")
    return value


def _array(value: object, subject: str) -> list[Any]:
    if not isinstance(value, list):
        raise _bad(f"{subject} must be a JSON array.")
    return value


def _instant(value: object, subject: str) -> PointInTime:
    text = _text(value, subject)
    point = PointInTime(text)
    if point.value != text:
        raise _bad(f"{subject} is not canonical instant text.")
    return point


def _evidence(value: object, subject: str) -> EquityLifecycleEvidence:
    item = _object(value, _EVIDENCE_KEYS, subject)
    return EquityLifecycleEvidence(
        _text(item["reference"], f"{subject} reference"),
        _instant(item["opens_at"], f"{subject} opens_at"),
        _instant(item["closes_at"], f"{subject} closes_at"),
        tuple(
            _text(entry, f"{subject} observation")
            for entry in _array(item["observations"], f"{subject} observations")
        ),
    )


def _deficiency(value: object) -> EquityEvidenceDeficiency:
    item = _object(value, _DEFICIENCY_KEYS, "A deficiency")
    return EquityEvidenceDeficiency(
        _instant(item["opens_at"], "A deficiency opens_at"),
        _instant(item["closes_at"], "A deficiency closes_at"),
        EquityEvidenceDeficiencyKind(_text(item["kind"], "A deficiency kind")),
    )


def _optional[T](value: object, build: Any) -> T | None:
    return None if value is None else build(value)


def _event(document: dict[str, Any]) -> EquityLifecycleEvent:
    return EquityLifecycleEvent(
        sequence=_integer(document["sequence"], "sequence"),
        kind=EquityLifecycleEventKind(_text(document["kind"], "kind")),
        from_state=EquityLifecycleState(_text(document["from_state"], "from_state")),
        to_state=EquityLifecycleState(_text(document["to_state"], "to_state")),
        evaluated_at=_instant(document["evaluated_at"], "evaluated_at"),
        evidence=_evidence(document["evidence"], "evidence"),
        touched=tuple(
            EquityLifecycleLevel(_text(level, "A touched level"))
            for level in _array(document["touched"], "touched")
        ),
        ordering=EquityIntrabarOrdering(_text(document["ordering"], "ordering")),
        reason=_optional(
            document["reason"],
            lambda v: EquityLifecycleCancellationReason(_text(v, "reason")),
        ),
        deficiencies=tuple(
            _deficiency(item) for item in _array(document["deficiencies"], "deficiencies")
        ),
        cited_sequence=_optional(
            document["cited_sequence"], lambda v: _integer(v, "cited_sequence")
        ),
        superseded_evidence=_optional(
            document["superseded_evidence"], lambda v: _evidence(v, "superseded_evidence")
        ),
        materiality=_optional(
            document["materiality"],
            lambda v: EquityRevisionMateriality(_text(v, "materiality")),
        ),
    )


def decode_equity_lifecycle_event_document(text: str) -> EquityLifecycleEvent:
    """Return the Core event a canonical ``/1`` document states, or refuse it.

    See the module docstring: every check is strict, and the document must
    re-encode to exactly its own text.
    """
    if not isinstance(text, str):
        raise _bad("A lifecycle-event document must be text.")
    try:
        document = json.loads(text)
    except ValueError as exc:
        raise _bad(f"The document is not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise _bad("A lifecycle-event document must be a JSON object.")
    if document.get("format") != EQUITY_RECOMMENDATION_LIFECYCLE_EVENT_FORMAT:
        raise IncompatibleEquityLifecycleEventDocumentError(
            f"Unsupported lifecycle-event format {document.get('format')!r}."
        )
    _object(document, _EVENT_KEYS, "A lifecycle-event document")
    try:
        event = _event(document)
    except InvalidEquityLifecycleEventDocumentError:
        raise
    except _DECODE_ERRORS as exc:
        raise _bad(f"The document is not a valid lifecycle event: {exc}") from exc
    if equity_lifecycle_event_document(event) != text:
        raise _bad("The document is not the canonical encoding of its event.")
    return event


# ---------------------------------------------------------------------------
# Verified rehydration, for every store
# ---------------------------------------------------------------------------


def equity_recorded_lifecycle(
    issuance: EquityRecordedIssuance,
    window: EquityLifecycleWindow,
    documents: Sequence[str],
) -> EquityRecordedLifecycle:
    """Return the verified lifecycle that stored ``documents`` state, or refuse them.

    ``window`` must be the authoritatively derived window of ``issuance``. A
    document that does not decode strictly is ``EVENT``; a history Core replay
    refuses is ``HISTORY``. Nothing is repaired or skipped.
    """
    integrity = InvalidStoredEquityRecommendationLifecycleError
    events = []
    for position, text in enumerate(documents, start=1):
        try:
            event = decode_equity_lifecycle_event_document(text)
        except InvalidEquityLifecycleEventDocumentError as exc:
            raise integrity(
                integrity.EVENT,
                f"Stored lifecycle event {position} does not decode ({type(exc).__name__}).",
            ) from exc
        events.append(event)
    try:
        lifecycle = EquityRecommendationLifecycle(window, tuple(events))
    except InvalidEquityRecommendationLifecycleError as exc:
        raise integrity(
            integrity.HISTORY, f"Core replay refuses the stored history: {exc}"
        ) from exc
    return EquityRecordedLifecycle(issuance, lifecycle)
