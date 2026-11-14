"""Content-bound lifecycle-append authorization and the shared append rule (EI-6B, D6B-5).

The EI4-D11 and EI-5D verified-write pattern, applied to lifecycle appends
(EI-6B design, sections 5 and 10).

``VerifiedEquityRecommendationLifecycleAppend``
-----------------------------------------------
It authorizes appending one exact batch of Core lifecycle events to one
authoritative issuance's lifecycle, from one exact prior history. At
construction it binds, as canonical content (``covered``):

- the authoritative issuance: its identity, its full issued-record content
  and its recording sequence;
- the lifecycle window, derived from that issuance's verified provenance;
- the canonical documents of every prior event and of every new event;
- the expected history position, ``len(prior)``.

It refuses construction unless the window agrees with the issued record, the
batch is non-empty, and Core replays ``prior + events`` with that window.

Only ``RecordEquityRecommendationLifecycleUseCase`` issues one, through the
private ``_issue_lifecycle_append``, which registers it in a module-private
weak registry. The public validator
``require_verified_equity_recommendation_lifecycle_append`` -- which a store
calls before opening anything -- accepts only a registered object of exactly
this type and re-derives its whole covered content, so an authorization built
directly, copied, made by ``dataclasses.replace`` or changed after issue is
refused. A store writes the validated ``event_documents``.

The append rule
---------------
``equity_lifecycle_append_decision(stored, authorization)`` decides every
store's append, once:

- the stored history equals the authorization's prior history: APPENDED (the
  store then writes the new events, all or nothing);
- it already equals prior plus the new events: EQUIVALENT (nothing written);
- anything else: CONFLICT (nothing written, nothing adopted).

Histories compare by their canonical documents. A stored lifecycle whose
parent issuance or window is not the authorization's is refused as an invalid
write: that is never a conflict to retry, and never equivalent.

Trust boundary: this stops ordinary unsupported writes within Northstar's
supported interfaces (ADR-024, section 14). It is not cryptographically
unforgeable. Nothing here persists anything, reads a clock or calls out.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from typing import Any, NoReturn

from northstar_core.recommendations.equity.lifecycle import (
    EquityLifecycleEvent,
    EquityLifecycleWindow,
    EquityRecommendationLifecycle,
    InvalidEquityRecommendationLifecycleError,
)

from northstar_application.application_services.equity_recommendation_generation_evidence import (
    canonical_equity_recommendation_json,
)
from northstar_application.application_services.equity_recommendation_issuance_records import (
    EquityRecordedIssuance,
)
from northstar_application.application_services.equity_recommendation_lifecycle_events import (
    equity_lifecycle_event_documents,
)
from northstar_application.ports.equity_recommendation_lifecycle_store import (
    EquityRecommendationLifecycleWriteOutcome,
    EquityRecordedLifecycle,
    InvalidEquityRecommendationLifecycleWriteError,
    UnverifiedEquityRecommendationLifecycleAppendError,
)

# Authorizations issued by the recording use case in this process, keyed by
# identity. Weak: an authorization no one holds can no longer be written.
_ISSUED_LIFECYCLE_APPENDS: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()

_W = EquityRecommendationLifecycleWriteOutcome


def _refuse(message: str) -> NoReturn:
    raise UnverifiedEquityRecommendationLifecycleAppendError(message)


def _window_document(window: EquityLifecycleWindow) -> dict[str, Any]:
    return {
        "levels": window.levels.document(),
        "eligible_from": window.eligible_from.value,
        "latest_entry_bar_opens_at": window.latest_entry_bar_opens_at.value,
        "entry_window_ends_at": window.entry_window_ends_at.value,
        "modeled_exit_at": window.modeled_exit_at.value,
    }


def _check(
    issuance: object, window: object, prior: object, events: object
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Refuse an append that is not exactly authorizable; return its documents and content."""
    if not isinstance(issuance, EquityRecordedIssuance):
        _refuse("A lifecycle append needs the authoritative EquityRecordedIssuance.")
    if not isinstance(window, EquityLifecycleWindow):
        _refuse("A lifecycle append needs the issuance's derived EquityLifecycleWindow.")
    for name, value in (("prior", prior), ("events", events)):
        if not isinstance(value, tuple) or not all(
            isinstance(event, EquityLifecycleEvent) for event in value
        ):
            _refuse(f"A lifecycle append's {name} must be a tuple of EquityLifecycleEvent.")
    assert isinstance(prior, tuple) and isinstance(events, tuple)
    if not events:
        _refuse("A lifecycle append adds at least one event.")
    record = issuance.recommendation
    if (
        window.levels,
        window.eligible_from,
        window.latest_entry_bar_opens_at,
        window.modeled_exit_at,
    ) != (
        record.levels,
        record.eligible_from,
        record.latest_entry_bar_opens_at,
        record.modeled_exit_at,
    ):
        _refuse("The window is not the issued record's.")
    try:
        EquityRecommendationLifecycle(window, prior + events)
    except InvalidEquityRecommendationLifecycleError as exc:
        _refuse(f"Core does not replay the prior history with the new events: {exc}")
    prior_documents = equity_lifecycle_event_documents(prior)
    event_documents = equity_lifecycle_event_documents(events)
    covered = canonical_equity_recommendation_json(
        {
            "issuance_id": record.issuance_id,
            "issuance": record.content(),
            "recording_sequence": issuance.recording_sequence,
            "window": _window_document(window),
            "prior": list(prior_documents),
            "events": list(event_documents),
            "position": len(prior),
        }
    )
    return prior_documents, event_documents, covered


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class VerifiedEquityRecommendationLifecycleAppend:
    """The recording use case's authorization to append exactly one batch of events.

    Equality is identity: two authorizations are never interchangeable. See the
    module docstring.
    """

    issuance: EquityRecordedIssuance
    window: EquityLifecycleWindow
    prior: tuple[EquityLifecycleEvent, ...]
    events: tuple[EquityLifecycleEvent, ...]
    prior_documents: tuple[str, ...] = field(init=False, repr=False)
    event_documents: tuple[str, ...] = field(init=False, repr=False)
    covered: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        prior_documents, event_documents, covered = _check(
            self.issuance, self.window, self.prior, self.events
        )
        object.__setattr__(self, "prior_documents", prior_documents)
        object.__setattr__(self, "event_documents", event_documents)
        object.__setattr__(self, "covered", covered)

    @property
    def issuance_id(self) -> str:
        """Return the authoritative issuance identity the append is for."""
        return self.issuance.recommendation.issuance_id

    @property
    def expected_position(self) -> int:
        """Return the number of events the stored history must already hold."""
        return len(self.prior)


def _issue_lifecycle_append(
    issuance: EquityRecordedIssuance,
    window: EquityLifecycleWindow,
    prior: tuple[EquityLifecycleEvent, ...],
    events: tuple[EquityLifecycleEvent, ...],
) -> VerifiedEquityRecommendationLifecycleAppend:
    """Private to the recording use case: issue after its checks passed."""
    authorization = VerifiedEquityRecommendationLifecycleAppend(issuance, window, prior, events)
    _ISSUED_LIFECYCLE_APPENDS[id(authorization)] = authorization
    return authorization


def require_verified_equity_recommendation_lifecycle_append(
    authorization: object,
) -> VerifiedEquityRecommendationLifecycleAppend:
    """Return ``authorization`` when it authorizes exactly its append; refuse otherwise.

    A store calls this before opening anything. It raises
    UnverifiedEquityRecommendationLifecycleAppendError for a missing, foreign
    or unissued authorization, or one whose content no longer matches.
    """
    kind = VerifiedEquityRecommendationLifecycleAppend
    if type(authorization) is not kind:
        _refuse(
            f"A {kind.__name__} issued by the recording use case is required, not "
            f"{type(authorization).__name__}."
        )
    if _ISSUED_LIFECYCLE_APPENDS.get(id(authorization)) is not authorization:
        _refuse(f"This {kind.__name__} was not issued by the recording use case.")
    assert isinstance(authorization, kind)
    prior_documents, event_documents, covered = _check(
        authorization.issuance, authorization.window, authorization.prior, authorization.events
    )
    if (prior_documents, event_documents, covered) != (
        authorization.prior_documents,
        authorization.event_documents,
        authorization.covered,
    ):
        _refuse("The append's content differs from the content this authorization covers.")
    return authorization


def equity_lifecycle_append_decision(
    stored: EquityRecordedLifecycle,
    authorization: VerifiedEquityRecommendationLifecycleAppend,
) -> EquityRecommendationLifecycleWriteOutcome:
    """Decide one append against the verified stored lifecycle (EI-6B design, section 10).

    ``authorization`` must already be validated. See the module docstring.
    """
    if not isinstance(stored, EquityRecordedLifecycle):
        raise TypeError("stored must be an EquityRecordedLifecycle.")
    if not isinstance(authorization, VerifiedEquityRecommendationLifecycleAppend):
        raise TypeError("authorization must be a VerifiedEquityRecommendationLifecycleAppend.")
    if stored.issuance != authorization.issuance:
        raise InvalidEquityRecommendationLifecycleWriteError(
            "The stored lifecycle belongs to another parent issuance than the authorization's."
        )
    if stored.lifecycle.window != authorization.window:
        raise InvalidEquityRecommendationLifecycleWriteError(
            "The stored lifecycle has another window than the authorization's."
        )
    documents = equity_lifecycle_event_documents(stored.lifecycle.events)
    if documents == authorization.prior_documents:
        return _W.APPENDED
    if documents == authorization.prior_documents + authorization.event_documents:
        return _W.EQUIVALENT
    return _W.CONFLICT
