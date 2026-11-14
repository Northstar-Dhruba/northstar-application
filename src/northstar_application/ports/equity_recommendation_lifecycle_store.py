"""Application port for durable equity recommendation lifecycles (EI-6B, ADR-024).

EI-6B persists the Core lifecycle events (EI-5B) of an authoritative MODELED
issuance (EI-6A), as Application-owned canonical documents
(``northstar.equity-recommendation-lifecycle-event/1``). The design is the
EI-6B Durable Recommendation Lifecycle Persistence Design v1.0, sections 4, 5
and 10. Application ships no durable implementation; EI-6B Infrastructure
implements this port in SQLite.

``EquityRecommendationLifecycleStore``
--------------------------------------
- ``load_lifecycle(issuance_id)`` returns an ``EquityRecordedLifecycle``: the
  re-derived parent issuance and its lifecycle, every event strictly decoded
  and the whole history replayed by Core with the authoritatively derived
  window. ``None`` means no such issuance. A known issuance with no events is
  an empty, ``RECORDED`` lifecycle. A raw, unverified event collection is never
  returned.
- ``append(authorization)`` accepts only a
  ``VerifiedEquityRecommendationLifecycleAppend``, validated with
  ``require_verified_equity_recommendation_lifecycle_append`` **before any
  connection opens**. Against the stored history it is:

  - ``APPENDED`` when the stored history equals the authorization's prior
    history: the new events are written, all or nothing;
  - ``EQUIVALENT`` when the stored history already equals prior plus the new
    events: nothing is written (an identical retry, or the loser of two
    identical concurrent requests);
  - ``CONFLICT`` otherwise: a stale writer, or another history committed
    first. Nothing is written and nothing is adopted.

  ``equity_lifecycle_append_decision`` states the rule once, for every store.

Failures
--------
- EquityRecommendationLifecycleStoreUnavailableError: temporary; nothing written.
- EquityRecommendationLifecycleCommitUnconfirmedError: the COMMIT outcome is
  unknown. Never evidence of a rollback; the caller reconciles by reload.
- UnverifiedEquityRecommendationLifecycleAppendError: a missing, foreign,
  unissued or changed authorization; raised before any connection opens.
- InvalidStoredEquityRecommendationLifecycleError: **permanent** corruption of
  the stored lifecycle (``EVENT``, ``HISTORY`` or ``WINDOW``); never retryable,
  never a conflict and never equivalent. Corrupt *parent issuance* authority
  stays the EI-6A error InvalidStoredEquityRecommendationIssuanceAuthorityError,
  so the two kinds of corruption remain distinct.
- InvalidEquityRecommendationLifecycleWriteError: the store refused, or
  reported, a write its contract does not allow; nothing written.

Nothing here imports a database, reads a clock or calls out.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from northstar_core.recommendations.equity.lifecycle import EquityRecommendationLifecycle

if TYPE_CHECKING:
    from northstar_application.application_services.equity_recommendation_issuance_records import (  # noqa: E501
        EquityRecordedIssuance,
    )
    from northstar_application.application_services.equity_recommendation_lifecycle_authorization import (  # noqa: E501
        VerifiedEquityRecommendationLifecycleAppend,
    )


class EquityRecommendationLifecycleStoreUnavailableError(RuntimeError):
    """Raised when the lifecycle store cannot be opened, read or locked; nothing was written."""


class EquityRecommendationLifecycleCommitUnconfirmedError(RuntimeError):
    """Raised when an append's commit outcome is unknown. It is never a rollback."""


class UnverifiedEquityRecommendationLifecycleAppendError(ValueError):
    """Raised when a lifecycle append is not authorized; nothing is written.

    The authorization is missing, of the wrong type, was not issued by the
    recording use case, or covers other content than offered.
    """


class InvalidEquityRecommendationLifecycleWriteError(ValueError):
    """Raised when a store refuses or reports a write its contract does not allow."""


class InvalidStoredEquityRecommendationLifecycleError(ValueError):
    """Raised when a stored lifecycle is corrupt (EI-6B design, section 4).

    A **permanent** integrity failure: never retryable, never ``CONFLICT`` and
    never ``EQUIVALENT``. Nothing is repaired. ``category`` is one of:

    - ``EVENT``: a stored document does not decode strictly, is not canonical,
      has an unsupported format, or disagrees with its stored columns;
    - ``HISTORY``: the stored sequence is not contiguous, or Core replay
      refuses the stored events;
    - ``WINDOW``: the lifecycle window cannot be derived from the parent
      issuance's provenance.

    Messages carry no stored document content.
    """

    EVENT = "EVENT"
    HISTORY = "HISTORY"
    WINDOW = "WINDOW"
    CATEGORIES = frozenset({EVENT, HISTORY, WINDOW})

    def __init__(self, category: str, message: str) -> None:
        if category not in self.CATEGORIES:
            raise ValueError(f"category must be one of {sorted(self.CATEGORIES)}.")
        super().__init__(f"{category}: {message}")
        self.category = category

    @property
    def retryable(self) -> bool:
        """Always False: corrupt stored history never clears by retrying."""
        return False


class EquityRecommendationLifecycleWriteOutcome(StrEnum):
    """What one authorized lifecycle append established (EI-6B design, section 10)."""

    APPENDED = "APPENDED"
    EQUIVALENT = "EQUIVALENT"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True, slots=True)
class EquityRecordedLifecycle:
    """An authoritative issuance's lifecycle, as the store holds it.

    ``issuance`` is the re-derived parent issuance (EI-6A). ``lifecycle`` is the
    Core lifecycle replayed from every stored event, with the window derived
    from that issuance's verified provenance; ``window`` and ``state`` are its.
    """

    issuance: EquityRecordedIssuance
    lifecycle: EquityRecommendationLifecycle

    def __post_init__(self) -> None:
        from northstar_application.application_services.equity_recommendation_issuance_records import (  # noqa: E501
            EquityRecordedIssuance,
        )

        if not isinstance(self.issuance, EquityRecordedIssuance):
            raise InvalidEquityRecommendationLifecycleWriteError(
                "A recorded lifecycle names an EquityRecordedIssuance."
            )
        if not isinstance(self.lifecycle, EquityRecommendationLifecycle):
            raise InvalidEquityRecommendationLifecycleWriteError(
                "A recorded lifecycle holds an EquityRecommendationLifecycle."
            )

    @property
    def issuance_id(self) -> str:
        """Return the parent issuance identity."""
        return self.issuance.recommendation.issuance_id

    @property
    def position(self) -> int:
        """Return the number of recorded events: the history position an append continues."""
        return len(self.lifecycle.events)


@dataclass(frozen=True, slots=True)
class EquityRecommendationLifecycleWrite:
    """A completed append decision and the authoritative history it concerns.

    ``recorded`` is the history after the append for APPENDED, the equal
    history for EQUIVALENT, and the stored history that refused it for
    CONFLICT.
    """

    outcome: EquityRecommendationLifecycleWriteOutcome
    recorded: EquityRecordedLifecycle

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, EquityRecommendationLifecycleWriteOutcome):
            raise InvalidEquityRecommendationLifecycleWriteError(
                "outcome must be an EquityRecommendationLifecycleWriteOutcome."
            )
        if not isinstance(self.recorded, EquityRecordedLifecycle):
            raise InvalidEquityRecommendationLifecycleWriteError(
                "A lifecycle write always reports the recorded lifecycle it concerns."
            )


class EquityRecommendationLifecycleStore(ABC):
    """Durable lifecycles of authoritative issuances: verified reads, authorized appends."""

    @abstractmethod
    def load_lifecycle(self, issuance_id: str) -> EquityRecordedLifecycle | None:
        """Return the verified, replayed lifecycle of ``issuance_id``, or None when unknown."""

    @abstractmethod
    def append(
        self, authorization: VerifiedEquityRecommendationLifecycleAppend
    ) -> EquityRecommendationLifecycleWrite:
        """Append exactly the events ``authorization`` covers, by compare-and-append."""
