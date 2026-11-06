"""Application port for authoritative equity intraday research timelines (EI-3C).

``EquityIntradayResearchTimelineStore`` is the durable source of truth for EI-3B
research timelines, under the owner decisions recorded on 2026-11-07 in the
EI-3 Decision Register (northstar-docs, product/Equity-Intraday-V1-EI3-
Decision-Register.md, section 11): EI3-D11, EI3-D12 and EI3-D13. It holds
synthetic research history only.

A timeline is keyed by strategy identity, listing and trading date. Its record
-- ``StoredEquityIntradayResearchTimeline`` in advance_equity_intraday_research_
timeline (Application) -- holds the canonical configuration document and every
entry, each with its canonical EI-3A decision document and fingerprints. An
implementation stores and returns records exactly; it never builds, evaluates
or repairs a research decision.

``load(strategy, listing, trading_date)`` returns the stored record with every
committed entry in sequence order, or None when no entry was ever committed:
a timeline is established only by its first committed entry (EI3-D11), so no
empty timeline exists. It never returns a partial record.

``append(timeline, expected_head_fingerprint)`` is an atomic compare-and-append.
``timeline`` is the complete intended record: every stored entry followed by
the new ones. In one write transaction the implementation:

1. reads the stored record under the write lock;
2. raises EquityIntradayResearchTimelineConfigurationConflictError when a
   stored record holds another configuration document;
3. raises EquityIntradayResearchTimelineStaleError unless the stored head
   fingerprint -- the last committed entry's, None when nothing is stored --
   equals ``expected_head_fingerprint``, and the intended record repeats
   every stored entry exactly;
4. inserts the identity, when none is stored, and every entry after the
   stored ones, and commits them all, or none of them.

Nothing stored is ever updated, replaced or deleted, and no mutable head is
kept: the head is the last committed entry.

Commit outcomes
---------------
``append`` returning means the commit is confirmed. Stale and configuration
conflict errors, and EquityIntradayResearchTimelineUnavailableError, mean
nothing was committed. EquityIntradayResearchTimelineCommitUnconfirmedError
means the outcome is unknown -- the failure came at or after the commit, or
the acknowledgment was lost -- and is never evidence of a rollback.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    from northstar_core.domain.value_objects import ListingReference
    from northstar_core.strategy import StrategyIdentity

    from northstar_application.application_services.advance_equity_intraday_research_timeline import (  # noqa: E501
        StoredEquityIntradayResearchTimeline,
    )


class EquityIntradayResearchTimelineUnavailableError(RuntimeError):
    """Raised when timeline storage is unavailable or holds data it cannot decode.

    Raised by ``append``, it means nothing was committed.
    """


class EquityIntradayResearchTimelineCommitUnconfirmedError(RuntimeError):
    """Raised by ``append`` when the commit outcome is unknown.

    The failure came at or after the commit, or its acknowledgment was lost.
    The entries may or may not be durable: this is never a rollback.
    """


class EquityIntradayResearchTimelineCommitUnresolvedError(RuntimeError):
    """Raised when an unconfirmed commit could not be resolved against stored history.

    Whether the attempted entries are durable is unknown. Nothing is claimed
    either way; a later authoritative load settles it.
    """

    def __init__(self, message: str, expected_head: str | None, attempted_head: str | None) -> None:
        super().__init__(message)
        self.expected_head = expected_head
        self.attempted_head = attempted_head


class EquityIntradayResearchTimelineStaleError(ValueError):
    """Raised by ``append`` when the stored head is not the expected one; nothing is written."""


class EquityIntradayResearchTimelineConfigurationConflictError(ValueError):
    """Raised when a stored timeline holds another configuration; nothing is written."""


class InvalidStoredEquityIntradayResearchTimelineError(ValueError):
    """Raised when stored timeline history fails verification. The whole load is refused."""


class EquityIntradayResearchDecisionNotReproducedError(
    InvalidStoredEquityIntradayResearchTimelineError
):
    """Raised when a stored decision does not re-derive exactly from its snapshot (EI3-D13)."""


class EquityIntradayResearchHistoricalEvaluationUnavailableError(
    InvalidStoredEquityIntradayResearchTimelineError
):
    """Raised when no compatible evaluation behaviour exists for a stored decision (EI3-D13)."""


class EquityIntradayResearchTimelineStore(ABC):
    """Durable, append-only store of authoritative research timelines."""

    @abstractmethod
    def load(
        self, strategy: StrategyIdentity, listing: ListingReference, trading_date: date
    ) -> StoredEquityIntradayResearchTimeline | None:
        """Return every committed entry of the timeline, or None when none was committed."""

    @abstractmethod
    def append(
        self,
        timeline: StoredEquityIntradayResearchTimeline,
        expected_head_fingerprint: str | None,
    ) -> None:
        """Commit ``timeline``'s new entries if the stored head is still the expected one."""
