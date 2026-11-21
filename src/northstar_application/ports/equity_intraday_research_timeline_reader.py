"""Read-only view of authoritative equity intraday research timelines (EI-7C).

EI-7 research inspection reads persisted EI-3C timelines; it never appends.
This protocol names exactly the read operation of the EI-3C timeline store,
``load``, so a holder of it cannot append research history. It adds no
operation and changes no EI-3C contract: every existing
``EquityIntradayResearchTimelineStore`` satisfies it structurally, and a
read-only adapter (EI-7D) satisfies it without any write method.

``load(strategy, listing, trading_date)`` returns the stored record with every
committed entry in sequence order, or None when no entry was ever committed
(EI3-D11: no empty timeline exists). It never returns a partial record, and
it never builds, evaluates or repairs a research decision: verification is
EI3-D13 rehydration in the Application.

Failures are the EI-3C ones: EquityIntradayResearchTimelineUnavailableError
when storage cannot be read, and InvalidStoredEquityIntradayResearchTimelineError
when a stored record cannot be constructed.

``runtime_checkable`` structural checks only that ``load`` exists, not how it
behaves; behaviour is proven by contract tests.

Nothing here imports a database, reads a clock or calls out.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from datetime import date

    from northstar_core.domain.value_objects import ListingReference
    from northstar_core.strategy import StrategyIdentity

    from northstar_application.application_services.advance_equity_intraday_research_timeline import (  # noqa: E501
        StoredEquityIntradayResearchTimeline,
    )


@runtime_checkable
class EquityIntradayResearchTimelineReader(Protocol):
    """Read a stored research timeline by strategy, listing and date; None when none exists."""

    def load(
        self, strategy: StrategyIdentity, listing: ListingReference, trading_date: date
    ) -> StoredEquityIntradayResearchTimeline | None:
        """Return every committed entry of the timeline, or None when none was committed."""
