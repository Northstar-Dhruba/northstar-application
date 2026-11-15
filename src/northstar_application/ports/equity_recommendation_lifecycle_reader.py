"""Read-only view of durable equity recommendation lifecycles (EI-6C).

EI-6C monitoring reads the authoritative lifecycle history to find where it
continues from; it never appends. This protocol names exactly the read
operation of the EI-6B lifecycle store, so a monitor holding it cannot write:
every lifecycle append stays with ``RecordEquityRecommendationLifecycleUseCase``
(EI-6B design, section 5). Any EI-6B store satisfies it structurally; it adds
no operation and changes no EI-6B contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from northstar_application.ports.equity_recommendation_lifecycle_store import (
        EquityRecordedLifecycle,
    )


@runtime_checkable
class EquityRecommendationLifecycleReader(Protocol):
    """Read the verified, replayed lifecycle of an issuance; None when it is unknown."""

    def load_lifecycle(self, issuance_id: str) -> EquityRecordedLifecycle | None:
        """Return the authoritative recorded lifecycle of ``issuance_id``, or None."""
