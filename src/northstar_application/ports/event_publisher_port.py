"""Application abstraction for approved post-transaction event delivery."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from northstar_application.execution.contracts import (
        ExecutionFailure,
        ExecutionRequest,
        ExecutionResult,
    )


class EventPublisherPort(ABC):
    """Provides post-transaction event publication in Application terms."""

    @abstractmethod
    def publish(self, request: ExecutionRequest) -> ExecutionResult | ExecutionFailure:
        """Deliver an approved publication request beyond the Application boundary."""
