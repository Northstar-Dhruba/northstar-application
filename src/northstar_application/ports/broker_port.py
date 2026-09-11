"""Application abstraction for broker-facing execution capabilities."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from northstar_application.execution.contracts import (
        ExecutionFailure,
        ExecutionRequest,
        ExecutionResult,
    )


class BrokerPort(ABC):
    """Provides broker-mediated execution actions in Application terms."""

    @abstractmethod
    def submit(self, request: ExecutionRequest) -> ExecutionResult | ExecutionFailure:
        """Submit an approved execution request to a broker."""

    @abstractmethod
    def cancel(self, request: ExecutionRequest) -> ExecutionResult | ExecutionFailure:
        """Request cancellation of an approved execution request at a broker."""

    @abstractmethod
    def replace(self, request: ExecutionRequest) -> ExecutionResult | ExecutionFailure:
        """Request replacement of an approved execution request at a broker."""
