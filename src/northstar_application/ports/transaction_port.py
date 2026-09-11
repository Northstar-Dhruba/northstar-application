"""Application abstraction for Execution transaction coordination."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from northstar_application.execution.contracts import (
        ExecutionFailure,
        ExecutionRequest,
        ExecutionResult,
    )


class TransactionPort(ABC):
    """Provides transaction lifecycle coordination for Execution workflows."""

    @abstractmethod
    def begin(self, request: ExecutionRequest) -> ExecutionResult | ExecutionFailure:
        """Begin the transaction scope for an execution workflow."""

    @abstractmethod
    def commit(self, request: ExecutionRequest) -> ExecutionResult | ExecutionFailure:
        """Complete the transaction scope for an execution workflow."""

    @abstractmethod
    def rollback(self, request: ExecutionRequest) -> ExecutionResult | ExecutionFailure:
        """Roll back the transaction scope after an incomplete workflow."""

    @abstractmethod
    def compensate(self, request: ExecutionRequest) -> ExecutionResult | ExecutionFailure:
        """Coordinate approved compensation for non-atomic external work."""
