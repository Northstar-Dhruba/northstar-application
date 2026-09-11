"""Application abstraction for execution workflow Order state coordination."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from northstar_application.execution.contracts import (
        ExecutionFailure,
        ExecutionRequest,
        ExecutionResult,
    )


class OrderPersistencePort(ABC):
    """Coordinates Order state retrieval and persistence for Execution workflows."""

    @abstractmethod
    def load(self, request: ExecutionRequest) -> ExecutionResult | ExecutionFailure:
        """Load the Order context required by an execution workflow."""

    @abstractmethod
    def save(self, request: ExecutionRequest) -> ExecutionResult | ExecutionFailure:
        """Persist Order state resulting from approved Domain behavior."""
