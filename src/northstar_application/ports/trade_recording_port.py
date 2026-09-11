"""Application abstraction for recording approved Trade outcomes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from northstar_application.execution.contracts import (
        ExecutionFailure,
        ExecutionRequest,
        ExecutionResult,
    )


class TradeRecordingPort(ABC):
    """Coordinates recording of Trade truth established by the Domain."""

    @abstractmethod
    def record(self, request: ExecutionRequest) -> ExecutionResult | ExecutionFailure:
        """Record Trade state established through approved Domain behavior."""
