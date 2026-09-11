"""Application contract for returning an Execution workflow conclusion."""

from __future__ import annotations

from dataclasses import dataclass

from northstar_application.execution.contracts.execution_failure import ExecutionFailure
from northstar_application.execution.contracts.execution_outcome import ExecutionOutcome


@dataclass(frozen=True, slots=True)
class ExecutionResponse:
    """Communicates an Execution outcome to an inbound Application boundary."""

    outcome: ExecutionOutcome
    failure: ExecutionFailure | None = None
