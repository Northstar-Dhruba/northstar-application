"""Application contract for the final outcome of an Execution workflow."""

from __future__ import annotations

from dataclasses import dataclass

from northstar_application.execution.contracts.execution_failure import ExecutionFailure
from northstar_application.execution.contracts.execution_result import ExecutionResult


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """Communicates the final Application-level status of an Execution use case."""

    request_id: str
    status: str
    correlation_id: str | None = None
    result: ExecutionResult | None = None
    failure: ExecutionFailure | None = None
