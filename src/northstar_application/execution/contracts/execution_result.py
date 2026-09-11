"""Application contract for a normalized Execution workflow-step result."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Communicates a protocol-independent result of one Execution workflow step."""

    request_id: str
    status: str
    correlation_id: str | None = None
    result_reference: str | None = None
    detail: str | None = None
