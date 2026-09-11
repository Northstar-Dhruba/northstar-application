"""Application contract for non-successful or uncertain Execution coordination."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionFailure:
    """Communicates an Application-level Execution failure classification."""

    request_id: str
    category: str
    correlation_id: str | None = None
    detail: str | None = None
    retryable: bool = False
