"""Application contract for initiating or continuing Execution coordination."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionRequest:
    """Expresses an Execution use-case intent and its workflow references."""

    operation: str
    request_id: str
    correlation_id: str | None = None
    order_reference: str | None = None
    trade_reference: str | None = None
    portfolio_reference: str | None = None
    strategy_reference: str | None = None
    participant_reference: str | None = None
    listing_reference: str | None = None
    external_result_reference: str | None = None
    caller_reference: str | None = None
