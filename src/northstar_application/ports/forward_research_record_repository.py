"""Application port for retrieving frozen forward research decisions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from northstar_core.foundation.value_objects import ExchangeCode, Symbol, Timeframe

if TYPE_CHECKING:
    from northstar_application.application_services.record_forward_research_decision import (
        ForwardResearchRecord,
    )


class InvalidForwardResearchRecordQueryError(ValueError):
    """Raised when a forward research record query is invalid."""


@dataclass(frozen=True, slots=True)
class ForwardResearchRecordQuery:
    """Request for every frozen decision recorded for one observation series.

    The query identifies one market series rather than one decision, because a
    caller measuring outcomes needs every decision frozen against that series
    and will retrieve the matching observations with the same identity.

    The query deliberately carries no decision-instant range, strategy filter
    or measurement status. Status is never stored, and narrower filtering is
    not required to measure a series.
    """

    symbol: Symbol
    exchange_code: ExchangeCode
    timeframe: Timeframe

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, Symbol):
            raise InvalidForwardResearchRecordQueryError(
                "ForwardResearchRecordQuery symbol must be a Symbol value."
            )
        if not isinstance(self.exchange_code, ExchangeCode):
            raise InvalidForwardResearchRecordQueryError(
                "ForwardResearchRecordQuery exchange code must be an ExchangeCode value."
            )
        if not isinstance(self.timeframe, Timeframe):
            raise InvalidForwardResearchRecordQueryError(
                "ForwardResearchRecordQuery timeframe must be a Timeframe value."
            )


class ForwardResearchRecordRepository(ABC):
    """Retrieves frozen forward research decisions in decision order.

    Implementations must return an immutable tuple ordered oldest to newest by
    decision instant. Records sharing one decision instant under different
    strategies are all returned; their relative order must be deterministic.

    A valid query matching no frozen decision returns an empty tuple. The
    abstract contract documents these obligations but cannot enforce ordering
    at runtime.
    """

    @abstractmethod
    def get_records(self, query: ForwardResearchRecordQuery) -> tuple[ForwardResearchRecord, ...]:
        """Return frozen decisions for one series, oldest to newest."""
