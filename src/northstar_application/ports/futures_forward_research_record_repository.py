"""Application port for retrieving frozen futures forward research decisions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from northstar_core.foundation.value_objects import Timeframe
from northstar_core.futures import FuturesContract

if TYPE_CHECKING:
    from northstar_application.application_services.futures_forward_research_record import (
        FuturesForwardResearchRecord,
    )

_SUPPORTED_TIMEFRAME = Timeframe("1d")


class InvalidFuturesForwardResearchRecordQueryError(ValueError):
    """Raised when a futures forward research record query is invalid."""


@dataclass(frozen=True, slots=True)
class FuturesForwardResearchRecordQuery:
    """Request for every frozen decision recorded for one concrete contract.

    The query names one observation series -- a contract at a timeframe --
    rather than one decision, because measuring outcomes needs every decision
    frozen against that series, and the matching observations are retrieved
    with the same identity. It spans every strategy that decided on the series.

    It deliberately carries no strategy, no decision-instant range, no as-of
    boundary and no horizon. Those belong to the forward run, and measurement
    status is never stored, so there is nothing to filter on.

    Forward research is session-daily, so the query accepts only ``1d``.
    """

    contract: FuturesContract
    timeframe: Timeframe

    def __post_init__(self) -> None:
        if not isinstance(self.contract, FuturesContract):
            raise InvalidFuturesForwardResearchRecordQueryError(
                "FuturesForwardResearchRecordQuery contract must be a FuturesContract value."
            )
        if not isinstance(self.timeframe, Timeframe):
            raise InvalidFuturesForwardResearchRecordQueryError(
                "FuturesForwardResearchRecordQuery timeframe must be a Timeframe value."
            )
        if self.timeframe != _SUPPORTED_TIMEFRAME:
            raise InvalidFuturesForwardResearchRecordQueryError(
                f"FuturesForwardResearchRecordQuery supports session-daily research only; "
                f"timeframe {self.timeframe} is not {_SUPPORTED_TIMEFRAME}."
            )


class FuturesForwardResearchRecordRepository(ABC):
    """Retrieves frozen futures forward research decisions in decision order.

    Implementations must return an immutable tuple holding every frozen record
    whose contract and timeframe equal the query's, and no other. Contracts
    match by value, so a query for ES December never returns MES December or ES
    March.

    Records are ordered oldest to newest by decision instant compared with
    PointInTime.compare(), not by stored text, and records sharing one instant
    under different strategies are ordered by strategy identity. A valid query
    matching no frozen decision returns an empty tuple. The abstract contract
    documents these obligations but cannot enforce ordering at runtime.
    """

    @abstractmethod
    def get_records(
        self, query: FuturesForwardResearchRecordQuery
    ) -> tuple[FuturesForwardResearchRecord, ...]:
        """Return frozen decisions for one contract and timeframe, oldest to newest."""
