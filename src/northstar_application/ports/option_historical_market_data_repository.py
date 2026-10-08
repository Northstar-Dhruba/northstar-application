"""Application port for reading canonical option daily market data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.options import OptionContract, OptionOHLCVBar


class InvalidOptionHistoricalMarketDataQueryError(ValueError):
    """Raised when an option historical market data query is invalid."""


@dataclass(frozen=True, slots=True)
class OptionHistoricalMarketDataQuery:
    """Request for one exact option contract's bars over an optional window.

    The query names the complete OptionContract, so a neighbouring strike, the
    other right or another expiry is never returned. ``start`` and ``end`` are
    both optional and, when present, both inclusive; chronology is compared
    with PointInTime.compare(), never by text.
    """

    contract: OptionContract
    timeframe: Timeframe
    start: PointInTime | None = None
    end: PointInTime | None = None

    def __post_init__(self) -> None:
        subject = "OptionHistoricalMarketDataQuery"
        if not isinstance(self.contract, OptionContract):
            raise InvalidOptionHistoricalMarketDataQueryError(
                f"{subject} contract must be an OptionContract value."
            )
        if not isinstance(self.timeframe, Timeframe):
            raise InvalidOptionHistoricalMarketDataQueryError(
                f"{subject} timeframe must be a Timeframe value."
            )
        for name in ("start", "end"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, PointInTime):
                raise InvalidOptionHistoricalMarketDataQueryError(
                    f"{subject} {name} must be a PointInTime value."
                )
        if self.start is not None and self.end is not None and self.start.compare(self.end) > 0:
            raise InvalidOptionHistoricalMarketDataQueryError(
                f"{subject} start must be before or equal to end."
            )

    def covers(self, point_in_time: PointInTime) -> bool:
        """Return whether an instant falls inside this query's inclusive window."""
        if self.start is not None and point_in_time.compare(self.start) < 0:
            return False
        return not (self.end is not None and point_in_time.compare(self.end) > 0)


class OptionHistoricalMarketDataRepository(ABC):
    """Retrieves one exact option contract's bars in chronological order.

    Implementations must return an immutable tuple containing exactly those bars
    whose contract and timeframe equal the query's and whose instant falls
    inside the inclusive window, ordered oldest to newest by
    PointInTime.compare(). Matching is by value equality on the complete
    contract. A valid query matching nothing returns an empty tuple, including
    when nothing has ever been stored. The port reads only what is stored: it
    performs no acquisition and reaches no provider or network.
    """

    @abstractmethod
    def get_bars(self, query: OptionHistoricalMarketDataQuery) -> tuple[OptionOHLCVBar, ...]:
        """Return one contract's bars within the window, oldest to newest."""
