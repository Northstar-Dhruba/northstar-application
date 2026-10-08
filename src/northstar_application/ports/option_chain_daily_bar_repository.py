"""Application port for every stored daily bar of one option expiration at one instant."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from northstar_core.derivatives import ExpirationDate
from northstar_core.foundation.value_objects import PointInTime
from northstar_core.options import OptionOHLCVBar, OptionProductReference


class InvalidOptionChainDailyBarQueryError(ValueError):
    """Raised when an option chain daily-bar query is invalid."""


@dataclass(frozen=True, slots=True)
class OptionChainDailyBarQuery:
    """Request for one expiration's daily bars stamped exactly at one instant."""

    product: OptionProductReference
    expiration_date: ExpirationDate
    as_of: PointInTime

    def __post_init__(self) -> None:
        subject = "OptionChainDailyBarQuery"
        if not isinstance(self.product, OptionProductReference):
            raise InvalidOptionChainDailyBarQueryError(
                f"{subject} product must be an OptionProductReference value."
            )
        if not isinstance(self.expiration_date, ExpirationDate):
            raise InvalidOptionChainDailyBarQueryError(
                f"{subject} expiration date must be an ExpirationDate value."
            )
        if not isinstance(self.as_of, PointInTime):
            raise InvalidOptionChainDailyBarQueryError(
                f"{subject} as-of must be a PointInTime value."
            )


class OptionChainDailyBarRepository(ABC):
    """Reads one expiration's canonical daily bars stamped exactly at one instant.

    Implementations must return an immutable tuple of exactly those stored bars
    whose contract has the query's product and expiration, whose timeframe is
    the daily ``1d`` and whose instant equals ``as_of``: no earlier, later or
    latest-available bar stands in for a missing one. The tuple holds each
    contract at most once, ordered by strike ascending, then CALL before PUT.
    A query matching nothing returns an empty tuple, including when nothing has
    ever been stored. The port reads only what is stored: it creates nothing,
    acquires nothing and reaches no provider, network or clock.
    """

    @abstractmethod
    def daily_bars_at(self, query: OptionChainDailyBarQuery) -> tuple[OptionOHLCVBar, ...]:
        """Return the expiration's daily bars stamped exactly at the instant."""
