"""Application port for resolving market trading session completion instants."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from northstar_core.foundation.value_objects import ExchangeCode, PointInTime


class TradingSessionResolutionError(RuntimeError):
    """Raised when trading session resolution cannot be performed reliably."""


class TradingSessionResolver(ABC):
    """Resolves the completion instant of a regular market trading session.

    Implementations map an exchange venue and calendar date to the exact
    PointInTime when that regular session concludes and its full factual
    observations become available for deterministic replay.

    If the specified date is a known non-trading day for an understood venue
    (such as a weekend, exchange holiday, or scheduled closure), implementations
    must return None.

    If the venue is unsupported, unknown, or the session resolution cannot
    be performed reliably due to missing or unresolvable calendar data,
    implementations must raise TradingSessionResolutionError rather than
    returning None.
    """

    @abstractmethod
    def resolve_session_close(
        self, exchange_code: ExchangeCode, trading_date: date
    ) -> PointInTime | None:
        """Return the regular session close PointInTime, or None if not a trading day."""
