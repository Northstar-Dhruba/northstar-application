"""Application port for acquiring provider-native daily futures observations.

Some providers publish a daily candle per contract directly, so there is no
minute history to fold. This port is where those candles enter Application. It
is additive: the minute-bar port, FuturesHistoricalMarketDataSource, and the
fold that consumes it are unchanged.

Why an observation rather than a FuturesOHLCVBar
------------------------------------------------
A canonical bar is stamped at its interval's completion, and for a daily bar
that instant is the trading session's close. That close comes from the session
calendar, not the provider. Providers stamp daily candles at arbitrary instants:
local midnight, the session open, or a UTC date. Letting a source build the bar
would either push the session calendar into Infrastructure a second time or let
a provider timestamp become the bar's PointInTime.

The source therefore returns what it actually knows: which session label a
candle belongs to, and the candle's values. Application matches the label to a
resolved FuturesTradingSession and stamps the bar at that session's close.

Why there is no open interest
-----------------------------
FuturesOHLCVBar deliberately has no open-interest field yet, so there is nowhere
canonical to persist one. An observation carrying it would carry a value that is
silently dropped. It is added here when the bar defines it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime

from northstar_core.derivatives import QuoteValue
from northstar_core.foundation.value_objects import Quantity
from northstar_core.futures import FuturesContract


class InvalidFuturesNativeDailyObservationError(ValueError):
    """Raised when a native daily futures observation is invalid."""


@dataclass(frozen=True, slots=True)
class FuturesNativeDailyObservation:
    """One provider-native daily candle for one contract, keyed by session label.

    ``trading_date`` is the exchange *session label* the candle summarises, in
    the same terms as FuturesTradingSession.trading_date. It is not a provider
    timestamp, and the observation deliberately carries no instant at all.

    ``contract`` is the exact dated contract the candle describes, echoed back
    by the source so that a mis-mapped provider instrument is caught rather
    than stored under the wrong identity.

    ``volume`` must be integral: it is a count of contracts, and the daily bar
    folded from minute bars is integral by construction. Price coherence
    (high and low bounding open and close) is left to FuturesOHLCVBar, which
    owns that rule.
    """

    contract: FuturesContract
    trading_date: date
    open: QuoteValue
    high: QuoteValue
    low: QuoteValue
    close: QuoteValue
    volume: Quantity

    def __post_init__(self) -> None:
        if not isinstance(self.contract, FuturesContract):
            raise InvalidFuturesNativeDailyObservationError(
                "FuturesNativeDailyObservation contract must be a FuturesContract value."
            )
        _validate_trading_date(self.trading_date)
        for field_name in ("open", "high", "low", "close"):
            if not isinstance(getattr(self, field_name), QuoteValue):
                raise InvalidFuturesNativeDailyObservationError(
                    f"FuturesNativeDailyObservation {field_name} must be a QuoteValue."
                )
        if not isinstance(self.volume, Quantity):
            raise InvalidFuturesNativeDailyObservationError(
                "FuturesNativeDailyObservation volume must be a Quantity value."
            )
        value = self.volume.value
        if value != value.to_integral_value():
            raise InvalidFuturesNativeDailyObservationError(
                f"FuturesNativeDailyObservation volume must be integral; received {value}."
            )

    def __str__(self) -> str:
        return (
            f"{self.contract} {self.trading_date.isoformat()} "
            f"O={self.open} H={self.high} L={self.low} C={self.close} V={self.volume}"
        )


def _validate_trading_date(value: date) -> None:
    if value is None:
        raise InvalidFuturesNativeDailyObservationError(
            "FuturesNativeDailyObservation trading date cannot be None."
        )
    if isinstance(value, datetime):
        raise InvalidFuturesNativeDailyObservationError(
            "FuturesNativeDailyObservation trading date must be a plain date, not a datetime."
        )
    if not isinstance(value, date):
        raise InvalidFuturesNativeDailyObservationError(
            "FuturesNativeDailyObservation trading date must be a date value."
        )


class FuturesNativeDailyMarketDataSource(ABC):
    """Acquires one futures contract's native daily candles over a label range.

    ``start_trading_date`` and ``end_trading_date`` are inclusive session
    labels. Implementations must return an immutable tuple of
    FuturesNativeDailyObservation values. Each one:

    - carries exactly the requested FuturesContract
    - has a ``trading_date`` inside the inclusive requested range
    - is the only observation for its ``trading_date``

    Order is not part of the contract. Observations are keyed by session label,
    and the caller places them in calendar order itself.

    A label with no candle is simply absent. The source decides nothing about
    whether a session *should* have a candle or whether the candle is final.
    The session calendar answers the first and the caller's chosen range the
    second.

    What Infrastructure owns
    ------------------------
    Mapping a FuturesContract to a provider instrument key, choosing the
    provider's daily interval, converting the provider's candle timestamp to the
    session label it summarises, authentication, and payload formats. None of it
    crosses this boundary. In particular, a provider timestamp is never passed
    through as an instant.

    The abstract contract documents these obligations but cannot enforce them
    at runtime. AcquireFuturesNativeDailyHistoryUseCase enforces them on every
    result.
    """

    @abstractmethod
    def fetch_daily_observations(
        self, contract: FuturesContract, start_trading_date: date, end_trading_date: date
    ) -> tuple[FuturesNativeDailyObservation, ...]:
        """Return the contract's native daily candles labelled within the range."""
