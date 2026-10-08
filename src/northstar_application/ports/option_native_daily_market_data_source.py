"""Application port for acquiring provider-native daily option observations.

A provider publishes one daily candle per exact option contract. This port is
where those candles enter Application, as provider-neutral observations keyed
by session label. Application -- not the source -- matches each label to a
resolved OptionTradingSession and stamps the canonical OptionOHLCVBar at that
session's close, so a provider timestamp never becomes a bar's instant.

The caller chooses the exact OptionContract and the inclusive range. There is no
contract selection, no nearest strike or expiry, no default range and no clock.

Why there is no open interest
-----------------------------
OptionOHLCVBar has no open-interest field: its units and revision behaviour are
not established. Provider open interest is preserved as Infrastructure evidence
and never crosses this boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime

from northstar_core.foundation.value_objects import Quantity
from northstar_core.options import OptionContract, OptionPremium


class InvalidOptionNativeDailyObservationError(ValueError):
    """Raised when a native daily option observation is invalid."""


class InvalidOptionDailyAcquisitionQueryError(ValueError):
    """Raised when an option daily acquisition query is invalid."""


def _require_label(value: object, subject: str, error: type[ValueError]) -> None:
    if value is None:
        raise error(f"{subject} cannot be None.")
    if isinstance(value, datetime) or not isinstance(value, date):
        raise error(f"{subject} must be a plain date.")


@dataclass(frozen=True, slots=True)
class OptionNativeDailyObservation:
    """One provider-native daily candle for one exact option contract.

    ``trading_date`` is the session label the candle summarises; the
    observation deliberately carries no instant. ``contract`` is echoed back by
    the source so that a mis-mapped provider instrument is caught rather than
    stored under the wrong identity. ``volume`` is a whole number of option
    contracts. The high must bound open, close and low, and the low must be
    bounded by open, close and high.
    """

    contract: OptionContract
    trading_date: date
    open: OptionPremium
    high: OptionPremium
    low: OptionPremium
    close: OptionPremium
    volume: Quantity

    def __post_init__(self) -> None:
        subject = "OptionNativeDailyObservation"
        if not isinstance(self.contract, OptionContract):
            raise InvalidOptionNativeDailyObservationError(
                f"{subject} contract must be an OptionContract value."
            )
        _require_label(
            self.trading_date, f"{subject} trading date", InvalidOptionNativeDailyObservationError
        )
        for field_name in ("open", "high", "low", "close"):
            if not isinstance(getattr(self, field_name), OptionPremium):
                raise InvalidOptionNativeDailyObservationError(
                    f"{subject} {field_name} must be an OptionPremium."
                )
        if not isinstance(self.volume, Quantity):
            raise InvalidOptionNativeDailyObservationError(
                f"{subject} volume must be a Quantity value."
            )
        value = self.volume.value
        if value != value.to_integral_value():
            raise InvalidOptionNativeDailyObservationError(
                f"{subject} volume must be a whole number of option contracts; received {value}."
            )
        if self.high < self.open or self.high < self.close or self.high < self.low:
            raise InvalidOptionNativeDailyObservationError(
                f"{subject} high must be greater than or equal to open, close, and low."
            )
        if self.low > self.open or self.low > self.close:
            raise InvalidOptionNativeDailyObservationError(
                f"{subject} low must be less than or equal to open, close, and high."
            )

    def __str__(self) -> str:
        return (
            f"{self.contract} {self.trading_date.isoformat()} "
            f"O={self.open} H={self.high} L={self.low} C={self.close} V={self.volume}"
        )


@dataclass(frozen=True, slots=True)
class OptionDailyAcquisitionQuery:
    """Request to acquire one exact option contract's daily history over sessions.

    ``start_trading_date`` and ``end_trading_date`` are inclusive session
    labels. Which labels are sessions is the option session resolver's answer;
    a range may name weekends and holidays, and those simply yield no sessions.
    The range is not checked against the contract's listed life.
    """

    contract: OptionContract
    start_trading_date: date
    end_trading_date: date

    def __post_init__(self) -> None:
        subject = "OptionDailyAcquisitionQuery"
        if not isinstance(self.contract, OptionContract):
            raise InvalidOptionDailyAcquisitionQueryError(
                f"{subject} contract must be an OptionContract value."
            )
        _require_label(
            self.start_trading_date,
            f"{subject} start trading date",
            InvalidOptionDailyAcquisitionQueryError,
        )
        _require_label(
            self.end_trading_date,
            f"{subject} end trading date",
            InvalidOptionDailyAcquisitionQueryError,
        )
        if self.start_trading_date > self.end_trading_date:
            raise InvalidOptionDailyAcquisitionQueryError(
                f"{subject} start trading date must not be after the end trading date."
            )

    def __str__(self) -> str:
        return (
            f"{self.contract} "
            f"[{self.start_trading_date.isoformat()} .. {self.end_trading_date.isoformat()}]"
        )


class OptionNativeDailyMarketDataSource(ABC):
    """Acquires one exact option contract's native daily candles over a label range.

    ``start_trading_date`` and ``end_trading_date`` are inclusive session
    labels. Implementations must return an immutable tuple of
    OptionNativeDailyObservation values. Each one:

    - carries exactly the requested OptionContract;
    - has a ``trading_date`` inside the inclusive requested range;
    - is the only observation for its ``trading_date``.

    Order is not part of the contract. A label with no candle is simply absent;
    the source decides nothing about whether a session should have a candle or
    whether a candle is final.

    Mapping an OptionContract to a provider instrument, choosing the provider's
    daily interval, reading a candle's session label, converting provider
    volume to option contracts, authentication and payload formats all belong
    to Infrastructure, and none of it crosses this boundary. The abstract
    contract documents these obligations; AcquireOptionNativeDailyHistoryUseCase
    enforces them on every result.
    """

    @abstractmethod
    def fetch_daily_observations(
        self, contract: OptionContract, start_trading_date: date, end_trading_date: date
    ) -> tuple[OptionNativeDailyObservation, ...]:
        """Return the contract's native daily candles labelled within the range."""
