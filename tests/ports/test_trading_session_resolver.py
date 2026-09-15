"""Contract tests for the TradingSessionResolver Application port."""

from __future__ import annotations

from datetime import date

from northstar_core.foundation.value_objects import ExchangeCode, PointInTime

from northstar_application.ports import (
    TradingSessionResolutionError,
    TradingSessionResolver,
)


class StubTradingSessionResolver(TradingSessionResolver):
    """Deterministic test double implementing TradingSessionResolver."""

    def __init__(
        self,
        schedule: dict[tuple[ExchangeCode, date], PointInTime | None],
        supported_exchanges: set[ExchangeCode] | None = None,
    ) -> None:
        self._schedule = schedule
        self._supported_exchanges = supported_exchanges

    def resolve_session_close(
        self, exchange_code: ExchangeCode, trading_date: date
    ) -> PointInTime | None:
        if self._supported_exchanges is not None and exchange_code not in self._supported_exchanges:
            raise TradingSessionResolutionError(
                f"Unsupported exchange venue: {exchange_code.value}"
            )
        if (exchange_code, trading_date) not in self._schedule:
            raise TradingSessionResolutionError(
                f"Trading session data unresolvable for {exchange_code.value} on {trading_date}."
            )
        return self._schedule[(exchange_code, trading_date)]


def test_resolver_returns_point_in_time_for_valid_trading_session() -> None:
    exchange = ExchangeCode("NASDAQ")
    session_date = date(2026, 9, 15)
    close_pit = PointInTime("2026-09-15T20:00:00Z")

    resolver = StubTradingSessionResolver(
        schedule={(exchange, session_date): close_pit},
        supported_exchanges={exchange},
    )

    result = resolver.resolve_session_close(exchange, session_date)

    assert result == close_pit
    assert isinstance(result, PointInTime)


def test_resolver_returns_none_for_known_non_trading_day() -> None:
    exchange = ExchangeCode("NASDAQ")
    holiday_date = date(2026, 12, 25)

    resolver = StubTradingSessionResolver(
        schedule={(exchange, holiday_date): None},
        supported_exchanges={exchange},
    )

    result = resolver.resolve_session_close(exchange, holiday_date)

    assert result is None


def test_resolver_raises_error_for_unsupported_exchange() -> None:
    supported_exchange = ExchangeCode("NASDAQ")
    unsupported_exchange = ExchangeCode("UNKNOWN")
    session_date = date(2026, 9, 15)

    resolver = StubTradingSessionResolver(
        schedule={(supported_exchange, session_date): PointInTime("2026-09-15T20:00:00Z")},
        supported_exchanges={supported_exchange},
    )

    import pytest

    with pytest.raises(TradingSessionResolutionError, match="Unsupported exchange venue"):
        resolver.resolve_session_close(unsupported_exchange, session_date)


def test_resolver_raises_error_when_calendar_data_is_unresolvable() -> None:
    exchange = ExchangeCode("NASDAQ")
    missing_date = date(2026, 9, 15)

    resolver = StubTradingSessionResolver(
        schedule={},
        supported_exchanges={exchange},
    )

    import pytest

    with pytest.raises(TradingSessionResolutionError, match="Trading session data unresolvable"):
        resolver.resolve_session_close(exchange, missing_date)


def test_resolver_supports_different_exchange_venues_independently() -> None:
    nasdaq = ExchangeCode("NASDAQ")
    nse = ExchangeCode("NSE")
    target_date = date(2026, 9, 15)
    nasdaq_close = PointInTime("2026-09-15T20:00:00Z")
    nse_close = PointInTime("2026-09-15T10:00:00Z")

    resolver = StubTradingSessionResolver(
        schedule={
            (nasdaq, target_date): nasdaq_close,
            (nse, target_date): nse_close,
        },
        supported_exchanges={nasdaq, nse},
    )

    assert resolver.resolve_session_close(nasdaq, target_date) == nasdaq_close
    assert resolver.resolve_session_close(nse, target_date) == nse_close
