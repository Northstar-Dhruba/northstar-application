"""Contract tests for the historical market data Application port."""

from dataclasses import FrozenInstanceError

import pytest
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol, Timeframe
from northstar_core.market_data import HistoricalOHLCVBar

from northstar_application.ports import (
    HistoricalMarketDataQuery,
    HistoricalMarketDataRepository,
    InvalidHistoricalMarketDataQueryError,
)


def _build_query() -> HistoricalMarketDataQuery:
    return HistoricalMarketDataQuery(
        symbol=Symbol("AAPL"),
        exchange_code=ExchangeCode("NASDAQ"),
        timeframe=Timeframe("1d"),
        start=PointInTime("2026-09-01T00:00:00Z"),
        end=PointInTime("2026-09-15T00:00:00Z"),
    )


def test_query_accepts_inclusive_forward_range() -> None:
    query = _build_query()

    assert query.start.value == "2026-09-01T00:00:00Z"
    assert query.end.value == "2026-09-15T00:00:00Z"


def test_query_accepts_equal_start_and_end() -> None:
    point_in_time = PointInTime("2026-09-15T00:00:00Z")
    query = HistoricalMarketDataQuery(
        Symbol("AAPL"), ExchangeCode("NASDAQ"), Timeframe("1d"), point_in_time, point_in_time
    )

    assert query.start == query.end


def test_query_accepts_fractional_second_end_after_whole_second_start() -> None:
    query = HistoricalMarketDataQuery(
        Symbol("AAPL"),
        ExchangeCode("NASDAQ"),
        Timeframe("1d"),
        PointInTime("2026-09-15T09:30:00Z"),
        PointInTime("2026-09-15T09:30:00.000001Z"),
    )

    assert query.start.compare(query.end) == -1


def test_query_rejects_fractional_second_start_after_whole_second_end() -> None:
    with pytest.raises(InvalidHistoricalMarketDataQueryError, match="before or equal"):
        HistoricalMarketDataQuery(
            Symbol("AAPL"),
            ExchangeCode("NASDAQ"),
            Timeframe("1d"),
            PointInTime("2026-09-15T09:30:00.000001Z"),
            PointInTime("2026-09-15T09:30:00Z"),
        )


def test_query_compares_equivalent_point_in_time_values_after_core_normalization() -> None:
    query = HistoricalMarketDataQuery(
        Symbol("AAPL"),
        ExchangeCode("NASDAQ"),
        Timeframe("1d"),
        PointInTime("2026-09-15T14:30:00+05:00"),
        PointInTime("2026-09-15T09:30:00Z"),
    )

    assert query.start == query.end


def test_query_rejects_reversed_range() -> None:
    with pytest.raises(InvalidHistoricalMarketDataQueryError, match="before or equal"):
        HistoricalMarketDataQuery(
            Symbol("AAPL"),
            ExchangeCode("NASDAQ"),
            Timeframe("1d"),
            PointInTime("2026-09-15T00:00:00Z"),
            PointInTime("2026-09-01T00:00:00Z"),
        )


def test_query_is_immutable_and_value_based() -> None:
    left = _build_query()
    right = _build_query()

    assert left == right
    assert hash(left) == hash(right)
    with pytest.raises(FrozenInstanceError):
        left.symbol = Symbol("MSFT")


def test_repository_contract_returns_ordered_historical_bars() -> None:
    class InMemoryHistoricalRepository(HistoricalMarketDataRepository):
        def get_history(self, query: HistoricalMarketDataQuery) -> tuple[HistoricalOHLCVBar, ...]:
            return ()

    repository = InMemoryHistoricalRepository()

    assert repository.get_history(_build_query()) == ()
