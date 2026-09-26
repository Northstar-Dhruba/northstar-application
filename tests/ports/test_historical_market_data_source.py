"""Contract tests for the HistoricalMarketDataSource Application port."""

from __future__ import annotations

from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol, Timeframe
from northstar_core.market_data import HistoricalOHLCVBar

from northstar_application.ports import (
    HistoricalMarketDataQuery,
    HistoricalMarketDataSource,
)


class DummyHistoricalMarketDataSource(HistoricalMarketDataSource):
    def fetch_history(self, query: HistoricalMarketDataQuery) -> tuple[HistoricalOHLCVBar, ...]:
        return ()


def test_source_port_contract_returns_tuple_of_bars() -> None:
    source = DummyHistoricalMarketDataSource()
    query = HistoricalMarketDataQuery(
        symbol=Symbol("AAPL"),
        exchange_code=ExchangeCode("NASDAQ"),
        timeframe=Timeframe("1d"),
        start=PointInTime("2026-09-01T00:00:00Z"),
        end=PointInTime("2026-09-15T00:00:00Z"),
    )

    result = source.fetch_history(query)

    assert result == ()
    assert isinstance(result, tuple)
