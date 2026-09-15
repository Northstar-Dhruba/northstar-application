"""Contract tests for the HistoricalMarketDataStore Application port."""

from __future__ import annotations

from northstar_core.market_data import HistoricalOHLCVBar

from northstar_application.ports import HistoricalMarketDataStore


class DummyHistoricalMarketDataStore(HistoricalMarketDataStore):
    def store(self, observations: tuple[HistoricalOHLCVBar, ...]) -> int:
        return len(observations)


def test_store_port_contract_accepts_tuple_of_bars() -> None:
    store = DummyHistoricalMarketDataStore()

    result = store.store(())

    assert result == 0
