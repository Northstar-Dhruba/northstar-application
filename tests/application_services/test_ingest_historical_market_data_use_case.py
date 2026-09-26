"""Unit tests for IngestHistoricalMarketDataUseCase."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from northstar_core.foundation.value_objects import (
    Currency,
    ExchangeCode,
    PointInTime,
    Price,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.market_data import HistoricalOHLCVBar

from northstar_application.application_services import (
    HistoricalDataContractViolationError,
    HistoricalMarketDataIngestionResult,
    IngestHistoricalMarketDataUseCase,
)
from northstar_application.ports import (
    HistoricalMarketDataQuery,
    HistoricalMarketDataSource,
    HistoricalMarketDataStore,
)


def _make_bar(
    symbol: str = "AAPL",
    exchange: str = "NASDAQ",
    timestamp: str = "2026-09-01T00:00:00Z",
    timeframe: str = "1d",
    currency: str = "USD",
    open_price: str = "100",
    high_price: str = "105",
    low_price: str = "95",
    close_price: str = "102",
    volume: str = "1000",
    adjusted_close: str | None = None,
) -> HistoricalOHLCVBar:
    c = Currency(currency)
    return HistoricalOHLCVBar(
        symbol=Symbol(symbol),
        exchange_code=ExchangeCode(exchange),
        point_in_time=PointInTime(timestamp),
        timeframe=Timeframe(timeframe),
        open=Price(open_price, c),
        high=Price(high_price, c),
        low=Price(low_price, c),
        close=Price(close_price, c),
        volume=Quantity(volume),
        adjusted_close=Price(adjusted_close, c) if adjusted_close else None,
    )


def _make_query(
    symbol: str = "AAPL",
    exchange: str = "NASDAQ",
    timeframe: str = "1d",
    start: str = "2026-09-01T00:00:00Z",
    end: str = "2026-09-10T00:00:00Z",
) -> HistoricalMarketDataQuery:
    return HistoricalMarketDataQuery(
        symbol=Symbol(symbol),
        exchange_code=ExchangeCode(exchange),
        timeframe=Timeframe(timeframe),
        start=PointInTime(start),
        end=PointInTime(end),
    )


class StubSource(HistoricalMarketDataSource):
    def __init__(self, bars: tuple[HistoricalOHLCVBar, ...] | object = ()) -> None:
        self.bars = bars
        self.last_query: HistoricalMarketDataQuery | None = None

    def fetch_history(self, query: HistoricalMarketDataQuery) -> tuple[HistoricalOHLCVBar, ...]:
        self.last_query = query
        if isinstance(self.bars, Exception):
            raise self.bars
        return self.bars  # type: ignore[return-value]


class RecordingStore(HistoricalMarketDataStore):
    def __init__(self, return_count: int | None = None) -> None:
        self.stored_batches: list[tuple[HistoricalOHLCVBar, ...]] = []
        self.return_count = return_count

    def store(self, observations: tuple[HistoricalOHLCVBar, ...]) -> int:
        self.stored_batches.append(observations)
        if self.return_count is not None:
            return self.return_count
        return len(observations)


class FailingStore(HistoricalMarketDataStore):
    def __init__(self, error: Exception) -> None:
        self.error = error

    def store(self, observations: tuple[HistoricalOHLCVBar, ...]) -> int:
        raise self.error


def test_successful_ingestion() -> None:
    query = _make_query()
    bars = (
        _make_bar(timestamp="2026-09-01T00:00:00Z"),
        _make_bar(timestamp="2026-09-02T00:00:00Z"),
        _make_bar(timestamp="2026-09-03T00:00:00Z"),
    )
    source = StubSource(bars)
    store = RecordingStore()
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    result = use_case.execute(query)

    assert isinstance(result, HistoricalMarketDataIngestionResult)
    assert result.query == query
    assert result.acquired_count == 3
    assert result.stored_count == 3
    assert source.last_query == query
    assert store.stored_batches == [bars]


def test_empty_acquisition_does_not_call_store_and_returns_zero_result() -> None:
    query = _make_query()
    source = StubSource(())
    store = RecordingStore()
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    result = use_case.execute(query)

    assert result.query == query
    assert result.acquired_count == 0
    assert result.stored_count == 0
    assert len(store.stored_batches) == 0


def test_repeated_ingestion_invokes_store_deterministically() -> None:
    query = _make_query()
    bars = (
        _make_bar(timestamp="2026-09-01T00:00:00Z"),
        _make_bar(timestamp="2026-09-02T00:00:00Z"),
    )
    source = StubSource(bars)
    store = RecordingStore()
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    result_1 = use_case.execute(query)
    assert result_1.acquired_count == 2
    assert result_1.stored_count == 2

    result_2 = use_case.execute(query)
    assert result_2.acquired_count == 2
    assert result_2.stored_count == 2

    assert len(store.stored_batches) == 2
    assert store.stored_batches[0] == bars
    assert store.stored_batches[1] == bars


def test_fractional_timestamp_observations_bounded_and_ordered_correctly() -> None:
    query = _make_query(
        start="2026-09-15T09:30:00Z",
        end="2026-09-15T09:30:00.000001Z",
    )
    bars = (
        _make_bar(timestamp="2026-09-15T09:30:00Z"),
        _make_bar(timestamp="2026-09-15T09:30:00.000001Z"),
    )
    source = StubSource(bars)
    store = RecordingStore()
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    result = use_case.execute(query)

    assert result.acquired_count == 2
    assert result.stored_count == 2
    assert store.stored_batches[0] == bars


def test_query_boundary_observations_at_exact_endpoints_are_accepted() -> None:
    query = _make_query(
        start="2026-09-01T00:00:00Z",
        end="2026-09-03T00:00:00Z",
    )
    bars = (
        _make_bar(timestamp="2026-09-01T00:00:00Z"),
        _make_bar(timestamp="2026-09-03T00:00:00Z"),
    )
    source = StubSource(bars)
    store = RecordingStore()
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    result = use_case.execute(query)

    assert result.acquired_count == 2
    assert result.stored_count == 2


def test_source_failure_propagates_unwrapped() -> None:
    query = _make_query()
    source = StubSource(RuntimeError("External provider unavailable"))
    store = RecordingStore()
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    with pytest.raises(RuntimeError, match="External provider unavailable"):
        use_case.execute(query)

    assert len(store.stored_batches) == 0


def test_store_failure_propagates_unwrapped() -> None:
    query = _make_query()
    bars = (_make_bar(timestamp="2026-09-01T00:00:00Z"),)
    source = StubSource(bars)
    store = FailingStore(RuntimeError("Storage disk failure"))
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    with pytest.raises(RuntimeError, match="Storage disk failure"):
        use_case.execute(query)


def test_defensive_validation_rejects_non_tuple_source_return() -> None:
    query = _make_query()
    bars_list = [_make_bar(timestamp="2026-09-01T00:00:00Z")]
    source = StubSource(bars_list)
    store = RecordingStore()
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    with pytest.raises(
        HistoricalDataContractViolationError, match="must return a tuple of HistoricalOHLCVBar"
    ):
        use_case.execute(query)


def test_defensive_validation_rejects_non_bar_element() -> None:
    query = _make_query()
    source = StubSource(("not_a_bar",))
    store = RecordingStore()
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    with pytest.raises(
        HistoricalDataContractViolationError, match="must be a HistoricalOHLCVBar instance"
    ):
        use_case.execute(query)


def test_defensive_validation_rejects_mismatched_symbol() -> None:
    query = _make_query(symbol="AAPL")
    bars = (_make_bar(symbol="MSFT", timestamp="2026-09-01T00:00:00Z"),)
    source = StubSource(bars)
    store = RecordingStore()
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    with pytest.raises(
        HistoricalDataContractViolationError, match="symbol MSFT does not match query symbol AAPL"
    ):
        use_case.execute(query)


def test_defensive_validation_rejects_mismatched_exchange() -> None:
    query = _make_query(exchange="NASDAQ")
    bars = (_make_bar(exchange="NYSE", timestamp="2026-09-01T00:00:00Z"),)
    source = StubSource(bars)
    store = RecordingStore()
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    with pytest.raises(
        HistoricalDataContractViolationError,
        match="exchange NYSE does not match query exchange NASDAQ",
    ):
        use_case.execute(query)


def test_defensive_validation_rejects_mismatched_timeframe() -> None:
    query = _make_query(timeframe="1d")
    bars = (_make_bar(timeframe="1h", timestamp="2026-09-01T00:00:00Z"),)
    source = StubSource(bars)
    store = RecordingStore()
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    with pytest.raises(
        HistoricalDataContractViolationError, match="timeframe 1h does not match query timeframe 1d"
    ):
        use_case.execute(query)


def test_defensive_validation_rejects_observation_before_start() -> None:
    query = _make_query(start="2026-09-05T00:00:00Z", end="2026-09-10T00:00:00Z")
    bars = (_make_bar(timestamp="2026-09-01T00:00:00Z"),)
    source = StubSource(bars)
    store = RecordingStore()
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    with pytest.raises(HistoricalDataContractViolationError, match="outside query bounds"):
        use_case.execute(query)


def test_defensive_validation_rejects_observation_after_end() -> None:
    query = _make_query(start="2026-09-01T00:00:00Z", end="2026-09-05T00:00:00Z")
    bars = (_make_bar(timestamp="2026-09-06T00:00:00Z"),)
    source = StubSource(bars)
    store = RecordingStore()
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    with pytest.raises(HistoricalDataContractViolationError, match="outside query bounds"):
        use_case.execute(query)


def test_defensive_validation_rejects_unordered_observations() -> None:
    query = _make_query(start="2026-09-01T00:00:00Z", end="2026-09-10T00:00:00Z")
    bars = (
        _make_bar(timestamp="2026-09-03T00:00:00Z"),
        _make_bar(timestamp="2026-09-02T00:00:00Z"),
    )
    source = StubSource(bars)
    store = RecordingStore()
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    with pytest.raises(
        HistoricalDataContractViolationError, match="strictly ordered from oldest to newest"
    ):
        use_case.execute(query)


def test_defensive_validation_rejects_duplicate_timestamp_observations() -> None:
    query = _make_query(start="2026-09-01T00:00:00Z", end="2026-09-10T00:00:00Z")
    bars = (
        _make_bar(timestamp="2026-09-01T00:00:00Z"),
        _make_bar(timestamp="2026-09-01T00:00:00Z"),
    )
    source = StubSource(bars)
    store = RecordingStore()
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    with pytest.raises(
        HistoricalDataContractViolationError, match="strictly ordered from oldest to newest"
    ):
        use_case.execute(query)


def test_defensive_validation_rejects_store_returning_non_int() -> None:
    query = _make_query()
    bars = (_make_bar(timestamp="2026-09-01T00:00:00Z"),)
    source = StubSource(bars)
    store = RecordingStore(return_count="1")  # type: ignore[arg-type]
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    with pytest.raises(
        HistoricalDataContractViolationError,
        match="HistoricalMarketDataStore.store\\(\\) must return an integer count",
    ):
        use_case.execute(query)


def test_defensive_validation_rejects_store_returning_boolean() -> None:
    query = _make_query()
    bars = (_make_bar(timestamp="2026-09-01T00:00:00Z"),)
    source = StubSource(bars)
    store = RecordingStore(return_count=True)  # type: ignore[arg-type]
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    with pytest.raises(
        HistoricalDataContractViolationError,
        match="HistoricalMarketDataStore.store\\(\\) must return an integer count",
    ):
        use_case.execute(query)


def test_defensive_validation_rejects_store_returning_partial_count() -> None:
    query = _make_query()
    bars = (
        _make_bar(timestamp="2026-09-01T00:00:00Z"),
        _make_bar(timestamp="2026-09-02T00:00:00Z"),
    )
    source = StubSource(bars)
    store = RecordingStore(return_count=1)
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    with pytest.raises(
        HistoricalDataContractViolationError,
        match="returned count 1, expected 2 for the complete batch",
    ):
        use_case.execute(query)


def test_defensive_validation_rejects_store_returning_excess_count() -> None:
    query = _make_query()
    bars = (_make_bar(timestamp="2026-09-01T00:00:00Z"),)
    source = StubSource(bars)
    store = RecordingStore(return_count=5)
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    with pytest.raises(
        HistoricalDataContractViolationError,
        match="returned count 5, expected 1 for the complete batch",
    ):
        use_case.execute(query)


def test_defensive_validation_rejects_store_returning_negative_count() -> None:
    query = _make_query()
    bars = (_make_bar(timestamp="2026-09-01T00:00:00Z"),)
    source = StubSource(bars)
    store = RecordingStore(return_count=-1)
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    with pytest.raises(
        HistoricalDataContractViolationError,
        match="returned count -1, expected 1 for the complete batch",
    ):
        use_case.execute(query)


def test_use_case_validates_constructor_arguments() -> None:
    source = StubSource(())
    store = RecordingStore()

    with pytest.raises(TypeError, match="source cannot be None"):
        IngestHistoricalMarketDataUseCase(source=None, store=store)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be a HistoricalMarketDataSource"):
        IngestHistoricalMarketDataUseCase(source="invalid", store=store)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="store cannot be None"):
        IngestHistoricalMarketDataUseCase(source=source, store=None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be a HistoricalMarketDataStore"):
        IngestHistoricalMarketDataUseCase(source=source, store="invalid")  # type: ignore[arg-type]


def test_use_case_validates_execute_argument() -> None:
    source = StubSource(())
    store = RecordingStore()
    use_case = IngestHistoricalMarketDataUseCase(source=source, store=store)

    with pytest.raises(TypeError, match="query cannot be None"):
        use_case.execute(None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be a HistoricalMarketDataQuery"):
        use_case.execute("invalid")  # type: ignore[arg-type]


def test_ingestion_result_immutability_and_validation() -> None:
    query = _make_query()
    result = HistoricalMarketDataIngestionResult(
        query=query,
        acquired_count=5,
        stored_count=5,
    )

    assert result.query == query
    assert result.acquired_count == 5
    assert result.stored_count == 5

    with pytest.raises(FrozenInstanceError):
        result.stored_count = 10  # type: ignore[misc]

    with pytest.raises(TypeError, match="query cannot be None"):
        HistoricalMarketDataIngestionResult(query=None, acquired_count=0, stored_count=0)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="query must be a HistoricalMarketDataQuery"):
        HistoricalMarketDataIngestionResult(query="invalid", acquired_count=0, stored_count=0)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="acquired_count must be an integer"):
        HistoricalMarketDataIngestionResult(query=query, acquired_count="5", stored_count=0)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="acquired_count must be an integer"):
        HistoricalMarketDataIngestionResult(query=query, acquired_count=True, stored_count=0)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="acquired_count must be non-negative"):
        HistoricalMarketDataIngestionResult(query=query, acquired_count=-1, stored_count=0)

    with pytest.raises(TypeError, match="stored_count must be an integer"):
        HistoricalMarketDataIngestionResult(query=query, acquired_count=0, stored_count="5")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="stored_count must be an integer"):
        HistoricalMarketDataIngestionResult(query=query, acquired_count=0, stored_count=False)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="stored_count must be non-negative"):
        HistoricalMarketDataIngestionResult(query=query, acquired_count=0, stored_count=-1)
