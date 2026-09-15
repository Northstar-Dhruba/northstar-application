"""Application use case for acquiring and ingesting historical market data."""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.market_data import HistoricalOHLCVBar

from northstar_application.ports import (
    HistoricalMarketDataQuery,
    HistoricalMarketDataSource,
    HistoricalMarketDataStore,
)


class HistoricalDataContractViolationError(ValueError):
    """Raised when an external historical data source or store violates the port contract."""


@dataclass(frozen=True, slots=True)
class HistoricalMarketDataIngestionResult:
    """Outcome of one historical market data ingestion execution."""

    query: HistoricalMarketDataQuery
    acquired_count: int
    stored_count: int

    def __post_init__(self) -> None:
        if self.query is None:
            raise TypeError("HistoricalMarketDataIngestionResult query cannot be None.")
        if not isinstance(self.query, HistoricalMarketDataQuery):
            raise TypeError(
                "HistoricalMarketDataIngestionResult query must be a HistoricalMarketDataQuery."
            )
        if not isinstance(self.acquired_count, int) or isinstance(self.acquired_count, bool):
            raise TypeError(
                "HistoricalMarketDataIngestionResult acquired_count must be an integer."
            )
        if self.acquired_count < 0:
            raise ValueError(
                "HistoricalMarketDataIngestionResult acquired_count must be non-negative."
            )
        if not isinstance(self.stored_count, int) or isinstance(self.stored_count, bool):
            raise TypeError("HistoricalMarketDataIngestionResult stored_count must be an integer.")
        if self.stored_count < 0:
            raise ValueError(
                "HistoricalMarketDataIngestionResult stored_count must be non-negative."
            )


class IngestHistoricalMarketDataUseCase:
    """Coordinates acquiring historical observations and persisting them to research storage."""

    def __init__(
        self,
        source: HistoricalMarketDataSource,
        store: HistoricalMarketDataStore,
    ) -> None:
        if source is None:
            raise TypeError("IngestHistoricalMarketDataUseCase source cannot be None.")
        if not isinstance(source, HistoricalMarketDataSource):
            raise TypeError(
                "IngestHistoricalMarketDataUseCase source must be a HistoricalMarketDataSource."
            )
        if store is None:
            raise TypeError("IngestHistoricalMarketDataUseCase store cannot be None.")
        if not isinstance(store, HistoricalMarketDataStore):
            raise TypeError(
                "IngestHistoricalMarketDataUseCase store must be a HistoricalMarketDataStore."
            )

        self._source = source
        self._store = store

    def execute(self, query: HistoricalMarketDataQuery) -> HistoricalMarketDataIngestionResult:
        """Fetch observations from the source, validate contract constraints, and persist."""
        if query is None:
            raise TypeError("IngestHistoricalMarketDataUseCase query cannot be None.")
        if not isinstance(query, HistoricalMarketDataQuery):
            raise TypeError(
                "IngestHistoricalMarketDataUseCase query must be a HistoricalMarketDataQuery."
            )

        observations = self._source.fetch_history(query)
        self._validate_observations(observations, query)

        if not observations:
            return HistoricalMarketDataIngestionResult(
                query=query,
                acquired_count=0,
                stored_count=0,
            )

        stored = self._store.store(observations)
        if not isinstance(stored, int) or isinstance(stored, bool):
            raise HistoricalDataContractViolationError(
                "HistoricalMarketDataStore.store() must return an integer count."
            )
        if stored != len(observations):
            raise HistoricalDataContractViolationError(
                f"HistoricalMarketDataStore.store() returned count {stored}, "
                f"expected {len(observations)} for the complete batch."
            )

        return HistoricalMarketDataIngestionResult(
            query=query,
            acquired_count=len(observations),
            stored_count=stored,
        )

    @staticmethod
    def _validate_observations(observations: object, query: HistoricalMarketDataQuery) -> None:
        if not isinstance(observations, tuple):
            raise HistoricalDataContractViolationError(
                "HistoricalMarketDataSource must return a tuple of HistoricalOHLCVBar."
            )

        for bar in observations:
            if not isinstance(bar, HistoricalOHLCVBar):
                raise HistoricalDataContractViolationError(
                    "Historical observation must be a HistoricalOHLCVBar instance."
                )
            if bar.symbol != query.symbol:
                raise HistoricalDataContractViolationError(
                    f"Historical observation symbol {bar.symbol} "
                    f"does not match query symbol {query.symbol}."
                )
            if bar.exchange_code != query.exchange_code:
                raise HistoricalDataContractViolationError(
                    f"Historical observation exchange {bar.exchange_code} "
                    f"does not match query exchange {query.exchange_code}."
                )
            if bar.timeframe != query.timeframe:
                raise HistoricalDataContractViolationError(
                    f"Historical observation timeframe {bar.timeframe} "
                    f"does not match query timeframe {query.timeframe}."
                )
            if (
                query.start.compare(bar.point_in_time) > 0
                or query.end.compare(bar.point_in_time) < 0
            ):
                raise HistoricalDataContractViolationError(
                    f"Historical observation point in time {bar.point_in_time} "
                    f"is outside query bounds [{query.start}, {query.end}]."
                )

        for i in range(len(observations) - 1):
            if observations[i].point_in_time.compare(observations[i + 1].point_in_time) >= 0:
                raise HistoricalDataContractViolationError(
                    "Historical observations must be strictly ordered from oldest to newest."
                )
