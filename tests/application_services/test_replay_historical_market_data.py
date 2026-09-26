"""Tests for deterministic historical replay orchestration."""

from __future__ import annotations

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
    ReplayHistoricalMarketDataUseCase,
)
from northstar_application.ports import (
    HistoricalMarketDataQuery,
    HistoricalMarketDataRepository,
)


def _query(
    start: str = "2026-01-01T10:00:00Z",
    end: str = "2026-01-01T10:10:00Z",
) -> HistoricalMarketDataQuery:
    return HistoricalMarketDataQuery(
        Symbol("AAPL"),
        ExchangeCode("NASDAQ"),
        Timeframe("1m"),
        PointInTime(start),
        PointInTime(end),
    )


def _bar(timestamp: str) -> HistoricalOHLCVBar:
    currency = Currency("USD")
    return HistoricalOHLCVBar(
        Symbol("AAPL"),
        ExchangeCode("NASDAQ"),
        PointInTime(timestamp),
        Timeframe("1m"),
        Price("100", currency),
        Price("105", currency),
        Price("95", currency),
        Price("102", currency),
        Quantity("1000"),
    )


class StubRepository(HistoricalMarketDataRepository):
    def __init__(self, observations: object) -> None:
        self.observations = observations
        self.calls: list[HistoricalMarketDataQuery] = []

    def get_history(self, query: HistoricalMarketDataQuery) -> tuple[HistoricalOHLCVBar, ...]:
        self.calls.append(query)
        return self.observations  # type: ignore[return-value]


class FailingRepository(HistoricalMarketDataRepository):
    def get_history(self, query: HistoricalMarketDataQuery) -> tuple[HistoricalOHLCVBar, ...]:
        raise RuntimeError("repository unavailable")


def test_empty_history_returns_empty_tuple() -> None:
    query = _query()
    repository = StubRepository(())

    result = ReplayHistoricalMarketDataUseCase(repository).execute(query)

    assert result == ()
    assert repository.calls == [query]


def test_one_bar_produces_one_snapshot() -> None:
    bar = _bar("2026-01-01T10:00:00Z")

    snapshots = ReplayHistoricalMarketDataUseCase(StubRepository((bar,))).execute(_query())

    assert len(snapshots) == 1
    assert snapshots[0].replay_instant == bar.point_in_time
    assert snapshots[0].observations == (bar,)


def test_bar_at_replay_instant_is_included() -> None:
    bar = _bar("2026-01-01T10:05:00Z")

    snapshots = ReplayHistoricalMarketDataUseCase(StubRepository((bar,))).execute(
        _query(start="2026-01-01T10:05:00Z", end="2026-01-01T10:05:00Z")
    )

    assert snapshots[0].observations == (bar,)


def test_chronological_bars_produce_cumulative_snapshots() -> None:
    bars = (
        _bar("2026-01-01T10:00:00Z"),
        _bar("2026-01-01T10:05:00Z"),
        _bar("2026-01-01T10:10:00Z"),
    )

    snapshots = ReplayHistoricalMarketDataUseCase(StubRepository(bars)).execute(_query())

    assert [snapshot.replay_instant.value for snapshot in snapshots] == [
        "2026-01-01T10:00:00Z",
        "2026-01-01T10:05:00Z",
        "2026-01-01T10:10:00Z",
    ]
    assert [len(snapshot.observations) for snapshot in snapshots] == [1, 2, 3]
    assert snapshots[0].observations == (bars[0],)
    assert snapshots[1].observations == bars[:2]
    assert snapshots[2].observations == bars


def test_snapshots_have_unique_increasing_replay_instants() -> None:
    bars = (
        _bar("2026-01-01T10:00:00Z"),
        _bar("2026-01-01T10:05:00Z"),
        _bar("2026-01-01T10:10:00Z"),
    )

    snapshots = ReplayHistoricalMarketDataUseCase(StubRepository(bars)).execute(_query())

    assert len({snapshot.replay_instant for snapshot in snapshots}) == len(snapshots)
    assert all(
        snapshots[index].replay_instant.compare(snapshots[index + 1].replay_instant) < 0
        for index in range(len(snapshots) - 1)
    )


def test_previous_snapshots_remain_unchanged_as_replay_advances() -> None:
    bars = (
        _bar("2026-01-01T10:00:00Z"),
        _bar("2026-01-01T10:05:00Z"),
    )

    snapshots = ReplayHistoricalMarketDataUseCase(StubRepository(bars)).execute(_query())

    assert snapshots[0].observations == (bars[0],)
    assert snapshots[0].observations is not snapshots[1].observations


def test_future_observations_are_rejected() -> None:
    future_bar = _bar("2026-01-01T10:11:00Z")

    with pytest.raises(HistoricalDataContractViolationError, match="outside query bounds"):
        ReplayHistoricalMarketDataUseCase(StubRepository((future_bar,))).execute(_query())


def test_repository_is_called_exactly_once_with_provided_query() -> None:
    query = _query()
    repository = StubRepository(())

    ReplayHistoricalMarketDataUseCase(repository).execute(query)

    assert repository.calls == [query]


def test_repository_operational_failure_propagates() -> None:
    with pytest.raises(RuntimeError, match="repository unavailable"):
        ReplayHistoricalMarketDataUseCase(FailingRepository()).execute(_query())


def test_malformed_repository_result_is_rejected() -> None:
    with pytest.raises(HistoricalDataContractViolationError, match="must return a tuple"):
        ReplayHistoricalMarketDataUseCase(StubRepository([])).execute(_query())


def test_out_of_order_repository_result_is_rejected() -> None:
    bars = (
        _bar("2026-01-01T10:05:00Z"),
        _bar("2026-01-01T10:00:00Z"),
    )

    with pytest.raises(HistoricalDataContractViolationError, match="ordered oldest to newest"):
        ReplayHistoricalMarketDataUseCase(StubRepository(bars)).execute(_query())


def test_duplicate_logical_observations_are_rejected() -> None:
    bar = _bar("2026-01-01T10:00:00Z")

    with pytest.raises(HistoricalDataContractViolationError, match="duplicate logical bars"):
        ReplayHistoricalMarketDataUseCase(StubRepository((bar, bar))).execute(_query())


def test_use_case_validates_repository_constructor_argument() -> None:
    with pytest.raises(TypeError, match="repository cannot be None"):
        ReplayHistoricalMarketDataUseCase(None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be a HistoricalMarketDataRepository"):
        ReplayHistoricalMarketDataUseCase("invalid")  # type: ignore[arg-type]
