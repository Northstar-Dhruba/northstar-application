"""Tests for the concrete historical snapshot evaluator."""

from __future__ import annotations

import pytest
from northstar_core.domain.exchange import Exchange
from northstar_core.domain.instrument import Instrument
from northstar_core.domain.listing import Listing
from northstar_core.domain.value_objects import ListingStatus, Tradability
from northstar_core.foundation.value_objects import (
    Currency,
    ExchangeCode,
    PointInTime,
    Price,
    Quantity,
    Symbol,
    Timeframe,
)
from northstar_core.market_data import HistoricalOHLCVBar, HistoricalReplaySnapshot
from northstar_core.strategy import AssetAnalysisGenerator, Strategy, StrategyIdentity

from northstar_application.application_services import (
    AnalyzeAssetResult,
    HistoricalSnapshotEvaluatorService,
)
from northstar_application.ports import ListingResolver


def _listing() -> Listing:
    return Listing(
        instrument=Instrument(Symbol("AAPL"), "Apple Inc.", "Equity"),
        exchange=Exchange(ExchangeCode("NASDAQ"), "NASDAQ"),
        currency=Currency("USD"),
        listing_status=ListingStatus("Active"),
        tradability=Tradability("Permitted"),
        description="Apple Inc.",
    )


class StubListingResolver(ListingResolver):
    def __init__(self) -> None:
        self.calls: list[tuple[Symbol, ExchangeCode, PointInTime]] = []

    def resolve_listing(
        self,
        symbol: Symbol,
        exchange_code: ExchangeCode,
        as_of: PointInTime,
    ) -> Listing:
        self.calls.append((symbol, exchange_code, as_of))
        return _listing()


def _snapshot(timeframe: str = "1d", count: int = 20) -> HistoricalReplaySnapshot:
    currency = Currency("USD")
    bars = tuple(
        HistoricalOHLCVBar(
            Symbol("AAPL"),
            ExchangeCode("NASDAQ"),
            PointInTime(f"2026-01-01T{10 + index // 60:02d}:{index % 60:02d}:00Z"),
            Timeframe(timeframe),
            Price(str(100 + index), currency),
            Price(str(105 + index), currency),
            Price(str(95 + index), currency),
            Price(str(102 + index), currency),
            Quantity(str(1000 + index)),
        )
        for index in range(count)
    )
    return HistoricalReplaySnapshot(bars[-1].point_in_time, bars)


def _evaluator(resolver: StubListingResolver) -> HistoricalSnapshotEvaluatorService:
    return HistoricalSnapshotEvaluatorService(
        listing_resolver=resolver,
        strategy=Strategy(StrategyIdentity("historical-test")),
        analysis_generator=AssetAnalysisGenerator(),
    )


def test_evaluator_builds_existing_analysis_result_from_historical_facts() -> None:
    resolver = StubListingResolver()
    snapshot = _snapshot()

    result = _evaluator(resolver).evaluate(snapshot)

    assert isinstance(result, AnalyzeAssetResult)
    assert result.market_observation_context.observed_at == snapshot.replay_instant
    assert result.market_observation_context.latest_price == snapshot.observations[-1].close
    assert result.market_observation_context.previous_close == snapshot.observations[-2].close
    assert len(result.market_observation_context.recent_closes) == 20
    assert len(result.market_observation_context.recent_volumes) == 20
    assert resolver.calls == [(Symbol("AAPL"), ExchangeCode("NASDAQ"), snapshot.replay_instant)]
    assert result.recommendation.action.value in {"BUY", "HOLD", "SELL"}
    assert result.explanation.recommendation is result.recommendation


def test_evaluator_uses_no_live_market_observation_source() -> None:
    resolver = StubListingResolver()
    evaluator = _evaluator(resolver)

    result = evaluator.evaluate(_snapshot())

    assert result.market_observation_context.observed_at.value == "2026-01-01T10:19:00Z"


def test_evaluator_rejects_unsupported_timeframe() -> None:
    with pytest.raises(ValueError, match="supports only daily bars"):
        _evaluator(StubListingResolver()).evaluate(_snapshot(timeframe="1h"))


def test_evaluator_requires_twenty_daily_bars() -> None:
    with pytest.raises(ValueError, match="at least 20 daily bars"):
        _evaluator(StubListingResolver()).evaluate(_snapshot(count=19))


def test_evaluator_rejects_incompatible_identity() -> None:
    snapshot = _snapshot()
    currency = Currency("USD")
    incompatible = HistoricalOHLCVBar(
        Symbol("MSFT"),
        ExchangeCode("NASDAQ"),
        PointInTime("2026-01-01T10:19:00Z"),
        Timeframe("1d"),
        Price("100", currency),
        Price("105", currency),
        Price("95", currency),
        Price("102", currency),
        Quantity("1000"),
    )
    mixed = HistoricalReplaySnapshot(
        snapshot.replay_instant, snapshot.observations[:-1] + (incompatible,)
    )

    with pytest.raises(ValueError, match="incompatible market identities"):
        _evaluator(StubListingResolver()).evaluate(mixed)


def test_evaluator_rejects_missing_or_invalid_listing_reference() -> None:
    class InvalidListingResolver(ListingResolver):
        def resolve_listing(
            self,
            symbol: Symbol,
            exchange_code: ExchangeCode,
            as_of: PointInTime,
        ) -> Listing:
            return "invalid"  # type: ignore[return-value]

    with pytest.raises(TypeError, match="must return a Listing"):
        HistoricalSnapshotEvaluatorService(
            InvalidListingResolver(),
            Strategy(StrategyIdentity("historical-test")),
            AssetAnalysisGenerator(),
        ).evaluate(_snapshot())
