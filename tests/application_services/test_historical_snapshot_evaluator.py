"""Tests for the concrete historical snapshot evaluator."""

from __future__ import annotations

import inspect

import pytest
from northstar_core.domain.value_objects import ListingReference
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

_USD = Currency("USD")
_EUR = Currency("EUR")


def _bar(
    index: int,
    *,
    currency: Currency = _USD,
    timeframe: str = "1d",
    symbol: str = "AAPL",
    exchange_code: str = "NASDAQ",
) -> HistoricalOHLCVBar:
    return HistoricalOHLCVBar(
        Symbol(symbol),
        ExchangeCode(exchange_code),
        PointInTime(f"2026-01-01T{10 + index // 60:02d}:{index % 60:02d}:00Z"),
        Timeframe(timeframe),
        Price(str(100 + index), currency),
        Price(str(105 + index), currency),
        Price(str(95 + index), currency),
        Price(str(102 + index), currency),
        Quantity(str(1000 + index)),
    )


def _snapshot(timeframe: str = "1d", count: int = 20) -> HistoricalReplaySnapshot:
    bars = tuple(_bar(index, timeframe=timeframe) for index in range(count))
    return HistoricalReplaySnapshot(bars[-1].point_in_time, bars)


def _evaluator() -> HistoricalSnapshotEvaluatorService:
    return HistoricalSnapshotEvaluatorService(
        strategy=Strategy(StrategyIdentity("historical-test")),
        analysis_generator=AssetAnalysisGenerator(),
    )


# ---------------------------------------------------------------------------
# Dependency shape
# ---------------------------------------------------------------------------


def test_evaluator_has_no_listing_resolver_dependency() -> None:
    parameters = inspect.signature(HistoricalSnapshotEvaluatorService.__init__).parameters

    assert set(parameters) == {"self", "strategy", "analysis_generator"}
    assert not hasattr(_evaluator(), "_listing_resolver")


def test_evaluator_constructs_from_strategy_and_generator_only() -> None:
    evaluator = HistoricalSnapshotEvaluatorService(
        Strategy(StrategyIdentity("historical-test")),
        AssetAnalysisGenerator(),
    )

    assert isinstance(evaluator.evaluate(_snapshot()), AnalyzeAssetResult)


# ---------------------------------------------------------------------------
# Identity derived from historical observations
# ---------------------------------------------------------------------------


def test_listing_reference_is_derived_from_snapshot_symbol_and_exchange_code() -> None:
    snapshot = _snapshot()

    result = _evaluator().evaluate(snapshot)

    reference = result.market_observation_context.listing_reference
    assert reference == ListingReference(Symbol("AAPL"), ExchangeCode("NASDAQ"))
    assert reference.symbol == snapshot.observations[-1].symbol
    assert reference.exchange_code == snapshot.observations[-1].exchange_code


def test_listing_reference_reaches_the_asset_analysis() -> None:
    result = _evaluator().evaluate(_snapshot())

    assert result.recommendation.asset_analysis.listing_reference == ListingReference(
        Symbol("AAPL"), ExchangeCode("NASDAQ")
    )


def test_evaluator_does_not_carry_a_listing_entity() -> None:
    result = _evaluator().evaluate(_snapshot())

    assert not hasattr(result.market_observation_context, "listing")
    assert not hasattr(result.recommendation.asset_analysis, "listing")


# ---------------------------------------------------------------------------
# Historical fact mapping
# ---------------------------------------------------------------------------


def test_evaluator_builds_existing_analysis_result_from_historical_facts() -> None:
    snapshot = _snapshot()

    result = _evaluator().evaluate(snapshot)

    context = result.market_observation_context
    assert isinstance(result, AnalyzeAssetResult)
    assert context.latest_price == snapshot.observations[-1].close
    assert context.previous_close == snapshot.observations[-2].close
    assert context.latest_volume == snapshot.observations[-1].volume
    assert context.daily_high == snapshot.observations[-1].high
    assert context.daily_low == snapshot.observations[-1].low
    assert len(context.recent_closes) == 20
    assert len(context.recent_volumes) == 20
    assert result.recommendation.action.value in {"BUY", "HOLD", "SELL"}
    assert result.explanation.recommendation is result.recommendation


def test_observed_at_is_the_latest_visible_bar_completion_instant() -> None:
    snapshot = _snapshot()

    result = _evaluator().evaluate(snapshot)

    assert result.market_observation_context.observed_at == snapshot.observations[-1].point_in_time
    assert result.market_observation_context.observed_at.value == "2026-01-01T10:19:00Z"


def test_recent_closes_and_volumes_include_the_latest_bar() -> None:
    snapshot = _snapshot()

    context = _evaluator().evaluate(snapshot).market_observation_context

    assert context.recent_closes[-1] == snapshot.observations[-1].close
    assert context.recent_volumes[-1] == snapshot.observations[-1].volume
    assert context.recent_closes == tuple(bar.close for bar in snapshot.observations[-20:])
    assert context.recent_volumes == tuple(bar.volume for bar in snapshot.observations[-20:])


def test_evaluator_uses_no_live_market_observation_source() -> None:
    result = _evaluator().evaluate(_snapshot())

    assert result.market_observation_context.observed_at.value == "2026-01-01T10:19:00Z"


def test_repeated_evaluation_is_deterministic() -> None:
    snapshot = _snapshot()

    first = _evaluator().evaluate(snapshot)
    second = _evaluator().evaluate(snapshot)

    assert first.market_observation_context == second.market_observation_context
    assert first.recommendation.action == second.recommendation.action
    assert first.recommendation.asset_analysis == second.recommendation.asset_analysis


def test_result_remains_an_analyze_asset_result() -> None:
    result = _evaluator().evaluate(_snapshot())

    assert isinstance(result, AnalyzeAssetResult)
    assert result.recommendation is not None
    assert result.explanation is not None
    assert result.market_observation_context is not None


# ---------------------------------------------------------------------------
# Preserved rejection rules
# ---------------------------------------------------------------------------


def test_evaluator_rejects_unsupported_timeframe() -> None:
    with pytest.raises(ValueError, match="supports only daily bars"):
        _evaluator().evaluate(_snapshot(timeframe="1h"))


def test_evaluator_requires_twenty_daily_bars() -> None:
    with pytest.raises(ValueError, match="at least 20 daily bars"):
        _evaluator().evaluate(_snapshot(count=19))


def test_evaluator_rejects_incompatible_symbol() -> None:
    snapshot = _snapshot()
    incompatible = HistoricalOHLCVBar(
        Symbol("MSFT"),
        ExchangeCode("NASDAQ"),
        PointInTime("2026-01-01T10:19:00Z"),
        Timeframe("1d"),
        Price("100", _USD),
        Price("105", _USD),
        Price("95", _USD),
        Price("102", _USD),
        Quantity("1000"),
    )
    mixed = HistoricalReplaySnapshot(
        snapshot.replay_instant, snapshot.observations[:-1] + (incompatible,)
    )

    with pytest.raises(ValueError, match="incompatible market identities"):
        _evaluator().evaluate(mixed)


def test_evaluator_rejects_incompatible_exchange_code() -> None:
    snapshot = _snapshot()
    incompatible = HistoricalOHLCVBar(
        Symbol("AAPL"),
        ExchangeCode("NYSE"),
        PointInTime("2026-01-01T10:19:00Z"),
        Timeframe("1d"),
        Price("100", _USD),
        Price("105", _USD),
        Price("95", _USD),
        Price("102", _USD),
        Quantity("1000"),
    )
    mixed = HistoricalReplaySnapshot(
        snapshot.replay_instant, snapshot.observations[:-1] + (incompatible,)
    )

    with pytest.raises(ValueError, match="incompatible market identities"):
        _evaluator().evaluate(mixed)


def test_evaluator_rejects_snapshot_of_wrong_type() -> None:
    with pytest.raises(TypeError, match="must be a HistoricalReplaySnapshot"):
        _evaluator().evaluate("snapshot")


def test_evaluator_rejects_none_snapshot() -> None:
    with pytest.raises(TypeError, match="snapshot cannot be None"):
        _evaluator().evaluate(None)


# ---------------------------------------------------------------------------
# Currency invariant is owned by Core
# ---------------------------------------------------------------------------


def test_mixed_currency_analysis_window_is_rejected() -> None:
    bars = tuple(_bar(index) for index in range(19)) + (_bar(19, currency=_EUR),)
    snapshot = HistoricalReplaySnapshot(bars[-1].point_in_time, bars)

    with pytest.raises(ValueError, match="must match the market observation currency"):
        _evaluator().evaluate(snapshot)


def test_cumulative_snapshot_may_span_a_historical_currency_transition() -> None:
    legacy = tuple(_bar(index, currency=_EUR) for index in range(10))
    current = tuple(_bar(index) for index in range(10, 30))
    snapshot = HistoricalReplaySnapshot(current[-1].point_in_time, legacy + current)

    result = _evaluator().evaluate(snapshot)

    assert isinstance(result, AnalyzeAssetResult)
    assert result.market_observation_context.latest_price.currency == _USD
