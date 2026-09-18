"""Tests for historical research evaluation orchestration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

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
from northstar_core.market_data import HistoricalReplaySnapshot
from northstar_core.strategy import (
    AssetAnalysisGenerator,
    ExplanationReason,
    MarketObservationContext,
    RecommendationExplanation,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services import (
    AnalyzeAssetResult,
    EvaluateHistoricalResearchUseCase,
    HistoricalResearchEvaluation,
)
from northstar_application.ports import HistoricalSnapshotEvaluator


def _context() -> MarketObservationContext:
    currency = Currency("USD")
    listing = Listing(
        instrument=Instrument(Symbol("AAPL"), "Apple Inc.", "Equity"),
        exchange=Exchange(ExchangeCode("NASDAQ"), "NASDAQ"),
        currency=currency,
        listing_status=ListingStatus("Active"),
        tradability=Tradability("Permitted"),
        description="Apple Inc.",
    )
    return MarketObservationContext(
        listing=listing,
        observed_at=PointInTime("2026-01-01T10:00:00Z"),
        latest_price=Price("130", currency),
        previous_close=Price("120", currency),
        latest_volume=Quantity("200"),
        daily_high=Price("135", currency),
        daily_low=Price("115", currency),
        recent_closes=tuple(Price("100", currency) for _ in range(20)),
        recent_volumes=tuple(Quantity("100") for _ in range(20)),
    )


def _result(replay_instant: str) -> AnalyzeAssetResult:
    context = _context()
    analysis = AssetAnalysisGenerator().generate(context)
    recommendation = Strategy(StrategyIdentity("historical-test")).evaluate(analysis)
    explanation = RecommendationExplanation(
        recommendation=recommendation,
        reasons=(ExplanationReason("Deterministic historical research result."),),
    )
    return AnalyzeAssetResult(recommendation, explanation, context)


class StubEvaluator(HistoricalSnapshotEvaluator):
    def __init__(self) -> None:
        self.snapshots: list[HistoricalReplaySnapshot] = []

    def evaluate(self, snapshot: HistoricalReplaySnapshot) -> AnalyzeAssetResult:
        self.snapshots.append(snapshot)
        return _result(snapshot.replay_instant.value)


class FailingEvaluator(HistoricalSnapshotEvaluator):
    def evaluate(self, snapshot: HistoricalReplaySnapshot) -> AnalyzeAssetResult:
        raise RuntimeError("historical evaluator unavailable")


def _snapshot(timestamp: str) -> HistoricalReplaySnapshot:
    from northstar_core.market_data import HistoricalOHLCVBar

    currency = Currency("USD")
    bar = HistoricalOHLCVBar(
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
    return HistoricalReplaySnapshot(PointInTime(timestamp), (bar,))


def test_empty_replay_returns_empty_evaluation_result() -> None:
    evaluator = StubEvaluator()

    result = EvaluateHistoricalResearchUseCase(evaluator).execute(())

    assert result == ()
    assert evaluator.snapshots == []


def test_one_snapshot_produces_one_evaluation_with_replay_instant() -> None:
    snapshot = _snapshot("2026-01-01T10:00:00Z")
    evaluator = StubEvaluator()

    result = EvaluateHistoricalResearchUseCase(evaluator).execute((snapshot,))

    assert len(result) == 1
    assert isinstance(result[0], HistoricalResearchEvaluation)
    assert result[0].replay_instant == snapshot.replay_instant
    assert result[0].result.recommendation is not None
    assert evaluator.snapshots == [snapshot]


def test_multiple_snapshots_are_evaluated_in_replay_order() -> None:
    snapshots = (
        _snapshot("2026-01-01T10:00:00Z"),
        _snapshot("2026-01-01T10:05:00Z"),
    )
    evaluator = StubEvaluator()

    result = EvaluateHistoricalResearchUseCase(evaluator).execute(snapshots)

    assert [item.replay_instant.value for item in result] == [
        "2026-01-01T10:00:00Z",
        "2026-01-01T10:05:00Z",
    ]
    assert evaluator.snapshots == list(snapshots)


def test_same_input_produces_deterministic_result() -> None:
    snapshot = _snapshot("2026-01-01T10:00:00Z")

    first = EvaluateHistoricalResearchUseCase(StubEvaluator()).execute((snapshot,))
    second = EvaluateHistoricalResearchUseCase(StubEvaluator()).execute((snapshot,))

    assert first[0].replay_instant == second[0].replay_instant
    assert first[0].result.recommendation.action == second[0].result.recommendation.action
    assert first[0].result.explanation.reasons == second[0].result.explanation.reasons


def test_evaluator_receives_only_the_supplied_replay_snapshot() -> None:
    snapshot = _snapshot("2026-01-01T10:00:00Z")
    evaluator = StubEvaluator()

    EvaluateHistoricalResearchUseCase(evaluator).execute((snapshot,))

    assert evaluator.snapshots == [snapshot]
    assert all(
        bar.point_in_time.compare(snapshot.replay_instant) <= 0
        for bar in evaluator.snapshots[0].observations
    )


def test_evaluator_failure_propagates() -> None:
    with pytest.raises(RuntimeError, match="historical evaluator unavailable"):
        EvaluateHistoricalResearchUseCase(FailingEvaluator()).execute(
            (_snapshot("2026-01-01T10:00:00Z"),)
        )


def test_previous_evaluation_results_remain_immutable() -> None:
    snapshots = (
        _snapshot("2026-01-01T10:00:00Z"),
        _snapshot("2026-01-01T10:05:00Z"),
    )
    result = EvaluateHistoricalResearchUseCase(StubEvaluator()).execute(snapshots)

    with pytest.raises(FrozenInstanceError):
        result[0].replay_instant = PointInTime("2026-01-01T11:00:00Z")  # type: ignore[misc]
    assert result[0].replay_instant.value == "2026-01-01T10:00:00Z"


def test_use_case_rejects_unordered_snapshots() -> None:
    snapshots = (
        _snapshot("2026-01-01T10:05:00Z"),
        _snapshot("2026-01-01T10:00:00Z"),
    )

    with pytest.raises(ValueError, match="chronologically ordered"):
        EvaluateHistoricalResearchUseCase(StubEvaluator()).execute(snapshots)


def test_use_case_rejects_invalid_evaluator_result() -> None:
    class InvalidEvaluator(HistoricalSnapshotEvaluator):
        def evaluate(self, snapshot: HistoricalReplaySnapshot) -> AnalyzeAssetResult:
            return "invalid"  # type: ignore[return-value]

    with pytest.raises(TypeError, match="must return an AnalyzeAssetResult"):
        EvaluateHistoricalResearchUseCase(InvalidEvaluator()).execute(
            (_snapshot("2026-01-01T10:00:00Z"),)
        )
