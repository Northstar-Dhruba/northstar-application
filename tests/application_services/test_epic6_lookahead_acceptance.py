"""Epic 6 acceptance: historical decisions are isolated from future observations.

The defining Epic 6 invariant is that a decision produced at instant T depends
only on observations completed at or before T. Appending later observations to
the underlying historical data must never change a decision already produced
from the original visible window.

These tests drive the real pipeline -- repository, replay, snapshot evaluation
and historical research evaluation -- rather than hand-built snapshots, so the
invariant is proven where it actually has to hold. Outcome measurement is
deliberately out of scope here: this file is about decision isolation only.
"""

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
from northstar_core.market_data import HistoricalOHLCVBar, HistoricalReplaySnapshot
from northstar_core.strategy import AssetAnalysisGenerator, Strategy, StrategyIdentity

from northstar_application.application_services import (
    EvaluateHistoricalResearchUseCase,
    HistoricalResearchEvaluation,
    HistoricalSnapshotEvaluatorService,
    ReplayHistoricalMarketDataUseCase,
)
from northstar_application.ports import (
    HistoricalMarketDataQuery,
    HistoricalMarketDataRepository,
)

_USD = Currency("USD")
_DAILY = Timeframe("1d")
_SYMBOL = Symbol("AAPL")
_EXCHANGE = ExchangeCode("NASDAQ")

_HISTORY_START_DAY = 1
_DECISION_DAY = 25
_FUTURE_START_DAY = 26
_FUTURE_END_DAY = 35
_MINIMUM_HISTORY = 20


def _instant(day: int) -> PointInTime:
    """Return a calendar-correct instant, rolling past the end of January."""
    month, day_of_month = (1, day) if day <= 31 else (2, day - 31)
    return PointInTime(f"2026-{month:02d}-{day_of_month:02d}T16:00:00Z")


def _bar(day: int, close: str) -> HistoricalOHLCVBar:
    return HistoricalOHLCVBar(
        _SYMBOL,
        _EXCHANGE,
        _instant(day),
        _DAILY,
        Price("50", _USD),
        Price("9000", _USD),
        Price("1", _USD),
        Price(close, _USD),
        Quantity("1000"),
    )


def _original_history() -> tuple[HistoricalOHLCVBar, ...]:
    """Return the calm history visible at the decision instant."""
    return tuple(_bar(day, str(100 + day)) for day in range(_HISTORY_START_DAY, _DECISION_DAY + 1))


def _divergent_future() -> tuple[HistoricalOHLCVBar, ...]:
    """Return extreme observations strictly after the decision instant.

    Closes are roughly two orders of magnitude away from the original window and
    alternate direction, so any leakage into signal generation would change the
    resulting recommendation rather than merely perturb it.
    """
    return tuple(
        _bar(day, str(8000 - day * 100) if day % 2 else str(200 + day * 10))
        for day in range(_FUTURE_START_DAY, _FUTURE_END_DAY + 1)
    )


class StubRepository(HistoricalMarketDataRepository):
    """Returns bounded, identity-matched observations as a conforming repository does."""

    def __init__(self, observations: tuple[HistoricalOHLCVBar, ...]) -> None:
        self.observations = observations

    def get_history(self, query: HistoricalMarketDataQuery) -> tuple[HistoricalOHLCVBar, ...]:
        return tuple(
            bar
            for bar in self.observations
            if bar.symbol == query.symbol
            and bar.exchange_code == query.exchange_code
            and bar.timeframe == query.timeframe
            and query.start.compare(bar.point_in_time) <= 0
            and query.end.compare(bar.point_in_time) >= 0
        )


def _replay(
    observations: tuple[HistoricalOHLCVBar, ...], through_day: int
) -> tuple[HistoricalReplaySnapshot, ...]:
    """Replay the stored history up to and including ``through_day``."""
    query = HistoricalMarketDataQuery(
        symbol=_SYMBOL,
        exchange_code=_EXCHANGE,
        timeframe=_DAILY,
        start=_instant(_HISTORY_START_DAY),
        end=_instant(through_day),
    )
    return ReplayHistoricalMarketDataUseCase(StubRepository(observations)).execute(query)


def _evaluate(
    snapshots: tuple[HistoricalReplaySnapshot, ...],
) -> tuple[HistoricalResearchEvaluation, ...]:
    """Evaluate every snapshot that satisfies the evaluator's warm-up requirement."""
    evaluable = tuple(
        snapshot for snapshot in snapshots if len(snapshot.observations) >= _MINIMUM_HISTORY
    )
    evaluator = HistoricalSnapshotEvaluatorService(
        strategy=Strategy(StrategyIdentity("lookahead-acceptance")),
        analysis_generator=AssetAnalysisGenerator(),
    )
    return EvaluateHistoricalResearchUseCase(evaluator).execute(evaluable)


def _decision(
    evaluations: tuple[HistoricalResearchEvaluation, ...],
) -> HistoricalResearchEvaluation:
    """Return the evaluation produced at the decision instant."""
    decision_instant = _instant(_DECISION_DAY)
    for evaluation in evaluations:
        if evaluation.result.market_observation_context.observed_at.compare(decision_instant) == 0:
            return evaluation
    raise AssertionError("No evaluation was produced at the decision instant.")


# ---------------------------------------------------------------------------
# Fixture integrity: the test must not be able to pass accidentally
# ---------------------------------------------------------------------------


def test_appended_bars_are_strictly_after_the_decision_instant() -> None:
    decision_instant = _instant(_DECISION_DAY)
    future = _divergent_future()

    assert future
    assert all(bar.point_in_time.compare(decision_instant) > 0 for bar in future)


def test_appended_bars_are_divergent_enough_to_change_a_decision() -> None:
    original_closes = {bar.close.amount for bar in _original_history()}
    future_closes = {bar.close.amount for bar in _divergent_future()}

    assert max(future_closes) > max(original_closes) * 10
    assert not original_closes & future_closes


def test_a_leaked_future_window_would_produce_a_different_decision() -> None:
    """Prove the equality assertions have teeth.

    If future observations ever reached the analysis window, the resulting
    decision-time evidence and recommendation would differ. This constructs
    that counterfactual directly, so the isolation assertions cannot be passing
    merely because the appended data is indistinguishable.
    """
    honest = _decision(_evaluate(_replay(_original_history(), _DECISION_DAY))).result

    leaked_snapshot = HistoricalReplaySnapshot(
        _instant(_FUTURE_END_DAY),
        _original_history() + _divergent_future(),
    )
    evaluator = HistoricalSnapshotEvaluatorService(
        strategy=Strategy(StrategyIdentity("lookahead-acceptance")),
        analysis_generator=AssetAnalysisGenerator(),
    )
    leaked = evaluator.evaluate(leaked_snapshot)

    assert leaked.market_observation_context != honest.market_observation_context
    assert leaked.market_observation_context.recent_closes != (
        honest.market_observation_context.recent_closes
    )
    assert leaked.recommendation.asset_analysis != honest.recommendation.asset_analysis


def test_original_history_supports_a_real_decision() -> None:
    evaluations = _evaluate(_replay(_original_history(), _DECISION_DAY))

    assert evaluations
    decision = _decision(evaluations)
    assert len(decision.result.market_observation_context.recent_closes) == _MINIMUM_HISTORY
    assert decision.result.recommendation.action.value in {"BUY", "HOLD", "SELL"}


# ---------------------------------------------------------------------------
# The decision snapshot never contains future observations
# ---------------------------------------------------------------------------


def test_decision_snapshot_excludes_appended_future_bars() -> None:
    decision_instant = _instant(_DECISION_DAY)
    extended = _original_history() + _divergent_future()

    snapshots = _replay(extended, _FUTURE_END_DAY)
    decision_snapshot = next(
        snapshot for snapshot in snapshots if snapshot.replay_instant.compare(decision_instant) == 0
    )

    assert decision_snapshot.observations
    assert all(
        bar.point_in_time.compare(decision_instant) <= 0 for bar in decision_snapshot.observations
    )
    future_closes = {bar.close for bar in _divergent_future()}
    assert not {bar.close for bar in decision_snapshot.observations} & future_closes


def test_decision_snapshot_is_identical_with_and_without_future_data() -> None:
    decision_instant = _instant(_DECISION_DAY)
    original_snapshot = _replay(_original_history(), _DECISION_DAY)[-1]

    extended = _original_history() + _divergent_future()
    extended_snapshot = next(
        snapshot
        for snapshot in _replay(extended, _FUTURE_END_DAY)
        if snapshot.replay_instant.compare(decision_instant) == 0
    )

    assert extended_snapshot == original_snapshot


# ---------------------------------------------------------------------------
# The decision itself is unchanged by future observations
# ---------------------------------------------------------------------------


def test_recommendation_is_unchanged_when_future_bars_are_appended() -> None:
    before = _decision(_evaluate(_replay(_original_history(), _DECISION_DAY)))

    extended = _original_history() + _divergent_future()
    after = _decision(_evaluate(_replay(extended, _FUTURE_END_DAY)))

    assert after.result.recommendation == before.result.recommendation


def test_asset_analysis_is_unchanged_when_future_bars_are_appended() -> None:
    before = _decision(_evaluate(_replay(_original_history(), _DECISION_DAY)))

    extended = _original_history() + _divergent_future()
    after = _decision(_evaluate(_replay(extended, _FUTURE_END_DAY)))

    assert after.result.recommendation.asset_analysis == (
        before.result.recommendation.asset_analysis
    )


def test_market_observation_context_is_unchanged_when_future_bars_are_appended() -> None:
    before = _decision(_evaluate(_replay(_original_history(), _DECISION_DAY)))

    extended = _original_history() + _divergent_future()
    after = _decision(_evaluate(_replay(extended, _FUTURE_END_DAY)))

    assert after.result.market_observation_context == (before.result.market_observation_context)


def test_every_captured_decision_value_is_unchanged_together() -> None:
    before = _decision(_evaluate(_replay(_original_history(), _DECISION_DAY)))

    extended = _original_history() + _divergent_future()
    after = _decision(_evaluate(_replay(extended, _FUTURE_END_DAY)))

    assert after.result.recommendation == before.result.recommendation
    assert after.result.recommendation.asset_analysis == (
        before.result.recommendation.asset_analysis
    )
    assert after.result.market_observation_context == (before.result.market_observation_context)
    assert after.result.explanation == before.result.explanation


def test_signal_and_action_are_unchanged_when_future_bars_are_appended() -> None:
    before = _decision(_evaluate(_replay(_original_history(), _DECISION_DAY)))

    extended = _original_history() + _divergent_future()
    after = _decision(_evaluate(_replay(extended, _FUTURE_END_DAY)))

    assert after.result.recommendation.action == before.result.recommendation.action
    assert after.result.recommendation.asset_analysis.summarized_signals == (
        before.result.recommendation.asset_analysis.summarized_signals
    )


def test_decision_time_evidence_is_unchanged_when_future_bars_are_appended() -> None:
    before = _decision(_evaluate(_replay(_original_history(), _DECISION_DAY))).result
    extended = _original_history() + _divergent_future()
    after = _decision(_evaluate(_replay(extended, _FUTURE_END_DAY))).result

    assert after.market_observation_context.observed_at == (
        before.market_observation_context.observed_at
    )
    assert after.market_observation_context.latest_price == (
        before.market_observation_context.latest_price
    )
    assert after.market_observation_context.recent_closes == (
        before.market_observation_context.recent_closes
    )
    assert after.market_observation_context.recent_volumes == (
        before.market_observation_context.recent_volumes
    )


# ---------------------------------------------------------------------------
# Every earlier decision in the window is equally isolated
# ---------------------------------------------------------------------------


def test_all_decisions_in_the_window_are_unchanged_by_future_data() -> None:
    before = _evaluate(_replay(_original_history(), _DECISION_DAY))

    extended = _original_history() + _divergent_future()
    after_all = _evaluate(_replay(extended, _FUTURE_END_DAY))
    decision_instant = _instant(_DECISION_DAY)
    after = tuple(
        evaluation
        for evaluation in after_all
        if evaluation.replay_instant.compare(decision_instant) <= 0
    )

    assert len(after) == len(before) > 1
    for original, extended_evaluation in zip(before, after, strict=True):
        assert extended_evaluation.replay_instant == original.replay_instant
        assert extended_evaluation.result == original.result


def test_future_data_does_produce_additional_later_decisions() -> None:
    before = _evaluate(_replay(_original_history(), _DECISION_DAY))

    extended = _original_history() + _divergent_future()
    after = _evaluate(_replay(extended, _FUTURE_END_DAY))

    assert len(after) > len(before)


@pytest.mark.parametrize("through_day", [_DECISION_DAY, _FUTURE_END_DAY])
def test_decision_is_stable_regardless_of_how_far_the_replay_window_extends(
    through_day: int,
) -> None:
    extended = _original_history() + _divergent_future()
    reference = _decision(_evaluate(_replay(_original_history(), _DECISION_DAY)))

    decision = _decision(_evaluate(_replay(extended, through_day)))

    assert decision.result.recommendation == reference.result.recommendation
    assert decision.result.market_observation_context == (
        reference.result.market_observation_context
    )
