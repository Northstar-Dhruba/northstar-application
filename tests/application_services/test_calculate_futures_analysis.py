"""Tests for the read-only futures analysis composition (INDIA-8H-B).

The composition adds no formula: every value is checked against the use case it
composes, run independently over the same in-memory facts.
"""

from __future__ import annotations

from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext

import pytest
from northstar_core.foundation.value_objects import Percentage, Timeframe
from northstar_core.strategy import (
    FuturesAssetAnalysisGenerator,
    FuturesRecommendationOutcomeUnavailableReason,
    ResearchHorizon,
    Strategy,
)
from test_get_futures_paper_trading_snapshot import (
    _ALPHA,
    _ES_DEC,
    _ES_MAR,
    _FESX_DEC,
    _PORTFOLIO,
    Economics,
    Fills,
    Forward,
    Market,
    Orders,
    World,
    _at,
    _bar,
    _daily,
    _phase_a,
)

from northstar_application.application_services import (
    CalculateFuturesAnalysisUseCase,
    CalculateFuturesHistoricalResearchMetricsUseCase,
    CalculateFuturesPaperPerformanceUseCase,
    FuturesAnalysis,
    RunFuturesHistoricalResearchUseCase,
)

_H1, _H5 = ResearchHorizon(1), ResearchHorizon(5)
_INSUFFICIENT = FuturesRecommendationOutcomeUnavailableReason.INSUFFICIENT_FUTURE_OBSERVATIONS


def _use_case(world: World) -> CalculateFuturesAnalysisUseCase:
    return CalculateFuturesAnalysisUseCase(
        Forward(world),
        Orders(world),
        Fills(world),
        Market(world),
        Economics(world),
        FuturesAssetAnalysisGenerator(),
    )


def _analysis(world: World, session: int, horizons=(_H1, _H5), contract=_ES_DEC) -> FuturesAnalysis:
    return _use_case(world).execute(contract, _ALPHA, _PORTFOLIO, horizons, _at(session))


def _research_run(world: World, session: int, horizons=(_H1, _H5), contract=_ES_DEC):
    return RunFuturesHistoricalResearchUseCase(
        Market(world), FuturesAssetAnalysisGenerator()
    ).execute(contract, Timeframe("1d"), Strategy(_ALPHA), horizons, _at(session))


@pytest.fixture
def operated() -> World:
    """Production paper sessions 25..29 over the snapshot tests' scripted market."""
    return _daily(World(*_phase_a()), 29)


def test_paper_and_research_compose_without_new_formulas(operated: World) -> None:
    analysis = _analysis(operated, 29)

    expected_paper = CalculateFuturesPaperPerformanceUseCase(
        Forward(operated), Orders(operated), Fills(operated), Market(operated), Economics(operated)
    ).execute(_ES_DEC, _ALPHA, _PORTFOLIO, _at(29))
    run = _research_run(operated, 29)
    assert analysis.performance == expected_paper
    assert analysis.missing_economics is None
    assert analysis.research_metrics == CalculateFuturesHistoricalResearchMetricsUseCase().execute(
        run
    )
    assert analysis.research_decision_count == len(run.analysis_results)
    assert analysis.horizons == (_H1, _H5)
    assert [m.horizon for m in analysis.research_metrics] == [_H1, _H5]


def test_actions_partition_the_research_decisions(operated: World) -> None:
    analysis = _analysis(operated, 29)
    run = _research_run(operated, 29)

    assert [m.action for m in analysis.action_metrics] == ["BUY", "SELL", "HOLD"]
    for metrics in analysis.action_metrics:
        mine = [r for r in run.analysis_results if r.recommendation.action.value == metrics.action]
        assert metrics.decision_count == len(mine)
        for horizon_metrics in metrics.horizons:
            outcomes = [
                o
                for o in run.outcomes
                if o.recommendation.action.value == metrics.action
                and o.horizon == horizon_metrics.horizon
            ]
            returns = sorted(o.forward_return.value for o in outcomes if o.forward_return)
            assert horizon_metrics.total_count == len(outcomes) == len(mine)
            assert horizon_metrics.measured_count == len(returns)
            assert horizon_metrics.insufficient_future_observations_count == sum(
                1 for o in outcomes if o.unavailable_reason is _INSUFFICIENT
            )
            if returns:
                with localcontext(Context(prec=28, rounding=ROUND_HALF_EVEN)):
                    mean = sum(returns, Decimal(0)) / len(returns)
                assert horizon_metrics.average_forward_return == Percentage(mean)
                assert horizon_metrics.minimum_forward_return == Percentage(returns[0])
                assert horizon_metrics.maximum_forward_return == Percentage(returns[-1])
            else:
                assert horizon_metrics.average_forward_return is None
    # The scripted market produces more than one action.
    assert sum(1 for m in analysis.action_metrics if m.decision_count) >= 2


def test_late_decisions_lack_future_observations_at_long_horizons(operated: World) -> None:
    analysis = _analysis(operated, 29)

    h1, h5 = analysis.research_metrics
    assert h5.insufficient_future_observations_count == 5
    assert h1.insufficient_future_observations_count == 1
    assert h5.measured_count == h1.measured_count - 4


def test_horizon_order_is_the_callers(operated: World) -> None:
    analysis = _analysis(operated, 29, horizons=(_H5, _H1))

    assert [m.horizon for m in analysis.research_metrics] == [_H5, _H1]
    assert [m.horizon for m in analysis.action_metrics[0].horizons] == [_H5, _H1]


def test_the_same_facts_give_an_equal_analysis(operated: World) -> None:
    assert _analysis(operated, 29) == _analysis(operated, 29)


def test_later_facts_never_change_an_earlier_cutoff(operated: World) -> None:
    before = _analysis(operated, 27)
    operated.bars.append(_bar(30, "1"))  # poison
    operated.bars.append(_bar(31, "99999"))

    assert _analysis(operated, 27) == before


def test_reading_persists_nothing(operated: World) -> None:
    facts, writes = operated.facts(), operated.writes

    _analysis(operated, 29)

    assert operated.facts() == facts and operated.writes == writes


def test_missing_economics_keeps_research() -> None:
    world = World(*_phase_a(contract=_ES_MAR), economics=())

    analysis = _analysis(world, 25, contract=_ES_MAR)

    assert analysis.performance is None and analysis.missing_economics == _ES_MAR
    assert analysis.research_decision_count == 6  # sessions 20..25 have the warm-up


def test_any_futures_contract_is_analysed_generically() -> None:
    world = World(*_phase_a(contract=_FESX_DEC))

    analysis = _analysis(world, 25, contract=_FESX_DEC)

    assert analysis.performance.settlement_currency.value == "EUR"
    assert analysis.performance.decisions.decision_count == 0
    assert analysis.research_decision_count == 6
    assert analysis.contract == _FESX_DEC


def test_no_bars_is_an_empty_analysis() -> None:
    analysis = _analysis(World(), 25)

    assert analysis.research_decision_count == 0
    assert all(
        m.total_count == 0 and m.average_forward_return is None for m in analysis.research_metrics
    )
    assert all(m.decision_count == 0 for m in analysis.action_metrics)
    assert analysis.performance.equity_curve == ()
