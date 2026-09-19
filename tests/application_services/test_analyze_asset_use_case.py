"""Contract tests for the AnalyzeAssetUseCase application service."""

from __future__ import annotations

import pytest
from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import (
    Currency,
    ExchangeCode,
    PointInTime,
    Price,
    Quantity,
    Symbol,
)
from northstar_core.strategy import (
    AssetAnalysis,
    AssetAnalysisGenerator,
    MarketObservationContext,
    Recommendation,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services import (
    AnalyzeAssetResult,
    AnalyzeAssetUseCase,
)
from northstar_application.ports import MarketObservationSource


def _build_listing_reference(symbol: str = "AAPL") -> ListingReference:
    return ListingReference(Symbol(symbol), ExchangeCode("NASDAQ"))


def _build_context(symbol: str = "AAPL") -> MarketObservationContext:
    currency = Currency("USD")
    return MarketObservationContext(
        listing_reference=_build_listing_reference(symbol),
        observed_at=PointInTime("2026-09-06T09:30:00Z"),
        latest_price=Price("130", currency),
        previous_close=Price("120", currency),
        latest_volume=Quantity("200"),
        daily_high=Price("135", currency),
        daily_low=Price("115", currency),
        recent_closes=tuple(Price("100", currency) for _ in range(15))
        + tuple(Price("130", currency) for _ in range(5)),
        recent_volumes=tuple(Quantity("100") for _ in range(20)),
    )


def _build_recommendation() -> Recommendation:
    return Strategy(StrategyIdentity("test-strategy")).evaluate(
        AssetAnalysisGenerator().generate(_build_context())
    )


class StubObservationSource(MarketObservationSource):
    def __init__(self, *, context: MarketObservationContext) -> None:
        self.context = context
        self.calls: list[Symbol] = []

    def get_observation_context(self, symbol: Symbol) -> MarketObservationContext:
        self.calls.append(symbol)
        return self.context


class RecordingAnalysisGenerator(AssetAnalysisGenerator):
    def __init__(self, asset_analysis: AssetAnalysis) -> None:
        self.asset_analysis = asset_analysis
        self.calls: list[MarketObservationContext] = []

    def generate(self, context: MarketObservationContext) -> AssetAnalysis:
        self.calls.append(context)
        return self.asset_analysis


class RecordingStrategy(Strategy):
    def __init__(self, recommendation: object) -> None:
        super().__init__(StrategyIdentity("recording-strategy"))
        self._recommendation = recommendation
        self.calls: list[AssetAnalysis] = []

    def evaluate(self, asset_analysis: AssetAnalysis) -> object:
        self.calls.append(asset_analysis)
        return self._recommendation


def test_execute_returns_strategy_recommendation() -> None:
    symbol = Symbol("AAPL")
    context = _build_context("AAPL")
    strategy = Strategy(StrategyIdentity("mvp"))
    use_case = AnalyzeAssetUseCase(
        strategy,
        StubObservationSource(context=context),
        AssetAnalysisGenerator(),
    )

    result = use_case.execute(symbol)

    expected = strategy.evaluate(AssetAnalysisGenerator().generate(context))

    assert isinstance(result, AnalyzeAssetResult)
    assert result.recommendation == expected
    assert str(result.recommendation) == "BUY AAPL by mvp at 2026-09-06T09:30:00Z"
    assert result.explanation.recommendation is result.recommendation
    assert result.explanation.reasons[0].supporting_signals == ("strong bullish",)
    assert result.market_observation_context is context


def test_execute_requests_context_generates_analysis_and_delegates_strategy() -> None:
    symbol = Symbol("AAPL")
    context = _build_context("AAPL")
    expected = AssetAnalysis(
        listing_reference=context.listing_reference,
        point_in_time=context.observed_at,
        summarized_signals=("strong bullish",),
    )
    strategy = RecordingStrategy(recommendation=_build_recommendation())
    source = StubObservationSource(context=context)
    generator = RecordingAnalysisGenerator(expected)
    use_case = AnalyzeAssetUseCase(strategy, source, generator)

    result = use_case.execute(symbol)

    assert result.recommendation is strategy._recommendation
    assert result.market_observation_context is context
    assert source.calls == [symbol]
    assert generator.calls == [context]
    assert strategy.calls == [expected]


def test_execute_calls_observation_source_once_per_invocation() -> None:
    symbol = Symbol("AAPL")
    source = StubObservationSource(context=_build_context("AAPL"))
    strategy = RecordingStrategy(recommendation=_build_recommendation())
    use_case = AnalyzeAssetUseCase(strategy, source, AssetAnalysisGenerator())

    use_case.execute(symbol)
    use_case.execute(symbol)

    assert source.calls == [symbol, symbol]


@pytest.mark.parametrize(
    ("strategy", "observation_source", "analysis_generator", "message"),
    [
        (
            None,
            StubObservationSource(context=_build_context()),
            AssetAnalysisGenerator(),
            "AnalyzeAssetUseCase strategy cannot be None.",
        ),
        (
            object(),
            StubObservationSource(context=_build_context()),
            AssetAnalysisGenerator(),
            "AnalyzeAssetUseCase strategy must be a Strategy instance.",
        ),
        (
            Strategy(StrategyIdentity("mvp")),
            None,
            AssetAnalysisGenerator(),
            "AnalyzeAssetUseCase observation_source cannot be None.",
        ),
        (
            Strategy(StrategyIdentity("mvp")),
            object(),
            AssetAnalysisGenerator(),
            "AnalyzeAssetUseCase observation_source must be a MarketObservationSource.",
        ),
        (
            Strategy(StrategyIdentity("mvp")),
            StubObservationSource(context=_build_context()),
            None,
            "AnalyzeAssetUseCase analysis_generator cannot be None.",
        ),
        (
            Strategy(StrategyIdentity("mvp")),
            StubObservationSource(context=_build_context()),
            object(),
            "AnalyzeAssetUseCase analysis_generator must be an AssetAnalysisGenerator.",
        ),
    ],
)
def test_constructor_rejects_invalid_dependencies(
    strategy: object,
    observation_source: object,
    analysis_generator: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        AnalyzeAssetUseCase(strategy, observation_source, analysis_generator)


def test_execute_rejects_invalid_symbol() -> None:
    use_case = AnalyzeAssetUseCase(
        Strategy(StrategyIdentity("mvp")),
        StubObservationSource(context=_build_context()),
        AssetAnalysisGenerator(),
    )

    with pytest.raises(TypeError, match="AnalyzeAssetUseCase symbol must be a Symbol value."):
        use_case.execute("AAPL")


def test_execute_is_deterministic_for_same_inputs() -> None:
    symbol = Symbol("AAPL")
    context = _build_context("AAPL")
    strategy = RecordingStrategy(recommendation=_build_recommendation())
    use_case = AnalyzeAssetUseCase(
        strategy,
        StubObservationSource(context=context),
        AssetAnalysisGenerator(),
    )

    first = use_case.execute(symbol)
    second = use_case.execute(symbol)

    assert first.recommendation is second.recommendation
    assert first.explanation == second.explanation
    assert first.market_observation_context is context
    assert second.market_observation_context is context
    assert strategy.calls[0] == strategy.calls[1]
