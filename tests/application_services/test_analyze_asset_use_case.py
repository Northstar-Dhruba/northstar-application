"""Contract tests for the AnalyzeAssetUseCase application service."""

from __future__ import annotations

import pytest
from northstar_core.domain.exchange import Exchange
from northstar_core.domain.instrument import Instrument
from northstar_core.domain.listing import Listing
from northstar_core.domain.value_objects import ListingStatus, Tradability
from northstar_core.foundation.value_objects import Currency, ExchangeCode, PointInTime, Symbol
from northstar_core.strategy import AssetAnalysis, Recommendation, Strategy, StrategyIdentity

from northstar_application.application_services import (
    AnalyzeAssetResult,
    AnalyzeAssetUseCase,
    AssetAnalysisInput,
)


def _build_listing(symbol: str = "AAPL") -> Listing:
    return Listing(
        instrument=Instrument(Symbol(symbol), "Apple Inc.", "Equity"),
        exchange=Exchange(ExchangeCode("NASDAQ"), "NASDAQ"),
        currency=Currency("USD"),
        listing_status=ListingStatus("Active"),
        tradability=Tradability("Permitted"),
        description="Apple Inc.",
    )


def _build_recommendation() -> Recommendation:
    asset_analysis = AssetAnalysis(
        listing=_build_listing(),
        point_in_time=PointInTime("2026-09-06T09:30:00Z"),
        summarized_signals=("strong bullish",),
    )
    return Strategy(StrategyIdentity("test-strategy")).evaluate(asset_analysis)


class StubProvider:
    def __init__(self, *, input_data: AssetAnalysisInput) -> None:
        self.input_data = input_data
        self.calls: list[Symbol] = []

    def get_analysis_input(self, symbol: Symbol) -> AssetAnalysisInput:
        self.calls.append(symbol)
        return self.input_data


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
    analysis_input = AssetAnalysisInput(
        listing=_build_listing("AAPL"),
        point_in_time=PointInTime("2026-09-06T09:30:00Z"),
        summarized_signals=("strong bullish",),
    )
    strategy = Strategy(StrategyIdentity("mvp"))
    use_case = AnalyzeAssetUseCase(strategy, StubProvider(input_data=analysis_input))

    result = use_case.execute(symbol)

    expected = strategy.evaluate(
        AssetAnalysis(
            listing=analysis_input.listing,
            point_in_time=analysis_input.point_in_time,
            summarized_signals=analysis_input.summarized_signals,
        )
    )

    assert isinstance(result, AnalyzeAssetResult)
    assert result.recommendation == expected
    assert str(result.recommendation) == "BUY AAPL by mvp at 2026-09-06T09:30:00Z"
    assert result.explanation.recommendation is result.recommendation
    assert result.explanation.reasons[0].supporting_signals == ("strong bullish",)


def test_execute_builds_asset_analysis_from_provider_input_and_delegates_strategy() -> None:
    symbol = Symbol("AAPL")
    analysis_input = AssetAnalysisInput(
        listing=_build_listing("AAPL"),
        point_in_time=PointInTime("2026-09-06T09:30:00Z"),
        summarized_signals=("strong bullish", "volume expansion"),
    )
    expected = AssetAnalysis(
        listing=analysis_input.listing,
        point_in_time=analysis_input.point_in_time,
        summarized_signals=analysis_input.summarized_signals,
    )
    strategy = RecordingStrategy(recommendation=_build_recommendation())
    provider = StubProvider(input_data=analysis_input)
    use_case = AnalyzeAssetUseCase(strategy, provider)

    result = use_case.execute(symbol)

    assert result.recommendation is strategy._recommendation
    assert provider.calls == [symbol]
    assert strategy.calls == [expected]


def test_execute_calls_provider_once_per_invocation() -> None:
    symbol = Symbol("AAPL")
    provider = StubProvider(
        input_data=AssetAnalysisInput(
            listing=_build_listing("AAPL"),
            point_in_time=PointInTime("2026-09-06T09:30:00Z"),
            summarized_signals=("strong bullish",),
        )
    )
    strategy = RecordingStrategy(recommendation=_build_recommendation())
    use_case = AnalyzeAssetUseCase(strategy, provider)

    use_case.execute(symbol)
    use_case.execute(symbol)

    assert provider.calls == [symbol, symbol]


@pytest.mark.parametrize(
    ("strategy", "observation_provider", "message"),
    [
        (
            None,
            StubProvider(
                input_data=AssetAnalysisInput(
                    listing=_build_listing("AAPL"),
                    point_in_time=PointInTime("2026-09-06T09:30:00Z"),
                    summarized_signals=("strong bullish",),
                )
            ),
            "AnalyzeAssetUseCase strategy cannot be None.",
        ),
        (
            object(),
            StubProvider(
                input_data=AssetAnalysisInput(
                    listing=_build_listing("AAPL"),
                    point_in_time=PointInTime("2026-09-06T09:30:00Z"),
                    summarized_signals=("strong bullish",),
                )
            ),
            "AnalyzeAssetUseCase strategy must be a Strategy instance.",
        ),
        (
            Strategy(StrategyIdentity("mvp")),
            None,
            "AnalyzeAssetUseCase observation_provider cannot be None.",
        ),
        (
            Strategy(StrategyIdentity("mvp")),
            object(),
            r"AnalyzeAssetUseCase observation_provider must provide get_analysis_input\(symbol\)\.",
        ),
    ],
)
def test_constructor_rejects_invalid_dependencies(
    strategy: object,
    observation_provider: object,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        AnalyzeAssetUseCase(strategy, observation_provider)


def test_execute_rejects_invalid_symbol() -> None:
    use_case = AnalyzeAssetUseCase(
        Strategy(StrategyIdentity("mvp")),
        StubProvider(
            input_data=AssetAnalysisInput(
                listing=_build_listing("AAPL"),
                point_in_time=PointInTime("2026-09-06T09:30:00Z"),
                summarized_signals=("strong bullish",),
            )
        ),
    )

    with pytest.raises(TypeError, match="AnalyzeAssetUseCase symbol must be a Symbol value."):
        use_case.execute("AAPL")


def test_execute_is_deterministic_for_same_inputs() -> None:
    symbol = Symbol("AAPL")
    analysis_input = AssetAnalysisInput(
        listing=_build_listing("AAPL"),
        point_in_time=PointInTime("2026-09-06T09:30:00Z"),
        summarized_signals=("strong bullish",),
    )
    strategy = RecordingStrategy(recommendation=_build_recommendation())
    use_case = AnalyzeAssetUseCase(strategy, StubProvider(input_data=analysis_input))

    first = use_case.execute(symbol)
    second = use_case.execute(symbol)

    assert first.recommendation is second.recommendation
    assert first.explanation == second.explanation
    assert strategy.calls[0] == strategy.calls[1]
