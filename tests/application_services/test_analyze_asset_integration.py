"""Deterministic end-to-end Application workflow coverage."""

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
    AssetAnalysisGenerator,
    MarketObservationContext,
    Strategy,
    StrategyIdentity,
)

from northstar_application.application_services import AnalyzeAssetUseCase
from northstar_application.ports import MarketObservationSource


class DeterministicObservationSource(MarketObservationSource):
    def __init__(self, context: MarketObservationContext) -> None:
        self.context = context

    def get_observation_context(self, symbol: Symbol) -> MarketObservationContext:
        assert symbol == self.context.listing_reference.symbol
        return self.context


def _build_context() -> MarketObservationContext:
    currency = Currency("USD")
    listing_reference = ListingReference(Symbol("AAPL"), ExchangeCode("NASDAQ"))
    closes = tuple(Price("100", currency) for _ in range(15)) + tuple(
        Price("130", currency) for _ in range(5)
    )
    volumes = tuple(Quantity("100") for _ in range(20))
    return MarketObservationContext(
        listing_reference=listing_reference,
        observed_at=PointInTime("2026-09-15T10:00:00Z"),
        latest_price=Price("130", currency),
        previous_close=Price("120", currency),
        latest_volume=Quantity("200"),
        daily_high=Price("135", currency),
        daily_low=Price("115", currency),
        recent_closes=closes,
        recent_volumes=volumes,
    )


def test_analyze_asset_application_workflow_preserves_complete_result() -> None:
    context = _build_context()
    use_case = AnalyzeAssetUseCase(
        strategy=Strategy(StrategyIdentity("integration-test")),
        observation_source=DeterministicObservationSource(context),
        analysis_generator=AssetAnalysisGenerator(),
    )

    result = use_case.execute(Symbol("AAPL"))

    assert result.market_observation_context is context
    assert result.recommendation.action.value == "BUY"
    assert result.recommendation.asset_analysis.listing_reference is context.listing_reference
    assert result.explanation.recommendation is result.recommendation
    assert result.explanation.reasons[0].supporting_signals == ("strong bullish",)
