"""Contract tests for the AnalyzeWatchlistUseCase application service."""

from __future__ import annotations

import pytest
from northstar_core.foundation.value_objects import Symbol

from northstar_application.application_services import (
    AnalyzeAssetUseCase,
    AnalyzeWatchlistFailureCode,
    AnalyzeWatchlistUseCase,
)


class StubAnalyzeAssetUseCase(AnalyzeAssetUseCase):
    def __init__(self, outcomes: dict[Symbol, object]) -> None:
        self.outcomes = outcomes
        self.calls: list[Symbol] = []

    def execute(self, symbol: Symbol) -> object:
        self.calls.append(symbol)
        outcome = self.outcomes[symbol]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_execute_analyzes_every_watchlist_symbol_in_order() -> None:
    symbols = (Symbol("AAPL"), Symbol("MSFT"), Symbol("GOOG"))
    outcomes = {symbol: object() for symbol in symbols}
    analyze_asset = StubAnalyzeAssetUseCase(outcomes)

    result = AnalyzeWatchlistUseCase(analyze_asset).execute(symbols)

    assert analyze_asset.calls == list(symbols)
    assert tuple(item.symbol for item in result.items) == symbols
    assert tuple(item.result for item in result.items) == tuple(outcomes.values())


def test_execute_isolates_one_asset_failure_and_continues() -> None:
    symbols = (Symbol("AAPL"), Symbol("MSFT"), Symbol("GOOG"))
    failure = RuntimeError("MSFT unavailable")
    outcomes = {symbols[0]: object(), symbols[1]: failure, symbols[2]: object()}
    analyze_asset = StubAnalyzeAssetUseCase(outcomes)

    result = AnalyzeWatchlistUseCase(analyze_asset).execute(symbols)

    assert len(result.items) == 3
    assert result.items[0].result is outcomes[symbols[0]]
    assert result.items[1].result is None
    assert result.items[1].failure is not None
    assert result.items[1].failure.code == AnalyzeWatchlistFailureCode.PROVIDER_UNAVAILABLE
    assert result.items[2].result is outcomes[symbols[2]]
    assert analyze_asset.calls == list(symbols)


@pytest.mark.parametrize("symbols", [None, [Symbol("AAPL")], ("AAPL",)])
def test_execute_rejects_invalid_watchlist_input(symbols: object) -> None:
    analyze_asset = StubAnalyzeAssetUseCase({})

    with pytest.raises(TypeError):
        AnalyzeWatchlistUseCase(analyze_asset).execute(symbols)


@pytest.mark.parametrize("analyze_asset", [None, object()])
def test_constructor_rejects_invalid_analyze_asset_dependency(analyze_asset: object) -> None:
    with pytest.raises(TypeError):
        AnalyzeWatchlistUseCase(analyze_asset)
