"""Application workflow for analyzing every symbol in an ordered watchlist."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from northstar_core.foundation.value_objects import Symbol

from northstar_application.application_services.analyze_asset import (
    AnalyzeAssetResult,
    AnalyzeAssetUseCase,
)


class AnalyzeWatchlistFailureCode(StrEnum):
    """Stable Application-level failure categories for one watchlist item."""

    UNKNOWN_SYMBOL = "UNKNOWN_SYMBOL"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    INCOMPLETE_MARKET_DATA = "INCOMPLETE_MARKET_DATA"
    ANALYSIS_FAILED = "ANALYSIS_FAILED"


@dataclass(frozen=True, slots=True)
class AnalyzeWatchlistFailure:
    """Sanitized failure information safe to cross the Application boundary."""

    code: AnalyzeWatchlistFailureCode


@dataclass(frozen=True, slots=True)
class AnalyzeWatchlistItemResult:
    """Outcome for one watchlist symbol."""

    symbol: Symbol
    result: AnalyzeAssetResult | None = None
    failure: AnalyzeWatchlistFailure | None = None

    @property
    def error(self) -> AnalyzeWatchlistFailure | None:
        """Backward-compatible alias for consumers of the previous contract."""
        return self.failure


@dataclass(frozen=True, slots=True)
class AnalyzeWatchlistResult:
    """Ordered outcomes for all symbols submitted in one watchlist."""

    items: tuple[AnalyzeWatchlistItemResult, ...]


class AnalyzeWatchlistUseCase:
    """Analyzes watchlist symbols sequentially through AnalyzeAssetUseCase."""

    def __init__(self, analyze_asset: AnalyzeAssetUseCase) -> None:
        if analyze_asset is None:
            raise TypeError("AnalyzeWatchlistUseCase analyze_asset cannot be None.")
        if not isinstance(analyze_asset, AnalyzeAssetUseCase):
            raise TypeError("AnalyzeWatchlistUseCase analyze_asset must be an AnalyzeAssetUseCase.")
        self._analyze_asset = analyze_asset

    def execute(self, symbols: tuple[Symbol, ...]) -> AnalyzeWatchlistResult:
        """Analyze each symbol, retaining order and isolating failures."""
        if symbols is None:
            raise TypeError("AnalyzeWatchlistUseCase symbols cannot be None.")
        if not isinstance(symbols, tuple):
            raise TypeError("AnalyzeWatchlistUseCase symbols must be a tuple.")
        if not all(isinstance(symbol, Symbol) for symbol in symbols):
            raise TypeError("AnalyzeWatchlistUseCase symbols must contain Symbol values.")

        items: list[AnalyzeWatchlistItemResult] = []
        for symbol in symbols:
            try:
                result = self._analyze_asset.execute(symbol)
            except Exception as error:
                items.append(
                    AnalyzeWatchlistItemResult(
                        symbol=symbol,
                        failure=AnalyzeWatchlistFailure(self._classify_failure(error)),
                    )
                )
            else:
                items.append(AnalyzeWatchlistItemResult(symbol=symbol, result=result))

        return AnalyzeWatchlistResult(items=tuple(items))

    @staticmethod
    def _classify_failure(error: Exception) -> AnalyzeWatchlistFailureCode:
        if isinstance(error, LookupError):
            return AnalyzeWatchlistFailureCode.UNKNOWN_SYMBOL
        if isinstance(error, RuntimeError):
            if "incomplete" in str(error).casefold():
                return AnalyzeWatchlistFailureCode.INCOMPLETE_MARKET_DATA
            return AnalyzeWatchlistFailureCode.PROVIDER_UNAVAILABLE
        return AnalyzeWatchlistFailureCode.ANALYSIS_FAILED
