"""Application port for evaluating one historical replay snapshot."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from northstar_core.market_data import HistoricalReplaySnapshot

if TYPE_CHECKING:
    from northstar_application.application_services.analyze_asset_result import AnalyzeAssetResult


class HistoricalSnapshotEvaluator(ABC):
    """Evaluates factual replay state without acquiring live market data."""

    @abstractmethod
    def evaluate(self, snapshot: HistoricalReplaySnapshot) -> AnalyzeAssetResult:
        """Return the existing analysis result for one legally visible snapshot."""
