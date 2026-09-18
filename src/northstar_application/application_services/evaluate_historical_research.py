"""Application orchestration for historical research evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.foundation.value_objects import PointInTime
from northstar_core.market_data import HistoricalReplaySnapshot

from northstar_application.application_services.analyze_asset_result import AnalyzeAssetResult
from northstar_application.ports import HistoricalSnapshotEvaluator


@dataclass(frozen=True, slots=True)
class HistoricalResearchEvaluation:
    """Existing analysis result associated with one historical replay instant."""

    replay_instant: PointInTime
    result: AnalyzeAssetResult


class EvaluateHistoricalResearchUseCase:
    """Evaluate each replay snapshot exactly once in chronological order."""

    def __init__(self, evaluator: HistoricalSnapshotEvaluator) -> None:
        if evaluator is None:
            raise TypeError("EvaluateHistoricalResearchUseCase evaluator cannot be None.")
        if not isinstance(evaluator, HistoricalSnapshotEvaluator):
            raise TypeError(
                "EvaluateHistoricalResearchUseCase evaluator must be a HistoricalSnapshotEvaluator."
            )
        self._evaluator = evaluator

    def execute(
        self, snapshots: tuple[HistoricalReplaySnapshot, ...]
    ) -> tuple[HistoricalResearchEvaluation, ...]:
        """Evaluate snapshots without acquiring data outside each replay snapshot."""
        if snapshots is None:
            raise TypeError("EvaluateHistoricalResearchUseCase snapshots cannot be None.")
        if not isinstance(snapshots, tuple):
            raise TypeError("EvaluateHistoricalResearchUseCase snapshots must be a tuple.")

        evaluations: list[HistoricalResearchEvaluation] = []
        previous_instant: PointInTime | None = None
        for snapshot in snapshots:
            if not isinstance(snapshot, HistoricalReplaySnapshot):
                raise TypeError(
                    "EvaluateHistoricalResearchUseCase snapshots must contain "
                    "HistoricalReplaySnapshot values."
                )
            if (
                previous_instant is not None
                and previous_instant.compare(snapshot.replay_instant) >= 0
            ):
                raise ValueError(
                    "EvaluateHistoricalResearchUseCase snapshots must be chronologically ordered."
                )

            result = self._evaluator.evaluate(snapshot)
            if not isinstance(result, AnalyzeAssetResult):
                raise TypeError("HistoricalSnapshotEvaluator must return an AnalyzeAssetResult.")
            evaluations.append(
                HistoricalResearchEvaluation(
                    replay_instant=snapshot.replay_instant,
                    result=result,
                )
            )
            previous_instant = snapshot.replay_instant

        return tuple(evaluations)
