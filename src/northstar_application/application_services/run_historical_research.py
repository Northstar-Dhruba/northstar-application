"""Application orchestration for one historical research run.

A run measures every evaluated decision against every requested research
horizon. It records factual measurements only: it never interprets BUY, SELL or
HOLD, never aggregates accuracy, win rate or profit and loss, and never models
orders, fills, positions or execution.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.strategy import ResearchHorizon

from northstar_application.application_services.evaluate_historical_research import (
    HistoricalResearchEvaluation,
)
from northstar_application.application_services.measure_recommendation_outcome import (
    MeasureRecommendationOutcomeUseCase,
    RecommendationOutcomeMeasurement,
)


@dataclass(frozen=True, slots=True)
class HistoricalResearchRun:
    """Complete factual measurement set for one historical research run.

    ``measurements`` holds one measurement for every evaluation and horizon
    pair, ordered by evaluation first and horizon second. Unmeasurable pairs
    remain present as explicit measurements so no evaluated decision is ever
    silently dropped from a run.

    ``timeframe`` and ``available_through`` record the run configuration that
    cannot be reconstructed from the measurements themselves: the observation
    interval a horizon counts, and the boundary beyond which no observation was
    available. Without them an unavailable measurement cannot be distinguished
    from a deliberately truncated data window.

    A non-empty run is semantically homogeneous: every evaluation shares one
    ListingReference and one StrategyIdentity, so the run describes one listed
    asset researched under one strategy. Those values are not duplicated as run
    fields; they remain owned by each evaluation's Recommendation.
    """

    evaluations: tuple[HistoricalResearchEvaluation, ...]
    horizons: tuple[ResearchHorizon, ...]
    measurements: tuple[RecommendationOutcomeMeasurement, ...]
    timeframe: Timeframe
    available_through: PointInTime

    def __post_init__(self) -> None:
        if not isinstance(self.evaluations, tuple):
            raise TypeError("HistoricalResearchRun evaluations must be a tuple.")
        if not all(
            isinstance(evaluation, HistoricalResearchEvaluation) for evaluation in self.evaluations
        ):
            raise TypeError(
                "HistoricalResearchRun evaluations must contain "
                "HistoricalResearchEvaluation values."
            )
        if not isinstance(self.timeframe, Timeframe):
            raise TypeError("HistoricalResearchRun timeframe must be a Timeframe.")
        if not isinstance(self.available_through, PointInTime):
            raise TypeError("HistoricalResearchRun available-through must be a PointInTime.")
        self._validate_homogeneous_identity()
        if not isinstance(self.horizons, tuple):
            raise TypeError("HistoricalResearchRun horizons must be a tuple.")
        if not all(isinstance(horizon, ResearchHorizon) for horizon in self.horizons):
            raise TypeError("HistoricalResearchRun horizons must contain ResearchHorizon values.")
        if not isinstance(self.measurements, tuple):
            raise TypeError("HistoricalResearchRun measurements must be a tuple.")
        if not all(
            isinstance(measurement, RecommendationOutcomeMeasurement)
            for measurement in self.measurements
        ):
            raise TypeError(
                "HistoricalResearchRun measurements must contain "
                "RecommendationOutcomeMeasurement values."
            )
        if len(self.measurements) != len(self.evaluations) * len(self.horizons):
            raise ValueError(
                "HistoricalResearchRun must contain one measurement for every "
                "evaluation and horizon pair."
            )

        expected_pairs = (
            (evaluation.result.recommendation, horizon)
            for evaluation in self.evaluations
            for horizon in self.horizons
        )
        for measurement, (recommendation, horizon) in zip(
            self.measurements, expected_pairs, strict=True
        ):
            if measurement.recommendation != recommendation:
                raise ValueError(
                    "HistoricalResearchRun measurements must follow evaluation order, "
                    "measuring each evaluation's own recommendation."
                )
            if measurement.horizon != horizon:
                raise ValueError(
                    "HistoricalResearchRun measurements must follow horizon order "
                    "within each evaluation."
                )

    def _validate_homogeneous_identity(self) -> None:
        """Require one listed asset and one strategy across a non-empty run."""
        if not self.evaluations:
            return

        first = self.evaluations[0].result.recommendation
        listing_reference = first.asset_analysis.listing_reference
        strategy_identity = first.strategy_identity
        for evaluation in self.evaluations[1:]:
            recommendation = evaluation.result.recommendation
            if recommendation.asset_analysis.listing_reference != listing_reference:
                raise ValueError(
                    "HistoricalResearchRun evaluations must share one listing reference."
                )
            if recommendation.strategy_identity != strategy_identity:
                raise ValueError(
                    "HistoricalResearchRun evaluations must share one strategy identity."
                )


class RunHistoricalResearchUseCase:
    """Measure a replay-ordered evaluation sequence across research horizons.

    Measurement itself is delegated to MeasureRecommendationOutcomeUseCase so
    horizon selection, look-ahead safety and unavailable-reason semantics are
    defined in exactly one place.
    """

    def __init__(self, measure_recommendation_outcome: MeasureRecommendationOutcomeUseCase) -> None:
        if measure_recommendation_outcome is None:
            raise TypeError(
                "RunHistoricalResearchUseCase measure_recommendation_outcome cannot be None."
            )
        if not isinstance(measure_recommendation_outcome, MeasureRecommendationOutcomeUseCase):
            raise TypeError(
                "RunHistoricalResearchUseCase measure_recommendation_outcome must be a "
                "MeasureRecommendationOutcomeUseCase."
            )
        self._measure_recommendation_outcome = measure_recommendation_outcome

    def execute(
        self,
        evaluations: tuple[HistoricalResearchEvaluation, ...],
        horizons: tuple[ResearchHorizon, ...],
        timeframe: Timeframe,
        available_through: PointInTime,
    ) -> HistoricalResearchRun:
        """Measure every evaluation against every horizon in deterministic order."""
        self._validate_evaluations(evaluations)
        self._validate_horizons(horizons)
        if not isinstance(timeframe, Timeframe):
            raise TypeError("RunHistoricalResearchUseCase timeframe must be a Timeframe.")
        if not isinstance(available_through, PointInTime):
            raise TypeError("RunHistoricalResearchUseCase available-through must be a PointInTime.")

        measurements = tuple(
            self._measure_recommendation_outcome.execute(
                evaluation.result,
                horizon,
                timeframe,
                available_through,
            )
            for evaluation in evaluations
            for horizon in horizons
        )
        return HistoricalResearchRun(
            evaluations=evaluations,
            horizons=horizons,
            measurements=measurements,
            timeframe=timeframe,
            available_through=available_through,
        )

    @staticmethod
    def _validate_evaluations(evaluations: object) -> None:
        if evaluations is None:
            raise TypeError("RunHistoricalResearchUseCase evaluations cannot be None.")
        if not isinstance(evaluations, tuple):
            raise TypeError("RunHistoricalResearchUseCase evaluations must be a tuple.")

        previous_instant: PointInTime | None = None
        for evaluation in evaluations:
            if not isinstance(evaluation, HistoricalResearchEvaluation):
                raise TypeError(
                    "RunHistoricalResearchUseCase evaluations must contain "
                    "HistoricalResearchEvaluation values."
                )
            if (
                previous_instant is not None
                and previous_instant.compare(evaluation.replay_instant) >= 0
            ):
                raise ValueError(
                    "RunHistoricalResearchUseCase evaluations must be chronologically ordered."
                )
            previous_instant = evaluation.replay_instant

    @staticmethod
    def _validate_horizons(horizons: object) -> None:
        if horizons is None:
            raise TypeError("RunHistoricalResearchUseCase horizons cannot be None.")
        if not isinstance(horizons, tuple):
            raise TypeError("RunHistoricalResearchUseCase horizons must be a tuple.")
        if not horizons:
            raise ValueError("RunHistoricalResearchUseCase requires at least one horizon.")

        seen: set[int] = set()
        for horizon in horizons:
            if not isinstance(horizon, ResearchHorizon):
                raise TypeError(
                    "RunHistoricalResearchUseCase horizons must contain ResearchHorizon values."
                )
            if horizon.observations in seen:
                raise ValueError("RunHistoricalResearchUseCase horizons cannot contain duplicates.")
            seen.add(horizon.observations)
