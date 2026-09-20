"""Application measurement of one frozen forward research decision.

Measuring a frozen decision uses exactly the same machinery as historical
research: the decision is evidence, and later observations record only what the
market subsequently did. Measurement applies no BUY, SELL or HOLD
interpretation and models no order, fill, position or profit and loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from northstar_core.foundation.value_objects import PointInTime
from northstar_core.strategy import ResearchHorizon

from northstar_application.application_services.measure_recommendation_outcome import (
    MeasureRecommendationOutcomeUseCase,
    RecommendationOutcomeMeasurement,
    RecommendationOutcomeUnavailableReason,
)
from northstar_application.application_services.record_forward_research_decision import (
    ForwardResearchRecord,
)


class ForwardResearchMeasurementState(StrEnum):
    """Stable Application-level states of one forward measurement attempt.

    PENDING and UNAVAILABLE both describe an unmeasured decision, but differ in
    expectation: a pending decision is expected to become measurable as further
    observations arrive, whereas an unavailable one will not become measurable
    for this horizon.
    """

    PENDING = "PENDING"
    MEASURED = "MEASURED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class ForwardResearchMeasurement:
    """One frozen decision paired with its measurement attempt at one horizon.

    ``state`` is derived from the measurement on every read and is never stored.
    A persisted state would go stale the moment further observations arrive, so
    a pending decision must always be re-derived from current data rather than
    remembered.
    """

    record: ForwardResearchRecord
    horizon: ResearchHorizon
    measurement: RecommendationOutcomeMeasurement

    def __post_init__(self) -> None:
        if not isinstance(self.record, ForwardResearchRecord):
            raise TypeError("ForwardResearchMeasurement record must be a ForwardResearchRecord.")
        if not isinstance(self.horizon, ResearchHorizon):
            raise TypeError("ForwardResearchMeasurement horizon must be a ResearchHorizon.")
        if not isinstance(self.measurement, RecommendationOutcomeMeasurement):
            raise TypeError(
                "ForwardResearchMeasurement measurement must be a RecommendationOutcomeMeasurement."
            )
        if self.measurement.horizon != self.horizon:
            raise ValueError(
                "ForwardResearchMeasurement measurement must measure the same horizon."
            )
        if self.measurement.recommendation != self.record.result.recommendation:
            raise ValueError(
                "ForwardResearchMeasurement measurement must measure the recorded recommendation."
            )

    @property
    def state(self) -> ForwardResearchMeasurementState:
        """Return the measurement state derived from the current measurement."""
        if self.measurement.outcome is not None:
            return ForwardResearchMeasurementState.MEASURED
        if (
            self.measurement.unavailable
            is RecommendationOutcomeUnavailableReason.INSUFFICIENT_FUTURE_OBSERVATIONS
        ):
            return ForwardResearchMeasurementState.PENDING
        return ForwardResearchMeasurementState.UNAVAILABLE

    @property
    def is_measured(self) -> bool:
        """Return whether a factual outcome was measured."""
        return self.state is ForwardResearchMeasurementState.MEASURED


class MeasureForwardResearchRecordUseCase:
    """Measure one frozen forward decision at one research horizon.

    Measurement is delegated to MeasureRecommendationOutcomeUseCase so horizon
    selection, look-ahead safety and unavailable-reason semantics stay defined
    in exactly one place for both historical and forward research.
    """

    def __init__(self, measure_recommendation_outcome: MeasureRecommendationOutcomeUseCase) -> None:
        if measure_recommendation_outcome is None:
            raise TypeError(
                "MeasureForwardResearchRecordUseCase measure_recommendation_outcome cannot be None."
            )
        if not isinstance(measure_recommendation_outcome, MeasureRecommendationOutcomeUseCase):
            raise TypeError(
                "MeasureForwardResearchRecordUseCase measure_recommendation_outcome must be a "
                "MeasureRecommendationOutcomeUseCase."
            )
        self._measure_recommendation_outcome = measure_recommendation_outcome

    def execute(
        self,
        record: ForwardResearchRecord,
        horizon: ResearchHorizon,
        available_through: PointInTime,
    ) -> ForwardResearchMeasurement:
        """Measure what the market did after one frozen decision."""
        if record is None:
            raise TypeError("MeasureForwardResearchRecordUseCase record cannot be None.")
        if not isinstance(record, ForwardResearchRecord):
            raise TypeError(
                "MeasureForwardResearchRecordUseCase record must be a ForwardResearchRecord."
            )

        measurement = self._measure_recommendation_outcome.execute(
            record.result,
            horizon,
            record.timeframe,
            available_through,
        )
        return ForwardResearchMeasurement(
            record=record,
            horizon=horizon,
            measurement=measurement,
        )
