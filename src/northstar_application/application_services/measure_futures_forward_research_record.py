"""Application measurement of one frozen futures forward research decision.

Measuring a frozen forward decision uses exactly the machinery historical
research uses: the decision is evidence, and later persisted bars record only
what the market subsequently did. Horizon selection, the strictly-later rule,
the evidence cutoff, the return and the return-basis policy all belong to
MeasureFuturesRecommendationOutcomeUseCase and FuturesRecommendationOutcome;
this module adds only the forward reading of that outcome.

The forward reading
-------------------
A historical run is a finished record, so an unreached horizon is simply
insufficient. A forward decision is still maturing, so the same fact reads as
PENDING: more bars are expected. The three states follow from the Core outcome
alone:

- a forward return exists: MEASURED;
- the horizon observation does not exist yet: PENDING;
- the horizon observation exists but the decision quote was zero or negative,
  so a percentage return cannot be expressed: UNAVAILABLE.

Core decides insufficiency before the return basis, so a decision taken on a
non-positive quote stays PENDING until its horizon observation arrives and only
then becomes UNAVAILABLE. Nothing here inspects a quote's sign.

What is and is not guaranteed
-----------------------------
Nothing about a measurement is stored; it is derived on every run from the
frozen record, the horizon, the persisted bars and an explicit cutoff. Once the
Nth later observation exists, bars appended after it cannot change the
measurement, because the horizon selects the Nth observation and nothing past
it. A bar back-filled *between* the decision and that Nth observation can,
however, shift which observation is Nth and so change a derived measurement.
That is known, accepted debt of derived measurement; this story does not lock
or persist measurements to prevent it.

BUY, HOLD and SELL are never read: a measurement is market movement, not
profit, and the same movement is recorded whichever way the decision leaned.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.foundation.value_objects import PointInTime
from northstar_core.strategy import (
    FuturesRecommendationOutcome,
    FuturesRecommendationOutcomeUnavailableReason,
    ResearchHorizon,
)

from northstar_application.application_services.futures_forward_research_record import (
    FuturesForwardResearchRecord,
)
from northstar_application.application_services.measure_forward_research_record import (
    ForwardResearchMeasurementState,
)
from northstar_application.application_services.measure_futures_recommendation_outcome import (
    MeasureFuturesRecommendationOutcomeUseCase,
)
from northstar_application.ports import FuturesHistoricalMarketDataRepository

_INSUFFICIENT = FuturesRecommendationOutcomeUnavailableReason.INSUFFICIENT_FUTURE_OBSERVATIONS
_UNDEFINED_BASIS = FuturesRecommendationOutcomeUnavailableReason.UNDEFINED_RETURN_BASIS


@dataclass(frozen=True, slots=True)
class FuturesForwardResearchMeasurement:
    """One frozen futures decision paired with its outcome at one horizon.

    ``state`` is derived from the outcome on every read and is never stored: a
    stored PENDING would go stale the moment the horizon observation arrived.

    The outcome must belong to exactly this record and horizon. Its
    recommendation must be the record's -- which also fixes the contract,
    strategy and decision instant -- and its decision quote must be the frozen
    decision evidence, never a quote read back from storage.
    """

    record: FuturesForwardResearchRecord
    horizon: ResearchHorizon
    outcome: FuturesRecommendationOutcome

    def __post_init__(self) -> None:
        subject = "FuturesForwardResearchMeasurement"
        if not isinstance(self.record, FuturesForwardResearchRecord):
            raise TypeError(f"{subject} record must be a FuturesForwardResearchRecord.")
        if not isinstance(self.horizon, ResearchHorizon):
            raise TypeError(f"{subject} horizon must be a ResearchHorizon.")
        if not isinstance(self.outcome, FuturesRecommendationOutcome):
            raise TypeError(f"{subject} outcome must be a FuturesRecommendationOutcome.")

        if self.outcome.recommendation != self.record.result.recommendation:
            raise ValueError(f"{subject} outcome must measure the recorded recommendation.")
        if self.outcome.horizon != self.horizon:
            raise ValueError(f"{subject} outcome must measure the same horizon.")
        if (
            self.outcome.decision_quote
            != self.record.result.market_observation_context.latest_quote
        ):
            raise ValueError(
                f"{subject} outcome decision quote must be the frozen decision evidence."
            )

    @property
    def state(self) -> ForwardResearchMeasurementState:
        """Return the forward state derived from the current outcome."""
        if self.outcome.forward_return is not None:
            return ForwardResearchMeasurementState.MEASURED
        reason = self.outcome.unavailable_reason
        if reason is _INSUFFICIENT:
            return ForwardResearchMeasurementState.PENDING
        if reason is _UNDEFINED_BASIS:
            return ForwardResearchMeasurementState.UNAVAILABLE
        raise ValueError(f"FuturesForwardResearchMeasurement has no state for reason {reason}.")

    @property
    def is_measured(self) -> bool:
        """Return whether a factual forward return was measured."""
        return self.state is ForwardResearchMeasurementState.MEASURED


class MeasureFuturesForwardResearchRecordUseCase:
    """Measure one frozen futures forward decision at one horizon, as of one cutoff.

    Measurement is delegated to MeasureFuturesRecommendationOutcomeUseCase, built
    over the same repository, so horizon selection, look-ahead safety and the
    unavailable-reason semantics stay defined in exactly one place for both
    historical and forward futures research.
    """

    def __init__(self, repository: FuturesHistoricalMarketDataRepository) -> None:
        if not isinstance(repository, FuturesHistoricalMarketDataRepository):
            raise TypeError(
                "MeasureFuturesForwardResearchRecordUseCase repository "
                "must be a FuturesHistoricalMarketDataRepository."
            )
        self._measure = MeasureFuturesRecommendationOutcomeUseCase(repository)

    def execute(
        self,
        record: FuturesForwardResearchRecord,
        horizon: ResearchHorizon,
        available_through: PointInTime,
    ) -> FuturesForwardResearchMeasurement:
        """Measure what the market did after one frozen decision, up to the cutoff.

        ``available_through`` is required and must not precede the decision
        instant; a cutoff at the decision instant is valid and reads PENDING,
        because no strictly later observation can exist by then.
        """
        subject = "MeasureFuturesForwardResearchRecordUseCase"
        if not isinstance(record, FuturesForwardResearchRecord):
            raise TypeError(f"{subject} record must be a FuturesForwardResearchRecord.")
        if not isinstance(horizon, ResearchHorizon):
            raise TypeError(f"{subject} horizon must be a ResearchHorizon.")
        if not isinstance(available_through, PointInTime):
            raise TypeError(f"{subject} available-through must be a PointInTime.")

        outcome = self._measure.execute(record.result, horizon, available_through)
        return FuturesForwardResearchMeasurement(record=record, horizon=horizon, outcome=outcome)
