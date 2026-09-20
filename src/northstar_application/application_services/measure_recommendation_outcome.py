"""Application orchestration for measuring one recommendation outcome.

Measurement is purely factual. It records what the market did after a
recommendation was produced and never interprets BUY, SELL or HOLD, and never
models orders, fills, positions or profit and loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.market_data import HistoricalOHLCVBar
from northstar_core.strategy import (
    Recommendation,
    RecommendationOutcome,
    ResearchHorizon,
)

from northstar_application.application_services.evaluate_historical_research import (
    HistoricalResearchEvaluation,
)
from northstar_application.application_services.ingest_historical_market_data import (
    HistoricalDataContractViolationError,
)
from northstar_application.ports import (
    HistoricalMarketDataQuery,
    HistoricalMarketDataRepository,
)


class RecommendationOutcomeUnavailableReason(StrEnum):
    """Stable Application-level reasons why an outcome could not be measured.

    Both reasons describe normal unmeasurable historical states rather than
    failures. A cumulative observation series may legitimately span a currency
    transition, which leaves the forward movement without a same-basis
    measurement.
    """

    INSUFFICIENT_FUTURE_OBSERVATIONS = "INSUFFICIENT_FUTURE_OBSERVATIONS"
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"


@dataclass(frozen=True, slots=True)
class RecommendationOutcomeMeasurement:
    """Outcome measurement attempt for one recommendation at one horizon.

    Exactly one of ``outcome`` and ``unavailable`` is present. A measurement is
    always returned so a recommendation is never silently omitted, and an
    unmeasurable recommendation is never represented as a zero movement.
    """

    recommendation: Recommendation
    horizon: ResearchHorizon
    outcome: RecommendationOutcome | None = None
    unavailable: RecommendationOutcomeUnavailableReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.recommendation, Recommendation):
            raise TypeError(
                "RecommendationOutcomeMeasurement recommendation must be a Recommendation."
            )
        if not isinstance(self.horizon, ResearchHorizon):
            raise TypeError("RecommendationOutcomeMeasurement horizon must be a ResearchHorizon.")
        if self.outcome is not None and not isinstance(self.outcome, RecommendationOutcome):
            raise TypeError(
                "RecommendationOutcomeMeasurement outcome must be a RecommendationOutcome."
            )
        if self.unavailable is not None and not isinstance(
            self.unavailable, RecommendationOutcomeUnavailableReason
        ):
            raise TypeError(
                "RecommendationOutcomeMeasurement unavailable must be a "
                "RecommendationOutcomeUnavailableReason."
            )
        if (self.outcome is None) == (self.unavailable is None):
            raise ValueError(
                "RecommendationOutcomeMeasurement must carry exactly one of outcome or unavailable."
            )
        if self.outcome is not None:
            if self.outcome.recommendation != self.recommendation:
                raise ValueError(
                    "RecommendationOutcomeMeasurement outcome must measure the same recommendation."
                )
            if self.outcome.horizon != self.horizon:
                raise ValueError(
                    "RecommendationOutcomeMeasurement outcome must measure the same horizon."
                )

    @property
    def is_measured(self) -> bool:
        """Return whether a factual outcome was measured."""
        return self.outcome is not None


class MeasureRecommendationOutcomeUseCase:
    """Measure one historical research evaluation at one research horizon.

    Future observations are used only to record what subsequently happened.
    They never re-enter signal generation, recommendation selection or the
    decision-time evidence already captured by the evaluation.
    """

    def __init__(self, repository: HistoricalMarketDataRepository) -> None:
        if repository is None:
            raise TypeError("MeasureRecommendationOutcomeUseCase repository cannot be None.")
        if not isinstance(repository, HistoricalMarketDataRepository):
            raise TypeError(
                "MeasureRecommendationOutcomeUseCase repository must be a "
                "HistoricalMarketDataRepository."
            )
        self._repository = repository

    def execute(
        self,
        evaluation: HistoricalResearchEvaluation,
        horizon: ResearchHorizon,
        timeframe: Timeframe,
        available_through: PointInTime,
    ) -> RecommendationOutcomeMeasurement:
        """Measure the forward market movement following one evaluated decision."""
        self._validate_inputs(evaluation, horizon, timeframe, available_through)

        context = evaluation.result.market_observation_context
        recommendation = evaluation.result.recommendation
        decision_instant = context.observed_at
        if recommendation.point_in_time.compare(decision_instant) != 0:
            raise ValueError(
                "MeasureRecommendationOutcomeUseCase recommendation instant must match "
                "the observed market context instant."
            )

        if available_through.compare(decision_instant) < 0:
            return self._unavailable(
                recommendation,
                horizon,
                RecommendationOutcomeUnavailableReason.INSUFFICIENT_FUTURE_OBSERVATIONS,
            )

        listing_reference = context.listing_reference
        query = HistoricalMarketDataQuery(
            symbol=listing_reference.symbol,
            exchange_code=listing_reference.exchange_code,
            timeframe=timeframe,
            start=decision_instant,
            end=available_through,
        )
        observations = self._repository.get_history(query)
        self._validate_observations(observations, query)

        future_observations = tuple(
            bar for bar in observations if bar.point_in_time.compare(decision_instant) > 0
        )
        if len(future_observations) < horizon.observations:
            return self._unavailable(
                recommendation,
                horizon,
                RecommendationOutcomeUnavailableReason.INSUFFICIENT_FUTURE_OBSERVATIONS,
            )

        selected = future_observations[horizon.observations - 1]
        decision_price = context.latest_price
        if selected.close.currency != decision_price.currency:
            return self._unavailable(
                recommendation,
                horizon,
                RecommendationOutcomeUnavailableReason.CURRENCY_MISMATCH,
            )

        return RecommendationOutcomeMeasurement(
            recommendation=recommendation,
            horizon=horizon,
            outcome=RecommendationOutcome(
                recommendation=recommendation,
                horizon=horizon,
                decision_price=decision_price,
                evaluation_instant=selected.point_in_time,
                evaluation_price=selected.close,
            ),
        )

    @staticmethod
    def _unavailable(
        recommendation: Recommendation,
        horizon: ResearchHorizon,
        reason: RecommendationOutcomeUnavailableReason,
    ) -> RecommendationOutcomeMeasurement:
        return RecommendationOutcomeMeasurement(
            recommendation=recommendation,
            horizon=horizon,
            unavailable=reason,
        )

    @staticmethod
    def _validate_inputs(
        evaluation: HistoricalResearchEvaluation,
        horizon: ResearchHorizon,
        timeframe: Timeframe,
        available_through: PointInTime,
    ) -> None:
        if evaluation is None:
            raise TypeError("MeasureRecommendationOutcomeUseCase evaluation cannot be None.")
        if not isinstance(evaluation, HistoricalResearchEvaluation):
            raise TypeError(
                "MeasureRecommendationOutcomeUseCase evaluation must be a "
                "HistoricalResearchEvaluation."
            )
        if horizon is None:
            raise TypeError("MeasureRecommendationOutcomeUseCase horizon cannot be None.")
        if not isinstance(horizon, ResearchHorizon):
            raise TypeError(
                "MeasureRecommendationOutcomeUseCase horizon must be a ResearchHorizon."
            )
        if timeframe is None:
            raise TypeError("MeasureRecommendationOutcomeUseCase timeframe cannot be None.")
        if not isinstance(timeframe, Timeframe):
            raise TypeError("MeasureRecommendationOutcomeUseCase timeframe must be a Timeframe.")
        if available_through is None:
            raise TypeError("MeasureRecommendationOutcomeUseCase available-through cannot be None.")
        if not isinstance(available_through, PointInTime):
            raise TypeError(
                "MeasureRecommendationOutcomeUseCase available-through must be a PointInTime."
            )

    @staticmethod
    def _validate_observations(observations: object, query: HistoricalMarketDataQuery) -> None:
        if not isinstance(observations, tuple):
            raise HistoricalDataContractViolationError(
                "HistoricalMarketDataRepository must return a tuple of HistoricalOHLCVBar."
            )

        seen_identities: set[tuple[str, str, str, str]] = set()
        for index, bar in enumerate(observations):
            if not isinstance(bar, HistoricalOHLCVBar):
                raise HistoricalDataContractViolationError(
                    "Historical repository observation must be a HistoricalOHLCVBar instance."
                )
            if bar.symbol != query.symbol:
                raise HistoricalDataContractViolationError(
                    f"Historical repository observation symbol {bar.symbol} "
                    f"does not match query symbol {query.symbol}."
                )
            if bar.exchange_code != query.exchange_code:
                raise HistoricalDataContractViolationError(
                    f"Historical repository observation exchange {bar.exchange_code} "
                    f"does not match query exchange {query.exchange_code}."
                )
            if bar.timeframe != query.timeframe:
                raise HistoricalDataContractViolationError(
                    f"Historical repository observation timeframe {bar.timeframe} "
                    f"does not match query timeframe {query.timeframe}."
                )
            if (
                query.start.compare(bar.point_in_time) > 0
                or query.end.compare(bar.point_in_time) < 0
            ):
                raise HistoricalDataContractViolationError(
                    f"Historical repository observation point in time {bar.point_in_time} "
                    f"is outside query bounds [{query.start}, {query.end}]."
                )

            identity = (
                bar.symbol.value,
                bar.exchange_code.value,
                bar.timeframe.value,
                bar.point_in_time.value,
            )
            if identity in seen_identities:
                raise HistoricalDataContractViolationError(
                    "Historical repository observations cannot contain duplicate logical bars."
                )
            seen_identities.add(identity)

            if index and observations[index - 1].point_in_time.compare(bar.point_in_time) > 0:
                raise HistoricalDataContractViolationError(
                    "Historical repository observations must be ordered oldest to newest."
                )
