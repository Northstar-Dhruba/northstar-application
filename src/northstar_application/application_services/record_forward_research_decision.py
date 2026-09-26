"""Application freezing of one live recommendation for later measurement.

A forward research record captures what a strategy advised at one decision
instant, together with the evidence it decided from, so the outcome can be
measured once later observations exist. Recording performs no interpretation,
reads no clock, and models no order, fill, position or profit and loss.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import (
    ExchangeCode,
    PointInTime,
    Symbol,
    Timeframe,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services._result_coherence import (
    validate_result_coherence,
)
from northstar_application.application_services.analyze_asset_result import AnalyzeAssetResult
from northstar_application.ports import ForwardResearchRecordStore


class ForwardResearchContractViolationError(ValueError):
    """Raised when a forward research store violates the port contract."""


@dataclass(frozen=True, slots=True)
class ForwardResearchRecord:
    """One frozen decision and the observation interval it was decided on.

    ``result`` is the complete decision-time evidence: the recommendation, its
    explanation, and the market observation context it was produced from.
    ``timeframe`` records the observation interval, which the analysis contracts
    do not carry and which later outcome measurement requires.

    Identity, decision instant and strategy are not duplicated as fields; they
    remain owned by the recorded result and are surfaced as derived properties.
    """

    result: AnalyzeAssetResult
    timeframe: Timeframe

    def __post_init__(self) -> None:
        if self.result is None:
            raise TypeError("ForwardResearchRecord result cannot be None.")
        if not isinstance(self.result, AnalyzeAssetResult):
            raise TypeError("ForwardResearchRecord result must be an AnalyzeAssetResult.")
        if self.timeframe is None:
            raise TypeError("ForwardResearchRecord timeframe cannot be None.")
        if not isinstance(self.timeframe, Timeframe):
            raise TypeError("ForwardResearchRecord timeframe must be a Timeframe.")

        validate_result_coherence(self.result, "ForwardResearchRecord")

    @property
    def listing_reference(self) -> ListingReference:
        """Return the listed asset the decision was made about."""
        return self.result.market_observation_context.listing_reference

    @property
    def strategy_identity(self) -> StrategyIdentity:
        """Return the strategy that produced the decision."""
        return self.result.recommendation.strategy_identity

    @property
    def decision_instant(self) -> PointInTime:
        """Return the completion instant of the observation decided upon."""
        return self.result.market_observation_context.observed_at

    @property
    def natural_key(self) -> tuple[Symbol, ExchangeCode, Timeframe, PointInTime, StrategyIdentity]:
        """Return the identity under which this decision is frozen."""
        return (
            self.listing_reference.symbol,
            self.listing_reference.exchange_code,
            self.timeframe,
            self.decision_instant,
            self.strategy_identity,
        )


class RecordForwardResearchDecisionUseCase:
    """Freeze one produced recommendation so its outcome can be measured later."""

    def __init__(self, store: ForwardResearchRecordStore) -> None:
        if store is None:
            raise TypeError("RecordForwardResearchDecisionUseCase store cannot be None.")
        if not isinstance(store, ForwardResearchRecordStore):
            raise TypeError(
                "RecordForwardResearchDecisionUseCase store must be a ForwardResearchRecordStore."
            )
        self._store = store

    def execute(self, result: AnalyzeAssetResult, timeframe: Timeframe) -> ForwardResearchRecord:
        """Freeze one decision and return the record that was persisted."""
        record = ForwardResearchRecord(result=result, timeframe=timeframe)

        stored = self._store.store((record,))
        if not isinstance(stored, int) or isinstance(stored, bool):
            raise ForwardResearchContractViolationError(
                "ForwardResearchRecordStore.store() must return an integer count."
            )
        if stored != 1:
            raise ForwardResearchContractViolationError(
                f"ForwardResearchRecordStore.store() returned count {stored}, expected 1 "
                "for the complete batch."
            )

        return record
