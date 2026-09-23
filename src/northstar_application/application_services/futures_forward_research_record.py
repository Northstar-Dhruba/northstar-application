"""Frozen futures forward research decision.

A forward research record freezes what a strategy advised about one concrete
futures contract at one decision instant, together with the evidence it decided
from, so the outcome can be measured once later observations exist. It is the
futures parallel of ForwardResearchRecord and differs in one field: the equity
record stores its timeframe separately because the equity observation context
carries none, whereas the futures context already does. The record therefore
holds exactly one value, the analysis result, and derives everything else.

Freezing performs no interpretation and reads no clock. BUY, HOLD and SELL are
directional research classifications; nothing here is an order, a position, a
margin requirement, a multiplier or a profit.
"""

from __future__ import annotations

from dataclasses import dataclass

from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.futures import FuturesContract
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services.futures_analysis_result import (
    FuturesAnalysisResult,
)
from northstar_application.application_services.replay_futures_historical_market_data import (
    UnsupportedFuturesReplayTimeframeError,
)

_SUPPORTED_TIMEFRAME = Timeframe("1d")


@dataclass(frozen=True, slots=True)
class FuturesForwardResearchRecord:
    """One frozen futures decision and the complete evidence it was decided on.

    ``result`` holds the recommendation and the market observation context it
    was produced from. Contract, timeframe, decision instant and strategy are
    not duplicated as fields; they are owned by the result and surfaced as
    derived properties, so a record cannot disagree with its own evidence.

    Equality is full value equality of the result. Two records can share a
    natural key and still be unequal -- that is precisely the conflict a store
    must refuse, because a frozen decision is never rewritten.

    The result already guarantees that its recommendation and context describe
    one contract at one instant. The record adds the one rule the result cannot:
    forward research is session-daily, and a hand-built result may carry any
    timeframe.
    """

    result: FuturesAnalysisResult

    def __post_init__(self) -> None:
        if not isinstance(self.result, FuturesAnalysisResult):
            raise TypeError("FuturesForwardResearchRecord result must be a FuturesAnalysisResult.")
        timeframe = self.result.market_observation_context.timeframe
        if timeframe != _SUPPORTED_TIMEFRAME:
            raise UnsupportedFuturesReplayTimeframeError(
                f"FuturesForwardResearchRecord supports session-daily research only; "
                f"timeframe {timeframe} is not {_SUPPORTED_TIMEFRAME}."
            )

    @property
    def contract(self) -> FuturesContract:
        """Return the concrete contract the decision was made about."""
        return self.result.recommendation.contract

    @property
    def timeframe(self) -> Timeframe:
        """Return the observation interval the decision was made on."""
        return self.result.market_observation_context.timeframe

    @property
    def decision_instant(self) -> PointInTime:
        """Return the completion instant of the observation decided upon."""
        return self.result.market_observation_context.observed_at

    @property
    def strategy_identity(self) -> StrategyIdentity:
        """Return the strategy that produced the decision."""
        return self.result.recommendation.strategy_identity

    @property
    def natural_key(self) -> tuple[FuturesContract, Timeframe, PointInTime, StrategyIdentity]:
        """Return the identity under which this decision is frozen.

        Every part compares by value. PointInTime canonicalises to UTC, so two
        spellings of one instant are one key; chronological ordering still
        belongs to PointInTime.compare(), never to this tuple.
        """
        return (self.contract, self.timeframe, self.decision_instant, self.strategy_identity)
