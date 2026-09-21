"""Application-owned abstractions for external capabilities."""

from northstar_application.ports.forward_research_record_repository import (
    ForwardResearchRecordQuery,
    ForwardResearchRecordRepository,
    InvalidForwardResearchRecordQueryError,
)
from northstar_application.ports.forward_research_record_store import (
    ForwardResearchRecordConflictError,
    ForwardResearchRecordStore,
)
from northstar_application.ports.historical_market_data_repository import (
    HistoricalMarketDataQuery,
    HistoricalMarketDataRepository,
    InvalidHistoricalMarketDataQueryError,
)
from northstar_application.ports.historical_market_data_source import HistoricalMarketDataSource
from northstar_application.ports.historical_market_data_store import HistoricalMarketDataStore
from northstar_application.ports.historical_snapshot_evaluator import HistoricalSnapshotEvaluator
from northstar_application.ports.market_observation_source import MarketObservationSource
from northstar_application.ports.trading_session_resolver import (
    TradingSessionResolutionError,
    TradingSessionResolver,
)

__all__ = [
    "ForwardResearchRecordConflictError",
    "ForwardResearchRecordQuery",
    "ForwardResearchRecordRepository",
    "ForwardResearchRecordStore",
    "InvalidForwardResearchRecordQueryError",
    "HistoricalMarketDataQuery",
    "HistoricalMarketDataRepository",
    "HistoricalMarketDataSource",
    "HistoricalMarketDataStore",
    "HistoricalSnapshotEvaluator",
    "InvalidHistoricalMarketDataQueryError",
    "MarketObservationSource",
    "TradingSessionResolutionError",
    "TradingSessionResolver",
]
