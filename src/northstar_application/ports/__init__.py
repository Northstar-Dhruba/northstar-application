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
from northstar_application.ports.futures_daily_bar_completion_resolver import (
    FuturesDailyBarCompletionResolver,
    FuturesSessionResolutionError,
)
from northstar_application.ports.futures_historical_market_data_repository import (
    FuturesHistoricalMarketDataQuery,
    FuturesHistoricalMarketDataRepository,
    InvalidFuturesHistoricalMarketDataQueryError,
)
from northstar_application.ports.futures_historical_market_data_store import (
    FuturesHistoricalMarketDataConflictError,
    FuturesHistoricalMarketDataStore,
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
from northstar_application.ports.paper_fill_repository import (
    InvalidPaperFillQueryError,
    PaperFillQuery,
    PaperFillRepository,
)
from northstar_application.ports.paper_fill_store import (
    PaperFillConflictError,
    PaperFillStore,
)
from northstar_application.ports.trading_session_resolver import (
    TradingSessionResolutionError,
    TradingSessionResolver,
)

__all__ = [
    "ForwardResearchRecordConflictError",
    "FuturesDailyBarCompletionResolver",
    "FuturesHistoricalMarketDataConflictError",
    "FuturesHistoricalMarketDataQuery",
    "FuturesHistoricalMarketDataRepository",
    "FuturesHistoricalMarketDataStore",
    "FuturesSessionResolutionError",
    "InvalidFuturesHistoricalMarketDataQueryError",
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
    "InvalidPaperFillQueryError",
    "PaperFillConflictError",
    "PaperFillQuery",
    "PaperFillRepository",
    "PaperFillStore",
    "TradingSessionResolutionError",
    "TradingSessionResolver",
]
