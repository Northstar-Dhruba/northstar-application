"""The closed vocabularies describing how one recommendation generation ended.

They live apart from the use case so that the generation audit
(futures_recommendation_generation_audit) can name them without importing the
use case that builds audits. The use case re-exports them unchanged.
"""

from __future__ import annotations

from enum import StrEnum


class FuturesTradeRecommendationGenerationStatus(StrEnum):
    """Whether generation ran to completion or was blocked before generating."""

    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"


class FuturesTradeRecommendationBlockReason(StrEnum):
    """Why generation was blocked; nothing was generated or recorded."""

    NO_PERSISTED_SESSION = "NO_PERSISTED_SESSION"
    SESSION_NOT_FINAL = "SESSION_NOT_FINAL"
    SESSION_NOT_OPERATOR_REVIEWED = "SESSION_NOT_OPERATOR_REVIEWED"
    NO_RESEARCH_FOR_SESSION = "NO_RESEARCH_FOR_SESSION"
    RESEARCH_HISTORY_NOT_VERIFIED = "RESEARCH_HISTORY_NOT_VERIFIED"
    CONTRACT_CUTOFF_UNAVAILABLE = "CONTRACT_CUTOFF_UNAVAILABLE"


class FuturesResearchHistoryCoverage(StrEnum):
    """Whether the research history behind the decision was verified complete."""

    VERIFIED = "VERIFIED"
    UNVERIFIED = "UNVERIFIED"


class FuturesResearchHistoryDeficiency(StrEnum):
    """Why research history is UNVERIFIED; the first that applies is reported."""

    INCOMPLETE_CALENDAR = "INCOMPLETE_CALENDAR"
    MISSING_BAR = "MISSING_BAR"
    UNAPPROVED_CANDLE = "UNAPPROVED_CANDLE"
