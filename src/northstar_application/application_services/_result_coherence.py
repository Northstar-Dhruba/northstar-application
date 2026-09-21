"""Shared coherence checks for one produced AnalyzeAssetResult.

An AnalyzeAssetResult carries three views of one decision: the recommendation,
the analysis it was derived from, and the market observation both were produced
against. Consumers read identity and instant from whichever view is convenient,
so a result whose views disagree would silently attribute a record, a
measurement or an execution to the wrong asset or the wrong instant.

The checks live here because four services need them and a fix to one must be a
fix to all. Each caller supplies its own subject so error messages stay
attributable, and each chooses how much coherence it requires: measurement
constrains only the recommendation instant, while recording and execution
require the full set.

Instants are compared with PointInTime.compare() so an offset-equivalent
spelling is judged chronologically rather than by its text.

This module is internal to the Application Services package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from northstar_application.application_services.analyze_asset_result import AnalyzeAssetResult


def validate_recommendation_instant(result: AnalyzeAssetResult, subject: str) -> None:
    """Require the recommendation to be dated at the observation it used."""
    context = result.market_observation_context
    if result.recommendation.point_in_time.compare(context.observed_at) != 0:
        raise ValueError(
            f"{subject} recommendation instant must match the observed market context instant."
        )


def validate_result_coherence(result: AnalyzeAssetResult, subject: str) -> None:
    """Require the recommendation, its analysis and the observation to agree."""
    validate_recommendation_instant(result, subject)

    context = result.market_observation_context
    analysis = result.recommendation.asset_analysis
    if analysis.point_in_time.compare(context.observed_at) != 0:
        raise ValueError(
            f"{subject} asset analysis instant must match the observed market context instant."
        )
    if analysis.listing_reference != context.listing_reference:
        raise ValueError(
            f"{subject} asset analysis listing must match the observed market context listing."
        )
