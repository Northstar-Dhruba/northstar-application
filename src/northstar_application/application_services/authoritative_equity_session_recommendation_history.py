"""The authoritative issued-history source over the issuance authority (EI-5D, ADR-024 6).

``AuthoritativeEquitySessionRecommendationHistorySource`` implements the
existing ``EquitySessionRecommendationHistorySource`` port that EI-4C already
consumes, over the issuance authority's **reader** only. It replaces nothing:
synthetic fixtures stay test-only, and the history contract is unchanged
(``EquitySessionRecommendationHistory``, ``EquityPriorSessionRecommendation``).

- ``session_history(exchange, date, as_of)`` returns the authority's current
  view of the session. The caller keeps refusing a view whose
  ``complete_through`` is before ``as_of``, as EI-4C already does: this source
  never states completeness the authority has not declared.
- A session with no declaration is unknown: None, never an empty tuple (EI5-D4;
  EI4-D9-H). An empty tuple means a declaration exists and lists no issuance.
- Every view names the issuance authority,
  ``northstar.equity-recommendation-issuance-authority/1`` (D5D-3). A view
  naming anything else, for another session, or one the authority cannot
  read raises EquityRecommendationHistoryUnavailableError. Nothing is
  repaired, inferred or backdated.

Records carry no lifecycle outcome: every issued recommendation stays open
for EI-4C's exposure and exclusivity checks (EI4-D8, unchanged).
"""

from __future__ import annotations

from datetime import date

from northstar_core.foundation.value_objects import ExchangeCode, PointInTime

from northstar_application.application_services.equity_recommendation_issuance_records import (
    EQUITY_RECOMMENDATION_ISSUANCE_AUTHORITY_REFERENCE,
)
from northstar_application.ports.equity_recommendation_issuance import (
    EquityRecommendationIssuanceReader,
    EquityRecommendationIssuanceUnavailableError,
)
from northstar_application.ports.equity_recommendation_references import (
    EquityRecommendationHistoryUnavailableError,
    EquitySessionRecommendationHistory,
    EquitySessionRecommendationHistorySource,
)


class AuthoritativeEquitySessionRecommendationHistorySource(
    EquitySessionRecommendationHistorySource
):
    """EI-4C's issued-history port, answered by the issuance authority's reader."""

    def __init__(self, reader: EquityRecommendationIssuanceReader) -> None:
        if not isinstance(reader, EquityRecommendationIssuanceReader):
            raise TypeError("reader must be an EquityRecommendationIssuanceReader.")
        self._reader = reader

    def session_history(
        self, exchange_code: ExchangeCode, trading_date: date, as_of: PointInTime
    ) -> EquitySessionRecommendationHistory | None:
        """Return the authority's current view of the session, or None when undeclared."""
        if not isinstance(as_of, PointInTime):
            raise TypeError("as_of must be a PointInTime.")
        try:
            view = self._reader.session_view(exchange_code, trading_date)
        except EquityRecommendationIssuanceUnavailableError as exc:
            raise EquityRecommendationHistoryUnavailableError(
                f"The issuance authority cannot be read: {exc}"
            ) from exc
        if view is None:
            return None
        if not isinstance(view, EquitySessionRecommendationHistory):
            raise EquityRecommendationHistoryUnavailableError(
                "The issuance authority answered something other than a session history."
            )
        if view.reference != EQUITY_RECOMMENDATION_ISSUANCE_AUTHORITY_REFERENCE:
            raise EquityRecommendationHistoryUnavailableError(
                f"The view names {view.reference!r}, not the issuance authority."
            )
        if (view.exchange_code, view.trading_date) != (exchange_code, trading_date):
            raise EquityRecommendationHistoryUnavailableError(
                "The issuance authority answered for another session."
            )
        return view
