"""Application port for resolving futures daily trading-session completion instants.

A daily FuturesOHLCVBar needs a completion instant, and that instant is a
property of the venue's session calendar rather than of any provider's payload.
This port is where Application asks for it, so no source adapter ever has to
derive one for itself.

Why this is not TradingSessionResolver
--------------------------------------
The equity port resolves the close of a *regular session* from an ExchangeCode.
Futures sessions are a different shape: a CME session runs from 17:00 Chicago
to 17:00 Chicago, so it opens on the previous civil date and there is no
regular session in the equity sense. Reusing the equity port would mean reusing
a name and a signature whose meaning does not hold here, and it would pin the
equity contract to an input type futures may later need to change. The two
ports therefore stay independent; nothing here imports or modifies the equity
one.

Superseded
----------
This port is superseded by FuturesTradingSessionResolver, which returns the
whole session window rather than only its completion instant, because daily
aggregation needs both boundaries. It is retained only until
northstar-infrastructure has migrated, and will then be removed. Nothing new
should implement or depend on it, and no adapter needs both.

FuturesSessionResolutionError now lives with the replacement port and is
re-exported here unchanged, so existing `except FuturesSessionResolutionError`
handlers keep catching exactly the same class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from northstar_core.foundation.value_objects import PointInTime
from northstar_core.futures import FuturesProductReference

from northstar_application.ports.futures_trading_session_resolver import (
    FuturesSessionResolutionError,
)

__all__ = ["FuturesDailyBarCompletionResolver", "FuturesSessionResolutionError"]


class FuturesDailyBarCompletionResolver(ABC):
    """Resolves the completion instant of one daily futures trading session.

    ``trading_date`` is the exchange's *session label*, not a civil day over
    which to aggregate. A CME session labelled 2026-09-15 opens at
    2026-09-14T22:00Z and completes at 2026-09-15T22:00Z, so the label and the
    opening civil date routinely differ. Implementations must treat the label
    as the session's own name and must never derive it by converting an instant
    to a UTC date.

    The returned PointInTime is the instant at which that session completes:
    the earliest logical instant at which the completed daily bar may take part
    in deterministic replay. It is deliberately none of the following, and
    implementations must not substitute them:

    - the official settlement time, which is a computed mark struck inside a
      closing window and differs per product
    - the settlement publication time, which is later again and may fall on the
      next calendar day for final settlements
    - the provider's arrival or ingestion time
    - any timestamp carried on a provider's payload

    Those four are genuinely distinct events, and collapsing any of them into
    this one would make a stored bar claim an availability it did not have.

    Implementations must honour these semantics:

    - A date that is a trading session for the product's venue resolves to that
      session's completion instant.
    - A date that is not a session for that venue -- a weekend, an exchange
      holiday or a scheduled closure -- returns None.
    - An unsupported or unknown venue, or a calendar that cannot be resolved
      reliably, raises FuturesSessionResolutionError rather than returning
      None. A missing answer and "no session that day" are different facts and
      must not share a representation.
    - The completion instant must come from the venue's session calendar, not
      from fixed-duration arithmetic on the session open. Sessions are not all
      the same length: an early close can end five hours sooner than a full
      session while the surrounding sessions are unaffected.

    The abstract contract documents these obligations but cannot enforce them
    at runtime.
    """

    @abstractmethod
    def resolve_completion(
        self, product: FuturesProductReference, trading_date: date
    ) -> PointInTime | None:
        """Return the session completion PointInTime, or None if not a session.

        ``product`` is a FuturesProductReference rather than a bare
        ExchangeCode. Session schedules resolve at venue level today, so an
        implementation is expected to read only ``product.exchange_code``. The
        product is nevertheless the smallest stable input: if a venue ever
        needs a per-product schedule, that arrives without changing this
        signature, and the reference already carries the exchange.

        It is deliberately not a FuturesContract. An expiry does not determine
        the daily session schedule, so taking one would invite a caller to
        believe otherwise.
        """
