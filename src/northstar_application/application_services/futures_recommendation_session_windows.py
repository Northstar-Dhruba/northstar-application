"""Application resolution of a recommendation's validity window in trading sessions.

F-4 takes a FuturesRecommendationSessionWindow of caller-resolved instants; this
service resolves them from the trading calendar and a verified contract cutoff:

- ``eligible_from``: the close of the first session that *opens strictly after*
  issuance. A session that opened at or before issuance can hold prices traded
  before the recommendation existed, so it is never eligible (F-2 B1), and a
  late issuance simply starts the window later.
- ``valid_through``: the close of the M-th such session, M being the profile's
  validity horizon. The horizon is never shortened.
- ``contract_cutoff``: the close of the contract's verified last session, from
  a FuturesContractCutoff -- never derived from the expiration date.

A horizon that runs past the cutoff still yields a well-formed window; F-4's
contract-life rejection stays authoritative. The cutoff is cross-checked
against the calendar: a last session the calendar does not resolve identically,
one dated after the expiration date, or one for another contract is
contradictory evidence and refuses resolution.

Sessions are only ever resolved, never computed from calendar days, and no
clock is read. Resolver errors -- for example a calendar year that is not
loaded -- propagate unchanged, and a calendar that yields fewer than M sessions
within the search span refuses resolution: no session is ever invented.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from northstar_core.foundation.value_objects import PointInTime
from northstar_core.futures import FuturesContract
from northstar_core.recommendations.futures import FuturesRecommendationSessionWindow

from northstar_application.application_services.futures_recommendation_profile import (
    FuturesTradeRecommendationGenerationProfile,
)
from northstar_application.ports.futures_contract_cutoff_source import FuturesContractCutoff
from northstar_application.ports.futures_trading_session_resolver import (
    FuturesTradingSession,
    FuturesTradingSessionResolver,
)

# A session may open on its label's previous civil day, so the search starts a
# day before issuance. Sessions are fetched in chunks until M are found.
_SEARCH_LEAD = timedelta(days=1)
_SEARCH_CHUNK = timedelta(days=31)


class FuturesRecommendationWindowResolutionError(ValueError):
    """Raised when a recommendation window cannot be resolved from trusted inputs."""


def _utc_date(instant: PointInTime) -> date:
    return datetime.fromisoformat(instant.value.replace("Z", "+00:00")).astimezone(UTC).date()


def _search_span(validity_sessions: int) -> timedelta:
    """Return the calendar span within which M sessions must be found.

    Generous on purpose -- a week per session plus two months of holidays and
    closures -- so that only a genuinely missing calendar refuses resolution.
    """
    return timedelta(days=7 * validity_sessions + 62)


class FuturesRecommendationSessionWindowResolver:
    """Resolves one recommendation's session window from the calendar and a cutoff."""

    def __init__(self, sessions: FuturesTradingSessionResolver) -> None:
        if not isinstance(sessions, FuturesTradingSessionResolver):
            raise TypeError(
                "FuturesRecommendationSessionWindowResolver sessions must be a "
                "FuturesTradingSessionResolver."
            )
        self._sessions = sessions

    def resolve(
        self,
        contract: FuturesContract,
        decision_instant: PointInTime,
        issued_at: PointInTime,
        profile: FuturesTradeRecommendationGenerationProfile,
        cutoff: FuturesContractCutoff,
    ) -> FuturesRecommendationSessionWindow:
        """Return the window of a recommendation on ``contract`` issued at ``issued_at``."""
        self._validate_types(contract, decision_instant, issued_at, profile, cutoff)
        if decision_instant.compare(issued_at) > 0:
            raise FuturesRecommendationWindowResolutionError(
                f"A recommendation cannot be issued at {issued_at}, before its research "
                f"decision at {decision_instant}."
            )
        self._verify_cutoff(contract, cutoff)
        eligible = self._eligible_sessions(contract, issued_at, profile.validity_sessions)
        return FuturesRecommendationSessionWindow(
            contract=contract,
            eligible_from=eligible[0].closes_at,
            valid_through=eligible[-1].closes_at,
            contract_cutoff=cutoff.last_session.closes_at,
        )

    @staticmethod
    def _validate_types(
        contract: object,
        decision_instant: object,
        issued_at: object,
        profile: object,
        cutoff: object,
    ) -> None:
        if not isinstance(contract, FuturesContract):
            raise TypeError("Window resolution contract must be a FuturesContract.")
        if not isinstance(decision_instant, PointInTime) or not isinstance(issued_at, PointInTime):
            raise TypeError("Window resolution instants must be PointInTime values.")
        if not isinstance(profile, FuturesTradeRecommendationGenerationProfile):
            raise TypeError(
                "Window resolution profile must be a FuturesTradeRecommendationGenerationProfile."
            )
        if not isinstance(cutoff, FuturesContractCutoff):
            raise TypeError("Window resolution cutoff must be a FuturesContractCutoff.")

    def _verify_cutoff(self, contract: FuturesContract, cutoff: FuturesContractCutoff) -> None:
        if cutoff.contract != contract:
            raise FuturesRecommendationWindowResolutionError(
                f"The contract cutoff is for {cutoff.contract}, not {contract}."
            )
        last = cutoff.last_session
        if last.trading_date > date.fromisoformat(contract.expiration_date.value):
            raise FuturesRecommendationWindowResolutionError(
                f"The verified last session {last.trading_date.isoformat()} of {contract} is "
                "after its expiration date."
            )
        resolved = self._sessions.resolve(contract.product, last.trading_date)
        if resolved != last:
            raise FuturesRecommendationWindowResolutionError(
                f"The verified last session {last} of {contract} does not match the trading "
                f"calendar ({resolved if resolved is not None else 'not a session'})."
            )

    def _eligible_sessions(
        self, contract: FuturesContract, issued_at: PointInTime, count: int
    ) -> tuple[FuturesTradingSession, ...]:
        start = _utc_date(issued_at) - _SEARCH_LEAD
        limit = start + _search_span(count)
        eligible: list[FuturesTradingSession] = []
        chunk_start = start
        while len(eligible) < count and chunk_start <= limit:
            chunk_end = min(chunk_start + _SEARCH_CHUNK, limit)
            for session in self._sessions.sessions_in_range(
                contract.product, chunk_start, chunk_end
            ):
                if session.opens_at.compare(issued_at) > 0 and len(eligible) < count:
                    eligible.append(session)
            chunk_start = chunk_end + timedelta(days=1)
        if len(eligible) < count:
            raise FuturesRecommendationWindowResolutionError(
                f"The trading calendar of {contract.product} resolves only {len(eligible)} of "
                f"{count} session(s) opening after {issued_at} within {limit.isoformat()}."
            )
        return tuple(eligible)
