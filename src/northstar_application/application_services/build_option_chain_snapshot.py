"""Application use case for building a point-in-time option chain snapshot.

The workflow is: resolve the one option session the trading date names, take
its close as ``as_of``, read the expiration's contracts Northstar knew were
listed by ``as_of``, read the expiration's daily bars stamped exactly at
``as_of``, and join the two with the listing set on the left.

Listed, observed, selected
--------------------------
The listing set decides membership. A bar never establishes it: a bar for a
contract whose listing Northstar first observed after ``as_of`` -- one acquired
later with hindsight -- is excluded, however exactly it is stamped. Every known
contract becomes an entry, with its exact bar or with none; a missing bar means
only that no canonical daily bar is stored, never that nothing traded or that a
provider returned no candle. Nothing is selected, ranked or filtered.

Market time, not availability
-----------------------------
``as_of`` is the session close a bar stamped at that close describes. It does
not claim Northstar held or finalized the bar at that instant: historical bars
are acquired later. Using a bar for a forward decision needs an availability
or finality gate that this use case does not provide.

Built on demand
---------------
Nothing is written. Listings and bars are immutable, so a rebuilt snapshot for
the same date always has the same entries, but a bar acquired later can fill an
entry that had none. The snapshot is a projection, not a frozen decision record.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from northstar_core.derivatives import ExpirationDate
from northstar_core.foundation.value_objects import PointInTime, Timeframe
from northstar_core.options import (
    InvalidOptionChainEntryError,
    InvalidOptionChainSnapshotError,
    OptionChainEntry,
    OptionChainSnapshot,
    OptionContract,
    OptionOHLCVBar,
    OptionProductReference,
    OptionRight,
)

from northstar_application.ports import (
    OptionChainDailyBarQuery,
    OptionChainDailyBarRepository,
    OptionListedContractQuery,
    OptionListedContractRepository,
    OptionTradingSession,
    OptionTradingSessionResolver,
)

_DAILY = Timeframe("1d")
_SUBJECT = "BuildOptionChainSnapshotUseCase"
_RIGHT_RANK = {OptionRight.CALL: 0, OptionRight.PUT: 1}


class InvalidOptionChainSnapshotQueryError(ValueError):
    """Raised when an option chain snapshot query is invalid."""


class OptionChainSessionNotFoundError(ValueError):
    """Raised when the requested trading date is not an option trading session."""

    def __init__(self, product: OptionProductReference, trading_date: date) -> None:
        super().__init__(
            f"{trading_date.isoformat()} is not a {product} option trading session "
            "(a weekend or a trading holiday)."
        )
        self.product = product
        self.trading_date = trading_date


class OptionChainListingNotKnownError(ValueError):
    """Raised when Northstar knew of no listed contract of the expiration by the instant."""

    def __init__(
        self, product: OptionProductReference, expiration_date: ExpirationDate, as_of: PointInTime
    ) -> None:
        super().__init__(
            f"No {product} {expiration_date} option listing was known by {as_of}. "
            "Listings count only once Northstar has observed them; later instruments "
            "syncs and stored bars do not reach back."
        )
        self.product = product
        self.expiration_date = expiration_date
        self.as_of = as_of


class OptionChainContractViolationError(ValueError):
    """Raised when a session resolver or chain repository violates its port contract."""


@dataclass(frozen=True, slots=True)
class OptionChainSnapshotQuery:
    """Request for one product's one expiration as of one trading date's session close.

    The expiration may fall on the trading date itself; one before it has
    expired and is refused.
    """

    product: OptionProductReference
    expiration_date: ExpirationDate
    trading_date: date

    def __post_init__(self) -> None:
        subject = "OptionChainSnapshotQuery"
        if not isinstance(self.product, OptionProductReference):
            raise InvalidOptionChainSnapshotQueryError(
                f"{subject} product must be an OptionProductReference value."
            )
        if not isinstance(self.expiration_date, ExpirationDate):
            raise InvalidOptionChainSnapshotQueryError(
                f"{subject} expiration date must be an ExpirationDate value."
            )
        if isinstance(self.trading_date, datetime) or not isinstance(self.trading_date, date):
            raise InvalidOptionChainSnapshotQueryError(
                f"{subject} trading date must be a plain date."
            )
        if self.expiration_date.value < self.trading_date.isoformat():
            raise InvalidOptionChainSnapshotQueryError(
                f"{subject} expiration {self.expiration_date} is before trading date "
                f"{self.trading_date.isoformat()}; the contracts have expired."
            )


class BuildOptionChainSnapshotUseCase:
    """Build one expiration's point-in-time chain from listing knowledge and stored bars."""

    def __init__(
        self,
        session_resolver: OptionTradingSessionResolver,
        listed_contracts: OptionListedContractRepository,
        chain_bars: OptionChainDailyBarRepository,
    ) -> None:
        for name, value, expected in (
            ("session_resolver", session_resolver, OptionTradingSessionResolver),
            ("listed_contracts", listed_contracts, OptionListedContractRepository),
            ("chain_bars", chain_bars, OptionChainDailyBarRepository),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"{_SUBJECT} {name} must be an {expected.__name__}.")
        self._session_resolver = session_resolver
        self._listed_contracts = listed_contracts
        self._chain_bars = chain_bars

    def execute(self, query: OptionChainSnapshotQuery) -> OptionChainSnapshot:
        """Return the expiration's chain as of the trading date's session close."""
        if not isinstance(query, OptionChainSnapshotQuery):
            raise TypeError(f"{_SUBJECT} query must be an OptionChainSnapshotQuery.")

        session = self._session(query)
        as_of = session.closes_at

        contracts = _validate_contracts(
            self._listed_contracts.listed_contracts(
                OptionListedContractQuery(query.product, query.expiration_date, as_of)
            ),
            query,
        )
        if not contracts:
            raise OptionChainListingNotKnownError(query.product, query.expiration_date, as_of)

        bars = _validate_bars(
            self._chain_bars.daily_bars_at(
                OptionChainDailyBarQuery(query.product, query.expiration_date, as_of)
            ),
            query,
            as_of,
        )
        # The listing set is the left side: a bar for a contract not known by
        # as_of is a hindsight bar and is excluded, never promoted to an entry.
        by_contract = {bar.contract: bar for bar in bars}
        try:
            return OptionChainSnapshot(
                product=query.product,
                expiration_date=query.expiration_date,
                as_of=as_of,
                entries=tuple(
                    OptionChainEntry(contract, by_contract.get(contract)) for contract in contracts
                ),
            )
        except (InvalidOptionChainEntryError, InvalidOptionChainSnapshotError) as exc:
            raise OptionChainContractViolationError(
                f"{_SUBJECT} could not build the chain from its repositories: {exc}"
            ) from exc

    def _session(self, query: OptionChainSnapshotQuery) -> OptionTradingSession:
        """Resolve exactly the trading date; no session means a weekend or holiday."""
        sessions = self._session_resolver.sessions_in_range(
            query.product, query.trading_date, query.trading_date
        )
        if not isinstance(sessions, tuple) or not all(
            isinstance(session, OptionTradingSession) for session in sessions
        ):
            raise OptionChainContractViolationError(
                "OptionTradingSessionResolver must return a tuple of OptionTradingSession."
            )
        if not sessions:
            raise OptionChainSessionNotFoundError(query.product, query.trading_date)
        if len(sessions) != 1 or sessions[0].trading_date != query.trading_date:
            raise OptionChainContractViolationError(
                "OptionTradingSessionResolver must return exactly the requested trading "
                f"date's session for {query.trading_date.isoformat()}."
            )
        return sessions[0]


def _chain_order(contract: OptionContract) -> tuple:
    return (contract.strike.value, _RIGHT_RANK[contract.right])


def _belongs(contract: OptionContract, query: OptionChainSnapshotQuery) -> bool:
    return contract.product == query.product and contract.expiration_date == query.expiration_date


def _validate_contracts(
    contracts: object, query: OptionChainSnapshotQuery
) -> tuple[OptionContract, ...]:
    """Require the listing answer to be canonical, unique and for the requested expiration."""
    subject = "OptionListedContractRepository"
    if not isinstance(contracts, tuple) or not all(
        isinstance(contract, OptionContract) for contract in contracts
    ):
        raise OptionChainContractViolationError(
            f"{subject} must return a tuple of OptionContract values."
        )
    for contract in contracts:
        if not _belongs(contract, query):
            raise OptionChainContractViolationError(
                f"{subject} returned {contract} for {query.product} {query.expiration_date}."
            )
    if len(set(contracts)) != len(contracts):
        raise OptionChainContractViolationError(f"{subject} returned a contract more than once.")
    keys = [_chain_order(contract) for contract in contracts]
    if keys != sorted(keys):
        raise OptionChainContractViolationError(
            f"{subject} must order contracts by strike ascending, then CALL before PUT."
        )
    return contracts


def _validate_bars(
    bars: object, query: OptionChainSnapshotQuery, as_of: PointInTime
) -> tuple[OptionOHLCVBar, ...]:
    """Require exactly-stamped daily bars of the requested expiration, one per contract."""
    subject = "OptionChainDailyBarRepository"
    if not isinstance(bars, tuple) or not all(isinstance(bar, OptionOHLCVBar) for bar in bars):
        raise OptionChainContractViolationError(
            f"{subject} must return a tuple of OptionOHLCVBar values."
        )
    for bar in bars:
        if not _belongs(bar.contract, query):
            raise OptionChainContractViolationError(
                f"{subject} returned a bar for {bar.contract} for "
                f"{query.product} {query.expiration_date}."
            )
        if bar.point_in_time != as_of:
            raise OptionChainContractViolationError(
                f"{subject} returned a bar for {bar.contract} stamped {bar.point_in_time}, "
                f"not {as_of}."
            )
        if bar.timeframe != _DAILY:
            raise OptionChainContractViolationError(
                f"{subject} returned a {bar.timeframe} bar for {bar.contract}, not {_DAILY}."
            )
    contracts = [bar.contract for bar in bars]
    if len(set(contracts)) != len(contracts):
        raise OptionChainContractViolationError(
            f"{subject} returned more than one bar for a contract."
        )
    keys = [_chain_order(contract) for contract in contracts]
    if keys != sorted(keys):
        raise OptionChainContractViolationError(
            f"{subject} must order bars by strike ascending, then CALL before PUT."
        )
    return bars
