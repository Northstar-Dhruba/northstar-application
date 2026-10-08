"""Application port for the option contracts Northstar knew were listed by an instant.

A listed contract is listing knowledge, not market observation: the port answers
which exact contracts of one expiration Northstar had observed listed on or
before ``known_by``. It exposes contracts only. Which provider reported them,
under which instrument key and lot, and exactly when each was first observed
are adapter concerns.

Northstar records when it first observed each listing, not continuous listing
membership. A contract observed listed on or before ``known_by`` is treated as
listed through its expiration unless Northstar holds explicit contrary
evidence. That is a Northstar reconstruction assumption, not an exchange
guarantee; a contract first observed after ``known_by`` is never returned.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from northstar_core.derivatives import ExpirationDate
from northstar_core.foundation.value_objects import PointInTime
from northstar_core.options import OptionContract, OptionProductReference


class InvalidOptionListedContractQueryError(ValueError):
    """Raised when an option listed-contract query is invalid."""


@dataclass(frozen=True, slots=True)
class OptionListedContractQuery:
    """Request for one expiration's contracts Northstar knew were listed by an instant."""

    product: OptionProductReference
    expiration_date: ExpirationDate
    known_by: PointInTime

    def __post_init__(self) -> None:
        subject = "OptionListedContractQuery"
        if not isinstance(self.product, OptionProductReference):
            raise InvalidOptionListedContractQueryError(
                f"{subject} product must be an OptionProductReference value."
            )
        if not isinstance(self.expiration_date, ExpirationDate):
            raise InvalidOptionListedContractQueryError(
                f"{subject} expiration date must be an ExpirationDate value."
            )
        if not isinstance(self.known_by, PointInTime):
            raise InvalidOptionListedContractQueryError(
                f"{subject} known-by instant must be a PointInTime value."
            )


class OptionListedContractRepository(ABC):
    """Reads the exact option contracts Northstar knew were listed by an instant.

    Implementations must return an immutable tuple of exactly those contracts
    whose product and expiration equal the query's and whose listing Northstar
    first observed at or before ``known_by``, compared as instants with
    PointInTime.compare(), never as text. The tuple holds each contract once,
    ordered by strike ascending, then CALL before PUT. A query matching nothing
    returns an empty tuple, including when nothing has ever been stored. The
    port reads only what is stored: it creates nothing and reaches no provider,
    instrument master, network or clock.
    """

    @abstractmethod
    def listed_contracts(self, query: OptionListedContractQuery) -> tuple[OptionContract, ...]:
        """Return the expiration's contracts known by the instant, in canonical order."""
