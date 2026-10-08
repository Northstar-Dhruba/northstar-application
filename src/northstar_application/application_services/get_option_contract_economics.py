"""Application fail-closed query for the economics of one option contract.

Option economics are keyed by the complete OptionContract -- product, exchange,
expiration, strike and right -- because every one of those can carry its own
point value. This query returns exactly the requested contract's economics or
fails; it never returns None, a default point value or another contract's
economics.

The repository is asked once, for the requested contract only. Nothing here
retries with a neighbouring strike, the other right, another expiry or another
product, because there is no fallback to make: a contract without its own
economics is not configured, and saying so is the point.

A repository answer that is not OptionContractEconomics, or that names any
other contract, is a broken adapter rather than missing configuration, and is
reported as a contract violation instead of being trusted. Nothing is persisted
and no clock is read.
"""

from __future__ import annotations

from northstar_core.options import OptionContract, OptionContractEconomics

from northstar_application.ports import OptionContractEconomicsRepository

_SUBJECT = "GetOptionContractEconomicsUseCase"


class OptionContractEconomicsNotFoundError(ValueError):
    """Raised when an option contract has no configured economics.

    ``contract`` names the complete option contract -- product, exchange,
    expiration, strike and right -- so callers never parse the message, and one
    strike or right without economics is never mistaken for another.
    """

    def __init__(self, message: str, contract: OptionContract | None = None) -> None:
        super().__init__(message)
        self.contract = contract


class OptionContractEconomicsContractViolationError(ValueError):
    """Raised when an OptionContractEconomicsRepository violates its contract."""


class GetOptionContractEconomicsUseCase:
    """Return the economics of exactly one option contract, or fail closed."""

    def __init__(self, repository: OptionContractEconomicsRepository) -> None:
        if not isinstance(repository, OptionContractEconomicsRepository):
            raise TypeError(f"{_SUBJECT} repository must be an OptionContractEconomicsRepository.")
        self._repository = repository

    def execute(self, contract: OptionContract) -> OptionContractEconomics:
        """Return the requested contract's own economics.

        Raises OptionContractEconomicsNotFoundError when none are configured,
        and OptionContractEconomicsContractViolationError when the repository
        answers with anything other than this contract's economics.
        """
        if not isinstance(contract, OptionContract):
            raise TypeError(f"{_SUBJECT} contract must be an OptionContract.")

        found = self._repository.get_economics(contract)
        if found is None:
            raise OptionContractEconomicsNotFoundError(
                f"Option contract economics not configured for {contract}.", contract
            )
        if not isinstance(found, OptionContractEconomics):
            raise OptionContractEconomicsContractViolationError(
                "OptionContractEconomicsRepository must return OptionContractEconomics or None."
            )
        if found.contract != contract:
            raise OptionContractEconomicsContractViolationError(
                f"OptionContractEconomicsRepository returned economics for {found.contract} "
                f"when asked for {contract}."
            )
        return found
