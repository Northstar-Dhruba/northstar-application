"""Application port for resolving option expiration dates from exchange rules.

An option expiration date is derived from the exchange's expiration rule and its
trading calendar: the rule names a nominal date for a period -- the Tuesday of
an expiry week, the last Tuesday of an expiry month -- and a holiday adjustment
may move the actual expiration to an earlier trading day. Both dates are kept,
so an adjusted expiry is visible rather than silently replacing the rule's
answer.

Resolving an expiration says which date the rule gives. It does not establish
that any contract was listed: strike availability, listing existence, provider
instruments, lot sizes and option-chain membership are all outside this port.
Nothing here generates an OptionContract.

Weekly and monthly are two explicit operations rather than one operation keyed
by an expiry-series type. Quarterly and half-yearly contracts expire on the
monthly rule's date and differ only in which months are listed, which is a
listing-cycle fact outside this port.

This is calendar-derived Application data: it belongs to no Core bounded
context and changes no option contract identity.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime

from northstar_core.derivatives import ExpirationDate
from northstar_core.options import OptionProductReference


class OptionExpirationResolutionError(RuntimeError):
    """Raised when an option expiration cannot be resolved reliably."""


class InvalidResolvedOptionExpirationError(ValueError):
    """Raised when a ResolvedOptionExpiration value is invalid."""


def _validate_product(value: OptionProductReference) -> OptionProductReference:
    if value is None:
        raise InvalidResolvedOptionExpirationError(
            "ResolvedOptionExpiration product cannot be None."
        )
    if not isinstance(value, OptionProductReference):
        raise InvalidResolvedOptionExpirationError(
            "ResolvedOptionExpiration product must be an OptionProductReference value."
        )
    return value


def _validate_nominal_date(value: date) -> date:
    if value is None:
        raise InvalidResolvedOptionExpirationError(
            "ResolvedOptionExpiration nominal date cannot be None."
        )
    if isinstance(value, datetime) or not isinstance(value, date):
        raise InvalidResolvedOptionExpirationError(
            "ResolvedOptionExpiration nominal date must be a plain date."
        )
    return value


def _validate_expiration_date(value: ExpirationDate) -> ExpirationDate:
    if value is None:
        raise InvalidResolvedOptionExpirationError(
            "ResolvedOptionExpiration expiration date cannot be None."
        )
    if not isinstance(value, ExpirationDate):
        raise InvalidResolvedOptionExpirationError(
            "ResolvedOptionExpiration expiration date must be an ExpirationDate value."
        )
    return value


def _validate_rule_source(value: str) -> str:
    if value is None:
        raise InvalidResolvedOptionExpirationError(
            "ResolvedOptionExpiration rule source cannot be None."
        )
    if not isinstance(value, str):
        raise InvalidResolvedOptionExpirationError(
            "ResolvedOptionExpiration rule source must be a string."
        )
    if not value.strip():
        raise InvalidResolvedOptionExpirationError(
            "ResolvedOptionExpiration rule source cannot be empty."
        )
    return value


@dataclass(frozen=True, slots=True)
class ResolvedOptionExpiration:
    """One option expiration resolved from an exchange expiration rule.

    ``nominal_date`` is the date the rule names for the requested period before
    any adjustment. ``expiration_date`` is the actual exchange expiration after
    the holiday adjustment, which only ever moves backward, so it is never
    after the nominal date. ``rule_source`` cites the rule epoch that applied.
    """

    product: OptionProductReference
    nominal_date: date
    expiration_date: ExpirationDate
    rule_source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "product", _validate_product(self.product))
        object.__setattr__(self, "nominal_date", _validate_nominal_date(self.nominal_date))
        object.__setattr__(self, "expiration_date", _validate_expiration_date(self.expiration_date))
        object.__setattr__(self, "rule_source", _validate_rule_source(self.rule_source))

        if date.fromisoformat(self.expiration_date.value) > self.nominal_date:
            raise InvalidResolvedOptionExpirationError(
                "ResolvedOptionExpiration expiration date cannot be after its nominal date."
            )

    @property
    def is_adjusted(self) -> bool:
        """Return whether the expiration was moved off its nominal date."""
        return self.expiration_date.value != self.nominal_date.isoformat()

    def __str__(self) -> str:
        return f"{self.product} {self.expiration_date} (nominal {self.nominal_date.isoformat()})"

    def __repr__(self) -> str:
        return (
            "ResolvedOptionExpiration("
            f"product={self.product!r}, "
            f"nominal_date={self.nominal_date!r}, "
            f"expiration_date={self.expiration_date!r}, "
            f"rule_source={self.rule_source!r}"
            ")"
        )


class OptionExpirationResolver(ABC):
    """Resolves the expiration dates an exchange rule gives an option product.

    Implementations must honour these semantics for both operations:

    - The answer is the requested period's nominal date under the rule epoch in
      force for that period, and the expiration date the rule's adjustment
      gives it.
    - Invalid coordinates -- an ISO week the ISO year does not have, a month
      outside 1..12, a non-integer -- raise OptionExpirationResolutionError.
    - An unsupported product or venue, a period with no rule epoch in force, a
      period spanning two rule epochs, and calendar data that cannot answer
      reliably all raise OptionExpirationResolutionError. Nothing is guessed,
      and no rule is assumed to apply before its effective date.
    - The answer never depends on a wall clock, a provider or the network.

    Resolving an expiration never establishes that a contract was listed.

    The abstract contract documents these obligations but cannot enforce them
    at runtime.
    """

    @abstractmethod
    def weekly_expiration(
        self, product: OptionProductReference, iso_year: int, iso_week: int
    ) -> ResolvedOptionExpiration:
        """Return the weekly expiration of one ISO week (``iso_year``, ``iso_week``)."""

    @abstractmethod
    def monthly_expiration(
        self, product: OptionProductReference, year: int, month: int
    ) -> ResolvedOptionExpiration:
        """Return the monthly expiration of one calendar month (``year``, ``month``)."""
