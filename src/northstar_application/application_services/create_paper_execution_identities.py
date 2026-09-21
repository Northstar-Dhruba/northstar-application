"""Application derivation of deterministic paper execution identities.

Paper trading needs an order identity and a fill identity before it can execute
anything, and the obvious sources -- a UUID, a clock, a database sequence --
would each make an execution irreproducible. Instead both identities are
derived from the decision that caused them, so the same decision always yields
the same identities and a retried pipeline converges on the same records
rather than accumulating duplicates.

The identities are opaque. Nothing may parse them, and their only guarantees
are determinism, distinctness between the order and fill namespaces, and
stability for one decision boundary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from northstar_core.paper_trading import (
    ExecutionIntent,
    PaperFillIdentity,
    PaperOrderIdentity,
)

_ORDER_NAMESPACE = "northstar.paper-order.v1"
_FILL_NAMESPACE = "northstar.paper-fill.v1"


def _canonical(parts: tuple[str, ...]) -> bytes:
    """Return an injective byte encoding of an ordered string tuple.

    Each part is length-prefixed, so no choice of separator can be forged by a
    value that happens to contain it. Identity strings have deliberately
    undefined formats and may contain any character, which makes plain
    delimiter joining unsafe: ``("a|b", "c")`` and ``("a", "b|c")`` would
    otherwise produce one encoding and therefore one identity.
    """
    encoded = bytearray()
    for part in parts:
        raw = part.encode("utf-8")
        encoded += str(len(raw)).encode("ascii")
        encoded += b":"
        encoded += raw
    return bytes(encoded)


def _digest(namespace: str, key: tuple[str, ...]) -> str:
    """Return the SHA-256 digest of one namespaced execution key.

    The namespace is encoded as the first field of the same injective
    encoding, so an order identity and a fill identity for one decision can
    never collide, and a future scheme version can never collide with this one.
    """
    return hashlib.sha256(_canonical((namespace, *key))).hexdigest()


def _execution_key(intent: ExecutionIntent) -> tuple[str, str, str, str, str]:
    """Return the canonical execution key for one decision boundary.

    The key answers "which decision is being executed", not "what was
    executed". Side, quantity, price and any fill instant are deliberately
    excluded: one portfolio executes one strategy's decision on one listing at
    one instant exactly once, so a retry that changed the side or the size must
    arrive at the same identities and be rejected by persistence as a
    conflicting execution rather than quietly stored as a second one.

    ``decided_at.value`` is already canonical UTC, and every identity value is
    already normalized by its own value object, so no further normalization is
    applied or needed here.
    """
    return (
        intent.portfolio_identity.identity,
        intent.listing_reference.symbol.value,
        intent.listing_reference.exchange_code.value,
        intent.strategy_identity.identity,
        intent.decided_at.value,
    )


@dataclass(frozen=True, slots=True)
class PaperExecutionIdentities:
    """The order and fill identities derived for one decision boundary."""

    order_identity: PaperOrderIdentity
    fill_identity: PaperFillIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.order_identity, PaperOrderIdentity):
            raise TypeError("PaperExecutionIdentities order identity must be a PaperOrderIdentity.")
        if not isinstance(self.fill_identity, PaperFillIdentity):
            raise TypeError("PaperExecutionIdentities fill identity must be a PaperFillIdentity.")


class CreatePaperExecutionIdentitiesUseCase:
    """Derive one decision's execution identities, reproducibly.

    The use case holds no dependencies and touches no clock, randomness,
    persistence or object memory identity: equivalent intents produce equal
    identities in any process, on any machine, at any time.
    """

    def execute(self, intent: ExecutionIntent) -> PaperExecutionIdentities:
        """Return the identities that one execution intent must execute under."""
        if intent is None:
            raise TypeError("CreatePaperExecutionIdentitiesUseCase intent cannot be None.")
        if not isinstance(intent, ExecutionIntent):
            raise TypeError(
                "CreatePaperExecutionIdentitiesUseCase intent must be an ExecutionIntent."
            )

        key = _execution_key(intent)
        return PaperExecutionIdentities(
            order_identity=PaperOrderIdentity(_digest(_ORDER_NAMESPACE, key)),
            fill_identity=PaperFillIdentity(_digest(_FILL_NAMESPACE, key)),
        )
