"""Application derivation of deterministic futures paper execution identities.

The futures parallel of CreatePaperExecutionIdentitiesUseCase, with the same
length-prefixed SHA-256 scheme under its own namespaces:

    northstar.futures-paper-order.v1
    northstar.futures-paper-fill.v1

The order identity is keyed by the paper portfolio plus the frozen forward
record's natural key, in this exact order:

    portfolio identity
    product code
    exchange code
    expiration date
    timeframe
    decision instant
    strategy identity

It is derived from the record rather than from a FuturesExecutionIntent because
the intent carries no timeframe, and the audited identity includes it.

The fill identity is derived from the order identity alone. There is exactly one
full fill per order, so the order already names the fill; side, contracts, fill
quote and fill instant are payload facts, not logical identity. A retry whose
payload changed therefore arrives at the same identities and is rejected by
persistence as a conflict rather than stored as a duplicate.

The identities are opaque. Nothing may parse them.
"""

from __future__ import annotations

import hashlib

from northstar_core.paper_trading import (
    PaperFillIdentity,
    PaperOrderIdentity,
    PaperPortfolioIdentity,
)

from northstar_application.application_services.futures_forward_research_record import (
    FuturesForwardResearchRecord,
)

_ORDER_NAMESPACE = "northstar.futures-paper-order.v1"
_FILL_NAMESPACE = "northstar.futures-paper-fill.v1"


def _canonical(parts: tuple[str, ...]) -> bytes:
    """Return an injective byte encoding of an ordered string tuple."""
    encoded = bytearray()
    for part in parts:
        raw = part.encode("utf-8")
        encoded += str(len(raw)).encode("ascii")
        encoded += b":"
        encoded += raw
    return bytes(encoded)


def _digest(namespace: str, key: tuple[str, ...]) -> str:
    return hashlib.sha256(_canonical((namespace, *key))).hexdigest()


def _order_key(
    record: FuturesForwardResearchRecord, portfolio_identity: PaperPortfolioIdentity
) -> tuple[str, ...]:
    # Every component is already canonical in its own value object.
    contract = record.contract
    return (
        portfolio_identity.identity,
        contract.product.product_code.value,
        contract.product.exchange_code.value,
        contract.expiration_date.value,
        record.timeframe.value,
        record.decision_instant.value,
        record.strategy_identity.identity,
    )


class FuturesPaperExecutionIdentityService:
    """Derive futures paper order and fill identities, reproducibly.

    The service holds no dependencies and touches no clock, randomness or
    persistence: equivalent inputs produce equal identities in any process.
    """

    def order_identity(
        self,
        record: FuturesForwardResearchRecord,
        portfolio_identity: PaperPortfolioIdentity,
    ) -> PaperOrderIdentity:
        """Return the order identity for one portfolio executing one frozen decision."""
        if record is None:
            raise TypeError("FuturesPaperExecutionIdentityService record cannot be None.")
        if not isinstance(record, FuturesForwardResearchRecord):
            raise TypeError(
                "FuturesPaperExecutionIdentityService record must be a "
                "FuturesForwardResearchRecord."
            )
        if portfolio_identity is None:
            raise TypeError(
                "FuturesPaperExecutionIdentityService portfolio identity cannot be None."
            )
        if not isinstance(portfolio_identity, PaperPortfolioIdentity):
            raise TypeError(
                "FuturesPaperExecutionIdentityService portfolio identity must be a "
                "PaperPortfolioIdentity."
            )

        return PaperOrderIdentity(_digest(_ORDER_NAMESPACE, _order_key(record, portfolio_identity)))

    def fill_identity(self, order_identity: PaperOrderIdentity) -> PaperFillIdentity:
        """Return the identity of the one full fill an order can have."""
        if order_identity is None:
            raise TypeError("FuturesPaperExecutionIdentityService order identity cannot be None.")
        if not isinstance(order_identity, PaperOrderIdentity):
            raise TypeError(
                "FuturesPaperExecutionIdentityService order identity must be a PaperOrderIdentity."
            )

        return PaperFillIdentity(_digest(_FILL_NAMESPACE, (order_identity.identity,)))
