"""Application ports for the MODELED recommendation issuance authority (EI-5D, ADR-024).

The issuance authority records MODELED issued recommendations and session
completeness declarations under one monotone recording sequence, and answers
the authoritative issued-history views that EI-4C consumes (EI5-D1 to EI5-D4;
EI-5 design, section 8). It is one database authority; EI-6 implements it in
SQLite, in the same file as the EI-4D records. Application ships no
implementation, and EI-5D exercises these ports against a labelled in-memory
test authority only.

Read and authorized-write ports (ADR-024, section 4)
----------------------------------------------------
``EquityRecommendationIssuanceReader`` is read-only. Read-only consumers --
the authoritative history source, lifecycle evaluation and verification --
receive only the reader:

- ``load_issuance(issuance_id)`` and ``load_session_issuance(strategy,
  listing, trading_date)`` return the recorded issuance, or None;
- ``session_view(exchange, date)`` is the current view: None when the session
  has no declaration, otherwise an ``EquitySessionRecommendationHistory`` at
  the current watermark, whose ``reference`` is the authority reference;
- ``session_view_at(exchange, date, watermark)`` is the view at an earlier
  watermark, so frozen history evidence can be reproduced. A watermark beyond
  the authority's current position raises
  EquityIssuanceWatermarkBeyondAuthorityError.

``EquityRecommendationIssuanceWriter`` accepts only the content-bound
authorizations that the issuance and declaration use cases issue
(``VerifiedEquityRecommendationIssuance``,
``VerifiedEquityIssuanceCompletenessDeclaration``). An implementation
validates them with the public ``require_verified_*`` validators **before any
connection opens**, and persists exactly the content they cover. A missing,
foreign, unissued or changed authorization raises
UnverifiedEquityRecommendationIssuanceWriteError; nothing is written. No
caller supplies a recording sequence or watermark.

The writer enforces the invariants of the EI-5 design, section 8.2, in one
serialized write (``equity_recommendation_issuance_rules`` states them).

Failures
--------
- EquityRecommendationIssuanceUnavailableError: the authority could not be
  read or opened; nothing was written.
- EquityRecommendationIssuanceCommitUnconfirmedError: the commit outcome is
  unknown. It is never evidence of a rollback; the caller reconciles by an
  authoritative reload (the EI4-D9-G pattern).

Nothing here reads a clock or calls a provider, broker or network.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import TYPE_CHECKING

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import ExchangeCode
from northstar_core.strategy import StrategyIdentity

from northstar_application.ports.equity_recommendation_references import (
    EquitySessionRecommendationHistory,
)

if TYPE_CHECKING:
    from northstar_application.application_services.equity_recommendation_issuance_authorization import (  # noqa: E501
        VerifiedEquityIssuanceCompletenessDeclaration,
        VerifiedEquityRecommendationIssuance,
    )
    from northstar_application.application_services.equity_recommendation_issuance_records import (  # noqa: E501
        EquityRecordedCompletenessDeclaration,
        EquityRecordedIssuance,
    )


class EquityRecommendationIssuanceUnavailableError(RuntimeError):
    """Raised when the issuance authority cannot be read or opened; nothing was written."""


class EquityRecommendationIssuanceCommitUnconfirmedError(RuntimeError):
    """Raised when a write's commit outcome is unknown. It is never a rollback."""


class EquityIssuanceWatermarkBeyondAuthorityError(ValueError):
    """Raised when a view is asked for at a watermark beyond the authority's position."""


class UnverifiedEquityRecommendationIssuanceWriteError(ValueError):
    """Raised when an issuance-authority write is not authorized (ADR-024, section 4).

    The authorization is missing, of the wrong type, was not issued by the
    issuance or declaration use case, or covers other content than offered.
    Nothing is written. No supported caller can produce it.
    """


class InvalidEquityRecommendationIssuanceWriteError(ValueError):
    """Raised when an authority reports a write that its contract does not allow."""


class EquityRecommendationIssuanceOutcome(StrEnum):
    """What one issuance write established (EI-5 design, section 8.3)."""

    ISSUED = "ISSUED"
    EQUIVALENT = "EQUIVALENT"
    GENERATION_ALREADY_ISSUED = "GENERATION_ALREADY_ISSUED"
    SESSION_STRATEGY_ALREADY_ISSUED = "SESSION_STRATEGY_ALREADY_ISSUED"
    BACKDATED_ISSUANCE = "BACKDATED_ISSUANCE"
    STORED_ISSUANCE_INVALID = "STORED_ISSUANCE_INVALID"


class EquityIssuanceDeclarationOutcome(StrEnum):
    """What one completeness-declaration write established (EI-5 design, section 8.3)."""

    DECLARED = "DECLARED"
    DECLARATION_EQUIVALENT = "DECLARATION_EQUIVALENT"
    COMPLETENESS_REGRESSION = "COMPLETENESS_REGRESSION"


_STORED_REQUIRED = frozenset(
    {
        EquityRecommendationIssuanceOutcome.ISSUED,
        EquityRecommendationIssuanceOutcome.EQUIVALENT,
        EquityRecommendationIssuanceOutcome.GENERATION_ALREADY_ISSUED,
        EquityRecommendationIssuanceOutcome.SESSION_STRATEGY_ALREADY_ISSUED,
    }
)


@dataclass(frozen=True, slots=True)
class EquityRecommendationIssuanceWrite:
    """A committed issuance write: its outcome and the recorded issuance it concerns.

    ``stored`` is the issuance now recorded for ISSUED, the equal recorded one
    for EQUIVALENT, the conflicting recorded one for the two
    ``*_ALREADY_ISSUED`` outcomes, and None for BACKDATED_ISSUANCE and
    STORED_ISSUANCE_INVALID.
    """

    outcome: EquityRecommendationIssuanceOutcome
    stored: EquityRecordedIssuance | None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, EquityRecommendationIssuanceOutcome):
            raise InvalidEquityRecommendationIssuanceWriteError(
                "outcome must be an EquityRecommendationIssuanceOutcome."
            )
        if (self.outcome in _STORED_REQUIRED) != (self.stored is not None):
            raise InvalidEquityRecommendationIssuanceWriteError(
                f"A {self.outcome.value} write reports a recorded issuance exactly when it "
                "concerns one."
            )


@dataclass(frozen=True, slots=True)
class EquityIssuanceDeclarationWrite:
    """A committed declaration write.

    ``recorded`` is the declaration recorded now (DECLARED), the equal one
    already recorded (DECLARATION_EQUIVALENT), or the recorded declaration of
    the session's current maximum (COMPLETENESS_REGRESSION).
    """

    outcome: EquityIssuanceDeclarationOutcome
    recorded: EquityRecordedCompletenessDeclaration

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, EquityIssuanceDeclarationOutcome):
            raise InvalidEquityRecommendationIssuanceWriteError(
                "outcome must be an EquityIssuanceDeclarationOutcome."
            )
        if self.recorded is None:
            raise InvalidEquityRecommendationIssuanceWriteError(
                "A declaration write always reports the recorded declaration it concerns."
            )


class EquityRecommendationIssuanceReader(ABC):
    """Read-only access to recorded issuances and authoritative session views."""

    @abstractmethod
    def load_issuance(self, issuance_id: str) -> EquityRecordedIssuance | None:
        """Return the recorded issuance with ``issuance_id``, or None."""

    @abstractmethod
    def load_session_issuance(
        self, strategy: StrategyIdentity, listing: ListingReference, trading_date: date
    ) -> EquityRecordedIssuance | None:
        """Return the recorded issuance for the strategy, listing and date, or None."""

    @abstractmethod
    def session_view(
        self, exchange_code: ExchangeCode, trading_date: date
    ) -> EquitySessionRecommendationHistory | None:
        """Return the session's current view, or None when it has no declaration."""

    @abstractmethod
    def session_view_at(
        self, exchange_code: ExchangeCode, trading_date: date, watermark: int
    ) -> EquitySessionRecommendationHistory | None:
        """Return the session's view at ``watermark``, or None when undeclared there."""


class EquityRecommendationIssuanceWriter(ABC):
    """Authorized writes to the issuance authority: only use-case-issued authorizations."""

    @abstractmethod
    def write_issuance(
        self, authorization: VerifiedEquityRecommendationIssuance
    ) -> EquityRecommendationIssuanceWrite:
        """Record exactly the issued recommendation ``authorization`` covers."""

    @abstractmethod
    def write_declaration(
        self, authorization: VerifiedEquityIssuanceCompletenessDeclaration
    ) -> EquityIssuanceDeclarationWrite:
        """Record exactly the completeness declaration ``authorization`` covers."""
