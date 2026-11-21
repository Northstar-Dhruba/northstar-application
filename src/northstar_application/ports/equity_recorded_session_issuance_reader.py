"""Bounded, read-only discovery of one session's recorded issuances (EI-7C, EI7-CF1).

``EquityRecordedSessionIssuanceReader`` is the additive EI-7 read port of the
EI-7 Decision Register, section 13.2 (EI7-CF1, refined by EI7-P1, EI7-P2,
EI7-P6, EI7-P7 and EI7-FC1). It is separate from
``EquityRecommendationIssuanceReader``, which is unchanged: no existing
implementation or test double must change, and the EI-6C monitor is never
given this port (EI6C-P7).

``recorded_session_issuances(exchange_code, trading_date, *, limit,
watermark=None)`` answers, from **one** read of the EI-6A issuance authority,
an ``EquityRecordedSessionIssuances``:

- ``recording_position`` N: the authority's observed ledger position at the
  read (the ledger is dense, 1 to N);
- ``recording_watermark`` W: ``watermark`` when given, otherwise N. A given
  watermark is never silently advanced to N;
- ``declarations``: the session's recorded completeness declarations with a
  recording sequence at or below W, in ascending recording sequence;
- ``issuances``: every recorded issuance of the session with a recording
  sequence at or below W, each re-derived and verified exactly as EI-6A
  verifies it, in ascending recording sequence -- never more than ``limit``.

An undeclared session is an ordinary result with no declaration at or below
W; it is never None. W and N are positions of the issuance ledger, never
lifecycle-event positions, evaluation times or recording times.

One exchange and one trading date: there is no list-all, no date range, no
strategy filter and no pagination.

Required order of checks (EI7-P6), inside the implementation's one read:

1. verify the whole recording-sequence ledger (LEDGER failure first);
2. establish N and the applicable W; a W above N raises
   EquityIssuanceWatermarkBeyondAuthorityError;
3. read and verify the session's declarations at or below W;
4. count the session's recorded issuances at or below W;
5. above ``limit``, raise EquityRecordedSessionDiscoveryLimitExceededError
   **before** re-deriving any issuance -- a complete result or a refusal,
   never a partial list or a silent truncation;
6. otherwise re-derive and verify every listed issuance.

So a LEDGER failure always takes precedence over overflow, and an overflow
may be reported before an individual corrupt issuance row is found -- an
accepted limitation (EI7-P6). An overflow is not a corruption verdict
(EI7-FC1). A successful result is never unverified.

A watermark above ``EQUITY_RECORDING_POSITION_MAXIMUM`` (2**63 - 1, the
SQLite INTEGER maximum) is ahead of any authority by definition: an
implementation refuses it with EquityIssuanceWatermarkBeyondAuthorityError
before binding it into a query (EI7-P7).

Failures (existing EI-6A types unless noted):

- EquityIssuanceWatermarkBeyondAuthorityError: W above N, or above the
  position maximum;
- EquityRecordedSessionDiscoveryLimitExceededError (EI-7C): more than
  ``limit`` recorded issuances at W;
- InvalidStoredEquityRecommendationIssuanceAuthorityError: corrupt ledger,
  declaration or issuance (LEDGER, DECLARATION, ISSUANCE);
- EquityRecommendationIssuanceUnavailableError: the authority cannot be read.

Nothing here writes, initializes a schema, reads a clock or calls out.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING

from northstar_core.foundation.value_objects import ExchangeCode, PointInTime

if TYPE_CHECKING:
    from northstar_application.application_services.equity_recommendation_issuance_records import (  # noqa: E501
        EquityRecordedCompletenessDeclaration,
        EquityRecordedIssuance,
    )

EQUITY_RECORDED_SESSION_DISCOVERY_LIMIT = 50
"""The approved maximum of recorded issuances per session (EI7-CF1, EI7-P2, EI7-FC1)."""

EQUITY_RECORDING_POSITION_MAXIMUM = 2**63 - 1
"""The greatest recording position an authority can hold (the SQLite INTEGER maximum)."""


class EquityRecordedSessionDiscoveryLimitExceededError(RuntimeError):
    """Raised when a session holds more recorded issuances at W than the limit allows.

    No partial list exists. It is not a corruption verdict and is never
    retryable: the same request at the same watermark refuses again.
    ``limit`` is the limit that was exceeded; no total count is claimed.
    """

    def __init__(self, limit: int, message: str | None = None) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer.")
        super().__init__(
            message or f"More than {limit} recorded issuances in the session; no list is returned."
        )
        self.limit = limit

    @property
    def retryable(self) -> bool:
        """Always False: the session's recorded issuances never shrink."""
        return False


class InvalidEquityRecordedSessionIssuancesError(ValueError):
    """Raised when a recorded-session result violates its contract.

    An implementation that answers another session, an unordered or
    out-of-watermark record, or an inconsistent position has reported state
    the authority cannot hold. Messages carry no stored content.
    """


def _fail(message: str) -> None:
    raise InvalidEquityRecordedSessionIssuancesError(message)


def _position(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{name} must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class EquityRecordedSessionIssuances:
    """One verified read of a session's recorded issuances and declarations at W.

    See the module docstring. Construction checks every structural invariant
    a correct authority guarantees; ``complete_through`` is the greatest
    declared instant at or below W, or None when the session is undeclared
    there.
    """

    exchange_code: ExchangeCode
    trading_date: date
    recording_position: int
    recording_watermark: int
    declarations: tuple[EquityRecordedCompletenessDeclaration, ...]
    issuances: tuple[EquityRecordedIssuance, ...]
    reference: str

    def __post_init__(self) -> None:
        from northstar_application.application_services.equity_recommendation_issuance_records import (  # noqa: E501
            EQUITY_RECOMMENDATION_ISSUANCE_AUTHORITY_REFERENCE,
            EquityRecordedCompletenessDeclaration,
            EquityRecordedIssuance,
        )

        subject = "EquityRecordedSessionIssuances"
        if not isinstance(self.exchange_code, ExchangeCode):
            _fail(f"{subject} exchange_code must be an ExchangeCode.")
        if isinstance(self.trading_date, datetime) or not isinstance(self.trading_date, date):
            _fail(f"{subject} trading_date must be a plain date.")
        _position(self.recording_position, f"{subject} recording_position")
        _position(self.recording_watermark, f"{subject} recording_watermark")
        if self.recording_watermark > self.recording_position:
            _fail(f"{subject} recording_watermark is beyond its recording_position.")
        if self.recording_position > EQUITY_RECORDING_POSITION_MAXIMUM:
            _fail(f"{subject} recording_position is beyond the position maximum.")
        if self.reference != EQUITY_RECOMMENDATION_ISSUANCE_AUTHORITY_REFERENCE:
            _fail(f"{subject} reference must be the issuance authority reference.")
        if not isinstance(self.declarations, tuple) or not all(
            isinstance(d, EquityRecordedCompletenessDeclaration) for d in self.declarations
        ):
            _fail(f"{subject} declarations must be a tuple of recorded declarations.")
        if not isinstance(self.issuances, tuple) or not all(
            isinstance(i, EquityRecordedIssuance) for i in self.issuances
        ):
            _fail(f"{subject} issuances must be a tuple of recorded issuances.")
        session = (self.exchange_code, self.trading_date)
        for recorded in self.declarations:
            found = recorded.declaration
            if (found.exchange_code, found.trading_date) != session:
                _fail(f"{subject} holds a declaration of another session.")
        for recorded in self.issuances:
            issued = recorded.recommendation
            if (issued.exchange_code, issued.trading_date) != session:
                _fail(f"{subject} holds an issuance of another session.")
        for name, records in (("declarations", self.declarations), ("issuances", self.issuances)):
            sequences = [r.recording_sequence for r in records]
            if sequences != sorted(sequences) or len(set(sequences)) != len(sequences):
                _fail(f"{subject} {name} must be in strictly ascending recording sequence.")
            if any(sequence > self.recording_watermark for sequence in sequences):
                _fail(f"{subject} lists {name} recorded after its recording watermark.")
        shared = [r.recording_sequence for r in self.declarations] + [
            r.recording_sequence for r in self.issuances
        ]
        if len(set(shared)) != len(shared):
            _fail(f"{subject} records share a recording sequence; the ledger is one sequence.")
        identities = [r.recommendation.issuance_id for r in self.issuances]
        if len(set(identities)) != len(identities):
            _fail(f"{subject} issuance identities must be distinct.")

    @property
    def complete_through(self) -> PointInTime | None:
        """Return the greatest declared ``complete_through`` at or below W, or None."""
        greatest: PointInTime | None = None
        for recorded in self.declarations:
            through = recorded.declaration.complete_through
            if greatest is None or through.compare(greatest) > 0:
                greatest = through
        return greatest

    @property
    def is_declared(self) -> bool:
        """Return whether the session has any declaration at or below W."""
        return bool(self.declarations)


class EquityRecordedSessionIssuanceReader(ABC):
    """Bounded, read-only discovery of one session's recorded issuances at one watermark."""

    @abstractmethod
    def recorded_session_issuances(
        self,
        exchange_code: ExchangeCode,
        trading_date: date,
        *,
        limit: int,
        watermark: int | None = None,
    ) -> EquityRecordedSessionIssuances:
        """Return the session's verified recorded issuances and declarations at W.

        Raises as the module docstring states. Never returns None and never a
        partial list.
        """
