"""Bounded session discovery: recorded and declared views (EI-7C, EI7-CF1).

Two Application-owned, immutable projections of **one** bounded read of the
EI-6A issuance authority through ``EquityRecordedSessionIssuanceReader``
(EI-7 Decision Register, section 13.2; design section 23.2):

- the **recorded view** lists every recorded issuance of the session at the
  recording watermark W, in ascending ``recording_sequence`` (EI7-P1), each
  marked ``covered_by_declaration`` when its ``issued_at`` is at or before the
  session's declared ``complete_through`` at W;
- the **declared view** lists exactly what the accepted EI-6A rule
  ``equity_issuance_session_view`` admits at W, in that rule's unchanged
  ``(issued_at, issuance_id)`` order, with the EI-6A history it produced.

Both state the observed recording position N and W, and the declaration state
at W, even for an undeclared session -- which is a result, never an error and
never "no recommendations". A session is only ever "declared complete through"
an instant, never complete.

The approved limit of 50 recorded issuances is passed to the reader. Above it
both projections refuse with DISCOVERY_LIMIT_EXCEEDED: never a partial list,
never a silent truncation, and never a corruption verdict (EI7-FC1). The bound
is a limit of this EI-7 projection layer only; the EI-6A authority and its
``session_view`` are unchanged and are not called here.

A requested watermark must be a non-negative integer. One above the recording
position -- or above the position maximum, which no authority can reach -- is
WATERMARK_AHEAD, refused here without calling the reader (EI7-P7). A reader
result for another session, or with a watermark other than the one requested,
is reported state the authority cannot hold: PERSISTED_AUTHORITY_INVALID.

Discovery is authoritative for issuance identities and session recording
facts only. It carries no lifecycle: the full lifecycle belongs to the
verified detail, and an optional per-item summary is the separate, independent
``GetEquityLifecycleSummaryUseCase`` read (read_equity_recommendation_detail),
never pinned to W and never atomic with this read.

The ledger-versus-overflow precedence of EI7-P6 happens inside the reader's
single read and is not observable here; this module only classifies what the
reader reports.

Nothing here writes, reads a clock or calls out.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from northstar_core.foundation.value_objects import ExchangeCode, PointInTime

from northstar_application.application_services.equity_read_query_errors import (
    EquityReadQueryRefusal,
    refuse_equity_read_query,
)
from northstar_application.application_services.equity_recommendation_issuance_records import (
    EquityRecordedIssuance,
)
from northstar_application.application_services.equity_recommendation_issuance_rules import (
    equity_declared_complete_through,
    equity_issuance_session_view,
)
from northstar_application.ports.equity_recommendation_issuance import (
    EquityIssuanceWatermarkBeyondAuthorityError,
    EquityRecommendationIssuanceUnavailableError,
    InvalidStoredEquityRecommendationIssuanceAuthorityError,
)
from northstar_application.ports.equity_recommendation_references import (
    EquitySessionRecommendationHistory,
)
from northstar_application.ports.equity_recorded_session_issuance_reader import (
    EQUITY_RECORDED_SESSION_DISCOVERY_LIMIT,
    EQUITY_RECORDING_POSITION_MAXIMUM,
    EquityRecordedSessionDiscoveryLimitExceededError,
    EquityRecordedSessionIssuanceReader,
    EquityRecordedSessionIssuances,
    InvalidEquityRecordedSessionIssuancesError,
)

_R = EquityReadQueryRefusal

EQUITY_READ_SUPPORTED_EXCHANGES = frozenset({ExchangeCode("NSE")})
"""The venues the V1 read queries answer for (EI-7 V1 boundaries: NSE only)."""


class InvalidEquitySessionViewError(ValueError):
    """Raised when a session view is constructed inconsistently."""


def _fail(message: str) -> None:
    raise InvalidEquitySessionViewError(message)


class EquitySessionDeclarationState(StrEnum):
    """Whether the session has a completeness declaration at the recording watermark."""

    NONE = "NONE"
    DECLARED = "DECLARED"


def _check_session_facts(
    subject: str,
    exchange_code: object,
    trading_date: object,
    recording_position: object,
    recording_watermark: object,
    declaration: object,
    complete_through: object,
) -> None:
    if not isinstance(exchange_code, ExchangeCode):
        _fail(f"{subject} exchange_code must be an ExchangeCode.")
    if isinstance(trading_date, datetime) or not isinstance(trading_date, date):
        _fail(f"{subject} trading_date must be a plain date.")
    for name, value in (
        ("recording_position", recording_position),
        ("recording_watermark", recording_watermark),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            _fail(f"{subject} {name} must be a non-negative integer.")
    assert isinstance(recording_position, int) and isinstance(recording_watermark, int)
    if recording_watermark > recording_position:
        _fail(f"{subject} recording_watermark is beyond its recording_position.")
    if not isinstance(declaration, EquitySessionDeclarationState):
        _fail(f"{subject} declaration must be an EquitySessionDeclarationState.")
    declared = declaration is EquitySessionDeclarationState.DECLARED
    if declared != isinstance(complete_through, PointInTime) or (
        not declared and complete_through is not None
    ):
        _fail(f"{subject} states complete_through exactly when the session is declared.")


@dataclass(frozen=True, slots=True)
class EquityRecordedSessionItem:
    """One recorded issuance in the recorded view, with its declaration coverage."""

    issuance: EquityRecordedIssuance
    covered_by_declaration: bool

    def __post_init__(self) -> None:
        if not isinstance(self.issuance, EquityRecordedIssuance):
            _fail("A recorded session item holds an EquityRecordedIssuance.")
        if not isinstance(self.covered_by_declaration, bool):
            _fail("covered_by_declaration must be a bool.")

    @property
    def issuance_id(self) -> str:
        """Return the authoritative issuance identity."""
        return self.issuance.recommendation.issuance_id

    @property
    def recording_sequence(self) -> int:
        """Return the authority's recording sequence of the issuance."""
        return self.issuance.recording_sequence


@dataclass(frozen=True, slots=True)
class EquityRecordedSessionIssuancesView:
    """The recorded view: every recorded issuance of the session at W, by sequence.

    ``items`` is empty when nothing is recorded at W -- a statement about
    recorded issuances, independent of ``declaration``.
    """

    exchange_code: ExchangeCode
    trading_date: date
    recording_position: int
    recording_watermark: int
    declaration: EquitySessionDeclarationState
    complete_through: PointInTime | None
    items: tuple[EquityRecordedSessionItem, ...]
    reference: str

    def __post_init__(self) -> None:
        subject = "EquityRecordedSessionIssuancesView"
        _check_session_facts(
            subject,
            self.exchange_code,
            self.trading_date,
            self.recording_position,
            self.recording_watermark,
            self.declaration,
            self.complete_through,
        )
        if not isinstance(self.items, tuple) or not all(
            isinstance(i, EquityRecordedSessionItem) for i in self.items
        ):
            _fail(f"{subject} items must be a tuple of EquityRecordedSessionItem.")
        if len(self.items) > EQUITY_RECORDED_SESSION_DISCOVERY_LIMIT:
            _fail(f"{subject} never lists more than the discovery limit.")
        sequences = [i.recording_sequence for i in self.items]
        if sequences != sorted(sequences) or len(set(sequences)) != len(sequences):
            _fail(f"{subject} items must be in strictly ascending recording sequence.")
        for item in self.items:
            issued = item.issuance.recommendation
            if (issued.exchange_code, issued.trading_date) != (
                self.exchange_code,
                self.trading_date,
            ):
                _fail(f"{subject} holds an issuance of another session.")
            if item.recording_sequence > self.recording_watermark:
                _fail(f"{subject} lists an issuance recorded after its watermark.")
            covered = self.complete_through is not None and (
                issued.issued_at.compare(self.complete_through) <= 0
            )
            if item.covered_by_declaration != covered:
                _fail(f"{subject} item coverage disagrees with complete_through.")
        if not isinstance(self.reference, str) or not self.reference:
            _fail(f"{subject} reference must be non-empty text.")


@dataclass(frozen=True, slots=True)
class EquityDeclaredSessionRecommendationsView:
    """The declared view: what the EI-6A session-view rule admits at W, in its order.

    ``history`` is the EI-6A ``EquitySessionRecommendationHistory`` at W, or
    None when the session is undeclared there. ``items`` are the verified
    recorded issuances it lists, in the history's ``(issued_at, issuance_id)``
    order; empty when undeclared or when the declaration covers none.
    """

    exchange_code: ExchangeCode
    trading_date: date
    recording_position: int
    recording_watermark: int
    declaration: EquitySessionDeclarationState
    complete_through: PointInTime | None
    history: EquitySessionRecommendationHistory | None
    items: tuple[EquityRecordedIssuance, ...]
    reference: str

    def __post_init__(self) -> None:
        subject = "EquityDeclaredSessionRecommendationsView"
        _check_session_facts(
            subject,
            self.exchange_code,
            self.trading_date,
            self.recording_position,
            self.recording_watermark,
            self.declaration,
            self.complete_through,
        )
        if not isinstance(self.items, tuple) or not all(
            isinstance(i, EquityRecordedIssuance) for i in self.items
        ):
            _fail(f"{subject} items must be a tuple of EquityRecordedIssuance.")
        if self.history is None:
            if self.declaration is not EquitySessionDeclarationState.NONE or self.items:
                _fail(f"{subject} without a history is undeclared and lists nothing.")
        else:
            history = self.history
            if not isinstance(history, EquitySessionRecommendationHistory):
                _fail(f"{subject} history must be an EquitySessionRecommendationHistory.")
            if (
                history.exchange_code,
                history.trading_date,
                history.recording_watermark,
                history.complete_through,
                history.reference,
            ) != (
                self.exchange_code,
                self.trading_date,
                self.recording_watermark,
                self.complete_through,
                self.reference,
            ):
                _fail(f"{subject} history is not this view's session, watermark or declaration.")
            if [r.recommendation_key for r in history.recommendations] != [
                i.recommendation.issuance_id for i in self.items
            ] or [r.recording_sequence for r in history.recommendations] != [
                i.recording_sequence for i in self.items
            ]:
                _fail(f"{subject} items are not exactly the history's records, in its order.")
        if not isinstance(self.reference, str) or not self.reference:
            _fail(f"{subject} reference must be non-empty text.")


def _require_session(exchange_code: object, trading_date: object) -> None:
    if not isinstance(exchange_code, ExchangeCode) or (
        exchange_code not in EQUITY_READ_SUPPORTED_EXCHANGES
    ):
        raise refuse_equity_read_query(_R.UNSUPPORTED_SCOPE, "the exchange is not supported")
    if isinstance(trading_date, datetime) or not isinstance(trading_date, date):
        raise refuse_equity_read_query(_R.UNSUPPORTED_SCOPE, "the trading date is not a date")


def _require_watermark(watermark: object) -> None:
    if watermark is None:
        return
    if isinstance(watermark, bool) or not isinstance(watermark, int) or watermark < 0:
        raise refuse_equity_read_query(
            _R.UNSUPPORTED_SCOPE, "the watermark must be a non-negative integer"
        )
    if watermark > EQUITY_RECORDING_POSITION_MAXIMUM:
        raise refuse_equity_read_query(
            _R.WATERMARK_AHEAD, "the watermark is beyond any recording position"
        )


def _bounded_read(
    reader: EquityRecordedSessionIssuanceReader,
    exchange_code: ExchangeCode,
    trading_date: date,
    watermark: int | None,
) -> EquityRecordedSessionIssuances:
    _require_session(exchange_code, trading_date)
    _require_watermark(watermark)
    try:
        result = reader.recorded_session_issuances(
            exchange_code,
            trading_date,
            limit=EQUITY_RECORDED_SESSION_DISCOVERY_LIMIT,
            watermark=watermark,
        )
    except EquityIssuanceWatermarkBeyondAuthorityError as exc:
        raise refuse_equity_read_query(
            _R.WATERMARK_AHEAD, "the watermark is beyond the recording position"
        ) from exc
    except EquityRecordedSessionDiscoveryLimitExceededError as exc:
        raise refuse_equity_read_query(
            _R.DISCOVERY_LIMIT_EXCEEDED,
            f"more than {EQUITY_RECORDED_SESSION_DISCOVERY_LIMIT} recorded issuances; "
            "no partial list is returned",
        ) from exc
    except (
        InvalidStoredEquityRecommendationIssuanceAuthorityError,
        InvalidEquityRecordedSessionIssuancesError,
    ) as exc:
        raise refuse_equity_read_query(
            _R.PERSISTED_AUTHORITY_INVALID, "the issuance authority is invalid"
        ) from exc
    except EquityRecommendationIssuanceUnavailableError as exc:
        raise refuse_equity_read_query(
            _R.DEPENDENCY_UNAVAILABLE, "the issuance authority is unavailable"
        ) from exc
    if not isinstance(result, EquityRecordedSessionIssuances):
        raise refuse_equity_read_query(
            _R.PERSISTED_AUTHORITY_INVALID, "the issuance authority answered no session result"
        )
    expected_watermark = result.recording_position if watermark is None else watermark
    if (
        result.exchange_code,
        result.trading_date,
        result.recording_watermark,
    ) != (exchange_code, trading_date, expected_watermark) or len(
        result.issuances
    ) > EQUITY_RECORDED_SESSION_DISCOVERY_LIMIT:
        raise refuse_equity_read_query(
            _R.PERSISTED_AUTHORITY_INVALID,
            "the issuance authority answered another session, watermark or bound",
        )
    return result


def _declaration(
    result: EquityRecordedSessionIssuances,
) -> tuple[EquitySessionDeclarationState, PointInTime | None]:
    complete_through = equity_declared_complete_through(
        result.declarations, result.exchange_code, result.trading_date, result.recording_watermark
    )
    if complete_through is None:
        return EquitySessionDeclarationState.NONE, None
    return EquitySessionDeclarationState.DECLARED, complete_through


def equity_recorded_session_view(
    result: EquityRecordedSessionIssuances,
) -> EquityRecordedSessionIssuancesView:
    """Project one bounded read into the recorded view (ascending recording sequence)."""
    if not isinstance(result, EquityRecordedSessionIssuances):
        raise TypeError("result must be an EquityRecordedSessionIssuances.")
    declaration, complete_through = _declaration(result)
    items = tuple(
        EquityRecordedSessionItem(
            recorded,
            complete_through is not None
            and recorded.recommendation.issued_at.compare(complete_through) <= 0,
        )
        for recorded in sorted(result.issuances, key=lambda r: r.recording_sequence)
    )
    return EquityRecordedSessionIssuancesView(
        exchange_code=result.exchange_code,
        trading_date=result.trading_date,
        recording_position=result.recording_position,
        recording_watermark=result.recording_watermark,
        declaration=declaration,
        complete_through=complete_through,
        items=items,
        reference=result.reference,
    )


def equity_declared_session_view(
    result: EquityRecordedSessionIssuances,
) -> EquityDeclaredSessionRecommendationsView:
    """Project one bounded read into the declared view through the EI-6A rule, unchanged."""
    if not isinstance(result, EquityRecordedSessionIssuances):
        raise TypeError("result must be an EquityRecordedSessionIssuances.")
    declaration, complete_through = _declaration(result)
    history = equity_issuance_session_view(
        result.exchange_code,
        result.trading_date,
        result.recording_watermark,
        [recorded.history_record() for recorded in result.issuances],
        result.declarations,
        result.reference,
    )
    by_identity = {r.recommendation.issuance_id: r for r in result.issuances}
    items = (
        ()
        if history is None
        else tuple(by_identity[record.recommendation_key] for record in history.recommendations)
    )
    return EquityDeclaredSessionRecommendationsView(
        exchange_code=result.exchange_code,
        trading_date=result.trading_date,
        recording_position=result.recording_position,
        recording_watermark=result.recording_watermark,
        declaration=declaration,
        complete_through=complete_through,
        history=history,
        items=items,
        reference=result.reference,
    )


class _SessionQuery:
    def __init__(self, reader: EquityRecordedSessionIssuanceReader) -> None:
        if not isinstance(reader, EquityRecordedSessionIssuanceReader):
            raise TypeError("reader must be an EquityRecordedSessionIssuanceReader.")
        self._reader = reader


class ListEquityRecordedSessionIssuancesUseCase(_SessionQuery):
    """Return the recorded view of one NSE session at W (EI7-CF1)."""

    def execute(
        self, exchange_code: ExchangeCode, trading_date: date, watermark: int | None = None
    ) -> EquityRecordedSessionIssuancesView:
        """Refuse with EquityReadQueryRefusedError as the module docstring states."""
        result = _bounded_read(self._reader, exchange_code, trading_date, watermark)
        return equity_recorded_session_view(result)


class ListEquityDeclaredSessionRecommendationsUseCase(_SessionQuery):
    """Return the declared view of one NSE session at W (EI7-CF1, EI-6A rule 5)."""

    def execute(
        self, exchange_code: ExchangeCode, trading_date: date, watermark: int | None = None
    ) -> EquityDeclaredSessionRecommendationsView:
        """Refuse with EquityReadQueryRefusedError as the module docstring states."""
        result = _bounded_read(self._reader, exchange_code, trading_date, watermark)
        return equity_declared_session_view(result)
