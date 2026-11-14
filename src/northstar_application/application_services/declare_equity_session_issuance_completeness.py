"""Session completeness declarations: the only operational declarer (EI-5D, ADR-024 7).

``DeclareEquitySessionIssuanceCompletenessUseCase.execute(exchange_code,
trading_date, complete_through)`` records the append-only fact "exchange
``X``, trading date ``D`` is complete through instant ``T``" in the issuance
authority (EI5-D4):

1. the venue day is read from the existing venue-day reference; the
   declaration must be for that venue day's own exchange and date, on a
   NORMAL_SESSION, with ``T`` inside the venue's local calendar day;
2. a content-bound declaration authorization is issued;
3. the writer records it under the authority's next recording sequence, or
   reports DECLARATION_EQUIVALENT (equal to the current maximum) or
   COMPLETENESS_REGRESSION (below it); a regression is refused and changes
   nothing.

The caller supplies no recording sequence or watermark: the authority assigns
the sequence (EI5-D4). A declaration is trusted because the declaring workflow
is trusted within ADR-024's single-host boundary; it is not a cryptographic
proof (ADR-024, section 14).

An unconfirmed commit is reconciled by reading the session's current view:
complete exactly through ``T`` settles as DECLARATION_EQUIVALENT; nothing
declared, or less, is retried within the bound; anything greater cannot show
whether this declaration was recorded, and
EquityIssuanceDeclarationUnresolvedError says nothing is claimed.

Nothing here reads a clock or calls a provider, broker or network.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from northstar_core.foundation.value_objects import ExchangeCode, PointInTime

from northstar_application.application_services.equity_recommendation_issuance_authorization import (  # noqa: E501
    InvalidEquityIssuanceCompletenessDeclarationError,
    _issue_declaration_write,
    equity_declaration_venue_day_problem,
)
from northstar_application.application_services.equity_recommendation_issuance_records import (
    EquityIssuanceCompletenessDeclaration,
    EquityRecordedCompletenessDeclaration,
)
from northstar_application.ports.equity_recommendation_issuance import (
    EquityIssuanceDeclarationOutcome,
    EquityIssuanceDeclarationWrite,
    EquityRecommendationIssuanceCommitUnconfirmedError,
    EquityRecommendationIssuanceReader,
    EquityRecommendationIssuanceUnavailableError,
    EquityRecommendationIssuanceWriter,
    InvalidEquityRecommendationIssuanceWriteError,
)
from northstar_application.ports.equity_research_window_reference import (
    EquityVenueDayReferenceSource,
)

EQUITY_ISSUANCE_DECLARATION_WRITES = 3

_D = EquityIssuanceDeclarationOutcome


class EquityIssuanceDeclarationUnresolvedError(RuntimeError):
    """Raised when an unconfirmed declaration commit could not be settled; nothing is claimed."""


@dataclass(frozen=True, slots=True)
class EquityIssuanceDeclarationResult:
    """The outcome of one declaration request.

    ``recorded`` is the declaration recorded now (DECLARED), the equal recorded
    one (DECLARATION_EQUIVALENT), or the recorded current maximum that refused
    a regression (COMPLETENESS_REGRESSION).
    """

    outcome: EquityIssuanceDeclarationOutcome
    declaration: EquityIssuanceCompletenessDeclaration
    recorded: EquityRecordedCompletenessDeclaration | None
    writes: int
    reconciled: bool
    detail: str

    @property
    def is_declared(self) -> bool:
        """Return whether the session is now declared complete through this instant."""
        return self.outcome is not _D.COMPLETENESS_REGRESSION


class DeclareEquitySessionIssuanceCompletenessUseCase:
    """Declares a session complete through an instant in the issuance authority.

    ``venue`` is the existing venue-day reference source; ``reader`` and
    ``writer`` are the authority's ports. See the module docstring.
    """

    def __init__(
        self,
        venue: EquityVenueDayReferenceSource,
        reader: EquityRecommendationIssuanceReader,
        writer: EquityRecommendationIssuanceWriter,
        max_writes: int = EQUITY_ISSUANCE_DECLARATION_WRITES,
    ) -> None:
        if not isinstance(venue, EquityVenueDayReferenceSource):
            raise TypeError("venue must be an EquityVenueDayReferenceSource.")
        if not isinstance(reader, EquityRecommendationIssuanceReader):
            raise TypeError("reader must be an EquityRecommendationIssuanceReader.")
        if not isinstance(writer, EquityRecommendationIssuanceWriter):
            raise TypeError("writer must be an EquityRecommendationIssuanceWriter.")
        if isinstance(max_writes, bool) or not isinstance(max_writes, int) or max_writes < 1:
            raise ValueError("max_writes must be an integer of at least 1.")
        self._venue = venue
        self._reader = reader
        self._writer = writer
        self._max_writes = max_writes

    def execute(
        self, exchange_code: ExchangeCode, trading_date: date, complete_through: PointInTime
    ) -> EquityIssuanceDeclarationResult:
        """Declare the session complete through ``complete_through``."""
        if not isinstance(exchange_code, ExchangeCode):
            raise TypeError("exchange_code must be an ExchangeCode.")
        if isinstance(trading_date, datetime) or not isinstance(trading_date, date):
            raise TypeError("trading_date must be a plain date.")
        if not isinstance(complete_through, PointInTime):
            raise TypeError("complete_through must be a PointInTime.")
        declaration = EquityIssuanceCompletenessDeclaration(
            exchange_code, trading_date, complete_through
        )
        venue_day = self._venue.venue_day(exchange_code, trading_date)
        problem = equity_declaration_venue_day_problem(declaration, venue_day)
        if problem is not None:
            raise InvalidEquityIssuanceCompletenessDeclarationError(
                f"The declaration is refused: {problem}."
            )
        authorization = _issue_declaration_write(declaration, venue_day)
        for write in range(1, self._max_writes + 1):
            try:
                written = self._writer.write_declaration(authorization)
            except EquityRecommendationIssuanceCommitUnconfirmedError as exc:
                settled = self._reconcile(declaration, exc, write)
                if settled is None:
                    continue
                return settled
            return self._interpret(declaration, written, write)
        raise EquityIssuanceDeclarationUnresolvedError(
            f"No write of {self._max_writes} was confirmed; each reload found the declaration "
            "not yet recorded."
        )

    def _reconcile(
        self,
        declaration: EquityIssuanceCompletenessDeclaration,
        error: Exception,
        writes: int,
    ) -> EquityIssuanceDeclarationResult | None:
        try:
            view = self._reader.session_view(declaration.exchange_code, declaration.trading_date)
        except EquityRecommendationIssuanceUnavailableError as reload_error:
            raise EquityIssuanceDeclarationUnresolvedError(
                f"The commit is unconfirmed ({error}) and the reload failed ({reload_error}); "
                "nothing is claimed."
            ) from reload_error
        if view is None or view.complete_through.compare(declaration.complete_through) < 0:
            return None
        if view.complete_through != declaration.complete_through:
            raise EquityIssuanceDeclarationUnresolvedError(
                f"The commit is unconfirmed ({error}) and the session is now declared through "
                f"{view.complete_through}, so whether this declaration was recorded is unknown."
            )
        return EquityIssuanceDeclarationResult(
            _D.DECLARATION_EQUIVALENT,
            declaration,
            None,
            writes,
            True,
            "The session is declared complete through this instant (this commit or an equal one).",
        )

    def _interpret(
        self,
        declaration: EquityIssuanceCompletenessDeclaration,
        written: EquityIssuanceDeclarationWrite,
        writes: int,
    ) -> EquityIssuanceDeclarationResult:
        if not isinstance(written, EquityIssuanceDeclarationWrite):
            raise InvalidEquityRecommendationIssuanceWriteError(
                "The writer must return an EquityIssuanceDeclarationWrite."
            )
        recorded = written.recorded
        through = recorded.declaration.complete_through
        session = (recorded.declaration.exchange_code, recorded.declaration.trading_date)
        if session != (declaration.exchange_code, declaration.trading_date):
            raise InvalidEquityRecommendationIssuanceWriteError(
                "The writer reported a declaration of another session."
            )
        relation = declaration.complete_through.compare(through)
        expected = {
            _D.DECLARED: relation == 0,
            _D.DECLARATION_EQUIVALENT: relation == 0,
            _D.COMPLETENESS_REGRESSION: relation < 0,
        }[written.outcome]
        if not expected:
            raise InvalidEquityRecommendationIssuanceWriteError(
                f"The writer reported {written.outcome.value} against a declaration through "
                f"{through}, which contradicts it."
            )
        detail = {
            _D.DECLARED: f"Declared at sequence {recorded.recording_sequence}.",
            _D.DECLARATION_EQUIVALENT: "The session was already declared through this instant.",
            _D.COMPLETENESS_REGRESSION: (
                f"The session is already declared through {through}; a lower declaration is "
                "refused and nothing was written."
            ),
        }[written.outcome]
        return EquityIssuanceDeclarationResult(
            written.outcome, declaration, recorded, writes, False, detail
        )
