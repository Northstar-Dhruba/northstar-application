"""EI-7C Application: bounded session discovery, recorded and declared views (EI7-CF1).

Application contract tests for the EI-7 Decision Register, section 13.2, and
design section 23.2. Everything is synthetic and in memory:

- ``_Sessions`` is a LABELLED SYNTHETIC IN-MEMORY TEST READER, not an adapter:
  it applies the EI7-P6 check order over plain lists. It proves how the
  Application classifies and projects what a reader reports -- **not** how the
  EI-7D SQLite adapter orders its own checks inside one transaction. The
  ledger-versus-overflow precedence on real SQLite is an EI-7D obligation.
- recorded issuances are shape-valid MODELED records derived from one real
  EI-5E issued recommendation (``ei6b._recording``) with distinct identities;
  the reader contract says an adapter re-derives each from its disposition,
  which these session tests do not need.

Nothing reads a clock, opens a connection or writes anything.
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime

import pytest
import test_issue_equity_trade_recommendation as ei5d
import test_record_equity_recommendation_lifecycle as ei6b
from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Symbol
from northstar_core.recommendations.equity import EquityIssuanceKind

from northstar_application.application_services import (
    EQUITY_RECOMMENDATION_ISSUANCE_AUTHORITY_REFERENCE,
    EquityDeclaredSessionRecommendationsView,
    EquityIssuanceCompletenessDeclaration,
    EquityIssuedRecommendation,
    EquityReadQueryRefusal,
    EquityReadQueryRefusedError,
    EquityRecordedCompletenessDeclaration,
    EquityRecordedIssuance,
    EquityRecordedSessionIssuancesView,
    EquityRecordedSessionItem,
    EquitySessionDeclarationState,
    InvalidEquitySessionViewError,
    ListEquityDeclaredSessionRecommendationsUseCase,
    ListEquityRecordedSessionIssuancesUseCase,
    equity_issuance_session_view,
    equity_recommendation_issuance_identity,
)
from northstar_application.ports import (
    EQUITY_RECORDED_SESSION_DISCOVERY_LIMIT,
    EQUITY_RECORDING_POSITION_MAXIMUM,
    EquityIssuanceWatermarkBeyondAuthorityError,
    EquityRecommendationIssuanceUnavailableError,
    EquityRecordedSessionDiscoveryLimitExceededError,
    EquityRecordedSessionIssuanceReader,
    EquityRecordedSessionIssuances,
    InvalidStoredEquityRecommendationIssuanceAuthorityError,
)

_R = EquityReadQueryRefusal
_REF = EQUITY_RECOMMENDATION_ISSUANCE_AUTHORITY_REFERENCE
_NSE = ExchangeCode("NSE")
_Integrity = InvalidStoredEquityRecommendationIssuanceAuthorityError

_BASE: list[EquityIssuedRecommendation] = []


def _base() -> EquityIssuedRecommendation:
    """One real, verified EI-5E issued recommendation, built once per module."""
    if not _BASE:
        _BASE.append(ei6b._recording().issued)
    return _BASE[0]


def _day() -> date:
    return _base().trading_date


def _variant(n: int, **changes: object) -> EquityIssuedRecommendation:
    """A shape-valid MODELED record with its own identity (TEST-ONLY session fixture)."""
    key = "eirk1-sha256:" + f"{n:064x}"
    proposal = "eirp1-sha256:" + f"{n:064x}"
    identity = equity_recommendation_issuance_identity(key, proposal, EquityIssuanceKind.MODELED)
    return dataclasses.replace(
        _base(), generation_key=key, proposal_fingerprint=proposal, issuance_id=identity, **changes
    )


def _before() -> PointInTime:
    """b = t - 1 us, the accepted fixture instant just before the base issuance t."""
    assert _base().issued_at == ei5d._T
    return ei5d._B


def _covering() -> PointInTime:
    """An instant at the base issuance: it covers it (inclusive)."""
    return _base().issued_at


class _Sessions(EquityRecordedSessionIssuanceReader):
    """LABELLED SYNTHETIC IN-MEMORY TEST READER -- not durable, not an adapter.

    ``records`` is the shared recording-sequence ledger, 1 to N. Checks run in
    the EI7-P6 order: ledger, position and watermark, declarations, bounded
    count, overflow, then per-issuance verification. ``ledger_corrupt``,
    ``issuance_corrupt`` and ``unavailable`` inject the matching failure.
    """

    def __init__(self) -> None:
        self.records: list[EquityRecordedIssuance | EquityRecordedCompletenessDeclaration] = []
        self.calls: list[tuple[object, ...]] = []
        self.ledger_corrupt = False
        self.issuance_corrupt = False
        self.unavailable = False

    def issue(self, issued: EquityIssuedRecommendation) -> EquityRecordedIssuance:
        recorded = EquityRecordedIssuance(issued, len(self.records) + 1)
        self.records.append(recorded)
        return recorded

    def declare(self, through: PointInTime, day: date | None = None) -> None:
        declaration = EquityIssuanceCompletenessDeclaration(_NSE, day or _day(), through)
        self.records.append(
            EquityRecordedCompletenessDeclaration(declaration, len(self.records) + 1)
        )

    def recorded_session_issuances(self, exchange_code, trading_date, *, limit, watermark=None):  # type: ignore[no-untyped-def]
        self.calls.append((exchange_code, trading_date, limit, watermark))
        if self.unavailable:
            raise EquityRecommendationIssuanceUnavailableError("test: authority unreadable")
        if self.ledger_corrupt:
            raise _Integrity(_Integrity.LEDGER, "test: ledger gap")
        position = len(self.records)
        selected = position if watermark is None else watermark
        if selected > position:
            raise EquityIssuanceWatermarkBeyondAuthorityError("test: watermark ahead")
        session = (exchange_code, trading_date)
        declarations = tuple(
            r
            for r in self.records
            if isinstance(r, EquityRecordedCompletenessDeclaration)
            and (r.declaration.exchange_code, r.declaration.trading_date) == session
            and r.recording_sequence <= selected
        )
        issuances = tuple(
            r
            for r in self.records
            if isinstance(r, EquityRecordedIssuance)
            and (r.recommendation.exchange_code, r.recommendation.trading_date) == session
            and r.recording_sequence <= selected
        )
        if len(issuances) > limit:
            raise EquityRecordedSessionDiscoveryLimitExceededError(limit)
        if self.issuance_corrupt:
            raise _Integrity(_Integrity.ISSUANCE, "test: does not re-derive")
        return EquityRecordedSessionIssuances(
            exchange_code, trading_date, position, selected, declarations, issuances, _REF
        )


def _recorded(reader: EquityRecordedSessionIssuanceReader, **options: object):  # type: ignore[no-untyped-def]
    return ListEquityRecordedSessionIssuancesUseCase(reader).execute(_NSE, _day(), **options)


def _declared(reader: EquityRecordedSessionIssuanceReader, **options: object):  # type: ignore[no-untyped-def]
    return ListEquityDeclaredSessionRecommendationsUseCase(reader).execute(_NSE, _day(), **options)


def _refusal(call, *args: object, **options: object) -> EquityReadQueryRefusedError:  # type: ignore[no-untyped-def]
    with pytest.raises(EquityReadQueryRefusedError) as raised:
        call(*args, **options)
    return raised.value


# ----------------------------------------------------------------------
# 1-3: empty, undeclared and declared sessions
# ----------------------------------------------------------------------


def test_s01_no_recorded_issuances_and_no_declaration_are_stated_explicitly() -> None:
    reader = _Sessions()
    recorded = _recorded(reader)
    declared = _declared(reader)
    for view in (recorded, declared):
        assert (view.recording_position, view.recording_watermark) == (0, 0)
        assert view.declaration is EquitySessionDeclarationState.NONE
        assert view.complete_through is None and view.items == ()
        assert view.reference == _REF
    assert declared.history is None  # undeclared is unknown, never an empty history


def test_s02_an_issuance_recorded_before_any_covering_declaration() -> None:
    reader = _Sessions()
    reader.declare(_before())  # complete only through t - 1 us
    issued = reader.issue(_variant(1))
    recorded = _recorded(reader)
    declared = _declared(reader)
    assert recorded.declaration is EquitySessionDeclarationState.DECLARED
    assert [(i.issuance_id, i.covered_by_declaration) for i in recorded.items] == [
        (issued.recommendation.issuance_id, False)
    ]
    # Absent from the declared view; the session is declared, never "complete".
    assert declared.declaration is EquitySessionDeclarationState.DECLARED
    assert declared.items == () and declared.history is not None
    assert declared.history.recommendations == ()


def test_s02_an_undeclared_session_with_recorded_issuances_states_its_watermark() -> None:
    reader = _Sessions()
    reader.issue(_variant(1))
    recorded = _recorded(reader)
    declared = _declared(reader)
    assert len(recorded.items) == 1 and recorded.declaration is EquitySessionDeclarationState.NONE
    assert (declared.recording_position, declared.recording_watermark) == (1, 1)
    assert declared.declaration is EquitySessionDeclarationState.NONE
    assert declared.items == () and declared.history is None


def test_s03_a_declaration_recorded_exactly_at_the_watermark_is_applied() -> None:
    reader = _Sessions()
    reader.issue(_variant(1))
    reader.declare(_covering())  # sequence 2
    recorded = _recorded(reader, watermark=2)
    declared = _declared(reader, watermark=2)
    assert recorded.complete_through == _covering()
    assert [i.covered_by_declaration for i in recorded.items] == [True]
    assert [i.recommendation.issuance_id for i in declared.items] == [recorded.items[0].issuance_id]
    at_one = _declared(reader, watermark=1)
    assert at_one.declaration is EquitySessionDeclarationState.NONE and at_one.items == ()


# ----------------------------------------------------------------------
# 4-5: watermark replay, position and watermark distinct
# ----------------------------------------------------------------------


def test_s04_s05_an_earlier_watermark_replays_exactly_and_keeps_n_and_w_distinct() -> None:
    reader = _Sessions()
    reader.declare(_before())  # 1
    first = reader.issue(_variant(1))  # 2
    reader.issue(_variant(2))  # 3
    reader.declare(_covering())  # 4
    replay = _recorded(reader, watermark=2)
    assert (replay.recording_position, replay.recording_watermark) == (4, 2)
    assert [i.issuance_id for i in replay.items] == [first.recommendation.issuance_id]
    assert replay.complete_through == _before()  # the later declaration is beyond W
    assert _recorded(reader, watermark=2) == replay  # same W, same answer
    current = _recorded(reader)
    assert (current.recording_position, current.recording_watermark) == (4, 4)
    assert reader.calls[-1][3] is None and reader.calls[0][3] == 2


def test_s05_a_requested_watermark_is_never_silently_advanced_to_the_position() -> None:
    class _Advancing(_Sessions):
        def recorded_session_issuances(self, exchange_code, trading_date, *, limit, watermark=None):  # type: ignore[no-untyped-def]
            return super().recorded_session_issuances(
                exchange_code, trading_date, limit=limit, watermark=None
            )

    reader = _Advancing()
    reader.issue(_variant(1))
    reader.issue(_variant(2))
    refused = _refusal(_recorded, reader, watermark=1)
    assert (refused.reason, refused.retryable) == (_R.PERSISTED_AUTHORITY_INVALID, False)


# ----------------------------------------------------------------------
# 6-7: recorded order versus the preserved EI-6A declared order (EI7-P1)
# ----------------------------------------------------------------------


def test_s06_s07_recorded_order_is_by_sequence_and_declared_order_is_ei6as() -> None:
    reader = _Sessions()
    variants = [_variant(n) for n in (3, 1, 2)]  # one instant, three identities
    for variant in variants:
        reader.issue(variant)
    reader.declare(_covering())
    recorded = _recorded(reader)
    declared = _declared(reader)
    by_sequence = [v.issuance_id for v in variants]
    by_identity = sorted(by_sequence)
    assert by_sequence != by_identity  # the fixture makes the two orders differ
    assert [i.issuance_id for i in recorded.items] == by_sequence
    assert [i.recording_sequence for i in recorded.items] == [1, 2, 3]
    assert [i.recommendation.issuance_id for i in declared.items] == by_identity
    # The declared view is exactly the unchanged EI-6A rule over the same records.
    history = equity_issuance_session_view(
        _NSE,
        _day(),
        4,
        [r.history_record() for r in reader.records if isinstance(r, EquityRecordedIssuance)],
        [r for r in reader.records if isinstance(r, EquityRecordedCompletenessDeclaration)],
    )
    assert declared.history == history
    assert len(reader.calls) == 2  # one bounded read per view, nothing else


# ----------------------------------------------------------------------
# 8-10: the bound -- complete or refused, never partial (EI7-P2, EI7-FC1)
# ----------------------------------------------------------------------


def _filled(count: int) -> _Sessions:
    reader = _Sessions()
    for n in range(1, count + 1):
        reader.issue(_variant(n))
    reader.declare(_covering())
    return reader


def test_s08_exactly_fifty_recorded_issuances_are_returned_complete_in_both_views() -> None:
    reader = _filled(EQUITY_RECORDED_SESSION_DISCOVERY_LIMIT)
    assert len(_recorded(reader).items) == 50
    assert len(_declared(reader).items) == 50
    assert {call[2] for call in reader.calls} == {50}  # the approved limit is what is asked


def test_s09_s10_fifty_one_refuse_both_views_with_no_partial_list() -> None:
    reader = _filled(EQUITY_RECORDED_SESSION_DISCOVERY_LIMIT + 1)
    for call in (_recorded, _declared):
        refused = _refusal(call, reader)
        assert (refused.reason, refused.retryable) == (_R.DISCOVERY_LIMIT_EXCEEDED, False)
        assert refused.reason is not _R.PERSISTED_AUTHORITY_INVALID  # not a corruption verdict
    # An earlier watermark under the bound is answered completely.
    assert len(_recorded(reader, watermark=50).items) == 50


def test_s10_a_reader_that_returns_more_than_the_limit_is_never_trusted() -> None:
    class _Overfull(_Sessions):
        def recorded_session_issuances(self, exchange_code, trading_date, *, limit, watermark=None):  # type: ignore[no-untyped-def]
            return super().recorded_session_issuances(
                exchange_code, trading_date, limit=limit + 10, watermark=watermark
            )

    reader = _Overfull()
    for n in range(1, 52):
        reader.issue(_variant(n))
    refused = _refusal(_recorded, reader)
    assert refused.reason is _R.PERSISTED_AUTHORITY_INVALID


# ----------------------------------------------------------------------
# 11-14: watermark and scope validation
# ----------------------------------------------------------------------


@pytest.mark.parametrize("watermark", [-1, True, "3", 1.0])
def test_s11_an_invalid_watermark_is_refused_before_any_read(watermark: object) -> None:
    reader = _filled(1)
    refused = _refusal(_recorded, reader, watermark=watermark)
    assert (refused.reason, refused.retryable) == (_R.UNSUPPORTED_SCOPE, False)
    assert reader.calls == []


def test_s12_a_watermark_ahead_of_the_authority_is_watermark_ahead() -> None:
    reader = _filled(2)  # position 3
    refused = _refusal(_declared, reader, watermark=4)
    assert (refused.reason, refused.retryable) == (_R.WATERMARK_AHEAD, False)
    assert isinstance(refused.__cause__, EquityIssuanceWatermarkBeyondAuthorityError)


@pytest.mark.parametrize("watermark", [2**63, 10**40])
def test_s13_a_watermark_above_sqlite_int64_is_ahead_and_never_reaches_a_reader(
    watermark: int,
) -> None:
    assert watermark > EQUITY_RECORDING_POSITION_MAXIMUM == 2**63 - 1
    reader = _filled(1)
    for call in (_recorded, _declared):
        refused = _refusal(call, reader, watermark=watermark)
        assert (refused.reason, refused.retryable) == (_R.WATERMARK_AHEAD, False)
    assert reader.calls == []  # never bound into anything


def test_s13_the_largest_representable_watermark_is_passed_on_and_judged_by_the_reader() -> None:
    reader = _filled(1)
    refused = _refusal(_recorded, reader, watermark=EQUITY_RECORDING_POSITION_MAXIMUM)
    assert refused.reason is _R.WATERMARK_AHEAD
    assert reader.calls[0][3] == EQUITY_RECORDING_POSITION_MAXIMUM


@pytest.mark.parametrize(
    "exchange, day",
    [
        (ExchangeCode("BSE"), "use-day"),
        ("NSE", "use-day"),
        (_NSE, "2026-08-12"),
        (_NSE, datetime(2026, 8, 12, 9, 15)),
    ],
)
def test_s14_an_unsupported_or_malformed_session_scope_is_refused(
    exchange: object, day: object
) -> None:
    reader = _filled(1)
    trading_date = _day() if day == "use-day" else day
    for use_case in (
        ListEquityRecordedSessionIssuancesUseCase(reader),
        ListEquityDeclaredSessionRecommendationsUseCase(reader),
    ):
        refused = _refusal(use_case.execute, exchange, trading_date)
        assert refused.reason is _R.UNSUPPORTED_SCOPE
    assert reader.calls == []


def test_s14_another_sessions_records_are_never_listed() -> None:
    reader = _Sessions()
    mine = reader.issue(_variant(1))
    reader.issue(_variant(2, trading_date=date(2026, 8, 13)))
    reader.issue(_variant(3, listing=ListingReference(Symbol("RELIANCE"), ExchangeCode("BSE"))))
    assert [i.issuance_id for i in _recorded(reader).items] == [mine.recommendation.issuance_id]


# ----------------------------------------------------------------------
# 15-16: integrity failures and outages (precedence documented for EI-7D)
# ----------------------------------------------------------------------


def test_s15_a_ledger_failure_is_persisted_authority_invalid_even_over_the_limit() -> None:
    # The labelled reader applies EI7-P6: LEDGER first. Real SQLite: EI-7D.
    reader = _filled(51)
    reader.ledger_corrupt = True
    refused = _refusal(_recorded, reader)
    assert (refused.reason, refused.retryable) == (_R.PERSISTED_AUTHORITY_INVALID, False)
    assert refused.__cause__.category == _Integrity.LEDGER  # type: ignore[union-attr]


def test_s15_overflow_may_precede_a_corrupt_row_but_never_an_unverified_success() -> None:
    over = _filled(51)
    over.issuance_corrupt = True
    assert _refusal(_recorded, over).reason is _R.DISCOVERY_LIMIT_EXCEEDED  # accepted (EI7-P6)
    under = _filled(50)
    under.issuance_corrupt = True
    assert _refusal(_recorded, under).reason is _R.PERSISTED_AUTHORITY_INVALID


def test_s15_a_reader_answering_another_session_is_persisted_authority_invalid() -> None:
    class _Elsewhere(_Sessions):
        def recorded_session_issuances(self, exchange_code, trading_date, *, limit, watermark=None):  # type: ignore[no-untyped-def]
            return EquityRecordedSessionIssuances(
                exchange_code, date(2026, 8, 13), 0, 0, (), (), _REF
            )

    assert _refusal(_recorded, _Elsewhere()).reason is _R.PERSISTED_AUTHORITY_INVALID


def test_s16_an_unavailable_authority_is_a_retryable_dependency_outage() -> None:
    reader = _filled(1)
    reader.unavailable = True
    refused = _refusal(_declared, reader)
    assert (refused.reason, refused.retryable) == (_R.DEPENDENCY_UNAVAILABLE, True)


# ----------------------------------------------------------------------
# 17-18: read-only use, no lifecycle in discovery, immutable views
# ----------------------------------------------------------------------


def test_s17_the_queries_hold_only_the_bounded_reader_and_call_only_it() -> None:
    reader = _filled(2)
    _recorded(reader)
    _declared(reader)
    assert all(len(call) == 4 for call in reader.calls)
    with pytest.raises(TypeError):
        ListEquityRecordedSessionIssuancesUseCase(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ListEquityDeclaredSessionRecommendationsUseCase(object())  # type: ignore[arg-type]


def test_s18_discovery_carries_no_lifecycle_and_claims_no_atomic_lifecycle_snapshot() -> None:
    for view_type in (
        EquityRecordedSessionIssuancesView,
        EquityDeclaredSessionRecommendationsView,
        EquityRecordedSessionItem,
    ):
        names = {f.name for f in dataclasses.fields(view_type)}
        assert not {n for n in names if "lifecycle" in n or "state" in n}


def test_views_are_immutable_and_hold_tuples() -> None:
    reader = _filled(2)
    recorded = _recorded(reader)
    declared = _declared(reader)
    assert isinstance(recorded.items, tuple) and isinstance(declared.items, tuple)
    with pytest.raises(dataclasses.FrozenInstanceError):
        recorded.recording_watermark = 0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        declared.items = ()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        recorded.items[0].covered_by_declaration = False  # type: ignore[misc]


def test_views_refuse_inconsistent_construction() -> None:
    reader = _filled(2)
    recorded = _recorded(reader)
    with pytest.raises(InvalidEquitySessionViewError):
        dataclasses.replace(recorded, items=tuple(reversed(recorded.items)))
    with pytest.raises(InvalidEquitySessionViewError):
        dataclasses.replace(
            recorded,
            items=(dataclasses.replace(recorded.items[0], covered_by_declaration=False),),
        )
    with pytest.raises(InvalidEquitySessionViewError):
        dataclasses.replace(recorded, recording_watermark=recorded.recording_position + 1)
    with pytest.raises(InvalidEquitySessionViewError):
        dataclasses.replace(recorded, declaration=EquitySessionDeclarationState.NONE)
    declared = _declared(reader)
    with pytest.raises(InvalidEquitySessionViewError):
        dataclasses.replace(declared, items=tuple(reversed(declared.items)))
    with pytest.raises(InvalidEquitySessionViewError):
        dataclasses.replace(declared, history=None)
