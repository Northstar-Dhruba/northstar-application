"""Caller-driven lifecycle monitoring of existing MODELED issuances (EI-6C, EI6C-P1 to P12).

The EI-6C Lifecycle Monitoring and Snapshot Selection Design v1.0.
``MonitorEquityRecommendationLifecyclesUseCase.execute(request)`` monitors the
lifecycles of existing authoritative issuances at a caller-supplied
monitoring instant ``T``. It reads no clock, keeps no state, writes nothing
itself and runs nothing in the background.

For each issuance in the requested scope (EI6C-P7):

1. load the authoritative issuance (EI-6A reader) and its stored disposition
   (for the research-window identity the catalog is scoped by);
2. load the authoritative lifecycle (EI-6B store, read only); its last event's cited
   snapshot, authenticated through the snapshot store, is the **anchor**;
3. list the session's eligible snapshots through the read-only catalog
   (``as_of <= T`` and ``issued_at <= T``; at most the governed limit) and
   re-verify every entry against the authenticated snapshot it names;
4. select deterministically (equity_lifecycle_monitoring_selection);
5. record **one snapshot per call** through
   ``RecordEquityRecommendationLifecycleUseCase`` -- the only writer -- in
   order, keeping every committed event when a later snapshot is refused.

Results are per issuance: a failure for one never suppresses another's.

What is not done
----------------
No second lifecycle state machine (Core decides through the recorder), no
direct lifecycle append, no batching, no thinning, no settlement allowance
(O-2 option A, Δ = 0), no checkpoint or cursor, no retry beyond the
recorder's own bounded reconciliation, no snapshot creation, and no research,
generation, declaration or issuance orchestration.

Determinism (design section 10): identical authoritative history, catalog,
``T`` and policy give identical selection and idempotent behaviour. Several
runs equal one later run only under the design's stated conditions; that is
not guaranteed in general, and history is never rewritten to achieve it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

from northstar_core.domain.value_objects import ListingReference
from northstar_core.foundation.value_objects import ExchangeCode, PointInTime, Timeframe
from northstar_core.recommendations.equity.lifecycle import (
    EquityLifecycleEvent,
    EquityLifecycleState,
)
from northstar_core.strategy import StrategyIdentity

from northstar_application.application_services.equity_candle_evidence_snapshot import (
    EquityCandleEvidenceClass,
    EquityCandleEvidenceSnapshot,
)
from northstar_application.application_services.equity_lifecycle_monitoring_selection import (
    EQUITY_LIFECYCLE_MONITORING_CATALOG_LIMIT,
    EQUITY_LIFECYCLE_MONITORING_POLICY,
    EquityLifecycleSelectionStop,
    EquityLifecycleSnapshotAnchor,
    select_equity_lifecycle_snapshots,
)
from northstar_application.application_services.equity_recommendation_generation_decoding import (
    IncompatibleEquityRecommendationRecordError,
    InvalidEquityRecommendationRecordError,
)
from northstar_application.application_services.record_equity_recommendation_lifecycle import (
    EquityLifecycleRecordingOutcome,
    EquityLifecycleRecordingRefusal,
    EquityLifecycleRecordingRefusedError,
    EquityLifecycleRecordingUnresolvedError,
    RecordEquityRecommendationLifecycleUseCase,
)
from northstar_application.ports.equity_candle_evidence_journal import (
    EquityCandleEvidenceSnapshotStore,
    EquityCandleEvidenceUnavailableError,
)
from northstar_application.ports.equity_candle_evidence_snapshot_catalog import (
    EquityCandleEvidenceSnapshotCatalog,
    EquityCandleEvidenceSnapshotCatalogEntry,
    EquityCandleEvidenceSnapshotCatalogLimitExceededError,
    EquityCandleEvidenceSnapshotCatalogScope,
    EquityCandleEvidenceSnapshotCatalogUnavailableError,
    InvalidEquityCandleEvidenceSnapshotCatalogError,
)
from northstar_application.ports.equity_recommendation_generation_store import (
    EquityRecommendationGenerationRecordReader,
    EquityRecommendationGenerationStoreUnavailableError,
)
from northstar_application.ports.equity_recommendation_issuance import (
    EquityRecommendationIssuanceReader,
    EquityRecommendationIssuanceUnavailableError,
    InvalidStoredEquityRecommendationIssuanceAuthorityError,
)
from northstar_application.ports.equity_recommendation_lifecycle_reader import (
    EquityRecommendationLifecycleReader,
)
from northstar_application.ports.equity_recommendation_lifecycle_store import (
    EquityRecommendationLifecycleStoreUnavailableError,
    InvalidStoredEquityRecommendationLifecycleError,
)

_FIVE_MINUTES = Timeframe("5m")
_O = EquityLifecycleRecordingOutcome
_R = EquityLifecycleRecordingRefusal
_Catalog = InvalidEquityCandleEvidenceSnapshotCatalogError


# ---------------------------------------------------------------------------
# Request and scope (EI6C-P7)
# ---------------------------------------------------------------------------


def _plain_date(value: object) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise TypeError("trading_date must be a plain date.")
    return value


@dataclass(frozen=True, slots=True)
class EquityLifecycleMonitoringExplicitScope:
    """EXPLICIT: the named issuances, de-duplicated and in identity order."""

    issuance_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.issuance_ids, tuple) or not self.issuance_ids:
            raise TypeError("issuance_ids must be a non-empty tuple.")
        if not all(isinstance(i, str) and i for i in self.issuance_ids):
            raise TypeError("issuance_ids must be non-empty text.")
        object.__setattr__(self, "issuance_ids", tuple(sorted(set(self.issuance_ids))))


@dataclass(frozen=True, slots=True)
class EquityLifecycleMonitoringStrategyListingScope:
    """STRATEGY_LISTING: the zero or one issuance of a strategy, listing and trading date.

    Finds an issuance not yet covered by a completeness declaration.
    """

    strategy: StrategyIdentity
    listing: ListingReference
    trading_date: date

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, StrategyIdentity):
            raise TypeError("strategy must be a StrategyIdentity.")
        if not isinstance(self.listing, ListingReference):
            raise TypeError("listing must be a ListingReference.")
        _plain_date(self.trading_date)


@dataclass(frozen=True, slots=True)
class EquityLifecycleMonitoringDeclaredSessionScope:
    """DECLARED_SESSION: the issuances listed by the session's declared view.

    ``watermark`` selects ``session_view_at``; None selects the current view.
    **Incomplete by design:** an issuance made after the latest declaration
    (for example at ``t`` after the declaration through ``t - 1 µs``) is not
    listed until a later declaration covers it.
    """

    exchange_code: ExchangeCode
    trading_date: date
    watermark: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.exchange_code, ExchangeCode):
            raise TypeError("exchange_code must be an ExchangeCode.")
        _plain_date(self.trading_date)
        if self.watermark is not None and (
            isinstance(self.watermark, bool)
            or not isinstance(self.watermark, int)
            or self.watermark < 0
        ):
            raise TypeError("watermark must be a non-negative integer or None.")


EquityLifecycleMonitoringScope = (
    EquityLifecycleMonitoringExplicitScope
    | EquityLifecycleMonitoringStrategyListingScope
    | EquityLifecycleMonitoringDeclaredSessionScope
)
_SCOPES = (
    EquityLifecycleMonitoringExplicitScope,
    EquityLifecycleMonitoringStrategyListingScope,
    EquityLifecycleMonitoringDeclaredSessionScope,
)


@dataclass(frozen=True, slots=True)
class EquityLifecycleMonitoringRequest:
    """One monitoring invocation: exactly one scope and the caller's monitoring instant."""

    scope: EquityLifecycleMonitoringScope
    monitoring_instant: PointInTime

    def __post_init__(self) -> None:
        if not isinstance(self.scope, _SCOPES):
            raise TypeError("scope must be one EI-6C monitoring scope.")
        if not isinstance(self.monitoring_instant, PointInTime):
            raise TypeError("monitoring_instant must be a PointInTime.")


# ---------------------------------------------------------------------------
# Outcomes and results (EI6C-P9, P11)
# ---------------------------------------------------------------------------


class EquityLifecycleMonitoringOutcome(StrEnum):
    """The final outcome of monitoring one issuance in one run."""

    APPENDED = "APPENDED"  # EI-6B: at least one event appended this run
    EQUIVALENT = "EQUIVALENT"  # EI-6B: the events were already recorded
    CONFLICT = "CONFLICT"  # EI-6B: another writer advanced the history; nothing adopted
    NO_NEW_EVENTS = "NO_NEW_EVENTS"  # EI-6B: evaluated; no event (a successful no-op)
    NO_ELIGIBLE_SNAPSHOT = "NO_ELIGIBLE_SNAPSHOT"  # nothing to evaluate (a successful no-op)
    STOPPED_AT_CONFLICT = "STOPPED_AT_CONFLICT"  # same-instant conflict
    STOPPED_AT_REGRESSION = "STOPPED_AT_REGRESSION"  # watermark regression
    STOPPED_AT_SNAPSHOT_REFUSAL = "STOPPED_AT_SNAPSHOT_REFUSAL"  # an EI-5C SNAPSHOT_* refusal
    CATALOG_LIMIT_EXCEEDED = "CATALOG_LIMIT_EXCEEDED"  # nothing recorded
    ISSUANCE_EVALUATION_FAILED = "ISSUANCE_EVALUATION_FAILED"  # untyped failure; fail closed
    REFUSED = "REFUSED"  # typed refusal: corruption, unavailability, provenance
    UNRESOLVED = "UNRESOLVED"  # commit uncertainty not resolved; nothing claimed

    @property
    def is_success(self) -> bool:
        """Whether the run completed for this issuance without a stop or failure."""
        return self in _SUCCESSES


_SUCCESSES = frozenset(
    {
        EquityLifecycleMonitoringOutcome.APPENDED,
        EquityLifecycleMonitoringOutcome.EQUIVALENT,
        EquityLifecycleMonitoringOutcome.NO_NEW_EVENTS,
        EquityLifecycleMonitoringOutcome.NO_ELIGIBLE_SNAPSHOT,
    }
)
_M = EquityLifecycleMonitoringOutcome


class EquityLifecycleMonitoringRefusal(StrEnum):
    """EI-6C's own refusal reasons, beside the recorder's EquityLifecycleRecordingRefusal."""

    CATALOG_UNAVAILABLE = "CATALOG_UNAVAILABLE"
    CATALOG_INVALID = "CATALOG_INVALID"
    SNAPSHOT_STORE_UNAVAILABLE = "SNAPSHOT_STORE_UNAVAILABLE"
    ANCHOR_SNAPSHOT_INVALID = "ANCHOR_SNAPSHOT_INVALID"
    GENERATION_NOT_ISSUABLE = "GENERATION_NOT_ISSUABLE"

    @property
    def retryable(self) -> bool:
        """Whether retrying the same request later may succeed."""
        return self in (
            EquityLifecycleMonitoringRefusal.CATALOG_UNAVAILABLE,
            EquityLifecycleMonitoringRefusal.SNAPSHOT_STORE_UNAVAILABLE,
        )


class EquityLifecycleMonitoringDiagnosticKind(StrEnum):
    """Non-durable diagnostics; they accompany any outcome."""

    LATE_HISTORICAL_SNAPSHOT_IGNORED = "LATE_HISTORICAL_SNAPSHOT_IGNORED"


@dataclass(frozen=True, slots=True)
class EquityLifecycleMonitoringDiagnostic:
    """A non-durable observation about one snapshot; nothing is persisted."""

    kind: EquityLifecycleMonitoringDiagnosticKind
    snapshot_id: str
    as_of: PointInTime


@dataclass(frozen=True, slots=True)
class EquityLifecycleMonitoringStep:
    """One recorder call: one snapshot and what EI-6B confirmed for it."""

    snapshot_id: str
    as_of: PointInTime
    outcome: EquityLifecycleRecordingOutcome
    events: tuple[EquityLifecycleEvent, ...]
    reconciled: bool


@dataclass(frozen=True, slots=True)
class EquityLifecycleMonitoringResult:
    """What one run established for one issuance.

    ``steps`` are the recorder calls made, in order; ``appended`` are the
    events EI-6B confirmed it appended in this run. ``stopped_at`` and
    ``stop_snapshot_ids`` name where processing stopped. ``refusal`` is the
    typed reason for ``REFUSED`` and the stop outcomes; ``cause`` describes an
    unexpected failure. Nothing here is persisted.
    """

    issuance_id: str
    outcome: EquityLifecycleMonitoringOutcome
    steps: tuple[EquityLifecycleMonitoringStep, ...] = ()
    appended: tuple[EquityLifecycleEvent, ...] = ()
    state: EquityLifecycleState | None = None
    stopped_at: PointInTime | None = None
    stop_snapshot_ids: tuple[str, ...] = ()
    refusal: EquityLifecycleRecordingRefusal | EquityLifecycleMonitoringRefusal | None = None
    retryable: bool = False
    cause: str | None = None
    diagnostics: tuple[EquityLifecycleMonitoringDiagnostic, ...] = ()

    @property
    def recorded_snapshot_ids(self) -> tuple[str, ...]:
        """Return the snapshots passed to the recorder, one per call, in order."""
        return tuple(step.snapshot_id for step in self.steps)


@dataclass(frozen=True, slots=True)
class EquityLifecycleMonitoringReport:
    """One invocation's results, one per resolved issuance, in identity order."""

    scope: EquityLifecycleMonitoringScope
    monitoring_instant: PointInTime
    policy: str
    results: tuple[EquityLifecycleMonitoringResult, ...]


class EquityLifecycleMonitoringScopeError(RuntimeError):
    """Raised when the requested scope itself cannot be resolved; no issuance is monitored."""

    def __init__(self, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class _Stop(Exception):  # noqa: N818 -- an internal control-flow signal, never raised out
    def __init__(self, result: EquityLifecycleMonitoringResult) -> None:
        super().__init__(result.outcome.value)
        self.result = result


# ---------------------------------------------------------------------------
# The use case
# ---------------------------------------------------------------------------


class MonitorEquityRecommendationLifecyclesUseCase:
    """Monitor existing issuances' lifecycles at a caller-supplied instant; see the module."""

    def __init__(
        self,
        issuances: EquityRecommendationIssuanceReader,
        generations: EquityRecommendationGenerationRecordReader,
        snapshots: EquityCandleEvidenceSnapshotStore,
        catalog: EquityCandleEvidenceSnapshotCatalog,
        lifecycles: EquityRecommendationLifecycleReader,
        recorder: RecordEquityRecommendationLifecycleUseCase,
    ) -> None:
        checks = (
            (issuances, EquityRecommendationIssuanceReader, "issuances"),
            (generations, EquityRecommendationGenerationRecordReader, "generations"),
            (snapshots, EquityCandleEvidenceSnapshotStore, "snapshots"),
            (catalog, EquityCandleEvidenceSnapshotCatalog, "catalog"),
            (lifecycles, EquityRecommendationLifecycleReader, "lifecycles"),
            (recorder, RecordEquityRecommendationLifecycleUseCase, "recorder"),
        )
        for value, kind, name in checks:
            if not isinstance(value, kind):
                raise TypeError(f"{name} must be a {kind.__name__}.")
        self._issuances = issuances
        self._generations = generations
        self._snapshots = snapshots
        self._catalog = catalog
        self._lifecycles = lifecycles
        self._recorder = recorder

    def execute(self, request: EquityLifecycleMonitoringRequest) -> EquityLifecycleMonitoringReport:
        """Monitor every issuance of ``request.scope`` at ``request.monitoring_instant``."""
        if not isinstance(request, EquityLifecycleMonitoringRequest):
            raise TypeError("request must be an EquityLifecycleMonitoringRequest.")
        instant = request.monitoring_instant
        results = tuple(
            self._monitor(issuance_id, instant)
            for issuance_id in sorted(set(self._resolve(request.scope)))
        )
        return EquityLifecycleMonitoringReport(
            request.scope, instant, EQUITY_LIFECYCLE_MONITORING_POLICY, results
        )

    # -- scope ---------------------------------------------------------------

    def _resolve(self, scope: EquityLifecycleMonitoringScope) -> tuple[str, ...]:
        if isinstance(scope, EquityLifecycleMonitoringExplicitScope):
            return scope.issuance_ids  # each is loaded, and typed, per issuance
        try:
            if isinstance(scope, EquityLifecycleMonitoringStrategyListingScope):
                found = self._issuances.load_session_issuance(
                    scope.strategy, scope.listing, scope.trading_date
                )
                return () if found is None else (found.recommendation.issuance_id,)
            view = (
                self._issuances.session_view(scope.exchange_code, scope.trading_date)
                if scope.watermark is None
                else self._issuances.session_view_at(
                    scope.exchange_code, scope.trading_date, scope.watermark
                )
            )
        except EquityRecommendationIssuanceUnavailableError as exc:
            raise EquityLifecycleMonitoringScopeError(
                f"The issuance authority is unavailable: {exc}", retryable=True
            ) from exc
        except InvalidStoredEquityRecommendationIssuanceAuthorityError as exc:
            raise EquityLifecycleMonitoringScopeError(
                f"The issuance authority is corrupt: {exc}", retryable=False
            ) from exc
        return () if view is None else tuple(r.recommendation_key for r in view.recommendations)

    # -- one issuance ----------------------------------------------------------

    def _monitor(self, issuance_id: str, instant: PointInTime) -> EquityLifecycleMonitoringResult:
        try:
            return self._run(issuance_id, instant)
        except _Stop as stop:
            return stop.result
        except Exception as exc:  # noqa: BLE001 -- fail this issuance closed, typed (EI6C-P9)
            return EquityLifecycleMonitoringResult(
                issuance_id, _M.ISSUANCE_EVALUATION_FAILED, cause=f"{type(exc).__name__}: {exc}"
            )

    def _refused(
        self,
        issuance_id: str,
        reason: EquityLifecycleRecordingRefusal | EquityLifecycleMonitoringRefusal,
        exc: BaseException,
        retryable: bool,
    ) -> _Stop:
        return _Stop(
            EquityLifecycleMonitoringResult(
                issuance_id,
                _M.REFUSED,
                refusal=reason,
                retryable=retryable,
                cause=f"{type(exc).__name__}: {exc}",
            )
        )

    def _run(self, issuance_id: str, instant: PointInTime) -> EquityLifecycleMonitoringResult:
        scope, issued_at = self._scope(issuance_id)
        anchor, state = self._anchor(issuance_id)
        entries = self._entries(issuance_id, scope, instant)
        try:
            selection = select_equity_lifecycle_snapshots(
                entries, scope, instant, issued_at, anchor
            )
        except InvalidEquityCandleEvidenceSnapshotCatalogError as exc:
            raise self._refused(
                issuance_id, EquityLifecycleMonitoringRefusal.CATALOG_INVALID, exc, False
            ) from exc
        diagnostics = tuple(
            EquityLifecycleMonitoringDiagnostic(
                EquityLifecycleMonitoringDiagnosticKind.LATE_HISTORICAL_SNAPSHOT_IGNORED,
                entry.snapshot_id,
                entry.as_of,
            )
            for entry in selection.late_historical
        )
        steps: list[EquityLifecycleMonitoringStep] = []

        def result(outcome: _M, **fields: object) -> EquityLifecycleMonitoringResult:
            appended = tuple(
                event for step in steps if step.outcome is _O.APPENDED for event in step.events
            )
            return EquityLifecycleMonitoringResult(
                issuance_id,
                outcome,
                tuple(steps),
                appended,
                diagnostics=diagnostics,
                **fields,  # type: ignore[arg-type]
            )

        for entry in selection.selected:
            try:
                recorded = self._recorder.execute(issuance_id, (entry.snapshot_id,))
            except EquityLifecycleRecordingRefusedError as exc:
                snapshot_refusal = exc.reason.value.startswith("SNAPSHOT_")
                return result(
                    _M.STOPPED_AT_SNAPSHOT_REFUSAL if snapshot_refusal else _M.REFUSED,
                    state=state,
                    stopped_at=entry.as_of if snapshot_refusal else None,
                    stop_snapshot_ids=(entry.snapshot_id,) if snapshot_refusal else (),
                    refusal=exc.reason,
                    retryable=exc.retryable,
                    cause=str(exc),
                )
            except EquityLifecycleRecordingUnresolvedError as exc:
                return result(_M.UNRESOLVED, state=state, cause=str(exc))
            except InvalidStoredEquityRecommendationLifecycleError as exc:
                return result(
                    _M.REFUSED, state=state, refusal=_R.STORED_LIFECYCLE_INVALID, cause=str(exc)
                )
            except Exception as exc:  # noqa: BLE001 -- untyped: fail closed, prefix kept
                return result(
                    _M.ISSUANCE_EVALUATION_FAILED,
                    state=state,
                    stopped_at=entry.as_of,
                    stop_snapshot_ids=(entry.snapshot_id,),
                    cause=f"{type(exc).__name__}: {exc}",
                )
            steps.append(
                EquityLifecycleMonitoringStep(
                    entry.snapshot_id,
                    entry.as_of,
                    recorded.outcome,
                    recorded.new_events,
                    recorded.reconciled,
                )
            )
            state = recorded.state
            if recorded.outcome is _O.CONFLICT:
                return result(_M.CONFLICT, state=state, stopped_at=entry.as_of)

        if selection.stop is not None:
            stop = (
                _M.STOPPED_AT_CONFLICT
                if selection.stop is EquityLifecycleSelectionStop.CONFLICT
                else _M.STOPPED_AT_REGRESSION
            )
            return result(
                stop,
                state=state,
                stopped_at=selection.stop_at,
                stop_snapshot_ids=selection.stop_snapshot_ids,
            )
        if not steps:
            return result(_M.NO_ELIGIBLE_SNAPSHOT, state=state)
        outcomes = {step.outcome for step in steps}
        final = (
            _M.APPENDED
            if _O.APPENDED in outcomes
            else _M.EQUIVALENT
            if _O.EQUIVALENT in outcomes
            else _M.NO_NEW_EVENTS
        )
        return result(final, state=state)

    def _scope(
        self, issuance_id: str
    ) -> tuple[EquityCandleEvidenceSnapshotCatalogScope, PointInTime]:
        try:
            recorded = self._issuances.load_issuance(issuance_id)
        except EquityRecommendationIssuanceUnavailableError as exc:
            raise self._refused(issuance_id, _R.AUTHORITY_UNAVAILABLE, exc, True) from exc
        except InvalidStoredEquityRecommendationIssuanceAuthorityError as exc:
            raise self._refused(issuance_id, _R.STORED_AUTHORITY_INVALID, exc, False) from exc
        if recorded is None:
            raise _Stop(
                EquityLifecycleMonitoringResult(
                    issuance_id, _M.REFUSED, refusal=_R.ISSUANCE_NOT_FOUND
                )
            )
        issued = recorded.recommendation
        try:
            disposition = self._generations.load_disposition(issued.generation_key)
        except EquityRecommendationGenerationStoreUnavailableError as exc:
            raise self._refused(issuance_id, _R.GENERATION_STORE_UNAVAILABLE, exc, True) from exc
        except IncompatibleEquityRecommendationRecordError as exc:
            raise self._refused(issuance_id, _R.STORED_GENERATION_INCOMPATIBLE, exc, False) from exc
        except InvalidEquityRecommendationRecordError as exc:
            raise self._refused(issuance_id, _R.STORED_GENERATION_INVALID, exc, False) from exc
        if disposition is None:
            raise _Stop(
                EquityLifecycleMonitoringResult(
                    issuance_id, _M.REFUSED, refusal=_R.GENERATION_NOT_FOUND
                )
            )
        if disposition.proposal is None:
            raise _Stop(
                EquityLifecycleMonitoringResult(
                    issuance_id,
                    _M.REFUSED,
                    refusal=EquityLifecycleMonitoringRefusal.GENERATION_NOT_ISSUABLE,
                )
            )
        scope = EquityCandleEvidenceSnapshotCatalogScope(
            issued.listing,
            issued.trading_date,
            _FIVE_MINUTES,
            disposition.candidate.research_window_policy_version,
            EquityCandleEvidenceClass.SYNTHETIC,
        )
        return scope, issued.issued_at

    def _anchor(
        self, issuance_id: str
    ) -> tuple[EquityLifecycleSnapshotAnchor | None, EquityLifecycleState | None]:
        try:
            stored = self._lifecycles.load_lifecycle(issuance_id)
        except EquityRecommendationLifecycleStoreUnavailableError as exc:
            raise self._refused(issuance_id, _R.STORE_UNAVAILABLE, exc, True) from exc
        except InvalidStoredEquityRecommendationLifecycleError as exc:
            raise self._refused(issuance_id, _R.STORED_LIFECYCLE_INVALID, exc, False) from exc
        except InvalidStoredEquityRecommendationIssuanceAuthorityError as exc:
            raise self._refused(issuance_id, _R.STORED_AUTHORITY_INVALID, exc, False) from exc
        if stored is None:
            raise _Stop(
                EquityLifecycleMonitoringResult(
                    issuance_id, _M.REFUSED, refusal=_R.ISSUANCE_NOT_FOUND
                )
            )
        lifecycle = stored.lifecycle
        if not lifecycle.events:
            return None, lifecycle.state
        last = lifecycle.events[-1]
        snapshot = self._authenticated(issuance_id, last.evidence.reference)
        if snapshot is None or snapshot.as_of.compare(last.evaluated_at) != 0:
            raise _Stop(
                EquityLifecycleMonitoringResult(
                    issuance_id,
                    _M.REFUSED,
                    state=lifecycle.state,
                    refusal=EquityLifecycleMonitoringRefusal.ANCHOR_SNAPSHOT_INVALID,
                    cause="The last event's cited snapshot is missing or not at its instant.",
                )
            )
        anchor = EquityLifecycleSnapshotAnchor(
            snapshot.snapshot_id,
            snapshot.as_of,
            snapshot.issued_at,
            snapshot.watermark,
            snapshot.manifest_fingerprint,
        )
        return anchor, lifecycle.state

    def _authenticated(
        self, issuance_id: str, snapshot_id: str
    ) -> EquityCandleEvidenceSnapshot | None:
        try:
            snapshot = self._snapshots.get_snapshot(snapshot_id)
        except EquityCandleEvidenceUnavailableError as exc:
            raise self._refused(
                issuance_id, EquityLifecycleMonitoringRefusal.SNAPSHOT_STORE_UNAVAILABLE, exc, True
            ) from exc
        if snapshot is not None and snapshot.snapshot_id != snapshot_id:
            return None
        return snapshot

    def _entries(
        self,
        issuance_id: str,
        scope: EquityCandleEvidenceSnapshotCatalogScope,
        instant: PointInTime,
    ) -> tuple[EquityCandleEvidenceSnapshotCatalogEntry, ...]:
        refusal = EquityLifecycleMonitoringRefusal
        try:
            entries = self._catalog.list_snapshots(
                scope, instant, EQUITY_LIFECYCLE_MONITORING_CATALOG_LIMIT
            )
        except EquityCandleEvidenceSnapshotCatalogLimitExceededError as exc:
            raise _Stop(
                EquityLifecycleMonitoringResult(
                    issuance_id, _M.CATALOG_LIMIT_EXCEEDED, cause=str(exc)
                )
            ) from exc
        except EquityCandleEvidenceSnapshotCatalogUnavailableError as exc:
            raise self._refused(issuance_id, refusal.CATALOG_UNAVAILABLE, exc, True) from exc
        except InvalidEquityCandleEvidenceSnapshotCatalogError as exc:
            raise self._refused(issuance_id, refusal.CATALOG_INVALID, exc, False) from exc
        if not isinstance(entries, tuple):
            raise self._refused(
                issuance_id,
                refusal.CATALOG_INVALID,
                _Catalog(_Catalog.METADATA, "The catalog answered something other than a tuple."),
                False,
            )
        if len(entries) > EQUITY_LIFECYCLE_MONITORING_CATALOG_LIMIT:
            raise _Stop(
                EquityLifecycleMonitoringResult(
                    issuance_id,
                    _M.CATALOG_LIMIT_EXCEEDED,
                    cause="The catalog returned more entries than the governed limit.",
                )
            )
        for entry in entries:
            self._verify(issuance_id, entry)
        return entries

    def _verify(self, issuance_id: str, entry: EquityCandleEvidenceSnapshotCatalogEntry) -> None:
        """Every catalog claim must equal the authenticated snapshot it names."""
        if not isinstance(entry, EquityCandleEvidenceSnapshotCatalogEntry):
            raise self._refused(
                issuance_id,
                EquityLifecycleMonitoringRefusal.CATALOG_INVALID,
                _Catalog(_Catalog.METADATA, "A catalog answer holds a non-entry value."),
                False,
            )
        snapshot = self._authenticated(issuance_id, entry.snapshot_id)
        claims = (
            entry.listing,
            entry.trading_date,
            entry.timeframe,
            entry.window_policy_version,
            entry.evidence_class,
            entry.as_of,
            entry.issued_at,
            entry.watermark,
            entry.manifest_fingerprint,
        )
        if snapshot is None or claims != (
            snapshot.listing,
            snapshot.trading_date,
            snapshot.timeframe,
            snapshot.window_policy_version,
            snapshot.evidence_class,
            snapshot.as_of,
            snapshot.issued_at,
            snapshot.watermark,
            snapshot.manifest_fingerprint,
        ):
            raise self._refused(
                issuance_id,
                EquityLifecycleMonitoringRefusal.CATALOG_INVALID,
                _Catalog(
                    _Catalog.INTEGRITY,
                    f"Catalog entry {entry.snapshot_id} does not match its authenticated snapshot.",
                ),
                False,
            )
