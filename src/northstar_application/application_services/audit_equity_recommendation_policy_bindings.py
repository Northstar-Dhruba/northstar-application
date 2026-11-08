"""Read-only historical policy-binding audit (EI4-D10-DQ7, ADR-022 section 9).

AuditEquityRecommendationPolicyBindingsUseCase classifies the policy labels of
stored generation dispositions against the immutable policy catalog. It is not
historical assessment replay. Replay (EquityRecommendationGenerationVerifier,
ADR-021 section 10) never consults the catalog, and this audit changes none of
its verdicts.

For each audited record it takes the frozen policy set its assessment holds,
binds the five components and the set exactly as publication would, and
classifies each of the six identities:

- HISTORICALLY_VERIFIED_BINDING: the catalog holds the same binding, AND
  recorded historical evidence (EquitySyntheticPolicyPublicationHistory) shows
  it was published before the record was committed;
- BINDING_NOT_ESTABLISHED: the label is not published; or it is published
  with the same content but no evidence shows it existed when the record was
  made. A binding that matches today is never taken as historical proof;
- HISTORICAL_POLICY_VERSION_CONFLICT: the catalog binds the label to other
  content, or another audited record holds other content under the same
  label;
- INVALID_CANONICAL_EVIDENCE: the stored record, or the catalog's binding,
  does not verify;
- INCOMPATIBLE_HISTORICAL_FORMAT: the stored record, or the catalog's
  binding, uses an unsupported format;
- AUDIT_EVIDENCE_UNAVAILABLE: the generation store, the catalog or the
  publication history cannot answer. This is an explicit uncertainty, never a
  verdict.

A record's own status is the most severe of its findings, in the order above
from INVALID_CANONICAL_EVIDENCE down to HISTORICALLY_VERIFIED_BINDING.

The audit only reads: it never publishes, backfills, repairs, re-keys or
writes anything. No historical publication evidence exists for any record
today (ADR-022 section 9), so without an evidence source it never answers
HISTORICALLY_VERIFIED_BINDING.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from northstar_application.application_services.equity_recommendation_generation_decoding import (
    IncompatibleEquityRecommendationRecordError,
    InvalidEquityRecommendationRecordError,
)
from northstar_application.application_services.equity_recommendation_generation_records import (
    EquityRecommendationGenerationDisposition,
)
from northstar_application.application_services.equity_synthetic_policy_bindings import (
    EquitySyntheticPolicyBindingIdentity,
    EquitySyntheticPolicyBundle,
    EquitySyntheticPolicyDefinitionBinding,
    EquitySyntheticPolicySetBinding,
    IncompatibleEquitySyntheticPolicyBindingError,
    InvalidEquitySyntheticPolicyBindingError,
)
from northstar_application.ports.equity_recommendation_generation_store import (
    EquityRecommendationGenerationStore,
    EquityRecommendationGenerationStoreUnavailableError,
)
from northstar_application.ports.equity_synthetic_policy_catalog_store import (
    EquitySyntheticPolicyCatalogStore,
    EquitySyntheticPolicyCatalogUnavailableError,
    EquitySyntheticPolicyPublicationHistory,
)


class EquityRecommendationPolicyBindingStatus(StrEnum):
    """The audit classification of one policy identity of one stored record."""

    HISTORICALLY_VERIFIED_BINDING = "HISTORICALLY_VERIFIED_BINDING"
    BINDING_NOT_ESTABLISHED = "BINDING_NOT_ESTABLISHED"
    HISTORICAL_POLICY_VERSION_CONFLICT = "HISTORICAL_POLICY_VERSION_CONFLICT"
    INVALID_CANONICAL_EVIDENCE = "INVALID_CANONICAL_EVIDENCE"
    INCOMPATIBLE_HISTORICAL_FORMAT = "INCOMPATIBLE_HISTORICAL_FORMAT"
    AUDIT_EVIDENCE_UNAVAILABLE = "AUDIT_EVIDENCE_UNAVAILABLE"


class EquityRecommendationPolicyBindingEvidence(StrEnum):
    """Which evidence a finding rests on."""

    STORED_RECORD = "STORED_RECORD"
    OTHER_STORED_RECORDS = "OTHER_STORED_RECORDS"
    CATALOG = "CATALOG"
    PUBLICATION_HISTORY = "PUBLICATION_HISTORY"


_S = EquityRecommendationPolicyBindingStatus
_E = EquityRecommendationPolicyBindingEvidence
# Most severe first.
_SEVERITY = (
    _S.INVALID_CANONICAL_EVIDENCE,
    _S.INCOMPATIBLE_HISTORICAL_FORMAT,
    _S.HISTORICAL_POLICY_VERSION_CONFLICT,
    _S.AUDIT_EVIDENCE_UNAVAILABLE,
    _S.BINDING_NOT_ESTABLISHED,
    _S.HISTORICALLY_VERIFIED_BINDING,
)
_Binding = EquitySyntheticPolicyDefinitionBinding | EquitySyntheticPolicySetBinding


@dataclass(frozen=True, slots=True)
class EquityRecommendationPolicyBindingFinding:
    """The classification of one policy identity of one record.

    ``recorded_fingerprint`` is the ``eipd1`` (component) or ``eirs1`` (set)
    fingerprint of the record's frozen content; ``catalog_fingerprint`` is
    the catalog's, when one was read. ``detail`` is diagnostic prose only.
    """

    identity: EquitySyntheticPolicyBindingIdentity
    status: EquityRecommendationPolicyBindingStatus
    evidence: EquityRecommendationPolicyBindingEvidence
    recorded_fingerprint: str
    catalog_fingerprint: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class EquityRecommendationPolicyBindingRecordAudit:
    """The audit of one stored disposition: its findings, or why none could be made."""

    generation_key: str
    findings: tuple[EquityRecommendationPolicyBindingFinding, ...]
    record_status: EquityRecommendationPolicyBindingStatus | None = None
    detail: str = ""

    @property
    def status(self) -> EquityRecommendationPolicyBindingStatus:
        """Return the most severe classification of the record."""
        statuses = {f.status for f in self.findings}
        if self.record_status is not None:
            statuses.add(self.record_status)
        return next(s for s in _SEVERITY if s in statuses)


@dataclass(frozen=True, slots=True)
class EquityRecommendationPolicyBindingAudit:
    """The audit of several stored dispositions. ``missing`` keys have no stored record."""

    records: tuple[EquityRecommendationPolicyBindingRecordAudit, ...]
    missing: tuple[str, ...]


def _fingerprint(binding: _Binding) -> str:
    if isinstance(binding, EquitySyntheticPolicySetBinding):
        return binding.policy_set_fingerprint
    return binding.fingerprint


class AuditEquityRecommendationPolicyBindingsUseCase:
    """Classifies stored records' policy labels against the catalog, reading only."""

    def __init__(
        self,
        records: EquityRecommendationGenerationStore,
        catalog: EquitySyntheticPolicyCatalogStore,
        publication_history: EquitySyntheticPolicyPublicationHistory | None = None,
    ) -> None:
        if not isinstance(records, EquityRecommendationGenerationStore):
            raise TypeError("records must be an EquityRecommendationGenerationStore.")
        if not isinstance(catalog, EquitySyntheticPolicyCatalogStore):
            raise TypeError("catalog must be an EquitySyntheticPolicyCatalogStore.")
        if publication_history is not None and not isinstance(
            publication_history, EquitySyntheticPolicyPublicationHistory
        ):
            raise TypeError(
                "publication_history must be an EquitySyntheticPolicyPublicationHistory or None."
            )
        self._records = records
        self._catalog = catalog
        self._history = publication_history

    def execute(self, generation_keys: Iterable[str]) -> EquityRecommendationPolicyBindingAudit:
        """Audit the stored dispositions named by ``generation_keys``; write nothing."""
        keys = tuple(generation_keys)
        if not all(isinstance(key, str) for key in keys):
            raise TypeError("generation_keys must be text.")
        audited: dict[str, EquityRecommendationPolicyBindingRecordAudit] = {}
        frozen: dict[str, tuple[_Binding, ...]] = {}
        missing: list[str] = []
        for key in dict.fromkeys(keys):
            loaded = self._load(key)
            if loaded is None:
                missing.append(key)
            elif isinstance(loaded, EquityRecommendationPolicyBindingRecordAudit):
                audited[key] = loaded
            else:
                frozen[key] = loaded
        recorded: dict[EquitySyntheticPolicyBindingIdentity, set[str]] = defaultdict(set)
        for bindings in frozen.values():
            for binding in bindings:
                recorded[binding.identity].add(_fingerprint(binding))
        for key, bindings in frozen.items():
            audited[key] = EquityRecommendationPolicyBindingRecordAudit(
                key,
                tuple(
                    self._finding(key, binding, reused=len(recorded[binding.identity]) > 1)
                    for binding in bindings
                ),
            )
        return EquityRecommendationPolicyBindingAudit(
            tuple(audited[key] for key in dict.fromkeys(keys) if key in audited), tuple(missing)
        )

    def _load(
        self, key: str
    ) -> tuple[_Binding, ...] | EquityRecommendationPolicyBindingRecordAudit | None:
        def failed(
            status: EquityRecommendationPolicyBindingStatus, detail: str
        ) -> EquityRecommendationPolicyBindingRecordAudit:
            return EquityRecommendationPolicyBindingRecordAudit(key, (), status, detail)

        try:
            disposition = self._records.load_disposition(key)
        except EquityRecommendationGenerationStoreUnavailableError as exc:
            return failed(_S.AUDIT_EVIDENCE_UNAVAILABLE, f"The record cannot be read: {exc}")
        except IncompatibleEquityRecommendationRecordError as exc:
            return failed(_S.INCOMPATIBLE_HISTORICAL_FORMAT, f"The record is unsupported: {exc}")
        except InvalidEquityRecommendationRecordError as exc:
            return failed(_S.INVALID_CANONICAL_EVIDENCE, f"The record does not verify: {exc}")
        if disposition is None:
            return None
        if not isinstance(disposition, EquityRecommendationGenerationDisposition) or (
            disposition.generation_key != key
        ):
            return failed(_S.INVALID_CANONICAL_EVIDENCE, "The store answered another record.")
        try:
            bundle = EquitySyntheticPolicyBundle.of(disposition.inputs.policy_set)
        except InvalidEquitySyntheticPolicyBindingError as exc:
            return failed(
                _S.INVALID_CANONICAL_EVIDENCE, f"The frozen policy set does not bind: {exc}"
            )
        return (*bundle.components, bundle.policy_set)

    def _finding(
        self, key: str, recorded: _Binding, *, reused: bool
    ) -> EquityRecommendationPolicyBindingFinding:
        identity = recorded.identity

        def finding(
            status: EquityRecommendationPolicyBindingStatus,
            evidence: EquityRecommendationPolicyBindingEvidence,
            detail: str,
            catalog: str | None = None,
        ) -> EquityRecommendationPolicyBindingFinding:
            return EquityRecommendationPolicyBindingFinding(
                identity, status, evidence, _fingerprint(recorded), catalog, detail
            )

        if reused:
            return finding(
                _S.HISTORICAL_POLICY_VERSION_CONFLICT,
                _E.OTHER_STORED_RECORDS,
                "Another audited record holds other content under this label.",
            )
        load = (
            self._catalog.load_policy_set
            if isinstance(recorded, EquitySyntheticPolicySetBinding)
            else self._catalog.load_definition
        )
        try:
            published = load(identity)
        except EquitySyntheticPolicyCatalogUnavailableError as exc:
            return finding(_S.AUDIT_EVIDENCE_UNAVAILABLE, _E.CATALOG, f"Catalog unavailable: {exc}")
        except IncompatibleEquitySyntheticPolicyBindingError as exc:
            return finding(_S.INCOMPATIBLE_HISTORICAL_FORMAT, _E.CATALOG, str(exc))
        except InvalidEquitySyntheticPolicyBindingError as exc:
            return finding(_S.INVALID_CANONICAL_EVIDENCE, _E.CATALOG, str(exc))
        if published is None:
            return finding(_S.BINDING_NOT_ESTABLISHED, _E.CATALOG, "The label is not published.")
        if published != recorded:
            return finding(
                _S.HISTORICAL_POLICY_VERSION_CONFLICT,
                _E.CATALOG,
                "The catalog binds this label to other content.",
                _fingerprint(published),
            )
        catalog = _fingerprint(published)
        if self._history is None:
            return finding(
                _S.BINDING_NOT_ESTABLISHED,
                _E.CATALOG,
                "The label is published with this content today, but no historical evidence "
                "shows it was published when the record was made.",
                catalog,
            )
        try:
            proven = self._history.published_before(identity, catalog, key)
        except EquitySyntheticPolicyCatalogUnavailableError as exc:
            return finding(
                _S.AUDIT_EVIDENCE_UNAVAILABLE,
                _E.PUBLICATION_HISTORY,
                f"Publication history unavailable: {exc}",
                catalog,
            )
        if proven is not None and not isinstance(proven, bool):
            raise TypeError("published_before must return True, False or None.")
        if proven is True:
            return finding(
                _S.HISTORICALLY_VERIFIED_BINDING,
                _E.PUBLICATION_HISTORY,
                "Recorded evidence shows this binding was published before the record.",
                catalog,
            )
        return finding(
            _S.BINDING_NOT_ESTABLISHED,
            _E.PUBLICATION_HISTORY,
            "Recorded evidence does not show this binding was published before the record.",
            catalog,
        )
