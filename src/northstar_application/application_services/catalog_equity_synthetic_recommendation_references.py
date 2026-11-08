"""Catalog-backed synthetic recommendation references (EI4-D10, ADR-022 section 7).

CatalogEquitySyntheticRecommendationReferences implements the existing
EquitySyntheticRecommendationReferenceSource port over the immutable policy
catalog. A requested version resolves only to a definition published under
exactly ``(EQUITY_SYNTHETIC_POLICY_NAMESPACE, policy type, version)``:

- not published: None. The port's "not held" answer, never a default;
- the catalog cannot answer: EquitySyntheticReferenceUnavailableError, the
  port's existing outage error;
- the stored binding does not verify, or the catalog answers another
  identity: EquitySyntheticPolicyReferenceConflictError, a permanent integrity
  conflict (reason ``POLICY_REFERENCE_CONFLICT``; ``incompatible`` for an
  unsupported binding format). It is never retried or masked by a fallback.

A read never publishes, and no label is changed. The resolver keeps the
approved namespace fixed: callers cannot choose another.

Price increments
----------------
``price_increment(version, listing, trading_date)`` returns the one amount the
label is published with, for every listing and date (EI4-D10-DQ3). No
listing-specific or NSE tick-size rule is applied or inferred.

Scope
-----
This is an additional implementation of the existing port. The EI-4C generator
is not rewired to require it, and it does not yet map
EquitySyntheticPolicyReferenceConflictError to a generation reason: both are
the S3 slice. Until then, a conflict raised here propagates out of generation,
and nothing is produced.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from northstar_core.domain.value_objects import ListingReference
from northstar_core.recommendations.equity import (
    EquityPriceIncrement,
    EquitySyntheticCostSchedule,
    EquitySyntheticGenerationPolicy,
    EquitySyntheticRiskPolicy,
    EquitySyntheticVersion,
)

from northstar_application.application_services.equity_synthetic_policy_bindings import (
    EQUITY_SYNTHETIC_POLICY_NAMESPACE,
    EquitySyntheticPolicyBindingIdentity,
    EquitySyntheticPolicyDefinitionBinding,
    EquitySyntheticPolicyType,
    IncompatibleEquitySyntheticPolicyBindingError,
    InvalidEquitySyntheticPolicyBindingError,
)
from northstar_application.ports.equity_recommendation_references import (
    EquitySyntheticEntryTimingPolicy,
    EquitySyntheticPolicyReferenceConflictError,
    EquitySyntheticRecommendationReferenceSource,
    EquitySyntheticReferenceUnavailableError,
)
from northstar_application.ports.equity_synthetic_policy_catalog_store import (
    EquitySyntheticPolicyCatalogStore,
    EquitySyntheticPolicyCatalogUnavailableError,
)

_T = EquitySyntheticPolicyType


class CatalogEquitySyntheticRecommendationReferences(EquitySyntheticRecommendationReferenceSource):
    """Resolves synthetic references only to published, verified catalog bindings."""

    def __init__(self, catalog: EquitySyntheticPolicyCatalogStore) -> None:
        if not isinstance(catalog, EquitySyntheticPolicyCatalogStore):
            raise TypeError("catalog must be an EquitySyntheticPolicyCatalogStore.")
        self._catalog = catalog

    @property
    def namespace(self) -> str:
        """Return the one namespace this resolver reads: the approved one."""
        return EQUITY_SYNTHETIC_POLICY_NAMESPACE

    def _resolve(self, policy_type: EquitySyntheticPolicyType, version: object) -> Any:
        if not isinstance(version, EquitySyntheticVersion):
            raise TypeError("version must be an EquitySyntheticVersion.")
        identity = EquitySyntheticPolicyBindingIdentity(
            EQUITY_SYNTHETIC_POLICY_NAMESPACE, policy_type, version
        )
        try:
            binding = self._catalog.load_definition(identity)
        except EquitySyntheticPolicyCatalogUnavailableError as exc:
            raise EquitySyntheticReferenceUnavailableError(
                f"The policy catalog cannot answer {policy_type.value} {version}: {exc}"
            ) from exc
        except IncompatibleEquitySyntheticPolicyBindingError as exc:
            raise EquitySyntheticPolicyReferenceConflictError(
                f"The published {policy_type.value} {version} uses an unsupported format: {exc}",
                incompatible=True,
            ) from exc
        except InvalidEquitySyntheticPolicyBindingError as exc:
            raise EquitySyntheticPolicyReferenceConflictError(
                f"The published {policy_type.value} {version} does not verify: {exc}"
            ) from exc
        if binding is None:
            return None
        if not isinstance(binding, EquitySyntheticPolicyDefinitionBinding) or (
            binding.identity != identity
        ):
            raise EquitySyntheticPolicyReferenceConflictError(
                f"The catalog answered another binding for {policy_type.value} {version}."
            )
        return binding.definition

    def generation_policy(
        self, version: EquitySyntheticVersion
    ) -> EquitySyntheticGenerationPolicy | None:
        """Return the published generation policy ``version``, or None."""
        return self._resolve(_T.GENERATION_POLICY, version)  # type: ignore[no-any-return]

    def risk_policy(self, version: EquitySyntheticVersion) -> EquitySyntheticRiskPolicy | None:
        """Return the published risk policy ``version``, or None."""
        return self._resolve(_T.RISK_POLICY, version)  # type: ignore[no-any-return]

    def cost_schedule(self, version: EquitySyntheticVersion) -> EquitySyntheticCostSchedule | None:
        """Return the published cost schedule ``version``, or None."""
        return self._resolve(_T.COST_SCHEDULE, version)  # type: ignore[no-any-return]

    def price_increment(
        self, version: EquitySyntheticVersion, listing: ListingReference, trading_date: date
    ) -> EquityPriceIncrement | None:
        """Return the one amount published under ``version``, for every listing and date."""
        if not isinstance(listing, ListingReference):
            raise TypeError("listing must be a ListingReference.")
        return self._resolve(_T.PRICE_INCREMENT, version)  # type: ignore[no-any-return]

    def timing_policy(
        self, version: EquitySyntheticVersion
    ) -> EquitySyntheticEntryTimingPolicy | None:
        """Return the published entry-timing policy ``version``, or None."""
        return self._resolve(_T.TIMING_POLICY, version)  # type: ignore[no-any-return]
