"""Application ports for generation audits of attempts that recorded nothing new.

A RECORDED audit is committed only by FuturesTradeRecommendationStore.record,
in the transaction that inserts its recommendations (I-2). Every other audit --
EXISTING_RETURNED, REJECTED or BLOCKED -- describes an attempt that inserted no
recommendation, and is recorded on its own here, in its own transaction:

- an audit whose identity is not stored is inserted, and 1 is returned;
- the same identity already stored with equal content is an idempotent no-op,
  and 0 is returned;
- the same identity with different content raises
  FuturesRecommendationGenerationAuditConflictError and stores nothing;
- a RECORDED audit is refused with
  InvalidFuturesRecommendationGenerationAuditRecordingError, so no standalone
  write can claim a recommendation was newly recorded;
- every identity an audit names as ``existing`` must already be recorded, for
  the audited contract, strategy, policy and profile; otherwise the audit is
  refused with InvalidFuturesRecommendationGenerationAuditRecordingError.
  ``proposed`` identities need not exist: a lost race proposed one that was
  never recorded under this attempt.

Nothing is registered: a blocked attempt may have run under a configuration
that was never recorded, and its audit does not register one.

The repository reads an audit back exactly as it was recorded, whichever
operation recorded it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from northstar_application.application_services.futures_recommendation_generation_audit import (  # noqa: E501
        FuturesRecommendationGenerationAudit,
    )


class FuturesRecommendationGenerationAuditConflictError(ValueError):
    """Raised when an audit identity is already recorded with different content."""


class InvalidFuturesRecommendationGenerationAuditRecordingError(ValueError):
    """Raised when an audit cannot be recorded on its own."""


def validate_standalone_futures_recommendation_generation_audit(
    audit: FuturesRecommendationGenerationAudit,
) -> None:
    """Refuse an audit that is not a standalone (non-RECORDED) generation audit.

    Every audit store implementation calls this before touching storage.
    """
    from northstar_application.application_services.futures_recommendation_generation_audit import (  # noqa: E501
        FuturesRecommendationGenerationAudit,
        FuturesRecommendationGenerationAuditOutcome,
    )

    if not isinstance(audit, FuturesRecommendationGenerationAudit):
        raise InvalidFuturesRecommendationGenerationAuditRecordingError(
            "A standalone audit must be a FuturesRecommendationGenerationAudit."
        )
    if audit.outcome is FuturesRecommendationGenerationAuditOutcome.RECORDED:
        raise InvalidFuturesRecommendationGenerationAuditRecordingError(
            "A RECORDED audit is committed only with its recommendations, never on its own."
        )


class FuturesRecommendationGenerationAuditStore(ABC):
    """Records audits of generation attempts that inserted no recommendation.

    Implementations must call
    validate_standalone_futures_recommendation_generation_audit first, then
    honour the semantics in this module's documentation. The abstract contract
    documents these obligations but cannot enforce them at runtime.
    """

    @abstractmethod
    def record_audit(self, audit: FuturesRecommendationGenerationAudit) -> int:
        """Record a standalone audit; return 1 when inserted, 0 when already recorded."""


class FuturesRecommendationGenerationAuditRepository(ABC):
    """Reads recorded generation audits."""

    @abstractmethod
    def get_audit(self, identity: str) -> FuturesRecommendationGenerationAudit | None:
        """Return the audit recorded under ``identity``, if any."""
