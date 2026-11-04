"""Application port classifying the provenance of a research window's references (EI-2B).

An EI-1C-3 research-window evaluation names its venue reference and listing
applicability reference as text. Text is not provenance: a reference string
can be renamed, so no classification may be inferred from how a reference is
spelled. ``EquityResearchReferenceProvenanceSource`` classifies a window's two
references from the reference sources that actually produced them, and

- returns an ``EquityResearchWindowProvenance`` naming exactly the window's
  two references, each either SYNTHETIC or ACCEPTED_PRODUCTION;
- raises UnknownEquityReferenceProvenanceError for any reference it cannot
  attribute to a known source -- an unknown source, a reference the source
  did not produce, or a missing reference. Unknown provenance is never
  classified, and never as production.

ACCEPTED_PRODUCTION states only that the reference came from Northstar's
accepted production reference data. It approves no candle, finality source or
trade; candle finality stays the EI-2A finality policy's.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from northstar_application.application_services.equity_research_window import (
        EquityResearchWindowEvaluation,
    )


class UnknownEquityReferenceProvenanceError(ValueError):
    """Raised when a reference cannot be attributed to a known reference source."""


class EquityReferenceProvenanceKind(StrEnum):
    """Where a reference came from."""

    SYNTHETIC = "SYNTHETIC"
    ACCEPTED_PRODUCTION = "ACCEPTED_PRODUCTION"


@dataclass(frozen=True, slots=True)
class EquityResearchWindowProvenance:
    """The provenance of one window's venue and applicability references."""

    venue_reference: str
    venue: EquityReferenceProvenanceKind
    applicability_reference: str
    applicability: EquityReferenceProvenanceKind

    def __post_init__(self) -> None:
        for name in ("venue_reference", "applicability_reference"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise UnknownEquityReferenceProvenanceError(f"{name} must be a non-empty string.")
        for name in ("venue", "applicability"):
            if not isinstance(getattr(self, name), EquityReferenceProvenanceKind):
                raise UnknownEquityReferenceProvenanceError(
                    f"{name} provenance must be an EquityReferenceProvenanceKind."
                )

    @property
    def is_accepted_production(self) -> bool:
        """Return whether both references are accepted production reference data."""
        production = EquityReferenceProvenanceKind.ACCEPTED_PRODUCTION
        return self.venue is production and self.applicability is production


class EquityResearchReferenceProvenanceSource(ABC):
    """Classifies a research window's references from the sources that produced them."""

    @abstractmethod
    def classify(self, window: EquityResearchWindowEvaluation) -> EquityResearchWindowProvenance:
        """Return the provenance of ``window``'s references, or raise if any is unknown."""
