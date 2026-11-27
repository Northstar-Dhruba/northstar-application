"""Read-only dependency probe of the EI-7 Equity read path (EI-7E P1, EI7E-D02).

``EquityReadDependencyProbe`` is the Application-owned port of the EI-7
Decision Register, section 15.3 (EI7E-D02, approved 2026-11-28), behind the
future ``GET /equity/health``. It answers one narrow question: can the
configured Equity read dependency be read safely, under the approved probe?

Three outcomes, never confused with one another
-----------------------------------------------
- **Readable.** ``probe()`` returns ``EquityReadDependencyProbeResult`` with
  status ``READABLE``: the approved narrow probe completed. Nothing more is
  claimed (EI7-SI2): not whole-database integrity, not the absence of
  corruption on pages the probe never read, not complete or verified
  recommendation, issuance, lifecycle or research history, not market data,
  a broker connection, an execution, a realized result or operational
  readiness. The result carries no path, time, version, table name, count or
  diagnostic payload.
- **Dependency unavailable.** ``probe()`` raises
  ``EquityReadDependencyUnavailableError``: the dependency could not be read
  now (for example a missing file, a lock, a hot journal, a refused
  journal mode or missing required tables, as an implementation detects
  them). It is retryable, and it says nothing about any stored record:
  never that the store is empty, never that a record exists or is missing.
- **Permanent storage integrity.** ``probe()`` raises the SI-A contract
  ``EquityReadStorageIntegrityError`` (CORRUPT or UNUSABLE) unchanged: never
  retryable, never wrapped, never reported as unavailable (EI7-SI1).

Which storage condition is which is decided by the implementation (EI-7E
P2, Infrastructure), never here; nothing in the Application imports a
database driver, holds a path or runs a query.

Boundaries
----------
- The port is observational. ``probe()`` takes no argument: no identity,
  session, date, watermark, generation key, path, connection, SQL or clock.
  An implementation receives its dependency through composition.
- It is separate from every existing reader port and depends on no
  issuance, generation, lifecycle, research or evidence authority, writer,
  recorder, monitor or scheduler.
- Not configured is not a probe outcome: a missing Equity setting is a
  runtime-configuration condition of the API (EQUITY_NOT_CONFIGURED,
  EI7-CF5-D), which never constructs a probe without a configured
  dependency.
- No public refusal reason is added. The API maps the outcomes to the
  approved vocabulary (EI-7E E1): READABLE to a successful health response,
  unavailability to DEPENDENCY_UNAVAILABLE, the SI-A contract to
  PERSISTED_AUTHORITY_INVALID.

Message safety
--------------
``EquityReadDependencyUnavailableError`` takes no free text: its message is
one fixed sentence. An implementation keeps the underlying exception as
``__cause__`` (``raise EquityReadDependencyUnavailableError() from exc``)
for internal diagnosis only; it is never part of the public message.

Nothing here writes, reads a clock or the environment, or calls out.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

_UNAVAILABLE_MESSAGE = "The Equity read dependency is unavailable; nothing was read."


class EquityReadDependencyProbeOutcome(StrEnum):
    """The one approved successful probe status (EI7E-D02)."""

    READABLE = "READABLE"


@dataclass(frozen=True, slots=True)
class EquityReadDependencyProbeResult:
    """A successful probe: the dependency was read safely by the narrow probe, nothing more.

    ``status`` is always ``EquityReadDependencyProbeOutcome.READABLE``; the result
    holds nothing else.
    """

    status: EquityReadDependencyProbeOutcome

    def __post_init__(self) -> None:
        if (
            not isinstance(self.status, EquityReadDependencyProbeOutcome)
            or self.status is not EquityReadDependencyProbeOutcome.READABLE
        ):
            raise ValueError("A probe result states the approved READABLE status only.")


class EquityReadDependencyUnavailableError(RuntimeError):
    """The Equity read dependency could not be read now; retryable (EI7E-D02).

    It names no record, path or engine message, and asserts nothing about
    stored data. It is not the SI-A permanent storage-integrity error.
    """

    def __init__(self) -> None:
        super().__init__(_UNAVAILABLE_MESSAGE)

    @property
    def retryable(self) -> bool:
        """Always True: an unavailable dependency may become readable again."""
        return True


class EquityReadDependencyProbe(ABC):
    """Observe whether the configured Equity read dependency is safely readable."""

    @abstractmethod
    def probe(self) -> EquityReadDependencyProbeResult:
        """Return READABLE, or raise as the module docstring states; never write or repair.

        Raises EquityReadDependencyUnavailableError when the dependency cannot
        be read now, and EquityReadStorageIntegrityError (SI-A) for a
        permanent storage-integrity failure.
        """
