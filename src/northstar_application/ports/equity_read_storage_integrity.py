"""Permanent read-storage integrity failure of the EI-7 read path (SI-A, EI7-SI1).

``EquityReadStorageIntegrityError`` is the Application-owned contract of the
EI-7 Decision Register, section 14.4 (EI7-SI1, approved 2026-11-28): a
**permanent** failure of the storage read boundary itself -- the storage
engine reports that the persisted store cannot be read with integrity, or is
not a usable store at all. A read-only reader raises it in place of its
family's unavailability error for exactly those conditions; every other
failure keeps its existing typed error.

What it says, and what it does not
----------------------------------
It describes the read boundary, never a record. It does not assert that the
requested issuance, lifecycle, generation record, timeline or snapshot exists,
is missing or is corrupt: when the storage cannot be read, none of that is
established. It is therefore not one of the decoded-authority integrity
errors (the issuance LEDGER, DECLARATION and ISSUANCE categories; the
lifecycle EVENT, HISTORY and WINDOW categories; the generation-record,
timeline and snapshot integrity errors), none of which is changed, and it is
not an outage: it never clears by retrying.

``condition`` restates the two approved classifications (section 14.4):

- ``CORRUPT``: a permanent storage-integrity failure (SQLite's
  ``SQLITE_CORRUPT`` family);
- ``UNUSABLE``: the configured storage target is unusable without
  intervention (SQLite's ``SQLITE_NOTADB``). It may be a misconfigured path
  rather than damaged storage; the classification does not diagnose which.

Which engine condition is which is decided by the Infrastructure reader from
result codes (SI-B), never here; nothing in the Application imports a
database driver.

Inheritance
-----------
It derives from ``Exception`` directly -- not from any reader-family
unavailability error (all ``RuntimeError``) and not from any record integrity
error (all ``ValueError``) -- so no existing handler of either can receive it
and present it as retryable or as a record verdict. The EI-7C read queries
map it, at their query boundary, to the existing refusal
``PERSISTED_AUTHORITY_INVALID``: not retryable, with a fixed message (EI-7C
``equity_read_query_errors``). No public error code is added.

Message safety
--------------
The constructor takes no free text. ``str(error)`` is a fixed sentence naming
only the condition: never a database path, SQL text, driver message, stored
payload or credential. A reader keeps the engine's own exception as
``__cause__`` (``raise EquityReadStorageIntegrityError(...) from exc``) for
internal diagnosis only; no query refusal repeats it.

Nothing here imports a database, reads a clock or calls out.
"""

from __future__ import annotations

from enum import StrEnum


class EquityReadStorageIntegrityCondition(StrEnum):
    """The two approved permanent storage classifications (EI7-SI1)."""

    CORRUPT = "CORRUPT"
    UNUSABLE = "UNUSABLE"


_MESSAGES = {
    EquityReadStorageIntegrityCondition.CORRUPT: (
        "The persisted read storage could not be read with integrity; nothing was read."
    ),
    EquityReadStorageIntegrityCondition.UNUSABLE: (
        "The configured read storage is not a usable store; nothing was read."
    ),
}


class EquityReadStorageIntegrityError(Exception):
    """A permanent failure of the storage read boundary (EI7-SI1); never retryable.

    See the module docstring. It names no record, path or engine message.
    """

    def __init__(self, condition: EquityReadStorageIntegrityCondition) -> None:
        if not isinstance(condition, EquityReadStorageIntegrityCondition):
            raise TypeError("condition must be an EquityReadStorageIntegrityCondition.")
        super().__init__(_MESSAGES[condition])
        self.condition = condition

    @property
    def retryable(self) -> bool:
        """Always False: a corrupt or unusable store never clears by retrying."""
        return False
