"""Typed refusals of the EI-7 private read queries (EI-7C, EI7-CF2).

Every EI-7 read query classifies its failures itself (EI-7 R-2; EI7-CF2), so a
future HTTP layer maps one Application type and never catches a port,
Infrastructure or storage error. The pattern is the existing EI-6B
``EquityLifecycleRecordingRefusedError``: one refusal type with a stable
``reason`` and a derived ``retryable``.

The reasons are the approved semantic codes of the EI-7 Decision Register,
section 13.3. Their HTTP statuses belong to EI-7E and are not represented
here; this module imports no web framework.

- ISSUANCE_NOT_FOUND: the authority holds no issuance with the identity;
- WATERMARK_AHEAD: a requested watermark is above the recording position,
  including any value above the position maximum (EI7-P7);
- DISCOVERY_LIMIT_EXCEEDED: more recorded issuances than the approved limit
  (EI7-P2, EI7-FC1); never a corruption verdict;
- EVIDENCE_CLASS_UNSUPPORTED: a retrieved snapshot is not SYNTHETIC
  (EI7-EC1); its content is never exposed;
- UNSUPPORTED_SCOPE: a malformed or unsupported request value;
- PERSISTED_AUTHORITY_INVALID: corrupt or inconsistent required persisted
  authority, a failed lineage comparison, or an unverifiable primary research
  timeline or configuration (EI7-CF4, EI7-P5, EI7-FC3);
- LIFECYCLE_HISTORY_INVALID: a corrupt lifecycle history of a detail;
- DEPENDENCY_UNAVAILABLE: a required reader could not be read.

Only DEPENDENCY_UNAVAILABLE is retryable. ``EQUITY_NOT_CONFIGURED`` is a
runtime-configuration condition of EI-7E (EI7-CF5) and is deliberately not a
query refusal: a query is never constructed without its readers.

A refusal's message names what failed, never stored document content,
evidence payloads or request input beyond the reason. The underlying port
error is kept as ``__cause__`` for diagnostics only.
"""

from __future__ import annotations

from enum import StrEnum


class EquityReadQueryRefusal(StrEnum):
    """Why an EI-7 read query refused the whole response (EI7-CF2 codes)."""

    ISSUANCE_NOT_FOUND = "ISSUANCE_NOT_FOUND"
    WATERMARK_AHEAD = "WATERMARK_AHEAD"
    DISCOVERY_LIMIT_EXCEEDED = "DISCOVERY_LIMIT_EXCEEDED"
    EVIDENCE_CLASS_UNSUPPORTED = "EVIDENCE_CLASS_UNSUPPORTED"
    UNSUPPORTED_SCOPE = "UNSUPPORTED_SCOPE"
    PERSISTED_AUTHORITY_INVALID = "PERSISTED_AUTHORITY_INVALID"
    LIFECYCLE_HISTORY_INVALID = "LIFECYCLE_HISTORY_INVALID"
    DEPENDENCY_UNAVAILABLE = "DEPENDENCY_UNAVAILABLE"


_RETRYABLE = frozenset({EquityReadQueryRefusal.DEPENDENCY_UNAVAILABLE})


class EquityReadQueryRefusedError(Exception):
    """An EI-7 read query refused the whole response; ``reason`` says why.

    Nothing was written. ``retryable`` is True only for an outage of a
    required reader.
    """

    def __init__(self, reason: EquityReadQueryRefusal, message: str) -> None:
        if not isinstance(reason, EquityReadQueryRefusal):
            raise TypeError("reason must be an EquityReadQueryRefusal.")
        if not isinstance(message, str) or not message:
            raise ValueError("A refusal states what failed.")
        super().__init__(f"{reason.value}: {message}")
        self.reason = reason

    @property
    def retryable(self) -> bool:
        """Return whether the same request may succeed once an outage clears."""
        return self.reason in _RETRYABLE


def refuse_equity_read_query(
    reason: EquityReadQueryRefusal, message: str
) -> EquityReadQueryRefusedError:
    """Return a refusal; callers raise it, chaining the port error as its cause."""
    return EquityReadQueryRefusedError(reason, message)
