"""Canonical UTC timestamp helpers for SloughGPT.

Every timestamp written to a file, a checkpoint, or an API payload must
round-trip through JavaScript's ``new Date(...)`` — i.e. be RFC 3339
compliant.  Two historical failure modes are guarded against here:

* ``datetime.isoformat() + "Z"`` produces ``2026-09-24T09:47:33.835728+00:00Z``
  (a numeric offset *and* a ``Z`` designator) which ``new Date`` rejects as
  ``Invalid Date``.
* naive datetimes (no offset) are interpreted as *local* time by JavaScript
  and by ``datetime.fromisoformat`` consumers, silently shifting the value.

Readers must go through :func:`parse_iso`/:func:`normalize_iso` rather than
:func:`datetime.fromisoformat` directly, so legacy values already on disk are
repaired instead of surfacing as ``Invalid Date`` in the UI.

Example:
    >>> from domain.shared import utc_now_iso, normalize_iso
    >>> utc_now_iso().endswith("Z") and "+" not in utc_now_iso()
    True
    >>> normalize_iso("2026-09-24T09:47:33.835728+00:00Z")
    '2026-09-24T09:47:33.835728Z'
"""

from __future__ import annotations

import re
from datetime import UTC, datetime

__all__ = ["is_valid_iso", "normalize_iso", "parse_iso", "to_iso", "utc_now_iso"]

# "...+00:00Z" / "...+02:00Z" — an offset immediately followed by a Z designator.
_DUPLICATE_Z = re.compile(r"(?P<offset>[+-]\d{2}:?\d{2})Z$")


def _emit(dt: datetime) -> str:
    """Serialize an aware datetime as UTC ``...T...Z`` (no numeric offset)."""
    return dt.astimezone(UTC).isoformat(timespec="microseconds").removesuffix("+00:00") + "Z"


def utc_now_iso() -> str:
    """Current UTC time as RFC 3339 with a ``Z`` suffix.

    Always microseconds precision, always UTC, always JS-parseable:
    ``2026-09-24T09:47:33.835728Z``.
    """
    return _emit(datetime.now(UTC))


def to_iso(value: datetime) -> str:
    """Serialize *value* as an RFC 3339 ``Z`` string in UTC.

    Naive datetimes are assumed to already be UTC (documented assumption —
    SloughGPT records everything in UTC).
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return _emit(value.astimezone(UTC))


def parse_iso(value: object) -> datetime | None:
    """Parse an ISO 8601 timestamp, tolerating legacy malformed input.

    Handles the ``+00:00Z`` bug, a bare ``Z``, naive strings (assumed UTC),
    and epoch-free garbage. Returns ``None`` when *value* is not a string or
    not parseable — never raises.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    text = _DUPLICATE_Z.sub(r"\g<offset>", text)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def normalize_iso(value: object) -> str:
    """Return a JS-safe RFC 3339 ``Z`` string, or ``""`` if unparseable.

    Use this on the read path for any timestamp that came from a file, a
    database row, or an older release — it repairs ``+00:00Z`` and drops
    values that would render as ``Invalid Date``.
    """
    parsed = parse_iso(value)
    return _emit(parsed) if parsed is not None else ""


def is_valid_iso(value: object) -> bool:
    """True when *value* parses as an ISO 8601 timestamp."""
    return parse_iso(value) is not None
