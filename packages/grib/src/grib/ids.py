"""Readable ids for cards and notes.

Format: ``YYYYMMDD_HHMMSS_<slug>`` (slug optional, <= 60 chars) — timestamp
gives free creation-order sort, slug gives scannability. Legacy uuid4 ids on
the existing board keep working: nothing here rewrites them, and every helper
treats uuid-format ids with the old [:8] display behaviour.

Charset is strictly lowercase alnum + underscore/hyphen so ids are safe in
URLs, filenames, and CLI args without quoting.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import UTC, datetime

MAX_ID_SLUG = 60

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

_READABLE_RE = re.compile(r"^\d{8}_\d{6}(_[0-9a-z-]+)?(-[0-9]+)?$")


def is_uuid_id(value: str) -> bool:
    return bool(_UUID_RE.match(value))


def is_readable_id(value: str) -> bool:
    return bool(_READABLE_RE.match(value))


def slugify(text: str, max_len: int = MAX_ID_SLUG) -> str:
    slug = text.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug[:max_len].strip("-")


def make_readable_id(title: str, taken: Iterable[str] = ()) -> str:
    """Build a readable id; on collision with ``taken`` append -2, -3, ..."""
    taken_set = set(taken)
    slug = slugify(title)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base = f"{ts}_{slug}" if slug else ts
    candidate = base
    n = 2
    while candidate in taken_set:
        candidate = f"{base}-{n}"
        n += 1
    return candidate


def display_id(value: str) -> str:
    """Short display form: legacy uuid ids keep their [:8] abbreviation,
    readable ids show in full (they are the copy-paste token for prefix
    lookups)."""
    if not value:
        return ""
    return value[:8] if is_uuid_id(value) else value
