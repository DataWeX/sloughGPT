"""Training data feed endpoints — serve live corpus records over HTTP.

The model's API conversations are appended continuously to a JSONL corpus
(``ConversationLogger`` → ``data/api_conversations/corpus.jsonl``). These
endpoints expose that stream as a paginated JSON feed with an offset cursor so
training consumers can point a URL at the flow and read only records written
since their last read position — a subscription feed, not a dataset snapshot.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from infrastructure.auth import require_auth_if_enabled
from schemas.common import raise_error, success_response

from domain.shared import find_repo_root

logger = logging.getLogger("slo")

router = APIRouter(tags=["training-feed"])

_MAX_RECORDS = 5000

# Registered feed sources: name -> corpus path relative to the repo root.
# Covers the live API-log corpus and feedback corrections; extend when new
# corpus.jsonl stores are wired into training.
_SOURCES: dict[str, str] = {
    "api-conversations": "data/api_conversations/corpus.jsonl",
    "feedback": "data/feedback.jsonl",
    "response-logs": "data/response_logs",
}


def list_feed_sources() -> list[dict[str, Any]]:
    """Summarize every registered feed source (name, total records, last write)."""
    sources: list[dict[str, Any]] = []
    for name in sorted(_SOURCES):
        path = _feed_path(name)
        try:
            lines = _source_lines(name)
            total = len(lines)
        except OSError:
            lines = []
            total = 0
        sources.append(
            {
                "name": name,
                "source": name,
                "total": total,
                "path": str(path),
                "last_updated": _iso_mtime(_source_paths(name)),
            }
        )
    return sources


def _iso_mtime(paths: list[Path]) -> str | None:
    """ISO-8601 modification time of the newest path, or None when unavailable."""
    try:
        from datetime import UTC, datetime

        mtimes = [path.stat().st_mtime for path in paths if path.exists()]
        if not mtimes:
            return None
        return datetime.fromtimestamp(max(mtimes), tz=UTC).isoformat()
    except OSError:
        return None


def _feed_path(name: str) -> Path:
    """Resolve a feed source name to its corpus path.

    Absolute values (used by tests) resolve directly; relative values resolve
    under the repo root. Unknown names are caught by callers via ``_SOURCES``.
    """
    value = _SOURCES[name]
    path = Path(value)
    if path.is_absolute():
        return path
    return find_repo_root(Path(__file__).resolve()) / path


def _source_paths(name: str) -> list[Path]:
    """Concrete files backing a feed source.

    A source may be a single JSONL file (``api-conversations``) or a directory
    of daily shards (``response-logs``). Directory shards are ordered by name so
    offsets advance chronologically across shards.
    """
    path = _feed_path(name)
    if path.is_dir():
        return sorted(path.glob("*.jsonl"))
    return [path]


def _source_lines(name: str) -> list[str]:
    """All raw corpus lines for a feed source (file or directory of shards)."""
    lines: list[str] = []
    for path in _source_paths(name):
        if not path.exists():
            continue
        lines.extend(path.read_text(encoding="utf-8").splitlines())
    return lines


def _read_feed(name: str, offset: int, limit: int) -> dict[str, Any]:
    """Slice one feed page and return the response payload.

    Args:
        name: feed source name (must exist in ``_SOURCES``).
        offset: starting raw line offset (0-based).
        limit: max records to return; 0 returns metadata only.

    Returns:
        dict with name/source/offset/next_offset/total/records.
    """
    try:
        lines = _source_lines(name)
    except OSError as exc:
        raise_error(f"Feed source unreadable: {exc}", "E_DOMAIN", status_code=500)
    total = len(lines)
    if limit and limit > 0:
        taken = lines[offset : offset + limit]
    else:
        taken = []
    records: list[dict[str, Any]] = []
    for line in taken:
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    next_offset = min(total, offset + (len(taken) if limit and limit > 0 else 0))
    return {
        "name": name,
        "source": name,
        "offset": offset,
        "next_offset": next_offset,
        "total": total,
        "records": records,
    }


@router.get("/training/feed")
async def training_feed(
    source: str = "api-conversations",
    offset: int = 0,
    limit: int = 500,
    auth_user: dict | None = Depends(require_auth_if_enabled),
):
    """Read records from a training feed past an offset cursor.

    Args:
        source: feed source name (default ``api-conversations`` = live API logs).
        offset: 0-based record offset to start from.
        limit: max records to return (0 = metadata only; cap ``_MAX_RECORDS``).

    Returns:
        StandardResponse with name/source/offset/next_offset/total/records.
    """
    if source not in _SOURCES:
        raise_error(f"Unknown feed source: {source}", "E_BAD_REQUEST", status_code=400)
    if offset < 0:
        raise_error("offset must be >= 0", "E_BAD_REQUEST", status_code=400)
    if limit < 0 or limit > _MAX_RECORDS:
        raise_error(f"limit must be between 0 and {_MAX_RECORDS}", "E_BAD_REQUEST", status_code=400)
    return success_response(data=_read_feed(source, offset, limit))


@router.get("/training/feeds")
async def training_feeds(
    auth_user: dict | None = Depends(require_auth_if_enabled),
):
    """List available training feed sources with record totals."""
    return success_response(data={"feeds": list_feed_sources(), "total": len(_SOURCES)})
