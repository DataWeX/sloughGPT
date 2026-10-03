"""In-memory session search index — extracted from routers/inference.py.

Caches session file contents for fast full-text search; only re-reads
files whose mtime changed since the last build. Thread-safe for
concurrent reads; rebuilds are serialized.

Lives in domain so BOTH the legacy ``GET /chat/sessions/search``
endpoint and the system-search SessionsAdapter share ONE index —
instantiating two would double-cache the ~1.4k session files.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

# domain/search/session_index.py -> repo root is two levels up.
_REPO_ROOT = Path(__file__).resolve().parents[2]
# Historical storage location (unchanged by the extraction): the
# FileRepository-backed session store writes under apps/data.
_DEFAULT_DATA_DIRS = [
    _REPO_ROOT / "apps" / "data" / "chat_sessions",
    _REPO_ROOT / "apps" / "data" / "conversations",
]


class SessionSearchIndex:
    """Caches session file contents in memory for fast search.

    Only re-reads files whose mtime has changed since last index build.
    Thread-safe for concurrent reads; rebuilds are serialized.
    """

    def __init__(self, max_age_seconds: float = 5.0, data_dirs: list[Path] | None = None) -> None:
        self._max_age = max_age_seconds
        self._lock = threading.Lock()
        self._last_build: float = 0.0
        self._data_dirs = data_dirs if data_dirs is not None else list(_DEFAULT_DATA_DIRS)
        # sid → {name, messages, created_at, updated_at, file_path}
        self._entries: dict[str, dict[str, Any]] = {}

    def search(self, q: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search cached sessions. Rebuilds index if stale."""
        now = time.monotonic()
        if now - self._last_build > self._max_age:
            self._rebuild()
        return self._query(q.lower().strip(), limit)

    def _rebuild(self) -> None:
        with self._lock:
            # Double-check after acquiring lock
            if time.monotonic() - self._last_build <= self._max_age:
                return
            self._scan_files()
            self._last_build = time.monotonic()

    def _scan_files(self) -> None:
        search_dirs = self._data_dirs

        # Collect all current file mtimes
        current_files: dict[str, Path] = {}
        for sdir in search_dirs:
            if not sdir.is_dir():
                continue
            for f in sdir.glob("*.json"):
                sid = f.stem
                if sid not in current_files:
                    current_files[sid] = f

        # Remove entries for deleted files
        stale = set(self._entries.keys()) - set(current_files.keys())
        for sid in stale:
            del self._entries[sid]

        # Add or update entries
        for sid, fpath in current_files.items():
            try:
                mtime = fpath.stat().st_mtime
            except OSError:
                continue
            existing = self._entries.get(sid)
            if existing and existing.get("_mtime") == mtime:
                continue  # unchanged
            try:
                data = json.loads(fpath.read_text())
                self._entries[sid] = {
                    "id": data.get("id") or data.get("session_id") or sid,
                    "name": data.get("name", "") or "",
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "messages": data.get("messages", []),
                    "_mtime": mtime,
                }
            except (json.JSONDecodeError, OSError):
                continue

    def _query(self, q_lower: str, limit: int) -> list[dict[str, Any]]:
        if not q_lower:
            return []
        results: list[dict[str, Any]] = []
        max_matches_per_session = 3

        # Sort by updated_at descending for most-recent-first
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: e.get("updated_at", ""),
            reverse=True,
        )

        for entry in sorted_entries:
            if len(results) >= limit:
                break
            name = entry["name"]
            messages = entry["messages"]
            matches: list[dict[str, str]] = []

            if q_lower in name.lower():
                matches.append(
                    {
                        "role": "session",
                        "content": name,
                        "timestamp": entry.get("updated_at", ""),
                    }
                )

            for msg in messages:
                if len(matches) >= max_matches_per_session:
                    break
                content = msg.get("content", "")
                if q_lower in content.lower():
                    matches.append(
                        {
                            "role": msg.get("role", "unknown"),
                            "content": content,
                            "timestamp": msg.get("timestamp", ""),
                        }
                    )

            if matches:
                results.append(
                    {
                        "id": entry["id"],
                        "name": name or entry["id"],
                        "created_at": entry.get("created_at", ""),
                        "updated_at": entry.get("updated_at", ""),
                        "match_count": len(matches),
                        "matches": matches[:3],
                    }
                )

        return results


_index: SessionSearchIndex | None = None


def get_session_index() -> SessionSearchIndex:
    """Process-wide singleton — the legacy endpoint and the search
    adapter must share one cache, not build two."""
    global _index
    if _index is None:
        _index = SessionSearchIndex()
    return _index
