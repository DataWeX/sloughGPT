"""File-backed store adapters — docs, dev notes, kanban cards.

This is the "development environment" half of system-wide search: the
repository's own knowledge, not tenant data. Files are read once and
cached by mtime (``capability = "indexed"``), so a keystroke never
rescans the tree.

Locators are ``file:path:line`` — the palette and page currently show
route: hits only (no file viewer exists yet); file hits are served by
GET /search for API consumers until a viewer lands.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from domain.search.matching import substring_score
from domain.search.types import SearchContext, SearchHit

logger = logging.getLogger(__name__)

_DETAIL_MAX = 160

# domain/search/adapters/filesources.py -> repo root is four levels up.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_NOTES_DIR = Path.home() / ".config" / "dev-notes"
_BOARD_FILE = _REPO_ROOT / ".kanban" / "board.jsonl"


class TextFileAdapter:
    """Substring search over text files (one hit per file: first match)."""

    capability = "indexed"
    workspace_scoped = False

    def __init__(self, store: str, roots: list[Path], patterns: tuple[str, ...] = ("*.md",)):
        self.store = store
        self._roots = roots
        self._patterns = patterns
        # path -> (mtime, lines); refreshed on mtime change, never rescanned otherwise.
        self._cache: dict[Path, tuple[float, list[str]]] = {}

    def _iter_files(self):
        for root in self._roots:
            if not root.exists():
                continue  # absent corpus (e.g. prod image without docs/) = empty, not an error
            if root.is_file():
                yield root, root.parent
                continue
            for pattern in self._patterns:
                for path in sorted(root.rglob(pattern)):
                    if path.is_file():
                        yield path, root

    def _lines(self, path: Path) -> tuple[float, list[str]]:
        """(mtime, lines) for a file — refreshed on mtime change, cached otherwise.

        Returns ``(0.0, [])`` when the file is unreadable or gone, so the
        caller can treat absence as "nothing to match" without special-casing
        the error.
        """
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return 0.0, []
        cached = self._cache.get(path)
        if cached and cached[0] == mtime:
            return cached[0], cached[1]
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as e:
            logger.debug("search cannot read %s: %s", path, e)
            return 0.0, []
        self._cache[path] = (mtime, lines)
        return mtime, lines

    async def search(self, q: str, limit: int, ctx: SearchContext) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for path, root in self._iter_files():
            rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            mtime, lines = self._lines(path)
            if not lines:
                continue
            for lineno, line in enumerate(lines, start=1):
                score = substring_score(q, rel, line)
                if score is None:
                    continue
                hits.append(
                    SearchHit(
                        id=f"{rel}:{lineno}",
                        store=self.store,
                        title=rel,
                        detail=line.strip()[:_DETAIL_MAX],
                        score=score,
                        locator=f"file:{rel}:{lineno}",
                        updated_at=mtime,
                    )
                )
                break  # one hit per file — first matching line
            if len(hits) >= limit:
                break
        return hits

    def sweep(self) -> dict[str, int]:
        """Two-pass staleness sweep, newest-first then oldest-first.

        The cache only refreshes a file when a query touches it, so a doc
        nobody queries drifts behind the index and a deleted doc keeps a
        ghost entry forever. This closes both holes with two passes, front
        to back:

        - Pass 1 (front, newest -> oldest): walk the on-disk corpus in
          recency order and pull in files not yet cached — the latest
          uploads land in the index first, before the sweep touches
          anything older.
        - Pass 2 (back, oldest -> newest): walk the *cache* oldest-first
          and re-stat every entry — changed files are re-read (brought to
          the updated state), deleted files are evicted (no ghost
          entries). Oldest get their re-read first, so a sweep cut short
          still leaves the stalest docs corrected.

        Returns per-outcome counts: ``{"new": n, "refreshed": n,
        "unchanged": n, "evicted": n}``. Absent roots are skipped, not an
        error — a prod image without docs/ sweeps to zeroes.
        """
        stats = {"new": 0, "refreshed": 0, "unchanged": 0, "evicted": 0}

        # Pass 1 — front, newest first: absorb files that appeared since the
        # last sweep. Existing entries are left alone; pass 2 owns them.
        disk: list[tuple[float, Path]] = []
        for path, _root in self._iter_files():
            try:
                disk.append((path.stat().st_mtime, path))
            except OSError:
                continue
        for _mtime, path in sorted(disk, key=lambda t: -t[0]):
            if path in self._cache:
                continue
            mtime, _ = self._lines(path)
            if not mtime:
                continue  # vanished between walk and read — nothing to index
            stats["new"] += 1  # empty files count too: they exist now

        # Pass 2 — back, oldest first: bring every cached entry to the
        # source's current state (refresh changed, evict deleted).
        for path in sorted(self._cache, key=lambda p: self._cache[p][0]):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                del self._cache[path]  # sorted() materialised the keys first
                stats["evicted"] += 1
                continue
            if self._cache[path][0] == mtime:
                stats["unchanged"] += 1
            else:
                self._lines(path)  # re-read -> updated state
                stats["refreshed"] += 1
        return stats


class JsonlRecordAdapter:
    """Search JSONL records by declared fields (never raw JSON text —
    punctuation and keys must not be searchable)."""

    capability = "indexed"
    workspace_scoped = False

    def __init__(
        self,
        store: str,
        path: Path,
        title_field: str,
        detail_field: str,
        id_field: str,
        ts_field: str | None = None,
    ):
        self.store = store
        self._path = path
        self._title_field = title_field
        self._detail_field = detail_field
        self._id_field = id_field
        # Optional recency field (e.g. ``updated_at``); records without a
        # usable value fall back to 0.0 and lose only the recency tiebreak.
        self._ts_field = ts_field
        self._cache: tuple[float, list[tuple[int, dict]]] | None = None

    def _records(self) -> list[tuple[int, dict]]:
        try:
            mtime = self._path.stat().st_mtime
        except OSError:
            return []  # absent board = empty, not an error
        if self._cache and self._cache[0] == mtime:
            return self._cache[1]
        records: list[tuple[int, dict]] = []
        try:
            for lineno, line in enumerate(
                self._path.read_text(errors="replace").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue  # tolerate non-card lines
                if isinstance(obj, dict):
                    records.append((lineno, obj))
        except OSError as e:
            logger.debug("search cannot read %s: %s", self._path, e)
            return []
        self._cache = (mtime, records)
        return records

    async def search(self, q: str, limit: int, ctx: SearchContext) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for lineno, rec in self._records():
            title = str(rec.get(self._title_field) or "")
            detail = str(rec.get(self._detail_field) or "")
            score = substring_score(q, title, detail)
            if score is None:
                continue
            hits.append(
                SearchHit(
                    id=str(rec.get(self._id_field) or f"line-{lineno}"),
                    store=self.store,
                    title=title or detail[:_DETAIL_MAX],
                    detail=detail[:_DETAIL_MAX],
                    score=score,
                    locator=f"file:{self._path}:{lineno}",
                    updated_at=self._ts(rec),
                )
            )
            if len(hits) >= limit:
                break
        return hits

    def _ts(self, rec: dict) -> float:
        """Epoch seconds from the declared timestamp field, or 0.0.

        Accepts numeric seconds or an ISO-8601 string; anything unparseable
        degrades to 0.0 (no recency signal) rather than raising — a malformed
        timestamp must never break a search.
        """
        if not self._ts_field:
            return 0.0
        raw = rec.get(self._ts_field)
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str) and raw:
            try:
                from datetime import datetime

                dt = raw.replace("Z", "+00:00")
                return datetime.fromisoformat(dt).timestamp()
            except ValueError:
                return 0.0
        return 0.0


def default_file_adapters() -> list:
    """The built-in repository corpus: docs/, dev notes, kanban board."""
    return [
        TextFileAdapter("docs", [_REPO_ROOT / "docs"]),
        TextFileAdapter("dev_notes", [_NOTES_DIR]),
        JsonlRecordAdapter(
            "kanban_cards",
            _BOARD_FILE,
            title_field="title",
            detail_field="description",
            id_field="id",
            ts_field="updated_at",
        ),
    ]
