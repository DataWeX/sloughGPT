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

    def _lines(self, path: Path) -> list[str]:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return []
        cached = self._cache.get(path)
        if cached and cached[0] == mtime:
            return cached[1]
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as e:
            logger.debug("search cannot read %s: %s", path, e)
            return []
        self._cache[path] = (mtime, lines)
        return lines

    async def search(self, q: str, limit: int, ctx: SearchContext) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for path, root in self._iter_files():
            rel = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
            lines = self._lines(path)
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
                    )
                )
                break  # one hit per file — first matching line
            if len(hits) >= limit:
                break
        return hits


class JsonlRecordAdapter:
    """Search JSONL records by declared fields (never raw JSON text —
    punctuation and keys must not be searchable)."""

    capability = "indexed"
    workspace_scoped = False

    def __init__(self, store: str, path: Path, title_field: str, detail_field: str, id_field: str):
        self.store = store
        self._path = path
        self._title_field = title_field
        self._detail_field = detail_field
        self._id_field = id_field
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
                )
            )
            if len(hits) >= limit:
                break
        return hits


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
        ),
    ]
