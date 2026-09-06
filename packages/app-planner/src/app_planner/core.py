#!/usr/bin/env python3
"""
notes — standalone development journal.

File-backed note-taking with YAML frontmatter. Each note is an individual
markdown file in ``~/.config/dev-notes/``.

Usage as a module::

    from notes import NoteStore
    store = NoteStore()
    note = store.create("Fix kernel boot", tags=["kernel", "bugfix"])
    store.search("kernel")

Usage as a CLI::

    notes new "Fix kernel boot" --tags kernel,bugfix --status done
    notes list
    notes search kernel
"""

from __future__ import annotations

import os
import re
import sys
import json
import logging
from datetime import datetime, timezone, date
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

from . import config

logger = logging.getLogger("dev-notes")

_MAX_TITLE_SLUG = 60


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Note:
    """A single development journal note."""

    id: str = ""
    title: str = ""
    created_at: str = ""
    updated_at: str = ""
    tags: list[str] = field(default_factory=list)
    status: str = "open"
    author: str = ""
    sprint: str = ""
    gh: str = ""
    assignee: str = ""
    body: str = ""

    @property
    def date_str(self) -> str:
        return self.created_at[:10] if self.created_at else ""

    @property
    def short_id(self) -> str:
        return self.id[:8] if self.id else ""

    @property
    def gh_url(self) -> str:
        """Full GitHub issue URL if gh is set."""
        if not self.gh:
            return ""
        if self.gh.startswith("http"):
            return self.gh
        if "/" in self.gh and "#" in self.gh:
            owner_repo, num = self.gh.rsplit("#", 1)
            return f"https://github.com/{owner_repo}/issues/{num}"
        if self.gh.startswith("#"):
            num = self.gh.lstrip("#")
            return f"https://github.com/issues/{num}"
        return ""

    def to_markdown(self) -> str:
        tags_str = ", ".join(self.tags) if self.tags else ""
        lines = [
            "---",
            f"title: {self.title}",
            f"created: {self.created_at}",
            f"updated: {self.updated_at}",
            f"tags: {tags_str}",
            f"status: {self.status}",
        ]
        if self.author:
            lines.append(f"author: {self.author}")
        if self.sprint:
            lines.append(f"sprint: {self.sprint}")
        if self.gh:
            lines.append(f"gh: {self.gh}")
        if self.assignee:
            lines.append(f"assignee: {self.assignee}")
        lines.extend(["---", "", self.body.rstrip(), ""])
        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, text: str, note_id: str = "") -> Note:
        lines = text.split("\n")
        meta: dict[str, str] = {}
        body_start = 0
        if lines and lines[0].strip() == "---":
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == "---":
                    body_start = i + 1
                    break
                if ":" in line:
                    key, _, val = line.partition(":")
                    meta[key.strip()] = val.strip()
        tags_raw = meta.get("tags", "")
        tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []
        body = "\n".join(lines[body_start:]).strip("\n").rstrip()
        return cls(
            id=note_id,
            title=meta.get("title", ""),
            created_at=meta.get("created", ""),
            updated_at=meta.get("updated", ""),
            tags=tags,
            status=meta.get("status", "open"),
            author=meta.get("author", ""),
            sprint=meta.get("sprint", ""),
            gh=meta.get("gh", ""),
            assignee=meta.get("assignee", ""),
            body=body,
        )


# ---------------------------------------------------------------------------
# Storage backends
# ---------------------------------------------------------------------------

class _FileBackend:
    def __init__(self, notes_dir: Path):
        self._dir = notes_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def all_notes(self) -> list[Note]:
        notes: list[Note] = []
        for path in sorted(self._dir.glob("*.md"), reverse=True):
            note = self._load(path)
            if note is not None:
                notes.append(note)
        return notes

    def put(self, note: Note) -> None:
        (self._dir / f"{note.id}.md").write_text(note.to_markdown())

    def delete_id(self, note_id: str) -> bool:
        path = self._dir / f"{note_id}.md"
        if path.exists():
            path.unlink()
            return True
        return False

    def find_by_prefix(self, prefix: str) -> list[str]:
        prefix_lower = prefix.lower()
        return [p.stem for p in self._dir.glob("*.md")
                if p.stem.lower().startswith(prefix_lower)]

    def rename_id(self, old_id: str, new_id: str) -> None:
        old_path = self._dir / f"{old_id}.md"
        new_path = self._dir / f"{new_id}.md"
        if old_path.exists():
            old_path.rename(new_path)

    def note_by_id(self, note_id: str) -> Note | None:
        path = self._dir / f"{note_id}.md"
        if not path.exists():
            return None
        try:
            text = path.read_text()
            return Note.from_markdown(text, note_id=path.stem)
        except Exception as e:
            logger.warning("Failed to load note %s: %s", path.name, e)
            return None

    def count(self) -> int:
        return sum(1 for _ in self._dir.glob("*.md"))

    def _load(self, path: Path) -> Note | None:
        try:
            text = path.read_text()
            return Note.from_markdown(text, note_id=path.stem)
        except Exception as e:
            logger.warning("Failed to load note %s: %s", path.name, e)
            return None


class _MogDBBackend:
    def __init__(self, notes_dir: Path):
        self._dir = notes_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._db = None
        self._col = None

    def _ensure(self):
        if self._db is None:
            from mogdb import MogDB
            self._db = MogDB(str(self._dir / "store"))
            self._col = self._db.collection("notes")
        return self._col

    @staticmethod
    def _doc_to_note(doc: dict) -> Note:
        tags_raw = doc.get("tags", "")
        if isinstance(tags_raw, str):
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        else:
            tags = list(tags_raw) if tags_raw else []
        return Note(
            id=doc.get("id", doc.get("_id", "")),
            title=doc.get("title", ""),
            created_at=doc.get("created_at", ""),
            updated_at=doc.get("updated_at", ""),
            tags=tags,
            status=doc.get("status", "open"),
            sprint=doc.get("sprint", ""),
            gh=doc.get("gh", ""),
            body=doc.get("body", ""),
        )

    @staticmethod
    def _note_to_doc(note: Note) -> dict:
        return {
            "id": note.id,
            "title": note.title,
            "created_at": note.created_at,
            "updated_at": note.updated_at,
            "tags": ",".join(note.tags),
            "status": note.status,
            "sprint": note.sprint,
            "gh": note.gh,
            "body": note.body,
        }

    def all_notes(self) -> list[Note]:
        col = self._ensure()
        docs = col.find(sort=[("created_at", -1)])
        return [self._doc_to_note(d) for d in docs]

    def put(self, note: Note) -> None:
        col = self._ensure()
        existing = col.find({"id": note.id}, limit=1)
        if existing:
            col.update_one({"id": note.id}, {"$set": self._note_to_doc(note)})
        else:
            col.insert_one(self._note_to_doc(note))

    def delete_id(self, note_id: str) -> bool:
        col = self._ensure()
        return col.delete_one({"id": note_id}) > 0

    def find_by_prefix(self, prefix: str) -> list[str]:
        col = self._ensure()
        prefix_lower = prefix.lower()
        ids: list[str] = []
        for doc in col.find():
            doc_id = doc.get("id", "")
            if doc_id.lower().startswith(prefix_lower):
                ids.append(doc_id)
        return ids

    def rename_id(self, old_id: str, new_id: str) -> None:
        col = self._ensure()
        col.update_one({"id": old_id}, {"$set": {"id": new_id}})

    def note_by_id(self, note_id: str) -> Note | None:
        col = self._ensure()
        doc = col.find_one({"id": note_id})
        if doc:
            return self._doc_to_note(doc)
        return None

    def count(self) -> int:
        col = self._ensure()
        return col.count()


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class NoteStore:
    """Note store with pluggable backends (``file`` or ``mogdb``)."""

    def __init__(self, notes_dir: Path | None = None, backend: str = "file"):
        self._dir = Path(notes_dir) if notes_dir is not None else config.default_notes_dir()
        self._backend = backend
        if backend == "mogdb":
            self._bk = _MogDBBackend(self._dir)
        else:
            self._bk = _FileBackend(self._dir)

    def create(self, title: str, tags: list[str] | None = None,
               status: str = "open", author: str = "", sprint: str = "", gh: str = "",
               assignee: str = "", body: str = "") -> Note:
        now = datetime.now(timezone.utc)
        slug = self._title_to_slug(title)
        ts = now.strftime("%Y%m%d_%H%M%S")
        note_id = f"{ts}_{slug}"
        note = Note(
            id=note_id,
            title=title,
            created_at=now.isoformat(),
            updated_at=now.isoformat(),
            tags=tags or [],
            status=status,
            author=author,
            sprint=sprint,
            gh=gh,
            assignee=assignee,
            body=body,
        )
        self._bk.put(note)
        logger.info("Created note: %s", note_id)
        return note

    def get(self, note_id: str) -> Note | None:
        """
        Resolve a note id prefix to a note.

        Exact and unambiguous prefixes return their note. An ambiguous prefix
        (e.g. a bare date like ``20260806`` matching several notes) resolves to
        the most recently updated match so commands like ``show`` work without
        forcing the caller to type the full id. Ambiguity is logged with the
        full match list.

        Args:
            note_id: full id or prefix.

        Returns:
            The matching note, or ``None`` if nothing matches.
        """
        matches = self._bk.find_by_prefix(note_id)
        if len(matches) == 1:
            return self._bk.note_by_id(matches[0])
        if len(matches) > 1:
            candidates: list[Note] = []
            for match_id in matches:
                note = self._bk.note_by_id(match_id)
                if note is not None:
                    candidates.append(note)
            if candidates:
                candidates.sort(
                    key=lambda n: n.updated_at or n.created_at or n.id,
                    reverse=True,
                )
                chosen = candidates[0]
                logger.warning(
                    "Ambiguous id '%s' (%d matches); using most recently updated: %s",
                    note_id, len(candidates), chosen.id,
                )
                return chosen
        return None

    def update(self, note_id: str, **kwargs: Any) -> Note | None:
        note = self.get(note_id)
        if note is None:
            return None
        old_id = note.id
        for key in ("title", "tags", "status", "author", "sprint", "gh", "assignee", "body"):
            if key in kwargs and kwargs[key] is not None:
                setattr(note, key, kwargs[key])
        note.updated_at = datetime.now(timezone.utc).isoformat()
        new_slug = self._title_to_slug(note.title)
        new_id = f"{note.created_at[:10].replace('-', '')}_{note.created_at[11:19].replace(':', '')}_{new_slug}"
        if new_id != old_id:
            self._bk.rename_id(old_id, new_id)
            note.id = new_id
        self._bk.put(note)
        return note

    def delete(self, note_id: str) -> bool:
        matches = self._bk.find_by_prefix(note_id)
        if len(matches) != 1:
            return False
        self._bk.delete_id(matches[0])
        logger.info("Deleted note: %s", matches[0])
        return True

    def list_notes(self, tag: str | None = None, status: str | None = None,
                   author: str | None = None, sprint: str | None = None, limit: int = 50,
                   today: bool = False) -> list[Note]:
        notes: list[Note] = []
        today_str = date.today().isoformat() if today else ""
        for note in self._bk.all_notes():
            if today and note.date_str != today_str:
                continue
            if tag and tag not in note.tags:
                continue
            if status and note.status != status:
                continue
            if author and note.author != author:
                continue
            if sprint and note.sprint != sprint:
                continue
            notes.append(note)
            if len(notes) >= limit:
                break
        return notes

    def search(self, query: str, limit: int = 20) -> list[Note]:
        q = query.lower()
        results: list[Note] = []
        for note in self._bk.all_notes():
            if (q in note.title.lower()
                    or q in " ".join(note.tags).lower()
                    or q in note.body.lower()
                    or q in (note.author or "").lower()):
                results.append(note)
                if len(results) >= limit:
                    break
        return results

    def today(self) -> list[Note]:
        today_str = date.today().isoformat()
        results: list[Note] = []
        for note in self._bk.all_notes():
            if note.date_str == today_str:
                results.append(note)
        return results

    def export_all(self, output_path: str | None = None) -> str:
        notes = self.list_notes(limit=9999)
        sections: list[str] = []
        for note in reversed(notes):
            sections.append(note.to_markdown())
            sections.append("\n---\n")
        content = "\n".join(sections)
        if output_path:
            Path(output_path).write_text(content)
            logger.info("Exported %d notes to %s", len(notes), output_path)
        return content

    def sprints(self) -> list[str]:
        seen: set[str] = set()
        for note in self._bk.all_notes():
            if note.sprint:
                seen.add(note.sprint)
        return sorted(seen)

    def sprint_report(self, sprint_name: str) -> str:
        """Generate a markdown sprint report with GitHub issue links."""
        notes = self.list_notes(sprint=sprint_name, limit=9999)
        if not notes:
            return f"No notes found for sprint '{sprint_name}'.\n"

        lines: list[str] = [
            f"# Sprint Report: {sprint_name}",
            f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
            f"**Notes:** {len(notes)}",
            "",
            "---",
            "",
        ]
        for i, note in enumerate(notes, 1):
            lines.append(f"## {i}. {note.title}")
            lines.append(f"**Status:** {note.status}  ")
            lines.append(f"**Date:** {note.date_str}  ")
            lines.append(f"**ID:** `{note.short_id}`  ")
            if note.tags:
                lines.append(f"**Tags:** {', '.join(note.tags)}  ")
            if note.gh:
                url = note.gh_url
                if url:
                    lines.append(f"**GitHub:** [{note.gh}]({url})  ")
                else:
                    lines.append(f"**GitHub:** {note.gh}  ")
            if note.body:
                lines.append("")
                lines.append(note.body.strip())
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def timeline(self, days: int = 7, tag: str | None = None,
                 status: str | None = None) -> list[tuple[str, list[Note]]]:
        """Return notes grouped by day for the last *days* days.

        Returns a list of ``(date_str, [notes])`` tuples, newest day first.
        Notes within each day are ordered newest first.
        """
        cutoff = date.today()
        groups: dict[str, list[Note]] = {}
        for note in self._bk.all_notes():
            nd = note.date_str
            if not nd:
                continue
            try:
                note_date = date.fromisoformat(nd)
            except ValueError:
                continue
            if (cutoff - note_date).days > days:
                continue
            if tag and tag not in note.tags:
                continue
            if status and note.status != status:
                continue
            groups.setdefault(nd, []).append(note)
        result = sorted(groups.items(), reverse=True)
        return result

    def count(self) -> int:
        return self._bk.count()

    @staticmethod
    def _title_to_slug(title: str) -> str:
        slug = title.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)
        slug = re.sub(r"-+", "-", slug)
        return slug[:_MAX_TITLE_SLUG].rstrip("-")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_store: NoteStore | None = None


def get_note_store(backend: str = "file") -> NoteStore:
    global _store
    if _store is None or _store._backend != backend:
        _store = NoteStore(backend=backend)
    return _store


def reset_note_store() -> None:
    global _store
    _store = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli_main(argv: list[str] | None = None) -> int:
    """Backward-compatible entry point delegating to the unified CLI."""
    from app_planner.cli import cli_main as unified_cli_main
    return unified_cli_main(argv)


if __name__ == "__main__":
    sys.exit(cli_main())
