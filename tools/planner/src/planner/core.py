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

logger = logging.getLogger("dev-notes")

_NOTES_DIR = Path.home() / ".config" / "dev-notes"
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
    sprint: str = ""
    gh: str = ""
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
        if self.sprint:
            lines.append(f"sprint: {self.sprint}")
        if self.gh:
            lines.append(f"gh: {self.gh}")
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
        body = "\n".join(lines[body_start:]).rstrip()
        return cls(
            id=note_id,
            title=meta.get("title", ""),
            created_at=meta.get("created", ""),
            updated_at=meta.get("updated", ""),
            tags=tags,
            status=meta.get("status", "open"),
            sprint=meta.get("sprint", ""),
            gh=meta.get("gh", ""),
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
        self._dir = notes_dir or _NOTES_DIR
        if backend == "mogdb":
            self._bk = _MogDBBackend(self._dir)
        else:
            self._bk = _FileBackend(self._dir)

    def create(self, title: str, tags: list[str] | None = None,
               status: str = "open", sprint: str = "", gh: str = "",
               body: str = "") -> Note:
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
            sprint=sprint,
            gh=gh,
            body=body,
        )
        self._bk.put(note)
        logger.info("Created note: %s", note_id)
        return note

    def get(self, note_id: str) -> Note | None:
        matches = self._bk.find_by_prefix(note_id)
        if len(matches) == 1:
            return self._bk.note_by_id(matches[0])
        if len(matches) > 1:
            logger.warning("Ambiguous id '%s': %s", note_id, matches)
        return None

    def update(self, note_id: str, **kwargs: Any) -> Note | None:
        note = self.get(note_id)
        if note is None:
            return None
        old_id = note.id
        for key in ("title", "tags", "status", "sprint", "gh", "body"):
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
                   sprint: str | None = None,
                   limit: int = 50) -> list[Note]:
        notes: list[Note] = []
        for note in self._bk.all_notes():
            if tag and tag not in note.tags:
                continue
            if status and note.status != status:
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
                    or q in note.body.lower()):
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
    if _store is None or backend != "file":
        _store = NoteStore(backend=backend)
    return _store


def reset_note_store() -> None:
    global _store
    _store = None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cli_main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Return exit code."""
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "kanban":
        from .kanban import cli_main as kcli
        return kcli(argv[1:])
    if argv and argv[0] == "gui":
        from .gui import main as gmain
        return gmain(argv[1:])
    import argparse

    parser = argparse.ArgumentParser(prog="planner")
    parser.add_argument("--backend", default="file", choices=["file", "mogdb"],
                        help="Storage backend")
    sub = parser.add_subparsers(dest="cmd")

    p_new = sub.add_parser("new", help="Create a new note")
    p_new.add_argument("title", help="Note title")
    p_new.add_argument("--tags", default="", help="Comma-separated tags")
    p_new.add_argument("--status", default="open", choices=["open", "wip", "done", "blocked"])
    p_new.add_argument("--sprint", default="", help="Sprint identifier (e.g. S1, 2026-Q3)")
    p_new.add_argument("--gh", default="", help="GitHub issue reference (e.g. owner/repo#123)")
    p_new.add_argument("--body", default="", help="Body text")

    p_list = sub.add_parser("list", help="List notes")
    p_list.add_argument("--tag", default=None, help="Filter by tag")
    p_list.add_argument("--status", default=None, choices=["open", "wip", "done", "blocked", None])
    p_list.add_argument("--sprint", default=None, help="Filter by sprint")
    p_list.add_argument("--limit", type=int, default=20, help="Max results")

    p_show = sub.add_parser("show", help="Show a note")
    p_show.add_argument("note_id", help="Note id or prefix")

    p_edit = sub.add_parser("edit", help="Edit a note")
    p_edit.add_argument("note_id", help="Note id or prefix")
    p_edit.add_argument("--title", default=None, help="New title")
    p_edit.add_argument("--tags", default=None, help="Comma-separated tags")
    p_edit.add_argument("--status", default=None, choices=["open", "wip", "done", "blocked"])
    p_edit.add_argument("--sprint", default=None, help="Sprint identifier")
    p_edit.add_argument("--gh", default=None, help="GitHub issue reference")
    p_edit.add_argument("--body", default=None, help="New body text")

    p_del = sub.add_parser("delete", aliases=["rm"], help="Delete a note")
    p_del.add_argument("note_id", help="Note id or prefix")

    p_search = sub.add_parser("search", help="Search notes")
    p_search.add_argument("query", help="Search string")
    p_search.add_argument("--limit", type=int, default=20)

    sub.add_parser("today", help="Show today's notes")

    p_export = sub.add_parser("export", help="Export all notes")
    p_export.add_argument("output", nargs="?", default=None, help="Output file")

    sub.add_parser("tags", help="List all tags")
    sub.add_parser("status", help="Status summary")

    p_timeline = sub.add_parser("timeline", help="Show notes grouped by day")
    p_timeline.add_argument("--days", type=int, default=7, help="Number of days to show (default 7)")
    p_timeline.add_argument("--tag", default=None, help="Filter by tag")
    p_timeline.add_argument("--status", default=None, choices=["open", "wip", "done", "blocked"])

    p_sprint = sub.add_parser("sprint", help="Sprint operations")
    p_sprint.add_argument("sprint_name", help="Sprint identifier (e.g. S1)")
    p_sprint.add_argument("action", nargs="?", default="list",
                          choices=["list", "report"],
                          help="list: show notes (default). report: full markdown report.")

    args = parser.parse_args(argv)

    if args.cmd is None:
        parser.print_help()
        return 1

    store = get_note_store(backend=args.backend)

    if args.cmd == "new":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        note = store.create(args.title, tags=tags, status=args.status,
                            sprint=args.sprint, gh=args.gh, body=args.body)
        sprint_tag = f" [{args.sprint}]" if args.sprint else ""
        print(f"Created: {note.short_id}  {note.title}{sprint_tag}")
        return 0

    if args.cmd == "list":
        notes = store.list_notes(tag=args.tag, status=args.status,
                                 sprint=args.sprint, limit=args.limit)
        if not notes:
            print("No notes found.")
            return 0
        by_date: dict[str, list[Note]] = {}
        for n in notes:
            by_date.setdefault(n.date_str, []).append(n)
        for date_str, day_notes in by_date.items():
            print(f"\n  {date_str}")
            for n in day_notes:
                tags_str = f"  [{', '.join(n.tags)}]" if n.tags else ""
                icons = {"open": "○", "wip": "◐", "done": "●", "blocked": "✕"}
                icon = icons.get(n.status, "?")
                print(f"    {icon} {n.short_id}  {n.title}{tags_str}")
        print(f"\n  {len(notes)} note(s)")
        return 0

    if args.cmd == "show":
        note = store.get(args.note_id)
        if note is None:
            print(f"Note not found: {args.note_id}")
            return 1
        tags_str = ", ".join(note.tags) if note.tags else "none"
        sprint_str = f"\n  sprint: {note.sprint}" if note.sprint else ""
        gh_str = f"\n  gh: {note.gh}" if note.gh else ""
        if note.gh_url:
            gh_str += f"\n  gh_url: {note.gh_url}"
        print(f"  {note.title}")
        print(f"  id: {note.id}")
        print(f"  created: {note.created_at}")
        print(f"  updated: {note.updated_at}")
        print(f"  status: {note.status}")
        print(f"  tags: {tags_str}{sprint_str}{gh_str}")
        print("")
        for line in note.body.split("\n"):
            print(f"  {line}")
        return 0

    if args.cmd == "edit":
        kwargs: dict[str, Any] = {}
        if args.title is not None:
            kwargs["title"] = args.title
        if args.tags is not None:
            kwargs["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
        if args.status is not None:
            kwargs["status"] = args.status
        if args.sprint is not None:
            kwargs["sprint"] = args.sprint
        if args.gh is not None:
            kwargs["gh"] = args.gh
        if args.body is not None:
            kwargs["body"] = args.body
        if not kwargs:
            print("No changes specified.")
            return 1
        updated = store.update(args.note_id, **kwargs)
        if updated is None:
            print(f"Note not found: {args.note_id}")
            return 1
        print(f"Updated: {updated.short_id}  {updated.title}")
        return 0

    if args.cmd in ("delete", "rm"):
        if store.delete(args.note_id):
            print(f"Deleted: {args.note_id}")
            return 0
        print(f"Note not found: {args.note_id}")
        return 1

    if args.cmd == "search":
        results = store.search(args.query, limit=args.limit)
        if not results:
            print(f"No notes matching '{args.query}'")
            return 0
        for n in results:
            tags_str = f"  [{', '.join(n.tags)}]" if n.tags else ""
            print(f"    {n.short_id}  {n.title}{tags_str}")
        print(f"\n  {len(results)} result(s)")
        return 0

    if args.cmd == "sprint":
        notes = store.list_notes(sprint=args.sprint_name, limit=9999)
        if not notes:
            print(f"No notes for sprint '{args.sprint_name}'.")
            return 0

        if args.action == "report":
            report = store.sprint_report(args.sprint_name)
            print(report)
        else:
            by_status: dict[str, list[Note]] = {}
            for n in notes:
                by_status.setdefault(n.status, []).append(n)
            for status in ["open", "wip", "done", "blocked"]:
                items = by_status.get(status, [])
                if not items:
                    continue
                icons = {"open": "○", "wip": "◐", "done": "●", "blocked": "✕"}
                icon = icons.get(status, "?")
                print(f"\n  {icon} {status.upper()} ({len(items)})")
                for n in items:
                    gh_tag = f"  #{n.gh}" if n.gh else ""
                    print(f"    {n.short_id}  {n.title}{gh_tag}")
            print(f"\n  {len(notes)} note(s) in sprint '{args.sprint_name}'")
        return 0

    if args.cmd == "today":
        notes = store.today()
        if not notes:
            print("No notes today.")
            return 0
        for n in notes:
            tags_str = f"  [{', '.join(n.tags)}]" if n.tags else ""
            icons = {"open": "○", "wip": "◐", "done": "●", "blocked": "✕"}
            icon = icons.get(n.status, "?")
            print(f"    {icon} {n.short_id}  {n.title}{tags_str}")
        print(f"\n  {len(notes)} note(s) today")
        return 0

    if args.cmd == "export":
        content = store.export_all(output_path=args.output)
        if args.output:
            print(f"Exported {store.count()} notes to {args.output}")
        else:
            print(content)
        return 0

    if args.cmd == "tags":
        tag_counts: dict[str, int] = {}
        for n in store.list_notes(limit=9999):
            for tag in n.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if not tag_counts:
            print("No tags found.")
            return 0
        for tag, count in sorted(tag_counts.items()):
            print(f"    {tag:20s}  {count} note(s)")
        return 0

    if args.cmd == "status":
        status_counts: dict[str, int] = {}
        for n in store.list_notes(limit=9999):
            status_counts[n.status] = status_counts.get(n.status, 0) + 1
        if not status_counts:
            print("No notes.")
            return 0
        icons = {"open": "○", "wip": "◐", "done": "●", "blocked": "✕"}
        for s in ["open", "wip", "done", "blocked"]:
            count = status_counts.get(s, 0)
            if count:
                icon = icons.get(s, "?")
                print(f"    {icon} {s:10s}  {count}")
        return 0

    if args.cmd == "timeline":
        groups = store.timeline(days=args.days, tag=args.tag, status=args.status)
        if not groups:
            print("No notes in the specified range.")
            return 0
        total = 0
        icons = {"open": "○", "wip": "◐", "done": "●", "blocked": "✕"}
        for date_str, day_notes in groups:
            print(f"\n  ══ {date_str} ══")
            for n in day_notes:
                icon = icons.get(n.status, "?")
                tags_s = f"  [{', '.join(n.tags)}]" if n.tags else ""
                sprint_s = f"  [{n.sprint}]" if n.sprint else ""
                print(f"    {icon} {n.short_id}  {n.title}{tags_s}{sprint_s}")
            total += len(day_notes)
        print(f"\n  {total} note(s) across {len(groups)} day(s)")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
