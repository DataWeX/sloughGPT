"""
app-planner — unified notes + kanban CLI.

One entry point for both notes and board operations. Every note mutation
optionally syncs to the kanban board so the two views stay in step without
a separate ``sync`` step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app_planner import config
from app_planner.core import NoteStore, get_note_store, reset_note_store
from app_planner.kanban import KanbanStore
from app_planner.sync import sync_notes_to_board


# ---------------------------------------------------------------------------
# Auto-sync helper
# ---------------------------------------------------------------------------

def _auto_sync(backend: str = "file", notes_dir: Path | None = None,
               board_dir: Path | None = None) -> tuple[int, int, int] | None:
    """Sync notes to board unless ``APP_PLANNER_NO_SYNC`` is set."""
    if sys.stdout.isatty() and sys.stdin.isatty():
        return None
    try:
        ns = NoteStore(
            notes_dir=notes_dir or config.default_notes_dir(),
            backend=backend,
        )
        bs = KanbanStore(board_dir=board_dir or config.default_board_dir())
        return sync_notes_to_board(ns, bs)
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Note subcommands
# ---------------------------------------------------------------------------

def _note_new(args: argparse.Namespace) -> int:
    store = get_note_store(backend=args.backend or config.default_backend())
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
    note = store.create(
        args.title,
        tags=tags,
        status=args.status,
        author=args.author,
        sprint=args.sprint,
        gh=args.gh,
        assignee=args.assignee,
        body=args.body,
    )
    sprint_tag = f" [{args.sprint}]" if args.sprint else ""
    assign_tag = f" @{args.assignee}" if args.assignee else ""
    print(f"Created: {note.short_id}  {note.title}{sprint_tag}{assign_tag}")
    if not args.no_sync:
        _auto_sync(args.backend, getattr(args, "notes_dir", None), getattr(args, "board_dir", None))
    return 0


def _note_list(args: argparse.Namespace) -> int:
    store = get_note_store(backend=args.backend or config.default_backend())
    notes = store.list_notes(
        tag=args.tag,
        status=args.status,
        author=args.author,
        sprint=args.sprint,
        limit=args.limit,
        today=args.today,
    )
    if not notes:
        print("No notes found.")
        return 0
    by_date: dict[str, list] = {}
    for n in notes:
        by_date.setdefault(n.date_str, []).append(n)
    for date_str, day_notes in by_date.items():
        print(f"\n  {date_str}")
        for n in day_notes:
            tags_str = f"  [{', '.join(n.tags)}]" if n.tags else ""
            author_str = f"  @{n.author}" if n.author else ""
            icon = config.STATUS_ICONS.get(n.status, "?")
            print(f"    {icon} {n.short_id}  {n.title}{tags_str}{author_str}")
    print(f"\n  {len(notes)} note(s)")
    return 0


def _note_show(args: argparse.Namespace) -> int:
    store = get_note_store(backend=args.backend or config.default_backend())
    note = store.get(args.note_id)
    if note is None:
        print(f"Note not found: {args.note_id}")
        return 1
    if note.id != args.note_id:
        print(f"  (matched {note.id})")
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
    print(f"  author: {note.author or 'unknown'}")
    print(f"  tags: {tags_str}{sprint_str}{gh_str}")
    if note.assignee:
        print(f"  assignee: {note.assignee}")
    print("")
    for line in note.body.split("\n"):
        print(f"  {line}")
    return 0


def _note_edit(args: argparse.Namespace) -> int:
    store = get_note_store(backend=args.backend or config.default_backend())
    kwargs: dict = {}
    if args.title is not None:
        kwargs["title"] = args.title
    if args.tags is not None:
        kwargs["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    if args.status is not None:
        kwargs["status"] = args.status
    if args.author is not None:
        kwargs["author"] = args.author
    if args.sprint is not None:
        kwargs["sprint"] = args.sprint
    if args.gh is not None:
        kwargs["gh"] = args.gh
    if args.assignee is not None:
        kwargs["assignee"] = args.assignee
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
    if not args.no_sync:
        _auto_sync(args.backend, getattr(args, "notes_dir", None), getattr(args, "board_dir", None))
    return 0


def _note_delete(args: argparse.Namespace) -> int:
    store = get_note_store(backend=args.backend or config.default_backend())
    if store.delete(args.note_id):
        print(f"Deleted: {args.note_id}")
        if not args.no_sync:
            _auto_sync(args.backend, getattr(args, "notes_dir", None), getattr(args, "board_dir", None))
        return 0
    print(f"Note not found: {args.note_id}")
    return 1


def _note_search(args: argparse.Namespace) -> int:
    store = get_note_store(backend=args.backend or config.default_backend())
    results = store.search(args.query, limit=args.limit)
    if not results:
        print(f"No notes matching '{args.query}'")
        return 0
    for n in results:
        tags_str = f"  [{', '.join(n.tags)}]" if n.tags else ""
        print(f"    {n.short_id}  {n.title}{tags_str}")
    print(f"\n  {len(results)} result(s)")
    return 0


def _note_today(args: argparse.Namespace) -> int:
    store = get_note_store(backend=args.backend or config.default_backend())
    notes = store.today()
    if not notes:
        print("No notes today.")
        return 0
    for n in notes:
        tags_str = f"  [{', '.join(n.tags)}]" if n.tags else ""
        icon = config.STATUS_ICONS.get(n.status, "?")
        print(f"    {icon} {n.short_id}  {n.title}{tags_str}")
    print(f"\n  {len(notes)} note(s) today")
    return 0


def _note_export(args: argparse.Namespace) -> int:
    store = get_note_store(backend=args.backend or config.default_backend())
    content = store.export_all(output_path=args.output)
    if args.output:
        print(f"Exported {store.count()} notes to {args.output}")
    else:
        print(content)
    return 0


def _note_tags(args: argparse.Namespace) -> int:
    store = get_note_store(backend=args.backend or config.default_backend())
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


def _note_status(args: argparse.Namespace) -> int:
    store = get_note_store(backend=args.backend or config.default_backend())
    status_counts: dict[str, int] = {}
    for n in store.list_notes(limit=9999):
        status_counts[n.status] = status_counts.get(n.status, 0) + 1
    if not status_counts:
        print("No notes.")
        return 0
    for s in config.STATUSES:
        count = status_counts.get(s, 0)
        if count:
            icon = config.STATUS_ICONS.get(s, "?")
            print(f"    {icon} {s:10s}  {count}")
    return 0


def _note_timeline(args: argparse.Namespace) -> int:
    store = get_note_store(backend=args.backend or config.default_backend())
    groups = store.timeline(days=args.days, tag=args.tag, status=args.status)
    if not groups:
        print("No notes in the specified range.")
        return 0
    total = 0
    for date_str, day_notes in groups:
        print(f"\n  ══ {date_str} ══")
        for n in day_notes:
            icon = config.STATUS_ICONS.get(n.status, "?")
            tags_s = f"  [{', '.join(n.tags)}]" if n.tags else ""
            sprint_s = f"  [{n.sprint}]" if n.sprint else ""
            print(f"    {icon} {n.short_id}  {n.title}{tags_s}{sprint_s}")
        total += len(day_notes)
    print(f"\n  {total} note(s) across {len(groups)} day(s)")
    return 0


def _note_sprint(args: argparse.Namespace) -> int:
    store = get_note_store(backend=args.backend or config.default_backend())
    notes = store.list_notes(sprint=args.sprint_name, limit=9999)
    if not notes:
        print(f"No notes for sprint '{args.sprint_name}'.")
        return 0
    if args.action == "report":
        report = store.sprint_report(args.sprint_name)
        print(report)
    else:
        by_status: dict[str, list] = {}
        for n in notes:
            by_status.setdefault(n.status, []).append(n)
        for status in config.STATUSES:
            items = by_status.get(status, [])
            if not items:
                continue
            icon = config.STATUS_ICONS.get(status, "?")
            print(f"\n  {icon} {status.upper()} ({len(items)})")
            for n in items:
                gh_tag = f"  #{n.gh}" if n.gh else ""
                print(f"    {n.short_id}  {n.title}{gh_tag}")
        print(f"\n  {len(notes)} note(s) in sprint '{args.sprint_name}'")
    return 0


# ---------------------------------------------------------------------------
# Kanban subcommands
# ---------------------------------------------------------------------------

def _kanban_init(args: argparse.Namespace) -> int:
    from app_planner.kanban import cli_main as kcli
    return kcli(["init", "--name", args.name, *(["--force"] if args.force else [])])


def _kanban_add(args: argparse.Namespace) -> int:
    from app_planner.kanban import cli_main as kcli
    return kcli([
        "add", args.title,
        *(["--column", args.column] if args.column else []),
        "--priority", args.priority,
        "--type", args.type,
        *([f"--effort={args.effort}"] if args.effort else []),
        *(["--sprint", args.sprint] if args.sprint else []),
        "--desc", args.desc,
        "--tags", args.tags,
        *(["--due", args.due] if args.due else []),
        *(["--assignee", args.assignee] if args.assignee else []),
    ])


def _kanban_list(args: argparse.Namespace) -> int:
    from app_planner.kanban import cli_main as kcli
    argv = ["list"]
    if args.column:
        argv += ["--column", args.column]
    if args.priority:
        argv += ["--priority", args.priority]
    if args.tag:
        argv += ["--tag", args.tag]
    if args.assignee:
        argv += ["--assignee", args.assignee]
    if args.due_before:
        argv += ["--due-before", args.due_before]
    if args.due_after:
        argv += ["--due-after", args.due_after]
    if args.overdue:
        argv += ["--overdue"]
    argv += ["--limit", str(args.limit)]
    return kcli(argv)


def _kanban_show(args: argparse.Namespace) -> int:
    from app_planner.kanban import cli_main as kcli
    return kcli(["show", args.card_id])


def _kanban_edit(args: argparse.Namespace) -> int:
    from app_planner.kanban import cli_main as kcli
    argv = ["edit", args.card_id]
    for flag, val in [
        ("--title", args.title),
        ("--desc", args.desc),
        ("--priority", args.priority),
        ("--type", args.type),
        ("--effort", args.effort),
        ("--sprint", args.sprint),
        ("--tags", args.tags),
        ("--due", args.due),
        ("--assignee", args.assignee),
    ]:
        if val is not None:
            argv += [flag, str(val)]
    return kcli(argv)


def _kanban_move(args: argparse.Namespace) -> int:
    from app_planner.kanban import cli_main as kcli
    return kcli(["move", args.card_id, args.column])


def _kanban_delete(args: argparse.Namespace) -> int:
    from app_planner.kanban import cli_main as kcli
    return kcli(["delete", args.card_id])


def _kanban_board(args: argparse.Namespace) -> int:
    from app_planner.kanban import cli_main as kcli
    return kcli(["board"])


def _kanban_note(args: argparse.Namespace) -> int:
    from app_planner.kanban import cli_main as kcli
    sub_argv = ["note", args.note_cmd, args.card_id]
    if args.note_cmd == "add":
        sub_argv += [args.text, *(["--author", args.author] if args.author else [])]
    elif args.note_cmd == "delete":
        sub_argv += [args.note_id]
    return kcli(sub_argv)


def _kanban_columns(args: argparse.Namespace) -> int:
    from app_planner.kanban import cli_main as kcli
    return kcli(["columns"])


def _kanban_column_add(args: argparse.Namespace) -> int:
    from app_planner.kanban import cli_main as kcli
    return kcli(["column-add", args.name, *(["--wip", str(args.wip)] if args.wip else [])])


def _kanban_column_rename(args: argparse.Namespace) -> int:
    from app_planner.kanban import cli_main as kcli
    return kcli(["column-rename", args.name, args.new_name])


def _kanban_column_rm(args: argparse.Namespace) -> int:
    from app_planner.kanban import cli_main as kcli
    return kcli(["column-rm", args.name, "--move-to", args.move_to])


def _kanban_archive(args: argparse.Namespace) -> int:
    from app_planner.kanban import cli_main as kcli
    return kcli(["archive"])


def _kanban_search(args: argparse.Namespace) -> int:
    from app_planner.kanban import cli_main as kcli
    return kcli(["search", args.query, "--limit", str(args.limit)])


def _kanban_stats(args: argparse.Namespace) -> int:
    from app_planner.kanban import cli_main as kcli
    return kcli(["stats"])


def _kanban_export(args: argparse.Namespace) -> int:
    from app_planner.kanban import cli_main as kcli
    return kcli(["export", *(["--output", args.output] if args.output else [])])


def _kanban_import(args: argparse.Namespace) -> int:
    from app_planner.kanban import cli_main as kcli
    return kcli(["import", args.input])


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def cli_main(argv: list[str] | None = None) -> int:
    """Unified entry point for notes + kanban."""
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="app-planner",
        description="Unified notes + kanban planner",
    )
    parser.add_argument("--backend", default=None, choices=config.BACKENDS,
                        help="Storage backend (default: config/env)")
    parser.add_argument("--notes-dir", default=None, help="Notes directory")
    parser.add_argument("--board-dir", default=None, help="Board directory")
    parser.add_argument("--no-sync", action="store_true",
                        help="Disable auto-sync after note mutations")

    sub = parser.add_subparsers(dest="cmd")

    # --- Notes ---
    p_new = sub.add_parser("new", help="Create a new note")
    p_new.add_argument("title", help="Note title")
    p_new.add_argument("--tags", default="", help="Comma-separated tags")
    p_new.add_argument("--status", default="open", choices=config.STATUSES)
    p_new.add_argument("--author", default="", help="Note author/owner")
    p_new.add_argument("--sprint", default="", help="Sprint identifier")
    p_new.add_argument("--gh", default="", help="GitHub issue reference")
    p_new.add_argument("--assignee", default="", help="Assignee")
    p_new.add_argument("--body", default="", help="Body text")

    p_list = sub.add_parser("list", help="List notes")
    p_list.add_argument("--tag", default=None, help="Filter by tag")
    p_list.add_argument("--status", default=None, choices=config.STATUSES)
    p_list.add_argument("--author", default=None, help="Filter by author")
    p_list.add_argument("--sprint", default=None, help="Filter by sprint")
    p_list.add_argument("--limit", type=int, default=20, help="Max results")
    p_list.add_argument("--today", action="store_true", help="Only today's notes")

    p_show = sub.add_parser("show", help="Show a note")
    p_show.add_argument("note_id", help="Note id or prefix")

    p_edit = sub.add_parser("edit", help="Edit a note")
    p_edit.add_argument("note_id", help="Note id or prefix")
    p_edit.add_argument("--title", default=None)
    p_edit.add_argument("--tags", default=None)
    p_edit.add_argument("--status", default=None, choices=config.STATUSES)
    p_edit.add_argument("--author", default=None)
    p_edit.add_argument("--sprint", default=None)
    p_edit.add_argument("--gh", default=None)
    p_edit.add_argument("--assignee", default=None)
    p_edit.add_argument("--body", default=None)

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
    p_timeline.add_argument("--days", type=int, default=7, help="Number of days to show")
    p_timeline.add_argument("--tag", default=None, help="Filter by tag")
    p_timeline.add_argument("--status", default=None, choices=config.STATUSES)

    p_sprint = sub.add_parser("sprint", help="Sprint operations")
    p_sprint.add_argument("sprint_name", help="Sprint identifier")
    p_sprint.add_argument("action", nargs="?", default="list",
                          choices=["list", "report"])

    # --- Kanban ---
    p_init = sub.add_parser("init", help="Initialize a new board")
    p_init.add_argument("--name", default="board", help="Board name")
    p_init.add_argument("--force", action="store_true", help="Recreate if exists")

    p_add = sub.add_parser("add", help="Add a card")
    p_add.add_argument("title", help="Card title")
    p_add.add_argument("--column", default="", help="Target column")
    p_add.add_argument("--priority", default="medium", choices=["low", "medium", "high"])
    p_add.add_argument("--type", default="task", choices=["task", "bug", "feature", "chore", "spike", "docs"])
    p_add.add_argument("--effort", type=float, default=0.0, help="Effort estimate")
    p_add.add_argument("--sprint", default="", help="Sprint identifier")
    p_add.add_argument("--desc", default="", help="Description")
    p_add.add_argument("--tags", default="", help="Comma-separated tags")
    p_add.add_argument("--due", default="", help="Due date (ISO)")
    p_add.add_argument("--assignee", default="", help="Assignee")

    p_list = sub.add_parser("cards", help="List cards")
    p_list.add_argument("--column", default=None, help="Filter by column")
    p_list.add_argument("--priority", default=None, choices=["low", "medium", "high"])
    p_list.add_argument("--tag", default=None, help="Filter by tag")
    p_list.add_argument("--assignee", default=None, help="Filter by assignee")
    p_list.add_argument("--due-before", default="", help="Due before YYYY-MM-DD")
    p_list.add_argument("--due-after", default="", help="Due after YYYY-MM-DD")
    p_list.add_argument("--overdue", action="store_true", help="Only overdue cards")
    p_list.add_argument("--limit", type=int, default=50)

    p_show = sub.add_parser("card", help="Show card details")
    p_show.add_argument("card_id", help="Card id or prefix")

    p_edit = sub.add_parser("edit-card", help="Edit a card")
    p_edit.add_argument("card_id", help="Card id or prefix")
    p_edit.add_argument("--title", default=None)
    p_edit.add_argument("--desc", default=None)
    p_edit.add_argument("--priority", default=None, choices=["low", "medium", "high"])
    p_edit.add_argument("--type", default=None, choices=["task", "bug", "feature", "chore", "spike", "docs"])
    p_edit.add_argument("--effort", type=float, default=None, help="Effort estimate")
    p_edit.add_argument("--sprint", default=None, help="Sprint identifier")
    p_edit.add_argument("--tags", default=None)
    p_edit.add_argument("--due", default=None)
    p_edit.add_argument("--assignee", default=None)

    p_move = sub.add_parser("move", help="Move card to another column")
    p_move.add_argument("card_id", help="Card id or prefix")
    p_move.add_argument("column", help="Target column")

    p_block = sub.add_parser("block", help="Block a card by another card")
    p_block.add_argument("card_id", help="Card id or prefix to block")
    p_block.add_argument("blocker_id", help="Blocker card id or prefix")

    p_unblock = sub.add_parser("unblock", help="Unblock a card")
    p_unblock.add_argument("card_id", help="Card id or prefix")
    p_unblock.add_argument("blocker_id", help="Blocker card id or prefix")

    p_blocked = sub.add_parser("blocked", help="List blocked cards")

    p_del = sub.add_parser("delete-card", aliases=["rm-card"], help="Delete a card")
    p_del.add_argument("card_id", help="Card id or prefix")

    sub.add_parser("board", help="Show ASCII kanban board")

    p_note = sub.add_parser("note", help="Manage notes on a card")
    p_note_sub = p_note.add_subparsers(dest="note_cmd")
    p_na = p_note_sub.add_parser("add", help="Add note")
    p_na.add_argument("card_id")
    p_na.add_argument("text", help="Note text")
    p_na.add_argument("--author", default="")
    p_nl = p_note_sub.add_parser("list", help="List notes")
    p_nl.add_argument("card_id")
    p_nd = p_note_sub.add_parser("delete", aliases=["rm"], help="Delete note")
    p_nd.add_argument("card_id")
    p_nd.add_argument("note_id")

    sub.add_parser("columns", help="List columns")
    p_ca = sub.add_parser("column-add", help="Add a column")
    p_ca.add_argument("name")
    p_ca.add_argument("--wip", type=int, default=0)
    p_cr = sub.add_parser("column-rename", help="Rename a column")
    p_cr.add_argument("name")
    p_cr.add_argument("new_name")
    p_crm = sub.add_parser("column-rm", help="Remove a column")
    p_crm.add_argument("name")
    p_crm.add_argument("--move-to", default="todo")

    sub.add_parser("archive", help="Archive (delete) all done cards")

    p_search = sub.add_parser("search-cards", help="Search cards")
    p_search.add_argument("query")
    p_search.add_argument("--limit", type=int, default=30)

    sub.add_parser("stats", help="Board statistics")

    p_export = sub.add_parser("export-board", help="Export board to JSON")
    p_export.add_argument("--output", default="", help="Output file path")

    p_import = sub.add_parser("import-board", help="Import board from JSON")
    p_import.add_argument("input", help="Input JSON file")

    sub.add_parser("sync", help="Sync notes to board")
    sub.add_parser("gui", help="Launch web GUI")

    args = parser.parse_args(argv)

    if args.cmd is None:
        parser.print_help()
        return 0

    # --- Notes ---
    if args.cmd == "new":
        return _note_new(args)
    if args.cmd == "list":
        return _note_list(args)
    if args.cmd == "show":
        return _note_show(args)
    if args.cmd == "edit":
        return _note_edit(args)
    if args.cmd in ("delete", "rm"):
        return _note_delete(args)
    if args.cmd == "search":
        return _note_search(args)
    if args.cmd == "today":
        return _note_today(args)
    if args.cmd == "export":
        return _note_export(args)
    if args.cmd == "tags":
        return _note_tags(args)
    if args.cmd == "status":
        return _note_status(args)
    if args.cmd == "timeline":
        return _note_timeline(args)
    if args.cmd == "sprint":
        return _note_sprint(args)

    # --- Kanban ---
    if args.cmd == "init":
        return _kanban_init(args)
    if args.cmd == "add":
        return _kanban_add(args)
    if args.cmd == "cards":
        return _kanban_list(args)
    if args.cmd == "card":
        return _kanban_show(args)
    if args.cmd == "edit-card":
        return _kanban_edit(args)
    if args.cmd == "move":
        return _kanban_move(args)
    if args.cmd == "delete-card" or args.cmd == "rm-card":
        return _kanban_delete(args)
    if args.cmd == "board":
        return _kanban_board(args)
    if args.cmd == "note":
        return _kanban_note(args)
    if args.cmd == "columns":
        return _kanban_columns(args)
    if args.cmd == "column-add":
        return _kanban_column_add(args)
    if args.cmd == "column-rename":
        return _kanban_column_rename(args)
    if args.cmd == "column-rm":
        return _kanban_column_rm(args)
    if args.cmd == "archive":
        return _kanban_archive(args)
    if args.cmd == "search-cards":
        return _kanban_search(args)
    if args.cmd == "stats":
        return _kanban_stats(args)
    if args.cmd == "export-board":
        return _kanban_export(args)
    if args.cmd == "import-board":
        return _kanban_import(args)

    # --- Meta ---
    if args.cmd == "sync":
        from app_planner.sync import cli_main as scli
        return scli(["--notes-dir", str(args.notes_dir or config.default_notes_dir()),
                     "--board-dir", str(args.board_dir or config.default_board_dir()),
                     "--backend", args.backend or config.default_backend()])
    if args.cmd == "gui":
        from app_planner.gui import main as gmain
        return gmain(["--notes-dir", str(args.notes_dir or config.default_notes_dir()),
                      "--board-dir", str(args.board_dir or config.default_board_dir()),
                      "--backend", args.backend or config.default_backend()])

    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
