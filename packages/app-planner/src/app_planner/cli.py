"""
app-planner — unified notes + kanban CLI.

One entry point for both notes and board operations. Every note mutation
optionally syncs to the kanban board so the two views stay in step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app_planner import config
from app_planner.store import PlannerStore, get_store

# ── Helpers ──────────────────────────────────────────────────────────────


def _get_store(args: argparse.Namespace) -> PlannerStore:
    board_dir = Path(args.board_dir) if hasattr(args, "board_dir") and args.board_dir else None
    notes_dir = Path(args.notes_dir) if hasattr(args, "notes_dir") and args.notes_dir else None
    return get_store(board_dir=board_dir, notes_dir=notes_dir)


def _auto_sync(store: PlannerStore) -> tuple[int, int, int] | None:
    """Best-effort sync after note mutations. Silently skips on error."""
    try:
        return store.sync()
    except Exception:  # noqa: BLE001
        return None


# ── Note subcommands ─────────────────────────────────────────────────────


def _note_new(args: argparse.Namespace) -> int:
    store = _get_store(args)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
    status = args.status or "open"
    if status not in config.STATUSES:
        print(f"Invalid status: {status}. Valid: {', '.join(config.STATUSES)}", file=sys.stderr)
        return 2
    note = store.create_note(
        args.title,
        tags=tags,
        status=status,
        body=args.body or "",
        sprint=args.sprint or "",
        gh=args.gh or "",
        assignee=args.author or "",
    )
    _auto_sync(store)
    print(f"Created note: {note.id[:8]} — {note.title}")
    return 0


def _note_list(args: argparse.Namespace) -> int:
    store = _get_store(args)
    notes = store.list_notes(tag=args.tag, status=args.status, limit=args.limit)
    for note in notes:
        icon = "\u25cf" if note.status == "done" else "\u25cb"
        tags = ",".join(note.tags[:3]) if note.tags else ""
        print(f"  {icon} [{note.id[:8]}] {note.title}" + (f"  ({tags})" if tags else ""))
    print(f"\n{len(notes)} note(s)")
    return 0


def _note_show(args: argparse.Namespace) -> int:
    store = _get_store(args)
    note = store.get_note(args.id)
    if not note:
        print(f"Note not found: {args.id}", file=sys.stderr)
        return 1
    print(f"Title:   {note.title}")
    print(f"Status:  {note.status}")
    print(f"Tags:    {', '.join(note.tags)}")
    print(f"Sprint:  {note.sprint or '-'}")
    print(f"GitHub:  {note.gh or '-'}")
    print(f"Created: {note.created_at}")
    print(f"Updated: {note.updated_at}")
    if note.body:
        print(f"\n{note.body}")
    return 0


def _note_update(args: argparse.Namespace) -> int:
    store = _get_store(args)
    updates: dict = {}
    if args.status:
        updates["status"] = args.status
    if args.tags:
        updates["tags"] = [t.strip() for t in args.tags.split(",") if t.strip()]
    if args.title:
        updates["title"] = args.title
    if args.body is not None:
        updates["body"] = args.body
    note = store.update_note(args.id, **updates)
    if not note:
        print(f"Note not found: {args.id}", file=sys.stderr)
        return 1
    _auto_sync(store)
    print(f"Updated: {note.id[:8]} — {note.title}")
    return 0


def _note_delete(args: argparse.Namespace) -> int:
    store = _get_store(args)
    if store.delete_note(args.id):
        _auto_sync(store)
        print(f"Deleted: {args.id[:8]}")
        return 0
    print(f"Note not found: {args.id}", file=sys.stderr)
    return 1


def _note_search(args: argparse.Namespace) -> int:
    store = _get_store(args)
    notes = store.search_notes(args.query)
    for note in notes:
        icon = "\u25cf" if note.status == "done" else "\u25cb"
        print(f"  {icon} [{note.id[:8]}] {note.title}")
    print(f"\n{len(notes)} result(s)")
    return 0


# ── Board subcommands ────────────────────────────────────────────────────


def _board_show(args: argparse.Namespace) -> int:
    store = _get_store(args)
    card_id = getattr(args, "id", None)
    if card_id:
        card = store.get_card(card_id)
        if not card:
            print(f"Card not found: {card_id}", file=sys.stderr)
            return 1
        print(f"Title:       {card.title}")
        print(f"Column:      {card.column}")
        print(f"Priority:    {card.priority}")
        print(f"Tags:        {', '.join(card.tags) if card.tags else 'none'}")
        print(f"Assignee:    {card.assignee or '-'}")
        print(f"Due:         {card.due_date or '-'}")
        print(f"Type:        {card.card_type or '-'}")
        print(f"Blocked by:  {', '.join(card.blocked_by) if card.blocked_by else 'none'}")
        print(f"Description: {card.description or '-'}")
        print(f"Sprint:      {card.sprint or '-'}")
        print(f"GitHub:      {card.gh or '-'}")
        return 0
    board = store.load_board()
    col_widths: dict[str, int] = {}
    for col in board.columns:
        name = col["name"]
        cards = [c for c in board.cards if c.column == name]
        col_widths[name] = max(len(name), max((len(c.title) for c in cards), default=0))

    for col in board.columns:
        name = col["name"]
        cards = [c for c in board.cards if c.column == name]
        wip = col.get("wip_limit", 0)
        wip_str = f" (WIP {wip})" if wip else ""
        print(f"\n\u250c\u2500 {name.upper()}{wip_str} \u2500" + "\u2500" * 40)
        if not cards:
            print("\u2502  (empty)")
        for card in cards:
            pri = {"low": " ", "medium": "!", "high": "!!", "critical": "!!!"}.get(
                card.priority, " "
            )
            print(f"\u2502  [{pri}] {card.title}")
    print()
    return 0


def _board_add(args: argparse.Namespace) -> int:
    store = _get_store(args)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
    card = store.add_card(
        args.title,
        column=args.column or "todo",
        priority=args.priority or "medium",
        description=args.description or "",
        tags=tags,
    )
    print(f"Created: {card.id[:8]} — {card.title}")
    return 0


def _board_move(args: argparse.Namespace) -> int:
    store = _get_store(args)
    board = store.load_board()
    valid_columns = [c["name"] for c in board.columns]
    if args.column not in valid_columns:
        print(f"Invalid column: {args.column}. Valid: {', '.join(valid_columns)}", file=sys.stderr)
        return 1
    if store.move_card(args.id, args.column):
        print(f"Moved {args.id[:8]} → {args.column}")
        return 0
    print(f"Card not found: {args.id}", file=sys.stderr)
    return 1


def _board_delete(args: argparse.Namespace) -> int:
    store = _get_store(args)
    if store.delete_card(args.id):
        print(f"Deleted: {args.id[:8]}")
        return 0
    print(f"Card not found: {args.id}", file=sys.stderr)
    return 1


def _board_tags(args: argparse.Namespace) -> int:
    store = _get_store(args)
    tags = store.get_tags()
    for tag, count in sorted(tags.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")
    return 0


def _board_stats(args: argparse.Namespace) -> int:
    store = _get_store(args)
    stats = store.get_stats()
    print(f"Total cards: {stats['total']}")
    for col, count in stats["byColumn"].items():
        print(f"  {col}: {count}")
    return 0


# ── Sync ─────────────────────────────────────────────────────────────────


def _sync(args: argparse.Namespace) -> int:
    store = _get_store(args)
    added, updated, total = store.sync()
    if not args.quiet:
        board = store.load_board()
        for card in board.cards:
            icon = "\u2713" if card.column == "done" else "\u25cb"
            print(f"  {icon} [{card.column:12s}] {card.title}")
        print(f"\n{added} new card(s), {updated} moved, {total} total")
    else:
        print(f"{added} new card(s), {updated} moved, {total} total")
    return 0


# ── CLI Parser ───────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app-planner",
        description="Unified notes + kanban planner.",
    )
    parser.add_argument("--notes-dir", default=None, help="Notes directory")
    parser.add_argument("--board-dir", default=None, help="Board directory")
    sub = parser.add_subparsers(dest="command")

    # Notes
    notes = sub.add_parser("note", aliases=["notes"], help="Note operations")
    note_sub = notes.add_subparsers(dest="subcommand")
    p = note_sub.add_parser("new", help="Create a note")
    p.add_argument("title")
    p.add_argument("--tags", default=None)
    p.add_argument("--status", default=None)
    p.add_argument("--body", default=None)
    p.add_argument("--sprint", default=None)
    p.add_argument("--gh", default=None)
    p.add_argument("--author", default=None)

    p = note_sub.add_parser("list", aliases=["ls"], help="List notes")
    p.add_argument("--tag", default=None)
    p.add_argument("--status", default=None)
    p.add_argument("--limit", type=int, default=50)

    p = note_sub.add_parser("show", help="Show a note")
    p.add_argument("id")

    p = note_sub.add_parser("update", help="Update a note")
    p.add_argument("id")
    p.add_argument("--title", default=None)
    p.add_argument("--status", default=None)
    p.add_argument("--tags", default=None)
    p.add_argument("--body", default=None)

    p = note_sub.add_parser("delete", aliases=["rm"], help="Delete a note")
    p.add_argument("id")

    p = note_sub.add_parser("search", help="Search notes")
    p.add_argument("query")

    # Board
    board = sub.add_parser("board", aliases=["kanban"], help="Board operations")
    board_sub = board.add_subparsers(dest="subcommand")
    p = board_sub.add_parser("show", aliases=["ls"], help="Show the board or a specific card")
    p.add_argument("id", nargs="?", default=None)

    p = board_sub.add_parser("add", help="Add a card")
    p.add_argument("title")
    p.add_argument("--column", default=None)
    p.add_argument("--priority", default=None)
    p.add_argument("--description", default=None)
    p.add_argument("--tags", default=None)

    p = board_sub.add_parser("move", help="Move a card")
    p.add_argument("id")
    p.add_argument("column")

    p = board_sub.add_parser("delete", aliases=["rm"], help="Delete a card")
    p.add_argument("id")

    board_sub.add_parser("tags", help="List tags")
    board_sub.add_parser("stats", help="Show stats")

    # Sync
    sync_p = sub.add_parser("sync", help="Sync notes to board")
    sync_p.add_argument("--quiet", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    cmd = args.command
    sub = getattr(args, "subcommand", None)

    dispatch = {
        ("note", "new"): _note_new,
        ("note", "list"): _note_list,
        ("note", "ls"): _note_list,
        ("note", "show"): _note_show,
        ("note", "update"): _note_update,
        ("note", "delete"): _note_delete,
        ("note", "rm"): _note_delete,
        ("note", "search"): _note_search,
        ("notes", "new"): _note_new,
        ("notes", "list"): _note_list,
        ("notes", "ls"): _note_list,
        ("notes", "show"): _note_show,
        ("notes", "update"): _note_update,
        ("notes", "delete"): _note_delete,
        ("notes", "rm"): _note_delete,
        ("notes", "search"): _note_search,
        ("board", "show"): _board_show,
        ("board", "ls"): _board_show,
        ("board", "add"): _board_add,
        ("board", "move"): _board_move,
        ("board", "delete"): _board_delete,
        ("board", "rm"): _board_delete,
        ("board", "tags"): _board_tags,
        ("board", "stats"): _board_stats,
        ("kanban", "show"): _board_show,
        ("kanban", "ls"): _board_show,
        ("kanban", "add"): _board_add,
        ("kanban", "move"): _board_move,
        ("kanban", "delete"): _board_delete,
        ("kanban", "rm"): _board_delete,
        ("kanban", "tags"): _board_tags,
        ("kanban", "stats"): _board_stats,
        ("sync", None): _sync,
    }

    handler = dispatch.get((cmd, sub))
    if handler:
        return handler(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())


# Backward compatibility
cli_main = main
