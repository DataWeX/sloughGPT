"""
planner CLI — command-line interface for the planner package.

Usage:
    planner notes new "Fix kernel boot" --tags kernel,bugfix --status done
    planner notes list
    planner board list
    planner board add "New task" --column todo --priority high
    planner sync
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from planner import config


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="planner",
        description="Planner — notes + kanban board management",
    )
    parser.add_argument("--backend", default=None, choices=config.BACKENDS,
                        help="Storage backend (default: config/env)")
    parser.add_argument("--notes-dir", default=None, help="Notes directory")
    parser.add_argument("--board-dir", default=None, help="Board directory")

    sub = parser.add_subparsers(dest="command")

    # Notes subcommand
    notes_parser = sub.add_parser("notes", help="Notes operations")
    notes_sub = notes_parser.add_subparsers(dest="notes_cmd")

    # notes new
    p_new = notes_sub.add_parser("new", help="Create a new note")
    p_new.add_argument("title", help="Note title")
    p_new.add_argument("--tags", default="", help="Comma-separated tags")
    p_new.add_argument("--status", default="open", choices=config.STATUSES)
    p_new.add_argument("--sprint", default="", help="Sprint identifier")
    p_new.add_argument("--gh", default="", help="GitHub issue reference")
    p_new.add_argument("--body", default="", help="Body text")

    # notes list
    p_list = notes_sub.add_parser("list", help="List notes")
    p_list.add_argument("--tag", default=None, help="Filter by tag")
    p_list.add_argument("--status", default=None, choices=config.STATUSES + [None])
    p_list.add_argument("--sprint", default=None, help="Filter by sprint")
    p_list.add_argument("--limit", type=int, default=20, help="Max results")
    p_list.add_argument("--today", action="store_true", help="Only today's notes")

    # notes show
    p_show = notes_sub.add_parser("show", help="Show a note")
    p_show.add_argument("note_id", help="Note id or prefix")

    # notes edit
    p_edit = notes_sub.add_parser("edit", help="Edit a note")
    p_edit.add_argument("note_id", help="Note id or prefix")
    p_edit.add_argument("--title", default=None, help="New title")
    p_edit.add_argument("--tags", default=None, help="Comma-separated tags")
    p_edit.add_argument("--status", default=None, choices=config.STATUSES)
    p_edit.add_argument("--sprint", default=None, help="Sprint identifier")
    p_edit.add_argument("--gh", default=None, help="GitHub issue reference")
    p_edit.add_argument("--body", default=None, help="New body text")

    # notes delete
    p_del = notes_sub.add_parser("delete", aliases=["rm"], help="Delete a note")
    p_del.add_argument("note_id", help="Note id or prefix")

    # notes search
    p_search = notes_sub.add_parser("search", help="Search notes")
    p_search.add_argument("query", help="Search string")
    p_search.add_argument("--limit", type=int, default=20)

    # Board subcommand
    board_parser = sub.add_parser("board", help="Board operations")
    board_sub = board_parser.add_subparsers(dest="board_cmd")

    # board list
    board_sub.add_parser("list", help="List all cards")

    # board add
    p_add = board_sub.add_parser("add", help="Add a card")
    p_add.add_argument("title", help="Card title")
    p_add.add_argument("--column", default="todo", choices=config.COLUMNS)
    p_add.add_argument("--priority", default="medium", choices=config.PRIORITY_LEVELS)
    p_add.add_argument("--description", default="", help="Card description")
    p_add.add_argument("--tags", default="", help="Comma-separated tags")
    p_add.add_argument("--assignee", default="", help="Assignee")
    p_add.add_argument("--due", default="", help="Due date")

    # board move
    p_move = board_sub.add_parser("move", help="Move a card")
    p_move.add_argument("card_id", help="Card ID")
    p_move.add_argument("column", choices=config.COLUMNS, help="Target column")

    # board delete
    p_del_card = board_sub.add_parser("delete", help="Delete a card")
    p_del_card.add_argument("card_id", help="Card ID")

    # board stats
    board_sub.add_parser("stats", help="Show board statistics")

    # Sync subcommand
    sub.add_parser("sync", help="Sync notes to board")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    # Resolve directories
    notes_dir = Path(args.notes_dir) if args.notes_dir else config.default_notes_dir()
    board_dir = Path(args.board_dir) if args.board_dir else config.default_board_dir()
    backend = args.backend or config.default_backend(notes_dir=notes_dir)

    if args.command == "notes":
        return _handle_notes(args, notes_dir, backend)
    elif args.command == "board":
        return _handle_board(args, board_dir)
    elif args.command == "sync":
        return _handle_sync(notes_dir, board_dir, backend)

    return 1


def _handle_notes(args, notes_dir: Path, backend: str) -> int:
    from planner.core import NoteStore, STATUS_ICONS

    store = NoteStore(notes_dir=notes_dir, backend=backend)

    if args.notes_cmd is None:
        print("Usage: planner notes <command>")
        return 1

    if args.notes_cmd == "new":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        note = store.create(args.title, tags=tags, status=args.status,
                            sprint=args.sprint, gh=args.gh, body=args.body)
        sprint_tag = f" [{args.sprint}]" if args.sprint else ""
        print(f"Created: {note.short_id}  {note.title}{sprint_tag}")
        return 0

    if args.notes_cmd == "list":
        notes = store.list_notes(tag=args.tag, status=args.status,
                                 sprint=args.sprint, limit=args.limit,
                                 today=args.today)
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
                icon = STATUS_ICONS.get(n.status, "?")
                print(f"    {icon} {n.short_id}  {n.title}{tags_str}")
        print(f"\n  {len(notes)} note(s)")
        return 0

    if args.notes_cmd == "show":
        note = store.get(args.note_id)
        if note is None:
            print(f"Note not found: {args.note_id}")
            return 1
        if note.id != args.note_id:
            print(f"  (matched {note.id})")
        tags_str = ", ".join(note.tags) if note.tags else "none"
        sprint_str = f"\n  sprint: {note.sprint}" if note.sprint else ""
        gh_str = f"\n  gh: {note.gh}" if note.gh else ""
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

    if args.notes_cmd == "edit":
        kwargs = {}
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

    if args.notes_cmd in ("delete", "rm"):
        if store.delete(args.note_id):
            print(f"Deleted: {args.note_id}")
            return 0
        print(f"Note not found: {args.note_id}")
        return 1

    if args.notes_cmd == "search":
        results = store.search(args.query, limit=args.limit)
        if not results:
            print(f"No notes matching '{args.query}'")
            return 0
        for n in results:
            tags_str = f"  [{', '.join(n.tags)}]" if n.tags else ""
            print(f"    {n.short_id}  {n.title}{tags_str}")
        print(f"\n  {len(results)} result(s)")
        return 0

    return 1


def _handle_board(args, board_dir: Path) -> int:
    from planner.store import Store

    store = Store(board_dir=board_dir)

    if args.board_cmd is None:
        print("Usage: planner board <command>")
        return 1

    if args.board_cmd == "list":
        board = store.load_board()
        if not board.cards:
            print("No cards on the board.")
            return 0
        by_column: dict[str, list] = {}
        for card in board.cards:
            by_column.setdefault(card.column, []).append(card)
        for col in config.COLUMNS:
            cards = by_column.get(col, [])
            if not cards:
                continue
            label = config.COLUMN_LABELS.get(col, col)
            print(f"\n  {label} ({len(cards)})")
            for card in cards:
                priority_icon = {"low": "\u2193", "medium": "\u2192", "high": "\u2191", "urgent": "\u2191\u2191"}.get(card.priority, "?")
                print(f"    {priority_icon} {card.id[:8]}  {card.title}")
        print(f"\n  {len(board.cards)} card(s) total")
        return 0

    if args.board_cmd == "add":
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        card = store.create_card(
            title=args.title,
            column=args.column,
            priority=args.priority,
            description=args.description,
            tags=tags,
            assignee=args.assignee,
            dueDate=args.due,
        )
        print(f"Created: {card.id[:8]}  {card.title}")
        return 0

    if args.board_cmd == "move":
        if store.move_card(args.card_id, args.column):
            print(f"Moved {args.card_id[:8]} to {args.column}")
            return 0
        print(f"Card not found: {args.card_id}")
        return 1

    if args.board_cmd == "delete":
        if store.delete_card(args.card_id):
            print(f"Deleted: {args.card_id[:8]}")
            return 0
        print(f"Card not found: {args.card_id}")
        return 1

    if args.board_cmd == "stats":
        stats = store.get_stats()
        print(f"Total cards: {stats['total']}")
        print("By column:")
        for col, count in stats["byColumn"].items():
            label = config.COLUMN_LABELS.get(col, col)
            print(f"  {label}: {count}")
        print("By priority:")
        for pri, count in stats["byPriority"].items():
            print(f"  {pri}: {count}")
        return 0

    return 1


def _handle_sync(notes_dir: Path, board_dir: Path, backend: str) -> int:
    from planner.core import NoteStore
    from planner.store import Store
    from planner.sync import sync_notes_to_board

    note_store = NoteStore(notes_dir=notes_dir, backend=backend)
    board_store = Store(board_dir=board_dir)

    added, updated, total = sync_notes_to_board(note_store, board_store)
    print(f"{added} new card(s) added, {updated} moved, {total} total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
