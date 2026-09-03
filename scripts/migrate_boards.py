#!/usr/bin/env python3
"""
Migrate the legacy board to the current JSONL card schema and write it back
to JSON.

Reads the old ``.kanban/board.json`` (snake_case card fields) and the existing
``.kanban/board.jsonl`` (mixed note-derived + new-format lines), normalizes
every card to the ``planner.store.Card`` schema (``dueDate``/``createdAt``/
``updatedAt``, ``column``, ``body`` folded into ``description``,
``completed`` status -> ``done`` column), merges both sources by title, writes
the consolidated board to ``board.jsonl`` (one card per line), and writes the
same board back to ``board.json`` as JSON so the human-readable file matches.

No files are deleted and the original board is preserved via git. Run with
``--dry-run`` to preview the merge without writing.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# New card schema field ordering (mirrors planner.store.Card).
NEW_FIELDS = [
    "id", "title", "description", "column", "priority", "tags",
    "assignee", "dueDate", "createdAt", "updatedAt", "root_hash", "notes",
]

# Maps a source status/column value to a board column.
STATUS_TO_COLUMN = {
    "done": "done",
    "completed": "done",
    "wip": "in_progress",
    "in_progress": "in_progress",
    "review": "review",
    "todo": "todo",
    "open": "todo",
    "blocked": "todo",
    "": "todo",
    None: "todo",
}

PRIORITY_LEVELS = ("critical", "high", "medium", "low")


def _slugify(title: str) -> str:
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug.strip("-")[:80]


def normalize(raw: dict) -> dict:
    """Fold one raw card (legacy json or old jsonl line) into the new schema."""
    status = raw.get("status") or ("completed" if raw.get("completed") else "")
    col = raw.get("column") or STATUS_TO_COLUMN.get(str(status).lower())

    description = raw.get("description") or raw.get("body") or ""
    tags = raw.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    priority = raw.get("priority") or "medium"
    if priority not in PRIORITY_LEVELS:
        priority = "medium"

    notes = raw.get("notes") or raw.get("comments") or []

    return {
        "id": raw.get("id") or _slugify(raw.get("title") or "untitled"),
        "title": raw.get("title") or "(untitled)",
        "description": description,
        "column": col,
        "priority": priority,
        "tags": list(tags),
        "assignee": raw.get("assignee") or "",
        "dueDate": raw.get("dueDate") or raw.get("due_date") or "",
        "createdAt": raw.get("createdAt") or raw.get("created_at") or "",
        "updatedAt": raw.get("updatedAt") or raw.get("updated_at") or "",
        "root_hash": raw.get("root_hash") or "",
        "notes": notes,
    }


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    cards = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue  # skip any legacy header line
        data = json.loads(line)
        if "title" in data:
            cards.append(data)
    return cards


def migrate(board_dir: Path, dry_run: bool = False) -> dict:
    board_json = board_dir / "board.json"
    board_jsonl = board_dir / "board.jsonl"

    columns = [
        {"name": "todo", "wip_limit": 0, "order": 0},
        {"name": "in_progress", "wip_limit": 3, "order": 1},
        {"name": "review", "wip_limit": 0, "order": 2},
        {"name": "done", "wip_limit": 0, "order": 3},
    ]

    # Source 1: legacy board.json (ordered).
    json_cards: list[dict] = []
    name = "board"
    if board_json.exists():
        data = json.loads(board_json.read_text())
        name = data.get("name", "board")
        raw_columns = data.get("columns")
        if raw_columns:
            columns = [
                {
                    "name": c.get("name"), "wip_limit": c.get("wip_limit", 0),
                    "order": c.get("order", i),
                }
                for i, c in enumerate(raw_columns)
            ]
        json_cards = [normalize(c) for c in data.get("cards", [])]

    # Source 2: existing jsonl cards.
    jsonl_cards = [normalize(c) for c in load_jsonl(board_jsonl)]

    # Merge by title (case-sensitive), preferring the newer jsonl entry.
    # A repeated title keeps the first record and drops later duplicates.
    title_idx: dict[str, int] = {}
    merged: list[dict] = []
    from_jsonl_added = 0
    for card in json_cards:
        if card["title"] in title_idx:
            continue
        title_idx[card["title"]] = len(merged)
        merged.append(card)
    for card in jsonl_cards:
        if card["title"] in title_idx:
            merged[title_idx[card["title"]]] = card
        else:
            title_idx[card["title"]] = len(merged)
            merged.append(card)
            from_jsonl_added += 1

    if dry_run:
        return {
            "name": name, "columns": columns, "cards": merged,
            "from_json": len(json_cards),
            "from_jsonl": len(jsonl_cards),
            "jsonl_new": from_jsonl_added,
            "rows": len(merged),
        }

    lines = [json.dumps(c, ensure_ascii=False) for c in merged]
    board_dir.mkdir(parents=True, exist_ok=True)
    board_jsonl.write_text("\n".join(lines) + "\n" if lines else "")

    board_doc = {"name": name, "columns": columns, "cards": merged}
    board_json.write_text(json.dumps(board_doc, ensure_ascii=False, indent=2) + "\n")

    return {
        "name": name, "columns": columns, "cards": merged,
        "from_json": len(json_cards),
        "from_jsonl": len(jsonl_cards),
        "jsonl_new": from_jsonl_added,
        "rows": len(merged),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="migrate_boards",
        description="Migrate legacy board.json + board.jsonl to the new card schema.",
    )
    parser.add_argument("--board-dir", default=".kanban", help="Board directory with board.json/board.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Preview the merge without writing")
    parser.add_argument("--json", dest="json_out", action="store_true", help="Emit machine-readable stats")
    args = parser.parse_args()

    stats = migrate(Path(args.board_dir), dry_run=args.dry_run)

    if args.json_out:
        print(json.dumps({k: v for k, v in stats.items() if k != "cards"}, indent=2))
        return

    mode = "dry run" if args.dry_run else "migrated"
    print(f"Boards {mode}:")
    print(f"  cards from board.json : {stats['from_json']}")
    print(f"  cards from board.jsonl: {stats['from_jsonl']}")
    print(f"  board.jsonl-only cards: {stats['jsonl_new']}")
    print(f"  merged card rows      : {stats['rows']}")
    if not args.dry_run:
        print("  wrote board.jsonl and board.json (new card schema)")


if __name__ == "__main__":
    main()
