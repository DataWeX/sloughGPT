#!/usr/bin/env python3
"""
Migrate planner to single JSONL store.

Reads board.json + ~/.config/dev-notes/*.md, merges into board.jsonl.
Cards absorb note fields (sprint, gh). Dev-notes files are deleted after migration.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path


def parse_note_file(filepath: Path) -> dict | None:
    """Parse a YAML frontmatter + markdown note file."""
    content = filepath.read_text()
    match = re.match(r"^---\n([\s\S]*?)\n---\n([\s\S]*)$", content)
    if not match:
        return None
    yaml_body, markdown_body = match.group(1), match.group(2)
    meta: dict[str, str] = {}
    for line in yaml_body.split("\n"):
        colon = line.index(":") if ":" in line else -1
        if colon > 0:
            key = line[:colon].strip()
            val = line[colon + 1:].strip()
            meta[key] = val
    tags_raw = meta.get("tags", "")
    tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else []
    return {
        "title": meta.get("title", filepath.stem),
        "tags": tags,
        "status": meta.get("status", "open"),
        "sprint": meta.get("sprint", ""),
        "gh": meta.get("gh", ""),
        "body": markdown_body.strip(),
        "created_at": meta.get("created", ""),
        "updated_at": meta.get("updated", ""),
    }


def migrate(repo_root: Path, dry_run: bool = False) -> dict:
    """
    Migrate board.json + dev-notes -> board.jsonl.

    Returns migration stats.
    """
    kanban_dir = repo_root / ".kanban"
    board_json = kanban_dir / "board.json"
    board_jsonl = kanban_dir / "board.jsonl"

    # Resolve notes directory
    notes_dir = Path(os.environ.get("PLANNER_NOTES_DIR", ""))
    if not notes_dir.exists():
        notes_dir = repo_root / ".dev-notes"
    if not notes_dir.exists():
        notes_dir = Path.home() / ".config" / "dev-notes"

    stats = {"cards_from_json": 0, "cards_from_notes": 0, "notes_merged": 0, "notes_deleted": 0}

    # Load existing board
    existing_cards = []
    columns = [
        {"name": "todo", "wip_limit": 0, "order": 0},
        {"name": "in_progress", "wip_limit": 3, "order": 1},
        {"name": "review", "wip_limit": 0, "order": 2},
        {"name": "done", "wip_limit": 0, "order": 3},
    ]

    if board_json.exists():
        data = json.loads(board_json.read_text())
        columns = data.get("columns", columns)
        for card in data.get("cards", []):
            # Normalize: rename "notes" -> "comments"
            if "notes" in card and "comments" not in card:
                card["comments"] = card.pop("notes")
            elif "notes" in card:
                card.pop("notes")
            # Add new fields if missing
            card.setdefault("sprint", "")
            card.setdefault("gh", "")
            existing_cards.append(card)
        stats["cards_from_json"] = len(existing_cards)

    # Build title -> card index
    title_to_idx = {c["title"]: i for i, c in enumerate(existing_cards)}

    # Load and merge dev-notes
    if notes_dir.exists():
        note_files = sorted(notes_dir.glob("*.md"))
        for nf in note_files:
            note = parse_note_file(nf)
            if note is None:
                continue
            title = note["title"]
            STATUS_MAP = {
                "done": "done", "wip": "in_progress", "review": "review",
                "todo": "todo", "open": "todo", "blocked": "todo", "": "todo",
            }
            col = STATUS_MAP.get(note["status"].lower(), "todo")

            if title in title_to_idx:
                # Update existing card with note data
                card = existing_cards[title_to_idx[title]]
                if note["body"] and not card.get("description"):
                    card["description"] = note["body"]
                if note["tags"] and not card.get("tags"):
                    card["tags"] = note["tags"]
                if note["sprint"]:
                    card["sprint"] = note["sprint"]
                if note["gh"]:
                    card["gh"] = note["gh"]
                if card.get("column", "todo") == "todo" and col != "todo":
                    card["column"] = col
                stats["notes_merged"] += 1
            else:
                # Create new card from note
                import re as _re
                slug = title.lower().strip()
                slug = _re.sub(r"[^\w\s-]", "", slug)
                slug = _re.sub(r"[\s_]+", "-", slug)
                slug = _re.sub(r"-+", "-", slug)[:60].rstrip("-")
                now = note.get("created_at") or note.get("updated_at") or ""
                card_id = f"{now.replace('-', '').replace(':', '').replace('T', '')[:15]}_{slug}" if now else f"migration_{slug}"
                existing_cards.append({
                    "id": card_id,
                    "title": title,
                    "description": note["body"],
                    "column": col,
                    "priority": "medium",
                    "tags": note["tags"],
                    "created_at": note["created_at"],
                    "updated_at": note["updated_at"],
                    "due_date": "",
                    "assignee": "",
                    "sprint": note["sprint"],
                    "gh": note["gh"],
                    "comments": [],
                })
                stats["cards_from_notes"] += 1

    # Write board.jsonl
    if not dry_run:
        kanban_dir.mkdir(parents=True, exist_ok=True)
        lines = []
        header = {"schema": "planner/1", "name": "board", "columns": columns}
        lines.append(json.dumps(header))
        for card in existing_cards:
            lines.append(json.dumps(card, default=str))
        board_jsonl.write_text("\n".join(lines) + "\n")

        # Backup and remove board.json
        if board_json.exists():
            shutil.move(str(board_json), str(board_json.with_suffix(".json.bak")))

        # Delete dev-notes files
        if notes_dir.exists():
            for nf in notes_dir.glob("*.md"):
                nf.unlink()
                stats["notes_deleted"] += 1
            # Remove empty dir
            if not any(notes_dir.iterdir()):
                notes_dir.rmdir()

    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Migrate planner to JSONL single store")
    parser.add_argument("--repo-root", default=".", help="Repository root")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    print(f"Migrating planner to JSONL single store...")
    print(f"  Repo root: {repo}")

    stats = migrate(repo, dry_run=args.dry_run)

    print(f"\nMigration {'(dry run) ' if args.dry_run else ''}complete:")
    print(f"  Cards from board.json: {stats['cards_from_json']}")
    print(f"  Cards created from notes: {stats['cards_from_notes']}")
    print(f"  Notes merged into existing cards: {stats['notes_merged']}")
    print(f"  Note files deleted: {stats['notes_deleted']}")


if __name__ == "__main__":
    main()
