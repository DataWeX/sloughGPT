#!/usr/bin/env python3
"""
CI alarm: the local kanban sync producer must be alive.

The canonical store lives outside CI (see AGENTS.md architecture), so
``scripts/kanban_sync.sh`` (kanban-sync.timer) mirrors it into
``.kanban/board.jsonl`` + ``.kanban/manifest.json``. If that timer dies,
board history on GitHub goes stale silently. This check reddens the build
instead: manifest missing or ``exported_at`` older than the budget.

Stdlib only, no app imports — CI runs it on a stock python3.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / ".kanban" / "manifest.json"
DEFAULT_MAX_AGE_MINUTES = 120


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--max-age-minutes", type=int, default=DEFAULT_MAX_AGE_MINUTES)
    args = parser.parse_args(argv)

    if not args.manifest.is_file():
        print(
            f"FAIL: no {args.manifest} — the sync producer never ran "
            f"(scripts/kanban_sync.sh / kanban-sync.timer)"
        )
        return 1

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        exported = datetime.fromisoformat(manifest["exported_at"])
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        print(f"FAIL: unreadable manifest {args.manifest}: {exc}")
        return 1
    if exported.tzinfo is None:
        exported = exported.replace(tzinfo=UTC)

    age = datetime.now(UTC) - exported
    limit = timedelta(minutes=args.max_age_minutes)
    if age > limit:
        print(f"FAIL: kanban export is stale: age={age} > {limit} — kanban-sync.timer looks dead")
        return 1

    print(
        f"ok: kanban export {age} old "
        f"({manifest.get('card_count', '?')} cards, "
        f"{manifest.get('note_count', '?')} notes)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
