#!/usr/bin/env python3
"""Pre-push board guard: catch done-row loss from stale-base board merges.

Card 6179df70. Recurring class (orig. b17dfec2d; 5 recurrences 2026-10-10):
a lane commits .kanban/board.jsonl from a stale base, the merge keeps their
copy of the row, and a card whose work LANDED silently reverts done -> todo.

Called by githooks/pre-push once per pushed ref range:

    check_board_reopen.py <base_sha> <local_sha>

Exit 0 — no card left `done`, or every such card is justified.
Exit 1 — at least one card moved done -> <other> (or its row vanished)
         without a justification marker in the pushed commit messages.

Justification (legitimate reopens): mention `board-reopen: <id-prefix>`
in one of the commits in <base_sha>..<local_sha>.

Fail-open on parse errors: this is a tripwire, CI and humans stay the
source of truth.
"""
from __future__ import annotations

import json
import subprocess
import sys


def _board_at(sha: str) -> dict[str, tuple[str, str]]:
    """id -> (column, title) at sha; last row wins for duplicate ids."""
    res = subprocess.run(
        ["git", "show", f"{sha}:.kanban/board.jsonl"],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        return {}
    cards: dict[str, tuple[str, str]] = {}
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "column" in obj and "id" in obj:
            cards[str(obj["id"])] = (
                str(obj.get("column", "")),
                str(obj.get("title", "") or ""),
            )
    return cards


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: check_board_reopen.py <base_sha> <local_sha>", file=sys.stderr)
        return 0  # fail-open on misuse
    base, local = argv[1], argv[2]
    try:
        before = _board_at(base)
        after = _board_at(local)
        if not before or not after:
            return 0  # board missing on either side — nothing to guard
        lost = [
            (cid, title, col_before)
            for cid, (col_before, title) in before.items()
            if col_before == "done"
            and after.get(cid, ("<gone>", ""))[0] != "done"
        ]
        if not lost:
            return 0
        msgs = subprocess.run(
            ["git", "log", "--format=%B", f"{base}..{local}"],
            capture_output=True,
            text=True,
        ).stdout
        unjustified = [
            (cid, title, after.get(cid, ("<gone>", "?"))[0])
            for cid, title, _ in lost
            if f"board-reopen: {cid[:8]}" not in msgs
            and f"board-reopen: {cid[:14]}" not in msgs
        ]
        if not unjustified:
            return 0
        print(
            "pre-push: BLOCKED — cards left the `done` column in this range"
            " (stale-base board merges silently revert landed work; card 6179df70):",
            file=sys.stderr,
        )
        for cid, title, col_now in unjustified:
            print(f"  {cid[:14]}  done -> {col_now}  | {title[:60]}", file=sys.stderr)
        print(file=sys.stderr)
        print("  If this is a LEGITIMATE reopen, add to one of the pushed commits:", file=sys.stderr)
        print("      board-reopen: <id-prefix>   (then amend/re-commit)", file=sys.stderr)
        print("  If a lane's stale-base board merge clobbered these rows, rebase the", file=sys.stderr)
        print("  board file onto origin/main instead of overwriting it.", file=sys.stderr)
        print("  bypass: git push --no-verify", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — tripwire must never wedge pushes
        print(f"pre-push: board guard skipped ({exc!r})", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
