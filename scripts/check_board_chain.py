#!/usr/bin/env python3
"""Board integrity gate: schema + uniqueness + hash-chain of .kanban/board.jsonl.

Card 20261029. Merge 5e5c8ce36 union-merged two board versions and landed on
main undetected: 975 duplicate card lines, 972 ids carrying two diverging
copies, the schema header pushed to EOF, and verify_chain broken at every one
of the 1030 chained indices. Nothing ran verify_chain against the REAL board
(only tmp-dir unit tests), so the corruption shipped silently and every later
board write compounded it.

This is that missing gate. Three checks, cheap -> deep:

  1. schema  — exactly one `planner/1` header, on the FIRST non-empty line;
  2. unique  — no duplicate card ids (the direct union-merge symptom);
  3. chain   — PlannerStore.verify_chain() == [] (content + order tamper-evident).

Usage::

    check_board_chain.py [path/to/board.jsonl]     # default: repo board

Exit 0 — board intact (or missing: nothing to guard).
Exit 1 — corruption found; diagnostics on stderr.

Env failures (app_planner unimportable) fail OPEN with a warning — the git
hook philosophy is "a wedged env must not wedge pushes"; CI is the hard gate
and its checkout always carries packages/app-planner/src next to this script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages" / "app-planner" / "src"))

try:
    from app_planner.store import PlannerStore
except Exception as exc:  # noqa: BLE001 — fail-open on env, never wedge a push
    print(f"check_board_chain: SKIPPED (app_planner unimportable: {exc!r})", file=sys.stderr)
    raise SystemExit(0) from exc

SCHEMA_NAME = "planner/1"


def check_board(board_file: Path) -> list[str]:
    """Return a list of corruption findings (empty = intact)."""
    if not board_file.exists():
        return []

    findings: list[str] = []
    try:
        raw_lines = [
            line for line in board_file.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
    except (OSError, UnicodeDecodeError) as exc:
        return [f"board unreadable: {exc!r}"]

    schema_lines: list[int] = []
    ids_seen: dict[str, int] = {}
    for lineno, raw in enumerate(raw_lines, start=1):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            findings.append(f"line {lineno}: unparseable JSON ({exc.msg})")
            continue
        if not isinstance(obj, dict):
            findings.append(f"line {lineno}: not a JSON object")
            continue
        if obj.get("schema") == SCHEMA_NAME:
            schema_lines.append(lineno)
            continue
        card_id = obj.get("id")
        if not card_id or "title" not in obj:
            findings.append(f"line {lineno}: neither schema header nor card object")
            continue
        if card_id in ids_seen:
            findings.append(
                f"line {lineno}: duplicate card id {card_id!r} "
                f"(first seen on line {ids_seen[card_id]})"
            )
        else:
            ids_seen[card_id] = lineno

    if not schema_lines:
        findings.append(f"missing `{SCHEMA_NAME}` schema header line")
    elif schema_lines[0] != 1:
        findings.append(
            f"schema header on line {schema_lines[0]}, expected line 1 "
            "(board was rewritten by a merge that appended instead of taking one side)"
        )
    if len(schema_lines) > 1:
        findings.append(f"schema header duplicated on lines {schema_lines}")

    # Deep check: the hash chain itself (also catches unsealed additions —
    # a card appended without compute_chains() has chain_index=-1 and fails).
    if not findings:  # chain verify on a structurally broken file is noise
        try:
            store = PlannerStore(board_dir=board_file.parent, notes_dir=board_file.parent)
            broken = store.verify_chain()
        except Exception as exc:  # noqa: BLE001 — report, don't crash the gate
            findings.append(f"chain verification errored: {exc!r}")
        else:
            if broken:
                shown = ", ".join(str(i) for i in broken[:8])
                suffix = ", ..." if len(broken) > 8 else ""
                findings.append(
                    f"hash chain broken at {len(broken)} card(s) [indices {shown}{suffix}] "
                    "(stale-base merge, duplicate rows, or an unsealed edit — "
                    "fix: re-seal with PlannerStore.compute_chains(), see card 20261029)"
                )
    return findings


def main(argv: list[str]) -> int:
    board_file = (
        Path(argv[1]) if len(argv) > 1 else REPO_ROOT / ".kanban" / "board.jsonl"
    ).resolve()
    findings = check_board(board_file)
    if not findings:
        return 0
    print(f"check_board_chain: FAIL — {board_file}", file=sys.stderr)
    for finding in findings[:20]:
        print(f"  {finding}", file=sys.stderr)
    if len(findings) > 20:
        print(f"  ... and {len(findings) - 20} more", file=sys.stderr)
    print(file=sys.stderr)
    print(
        "  This is the gate that would have caught merge 5e5c8ce36 (card 20261029).",
        file=sys.stderr,
    )
    print("  bypass: git push --no-verify   (CI will still red)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
