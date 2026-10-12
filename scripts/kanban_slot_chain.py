#!/usr/bin/env python3
"""Kanban slot chain CLI — per-slot parent-linked hashes (never card content).

Core logic lives in ``app_planner.slot_chain`` — the store keeps the chain
alive with a single ``apply_board()`` line in ``_atomic_write``; this CLI is
the manual path (initial migration / drift repair) plus ``verify``, the gate.

Usage:
  scripts/kanban_slot_chain.py apply   [--board .kanban/board.jsonl]
  scripts/kanban_slot_chain.py verify  [--board .kanban/board.jsonl]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "app-planner" / "src"))

from app_planner.slot_chain import (  # noqa: E402
    SlotChainError,
    apply_board,
    journal_path,
    load_board,
    load_journal,
    verify_board,
)

# Every message shape verify_board emits, matched on a distinctive substring.
# An unrecognised message falls through to its own text and still gets its own
# row, so a newly added class cannot be silently bucketed away.
_CLASSES: tuple[tuple[str, str], ...] = (
    ("rehash mismatch", "journal node rehash mismatch"),
    ("non-empty parent", "journal first node has non-empty parent"),
    ("chain broken", "journal chain broken"),
    ("missing slot fields", "card missing slot fields (apply repairs)"),
    ("never journaled", "card slot never journaled (apply repairs)"),
    ("tip belongs to", "slot tip owned by a different card (apply repairs)"),
    ("notes changed since last apply", "card notes changed since apply (apply repairs)"),
    ("slot_position stale", "card slot_position stale (apply repairs)"),
    ("slot chain fields stale", "card slot chain fields stale (apply repairs)"),
    ("note_hashes out of sync", "card note_hashes out of sync with notes"),
    ("retired field", "card carries a retired field"),
    ("tray_roots", "header tray_roots != journal-derived roots"),
    ("slot_history missing", "header slot_history missing/wrong"),
)

_BREAK_AT = re.compile(r"chain broken at (\S+)$")


def _classify(problem: str) -> str:
    for needle, label in _CLASSES:
        if needle in problem:
            return label
    return problem if len(problem) <= 60 else problem[:57] + "..."


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _print_class_summary(failures: list[str]) -> None:
    """Group by root cause, not by count.

    The problem list is truncated at 20, and those 20 are whatever class sorts
    first — so this output showed 20 near-identical chain breaks and a bare
    ``... 1793 more``. Three fifths of the failures turned out to be a
    different class entirely, discoverable only by calling verify_board()
    directly and re-deriving the grouping by hand.
    """
    counts = Counter(_classify(f) for f in failures)
    if len(counts) < 2:
        return
    print("  by root cause:")
    width = max(len(label) for label in counts)
    for label, n in counts.most_common():
        print(f"    {n:6d}  {label:<{width}}")


def _chain_break_timeline(failures: list[str], board: str | Path) -> None:
    """Say whether chain breaks are residue or an open wound.

    A chain break is immutable history: the journal is append-only, so a fork
    written last week cannot un-happen. Yet ``verify`` printed it exactly like
    one written a second ago. The live board carried 788 forks left by
    overlapping ``apply_board`` passes (card d708578d, closed by 8ca177458),
    and all 788 still failed the gate with no date context — which cost a full
    re-diagnosis of a bug that was already fixed on this very branch.

    The distinguishing fact is free: the store appends a journal node on every
    board write, so if writes have kept landing without producing a new break,
    the breaks are residue rather than damage in progress. Report that, and
    nobody has to guess again.
    """
    hits = [m.group(1) for f in failures if (m := _BREAK_AT.search(f))]
    if not hits:
        return
    stamps = sorted(set(hits))
    latest = stamps[-1]
    latest_dt = _parse_ts(latest)
    try:
        nodes, _ = load_journal(journal_path(board))
    except OSError:
        return
    since = sum(
        1
        for n in nodes
        if latest_dt is not None
        and (n_dt := _parse_ts(n.get("ts", ""))) is not None
        and n_dt > latest_dt
    )
    print(f"  chain breaks: {len(hits)} across {len(stamps)} distinct write event(s)")
    print(f"    last break   : {latest}")
    if since:
        print(f"    journal since: {since} record(s) appended, none of them broke a chain")
        print(
            "    verdict      : residue — the journal is append-only, so a recorded\n"
            "                   fork cannot un-happen. `apply` repairs the other\n"
            "                   classes but cannot rewrite this history."
        )
    else:
        print("    journal since: nothing appended since the last break")
        print(
            "    verdict      : FRESH — no write has landed since, so this may\n"
            "                   still be accumulating. Inspect the writer."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["apply", "verify"])
    parser.add_argument("--board", default=".kanban/board.jsonl")
    args = parser.parse_args()
    try:
        if args.mode == "apply":
            stats = apply_board(args.board)
            print(
                "apply ok: cards={cards} nodes_created={nodes_created} "
                "nodes_reused={nodes_reused} tips={tips} trays={trays}".format(**stats)
            )
        else:
            failures = verify_board(args.board)
            if not failures:
                raws, objs, cards, meta_idx = load_board(args.board)
                nodes, tips = load_journal(journal_path(args.board))
                print(
                    f"verify ok: cards={len(cards)} journal_nodes={len(nodes)} "
                    f"slots={len(tips)} trays={len(objs[meta_idx].get('tray_roots', {}))}"
                )
                return
    except SlotChainError as exc:
        print(f"error: {exc}")
        sys.exit(2)
    if args.mode == "verify":
        if failures:
            print(f"verify FAIL: {len(failures)} problem(s)")
            _print_class_summary(failures)
            print(f"  first {min(len(failures), 20)}:")
            for problem in failures[:20]:
                print("  -", problem)
            if len(failures) > 20:
                print(f"  ... {len(failures) - 20} more")
            _chain_break_timeline(failures, args.board)
            sys.exit(1)


if __name__ == "__main__":
    main()
