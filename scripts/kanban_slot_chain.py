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
import sys
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
            for problem in failures[:20]:
                print("  -", problem)
            if len(failures) > 20:
                print(f"  ... {len(failures) - 20} more")
            sys.exit(1)


if __name__ == "__main__":
    main()
