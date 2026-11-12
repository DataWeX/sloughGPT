#!/usr/bin/env python3
"""Work history: record / history / compare — score card-close outcomes.

    python scripts/work_history.py record    # snapshot the board's done/queue state
    python scripts/work_history.py history   # list snapshots
    python scripts/work_history.py compare   # last two snapshots, scalar deltas

Card 20261112_064224 (minimal build, user ruling 2026-11-12 "minimal and
focused"): derives from EXISTING artifacts only — ~/.config/kanban/board.jsonl
(done cards' CLOSED evidence + created/updated stamps). No git source, no
telemetry, nothing written near notification surfaces. Snapshots are stamped
JSON under ~/.local/share/sloughgpt/work_history (override: SLO_WORK_HISTORY_DIR).

Metrics score OUTCOMES, not raw activity: closures with evidence, closures
whose triage reached "zero attributable" (the three-strike verdict — scoring
raw reds would punish honest RED-finding), queue size, creation->close latency.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

DEFAULT_BOARD = Path.home() / ".config" / "kanban" / "board.jsonl"
DEFAULT_OUT = Path(
    os.environ.get("SLO_WORK_HISTORY_DIR")
    or Path.home() / ".local" / "share" / "sloughgpt" / "work_history"
)
QUEUE_COLUMNS = ("todo", "wip", "review")
_ZERO_ATTR = re.compile(r"zero[- ]attributable", re.IGNORECASE)


def snapshot_from_board(board_path: Path | str) -> dict:
    """One deterministic snapshot of board state (recorded_at added by record)."""
    queue = dict.fromkeys(QUEUE_COLUMNS, 0)
    closed = {"total": 0, "with_evidence": 0, "zero_attributable": 0}
    reopened = 0
    latencies: list[float] = []

    for line in Path(board_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            card = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(card, dict):
            continue
        column = card.get("column")
        if column in QUEUE_COLUMNS:
            queue[column] += 1
            continue
        if column != "done":
            continue
        desc = card.get("description") or ""
        closed["total"] += 1
        if "CLOSED " in desc:
            closed["with_evidence"] += 1
        if _ZERO_ATTR.search(desc):
            closed["zero_attributable"] += 1
        if "reopen" in desc.lower():
            reopened += 1
        try:
            created = datetime.fromisoformat(card["created_at"])
            updated = datetime.fromisoformat(card["updated_at"])
            latencies.append((updated - created).total_seconds())
        except (KeyError, TypeError, ValueError):
            pass

    avg = sum(latencies) / len(latencies) if latencies else None
    return {
        "schema": 1,
        "closed": closed,
        "queue": queue,
        "avg_close_latency_s": avg,
        "reopened": reopened,
    }


def _snapshots(out_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(out_dir.glob("*.json"))]


def compare(snaps: list[dict]) -> list[str]:
    """Scalar deltas between the last two snapshots."""
    old, new = snaps[-2], snaps[-1]
    rows: list[tuple[str, float, float]] = [
        ("closed", old["closed"]["total"], new["closed"]["total"]),
        ("with_evidence", old["closed"]["with_evidence"], new["closed"]["with_evidence"]),
        (
            "zero_attributable",
            old["closed"]["zero_attributable"],
            new["closed"]["zero_attributable"],
        ),
        ("todo", old["queue"]["todo"], new["queue"]["todo"]),
        ("wip", old["queue"]["wip"], new["queue"]["wip"]),
    ]
    out = []
    for name, a, b in rows:
        delta = int(b - a)
        out.append(f"{name}: {a} -> {b} ({delta:+d})")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_r = sub.add_parser("record", help="snapshot current board state")
    p_r.add_argument("--board", default=str(DEFAULT_BOARD))
    p_r.add_argument("--out", default=str(DEFAULT_OUT))
    p_h = sub.add_parser("history", help="list snapshots")
    p_h.add_argument("--out", default=str(DEFAULT_OUT))
    p_c = sub.add_parser("compare", help="deltas between the last two snapshots")
    p_c.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    out = Path(args.out)
    if args.cmd == "record":
        snap = snapshot_from_board(args.board)
        snap["recorded_at"] = datetime.now(UTC).isoformat()
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f") + "Z"
        (out / f"{stamp}.json").write_text(json.dumps(snap, indent=2, sort_keys=True) + "\n")
        c, q = snap["closed"], snap["queue"]
        print(
            f"{stamp} closed={c['total']} evidence={c['with_evidence']} "
            f"todo={q['todo']} wip={q['wip']}"
        )
        return 0
    if args.cmd == "history":
        for snap in _snapshots(out):
            c, q = snap["closed"], snap["queue"]
            print(
                f"{snap['recorded_at']} closed={c['total']} evidence={c['with_evidence']} "
                f"todo={q['todo']} wip={q['wip']}"
            )
        return 0
    # compare
    snaps = _snapshots(out)
    if len(snaps) < 2:
        print(f"need at least 2 snapshots in {out} (have {len(snaps)})")
        return 1
    for line in compare(snaps):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
