#!/usr/bin/env python3
"""Session-trajectory exporter: OpenCode sessions → filtered chat-sft JSONL.

Card 20261123_135823 (the learner feeds the executor's domain): reads this
repo's OpenCode sessions through the documented API (``opencode api``),
extracts plan → tool_calls → observation → synthesis turns, applies the
step-2 seed filter discipline, and writes chat-sft adapter records
(``{messages, source: "session", split}``) for ``chat_sft_adapter.build``.

Usage (project env, repo root):

    ./scripts/python scripts/export_session_trajectories.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from domain.training._internal.session_exporter import (
    assign_split,
    export_session,
    extract_trajectories,
    filter_records,
    is_repo_session,
    list_sessions,
    repo_roots,
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-dir",
        default=None,
        help="session directory filter (default: this repo's checkout family — "
        "main worktree + every linked worktree)",
    )
    parser.add_argument(
        "--out",
        default=str(repo_root / "tests" / "fixtures" / "session_trajectories.jsonl"),
        help="output JSONL path (chat-sft adapter records)",
    )
    args = parser.parse_args()

    session_filter = args.repo_dir or repo_roots()
    sessions = list_sessions()
    ours = [s for s in sessions if is_repo_session(s, session_filter)]
    print(f"sessions: {len(ours)} in repo of {len(sessions)} total")

    records: list[dict] = []
    totals = {
        "total": 0,
        "no_tool": 0,
        "no_synthesis": 0,
        "refusal": 0,
        "degenerate": 0,
        "duplicate": 0,
        "kept": 0,
    }
    failures: list[tuple[str, str]] = []
    for info in ours:
        session_id = str(info.get("id") or "")
        try:
            messages = export_session(session_id)
            kept, stats = filter_records(extract_trajectories(session_id, messages))
        except Exception as exc:  # one broken session must not fudge the report
            failures.append((session_id, str(exc)))
            continue
        for key in totals:
            totals[key] += stats[key]
        for record in kept:
            record["split"] = assign_split(record["id"])
            records.append(record)

    if failures:
        print(f"FAILED to export {len(failures)} session(s) — nothing written:")
        for session_id, error in failures:
            print(f"  {session_id}: {error}")
        return 1

    records.sort(key=lambda r: r["id"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))

    train = sum(1 for r in records if r["split"] == "train")
    heldout = len(records) - train
    print(f"candidates: {totals['total']} turns across {len(ours)} sessions")
    print(
        "dropped:   "
        + ", ".join(
            f"{key}={totals[key]}"
            for key in ("no_tool", "no_synthesis", "refusal", "degenerate", "duplicate")
        )
    )
    print(f"kept:      {len(records)} records (train={train}, heldout={heldout})")
    print(f"wrote:     {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
