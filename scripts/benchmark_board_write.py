"""Board-write benchmark — cost of the OCC guard on PlannerStore JSONL writes.

What it measures:
  1. single-process add throughput (ops/s): the UNCONTENDED OCC path —
     sha256 generation token + non-blocking flock barrier per card add.
  2. single-process note-create throughput (ops/s): the notes-journal path
     (same guard, previously a bare non-atomic write_text).
  3. multi-process add hammer (aggregate ops/s under real contention):
     N cards from --workers processes against one board file. Includes an
     INTEGRITY check — every card must survive on disk. A lost card is a
     clobber regression (the bug this guard exists for) and fails the run.

No regression baseline exists yet for board writes; run this on main before
an OCC/store change and after to compare ops/s and integrity.

Usage:
    python scripts/benchmark_board_write.py [--ops 200] [--workers 8] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "packages/app-planner/src"))

from app_planner.store import PlannerStore  # noqa: E402


def _add_burst(payload: tuple[str, str, int, int]) -> int:
    board_dir, notes_dir, count, seed = payload
    store = PlannerStore(board_dir=Path(board_dir), notes_dir=Path(notes_dir))
    for i in range(count):
        store.add_card(f"bench-{seed}-{i}", column="todo")
    return count


def bench_single_process(ops: int) -> dict[str, float]:
    with tempfile.TemporaryDirectory() as d:
        store = PlannerStore(board_dir=Path(d) / "board", notes_dir=Path(d) / "notes")
        t0 = time.perf_counter()
        for i in range(ops):
            store.add_card(f"single-{i}", column="todo")
        add_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        for i in range(ops):
            store.create_note(f"note-{i}", status="open")
        note_s = time.perf_counter() - t0

        survived = len(store.load_board().cards)
        assert survived == ops, f"single-process lost {ops - survived} cards!"
        return {
            "add_ops_per_s": round(ops / add_s, 1),
            "note_ops_per_s": round(ops / note_s, 1),
        }


def bench_multi_process(total: int, workers: int) -> dict[str, float | int]:
    per = max(1, total // workers)
    with tempfile.TemporaryDirectory() as d:
        board, notes = str(Path(d) / "board"), str(Path(d) / "notes")
        payloads = [(board, notes, per, w) for w in range(workers)]
        t0 = time.perf_counter()
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            list(ex.map(_add_burst, payloads))
        wall = time.perf_counter() - t0

        store = PlannerStore(board_dir=Path(board), notes_dir=Path(notes))
        survived = len(store.load_board().cards)
        expected = per * workers
        if survived != expected:  # clobber regression: fail loudly
            raise SystemExit(
                f"[FAIL] {expected - survived} cards lost under {workers}-way "
                f"contention ({survived}/{expected} survived)"
            )
        return {
            "aggregate_ops_per_s": round(expected / wall, 1),
            "wall_s": round(wall, 3),
            "cards_survived": survived,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", type=int, default=200, help="ops per single-process phase")
    parser.add_argument("--workers", type=int, default=8, help="hammer process count")
    parser.add_argument("--json", type=Path, default=None, help="write metrics JSON")
    args = parser.parse_args()

    single = bench_single_process(args.ops)
    multi = bench_multi_process(args.ops, args.workers)

    print("board-write benchmark (OCC PlannerStore)")
    print(f"  single-process add      : {single['add_ops_per_s']:>9.1f} ops/s ({args.ops} adds)")
    print(f"  single-process note     : {single['note_ops_per_s']:>9.1f} ops/s ({args.ops} creates)")
    print(
        f"  {args.workers}-process hammer       : "
        f"{multi['aggregate_ops_per_s']:>9.1f} ops/s aggregate "
        f"(wall {multi['wall_s']}s, integrity {multi['cards_survived']}/{args.workers}×{args.ops // args.workers} ok)"
    )

    if args.json:
        args.json.write_text(json.dumps({"single": single, "multi": multi}, indent=2))
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
