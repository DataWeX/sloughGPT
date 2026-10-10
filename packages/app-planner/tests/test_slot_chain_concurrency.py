"""Concurrency: ``apply_board`` must not fork the chain when sessions overlap.

Two writers that both derive a node from the same journal tail each append it,
leaving two records that claim the same parent — ``verify_board`` reports that
as ``journal: <key> chain broken``. The ``flock`` in ``append_lines`` does not
prevent it: it serializes bytes, but by the time it is taken the parent has
already been decided from a stale read.

The live journal recorded 788 such forks between 2026-10-04 and 2026-10-10
(card d708578d), because ``store._atomic_write`` runs ``apply_board`` on *every*
board write and a single pass already spans ~1s — so overlapping passes are the
norm, not a coincidence.

The window here is widened deterministically instead of being left to the
scheduler: each child stalls between reading the tail and appending to it,
which is exactly the interval the fix closes. Without that stall these tests
pass on the unfixed code too, since the race is then merely rare.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import app_planner
from app_planner import atomic
from app_planner.slot_chain import apply_board, journal_path, verify_board

HEADER = {
    "schema": "planner/1",
    "name": "Main",
    "columns": [
        {"name": "todo", "wip_limit": 5, "order": 0},
        {"name": "in_progress", "wip_limit": 3, "order": 1},
    ],
}

# Reads the tail, stalls inside the read→derive→append window, then applies.
# The stall is what makes the race reproducible: without it every child would
# append before the next one had even started.
CHILD = """
import sys, time
from pathlib import Path

from app_planner import slot_chain as sc

_orig = sc.load_journal
_reads = {"n": 0}


def _slow(path):
    result = _orig(path)
    _reads["n"] += 1
    if _reads["n"] == 1:
        time.sleep(float(sys.argv[3]))
    return result


sc.load_journal = _slow

board = Path(sys.argv[1])
print("READY", flush=True)
sys.stdin.readline()  # released by the parent once every child is up
for _ in range(int(sys.argv[2])):
    sc.apply_board(board, notes_home=sys.argv[4])
"""


def _write_board(path: Path, card_ids: list[str]) -> None:
    lines = [HEADER]
    lines += [
        {"id": cid, "title": cid, "column": "todo", "priority": "medium", "tags": [], "notes": []}
        for cid in card_ids
    ]
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")


def _chain_breaks(failures: list[str]) -> list[str]:
    return [f for f in failures if "chain broken" in f]


# ── append_lines_checked: the two arms of the contract ───────────────────


def test_checked_append_writes_snapshot_when_tail_is_unchanged(tmp_path):
    journal = tmp_path / "j.jsonl"
    atomic.append_lines(journal, ['{"n": 1}\n'])
    size = journal.stat().st_size

    rebased: list[int] = []
    atomic.append_lines_checked(
        journal,
        ['{"n": 2}\n'],
        size,
        lambda: rebased.append(1) or ['{"n": 3}\n'],
    )

    assert rebased == [], "tail did not move, so rebase must not run"
    assert journal.read_text(encoding="utf-8").splitlines()[-1] == '{"n": 2}'


def test_checked_append_discards_snapshot_when_tail_moved(tmp_path):
    journal = tmp_path / "j.jsonl"
    atomic.append_lines(journal, ['{"n": 1}\n'])
    size = journal.stat().st_size
    # Another writer lands after we read the tail...
    atomic.append_lines(journal, ['{"n": 2}\n'])

    rebased: list[int] = []
    atomic.append_lines_checked(
        journal,
        ['{"n": 99}\n'],  # ...so the payload we derived from that read is stale
        size,
        lambda: rebased.append(1) or ['{"n": 3}\n'],
    )

    assert rebased == [1], "tail moved, so rebase must run under the lock"
    lines = journal.read_text(encoding="utf-8").splitlines()
    assert lines == ['{"n": 1}', '{"n": 2}', '{"n": 3}'], "stale payload must not be written"


def test_checked_append_creates_missing_journal(tmp_path):
    journal = tmp_path / "nested" / "j.jsonl"
    atomic.append_lines_checked(journal, ['{"n": 1}\n'], 0, lambda: [])
    assert journal.read_text(encoding="utf-8") == '{"n": 1}\n'


# ── end to end: concurrent apply_board ────────────────────────────────────


def test_concurrent_apply_board_does_not_fork_the_chain(tmp_path):
    board = tmp_path / "board.jsonl"
    notes = tmp_path / "notes"
    notes.mkdir()
    _write_board(board, ["c1", "c2", "c3"])
    # A cold journal is the harshest case: every child derives the same nodes
    # from the same empty tail, so an unlocked read produces one fork per slot.
    assert not journal_path(board).exists()

    child = tmp_path / "child.py"
    child.write_text(CHILD, encoding="utf-8")
    src = str(Path(app_planner.__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": src, "PYTHONNOUSERSITE": "1"}

    procs = [
        subprocess.Popen(
            [sys.executable, str(child), str(board), "1", "0.5", str(notes)],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    try:
        # Barrier: release every child at once so all four read the same tail
        # and then stall together inside the vulnerable window.
        for proc in procs:
            ready = proc.stdout.readline().strip()
            assert ready == "READY", f"child did not start: {ready!r}"
        for proc in procs:
            proc.stdin.write("\n")
            proc.stdin.flush()
        for proc in procs:
            _, err = proc.communicate(timeout=60)
            assert proc.returncode == 0, f"child failed: {err}"
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()

    nodes = _load_nodes(board)
    breaks = _chain_breaks(verify_board(board, notes_home=notes))
    assert breaks == [], f"{len(breaks)} chain forks under concurrent apply: {breaks[:3]}"

    # One node per slot, not one per child: the losers re-derived instead of
    # appending a duplicate parent.
    slots = [n["slot_key"] for n in nodes]
    assert len(slots) == len(set(slots)), f"duplicate slot nodes written: {slots}"


def test_sequential_apply_still_reuses_nodes(tmp_path):
    """The lock must not break idempotence: a second pass writes nothing."""
    board = tmp_path / "board.jsonl"
    notes = tmp_path / "notes"
    notes.mkdir()
    _write_board(board, ["c1", "c2"])

    first = apply_board(board, notes_home=notes)
    before = journal_path(board).stat().st_size
    second = apply_board(board, notes_home=notes)

    assert first["nodes_created"] == 2
    assert second["nodes_created"] == 0 and second["nodes_reused"] == 2
    assert journal_path(board).stat().st_size == before, "re-apply must append nothing"
    assert verify_board(board, notes_home=notes) == []


def _load_nodes(board: Path) -> list[dict]:
    """Every record in the journal, in file order."""
    journal = journal_path(board)
    if not journal.exists():
        return []
    return [
        json.loads(raw) for raw in journal.read_text(encoding="utf-8").splitlines() if raw.strip()
    ]
