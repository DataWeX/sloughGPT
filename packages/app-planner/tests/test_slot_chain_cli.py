"""The ``kanban_slot_chain.py`` CLI must run, and must date its verdicts.

Two things were missing and both cost real time:

* Nothing invoked the script end-to-end, so a typo in a module-level name
  (``_BROKEN_AT`` defined, ``_BREAK_AT`` called) shipped as a working-looking
  diff and only surfaced as a ``NameError`` on the first real run. Importing
  ``app_planner.slot_chain`` does not exercise it — the script is its own
  entry point.
* ``verify`` printed chain breaks with no date context at all, so 788 forks
  left by the concurrency race closed in ``8ca177458`` (card d708578d) read
  as a live fault and were re-diagnosed from scratch on a branch that already
  contained the fix. The journal is append-only, so a recorded fork cannot
  un-happen; what distinguishes residue from damage in progress is whether
  writes have kept landing without producing a new break.

These tests fork a journal the way the race did, then assert the CLI both
survives and reports which side of that line it is on.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from app_planner.slot_chain import (
    apply_board,
    journal_path,
    load_journal,
    node_hash,
    sha256,
)

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "kanban_slot_chain.py"

HEADER = {
    "schema": "planner/1",
    "name": "Main",
    "columns": [
        {"name": "todo", "wip_limit": 5, "order": 0},
        {"name": "in_progress", "wip_limit": 3, "order": 1},
    ],
}


def _card(card_id: str, title: str, column: str = "todo") -> dict:
    return {
        "id": card_id,
        "title": title,
        "column": column,
        "priority": "medium",
        "tags": [],
        "notes": [],
        "created_at": "2026-10-02T00:00:00+00:00",
        "updated_at": "2026-10-02T00:00:00+00:00",
    }


def _write_board(path: Path, cards: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(obj) for obj in [HEADER, *cards]) + "\n", encoding="utf-8")


def _run(board: Path, mode: str = "verify") -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), mode, "--board", str(board)],
        capture_output=True,
        text=True,
        timeout=180,
    )


def _fork_journal(board: Path, ts: str) -> str:
    """Append a node claiming the same parent as the current tip.

    That is exactly what two overlapping ``apply_board`` passes left behind:
    each read the same tail and derived from it, so the journal ends up with
    two records naming one parent. The forged node is self-consistent (it
    rehashes correctly) but not linked, which is what makes it a fork rather
    than tampering.
    """
    jp = journal_path(board)
    nodes, _ = load_journal(jp)
    tip = nodes[-1]
    forged = dict(tip)
    forged["notes_hash"] = sha256("someone else's notes")
    forged["slot_hash"] = node_hash(forged["slot_prev"], forged["slot_key"], forged["notes_hash"])
    forged["ts"] = ts
    with jp.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(forged) + "\n")
    return ts


def test_cli_script_actually_runs_and_reports_a_clean_board(tmp_path):
    """The cheapest possible guard: a healthy board must verify OK.

    This alone would have caught the undefined-name typo, which only ever
    executed on the failure path.
    """
    board = tmp_path / "board.jsonl"
    _write_board(board, [_card("c1", "A"), _card("c2", "B")])
    apply_board(board)

    result = _run(board)

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("verify ok:"), result.stdout


def test_cli_groups_failures_by_root_cause(tmp_path):
    """The old output truncated at 20 identical-looking lines plus a bare
    ``... N more``, hiding that most failures were a different class."""
    board = tmp_path / "board.jsonl"
    _write_board(board, [_card("c1", "A")])
    apply_board(board)
    _fork_journal(board, "2026-10-09T09:50:46+00:00")

    # a second, unrelated class: strip the slot fields off a fresh card
    _write_board(board, [_card("c1", "A"), _card("c2", "B")])

    result = _run(board)

    assert result.returncode == 1
    assert "by root cause:" in result.stdout, result.stdout
    assert "journal chain broken" in result.stdout
    assert "apply repairs" in result.stdout


def test_cli_dates_chain_breaks_as_residue(tmp_path):
    """Writes landing after the last break mean the forks are historical."""
    board = tmp_path / "board.jsonl"
    _write_board(board, [_card("c1", "A"), _card("c2", "B")])
    apply_board(board)
    # Earliest timestamp of the recorded race, so every real node postdates it.
    _fork_journal(board, "2026-10-09T09:50:46+00:00")

    result = _run(board)

    assert result.returncode == 1
    assert "chain breaks: " in result.stdout, result.stdout
    assert "distinct write event(s)" in result.stdout
    assert "last break   : 2026-10-09T09:50:46+00:00" in result.stdout
    assert "record(s) appended, none of them broke a chain" in result.stdout
    assert "verdict      : residue" in result.stdout
    assert "cannot un-happen" in result.stdout


def test_cli_flags_a_fresh_break_instead_of_calling_it_residue(tmp_path):
    """The converse: nothing written since the break, so it may still be live.

    Without this the verdict would be decorative — it has to be able to say
    the uncomfortable thing.
    """
    board = tmp_path / "board.jsonl"
    _write_board(board, [_card("c1", "A")])
    apply_board(board)
    # Far in the future: no journal record can postdate it.
    _fork_journal(board, "2999-01-01T00:00:00+00:00")

    result = _run(board)

    assert result.returncode == 1
    assert "nothing appended since the last break" in result.stdout, result.stdout
    assert "verdict      : FRESH" in result.stdout
    assert "Inspect the writer" in result.stdout


def test_cli_apply_is_idempotent_through_the_script(tmp_path):
    """`apply` via the CLI must not grow the journal on a second run."""
    board = tmp_path / "board.jsonl"
    _write_board(board, [_card("c1", "A"), _card("c2", "B")])

    first = _run(board, "apply")
    assert first.returncode == 0, first.stderr
    assert "nodes_created=2" in first.stdout, first.stdout

    second = _run(board, "apply")
    assert second.returncode == 0, second.stderr
    assert "nodes_created=0" in second.stdout, second.stdout
    assert "nodes_reused=2" in second.stdout, second.stdout
