"""Tests for scripts/work_history.py — board-derived work-history snapshots.

Card 20261112_064224. Source of truth is the kanban board (done cards' CLOSED
evidence + created/updated stamps); no new telemetry, no git source.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
from work_history import main, snapshot_from_board  # noqa: E402

FIXED_BOARD = """\
{"id": "card_a", "title": "Card A", "column": "done", "tags": ["tests"], "created_at": "2026-11-10T00:00:00+00:00", "updated_at": "2026-11-12T00:00:00+00:00", "description": "CLOSED 2026-11-12 — GREEN 31 passed / 0 failed; closure 48360 passed, all 13 triaged, zero-attributable. `ed3514237`"}
{"id": "card_b", "title": "Card B", "column": "done", "tags": [], "created_at": "2026-11-11T00:00:00+00:00", "updated_at": "2026-11-11T12:00:00+00:00", "description": "CLOSED 2026-11-11 — evidence"}
{"id": "card_c", "title": "Card C", "column": "todo", "tags": [], "created_at": "2026-11-12T00:00:00+00:00", "updated_at": "2026-11-12T00:00:00+00:00", "description": "open card, no evidence"}
{"id": "card_d", "title": "Card D", "column": "wip", "tags": [], "created_at": "2026-11-12T00:00:00+00:00", "updated_at": "2026-11-12T00:00:00+00:00", "description": ""}
"""


@pytest.fixture()
def board_file(tmp_path):
    p = tmp_path / "board.jsonl"
    p.write_text(FIXED_BOARD)
    return p


def test_snapshot_counts_and_evidence(board_file):
    s = snapshot_from_board(board_file)
    assert s["schema"] == 1
    assert s["closed"]["total"] == 2
    assert s["closed"]["with_evidence"] == 2
    assert s["closed"]["zero_attributable"] == 1
    assert s["queue"] == {"todo": 1, "wip": 1, "review": 0}
    # card_a: created -> closed = 48 h; card_b: 12 h -> mean 30 h
    assert s["avg_close_latency_s"] == pytest.approx(30 * 3600)
    assert s["reopened"] == 0


def test_record_is_deterministic(board_file, tmp_path):
    out = tmp_path / "hist"
    assert main(["record", "--board", str(board_file), "--out", str(out)]) == 0
    assert main(["record", "--board", str(board_file), "--out", str(out)]) == 0
    files = sorted(out.glob("*.json"))
    assert len(files) == 2
    a, b = (json.loads(f.read_text()) for f in files)
    a.pop("recorded_at")
    b.pop("recorded_at")
    assert a == b  # same board state -> identical payload apart from timestamp


def test_history_lists_snapshots(board_file, tmp_path, capsys):
    out = tmp_path / "hist"
    main(["record", "--board", str(board_file), "--out", str(out)])
    assert main(["history", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "closed=2" in text and "todo=1" in text


def test_compare_yields_deltas(board_file, tmp_path, capsys):
    out = tmp_path / "hist"
    main(["record", "--board", str(board_file), "--out", str(out)])
    # second window: card_c closes too (queue shrinks, closed grows)
    board_file.write_text(FIXED_BOARD.replace('"column": "todo"', '"column": "done"'))
    main(["record", "--board", str(board_file), "--out", str(out)])
    assert main(["compare", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "closed: 2 -> 3 (+1)" in text
    assert "todo: 1 -> 0 (-1)" in text


def test_compare_needs_two_snapshots(tmp_path, capsys):
    out = tmp_path / "hist"
    out.mkdir()
    assert main(["compare", "--out", str(out)]) == 1
    assert "need at least 2" in capsys.readouterr().out
