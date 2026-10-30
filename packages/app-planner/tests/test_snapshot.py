"""
Tests for app_planner.snapshot — dated byte-exact export of the canonical
board into ``<repo>/.kanban/snapshots/``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from app_planner.snapshot import SnapshotError, snapshot
from app_planner.store import PlannerStore


def _canonical_board(tmp_path: Path) -> Path:
    board_dir = tmp_path / "canonical-board"
    store = PlannerStore(board_dir=board_dir, notes_dir=tmp_path / "cn-notes")
    store.add_card("snap me")
    return board_dir


def test_snapshot_copies_canonical_board(tmp_path):
    board_dir = _canonical_board(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()

    result = snapshot(repo, board_dir=board_dir)

    dest = Path(result["snapshot"])
    assert dest.parent == repo / ".kanban" / "snapshots"
    assert dest.name.startswith("board-") and dest.name.endswith(".jsonl")
    # Byte-exact copy of the canonical board at snapshot time.
    assert dest.read_bytes() == (board_dir / "board.jsonl").read_bytes()
    assert result["bytes"] == (board_dir / "board.jsonl").stat().st_size


def test_snapshot_missing_canonical_board_raises(tmp_path):
    with pytest.raises(SnapshotError, match="no canonical board"):
        snapshot(tmp_path / "repo", board_dir=tmp_path / "does-not-exist")


def test_snapshot_distinct_stamps_same_run(tmp_path):
    """Two snapshots never clobber each other, even within one second."""
    board_dir = _canonical_board(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()

    first = Path(snapshot(repo, board_dir=board_dir)["snapshot"])
    second = Path(snapshot(repo, board_dir=board_dir)["snapshot"])

    assert first != second, "same-second snapshots must get distinct paths"
    assert first.is_file() and second.is_file()
    expected = (board_dir / "board.jsonl").read_bytes()
    assert first.read_bytes() == expected
    assert second.read_bytes() == expected
