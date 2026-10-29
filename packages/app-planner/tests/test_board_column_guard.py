"""Read-path column guard tests (card b538ecbd).

``load_board()`` must never hand a card with an undeclared column to
renderers: ``board show`` iterates the *header* columns only, so a
hand-typed column value (the board file is hand-editable by design) made
the card silently invisible while stats still counted it. The guard
warns and re-homes such cards under a declared column instead of
raising — one typo must not break the whole board.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from app_planner.store import PlannerStore

HEADER = {
    "schema": "planner/1",
    "name": "Main",
    "columns": [
        {"name": "todo", "order": 0, "wip_limit": 5},
        {"name": "in_progress", "order": 1, "wip_limit": 3},
        {"name": "review", "order": 2, "wip_limit": 2},
        {"name": "done", "order": 3, "wip_limit": 0},
    ],
}
DECLARED = ["todo", "in_progress", "review", "done"]


def _write_board(board_dir: Path, records: list[dict]) -> PlannerStore:
    board_dir.mkdir(parents=True, exist_ok=True)
    path = board_dir / "board.jsonl"
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records),
        encoding="utf-8",
    )
    return PlannerStore(board_dir=board_dir, notes_dir=board_dir / "notes")


def test_undeclared_column_is_reported_and_re_homed(tmp_path, caplog):
    store = _write_board(
        tmp_path / "board",
        [
            HEADER,
            {"id": "good-1", "title": "Good card", "column": "todo"},
            {"id": "phantom-1", "title": "Phantom card", "column": "in-progress"},
        ],
    )
    with caplog.at_level(logging.WARNING, logger="app_planner.store"):
        board = store.load_board()

    by_id = {c.id: c for c in board.cards}
    assert "phantom-1" in by_id, "phantom card must stay visible"
    assert by_id["phantom-1"].column == "todo"
    assert by_id["good-1"].column == "todo", "valid cards untouched"

    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("phantom-1" in w and "in-progress" in w for w in warnings), warnings


def test_every_loaded_card_is_in_a_declared_column(tmp_path):
    store = _write_board(
        tmp_path / "board",
        [
            HEADER,
            {"id": "a", "title": "A", "column": "todo"},
            {"id": "b", "title": "B", "column": "in_progress"},
            {"id": "c", "title": "C", "column": "TYPO"},
        ],
    )
    board = store.load_board()
    assert [c.column for c in board.cards] != []
    assert all(c.column in DECLARED for c in board.cards)


def test_clean_board_loads_without_warnings(tmp_path, caplog):
    store = _write_board(
        tmp_path / "board",
        [
            HEADER,
            {"id": "a", "title": "A", "column": "in_progress"},
            {"id": "b", "title": "B", "column": "done"},
        ],
    )
    with caplog.at_level(logging.WARNING, logger="app_planner.store"):
        board = store.load_board()
    assert len(board.cards) == 2
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []


def test_headerless_board_falls_back_to_default_vocabulary(tmp_path, caplog):
    store = _write_board(
        tmp_path / "board",
        [{"id": "a", "title": "A", "column": "nonsense"}],
    )
    with caplog.at_level(logging.WARNING, logger="app_planner.store"):
        board = store.load_board()
    # fallback vocabulary (Board().columns) starts with 'todo'
    assert [c.column for c in board.cards] == ["todo"]


def test_repo_board_every_card_declared(tmp_path):
    """Canary for the shared board file: no phantom columns survive load.

    The board is hand-edited by design across sessions; if a future
    merge or edit reintroduces an undeclared column, this fails loudly
    instead of letting cards go invisible (skip if the checkout carries
    no board file).
    """
    repo_root = Path(__file__).resolve().parents[3]
    board_dir = repo_root / ".kanban"
    if not (board_dir / "board.jsonl").exists():
        pytest.skip("no board.jsonl in this checkout")
    store = PlannerStore(board_dir=board_dir, notes_dir=tmp_path / "notes")
    board = store.load_board()
    assert board.cards, "expected a populated board"
    declared = {c.get("name") for c in board.columns}
    undeclared = [(c.id, c.column) for c in board.cards if c.column not in declared]
    assert undeclared == [], f"cards outside the header vocabulary: {undeclared}"
