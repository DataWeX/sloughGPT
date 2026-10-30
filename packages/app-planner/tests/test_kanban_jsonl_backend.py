"""
KanbanStore JSONL migration tests — the GUI store reads/writes the live
``.kanban/board.jsonl`` (PlannerStore format) losslessly, with the legacy
``board.json`` as a read-only fallback.

Guarantees under test:
- tolerant parse: web-shaped cards (gh, chain_*, root_hash, unknown note
  keys) load and round-trip without destroying the unknown fields;
- surgical saves: only lines the caller changed are rewritten — header,
  foreign schema rows, unparsable lines, and concurrent additions survive
  byte-for-byte;
- wipe protection: a corrupt/empty read can never blank the live board;
- legacy migration: a board.json-only directory loads, and the first save
  lands board.jsonl without touching board.json;
- engine seam: boards on the KanbanEngine vocabulary are validated through
  it; the legacy4-column vocabulary keeps raw moves (no mixed enforcement).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app_planner.kanban import (
    Board,
    KanbanStore,
    _clone_board,
)

_HEADER = {
    "schema": "planner/1",
    "name": "Main",
    "columns": [
        {"name": "todo", "order": 0, "wip_limit": 5},
        {"name": "wip", "order": 1, "wip_limit": 3},
        {"name": "review", "order": 2, "wip_limit": 2},
        {"name": "done", "order": 3, "wip_limit": 0},
    ],
}

_CARD_A = {
    "id": "card-aaa-move-me",
    "title": "Web-shaped card A",
    "description": "carries live-board fields",
    "column": "todo",
    "priority": "high",
    "card_type": "",
    "assignee": "",
    "blocked_by": [],
    "created_at": "2026-10-01T00:00:00+00:00",
    "updated_at": "2026-10-01T00:00:00+00:00",
    "due_date": "",
    "gh": "n/a",
    "root_hash": "frozen-at-creation",
    "chain_hash": "c" * 64,
    "chain_index": 7,
    "chain_prev": "b" * 64,
    "notes": [
        {
            "id": "note-raw-1",
            "text": "web note",
            "action": "add",
            "at": "2026-10-02T10:00:00+00:00",
        }
    ],
}

_CARD_B = {
    "id": "card-bbb-untouched",
    "title": "Web-shaped card B",
    "column": "done",
    "priority": "low",
    "card_type": "chore",
    "chain_hash": "d" * 64,
    "chain_index": 8,
    "chain_prev": "c" * 64,
}

_CARD_C_STRAY = {
    "id": "card-ccc-stray-col",
    "title": "Stray column card",
    "column": "in-progress",
    "priority": "medium",
}

_FOREIGN_ROW = {
    "schema": "planner/1",
    "slot_history": {"todo": ["a", "b"]},
    "tray_roots": [],
}

_UNPARSABLE = "not-json{{{{"


def _line(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def _seed(dir_path: Path) -> Path:
    lines = [
        _line(_HEADER),
        _line(_CARD_A),
        _line(_CARD_B),
        _line(_CARD_C_STRAY),
        _line(_FOREIGN_ROW),
        _UNPARSABLE,
    ]
    board = dir_path / "board.jsonl"
    board.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return board


@pytest.fixture
def board_dir(tmp_path: Path) -> Path:
    d = tmp_path / ".kanban"
    d.mkdir()
    _seed(d)
    return d


@pytest.fixture
def store(board_dir: Path) -> KanbanStore:
    return KanbanStore(board_dir=board_dir)


# ── Read path ───────────────────────────────────────────────────────────


def test_loads_jsonl_with_unknown_fields(store: KanbanStore):
    board = store.load_board()
    assert board.name == "Main"
    assert [c.name for c in board.columns] == ["todo", "wip", "review", "done"]
    by_id = {c.id: c for c in board.cards}
    assert set(by_id) == {"card-aaa-move-me", "card-bbb-untouched", "card-ccc-stray-col"}
    card = by_id["card-aaa-move-me"]
    d = card.to_dict()
    assert d["gh"] == "n/a"
    assert d["root_hash"] == "frozen-at-creation"
    assert d["chain_hash"] == "c" * 64
    assert d["chain_index"] == 7
    note = card.notes[0]
    assert note.text == "web note"
    assert note.to_dict()["action"] == "add"
    assert note.to_dict()["created_at"] == "2026-10-02T10:00:00+00:00"  # "at" alias


def test_stray_column_coerced_for_display_only(store: KanbanStore, board_dir: Path):
    board = store.load_board()
    stray = next(c for c in board.cards if c.id == "card-ccc-stray-col")
    assert stray.column == "todo"  # display coerced to declared vocabulary
    objs = [
        json.loads(x)
        for x in (board_dir / "board.jsonl").read_text(encoding="utf-8").splitlines()
        if x.strip().startswith("{")
    ]
    stray_raw = next(o for o in objs if isinstance(o, dict) and o.get("id") == "card-ccc-stray-col")
    assert stray_raw["column"] == "in-progress"  # file untouched


# ── Surgical writes ─────────────────────────────────────────────────────


def test_move_rewrites_only_that_card_line(store: KanbanStore, board_dir: Path):
    path = board_dir / "board.jsonl"
    before = path.read_text(encoding="utf-8").splitlines()
    moved = store.move_card("card-aaa-move-me", "wip")
    assert moved is not None and moved.column == "wip"
    after = path.read_text(encoding="utf-8").splitlines()
    assert len(after) == len(before)
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(changed) == 1
    rewritten = json.loads(after[changed[0]])
    assert rewritten["id"] == "card-aaa-move-me"
    assert rewritten["column"] == "wip"
    assert rewritten["chain_hash"] == "c" * 64
    assert rewritten["root_hash"] == "frozen-at-creation"
    assert rewritten["notes"][0]["action"] == "add"  # unknown note keys survive
    # untouched lines are byte-identical (header, B, stray, foreign, unparsable)
    for i, line in enumerate(before):
        if i != changed[0]:
            assert after[i] == line


def test_add_appends_one_line_preserving_foreign_rows(store: KanbanStore, board_dir: Path):
    path = board_dir / "board.jsonl"
    before = path.read_text(encoding="utf-8").splitlines()
    card = store.add_card("Fresh from the GUI", column="todo")
    after = path.read_text(encoding="utf-8").splitlines()
    assert len(after) == len(before) + 1
    assert after[: len(before)] == before  # every prior line byte-identical
    assert json.loads(after[-1])["id"] == card.id


def test_delete_last_card_allowed(store: KanbanStore, board_dir: Path):
    for cid in ("card-aaa-move-me", "card-bbb-untouched", "card-ccc-stray-col"):
        assert store.delete_card(cid) is True
    assert store.load_board().cards == []
    raw = (board_dir / "board.jsonl").read_text(encoding="utf-8")
    assert _line(_FOREIGN_ROW) in raw  # foreign rows survive a full delete-through


def test_refuses_wholesale_wipe_without_flag(store: KanbanStore):
    with pytest.raises(RuntimeError, match="refusing to clear"):
        store._bk.save(Board())
    assert len(store.load_board().cards) == 3  # board intact after refusal
    store._bk.save(Board(), allow_wipe=True)
    assert store.load_board().cards == []


def test_concurrent_addition_survives_surgical_save(store: KanbanStore, board_dir: Path):
    board = store.load_board()
    baseline = _clone_board(board)
    path = board_dir / "board.jsonl"
    foreign = {
        "id": "card-zzz-concurrent",
        "title": "Added by another session",
        "column": "review",
        "priority": "high",
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_line(foreign) + "\n")
    card = next(c for c in board.cards if c.id == "card-aaa-move-me")
    card.column = "wip"
    store._bk.save(board, baseline=baseline)
    ids = {c.id for c in store.load_board().cards}
    assert "card-zzz-concurrent" in ids  # other session's line preserved
    assert next(c for c in store.load_board().cards if c.id == card.id).column == "wip"


# ── Legacy board.json fallback + migration ──────────────────────────────


def test_legacy_board_json_load_and_first_save_migration(tmp_path: Path):
    legacy = {
        "name": "legacy",
        "columns": [
            {"name": "todo", "wip_limit": 0, "order": 0},
            {"name": "done", "wip_limit": 0, "order": 1},
        ],
        "cards": [
            {
                "id": "legacy-1",
                "title": "Old card with unknown keys",
                "column": "todo",
                "root_hash": "keep-me",
            }
        ],
    }
    legacy_bytes = json.dumps(legacy, indent=2)
    (tmp_path / "board.json").write_text(legacy_bytes, encoding="utf-8")
    store = KanbanStore(board_dir=tmp_path)

    board = store.load_board()
    assert [c.id for c in board.cards] == ["legacy-1"]
    assert board.cards[0].to_dict()["root_hash"] == "keep-me"  # unknown key survived

    store.add_card("First jsonl card", column="todo")
    live = tmp_path / "board.jsonl"
    assert live.exists()
    live_lines = [json.loads(x) for x in live.read_text(encoding="utf-8").splitlines() if x.strip()]
    header = live_lines[0]
    assert header["schema"] == "planner/1"
    ids = {c.get("id") for c in live_lines[1:]}
    assert "legacy-1" in ids
    assert len(live_lines) == 3  # header + legacy card + new card
    assert (tmp_path / "board.json").read_text(encoding="utf-8") == legacy_bytes


# ── Engine vocabulary seam ──────────────────────────────────────────────


def _write_engine_board(dir_path: Path) -> None:
    header = {
        "schema": "planner/1",
        "name": "engine",
        "columns": [
            {"name": "inbox", "order": 0, "wip_limit": 0},
            {"name": "ready", "order": 1, "wip_limit": 0},
            {"name": "doing", "order": 2, "wip_limit": 3},
            {"name": "review", "order": 3, "wip_limit": 0},
            {"name": "verify", "order": 4, "wip_limit": 0},
            {"name": "done", "order": 5, "wip_limit": 0},
        ],
    }
    card = {
        "id": "eng-1",
        "title": "Engine board card",
        "column": "ready",
        "priority": "medium",
    }
    (dir_path / "board.jsonl").write_text(
        _line(header) + "\n" + _line(card) + "\n", encoding="utf-8"
    )


def test_engine_vocab_board_validates_transitions(tmp_path: Path):
    _write_engine_board(tmp_path)
    store = KanbanStore(board_dir=tmp_path)
    # ready → review is illegal in the engine (ready → doing|blocked only)
    assert store.move_card("eng-1", "review") is None
    assert store.get_card("eng-1").column == "ready"
    # ready → doing is legal
    moved = store.move_card("eng-1", "doing")
    assert moved is not None and moved.column == "doing"


def test_legacy_vocab_board_keeps_raw_moves(store: KanbanStore):
    # todo/wip/review/done is not engine vocabulary: moves stay raw, no
    # half-enforcement, no ValueError from CardState coercion
    moved = store.move_card("card-bbb-untouched", "todo")
    assert moved is not None and moved.column == "todo"
