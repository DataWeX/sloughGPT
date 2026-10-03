"""Slot-chain tests — app_planner.slot_chain + the store's one-line write hook.

Covers: apply fills slot fields, re-apply is idempotent/byte-stable, card
body edits never rehash, move/delete produce branch parents, note inputs feed
note_hashes, verify catches drift/tampering, and the single
``PlannerStore._atomic_write`` line that keeps the chain alive across
add/update/move (so store writes never lose slot fields).
"""

from __future__ import annotations

import json
from pathlib import Path

from app_planner.slot_chain import (
    apply_board,
    journal_path,
    load_journal,
    sha256,
    verify_board,
)
from app_planner.store import PlannerStore

HEADER = {
    "schema": "planner/1",
    "name": "Main",
    "columns": [
        {"name": "todo", "wip_limit": 5, "order": 0},
        {"name": "in_progress", "wip_limit": 3, "order": 1},
        {"name": "review", "wip_limit": 2, "order": 2},
        {"name": "done", "wip_limit": 0, "order": 3},
    ],
}


def _write_lines(path: Path, lines: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")


def _card(card_id: str, title: str, column: str = "todo", **extra) -> dict:
    return {
        "id": card_id,
        "title": title,
        "column": column,
        "priority": "medium",
        "tags": [],
        "notes": [],
        "created_at": "2026-10-02T00:00:00+00:00",
        "updated_at": "2026-10-02T00:00:00+00:00",
        **extra,
    }


def _read_lines(path: Path) -> list[dict]:
    return [json.loads(raw) for raw in path.read_text(encoding="utf-8").splitlines() if raw.strip()]


def _read_cards(path: Path) -> list[dict]:
    return [obj for obj in _read_lines(path) if "columns" not in obj and "title" in obj]


def _mutate(path: Path, fn) -> None:
    """Simulate an arbitrary chain-unaware writer editing the board."""
    _write_lines(path, fn(_read_lines(path)))


# ── apply: fills fields, idempotent ──────────────────────────────────────


def test_apply_chains_card_fields(tmp_path):
    board = tmp_path / "board.jsonl"
    _write_lines(board, [HEADER, _card("c1", "First card")])

    stats = apply_board(board)

    assert stats == {"cards": 1, "nodes_created": 1, "nodes_reused": 0, "tips": 1, "trays": 1}
    (stored,) = _read_cards(board)
    for key in ("slot_position", "note_hashes", "notes_hash", "slot_prev", "slot_hash"):
        assert key in stored, f"missing {key} after apply"
    assert stored["slot_position"] == 0
    assert stored["slot_prev"] == ""  # first occupant of its slot
    assert stored["note_hashes"] == []  # no notes -> no inputs
    assert stored["notes_hash"] == ""

    nodes, _ = load_journal(journal_path(board))
    assert len(nodes) == 1
    assert nodes[0]["slot_hash"] == stored["slot_hash"]
    assert verify_board(board) == []


def test_reapply_is_idempotent_and_byte_stable(tmp_path):
    board = tmp_path / "board.jsonl"
    _write_lines(board, [HEADER, _card("c1", "A"), _card("c2", "B")])
    apply_board(board)
    snapshot = board.read_bytes()

    stats = apply_board(board)

    assert stats["nodes_created"] == 0
    assert stats["nodes_reused"] == 2
    assert board.read_bytes() == snapshot  # no-op runs touch nothing
    assert verify_board(board) == []


def test_card_body_edits_never_rehash(tmp_path):
    board = tmp_path / "board.jsonl"
    _write_lines(board, [HEADER, _card("c1", "Original title")])
    apply_board(board)
    before = _read_cards(board)[0]["slot_hash"]

    # a writer (UI/CLI/hand-edit) rewrites the line with a new title
    _mutate(board, lambda lines: [
        {**line, "title": "Renamed"} if line.get("id") == "c1" else line for line in lines
    ])
    stats = apply_board(board)

    assert stats["nodes_created"] == 0  # same slot + same notes -> same state
    (stored,) = _read_cards(board)
    assert stored["title"] == "Renamed"
    assert stored["slot_hash"] == before  # card body never enters the hash
    nodes, _ = load_journal(journal_path(board))
    assert len(nodes) == 1
    assert verify_board(board) == []


# ── move / delete: branch-on-removal ─────────────────────────────────────


def test_move_gives_displaced_cards_branch_parents(tmp_path):
    board = tmp_path / "board.jsonl"
    _write_lines(board, [HEADER, _card("c-a", "A"), _card("c-b", "B"), _card("c-c", "C")])
    apply_board(board)
    _, tips_before = load_journal(journal_path(board))
    b_old_node = tips_before["todo:1"]  # B's node before the move

    _mutate(board, lambda lines: [
        {**line, "column": "done"} if line.get("id") == "c-b" else line for line in lines
    ])
    apply_board(board)

    cards = {c["id"]: c for c in _read_cards(board)}
    # C shifted up todo:2 -> todo:1: its parent is the displaced occupant (B)
    assert cards["c-c"]["slot_position"] == 1
    assert cards["c-c"]["slot_prev"] == b_old_node["slot_hash"]
    # B now occupies done:0 (first occupant there)
    assert cards["c-b"]["column"] == "done"
    assert cards["c-b"]["slot_position"] == 0
    assert cards["c-b"]["slot_prev"] == ""
    # B's old node is retained as the root of its own branch
    nodes, _ = load_journal(journal_path(board))
    assert any(n["slot_hash"] == b_old_node["slot_hash"] for n in nodes)
    assert verify_board(board) == []


def test_delete_keeps_removed_slot_as_branch_root(tmp_path):
    board = tmp_path / "board.jsonl"
    _write_lines(board, [HEADER, _card("c-a", "A"), _card("c-b", "B")])
    apply_board(board)
    _, tips_before = load_journal(journal_path(board))
    a_node = tips_before["todo:0"]

    _mutate(board, lambda lines: [line for line in lines if line.get("id") != "c-a"])
    apply_board(board)

    nodes, _ = load_journal(journal_path(board))
    assert any(n["slot_hash"] == a_node["slot_hash"] for n in nodes)  # branch root survives
    (b,) = _read_cards(board)
    assert b["title"] == "B"
    assert b["slot_position"] == 0
    assert b["slot_prev"] == a_node["slot_hash"]  # new occupant parents to removed one
    assert verify_board(board) == []


# ── note inputs ──────────────────────────────────────────────────────────


def test_inline_notes_feed_note_hashes_and_aggregate(tmp_path):
    board = tmp_path / "board.jsonl"
    _write_lines(board, [HEADER, _card("c1", "With notes", notes=["hello", "world"])])

    apply_board(board)

    (stored,) = _read_cards(board)
    assert stored["note_hashes"] == [sha256("hello"), sha256("world")]
    assert stored["notes_hash"]  # aggregate over the inputs
    assert verify_board(board) == []


def test_linked_dev_note_hash_is_included(tmp_path):
    board = tmp_path / "board.jsonl"
    _write_lines(board, [HEADER, _card("c-linked", "Linked")])
    notes_home = tmp_path / "dev-notes"
    notes_home.mkdir()
    digest = sha256("frontmatter+filename")
    (notes_home / "a-note.md").write_text(
        f"---\nid: a-note\ntitle: A Note\nboard_id: c-linked\nhash: {digest}\n---\nbody\n",
        encoding="utf-8",
    )

    apply_board(board, notes_home=notes_home)

    (stored,) = _read_cards(board)
    assert stored["note_hashes"] == [digest]
    assert verify_board(board, notes_home=notes_home) == []


# ── verify is the gate ───────────────────────────────────────────────────


def test_verify_detects_slot_tampering(tmp_path):
    board = tmp_path / "board.jsonl"
    _write_lines(board, [HEADER, _card("c1", "Tamper me")])
    apply_board(board)

    _mutate(board, lambda lines: [
        {**line, "slot_hash": "deadbeef"} if line.get("id") == "c1" else line for line in lines
    ])

    failures = verify_board(board)
    assert failures, "tampered slot_hash must fail verify"


def test_apply_strips_retired_chain_fields(tmp_path):
    board = tmp_path / "board.jsonl"
    _write_lines(
        board,
        [HEADER, _card("c1", "Legacy", chain_hash="abc", chain_prev="def", chain_index=3)],
    )

    apply_board(board)

    (stored,) = _read_cards(board)
    assert "chain_hash" not in stored
    assert "chain_prev" not in stored
    assert "chain_index" not in stored
    assert verify_board(board) == []


def test_headerless_board_apply_is_noop_and_verify_flags(tmp_path):
    board = tmp_path / "board.jsonl"
    _write_lines(board, [_card("c1", "No header yet")])

    stats = apply_board(board)  # quiet no-op: legacy headerless writes keep working

    assert stats["cards"] == 0
    assert "slot_hash" not in _read_cards(board)[0]
    assert verify_board(board), "verify must still flag the headerless board"


# ── the store's one-line hook ────────────────────────────────────────────


def test_store_writes_keep_the_chain(tmp_path):
    """PlannerStore._atomic_write ends with one apply_board() line: add,
    update, and move never lose (or staleness) the slot fields — even though
    the Card dataclass knows nothing about slots (stripped line is healed
    in the same write)."""
    store = PlannerStore(board_dir=tmp_path / "kanban", notes_dir=tmp_path / "notes")
    card = store.add_card("First card")
    board = tmp_path / "kanban" / "board.jsonl"

    (stored,) = _read_cards(board)
    assert stored["slot_hash"], "add must chain immediately"
    assert stored["slot_position"] == 0
    before = stored["slot_hash"]

    store.update_card(card.id, title="Renamed")
    (stored,) = _read_cards(board)
    assert stored["title"] == "Renamed"
    assert stored["slot_hash"] == before  # body edit: same slot state, hash kept

    store.move_card(card.id, "done")
    stored = _read_cards(board)[0]
    assert stored["column"] == "done"
    assert stored["slot_position"] == 0
    assert stored["slot_prev"] == ""  # first occupant of done:0
    assert stored["slot_hash"]

    nodes, _ = load_journal(journal_path(board))
    assert len(nodes) == 2  # todo:0 + done:0
    assert verify_board(board) == []
