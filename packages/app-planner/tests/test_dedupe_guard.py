"""Tests for duplicate-id collapse + the read-only board-integrity guard.

Card 89c58cfb: historical line-level edits left some cards with two or three
lines sharing one ``id``. ``store.dedupe_cards()`` collapses them to the
latest-``updated_at`` survivor and reseals the hash chain; the opaque
``slot_history``/``tray_roots`` trailer is preserved byte-for-byte. The
``verify_board_integrity`` guard surfaces any regression as a finding — it
never raises, so it can never crash a writer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app_planner.store import PlannerStore, reset_store
from app_planner.verify import verify_board_integrity

HEADER = {
    "schema": "planner/1",
    "name": "Main",
    "columns": [
        {"name": "todo", "order": 0, "wip_limit": 5},
        {"name": "wip", "order": 1, "wip_limit": 3},
        {"name": "review", "order": 2, "wip_limit": 2},
        {"name": "done", "order": 3, "wip_limit": 0},
    ],
}
# Opaque trailer written by external tooling — invisible to the store's card
# model; a rebuild must not destroy it.
TRAILER = {
    "schema": "planner/1",
    "name": "Main",
    "columns": HEADER["columns"],
    "slot_history": "slot_history.jsonl",
    "tray_roots": {"done": "abc", "todo": "def"},
}


@pytest.fixture(autouse=True)
def _reset():
    reset_store()
    yield
    reset_store()


def _card(cid, title, column="todo", updated="2026-01-01T00:00:00+00:00", **over):
    base = {
        "id": cid,
        "title": title,
        "column": column,
        "updated_at": updated,
        "chain_index": -1,
        "chain_prev": "",
        "chain_hash": "",
    }
    base.update(over)
    return base


def _write_board(board_dir: Path, records: list[dict]) -> PlannerStore:
    board_dir.mkdir(parents=True, exist_ok=True)
    (board_dir / "board.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    return PlannerStore(board_dir=board_dir, notes_dir=board_dir / "notes")


def _card_ids_on_disk(store: PlannerStore) -> list[str]:
    lines = [
        json.loads(raw)
        for raw in store._board_file.read_text().splitlines()
        if raw.strip()
    ]
    return [o["id"] for o in lines if isinstance(o, dict) and o.get("title")]


class TestDedupeCards:
    def test_drops_surplus_keeps_latest_updated_at(self, tmp_path):
        store = _write_board(
            tmp_path / "board",
            [
                HEADER,
                _card("dup1", "Magazine", "todo", updated="2026-01-01T00:00:00+00:00"),
                _card("dup1", "Magazine", "wip", updated="2026-01-02T00:00:00+00:00"),
                _card("dup1", "Magazine", "done", updated="2026-01-03T00:00:00+00:00"),
                _card("solo", "Kept", "todo"),
            ],
        )
        result = store.dedupe_cards()
        assert result["duplicates"] == 2
        assert result["unique"] == 2

        remaining = {c.id: c for c in store.load_board().cards}
        assert set(remaining) == {"dup1", "solo"}
        assert remaining["dup1"].column == "done"  # latest updated_at wins

    def test_tie_keeps_first_occurrence(self, tmp_path):
        ts = "2026-01-01T00:00:00+00:00"
        store = _write_board(
            tmp_path / "board",
            [
                HEADER,
                _card("twin", "Twin", "todo", updated=ts, assignee=""),
                _card("twin", "Twin", "wip", updated=ts, assignee="session"),
            ],
        )
        store.dedupe_cards()
        survivors = [c for c in store.load_board().cards if c.id == "twin"]
        assert len(survivors) == 1
        # First occurrence wins the tie (>= keeps incumbent).
        assert survivors[0].assignee == ""

    def test_preserves_opaque_trailer_byte_for_byte(self, tmp_path):
        store = _write_board(
            tmp_path / "board",
            [
                HEADER,
                _card("dup1", "A", "todo", updated="2026-01-01T00:00:00+00:00"),
                _card("dup1", "A", "done", updated="2026-01-02T00:00:00+00:00"),
                TRAILER,
            ],
        )
        store.dedupe_cards()
        text = store._board_file.read_text(encoding="utf-8")

        def _true_trailers(t: str) -> list[str]:
            # A true trailer has slot_history AND no card id — a card whose
            # *title* mentions "slot_history" must not be mistaken for one.
            out = []
            for line in t.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("slot_history") and obj.get("id") is None:
                    out.append(line)
            return out

        # _reorder_cards matches only lines with a card id, so the trailer
        # (no "id" key) passes through untouched, in its original key order.
        trailer_line = json.dumps(TRAILER, ensure_ascii=False)
        assert trailer_line in text
        assert _true_trailers(text) == [trailer_line]

    def test_reseals_chain_over_survivors(self, tmp_path):
        store = _write_board(
            tmp_path / "board",
            [
                HEADER,
                _card("dup1", "A", "todo", updated="2026-01-01T00:00:00+00:00"),
                _card("dup1", "A", "done", updated="2026-01-02T00:00:00+00:00"),
                _card("solo", "B", "todo"),
            ],
        )
        store.dedupe_cards()
        # Chain is dense (no gaps) and every link verifies.
        cards = store.load_board().cards
        assert [c.chain_index for c in cards] == list(range(len(cards)))
        assert store.verify_chain() == []

    def test_idempotent_clean_board_no_write(self, tmp_path):
        store = _write_board(
            tmp_path / "board",
            [HEADER, _card("a", "A"), _card("b", "B")],
        )
        before = store._board_file.read_text(encoding="utf-8")
        result = store.dedupe_cards()
        assert result == {"duplicates": 0, "unique": 2}
        assert store._board_file.read_text(encoding="utf-8") == before


class TestBoardIntegrityGuard:
    def test_clean_board_ok(self, tmp_path):
        board = _write_board(
            tmp_path / "board",
            [HEADER, _card("a", "A"), _card("b", "B")],
        )
        findings = verify_board_integrity(board._board_file)
        assert all(f.ok for f in findings)
        assert any("unique" in f.detail for f in findings)

    def test_duplicate_id_flagged_not_raised(self, tmp_path):
        board = _write_board(
            tmp_path / "board",
            [
                HEADER,
                _card("dup", "A", "todo"),
                _card("dup", "A", "done"),
                _card("solo", "B"),
            ],
        )
        findings = verify_board_integrity(board._board_file)  # must not raise
        bad = [f for f in findings if not f.ok]
        assert bad, "duplicates must produce a finding"
        assert any("appears 2x" in f.detail for f in bad)
        assert any("surplus" in f.detail for f in bad)

    def test_missing_board_is_ok_not_error(self, tmp_path):
        findings = verify_board_integrity(tmp_path / "nope.jsonl")
        assert all(f.ok for f in findings)

    def test_unparsable_lines_tolerated(self, tmp_path):
        path = tmp_path / "board.jsonl"
        path.write_text(
            json.dumps(HEADER) + "\n"
            + "not json at all\n"
            + json.dumps(_card("a", "A")) + "\n"
        )
        findings = verify_board_integrity(path)  # must not raise
        assert all(f.ok for f in findings)
