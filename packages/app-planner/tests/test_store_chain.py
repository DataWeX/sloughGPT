"""Tests for hash-chained card ordering — chain_index/chain_prev/chain_hash.

The chain gives the board a canonical order (topological over blocked_by,
stable by (created_at, id)) plus tamper evidence: verify_chain() returns the
chain_index of every card whose stored link fails recomputation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app_planner.store import CHAIN_GENESIS, PlannerStore, reset_store


@pytest.fixture(autouse=True)
def _reset():
    reset_store()
    yield
    reset_store()


@pytest.fixture
def store(tmp_path: Path) -> PlannerStore:
    return PlannerStore(board_dir=tmp_path / "board", notes_dir=tmp_path / "notes")


def _file_card_ids(store: PlannerStore) -> list[str]:
    lines = [json.loads(raw) for raw in store._board_file.read_text().splitlines() if raw.strip()]
    return [obj["id"] for obj in lines if isinstance(obj, dict) and obj.get("title")]


class TestChainCompute:
    def test_card_has_chain_fields_default(self, store: PlannerStore):
        card = store.add_card("x")
        assert card.chain_index == -1
        assert card.chain_prev == ""
        assert card.chain_hash == ""

    def test_compute_assigns_dense_indices_and_orders_file(self, store: PlannerStore):
        a = store.add_card("a")
        b = store.add_card("b")
        c = store.add_card("c")
        cycles = store.compute_chains()
        assert cycles == []
        cards = store.load_board().cards
        assert [x.chain_index for x in cards] == [0, 1, 2]
        assert cards[0].chain_prev == CHAIN_GENESIS
        # file sequence follows the chain
        assert _file_card_ids(store) == [a.id, b.id, c.id]

    def test_compute_is_idempotent(self, store: PlannerStore):
        store.add_card("a")
        store.add_card("b")
        store.compute_chains()
        first = [(c.chain_prev, c.chain_hash) for c in store.load_board().cards]
        store.compute_chains()
        second = [(c.chain_prev, c.chain_hash) for c in store.load_board().cards]
        assert first == second

    def test_chain_deterministic_across_stores(self, tmp_path: Path):
        def build(root: Path) -> list[tuple[str, str]]:
            (root / "board").mkdir(parents=True)
            lines = [
                json.dumps(
                    {
                        "schema": "planner/1",
                        "name": "Main",
                        "columns": [{"name": "todo", "wip_limit": 0, "order": 0}],
                    }
                )
            ]
            for i in range(3):
                lines.append(
                    json.dumps(
                        {
                            "id": f"card-{i}",
                            "title": f"t{i}",
                            "column": "todo",
                            "blocked_by": [],
                            "created_at": "2026-01-01T00:00:00+00:00",
                            "updated_at": "2026-01-01T00:00:00+00:00",
                        },
                        sort_keys=True,
                    )
                )
            (root / "board" / "board.jsonl").write_text("\n".join(lines) + "\n")
            s = PlannerStore(board_dir=root / "board", notes_dir=root / "notes")
            s.compute_chains()
            return [(c.chain_prev, c.chain_hash) for c in s.load_board().cards]

        assert build(tmp_path / "one") == build(tmp_path / "two")

    def test_topo_beats_creation_order(self, store: PlannerStore):
        blocked = store.add_card("blocked")  # created FIRST
        blocker = store.add_card("blocker")  # created SECOND
        store.block_card(blocked.id, blocker.id)
        store.compute_chains()
        idx = {c.id: c.chain_index for c in store.load_board().cards}
        assert idx[blocker.id] < idx[blocked.id]

    def test_cycle_reported_and_still_indexed(self, store: PlannerStore):
        a = store.add_card("a")
        b = store.add_card("b")
        store.block_card(a.id, b.id)
        store.block_card(b.id, a.id)
        cycles = store.compute_chains()
        assert set(cycles) == {a.id, b.id}
        assert sorted(c.chain_index for c in store.load_board().cards) == [0, 1]

    def test_unknown_blocker_ignored(self, store: PlannerStore):
        card = store.add_card("solo")
        card.blocked_by = ["does-not-exist"]
        store.update_card(card.id, blocked_by=card.blocked_by)
        cycles = store.compute_chains()
        assert cycles == []
        assert store.load_board().cards[0].chain_index == 0


class TestChainVerify:
    def test_verify_passes_after_compute(self, store: PlannerStore):
        store.add_card("a")
        store.add_card("b")
        store.compute_chains()
        assert store.verify_chain() == []

    def test_verify_flags_mutation(self, store: PlannerStore):
        store.add_card("a")
        store.add_card("b")
        store.compute_chains()
        # A legitimate update now re-seals the board (TestChainSelfHeal), so
        # tamper out-of-band — bypass the store entirely — to prove the
        # chain still catches content edits made behind its back.
        text = store._board_file.read_text()
        tampered = text.replace('"title": "a"', '"title": "tampered"', 1)
        assert tampered != text
        store._atomic_write(tampered)
        assert store.verify_chain() == [0]

    def test_verify_flags_unchained(self, store: PlannerStore):
        store.add_card("a")
        assert store.verify_chain() == [-1]

    def test_verify_flags_reordered_file(self, store: PlannerStore):
        a = store.add_card("a")
        b = store.add_card("b")
        store.compute_chains()
        # swap the two card lines without touching their payloads
        text = store._board_file.read_text()
        line_a = next(raw for raw in text.splitlines() if f'"id": "{a.id}"' in raw)
        line_b = next(raw for raw in text.splitlines() if f'"id": "{b.id}"' in raw)
        store._atomic_write(
            text.replace(line_a, "\x00").replace(line_b, line_a).replace("\x00", line_b)
        )
        assert store.verify_chain() != []


class TestChainSelfHeal:
    """A sealed board re-seals itself after every store mutation.

    Card 835122e0: surgical rewrites previously left chain_hash stale
    until the next sync, so the chain could not serve as tamper evidence
    between syncs. Legacy unsealed boards must stay unsealed — mutations
    never introduce chain fields where they did not exist.
    """

    @pytest.fixture
    def sealed(self, store: PlannerStore) -> PlannerStore:
        store.add_card("a")
        store.add_card("b")
        store.compute_chains()
        assert store.verify_chain() == []
        return store

    def test_move_reseals(self, sealed: PlannerStore):
        first = sealed.load_board().cards[0]
        sealed.move_card(first.id, "wip")
        assert sealed.verify_chain() == []

    def test_update_reseals(self, sealed: PlannerStore):
        first = sealed.load_board().cards[0]
        sealed.update_card(first.id, title="renamed")
        assert sealed.verify_chain() == []

    def test_add_reseals(self, sealed: PlannerStore):
        sealed.add_card("c")
        cards = sealed.load_board().cards
        assert len(cards) == 3
        assert [c.chain_index for c in cards] == [0, 1, 2]
        assert sealed.verify_chain() == []

    def test_delete_reseals(self, sealed: PlannerStore):
        first = sealed.load_board().cards[0]
        sealed.delete_card(first.id)
        assert len(sealed.load_board().cards) == 1
        assert sealed.verify_chain() == []

    def test_block_reseals(self, sealed: PlannerStore):
        blocker, blocked = sealed.load_board().cards[:2]
        sealed.block_card(blocked.id, blocker.id)
        assert sealed.verify_chain() == []

    def test_legacy_board_never_auto_seals(self, store: PlannerStore):
        a = store.add_card("a")
        b = store.add_card("b")
        store.move_card(a.id, "wip")
        store.update_card(b.id, title="renamed")
        store.add_card("c")
        store.delete_card(a.id)
        cards = store.load_board().cards
        assert len(cards) == 2
        assert all(c.chain_hash == "" and c.chain_index == -1 for c in cards)
        assert store.verify_chain() == [-1]


class TestChainSync:
    def test_sync_chains_and_orders(self, store: PlannerStore):
        store.create_note("note-card", status="todo")
        added, _updated, total = store.sync()
        assert total == 1
        assert added == 1
        cards = store.load_board().cards
        assert [c.chain_index for c in cards] == [0]
        assert store.verify_chain() == []
        assert _file_card_ids(store) == [cards[0].id]
