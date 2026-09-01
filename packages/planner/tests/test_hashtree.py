"""Tests for planner.hashtree — cryptographic hash tree for kanban cards."""

import json
from pathlib import Path

import pytest

from planner.hashtree import (
    _hash,
    _derive_note_hash,
    CardSlotHash,
    NoteHash,
    HashHistoryEntry,
    HashCommit,
    HashTree,
    HashTreeStore,
    create_hash_tree,
)


# ══════════════════════════════════════════════════════════════════════════════
# Hash functions
# ══════════════════════════════════════════════════════════════════════════════


class TestHashFunctions:
    def test_hash_returns_hex_string(self):
        result = _hash("hello")
        assert len(result) == 64  # SHA-256 hex digest
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_deterministic(self):
        assert _hash("test") == _hash("test")

    def test_hash_different_inputs(self):
        assert _hash("a") != _hash("b")

    def test_derive_note_hash(self):
        h = _derive_note_hash("root123", "note content")
        assert len(h) == 64
        assert isinstance(h, str)

    def test_derive_note_hash_deterministic(self):
        assert _derive_note_hash("r", "c") == _derive_note_hash("r", "c")

    def test_derive_note_hash_depends_on_root(self):
        h1 = _derive_note_hash("root1", "content")
        h2 = _derive_note_hash("root2", "content")
        assert h1 != h2

    def test_derive_note_hash_depends_on_content(self):
        h1 = _derive_note_hash("root", "content1")
        h2 = _derive_note_hash("root", "content2")
        assert h1 != h2


# ══════════════════════════════════════════════════════════════════════════════
# Dataclass round-trips
# ══════════════════════════════════════════════════════════════════════════════


class TestCardSlotHash:
    def test_to_dict_and_back(self):
        csh = CardSlotHash(
            root="abc123", card_id="c1", slot_id="s1",
            tray="todo", position=0, placed_at="2024-01-01", created_at="2024-01-01",
        )
        d = csh.to_dict()
        restored = CardSlotHash.from_dict(d)
        assert restored.root == "abc123"
        assert restored.tray == "todo"

    def test_from_dict_defaults(self):
        d = {"root": "x", "card_id": "c", "slot_id": "s", "tray": "t"}
        csh = CardSlotHash.from_dict(d)
        assert csh.position == 0
        assert csh.placed_at == ""


class TestNoteHash:
    def test_to_dict_and_back(self):
        nh = NoteHash(note_id="n1", hash_value="h1", root_ref="r1", version=2)
        d = nh.to_dict()
        restored = NoteHash.from_dict(d)
        assert restored.note_id == "n1"
        assert restored.version == 2


class TestHashHistoryEntry:
    def test_to_dict_and_back(self):
        hhe = HashHistoryEntry(
            root_ref="r1", old_hash="old", new_hash="new",
            change_type="note_edit", note_id="n1", timestamp="t1",
        )
        d = hhe.to_dict()
        restored = HashHistoryEntry.from_dict(d)
        assert restored.change_type == "note_edit"


class TestHashCommit:
    def test_to_dict_and_back(self):
        hc = HashCommit(
            commit_hash="ch1", parent_hash="ph1", root_ref="r1",
            changes=[{"type": "add"}], color="#abc123", pixel="deadbeef",
        )
        d = hc.to_dict()
        restored = HashCommit.from_dict(d)
        assert restored.commit_hash == "ch1"
        assert restored.color == "#abc123"


# ══════════════════════════════════════════════════════════════════════════════
# HashTree
# ══════════════════════════════════════════════════════════════════════════════


def _make_tree():
    root = CardSlotHash(
        root="test_root", card_id="c1", slot_id="s1",
        tray="todo", position=0, placed_at="2024-01-01", created_at="2024-01-01",
    )
    return HashTree(root=root)


class TestHashTreeAddNote:
    def test_add_note(self):
        tree = _make_tree()
        nh = tree.add_note("n1", "hello world")
        assert nh.note_id == "n1"
        assert nh.version == 1
        assert len(tree.notes) == 1
        assert len(tree.history) == 1
        assert tree.history[0].change_type == "note_add"

    def test_add_note_updates_existing(self):
        tree = _make_tree()
        tree.add_note("n1", "v1")
        nh = tree.add_note("n1", "v2")
        assert nh.version == 2
        assert len(tree.notes) == 1
        assert len(tree.history) == 2
        assert tree.history[1].change_type == "note_edit"

    def test_add_multiple_notes(self):
        tree = _make_tree()
        tree.add_note("n1", "content1")
        tree.add_note("n2", "content2")
        assert len(tree.notes) == 2


class TestHashTreeDeleteNote:
    def test_delete_note(self):
        tree = _make_tree()
        tree.add_note("n1", "hello")
        assert tree.delete_note("n1") is True
        assert len(tree.notes) == 0
        assert len(tree.history) == 2  # add + delete

    def test_delete_nonexistent_returns_false(self):
        tree = _make_tree()
        assert tree.delete_note("missing") is False


class TestHashTreeGetAllHashes:
    def test_get_all_hashes(self):
        tree = _make_tree()
        tree.add_note("n1", "content1")
        tree.add_note("n2", "content2")
        hashes = tree.get_all_hashes()
        assert "n1" in hashes
        assert "n2" in hashes
        assert len(hashes) == 2

    def test_empty_tree(self):
        tree = _make_tree()
        assert tree.get_all_hashes() == {}


class TestHashTreeVerify:
    def test_verify_returns_true(self):
        tree = _make_tree()
        tree.add_note("n1", "content")
        assert tree.verify() is True


class TestHashTreeCommit:
    def test_commit_creates_immutable_commit(self):
        tree = _make_tree()
        tree.add_note("n1", "content")
        hc = tree.commit([{"type": "note_add", "note_id": "n1"}])
        assert len(tree.commits) == 1
        assert len(hc.commit_hash) == 64
        assert hc.parent_hash == ""
        assert hc.changes[0]["type"] == "note_add"

    def test_commit_chains_parent(self):
        tree = _make_tree()
        c1 = tree.commit()
        c2 = tree.commit()
        assert c2.parent_hash == c1.commit_hash

    def test_commit_generates_color_and_pixel(self):
        tree = _make_tree()
        hc = tree.commit()
        assert hc.color.startswith("#")
        assert len(hc.color) == 7
        assert len(hc.pixel) == 16


class TestHashTreeVectorize:
    def test_vectorize(self):
        tree = _make_tree()
        tree.add_note("n1", "content")
        tree.commit()
        v = tree.vectorize()
        assert "root_hash" in v
        assert "note_hashes" in v
        assert "pixels" in v
        assert v["commit_count"] == 1


class TestHashTreeRoundTrip:
    def test_to_dict_and_back(self):
        tree = _make_tree()
        tree.add_note("n1", "content")
        tree.commit()
        d = tree.to_dict()
        restored = HashTree.from_dict(d)
        assert len(restored.notes) == 1
        assert len(restored.commits) == 1
        assert restored.root.card_id == "c1"


# ══════════════════════════════════════════════════════════════════════════════
# create_hash_tree factory
# ══════════════════════════════════════════════════════════════════════════════


class TestCreateHashTree:
    def test_create_hash_tree(self):
        tree = create_hash_tree("c1", "my card content", "todo", 0)
        assert tree.root.card_id == "c1"
        assert tree.root.tray == "todo"
        assert tree.root.position == 0
        assert len(tree.root.root) == 64

    def test_create_hash_tree_unique_roots(self):
        t1 = create_hash_tree("c1", "content", "todo", 0)
        t2 = create_hash_tree("c2", "content", "todo", 0)
        assert t1.root.root != t2.root.root


# ══════════════════════════════════════════════════════════════════════════════
# HashTreeStore
# ══════════════════════════════════════════════════════════════════════════════


class TestHashTreeStore:
    def test_save_and_get(self, tmp_path):
        store = HashTreeStore(tmp_path / "ht")
        tree = create_hash_tree("c1", "content", "todo", 0)
        store.save(tree)
        got = store.get("c1")
        assert got is not None
        assert got.root.card_id == "c1"

    def test_get_nonexistent_returns_none(self, tmp_path):
        store = HashTreeStore(tmp_path / "ht")
        assert store.get("missing") is None

    def test_delete(self, tmp_path):
        store = HashTreeStore(tmp_path / "ht")
        tree = create_hash_tree("c1", "content", "todo", 0)
        store.save(tree)
        assert store.delete("c1") is True
        assert store.get("c1") is None

    def test_delete_nonexistent_returns_false(self, tmp_path):
        store = HashTreeStore(tmp_path / "ht")
        assert store.delete("missing") is False

    def test_list_trees(self, tmp_path):
        store = HashTreeStore(tmp_path / "ht")
        store.save(create_hash_tree("c1", "a", "todo", 0))
        store.save(create_hash_tree("c2", "b", "doing", 0))
        assert len(store.list_trees()) == 2

    def test_get_commits(self, tmp_path):
        store = HashTreeStore(tmp_path / "ht")
        tree = create_hash_tree("c1", "content", "todo", 0)
        tree.commit()
        tree.commit()
        store.save(tree)
        commits = store.get_commits("c1")
        assert len(commits) == 2

    def test_get_commits_nonexistent(self, tmp_path):
        store = HashTreeStore(tmp_path / "ht")
        assert store.get_commits("missing") == []

    def test_verify_all(self, tmp_path):
        store = HashTreeStore(tmp_path / "ht")
        t1 = create_hash_tree("c1", "a", "todo", 0)
        t1.add_note("n1", "content")
        store.save(t1)
        t2 = create_hash_tree("c2", "b", "todo", 0)
        store.save(t2)
        results = store.verify_all()
        assert results["c1"] is True
        assert results["c2"] is True

    def test_persistence_across_instances(self, tmp_path):
        store1 = HashTreeStore(tmp_path / "ht")
        store1.save(create_hash_tree("c1", "content", "todo", 0))
        store2 = HashTreeStore(tmp_path / "ht")
        assert store2.get("c1") is not None

    def test_corrupted_file_does_not_crash(self, tmp_path):
        store_dir = tmp_path / "ht"
        store_dir.mkdir(parents=True, exist_ok=True)
        (store_dir / "trees.jsonl").write_text("NOT JSON{{{}\n")
        store = HashTreeStore(store_dir)
        assert store.list_trees() == []
