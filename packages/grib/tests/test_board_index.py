"""Tests for the board index — the read-only MogDB query layer over board.jsonl.

The contract under test, card ``20261101_095701_kanban-read-side-...``:

- JSONL stays the source of truth; the index never writes ``board.jsonl``.
- ``sync()`` mirrors what ``board show`` renders (defensive ``load_board``).
- Reads are freshness-checked: a card added / deleted behind the index's back
  is picked up (re-sync + prune) without an explicit ``sync()`` call.
- Any MogDB failure silently falls back to a plain file read.
- Isolation: the index derives from ``board_dir``; a custom board dir never
  touches the canonical ``~/.config/kanban`` store.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from grib.board_index import BoardIndex, filter_cards
from grib.cli import main as cli_main
from grib.store import PlannerStore, reset_store


@pytest.fixture(autouse=True)
def _reset():
    reset_store()
    yield
    reset_store()


@pytest.fixture
def store(tmp_path):
    return PlannerStore(board_dir=tmp_path / "board", notes_dir=tmp_path / "board")


def _seed(store: PlannerStore) -> None:
    store.add_card("Port readers to MogDB", column="done", priority="high", tags=["mogdb"])
    store.add_card("Write index tests", column="wip", priority="medium", tags=["testing", "mogdb"])
    store.add_card("Design doc", column="todo", priority="low", tags=["docs"])


def _run(store_dir, *parts) -> tuple[int, str]:
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            code = cli_main(["--board-dir", str(store_dir), "--notes-dir", str(store_dir), *parts])
        except SystemExit as e:
            code = e.code or 1
    return code, buf.getvalue()


class TestRoundtrip:
    def test_sync_and_query_all(self, store, tmp_path):
        _seed(store)
        index = BoardIndex(board_dir=tmp_path / "board")
        try:
            assert index.sync() == 3
            assert len(index.query()) == 3
        finally:
            index.close()

    def test_filter_by_column_priority_assignee(self, store, tmp_path):
        _seed(store)
        index = BoardIndex(board_dir=tmp_path / "board")
        try:
            index.sync()
            assert [c["title"] for c in index.query(column="done")] == ["Port readers to MogDB"]
            assert len(index.query(priority="medium")) == 1
            assert index.query(column="todo")[0]["title"] == "Design doc"
            store.update_card(
                [c.id for c in store.load_board().cards if c.title == "Design doc"][0],
                assignee="alex",
            )
            assert index.query(assignee="alex")[0]["title"] == "Design doc"
        finally:
            index.close()

    def test_filter_by_tag_and_text(self, store, tmp_path):
        _seed(store)
        index = BoardIndex(board_dir=tmp_path / "board")
        try:
            index.sync()
            # tag matches BOTH cards carrying it (array containment post-filter)
            assert len(index.query(tag="mogdb")) == 2
            # text is case-insensitive over title+description
            assert "design-doc" in index.query(text="design")[0]["id"]
            assert index.query(text="DOCS") == []
            assert len(index.query(tag="mogdb", column="wip")) == 1
        finally:
            index.close()

    def test_limit(self, store, tmp_path):
        _seed(store)
        index = BoardIndex(board_dir=tmp_path / "board")
        try:
            index.sync()
            assert len(index.query(limit=2)) == 2
        finally:
            index.close()

    def test_query_result_has_card_fields(self, store, tmp_path):
        _seed(store)
        index = BoardIndex(board_dir=tmp_path / "board")
        try:
            index.sync()
            card = index.query(text="mogdb", column="done")[0]
            for field in ("id", "title", "column", "priority", "tags", "created_at"):
                assert field in card, f"missing {field}"
            # MogDB bookkeeping keys never leak
            assert "_created" not in card and "_updated" not in card and "_id" not in card
        finally:
            index.close()


class TestStalenessAndPruning:
    def test_new_card_picked_up_without_explicit_sync(self, store, tmp_path):
        _seed(store)
        index = BoardIndex(board_dir=tmp_path / "board")
        try:
            index.sync()
            assert len(index.query()) == 3
            store.add_card("Fresh card", column="todo")
            # No explicit sync — the mtime check must re-sync transparently.
            assert len(index.query()) == 4
        finally:
            index.close()

    def test_deleted_card_pruned(self, store, tmp_path):
        _seed(store)
        index = BoardIndex(board_dir=tmp_path / "board")
        try:
            index.sync()
            board = store.load_board()
            target = [c.id for c in board.cards if c.title == "Design doc"][0]
            store.delete_card(target)
            titles = [c["title"] for c in index.query()]
            assert "Design doc" not in titles
            assert len(titles) == 2
        finally:
            index.close()

    def test_upsert_replaces_same_id(self, store, tmp_path):
        _seed(store)
        index = BoardIndex(board_dir=tmp_path / "board")
        try:
            index.sync()
            board = store.load_board()
            cid = [c.id for c in board.cards if c.title == "Design doc"][0]
            store.update_card(cid, title="Design doc v2", column="wip")
            found = index.query(text="v2")
            assert len(found) == 1
            assert found[0]["column"] == "wip"
            assert len(index.query()) == 3  # no duplicate
        finally:
            index.close()


class TestFallback:
    def test_query_falls_back_when_mogdb_import_fails(self, store, tmp_path, monkeypatch):
        _seed(store)

        # Make the lazy `from mogdb import MogDB` raise, as if the package
        # were absent. The index must silently serve a file read.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mogdb":
                raise ImportError("mogdb unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        index = BoardIndex(board_dir=tmp_path / "board")
        try:
            cards = index.query(column="done")
            assert [c["title"] for c in cards] == ["Port readers to MogDB"]
            assert len(index.query()) == 3
        finally:
            index.close()

    def test_sync_raises_cleanly_when_unavailable(self, store, tmp_path, monkeypatch):
        _seed(store)
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mogdb":
                raise ImportError("mogdb unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        index = BoardIndex(board_dir=tmp_path / "board")
        try:
            with pytest.raises(ImportError):
                index.sync()
        finally:
            index.close()

    def test_cli_query_works_with_mogdb_unavailable(self, store, tmp_path, monkeypatch):
        _seed(store)
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "mogdb":
                raise ImportError("mogdb unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        code, out = _run(tmp_path / "board", "board", "query", "--column", "done")
        assert code == 0
        assert "Port readers to MogDB" in out


class TestReadOnlyAndIsolation:
    def test_index_never_writes_board_jsonl(self, store, tmp_path):
        _seed(store)
        board_file = tmp_path / "board" / "board.jsonl"
        before = board_file.read_bytes()
        index = BoardIndex(board_dir=tmp_path / "board")
        try:
            index.sync()
            index.query()
            index.query(column="wip")
        finally:
            index.close()
        assert board_file.read_bytes() == before, "index mutated board.jsonl"

    def test_index_dir_derived_from_board_dir(self, store, tmp_path):
        _seed(store)
        index = BoardIndex(board_dir=tmp_path / "board")
        try:
            index.sync()
        finally:
            index.close()
        assert index.index_dir == tmp_path / "board" / "index"
        assert (tmp_path / "board" / "index").exists()

    def test_isolated_from_canonical_store(self, store, tmp_path, monkeypatch):
        """A custom board dir's index must not touch ~/.config/kanban."""
        _seed(store)
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
        index = BoardIndex(board_dir=tmp_path / "board")
        try:
            index.sync()
        finally:
            index.close()
        # canonical fallback dir was never created or written
        assert not (fake_home / ".config" / "kanban").exists()


class TestFilterHelper:
    def test_filter_cards_no_filters_passthrough(self):
        docs = [{"id": "a", "title": "x", "tags": []}]
        assert filter_cards(docs) == docs

    def test_filter_cards_tag_containment(self):
        docs = [{"id": "a", "tags": ["x", "y"]}, {"id": "b", "tags": ["z"]}]
        assert [d["id"] for d in filter_cards(docs, tag="y")] == ["a"]

    def test_filter_cards_text_searches_description_too(self):
        docs = [{"id": "a", "title": "one", "description": "find ME here"}]
        assert len(filter_cards(docs, text="me")) == 1
