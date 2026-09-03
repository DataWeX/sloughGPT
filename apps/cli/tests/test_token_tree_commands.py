"""Tests for the CLI token-tree commands (apps/cli/src/commands/token_tree.py)."""
import sys
import os
import pytest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unittest.mock import MagicMock  # noqa: E402

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the quick brown fox is quick. "
    "the lazy dog sleeps. quick brown foxes are quick."
)


@pytest.fixture(autouse=True)
def mock_log(monkeypatch):
    fake_log = MagicMock()
    import commands.token_tree as mod
    monkeypatch.setattr(mod, "log", fake_log)
    return fake_log


@pytest.fixture()
def corpus_file(tmp_path: Path) -> Path:
    p = tmp_path / "corpus.txt"
    p.write_text(CORPUS)
    return p


@pytest.fixture()
def tree_args(tmp_path: Path, corpus_file: Path) -> MagicMock:
    args = MagicMock()
    args.corpus = str(corpus_file)
    args.vocab_size = 64
    args.embed_dim = 16
    args.min_freq = 2
    args.output = str(tmp_path / "tree")
    return args


class TestResolveCorpus:
    def test_explicit_path(self, corpus_file):
        from commands.token_tree import _resolve_corpus_file
        assert _resolve_corpus_file(str(corpus_file)) == corpus_file

    def test_dataset_name_via_datasets_dir(self, tmp_path, monkeypatch):
        from commands.token_tree import _resolve_corpus_file
        d = tmp_path / "data" / "demo"
        d.mkdir(parents=True)
        f = d / "input.txt"
        f.write_text("x")
        monkeypatch.chdir(tmp_path)
        assert _resolve_corpus_file("demo").resolve() == f

    def test_missing_exits(self, monkeypatch, tmp_path):
        from commands.token_tree import _resolve_corpus_file
        monkeypatch.chdir(tmp_path)
        with pytest.raises(SystemExit) as exc:
            _resolve_corpus_file("nope")
        assert exc.value.code == 2


class TestResolveToken:
    def test_numeric_id(self):
        from commands.token_tree import _resolve_token
        from domains.training.token_tree import TokenTree
        tree = TokenTree()
        tree.stoi = {"a": 7}
        assert _resolve_token(tree, "7") == 7

    def test_word_resolves_to_suffixed_form(self):
        from commands.token_tree import _resolve_token
        from domains.training.token_tree import TokenTree
        tree = TokenTree()
        tree.stoi = {"quick</w>": 3, " quick</w>": 5}
        assert _resolve_token(tree, "quick") == 3

    def test_special_token_whole(self):
        from commands.token_tree import _resolve_token
        from domains.training.token_tree import TokenTree
        tree = TokenTree()
        tree.stoi = {"<PAD>": 0, "pad</w>": 1}
        assert _resolve_token(tree, "<PAD>") == 0

    def test_unknown_exits(self):
        from commands.token_tree import _resolve_token
        from domains.training.token_tree import TokenTree
        tree = TokenTree()
        tree.stoi = {"a</w>": 1}
        with pytest.raises(SystemExit):
            _resolve_token(tree, "zzz")


class TestCmdTrainAndQuery:
    def test_train_saves_files(self, corpus_file, tree_args):
        from commands.token_tree import cmd_token_tree_train
        cmd_token_tree_train(tree_args)
        assert Path(str(tree_args.output) + ".meta.json").exists()
        assert Path(str(tree_args.output) + ".points.json").exists()

    def test_train_round_trips_through_cli_helpers(self, corpus_file, tree_args):
        from commands.token_tree import cmd_token_tree_train, _load_tree
        cmd_token_tree_train(tree_args)
        tree = _load_tree(tree_args.output)
        ids = tree.encode("the quick brown fox")
        assert tree.decode(ids) == "the quick brown fox"

    def test_stdin_encode_reads_stdin(self, corpus_file, tree_args, monkeypatch, mock_log):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_encode
        cmd_token_tree_train(tree_args)
        monkeypatch.setattr("sys.stdin", type("S", (), {"read": lambda self: "the lazy dog"})())
        args = MagicMock()
        args.tree = tree_args.output
        args.text = None
        cmd_token_tree_encode(args)
        assert mock_log.table.called

    def test_similar_runs_on_trained_tree(self, corpus_file, tree_args, mock_log):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_similar
        cmd_token_tree_train(tree_args)
        args = MagicMock()
        args.tree = tree_args.output
        args.token = "quick"
        args.top_k = 3
        cmd_token_tree_similar(args)
        assert any("quick" in str(c) for c in mock_log.header.call_args_list)

    def test_lineage_decomposes_token(self, corpus_file, tree_args, mock_log):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_lineage
        cmd_token_tree_train(tree_args)
        args = MagicMock()
        args.tree = tree_args.output
        args.token = "brown"
        cmd_token_tree_lineage(args)
        assert mock_log.header.called

    def test_vocab_lists_special_and_merged_flags(self, corpus_file, tree_args, mock_log):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_vocab
        cmd_token_tree_train(tree_args)
        args = MagicMock()
        args.tree = tree_args.output
        args.offset = 0
        args.limit = 500
        cmd_token_tree_vocab(args)
        assert any("Vocabulary" in str(c) for c in mock_log.header.call_args_list)

    def test_vocab_respects_paging(self, corpus_file, tree_args, mock_log):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_vocab
        cmd_token_tree_train(tree_args)
        args = MagicMock()
        args.tree = tree_args.output
        args.offset = 0
        args.limit = 5
        cmd_token_tree_vocab(args)
        assert any("Showing 1" in str(c) for c in mock_log.info.call_args_list)


class TestCmdEmbeddingAndPath:
    def test_embedding_shows_dim_and_norm(self, corpus_file, tree_args, mock_log):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_embedding
        cmd_token_tree_train(tree_args)
        args = MagicMock()
        args.tree = tree_args.output
        args.token = "quick"
        args.top_k = 3
        cmd_token_tree_embedding(args)
        assert any("Embedding of" in str(c) for c in mock_log.header.call_args_list)

    def test_embedding_no_points_exits(self, corpus_file, tree_args):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_embedding
        tree_args.embed_dim = 0
        cmd_token_tree_train(tree_args)
        args = MagicMock()
        args.tree = tree_args.output
        args.token = "quick"
        args.top_k = 3
        with pytest.raises(SystemExit) as exc:
            cmd_token_tree_embedding(args)
        assert exc.value.code == 2

    def test_embedding_unknown_token_exits(self, corpus_file, tree_args):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_embedding
        cmd_token_tree_train(tree_args)
        args = MagicMock()
        args.tree = tree_args.output
        args.token = "zzz-no-such-token"
        args.top_k = 3
        with pytest.raises(SystemExit):
            cmd_token_tree_embedding(args)

    def test_path_traces_steps_and_ids(self, corpus_file, tree_args, mock_log):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_path
        cmd_token_tree_train(tree_args)
        args = MagicMock()
        args.tree = tree_args.output
        args.text = "the quick brown"
        cmd_token_tree_path(args)
        assert any("Path" in str(c) for c in mock_log.header.call_args_list)

    def test_path_stdin_reads_stdin(self, corpus_file, tree_args, monkeypatch, mock_log):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_path
        cmd_token_tree_train(tree_args)
        monkeypatch.setattr("sys.stdin", type("S", (), {"read": lambda self: "the lazy dog"})())
        args = MagicMock()
        args.tree = tree_args.output
        args.text = None
        cmd_token_tree_path(args)
        assert mock_log.table.called

    def test_path_round_trips_through_cli(self, corpus_file, tree_args):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_path, _load_tree
        cmd_token_tree_train(tree_args)
        args = MagicMock()
        args.tree = tree_args.output
        args.text = "the quick brown fox"
        cmd_token_tree_path(args)
        tree = _load_tree(tree_args.output)
        assert tree.decode(tree.encode("the quick brown fox")) == "the quick brown fox"


class TestLoadTree:
    def test_missing_tree_exits(self, tmp_path):
        from commands.token_tree import _load_tree
        with pytest.raises(SystemExit) as exc:
            _load_tree(str(tmp_path / "missing"))
        assert exc.value.code == 2


class TestCmdMatrix:
    def test_matrix_shows_shape_and_norm_stats(self, corpus_file, tree_args, mock_log):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_matrix
        cmd_token_tree_train(tree_args)
        args = MagicMock()
        args.tree = tree_args.output
        args.top_k = 3
        cmd_token_tree_matrix(args)
        assert any("Embedding matrix" in str(c) for c in mock_log.header.call_args_list)

    def test_matrix_energy_tables(self, corpus_file, tree_args, mock_log):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_matrix
        cmd_token_tree_train(tree_args)
        args = MagicMock()
        args.tree = tree_args.output
        args.top_k = 3
        cmd_token_tree_matrix(args)
        assert any("Most energetic" in str(c) for c in mock_log.header.call_args_list)

    def test_matrix_no_embeddings_exits(self, corpus_file, tree_args):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_matrix
        tree_args.embed_dim = 0
        cmd_token_tree_train(tree_args)
        args = MagicMock()
        args.tree = tree_args.output
        args.top_k = 3
        with pytest.raises(SystemExit) as exc:
            cmd_token_tree_matrix(args)
        assert exc.value.code == 2


class TestCmdCompare:
    def _save_two(self, tmp_path, monkeypatch):
        """Train and save two distinct trees in a temp save dir."""
        monkeypatch.setattr("domains.training.token_tree_manager._SAVE_DIR", tmp_path)
        from domains.training.token_tree_manager import get_token_tree_manager
        mgr = get_token_tree_manager()
        mgr.train(["alpha alpha beta gamma gamma delta"], vocab_size=32, min_frequency=1)
        mgr.save("tree-a")
        mgr.train(["beta beta epsilon zeta zeta"], vocab_size=32, min_frequency=1)
        mgr.save("tree-b")
        return mgr

    def test_compare_prints_overlap(self, tmp_path, monkeypatch, mock_log):
        self._save_two(tmp_path, monkeypatch)
        from commands.token_tree import cmd_token_tree_compare
        args = MagicMock()
        args.a = "tree-a"
        args.b = "tree-b"
        args.top_n = 3
        cmd_token_tree_compare(args)
        assert any("Compare" in str(c) for c in mock_log.header.call_args_list)

    def test_compare_prints_examples(self, tmp_path, monkeypatch, mock_log):
        self._save_two(tmp_path, monkeypatch)
        from commands.token_tree import cmd_token_tree_compare
        args = MagicMock()
        args.a = "tree-a"
        args.b = "tree-b"
        args.top_n = 3
        cmd_token_tree_compare(args)
        assert any("Top shared tokens" in str(c) for c in mock_log.header.call_args_list)

    def test_compare_missing_exits(self, tmp_path, monkeypatch):
        self._save_two(tmp_path, monkeypatch)
        from commands.token_tree import cmd_token_tree_compare
        args = MagicMock()
        args.a = "tree-a"
        args.b = "ghost"
        args.top_n = 3
        with pytest.raises(SystemExit) as exc:
            cmd_token_tree_compare(args)
        assert exc.value.code == 2


class TestCmdMerges:
    def test_merges_prints_ranked_pairs(self, corpus_file, tree_args, mock_log):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_merges
        cmd_token_tree_train(tree_args)
        args = MagicMock()
        args.tree = tree_args.output
        args.top_n = 5
        args.query = ""
        cmd_token_tree_merges(args)
        assert any("Merges" in str(c) for c in mock_log.header.call_args_list)

    def test_merges_filters_by_query(self, corpus_file, tree_args, mock_log):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_merges
        cmd_token_tree_train(tree_args)
        args = MagicMock()
        args.tree = tree_args.output
        args.top_n = 20
        args.query = "qu"
        cmd_token_tree_merges(args)
        assert any("Merges" in str(c) for c in mock_log.header.call_args_list)

    def test_merges_missing_tree_exits(self, tmp_path):
        from commands.token_tree import cmd_token_tree_merges
        args = MagicMock()
        args.tree = str(tmp_path / "missing")
        args.top_n = 5
        args.query = ""
        with pytest.raises(SystemExit) as exc:
            cmd_token_tree_merges(args)
        assert exc.value.code == 2


class TestCmdSavedTrees:
    def _save_one(self, tmp_path, monkeypatch):
        monkeypatch.setattr("domains.training.token_tree_manager._SAVE_DIR", tmp_path)
        from domains.training.token_tree_manager import get_token_tree_manager
        mgr = get_token_tree_manager()
        mgr.train(["alpha alpha beta gamma gamma"], vocab_size=32, min_frequency=1)
        mgr.save("tree-a")
        return mgr

    def test_saved_lists_trees(self, tmp_path, monkeypatch, mock_log):
        self._save_one(tmp_path, monkeypatch)
        from commands.token_tree import cmd_token_tree_saved
        cmd_token_tree_saved(MagicMock())
        assert any("Saved token trees" in str(c) for c in mock_log.header.call_args_list)

    def test_saved_empty(self, tmp_path, monkeypatch, mock_log):
        monkeypatch.setattr("domains.training.token_tree_manager._SAVE_DIR", tmp_path)
        from commands.token_tree import cmd_token_tree_saved
        cmd_token_tree_saved(MagicMock())
        assert any("No saved token trees" in str(c) for c in mock_log.info.call_args_list)

    def test_save_uses_existing_current_tree(self, tmp_path, monkeypatch, mock_log):
        monkeypatch.setattr("domains.training.token_tree_manager._SAVE_DIR", tmp_path)
        from domains.training.token_tree_manager import get_token_tree_manager
        mgr = get_token_tree_manager()
        mgr.train(["alpha alpha beta beta"], vocab_size=32, min_frequency=1)
        from commands.token_tree import cmd_token_tree_save
        args = MagicMock()
        args.name = "mine"
        args.tree = None
        cmd_token_tree_save(args)
        names = [t["name"] for t in mgr.list_saved()]
        assert "mine" in names
        assert any("mine" in str(c) for c in mock_log.success.call_args_list)

    def test_save_adopts_tree_from_path(self, tmp_path, monkeypatch, mock_log):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_save
        tree_args = MagicMock()
        tree_args.corpus = str(tmp_path / "corpus.txt")
        (tmp_path / "corpus.txt").write_text("the quick brown fox jumps over the lazy dog")
        tree_args.vocab_size = 32
        tree_args.embed_dim = 16
        tree_args.min_freq = 1
        tree_args.output = str(tmp_path / "trained")
        cmd_token_tree_train(tree_args)

        monkeypatch.setattr("domains.training.token_tree_manager._SAVE_DIR", tmp_path / "save")
        from domains.training.token_tree_manager import get_token_tree_manager
        mgr = get_token_tree_manager()
        args = MagicMock()
        args.name = "adopted"
        args.tree = tree_args.output
        cmd_token_tree_save(args)
        names = [t["name"] for t in mgr.list_saved()]
        assert "adopted" in names

    def test_save_invalid_name_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr("domains.training.token_tree_manager._SAVE_DIR", tmp_path)
        from commands.token_tree import cmd_token_tree_save
        args = MagicMock()
        args.name = "../evil"
        args.tree = None
        with pytest.raises(SystemExit) as exc:
            cmd_token_tree_save(args)
        assert exc.value.code == 2

    def test_save_missing_tree_path_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr("domains.training.token_tree_manager._SAVE_DIR", tmp_path)
        from commands.token_tree import cmd_token_tree_save
        args = MagicMock()
        args.name = "mine"
        args.tree = str(tmp_path / "missing")
        with pytest.raises(SystemExit) as exc:
            cmd_token_tree_save(args)
        assert exc.value.code == 2

    def test_load_makes_tree_current(self, tmp_path, monkeypatch, mock_log):
        mgr = self._save_one(tmp_path, monkeypatch)
        from commands.token_tree import cmd_token_tree_load
        args = MagicMock()
        args.name = "tree-a"
        cmd_token_tree_load(args)
        assert any("Loaded" in str(c) for c in mock_log.success.call_args_list)
        assert mgr.is_trained()

    def test_load_missing_exits(self, tmp_path, monkeypatch):
        self._save_one(tmp_path, monkeypatch)
        from commands.token_tree import cmd_token_tree_load
        args = MagicMock()
        args.name = "ghost"
        with pytest.raises(SystemExit) as exc:
            cmd_token_tree_load(args)
        assert exc.value.code == 2

    def test_delete_removes_tree(self, tmp_path, monkeypatch, mock_log):
        mgr = self._save_one(tmp_path, monkeypatch)
        from commands.token_tree import cmd_token_tree_delete
        args = MagicMock()
        args.name = "tree-a"
        cmd_token_tree_delete(args)
        assert mgr.list_saved() == []
        assert any("Deleted" in str(c) for c in mock_log.success.call_args_list)

    def test_delete_missing_exits(self, tmp_path, monkeypatch):
        self._save_one(tmp_path, monkeypatch)
        from commands.token_tree import cmd_token_tree_delete
        args = MagicMock()
        args.name = "ghost"
        with pytest.raises(SystemExit) as exc:
            cmd_token_tree_delete(args)
        assert exc.value.code == 2
