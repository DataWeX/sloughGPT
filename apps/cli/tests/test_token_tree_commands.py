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
        d = tmp_path / "datasets" / "demo"
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

    def test_train_round_trips_through_cli_helpers(self, corpus_file, tree_args, capsys):
        from commands.token_tree import cmd_token_tree_train, _load_tree
        cmd_token_tree_train(tree_args)
        tree = _load_tree(tree_args.output)
        ids = tree.encode("the quick brown fox")
        assert tree.decode(ids) == "the quick brown fox"

    def test_stdin_encode_reads_stdin(self, corpus_file, tree_args, monkeypatch, capsys):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_encode
        cmd_token_tree_train(tree_args)
        monkeypatch.setattr("sys.stdin", type("S", (), {"read": lambda self: "the lazy dog"})())
        args = MagicMock()
        args.tree = tree_args.output
        args.text = None
        cmd_token_tree_encode(args)
        out = capsys.readouterr().out
        assert "lazy" in out

    def test_similar_runs_on_trained_tree(self, corpus_file, tree_args, capsys):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_similar
        cmd_token_tree_train(tree_args)
        args = MagicMock()
        args.tree = tree_args.output
        args.token = "quick"
        args.top_k = 3
        cmd_token_tree_similar(args)
        out = capsys.readouterr().out
        assert "quick" in out

    def test_lineage_decomposes_token(self, corpus_file, tree_args, capsys):
        from commands.token_tree import cmd_token_tree_train, cmd_token_tree_lineage
        cmd_token_tree_train(tree_args)
        args = MagicMock()
        args.tree = tree_args.output
        args.token = "brown"
        cmd_token_tree_lineage(args)
        out = capsys.readouterr().out
        assert "brown" in out


class TestLoadTree:
    def test_missing_tree_exits(self, tmp_path, monkeypatch):
        from commands.token_tree import _load_tree
        with pytest.raises(SystemExit) as exc:
            _load_tree(str(tmp_path / "missing"))
        assert exc.value.code == 2
