"""
Tests for TokenizerManager (domains/training/tokenizer_manager.py).
"""

import os
import tempfile

import pytest

from domain.training._internal.tokenizer_manager import TokenizerManager, get_tokenizer_manager


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton before and after each test."""
    TokenizerManager._instance = None
    yield
    TokenizerManager._instance = None


class TestTokenizerManager:
    """Tests for the TokenizerManager singleton."""

    def test_get_instance(self):
        mgr = TokenizerManager.get_instance()
        assert mgr is not None
        assert isinstance(mgr, TokenizerManager)

    def test_singleton(self):
        mgr1 = TokenizerManager.get_instance()
        mgr2 = TokenizerManager.get_instance()
        assert mgr1 is mgr2

    def test_global_helper(self):
        mgr = get_tokenizer_manager()
        assert isinstance(mgr, TokenizerManager)

    def test_global_singleton_consistency(self):
        mgr1 = get_tokenizer_manager()
        mgr2 = get_tokenizer_manager()
        assert mgr1 is mgr2

    def test_get_tokenizer_creates_default(self):
        mgr = TokenizerManager.get_instance()
        tok = mgr.get_tokenizer()
        assert tok is not None
        assert hasattr(tok, "encode")
        assert hasattr(tok, "decode")

    def test_is_trained_false_initially(self):
        mgr = TokenizerManager.get_instance()
        assert mgr.is_trained() is False

    def test_train_and_is_trained(self):
        mgr = TokenizerManager.get_instance()
        texts = ["hello world", "test data", "machine learning"]
        mgr.train(texts, vocab_size=50, min_frequency=1)
        assert mgr.is_trained() is True

    def test_train_returns_stats(self):
        mgr = TokenizerManager.get_instance()
        texts = ["hello world", "test data"]
        stats = mgr.train(texts, vocab_size=50, min_frequency=1)
        assert "vocab_size" in stats
        assert stats["vocab_size"] > 0

    def test_tokenize_untrained_still_works(self):
        mgr = TokenizerManager.get_instance()
        ids = mgr.tokenize("test")
        assert isinstance(ids, list)
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_tokenize_and_detokenize(self):
        mgr = TokenizerManager.get_instance()
        mgr.train(["hello world test"], vocab_size=50, min_frequency=1)
        ids = mgr.tokenize("hello")
        decoded = mgr.detokenize(ids)
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_vocab_size_property(self):
        mgr = TokenizerManager.get_instance()
        assert mgr.vocab_size == 0
        mgr.train(["test"], vocab_size=50, min_frequency=1)
        assert mgr.vocab_size > 0

    def test_stats_before_training(self):
        mgr = TokenizerManager.get_instance()
        stats = mgr.stats()
        assert stats["trained"] is False
        assert stats["vocab_size"] == 0

    def test_stats_after_training(self):
        mgr = TokenizerManager.get_instance()
        mgr.train(["hello world"], vocab_size=50, min_frequency=1)
        stats = mgr.stats()
        assert stats["trained"] is True
        assert stats["vocab_size"] > 0

    def test_save_and_load_roundtrip(self):
        mgr = TokenizerManager.get_instance()
        mgr.train(["hello world test data"], vocab_size=50, min_frequency=1)
        orig_vocab = mgr.vocab_size
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            mgr.save(path)
            mgr2 = TokenizerManager.get_instance()
            mgr2.load(path)
            assert mgr2.vocab_size == orig_vocab
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_to_dict_and_from_dict(self):
        mgr = TokenizerManager.get_instance()
        mgr.train(["hello world"], vocab_size=50, min_frequency=1)
        data = mgr.to_dict()
        assert "merges" in data
        assert len(data.get("vocab", [])) > 0
        mgr2 = TokenizerManager.get_instance()
        mgr2.from_dict(data)
        assert mgr2.vocab_size == len(data.get("vocab", []))

    def test_adopt(self):
        mgr = TokenizerManager.get_instance()
        mgr.train(["test"], vocab_size=50, min_frequency=1)
        tok = mgr._tokenizer
        mgr.reset()
        assert mgr.is_trained() is False
        mgr.adopt(tok)
        assert mgr.is_trained() is True

    def test_adopt_invalid(self):
        mgr = TokenizerManager.get_instance()
        mgr.adopt(None)
        assert mgr.is_trained() is False

    def test_set_tokenizer(self):
        mgr = TokenizerManager.get_instance()
        mgr.train(["test"], vocab_size=50, min_frequency=1)
        tok = mgr._tokenizer
        mgr.reset()
        mgr.set_tokenizer(tok)
        assert mgr.is_trained() is True

    def test_reset(self):
        mgr = TokenizerManager.get_instance()
        mgr.train(["test"], vocab_size=50, min_frequency=1)
        mgr.reset()
        assert mgr.is_trained() is False
        assert mgr.vocab_size == 0

    def test_borrow_from_autotrain_no_state(self):
        mgr = TokenizerManager.get_instance()
        mgr.reset()
        # Reset auto-train state if the module has been imported (avoids cross-test pollution
        # from tests that call POST /auto-train/start and leave state.student_tokenizer set)
        import sys as _sys

        if "routers.auto_train" in _sys.modules:
            _sys.modules["routers.auto_train"].state.student_tokenizer = None
        result = mgr.borrow_from_autotrain()
        assert result is False

    def test_to_dict_preserves_merges(self):
        mgr = TokenizerManager.get_instance()
        mgr.train(["hello world test data"], vocab_size=50, min_frequency=1)
        data = mgr.to_dict()
        mgr2 = TokenizerManager.get_instance()
        mgr2.from_dict(data)
        ids = mgr2.tokenize("hello world")
        assert len(ids) > 0


# ── Checkpoint binding (card 07c4e10a) ──────────────────────────────────────


def _write_soul_with_tree(tmp_path, corpus, vocab_size=64, name="demo"):
    """Write a `.soul` + sidecar whose metadata embeds a trained TokenTree.

    Mirrors exactly what ``train_pipeline`` writes:
    ``metadata["tokenizer"] = {"type": "token_tree", "tree": <tree.to_dict()>}``
    plus the ``tokenizer_format`` tag.
    """
    import json

    from domain.training._internal.token_tree import TokenTree

    tree = TokenTree()
    tree.train(corpus, vocab_size=vocab_size)
    soul = tmp_path / f"{name}.soul"
    soul.write_bytes(b"not-a-real-soul-but-present")
    sidecar = {
        "soul_name": name,
        "metadata": {
            "vocab_size": tree.vocab_size,
            "tokenizer": {"type": "token_tree", "tree": tree.to_dict()},
            "tokenizer_format": "native/token-tree",
        },
    }
    (tmp_path / f"{name}.soul.meta.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return soul, tree


@pytest.fixture
def root_confined(tmp_path, monkeypatch):
    """Redirect the checkpoint search roots to tmp_path (traversal-safe path)."""
    from domain.training._internal import checkpoints

    monkeypatch.setattr(checkpoints, "ckpt_roots", lambda: (tmp_path,))
    return tmp_path


class TestCheckpointBinding:
    """The inspector must resolve its tokenizer from a checkpoint's identity."""

    def test_no_checkpoint_returns_singleton(self):
        mgr = TokenizerManager.get_instance()
        mgr.train(["hello world test"], vocab_size=50, min_frequency=1)
        tok, source = mgr.tokenizer_for()
        assert source["kind"] == "singleton"
        assert tok is mgr.get_tokenizer()

    def test_checkpoint_returns_embedded_tokenizer(self, root_confined):
        soul, tree = _write_soul_with_tree(root_confined, ["hello world", "hello there"])
        mgr = TokenizerManager.get_instance()
        tok, source = mgr.tokenizer_for(soul.name)
        assert source["kind"] == "checkpoint"
        assert source["path"].endswith("demo.soul")
        assert source["variant"] in ("tree", "char")
        # It is the CHECKPOINT's tokenizer, byte-identical to the embedded one.
        text = "hello world"
        assert tok.encode(text) == tree.encode(text)
        assert tok.decode(tree.encode(text)) == text

    def test_checkpoint_does_not_mutate_singleton(self, root_confined):
        soul, _ = _write_soul_with_tree(root_confined, ["hello world"])
        mgr = TokenizerManager.get_instance()
        mgr.train(["something else entirely"], vocab_size=50, min_frequency=1)
        before = mgr.get_tokenizer()
        mgr.tokenizer_for(soul.name)
        assert mgr.get_tokenizer() is before

    def test_missing_checkpoint_raises(self, root_confined):
        mgr = TokenizerManager.get_instance()
        with pytest.raises(ValueError, match="not found"):
            mgr.tokenizer_for("nope.soul")

    def test_sidecar_without_tokenizer_raises(self, root_confined):
        import json

        soul = root_confined / "bare.soul"
        soul.write_bytes(b"present")
        (root_confined / "bare.soul.meta.json").write_text(
            json.dumps({"soul_name": "bare", "metadata": {"vocab_size": 10}}), encoding="utf-8"
        )
        mgr = TokenizerManager.get_instance()
        with pytest.raises(ValueError, match="tokenizer"):
            mgr.tokenizer_for("bare.soul")

    def test_stats_for_checkpoint_tokenizer(self, root_confined):
        soul, tree = _write_soul_with_tree(root_confined, ["hello world", "hello there"])
        mgr = TokenizerManager.get_instance()
        tok, source = mgr.tokenizer_for(soul.name)
        stats = mgr.stats_for(tok, source)
        assert stats["trained"] is True
        assert stats["vocab_size"] == tree.vocab_size
        assert stats["kind"] == "checkpoint"

    def test_vocab_and_merges_normalized_for_checkpoint(self, root_confined):
        soul, tree = _write_soul_with_tree(root_confined, ["hello world", "hello there"])
        mgr = TokenizerManager.get_instance()
        tok, _ = mgr.tokenizer_for(soul.name)
        vocab = mgr.vocab_for(tok)
        merges = mgr.merges_for(tok)
        assert vocab == list(tree.vocab)
        assert merges == [tuple(m) for m in tree.merges]
