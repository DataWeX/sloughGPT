"""
Tests for TokenizerManager (domains/training/tokenizer_manager.py).
"""

import os
import tempfile
import pytest

from domains.training.tokenizer_manager import TokenizerManager, get_tokenizer_manager


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
        if 'routers.auto_train' in _sys.modules:
            _sys.modules['routers.auto_train'].state.student_tokenizer = None
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
