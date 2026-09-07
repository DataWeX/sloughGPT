"""Tests for training.tokenizer_manager — TokenizerManager class."""

from __future__ import annotations

import os
import tempfile
import pytest
import numpy as np
from domains.training.tokenizer_manager import TokenizerManager


# ── TokenizerManager ───────────────────────────────────────────────────────


class TestTokenizerManager:

    def test_init(self):
        mgr = TokenizerManager()
        assert mgr._algo == "bpe"

    def test_get_instance(self):
        mgr = TokenizerManager.get_instance()
        assert isinstance(mgr, TokenizerManager)

    def test_get_tokenizer(self):
        mgr = TokenizerManager()
        tok = mgr.get_tokenizer()
        assert tok is not None

    def test_train_bpe(self):
        mgr = TokenizerManager()
        result = mgr.train(["hello world"], vocab_size=32)
        assert result is not None

    def test_tokenize(self):
        mgr = TokenizerManager()
        mgr.train(["hello world"], vocab_size=32)
        ids = mgr.tokenize("hello world")
        assert isinstance(ids, list)
        assert len(ids) > 0

    def test_detokenize(self):
        mgr = TokenizerManager()
        mgr.train(["hello world"], vocab_size=32)
        ids = mgr.tokenize("hello world")
        text = mgr.detokenize(ids)
        assert isinstance(text, str)

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = TokenizerManager()
            mgr.train(["hello world"], vocab_size=32)
            path = os.path.join(tmpdir, "test_tokenizer.json")
            mgr.save(path)

            mgr2 = TokenizerManager()
            mgr2.load(path)
            ids1 = mgr.tokenize("hello")
            ids2 = mgr2.tokenize("hello")
            assert ids1 == ids2

    def test_stats(self):
        mgr = TokenizerManager()
        mgr.train(["hello world"], vocab_size=32)
        stats = mgr.stats()
        assert "vocab_size" in stats

    def test_vocab_size(self):
        mgr = TokenizerManager()
        mgr.train(["hello world"], vocab_size=32)
        assert mgr.vocab_size > 0

    def test_encode_decode(self):
        mgr = TokenizerManager()
        mgr.train(["hello world"], vocab_size=32)
        ids = mgr.tokenize("hello world")
        text = mgr.detokenize(ids)
        assert isinstance(text, str)
