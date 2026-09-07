"""Tests for training.tokenizer — SloBPE class."""

from __future__ import annotations

import os
import tempfile
import pytest
import numpy as np
from domains.training.tokenizer import SloBPE


# ── SloBPE init ─────────────────────────────────────────────────────────────


class TestSloBPEInit:

    def test_default(self):
        tok = SloBPE()
        assert tok.vocab_size == 0
        assert tok._pretokenizer == "gpt2"

    def test_whitespace(self):
        tok = SloBPE(pretokenizer="whitespace")
        assert tok._pretokenizer == "whitespace"

    def test_properties(self):
        tok = SloBPE()
        assert tok.vocab_size == 0
        assert tok.pad_id == 0
        assert tok.unk_id == 1
        assert tok.bos_id == 2
        assert tok.eos_id == 3


# ── SloBPE.train ────────────────────────────────────────────────────────────


class TestSloBPETrain:

    def test_train(self):
        tok = SloBPE()
        tok.train(["hello world"], vocab_size=32)
        assert tok.vocab_size > 0

    def test_train_empty(self):
        tok = SloBPE()
        with pytest.raises(ValueError, match="at least one text"):
            tok.train([])

    def test_encode_decode(self):
        tok = SloBPE()
        tok.train(["hello world", "hello there"], vocab_size=32)
        ids = tok.encode("hello world")
        assert isinstance(ids, list)
        assert len(ids) > 0
        text = tok.decode(ids)
        assert isinstance(text, str)

    def test_encode_empty(self):
        tok = SloBPE()
        tok.train(["hello"], vocab_size=32)
        ids = tok.encode("")
        assert ids == []

    def test_batch_encode(self):
        tok = SloBPE()
        tok.train(["hello world", "foo bar"], vocab_size=32)
        batch = [tok.encode(t) for t in ["hello", "world"]]
        assert len(batch) == 2
        for ids in batch:
            assert len(ids) > 0

    def test_merges(self):
        tok = SloBPE()
        tok.train(["hello world hello world"], vocab_size=32)
        assert isinstance(tok.merges, list)


# ── SloBPE.save/load ───────────────────────────────────────────────────────


class TestSloBPESaveLoad:

    def test_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = SloBPE()
            tok.train(["hello world"], vocab_size=32)
            path = os.path.join(tmpdir, "test_tok")
            tok.save(path)

            tok2 = SloBPE.load(path)
            assert tok2.vocab_size == tok.vocab_size

    def test_encode_after_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tok = SloBPE()
            tok.train(["hello world"], vocab_size=32)
            path = os.path.join(tmpdir, "test_tok")
            tok.save(path)

            tok2 = SloBPE.load(path)
            ids1 = tok.encode("hello")
            ids2 = tok2.encode("hello")
            assert ids1 == ids2


# ── SloBPE with BOS/EOS ────────────────────────────────────────────────────


class TestSloBPEBosEos:

    def test_encode_with_bos_eos(self):
        tok = SloBPE()
        tok.train(["hello world"], vocab_size=32)
        ids = tok.encode("hello", add_bos=True, add_eos=True)
        assert ids[0] == tok.bos_id
        assert ids[-1] == tok.eos_id


# ── pretokenize helpers ────────────────────────────────────────────────────


class TestPretokenizeHelpers:

    def test_gpt2_pretokenize(self):
        from domains.training.tokenizer import gpt2_pretokenize
        result = gpt2_pretokenize("hello world")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_default_pretokenize(self):
        from domains.training.tokenizer import default_pretokenize
        result = default_pretokenize("hello world")
        assert isinstance(result, list)
        assert len(result) > 0
