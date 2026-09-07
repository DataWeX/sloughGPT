"""Tests for training.chat_trainer — ChatTrainConfig and ChatTextDataset."""

from __future__ import annotations

import pytest
import numpy as np
from domains.training.chat_trainer import (
    ChatTrainConfig,
    ChatTextDataset,
    _build_vocab,
)


# ── ChatTrainConfig ────────────────────────────────────────────────────────


class TestChatTrainConfig:

    def test_default(self):
        config = ChatTrainConfig()
        assert config.n_embed == 128
        assert config.n_layer == 4
        assert config.epochs == 10

    def test_custom(self):
        config = ChatTrainConfig(n_embed=64, n_layer=2)
        assert config.n_embed == 64
        assert config.n_layer == 2


# ── ChatTextDataset ────────────────────────────────────────────────────────


class TestChatTextDataset:

    def test_init(self):
        text = "hello world"
        stoi = {"h": 1, "e": 2, "l": 3, "o": 4, " ": 5, "w": 6, "r": 7, "d": 8, "\x00": 0}
        dataset = ChatTextDataset(text, block_size=5, stoi=stoi)
        assert len(dataset) > 0

    def test_get_batch(self):
        text = "hello world " * 10
        stoi = {"h": 1, "e": 2, "l": 3, "o": 4, " ": 5, "w": 6, "r": 7, "d": 8, "\x00": 0}
        dataset = ChatTextDataset(text, block_size=5, stoi=stoi)
        rng = np.random.default_rng(42)
        x, y = dataset.get_batch(4, rng)
        assert x.shape == (4, 5)
        assert y.shape == (4, 5)


# ── _build_vocab ────────────────────────────────────────────────────────────


class TestBuildVocab:

    def test_build_vocab(self):
        pairs = [
            {"user_msg": "hello", "assistant_msg": "world"},
            {"user_msg": "test", "assistant_msg": "data"},
        ]
        stoi, itos = _build_vocab(pairs)
        assert "\x00" in stoi
        assert len(stoi) > 0
        assert len(itos) > 0
