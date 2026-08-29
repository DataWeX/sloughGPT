"""Tests for domains.multimodal.engine — MultimodalOutput, TextDecoder,
ReplayBuffer, augment_image.

Covers: dataclass, vocab encode/decode roundtrip, replay buffer add/sample/evict,
image augmentation shape preservation. Excludes heavy model init.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.multimodal.engine import (
    MultimodalOutput,
    TextDecoder,
    ReplayBuffer,
    augment_image,
)


class TestMultimodalOutput:
    def test_creation(self):
        o = MultimodalOutput(text="a cat", confidence=0.9)
        assert o.text == "a cat"
        assert o.confidence == 0.9


class TestTextDecoder:
    def test_build_vocab(self):
        td = TextDecoder()
        td.build_vocab(["hello world", "test data"])
        assert td.vocab_size > 0

    def test_encode_decode_roundtrip(self):
        td = TextDecoder()
        td.build_vocab(["hello"])
        ids = td.encode("hello")
        text = td.decode(ids)
        assert text == "hello"

    def test_vocab_size(self):
        td = TextDecoder()
        td.build_vocab(["abc"])
        assert td.vocab_size > 0


class TestReplayBuffer:
    def test_add_and_size(self):
        buf = ReplayBuffer(capacity=10)
        img = np.ones((1, 8, 8, 3))
        buf.add(img, "caption a")
        assert buf.size == 1

    def test_capacity_eviction(self):
        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            buf.add(np.ones((1, 4, 4, 3)), f"cap {i}")
        assert buf.size == 3
        assert buf.captions[-1] == "cap 4"

    def test_sample_all(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(3):
            buf.add(np.ones((1, 4, 4, 3)), f"cap {i}")
        imgs, caps = buf.sample(10)
        assert len(imgs) == 3
        assert len(caps) == 3

    def test_sample_n(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(5):
            buf.add(np.ones((1, 4, 4, 3)), f"cap {i}")
        imgs, caps = buf.sample(3)
        assert len(imgs) == 3
        assert len(caps) == 3

    def test_diverse_sampling(self):
        buf = ReplayBuffer(capacity=10)
        for i in range(10):
            buf.add(np.ones((1, 4, 4, 3)), "same caption")
        imgs, caps = buf.sample(5)
        assert len(imgs) == 5

    def test_eviction_updates_counts(self):
        buf = ReplayBuffer(capacity=2)
        buf.add(np.ones((1, 4, 4, 3)), "a")
        buf.add(np.ones((1, 4, 4, 3)), "a")
        buf.add(np.ones((1, 4, 4, 3)), "b")
        assert buf.size == 2
        assert "a" not in buf._counts or buf._counts.get("a", 0) == 1


class TestAugmentImage:
    def test_preserves_shape(self):
        img = np.random.rand(1, 32, 32, 3).astype(np.float32)
        result = augment_image(img)
        assert result.shape == img.shape

    def test_output_range(self):
        img = np.random.rand(1, 16, 16, 3).astype(np.float32) * 0.5 + 0.25
        result = augment_image(img)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_does_not_modify_original(self):
        img = np.ones((1, 8, 8, 3), dtype=np.float32) * 0.5
        original = img.copy()
        augment_image(img)
        np.testing.assert_array_equal(img, original)
