"""Tests for domains.infrastructure.slnc.compiler — helper functions."""

import numpy as np
from domains.infrastructure.slnc.compiler import (
    _crc32, _xxhash64, SLNCCompiler,
)


class TestCrc32:
    def test_deterministic(self):
        assert _crc32(b"hello") == _crc32(b"hello")

    def test_different_inputs(self):
        assert _crc32(b"hello") != _crc32(b"world")

    def test_returns_int(self):
        result = _crc32(b"test")
        assert isinstance(result, int)
        assert 0 <= result <= 0xFFFFFFFF


class TestXxhash64:
    def test_deterministic(self):
        assert _xxhash64(b"hello") == _xxhash64(b"hello")

    def test_different_inputs(self):
        assert _xxhash64(b"hello") != _xxhash64(b"world")


class TestSLNCCompiler:
    def test_init(self):
        comp = SLNCCompiler()
        assert comp is not None

    def test_compute_block_size_gpt2(self):
        comp = SLNCCompiler()
        config = {"n_embd": 768, "n_inner": 3072}
        size = comp._compute_block_size(config)
        assert size > 0

    def test_compute_block_size_llama(self):
        comp = SLNCCompiler()
        config = {"hidden_size": 768, "intermediate_size": 2048, "rope_theta": 10000.0}
        size = comp._compute_block_size(config)
        assert size > 0

    def test_order_tensors(self):
        comp = SLNCCompiler()
        config = {"n_layer": 2}
        weights = {
            "h.0.attn.c_attn.weight": np.zeros((3, 3)),
            "h.1.attn.c_attn.weight": np.zeros((3, 3)),
            "wte.weight": np.zeros((10, 3)),
        }
        ordered = comp._order_tensors(config, weights)
        assert isinstance(ordered, list)
        assert len(ordered) == 3
