"""Tests for NumpyBE compute backend."""

import numpy as np
import pytest

from domains.infrastructure.numpy_backend import NumpyBE
from domains.infrastructure.arch_config import ArchConfig, LLAMA_WEIGHT_MAP


def _make_tiny_arch():
    """Build minimal ArchConfig for testing."""
    arch = ArchConfig(
        name="tiny_test",
        norm="rms_norm",
        positional="rope",
        activation="swiglu",
        attention="gqa",
        weight_map=LLAMA_WEIGHT_MAP,
    )
    arch.n_layers = 2
    arch.n_head = 4
    arch.n_kv_head = 2
    arch.n_embed = 64
    arch.head_dim = 16
    arch.rope_base = 10000.0
    arch.tied_weights = False
    return arch


def _make_weights(arch):
    """Create minimal weight dict for a tiny model."""
    E = arch.n_embed
    H = arch.n_head
    KV = arch.n_kv_head
    D = arch.head_dim
    V = 128
    FF = 128
    L = arch.n_layers
    weights = {}
    # Embeddings
    weights["model.embed_tokens.weight"] = np.random.randn(V, E).astype(np.float32) * 0.02
    # Per-layer
    for i in range(L):
        weights[f"model.layers.{i}.self_attn.q_proj.weight"] = np.random.randn(H * D, E).astype(np.float32) * 0.02
        weights[f"model.layers.{i}.self_attn.k_proj.weight"] = np.random.randn(KV * D, E).astype(np.float32) * 0.02
        weights[f"model.layers.{i}.self_attn.v_proj.weight"] = np.random.randn(KV * D, E).astype(np.float32) * 0.02
        weights[f"model.layers.{i}.self_attn.o_proj.weight"] = np.random.randn(E, H * D).astype(np.float32) * 0.02
        weights[f"model.layers.{i}.mlp.gate_proj.weight"] = np.random.randn(FF, E).astype(np.float32) * 0.02
        weights[f"model.layers.{i}.mlp.up_proj.weight"] = np.random.randn(FF, E).astype(np.float32) * 0.02
        weights[f"model.layers.{i}.mlp.down_proj.weight"] = np.random.randn(E, FF).astype(np.float32) * 0.02
        weights[f"model.layers.{i}.input_layernorm.weight"] = np.ones(E, dtype=np.float32)
        weights[f"model.layers.{i}.post_attention_layernorm.weight"] = np.ones(E, dtype=np.float32)
    # Final norm and lm_head
    weights["model.norm.weight"] = np.ones(E, dtype=np.float32)
    weights["lm_head.weight"] = np.random.randn(V, E).astype(np.float32) * 0.02
    return weights


class TestNumpyBEConstruction:
    def test_from_weights(self):
        arch = _make_tiny_arch()
        weights = _make_weights(arch)
        be = NumpyBE.from_weights(weights, arch)
        assert be.backend_name() == "numpy"
        assert be.vocab_size() == 128
        assert be.n_layers() == 2

    def test_flat_lookup(self):
        arch = _make_tiny_arch()
        weights = _make_weights(arch)
        be = NumpyBE(weights, arch)
        assert "layers.0.q.weight" in be._flat
        assert "layers.1.q.weight" in be._flat


class TestNumpyBETensorPrimitives:
    def setup_method(self):
        arch = _make_tiny_arch()
        weights = _make_weights(arch)
        self.be = NumpyBE(weights, arch)

    def test_matmul(self):
        a = np.random.randn(2, 4).astype(np.float32)
        b = np.random.randn(4, 3).astype(np.float32)
        result = self.be.matmul(a, b)
        np.testing.assert_allclose(result, a @ b, atol=1e-6)

    def test_softmax(self):
        x = np.array([[1.0, 2.0, 3.0]])
        result = self.be.softmax(x)
        assert result.shape == x.shape
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)
        assert result[0, 2] > result[0, 0]

    def test_rmsnorm(self):
        x = np.array([[2.0, 4.0, 6.0]], dtype=np.float32)
        w = np.ones(3, dtype=np.float32)
        result = self.be.rmsnorm(x, w)
        assert result.shape == x.shape
        assert result.dtype == np.float32

    def test_silu(self):
        x = np.array([0.0, 1.0, -1.0])
        result = self.be.silu(x)
        assert result[0] == 0.0
        assert 0 < result[1] < 1
        assert result[2] < 0

    def test_gelu(self):
        x = np.array([0.0, 1.0, -1.0])
        result = self.be.gelu(x)
        assert abs(result[0]) < 1e-6
        assert 0 < result[1] < 1

    def test_rope(self):
        x = np.random.randn(1, 4, 4, 16).astype(np.float32)
        cos = np.ones((4, 1, 8), dtype=np.float32)
        sin = np.zeros((4, 1, 8), dtype=np.float32)
        result = self.be.rope(x, cos, sin)
        assert result.shape == x.shape

    def test_repeat_kv(self):
        x = np.random.randn(1, 2, 4, 16).astype(np.float32)
        result = self.be.repeat_kv(x, 2)
        assert result.shape == (1, 4, 4, 16)

    def test_repeat_kv_noop(self):
        x = np.random.randn(1, 4, 4, 16).astype(np.float32)
        result = self.be.repeat_kv(x, 1)
        assert result is x

    def test_argmax(self):
        x = np.array([1.0, 5.0, 3.0])
        assert self.be.argmax(x) == 1

    def test_clip(self):
        x = np.array([1.0, 5.0, 10.0])
        result = self.be.clip(x, 2.0, 8.0)
        np.testing.assert_array_equal(result, [2.0, 5.0, 8.0])

    def test_from_to_numpy(self):
        arr = np.array([1, 2, 3])
        assert self.be.from_numpy(arr) is arr
        assert self.be.to_numpy(arr) is arr


class TestNumpyBEForward:
    def test_forward_produces_logits(self):
        arch = _make_tiny_arch()
        weights = _make_weights(arch)
        be = NumpyBE.from_weights(weights, arch)
        tokens = np.array([[1, 2, 3]], dtype=np.int64)
        logits = be.forward(tokens)
        assert logits.shape == (1, 3, 128)
        assert logits.dtype == np.float32

    def test_forward_1d_input(self):
        arch = _make_tiny_arch()
        weights = _make_weights(arch)
        be = NumpyBE.from_weights(weights, arch)
        tokens = np.array([1, 2], dtype=np.int64)
        logits = be.forward(tokens)
        assert logits.ndim == 3


class TestNumpyBEWarmup:
    def test_warmup(self):
        arch = _make_tiny_arch()
        weights = _make_weights(arch)
        be = NumpyBE.from_weights(weights, arch)
        assert not be._warmed
        be.warmup()
        assert be._warmed

    def test_warmup_idempotent(self):
        arch = _make_tiny_arch()
        weights = _make_weights(arch)
        be = NumpyBE.from_weights(weights, arch)
        be.warmup()
        be.warmup()
        assert be._warmed
