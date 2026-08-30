"""Tests for VectorBE compute backend with shared memory multiprocessing."""

import numpy as np
import pytest

from domains.infrastructure.vector_backend import VectorBE
from domains.infrastructure.arch_config import ArchConfig


def _make_arch():
    arch = ArchConfig(
        name="tiny_vector",
        norm="rms_norm",
        positional="rope",
        activation="swiglu",
        attention="gqa",
    )
    arch.n_layers = 1
    arch.n_heads = 4
    arch.n_kv_heads = 2
    arch.n_embed = 32
    arch.head_dim = 8
    arch.vocab_size = 64
    arch.rope_base = 10000.0
    arch.tied_weights = False
    arch.n_head = 4
    arch.n_kv_head = 2
    return arch


def _make_weights(arch):
    E = arch.n_embed
    H = arch.n_head
    KV = arch.n_kv_head
    D = arch.head_dim
    V = arch.vocab_size
    FF = 48
    weights = {}
    weights["embed_tokens.weight"] = np.random.randn(V, E).astype(np.float32) * 0.02
    i = 0
    weights[f"model.layers.{i}.self_attn.q_proj.weight"] = np.random.randn(H * D, E).astype(np.float32) * 0.02
    weights[f"model.layers.{i}.self_attn.k_proj.weight"] = np.random.randn(KV * D, E).astype(np.float32) * 0.02
    weights[f"model.layers.{i}.self_attn.v_proj.weight"] = np.random.randn(KV * D, E).astype(np.float32) * 0.02
    weights[f"model.layers.{i}.self_attn.o_proj.weight"] = np.random.randn(E, H * D).astype(np.float32) * 0.02
    weights[f"model.layers.{i}.mlp.gate_proj.weight"] = np.random.randn(FF, E).astype(np.float32) * 0.02
    weights[f"model.layers.{i}.mlp.up_proj.weight"] = np.random.randn(FF, E).astype(np.float32) * 0.02
    weights[f"model.layers.{i}.mlp.down_proj.weight"] = np.random.randn(E, FF).astype(np.float32) * 0.02
    weights[f"model.layers.{i}.input_layernorm.weight"] = np.ones(E, dtype=np.float32)
    weights[f"model.layers.{i}.post_attention_layernorm.weight"] = np.ones(E, dtype=np.float32)
    weights["model.norm.weight"] = np.ones(E, dtype=np.float32)
    weights["lm_head.weight"] = np.random.randn(V, E).astype(np.float32) * 0.02
    return weights


@pytest.fixture
def vector_be():
    arch = _make_arch()
    weights = _make_weights(arch)
    be = VectorBE.from_weights(weights, arch)
    yield be
    be.__del__()


class TestVectorBEConstruction:
    def test_from_weights(self, vector_be):
        assert "vector" in vector_be.backend_name()
        assert vector_be.vocab_size() == 64
        assert vector_be.n_layers() == 1

    def test_arch_stored(self, vector_be):
        assert vector_be._arch.n_layers == 1
        assert vector_be._arch.n_head == 4


class TestVectorBESimpleOps:
    def test_silu(self, vector_be):
        x = np.array([0.0, 1.0, -1.0])
        result = vector_be.silu(x)
        assert result[0] == 0.0
        assert 0 < result[1] < 1

    def test_gelu(self, vector_be):
        x = np.array([0.0, 1.0, -1.0])
        result = vector_be.gelu(x)
        assert abs(result[0]) < 1e-6
        assert 0 < result[1] < 1

    def test_rope(self, vector_be):
        x = np.random.randn(1, 4, 4, 8).astype(np.float32)
        cos = np.ones((4, 1, 4), dtype=np.float32)
        sin = np.zeros((4, 1, 4), dtype=np.float32)
        result = vector_be.rope(x, cos, sin)
        assert result.shape == x.shape

    def test_repeat_kv(self, vector_be):
        x = np.random.randn(1, 2, 4, 8).astype(np.float32)
        result = vector_be.repeat_kv(x, 2)
        assert result.shape == (1, 4, 4, 8)

    def test_repeat_kv_noop(self, vector_be):
        x = np.random.randn(1, 4, 4, 8).astype(np.float32)
        result = vector_be.repeat_kv(x, 1)
        assert result is x

    def test_argmax(self, vector_be):
        x = np.array([1.0, 5.0, 3.0])
        assert vector_be.argmax(x) == 1

    def test_clip(self, vector_be):
        x = np.array([1.0, 5.0, 10.0])
        result = vector_be.clip(x, 2.0, 8.0)
        np.testing.assert_array_equal(result, [2.0, 5.0, 8.0])

    def test_from_to_numpy(self, vector_be):
        arr = np.array([1, 2, 3])
        assert vector_be.from_numpy(arr) is arr
        assert vector_be.to_numpy(arr) is arr


class TestVectorBESharedMemoryOps:
    def test_matmul_2d(self, vector_be):
        a = np.random.randn(4, 8).astype(np.float32)
        b = np.random.randn(8, 3).astype(np.float32)
        result = vector_be.matmul(a, b)
        np.testing.assert_allclose(result, a @ b, atol=1e-5)

    def test_matmul_fallback(self, vector_be):
        a = np.random.randn(4, 8, 3).astype(np.float32)
        b = np.random.randn(3, 2).astype(np.float32)
        result = vector_be.matmul(a, b)
        np.testing.assert_allclose(result, a @ b, atol=1e-5)

    def test_softmax_2d(self, vector_be):
        x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = vector_be.softmax(x)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_softmax_fallback(self, vector_be):
        x = np.random.randn(2, 3, 4).astype(np.float32)
        result = vector_be.softmax(x, axis=-1)
        np.testing.assert_allclose(result.sum(axis=-1), np.ones((2, 3)), atol=1e-6)

    def test_rmsnorm_2d(self, vector_be):
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        result = vector_be.rmsnorm(x, w)
        assert result.shape == x.shape
        assert result.dtype == np.float32

    def test_rmsnorm_1d_fallback(self, vector_be):
        x = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        w = np.ones(3, dtype=np.float32)
        result = vector_be.rmsnorm(x, w)
        assert result.shape == x.shape


class TestVectorBEForward:
    def test_forward_produces_logits(self, vector_be):
        tokens = np.array([[1, 2, 3]], dtype=np.int64)
        logits = vector_be.forward(tokens)
        assert logits.shape == (1, 3, 64)
        assert logits.dtype == np.float32

    def test_generate_yields_tokens(self, vector_be):
        tokens = np.array([[1]], dtype=np.int64)
        gen = list(vector_be.generate_stream(tokens, max_new_tokens=5, temperature=0.0))
        assert len(gen) <= 5
        assert all(isinstance(t, int) for t in gen)

    def test_generate_returns_metrics(self, vector_be):
        tokens = np.array([[1]], dtype=np.int64)
        all_ids, metrics = vector_be.generate(tokens, max_new_tokens=3, temperature=0.0)
        assert "n_tokens" in metrics
        assert "ttft_ms" in metrics
        assert "tokens_per_sec" in metrics
        assert all_ids.shape[1] >= 1

    def test_generate_eos_stops(self, vector_be):
        tokens = np.array([[1]], dtype=np.int64)
        all_ids, metrics = vector_be.generate(
            tokens, max_new_tokens=50, temperature=0.0, eos_token=1
        )
        assert metrics["n_tokens"] <= 50
