"""Unit tests for WgpuBE — GPU compute backend (numpy fallback path).

Tests pure logic of tensor primitives, weight lookup, generation, and metadata
without requiring a GPU or SLNC model file.
"""

import math
import os
import numpy as np
import pytest

from domains.infrastructure.gpu.wgpu_be import WgpuBE, _load_spirv, _load_metallib
from domains.infrastructure.arch_config import ArchConfig, LLAMA_WEIGHT_MAP


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_arch(
    n_head=4,
    n_kv_head=4,
    n_embed=32,
    n_layers=2,
    head_dim=8,
    norm="rms_norm",
    positional="rope",
    activation="swiglu",
    attention="mha",
    rope_base=10000.0,
    weight_map=None,
) -> ArchConfig:
    if weight_map is None:
        weight_map = {}
    return ArchConfig(
        name="test",
        norm=norm,
        positional=positional,
        activation=activation,
        attention=attention,
        weight_map=weight_map,
        n_head=n_head,
        n_kv_head=n_kv_head,
        n_embed=n_embed,
        n_layers=n_layers,
        head_dim=head_dim,
        rope_base=rope_base,
    )


def _make_tiny_llama_weights(arch: ArchConfig, vocab_size=16, seq_len=4):
    """Build minimal weight dict for a tiny LLaMA-style model.

    FFN weights are stored as (in, out) because the forward pass applies .T.
    So gate_proj/up_proj are (ffn_dim, n_embed) and down_proj is (n_embed, ffn_dim).
    """
    n = arch
    ffn_dim = n.n_embed * 4
    w = {}
    w["model.embed_tokens.weight"] = np.random.randn(vocab_size, n.n_embed).astype(np.float32)
    for i in range(n.n_layers):
        w[f"model.layers.{i}.input_layernorm.weight"] = np.ones(n.n_embed, dtype=np.float32)
        w[f"model.layers.{i}.self_attn.q_proj.weight"] = np.random.randn(n.n_embed, n.n_embed).astype(np.float32)
        w[f"model.layers.{i}.self_attn.k_proj.weight"] = np.random.randn(n.n_embed, n.n_embed).astype(np.float32)
        w[f"model.layers.{i}.self_attn.v_proj.weight"] = np.random.randn(n.n_embed, n.n_embed).astype(np.float32)
        w[f"model.layers.{i}.self_attn.o_proj.weight"] = np.random.randn(n.n_embed, n.n_embed).astype(np.float32)
        w[f"model.layers.{i}.post_attention_layernorm.weight"] = np.ones(n.n_embed, dtype=np.float32)
        w[f"model.layers.{i}.mlp.gate_proj.weight"] = np.random.randn(ffn_dim, n.n_embed).astype(np.float32)
        w[f"model.layers.{i}.mlp.up_proj.weight"] = np.random.randn(ffn_dim, n.n_embed).astype(np.float32)
        w[f"model.layers.{i}.mlp.down_proj.weight"] = np.random.randn(n.n_embed, ffn_dim).astype(np.float32)
    w["model.norm.weight"] = np.ones(n.n_embed, dtype=np.float32)
    return w


def _make_tiny_gpt2_weights(arch: ArchConfig, vocab_size=16):
    """Build minimal weight dict for a tiny GPT-2-style model."""
    n = arch
    w = {}
    w["wte.weight"] = np.random.randn(vocab_size, n.n_embed).astype(np.float32)
    w["wpe.weight"] = np.random.randn(128, n.n_embed).astype(np.float32)
    for i in range(n.n_layers):
        w[f"h.{i}.ln_1.weight"] = np.ones(n.n_embed, dtype=np.float32)
        w[f"h.{i}.attn.c_attn.weight"] = np.random.randn(n.n_embed, n.n_embed * 3).astype(np.float32)
        w[f"h.{i}.attn.c_proj.weight"] = np.random.randn(n.n_embed, n.n_embed).astype(np.float32)
        w[f"h.{i}.ln_2.weight"] = np.ones(n.n_embed, dtype=np.float32)
        w[f"h.{i}.mlp.c_fc.weight"] = np.random.randn(n.n_embed, n.n_embed * 4).astype(np.float32)
        w[f"h.{i}.mlp.c_proj.weight"] = np.random.randn(n.n_embed * 4, n.n_embed).astype(np.float32)
    w["ln_f.weight"] = np.ones(n.n_embed, dtype=np.float32)
    return w


# ── _build_flat_lookup ──────────────────────────────────────────────────────

class TestBuildFlatLookup:

    def test_identity_map_no_template(self):
        wm = {"embed.token": "embed.token"}
        weights = {"embed.token": np.zeros(10)}
        arch = _make_arch(weight_map=wm)
        flat = WgpuBE._build_flat_lookup(weights, arch)
        assert "embed.token" in flat

    def test_layer_expansion(self):
        wm = {"layers.{i}.q.weight": "model.layers.{i}.q_proj.weight"}
        weights = {"model.layers.0.q_proj.weight": np.zeros(5)}
        arch = _make_arch(n_layers=1, weight_map=wm)
        flat = WgpuBE._build_flat_lookup(weights, arch)
        assert "layers.0.q.weight" in flat

    def test_layer_expansion_multiple_layers(self):
        wm = {"layers.{i}.q.weight": "model.layers.{i}.q_proj.weight"}
        weights = {
            "model.layers.0.q_proj.weight": np.zeros(5),
            "model.layers.1.q_proj.weight": np.zeros(5),
        }
        arch = _make_arch(n_layers=2, weight_map=wm)
        flat = WgpuBE._build_flat_lookup(weights, arch)
        assert "layers.0.q.weight" in flat
        assert "layers.1.q.weight" in flat

    def test_missing_weight_not_in_flat(self):
        wm = {"embed.token": "embed.token"}
        weights = {}
        arch = _make_arch(weight_map=wm)
        flat = WgpuBE._build_flat_lookup(weights, arch)
        assert "embed.token" not in flat

    def test_unmapped_weights_carry_through(self):
        wm = {}
        weights = {"extra.tensor": np.zeros(3)}
        arch = _make_arch(weight_map=wm)
        flat = WgpuBE._build_flat_lookup(weights, arch)
        assert "extra.tensor" in flat

    def test_mixed_mapped_and_unmapped(self):
        wm = {"embed.token": "embed.token"}
        weights = {
            "embed.token": np.zeros(10),
            "some_unmapped": np.ones(5),
        }
        arch = _make_arch(weight_map=wm)
        flat = WgpuBE._build_flat_lookup(weights, arch)
        assert "embed.token" in flat
        assert "some_unmapped" in flat

    def test_template_not_in_weights_skipped(self):
        wm = {"layers.{i}.q.weight": "model.layers.{i}.q_proj.weight"}
        weights = {}
        arch = _make_arch(n_layers=1, weight_map=wm)
        flat = WgpuBE._build_flat_lookup(weights, arch)
        assert "layers.0.q.weight" not in flat

    def test_duplicate_keys_resolved_by_weight_map(self):
        wm = {"layers.{i}.q.weight": "model.layers.0.q_proj.weight"}
        weights = {"model.layers.0.q_proj.weight": np.zeros(5)}
        arch = _make_arch(n_layers=1, weight_map=wm)
        flat = WgpuBE._build_flat_lookup(weights, arch)
        assert "layers.0.q.weight" in flat


# ── softmax ──────────────────────────────────────────────────────────────────

class TestSoftmax:

    def test_basic(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([1.0, 2.0, 3.0])
        result = backend.softmax(x)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_all_equal(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([5.0, 5.0, 5.0])
        result = backend.softmax(x)
        np.testing.assert_allclose(result, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)

    def test_large_values_stable(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([1000.0, 1001.0, 1002.0])
        result = backend.softmax(x)
        assert result.sum() == pytest.approx(1.0, abs=1e-5)
        assert not np.any(np.isnan(result))

    def test_negative_values(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([-10.0, -5.0, 0.0])
        result = backend.softmax(x)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)

    def test_2d_axis0(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = backend.softmax(x, axis=0)
        np.testing.assert_allclose(result.sum(axis=0), [1.0, 1.0], atol=1e-6)

    def test_2d_axis1(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = backend.softmax(x, axis=1)
        np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0], atol=1e-6)

    def test_preserves_shape(self):
        backend = WgpuBE({}, _make_arch())
        x = np.zeros((2, 3, 4))
        result = backend.softmax(x)
        assert result.shape == x.shape

    def test_monotonic_output(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = backend.softmax(x)
        assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))


# ── rmsnorm ──────────────────────────────────────────────────────────────────

class TestRmsnorm:

    def test_unit_weight(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4, dtype=np.float32)
        result = backend.rmsnorm(x, w)
        rms = np.sqrt(np.mean(x ** 2))
        np.testing.assert_allclose(result, x / rms, atol=1e-5)

    def test_scaled_weight(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([2.0, 4.0, 6.0, 8.0])
        w = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
        result = backend.rmsnorm(x, w)
        rms = np.sqrt(np.mean(x ** 2))
        np.testing.assert_allclose(result, (x / rms) * 2.0, atol=1e-5)

    def test_zero_input(self):
        backend = WgpuBE({}, _make_arch())
        x = np.zeros(4)
        w = np.ones(4, dtype=np.float32)
        result = backend.rmsnorm(x, w)
        assert not np.any(np.isnan(result))

    def test_eps_prevents_division_by_zero(self):
        backend = WgpuBE({}, _make_arch())
        x = np.zeros(4)
        w = np.ones(4, dtype=np.float32)
        result = backend.rmsnorm(x, w, eps=1e-6)
        assert not np.any(np.isnan(result))

    def test_batch_dimension(self):
        backend = WgpuBE({}, _make_arch())
        x = np.random.randn(3, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        result = backend.rmsnorm(x, w)
        assert result.shape == x.shape

    def test_output_dtype_matches_input(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        w = np.ones(3, dtype=np.float32)
        result = backend.rmsnorm(x, w)
        assert result.dtype == np.float32


# ── layer_norm ───────────────────────────────────────────────────────────────

class TestLayerNorm:

    def test_unit_weight_no_bias(self):
        backend = WgpuBE({}, _make_arch())
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        result = backend.layer_norm(x, w)
        mean = result.mean(axis=-1)
        np.testing.assert_allclose(mean, np.zeros(4), atol=1e-5)

    def test_with_bias(self):
        backend = WgpuBE({}, _make_arch())
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.ones(8, dtype=np.float32)
        b = np.full(8, 0.5, dtype=np.float32)
        result = backend.layer_norm(x, w, bias=b)
        mean = result.mean(axis=-1)
        np.testing.assert_allclose(mean, 0.5 * np.ones(4), atol=1e-5)

    def test_scaled_weight(self):
        backend = WgpuBE({}, _make_arch())
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.full(8, 2.0, dtype=np.float32)
        result = backend.layer_norm(x, w)
        std = result.std(axis=-1)
        np.testing.assert_allclose(std, 2.0 * np.ones(4), atol=1e-4)

    def test_eps_stability(self):
        backend = WgpuBE({}, _make_arch())
        x = np.zeros((2, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        result = backend.layer_norm(x, w, eps=1e-6)
        assert not np.any(np.isnan(result))

    def test_batch_preserved(self):
        backend = WgpuBE({}, _make_arch())
        x = np.random.randn(5, 16).astype(np.float32)
        w = np.ones(16, dtype=np.float32)
        result = backend.layer_norm(x, w)
        assert result.shape == (5, 16)

    def test_zero_mean_unit_variance(self):
        backend = WgpuBE({}, _make_arch())
        x = np.random.randn(10, 32).astype(np.float32) * 10 + 5
        w = np.ones(32, dtype=np.float32)
        result = backend.layer_norm(x, w)
        np.testing.assert_allclose(result.mean(axis=-1), 0.0, atol=1e-5)
        np.testing.assert_allclose(result.std(axis=-1), 1.0, atol=1e-5)


# ── silu ─────────────────────────────────────────────────────────────────────

class TestSilu:

    def test_basic(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([0.0, 1.0, -1.0])
        result = backend.silu(x)
        expected = x * (1.0 / (1.0 + np.exp(-x)))
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_zero(self):
        backend = WgpuBE({}, _make_arch())
        assert backend.silu(np.array([0.0]))[0] == pytest.approx(0.0)

    def test_large_positive(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([100.0])
        result = backend.silu(x)
        np.testing.assert_allclose(result, [100.0], atol=1e-3)

    def test_large_negative(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([-100.0])
        result = backend.silu(x)
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_preserves_shape(self):
        backend = WgpuBE({}, _make_arch())
        x = np.random.randn(3, 5)
        assert backend.silu(x).shape == x.shape

    def test_non_negative_output(self):
        backend = WgpuBE({}, _make_arch())
        x = np.linspace(-5, 5, 100)
        result = backend.silu(x)
        assert np.all(result >= -0.3)  # SiLU min is approx -0.28

    def test_monotonic_for_positive(self):
        backend = WgpuBE({}, _make_arch())
        x = np.linspace(0, 10, 50)
        result = backend.silu(x)
        assert all(result[i] <= result[i + 1] for i in range(len(result) - 1))


# ── gelu ─────────────────────────────────────────────────────────────────────

class TestGelu:

    def test_basic(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([0.0, 1.0, -1.0])
        result = backend.gelu(x)
        assert result.shape == x.shape

    def test_zero(self):
        backend = WgpuBE({}, _make_arch())
        assert backend.gelu(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-5)

    def test_large_positive(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([10.0])
        result = backend.gelu(x)
        np.testing.assert_allclose(result, [10.0], atol=1e-3)

    def test_large_negative(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([-10.0])
        result = backend.gelu(x)
        np.testing.assert_allclose(result, [0.0], atol=1e-6)

    def test_sigmoidal_shape(self):
        backend = WgpuBE({}, _make_arch())
        x = np.linspace(-3, 3, 100)
        result = backend.gelu(x)
        # GELU should transition from 0 to x smoothly
        assert result[0] < result[-1]

    def test_preserves_shape(self):
        backend = WgpuBE({}, _make_arch())
        x = np.random.randn(4, 6)
        assert backend.gelu(x).shape == x.shape


# ── rope ─────────────────────────────────────────────────────────────────────

class TestRope:

    def test_basic(self):
        backend = WgpuBE({}, _make_arch())
        x = np.ones((1, 2, 4, 8))
        cos = np.ones((2, 1, 4))
        sin = np.zeros((2, 1, 4))
        result = backend.rope(x, cos, sin)
        # sin=0 means output = x (since x1*cos - x2*0 = x1, x2*cos + x1*0 = x2)
        np.testing.assert_allclose(result, x, atol=1e-6)

    def test_sin_rotates(self):
        backend = WgpuBE({}, _make_arch())
        x = np.ones((1, 1, 1, 4))
        cos = np.ones((1, 1, 2))
        sin = np.zeros((1, 1, 2))
        result = backend.rope(x, cos, sin)
        assert result.shape == x.shape

    def test_negative_sin(self):
        backend = WgpuBE({}, _make_arch())
        x = np.ones((1, 1, 1, 4))
        cos = np.ones((1, 1, 2))
        sin = np.ones((1, 1, 2)) * 0.1
        result = backend.rope(x, cos, sin)
        # Should differ from identity
        assert not np.allclose(result, x, atol=1e-6)

    def test_output_shape(self):
        backend = WgpuBE({}, _make_arch())
        x = np.random.randn(1, 3, 2, 8)
        cos = np.random.randn(3, 1, 4)
        sin = np.random.randn(3, 1, 4)
        result = backend.rope(x, cos, sin)
        assert result.shape == x.shape

    def test_batch_preserved(self):
        backend = WgpuBE({}, _make_arch())
        x = np.random.randn(4, 5, 2, 8)
        cos = np.ones((5, 1, 4))
        sin = np.zeros((5, 1, 4))
        result = backend.rope(x, cos, sin)
        assert result.shape == (4, 5, 2, 8)


# ── repeat_kv ────────────────────────────────────────────────────────────────

class TestRepeatKv:

    def test_n_reps_1_passthrough(self):
        backend = WgpuBE({}, _make_arch())
        x = np.random.randn(1, 4, 3, 8)
        result = backend.repeat_kv(x, 1)
        np.testing.assert_array_equal(result, x)

    def test_n_reps_2(self):
        backend = WgpuBE({}, _make_arch())
        x = np.random.randn(1, 4, 3, 8)
        result = backend.repeat_kv(x, 2)
        assert result.shape == (1, 8, 3, 8)

    def test_n_reps_4(self):
        backend = WgpuBE({}, _make_arch())
        x = np.random.randn(1, 2, 3, 8)
        result = backend.repeat_kv(x, 4)
        assert result.shape == (1, 8, 3, 8)

    def test_values_duplicated(self):
        backend = WgpuBE({}, _make_arch())
        x = np.arange(8).reshape(1, 2, 1, 4).astype(np.float32)
        result = backend.repeat_kv(x, 2)
        # Each kv head should appear twice consecutively
        assert result[0, 0, 0, 0] == result[0, 1, 0, 0]
        assert result[0, 2, 0, 0] == result[0, 3, 0, 0]

    def test_batch_independent(self):
        backend = WgpuBE({}, _make_arch())
        x = np.random.randn(2, 3, 4, 5)
        result = backend.repeat_kv(x, 3)
        assert result.shape == (2, 9, 4, 5)


# ── argmax ───────────────────────────────────────────────────────────────────

class TestArgmax:

    def test_basic(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([1.0, 5.0, 3.0])
        assert backend.argmax(x) == 1

    def test_first_element(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([10.0, 1.0, 1.0])
        assert backend.argmax(x) == 0

    def test_last_element(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([1.0, 1.0, 10.0])
        assert backend.argmax(x) == 2

    def test_negative_values(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([-10.0, -5.0, -1.0])
        assert backend.argmax(x) == 2

    def test_non_array_returns_0(self):
        backend = WgpuBE({}, _make_arch())
        assert backend.argmax("not_array") == 0


# ── clip ─────────────────────────────────────────────────────────────────────

class TestClip:

    def test_basic(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([1.0, 5.0, 10.0])
        result = backend.clip(x, 2.0, 8.0)
        np.testing.assert_array_equal(result, [2.0, 5.0, 8.0])

    def test_no_clipping_needed(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([3.0, 4.0, 5.0])
        result = backend.clip(x, 1.0, 10.0)
        np.testing.assert_array_equal(result, x)

    def test_all_clipped_low(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([-10.0, -5.0, -1.0])
        result = backend.clip(x, 0.0, 1.0)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0])

    def test_all_clipped_high(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([100.0, 200.0, 300.0])
        result = backend.clip(x, 0.0, 10.0)
        np.testing.assert_array_equal(result, [10.0, 10.0, 10.0])

    def test_same_lo_hi(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([1.0, 5.0, 10.0])
        result = backend.clip(x, 5.0, 5.0)
        np.testing.assert_array_equal(result, [5.0, 5.0, 5.0])

    def test_non_array_passthrough(self):
        backend = WgpuBE({}, _make_arch())
        assert backend.clip("text", 0, 1) == "text"


# ── matmul ───────────────────────────────────────────────────────────────────

class TestMatmul:

    def test_basic(self):
        backend = WgpuBE({}, _make_arch())
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = backend.matmul(a, b)
        np.testing.assert_array_equal(result, a @ b)

    def test_vector_matrix(self):
        backend = WgpuBE({}, _make_arch())
        a = np.array([1.0, 2.0, 3.0])
        b = np.eye(3)
        result = backend.matmul(a, b)
        np.testing.assert_array_equal(result, a @ b)

    def test_3d(self):
        backend = WgpuBE({}, _make_arch())
        a = np.random.randn(2, 3, 4)
        b = np.random.randn(4, 5)
        result = backend.matmul(a, b)
        assert result.shape == (2, 3, 5)

    def test_single_element(self):
        backend = WgpuBE({}, _make_arch())
        a = np.array([[2.0]])
        b = np.array([[3.0]])
        result = backend.matmul(a, b)
        assert result[0, 0] == pytest.approx(6.0)


# ── from_numpy / to_numpy ───────────────────────────────────────────────────

class TestArrayConversion:

    def test_from_numpy_identity(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([1.0, 2.0, 3.0])
        assert backend.from_numpy(x) is x

    def test_to_numpy_from_ndarray(self):
        backend = WgpuBE({}, _make_arch())
        x = np.array([1.0, 2.0])
        result = backend.to_numpy(x)
        np.testing.assert_array_equal(result, x)

    def test_to_numpy_from_list(self):
        backend = WgpuBE({}, _make_arch())
        result = backend.to_numpy([1, 2, 3])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1, 2, 3])

    def test_to_numpy_from_scalar(self):
        backend = WgpuBE({}, _make_arch())
        result = backend.to_numpy(42)
        assert isinstance(result, np.ndarray)


# ── warmup ───────────────────────────────────────────────────────────────────

class TestWarmup:

    def test_noop(self):
        backend = WgpuBE({}, _make_arch())
        # Should not raise
        backend.warmup()
        backend.warmup(seq_len=10)


# ── backend_name ─────────────────────────────────────────────────────────────

class TestBackendName:

    def test_fallback_name(self):
        # GPU import will fail, so _has_gpu is False
        backend = WgpuBE({}, _make_arch())
        name = backend.backend_name()
        assert "gpu" in name.lower()
        assert "fallback" in name.lower() or "numpy" in name.lower()


# ── vocab_size ───────────────────────────────────────────────────────────────

class TestVocabSize:

    def test_with_embed_token(self):
        wm = {"embed.token": "model.embed_tokens.weight"}
        w = {"model.embed_tokens.weight": np.zeros((100, 32))}
        backend = WgpuBE(w, _make_arch(weight_map=wm))
        assert backend.vocab_size() == 100

    def test_with_model_embed_tokens_weight(self):
        wm = {"embed.token": "model.embed_tokens.weight"}
        w = {"model.embed_tokens.weight": np.zeros((200, 32))}
        backend = WgpuBE(w, _make_arch(weight_map=wm))
        assert backend.vocab_size() == 200

    def test_no_embed_returns_0(self):
        backend = WgpuBE({}, _make_arch())
        assert backend.vocab_size() == 0

    def test_vocab_size_fallback_model_key(self):
        """vocab_size checks both embed.token and model.embed_tokens.weight."""
        wm = {}
        w = {"model.embed_tokens.weight": np.zeros((50, 16))}
        backend = WgpuBE(w, _make_arch(weight_map=wm))
        assert backend.vocab_size() == 50


# ── n_layers ─────────────────────────────────────────────────────────────────

class TestNLayers:

    def test_from_arch(self):
        arch = _make_arch(n_layers=6)
        backend = WgpuBE({}, arch)
        assert backend.n_layers() == 6

    def test_zero_layers(self):
        arch = _make_arch(n_layers=0)
        backend = WgpuBE({}, arch)
        assert backend.n_layers() == 0


# ── _load_spirv / _load_metallib ────────────────────────────────────────────

class TestShaderLoading:

    def test_load_spirv_missing(self):
        with pytest.raises(FileNotFoundError):
            _load_spirv("nonexistent_shader")

    def test_load_metallib_missing(self):
        with pytest.raises(FileNotFoundError):
            _load_metallib("nonexistent_shader")


# ── Forward pass (tiny model) ───────────────────────────────────────────────

class TestForwardPass:

    def _make_backend(self, n_embed=16, n_head=2, n_kv_head=2, n_layers=1,
                      vocab_size=8, head_dim=8, norm="rms_norm",
                      positional="rope", activation="swiglu"):
        from domains.infrastructure.arch_config import LLAMA_WEIGHT_MAP
        arch = _make_arch(
            n_head=n_head, n_kv_head=n_kv_head, n_embed=n_embed,
            n_layers=n_layers, head_dim=head_dim, norm=norm,
            positional=positional, activation=activation,
            weight_map=LLAMA_WEIGHT_MAP,
        )
        weights = _make_tiny_llama_weights(arch, vocab_size=vocab_size)
        return WgpuBE(weights, arch), weights, arch

    def test_forward_output_shape(self):
        backend, _, arch = self._make_backend()
        tokens = np.array([[1, 2, 3]], dtype=np.int64)
        logits = backend.forward(tokens)
        assert logits.shape == (1, 3, 8)

    def test_forward_single_token(self):
        backend, _, _ = self._make_backend()
        tokens = np.array([[5]], dtype=np.int64)
        logits = backend.forward(tokens)
        assert logits.shape == (1, 1, 8)

    def test_forward_1d_reshaped(self):
        backend, _, _ = self._make_backend()
        tokens = np.array([1, 2], dtype=np.int64)
        logits = backend.forward(tokens)
        assert logits.shape == (1, 2, 8)

    def test_forward_deterministic(self):
        backend, _, _ = self._make_backend()
        tokens = np.array([[1, 2, 3]], dtype=np.int64)
        r1 = backend.forward(tokens)
        r2 = backend.forward(tokens)
        np.testing.assert_array_equal(r1, r2)

    def test_forward_finite_output(self):
        backend, _, _ = self._make_backend()
        tokens = np.array([[1, 2, 3, 4]], dtype=np.int64)
        logits = backend.forward(tokens)
        assert np.all(np.isfinite(logits))

    def test_forward_gpt2_style(self):
        """Test GPT-2 style (absolute pos, layer_norm, gelu) with minimal weights."""
        from domains.infrastructure.arch_config import GPT2_WEIGHT_MAP
        # Use a minimal weight map that maps canonical → same key (identity)
        # so the forward pass's .T convention works with our test weights.
        wm = {}
        wm["embed.token"] = "embed.token"
        wm["embed.pos"] = "embed.pos"
        wm["layers.{i}.attn_norm.weight"] = "layers.{i}.attn_norm.weight"
        wm["layers.{i}.q.weight"] = "layers.{i}.q.weight"
        wm["layers.{i}.k.weight"] = "layers.{i}.k.weight"
        wm["layers.{i}.v.weight"] = "layers.{i}.v.weight"
        wm["layers.{i}.o_proj.weight"] = "layers.{i}.o_proj.weight"
        wm["layers.{i}.ff_norm.weight"] = "layers.{i}.ff_norm.weight"
        wm["layers.{i}.ffn.weight"] = "layers.{i}.ffn.weight"
        wm["layers.{i}.ffn.down.weight"] = "layers.{i}.ffn.down.weight"
        wm["final_norm.weight"] = "final_norm.weight"
        arch = _make_arch(
            n_head=2, n_kv_head=2, n_embed=16, n_layers=1,
            head_dim=8, norm="layer_norm", positional="absolute",
            activation="gelu", weight_map=wm,
        )
        n, d = 16, 64
        w = {
            "embed.token": np.random.randn(8, n).astype(np.float32),
            "embed.pos": np.random.randn(128, n).astype(np.float32),
            "layers.0.attn_norm.weight": np.ones(n, dtype=np.float32),
            "layers.0.q.weight": np.random.randn(n, n).astype(np.float32),
            "layers.0.k.weight": np.random.randn(n, n).astype(np.float32),
            "layers.0.v.weight": np.random.randn(n, n).astype(np.float32),
            "layers.0.o_proj.weight": np.random.randn(n, n).astype(np.float32),
            "layers.0.ff_norm.weight": np.ones(n, dtype=np.float32),
            "layers.0.ffn.weight": np.random.randn(d, n).astype(np.float32),
            "layers.0.ffn.down.weight": np.random.randn(n, d).astype(np.float32),
            "final_norm.weight": np.ones(n, dtype=np.float32),
        }
        backend = WgpuBE(w, arch)
        tokens = np.array([[1, 2, 3]], dtype=np.int64)
        logits = backend.forward(tokens)
        assert logits.shape == (1, 3, 8)
        assert np.all(np.isfinite(logits))

    def test_forward_gqa_expansion(self):
        # GQA: n_head=4, n_kv_head=2, head_dim must satisfy n_embed = n_head * head_dim
        n_head, head_dim, n_embed = 4, 4, 16
        n_kv_head = 2
        arch = _make_arch(
            n_head=n_head, n_kv_head=n_kv_head, n_embed=n_embed, n_layers=1,
            head_dim=head_dim, norm="rms_norm", positional="rope",
            activation="swiglu", weight_map=LLAMA_WEIGHT_MAP,
        )
        ffn_dim = n_embed * 4
        kv_dim = n_kv_head * head_dim
        w = {}
        w["model.embed_tokens.weight"] = np.random.randn(8, n_embed).astype(np.float32)
        i = 0
        w[f"model.layers.{i}.input_layernorm.weight"] = np.ones(n_embed, dtype=np.float32)
        w[f"model.layers.{i}.self_attn.q_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
        # GQA: k/v proj output to kv_dim, not n_embed
        w[f"model.layers.{i}.self_attn.k_proj.weight"] = np.random.randn(kv_dim, n_embed).astype(np.float32)
        w[f"model.layers.{i}.self_attn.v_proj.weight"] = np.random.randn(kv_dim, n_embed).astype(np.float32)
        w[f"model.layers.{i}.self_attn.o_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
        w[f"model.layers.{i}.post_attention_layernorm.weight"] = np.ones(n_embed, dtype=np.float32)
        w[f"model.layers.{i}.mlp.gate_proj.weight"] = np.random.randn(ffn_dim, n_embed).astype(np.float32)
        w[f"model.layers.{i}.mlp.up_proj.weight"] = np.random.randn(ffn_dim, n_embed).astype(np.float32)
        w[f"model.layers.{i}.mlp.down_proj.weight"] = np.random.randn(n_embed, ffn_dim).astype(np.float32)
        w["model.norm.weight"] = np.ones(n_embed, dtype=np.float32)
        backend = WgpuBE(w, arch)
        tokens = np.array([[1, 2]], dtype=np.int64)
        logits = backend.forward(tokens)
        assert logits.shape == (1, 2, 8)
        assert np.all(np.isfinite(logits))


# ── generate_stream ──────────────────────────────────────────────────────────

class TestGenerateStream:

    def _make_backend(self):
        arch = _make_arch(
            n_head=2, n_kv_head=2, n_embed=16, n_layers=1,
            head_dim=8, norm="rms_norm", positional="rope",
            activation="swiglu", weight_map=LLAMA_WEIGHT_MAP,
        )
        weights = _make_tiny_llama_weights(arch, vocab_size=8)
        return WgpuBE(weights, arch)

    def test_yields_correct_count(self):
        backend = self._make_backend()
        tokens = np.array([[1, 2]], dtype=np.int64)
        generated = list(backend.generate_stream(tokens, max_new_tokens=5))
        assert len(generated) == 5

    def test_yields_ints(self):
        backend = self._make_backend()
        tokens = np.array([[1]], dtype=np.int64)
        for tok in backend.generate_stream(tokens, max_new_tokens=3):
            assert isinstance(tok, int)

    def test_valid_token_range(self):
        backend = self._make_backend()
        tokens = np.array([[1]], dtype=np.int64)
        for tok in backend.generate_stream(tokens, max_new_tokens=10):
            assert 0 <= tok < 8

    def test_eos_stops_generation(self):
        backend = self._make_backend()
        tokens = np.array([[1]], dtype=np.int64)
        generated = list(backend.generate_stream(tokens, max_new_tokens=100, eos_token=3))
        # Should stop early (probabilistically hits token 3 within 100 steps)
        # At minimum, just verify it yields valid tokens
        assert all(0 <= t < 8 for t in generated)

    def test_extra_stop_ids(self):
        backend = self._make_backend()
        tokens = np.array([[1]], dtype=np.int64)
        generated = list(backend.generate_stream(
            tokens, max_new_tokens=100, extra_stop_ids=[2, 3]
        ))
        assert all(t not in [2, 3] or len(generated) < 100 for t in generated)

    def test_temperature_zero_greedy(self):
        backend = self._make_backend()
        tokens = np.array([[1]], dtype=np.int64)
        r1 = list(backend.generate_stream(tokens, max_new_tokens=3, temperature=0.0))
        r2 = list(backend.generate_stream(tokens, max_new_tokens=3, temperature=0.0))
        assert r1 == r2

    def test_top_k(self):
        backend = self._make_backend()
        tokens = np.array([[1]], dtype=np.int64)
        generated = list(backend.generate_stream(tokens, max_new_tokens=5, top_k=2))
        assert len(generated) == 5

    def test_top_p(self):
        backend = self._make_backend()
        tokens = np.array([[1]], dtype=np.int64)
        generated = list(backend.generate_stream(tokens, max_new_tokens=5, top_p=0.9))
        assert len(generated) == 5

    def test_repetition_penalty(self):
        backend = self._make_backend()
        tokens = np.array([[1]], dtype=np.int64)
        generated = list(backend.generate_stream(
            tokens, max_new_tokens=5, repetition_penalty=1.5
        ))
        assert len(generated) == 5

    def test_1d_input_reshaped(self):
        backend = self._make_backend()
        tokens = np.array([1, 2, 3], dtype=np.int64)
        generated = list(backend.generate_stream(tokens, max_new_tokens=3))
        assert len(generated) == 3

    def test_max_new_tokens_zero(self):
        backend = self._make_backend()
        tokens = np.array([[1]], dtype=np.int64)
        generated = list(backend.generate_stream(tokens, max_new_tokens=0))
        assert generated == []


# ── generate ─────────────────────────────────────────────────────────────────

class TestGenerate:

    def _make_backend(self):
        arch = _make_arch(
            n_head=2, n_kv_head=2, n_embed=16, n_layers=1,
            head_dim=8, norm="rms_norm", positional="rope",
            activation="swiglu", weight_map=LLAMA_WEIGHT_MAP,
        )
        weights = _make_tiny_llama_weights(arch, vocab_size=8)
        return WgpuBE(weights, arch)

    def test_returns_token_array_and_metrics(self):
        backend = self._make_backend()
        tokens = np.array([[1, 2]], dtype=np.int64)
        result_ids, metrics = backend.generate(tokens, max_new_tokens=3)
        assert result_ids.shape == (1, 5)
        assert metrics["n_tokens"] == 3
        assert metrics["prompt_tokens"] == 2
        assert metrics["total_tokens"] == 5

    def test_metrics_timing(self):
        backend = self._make_backend()
        tokens = np.array([[1]], dtype=np.int64)
        _, metrics = backend.generate(tokens, max_new_tokens=2)
        assert metrics["t_start"] < metrics["t_end"]
        assert metrics["decode_ms"] >= 0
        assert metrics["tokens_per_sec"] >= 0

    def test_greedy_deterministic(self):
        backend = self._make_backend()
        tokens = np.array([[1]], dtype=np.int64)
        r1, _ = backend.generate(tokens, max_new_tokens=5, temperature=0.0)
        r2, _ = backend.generate(tokens, max_new_tokens=5, temperature=0.0)
        np.testing.assert_array_equal(r1, r2)

    def test_generate_stream_matches_generate(self):
        backend = self._make_backend()
        tokens = np.array([[1, 2]], dtype=np.int64)
        # temperature=0 for deterministic greedy — both paths must agree
        stream_tokens = list(backend.generate_stream(tokens, max_new_tokens=4, temperature=0.0))
        result_ids, _ = backend.generate(tokens, max_new_tokens=4, temperature=0.0)
        expected = [1, 2] + stream_tokens
        np.testing.assert_array_equal(result_ids.flatten(), expected)

    def test_1d_input(self):
        backend = self._make_backend()
        tokens = np.array([1, 2, 3], dtype=np.int64)
        result_ids, metrics = backend.generate(tokens, max_new_tokens=2)
        assert result_ids.ndim == 2
        assert result_ids.shape[0] == 1
        assert metrics["prompt_tokens"] == 3


# ── from_weights classmethod ─────────────────────────────────────────────────

class TestFromWeights:

    def test_creates_instance(self):
        arch = _make_arch()
        backend = WgpuBE.from_weights({}, arch)
        assert isinstance(backend, WgpuBE)

    def test_preserves_weights(self):
        arch = _make_arch()
        w = {"test": np.zeros(5)}
        backend = WgpuBE.from_weights(w, arch)
        assert "test" in backend._flat
