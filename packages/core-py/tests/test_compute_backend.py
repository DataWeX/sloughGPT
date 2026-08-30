"""Tests for ComputeBackend protocol and NumpyBE implementation."""

import numpy as np
import pytest

from domains.infrastructure.compute_backend import (
    ComputeBackend,
    create_backend,
    get_backend,
    register_backend,
    _BACKENDS,
)
from domains.infrastructure.arch_config import ArchConfig, build_arch, LLAMA_WEIGHT_MAP


# ── Fixtures ──────────────────────────────────────────────────────────────

def _make_qwen_arch():
    """Build ArchConfig matching Qwen2.5-0.5B (reduced to 2 layers for tests)."""
    return ArchConfig(
        name="qwen2.5-0.5b-test",
        norm="rms_norm",
        positional="rope",
        activation="swiglu",
        attention="gqa",
        weight_map=LLAMA_WEIGHT_MAP,
        n_head=14,
        n_kv_head=2,
        n_embed=896,
        n_layers=2,
        head_dim=64,
        rope_base=1000000.0,
    )


def _make_random_weights(arch: ArchConfig):
    """Create random weight dict matching the arch's weight map."""
    weights = {}
    W = arch.weight_map
    for canonical, mapped in W.items():
        if "{i}" in canonical:
            for layer in range(arch.n_layers):
                actual = mapped.replace("{i}", str(layer))
                if "weight" in canonical:
                    # Determine shape from name
                    if "q.weight" in canonical:
                        shape = (arch.n_head * arch.head_dim, arch.n_embed)
                    elif "k.weight" in canonical or "v.weight" in canonical:
                        shape = (arch.n_kv_head * arch.head_dim, arch.n_embed)
                    elif "o_proj.weight" in canonical:
                        shape = (arch.n_embed, arch.n_head * arch.head_dim)
                    elif "gate.weight" in canonical or "up.weight" in canonical:
                        shape = (arch.n_embed * 2, arch.n_embed)  # SwiGLU doubles
                    elif "down.weight" in canonical:
                        shape = (arch.n_embed, arch.n_embed * 2)
                    elif "attn_norm" in canonical or "ff_norm" in canonical:
                        shape = (arch.n_embed,)
                    else:
                        shape = (arch.n_embed, arch.n_embed)
                    weights[actual] = np.random.randn(*shape).astype(np.float32) * 0.02
                elif "bias" in canonical:
                    if "q.bias" in canonical:
                        dim = arch.n_head * arch.head_dim
                    elif "k.bias" in canonical or "v.bias" in canonical:
                        dim = arch.n_kv_head * arch.head_dim
                    elif "o_proj.bias" in canonical:
                        dim = arch.n_embed
                    else:
                        dim = arch.n_embed
                    weights[actual] = np.zeros(dim, dtype=np.float32)
        else:
            if "embed.token" in canonical:
                weights[mapped] = np.random.randn(151936, arch.n_embed).astype(np.float32) * 0.02
            elif "embed.pos" in canonical:
                weights[mapped] = np.zeros((32768, arch.n_embed), dtype=np.float32)
            elif "final_norm" in canonical:
                weights[mapped] = np.ones(arch.n_embed, dtype=np.float32)
            elif "lm_head" in canonical:
                weights[mapped] = np.random.randn(151936, arch.n_embed).astype(np.float32) * 0.02
    return weights


@pytest.fixture
def qwen_arch():
    return _make_qwen_arch()


@pytest.fixture
def qwen_weights(qwen_arch):
    return _make_random_weights(qwen_arch)


@pytest.fixture
def numpy_backend(qwen_weights, qwen_arch):
    from domains.infrastructure.numpy_backend import NumpyBE
    return NumpyBE.from_weights(qwen_weights, qwen_arch)


# ── Protocol compliance ──────────────────────────────────────────────────

class TestComputeBackendProtocol:
    """Verify the protocol is well-defined and NumpyBE conforms."""

    def test_numpy_backend_is_compute_backend(self, numpy_backend):
        assert isinstance(numpy_backend, ComputeBackend)

    def test_backend_name(self, numpy_backend):
        assert numpy_backend.backend_name() == "numpy"

    def test_n_layers(self, numpy_backend):
        assert numpy_backend.n_layers() == 2

    def test_vocab_size(self, numpy_backend):
        assert numpy_backend.vocab_size() == 151936


# ── Registry ─────────────────────────────────────────────────────────────

class TestRegistry:
    """Backend registration and lookup."""

    def test_numpy_registered(self):
        assert "numpy" in _BACKENDS

    def test_get_backend(self):
        cls = get_backend("numpy")
        assert cls is not None

    def test_unknown_backend_raises(self):
        with pytest.raises(KeyError, match="Unknown compute backend"):
            get_backend("nonexistent")

    def test_register_custom_backend(self):
        class FakeBE(ComputeBackend):
            def from_weights(cls, w, a): return cls()
            def warmup(self, s=1): pass
            def matmul(self, a, b): pass
            def softmax(self, x, axis=-1): pass
            def rmsnorm(self, x, w, eps=1e-6): pass
            def silu(self, x): pass
            def gelu(self, x): pass
            def rope(self, x, cos, sin): pass
            def repeat_kv(self, x, n): pass
            def argmax(self, x): pass
            def clip(self, x, lo, hi): pass
            def from_numpy(self, a): pass
            def to_numpy(self, t): pass
            def forward(self, token_ids, **kw): pass
            def generate_stream(self, token_ids, **kw): yield 0
            def generate(self, token_ids, **kw): pass
            def backend_name(self): return "fake"
            def vocab_size(self): return 0
            def n_layers(self): return 0

        register_backend("fake_test", FakeBE)
        assert "fake_test" in _BACKENDS
        del _BACKENDS["fake_test"]


# ── NumpyBE tensor primitives ────────────────────────────────────────────

class TestNumpyBEPrimitives:
    """Test individual tensor operations."""

    def test_matmul(self, numpy_backend):
        a = np.random.randn(2, 4).astype(np.float32)
        b = np.random.randn(4, 3).astype(np.float32)
        result = numpy_backend.matmul(a, b)
        np.testing.assert_allclose(result, a @ b)

    def test_softmax(self, numpy_backend):
        x = np.array([[1.0, 2.0, 3.0]])
        result = numpy_backend.softmax(x)
        np.testing.assert_allclose(result.sum(), 1.0, atol=1e-6)
        assert result.argmax() == 2

    def test_softmax_axis(self, numpy_backend):
        x = np.random.randn(3, 4).astype(np.float32)
        result = numpy_backend.softmax(x, axis=0)
        np.testing.assert_allclose(result.sum(axis=0), np.ones(4), atol=1e-6)

    def test_rmsnorm(self, numpy_backend):
        x = np.ones((2, 4), dtype=np.float32)
        w = np.ones(4, dtype=np.float32)
        result = numpy_backend.rmsnorm(x, w)
        # RMSNorm of all-ones: x/sqrt(mean(x^2)) * w = 1/sqrt(1) * 1 = 1
        np.testing.assert_allclose(result, np.ones((2, 4)), atol=1e-5)

    def test_layer_norm(self, numpy_backend):
        x = np.ones((2, 4), dtype=np.float32) * 5.0
        w = np.ones(4, dtype=np.float32)
        b = np.zeros(4, dtype=np.float32)
        result = numpy_backend.layer_norm(x, w, b)
        # LayerNorm of constant: (x - mean) / std * w + b = 0
        np.testing.assert_allclose(result, np.zeros((2, 4)), atol=1e-5)

    def test_silu(self, numpy_backend):
        x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        result = numpy_backend.silu(x)
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[1] > result[0]  # silu(1) > silu(0)

    def test_gelu(self, numpy_backend):
        x = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        result = numpy_backend.gelu(x)
        assert result[0] == pytest.approx(0.0, abs=1e-6)

    def test_argmax(self, numpy_backend):
        x = np.array([1.0, 3.0, 2.0])
        assert numpy_backend.argmax(x) == 1

    def test_clip(self, numpy_backend):
        x = np.array([-1.0, 0.5, 2.0])
        result = numpy_backend.clip(x, 0.0, 1.0)
        np.testing.assert_array_equal(result, [0.0, 0.5, 1.0])

    def test_repeat_kv(self, numpy_backend):
        # (batch=1, kv_heads=2, seq=3, dim=4)
        x = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
        result = numpy_backend.repeat_kv(x, n_reps=3)
        assert result.shape == (1, 6, 3, 4)  # 2*3=6 heads

    def test_repeat_kv_noop(self, numpy_backend):
        x = np.ones((1, 4, 3, 2), dtype=np.float32)
        result = numpy_backend.repeat_kv(x, n_reps=1)
        np.testing.assert_array_equal(result, x)

    def test_rope(self, numpy_backend):
        # (batch=1, seq=2, heads=3, dim=4)
        x = np.ones((1, 2, 3, 4), dtype=np.float32)
        cos = np.ones((2, 1, 2), dtype=np.float32)
        sin = np.zeros((2, 1, 2), dtype=np.float32)
        result = numpy_backend.rope(x, cos, sin)
        assert result.shape == (1, 2, 3, 4)

    def test_from_numpy_passthrough(self, numpy_backend):
        arr = np.array([1, 2, 3])
        assert numpy_backend.from_numpy(arr) is arr

    def test_to_numpy_passthrough(self, numpy_backend):
        arr = np.array([1, 2, 3])
        assert numpy_backend.to_numpy(arr) is arr


# ── NumpyBE forward pass ─────────────────────────────────────────────────

class TestNumpyBEForward:
    """Test full forward pass with random weights."""

    def test_forward_shape(self, numpy_backend):
        token_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        logits = numpy_backend.forward(token_ids)
        assert logits.shape == (1, 5, 151936)
        assert logits.dtype == np.float32

    def test_forward_single_token(self, numpy_backend):
        token_ids = np.array([[1]], dtype=np.int64)
        logits = numpy_backend.forward(token_ids)
        assert logits.shape == (1, 1, 151936)

    def test_forward_deterministic(self, numpy_backend):
        token_ids = np.array([[1, 2, 3]], dtype=np.int64)
        logits1 = numpy_backend.forward(token_ids)
        logits2 = numpy_backend.forward(token_ids)
        np.testing.assert_array_equal(logits1, logits2)


# ── NumpyBE generation ───────────────────────────────────────────────────

class TestNumpyBEGeneration:
    """Test generate_stream and generate."""

    def test_generate_stream_yields_tokens(self, numpy_backend):
        token_ids = np.array([[1, 2, 3]], dtype=np.int64)
        tokens = list(numpy_backend.generate_stream(token_ids, max_new_tokens=2))
        assert len(tokens) == 2
        assert all(isinstance(t, int) for t in tokens)

    def test_generate_stream_respects_max(self, numpy_backend):
        token_ids = np.array([[1]], dtype=np.int64)
        tokens = list(numpy_backend.generate_stream(token_ids, max_new_tokens=2))
        assert len(tokens) == 2

    def test_generate_returns_metrics(self, numpy_backend):
        token_ids = np.array([[1, 2]], dtype=np.int64)
        result_ids, metrics = numpy_backend.generate(token_ids, max_new_tokens=2)
        assert result_ids.shape == (1, 4)  # 2 prompt + 2 gen
        assert metrics["n_tokens"] == 2
        assert metrics["prompt_tokens"] == 2
        assert metrics["tokens_per_sec"] > 0

    def test_generate_greedy(self, numpy_backend):
        token_ids = np.array([[1]], dtype=np.int64)
        r1, _ = numpy_backend.generate(token_ids, max_new_tokens=2, temperature=0.0)
        r2, _ = numpy_backend.generate(token_ids, max_new_tokens=2, temperature=0.0)
        np.testing.assert_array_equal(r1, r2)


# ── Factory function ─────────────────────────────────────────────────────

class TestFactory:
    """Test create_backend_from_slnc."""

    def test_create_from_slnc(self):
        from domains.infrastructure.numpy_backend import create_backend_from_slnc
        slnc_path = "/home/mana/Documents/Default Project/sloughGPT/models/hf-cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/model.slnc"
        import os
        if not os.path.exists(slnc_path):
            pytest.skip("SLNC model not available")

        backend = create_backend_from_slnc(slnc_path, "numpy")
        assert backend.backend_name() == "numpy"
        assert backend.n_layers() == 24
        assert backend.vocab_size() == 151936

        # Verify it can do a forward pass
        token_ids = np.array([[1, 2, 3]], dtype=np.int64)
        logits = backend.forward(token_ids)
        assert logits.shape[2] == 151936
