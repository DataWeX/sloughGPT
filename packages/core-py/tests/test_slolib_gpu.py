"""Tests for the SloLib GPU accelerator layer (domains/slolib/gpu/__init__.py).

In this environment torch/cupy/pyopencl are all unavailable, so the CPU backend
(_CPUBackend) plus the base-class numpy implementations are the production path.
Every reachable numpy op is verified against an independent reference, the
backend-detection priority logic is exercised by monkeypatching the availability
probes, and the GPU backend classes are exercised both in their numpy-fallback /
is_available=False state and with numpy-backed proxies installed in
``sys.modules`` for cupy (CUDA) and pyopencl (OpenCL) so their dispatch arms
execute even without the real libraries. The Metal backend is torch-free — it
detects MPS via a platform check and computes in numpy.
"""

import importlib.util
import os
import sys
import time

import numpy as np
import pytest

from domains.slolib import gpu as slib


_TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


# =============================================================================
# Independent numpy references
# =============================================================================

def _ref_softmax(a, axis=-1):
    e = np.exp(a - np.max(a, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _ref_log_softmax(a, axis=-1):
    m = np.max(a, axis=axis, keepdims=True)
    return a - m - np.log(np.exp(a - m).sum(axis=axis, keepdims=True))


def _ref_layernorm(a, w, b, eps=1e-5):
    mean = a.mean(axis=-1, keepdims=True)
    var = a.var(axis=-1, keepdims=True)
    return ((a - mean) / np.sqrt(var + eps)) * w + b


def _ref_rmsnorm(a, w, eps=1e-5):
    return (a / np.sqrt(np.mean(a * a, axis=-1, keepdims=True) + eps)) * w


def _ref_gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def _ref_scaled_dot_attention(q, k, v, mask=None, scale=None, causal=False):
    if scale is None:
        scale = 1.0 / np.sqrt(k.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    if causal:
        n, s = q.shape[-2], k.shape[-2]
        scores = scores + np.triu(np.full((n, s), -1e9, dtype=np.float32), k=1)[None, None]
    return np.matmul(_ref_softmax(scores, axis=-1), v)


def _ref_conv2d(x, weight, bias=None, stride=1, padding=0):
    n, c, h, w = x.shape
    oc, ic, kh, kw = weight.shape
    if padding > 0:
        x = np.pad(x, [(0, 0), (0, 0), (padding, padding), (padding, padding)], mode="constant")
    oh = (x.shape[2] - kh) // stride + 1
    ow = (x.shape[3] - kw) // stride + 1
    out = np.zeros((n, oc, oh, ow), dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            patch = x[:, :, i:i + (oh - 1) * stride + 1, j:j + (ow - 1) * stride + 1]
            if stride != 1:
                patch = patch[:, :, ::stride, ::stride]
            out += np.tensordot(patch, weight[:, :, i, j], axes=([1], [1])).transpose(0, 3, 1, 2)
    if bias is not None:
        out = out + bias[None, :, None, None]
    return out


def _ref_max_pool(x, k, stride):
    n, c, h, w = x.shape
    oh = (h - k) // stride + 1
    ow = (w - k) // stride + 1
    out = np.zeros((n, c, oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[:, :, i, j] = x[:, :, i * stride:i * stride + k, j * stride:j * stride + k].max(axis=(2, 3))
    return out


def _ref_avg_pool(x, k, stride):
    n, c, h, w = x.shape
    oh = (h - k) // stride + 1
    ow = (w - k) // stride + 1
    out = np.zeros((n, c, oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[:, :, i, j] = x[:, :, i * stride:i * stride + k, j * stride:j * stride + k].mean(axis=(2, 3))
    return out


def _ref_cross_entropy(logits, targets):
    flat = logits.reshape(-1, logits.shape[-1])
    m = flat.max(axis=-1, keepdims=True)
    lp = flat - m - np.log(np.exp(flat - m).sum(axis=-1, keepdims=True))
    t = targets.astype(np.int64).flatten()
    valid = (t >= 0) & (t < flat.shape[1])
    idx = np.arange(len(t))[valid]
    return float(-lp[idx, t[valid]].mean()) if idx.size else 0.0


def _ref_batchnorm2d(x, gamma, beta, rmean, rvar, eps=1e-5, momentum=0.1, training=True):
    if training:
        mean = x.mean(axis=(0, 2, 3), keepdims=True)
        var = x.var(axis=(0, 2, 3), keepdims=True)
        rmean[...] = momentum * mean.squeeze() + (1 - momentum) * rmean
        rvar[...] = momentum * var.squeeze() + (1 - momentum) * rvar
    else:
        mean = rmean.reshape(1, -1, 1, 1)
        var = rvar.reshape(1, -1, 1, 1)
    return ((x - mean) / np.sqrt(var + eps)) * gamma[:, None, None] + beta[:, None, None]


def _ref_batchnorm1d(x, gamma, beta, rmean, rvar, eps=1e-5, momentum=0.1, training=True):
    if training:
        mean = x.mean(axis=0, keepdims=True)
        var = x.var(axis=0, keepdims=True)
        rmean[...] = momentum * mean.squeeze() + (1 - momentum) * rmean
        rvar[...] = momentum * var.squeeze() + (1 - momentum) * rvar
    else:
        mean = rmean.reshape(1, -1)
        var = rvar.reshape(1, -1)
    return ((x - mean) / np.sqrt(var + eps)) * gamma + beta


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def _reset_backend():
    slib.reset_accelerator()
    yield
    slib.reset_accelerator()


@pytest.fixture
def cpu():
    return slib._CPUBackend()


# =============================================================================
# _BufferPool
# =============================================================================

class TestBufferPool:
    def test_get_miss_creates_array(self):
        pool = slib._BufferPool()
        arr = pool.get((4, 8), np.float32)
        assert arr.shape == (4, 8)
        assert arr.dtype == np.float32
        assert pool.stats() == {"hits": 0, "misses": 1}

    def test_put_then_get_reuses_array(self):
        pool = slib._BufferPool()
        first = pool.get((2, 3))
        pool.put(first)
        second = pool.get((2, 3))
        assert second is first
        assert pool.stats() == {"hits": 1, "misses": 1}

    def test_different_shape_is_a_miss(self):
        pool = slib._BufferPool()
        pool.put(pool.get((1, 1)))
        pool.get((2, 2))
        assert pool.stats() == {"hits": 0, "misses": 2}

    def test_pool_size_cap(self):
        pool = slib._BufferPool(max_pool_size=3)
        for _ in range(6):
            pool.put(np.empty((5, 5), dtype=np.float32))
        # Only the last 3 fit; the first 3 are dropped
        assert sum(len(b) for b in pool._pool.values()) == 3

    def test_clear_resets_stats(self):
        pool = slib._BufferPool()
        pool.get((1, 1))
        pool.clear()
        assert pool._pool == {}
        assert pool.stats() == {"hits": 0, "misses": 0}


# =============================================================================
# Accelerator selection / detection
# =============================================================================

class TestBackendSelection:
    def test_default_detection_returns_cpu(self):
        acc = slib.get_accelerator()
        assert isinstance(acc, slib._CPUBackend)

    def test_get_accelerator_is_cached(self):
        a = slib.get_accelerator()
        b = slib.get_accelerator()
        assert a is b

    def test_reset_accelerator(self):
        a = slib.get_accelerator()
        slib.reset_accelerator()
        b = slib.get_accelerator()
        assert a is not b

    def test_cuda_has_highest_priority(self, monkeypatch):
        monkeypatch.setattr(slib._CUDABackend, "is_available", lambda self: True)
        monkeypatch.setattr(slib._CUDABackend, "vram_gb", lambda self: 16.0)
        acc = slib._detect_best_backend()
        assert isinstance(acc, slib._CUDABackend)

    def test_metal_selected_on_darwin(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(slib._MetalBackend, "is_available", lambda self: True)
        monkeypatch.setattr(slib._MetalBackend, "vram_gb", lambda self: 32.0)
        acc = slib._detect_best_backend()
        assert isinstance(acc, slib._MetalBackend)

    def test_opencl_selected_when_available(self, monkeypatch):
        monkeypatch.setattr(slib._OpenCLBackend, "is_available", lambda self: True)
        monkeypatch.setattr(slib._OpenCLBackend, "vram_gb", lambda self: 6.0)
        acc = slib._detect_best_backend()
        assert isinstance(acc, slib._OpenCLBackend)

    def test_metal_exception_is_swallowed(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(slib._MetalBackend, "is_available", lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        acc = slib._detect_best_backend()
        assert isinstance(acc, slib._CPUBackend)

    def test_cuda_exception_is_swallowed(self, monkeypatch):
        monkeypatch.setattr(slib._CUDABackend, "is_available", lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        acc = slib._detect_best_backend()
        assert isinstance(acc, slib._CPUBackend)

    def test_opencl_exception_is_swallowed(self, monkeypatch):
        monkeypatch.setattr(slib._OpenCLBackend, "is_available", lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        acc = slib._detect_best_backend()
        assert isinstance(acc, slib._CPUBackend)

    def test_cuda_beats_metal_on_priority(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setattr(slib._MetalBackend, "is_available", lambda self: True)
        monkeypatch.setattr(slib._MetalBackend, "vram_gb", lambda self: 64.0)
        monkeypatch.setattr(slib._CUDABackend, "is_available", lambda self: True)
        monkeypatch.setattr(slib._CUDABackend, "vram_gb", lambda self: 1.0)
        acc = slib._detect_best_backend()
        assert isinstance(acc, slib._CUDABackend)

    def test_openblas_priority_beats_plain_simd(self, monkeypatch):
        monkeypatch.setattr(slib._CPUBackend, "has_openblas", lambda self: True)
        monkeypatch.setattr(slib._CPUBackend, "openblas_threads", lambda self: 8)
        monkeypatch.setattr(slib._CPUBackend, "has_simd", lambda self: True)
        acc = slib._detect_best_backend()
        assert isinstance(acc, slib._CPUBackend)

    def test_set_accelerator_precision_fp16_returns_fp32_on_cpu(self):
        assert slib.set_accelerator_precision("fp16") == "fp32"

    def test_set_accelerator_precision_auto_returns_fp32(self):
        assert slib.set_accelerator_precision("auto") == "fp32"


# =============================================================================
# Precision control on the base class
# =============================================================================

class TestPrecision:
    def test_fp32_mode(self, cpu):
        assert cpu.set_precision("fp32") == "fp32"
        assert cpu._fp16_mode is False

    def test_unknown_mode_defaults_fp32(self, cpu):
        assert cpu.set_precision("bogus") == "fp32"

    def test_fp16_requested_but_unavailable(self, cpu):
        cpu._fp16_available = False
        assert cpu.set_precision("fp16") == "fp32"

    def test_auto_benchmark_runs_and_returns_fp32(self, cpu, monkeypatch):
        cpu._fp16_available = True

        def fake(a, b):
            if cpu._fp16_mode:
                time.sleep(0.02)
            return a

        monkeypatch.setattr(cpu, "matmul", fake)
        assert cpu.set_precision("auto") == "fp32"
        assert cpu._fp16_mode is False

    def test_auto_benchmark_picks_fp16_when_faster(self, cpu, monkeypatch):
        cpu._fp16_available = True

        def fake(a, b):
            if not cpu._fp16_mode:
                time.sleep(0.02)
            return a

        monkeypatch.setattr(cpu, "matmul", fake)
        assert cpu._prec_benchmark() == "fp16"
        assert cpu._fp16_mode is True

    def test_prec_benchmark_exception_falls_back_fp32(self, cpu, monkeypatch):
        cpu._fp16_available = True

        def boom(a, b):
            raise RuntimeError("matmul failed")

        monkeypatch.setattr(cpu, "matmul", boom)
        assert cpu._prec_benchmark() == "fp32"
        assert cpu._fp16_mode is False

    def test_prec_benchmark_skips_when_unavailable(self, cpu):
        cpu._fp16_available = False
        assert cpu._prec_benchmark() == "fp32"

    def test_fp16_available_mode_roundtrip(self, cpu):
        cpu._fp16_available = True
        assert cpu.set_precision("fp16") == "fp16"
        assert cpu._fp16_mode is True
        assert cpu.set_precision("fp32") == "fp32"
        assert cpu._fp16_mode is False


class TestAcceleratorBase:
    def test_base_softmax(self):
        base = slib._Accelerator()
        a = np.array([1.0, 2.0, 3.0])
        assert np.allclose(base.softmax(a), _ref_softmax(a))

    def test_base_log_softmax(self):
        base = slib._Accelerator()
        a = np.array([1.0, 2.0, 3.0])
        assert np.allclose(base.log_softmax(a), _ref_log_softmax(a))

    def test_base_vram_gb(self):
        assert slib._Accelerator().vram_gb() == 0.0

    def test_base_memory_hint(self):
        hint = slib._Accelerator().memory_hint()
        assert hint == {"tier": "lite"}

    def test_base_defaults(self):
        base = slib._Accelerator()
        assert base.name == "base"
        assert base.device_type == "cpu"
        assert base.is_available() is True
        assert base._fp16_mode is False
        assert base._fp16_available is False
        assert base.set_precision("fp16") == "fp32"


# =============================================================================
# Device info / transfer
# =============================================================================

class TestDeviceBasics:
    def test_is_available_true(self, cpu):
        assert cpu.is_available() is True

    def test_vram_gb_default(self, cpu):
        assert cpu.vram_gb() >= 0

    def test_memory_hint_base(self, cpu):
        hint = cpu.memory_hint()
        assert hint["tier"] == cpu.compute_tier

    def test_sync_is_noop(self, cpu):
        cpu.sync()
        assert True

    def test_to_device_float32(self, cpu):
        arr = np.arange(6, dtype=np.float64).reshape(2, 3)
        out = cpu.to_device(arr)
        assert out.dtype == np.float32
        assert np.array_equal(out, arr)

    def test_from_device_ndarray(self, cpu):
        arr = np.arange(3, dtype=np.float32)
        out = cpu.from_device(arr)
        assert np.array_equal(out, arr)

    def test_vram_gb_uses_psutil(self, cpu, monkeypatch):
        class _VM:
            total = 32 * 1024 ** 3

        class _FakePsutil:
            @staticmethod
            def virtual_memory():
                return _VM()

        monkeypatch.setitem(sys.modules, "psutil", _FakePsutil())
        assert cpu.vram_gb() == 16.0

    def test_vram_gb_psutil_installed_but_unavailable(self, cpu, monkeypatch):
        class _Boom:
            @staticmethod
            def virtual_memory():
                raise RuntimeError("no memory info")

        monkeypatch.setitem(sys.modules, "psutil", _Boom())
        assert cpu.vram_gb() == 8.0


# =============================================================================
# Elementary ops
# =============================================================================

class TestElementaryOps:
    def test_arith(self, cpu):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[0.5, 1.0], [2.0, 2.0]])
        assert np.array_equal(cpu.add(a, b), a + b)
        assert np.array_equal(cpu.sub(a, b), a - b)
        assert np.array_equal(cpu.mul(a, b), a * b)
        assert np.array_equal(cpu.div(a, b), a / b)

    def test_matmul(self, cpu):
        a = np.random.randn(3, 4)
        b = np.random.randn(4, 5)
        assert np.allclose(cpu.matmul(a, b), a @ b)

    def test_pow_sqrt_exp_log(self, cpu):
        a = np.array([1.0, 4.0, 9.0])
        assert np.allclose(cpu.pow(a, 2), a ** 2)
        assert np.allclose(cpu.sqrt(a), np.sqrt(a))
        assert np.allclose(cpu.exp(a), np.exp(a))
        assert np.allclose(cpu.log(a), np.log(a))

    def test_reductions(self, cpu):
        a = np.arange(12, dtype=np.float32).reshape(3, 4)
        assert np.allclose(cpu.sum(a), a.sum())
        assert np.allclose(cpu.sum(a, axis=1), a.sum(axis=1))
        assert np.allclose(cpu.mean(a), a.mean())
        assert np.allclose(cpu.mean(a, axis=0), a.mean(axis=0))
        assert np.allclose(cpu.max(a), a.max())
        assert np.allclose(cpu.max(a, axis=1), a.max(axis=1))
        assert np.allclose(cpu.min(a), a.min())
        assert np.allclose(cpu.min(a, axis=0), a.min(axis=0))

    def test_abs_neg_clamp(self, cpu):
        a = np.array([-3.0, 1.0, 5.0])
        assert np.array_equal(cpu.abs(a), np.abs(a))
        assert np.array_equal(cpu.neg(a), -a)
        assert np.array_equal(cpu.clamp(a, -1.0, 2.0), np.clip(a, -1.0, 2.0))

    def test_where(self, cpu):
        cond = np.array([True, False, True])
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert np.array_equal(cpu.where(cond, a, b), np.where(cond, a, b))

    def test_gather_axis0(self, cpu):
        a = np.arange(9, dtype=np.float32).reshape(3, 3)
        idx = np.array([[2], [0], [1]])
        assert np.array_equal(cpu.gather(a, 0, idx), a[[2, 0, 1]])

    def test_gather_axis1(self, cpu):
        a = np.arange(9, dtype=np.float32).reshape(3, 3)
        idx = np.array([[2, 1], [0, 2], [1, 0]])
        assert np.array_equal(cpu.gather(a, 1, idx), np.take_along_axis(a, idx, axis=1))

    def test_scatter(self, cpu):
        a = np.zeros((3, 3), dtype=np.float32)
        idx = np.array([[2, 0], [1, 2], [0, 1]])
        src = np.ones((3, 2), dtype=np.float32) * 7
        out = cpu.scatter(a, 1, idx, src)
        assert out[0, 2] == 7 and out[0, 0] == 7 and out[2, 1] == 7
        assert out[1, 1] == 7 and out[1, 2] == 7

    def test_pad_constant(self, cpu):
        a = np.ones((1, 2, 2))
        out = cpu.pad(a, ((0, 0), (1, 1), (1, 1)))
        assert out.shape == (1, 4, 4)
        assert out[0, 0, 0] == 0.0
        assert out[0, 1, 1] == 1.0

    def test_pad_constant_values(self, cpu):
        a = np.ones((2, 2))
        out = cpu.pad(a, ((1, 1), (1, 1)), constant_values=3.0)
        assert out[0, 0] == 3.0


# =============================================================================
# Activations + norms
# =============================================================================

class TestActivationsAndNorms:
    def test_softmax_1d(self, cpu):
        a = np.array([1.0, 2.0, 3.0])
        assert np.allclose(cpu.softmax(a), _ref_softmax(a))

    def test_softmax_2d_axis0(self, cpu):
        a = np.random.randn(3, 5)
        assert np.allclose(cpu.softmax(a, axis=0), _ref_softmax(a, axis=0))

    def test_log_softmax(self, cpu):
        a = np.random.randn(4, 6)
        assert np.allclose(cpu.log_softmax(a), _ref_log_softmax(a))

    def test_layer_norm(self, cpu):
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.random.randn(8).astype(np.float32)
        b = np.random.randn(8).astype(np.float32)
        assert np.allclose(cpu.layer_norm(x, w, b), _ref_layernorm(x, w, b), atol=1e-5)

    def test_rms_norm(self, cpu):
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.random.randn(8).astype(np.float32)
        assert np.allclose(cpu.rms_norm(x, w), _ref_rmsnorm(x, w), atol=1e-5)

    def test_gelu(self, cpu):
        x = np.linspace(-3, 3, 101).astype(np.float32)
        assert np.allclose(cpu.gelu(x), _ref_gelu(x), atol=1e-5)

    def test_silu(self, cpu):
        x = np.linspace(-3, 3, 101).astype(np.float32)
        assert np.allclose(cpu.silu(x), x / (1 + np.exp(-x)), atol=1e-5)

    def test_relu(self, cpu):
        a = np.array([-2.0, 0.0, 3.0])
        assert np.array_equal(cpu.relu(a), np.maximum(a, 0))

    def test_sigmoid_clips(self, cpu):
        a = np.array([-1000.0, 0.0, 1000.0])
        assert np.allclose(cpu.sigmoid(a), [0.0, 0.5, 1.0], atol=1e-6)

    def test_tanh(self, cpu):
        a = np.array([-1.0, 0.0, 1.0])
        assert np.allclose(cpu.tanh(a), np.tanh(a))


class TestFusedOps:
    def test_fused_add_mul(self, cpu):
        a, b, c = np.random.randn(3, 4), np.random.randn(3, 4), np.random.randn(3, 4)
        assert np.allclose(cpu.fused_add_mul(a, b, c), a + b * c)

    def test_fused_mul_add(self, cpu):
        a, b, c = np.random.randn(3, 4), np.random.randn(3, 4), np.random.randn(3, 4)
        assert np.allclose(cpu.fused_mul_add(a, b, c), a * b + c)

    def test_fused_layernorm_gelu(self, cpu):
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.random.randn(8).astype(np.float32)
        b = np.random.randn(8).astype(np.float32)
        expected = _ref_gelu(_ref_layernorm(x, w, b))
        assert np.allclose(cpu.fused_layernorm_gelu(x, w, b), expected, atol=1e-4)

    def test_fused_layernorm_silu(self, cpu):
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.random.randn(8).astype(np.float32)
        b = np.random.randn(8).astype(np.float32)
        normed = _ref_layernorm(x, w, b)
        assert np.allclose(cpu.fused_layernorm_silu(x, w, b), normed / (1 + np.exp(-normed)), atol=1e-5)

    def test_fused_layer_norm_gelu_alias(self, cpu):
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.random.randn(8).astype(np.float32)
        b = np.random.randn(8).astype(np.float32)
        assert np.allclose(cpu.fused_layer_norm_gelu(x, w, b), cpu.fused_layernorm_gelu(x, w, b), atol=1e-6)


# =============================================================================
# Attention
# =============================================================================

class TestAttention:
    def test_scaled_dot_attention_basic(self, cpu):
        q = np.random.randn(2, 4, 3, 8).astype(np.float32)
        k = np.random.randn(2, 4, 5, 8).astype(np.float32)
        v = np.random.randn(2, 4, 5, 8).astype(np.float32)
        out = cpu.scaled_dot_attention(q, k, v)
        assert np.allclose(out, _ref_scaled_dot_attention(q, k, v), atol=1e-5)

    def test_scaled_dot_attention_mask_and_scale(self, cpu):
        q = np.random.randn(1, 1, 3, 8).astype(np.float32)
        k = np.random.randn(1, 1, 5, 8).astype(np.float32)
        v = np.random.randn(1, 1, 5, 8).astype(np.float32)
        mask = np.full((1, 1, 3, 5), -1e9, dtype=np.float32)
        mask[..., :2] = 0.0
        out = cpu.scaled_dot_attention(q, k, v, mask=mask, scale=0.5)
        assert np.allclose(out, _ref_scaled_dot_attention(q, k, v, mask=mask, scale=0.5), atol=1e-5)

    def test_scaled_dot_attention_causal(self, cpu):
        q = np.random.randn(2, 1, 4, 16).astype(np.float32)
        k = np.random.randn(2, 1, 4, 16).astype(np.float32)
        v = np.random.randn(2, 1, 4, 16).astype(np.float32)
        out = cpu.scaled_dot_attention(q, k, v, causal=True)
        assert np.allclose(out, _ref_scaled_dot_attention(q, k, v, causal=True), atol=1e-5)

    def test_scaled_dot_attention_online_softmax_path(self, cpu):
        # N*S > 65536 forces the tiled fused path
        q = np.random.randn(1, 1, 64, 16).astype(np.float32)
        k = np.random.randn(1, 1, 1025, 16).astype(np.float32)
        v = np.random.randn(1, 1, 1025, 16).astype(np.float32)
        out = cpu.scaled_dot_attention(q, k, v)
        assert np.allclose(out, _ref_scaled_dot_attention(q, k, v), atol=1e-3)

    def test_multi_head_attention(self, cpu):
        B, N, C, H = 2, 6, 16, 4
        q = np.random.randn(B, N, C).astype(np.float32)
        k = np.random.randn(B, N + 1, C).astype(np.float32)
        v = np.random.randn(B, N + 1, C).astype(np.float32)
        out, attn = cpu.multi_head_attention(q, k, v, H)
        assert out.shape == (B, N, C)
        assert attn.shape == (B, H, N, N + 1)
        assert np.all(np.isfinite(out))

    def test_multi_head_attention_causal(self, cpu):
        B, N, C, H = 1, 5, 8, 2
        q = np.random.randn(B, N, C).astype(np.float32)
        k = np.random.randn(B, N, C).astype(np.float32)
        v = np.random.randn(B, N, C).astype(np.float32)
        out, _ = cpu.multi_head_attention(q, k, v, H, causal=True)
        assert out.shape == (B, N, C)


class TestFusedSoftmaxAttention:
    def test_small_path_matches_direct(self, cpu):
        q = np.random.randn(1, 2, 4, 8).astype(np.float32)
        k = np.random.randn(1, 2, 6, 8).astype(np.float32)
        v = np.random.randn(1, 2, 6, 8).astype(np.float32)
        out = cpu.fused_softmax_attention(q, k, v)
        assert np.allclose(out, _ref_scaled_dot_attention(q, k, v), atol=1e-5)

    def test_small_path_mask_causal(self, cpu):
        q = np.random.randn(1, 1, 4, 8).astype(np.float32)
        k = np.random.randn(1, 1, 6, 8).astype(np.float32)
        v = np.random.randn(1, 1, 6, 8).astype(np.float32)
        mask = np.zeros((1, 1, 4, 6), dtype=np.float32)
        mask[..., -1] = -1e9
        out = cpu.fused_softmax_attention(q, k, v, mask=mask, causal=True)
        assert np.allclose(out, _ref_scaled_dot_attention(q, k, v, mask=mask, causal=True), atol=1e-4)

    def test_tiled_path_matches_direct(self, cpu):
        q = np.random.randn(1, 1, 64, 8).astype(np.float32)
        k = np.random.randn(1, 1, 1024, 8).astype(np.float32)
        v = np.random.randn(1, 1, 1024, 8).astype(np.float32)
        out = cpu.fused_softmax_attention(q, k, v)
        assert np.allclose(out, _ref_scaled_dot_attention(q, k, v), atol=1e-3)

    def test_tiled_path_causal(self, cpu):
        q = np.random.randn(1, 1, 32, 8).astype(np.float32)
        k = np.random.randn(1, 1, 2048, 8).astype(np.float32)
        v = np.random.randn(1, 1, 2048, 8).astype(np.float32)
        out = cpu.fused_softmax_attention(q, k, v, causal=True)
        assert np.allclose(out, _ref_scaled_dot_attention(q, k, v, causal=True), atol=1e-3)

    def test_tiled_path_mask(self, cpu):
        q = np.random.randn(1, 1, 64, 8).astype(np.float32)
        k = np.random.randn(1, 1, 1024, 8).astype(np.float32)
        v = np.random.randn(1, 1, 1024, 8).astype(np.float32)
        mask = np.zeros((1, 1, 64, 1024), dtype=np.float32)
        mask[..., 1000:] = -1e9
        out = cpu.fused_softmax_attention(q, k, v, mask=mask)
        assert np.allclose(out, _ref_scaled_dot_attention(q, k, v, mask=mask), atol=1e-3)


# =============================================================================
# Convolution / pooling
# =============================================================================

class TestConvAndPool:
    def test_conv2d_basic(self, cpu):
        x = np.random.randn(1, 2, 5, 5).astype(np.float32)
        w = np.random.randn(3, 2, 2, 2).astype(np.float32)
        out = cpu.conv2d(x, w)
        assert np.allclose(out, _ref_conv2d(x, w), atol=1e-4)

    def test_conv2d_stride_padding_bias(self, cpu):
        x = np.random.randn(1, 2, 6, 6).astype(np.float32)
        w = np.random.randn(3, 2, 2, 2).astype(np.float32)
        b = np.random.randn(3).astype(np.float32)
        out = cpu.conv2d(x, w, bias=b, stride=2, padding=1)
        assert np.allclose(out, _ref_conv2d(x, w, bias=b, stride=2, padding=1), atol=1e-4)

    def test_conv2d_batch(self, cpu):
        x = np.random.randn(2, 1, 4, 4).astype(np.float32)
        w = np.random.randn(2, 1, 2, 2).astype(np.float32)
        out = cpu.conv2d(x, w, stride=1, padding=1)
        assert np.allclose(out, _ref_conv2d(x, w, stride=1, padding=1), atol=1e-4)

    def test_im2col_shape(self, cpu):
        x = np.random.randn(1, 2, 4, 4).astype(np.float32)
        cols = cpu._im2col(x, 2, 2, 1)
        assert cols.shape == (1 * 3 * 3, 2 * 2 * 2)

    def test_max_pool2d(self, cpu):
        x = np.random.randn(1, 2, 6, 6).astype(np.float32)
        assert np.allclose(cpu.max_pool2d(x, 2, 2), _ref_max_pool(x, 2, 2))

    def test_max_pool2d_stride_1(self, cpu):
        x = np.random.randn(1, 1, 4, 4).astype(np.float32)
        assert np.allclose(cpu.max_pool2d(x, 2, 1), _ref_max_pool(x, 2, 1))

    def test_avg_pool2d(self, cpu):
        x = np.random.randn(1, 2, 6, 6).astype(np.float32)
        assert np.allclose(cpu.avg_pool2d(x, 2, 2), _ref_avg_pool(x, 2, 2))


# =============================================================================
# Embedding / losses / misc
# =============================================================================

class TestEmbeddingAndLoss:
    def test_embedding_lookup(self, cpu):
        weight = np.random.randn(10, 4).astype(np.float32)
        idx = np.array([[1, 2], [0, 5]])
        out = cpu.embedding_lookup(idx, weight)
        assert np.array_equal(out, weight[idx])

    def test_embedding_lookup_clips(self, cpu):
        weight = np.random.randn(10, 4).astype(np.float32)
        idx = np.array([-3, 100])
        out = cpu.embedding_lookup(idx, weight)
        assert np.array_equal(out[0], weight[0])
        assert np.array_equal(out[1], weight[9])

    def test_embedding_alias(self, cpu):
        weight = np.random.randn(5, 3).astype(np.float32)
        idx = np.array([0, 4])
        assert np.array_equal(cpu.embedding(idx, weight), weight[idx])

    def test_one_hot(self, cpu):
        idx = np.array([[0, 3], [1, 2]])
        out = cpu.one_hot(idx, 4)
        assert out.shape == (2, 2, 4)
        assert out[0, 0, 0] == 1 and out[1, 1, 2] == 1

    def test_one_hot_oob_ignored(self, cpu):
        idx = np.array([0, 9, -1])
        out = cpu.one_hot(idx, 3)
        assert out.shape == (3, 3)
        assert out[1].sum() == 0 and out[2].sum() == 0

    def test_cross_entropy(self, cpu):
        logits = np.random.randn(4, 6).astype(np.float32)
        targets = np.array([0, 2, 5, 3])
        assert np.isclose(cpu.cross_entropy(logits, targets), _ref_cross_entropy(logits, targets), atol=1e-6)

    def test_cross_entropy_skips_oob(self, cpu):
        logits = np.random.randn(4, 6).astype(np.float32)
        targets = np.array([0, 99, 5, -1])
        expected = _ref_cross_entropy(logits, targets)
        assert np.isclose(cpu.cross_entropy(logits, targets), expected, atol=1e-6)

    def test_cross_entropy_all_oob_zero(self, cpu):
        logits = np.random.randn(3, 4).astype(np.float32)
        targets = np.array([99, 100, -1])
        assert cpu.cross_entropy(logits, targets) == 0.0


class TestTensorOps:
    def test_concat(self, cpu):
        a = np.ones((2, 3))
        b = np.zeros((2, 3))
        assert np.array_equal(cpu.concat([a, b], axis=0), np.concatenate([a, b], axis=0))

    def test_stack(self, cpu):
        a = np.ones((2, 3))
        b = np.zeros((2, 3))
        assert np.array_equal(cpu.stack([a, b], axis=0), np.stack([a, b], axis=0))

    def test_permute(self, cpu):
        a = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        assert np.array_equal(cpu.permute(a, (2, 0, 1)), np.transpose(a, (2, 0, 1)))

    def test_reshape(self, cpu):
        a = np.arange(6, dtype=np.float32)
        assert np.array_equal(cpu.reshape(a, (2, 3)), a.reshape(2, 3))

    def test_transpose_axes(self, cpu):
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        assert np.array_equal(cpu.transpose(a, (1, 0)), a.T)

    def test_transpose_default(self, cpu):
        a = np.arange(6, dtype=np.float32).reshape(2, 3)
        assert np.array_equal(cpu.transpose(a), a.T)

    def test_topk_largest(self, cpu):
        a = np.random.randn(4, 10).astype(np.float32)
        values, indices = cpu.topk(a, 3, dim=-1, largest=True)
        ref_idx = np.argsort(a, axis=-1)[..., ::-1][..., :3]
        assert np.array_equal(indices, ref_idx)
        assert np.allclose(values, np.take_along_axis(a, ref_idx, axis=-1))

    def test_topk_smallest(self, cpu):
        a = np.random.randn(5,).astype(np.float32)
        values, indices = cpu.topk(a, 2, dim=-1, largest=False)
        assert np.array_equal(indices, np.argsort(a)[:2])
        assert np.allclose(values, np.sort(a)[:2])

    def test_multinomial(self, cpu):
        probs = np.array([0.7, 0.2, 0.1])
        out = cpu.multinomial(probs, 5, replacement=True)
        assert out.shape == (1, 5)
        assert np.all((out >= 0) & (out < 3))

    def test_multinomial_without_replacement(self, cpu):
        probs = np.array([0.7, 0.2, 0.1])
        out = cpu.multinomial(probs, 2, replacement=False)
        assert out.shape == (1, 2)
        assert len(np.unique(out)) == 2

    def test_multinomial_zero_total_raises(self, cpu):
        # total == 0 leaves p as all-zeros, which numpy rejects
        with pytest.raises(ValueError):
            cpu.multinomial(np.array([0.0, 0.0]), 1)


class TestDropoutAndNorm:
    def test_dropout_off(self, cpu):
        x = np.random.randn(3, 4)
        assert np.array_equal(cpu.dropout(x, p=0.5, training=False), x)

    def test_dropout_p0(self, cpu):
        x = np.random.randn(3, 4)
        assert np.array_equal(cpu.dropout(x, p=0.0, training=True), x)

    def test_dropout_training_scales(self, cpu):
        rng = np.random.default_rng(7)
        x = rng.standard_normal((5, 6)).astype(np.float32)
        p = 0.5
        np.random.seed(123)
        mask = (np.random.rand(*x.shape) > p).astype(np.float32)
        np.random.seed(123)
        out = cpu.dropout(x.copy(), p=p, training=True)
        assert out.shape == x.shape
        assert np.allclose(out, x * mask / (1 - p), atol=1e-6)

    def test_batch_norm_2d_training(self, cpu):
        x = np.random.randn(2, 3, 4, 4).astype(np.float32)
        gamma = np.random.randn(3).astype(np.float32)
        beta = np.random.randn(3).astype(np.float32)
        rm = np.zeros(3, dtype=np.float32)
        rv = np.ones(3, dtype=np.float32)
        rm_ref = rm.copy()
        rv_ref = rv.copy()
        out = cpu.batch_norm_2d(x, gamma, beta, rm, rv)
        exp = _ref_batchnorm2d(x, gamma, beta, rm_ref, rv_ref)
        assert np.allclose(out, exp, atol=1e-5)
        assert np.allclose(rm, rm_ref, atol=1e-6)
        assert np.allclose(rv, rv_ref, atol=1e-6)

    def test_batch_norm_2d_eval(self, cpu):
        x = np.random.randn(2, 3, 4, 4).astype(np.float32)
        gamma = np.ones(3, dtype=np.float32)
        beta = np.zeros(3, dtype=np.float32)
        rm = np.array([0.5, -0.5, 1.0], dtype=np.float32)
        rv = np.array([2.0, 1.0, 0.5], dtype=np.float32)
        exp = _ref_batchnorm2d(x, gamma, beta, rm, rv, training=False)
        assert np.allclose(cpu.batch_norm_2d(x, gamma, beta, rm, rv, training=False), exp, atol=1e-5)

    def test_batch_norm_1d_training(self, cpu):
        x = np.random.randn(5, 3).astype(np.float32)
        gamma = np.random.randn(3).astype(np.float32)
        beta = np.random.randn(3).astype(np.float32)
        rm = np.zeros(3, dtype=np.float32)
        rv = np.ones(3, dtype=np.float32)
        rm_ref, rv_ref = rm.copy(), rv.copy()
        out = cpu.batch_norm_1d(x, gamma, beta, rm, rv)
        exp = _ref_batchnorm1d(x, gamma, beta, rm_ref, rv_ref)
        assert np.allclose(out, exp, atol=1e-5)
        assert np.allclose(rm, rm_ref, atol=1e-6)

    def test_batch_norm_1d_eval(self, cpu):
        x = np.random.randn(5, 3).astype(np.float32)
        gamma = np.ones(3, dtype=np.float32)
        beta = np.zeros(3, dtype=np.float32)
        rm = np.array([1.0, 0.0, -1.0], dtype=np.float32)
        rv = np.array([1.0, 4.0, 0.25], dtype=np.float32)
        exp = _ref_batchnorm1d(x, gamma, beta, rm, rv, training=False)
        assert np.allclose(cpu.batch_norm_1d(x, gamma, beta, rm, rv, training=False), exp, atol=1e-5)


# =============================================================================
# _CPUBackend specifics
# =============================================================================

class TestCPUBackend:
    def test_has_openblas_true(self, monkeypatch):
        import ctypes.util
        monkeypatch.setattr(ctypes.util, "find_library", lambda name: "libopenblas.so")
        assert slib._CPUBackend().has_openblas() is True

    def test_has_openblas_false(self, monkeypatch):
        import ctypes.util
        monkeypatch.setattr(ctypes.util, "find_library", lambda name: None)
        assert slib._CPUBackend().has_openblas() is False

    def test_has_openblas_exception(self, monkeypatch):
        import ctypes.util
        def boom(name):
            raise OSError("no lib")
        monkeypatch.setattr(ctypes.util, "find_library", boom)
        assert slib._CPUBackend().has_openblas() is False

    def test_has_simd_x86(self, monkeypatch):
        monkeypatch.setattr("platform.machine", lambda: "x86_64")
        assert slib._CPUBackend().has_simd() is True

    def test_has_simd_arm(self, monkeypatch):
        monkeypatch.setattr("platform.machine", lambda: "aarch64")
        assert slib._CPUBackend().has_simd() is True

    def test_has_simd_other(self, monkeypatch):
        monkeypatch.setattr("platform.machine", lambda: "riscv64")
        assert slib._CPUBackend().has_simd() is False

    def test_has_simd_exception(self, monkeypatch):
        def boom():
            raise OSError("no platform")
        monkeypatch.setattr("platform.machine", boom)
        assert slib._CPUBackend().has_simd() is False

    def test_openblas_threads_env(self, cpu, monkeypatch):
        monkeypatch.setenv("OPENBLAS_NUM_THREADS", "4")
        assert cpu.openblas_threads() == 4

    def test_openblas_threads_omp_env(self, cpu, monkeypatch):
        cpu._openblas_threads_cache = None
        monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
        monkeypatch.setenv("OMP_NUM_THREADS", "6")
        assert cpu.openblas_threads() == 6

    def test_openblas_threads_cached(self, cpu):
        cpu._openblas_threads_cache = 9
        assert cpu.openblas_threads() == 9

    def test_openblas_threads_exception(self, cpu, monkeypatch):
        cpu._openblas_threads_cache = None
        monkeypatch.setattr("os.environ.get", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
        assert cpu.openblas_threads() == 1

    def test_openblas_threads_resource_manager_fallback(self, cpu, monkeypatch):
        cpu._openblas_threads_cache = None
        monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        monkeypatch.setitem(sys.modules, "domains.infrastructure.resource_manager", None)
        n = cpu.openblas_threads()
        assert n == max(1, (os.cpu_count() or 1) - 1)

    def test_vram_gb_psutil_fallback(self, cpu, monkeypatch):
        monkeypatch.setitem(sys.modules, "psutil", None)
        assert cpu.vram_gb() == 8.0

    def test_compute_tier_full(self, cpu, monkeypatch):
        monkeypatch.setattr(cpu, "openblas_threads", lambda: 8)
        monkeypatch.setattr(cpu, "vram_gb", lambda: 20.0)
        assert cpu.compute_tier == "full"

    def test_compute_tier_medium(self, cpu, monkeypatch):
        monkeypatch.setattr(cpu, "openblas_threads", lambda: 4)
        monkeypatch.setattr(cpu, "vram_gb", lambda: 8.0)
        assert cpu.compute_tier == "medium"

    def test_compute_tier_lite(self, cpu, monkeypatch):
        monkeypatch.setattr(cpu, "openblas_threads", lambda: 2)
        monkeypatch.setattr(cpu, "vram_gb", lambda: 4.0)
        assert cpu.compute_tier == "lite"

    def test_memory_hint(self, cpu, monkeypatch):
        monkeypatch.setattr(cpu, "openblas_threads", lambda: 8)
        monkeypatch.setattr(cpu, "vram_gb", lambda: 20.0)
        monkeypatch.setattr(cpu, "has_openblas", lambda: True)
        hint = cpu.memory_hint()
        assert hint["tier"] == "full"
        assert hint["threads"] == 8
        assert hint["max_batch"] == 8
        assert hint["max_seq_len"] == 512
        assert hint["recommend_openmp"] is True
        assert hint["recommend_quantization"] is False

    def test_cpu_softmax_matches_base(self, cpu):
        a = np.random.randn(3, 7).astype(np.float32)
        assert np.allclose(cpu.softmax(a), _ref_softmax(a))


# =============================================================================
# GPU backend numpy-fallback / unavailable state
# =============================================================================

class TestGpuBackends:
    def test_metal_not_available_without_mps(self, monkeypatch):
        monkeypatch.setattr(
            "domains.infrastructure.ml_types._mps_available", lambda: False
        )
        assert slib._MetalBackend().is_available() is False

    def test_metal_is_available_when_mps_detected(self, monkeypatch):
        monkeypatch.setattr(
            "domains.infrastructure.ml_types._mps_available", lambda: True
        )
        backend = slib._MetalBackend()
        assert backend.is_available() is True

    def test_metal_sync_is_noop(self):
        backend = slib._MetalBackend()
        assert backend.sync() is None

    def test_cuda_not_available(self):
        assert slib._CUDABackend().is_available() is False

    def test_cuda_fallback_matmul(self):
        backend = slib._CUDABackend()
        a = np.random.randn(3, 4)
        b = np.random.randn(4, 5)
        assert np.allclose(backend.matmul(a, b), a @ b)

    def test_cuda_fallback_to_device_from_device(self):
        backend = slib._CUDABackend()
        arr = np.arange(6, dtype=np.float64)
        assert backend.to_device(arr).dtype == np.float32
        assert np.array_equal(backend.from_device(arr), arr)

    def test_cuda_fallback_vram_and_tier(self):
        backend = slib._CUDABackend()
        assert backend.vram_gb() == 4.0
        assert backend.compute_tier == "medium"

    def test_cuda_fallback_scaled_dot(self):
        backend = slib._CUDABackend()
        q = np.random.randn(1, 1, 3, 8).astype(np.float32)
        k = np.random.randn(1, 1, 4, 8).astype(np.float32)
        v = np.random.randn(1, 1, 4, 8).astype(np.float32)
        assert np.allclose(backend.scaled_dot_attention(q, k, v), _ref_scaled_dot_attention(q, k, v), atol=1e-5)

    def test_cuda_fallback_layer_norm_gelu(self):
        backend = slib._CUDABackend()
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.random.randn(8).astype(np.float32)
        b = np.random.randn(8).astype(np.float32)
        assert np.allclose(backend.layer_norm(x, w, b), _ref_layernorm(x, w, b), atol=1e-5)
        assert np.allclose(backend.gelu(x), _ref_gelu(x), atol=1e-5)

    def test_cuda_memory_hint(self):
        backend = slib._CUDABackend()
        hint = backend.memory_hint()
        assert hint["tier"] == "medium"
        assert hint["vram_gb"] == 4.0
        assert hint["recommend_flash_attention"] is True

    def test_opencl_not_available(self):
        assert slib._OpenCLBackend().is_available() is False

    def test_opencl_vram_zero_without_context(self):
        assert slib._OpenCLBackend().vram_gb() == 0.0

    def test_opencl_sync_with_fake_queue(self):
        backend = slib._OpenCLBackend()
        class _Q:
            def finish(self):
                pass
        backend._queue = _Q()
        backend.sync()

    def test_opencl_matmul_numpy(self):
        backend = slib._OpenCLBackend()
        a = np.random.randn(3, 4)
        b = np.random.randn(4, 5)
        assert np.allclose(backend.matmul(a, b), a @ b)

    def test_opencl_matmul_non_ndarray(self):
        backend = slib._OpenCLBackend()
        assert np.allclose(backend.matmul([[1, 2]], [[3], [4]]), [[11]])


class TestCudaWithFakeCupy:
    """Exercise the cupy-present dispatch arms with a numpy-backed cupy proxy."""

    class _CupArr:
        def __init__(self, data):
            self.data = np.asarray(data)

        @property
        def shape(self):
            return self.data.shape

        def get(self):
            return self.data

        def max(self, *a, **k):
            return TestCudaWithFakeCupy._CupArr(self.data.max(*a, **k))

        def sum(self, *a, **k):
            return TestCudaWithFakeCupy._CupArr(self.data.sum(*a, **k))

        def mean(self, *a, **k):
            return TestCudaWithFakeCupy._CupArr(self.data.mean(*a, **k))

        def var(self, *a, **k):
            return TestCudaWithFakeCupy._CupArr(self.data.var(*a, **k))

        def __add__(self, o):
            return TestCudaWithFakeCupy._CupArr(self.data + TestCudaWithFakeCupy._val(o))

        def __radd__(self, o):
            return TestCudaWithFakeCupy._CupArr(TestCudaWithFakeCupy._val(o) + self.data)

        def __sub__(self, o):
            return TestCudaWithFakeCupy._CupArr(self.data - TestCudaWithFakeCupy._val(o))

        def __rsub__(self, o):
            return TestCudaWithFakeCupy._CupArr(TestCudaWithFakeCupy._val(o) - self.data)

        def __mul__(self, o):
            return TestCudaWithFakeCupy._CupArr(self.data * TestCudaWithFakeCupy._val(o))

        def __rmul__(self, o):
            return TestCudaWithFakeCupy._CupArr(TestCudaWithFakeCupy._val(o) * self.data)

        def __truediv__(self, o):
            return TestCudaWithFakeCupy._CupArr(self.data / TestCudaWithFakeCupy._val(o))

        def __pow__(self, o):
            return TestCudaWithFakeCupy._CupArr(self.data ** o)

        def __neg__(self):
            return TestCudaWithFakeCupy._CupArr(-self.data)

        def __eq__(self, o):
            return self.data == TestCudaWithFakeCupy._val(o)

    @staticmethod
    def _val(x):
        return x.data if isinstance(x, TestCudaWithFakeCupy._CupArr) else x

    @classmethod
    def _fake_cupy(cls, total_gb=16.0):
        class _Stream:
            @staticmethod
            def synchronize():
                return None

        class _StreamHolder:
            null = _Stream()

        class _Device:
            def mem_info(self):
                return (0, int(total_gb * 1024 ** 3))

        class _Cuda:
            Stream = _StreamHolder
            Device = _Device

        class _FakeCupy:
            float16 = np.float16
            float32 = np.float32
            pi = np.pi
            cuda = _Cuda()

            @classmethod
            def asarray(cls, a, dtype=None, **k):
                return cls._CupArr(np.asarray(a, dtype=dtype))

            @classmethod
            def asnumpy(cls, a):
                return cls._val(a) if isinstance(a, cls._CupArr) else np.asarray(a)

            @classmethod
            def matmul(cls, a, b):
                return cls._CupArr(np.matmul(cls._val(a), cls._val(b)))

            @classmethod
            def einsum(cls, expr, *a):
                return cls._CupArr(np.einsum(expr, *[cls._val(x) for x in a]))

            @classmethod
            def exp(cls, a):
                return cls._CupArr(np.exp(cls._val(a)))

            @classmethod
            def tanh(cls, a):
                return cls._CupArr(np.tanh(cls._val(a)))

            @classmethod
            def sqrt(cls, a):
                return cls._CupArr(np.sqrt(cls._val(a)))

            @classmethod
            def where(cls, c, a, b):
                return cls._CupArr(np.where(cls._val(c), cls._val(a), cls._val(b)))

            @classmethod
            def triu(cls, a, k=0):
                return cls._CupArr(np.triu(cls._val(a), k))

            @classmethod
            def ones(cls, shape, dtype=None):
                return cls._CupArr(np.ones(shape, dtype=dtype))

        _FakeCupy._CupArr = cls._CupArr
        _FakeCupy._val = cls._val
        return _FakeCupy

    def test_cuda_available_with_fake_cupy(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cupy", self._fake_cupy())
        backend = slib._CUDABackend()
        assert backend.is_available() is True
        assert backend._cp is not None

    def test_cuda_sync_with_fake_cupy(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cupy", self._fake_cupy())
        backend = slib._CUDABackend()
        backend.is_available()
        backend.sync()

    def test_cuda_vram_full_tier(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cupy", self._fake_cupy(total_gb=16.0))
        backend = slib._CUDABackend()
        backend.is_available()
        assert backend.vram_gb() == 16.0
        assert backend.compute_tier == "full"
        hint = backend.memory_hint()
        assert hint["tier"] == "full"
        assert hint["max_seq_len"] == 2048
        assert hint["recommend_quantization"] is False

    def test_cuda_vram_lite_tier(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cupy", self._fake_cupy(total_gb=2.0))
        backend = slib._CUDABackend()
        backend.is_available()
        assert backend.vram_gb() == 2.0
        assert backend.compute_tier == "lite"
        assert backend.memory_hint()["max_seq_len"] == 512

    def test_cuda_dtype_fp32_and_fp16(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cupy", self._fake_cupy())
        backend = slib._CUDABackend()
        backend.is_available()
        backend._fp16_mode = False
        assert backend._dtype() is np.float32
        backend._fp16_mode = True
        assert backend._dtype() is np.float16

    def test_cuda_vram_exception_falls_back(self, monkeypatch):
        class _BoomDevice:
            def mem_info(self):
                raise RuntimeError("no device")

        class _BoomCuda:
            Device = _BoomDevice

        class _Fake:
            float16 = np.float16
            float32 = np.float32
            cuda = _BoomCuda()

        monkeypatch.setitem(sys.modules, "cupy", _Fake())
        backend = slib._CUDABackend()
        backend.is_available()
        assert backend.vram_gb() == 4.0

    def test_cuda_to_from_device_with_fake_cupy(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cupy", self._fake_cupy())
        backend = slib._CUDABackend()
        backend.is_available()
        arr = np.arange(6, dtype=np.float64)
        dev = backend.to_device(arr)
        assert hasattr(dev, "get")
        assert np.array_equal(dev.get(), arr.astype(np.float32))
        out = backend.from_device(dev)
        assert out.dtype == np.float32
        assert np.array_equal(out, arr.astype(np.float32))

    def test_cuda_matmul_with_fake_cupy(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cupy", self._fake_cupy())
        backend = slib._CUDABackend()
        backend.is_available()
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        assert np.allclose(backend.matmul(a, b), a @ b, atol=1e-5)
        backend._fp16_mode = True
        assert np.allclose(backend.matmul(a, b), a @ b, atol=1e-4)

    def test_cuda_scaled_dot_causal_with_fake_cupy(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cupy", self._fake_cupy())
        backend = slib._CUDABackend()
        backend.is_available()
        q = np.random.randn(1, 1, 3, 8).astype(np.float32)
        k = np.random.randn(1, 1, 4, 8).astype(np.float32)
        v = np.random.randn(1, 1, 4, 8).astype(np.float32)
        out = backend.scaled_dot_attention(q, k, v, causal=True)
        ref = _ref_scaled_dot_attention(q, k, v, causal=True)
        assert np.allclose(out, ref, atol=1e-5)

    def test_cuda_scaled_dot_mask_with_fake_cupy(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cupy", self._fake_cupy())
        backend = slib._CUDABackend()
        backend.is_available()
        q = np.random.randn(1, 1, 3, 8).astype(np.float32)
        k = np.random.randn(1, 1, 4, 8).astype(np.float32)
        v = np.random.randn(1, 1, 4, 8).astype(np.float32)
        mask = np.full((1, 1, 3, 4), -1e9, dtype=np.float32)
        mask[..., :2] = 0.0
        out = backend.scaled_dot_attention(q, k, v, mask=mask)
        ref = _ref_scaled_dot_attention(q, k, v, mask=mask)
        assert np.allclose(out, ref, atol=1e-5)

    def test_cuda_layer_norm_gelu_with_fake_cupy(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cupy", self._fake_cupy())
        backend = slib._CUDABackend()
        backend.is_available()
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.random.randn(8).astype(np.float32)
        b = np.random.randn(8).astype(np.float32)
        assert np.allclose(backend.layer_norm(x, w, b), _ref_layernorm(x, w, b), atol=1e-5)
        assert np.allclose(backend.gelu(x), _ref_gelu(x), atol=1e-5)


# =============================================================================
# Metal backend dispatch arms with a numpy-backed torch proxy
# =============================================================================

class TestMetalBackendNumpy:
    """Exercise the numpy Metal backend.

    ``_MetalBackend`` is torch-free: availability is a platform check
    (``ml_types._mps_available``), device transfer is fp32 numpy, and all
    compute ops fall through to the base ``_Accelerator`` numpy
    implementations. These tests cover both arms of ``is_available`` and
    verify every op matches an independent numpy reference without torch.
    """

    def test_metal_not_available_on_this_platform(self):
        backend = slib._MetalBackend()
        assert backend.is_available() is (sys.platform == "darwin")

    def test_metal_available_when_mps_detected(self, monkeypatch):
        monkeypatch.setattr(
            "domains.infrastructure.ml_types._mps_available", lambda: True
        )
        backend = slib._MetalBackend()
        assert backend.is_available() is True

    def test_metal_unavailable_when_mps_absent(self, monkeypatch):
        monkeypatch.setattr(
            "domains.infrastructure.ml_types._mps_available", lambda: False
        )
        backend = slib._MetalBackend()
        assert backend.is_available() is False

    def test_metal_fp32_only(self):
        backend = slib._MetalBackend()
        assert backend._fp16_available is False
        assert backend.set_precision("fp16") == "fp32"
        assert backend.set_precision("auto") == "fp32"

    def test_metal_to_from_device_fp32(self):
        backend = slib._MetalBackend()
        arr = np.arange(6, dtype=np.float64)
        dev = backend.to_device(arr)
        assert dev.dtype == np.float32
        assert np.array_equal(backend.from_device(dev), arr.astype(np.float32))

    def test_metal_device_transfer_passthrough(self):
        backend = slib._MetalBackend()
        arr = np.array([3.0, 4.0])
        assert np.array_equal(backend.from_device(arr), arr)
        assert backend.from_device(arr) is not arr
        assert np.array_equal(
            backend.to_device([1, 2, 3]), np.asarray([1, 2, 3], dtype=np.float32)
        )

    def test_metal_sync_noop(self):
        backend = slib._MetalBackend()
        assert backend.sync() is None

    def test_metal_matmul(self):
        backend = slib._MetalBackend()
        a = np.random.randn(3, 4).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        assert np.allclose(backend.matmul(a, b), a @ b, atol=1e-5)

    def test_metal_elementary_ops(self):
        backend = slib._MetalBackend()
        a = np.random.randn(4, 5).astype(np.float32)
        b = np.random.randn(4, 5).astype(np.float32)
        assert np.allclose(backend.add(a, b), a + b, atol=1e-5)
        assert np.allclose(backend.neg(a), -a, atol=1e-5)
        assert np.allclose(backend.mul(a, b), a * b, atol=1e-5)
        assert np.allclose(backend.pow(a, 2), a ** 2, atol=1e-5)

    def test_metal_sum_mean(self):
        backend = slib._MetalBackend()
        a = np.random.randn(3, 4).astype(np.float32)
        assert np.allclose(backend.sum(a), a.sum())
        assert np.allclose(backend.sum(a, axis=1), a.sum(axis=1))
        assert np.allclose(backend.mean(a), a.mean())
        assert np.allclose(backend.mean(a, axis=0), a.mean(axis=0))

    def test_metal_activations(self):
        backend = slib._MetalBackend()
        x = np.random.randn(4, 8).astype(np.float32)
        assert np.allclose(backend.sigmoid(x), 1 / (1 + np.exp(-x)), atol=1e-5)
        assert np.allclose(backend.tanh(x), np.tanh(x), atol=1e-5)
        assert np.allclose(backend.relu(x), np.maximum(x, 0), atol=1e-5)
        assert np.allclose(backend.gelu(x), _ref_gelu(x), atol=1e-5)
        assert np.allclose(backend.silu(x), x / (1 + np.exp(-x)), atol=1e-5)
        assert np.allclose(backend.softmax(x), _ref_softmax(x), atol=1e-5)

    def test_metal_layer_rms_norm(self):
        backend = slib._MetalBackend()
        x = np.random.randn(4, 8).astype(np.float32)
        w = np.random.randn(8).astype(np.float32)
        b = np.random.randn(8).astype(np.float32)
        assert np.allclose(backend.layer_norm(x, w, b), _ref_layernorm(x, w, b), atol=1e-4)
        assert np.allclose(backend.rms_norm(x, w), _ref_rmsnorm(x, w), atol=1e-4)

    def test_metal_scaled_dot_causal_and_mask(self):
        backend = slib._MetalBackend()
        q = np.random.randn(1, 1, 3, 8).astype(np.float32)
        k = np.random.randn(1, 1, 4, 8).astype(np.float32)
        v = np.random.randn(1, 1, 4, 8).astype(np.float32)
        out = backend.scaled_dot_attention(q, k, v, causal=True)
        assert np.allclose(out, _ref_scaled_dot_attention(q, k, v, causal=True), atol=1e-5)
        mask = np.full((1, 1, 3, 4), -1e9, dtype=np.float32)
        mask[..., :2] = 0.0
        out = backend.scaled_dot_attention(q, k, v, mask=mask)
        assert np.allclose(out, _ref_scaled_dot_attention(q, k, v, mask=mask), atol=1e-5)

    def test_metal_cross_entropy(self):
        backend = slib._MetalBackend()
        logits = np.random.randn(4, 8).astype(np.float32)
        targets = np.array([0, 1, 7, 3])
        assert np.allclose(backend.cross_entropy(logits, targets),
                           _ref_cross_entropy(logits, targets), atol=1e-5)

    def test_metal_conv_maxpool_embedding_dropout(self):
        backend = slib._MetalBackend()
        x = np.random.randn(1, 2, 6, 6).astype(np.float32)
        weight = np.random.randn(3, 2, 3, 3).astype(np.float32)
        bias = np.random.randn(3).astype(np.float32)
        assert np.allclose(backend.conv2d(x, weight, bias, stride=1, padding=1),
                           _ref_conv2d(x, weight, bias, 1, 1), atol=1e-4)
        assert np.allclose(backend.max_pool2d(x, 2, 2), _ref_max_pool(x, 2, 2), atol=1e-5)
        idx = np.array([[0, 2], [1, 3]])
        emb = np.random.randn(4, 5).astype(np.float32)
        assert np.allclose(backend.embedding(idx, emb), emb[idx], atol=1e-5)
        assert np.allclose(backend.dropout(x, p=0.5, training=False), x, atol=1e-5)
        assert np.allclose(backend.dropout(x, p=0.0, training=True), x, atol=1e-5)
        assert backend.dropout(x, p=0.5, training=True).shape == x.shape

    def test_metal_abs_exp_sqrt_max(self):
        backend = slib._MetalBackend()
        a = np.abs(np.random.randn(3, 4)).astype(np.float32) + 0.1
        assert np.allclose(backend.abs(a), np.abs(a), atol=1e-5)
        assert np.allclose(backend.exp(a), np.exp(a), atol=1e-5)
        assert np.allclose(backend.sqrt(a), np.sqrt(a), atol=1e-5)
        assert np.allclose(backend.max(a), a.max())
        assert np.allclose(backend.max(a, axis=0), a.max(axis=0))


# =============================================================================
# OpenCL backend dispatch arms with a numpy-backed pyopencl proxy
# =============================================================================

class TestOpenCLWithFakeOpenCL:
    """Exercise the pyopencl-present dispatch arms with a minimal proxy.

    The real ``pyopencl`` is unavailable in this environment, so a minimal proxy
    mirrors the surface ``_OpenCLBackend`` touches: ``create_some_context``,
    ``CommandQueue``, ``Buffer``, ``enqueue_copy``, ``mem_flags``,
    ``get_platforms`` and device ``global_mem_size``.
    """

    class _Buf:
        def __init__(self, data):
            self.data = np.asarray(data)

        @property
        def shape(self):
            return self.data.shape

    @classmethod
    def _fake_opencl(cls, mem_bytes=8 * 1024 ** 3):
        _Buf = cls._Buf

        class _MemFlags:
            READ_WRITE = 1
            COPY_HOST_PTR = 2

        class _Dev:
            global_mem_size = mem_bytes

        class _Platform:
            def get_devices(self, *a, **k):
                return [_Dev()]

        class _Ctx:
            devices = [_Dev()]

        class _Queue:
            def __init__(self, ctx):
                self.context = ctx

            def finish(self):
                return None

        class _FakeCL:
            mem_flags = _MemFlags()

            @staticmethod
            def create_some_context(interactive=False):
                return _Ctx()

            @staticmethod
            def CommandQueue(ctx):
                return _Queue(ctx)

            @staticmethod
            def Buffer(context, flags, hostbuf):
                return _Buf(hostbuf)

            @staticmethod
            def enqueue_copy(queue, dest, src):
                dest[...] = np.asarray(src.data if isinstance(src, _Buf) else src).astype(np.float32)
                return None

            @staticmethod
            def get_platforms():
                return [_Platform()]

        return _FakeCL()

    def test_opencl_available_with_fake(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyopencl", self._fake_opencl())
        backend = slib._OpenCLBackend()
        assert backend.is_available() is True
        assert backend._cl is not None

    def test_opencl_vram_full_tier(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyopencl", self._fake_opencl(mem_bytes=8 * 1024 ** 3))
        backend = slib._OpenCLBackend()
        backend.is_available()
        assert backend.vram_gb() == 8.0
        assert backend.compute_tier == "full"
        hint = backend.memory_hint()
        assert hint["tier"] == "full"
        assert hint["max_batch"] == int(16 * (8.0 / 4))
        assert hint["recommend_quantization"] is False

    def test_opencl_vram_medium_tier(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyopencl", self._fake_opencl(mem_bytes=2 * 1024 ** 3))
        backend = slib._OpenCLBackend()
        backend.is_available()
        assert backend.compute_tier == "medium"
        assert backend.memory_hint()["max_seq_len"] == 256

    def test_opencl_vram_lite_tier(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyopencl", self._fake_opencl(mem_bytes=1 * 1024 ** 3))
        backend = slib._OpenCLBackend()
        backend.is_available()
        assert backend.compute_tier == "lite"
        assert backend.memory_hint()["recommend_quantization"] is True

    def test_opencl_vram_exception_falls_back(self, monkeypatch):
        class _BoomDev:
            global_mem_size = "boom"

        class _BoomPlatform:
            def get_devices(self, *a, **k):
                return [_BoomDev()]

        class _Ctx:
            devices = [_BoomDev()]

        class _FakeCL:
            @staticmethod
            def create_some_context(interactive=False):
                return _Ctx()

            @staticmethod
            def get_platforms():
                return [_BoomPlatform()]

        monkeypatch.setitem(sys.modules, "pyopencl", _FakeCL())
        backend = slib._OpenCLBackend()
        backend.is_available()
        assert backend.vram_gb() == 1.0

    def test_opencl_to_from_device_roundtrip(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyopencl", self._fake_opencl())
        backend = slib._OpenCLBackend()
        backend.is_available()
        arr = np.arange(6, dtype=np.float64)
        dev = backend.to_device(arr)
        assert isinstance(dev, self._Buf)
        assert np.array_equal(dev.data, arr.astype(np.float32))
        out = backend.from_device(dev)
        assert out.dtype == np.float32
        assert np.array_equal(out, arr.astype(np.float32))

    def test_opencl_lazy_context_import(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pyopencl", self._fake_opencl())
        backend = slib._OpenCLBackend()
        backend._ensure_context()
        assert backend._cl is not None
        assert backend._queue is not None


# =============================================================================
# Module-level convenience functions
# =============================================================================

class TestModuleFunctions:
    def test_to_gpu_from_gpu_roundtrip(self):
        arr = np.arange(6, dtype=np.float64)
        assert np.array_equal(slib.from_gpu(slib.to_gpu(arr)), arr)

    def test_gelu_module(self):
        x = np.linspace(-2, 2, 9).astype(np.float32)
        assert np.allclose(slib.gelu(x), _ref_gelu(x), atol=1e-5)

    def test_silu_module(self):
        x = np.linspace(-2, 2, 9).astype(np.float32)
        assert np.allclose(slib.silu(x), x / (1 + np.exp(-x)), atol=1e-5)

    def test_softmax_module(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert np.allclose(slib.softmax(a), _ref_softmax(a))


class TestBenchmarkAccelerators:
    def test_benchmark_cpu_ok(self):
        results = slib.benchmark_accelerators()
        assert set(results) == {"cpu", "metal", "cuda", "opencl"}
        assert results["cpu"]["status"] == "ok"
        assert results["cpu"]["gflops"] > 0
        assert results["metal"]["status"] == "unavailable"
        assert results["cuda"]["status"] == "unavailable"
        assert results["opencl"]["status"] == "unavailable"

    def test_benchmark_cpu_error_path(self, monkeypatch):
        class _BrokenCPU:
            name = "cpu"
            def is_available(self):
                return True
            def matmul(self, a, b):
                raise RuntimeError("no blas")
            def layer_norm(self, a, b, c):
                raise RuntimeError("no blas")
            def gelu(self, a):
                raise RuntimeError("no blas")
            def sync(self):
                pass
            def vram_gb(self):
                return 1.0
            compute_tier = "lite"
        monkeypatch.setattr(slib, "_CPUBackend", _BrokenCPU)
        results = slib.benchmark_accelerators()
        assert results["cpu"]["status"].startswith("error:")
        assert results["cpu"]["gflops"] == 0.0
