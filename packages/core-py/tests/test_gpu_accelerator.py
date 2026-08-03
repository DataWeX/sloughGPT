"""Tests for the SloNet GPU acceleration layer (domains/training/gpu/accelerator.py).

The three backend classes (CPU, CUDA, Metal) are all pure-numpy compute in this
environment — cupy and MPS are unavailable, so the numpy fallback branches are
the actual production code path. Each accelerator is instantiated directly and
every compute op is verified against an independent numpy reference. The
cupy-dispatch branches (_cp is not None) are the documented environmental floor
and are never executed here. CUDA backend detection (CUDA_VISIBLE_DEVICES env)
and the global get_accelerator/to_gpu/from_gpu helpers are covered, along with
the Cholesky solvers and the power-iteration dominant_eigen.
"""

import importlib.util

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view

from domains.training.gpu import accelerator as acc


_CUPY_AVAILABLE = importlib.util.find_spec("cupy") is not None


# =============================================================================
# Independent numpy references
# =============================================================================

def _ref_softmax(a, axis=-1):
    e = np.exp(a - np.max(a, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _ref_layernorm(a, w, b, eps=1e-5):
    mean = a.mean(axis=-1, keepdims=True)
    var = a.var(axis=-1, keepdims=True)
    return ((a - mean) / np.sqrt(var + eps)) * w + b


def _ref_rmsnorm(a, w, eps=1e-5):
    return (a / np.sqrt(np.mean(a * a, axis=-1, keepdims=True) + eps)) * w


def _ref_attention(q, k, v, scale=1.0):
    scores = np.matmul(q, k.T) * scale
    return np.matmul(_ref_softmax(scores, axis=-1), v)


def _ref_scaled_dot_attention(q, k, v, mask=None, scale=None):
    if scale is None:
        scale = 1.0 / np.sqrt(k.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -2, -1)) * scale
    if mask is not None:
        scores = scores + mask
    return np.matmul(_ref_softmax(scores, axis=-1), v)


def _ref_conv2d(input, weight, bias=None, stride=1, padding=0):
    """Reference convolution via per-kernel-offset accumulation.

    Deliberately different structure from the implementation: each kernel
    offset contributes one scaled, strided slice of the (padded) input plane
    instead of extracting one patch per output position.
    """
    x = np.pad(input, [(0, 0), (0, 0), (padding, padding), (padding, padding)], mode="constant")
    n, c, h, w = x.shape
    oc, ic, kh, kw = weight.shape
    oh = (h - kh) // stride + 1
    ow = (w - kw) // stride + 1
    out = np.zeros((n, oc, oh, ow), dtype=np.float32)
    for i in range(n):
        for j in range(oc):
            plane = np.zeros((oh, ow), dtype=np.float32)
            for ci in range(ic):
                for a in range(kh):
                    for b in range(kw):
                        plane += weight[j, ci, a, b] * x[i, ci, a:a + oh * stride:stride, b:b + ow * stride:stride]
            if bias is not None:
                plane += bias[j]
            out[i, j] = plane
    return out


def _ref_maxpool_sliding(x, k, st):
    """Vectorized reference for in-bounds pooling (no padding)."""
    win = sliding_window_view(x, (k, k), axis=(2, 3))
    return win[:, :, ::st, ::st].max(axis=(-2, -1))


def _ref_maxpool2d(input, kernel_size, stride=None, padding=0):
    """Reference max-pool with clipped edge windows (matches Metal semantics)."""
    x = np.asarray(input)
    n, c, h, w = x.shape
    kh = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    kw = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
    st = stride if stride is not None else kh
    sh = st if isinstance(st, int) else st[0]
    sw = st if isinstance(st, int) else st[1]
    oh = (h + 2 * padding - kh) // sh + 1
    ow = (w + 2 * padding - kw) // sw + 1
    out = np.full((n, c, oh, ow), -np.inf, dtype=np.float32)
    for i in range(n):
        for ci in range(c):
            for a in range(oh):
                for b in range(ow):
                    ih = a * sh - padding
                    iw = b * sw - padding
                    win = x[i, ci, max(0, ih):min(h, ih + kh), max(0, iw):min(w, iw + kw)]
                    if win.size > 0:
                        out[i, ci, a, b] = win.max()
    return out


def _ref_cross_entropy(logits, targets):
    l = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(l) / np.exp(l).sum(axis=-1, keepdims=True)
    return float(-np.log(np.maximum(p[np.arange(len(targets)), targets.astype(int)], 1e-10)).mean())


def _pd_matrix(n, seed):
    rng = np.random.RandomState(seed)
    x = rng.randn(n, n)
    return x @ x.T + n * np.eye(n)


# =============================================================================
# CPU accelerator
# =============================================================================

class TestCPUAccelerator:
    def setup_method(self):
        self.cpu = acc._CPUAccelerator()

    def test_is_available_true(self):
        assert self.cpu.is_available() is True

    def test_device_transfer_roundtrip(self):
        arr = np.arange(6, dtype=np.float64).reshape(2, 3)
        gpu = self.cpu.to_device(arr)
        assert gpu.dtype == np.float32
        assert not np.shares_memory(gpu, arr)
        assert self.cpu.from_device(gpu) is gpu

    def test_matmul_matches_numpy(self):
        a = np.random.rand(2, 3)
        b = np.random.rand(3, 4)
        np.testing.assert_allclose(self.cpu.matmul(a, b), np.matmul(a, b))

    def test_add_matches_numpy(self):
        a = np.random.rand(2, 3)
        b = np.random.rand(2, 3)
        np.testing.assert_allclose(self.cpu.add(a, b), a + b)

    def test_softmax_rows_sum_to_one(self):
        a = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 0.5]])
        out = self.cpu.softmax(a)
        np.testing.assert_allclose(out.sum(axis=-1), np.ones(2), rtol=1e-6)
        np.testing.assert_allclose(out, _ref_softmax(a))

    def test_softmax_axis_zero(self):
        a = np.random.rand(3, 4)
        np.testing.assert_allclose(self.cpu.softmax(a, axis=0), _ref_softmax(a, axis=0))

    def test_gelu_matches_formula(self):
        a = np.random.rand(5, 5) - 0.5
        ref = 0.5 * a * (1 + np.tanh(np.sqrt(2 / np.pi) * (a + 0.044715 * a ** 3)))
        np.testing.assert_allclose(self.cpu.gelu(a), ref)

    def test_silu_matches_formula(self):
        a = np.random.rand(4, 4) - 0.5
        np.testing.assert_allclose(self.cpu.silu(a), a / (1 + np.exp(-a)))

    def test_layernorm_matches_reference(self):
        a = np.random.rand(3, 5)
        w = np.random.rand(5)
        b = np.random.rand(5)
        np.testing.assert_allclose(self.cpu.layernorm(a, w, b), _ref_layernorm(a, w, b))

    def test_rmsnorm_matches_reference(self):
        a = np.random.rand(3, 5)
        w = np.random.rand(5)
        np.testing.assert_allclose(self.cpu.rmsnorm(a, w), _ref_rmsnorm(a, w))

    def test_attention_matches_reference(self):
        q = np.random.rand(2, 3)
        k = np.random.rand(2, 3)
        v = np.random.rand(2, 4)
        np.testing.assert_allclose(self.cpu.attention(q, k, v, scale=0.5), _ref_attention(q, k, v, 0.5))

    def test_scaled_dot_attention_batched_with_mask(self):
        q = np.random.rand(2, 3, 4)
        k = np.random.rand(2, 3, 4)
        v = np.random.rand(2, 3, 5)
        mask = np.broadcast_to(np.triu(np.full((3, 3), -1e9), k=1), (2, 3, 3))
        np.testing.assert_allclose(
            self.cpu.scaled_dot_attention(q, k, v, mask=mask),
            _ref_scaled_dot_attention(q, k, v, mask=mask),
        )

    def test_scaled_dot_attention_no_mask_explicit_scale(self):
        q = np.random.rand(2, 3, 4)
        k = np.random.rand(2, 3, 4)
        v = np.random.rand(2, 3, 5)
        np.testing.assert_allclose(
            self.cpu.scaled_dot_attention(q, k, v, scale=1.0),
            _ref_scaled_dot_attention(q, k, v, scale=1.0),
        )

    def test_conv2d_single_channel_no_bias(self):
        x = np.random.rand(1, 1, 3, 3)
        w = np.random.rand(1, 1, 2, 2)
        np.testing.assert_allclose(self.cpu.conv2d(x, w), _ref_conv2d(x, w), atol=1e-5)

    def test_conv2d_with_bias(self):
        x = np.random.rand(1, 1, 3, 3)
        w = np.random.rand(1, 1, 2, 2)
        b = np.random.rand(1)
        np.testing.assert_allclose(self.cpu.conv2d(x, w, b), _ref_conv2d(x, w, b), atol=1e-5)

    def test_conv2d_stride(self):
        x = np.random.rand(1, 1, 3, 3)
        w = np.random.rand(1, 1, 2, 2)
        np.testing.assert_allclose(self.cpu.conv2d(x, w, stride=2), _ref_conv2d(x, w, stride=2), atol=1e-5)

    def test_conv2d_padded_multichannel(self):
        x = np.random.rand(2, 2, 4, 4)
        w = np.random.rand(3, 2, 3, 3)
        b = np.random.rand(3)
        np.testing.assert_allclose(
            self.cpu.conv2d(x, w, b, stride=1, padding=1),
            _ref_conv2d(x, w, b, stride=1, padding=1),
            atol=1e-5,
        )

    def test_max_pool2d_default(self):
        x = np.random.rand(1, 1, 4, 4)
        np.testing.assert_allclose(self.cpu.max_pool2d(x), _ref_maxpool_sliding(x, 2, 2))

    def test_max_pool2d_stride_one(self):
        x = np.random.rand(2, 1, 4, 4)
        np.testing.assert_allclose(self.cpu.max_pool2d(x, kernel_size=2, stride=1), _ref_maxpool_sliding(x, 2, 1))

    def test_max_pool2d_kernel_three(self):
        x = np.random.rand(1, 2, 6, 6)
        np.testing.assert_allclose(self.cpu.max_pool2d(x, kernel_size=3, stride=2), _ref_maxpool_sliding(x, 3, 2))

    def test_embedding_2d_indices_clipped(self):
        weight = np.random.rand(5, 4)
        indices = np.array([[0, -1, 2], [3, 7, 4]])
        out = self.cpu.embedding(indices, weight)
        assert out.shape == (2, 3, 4)
        clipped = np.clip(indices.astype(int).flatten(), 0, 4)
        np.testing.assert_allclose(out, weight[clipped].reshape(2, 3, 4))

    def test_cross_entropy_matches_reference(self):
        logits = np.random.rand(4, 7)
        targets = np.array([0, 3, 6, 1])
        np.testing.assert_allclose(self.cpu.cross_entropy(logits, targets), _ref_cross_entropy(logits, targets))

    def test_cross_entropy_skips_out_of_range_targets(self):
        logits = np.random.rand(3, 5)
        targets = np.array([2, 5, 0])
        result = self.cpu.cross_entropy(logits, targets)
        p = _ref_softmax(logits)
        expected = (-np.log(p[0, 2]) - np.log(p[2, 0])) / 3
        np.testing.assert_allclose(result, expected)

    def test_dropout_eval_mode_returns_input(self):
        a = np.random.rand(4, 4)
        assert self.cpu.dropout(a, 0.5, training=False) is a

    def test_dropout_p_zero_returns_input(self):
        a = np.random.rand(4, 4)
        assert self.cpu.dropout(a, 0.0, training=True) is a

    def test_dropout_training_scales_and_masks(self):
        np.random.seed(3)
        a = np.ones(4000, dtype=np.float32)
        out = self.cpu.dropout(a, 0.5, training=True)
        frac = np.mean(out != 0)
        assert abs(frac - 0.5) < 0.03
        np.testing.assert_allclose(out.mean(), 1.0, atol=0.05)
        kept = out[out != 0]
        np.testing.assert_allclose(kept, 2.0)


# =============================================================================
# Metal accelerator (pure-numpy compute, backend unavailable here)
# =============================================================================

class TestMetalAccelerator:
    def setup_method(self):
        self.metal = acc._MetalAccelerator()

    def test_device_transfer(self):
        arr = np.random.rand(3, 3)
        gpu = self.metal.to_device(arr)
        assert gpu.dtype == np.float32
        np.testing.assert_allclose(gpu, arr)
        np.testing.assert_allclose(self.metal.from_device(gpu), arr)

    def test_matmul_and_add(self):
        a = np.random.rand(2, 3)
        b = np.random.rand(3, 4)
        np.testing.assert_allclose(self.metal.matmul(a, b), np.matmul(a, b))
        np.testing.assert_allclose(self.metal.add(a, a[:2, :3]), a + a[:2, :3])

    def test_softmax(self):
        a = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 0.5]])
        np.testing.assert_allclose(self.metal.softmax(a), _ref_softmax(a))
        np.testing.assert_allclose(self.metal.softmax(a, axis=0), _ref_softmax(a, axis=0))

    def test_gelu_and_silu(self):
        a = np.random.rand(4, 4) - 0.5
        ref_gelu = 0.5 * a * (1 + np.tanh(np.sqrt(2 / np.pi) * (a + 0.044715 * a ** 3)))
        np.testing.assert_allclose(self.metal.gelu(a), ref_gelu)
        np.testing.assert_allclose(self.metal.silu(a), a / (1 + np.exp(-np.clip(a, -500, 500))))

    def test_layernorm_and_rmsnorm(self):
        a = np.random.rand(3, 5)
        w = np.random.rand(5)
        b = np.random.rand(5)
        np.testing.assert_allclose(self.metal.layernorm(a, w, b), _ref_layernorm(a, w, b))
        np.testing.assert_allclose(self.metal.rmsnorm(a, w), _ref_rmsnorm(a, w))

    def test_attention(self):
        q = np.random.rand(2, 3)
        k = np.random.rand(2, 3)
        v = np.random.rand(2, 4)
        np.testing.assert_allclose(self.metal.attention(q, k, v, scale=0.5), _ref_attention(q, k, v, 0.5))

    def test_check_metal_exception_returns_false(self, monkeypatch):
        def boom():
            raise ImportError("mps unavailable")

        monkeypatch.setattr("domains.infrastructure.ml_types._mps_available", boom)
        assert acc._MetalAccelerator()._available is False

    def test_scaled_dot_attention_explicit_scale_no_mask(self):
        q = np.random.rand(2, 3, 4)
        k = np.random.rand(2, 3, 4)
        v = np.random.rand(2, 3, 5)
        np.testing.assert_allclose(
            self.metal.scaled_dot_attention(q, k, v, scale=1.0),
            _ref_scaled_dot_attention(q, k, v, scale=1.0),
        )

    def test_scaled_dot_attention_batched_with_mask(self):
        q = np.random.rand(2, 3, 4)
        k = np.random.rand(2, 3, 4)
        v = np.random.rand(2, 3, 5)
        mask = np.broadcast_to(np.triu(np.full((3, 3), -1e9), k=1), (2, 3, 3))
        np.testing.assert_allclose(
            self.metal.scaled_dot_attention(q, k, v, mask=mask),
            _ref_scaled_dot_attention(q, k, v, mask=mask),
        )

    def test_dropout_p_zero_returns_input(self):
        a = np.random.rand(4, 4)
        assert self.metal.dropout(a, 0.0) is a

    def test_dropout_p_negative_returns_input(self):
        a = np.random.rand(4, 4)
        assert self.metal.dropout(a, -0.5) is a

    def test_dropout_scales_and_masks(self):
        np.random.seed(9)
        a = np.ones(4000, dtype=np.float32)
        out = self.metal.dropout(a, 0.5)
        frac = np.mean(out != 0)
        assert abs(frac - 0.5) < 0.03
        np.testing.assert_allclose(out.mean(), 1.0, atol=0.05)
        np.testing.assert_allclose(out[out != 0], 2.0)

    def test_embedding_in_range(self):
        weight = np.random.rand(5, 4)
        indices = np.array([[0, 2], [3, 1]])
        out = self.metal.embedding(indices, weight)
        assert out.shape == (2, 2, 4)
        np.testing.assert_allclose(out, weight[indices.astype(int)])

    def test_embedding_out_of_range_raises(self):
        weight = np.random.rand(5, 4)
        with pytest.raises(IndexError):
            self.metal.embedding(np.array([0, 5]), weight)

    def test_cross_entropy_matches_reference(self):
        logits = np.random.rand(4, 7)
        targets = np.array([0, 3, 6, 1])
        np.testing.assert_allclose(self.metal.cross_entropy(logits, targets), _ref_cross_entropy(logits, targets))

    def test_conv2d_matches_reference(self):
        x = np.random.rand(1, 1, 3, 3)
        w = np.random.rand(1, 1, 2, 2)
        b = np.random.rand(1)
        np.testing.assert_allclose(self.metal.conv2d(x, w, b), _ref_conv2d(x, w, b), atol=1e-5)

    def test_max_pool2d_default(self):
        x = np.random.rand(1, 1, 4, 4)
        np.testing.assert_allclose(self.metal.max_pool2d(x), _ref_maxpool2d(x, 2))

    def test_max_pool2d_tuple_kernel_and_stride(self):
        x = np.random.rand(1, 1, 5, 6)
        np.testing.assert_allclose(
            self.metal.max_pool2d(x, kernel_size=(2, 3), stride=(1, 2)),
            _ref_maxpool2d(x, (2, 3), stride=(1, 2)),
        )

    def test_max_pool2d_with_padding(self):
        x = np.random.rand(1, 1, 4, 4)
        np.testing.assert_allclose(
            self.metal.max_pool2d(x, kernel_size=2, stride=2, padding=1),
            _ref_maxpool2d(x, 2, stride=2, padding=1),
        )


# =============================================================================
# CUDA accelerator (numpy fallback path, cupy not installed)
# =============================================================================

class TestCUDAFallbackAccelerator:
    def setup_method(self):
        self.cuda = acc._CUDAAccelerator()

    def test_is_available_false_without_cupy(self):
        assert self.cuda._cp is None
        assert self.cuda.is_available() is False

    @pytest.mark.skipif(_CUPY_AVAILABLE, reason="cupy installed; env detection bypassed")
    def test_check_cuda_env_true_branch(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        cuda = acc._CUDAAccelerator()
        assert cuda._available is True
        assert cuda._cp is None
        assert cuda.is_available() is False

    @pytest.mark.skipif(_CUPY_AVAILABLE, reason="cupy installed; env detection bypassed")
    def test_check_cuda_env_false_branches(self, monkeypatch):
        for value in ("", "-1"):
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", value)
            assert acc._CUDAAccelerator()._available is False
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert acc._CUDAAccelerator()._available is False

    def test_device_transfer_fallback(self):
        arr = np.random.rand(3, 3)
        gpu = self.cuda.to_device(arr)
        assert gpu.dtype == np.float32
        np.testing.assert_allclose(gpu, arr)
        np.testing.assert_allclose(self.cuda.from_device(gpu), arr)

    def test_matmul_and_add_fallback(self):
        a = np.random.rand(2, 3)
        b = np.random.rand(3, 4)
        np.testing.assert_allclose(self.cuda.matmul(a, b), np.matmul(a, b))
        np.testing.assert_allclose(self.cuda.add(a, a[:2, :3]), a + a[:2, :3])

    def test_softmax_ndarray_input(self):
        a = np.random.rand(2, 3)
        out = self.cuda.softmax(a)
        assert isinstance(out, np.ndarray)
        np.testing.assert_allclose(out, _ref_softmax(a))

    def test_softmax_list_input_converted(self):
        out = self.cuda.softmax([[1.0, 2.0], [3.0, 0.0]])
        np.testing.assert_allclose(out, _ref_softmax(np.array([[1.0, 2.0], [3.0, 0.0]])))

    def test_gelu_fallback(self):
        a = np.random.rand(4, 4) - 0.5
        ref = 0.5 * a * (1 + np.tanh(np.sqrt(2 / np.pi) * (a + 0.044715 * a ** 3)))
        np.testing.assert_allclose(self.cuda.gelu(a), ref)

    def test_layernorm_fallback(self):
        a = np.random.rand(3, 5)
        w = np.random.rand(5)
        b = np.random.rand(5)
        np.testing.assert_allclose(self.cuda.layernorm(a, w, b), _ref_layernorm(a, w, b))

    def test_attention_fallback(self):
        q = np.random.rand(2, 3)
        k = np.random.rand(2, 3)
        v = np.random.rand(2, 4)
        np.testing.assert_allclose(self.cuda.attention(q, k, v, scale=0.5), _ref_attention(q, k, v, 0.5))

    def test_conv2d_fallback(self):
        x = np.random.rand(1, 1, 3, 3)
        w = np.random.rand(1, 1, 2, 2)
        b = np.random.rand(1)
        np.testing.assert_allclose(self.cuda.conv2d(x, w, b, stride=1, padding=1), _ref_conv2d(x, w, b, padding=1), atol=1e-5)

    def test_conv2d_impl_static_no_bias(self):
        x = np.random.rand(2, 2, 4, 4)
        w = np.random.rand(1, 2, 3, 3)
        np.testing.assert_allclose(
            acc._CUDAAccelerator._conv2d_impl(x, w, None, 2, 0),
            _ref_conv2d(x, w, None, stride=2, padding=0),
            atol=1e-5,
        )


# =============================================================================
# Global accelerator helpers
# =============================================================================

class TestGlobalAccelerator:
    def test_reset_and_get_caches_singleton(self):
        acc.reset_accelerator()
        first = acc.get_accelerator()
        assert acc.get_accelerator() is first

    def test_get_accelerator_prefers_cpu_without_gpu_backend(self):
        if _CUPY_AVAILABLE or acc._MetalAccelerator().is_available():
            pytest.skip("a GPU backend is available; CPU selection not expected")
        acc.reset_accelerator()
        assert isinstance(acc.get_accelerator(), acc._CPUAccelerator)

    def test_to_gpu_from_gpu_roundtrip(self):
        acc.reset_accelerator()
        arr = np.random.rand(2, 4).astype(np.float64)
        gpu = acc.to_gpu(arr)
        assert gpu.dtype == np.float32
        np.testing.assert_allclose(acc.from_gpu(gpu), arr)


# =============================================================================
# Cholesky solvers
# =============================================================================

class TestSolvers:
    def test_cholesky_matches_numpy(self):
        a = _pd_matrix(5, 7)
        l = acc.cholesky(a)
        np.testing.assert_allclose(l, np.linalg.cholesky(a), atol=2e-4)
        np.testing.assert_allclose(l @ l.T, a, atol=2e-4)

    def test_solve_triangular_lower(self):
        a = np.tril(np.random.RandomState(1).rand(5, 5)) + np.eye(5)
        b = np.random.RandomState(2).rand(5)
        np.testing.assert_allclose(acc.solve_triangular(a, b, lower=True), np.linalg.solve(a, b), atol=1e-4)

    def test_solve_triangular_upper(self):
        a = np.triu(np.random.RandomState(3).rand(5, 5)) + np.eye(5)
        b = np.random.RandomState(4).rand(5)
        np.testing.assert_allclose(acc.solve_triangular(a, b, lower=False), np.linalg.solve(a, b), atol=1e-4)

    def test_solve_cholesky_matches_numpy(self):
        a = _pd_matrix(5, 11)
        b = np.random.RandomState(5).rand(5)
        np.testing.assert_allclose(acc.solve_cholesky(a, b), np.linalg.solve(a, b), atol=1e-3)


# =============================================================================
# Power-iteration dominant eigen decomposition
# =============================================================================

class TestDominantEigen:
    def test_dominant_eigen_defaults(self):
        np.random.seed(5)
        a = _pd_matrix(4, 1)
        evals, evecs = acc.dominant_eigen(a)
        true = np.linalg.eigvalsh(a)[-1]
        assert abs(evals[0] - true) < 1e-3
        assert np.linalg.norm(a @ evecs[:, 0] - evals[0] * evecs[:, 0]) < 1e-2

    def test_dominant_eigen_two_components(self):
        np.random.seed(6)
        a = _pd_matrix(4, 2)
        evals, evecs = acc.dominant_eigen(a, n_eigen=2, max_iter=2000, tol=1e-9)
        true = np.sort(np.linalg.eigvalsh(a))[::-1][:2]
        np.testing.assert_allclose(evals, true, atol=1e-3)
        for e in range(2):
            assert np.linalg.norm(a @ evecs[:, e] - evals[e] * evecs[:, e]) < 1e-2

    def test_dominant_eigen_loop_exhausts_max_iter(self):
        np.random.seed(4)
        a = _pd_matrix(4, 4)
        evals, evecs = acc.dominant_eigen(a, n_eigen=1, max_iter=1, tol=1e-12)
        assert np.isfinite(evals[0])
        assert np.linalg.norm(evecs[:, 0]) > 0

    def test_dominant_eigen_deterministic_with_seed(self):
        np.random.seed(8)
        a = _pd_matrix(4, 3)
        first = acc.dominant_eigen(a, n_eigen=2, max_iter=2000, tol=1e-9)
        np.random.seed(8)
        second = acc.dominant_eigen(a, n_eigen=2, max_iter=2000, tol=1e-9)
        np.testing.assert_allclose(first[0], second[0])
