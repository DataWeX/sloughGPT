"""Tests for domains.training.gpu.accelerator (SloNet multi-backend acceleration layer)."""

import os
import sys
import types

import numpy as np
import pytest

import domains.training.gpu.accelerator as accel_mod
from domains.training.gpu.accelerator import (
    _CPUAccelerator,
    _CUDAAccelerator,
    _MetalAccelerator,
    cholesky,
    dominant_eigen,
    from_gpu,
    get_accelerator,
    reset_accelerator,
    solve_cholesky,
    solve_triangular,
    to_gpu,
)


def _set_metal(monkeypatch, available):
    fake = types.ModuleType("domains.infrastructure.ml_types")
    fake._mps_available = lambda: available
    monkeypatch.setitem(sys.modules, "domains.infrastructure.ml_types", fake)


class _ArrayLike:
    """Minimal cupy-like array exposing .get()."""

    def __init__(self, arr):
        self._arr = np.asarray(arr, dtype=np.float32)

    def get(self):
        return self._arr

    def __add__(self, other):
        return self._arr + np.asarray(other)

    def __radd__(self, other):
        return np.asarray(other) + self._arr

    def __array__(self, dtype=None):
        return self._arr.astype(dtype) if dtype is not None else self._arr


def _fake_cupy(monkeypatch):
    fake = types.ModuleType("cupy")
    fake.asarray = lambda arr: np.asarray(arr, dtype=np.float32)
    fake.matmul = lambda a, b: np.matmul(np.asarray(a), np.asarray(b))
    monkeypatch.setitem(sys.modules, "cupy", fake)
    return fake


def _unset_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)


@pytest.fixture(autouse=True)
def _reset():
    reset_accelerator()
    yield
    reset_accelerator()


class TestMetalAccelerator:
    def test_check_metal_import_failure(self, monkeypatch):
        """Import failure yields unavailable metal accelerator."""
        fake = types.ModuleType("domains.infrastructure.ml_types")
        monkeypatch.setitem(sys.modules, "domains.infrastructure.ml_types", fake)
        acc = _MetalAccelerator()
        assert acc._available is False
        assert acc.is_available() is False

    def test_check_metal_available(self, monkeypatch):
        """MPS availability is detected via ml_types shim."""
        _set_metal(monkeypatch, True)
        acc = _MetalAccelerator()
        assert acc._available is True
        assert acc.is_available() is True
        assert acc.name == "metal"
        assert acc.device_type == "gpu"

    def test_device_transfer(self, monkeypatch):
        _set_metal(monkeypatch, True)
        acc = _MetalAccelerator()
        arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
        dev = acc.to_device(arr)
        assert dev.dtype == np.float32
        assert dev is not arr
        back = acc.from_device(dev)
        np.testing.assert_allclose(back, arr.astype(np.float32))

    def test_matmul_add(self, monkeypatch):
        _set_metal(monkeypatch, True)
        acc = _MetalAccelerator()
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        np.testing.assert_allclose(acc.matmul(a, b), np.matmul(a, b))
        np.testing.assert_allclose(acc.add(a, b), a + b)

    def test_activations(self, monkeypatch):
        _set_metal(monkeypatch, True)
        acc = _MetalAccelerator()
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        sm = acc.softmax(x)
        assert abs(sm.sum() - 1.0) < 1e-6
        np.testing.assert_allclose(acc.gelu(x), 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))))
        np.testing.assert_allclose(acc.silu(x), x / (1 + np.exp(-np.clip(x, -500, 500))))

    def test_norms(self, monkeypatch):
        _set_metal(monkeypatch, True)
        acc = _MetalAccelerator()
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        weight = np.ones(3)
        bias = np.zeros(3)
        ln = acc.layernorm(arr, weight, bias)
        assert ln.shape == arr.shape
        rn = acc.rmsnorm(arr, weight)
        assert rn.shape == arr.shape

    def test_attention(self, monkeypatch):
        _set_metal(monkeypatch, True)
        acc = _MetalAccelerator()
        q = np.array([[1.0, 0.0]])
        k = np.array([[1.0, 0.0], [0.0, 1.0]])
        v = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = acc.attention(q, k, v, scale=0.5)
        assert out.shape == (1, 2)

    def test_scaled_dot_attention(self, monkeypatch):
        _set_metal(monkeypatch, True)
        acc = _MetalAccelerator()
        q = np.array([[[1.0, 0.0]]])
        k = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        v = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        out = acc.scaled_dot_attention(q, k, v)
        assert out.shape == (1, 1, 2)
        mask = np.zeros((1, 1, 2))
        out2 = acc.scaled_dot_attention(q, k, v, mask=mask, scale=1.0)
        assert out2.shape == (1, 1, 2)

    def test_dropout(self, monkeypatch):
        _set_metal(monkeypatch, True)
        acc = _MetalAccelerator()
        arr = np.ones((100, 100))
        same = acc.dropout(arr, p=0.0)
        assert np.array_equal(same, arr)
        dropped = acc.dropout(arr, p=0.5)
        assert dropped.shape == arr.shape

    def test_embedding(self, monkeypatch):
        _set_metal(monkeypatch, True)
        acc = _MetalAccelerator()
        weight = np.arange(12, dtype=np.float32).reshape(4, 3)
        idx = np.array([[0, 2], [3, 1]])
        out = acc.embedding(idx, weight)
        np.testing.assert_allclose(out, weight[idx])

    def test_cross_entropy(self, monkeypatch):
        _set_metal(monkeypatch, True)
        acc = _MetalAccelerator()
        logits = np.array([[2.0, 0.5, 0.1], [0.1, 3.0, 0.2]])
        targets = np.array([0, 1])
        loss = acc.cross_entropy(logits, targets)
        assert loss > 0.0
        assert loss < 1.0

    def test_conv2d(self, monkeypatch):
        _set_metal(monkeypatch, True)
        acc = _MetalAccelerator()
        inp = np.ones((1, 1, 3, 3), dtype=np.float32)
        w = np.ones((2, 1, 2, 2), dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)
        out = acc.conv2d(inp, w, b, stride=1, padding=0)
        assert out.shape == (1, 2, 2, 2)
        padded = acc.conv2d(inp, w, None, stride=2, padding=1)
        assert padded.shape == (1, 2, 2, 2)

    def test_max_pool2d(self, monkeypatch):
        _set_metal(monkeypatch, True)
        acc = _MetalAccelerator()
        inp = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
        out = acc.max_pool2d(inp, kernel_size=2)
        assert out.shape == (1, 1, 1, 1)
        assert out[0, 0, 0, 0] == 4.0
        tup = acc.max_pool2d(inp, kernel_size=(2, 2), stride=1, padding=0)
        assert tup.shape == (1, 1, 1, 1)


class TestCUDAAccelerator:
    def test_no_cuda_no_env(self, monkeypatch):
        _unset_cupy(monkeypatch)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        acc = _CUDAAccelerator()
        assert acc._cp is None
        assert acc._available is False
        assert acc.is_available() is False

    def test_cuda_env_present(self, monkeypatch):
        _unset_cupy(monkeypatch)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        acc = _CUDAAccelerator()
        assert acc._available is True
        assert acc.is_available() is False

    def test_cuda_env_disabled_values(self, monkeypatch):
        _unset_cupy(monkeypatch)
        for val in ("", "-1"):
            monkeypatch.setenv("CUDA_VISIBLE_DEVICES", val)
            acc = _CUDAAccelerator()
            assert acc._available is False

    def test_cupy_available(self, monkeypatch):
        _fake_cupy(monkeypatch)
        acc = _CUDAAccelerator()
        assert acc._cp is not None
        assert acc._available is True
        assert acc.is_available() is True

    def test_device_transfer(self, monkeypatch):
        _fake_cupy(monkeypatch)
        acc = _CUDAAccelerator()
        dev = acc.to_device(np.ones((2, 2)))
        assert dev.dtype == np.float32
        wrapped = _ArrayLike(dev)
        back = acc.from_device(wrapped)
        np.testing.assert_allclose(back, np.ones((2, 2)))
        plain = acc.from_device(dev)
        np.testing.assert_allclose(plain, np.ones((2, 2)))

    def test_matmul_add(self, monkeypatch):
        _fake_cupy(monkeypatch)
        acc = _CUDAAccelerator()
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        np.testing.assert_allclose(acc.matmul(a, b), np.matmul(a, b))
        np.testing.assert_allclose(acc.add(a, b), a + b)

    def test_softmax(self, monkeypatch):
        _fake_cupy(monkeypatch)
        acc = _CUDAAccelerator()
        x = np.array([-1.0, 0.0, 2.0])
        sm = acc.softmax(x)
        assert abs(sm.sum() - 1.0) < 1e-6

    def test_gelu(self, monkeypatch):
        _fake_cupy(monkeypatch)
        acc = _CUDAAccelerator()
        x = np.array([0.5, -1.0])
        np.testing.assert_allclose(acc.gelu(x), 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))))

    def test_layernorm_attention(self, monkeypatch):
        _fake_cupy(monkeypatch)
        acc = _CUDAAccelerator()
        arr = np.array([[1.0, 2.0, 3.0]])
        ln = acc.layernorm(arr, np.ones(3), np.zeros(3))
        assert ln.shape == arr.shape
        q = np.array([[1.0, 0.0]])
        k = np.array([[1.0, 0.0], [0.0, 1.0]])
        v = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert acc.attention(q, k, v, scale=0.5).shape == (1, 2)

    def test_conv2d(self, monkeypatch):
        _fake_cupy(monkeypatch)
        acc = _CUDAAccelerator()
        inp = np.ones((1, 1, 3, 3), dtype=np.float32)
        w = np.ones((1, 1, 2, 2), dtype=np.float32)
        out = acc.conv2d(inp, w, None, stride=1, padding=0)
        assert out.shape == (1, 1, 2, 2)
        out_b = acc.conv2d(inp, w, np.zeros(1, dtype=np.float32), stride=1, padding=1)
        assert out_b.shape == (1, 1, 4, 4)

    def test_cpu_fallback_paths(self, monkeypatch):
        _unset_cupy(monkeypatch)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        acc = _CUDAAccelerator()
        assert acc.to_device(np.ones(2)).dtype == np.float32
        np.testing.assert_allclose(acc.from_device(np.ones(2)), np.ones(2))
        np.testing.assert_allclose(acc.matmul(np.ones((1, 2)), np.ones((2, 1))), [[2.0]])
        np.testing.assert_allclose(acc.add(np.ones(2), np.ones(2)), np.ones(2) * 2)
        sm = acc.softmax(np.array([0.0, 1.0]))
        assert abs(sm.sum() - 1.0) < 1e-6


class TestCPUAccelerator:
    def test_all_ops(self):
        acc = _CPUAccelerator()
        assert acc.name == "cpu"
        assert acc.device_type == "cpu"
        assert acc.is_available() is True
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        dev = acc.to_device(arr)
        assert dev.dtype == np.float32
        assert acc.from_device(dev) is dev
        np.testing.assert_allclose(acc.matmul(arr, arr.T), np.matmul(arr, arr.T))
        np.testing.assert_allclose(acc.add(arr, arr), arr * 2)

    def test_softmax_gelu_silu(self):
        acc = _CPUAccelerator()
        x = np.array([-1.0, 0.0, 1.0])
        sm = acc.softmax(x)
        assert abs(sm.sum() - 1.0) < 1e-6
        np.testing.assert_allclose(acc.gelu(x), 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))))
        np.testing.assert_allclose(acc.silu(x), x / (1 + np.exp(-x)))

    def test_norms(self):
        acc = _CPUAccelerator()
        arr = np.array([[1.0, 2.0, 3.0]])
        assert acc.layernorm(arr, np.ones(3), np.zeros(3)).shape == arr.shape
        assert acc.rmsnorm(arr, np.ones(3)).shape == arr.shape

    def test_attention_and_scaled(self):
        acc = _CPUAccelerator()
        q = np.array([[1.0, 0.0]])
        k = np.array([[1.0, 0.0], [0.0, 1.0]])
        v = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert acc.attention(q, k, v, scale=0.5).shape == (1, 2)
        q3 = np.array([[[1.0, 0.0]]])
        k3 = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        v3 = np.array([[[1.0, 2.0], [3.0, 4.0]]])
        assert acc.scaled_dot_attention(q3, k3, v3).shape == (1, 1, 2)
        assert acc.scaled_dot_attention(q3, k3, v3, mask=np.zeros((1, 1, 2)), scale=1.0).shape == (1, 1, 2)
    def test_conv2d_and_pool(self):
        acc = _CPUAccelerator()
        inp = np.ones((1, 1, 4, 4), dtype=np.float32)
        w = np.ones((1, 1, 2, 2), dtype=np.float32)
        assert acc.conv2d(inp, w, None, stride=2, padding=0).shape == (1, 1, 2, 2)
        assert acc.conv2d(inp, w, np.zeros(1, dtype=np.float32), stride=1, padding=1).shape == (1, 1, 5, 5)
        pool_in = np.array([[[[1.0, 2.0], [3.0, 4.0]]]])
        assert acc.max_pool2d(pool_in, kernel_size=2, stride=2)[0, 0, 0, 0] == 4.0

    def test_embedding_clips(self):
        acc = _CPUAccelerator()
        weight = np.arange(12, dtype=np.float32).reshape(4, 3)
        idx = np.array([[0, 5], [2, -1]])
        out = acc.embedding(idx, weight)
        assert out.shape == (2, 2, 3)

    def test_cross_entropy(self):
        acc = _CPUAccelerator()
        logits = np.array([[2.0, 0.5], [0.1, 3.0]])
        targets = np.array([0, 1])
        loss = acc.cross_entropy(logits, targets)
        assert loss > 0.0 and loss < 1.0
        oob = acc.cross_entropy(logits, np.array([0, 99]))
        assert oob > 0.0

    def test_dropout(self):
        acc = _CPUAccelerator()
        arr = np.ones((50, 50))
        assert np.array_equal(acc.dropout(arr, 0.0), arr)
        assert np.array_equal(acc.dropout(arr, 0.5, training=False), arr)
        out = acc.dropout(arr, 0.5, training=True)
        assert out.shape == arr.shape


class TestGlobalAccelerator:
    def test_cpu_fallback(self, monkeypatch):
        _unset_cupy(monkeypatch)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        _set_metal(monkeypatch, False)
        acc = get_accelerator()
        assert isinstance(acc, _CPUAccelerator)

    def test_metal_priority(self, monkeypatch):
        _unset_cupy(monkeypatch)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        _set_metal(monkeypatch, True)
        acc = get_accelerator()
        assert isinstance(acc, _MetalAccelerator)

    def test_cuda_priority(self, monkeypatch):
        _fake_cupy(monkeypatch)
        acc = get_accelerator()
        assert isinstance(acc, _CUDAAccelerator)

    def test_cached(self, monkeypatch):
        _unset_cupy(monkeypatch)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        _set_metal(monkeypatch, False)
        a = get_accelerator()
        b = get_accelerator()
        assert a is b

    def test_to_from_gpu(self, monkeypatch):
        _unset_cupy(monkeypatch)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        _set_metal(monkeypatch, False)
        arr = np.array([1.0, 2.0, 3.0])
        dev = to_gpu(arr)
        assert dev.dtype == np.float32
        np.testing.assert_allclose(from_gpu(dev), arr)

    def test_reset(self, monkeypatch):
        _unset_cupy(monkeypatch)
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        _set_metal(monkeypatch, False)
        get_accelerator()
        reset_accelerator()
        assert accel_mod._accelerator is None
        assert get_accelerator() is not None


class TestSolver:
    def test_cholesky(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        L = cholesky(A)
        np.testing.assert_allclose(L @ L.T, A, atol=1e-5)

    def test_solve_triangular_lower(self):
        L = np.array([[2.0, 0.0], [1.0, 3.0]])
        b = np.array([4.0, 7.0])
        x = solve_triangular(L, b, lower=True)
        np.testing.assert_allclose(L @ x, b, atol=1e-5)

    def test_solve_triangular_upper(self):
        U = np.array([[2.0, 1.0], [0.0, 3.0]])
        b = np.array([5.0, 6.0])
        x = solve_triangular(U, b, lower=False)
        np.testing.assert_allclose(U @ x, b, atol=1e-5)

    def test_solve_cholesky(self):
        A = np.array([[4.0, 1.0], [1.0, 3.0]])
        b = np.array([1.0, 2.0])
        x = solve_cholesky(A, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-5)

    def test_dominant_eigen(self):
        A = np.array([[3.0, 0.0], [0.0, 1.0]])
        vals, vecs = dominant_eigen(A, n_eigen=2)
        assert vals[0] == pytest.approx(3.0, abs=0.05)
        assert vals[1] == pytest.approx(1.0, abs=0.05)
        assert vecs.shape == (2, 2)

    def test_dominant_eigen_converges(self):
        A = np.array([[4.0, 1.0], [1.0, 4.0]])
        vals, _ = dominant_eigen(A, n_eigen=1, max_iter=200, tol=1e-8)
        assert vals[0] == pytest.approx(5.0, abs=0.01)
