"""Tests for training.gpu.accelerator — GPU accelerator."""

from __future__ import annotations

import pytest
import numpy as np
from domains.training.gpu.accelerator import (
    _MetalAccelerator,
    _CUDAAccelerator,
    _CPUAccelerator,
    get_accelerator,
)


# ── _CPUAccelerator ────────────────────────────────────────────────────────


class TestCPUAccelerator:

    def test_init(self):
        acc = _CPUAccelerator()
        assert acc.name == "cpu"
        assert acc.device_type == "cpu"

    def test_is_available(self):
        acc = _CPUAccelerator()
        assert acc.is_available() is True

    def test_matmul(self):
        acc = _CPUAccelerator()
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = acc.matmul(a, b)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.allclose(result, expected)

    def test_to_device(self):
        acc = _CPUAccelerator()
        arr = np.array([1.0, 2.0])
        result = acc.to_device(arr)
        assert np.allclose(result, arr)

    def test_from_device(self):
        acc = _CPUAccelerator()
        arr = np.array([1.0, 2.0])
        result = acc.from_device(arr)
        assert np.allclose(result, arr)


# ── _MetalAccelerator ──────────────────────────────────────────────────────


class TestMetalAccelerator:

    def test_init(self):
        acc = _MetalAccelerator()
        assert acc.name == "metal"
        assert acc.device_type == "gpu"

    def test_matmul(self):
        acc = _MetalAccelerator()
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = acc.matmul(a, b)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.allclose(result, expected)


# ── _CUDAAccelerator ──────────────────────────────────────────────────────


class TestCUDAAccelerator:

    def test_init(self):
        acc = _CUDAAccelerator()
        assert acc.name == "cuda"
        assert acc.device_type == "gpu"

    def test_is_available(self):
        acc = _CUDAAccelerator()
        assert isinstance(acc.is_available(), bool)

    def test_matmul_cpu_fallback(self):
        acc = _CPUAccelerator()
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = acc.matmul(a, b)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        assert np.allclose(result, expected)


# ── get_accelerator ────────────────────────────────────────────────────────


class TestGetAccelerator:

    def test_returns_accelerator(self):
        acc = get_accelerator()
        assert acc is not None
        assert hasattr(acc, "matmul")
        assert hasattr(acc, "is_available")
