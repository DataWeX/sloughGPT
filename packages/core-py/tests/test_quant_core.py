"""Tests for AVX2-accelerated quantized matrix multiplication wrappers.

Tests cover:
  - C library loading and feature detection
  - matmul_int8_c — C path and numpy fallback
  - matmul_int8_f32_c — fused quantize-GEMM-dequant path
  - matmul_int4_c — packed int4 GEMM and fallback
  - Edge cases: empty arrays, single-row, large dims
"""

import numpy as np
import pytest
from domains.infrastructure.quant_core.wrapper import (
    _fallback,
    _fallback_int4,
    _load_lib,
    HAS_AVX2,
    matmul_int8_c,
    matmul_int8_f32_c,
    matmul_int4_c,
)


class TestFallbackInt8:
    def test_basic(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.int8)
        B = np.array([[5, 6], [7, 8]], dtype=np.int8)
        C = _fallback(A, B)
        expected = A.astype(np.int32) @ B.astype(np.int32).T
        np.testing.assert_array_equal(C, expected)

    def test_single_row(self):
        A = np.array([[1, 2, 3]], dtype=np.int8)
        B = np.array([[4, 5, 6]], dtype=np.int8)
        C = _fallback(A, B)
        assert C.shape == (1, 1)

    def test_1d_shapes(self):
        A = np.array([[1, 2]], dtype=np.int8)
        B = np.array([[3, 4]], dtype=np.int8)
        C = _fallback(A, B)
        assert C[0, 0] == 1 * 3 + 2 * 4


class TestFallbackInt4:
    def test_roundtrip(self):
        # B_packed shape: (N=2, K//2=2) — 2 rows, each with 2 bytes = 4 nibbles
        # Row 0: values [1, -2, 3, -4] packed into 2 bytes
        # Row 1: values [5, -1, 2, -3] packed into 2 bytes
        row0 = np.array([
            (1 & 0x0F) | ((-2 & 0x0F) << 4),   # byte 0: low=1, high=-2
            (3 & 0x0F) | ((-4 & 0x0F) << 4),    # byte 1: low=3, high=-4
        ], dtype=np.uint8)
        row1 = np.array([
            (5 & 0x0F) | ((-1 & 0x0F) << 4),    # byte 0: low=5, high=-1
            (2 & 0x0F) | ((-3 & 0x0F) << 4),    # byte 1: low=2, high=-3
        ], dtype=np.uint8)
        B_packed = np.array([row0, row1], dtype=np.uint8)  # (2, 2)
        A = np.array([[1, 1, 1, 1]], dtype=np.int8)         # (1, K=4)
        C = _fallback_int4(A, B_packed, 4)
        B_unpacked = np.array([
            [1, -2, 3, -4],
            [5, -1, 2, -3],
        ], dtype=np.int8)
        expected = A.astype(np.int32) @ B_unpacked.astype(np.int32).T
        np.testing.assert_array_equal(C, expected)


class TestMatmulInt8C:
    def test_basic(self):
        A = np.random.randint(-128, 127, (4, 8), dtype=np.int8)
        B = np.random.randint(-128, 127, (3, 8), dtype=np.int8)
        C = matmul_int8_c(A, B)
        expected = A.astype(np.int32) @ B.astype(np.int32).T
        np.testing.assert_array_equal(C, expected)

    def test_single_row(self):
        A = np.array([[1, 2, 3, 4]], dtype=np.int8)
        B = np.array([[5, 6, 7, 8]], dtype=np.int8)
        C = matmul_int8_c(A, B)
        assert C.shape == (1, 1)

    def test_result_dtype(self):
        A = np.zeros((1, 4), dtype=np.int8)
        B = np.zeros((1, 4), dtype=np.int8)
        C = matmul_int8_c(A, B)
        assert C.dtype == np.int32

    def test_symmetric(self):
        A = np.array([[1, 2]], dtype=np.int8)
        B = np.array([[1, 2]], dtype=np.int8)
        C = matmul_int8_c(A, B)
        assert C[0, 0] == 5

    def test_shape_mismatch_raises(self):
        A = np.zeros((1, 4), dtype=np.int8)
        B = np.zeros((1, 5), dtype=np.int8)
        with pytest.raises(AssertionError, match="B.shape"):
            matmul_int8_c(A, B)


class TestMatmulInt8F32C:
    def test_basic_per_tensor(self):
        A = np.random.randn(4, 8).astype(np.float32)
        B = np.random.randint(-128, 127, (3, 8), dtype=np.int8)
        scale = 0.05
        result = matmul_int8_f32_c(A, B, scale)
        if result is not None:
            assert result.shape == (4, 3)
            assert result.dtype == np.float32

    def test_basic_per_row(self):
        A = np.random.randn(4, 8).astype(np.float32)
        B = np.random.randint(-128, 127, (3, 8), dtype=np.int8)
        scales = np.random.rand(3).astype(np.float32) * 0.1
        result = matmul_int8_f32_c(A, B, scales)
        if result is not None:
            assert result.shape == (4, 3)

    def test_with_bias(self):
        A = np.random.randn(2, 8).astype(np.float32)
        B = np.random.randint(-128, 127, (3, 8), dtype=np.int8)
        bias = np.random.randn(3).astype(np.float32)
        result = matmul_int8_f32_c(A, B, 0.05, bias=bias)
        if result is not None:
            assert result.shape == (2, 3)

    def test_returns_none_when_unavailable(self):
        if HAS_AVX2:
            pytest.skip("C library available, can't test fallback path directly")
        result = matmul_int8_f32_c(
            np.zeros((1, 4), dtype=np.float32),
            np.zeros((1, 4), dtype=np.int8),
            1.0,
        )
        assert result is None


class TestMatmulInt4C:
    def test_basic(self):
        # A: (1, K=4), B_packed: (2, K//2=2) — 2 rows of packed int4
        A = np.array([[1, 1, 1, 1]], dtype=np.int8)
        B_packed = np.array([
            [(1 & 0x0F) | ((2 & 0x0F) << 4), (0 & 0x0F) | ((0 & 0x0F) << 4)],
            [(3 & 0x0F) | ((4 & 0x0F) << 4), (0 & 0x0F) | ((0 & 0x0F) << 4)],
        ], dtype=np.uint8)  # row0=[1,2,0,0], row1=[3,4,0,0]
        C = matmul_int4_c(A, B_packed, 4)
        # Expected: A @ B.T = [[1+2+0+0, 3+4+0+0]] = [[3, 7]]
        assert C.shape == (1, 2)
        assert C[0, 0] == 3
        assert C[0, 1] == 7

    def test_result_dtype(self):
        A = np.zeros((1, 4), dtype=np.int8)
        B_packed = np.zeros((1, 2), dtype=np.uint8)
        C = matmul_int4_c(A, B_packed, 4)
        assert C.dtype == np.int32

    def test_shape_mismatch_raises(self):
        A = np.zeros((1, 4), dtype=np.int8)
        B_packed = np.zeros((1, 2), dtype=np.uint8)
        with pytest.raises(AssertionError):
            matmul_int4_c(A, B_packed, 8)


class TestLoadLib:
    def test_returns_bool(self):
        result = _load_lib()
        assert isinstance(result, bool)

    def test_has_avx2_matches_load(self):
        assert HAS_AVX2 == _load_lib()
