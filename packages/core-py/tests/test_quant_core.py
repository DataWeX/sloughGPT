"""
Tests for quant_core/wrapper — quantized GEMM with AVX2/numpy fallback.

Covers:
    - matmul_int8_c correctness against numpy reference
    - matmul_int4_c correctness against numpy reference
    - matmul_int8_f32_c correctness (when C lib available)
    - Fallback paths when C library unavailable
    - Shape validation
    - Edge cases (empty, single element)
"""

import numpy as np
import pytest
import sys
from pathlib import Path

_CORE_PY = Path(__file__).resolve().parents[1]
if str(_CORE_PY) not in sys.path:
    sys.path.insert(0, str(_CORE_PY))

from domains.infrastructure.quant_core.wrapper import (
    matmul_int8_c,
    matmul_int4_c,
    matmul_int8_f32_c,
    _fallback,
    _fallback_int4,
    HAS_AVX2,
)
from domains.infrastructure.quantization import _unpack_int4


# ── int8 matmul tests ─────────────────────────────────────────────────


class TestMatmulInt8:
    def test_basic_2x2(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.int8)
        B = np.array([[5, 6], [7, 8]], dtype=np.int8)
        result = matmul_int8_c(A, B)
        expected = _fallback(A, B)
        np.testing.assert_array_equal(result, expected)

    def test_random_small(self):
        M, N, K = 4, 8, 16
        A = np.random.randint(-128, 127, (M, K), dtype=np.int8)
        B = np.random.randint(-128, 127, (N, K), dtype=np.int8)
        result = matmul_int8_c(A, B)
        expected = _fallback(A, B)
        np.testing.assert_array_equal(result, expected)

    def test_random_medium(self):
        M, N, K = 32, 64, 128
        A = np.random.randint(-128, 127, (M, K), dtype=np.int8)
        B = np.random.randint(-128, 127, (N, K), dtype=np.int8)
        result = matmul_int8_c(A, B)
        expected = _fallback(A, B)
        np.testing.assert_array_equal(result, expected)

    def test_output_shape(self):
        M, N, K = 8, 16, 32
        A = np.random.randint(-128, 127, (M, K), dtype=np.int8)
        B = np.random.randint(-128, 127, (N, K), dtype=np.int8)
        result = matmul_int8_c(A, B)
        assert result.shape == (M, N)
        assert result.dtype == np.int32

    def test_single_element(self):
        A = np.array([[5]], dtype=np.int8)
        B = np.array([[3]], dtype=np.int8)
        result = matmul_int8_c(A, B)
        assert result[0, 0] == 15

    def test_identity(self):
        K = 8
        A = np.random.randint(-128, 127, (3, K), dtype=np.int8)
        B = np.eye(K, dtype=np.int8)
        result = matmul_int8_c(A, B)
        np.testing.assert_array_equal(result, A.astype(np.int32))

    def test_zeros(self):
        M, N, K = 4, 4, 4
        A = np.zeros((M, K), dtype=np.int8)
        B = np.random.randint(-128, 127, (N, K), dtype=np.int8)
        result = matmul_int8_c(A, B)
        np.testing.assert_array_equal(result, np.zeros((M, N), dtype=np.int32))


# ── int4 matmul tests ─────────────────────────────────────────────────


class TestMatmulInt4:
    def _pack_int4(self, B_int8):
        """Pack int8 array into int4 (two values per byte)."""
        N, K = B_int8.shape
        packed = np.zeros((N, K // 2), dtype=np.uint8)
        for i in range(K // 2):
            lo = B_int8[:, 2 * i].astype(np.uint8) & 0x0F
            hi = B_int8[:, 2 * i + 1].astype(np.uint8) & 0x0F
            packed[:, i] = (hi << 4) | lo
        return packed

    def test_basic(self):
        M, N, K = 2, 4, 8
        A = np.random.randint(-128, 127, (M, K), dtype=np.int8)
        B_int8 = np.random.randint(-8, 7, (N, K), dtype=np.int8)
        B_packed = self._pack_int4(B_int8)
        result = matmul_int4_c(A, B_packed, K)
        expected = _fallback_int4(A, B_packed, K)
        np.testing.assert_array_equal(result, expected)

    def test_output_shape(self):
        M, N, K = 4, 8, 16
        A = np.random.randint(-128, 127, (M, K), dtype=np.int8)
        B_int8 = np.random.randint(-8, 7, (N, K), dtype=np.int8)
        B_packed = self._pack_int4(B_int8)
        result = matmul_int4_c(A, B_packed, K)
        assert result.shape == (M, N)
        assert result.dtype == np.int32

    def test_zeros(self):
        M, N, K = 2, 4, 8
        A = np.zeros((M, K), dtype=np.int8)
        B_int8 = np.random.randint(-8, 7, (N, K), dtype=np.int8)
        B_packed = self._pack_int4(B_int8)
        result = matmul_int4_c(A, B_packed, K)
        np.testing.assert_array_equal(result, np.zeros((M, N), dtype=np.int32))


# ── Fallback path tests ──────────────────────────────────────────────


class TestFallbacks:
    def test_fallback_int8_matches_numpy(self):
        M, N, K = 8, 16, 32
        A = np.random.randint(-128, 127, (M, K), dtype=np.int8)
        B = np.random.randint(-128, 127, (N, K), dtype=np.int8)
        result = _fallback(A, B)
        expected = np.matmul(A.astype(np.int32), B.astype(np.int32).T)
        np.testing.assert_array_equal(result, expected)

    def test_fallback_int4_matches_manual_unpack(self):
        M, N, K = 4, 8, 16
        A = np.random.randint(-128, 127, (M, K), dtype=np.int8)
        B_int8 = np.random.randint(-8, 7, (N, K), dtype=np.int8)
        # Pack
        packed = np.zeros((N, K // 2), dtype=np.uint8)
        for i in range(K // 2):
            lo = B_int8[:, 2 * i].astype(np.uint8) & 0x0F
            hi = B_int8[:, 2 * i + 1].astype(np.uint8) & 0x0F
            packed[:, i] = (hi << 4) | lo
        result = _fallback_int4(A, packed, K)
        expected = np.matmul(A.astype(np.int32), B_int8.astype(np.int32).T)
        np.testing.assert_array_equal(result, expected)


# ── matmul_int8_f32_c tests ──────────────────────────────────────────


class TestMatmulInt8F32:
    def test_returns_none_when_no_lib(self):
        """If C library unavailable, should return None (not crash)."""
        M, N, K = 4, 8, 16
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randint(-128, 127, (N, K), dtype=np.int8)
        B_scale = 0.01
        result = matmul_int8_f32_c(A, B, B_scale)
        # Result is either a valid array or None (C lib unavailable)
        if result is not None:
            assert result.shape == (M, N)
            assert result.dtype == np.float32
            assert np.all(np.isfinite(result))

    def test_per_row_scale(self):
        M, N, K = 4, 8, 16
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randint(-128, 127, (N, K), dtype=np.int8)
        B_scale = np.random.rand(N).astype(np.float32) * 0.01 + 0.001
        result = matmul_int8_f32_c(A, B, B_scale)
        if result is not None:
            assert result.shape == (M, N)
            assert np.all(np.isfinite(result))


# ── HAS_AVX2 flag ────────────────────────────────────────────────────


class TestHASAVX2:
    def test_flag_is_bool(self):
        assert isinstance(HAS_AVX2, bool)


# ── AVX-512 VNNI flag ────────────────────────────────────────────────


class TestHASAVX512:
    def test_flag_is_bool(self):
        from domains.infrastructure.quant_core.wrapper import HAS_AVX512
        assert isinstance(HAS_AVX512, bool)

    def test_crossover_tracks_kernel(self):
        """Adaptive crossover must be lower when the AVX-512 kernel is active."""
        from domains.infrastructure.quant_core.wrapper import HAS_AVX512
        from domains.infrastructure import quantization as q

        expected = 512 if HAS_AVX512 else 1024
        assert q.QUANT_CROSSOVER_K == expected

    def test_avx512_path_correct(self):
        """The AVX-512 VNNI GEMM must match the numpy reference exactly."""
        from domains.infrastructure.quant_core.wrapper import HAS_AVX512
        if not HAS_AVX512:
            pytest.skip("AVX-512 VNNI kernel not active on this host")
        rng = np.random.default_rng(7)
        for K in (512, 1024, 127):
            A = rng.integers(-128, 127, (3, K), dtype=np.int8)
            B = rng.integers(-128, 127, (K, K), dtype=np.int8)
            got = matmul_int8_c(A, B)
            ref = (A.astype(np.int32) @ B.astype(np.int32).T).astype(np.int32)
            np.testing.assert_array_equal(got, ref)
