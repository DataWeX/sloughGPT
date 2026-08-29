"""Tests for domains.infrastructure.quant_core.wrapper — numpy fallback matmul."""

import numpy as np
from domains.infrastructure.quant_core.wrapper import _fallback, _fallback_int4


class TestFallback:
    def test_basic_matmul(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.int8)
        B = np.array([[5, 6], [7, 8]], dtype=np.int8)
        result = _fallback(A, B)
        expected = np.matmul(A.astype(np.int32), B.astype(np.int32).T)
        np.testing.assert_array_equal(result, expected)

    def test_single_row(self):
        A = np.array([[1, 2, 3]], dtype=np.int8)
        B = np.array([[4, 5, 6]], dtype=np.int8)
        result = _fallback(A, B)
        assert result.shape == (1, 1)

    def test_zeros(self):
        A = np.zeros((2, 3), dtype=np.int8)
        B = np.zeros((4, 3), dtype=np.int8)
        result = _fallback(A, B)
        np.testing.assert_array_equal(result, np.zeros((2, 4), dtype=np.int32))


class TestFallbackInt4:
    def test_basic(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.int8)
        B_packed = np.array([[0x12], [0x34]], dtype=np.uint8)
        result = _fallback_int4(A, B_packed, K=2)
        assert result.shape[0] == 2
        assert result.dtype == np.int32
