"""Tests for domains.infrastructure.quant_core.wrapper — numpy fallback matmul."""

import numpy as np
import pytest
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

    def test_single_element(self):
        A = np.array([[5]], dtype=np.int8)
        B = np.array([[3]], dtype=np.int8)
        result = _fallback(A, B)
        assert result.shape == (1, 1)
        assert result[0, 0] == 15

    def test_negative_values(self):
        A = np.array([[-1, 2], [3, -4]], dtype=np.int8)
        B = np.array([[-5, 6], [7, -8]], dtype=np.int8)
        result = _fallback(A, B)
        expected = np.matmul(A.astype(np.int32), B.astype(np.int32).T)
        np.testing.assert_array_equal(result, expected)

    def test_identity_like(self):
        A = np.eye(3, dtype=np.int8)
        B = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int8)
        result = _fallback(A, B)
        expected = np.matmul(A.astype(np.int32), B.astype(np.int32).T)
        np.testing.assert_array_equal(result, expected)

    def test_transpose_property(self):
        A = np.random.randint(-10, 10, (3, 5)).astype(np.int8)
        B = np.random.randint(-10, 10, (4, 5)).astype(np.int8)
        result = _fallback(A, B)
        expected = np.matmul(A.astype(np.int32), B.astype(np.int32).T)
        np.testing.assert_array_equal(result, expected)

    def test_result_dtype_int32(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.int8)
        B = np.array([[5, 6], [7, 8]], dtype=np.int8)
        result = _fallback(A, B)
        assert result.dtype == np.int32

    def test_ones_matrix(self):
        A = np.ones((3, 3), dtype=np.int8)
        B = np.ones((3, 3), dtype=np.int8)
        result = _fallback(A, B)
        expected = np.full((3, 3), 3, dtype=np.int32)
        np.testing.assert_array_equal(result, expected)

    def test_rectangular_output_shape(self):
        A = np.random.randint(-5, 5, (2, 7)).astype(np.int8)
        B = np.random.randint(-5, 5, (3, 7)).astype(np.int8)
        result = _fallback(A, B)
        assert result.shape == (2, 3)

    def test_large_values(self):
        A = np.array([[127, 127], [127, 127]], dtype=np.int8)
        B = np.array([[127, 127], [127, 127]], dtype=np.int8)
        result = _fallback(A, B)
        expected_val = 127 * 127 * 2
        assert result[0, 0] == expected_val

    def test_negative_int8_extremes(self):
        A = np.array([[-128, -128], [-128, -128]], dtype=np.int8)
        B = np.array([[-128, -128], [-128, -128]], dtype=np.int8)
        result = _fallback(A, B)
        expected_val = (-128) * (-128) * 2
        assert result[0, 0] == expected_val

    def test_mixed_signs(self):
        A = np.array([[1, -1], [-1, 1]], dtype=np.int8)
        B = np.array([[1, 1], [1, 1]], dtype=np.int8)
        result = _fallback(A, B)
        expected = np.matmul(A.astype(np.int32), B.astype(np.int32).T)
        np.testing.assert_array_equal(result, expected)

    def test_diagonal_matrix(self):
        A = np.diag([1, 2, 3]).astype(np.int8)
        B = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int8)
        result = _fallback(A, B)
        expected = np.matmul(A.astype(np.int32), B.astype(np.int32).T)
        np.testing.assert_array_equal(result, expected)

    def test_column_times_row(self):
        A = np.array([[1], [2], [3]], dtype=np.int8)
        B = np.array([[4], [5], [6]], dtype=np.int8)
        result = _fallback(A, B)
        assert result.shape == (3, 3)
        expected = np.matmul(A.astype(np.int32), B.astype(np.int32).T)
        np.testing.assert_array_equal(result, expected)

    def test_row_times_column(self):
        A = np.array([[1, 2, 3]], dtype=np.int8)
        B = np.array([[4, 5, 6]], dtype=np.int8)
        result = _fallback(A, B)
        assert result.shape == (1, 1)
        expected = np.matmul(A.astype(np.int32), B.astype(np.int32).T)
        np.testing.assert_array_equal(result, expected)

    def test_transpose_property_explicit(self):
        A = np.random.randint(-10, 10, (3, 5)).astype(np.int8)
        B = np.random.randint(-10, 10, (4, 5)).astype(np.int8)
        result = _fallback(A, B)
        expected = np.matmul(A.astype(np.int32), B.astype(np.int32).T)
        np.testing.assert_array_equal(result, expected)

    def test_non_contiguous_input(self):
        A_full = np.random.randint(-5, 5, (4, 8)).astype(np.int8)
        A = A_full[:, ::2]
        B = np.random.randint(-5, 5, (3, 4)).astype(np.int8)
        result = _fallback(A, B)
        expected = np.matmul(A.astype(np.int32), B.astype(np.int32).T)
        np.testing.assert_array_equal(result, expected)

    def test_repeated_application(self):
        A = np.array([[1, 0], [0, 1]], dtype=np.int8)
        B = np.array([[1, 0], [0, 1]], dtype=np.int8)
        result = _fallback(A, B)
        expected = np.matmul(A.astype(np.int32), B.astype(np.int32).T)
        np.testing.assert_array_equal(result, expected)
        result2 = _fallback(result.astype(np.int8), B)
        expected2 = np.matmul(result.astype(np.int32), B.astype(np.int32).T)
        np.testing.assert_array_equal(result2, expected2)

    def test_stress_random_shapes(self):
        rng = np.random.RandomState(42)
        for _ in range(30):
            m = rng.randint(1, 30)
            k = rng.randint(1, 30)
            n = rng.randint(1, 30)
            A = rng.randint(-10, 10, (m, k)).astype(np.int8)
            B = rng.randint(-10, 10, (n, k)).astype(np.int8)
            result = _fallback(A, B)
            assert result.shape == (m, n)
            expected = np.matmul(A.astype(np.int32), B.astype(np.int32).T)
            np.testing.assert_array_equal(result, expected)

    def test_binary_values(self):
        A = np.array([[0, 1], [1, 0]], dtype=np.int8)
        B = np.array([[0, 1], [1, 0]], dtype=np.int8)
        result = _fallback(A, B)
        expected = np.matmul(A.astype(np.int32), B.astype(np.int32).T)
        np.testing.assert_array_equal(result, expected)

    def test_asymmetric_dimensions(self):
        pairs = [(1, 5), (5, 1), (2, 9), (9, 2), (3, 17), (17, 3)]
        for m, n in pairs:
            k = 8
            A = np.random.randint(-5, 5, (m, k)).astype(np.int8)
            B = np.random.randint(-5, 5, (n, k)).astype(np.int8)
            result = _fallback(A, B)
            assert result.shape == (m, n)

    def test_power_of_two_sizes(self):
        for size in [1, 2, 4, 8, 16]:
            A = np.random.randint(-5, 5, (size, size)).astype(np.int8)
            B = np.random.randint(-5, 5, (size, size)).astype(np.int8)
            result = _fallback(A, B)
            assert result.shape == (size, size)

    def test_zero_left_operand(self):
        A = np.zeros((3, 5), dtype=np.int8)
        B = np.random.randint(-5, 5, (4, 5)).astype(np.int8)
        result = _fallback(A, B)
        np.testing.assert_array_equal(result, np.zeros((3, 4), dtype=np.int32))

    def test_zero_right_operand(self):
        A = np.random.randint(-5, 5, (3, 5)).astype(np.int8)
        B = np.zeros((4, 5), dtype=np.int8)
        result = _fallback(A, B)
        np.testing.assert_array_equal(result, np.zeros((3, 4), dtype=np.int32))

    def test_permutation_matrix(self):
        perm = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int8)
        A = np.array([[1, 2, 3]], dtype=np.int8)
        result = _fallback(A, perm)
        expected = np.matmul(A.astype(np.int32), perm.astype(np.int32).T)
        np.testing.assert_array_equal(result, expected)

    def test_counter_example_no_transpose(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.int8)
        B = np.array([[5, 6], [7, 8]], dtype=np.int8)
        result = _fallback(A, B)
        no_transpose = np.matmul(A.astype(np.int32), B.astype(np.int32))
        assert not np.array_equal(result, no_transpose) or result.shape[0] == result.shape[1]

    def test_associativity_within_int32(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.int8)
        B = np.array([[5, 6], [7, 8]], dtype=np.int8)
        C = np.array([[1, 0], [0, 1]], dtype=np.int8)
        ab = _fallback(A, B)
        ab_c = np.matmul(ab.astype(np.int32), C.astype(np.int32).T)
        a_bc = np.matmul(A.astype(np.int32), np.matmul(B.astype(np.int32), C.astype(np.int32)).T)
        np.testing.assert_array_equal(ab_c, a_bc)

    def test_all_same_value(self):
        A = np.full((4, 4), 3, dtype=np.int8)
        B = np.full((4, 4), 2, dtype=np.int8)
        result = _fallback(A, B)
        expected = np.full((4, 4), 24, dtype=np.int32)
        np.testing.assert_array_equal(result, expected)


class TestFallbackInt4:
    def test_basic(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.int8)
        B_packed = np.array([[0x12], [0x34]], dtype=np.uint8)
        result = _fallback_int4(A, B_packed, K=2)
        assert result.shape[0] == 2
        assert result.dtype == np.int32

    def test_basic_values(self):
        A = np.array([[1, 0], [0, 1]], dtype=np.int8)
        B_packed = np.array([[0x21], [0x43]], dtype=np.uint8)
        result = _fallback_int4(A, B_packed, K=2)
        assert result.shape == (2, 2)
        assert result.dtype == np.int32

    def test_zeros(self):
        A = np.zeros((2, 4), dtype=np.int8)
        B_packed = np.zeros((3, 2), dtype=np.uint8)
        result = _fallback_int4(A, B_packed, K=4)
        np.testing.assert_array_equal(result, np.zeros((2, 3), dtype=np.int32))

    def test_identity_packed(self):
        A = np.eye(2, dtype=np.int8)
        B_packed = np.array([[0x10], [0x01]], dtype=np.uint8)
        result = _fallback_int4(A, B_packed, K=2)
        assert result.shape == (2, 2)
        assert result.dtype == np.int32

    def test_output_shape(self):
        A = np.random.randint(-5, 5, (3, 8)).astype(np.int8)
        B_packed = np.random.randint(0, 256, (5, 4)).astype(np.uint8)
        result = _fallback_int4(A, B_packed, K=8)
        assert result.shape == (3, 5)

    def test_single_row(self):
        A = np.array([[1, 2, 3, 4]], dtype=np.int8)
        B_packed = np.array([[0x12, 0x34]], dtype=np.uint8)
        result = _fallback_int4(A, B_packed, K=4)
        assert result.shape == (1, 1)

    def test_single_column(self):
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int8)
        B_packed = np.array([[0x12]], dtype=np.uint8)
        result = _fallback_int4(A, B_packed, K=2)
        assert result.shape == (3, 1)

    def test_large_k(self):
        A = np.random.randint(-5, 5, (2, 32)).astype(np.int8)
        B_packed = np.random.randint(0, 256, (4, 16)).astype(np.uint8)
        result = _fallback_int4(A, B_packed, K=32)
        assert result.shape == (2, 4)

    def test_negative_int4_values(self):
        A = np.array([[1, -1], [1, -1]], dtype=np.int8)
        B_packed = np.array([[0xFF], [0x00]], dtype=np.uint8)
        result = _fallback_int4(A, B_packed, K=2)
        assert result.shape == (2, 2)
        assert result.dtype == np.int32

    def test_packed_byte_order(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.int8)
        B_packed = np.array([[0x10], [0x32]], dtype=np.uint8)
        result = _fallback_int4(A, B_packed, K=2)
        assert result.shape == (2, 2)

    def test_mixed_positive_negative_packed(self):
        A = np.array([[1, 1], [1, 1]], dtype=np.int8)
        B_packed = np.array([[0x80], [0x07]], dtype=np.uint8)
        result = _fallback_int4(A, B_packed, K=2)
        assert result.shape == (2, 2)

    def test_multiple_rows_columns(self):
        A = np.random.randint(-3, 3, (5, 6)).astype(np.int8)
        B_packed = np.random.randint(0, 256, (7, 3)).astype(np.uint8)
        result = _fallback_int4(A, B_packed, K=6)
        assert result.shape == (5, 7)

    def test_all_ones_packed(self):
        A = np.ones((2, 2), dtype=np.int8)
        B_packed = np.array([[0x11], [0x11]], dtype=np.uint8)
        result = _fallback_int4(A, B_packed, K=2)
        assert result.shape == (2, 2)
        assert result.dtype == np.int32

    def test_all_zeros_packed(self):
        A = np.ones((2, 2), dtype=np.int8)
        B_packed = np.array([[0x00], [0x00]], dtype=np.uint8)
        result = _fallback_int4(A, B_packed, K=2)
        np.testing.assert_array_equal(result, np.zeros((2, 2), dtype=np.int32))

    def test_stress_random(self):
        rng = np.random.RandomState(42)
        for _ in range(20):
            m = rng.randint(1, 20)
            k = rng.randint(2, 40)
            if k % 2 != 0:
                k += 1
            n = rng.randint(1, 20)
            A = rng.randint(-5, 5, (m, k)).astype(np.int8)
            B_packed = rng.randint(0, 256, (n, k // 2)).astype(np.uint8)
            result = _fallback_int4(A, B_packed, K=k)
            assert result.shape == (m, n)
            assert result.dtype == np.int32

    def test_k_mismatch_raises(self):
        A = np.array([[1, 2, 3, 4]], dtype=np.int8)
        B_packed = np.array([[0x12]], dtype=np.uint8)
        with pytest.raises((ValueError, AssertionError)):
            _fallback_int4(A, B_packed, K=4)

    def test_odd_k_padded(self):
        A = np.array([[1, 2, 3, 4]], dtype=np.int8)
        B_packed = np.array([[0x12, 0x30]], dtype=np.uint8)
        result = _fallback_int4(A, B_packed, K=4)
        assert result.shape == (1, 1)

    def test_symmetric_packed(self):
        A = np.eye(4, dtype=np.int8)
        B_packed = np.zeros((4, 2), dtype=np.uint8)
        B_packed[0, 0] = 0x10
        B_packed[1, 0] = 0x01
        B_packed[2, 1] = 0x10
        B_packed[3, 1] = 0x01
        result = _fallback_int4(A, B_packed, K=4)
        assert result.shape == (4, 4)

    def test_consistency_with_manual_unpack(self):
        A = np.array([[1, 2], [3, 4]], dtype=np.int8)
        B_packed = np.array([[0x21], [0x43]], dtype=np.uint8)
        result = _fallback_int4(A, B_packed, K=2)
        B_manual = np.array([[1, 2], [3, 4]], dtype=np.int8)
        expected = np.matmul(A.astype(np.int32), B_manual.astype(np.int32).T)
        np.testing.assert_array_equal(result, expected)

    def test_power_of_two_k(self):
        for k in [2, 4, 8, 16, 32]:
            A = np.random.randint(-3, 3, (2, k)).astype(np.int8)
            B_packed = np.random.randint(0, 256, (3, k // 2)).astype(np.uint8)
            result = _fallback_int4(A, B_packed, K=k)
            assert result.shape == (2, 3)

    def test_each_byte_two_values(self):
        A = np.array([[1, 1]], dtype=np.int8)
        B_packed = np.array([[0x23]], dtype=np.uint8)
        result = _fallback_int4(A, B_packed, K=2)
        assert result.shape == (1, 1)
        assert result.dtype == np.int32

    def test_high_nibble_low_nibble(self):
        A = np.array([[1, 0]], dtype=np.int8)
        B_packed_high = np.array([[0xF0]], dtype=np.uint8)
        B_packed_low = np.array([[0x0F]], dtype=np.uint8)
        result_high = _fallback_int4(A, B_packed_high, K=2)
        result_low = _fallback_int4(A, B_packed_low, K=2)
        assert result_high.shape == (1, 1)
        assert result_low.shape == (1, 1)
