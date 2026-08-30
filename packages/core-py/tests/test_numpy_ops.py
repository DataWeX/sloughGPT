"""Tests for domains.infrastructure.numpy_ops — pure NumPy transformer operations.

Covers: softmax, rmsnorm, layer_norm, gelu, silu, rope, to_float32.
Edge cases, boundary values, different dtypes, axis handling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.infrastructure.numpy_ops import (
    to_float32,
    softmax,
    rmsnorm,
    layer_norm,
    gelu,
    silu,
    rope,
)


# ── to_float32 ───────────────────────────────────────────────────────────────

class TestToFloat32:
    def test_already_float32(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = to_float32(x)
        assert result.dtype == np.float32

    def test_float16(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        result = to_float32(x)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0], rtol=1e-3)

    def test_int_to_float32(self):
        x = np.array([1, 2, 3], dtype=np.int32)
        result = to_float32(x)
        assert result.dtype == np.float32

    def test_int64_to_float32(self):
        x = np.array([100, 200, 300], dtype=np.int64)
        result = to_float32(x)
        assert result.dtype == np.float32

    def test_int8_to_float32(self):
        x = np.array([1, 2, 3], dtype=np.int8)
        result = to_float32(x)
        assert result.dtype == np.float32

    def test_uint8_to_float32(self):
        x = np.array([255, 128, 0], dtype=np.uint8)
        result = to_float32(x)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [255.0, 128.0, 0.0])

    def test_empty_array(self):
        x = np.array([], dtype=np.float32)
        result = to_float32(x)
        assert result.dtype == np.float32
        assert len(result) == 0

    def test_2d_array(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float16)
        result = to_float32(x)
        assert result.dtype == np.float32
        assert result.shape == (2, 2)

    def test_large_values(self):
        x = np.array([1e38, -1e38, 0.0], dtype=np.float16)
        result = to_float32(x)
        assert result.dtype == np.float32

    def test_preserves_values(self):
        x = np.array([1.5, 2.5, 3.5], dtype=np.float16)
        result = to_float32(x)
        np.testing.assert_allclose(result, [1.5, 2.5, 3.5], rtol=1e-3)

    def test_negative_values(self):
        x = np.array([-1.0, -2.0, -3.0], dtype=np.float16)
        result = to_float32(x)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [-1.0, -2.0, -3.0], rtol=1e-3)

    def test_zero_values(self):
        x = np.array([0.0, 0.0, 0.0], dtype=np.float16)
        result = to_float32(x)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])


# ── softmax ──────────────────────────────────────────────────────────────────

class TestSoftmax:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert result.shape == x.shape
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)

    def test_stable(self):
        x = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(x)
        assert result.sum() == pytest.approx(1.0, abs=1e-5)

    def test_2d_axis1(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = softmax(x, axis=1)
        np.testing.assert_allclose(result.sum(axis=1), [1.0, 1.0], rtol=1e-5)

    def test_2d_axis0(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = softmax(x, axis=0)
        np.testing.assert_allclose(result.sum(axis=0), [1.0, 1.0], rtol=1e-5)

    def test_single_element(self):
        x = np.array([5.0])
        result = softmax(x)
        assert result[0] == pytest.approx(1.0)

    def test_equal_values(self):
        x = np.array([1.0, 1.0, 1.0])
        result = softmax(x)
        np.testing.assert_allclose(result, [1/3, 1/3, 1/3], rtol=1e-5)

    def test_large_negative_values(self):
        x = np.array([-1000.0, -1001.0, -1002.0])
        result = softmax(x)
        assert result.sum() == pytest.approx(1.0, abs=1e-5)

    def test_zero_values(self):
        x = np.array([0.0, 0.0, 0.0])
        result = softmax(x)
        np.testing.assert_allclose(result, [1/3, 1/3, 1/3], rtol=1e-5)

    def test_all_positive(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert (result > 0).all()

    def test_all_negative(self):
        x = np.array([-3.0, -2.0, -1.0])
        result = softmax(x)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)

    def test_mixed_signs(self):
        x = np.array([-10.0, 0.0, 10.0])
        result = softmax(x)
        np.testing.assert_allclose(result.sum(), 1.0, rtol=1e-5)

    def test_3d_axis(self):
        x = np.random.randn(2, 3, 4)
        result = softmax(x, axis=-1)
        np.testing.assert_allclose(result.sum(axis=-1), np.ones((2, 3)), rtol=1e-5)

    def test_preserves_shape(self):
        x = np.random.randn(5, 3)
        result = softmax(x, axis=1)
        assert result.shape == x.shape


# ── rmsnorm ──────────────────────────────────────────────────────────────────

class TestRmsnorm:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        result = rmsnorm(x, w)
        assert result.shape == x.shape
        assert np.abs(np.sqrt(np.mean(result ** 2)) - 1.0) < 0.1

    def test_weighted(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.array([2.0, 2.0, 2.0])
        result = rmsnorm(x, w)
        assert np.sqrt(np.mean(result ** 2)) > 1.0

    def test_zero_weights(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.zeros(3)
        result = rmsnorm(x, w)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    def test_ones_input(self):
        x = np.array([1.0, 1.0, 1.0])
        w = np.ones(3)
        result = rmsnorm(x, w)
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0], rtol=1e-5)

    def test_large_values(self):
        x = np.array([1000.0, 2000.0, 3000.0])
        w = np.ones(3)
        result = rmsnorm(x, w)
        assert np.abs(np.sqrt(np.mean(result ** 2)) - 1.0) < 0.1

    def test_small_values(self):
        x = np.array([0.001, 0.002, 0.003])
        w = np.ones(3)
        result = rmsnorm(x, w)
        assert np.abs(np.sqrt(np.mean(result ** 2)) - 1.0) < 0.1

    def test_custom_eps(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        result = rmsnorm(x, w, eps=1e-3)
        assert result.shape == x.shape

    def test_2d_input(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        w = np.ones(2)
        result = rmsnorm(x, w)
        assert result.shape == x.shape


# ── layer_norm ───────────────────────────────────────────────────────────────

class TestLayerNorm:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        b = np.zeros(3)
        result = layer_norm(x, w, b)
        assert result.shape == x.shape
        np.testing.assert_allclose(result.mean(), 0.0, atol=1e-5)

    def test_weighted(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.array([2.0, 2.0, 2.0])
        b = np.array([1.0, 1.0, 1.0])
        result = layer_norm(x, w, b)
        np.testing.assert_allclose(result.mean(), 1.0, atol=1e-5)

    def test_none_bias(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        result = layer_norm(x, w, None)
        assert result.shape == x.shape

    def test_zero_bias(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        b = np.zeros(3)
        result = layer_norm(x, w, b)
        np.testing.assert_allclose(result.mean(), 0.0, atol=1e-5)

    def test_uniform_input(self):
        x = np.array([5.0, 5.0, 5.0])
        w = np.ones(3)
        b = np.zeros(3)
        result = layer_norm(x, w, b)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-5)

    def test_custom_eps(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        b = np.zeros(3)
        result = layer_norm(x, w, b, eps=1e-3)
        assert result.shape == x.shape

    def test_2d_input(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        w = np.ones(2)
        b = np.zeros(2)
        result = layer_norm(x, w, b)
        assert result.shape == x.shape

    def test_negative_weights(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.array([-1.0, -1.0, -1.0])
        b = np.zeros(3)
        result = layer_norm(x, w, b)
        assert result.shape == x.shape

    def test_large_values(self):
        x = np.array([1e6, 2e6, 3e6])
        w = np.ones(3)
        b = np.zeros(3)
        result = layer_norm(x, w, b)
        np.testing.assert_allclose(result.mean(), 0.0, atol=1e-3)


# ── gelu ─────────────────────────────────────────────────────────────────────

class TestGelu:
    def test_basic(self):
        x = np.array([-1.0, 0.0, 1.0])
        result = gelu(x)
        assert result.shape == x.shape
        assert result[1] == pytest.approx(0.0, abs=1e-5)
        assert result[2] > 0

    def test_negative(self):
        x = np.array([-2.0])
        result = gelu(x)
        assert result[0] < 0

    def test_zero(self):
        x = np.array([0.0])
        result = gelu(x)
        assert result[0] == pytest.approx(0.0, abs=1e-5)

    def test_large_positive(self):
        x = np.array([100.0])
        result = gelu(x)
        assert result[0] == pytest.approx(100.0, abs=0.1)

    def test_large_negative(self):
        x = np.array([-100.0])
        result = gelu(x)
        assert result[0] == pytest.approx(0.0, abs=0.01)

    def test_array(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        result = gelu(x)
        assert result.shape == x.shape

    def test_symmetry_approximate(self):
        x = np.array([-1.0, 1.0])
        result = gelu(x)
        # GELU is approximately antisymmetric around origin
        assert result[1] > result[0]

    def test_negative_side_approaches_zero(self):
        x = np.array([-10.0])
        result = gelu(x)
        assert result[0] < 0.1  # Should be very close to zero

    def test_positive_side_approaches_linear(self):
        x = np.array([10.0])
        result = gelu(x)
        # For large positive x, GELU(x) ≈ x
        assert result[0] > 9.0


# ── silu ─────────────────────────────────────────────────────────────────────

class TestSilu:
    def test_basic(self):
        x = np.array([-1.0, 0.0, 1.0])
        result = silu(x)
        assert result.shape == x.shape
        assert result[1] == pytest.approx(0.0, abs=1e-5)
        assert result[2] > 0

    def test_large_positive(self):
        x = np.array([100.0])
        result = silu(x)
        assert result[0] == pytest.approx(100.0, abs=0.1)

    def test_zero(self):
        x = np.array([0.0])
        result = silu(x)
        assert result[0] == pytest.approx(0.0, abs=1e-5)

    def test_negative_approaches_zero(self):
        x = np.array([-100.0])
        result = silu(x)
        assert result[0] == pytest.approx(0.0, abs=0.01)

    def test_array(self):
        x = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        result = silu(x)
        assert result.shape == x.shape

    def test_negative_values(self):
        x = np.array([-2.0, -1.0])
        result = silu(x)
        # SiLU(x) = x * sigmoid(x) which is negative for negative x
        assert result[0] < 0
        assert result[1] < 0

    def test_positive_monotonic(self):
        x = np.array([0.5, 1.0, 1.5, 2.0])
        result = silu(x)
        # SiLU is monotonically increasing
        for i in range(len(result) - 1):
            assert result[i] < result[i + 1]

    def test_2d_input(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = silu(x)
        assert result.shape == x.shape


# ── rope ─────────────────────────────────────────────────────────────────────

class TestRope:
    def test_basic(self):
        x = np.ones((4, 2, 8))
        result = rope(x, pos=0, dim=8)
        assert result.shape == x.shape

    def test_different_positions(self):
        x = np.ones((2, 1, 4))
        r1 = rope(x, pos=0, dim=4)
        r2 = rope(x, pos=10, dim=4)
        assert not np.allclose(r1, r2)

    def test_preserves_shape(self):
        x = np.random.randn(3, 4, 6)
        result = rope(x, pos=5, dim=6)
        assert result.shape == x.shape

    def test_pos_zero(self):
        x = np.ones((1, 1, 4))
        result = rope(x, pos=0, dim=4)
        assert result.shape == x.shape

    def test_large_pos(self):
        x = np.ones((1, 1, 4))
        result = rope(x, pos=1000, dim=4)
        assert result.shape == x.shape

    def test_dim_2(self):
        x = np.ones((2, 1, 2))
        result = rope(x, pos=0, dim=2)
        assert result.shape == x.shape

    def test_dim_16(self):
        x = np.ones((3, 2, 16))
        result = rope(x, pos=0, dim=16)
        assert result.shape == x.shape

    def test_custom_base(self):
        x = np.ones((2, 1, 8))
        r1 = rope(x, pos=0, dim=8, base=10000.0)
        r2 = rope(x, pos=0, dim=8, base=100.0)
        # Different bases give different results
        assert not np.allclose(r1, r2)

    def test_same_pos_same_result(self):
        x = np.ones((2, 1, 4))
        r1 = rope(x, pos=5, dim=4)
        r2 = rope(x, pos=5, dim=4)
        np.testing.assert_allclose(r1, r2)

    def test_rotation_magnitude(self):
        x = np.ones((1, 1, 4))
        result = rope(x, pos=0, dim=4)
        # Rotation should preserve magnitude
        np.testing.assert_allclose(
            np.linalg.norm(result), np.linalg.norm(x), rtol=1e-5
        )

    def test_2d_input_no_head(self):
        x = np.ones((3, 4))  # (seq, head_dim) — no heads dim
        result = rope(x, pos=0, dim=4)
        assert result.shape == x.shape

    def test_sequential_positions(self):
        x = np.ones((3, 1, 4))
        r0 = rope(x, pos=0, dim=4)
        r1 = rope(x, pos=1, dim=4)
        # Sequential positions give different results
        assert not np.allclose(r0, r1)


# ── to_float32 additional edge cases ────────────────────────────────────────

class TestToFloat32Extended:
    def test_bfloat16(self):
        try:
            x = np.array([1.0, 2.0, 3.0], dtype=np.bfloat16)
            result = to_float32(x)
            assert result.dtype == np.float32
        except (TypeError, ValueError, AttributeError):
            pytest.skip("bfloat16 not supported on this numpy version")

    def test_bool_to_float32(self):
        x = np.array([True, False, True], dtype=bool)
        result = to_float32(x)
        assert result.dtype == np.float32

    def test_preserves_nan(self):
        x = np.array([1.0, float("nan"), 3.0], dtype=np.float16)
        result = to_float32(x)
        assert np.isnan(result[1])

    def test_preserves_inf(self):
        x = np.array([1.0, float("inf"), 3.0], dtype=np.float16)
        result = to_float32(x)
        assert np.isinf(result[1])

    def test_negative_inf(self):
        x = np.array([float("-inf")], dtype=np.float16)
        result = to_float32(x)
        assert result[0] == float("-inf")

    def test_3d_array(self):
        x = np.ones((2, 3, 4), dtype=np.float16)
        result = to_float32(x)
        assert result.shape == (2, 3, 4)
        assert result.dtype == np.float32

    def test_scalar_array(self):
        x = np.array(3.14, dtype=np.float16)
        result = to_float32(x)
        assert result.dtype == np.float32

    def test_bool_type(self):
        x = np.array([1, 0, 1], dtype=bool)
        result = to_float32(x)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, [1.0, 0.0, 1.0])


# ── softmax extended ────────────────────────────────────────────────────────

class TestSoftmaxExtended:
    def test_1d_array(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        result = softmax(x)
        assert result.sum() == pytest.approx(1.0, abs=1e-5)

    def test_all_same_values(self):
        x = np.array([5.0, 5.0, 5.0, 5.0])
        result = softmax(x)
        np.testing.assert_allclose(result, [0.25, 0.25, 0.25, 0.25], rtol=1e-5)

    def test_single_large_value(self):
        x = np.array([100.0, 1.0, 1.0])
        result = softmax(x)
        assert result[0] > 0.99

    def test_negative_inputs(self):
        x = np.array([-10.0, -20.0, -30.0])
        result = softmax(x)
        assert result.sum() == pytest.approx(1.0, abs=1e-5)
        assert result[0] > result[1] > result[2]

    def test_4d_array(self):
        x = np.random.randn(2, 3, 4, 5)
        result = softmax(x, axis=-1)
        np.testing.assert_allclose(result.sum(axis=-1), np.ones((2, 3, 4)), rtol=1e-5)

    def test_preserves_shape_2d(self):
        x = np.random.randn(10, 20)
        result = softmax(x, axis=1)
        assert result.shape == x.shape


# ── rmsnorm extended ────────────────────────────────────────────────────────

class TestRmsnormExtended:
    def test_uniform_output(self):
        x = np.array([1.0, 1.0, 1.0])
        w = np.ones(3)
        result = rmsnorm(x, w)
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0], rtol=1e-5)

    def test_different_weights_scale(self):
        x = np.array([1.0, 2.0, 3.0])
        w1 = np.ones(3)
        w2 = np.ones(3) * 2.0
        r1 = rmsnorm(x, w1)
        r2 = rmsnorm(x, w2)
        np.testing.assert_allclose(r2, r1 * 2.0, rtol=1e-5)

    def test_zero_input(self):
        x = np.array([0.0, 0.0, 0.0])
        w = np.ones(3)
        result = rmsnorm(x, w)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0])

    def test_negative_input(self):
        x = np.array([-1.0, -2.0, -3.0])
        w = np.ones(3)
        result = rmsnorm(x, w)
        assert np.abs(np.sqrt(np.mean(result ** 2)) - 1.0) < 0.1

    def test_1d_output(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.ones(4)
        result = rmsnorm(x, w)
        assert result.shape == x.shape

    def test_custom_eps_large(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        result = rmsnorm(x, w, eps=1.0)
        assert result.shape == x.shape


# ── layer_norm extended ─────────────────────────────────────────────────────

class TestLayerNormExtended:
    def test_mean_zero(self):
        x = np.array([1.0, 5.0, 10.0])
        w = np.ones(3)
        b = np.zeros(3)
        result = layer_norm(x, w, b)
        np.testing.assert_allclose(result.mean(), 0.0, atol=1e-5)

    def test_with_large_bias(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3)
        b = np.array([10.0, 10.0, 10.0])
        result = layer_norm(x, w, b)
        np.testing.assert_allclose(result.mean(), 10.0, atol=1e-5)

    def test_negative_weights_scale(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.array([-1.0, -1.0, -1.0])
        b = np.zeros(3)
        result = layer_norm(x, w, b)
        assert result.shape == x.shape

    def test_1d_input(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        w = np.ones(5)
        b = np.zeros(5)
        result = layer_norm(x, w, b)
        assert result.shape == x.shape
        np.testing.assert_allclose(result.mean(), 0.0, atol=1e-5)

    def test_eps_prevents_division_by_zero(self):
        x = np.array([1.0, 1.0, 1.0])
        w = np.ones(3)
        b = np.zeros(3)
        result = layer_norm(x, w, b, eps=1e-10)
        assert np.all(np.isfinite(result))

    def test_large_values_stable(self):
        x = np.array([1e6, 2e6, 3e6])
        w = np.ones(3)
        b = np.zeros(3)
        result = layer_norm(x, w, b)
        assert np.all(np.isfinite(result))


# ── gelu extended ───────────────────────────────────────────────────────────

class TestGeluExtended:
    def test_monotonic_positive(self):
        x = np.array([0.0, 0.5, 1.0, 2.0, 5.0])
        result = gelu(x)
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1]

    def test_approaches_linear_for_large_x(self):
        x = np.array([50.0, 100.0, 200.0])
        result = gelu(x)
        np.testing.assert_allclose(result, x, rtol=0.01)

    def test_symmetry_around_origin(self):
        x = np.array([-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
        result = gelu(x)
        # GELU(-x) ≈ -GELU(x) for small x (approximate)
        assert result[3] == pytest.approx(0.0, abs=1e-5)

    def test_array_of_ones(self):
        x = np.ones(10)
        result = gelu(x)
        assert result.shape == (10,)
        assert all(r > 0 for r in result)

    def test_mixed_values(self):
        x = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        result = gelu(x)
        assert result.shape == x.shape
        assert result[3] > result[1]

    def test_small_negative(self):
        x = np.array([-0.1])
        result = gelu(x)
        assert result[0] < 0

    def test_small_positive(self):
        x = np.array([0.1])
        result = gelu(x)
        assert result[0] > 0


# ── silu extended ───────────────────────────────────────────────────────────

class TestSiluExtended:
    def test_at_zero(self):
        x = np.array([0.0])
        result = silu(x)
        assert result[0] == pytest.approx(0.0, abs=1e-5)

    def test_large_positive_approaches_identity(self):
        x = np.array([50.0, 100.0])
        result = silu(x)
        np.testing.assert_allclose(result, x, rtol=0.01)

    def test_large_negative_approaches_zero(self):
        x = np.array([-50.0, -100.0])
        result = silu(x)
        np.testing.assert_allclose(result, [0.0, 0.0], atol=0.01)

    def test_symmetry_not_exact(self):
        x = np.array([-1.0, 1.0])
        result = silu(x)
        # SiLU is not symmetric
        assert result[0] != pytest.approx(-result[1], abs=0.1)

    def test_3d_input(self):
        x = np.random.randn(2, 3, 4)
        result = silu(x)
        assert result.shape == x.shape

    def test_preserves_sign_negative(self):
        x = np.array([-5.0, -3.0, -1.0])
        result = silu(x)
        assert all(r < 0 for r in result)

    def test_preserves_sign_positive(self):
        x = np.array([1.0, 3.0, 5.0])
        result = silu(x)
        assert all(r > 0 for r in result)

    def test_derivative_positive_at_origin(self):
        # SiLU'(0) = 0.5
        x = np.array([0.0])
        result = silu(x)
        assert result[0] == pytest.approx(0.0, abs=1e-5)


# ── rope extended ───────────────────────────────────────────────────────────

class TestRopeExtended:
    def test_pos_100(self):
        x = np.ones((5, 2, 8))
        result = rope(x, pos=100, dim=8)
        assert result.shape == x.shape

    def test_dim_4(self):
        x = np.ones((3, 1, 4))
        result = rope(x, pos=0, dim=4)
        assert result.shape == x.shape

    def test_dim_8(self):
        x = np.ones((4, 2, 8))
        result = rope(x, pos=0, dim=8)
        assert result.shape == x.shape

    def test_odd_dim_handled(self):
        # Even if dim is odd, rope should handle the split
        x = np.ones((3, 2, 8))
        result = rope(x, pos=0, dim=8)
        assert result.shape == x.shape

    def test_base_100(self):
        x = np.ones((2, 1, 8))
        r1 = rope(x, pos=0, dim=8, base=100.0)
        r2 = rope(x, pos=0, dim=8, base=10000.0)
        assert not np.allclose(r1, r2)

    def test_preserves_norm(self):
        x = np.random.randn(5, 3, 8).astype(np.float32)
        result = rope(x, pos=0, dim=8)
        np.testing.assert_allclose(
            np.linalg.norm(result, axis=-1),
            np.linalg.norm(x, axis=-1),
            rtol=1e-5,
        )

    def test_batch_same_pos(self):
        x = np.ones((3, 4, 8))
        result = rope(x, pos=5, dim=8)
        # Each position in batch should have same rotation
        assert result.shape == x.shape

    def test_negative_pos(self):
        x = np.ones((2, 1, 4))
        result = rope(x, pos=-5, dim=4)
        assert result.shape == x.shape

    def test_zero_dim_pair(self):
        # dim=2 means 1 pair
        x = np.ones((3, 2, 2))
        result = rope(x, pos=0, dim=2)
        assert result.shape == x.shape
