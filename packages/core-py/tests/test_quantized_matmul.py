"""
Tests for quantized matmul kernels (int8 GEMM).

Tests:
  - quantize_activation: float32 -> int8
  - int8_matmul: int8 x int8 -> float32
  - quantized_linear: full quantized linear layer
  - Numerical accuracy vs float32 matmul
  - Quantine engine, TensorInfo, QuantMeta
  - int4 packing/unpacking
  - _cosine_similarity
  - quantize_state_dict
  - quantize_kv_tensor / dequantize_kv_tensor
  - QuantizedLinear wrapper
"""

import time
import json

import numpy as np
import pytest

from domains.infrastructure.quantization import (
    Quantine,
    TensorInfo,
    QuantMeta,
    QuantMode,
    QuantDtype,
    quantize_activation,
    int8_matmul,
    quantized_linear,
    quantize_kv_tensor,
    dequantize_kv_tensor,
    quantize_state_dict,
    _pack_int4,
    _unpack_int4,
    _dequantize,
    _cosine_similarity,
)


# ── quantize_activation ────────────────────────────────────────────────


class TestQuantizeActivation:
    """Test float32 -> int8 activation quantization."""

    def test_basic(self):
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        q = quantize_activation(x, scale=1.0)
        np.testing.assert_array_equal(q, [-1, 0, 1])

    def test_with_scale(self):
        x = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
        q = quantize_activation(x, scale=2.0 / 127)
        assert q[0] == -127
        assert q[1] == 0
        assert q[2] == 127

    def test_clipping(self):
        x = np.array([-1000.0, 0.0, 1000.0], dtype=np.float32)
        q = quantize_activation(x, scale=1.0)
        assert q[0] == -128
        assert q[2] == 127

    def test_preserves_shape(self):
        x = np.random.randn(3, 4, 5).astype(np.float32)
        q = quantize_activation(x, scale=0.1)
        assert q.shape == x.shape
        assert q.dtype == np.int8

    def test_zero_point_nonzero(self):
        x = np.array([0.0, 1.0, 2.0], dtype=np.float32)
        q = quantize_activation(x, scale=1.0, zero_point=10)
        assert q[0] == 10
        assert q[1] == 11
        assert q[2] == 12

    def test_symmetric_zero_point(self):
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        q = quantize_activation(x, scale=1.0, zero_point=0)
        np.testing.assert_array_equal(q, [-1, 0, 1])

    def test_large_scale(self):
        x = np.array([100.0, 200.0, 300.0], dtype=np.float32)
        q = quantize_activation(x, scale=10.0)
        assert q[0] == 10
        assert q[1] == 20
        assert q[2] == 30

    def test_negative_values(self):
        x = np.array([-5.0, -3.0, -1.0], dtype=np.float32)
        q = quantize_activation(x, scale=1.0)
        assert q[0] == -5
        assert q[1] == -3
        assert q[2] == -1

    def test_2d_array(self):
        x = np.random.randn(4, 8).astype(np.float32)
        q = quantize_activation(x, scale=0.1)
        assert q.shape == (4, 8)

    def test_single_element(self):
        x = np.array([3.0], dtype=np.float32)
        q = quantize_activation(x, scale=1.0)
        assert q[0] == 3

    def test_all_zeros(self):
        x = np.zeros(10, dtype=np.float32)
        q = quantize_activation(x, scale=1.0)
        np.testing.assert_array_equal(q, np.zeros(10, dtype=np.int8))


# ── int8_matmul ────────────────────────────────────────────────────────


class TestInt8Matmul:
    """Test int8 x int8 -> float32 matmul."""

    def test_identity(self):
        a = np.array([[1, 2, 3]], dtype=np.int8)
        b = np.eye(3, dtype=np.int8)
        result = int8_matmul(a, b, a_scale=1.0, b_scale=1.0)
        np.testing.assert_array_equal(result, [[1, 2, 3]])

    def test_symmetric_basic(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int8)
        b = np.array([[5, 7], [6, 8]], dtype=np.int8)
        result = int8_matmul(a, b, a_scale=1.0, b_scale=1.0)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_with_scales(self):
        a = np.array([[1, 2]], dtype=np.int8)
        b = np.array([[3, 4]], dtype=np.int8)
        result = int8_matmul(a, b, a_scale=0.1, b_scale=0.2)
        expected = np.array([[0.22]], dtype=np.float32)
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_matches_float32(self):
        rng = np.random.RandomState(42)
        a_fp = rng.randn(8, 16).astype(np.float32) * 0.5
        b_fp = rng.randn(4, 16).astype(np.float32) * 0.5
        a_scale = np.max(np.abs(a_fp)) / 127
        b_scale = np.max(np.abs(b_fp)) / 127
        a_int = np.clip(np.round(a_fp / a_scale), -128, 127).astype(np.int8)
        b_int = np.clip(np.round(b_fp / b_scale), -128, 127).astype(np.int8)
        y_fp32 = a_fp @ b_fp.T
        y_int8 = int8_matmul(a_int, b_int, a_scale=a_scale, b_scale=b_scale)
        cosine = np.dot(y_fp32.flatten(), y_int8.flatten()) / (
            np.linalg.norm(y_fp32) * np.linalg.norm(y_int8)
        )
        assert cosine > 0.99, f"Cosine similarity too low: {cosine}"

    def test_zeros_input(self):
        a = np.zeros((2, 3), dtype=np.int8)
        b = np.eye(3, dtype=np.int8)
        result = int8_matmul(a, b, a_scale=1.0, b_scale=1.0)
        np.testing.assert_array_equal(result, np.zeros((2, 3)))

    def test_asymmetric_basic(self):
        """Asymmetric matmul with nonzero zero_points produces valid output."""
        a = np.array([[1, 2]], dtype=np.int8)
        b = np.array([[3, 4]], dtype=np.int8)
        result = int8_matmul(a, b, a_scale=0.1, b_scale=0.2,
                             a_zero_point=1, b_zero_point=1)
        # (1,2) @ (2,1).T -> (1,1)
        assert result.dtype == np.float32
        assert result.shape == (1, 1)

    def test_per_channel_b_scale(self):
        """b_scale as per-channel (N,) array."""
        a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
        b = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int8)
        b_scale = np.array([0.1, 0.2, 0.3])
        result = int8_matmul(a, b, a_scale=1.0, b_scale=b_scale)
        assert result.shape == (2, 3)

    def test_large_values(self):
        """Test with large int8 values (multi-element K dimension)."""
        a = np.array([[127, -128]], dtype=np.int8)
        b = np.array([[127, -128], [-128, 127]], dtype=np.int8)
        result = int8_matmul(a, b, a_scale=1.0, b_scale=1.0)
        assert result.shape == (1, 2)

    def test_ones_input(self):
        a = np.ones((1, 4), dtype=np.int8)
        b = np.ones((2, 4), dtype=np.int8)
        result = int8_matmul(a, b, a_scale=1.0, b_scale=1.0)
        assert result.shape == (1, 2)
        np.testing.assert_array_equal(result, [[4.0, 4.0]])

    def test_scale_product(self):
        """Verify scale product affects magnitude."""
        a = np.array([[1, 0]], dtype=np.int8)
        b = np.array([[1, 0]], dtype=np.int8)
        r1 = int8_matmul(a, b, a_scale=1.0, b_scale=1.0)
        r2 = int8_matmul(a, b, a_scale=2.0, b_scale=3.0)
        np.testing.assert_allclose(r2, r1 * 6.0)

    def test_multi_batch(self):
        a = np.random.randint(-10, 10, (4, 8)).astype(np.int8)
        b = np.random.randint(-10, 10, (3, 8)).astype(np.int8)
        result = int8_matmul(a, b, a_scale=0.1, b_scale=0.1)
        assert result.shape == (4, 3)

    def test_asymmetric_zero_point_effects(self):
        """Verify asymmetric zero points change the result."""
        a = np.array([[5, 5]], dtype=np.int8)
        b = np.array([[5, 5]], dtype=np.int8)
        r_sym = int8_matmul(a, b, a_scale=1.0, b_scale=1.0, a_zero_point=0, b_zero_point=0)
        r_asym = int8_matmul(a, b, a_scale=1.0, b_scale=1.0, a_zero_point=5, b_zero_point=5)
        # Results should differ due to zero point compensation
        assert not np.allclose(r_sym, r_asym)


# ── quantized_linear ──────────────────────────────────────────────────


class TestQuantizedLinear:
    """Test full quantized linear layer."""

    def test_basic(self):
        x = np.random.randn(1, 16).astype(np.float32)
        w = np.random.randn(8, 16).astype(np.float32) * 0.02
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test", w)
        y_fp32 = x @ w.T
        y_quant = quantized_linear(x, info.array, info.meta.scale)
        cosine = np.dot(y_fp32.flatten(), y_quant.flatten()) / (
            np.linalg.norm(y_fp32) * np.linalg.norm(y_quant)
        )
        assert cosine > 0.99

    def test_with_bias(self):
        x = np.random.randn(1, 16).astype(np.float32)
        w = np.random.randn(8, 16).astype(np.float32) * 0.02
        bias = np.random.randn(8).astype(np.float32) * 0.01
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test", w)
        y_fp32 = x @ w.T + bias
        y_quant = quantized_linear(x, info.array, info.meta.scale, bias=bias)
        cosine = np.dot(y_fp32.flatten(), y_quant.flatten()) / (
            np.linalg.norm(y_fp32) * np.linalg.norm(y_quant)
        )
        assert cosine > 0.99

    def test_batch(self):
        x = np.random.randn(4, 128, 768).astype(np.float32)
        w = np.random.randn(768, 768).astype(np.float32) * 0.02
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test", w)
        y_fp32 = x @ w.T
        y_quant = quantized_linear(x, info.array, info.meta.scale)
        assert y_quant.shape == y_fp32.shape
        cosine = np.dot(y_fp32.flatten(), y_quant.flatten()) / (
            np.linalg.norm(y_fp32) * np.linalg.norm(y_quant)
        )
        assert cosine > 0.99

    def test_correctness_at_scale(self):
        x = np.random.randn(1, 128, 768).astype(np.float32)
        w = np.random.randn(768, 768).astype(np.float32) * 0.02
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test", w)
        y_fp32 = x @ w.T
        y_quant = quantized_linear(x, info.array, info.meta.scale)
        cosine = np.dot(y_fp32.flatten(), y_quant.flatten()) / (
            np.linalg.norm(y_fp32) * np.linalg.norm(y_quant)
        )
        assert cosine > 0.95, f"Cosine similarity too low: {cosine}"
        assert y_quant.shape == y_fp32.shape

    def test_no_bias(self):
        x = np.random.randn(2, 8).astype(np.float32)
        w = np.random.randn(4, 8).astype(np.float32) * 0.05
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test", w)
        y = quantized_linear(x, info.array, info.meta.scale)
        assert y.shape == (2, 4)

    def test_1d_input(self):
        x = np.random.randn(16).astype(np.float32)
        w = np.random.randn(8, 16).astype(np.float32) * 0.02
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test", w)
        y = quantized_linear(x, info.array, info.meta.scale)
        assert y.shape == (8,)

    def test_per_channel_scale(self):
        x = np.random.randn(1, 16).astype(np.float32)
        w = np.random.randn(8, 16).astype(np.float32) * 0.02
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test", w)
        if info.meta.is_per_channel:
            y = quantized_linear(x, info.array, info.meta.scale)
            assert y.shape == (1, 8)


# ── Quantine engine ───────────────────────────────────────────────────


class TestQuantine:
    def test_init_symmetric(self):
        engine = Quantine(bits=8, mode="symmetric")
        assert engine._bits == 8
        assert engine._mode == QuantMode.SYMMETRIC

    def test_init_asymmetric(self):
        engine = Quantine(bits=8, mode="asymmetric")
        assert engine._mode == QuantMode.ASYMMETRIC

    def test_init_invalid_bits(self):
        with pytest.raises(ValueError, match="bits"):
            Quantine(bits=16)

    def test_should_skip(self):
        engine = Quantine(bits=8, mode="symmetric")
        assert engine.should_skip("tok_emb.weight")
        assert engine.should_skip("pos_emb.weight")
        assert engine.should_skip("norm.weight")
        assert not engine.should_skip("blocks.0.q_proj.weight")

    def test_is_sensitive(self):
        engine = Quantine(bits=8, mode="symmetric")
        assert engine.is_sensitive("attn_norm.weight")
        assert engine.is_sensitive("ff_norm.weight")
        assert not engine.is_sensitive("blocks.0.q_proj.weight")

    def test_quantize_1d_array(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(64).astype(np.float32) * 0.02
        info = engine.quantize("test", arr)
        assert info.is_quantized
        assert info.meta.bits == 8

    def test_quantize_2d_per_channel(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(32, 64).astype(np.float32) * 0.02
        info = engine.quantize("test", arr)
        assert info.is_quantized
        assert info.meta.is_per_channel

    def test_quantize_skips_prefixed(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(8).astype(np.float32)
        info = engine.quantize("tok_emb.weight", arr)
        assert not info.is_quantized
        assert np.array_equal(info.array, arr)

    def test_error_report(self):
        engine = Quantine(bits=8, mode="symmetric")
        engine.quantize("a", np.random.randn(8).astype(np.float32))
        engine.quantize("b", np.random.randn(16).astype(np.float32))
        report = engine.error_report()
        assert "a" in report
        assert "b" in report

    def test_summary(self):
        engine = Quantine(bits=8, mode="symmetric")
        engine.quantize("a", np.random.randn(8).astype(np.float32))
        s = engine.summary()
        assert s["tensors"] == 1
        assert s["bits"] == 8
        assert s["mode"] == "symmetric"

    def test_summary_empty(self):
        engine = Quantine(bits=8, mode="symmetric")
        s = engine.summary()
        assert s["tensors"] == 0

    def test_quantize_with_scale(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(8).astype(np.float32)
        info = engine.quantize_with_scale("test", arr, scale=0.01, zero_point=0)
        assert info.is_quantized
        assert info.meta.scale == pytest.approx(0.01)

    def test_save_load_metadata(self, tmp_path):
        engine = Quantine(bits=8, mode="symmetric")
        engine.quantize("a", np.random.randn(8).astype(np.float32))
        path = str(tmp_path / "meta.json")
        engine.save_metadata(path)
        engine2 = Quantine(bits=8, mode="symmetric")
        engine2.load_metadata(path)
        assert "a" in engine2._error_report

    def test_save_load_weights(self, tmp_path):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(8).astype(np.float32) * 0.02
        info = engine.quantize("test", arr)
        path = str(tmp_path / "weights.npz")
        engine.save_weights(path, {"test": info})
        loaded = engine.load_weights(path)
        assert "test" in loaded
        assert loaded["test"].is_quantized

    def test_quantize_4bit(self):
        engine = Quantine(bits=4, mode="symmetric")
        arr = np.random.randn(16).astype(np.float32) * 0.02
        info = engine.quantize("test", arr)
        assert info.is_quantized
        assert info.meta.bits == 4

    def test_should_not_skip_normal(self):
        engine = Quantine(bits=8, mode="symmetric")
        assert not engine.should_skip("blocks.0.attn.W_q.weight")

    def test_should_not_skip_ff(self):
        engine = Quantine(bits=8, mode="symmetric")
        assert not engine.should_skip("blocks.0.ff.w1.weight")

    def test_quantize_with_scale_skip_prefix(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(8).astype(np.float32)
        info = engine.quantize_with_scale("tok_emb.weight", arr, scale=0.01)
        assert not info.is_quantized

    def test_dequantize_to_float(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(16).astype(np.float32) * 0.02
        info = engine.quantize("test", arr)
        deq = engine.dequantize_to_float(info)
        cosine = _cosine_similarity(arr, deq)
        assert cosine > 0.95

    def test_quantize_2d_asymmetric(self):
        engine = Quantine(bits=8, mode="asymmetric")
        arr = np.random.randn(8, 16).astype(np.float32) * 0.02
        info = engine.quantize("test", arr)
        assert info.is_quantized

    def test_summary_has_all_keys(self):
        engine = Quantine(bits=8, mode="symmetric")
        engine.quantize("a", np.random.randn(8).astype(np.float32))
        s = engine.summary()
        assert "tensors" in s
        assert "bits" in s
        assert "mode" in s
        assert "avg_mse" in s
        assert "avg_cosine_sim" in s
        assert "worst_tensor" in s

    def test_error_report_dict_format(self):
        engine = Quantine(bits=8, mode="symmetric")
        engine.quantize("a", np.random.randn(8).astype(np.float32))
        report = engine.error_report()
        assert isinstance(report["a"], dict)

    def test_quantize_preserves_shape(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(32, 64).astype(np.float32) * 0.02
        info = engine.quantize("test", arr)
        assert info.array.shape == arr.shape

    def test_clip_percentile(self):
        engine = Quantine(bits=8, mode="symmetric", clip_percentile=0.999)
        arr = np.random.randn(128).astype(np.float32) * 0.02
        arr[0] = 100.0  # outlier
        info = engine.quantize("test", arr)
        assert info.is_quantized


# ── TensorInfo ─────────────────────────────────────────────────────────


class TestTensorInfo:
    def test_not_quantized(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        info = TensorInfo(name="t", array=arr)
        assert not info.is_quantized
        assert info.shape == (2,)
        assert info.dtype == np.float32
        assert info.nbytes == arr.nbytes

    def test_as_float_not_quantized(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        info = TensorInfo(name="t", array=arr)
        result = info.as_float()
        np.testing.assert_array_equal(result, arr)

    def test_as_float_int8_not_quantized(self):
        arr = np.array([1, 2], dtype=np.int8)
        info = TensorInfo(name="t", array=arr)
        result = info.as_float()
        assert result.dtype == np.float32

    def test_quantized_properties(self):
        meta = QuantMeta(
            scale=0.01, zero_point=0, bits=8, mode="symmetric",
            dtype_code=5, original_shape=(8,), original_dtype="float32",
        )
        arr = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int8)
        info = TensorInfo(name="t", array=arr, meta=meta)
        assert info.is_quantized
        assert info.shape == (8,)
        assert info.dtype == np.float32

    def test_compression_ratio_not_quantized(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        info = TensorInfo(name="t", array=arr)
        assert info.compression_ratio() == 1.0

    def test_compression_ratio_quantized(self):
        meta = QuantMeta(
            scale=0.01, zero_point=0, bits=8, mode="symmetric",
            dtype_code=5, original_shape=(100,), original_dtype="float32",
        )
        arr = np.zeros(100, dtype=np.int8)
        info = TensorInfo(name="t", array=arr, meta=meta)
        ratio = info.compression_ratio()
        assert ratio == pytest.approx(4.0, rel=0.01)

    def test_quantized_bytes(self):
        meta = QuantMeta(
            scale=0.01, zero_point=0, bits=8, mode="symmetric",
            dtype_code=5, original_shape=(10,), original_dtype="float32",
        )
        arr = np.zeros(10, dtype=np.int8)
        info = TensorInfo(name="t", array=arr, meta=meta)
        assert info.quantized_bytes() == 10

    def test_name(self):
        arr = np.array([1.0], dtype=np.float32)
        info = TensorInfo(name="my_tensor", array=arr)
        assert info.name == "my_tensor"

    def test_nbytes(self):
        arr = np.random.randn(100).astype(np.float32)
        info = TensorInfo(name="t", array=arr)
        assert info.nbytes == 400

    def test_quantized_bytes_equals_array_nbytes(self):
        meta = QuantMeta(
            scale=0.01, zero_point=0, bits=8, mode="symmetric",
            dtype_code=5, original_shape=(10,), original_dtype="float32",
        )
        arr = np.zeros(10, dtype=np.int8)
        info = TensorInfo(name="t", array=arr, meta=meta)
        assert info.quantized_bytes() == arr.nbytes


# ── QuantMeta ──────────────────────────────────────────────────────────


class TestQuantMeta:
    def test_to_dict(self):
        meta = QuantMeta(
            scale=0.01, zero_point=5, bits=8, mode="symmetric",
            dtype_code=5, original_shape=(10,), original_dtype="float32",
            mse=0.001, max_abs_error=0.1, cosine_sim=0.99,
        )
        d = meta.to_dict()
        assert d["scale"] == 0.01
        assert d["zero_point"] == 5
        assert d["bits"] == 8
        assert d["mse"] == 0.001

    def test_from_dict(self):
        d = {
            "scale": 0.01, "zero_point": 0, "bits": 8, "mode": "symmetric",
            "dtype_code": 5, "original_shape": [10], "original_dtype": "float32",
        }
        meta = QuantMeta.from_dict(d)
        assert meta.scale == 0.01
        assert meta.original_shape == (10,)

    def test_from_dict_per_channel(self):
        d = {
            "scale": [0.1, 0.2, 0.3], "zero_point": 0, "bits": 8, "mode": "symmetric",
            "dtype_code": 5, "original_shape": [3, 10], "original_dtype": "float32",
        }
        meta = QuantMeta.from_dict(d)
        assert isinstance(meta.scale, np.ndarray)
        assert meta.is_per_channel

    def test_is_per_channel_float(self):
        meta = QuantMeta(
            scale=0.01, zero_point=0, bits=8, mode="symmetric",
            dtype_code=5, original_shape=(10,), original_dtype="float32",
        )
        assert not meta.is_per_channel

    def test_to_dict_roundtrip(self):
        meta = QuantMeta(
            scale=0.05, zero_point=3, bits=8, mode="asymmetric",
            dtype_code=5, original_shape=(16, 32), original_dtype="float32",
        )
        d = meta.to_dict()
        meta2 = QuantMeta.from_dict(d)
        assert meta2.scale == meta.scale
        assert meta2.zero_point == meta.zero_point
        assert meta2.original_shape == meta.original_shape

    def test_to_dict_with_per_channel_scale(self):
        meta = QuantMeta(
            scale=np.array([0.1, 0.2]), zero_point=0, bits=8, mode="symmetric",
            dtype_code=5, original_shape=(2, 10), original_dtype="float32",
        )
        d = meta.to_dict()
        assert isinstance(d["scale"], list)

    def test_from_dict_with_optional_fields(self):
        d = {
            "scale": 0.01, "zero_point": 0, "bits": 8, "mode": "symmetric",
            "dtype_code": 5, "original_shape": [10], "original_dtype": "float32",
            "mse": 0.001, "max_abs_error": 0.1, "cosine_sim": 0.99,
        }
        meta = QuantMeta.from_dict(d)
        assert meta.mse == 0.001
        assert meta.max_abs_error == 0.1
        assert meta.cosine_sim == 0.99

    def test_from_dict_without_optional_fields(self):
        d = {
            "scale": 0.01, "zero_point": 0, "bits": 8, "mode": "symmetric",
            "dtype_code": 5, "original_shape": [10], "original_dtype": "float32",
        }
        meta = QuantMeta.from_dict(d)
        assert meta.mse == 0.0
        assert meta.max_abs_error == 0.0
        assert meta.cosine_sim == 1.0

    def test_to_dict_has_all_keys(self):
        meta = QuantMeta(
            scale=0.01, zero_point=0, bits=8, mode="symmetric",
            dtype_code=5, original_shape=(10,), original_dtype="float32",
        )
        d = meta.to_dict()
        assert "scale" in d
        assert "zero_point" in d
        assert "bits" in d
        assert "mode" in d
        assert "dtype_code" in d
        assert "original_shape" in d
        assert "original_dtype" in d
        assert "mse" in d
        assert "max_abs_error" in d
        assert "cosine_sim" in d


# ── int4 packing / unpacking ──────────────────────────────────────────


class TestInt4Packing:
    def test_pack_int4_basic(self):
        arr = np.array([0, 1, 2, 3], dtype=np.int8)
        packed = _pack_int4(arr)
        assert packed.dtype == np.int8
        assert len(packed) == 2

    def test_pack_int4_odd_length(self):
        arr = np.array([0, 1, 2], dtype=np.int8)
        packed = _pack_int4(arr)
        assert len(packed) == 2  # padded to even

    def test_pack_int4_requires_1d(self):
        arr = np.array([[0, 1], [2, 3]], dtype=np.int8)
        with pytest.raises(AssertionError):
            _pack_int4(arr)

    def test_unpack_int4_basic(self):
        arr = np.array([0, 1, 2, 3], dtype=np.int8)
        packed = _pack_int4(arr)
        unpacked = _unpack_int4(packed, 4, signed=False)
        np.testing.assert_array_equal(unpacked, arr)

    def test_unpack_int4_signed(self):
        arr = np.array([-8, -1, 0, 7], dtype=np.int8)
        packed = _pack_int4(arr)
        unpacked = _unpack_int4(packed, 4, signed=True)
        np.testing.assert_array_equal(unpacked, arr)

    def test_pack_unpack_roundtrip(self):
        rng = np.random.RandomState(42)
        arr = rng.randint(-8, 8, size=16).astype(np.int8)
        packed = _pack_int4(arr)
        unpacked = _unpack_int4(packed, 16, signed=True)
        np.testing.assert_array_equal(unpacked, arr)

    def test_pack_unsigned(self):
        arr = np.array([0, 5, 10, 15], dtype=np.int8)
        packed = _pack_int4(arr)
        unpacked = _unpack_int4(packed, 4, signed=False)
        np.testing.assert_array_equal(unpacked, arr)

    def test_unpack_int4_requires_1d(self):
        packed = np.array([[1, 2]], dtype=np.int8)
        with pytest.raises(AssertionError):
            _unpack_int4(packed, 4)

    def test_pack_empty(self):
        arr = np.array([], dtype=np.int8)
        packed = _pack_int4(arr)
        assert len(packed) == 0

    def test_pack_single_element(self):
        arr = np.array([5], dtype=np.int8)
        packed = _pack_int4(arr)
        unpacked = _unpack_int4(packed, 1, signed=False)
        assert unpacked[0] == 5

    def test_pack_large_values(self):
        arr = np.array([-8, -8, 7, 7], dtype=np.int8)
        packed = _pack_int4(arr)
        unpacked = _unpack_int4(packed, 4, signed=True)
        np.testing.assert_array_equal(unpacked, arr)

    def test_roundtrip_various_lengths(self):
        for length in [2, 4, 6, 8, 10, 12, 16]:
            rng = np.random.RandomState(42)
            arr = rng.randint(-8, 8, size=length).astype(np.int8)
            packed = _pack_int4(arr)
            unpacked = _unpack_int4(packed, length, signed=True)
            np.testing.assert_array_equal(unpacked, arr)


# ── _dequantize ────────────────────────────────────────────────────────


class TestDequantize:
    def test_symmetric(self):
        quantized = np.array([0, 64, 127], dtype=np.int8)
        result = _dequantize(quantized, scale=0.01, zero_point=0, bits=8, original_shape=(3,))
        expected = np.array([0.0, 0.64, 1.27])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_asymmetric(self):
        quantized = np.array([-128, 0, 127], dtype=np.int8)
        result = _dequantize(quantized, scale=0.01, zero_point=0, bits=8, original_shape=(3,))
        assert result.shape == (3,)

    def test_int4_dequantize(self):
        arr = np.array([0, 1, 2, 3], dtype=np.int8)
        packed = _pack_int4(arr)
        result = _dequantize(packed, scale=1.0, zero_point=0, bits=4, original_shape=(4,))
        assert result.shape == (4,)

    def test_per_channel_scale(self):
        quantized = np.array([[10, 20], [30, 40]], dtype=np.int8)
        scale = np.array([0.1, 0.01])
        result = _dequantize(quantized, scale=scale, zero_point=0, bits=8, original_shape=(2, 2))
        assert result.shape == (2, 2)

    def test_asymmetric_nonzero_zero_point(self):
        quantized = np.array([0, 50, 127], dtype=np.int8)
        result = _dequantize(quantized, scale=0.01, zero_point=10, bits=8, original_shape=(3,))
        assert result.shape == (3,)
        # (0 - 10) * 0.01 = -0.1
        assert result[0] == pytest.approx(-0.1, abs=1e-6)

    def test_int4_per_channel(self):
        quantized = np.array([[10, 20], [30, 40]], dtype=np.int8)
        scale = np.array([0.1, 0.01])
        result = _dequantize(quantized, scale=scale, zero_point=0, bits=4, original_shape=(2, 2))
        assert result.shape == (2, 2)


# ── _cosine_similarity ────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(a, a) == pytest.approx(1.0)

    def test_opposite(self):
        a = np.array([1.0, 2.0])
        b = np.array([-1.0, -2.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_both_zero(self):
        a = np.zeros(3)
        b = np.zeros(3)
        assert _cosine_similarity(a, b) == 1.0

    def test_one_zero(self):
        a = np.array([1.0, 2.0])
        b = np.zeros(2)
        assert _cosine_similarity(a, b) == 0.0

    def test_similar_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])
        sim = _cosine_similarity(a, b)
        assert sim > 0.99

    def test_single_element(self):
        assert _cosine_similarity(np.array([5.0]), np.array([5.0])) == pytest.approx(1.0)
        assert _cosine_similarity(np.array([5.0]), np.array([-5.0])) == pytest.approx(-1.0)

    def test_large_vectors(self):
        rng = np.random.RandomState(42)
        a = rng.randn(1000).astype(np.float64)
        assert _cosine_similarity(a, a) == pytest.approx(1.0)

    def test_different_magnitude_same_direction(self):
        a = np.array([1.0, 2.0])
        b = np.array([10.0, 20.0])
        assert _cosine_similarity(a, b) == pytest.approx(1.0)


# ── quantize_state_dict ───────────────────────────────────────────────


class TestQuantizeStateDict:
    def test_basic(self):
        sd = {
            "a": np.random.randn(8).astype(np.float32),
            "b": np.random.randn(16, 32).astype(np.float32),
        }
        result = quantize_state_dict(sd, bits=8, mode="symmetric")
        assert "a" in result
        assert "b" in result
        assert result["a"].is_quantized

    def test_empty(self):
        result = quantize_state_dict({})
        assert result == {}

    def test_single_tensor(self):
        sd = {"w": np.random.randn(16).astype(np.float32) * 0.02}
        result = quantize_state_dict(sd, bits=8, mode="symmetric")
        assert len(result) == 1
        assert result["w"].is_quantized

    def test_4bit(self):
        sd = {"w": np.random.randn(16).astype(np.float32) * 0.02}
        result = quantize_state_dict(sd, bits=4, mode="symmetric")
        assert result["w"].is_quantized

    def test_skip_prefix(self):
        sd = {
            "tok_emb.weight": np.random.randn(8).astype(np.float32),
            "blocks.0.q_proj.weight": np.random.randn(16, 8).astype(np.float32),
        }
        result = quantize_state_dict(sd, bits=8, mode="symmetric")
        assert not result["tok_emb.weight"].is_quantized
        assert result["blocks.0.q_proj.weight"].is_quantized

    def test_returns_tensor_info(self):
        sd = {"w": np.random.randn(8).astype(np.float32) * 0.02}
        result = quantize_state_dict(sd, bits=8, mode="symmetric")
        assert isinstance(result["w"], TensorInfo)


# ── quantize_kv_tensor / dequantize_kv_tensor ────────────────────────


class TestKVQuantization:
    def test_quantize_dequantize_roundtrip(self):
        x = np.random.randn(1, 4, 8, 16).astype(np.float32)
        x_int8, scale = quantize_kv_tensor(x)
        assert x_int8.dtype == np.int8
        assert x_int8.shape == x.shape
        assert scale.shape == (1, 4, 8, 1)
        x_deq = dequantize_kv_tensor(x_int8, scale)
        assert x_deq.shape == x.shape
        # Should be close but not exact (int8 quantization error)
        cosine = np.dot(x.flatten(), x_deq.flatten()) / (
            np.linalg.norm(x) * np.linalg.norm(x_deq)
        )
        assert cosine > 0.99

    def test_zero_input(self):
        x = np.zeros((1, 2, 4, 8), dtype=np.float32)
        x_int8, scale = quantize_kv_tensor(x)
        x_deq = dequantize_kv_tensor(x_int8, scale)
        np.testing.assert_allclose(x_deq, 0.0, atol=1e-6)

    def test_scale_shape(self):
        x = np.random.randn(2, 3, 4, 8).astype(np.float32)
        _, scale = quantize_kv_tensor(x)
        assert scale.shape == (2, 3, 4, 1)

    def test_int8_range(self):
        x = np.random.randn(1, 2, 4, 16).astype(np.float32)
        x_int8, _ = quantize_kv_tensor(x)
        assert x_int8.min() >= -128
        assert x_int8.max() <= 127

    def test_large_values(self):
        x = np.array([[[[1000.0, -1000.0, 500.0, -500.0]]]]).astype(np.float32)
        x_int8, scale = quantize_kv_tensor(x)
        x_deq = dequantize_kv_tensor(x_int8, scale)
        cosine = np.dot(x.flatten(), x_deq.flatten()) / (
            np.linalg.norm(x) * np.linalg.norm(x_deq)
        )
        assert cosine > 0.99

    def test_ones_input(self):
        x = np.ones((1, 1, 1, 8), dtype=np.float32)
        x_int8, scale = quantize_kv_tensor(x)
        x_deq = dequantize_kv_tensor(x_int8, scale)
        np.testing.assert_allclose(x_deq, 1.0, atol=0.1)


# ── SloLinearQuantized (slonet integration) ──────────────────────────


class TestSloLinearQuantized:
    """Test that SloLinear uses int8 GEMM when quantized weight is set."""

    def test_quantized_forward_matches_float(self):
        from domains.training.slonet import SloLinear, Tensor
        layer = SloLinear(16, 8, bias=True)
        x = Tensor(np.random.randn(1, 16).astype(np.float32))
        y_float = layer.forward_numpy(x.data)
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize(layer.name, layer.weight.data.copy())
        assert info.is_quantized
        layer.set_quantized_weight(info)
        y_quant = layer.forward_numpy(x.data)
        cosine = np.dot(y_float.flatten(), y_quant.flatten()) / (
            np.linalg.norm(y_float) * np.linalg.norm(y_quant)
        )
        assert cosine > 0.95, f"Cosine similarity too low: {cosine}"

    def test_quantized_forward_uses_int8_matmul(self):
        from domains.training.slonet import SloLinear, Tensor
        layer = SloLinear(16, 8, bias=True)
        x = Tensor(np.random.randn(1, 16).astype(np.float32))
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize(layer.name, layer.weight.data.copy())
        assert info.is_quantized
        layer.set_quantized_weight(info)
        from domains.infrastructure.quantization import quantized_linear
        bias_arr = layer.bias.data if layer.use_bias else None
        y_direct = quantized_linear(x.data, info.array, info.meta.scale,
                                     info.meta.zero_point, bias_arr)
        y_via_layer = layer.forward_numpy(x.data)
        np.testing.assert_allclose(y_direct, y_via_layer, atol=1e-5)
        assert y_via_layer.shape == y_direct.shape

    def test_autograd_tensor_forward_quantized(self):
        from domains.training.slonet import SloLinear, Tensor
        layer = SloLinear(16, 8, bias=True)
        x = Tensor(np.random.randn(2, 16).astype(np.float32))
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize(layer.name, layer.weight.data.copy())
        layer.set_quantized_weight(info)
        y = layer.forward(x)
        assert isinstance(y, Tensor)
        assert y.data.shape == (2, 8)

    def test_no_quantize_uses_float(self):
        from domains.training.slonet import SloLinear, Tensor
        layer = SloLinear(16, 8, bias=True)
        x = Tensor(np.random.randn(1, 16).astype(np.float32))
        y = layer.forward(x)
        assert isinstance(y, Tensor)
        assert y.data.shape == (1, 8)
        y_np = layer.forward_numpy(x.data)
        np.testing.assert_allclose(y.data, y_np, atol=1e-5)


# ── QuantizedLinear wrapper ──────────────────────────────────────────


class TestQuantizedLinearWrapper:
    def test_dequantize(self):
        from domains.infrastructure.quantization import QuantizedLinear
        w = np.random.randn(8, 16).astype(np.float32) * 0.02
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test", w)
        ql = QuantizedLinear(
            weight_int8=info.array,
            scale=info.meta.scale,
            zero_point=info.meta.zero_point,
            bias=None,
            bits=8,
            original_shape=info.meta.original_shape,
        )
        w_deq = ql.dequantize()
        cosine = np.dot(w.flatten(), w_deq.flatten()) / (
            np.linalg.norm(w) * np.linalg.norm(w_deq)
        )
        assert cosine > 0.95

    def test_forward_numpy(self):
        from domains.infrastructure.quantization import QuantizedLinear
        w = np.random.randn(8, 16).astype(np.float32) * 0.02
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test", w)
        ql = QuantizedLinear(
            weight_int8=info.array,
            scale=info.meta.scale,
            zero_point=info.meta.zero_point,
            bias=None,
            bits=8,
            original_shape=info.meta.original_shape,
        )
        x = np.random.randn(1, 16).astype(np.float32)
        result = ql.forward_numpy(x)
        assert result.shape == (1, 8)

    def test_forward_numpy_with_bias(self):
        from domains.infrastructure.quantization import QuantizedLinear
        w = np.random.randn(4, 8).astype(np.float32) * 0.02
        bias = np.random.randn(4).astype(np.float32) * 0.01
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test", w)
        ql = QuantizedLinear(
            weight_int8=info.array,
            scale=info.meta.scale,
            zero_point=info.meta.zero_point,
            bias=bias,
            bits=8,
            original_shape=info.meta.original_shape,
        )
        x = np.random.randn(2, 8).astype(np.float32)
        result = ql.forward_numpy(x)
        assert result.shape == (2, 4)

    def test_call(self):
        from domains.infrastructure.quantization import QuantizedLinear
        w = np.random.randn(4, 8).astype(np.float32) * 0.02
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test", w)
        ql = QuantizedLinear(
            weight_int8=info.array,
            scale=info.meta.scale,
            zero_point=info.meta.zero_point,
            bias=None,
            bits=8,
            original_shape=info.meta.original_shape,
        )
        x = np.random.randn(1, 8).astype(np.float32)
        result = ql(x)
        assert result.shape == (1, 4)

    def test_dequantize_cached(self):
        from domains.infrastructure.quantization import QuantizedLinear
        w = np.random.randn(4, 8).astype(np.float32) * 0.02
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test", w)
        ql = QuantizedLinear(
            weight_int8=info.array,
            scale=info.meta.scale,
            zero_point=info.meta.zero_point,
            bias=None,
            bits=8,
            original_shape=info.meta.original_shape,
        )
        w1 = ql.dequantize()
        w2 = ql.dequantize()
        assert w1 is w2  # cached

    def test_mode_stored(self):
        from domains.infrastructure.quantization import QuantizedLinear
        w = np.random.randn(4, 8).astype(np.float32) * 0.02
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test", w)
        ql = QuantizedLinear(
            weight_int8=info.array,
            scale=info.meta.scale,
            zero_point=info.meta.zero_point,
            bias=None,
            bits=8,
            original_shape=info.meta.original_shape,
            mode="symmetric",
        )
        assert ql.mode == "symmetric"

    def test_bits_stored(self):
        from domains.infrastructure.quantization import QuantizedLinear
        w = np.random.randn(4, 8).astype(np.float32) * 0.02
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test", w)
        ql = QuantizedLinear(
            weight_int8=info.array,
            scale=info.meta.scale,
            zero_point=info.meta.zero_point,
            bias=None,
            bits=8,
            original_shape=info.meta.original_shape,
        )
        assert ql.bits == 8


# ── QuantMode / QuantDtype ───────────────────────────────────────────


class TestQuantMode:
    def test_symmetric(self):
        assert QuantMode.SYMMETRIC.value == "symmetric"

    def test_asymmetric(self):
        assert QuantMode.ASYMMETRIC.value == "asymmetric"

    def test_from_value(self):
        assert QuantMode("symmetric") is QuantMode.SYMMETRIC
        assert QuantMode("asymmetric") is QuantMode.ASYMMETRIC

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            QuantMode("unknown")


class TestQuantDtype:
    def test_int8(self):
        assert QuantDtype.INT8.value == "int8"

    def test_uint8(self):
        assert QuantDtype.UINT8.value == "uint8"

    def test_int4(self):
        assert QuantDtype.INT4.value == "int4"

    def test_from_value(self):
        assert QuantDtype("int8") is QuantDtype.INT8
        assert QuantDtype("int4") is QuantDtype.INT4

    def test_all_members(self):
        assert len(QuantDtype) == 3
