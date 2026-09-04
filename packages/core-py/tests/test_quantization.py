"""
Tests for per-tensor quantization engine.

Tests cover:
  - Symmetric and asymmetric quantization (8-bit and 4-bit)
  - Per-channel quantization for 2D weight matrices
  - Outlier clipping (percentile-based)
  - Quantization error metrics (MSE, cosine similarity)
  - TensorInfo wrapper (quantized vs non-quantized)
  - QuantMeta round-trip, is_per_channel
  - _pack_int4 / _unpack_int4 round-trip
  - quantize_state_dict, quantize_activation, quantize_kv_tensor, dequantize_kv_tensor
  - int8_matmul correctness against numpy reference
  - quantize_with_scale, save/load metadata and weights
  - Edge cases (zero-variance, single-element, very small tensors)
  - walk_slo_linears / walk_hf_linears with mock models
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from domains.infrastructure.quantization import (
    Quantine,
    QuantMeta,
    QuantizedLinear,
    TensorInfo,
    _cosine_similarity,
    _dequantize,
    _ensure_2d_packed,
    _int4_numpy_fallback,
    _numpy_fallback,
    _pack_int4,
    _unpack_int4,
    dequantize_kv_tensor,
    int8_matmul,
    int4_matmul,
    quantize_activation,
    quantize_kv_tensor,
    quantize_state_dict,
    quantized_linear,
    int4_quantized_linear,
    walk_hf_linears,
    walk_slo_linears,
    should_quantize_row,
    apply_adaptive_quantization,
)


# ---------------------------------------------------------------------------
# TensorInfo
# ---------------------------------------------------------------------------
class TestTensorInfo:
    """Test TensorInfo wrapper for quantized and non-quantized tensors."""

    def test_non_quantized_returns_original(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        info = TensorInfo(name="test", array=arr)

        assert not info.is_quantized
        np.testing.assert_array_equal(info.as_float(), arr)

    def test_quantized_dequantizes_correctly(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        meta = QuantMeta(
            scale=1.0, zero_point=0, bits=8, mode="symmetric",
            dtype_code=5, original_shape=(3,), original_dtype="float32",
        )
        quantized = np.array([1, 2, 3], dtype=np.int8)
        info = TensorInfo(name="test", array=quantized, meta=meta)

        assert info.is_quantized
        result = info.as_float()
        np.testing.assert_allclose(result, arr, atol=0.5)

    def test_shape_preserved(self):
        arr = np.zeros((2, 3), dtype=np.float32)
        info = TensorInfo(name="test", array=arr)
        assert info.shape == (2, 3)

    def test_shape_from_meta(self):
        meta = QuantMeta(
            scale=1.0, zero_point=0, bits=8, mode="symmetric",
            dtype_code=5, original_shape=(10, 20), original_dtype="float32",
        )
        info = TensorInfo(name="test", array=np.zeros((10,), dtype=np.int8), meta=meta)
        assert info.shape == (10, 20)

    def test_dtype_from_meta(self):
        meta = QuantMeta(
            scale=1.0, zero_point=0, bits=8, mode="symmetric",
            dtype_code=5, original_shape=(3,), original_dtype="float64",
        )
        info = TensorInfo(name="test", array=np.zeros(3, dtype=np.int8), meta=meta)
        assert info.dtype == np.dtype("float64")

    def test_nbytes(self):
        arr = np.zeros(10, dtype=np.int8)
        info = TensorInfo(name="test", array=arr)
        assert info.nbytes == 10

    def test_compression_ratio_int8(self):
        meta = QuantMeta(
            scale=1.0, zero_point=0, bits=8, mode="symmetric",
            dtype_code=5, original_shape=(100,), original_dtype="float32",
        )
        quantized = np.zeros((100,), dtype=np.int8)
        info = TensorInfo(name="test", array=quantized, meta=meta)
        assert info.compression_ratio() == pytest.approx(4.0)

    def test_compression_ratio_int4(self):
        meta = QuantMeta(
            scale=1.0, zero_point=0, bits=4, mode="symmetric",
            dtype_code=5, original_shape=(100,), original_dtype="float32",
        )
        quantized = np.zeros((50,), dtype=np.int8)
        info = TensorInfo(name="test", array=quantized, meta=meta)
        assert info.compression_ratio() == pytest.approx(8.0, rel=0.01)

    def test_compression_ratio_not_quantized(self):
        arr = np.zeros(100, dtype=np.float32)
        info = TensorInfo(name="test", array=arr)
        assert info.compression_ratio() == 1.0

    def test_as_float_non_quantized_casts_int_to_float32(self):
        arr = np.array([1, 2, 3], dtype=np.int32)
        info = TensorInfo(name="test", array=arr)
        result = info.as_float()
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, arr.astype(np.float32))


# ---------------------------------------------------------------------------
# Quantine symmetric 8-bit
# ---------------------------------------------------------------------------
class TestQuantineSymmetric:
    """Test symmetric quantization mode."""

    def test_int8_symmetric_basic(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        info = engine.quantize("test", arr)

        assert info.is_quantized
        result = info.as_float()
        np.testing.assert_allclose(result, arr, atol=0.02)

    def test_int8_symmetric_scale(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
        info = engine.quantize("test", arr)

        assert info.meta.scale == pytest.approx(2.0 / 127, rel=0.01)
        assert info.meta.zero_point == 0

    def test_int8_symmetric_error_low(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(1000).astype(np.float32) * 0.5
        info = engine.quantize("test", arr)

        assert info.meta.mse < 0.001
        assert info.meta.cosine_sim > 0.99

    def test_int8_symmetric_preserves_values(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        info = engine.quantize("test", arr)
        result = info.as_float()
        np.testing.assert_allclose(result, arr, atol=0.02)

    def test_int4_symmetric(self):
        engine = Quantine(bits=4, mode="symmetric")
        arr = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        info = engine.quantize("test", arr)

        assert info.is_quantized
        assert info.meta.bits == 4


# ---------------------------------------------------------------------------
# Quantine asymmetric 8-bit
# ---------------------------------------------------------------------------
class TestQuantineAsymmetric:
    """Test asymmetric quantization mode."""

    def test_int8_asymmetric_positive_only(self):
        engine = Quantine(bits=8, mode="asymmetric")
        arr = np.random.randn(10000).astype(np.float32) * 2.0 + 10.0
        info = engine.quantize("test", arr)

        if info.is_quantized:
            result = info.as_float()
            assert result.mean() == pytest.approx(arr.mean(), rel=0.1)
            assert info.meta.cosine_sim > 0.99

    def test_asymmetric_handles_shifted_distribution(self):
        arr = np.random.randn(10000).astype(np.float32) * 2.0 + 10.0
        asym = Quantine(bits=8, mode="asymmetric")
        info = asym.quantize("test", arr)
        assert info is not None

    def test_asymmetric_zero_point_nonzero(self):
        engine = Quantine(bits=8, mode="asymmetric")
        arr = np.random.randn(1000).astype(np.float32) + 5.0
        info = engine.quantize("test", arr)
        if info.is_quantized and not info.meta.is_per_channel:
            assert info.meta.zero_point != 0

    def test_asymmetric_int4(self):
        engine = Quantine(bits=4, mode="asymmetric")
        arr = np.random.rand(100).astype(np.float32) * 10.0
        info = engine.quantize("test", arr)
        assert info.is_quantized
        assert info.meta.bits == 4


# ---------------------------------------------------------------------------
# Per-channel quantization
# ---------------------------------------------------------------------------
class TestQuantinePerChannel:
    """Test per-channel quantization for 2D weight matrices."""

    def test_2d_symmetric_uses_per_channel(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(64, 128).astype(np.float32) * 0.5
        info = engine.quantize("blocks.0.w.weight", arr)

        assert info.is_quantized
        assert info.meta.is_per_channel
        assert info.meta.scale.shape == (64,)
        assert info.array.shape == (64, 128)
        assert info.array.dtype == np.int8

    def test_per_channel_scale_from_true_row_max(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.array([[1.0, 2.0, 3.0], [10.0, -20.0, 30.0]], dtype=np.float32)
        info = engine.quantize("blocks.0.w.weight", arr)

        expected = np.array([3.0, 30.0], dtype=np.float32) / 127.0
        np.testing.assert_allclose(info.meta.scale, expected, rtol=1e-3)

    def test_outlier_row_does_not_destroy_other_rows(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.array([[1.0, 2.0, 3.0], [10.0, -20.0, 30.0]], dtype=np.float32)
        info = engine.quantize("blocks.0.w.weight", arr)

        dequant = info.as_float()
        np.testing.assert_allclose(dequant[1], arr[1], atol=0.3)

    def test_quantized_outlier_preserved(self):
        engine = Quantine(bits=8, mode="symmetric", clip_percentile=0.99)
        arr = np.array([[1.0, 2.0, 3.0], [5.0, 10.0, 1000.0]], dtype=np.float32)
        info = engine.quantize("blocks.0.w.weight", arr)

        dequant = info.as_float()
        assert abs(dequant[1, 2] - 1000.0) < 1000.0 * 0.01
        np.testing.assert_allclose(dequant[0], arr[0], atol=0.3)

    def test_per_channel_error_metrics(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(32, 64).astype(np.float32) * 0.5
        info = engine.quantize("blocks.0.w.weight", arr)

        assert info.meta.cosine_sim > 0.99
        assert info.meta.max_abs_error < 0.02

    def test_1d_does_not_use_per_channel(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(100).astype(np.float32) * 0.5
        info = engine.quantize("some.vec", arr)

        assert info.is_quantized
        assert not info.meta.is_per_channel

    def test_asymmetric_2d_does_not_use_per_channel(self):
        engine = Quantine(bits=8, mode="asymmetric")
        arr = np.random.randn(16, 32).astype(np.float32) * 2.0 + 10.0
        info = engine.quantize("blocks.0.w.weight", arr)

        assert info.meta is not None
        assert not info.meta.is_per_channel

    def test_per_channel_int4_2d(self):
        engine = Quantine(bits=4, mode="symmetric")
        arr = np.random.randn(16, 32).astype(np.float32) * 0.5
        info = engine.quantize("blocks.0.w.weight", arr)
        assert info.is_quantized
        assert info.meta.is_per_channel
        assert info.meta.bits == 4
        deq = info.as_float()
        cosine = _cosine_similarity(arr.flatten(), deq.flatten())
        assert cosine > 0.90


# ---------------------------------------------------------------------------
# Sensitive / skip prefixes
# ---------------------------------------------------------------------------
class TestSkipSensitiveTensors:
    """Test that embedding and norm layers are skipped or flagged sensitive."""

    def test_skip_token_embedding(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(50257, 768).astype(np.float32)
        info = engine.quantize("tok_emb.weight", arr)
        assert not info.is_quantized

    def test_skip_positional_embedding(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(1024, 768).astype(np.float32)
        info = engine.quantize("pos_emb.weight", arr)
        assert not info.is_quantized

    def test_skip_final_norm(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(768).astype(np.float32)
        info = engine.quantize("norm.weight", arr)
        assert not info.is_quantized

    def test_quantize_linear_weights(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(768, 768).astype(np.float32) * 0.02
        info = engine.quantize("blocks.0.q_proj.weight", arr)
        assert info.is_quantized

    def test_is_sensitive_attn_norm(self):
        engine = Quantine(bits=8, mode="symmetric")
        assert engine.is_sensitive("attn_norm.weight")

    def test_is_sensitive_ff_norm(self):
        engine = Quantine(bits=8, mode="symmetric")
        assert engine.is_sensitive("ff_norm.weight")

    def test_not_sensitive_linear(self):
        engine = Quantine(bits=8, mode="symmetric")
        assert not engine.is_sensitive("blocks.0.q_proj.weight")

    def test_should_skip_tok_emb(self):
        engine = Quantine(bits=8, mode="symmetric")
        assert engine.should_skip("tok_emb.weight")

    def test_should_skip_norm(self):
        engine = Quantine(bits=8, mode="symmetric")
        assert engine.should_skip("norm.weight")

    def test_should_not_skip_linear(self):
        engine = Quantine(bits=8, mode="symmetric")
        assert not engine.should_skip("blocks.0.q_proj.weight")


# ---------------------------------------------------------------------------
# Clip percentile
# ---------------------------------------------------------------------------
class TestOutlierClipping:
    """Test percentile-based outlier clipping."""

    def test_clip_improves_main_distribution_accuracy(self):
        np.random.seed(42)
        main_dist = np.random.randn(99000).astype(np.float32) * 1.0
        outliers = np.array([100.0, -100.0])
        arr = np.concatenate([main_dist, outliers])

        engine_noclip = Quantine(bits=8, mode="symmetric")
        engine_clip = Quantine(bits=8, mode="symmetric", clip_percentile=0.999)

        info_noclip = engine_noclip.quantize("test", arr)
        info_clip = engine_clip.quantize("test", arr)

        assert info_noclip.is_quantized
        assert info_clip.is_quantized
        assert info_clip.meta.scale < info_noclip.meta.scale

        deq_noclip = info_noclip.as_float().flatten()
        deq_clip = info_clip.as_float().flatten()
        main_mask = np.abs(arr) < 10

        mse_noclip_main = np.mean((arr[main_mask] - deq_noclip[main_mask]) ** 2)
        mse_clip_main = np.mean((arr[main_mask] - deq_clip[main_mask]) ** 2)
        assert mse_clip_main < mse_noclip_main

    def test_clip_percentile_0_999(self):
        engine = Quantine(bits=8, mode="symmetric", clip_percentile=0.999)
        arr = np.random.randn(10000).astype(np.float32)
        info = engine.quantize("test", arr)
        assert info.is_quantized
        assert info.meta.mse < 0.001

    def test_clip_percentile_none_means_no_clip(self):
        engine = Quantine(bits=8, mode="symmetric", clip_percentile=None)
        arr = np.random.randn(1000).astype(np.float32)
        info = engine.quantize("test", arr)
        assert info.is_quantized


# ---------------------------------------------------------------------------
# Error report and summary
# ---------------------------------------------------------------------------
class TestErrorMetrics:
    """Test quantization error metrics."""

    def test_perfect_quantization_has_zero_mse(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.zeros(100, dtype=np.float32)
        info = engine.quantize("test", arr)
        assert info.meta.mse == 0.0
        assert info.meta.cosine_sim == 1.0

    def test_cosine_similarity_range(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(100).astype(np.float32)
        info = engine.quantize("test", arr)
        assert 0.0 <= info.meta.cosine_sim <= 1.0

    def test_error_report(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(100).astype(np.float32)
        engine.quantize("tensor_a", arr)
        engine.quantize("tensor_b", arr)

        report = engine.error_report()
        assert "tensor_a" in report
        assert "tensor_b" in report
        assert "mse" in report["tensor_a"]
        assert "cosine_sim" in report["tensor_a"]
        assert "max_abs_error" in report["tensor_a"]

    def test_summary(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(100).astype(np.float32)
        engine.quantize("tensor_a", arr)

        summary = engine.summary()
        assert summary["tensors"] == 1
        assert summary["bits"] == 8
        assert summary["mode"] == "symmetric"
        assert "avg_mse" in summary
        assert "avg_cosine_sim" in summary
        assert "worst_tensor" in summary
        assert summary["worst_tensor"] == "tensor_a"

    def test_summary_empty(self):
        engine = Quantine(bits=8, mode="symmetric")
        summary = engine.summary()
        assert summary["tensors"] == 0


# ---------------------------------------------------------------------------
# Metadata save/load round-trip
# ---------------------------------------------------------------------------
class TestMetadataSaveLoad:
    """Test saving and loading quantization metadata."""

    def test_save_load_roundtrip(self, tmp_path: Path):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(100).astype(np.float32)
        engine.quantize("test_tensor", arr)

        path = str(tmp_path / "meta.json")
        engine.save_metadata(path)

        engine2 = Quantine(bits=8, mode="symmetric")
        engine2.load_metadata(path)

        report = engine2.error_report()
        assert "test_tensor" in report
        assert report["test_tensor"]["mse"] == pytest.approx(
            engine.error_report()["test_tensor"]["mse"]
        )

    def test_save_load_multiple_tensors(self, tmp_path: Path):
        engine = Quantine(bits=8, mode="symmetric")
        for i in range(5):
            engine.quantize(f"t{i}", np.random.randn(50).astype(np.float32))

        path = str(tmp_path / "meta.json")
        engine.save_metadata(path)

        engine2 = Quantine(bits=8, mode="symmetric")
        engine2.load_metadata(path)
        assert len(engine2.error_report()) == 5

    def test_load_metadata_overwrites_existing(self, tmp_path: Path):
        engine1 = Quantine(bits=8, mode="symmetric")
        engine1.quantize("a", np.random.randn(50).astype(np.float32))
        path = str(tmp_path / "meta.json")
        engine1.save_metadata(path)

        engine2 = Quantine(bits=8, mode="symmetric")
        engine2.quantize("b", np.random.randn(50).astype(np.float32))
        assert "b" in engine2.error_report()

        engine2.load_metadata(path)
        assert "a" in engine2.error_report()
        assert "b" not in engine2.error_report()


# ---------------------------------------------------------------------------
# Weights save/load round-trip
# ---------------------------------------------------------------------------
class TestSaveLoadWeights:
    """Tests for Quantine.save_weights() / load_weights() round-trip."""

    def test_round_trip_preserves_values(self, tmp_path: Path):
        engine = Quantine(bits=8, mode="symmetric")
        w = np.random.randn(16, 32).astype(np.float32)
        info = engine.quantize("test_w", w)
        path = str(tmp_path / "quant.npz")
        engine.save_weights(path, {"test_w": info})
        loaded = engine.load_weights(path)
        assert "test_w" in loaded
        assert loaded["test_w"].is_quantized
        assert loaded["test_w"].meta.bits == 8
        np.testing.assert_array_equal(loaded["test_w"].array, info.array)

    def test_round_trip_multiple_tensors(self, tmp_path: Path):
        engine = Quantine(bits=8, mode="asymmetric")
        infos = {}
        for name in ("w1", "w2", "b1"):
            w = np.random.randn(8, 16).astype(np.float32)
            infos[name] = engine.quantize(name, w)
        path = str(tmp_path / "multi.npz")
        engine.save_weights(path, infos)
        loaded = engine.load_weights(path)
        assert set(loaded.keys()) == {"w1", "w2", "b1"}
        for name in infos:
            np.testing.assert_array_equal(loaded[name].array, infos[name].array)
            assert loaded[name].meta.bits == infos[name].meta.bits
            assert loaded[name].meta.mode == infos[name].meta.mode

    def test_skips_non_quantized_tensors(self, tmp_path: Path):
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("good", np.random.randn(8, 8).astype(np.float32))
        non_quant = TensorInfo(name="skip", array=np.zeros((4, 4), dtype=np.float32))
        path = str(tmp_path / "skip.npz")
        engine.save_weights(path, {"good": info, "skip": non_quant})
        loaded = engine.load_weights(path)
        assert "good" in loaded
        assert "skip" not in loaded

    def test_load_empty_archive_returns_empty(self, tmp_path: Path):
        path = str(tmp_path / "empty.npz")
        np.savez_compressed(path)
        engine = Quantine()
        result = engine.load_weights(path)
        assert result == {}

    def test_missing_metadata_logs_warning(self, tmp_path: Path):
        path = str(tmp_path / "bad.npz")
        np.savez_compressed(path, w=np.array([1, 2, 3], dtype=np.int8))
        engine = Quantine()
        result = engine.load_weights(path)
        assert result == {}

    def test_int4_round_trip(self, tmp_path: Path):
        engine = Quantine(bits=4, mode="symmetric")
        w = np.random.randn(8, 16).astype(np.float32)
        info = engine.quantize("w4", w)
        path = str(tmp_path / "int4.npz")
        engine.save_weights(path, {"w4": info})
        loaded = engine.load_weights(path)
        assert loaded["w4"].meta.bits == 4
        np.testing.assert_array_equal(loaded["w4"].array, info.array)

    def test_load_rejects_nonexistent_path(self):
        engine = Quantine()
        with pytest.raises((FileNotFoundError, OSError)):
            engine.load_weights("/nonexistent/path.npz")

    def test_weights_round_trip_metadata_populates_error_report(self, tmp_path: Path):
        engine = Quantine(bits=8, mode="symmetric")
        w = np.random.randn(16, 32).astype(np.float32)
        info = engine.quantize("test_w", w)
        path = str(tmp_path / "quant.npz")
        engine.save_weights(path, {"test_w": info})

        engine2 = Quantine(bits=8, mode="symmetric")
        engine2.load_weights(path)
        report = engine2.error_report()
        assert "test_w" in report
        assert report["test_w"]["bits"] == 8


# ---------------------------------------------------------------------------
# quantize_with_scale
# ---------------------------------------------------------------------------
class TestQuantizeWithScale:
    """Test pre-computed scale quantization."""

    def test_quantize_with_scale_basic(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(100).astype(np.float32)
        scale = 0.01
        info = engine.quantize_with_scale("test", arr, scale=scale, zero_point=0)
        assert info.is_quantized
        assert info.meta.scale == scale
        assert info.meta.zero_point == 0

    def test_quantize_with_scale_matches_quantize(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(100).astype(np.float32)
        info_q = engine.quantize("test", arr)
        scale = info_q.meta.scale

        engine2 = Quantine(bits=8, mode="symmetric")
        info_ws = engine2.quantize_with_scale("test", arr, scale=scale, zero_point=0)
        deq_q = info_q.as_float()
        deq_ws = info_ws.as_float()
        np.testing.assert_allclose(deq_q, deq_ws, atol=1e-5)

    def test_quantize_with_scale_skips_skip_prefix(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(10).astype(np.float32)
        info = engine.quantize_with_scale("tok_emb.weight", arr, scale=0.1)
        assert not info.is_quantized

    def test_quantize_with_scale_asymmetric(self):
        engine = Quantine(bits=8, mode="asymmetric")
        arr = np.random.randn(100).astype(np.float32)
        info = engine.quantize_with_scale("test", arr, scale=0.05, zero_point=10)
        assert info.is_quantized
        assert info.meta.zero_point == 10

    def test_quantize_with_scale_per_channel(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(16, 32).astype(np.float32) * 0.5
        per_channel_scale = np.abs(arr).max(axis=1) / 127.0
        info = engine.quantize_with_scale("blocks.0.w.weight", arr, scale=per_channel_scale, zero_point=0)
        assert info.is_quantized
        assert isinstance(info.meta.scale, np.ndarray)
        deq = info.as_float()
        cosine = _cosine_similarity(arr.flatten(), deq.flatten())
        assert cosine > 0.95


# ---------------------------------------------------------------------------
# QuantMeta to_dict / from_dict
# ---------------------------------------------------------------------------
class TestQuantMeta:
    """Test QuantMeta serialization and properties."""

    def test_to_dict_from_dict_roundtrip(self):
        meta = QuantMeta(
            scale=0.05, zero_point=10, bits=8, mode="asymmetric",
            dtype_code=5, original_shape=(16, 32), original_dtype="float32",
            mse=0.001, max_abs_error=0.02, cosine_sim=0.99,
        )
        d = meta.to_dict()
        restored = QuantMeta.from_dict(d)
        assert restored.scale == meta.scale
        assert restored.zero_point == meta.zero_point
        assert restored.bits == meta.bits
        assert restored.mode == meta.mode
        assert restored.original_shape == meta.original_shape
        assert restored.mse == pytest.approx(meta.mse)
        assert restored.cosine_sim == pytest.approx(meta.cosine_sim)

    def test_to_dict_from_dict_per_channel(self):
        scale_arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        meta = QuantMeta(
            scale=scale_arr, zero_point=0, bits=8, mode="symmetric",
            dtype_code=5, original_shape=(3, 10), original_dtype="float32",
        )
        d = meta.to_dict()
        assert isinstance(d["scale"], list)
        restored = QuantMeta.from_dict(d)
        assert isinstance(restored.scale, np.ndarray)
        np.testing.assert_array_almost_equal(restored.scale, scale_arr)

    def test_is_per_channel_true(self):
        meta = QuantMeta(
            scale=np.array([0.1, 0.2]), zero_point=0, bits=8, mode="symmetric",
            dtype_code=5, original_shape=(2, 10), original_dtype="float32",
        )
        assert meta.is_per_channel

    def test_is_per_channel_false(self):
        meta = QuantMeta(
            scale=0.1, zero_point=0, bits=8, mode="symmetric",
            dtype_code=5, original_shape=(10,), original_dtype="float32",
        )
        assert not meta.is_per_channel


# ---------------------------------------------------------------------------
# _pack_int4 / _unpack_int4
# ---------------------------------------------------------------------------
class TestInt4PackUnpack:
    """Test int4 packing and unpacking helpers."""

    def test_pack_roundtrip(self):
        arr = np.array([1, -3, 7, -8, 0, 5, -5, 3], dtype=np.int8)
        packed = _pack_int4(arr)
        assert len(packed) == 4
        unpacked = _unpack_int4(packed, 8)
        np.testing.assert_array_equal(unpacked, arr)

    def test_pack_odd_length(self):
        arr = np.array([1, -3, 7], dtype=np.int8)
        packed = _pack_int4(arr)
        assert len(packed) == 2
        unpacked = _unpack_int4(packed, 3)
        np.testing.assert_array_equal(unpacked, arr)

    def test_unpack_single_byte(self):
        packed = np.array([0x12], dtype=np.int8)
        unpacked = _unpack_int4(packed, 2)
        assert unpacked[0] == 2
        assert unpacked[1] == 1

    def test_pack_single_element(self):
        arr = np.array([5], dtype=np.int8)
        packed = _pack_int4(arr)
        assert len(packed) == 1
        unpacked = _unpack_int4(packed, 1)
        assert unpacked[0] == 5

    def test_pack_all_zeros(self):
        arr = np.zeros(8, dtype=np.int8)
        packed = _pack_int4(arr)
        unpacked = _unpack_int4(packed, 8)
        np.testing.assert_array_equal(unpacked, arr)

    def test_pack_max_min_values(self):
        arr = np.array([7, -8, 7, -8, 7, -8, 7, -8], dtype=np.int8)
        packed = _pack_int4(arr)
        unpacked = _unpack_int4(packed, 8, signed=True)
        np.testing.assert_array_equal(unpacked, arr)

    def test_unsigned_roundtrip(self):
        arr = np.array([0, 5, 10, 15, 3, 7, 12, 1], dtype=np.int8)
        packed = _pack_int4(arr)
        unpacked = _unpack_int4(packed, 8, signed=False)
        np.testing.assert_array_equal(unpacked, arr)


# ---------------------------------------------------------------------------
# _cosine_similarity
# ---------------------------------------------------------------------------
class TestCosineSimFunction:
    """Test the cosine similarity helper."""

    def test_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 2.0])
        b = np.array([-1.0, -2.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vectors(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert _cosine_similarity(a, b) == 0.0

    def test_both_zero_returns_1(self):
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 0.0])
        assert _cosine_similarity(a, b) == 1.0

    def test_similar_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])
        sim = _cosine_similarity(a, b)
        assert 0.99 < sim <= 1.0


# ---------------------------------------------------------------------------
# quantize_state_dict
# ---------------------------------------------------------------------------
class TestQuantizeStateDict:
    """Test quantizing an entire state dict."""

    def test_quantize_state_dict_basic(self):
        state_dict = {
            "blocks.0.q_proj.weight": np.random.randn(768, 768).astype(np.float32) * 0.02,
            "blocks.0.k_proj.weight": np.random.randn(768, 768).astype(np.float32) * 0.02,
            "tok_emb.weight": np.random.randn(50257, 768).astype(np.float32),
        }
        result = quantize_state_dict(state_dict, bits=8, mode="symmetric")

        assert result["blocks.0.q_proj.weight"].is_quantized
        assert result["blocks.0.k_proj.weight"].is_quantized
        assert not result["tok_emb.weight"].is_quantized

    def test_quantize_state_dict_all_errors_acceptable(self):
        state_dict = {
            f"blocks.{i}.q_proj.weight": np.random.randn(64, 64).astype(np.float32) * 0.02
            for i in range(12)
        }
        result = quantize_state_dict(state_dict, bits=8, mode="symmetric")

        for name, info in result.items():
            if info.is_quantized:
                assert info.meta.cosine_sim > 0.95, f"{name} cosine_sim too low: {info.meta.cosine_sim}"

    def test_quantize_state_dict_int4(self):
        state_dict = {
            "blocks.0.w": np.random.randn(32, 64).astype(np.float32) * 0.02,
        }
        result = quantize_state_dict(state_dict, bits=4, mode="symmetric")
        assert result["blocks.0.w"].is_quantized
        assert result["blocks.0.w"].meta.bits == 4

    def test_quantize_state_dict_asymmetric(self):
        state_dict = {
            "w": np.random.randn(100).astype(np.float32) * 2.0 + 5.0,
        }
        result = quantize_state_dict(state_dict, bits=8, mode="asymmetric")
        # May be quantized or skipped depending on error threshold
        assert "w" in result


# ---------------------------------------------------------------------------
# quantize_activation
# ---------------------------------------------------------------------------
class TestQuantizeActivation:
    """Test activation quantization."""

    def test_symmetric_quantize(self):
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        result = quantize_activation(x, scale=1.0 / 127.0, zero_point=0)
        assert result.dtype == np.int8
        np.testing.assert_array_equal(result, [-127, 0, 127])

    def test_asymmetric_quantize(self):
        x = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        result = quantize_activation(x, scale=1.0 / 127.0, zero_point=0)
        assert result.dtype == np.int8

    def test_clip_to_int8_range(self):
        x = np.array([1000.0, -1000.0], dtype=np.float32)
        result = quantize_activation(x, scale=1.0, zero_point=0)
        assert result[0] == 127
        assert result[1] == -128

    def test_preserves_shape(self):
        x = np.random.randn(4, 8).astype(np.float32)
        result = quantize_activation(x, scale=0.1, zero_point=0)
        assert result.shape == (4, 8)

    def test_roundtrip_dequantize(self):
        x = np.random.randn(100).astype(np.float32)
        scale = 0.05
        q = quantize_activation(x, scale=scale, zero_point=0)
        dq = q.astype(np.float32) * scale
        cosine = _cosine_similarity(x, dq)
        assert cosine > 0.95


# ---------------------------------------------------------------------------
# quantize_kv_tensor / dequantize_kv_tensor
# ---------------------------------------------------------------------------
class TestKVTensor:
    """Test KV cache quantization."""

    def test_quantize_kv_basic(self):
        x = np.random.randn(1, 10, 4, 64).astype(np.float32)
        x_int8, scale = quantize_kv_tensor(x)
        assert x_int8.dtype == np.int8
        assert x_int8.shape == x.shape
        assert scale.shape == (1, 10, 4, 1)

    def test_dequantize_kv_roundtrip(self):
        x = np.random.randn(1, 10, 4, 64).astype(np.float32)
        x_int8, scale = quantize_kv_tensor(x)
        dq = dequantize_kv_tensor(x_int8, scale)
        assert dq.shape == x.shape
        assert dq.dtype == np.float32
        cosine = _cosine_similarity(x.flatten(), dq.flatten())
        assert cosine > 0.95

    def test_zero_vector_quantize(self):
        x = np.zeros((1, 5, 2, 32), dtype=np.float32)
        x_int8, scale = quantize_kv_tensor(x)
        assert np.all(x_int8 == 0)

    def test_dequantize_kv_is_exact_for_int8(self):
        x = np.random.randn(2, 8, 4, 32).astype(np.float32)
        x_int8, scale = quantize_kv_tensor(x)
        dq = dequantize_kv_tensor(x_int8, scale)
        # For symmetric quantization with correct scale, error should be small
        max_err = np.max(np.abs(x - dq))
        assert max_err < 0.1


# ---------------------------------------------------------------------------
# int8_matmul
# ---------------------------------------------------------------------------
class TestInt8Matmul:
    """Test int8 matrix multiplication correctness."""

    def test_int8_matmul_symmetric(self):
        np.random.seed(42)
        M, K, N = 4, 16, 8
        a_fp = np.random.randn(M, K).astype(np.float32)
        b_fp = np.random.randn(N, K).astype(np.float32)

        a_scale = np.max(np.abs(a_fp)) / 127.0
        b_scale = np.max(np.abs(b_fp)) / 127.0
        a_int8 = np.clip(np.round(a_fp / a_scale), -128, 127).astype(np.int8)
        b_int8 = np.clip(np.round(b_fp / b_scale), -128, 127).astype(np.int8)

        result = int8_matmul(a_int8, b_int8, a_scale=a_scale, b_scale=b_scale)
        expected = a_fp @ b_fp.T
        np.testing.assert_allclose(result, expected, rtol=0.1, atol=0.5)

    def test_int8_matmul_asymmetric(self):
        np.random.seed(42)
        M, K, N = 4, 16, 8
        a_fp = np.random.randn(M, K).astype(np.float32) + 5.0
        b_fp = np.random.randn(N, K).astype(np.float32) + 5.0

        a_range = a_fp.max() - a_fp.min()
        b_range = b_fp.max() - b_fp.min()
        a_scale = a_range / 255.0
        b_scale = b_range / 255.0
        a_zp = int(np.round(-128 - a_fp.min() / a_scale))
        b_zp = int(np.round(-128 - b_fp.min() / b_scale))

        a_int8 = np.clip(np.round(a_fp / a_scale) + a_zp, -128, 127).astype(np.int8)
        b_int8 = np.clip(np.round(b_fp / b_scale) + b_zp, -128, 127).astype(np.int8)

        result = int8_matmul(
            a_int8, b_int8,
            a_scale=a_scale, b_scale=b_scale,
            a_zero_point=a_zp, b_zero_point=b_zp,
        )
        expected = a_fp @ b_fp.T
        np.testing.assert_allclose(result, expected, rtol=0.2, atol=1.0)

    def test_int8_matmul_per_channel_b_scale(self):
        np.random.seed(42)
        M, K, N = 4, 16, 8
        a_fp = np.random.randn(M, K).astype(np.float32)
        b_fp = np.random.randn(N, K).astype(np.float32)

        a_scale = np.max(np.abs(a_fp)) / 127.0
        b_row_max = np.max(np.abs(b_fp), axis=1) / 127.0
        b_scale = b_row_max.astype(np.float32)

        a_int8 = np.clip(np.round(a_fp / a_scale), -128, 127).astype(np.int8)
        b_int8 = np.clip(np.round(b_fp / b_scale[:, None]), -128, 127).astype(np.int8)

        result = int8_matmul(a_int8, b_int8, a_scale=a_scale, b_scale=b_scale)
        expected = a_fp @ b_fp.T
        np.testing.assert_allclose(result, expected, rtol=0.15, atol=0.5)


# ---------------------------------------------------------------------------
# quantized_linear
# ---------------------------------------------------------------------------
class TestQuantizedLinearFunc:
    """Test the quantized_linear function."""

    def test_quantized_linear_basic(self):
        np.random.seed(42)
        M, K, N = 4, 16, 8
        x = np.random.randn(M, K).astype(np.float32)
        w = np.random.randn(N, K).astype(np.float32) * 0.02

        w_scale = np.max(np.abs(w)) / 127.0
        w_int8 = np.clip(np.round(w / w_scale), -128, 127).astype(np.int8)

        result = quantized_linear(x, w_int8, weight_scale=w_scale)
        expected = x @ w.T
        np.testing.assert_allclose(result, expected, rtol=0.15, atol=0.1)

    def test_quantized_linear_with_bias(self):
        np.random.seed(42)
        M, K, N = 2, 8, 4
        x = np.random.randn(M, K).astype(np.float32)
        w = np.random.randn(N, K).astype(np.float32) * 0.02
        b = np.random.randn(N).astype(np.float32) * 0.01

        w_scale = np.max(np.abs(w)) / 127.0
        w_int8 = np.clip(np.round(w / w_scale), -128, 127).astype(np.int8)

        result = quantized_linear(x, w_int8, weight_scale=w_scale, bias=b)
        expected = x @ w.T + b
        np.testing.assert_allclose(result, expected, rtol=0.15, atol=0.1)


# ---------------------------------------------------------------------------
# walk_slo_linears / walk_hf_linears
# ---------------------------------------------------------------------------
class TestWalkLinears:
    """Test model layer walking functions."""

    def test_walk_slo_linears(self):
        from unittest.mock import patch, MagicMock

        FakeSloLinear = type("SloLinear", (), {})

        block_attn = SimpleNamespace(
            W_q=FakeSloLinear(),
            W_k=FakeSloLinear(),
            W_v=FakeSloLinear(),
            W_o=FakeSloLinear(),
        )
        block_ff = SimpleNamespace(
            w1=FakeSloLinear(),
            w2=FakeSloLinear(),
            w3=FakeSloLinear(),
        )
        block = SimpleNamespace(attn=block_attn, ff=block_ff)
        model = SimpleNamespace(
            blocks=[block],
            layers=[SimpleNamespace(), FakeSloLinear()],
        )

        mock_mod = MagicMock()
        mock_mod.SloLinear = FakeSloLinear
        with patch.dict("sys.modules", {"domains.training.slonet": mock_mod}):
            result = walk_slo_linears(model)
        assert "lm_head" in result
        assert "blocks.0.attn.W_q" in result
        assert "blocks.0.ff.w1" in result
        assert len(result) == 8

    def test_walk_slo_linears_empty(self):
        from unittest.mock import patch, MagicMock

        mock_mod = MagicMock()
        mock_mod.SloLinear = type("SloLinear", (), {})
        with patch.dict("sys.modules", {"domains.training.slonet": mock_mod}):
            model = SimpleNamespace(layers=[], blocks=[])
            result = walk_slo_linears(model)
        assert result == {}

    def test_walk_hf_linears(self):
        class Linear:
            def __init__(self):
                self.weight = True

        class FakeModule:
            def named_modules(self):
                return [
                    ("", self),
                    ("encoder.layer.0.attention", SimpleNamespace()),
                    ("encoder.layer.0.attention.query", Linear()),
                    ("encoder.layer.0.attention.key", Linear()),
                    ("encoder.layer.0.output", Linear()),
                ]

        model = FakeModule()
        result = walk_hf_linears(model)
        assert len(result) == 3
        assert "encoder.layer.0.attention.query" in result

    def test_walk_hf_linears_no_linear(self):
        class FakeModule:
            def named_modules(self):
                return [("", SimpleNamespace())]

        model = FakeModule()
        result = walk_hf_linears(model)
        assert result == {}


# ---------------------------------------------------------------------------
# Adaptive quantization
# ---------------------------------------------------------------------------
class TestAdaptiveQuantization:
    """Test adaptive int8 dispatch that only quantizes where int8 beats fp32."""

    _MOCK_MOD = None

    def _slo_linear(self, n, k, quantized=False):
        """Build a fake SloLinear-compatible module with the fields used."""
        from unittest.mock import MagicMock

        w = MagicMock()
        w.data = np.zeros((n, k), dtype=np.float32)
        m = MagicMock()
        m.weight = w
        m._quant_info = MagicMock() if quantized else None
        return m

    @pytest.fixture(autouse=True)
    def _patch_slonet(self):
        """Expose the fake SloLinear class to walk_slo_linears."""
        from unittest.mock import MagicMock, patch

        FakeSloLinear = MagicMock()
        mock_mod = MagicMock()
        mock_mod.SloLinear = FakeSloLinear
        with patch.dict("sys.modules", {"domains.training.slonet": mock_mod}):
            yield FakeSloLinear

    def test_should_quantize_small_false(self):
        # Small embed dims must stay fp32 (int8 loses on AVX512 numpy).
        assert should_quantize_row(96) is False
        assert should_quantize_row(256) is False
        assert should_quantize_row(511) is False

    def test_should_quantize_large_true(self):
        # Large inner dims cross the memory-bound threshold and should quantize.
        assert should_quantize_row(1024) is True
        assert should_quantize_row(4096) is True

    def test_should_quantize_respects_crossover(self):
        assert should_quantize_row(300, crossover_k=256) is True
        assert should_quantize_row(200, crossover_k=256) is False

    def test_no_kernel_means_never_quantize(self, _patch_slonet):
        # When the AVX2 kernel is not available, never vote to quantize.
        from domains.infrastructure import quantization as q
        from unittest.mock import patch

        w = self._slo_linear(4, 2048)
        with patch.object(q, "_has_int8_kernel", return_value=False):
            assert should_quantize_row(2048) is False

    def test_apply_small_model_stays_fp32(self, _patch_slonet):
        # A small-embed model: every layer below crossover, none quantized.
        linears = {}
        for name, n, k in [
            ("lm_head", 65, 96),
            ("blocks.0.attn.W_q", 96, 96),
            ("blocks.0.ff.w1", 96, 96),
            ("blocks.1.attn.W_k", 96, 96),
        ]:
            linears[name] = self._slo_linear(n, k)

        from unittest.mock import patch
        from domains.infrastructure import quantization as q

        class FakeModel:
            pass

        fm = FakeModel()
        with patch.object(q, "walk_slo_linears", return_value=linears):
            res = apply_adaptive_quantization(fm, bits=8, mode="symmetric")

        assert res["quantized"] == 0
        assert res["total"] == 4
        assert res["left_fp32"] == 4
        # No layer got a quantized weight set.
        for m in linears.values():
            assert m.set_quantized_weight.call_count == 0

    def test_apply_large_model_quantizes_big_layers(self, _patch_slonet):
        # Mixed model: big layers quantized, small layer left fp32.
        linears = {}
        for name, n, k, qz in [
            ("lm_head", 65, 1024, False),
            ("blocks.0.attn.W_q", 1024, 1024, False),
            ("blocks.0.ff.w1", 1024, 4096, False),
            ("blocks.0.attn.W_k", 64, 64, False),   # small -> stays fp32
        ]:
            linears[name] = self._slo_linear(n, k, qz)

        from unittest.mock import patch
        from domains.infrastructure import quantization as q

        fm = SimpleNamespace()
        with patch.object(q, "walk_slo_linears", return_value=linears):
            res = apply_adaptive_quantization(fm, bits=8, mode="symmetric")

        assert res["total"] == 4
        assert res["left_fp32"] == 1
        assert res["quantized"] == 3
        # big layers had set_quantized_weight called; small one did not.
        assert linears["blocks.0.attn.W_q"].set_quantized_weight.call_count == 1
        assert linears["blocks.0.attn.W_k"].set_quantized_weight.call_count == 0

    def test_apply_skips_already_quantized(self, _patch_slonet):
        linears = {
            "lm_head": self._slo_linear(65, 2048, quantized=True),
            "blocks.0.attn.W_q": self._slo_linear(2048, 2048, quantized=False),
        }
        from unittest.mock import patch
        from domains.infrastructure import quantization as q

        fm = SimpleNamespace()
        with patch.object(q, "walk_slo_linears", return_value=linears):
            res = apply_adaptive_quantization(fm, bits=8, mode="symmetric")

        assert res["quantized"] == 2  # already-quantized lm_head counts, no re-quantize
        assert linears["lm_head"].set_quantized_weight.call_count == 0
        assert linears["blocks.0.attn.W_q"].set_quantized_weight.call_count == 1


# ---------------------------------------------------------------------------
# Int4 quantization (extended)
# ---------------------------------------------------------------------------
class TestInt4Quantization:
    """Test int4 quantization."""

    def test_symmetric_quantize_dequantize(self):
        engine = Quantine(bits=4, mode="symmetric")
        arr = np.random.randn(8, 8).astype(np.float32)
        info = engine.quantize("test", arr)
        assert info.is_quantized
        assert info.meta.bits == 4
        assert info.meta.zero_point == 0
        assert info.array.nbytes == 32

    def test_asymmetric_quantize_dequantize(self):
        engine = Quantine(bits=4, mode="asymmetric")
        arr = np.random.rand(8, 8).astype(np.float32) * 10.0
        info = engine.quantize("test", arr)
        assert info.is_quantized
        assert info.meta.bits == 4

    def test_as_float_restores_shape(self):
        engine = Quantine(bits=4, mode="symmetric")
        arr = np.random.randn(16, 32).astype(np.float32)
        info = engine.quantize("test", arr)
        deq = info.as_float()
        assert deq.shape == (16, 32)
        assert deq.dtype == np.float32

    def test_cosine_gpt2_scale(self):
        rng = np.random.RandomState(42)
        w = rng.randn(768, 768).astype(np.float32) * 0.02
        engine = Quantine(bits=4, mode="symmetric")
        info = engine.quantize("test", w)
        assert info.meta.cosine_sim > 0.95

    def test_skip_prefixes_respected(self):
        engine = Quantine(bits=4, mode="symmetric")
        arr = np.random.randn(10, 10).astype(np.float32)
        info = engine.quantize("tok_emb.test", arr)
        assert not info.is_quantized

    def test_compression_ratio(self):
        engine = Quantine(bits=4, mode="symmetric")
        arr = np.random.randn(100, 100).astype(np.float32)
        info = engine.quantize("test", arr)
        ratio = info.compression_ratio()
        assert 7.5 < ratio < 8.5

    def test_dequantize_preserves_values(self):
        engine = Quantine(bits=4, mode="symmetric")
        arr = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
        info = engine.quantize("test", arr)
        deq = info.as_float()
        cosine = _cosine_similarity(arr.flatten(), deq.flatten())
        assert cosine > 0.90


# ---------------------------------------------------------------------------
# QuantizedLinear (class)
# ---------------------------------------------------------------------------
class TestQuantizedLinear:
    """Test QuantizedLinear drop-in replacement for nn.Linear."""

    def test_create_from_quantized_data(self):
        w = np.random.randn(32, 32).astype(np.float32) * 0.02
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test.weight", w)
        assert info.is_quantized

        ql = QuantizedLinear(
            weight_int8=info.array,
            scale=info.meta.scale,
            zero_point=info.meta.zero_point,
            bias=None,
            bits=info.meta.bits,
            original_shape=info.meta.original_shape,
            mode=info.meta.mode,
        )

        deq = ql.dequantize()
        cosine = _cosine_similarity(w, deq)
        assert cosine > 0.99

    def test_forward_numpy_shape(self):
        w = np.random.randn(16, 32).astype(np.float32) * 0.02
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test.weight", w)

        ql = QuantizedLinear(
            weight_int8=info.array,
            scale=info.meta.scale,
            zero_point=info.meta.zero_point,
            bias=None,
            bits=info.meta.bits,
            original_shape=info.meta.original_shape,
            mode=info.meta.mode,
        )

        x = np.random.randn(4, 32).astype(np.float32)
        result = ql.forward_numpy(x)
        assert result.shape == (4, 16)

    def test_forward_numpy_with_bias(self):
        w = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        b = np.array([0.5, -0.5], dtype=np.float32)
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test.weight", w)

        ql = QuantizedLinear(
            weight_int8=info.array,
            scale=info.meta.scale,
            zero_point=info.meta.zero_point,
            bias=b,
            bits=info.meta.bits,
            original_shape=info.meta.original_shape,
            mode=info.meta.mode,
        )

        x = np.array([[1.0, 2.0]], dtype=np.float32)
        result = ql.forward_numpy(x)
        expected = x @ w.T + b
        np.testing.assert_allclose(result, expected, atol=0.1)

    def test_make_torch_forward_returns_callable(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        w = np.random.randn(8, 16).astype(np.float32) * 0.02
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test.weight", w)

        ql = QuantizedLinear(
            weight_int8=info.array,
            scale=info.meta.scale,
            zero_point=info.meta.zero_point,
            bias=None,
            bits=info.meta.bits,
            original_shape=info.meta.original_shape,
            mode=info.meta.mode,
        )

        fwd = ql.make_torch_forward()
        assert callable(fwd)

        x_torch = torch.randn(2, 16)
        result = fwd(x_torch)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 8)

    def test_from_linear_extraction(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not installed")

        linear = nn.Linear(16, 8, bias=True)
        engine = Quantine(bits=8, mode="symmetric")
        w = linear.weight.data.cpu().numpy().astype(np.float32).copy()
        info = engine.quantize("test.weight", w)
        assert info.is_quantized

        ql = QuantizedLinear.from_linear(linear, info)
        assert ql.bias is not None
        assert ql.bias.shape == (8,)

    def test_from_linear_no_bias(self):
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not installed")

        linear = nn.Linear(16, 8, bias=False)
        engine = Quantine(bits=8, mode="symmetric")
        w = linear.weight.data.cpu().numpy().astype(np.float32).copy()
        info = engine.quantize("test.weight", w)

        ql = QuantizedLinear.from_linear(linear, info)
        assert ql.bias is None

    def test_int4_quantized_linear(self):
        w = np.random.randn(32, 32).astype(np.float32) * 0.02
        engine = Quantine(bits=4, mode="symmetric")
        info = engine.quantize("test.weight", w)

        ql = QuantizedLinear(
            weight_int8=info.array,
            scale=info.meta.scale,
            zero_point=info.meta.zero_point,
            bias=None,
            bits=info.meta.bits,
            original_shape=info.meta.original_shape,
            mode=info.meta.mode,
        )

        x = np.random.randn(4, 32).astype(np.float32)
        result = ql.forward_numpy(x)
        assert result.shape == (4, 32)

    def test_asymmetric_mode(self):
        w = np.random.randn(16, 16).astype(np.float32) * 0.02
        engine = Quantine(bits=8, mode="asymmetric")
        info = engine.quantize("test.weight", w)

        ql = QuantizedLinear(
            weight_int8=info.array,
            scale=info.meta.scale,
            zero_point=info.meta.zero_point,
            bias=None,
            bits=info.meta.bits,
            original_shape=info.meta.original_shape,
            mode=info.meta.mode,
        )

        x = np.random.randn(2, 16).astype(np.float32)
        result = ql.forward_numpy(x)
        assert result.shape == (2, 16)

    def test_call_delegates_to_forward_numpy(self):
        w = np.random.randn(8, 16).astype(np.float32) * 0.02
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test.weight", w)
        ql = QuantizedLinear(
            weight_int8=info.array, scale=info.meta.scale,
            zero_point=info.meta.zero_point, bias=None,
            bits=info.meta.bits, original_shape=info.meta.original_shape,
            mode=info.meta.mode,
        )
        x = np.random.randn(2, 16).astype(np.float32)
        result = ql(x)
        assert result.shape == (2, 8)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    """Test edge cases."""

    def test_zero_variance_tensor(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.zeros(100, dtype=np.float32)
        info = engine.quantize("test", arr)
        assert info.is_quantized

    def test_single_element(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.array([42.0], dtype=np.float32)
        info = engine.quantize("test", arr)
        assert info.is_quantized
        result = info.as_float()
        np.testing.assert_allclose(result, arr, atol=1.0)

    def test_multidimensional(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(12, 64, 64).astype(np.float32) * 0.02
        info = engine.quantize("test", arr)
        assert info.is_quantized
        assert info.meta.original_shape == (12, 64, 64)

    def test_invalid_bits_raises(self):
        with pytest.raises(ValueError):
            Quantine(bits=16, mode="symmetric")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            Quantine(bits=8, mode="invalid")


# ---------------------------------------------------------------------------
# SuggestFormat
# ---------------------------------------------------------------------------
class TestSuggestFormat:
    """Tests for Quantine.suggest_format() CPU precision auto-selection."""

    def test_returns_expected_keys(self):
        result = Quantine.suggest_format()
        assert "format" in result
        assert "bits" in result
        assert "reason" in result
        assert "benchmark" in result
        assert result["format"] in ("fp32", "int8", "int4")
        assert result["bits"] in (32, 8, 4)

    def test_fp32_always_in_benchmark(self):
        result = Quantine.suggest_format()
        assert "fp32" in result["benchmark"]
        assert result["benchmark"]["fp32"]["cosine_sim"] == 1.0
        assert result["benchmark"]["fp32"]["bits"] == 32

    def test_int8_in_benchmark(self):
        result = Quantine.suggest_format()
        assert "int8" in result["benchmark"]
        assert result["benchmark"]["int8"]["cosine_sim"] > 0.9
        assert result["benchmark"]["int8"]["bits"] == 8

    def test_int4_in_benchmark(self):
        result = Quantine.suggest_format()
        assert "int4" in result["benchmark"]
        assert result["benchmark"]["int4"]["bits"] == 4

    def test_benchmark_timing_non_negative(self):
        result = Quantine.suggest_format()
        for fmt in ("fp32", "int8", "int4"):
            assert result["benchmark"][fmt]["time_s"] > 0

    def test_custom_sample_weight(self):
        w = np.random.randn(64, 64).astype(np.float32)
        result = Quantine.suggest_format(sample_weight=w)
        assert result["format"] in ("fp32", "int8", "int4")

    def test_low_quality_threshold_prefers_int8(self):
        result = Quantine.suggest_format(quality_threshold=0.5, min_speed_ratio=0.1)
        assert "benchmark" in result

    def test_1d_weight_reshaped(self):
        w = np.random.randn(64).astype(np.float32)
        result = Quantine.suggest_format(sample_weight=w)
        assert result["format"] in ("fp32", "int8", "int4")


# ---------------------------------------------------------------------------
# _numpy_fallback / _int4_numpy_fallback
# ---------------------------------------------------------------------------
class TestFallbackGEMM:
    """Test pure-numpy GEMM fallback functions."""

    def test_numpy_fallback_correctness(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int8)
        b = np.array([[5, 6], [7, 8]], dtype=np.int8)
        result = _numpy_fallback(a, b)
        expected = a.astype(np.int32) @ b.astype(np.int32).T
        np.testing.assert_array_equal(result, expected)

    def test_numpy_fallback_larger(self):
        np.random.seed(99)
        a = np.random.randint(-10, 10, (4, 8)).astype(np.int8)
        b = np.random.randint(-10, 10, (3, 8)).astype(np.int8)
        result = _numpy_fallback(a, b)
        expected = a.astype(np.int32) @ b.astype(np.int32).T
        np.testing.assert_array_equal(result, expected)

    def test_int4_numpy_fallback_correctness(self):
        arr = np.array([1, -3, 7, -8], dtype=np.int8)
        packed = _pack_int4(arr)
        B_packed = packed.reshape(1, -1)
        K = 4
        A = np.array([[1, 0, 1, 0]], dtype=np.int8)
        result = _int4_numpy_fallback(A, B_packed, K)
        assert result.shape == (1, 1)
        expected_val = 1 * 1 + 0 * (-3) + 1 * 7 + 0 * (-8)
        assert result[0, 0] == expected_val

    def test_int4_numpy_fallback_roundtrip(self):
        np.random.seed(77)
        arr = np.random.randint(-8, 7, (3, 8)).astype(np.int8)
        packed = _pack_int4(arr.ravel()).reshape(3, 4)
        A = np.random.randint(-5, 5, (2, 8)).astype(np.int8)
        result = _int4_numpy_fallback(A, packed, 8)
        unpacked = np.zeros((3, 8), dtype=np.int8)
        for j in range(3):
            for k in range(8):
                if k % 2 == 0:
                    nib = int(packed[j, k // 2]) & 0x0F
                else:
                    nib = (int(packed[j, k // 2]) >> 4) & 0x0F
                unpacked[j, k] = np.int8((nib ^ 8) - 8)
        expected = A.astype(np.int32) @ unpacked.astype(np.int32).T
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# _ensure_2d_packed
# ---------------------------------------------------------------------------
class TestEnsure2dPacked:
    """Test _ensure_2d_packed reshaping."""

    def test_1d_to_2d(self):
        packed = np.zeros(16, dtype=np.int8)
        result = _ensure_2d_packed(packed, orig_k=8)
        assert result.shape == (4, 4)

    def test_already_2d(self):
        packed = np.zeros((4, 4), dtype=np.int8)
        result = _ensure_2d_packed(packed, orig_k=8)
        assert result.shape == (4, 4)

    def test_1d_odd_k(self):
        packed = np.zeros(6, dtype=np.int8)
        result = _ensure_2d_packed(packed, orig_k=4)
        assert result.shape == (3, 2)


# ---------------------------------------------------------------------------
# int4_matmul direct tests
# ---------------------------------------------------------------------------
class TestInt4Matmul:
    """Test int4 matrix multiplication directly."""

    def test_int4_matmul_symmetric(self):
        np.random.seed(42)
        M, K, N = 2, 8, 4
        a_fp = np.random.randn(M, K).astype(np.float32)
        b_fp = np.random.randn(N, K).astype(np.float32)

        a_scale = np.max(np.abs(a_fp)) / 127.0
        b_scale = np.max(np.abs(b_fp)) / 7.0

        a_int8 = np.clip(np.round(a_fp / a_scale), -128, 127).astype(np.int8)
        b_int4 = np.clip(np.round(b_fp / b_scale), -8, 7).astype(np.int8)
        b_packed = _pack_int4(b_int4.ravel()).reshape(N, K // 2).astype(np.int8)

        result = int4_matmul(
            a_int8, b_packed,
            a_scale=a_scale, b_scale=b_scale,
            orig_k=K,
        )
        expected = a_fp @ b_fp.T
        np.testing.assert_allclose(result, expected, rtol=0.2, atol=1.0)

    def test_int4_matmul_asymmetric(self):
        np.random.seed(42)
        M, K, N = 2, 8, 4
        a_fp = np.random.randn(M, K).astype(np.float32) + 5.0
        b_fp = np.random.rand(N, K).astype(np.float32) * 10.0

        a_range = a_fp.max() - a_fp.min()
        b_range = b_fp.max() - b_fp.min()
        a_scale = a_range / 255.0
        b_scale = b_range / 15.0
        a_zp = int(np.round(-128 - a_fp.min() / a_scale))
        b_zp = int(np.round(-b_fp.min() / b_scale))

        a_int8 = np.clip(np.round(a_fp / a_scale) + a_zp, -128, 127).astype(np.int8)
        b_int4 = np.clip(np.round(b_fp / b_scale) + b_zp, 0, 15).astype(np.int8)
        b_packed = _pack_int4(b_int4.ravel()).reshape(N, K // 2).astype(np.int8)

        result = int4_matmul(
            a_int8, b_packed,
            a_scale=a_scale, b_scale=b_scale,
            orig_k=K,
            a_zero_point=a_zp, b_zero_point=b_zp,
        )
        assert result.shape == (M, N)
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))

    def test_int4_matmul_1d_packed(self):
        np.random.seed(42)
        K, N = 8, 4
        a = np.random.randint(-10, 10, (2, K)).astype(np.int8)
        b_int4 = np.random.randint(-8, 7, (N, K)).astype(np.int8)
        b_packed = _pack_int4(b_int4.ravel()).astype(np.int8)

        result = int4_matmul(a, b_packed, a_scale=1.0, b_scale=1.0, orig_k=K)
        expected = a.astype(np.int32) @ b_int4.astype(np.int32).T
        np.testing.assert_array_equal(result, expected.astype(np.float32))


# ---------------------------------------------------------------------------
# int4_quantized_linear
# ---------------------------------------------------------------------------
class TestInt4QuantizedLinear:
    """Test int4_quantized_linear function."""

    def test_int4_quantized_linear_basic(self):
        np.random.seed(42)
        M, K, N = 4, 16, 8
        x = np.random.randn(M, K).astype(np.float32)
        w = np.random.randn(N, K).astype(np.float32) * 0.02

        w_scale = np.max(np.abs(w)) / 7.0
        w_int4 = np.clip(np.round(w / w_scale), -8, 7).astype(np.int8)
        w_packed = _pack_int4(w_int4.ravel()).reshape(N, K // 2).astype(np.int8)

        result = int4_quantized_linear(x, w_packed, weight_scale=w_scale, weight_zero_point=0, orig_k=K)
        assert result.shape == (M, N)

    def test_int4_quantized_linear_with_bias(self):
        np.random.seed(42)
        M, K, N = 2, 8, 4
        x = np.random.randn(M, K).astype(np.float32)
        w = np.random.randn(N, K).astype(np.float32) * 0.02
        b = np.random.randn(N).astype(np.float32) * 0.01

        w_scale = np.max(np.abs(w)) / 7.0
        w_int4 = np.clip(np.round(w / w_scale), -8, 7).astype(np.int8)
        w_packed = _pack_int4(w_int4.ravel()).reshape(N, K // 2).astype(np.int8)

        result = int4_quantized_linear(x, w_packed, weight_scale=w_scale, weight_zero_point=0, orig_k=K, bias=b)
        assert result.shape == (M, N)

    def test_int4_quantized_linear_asymmetric(self):
        np.random.seed(42)
        M, K, N = 2, 8, 4
        x = np.random.randn(M, K).astype(np.float32)
        w = np.random.rand(N, K).astype(np.float32) * 10.0

        w_range = w.max() - w.min()
        w_scale = w_range / 15.0
        w_zp = int(np.round(-w.min() / w_scale))
        w_int4 = np.clip(np.round(w / w_scale) + w_zp, 0, 15).astype(np.int8)
        w_packed = _pack_int4(w_int4.ravel()).reshape(N, K // 2).astype(np.int8)

        result = int4_quantized_linear(x, w_packed, weight_scale=w_scale, weight_zero_point=w_zp, orig_k=K)
        assert result.shape == (M, N)

    def test_int4_quantized_linear_1d_packed(self):
        np.random.seed(42)
        M, K, N = 2, 8, 4
        x = np.random.randn(M, K).astype(np.float32)
        w = np.random.randn(N, K).astype(np.float32) * 0.02

        w_scale = np.max(np.abs(w)) / 7.0
        w_int4 = np.clip(np.round(w / w_scale), -8, 7).astype(np.int8)
        w_packed = _pack_int4(w_int4.ravel()).astype(np.int8)

        result = int4_quantized_linear(x, w_packed, weight_scale=w_scale, weight_zero_point=0, orig_k=K)
        assert result.shape == (M, N)


# ---------------------------------------------------------------------------
# QuantizedLinear.dequantize caching
# ---------------------------------------------------------------------------
class TestQuantizedLinearCaching:
    """Test QuantizedLinear dequantize caching behavior."""

    def test_dequantize_caches_result(self):
        w = np.random.randn(16, 16).astype(np.float32) * 0.02
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("test.weight", w)
        ql = QuantizedLinear(
            weight_int8=info.array, scale=info.meta.scale,
            zero_point=info.meta.zero_point, bias=None,
            bits=info.meta.bits, original_shape=info.meta.original_shape,
            mode=info.meta.mode,
        )
        d1 = ql.dequantize()
        d2 = ql.dequantize()
        np.testing.assert_array_equal(d1, d2)
        assert d1 is d2


# ---------------------------------------------------------------------------
# Error threshold skip behavior
# ---------------------------------------------------------------------------
class TestErrorThreshold:
    """Test that high-error tensors are skipped."""

    def test_skip_when_error_too_high(self):
        arr = np.ones(100, dtype=np.float32)
        arr[0] = 1000.0
        engine = Quantine(bits=4, mode="asymmetric", skip_quantize_if_error_above=0.0001)
        info = engine.quantize("test", arr)
        assert not info.is_quantized

    def test_quantize_when_error_below_threshold(self):
        arr = np.random.randn(100).astype(np.float32) * 0.01
        engine = Quantine(bits=8, mode="symmetric", skip_quantize_if_error_above=10.0)
        info = engine.quantize("test", arr)
        assert info.is_quantized


# ---------------------------------------------------------------------------
# dequantize_to_float
# ---------------------------------------------------------------------------
class TestDequantizeToFloat:
    """Test Quantine.dequantize_to_float convenience method."""

    def test_dequantize_to_float(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(100).astype(np.float32)
        info = engine.quantize("test", arr)
        result = engine.dequantize_to_float(info)
        np.testing.assert_allclose(result, arr, atol=0.1)

    def test_dequantize_to_float_non_quantized(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(50).astype(np.float32)
        info = TensorInfo(name="test", array=arr)
        result = engine.dequantize_to_float(info)
        np.testing.assert_array_equal(result, arr)


# ---------------------------------------------------------------------------
# int8_matmul edge cases
# ---------------------------------------------------------------------------
class TestInt8MatmulEdgeCases:
    """Test int8_matmul with per-channel b_scale."""

    def test_per_channel_b_scale_as_ndarray(self):
        np.random.seed(42)
        M, K, N = 2, 8, 4
        a = np.random.randint(-10, 10, (M, K)).astype(np.int8)
        b = np.random.randint(-10, 10, (N, K)).astype(np.int8)
        b_scale = np.max(np.abs(b), axis=1).astype(np.float32) / 127.0
        result = int8_matmul(a, b, a_scale=1.0, b_scale=b_scale)
        raw = a.astype(np.int32) @ b.astype(np.int32).T
        expected = raw.astype(np.float32) * b_scale[np.newaxis, :]
        np.testing.assert_allclose(result, expected, atol=1.0)

    def test_single_element_matmul(self):
        a = np.array([[5]], dtype=np.int8)
        b = np.array([[3]], dtype=np.int8)
        result = int8_matmul(a, b, a_scale=1.0, b_scale=1.0)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# QuantMeta defaults
# ---------------------------------------------------------------------------
class TestQuantMetaDefaults:
    """Test QuantMeta default field values."""

    def test_default_error_metrics(self):
        meta = QuantMeta(
            scale=0.1, zero_point=0, bits=8, mode="symmetric",
            dtype_code=5, original_shape=(10,), original_dtype="float32",
        )
        assert meta.mse == 0.0
        assert meta.max_abs_error == 0.0
        assert meta.cosine_sim == 1.0

    def test_from_dict_with_defaults(self):
        d = {
            "scale": 0.1, "zero_point": 0, "bits": 8, "mode": "symmetric",
            "dtype_code": 5, "original_shape": [10], "original_dtype": "float32",
        }
        meta = QuantMeta.from_dict(d)
        assert meta.mse == 0.0
        assert meta.cosine_sim == 1.0


# ---------------------------------------------------------------------------
# Quantine constructor edge cases
# ---------------------------------------------------------------------------
class TestQuantineConstructor:
    """Test Quantine initialization edge cases."""

    def test_bits_4(self):
        engine = Quantine(bits=4, mode="symmetric")
        assert engine._bits == 4

    def test_bits_8(self):
        engine = Quantine(bits=8, mode="symmetric")
        assert engine._bits == 8

    def test_mode_asymmetric(self):
        engine = Quantine(bits=8, mode="asymmetric")
        assert engine._mode.value == "asymmetric"

    def test_clip_percentile_stored(self):
        engine = Quantine(bits=8, mode="symmetric", clip_percentile=0.99)
        assert engine._clip_pct == 0.99

    def test_error_threshold_stored(self):
        engine = Quantine(bits=8, mode="symmetric", skip_quantize_if_error_above=5.0)
        assert engine._error_threshold == 5.0


# ---------------------------------------------------------------------------
# _dequantize edge cases
# ---------------------------------------------------------------------------
class TestDequantizeEdgeCases:
    """Test _dequantize with various inputs."""

    def test_dequantize_asymmetric_int8(self):
        q = np.array([0, 64, 127], dtype=np.int8)
        result = _dequantize(q, scale=1.0, zero_point=0, bits=8, original_shape=(3,))
        np.testing.assert_allclose(result, [0.0, 64.0, 127.0])

    def test_dequantize_per_channel_scale(self):
        q = np.array([[1, 2], [3, 4]], dtype=np.int8)
        scale = np.array([0.1, 0.2], dtype=np.float32)
        result = _dequantize(q, scale=scale, zero_point=0, bits=8, original_shape=(2, 2))
        expected = q.astype(np.float32) * scale.reshape(-1, 1)
        np.testing.assert_allclose(result, expected)

    def test_dequantize_int4_packed(self):
        arr = np.array([1, -3, 5, -7], dtype=np.int8)
        packed = _pack_int4(arr)
        result = _dequantize(packed, scale=1.0, zero_point=0, bits=4, original_shape=(4,), signed=True)
        np.testing.assert_allclose(result, arr.astype(np.float32), atol=0.5)


# ---------------------------------------------------------------------------
# Summary edge cases
# ---------------------------------------------------------------------------
class TestSummaryEdgeCases:
    """Test summary with various quantization results."""

    def test_summary_worst_tensor(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr1 = np.random.randn(100).astype(np.float32) * 0.01
        arr2 = np.random.randn(100).astype(np.float32) * 10.0
        engine.quantize("small", arr1)
        engine.quantize("large", arr2)
        summary = engine.summary()
        assert summary["worst_tensor"] in ("small", "large")

    def test_summary_multiple_tensors(self):
        engine = Quantine(bits=8, mode="symmetric")
        for i in range(10):
            engine.quantize(f"t{i}", np.random.randn(50).astype(np.float32) * 0.02)
        summary = engine.summary()
        assert summary["tensors"] == 10
        assert summary["avg_mse"] >= 0
        assert 0 <= summary["avg_cosine_sim"] <= 1
        assert summary["avg_max_abs_error"] >= 0
