"""
Tests for per-tensor quantization engine.

Tests cover:
  - Symmetric and asymmetric quantization
  - Outlier clipping (percentile-based)
  - Quantization error metrics (MSE, cosine similarity)
  - TensorInfo wrapper (quantized vs non-quantized)
  - WeightManager integration
  - Edge cases (zero-variance, single-element, very small tensors)
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from domains.infrastructure.quantization import (
    QuantEngine,
    QuantMeta,
    TensorInfo,
    _cosine_similarity,
    quantize_state_dict,
)


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

    def test_compression_ratio(self):
        meta = QuantMeta(
            scale=1.0, zero_point=0, bits=8, mode="symmetric",
            dtype_code=5, original_shape=(100,), original_dtype="float32",
        )
        quantized = np.zeros((100,), dtype=np.int8)
        info = TensorInfo(name="test", array=quantized, meta=meta)

        # float32 = 400 bytes, int8 = 100 bytes
        assert info.compression_ratio() == pytest.approx(4.0)


class TestQuantEngineSymmetric:
    """Test symmetric quantization mode."""

    def test_int8_symmetric_basic(self):
        engine = QuantEngine(bits=8, mode="symmetric")
        arr = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        info = engine.quantize("test", arr)

        assert info.is_quantized
        result = info.as_float()
        np.testing.assert_allclose(result, arr, atol=0.02)

    def test_int8_symmetric_scale(self):
        engine = QuantEngine(bits=8, mode="symmetric")
        arr = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
        info = engine.quantize("test", arr)

        assert info.meta.scale == pytest.approx(2.0 / 127, rel=0.01)
        assert info.meta.zero_point == 0

    def test_int8_symmetric_error_low(self):
        engine = QuantEngine(bits=8, mode="symmetric")
        arr = np.random.randn(1000).astype(np.float32) * 0.5
        info = engine.quantize("test", arr)

        assert info.meta.mse < 0.001
        assert info.meta.cosine_sim > 0.99

    def test_int4_symmetric(self):
        engine = QuantEngine(bits=4, mode="symmetric")
        arr = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        info = engine.quantize("test", arr)

        assert info.is_quantized
        assert info.meta.bits == 4


class TestQuantEngineAsymmetric:
    """Test asymmetric quantization mode."""

    def test_int8_asymmetric_positive_only(self):
        engine = QuantEngine(bits=8, mode="asymmetric")
        arr = np.random.randn(10000).astype(np.float32) * 2.0 + 10.0
        info = engine.quantize("test", arr)

        # Should be quantized (or skipped if error too high)
        if info.is_quantized:
            result = info.as_float()
            # Dequantized values should be in the right ballpark
            assert result.mean() == pytest.approx(arr.mean(), rel=0.1)
            assert info.meta.cosine_sim > 0.99

    def test_asymmetric_handles_shifted_distribution(self):
        # Asymmetric should at least produce valid quantization
        arr = np.random.randn(10000).astype(np.float32) * 2.0 + 10.0

        asym = QuantEngine(bits=8, mode="asymmetric")
        info = asym.quantize("test", arr)

        # Whether quantized or skipped, should not crash
        assert info is not None


class TestOutlierClipping:
    """Test percentile-based outlier clipping."""

    def test_clip_improves_main_distribution_accuracy(self):
        # Clipping computes scale from the clipped range (excluding outliers),
        # so the main distribution gets better resolution. The tradeoff is that
        # outliers get saturated. To verify the main distribution benefit,
        # we measure error only on the non-outlier values.
        np.random.seed(42)
        main_dist = np.random.randn(99000).astype(np.float32) * 1.0
        outliers = np.array([100.0, -100.0])
        arr = np.concatenate([main_dist, outliers])

        engine_noclip = QuantEngine(bits=8, mode="symmetric")
        engine_clip = QuantEngine(bits=8, mode="symmetric", clip_percentile=0.999)

        info_noclip = engine_noclip.quantize("test", arr)
        info_clip = engine_clip.quantize("test", arr)

        # Both should be quantized
        assert info_noclip.is_quantized, "no-clip quantization should apply"
        assert info_clip.is_quantized, "clip quantization should apply"

        # Clipped version should have smaller scale (better resolution)
        assert info_clip.meta.scale < info_noclip.meta.scale

        # Dequantize and measure error on main distribution only (non-outliers)
        deq_noclip = info_noclip.as_float().flatten()
        deq_clip = info_clip.as_float().flatten()
        main_mask = np.abs(arr) < 10  # non-outlier values

        mse_noclip_main = np.mean((arr[main_mask] - deq_noclip[main_mask]) ** 2)
        mse_clip_main = np.mean((arr[main_mask] - deq_clip[main_mask]) ** 2)

        # Clipped version should have lower MSE on the main distribution
        assert mse_clip_main < mse_noclip_main

    def test_clip_percentile_0_999(self):
        engine = QuantEngine(bits=8, mode="symmetric", clip_percentile=0.999)
        arr = np.random.randn(10000).astype(np.float32)
        info = engine.quantize("test", arr)

        assert info.is_quantized
        assert info.meta.mse < 0.001


class TestSkipSensitiveTensors:
    """Test that embedding and norm layers are skipped."""

    def test_skip_token_embedding(self):
        engine = QuantEngine(bits=8, mode="symmetric")
        arr = np.random.randn(50257, 768).astype(np.float32)
        info = engine.quantize("tok_emb.weight", arr)

        assert not info.is_quantized

    def test_skip_positional_embedding(self):
        engine = QuantEngine(bits=8, mode="symmetric")
        arr = np.random.randn(1024, 768).astype(np.float32)
        info = engine.quantize("pos_emb.weight", arr)

        assert not info.is_quantized

    def test_skip_final_norm(self):
        engine = QuantEngine(bits=8, mode="symmetric")
        arr = np.random.randn(768).astype(np.float32)
        info = engine.quantize("norm.weight", arr)

        assert not info.is_quantized

    def test_quantize_linear_weights(self):
        engine = QuantEngine(bits=8, mode="symmetric")
        arr = np.random.randn(768, 768).astype(np.float32) * 0.02
        info = engine.quantize("blocks.0.q_proj.weight", arr)

        assert info.is_quantized


class TestErrorMetrics:
    """Test quantization error metrics."""

    def test_perfect_quantization_has_zero_mse(self):
        engine = QuantEngine(bits=8, mode="symmetric")
        arr = np.zeros(100, dtype=np.float32)
        info = engine.quantize("test", arr)

        assert info.meta.mse == 0.0
        assert info.meta.cosine_sim == 1.0

    def test_cosine_similarity_range(self):
        engine = QuantEngine(bits=8, mode="symmetric")
        arr = np.random.randn(100).astype(np.float32)
        info = engine.quantize("test", arr)

        assert 0.0 <= info.meta.cosine_sim <= 1.0

    def test_error_report(self):
        engine = QuantEngine(bits=8, mode="symmetric")
        arr = np.random.randn(100).astype(np.float32)
        engine.quantize("tensor_a", arr)
        engine.quantize("tensor_b", arr)

        report = engine.error_report()
        assert "tensor_a" in report
        assert "tensor_b" in report
        assert "mse" in report["tensor_a"]

    def test_summary(self):
        engine = QuantEngine(bits=8, mode="symmetric")
        arr = np.random.randn(100).astype(np.float32)
        engine.quantize("tensor_a", arr)

        summary = engine.summary()
        assert summary["tensors"] == 1
        assert summary["bits"] == 8
        assert "avg_mse" in summary
        assert "avg_cosine_sim" in summary


class TestQuantizeStateDict:
    """Test quantizing an entire state dict."""

    def test_quantize_state_dict_basic(self):
        state_dict = {
            "blocks.0.q_proj.weight": np.random.randn(768, 768).astype(np.float32) * 0.02,
            "blocks.0.k_proj.weight": np.random.randn(768, 768).astype(np.float32) * 0.02,
            "tok_emb.weight": np.random.randn(50257, 768).astype(np.float32),
        }
        result = quantize_state_dict(state_dict, bits=8, mode="symmetric")

        # Linear weights quantized, embedding skipped
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


class TestMetadataSaveLoad:
    """Test saving and loading quantization metadata."""

    def test_save_load_roundtrip(self):
        engine = QuantEngine(bits=8, mode="symmetric")
        arr = np.random.randn(100).astype(np.float32)
        engine.quantize("test_tensor", arr)

        with tempfile.NamedTemporaryFile(suffix=".json", mode="w") as f:
            engine.save_metadata(f.name)

            engine2 = QuantEngine(bits=8, mode="symmetric")
            engine2.load_metadata(f.name)

            report = engine2.error_report()
            assert "test_tensor" in report
            assert report["test_tensor"]["mse"] == pytest.approx(
                engine.error_report()["test_tensor"]["mse"]
            )


class TestEdgeCases:
    """Test edge cases."""

    def test_zero_variance_tensor(self):
        engine = QuantEngine(bits=8, mode="symmetric")
        arr = np.zeros(100, dtype=np.float32)
        info = engine.quantize("test", arr)

        # Should still be quantized (scale=very small, all zeros)
        assert info.is_quantized

    def test_single_element(self):
        engine = QuantEngine(bits=8, mode="symmetric")
        arr = np.array([42.0], dtype=np.float32)
        info = engine.quantize("test", arr)

        assert info.is_quantized
        result = info.as_float()
        np.testing.assert_allclose(result, arr, atol=1.0)

    def test_multidimensional(self):
        engine = QuantEngine(bits=8, mode="symmetric")
        arr = np.random.randn(12, 64, 64).astype(np.float32) * 0.02
        info = engine.quantize("test", arr)

        assert info.is_quantized
        assert info.meta.original_shape == (12, 64, 64)

    def test_invalid_bits_raises(self):
        with pytest.raises(ValueError):
            QuantEngine(bits=16, mode="symmetric")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            QuantEngine(bits=8, mode="invalid")


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
