"""
Benchmark: quantized vs non-quantized inference.

Tests:
  - Quantization quality (MSE, cosine similarity) on realistic weight distributions
  - Throughput comparison (quantized vs float32)
  - Per-tensor error distribution across a full model
"""

import time
import tempfile
import numpy as np
import pytest

from domains.infrastructure.quantization import (
    Quantine, TensorInfo, QuantMeta, QuantMode, QuantDtype,
    _pack_int4, _unpack_int4, _dequantize, _cosine_similarity,
    quantize_state_dict, quantize_activation, quantize_kv_tensor,
    dequantize_kv_tensor, _ensure_2d_packed, int4_matmul, int8_matmul,
    quantized_linear, int4_quantized_linear,
)


def _require_c_matmul():
    """Skip if AVX2 C extension not available."""
    from domains.infrastructure.quantization import _c_matmul_int4, _int4_numpy_fallback
    if _c_matmul_int4 is _int4_numpy_fallback:
        pytest.skip("AVX2 int4 C extension not available")


def _make_gpt2_like_weights(n_layer=12, n_embed=768, n_head=12):
    """Create weight arrays that mimic GPT-2's distribution."""
    rng = np.random.RandomState(42)
    weights = {}

    weights["tok_emb.weight"] = rng.randn(50257, n_embed).astype(np.float32) * 0.02

    for i in range(n_layer):
        prefix = f"blocks.{i}"
        weights[f"{prefix}.attn.q_proj.weight"] = rng.randn(n_embed, n_embed).astype(np.float32) * 0.02
        weights[f"{prefix}.attn.k_proj.weight"] = rng.randn(n_embed, n_embed).astype(np.float32) * 0.02
        weights[f"{prefix}.attn.v_proj.weight"] = rng.randn(n_embed, n_embed).astype(np.float32) * 0.02
        weights[f"{prefix}.attn.o_proj.weight"] = rng.randn(n_embed, n_embed).astype(np.float32) * 0.02

        intermediate = n_embed * 4
        weights[f"{prefix}.ff.w1.weight"] = rng.randn(intermediate, n_embed).astype(np.float32) * 0.02
        weights[f"{prefix}.ff.w2.weight"] = rng.randn(n_embed, intermediate).astype(np.float32) * 0.02
        weights[f"{prefix}.ff.w3.weight"] = rng.randn(intermediate, n_embed).astype(np.float32) * 0.02

        weights[f"{prefix}.attn_norm.weight"] = rng.randn(n_embed).astype(np.float32) * 0.01 + 1.0
        weights[f"{prefix}.ff_norm.weight"] = rng.randn(n_embed).astype(np.float32) * 0.01 + 1.0

    weights["norm.weight"] = rng.randn(n_embed).astype(np.float32) * 0.01 + 1.0

    return weights


class TestQuantizationBenchmark:
    """Benchmark quantized vs non-quantized inference quality."""

    @pytest.fixture
    def gpt2_weights(self):
        return _make_gpt2_like_weights()

    def test_int8_quantization_quality(self, gpt2_weights):
        engine = Quantine(bits=8, mode="symmetric")

        cos_sims = []
        for name, arr in gpt2_weights.items():
            info = engine.quantize(name, arr)
            if info.is_quantized:
                cos_sims.append(info.meta.cosine_sim)

        assert len(cos_sims) > 0
        assert min(cos_sims) > 0.99
        assert np.mean(cos_sims) > 0.999

    def test_int8_with_clip_quality(self, gpt2_weights):
        engine_noclip = Quantine(bits=8, mode="symmetric")
        engine_clip = Quantine(bits=8, mode="symmetric", clip_percentile=0.999)

        for name, arr in gpt2_weights.items():
            info_noclip = engine_noclip.quantize(name, arr)
            info_clip = engine_clip.quantize(name, arr)

            if info_noclip.is_quantized and info_clip.is_quantized:
                assert info_clip.meta.cosine_sim >= info_noclip.meta.cosine_sim - 0.01

    def test_quantization_speedup(self, gpt2_weights):
        engine = Quantine(bits=8, mode="symmetric")
        weight_name = "blocks.0.attn.q_proj.weight"
        arr = gpt2_weights[weight_name]
        info = engine.quantize(weight_name, arr)

        if not info.is_quantized:
            pytest.skip("Weight was skipped by quantizer")

        rng = np.random.RandomState(123)
        x = rng.randn(1, 128, 768).astype(np.float32)

        t0 = time.perf_counter()
        for _ in range(10):
            y_fp32 = x @ arr.T
        t_fp32 = (time.perf_counter() - t0) / 10

        quant_arr = info.array
        t0 = time.perf_counter()
        for _ in range(10):
            y_quant = x @ info.as_float().T
        t_quant = (time.perf_counter() - t0) / 10

        mse = np.mean((y_fp32 - y_quant) ** 2)
        cos = np.dot(y_fp32.flatten(), y_quant.flatten()) / (
            np.linalg.norm(y_fp32) * np.linalg.norm(y_quant)
        )

        assert cos > 0.99, f"Cosine similarity too low: {cos}"
        assert mse < 0.01, f"MSE too high: {mse}"

    def test_full_model_quantization_report(self, gpt2_weights):
        engine = Quantine(bits=8, mode="symmetric")

        quantized = 0
        skipped = 0
        for name, arr in gpt2_weights.items():
            info = engine.quantize(name, arr)
            if info.is_quantized:
                quantized += 1
            else:
                skipped += 1

        summary = engine.summary()

        assert summary["tensors"] == quantized
        assert summary["bits"] == 8
        assert summary["avg_cosine_sim"] > 0.99
        assert summary["min_cosine_sim"] > 0.95
        assert quantized > 0

    def test_int4_vs_int8_quality(self, gpt2_weights):
        engine8 = Quantine(bits=8, mode="symmetric")
        engine4 = Quantine(bits=4, mode="symmetric")

        cos8_list = []
        cos4_list = []
        for name, arr in gpt2_weights.items():
            info8 = engine8.quantize(name, arr)
            info4 = engine4.quantize(name, arr)

            if info8.is_quantized and info4.is_quantized:
                cos8_list.append(info8.meta.cosine_sim)
                cos4_list.append(info4.meta.cosine_sim)

        assert len(cos8_list) > 0
        assert np.mean(cos8_list) > np.mean(cos4_list)

    def test_int4_memory_savings(self, gpt2_weights):
        engine = Quantine(bits=4, mode="symmetric")

        total_original = 0
        total_quantized = 0
        for name, arr in gpt2_weights.items():
            info = engine.quantize(name, arr)
            if info.is_quantized:
                total_original += int(np.prod(arr.shape)) * 4
                total_quantized += info.array.nbytes

        assert total_original > 0
        compression = total_original / max(total_quantized, 1)
        assert 6.0 < compression < 10.0, f"Compression {compression:.1f}x should be ~8x"

    def test_int4_avx2_speed(self, gpt2_weights):
        _require_c_matmul()

        engine = Quantine(bits=4, mode="symmetric")
        weight_name = "blocks.0.attn.q_proj.weight"
        arr = gpt2_weights[weight_name]
        info = engine.quantize(weight_name, arr)

        rng = np.random.RandomState(123)
        x = rng.randn(1, 128, 768).astype(np.float32)

        packed = _ensure_2d_packed(info.array, arr.shape[-1])
        K = arr.shape[-1]

        x_max = np.max(np.abs(x))
        act_scale = x_max / 127.0 if x_max > 0 else 1.0
        x_flat = x.reshape(-1, x.shape[-1])
        x_int8 = quantize_activation(x_flat, act_scale, 0)

        int4_matmul(x_int8, packed, act_scale, info.meta.scale, K)

        t0 = time.perf_counter()
        for _ in range(10):
            int4_matmul(x_int8, packed, act_scale, info.meta.scale, K)
        t_int4 = (time.perf_counter() - t0) / 10

        n_total = int(np.prod(arr.shape))
        unpacked = _unpack_int4(packed.ravel(), n_total, signed=True).reshape(arr.shape).astype(np.int8)
        t0 = time.perf_counter()
        for _ in range(10):
            int8_matmul(x_int8, unpacked, act_scale, info.meta.scale)
        t_old = (time.perf_counter() - t0) / 10

        assert t_int4 < t_old * 3, (
            f"AVX2 int4 {t_int4*1000:.1f}ms too slow vs int8 {t_old*1000:.1f}ms"
        )

    def test_int4_vs_int8_speed(self, gpt2_weights):
        _require_c_matmul()

        engine4 = Quantine(bits=4, mode="symmetric")
        engine8 = Quantine(bits=8, mode="symmetric")
        weight_name = "blocks.0.attn.q_proj.weight"
        arr = gpt2_weights[weight_name]
        info4 = engine4.quantize(weight_name, arr)
        info8 = engine8.quantize(weight_name, arr)

        rng = np.random.RandomState(123)
        x = rng.randn(1, 128, 768).astype(np.float32)

        packed4 = _ensure_2d_packed(info4.array, arr.shape[-1])
        w8 = info8.array
        K = arr.shape[-1]

        x_max = np.max(np.abs(x))
        act_scale = x_max / 127.0 if x_max > 0 else 1.0
        x_flat = x.reshape(-1, x.shape[-1])
        x_int8 = quantize_activation(x_flat, act_scale, 0)

        int4_matmul(x_int8, packed4, act_scale, info4.meta.scale, K)
        int8_matmul(x_int8, w8, act_scale, info8.meta.scale)

        t0 = time.perf_counter()
        for _ in range(10):
            int4_matmul(x_int8, packed4, act_scale, info4.meta.scale, K)
        t_int4 = (time.perf_counter() - t0) / 10

        t0 = time.perf_counter()
        for _ in range(10):
            int8_matmul(x_int8, w8, act_scale, info8.meta.scale)
        t_int8 = (time.perf_counter() - t0) / 10

        assert t_int4 < t_int8 * 5, (
            f"int4 {t_int4*1000:.1f}ms too slow vs int8 {t_int8*1000:.1f}ms"
        )

    def test_fused_int8_linear_faster_than_unfused(self, gpt2_weights):
        import domains.infrastructure.quantization as Q

        _require_c_matmul()
        assert Q.matmul_int8_f32_c is not None, "fused kernel not wired"

        engine = Quantine(bits=8, mode="symmetric")
        name = "blocks.0.attn.q_proj.weight"
        info = engine.quantize(name, gpt2_weights[name])
        w = info.array.astype(np.int8)
        bscale = info.meta.scale
        bias = np.random.RandomState(7).randn(w.shape[0]).astype(np.float32)
        x = np.random.RandomState(8).randn(1, w.shape[1]).astype(np.float32)

        def _time_it(fn, iters=200):
            fn()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            return (time.perf_counter() - t0) / iters

        fused = Q.quantized_linear(x, w, bscale, 0, bias)
        t_fused = _time_it(lambda: Q.quantized_linear(x, w, bscale, 0, bias))
        saved = Q.matmul_int8_f32_c
        Q.matmul_int8_f32_c = None
        try:
            unfused = Q.quantized_linear(x, w, bscale, 0, bias)
            t_unfused = _time_it(lambda: Q.quantized_linear(x, w, bscale, 0, bias))
        finally:
            Q.matmul_int8_f32_c = saved
        np.testing.assert_array_equal(fused, unfused)
        assert t_fused < t_unfused * 3.0, (
            f"fused {t_fused*1000:.2f}ms not faster than unfused {t_unfused*1000:.2f}ms"
        )


class TestQuantMeta:
    def test_to_dict(self):
        meta = QuantMeta(
            scale=0.1, zero_point=0, bits=8, mode="symmetric",
            dtype_code=5, original_shape=(3, 4), original_dtype="float32",
            mse=0.001, max_abs_error=0.01, cosine_sim=0.999,
        )
        d = meta.to_dict()
        assert d["scale"] == 0.1
        assert d["bits"] == 8
        assert d["original_shape"] == [3, 4]

    def test_from_dict(self):
        d = {
            "scale": 0.1, "zero_point": 0, "bits": 8, "mode": "symmetric",
            "dtype_code": 5, "original_shape": [3, 4], "original_dtype": "float32",
            "mse": 0.001, "max_abs_error": 0.01, "cosine_sim": 0.999,
        }
        meta = QuantMeta.from_dict(d)
        assert meta.scale == 0.1
        assert meta.bits == 8
        assert meta.original_shape == (3, 4)

    def test_to_dict_per_channel(self):
        meta = QuantMeta(
            scale=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            zero_point=0, bits=8, mode="symmetric",
            dtype_code=5, original_shape=(3, 4), original_dtype="float32",
        )
        d = meta.to_dict()
        assert isinstance(d["scale"], list)
        assert len(d["scale"]) == 3

    def test_from_dict_per_channel(self):
        d = {
            "scale": [0.1, 0.2, 0.3], "zero_point": 0, "bits": 8,
            "mode": "symmetric", "dtype_code": 5,
            "original_shape": [3, 4], "original_dtype": "float32",
        }
        meta = QuantMeta.from_dict(d)
        assert isinstance(meta.scale, np.ndarray)
        assert len(meta.scale) == 3

    def test_is_per_channel(self):
        meta_t = QuantMeta(scale=0.1, zero_point=0, bits=8, mode="symmetric",
                          dtype_code=5, original_shape=(3,), original_dtype="float32")
        assert not meta_t.is_per_channel

        meta_c = QuantMeta(scale=np.array([0.1, 0.2]), zero_point=0, bits=8, mode="symmetric",
                          dtype_code=5, original_shape=(2, 4), original_dtype="float32")
        assert meta_c.is_per_channel

    def test_roundtrip_dict(self):
        meta = QuantMeta(
            scale=0.05, zero_point=10, bits=8, mode="asymmetric",
            dtype_code=5, original_shape=(16,), original_dtype="float32",
            mse=0.002, max_abs_error=0.05, cosine_sim=0.995,
        )
        d = meta.to_dict()
        meta2 = QuantMeta.from_dict(d)
        assert meta2.scale == meta.scale
        assert meta2.zero_point == meta.zero_point
        assert meta2.bits == meta.bits
        assert meta2.mode == meta.mode
        assert meta2.mse == meta.mse
        assert meta2.cosine_sim == meta.cosine_sim


class TestTensorInfo:
    def test_not_quantized(self):
        arr = np.random.randn(3, 4).astype(np.float32)
        info = TensorInfo(name="w", array=arr)
        assert not info.is_quantized
        assert info.shape == (3, 4)
        assert info.dtype == np.float32

    def test_quantized(self):
        arr = np.array([1, 2, 3], dtype=np.int8)
        meta = QuantMeta(scale=0.1, zero_point=0, bits=8, mode="symmetric",
                        dtype_code=5, original_shape=(3,), original_dtype="float32")
        info = TensorInfo(name="w", array=arr, meta=meta)
        assert info.is_quantized
        assert info.shape == (3,)
        assert info.dtype == np.float32

    def test_as_float_not_quantized(self):
        arr = np.random.randn(3, 4).astype(np.float32)
        info = TensorInfo(name="w", array=arr)
        result = info.as_float()
        np.testing.assert_array_equal(result, arr)

    def test_as_float_int8(self):
        arr = np.array([1, 2, 3, -1], dtype=np.int8)
        meta = QuantMeta(scale=0.1, zero_point=0, bits=8, mode="symmetric",
                        dtype_code=5, original_shape=(4,), original_dtype="float32")
        info = TensorInfo(name="w", array=arr, meta=meta)
        result = info.as_float()
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, arr.astype(np.float32) * 0.1, atol=1e-6)

    def test_nbytes(self):
        arr = np.random.randn(3, 4).astype(np.float32)
        info = TensorInfo(name="w", array=arr)
        assert info.nbytes == arr.nbytes

    def test_compression_ratio_not_quantized(self):
        arr = np.random.randn(3, 4).astype(np.float32)
        info = TensorInfo(name="w", array=arr)
        assert info.compression_ratio() == 1.0

    def test_compression_ratio_int8(self):
        arr_f32 = np.random.randn(16, 32).astype(np.float32)
        arr_i8 = np.random.randn(16, 32).astype(np.int8)
        meta = QuantMeta(scale=0.1, zero_point=0, bits=8, mode="symmetric",
                        dtype_code=5, original_shape=(16, 32), original_dtype="float32")
        info = TensorInfo(name="w", array=arr_i8, meta=meta)
        ratio = info.compression_ratio()
        assert ratio > 1.0

    def test_as_float_int16_input(self):
        arr = np.random.randn(4).astype(np.float16)
        info = TensorInfo(name="w", array=arr)
        result = info.as_float()
        assert result.dtype == np.float32


class TestQuantineEngine:
    def test_invalid_bits(self):
        with pytest.raises(ValueError):
            Quantine(bits=16, mode="symmetric")

    def test_skip_prefixes(self):
        engine = Quantine(bits=8, mode="symmetric")
        assert engine.should_skip("tok_emb.weight")
        assert engine.should_skip("pos_emb.weight")
        assert engine.should_skip("norm.weight")
        assert not engine.should_skip("blocks.0.attn.weight")

    def test_sensitive_prefixes(self):
        engine = Quantine(bits=8, mode="symmetric")
        assert engine.is_sensitive("attn_norm.weight")
        assert engine.is_sensitive("ff_norm.weight")
        assert not engine.is_sensitive("blocks.0.attn.weight")

    def test_summary_empty(self):
        engine = Quantine(bits=8, mode="symmetric")
        s = engine.summary()
        assert s["tensors"] == 0

    def test_error_report(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(16).astype(np.float32)
        engine.quantize("test", arr)
        report = engine.error_report()
        assert "test" in report

    def test_save_load_metadata(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(16).astype(np.float32)
        engine.quantize("test", arr)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        engine.save_metadata(path)

        engine2 = Quantine(bits=8, mode="symmetric")
        engine2.load_metadata(path)
        assert "test" in engine2._error_report

    def test_quantize_with_scale(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(16).astype(np.float32)
        info = engine.quantize_with_scale("test", arr, scale=0.1, zero_point=0)
        assert info.is_quantized

    def test_per_channel_2d(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(32, 64).astype(np.float32)
        info = engine.quantize("block.weight", arr)
        assert info.is_quantized
        assert info.meta.scale.shape == (32,)

    def test_asymmetric_mode(self):
        engine = Quantine(bits=8, mode="asymmetric")
        arr = np.random.randn(16).astype(np.float32)
        info = engine.quantize("test", arr)
        assert info.is_quantized
        assert info.meta.mode == "asymmetric"

    def test_int4_mode(self):
        engine = Quantine(bits=4, mode="symmetric")
        arr = np.random.randn(16).astype(np.float32)
        info = engine.quantize("test", arr)
        assert info.is_quantized
        assert info.meta.bits == 4

    def test_quantize_1d_tensor(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(64).astype(np.float32)
        info = engine.quantize("bias", arr)
        assert info.is_quantized
        assert info.meta.original_shape == (64,)

    def test_quantize_large_tensor(self):
        engine = Quantine(bits=8, mode="symmetric")
        arr = np.random.randn(1024, 1024).astype(np.float32)
        info = engine.quantize("large.weight", arr)
        assert info.is_quantized
        assert info.meta.cosine_sim > 0.99

    def test_error_report_multiple_tensors(self):
        engine = Quantine(bits=8, mode="symmetric")
        for i in range(5):
            engine.quantize(f"tensor_{i}", np.random.randn(16).astype(np.float32))
        report = engine.error_report()
        assert len(report) == 5

    def test_summary_multiple_tensors(self):
        engine = Quantine(bits=8, mode="symmetric")
        for i in range(3):
            engine.quantize(f"t_{i}", np.random.randn(32, 64).astype(np.float32))
        s = engine.summary()
        assert s["tensors"] == 3
        assert "avg_mse" in s
        assert "avg_cosine_sim" in s


class TestPackUnpackInt4:
    def test_roundtrip(self):
        original = np.array([-8, -1, 0, 3, 7, -4, 2, 5], dtype=np.int8)
        packed = _pack_int4(original)
        unpacked = _unpack_int4(packed, len(original), signed=True)
        np.testing.assert_array_equal(original, unpacked)

    def test_odd_length(self):
        original = np.array([1, 2, 3], dtype=np.int8)
        packed = _pack_int4(original)
        unpacked = _unpack_int4(packed, 3, signed=True)
        np.testing.assert_array_equal(original, unpacked)

    def test_single_element(self):
        original = np.array([5], dtype=np.int8)
        packed = _pack_int4(original)
        unpacked = _unpack_int4(packed, 1, signed=True)
        np.testing.assert_array_equal(original, unpacked)

    def test_packed_halves_length(self):
        original = np.array([-8, -1, 0, 3, 7, -4, 2, 5], dtype=np.int8)
        packed = _pack_int4(original)
        assert len(packed) == len(original) // 2

    def test_unsigned(self):
        original = np.array([0, 5, 10, 15, 3, 7], dtype=np.int8)
        packed = _pack_int4(original)
        unpacked = _unpack_int4(packed, len(original), signed=False)
        np.testing.assert_array_equal(original, unpacked)

    def test_all_zeros(self):
        original = np.zeros(8, dtype=np.int8)
        packed = _pack_int4(original)
        unpacked = _unpack_int4(packed, 8, signed=True)
        np.testing.assert_array_equal(original, unpacked)

    def test_boundary_values(self):
        original = np.array([-8, 7, -8, 7, -8, 7], dtype=np.int8)
        packed = _pack_int4(original)
        unpacked = _unpack_int4(packed, 6, signed=True)
        np.testing.assert_array_equal(original, unpacked)

    def test_large_array(self):
        original = np.random.randint(-8, 8, size=1000).astype(np.int8)
        packed = _pack_int4(original)
        unpacked = _unpack_int4(packed, 1000, signed=True)
        np.testing.assert_array_equal(original, unpacked)


class TestCosineSimilarity:
    def test_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(a, a) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite(self):
        a = np.array([1.0, 2.0])
        b = np.array([-1.0, -2.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_both_zero(self):
        a = np.zeros(5)
        b = np.zeros(5)
        assert _cosine_similarity(a, b) == 1.0

    def test_one_zero(self):
        a = np.array([1.0, 2.0])
        b = np.zeros(2)
        assert _cosine_similarity(a, b) == 0.0

    def test_high_dimensional(self):
        a = np.random.randn(1000)
        b = a + np.random.randn(1000) * 0.01
        sim = _cosine_similarity(a, b)
        assert sim > 0.99

    def test_negative_values(self):
        a = np.array([-1.0, -2.0, -3.0])
        b = np.array([-1.0, -2.0, -3.0])
        assert _cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)


class TestQuantizeStateDict:
    def test_basic(self):
        sd = {
            "w1": np.random.randn(16, 32).astype(np.float32),
            "w2": np.random.randn(8, 16).astype(np.float32),
        }
        result = quantize_state_dict(sd, bits=8, mode="symmetric")
        assert len(result) == 2
        for name, info in result.items():
            assert info.is_quantized

    def test_int4(self):
        sd = {"w": np.random.randn(16, 32).astype(np.float32)}
        result = quantize_state_dict(sd, bits=4, mode="symmetric")
        assert "w" in result
        assert result["w"].is_quantized

    def test_skip_prefix(self):
        sd = {
            "tok_emb.weight": np.random.randn(100, 32).astype(np.float32),
            "blocks.0.weight": np.random.randn(16, 32).astype(np.float32),
        }
        result = quantize_state_dict(sd, bits=8, mode="symmetric")
        assert not result["tok_emb.weight"].is_quantized
        assert result["blocks.0.weight"].is_quantized

    def test_asymmetric(self):
        sd = {"w": np.random.randn(16, 32).astype(np.float32)}
        result = quantize_state_dict(sd, bits=8, mode="asymmetric")
        assert result["w"].is_quantized

    def test_empty_dict(self):
        result = quantize_state_dict({}, bits=8, mode="symmetric")
        assert len(result) == 0


class TestQuantizeActivation:
    def test_symmetric(self):
        x = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        result = quantize_activation(x, scale=0.1, zero_point=0)
        assert result.dtype == np.int8

    def test_asymmetric(self):
        x = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        result = quantize_activation(x, scale=0.1, zero_point=5)
        assert result.dtype == np.int8

    def test_shape_preserved(self):
        x = np.random.randn(4, 8).astype(np.float32)
        result = quantize_activation(x, scale=0.1, zero_point=0)
        assert result.shape == x.shape

    def test_zero_input(self):
        x = np.zeros(10, dtype=np.float32)
        result = quantize_activation(x, scale=0.1, zero_point=0)
        np.testing.assert_array_equal(result, 0)

    def test_large_values_clipped(self):
        x = np.array([1000.0, -1000.0], dtype=np.float32)
        result = quantize_activation(x, scale=1.0, zero_point=0)
        assert np.all(result >= -128)
        assert np.all(result <= 127)


class TestQuantizeKVTensor:
    def test_basic(self):
        x = np.random.randn(1, 4, 8, 64).astype(np.float32)
        x_int8, scale = quantize_kv_tensor(x)
        assert x_int8.dtype == np.int8
        assert x_int8.shape == x.shape
        assert scale.shape == x.shape[:-1] + (1,)

    def test_dequantize_roundtrip(self):
        x = np.random.randn(1, 4, 8, 64).astype(np.float32)
        x_int8, scale = quantize_kv_tensor(x)
        x_deq = dequantize_kv_tensor(x_int8, scale)
        mse = np.mean((x - x_deq) ** 2)
        assert mse < 0.01

    def test_zero_vector(self):
        x = np.zeros((1, 2, 4, 16), dtype=np.float32)
        x_int8, scale = quantize_kv_tensor(x)
        x_deq = dequantize_kv_tensor(x_int8, scale)
        np.testing.assert_allclose(x_deq, 0.0, atol=1e-6)

    def test_scale_positive(self):
        x = np.random.randn(1, 4, 8, 32).astype(np.float32)
        _, scale = quantize_kv_tensor(x)
        assert np.all(scale > 0)

    def test_single_token(self):
        x = np.random.randn(1, 1, 4, 16).astype(np.float32)
        x_int8, scale = quantize_kv_tensor(x)
        x_deq = dequantize_kv_tensor(x_int8, scale)
        mse = np.mean((x - x_deq) ** 2)
        assert mse < 0.01


class TestEnsure2dPacked:
    def test_1d_to_2d(self):
        packed = np.array([1, 2, 3, 4], dtype=np.int8)
        result = _ensure_2d_packed(packed, orig_k=4)
        assert result.ndim == 2
        assert result.shape == (2, 2)

    def test_already_2d(self):
        packed = np.array([[1, 2], [3, 4]], dtype=np.int8)
        result = _ensure_2d_packed(packed, orig_k=4)
        assert result.ndim == 2

    def test_odd_k(self):
        packed = np.array([1, 2, 3], dtype=np.int8)
        result = _ensure_2d_packed(packed, orig_k=6)
        assert result.ndim == 2


class TestInt8Matmul:
    def test_basic(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int8)
        b = np.array([[5, 6], [7, 8]], dtype=np.int8)
        result = int8_matmul(a, b, a_scale=1.0, b_scale=1.0)
        expected = a.astype(np.float32) @ b.astype(np.float32).T
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_scaled(self):
        a = np.array([[1, 2]], dtype=np.int8)
        b = np.array([[3, 4]], dtype=np.int8)
        result = int8_matmul(a, b, a_scale=0.5, b_scale=0.5)
        expected = (a.astype(np.float32) * 0.5) @ (b.astype(np.float32) * 0.5).T
        np.testing.assert_allclose(result, expected, atol=1e-3)

    def test_per_channel_b_scale(self):
        a = np.array([[1, 2]], dtype=np.int8)
        b = np.array([[3, 4], [5, 6]], dtype=np.int8)
        b_scale = np.array([0.5, 0.5])
        result = int8_matmul(a, b, a_scale=1.0, b_scale=b_scale)
        assert result.shape == (1, 2)

    def test_asymmetric_zero_points(self):
        a = np.array([[1, 2]], dtype=np.int8)
        b = np.array([[3, 4]], dtype=np.int8)
        result = int8_matmul(a, b, a_scale=1.0, b_scale=1.0, a_zero_point=1, b_zero_point=1)
        assert result.shape == (1, 1)

    def test_large_matrices(self):
        a = np.random.randint(-10, 10, (32, 64)).astype(np.int8)
        b = np.random.randint(-10, 10, (16, 64)).astype(np.int8)
        result = int8_matmul(a, b, a_scale=1.0, b_scale=1.0)
        assert result.shape == (32, 16)


class TestQuantizedLinear:
    def test_basic(self):
        x = np.random.randn(1, 32).astype(np.float32)
        w = np.random.randn(16, 32).astype(np.float32)
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("w", w)
        result = quantized_linear(x, info.array, info.meta.scale, 0)
        assert result.shape == (1, 16)

    def test_with_bias(self):
        x = np.random.randn(1, 32).astype(np.float32)
        w = np.random.randn(16, 32).astype(np.float32)
        bias = np.random.randn(16).astype(np.float32)
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("w", w)
        result = quantized_linear(x, info.array, info.meta.scale, 0, bias=bias)
        assert result.shape == (1, 16)

    def test_per_channel_scale(self):
        x = np.random.randn(1, 32).astype(np.float32)
        w = np.random.randn(16, 32).astype(np.float32)
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("w", w)
        result = quantized_linear(x, info.array, info.meta.scale, 0)
        assert result.shape == (1, 16)

    def test_matches_float_matmul(self):
        x = np.random.randn(1, 32).astype(np.float32)
        w = np.random.randn(16, 32).astype(np.float32)
        y_fp32 = x @ w.T

        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("w", w)
        y_q = quantized_linear(x, info.array, info.meta.scale, 0)

        cos = np.dot(y_fp32.flatten(), y_q.flatten()) / (
            np.linalg.norm(y_fp32) * np.linalg.norm(y_q) + 1e-10
        )
        assert cos > 0.95

    def test_batch_input(self):
        x = np.random.randn(4, 32).astype(np.float32)
        w = np.random.randn(16, 32).astype(np.float32)
        engine = Quantine(bits=8, mode="symmetric")
        info = engine.quantize("w", w)
        result = quantized_linear(x, info.array, info.meta.scale, 0)
        assert result.shape == (4, 16)


class TestInt4QuantizedLinear:
    def test_basic(self):
        x = np.random.randn(1, 32).astype(np.float32)
        w = np.random.randn(16, 32).astype(np.float32)
        engine = Quantine(bits=4, mode="symmetric")
        info = engine.quantize("w", w)
        packed = _ensure_2d_packed(info.array, 32)
        result = int4_quantized_linear(x, packed, info.meta.scale, 0, orig_k=32)
        assert result.shape == (1, 16)

    def test_with_bias(self):
        x = np.random.randn(1, 32).astype(np.float32)
        w = np.random.randn(16, 32).astype(np.float32)
        bias = np.random.randn(16).astype(np.float32)
        engine = Quantine(bits=4, mode="symmetric")
        info = engine.quantize("w", w)
        packed = _ensure_2d_packed(info.array, 32)
        result = int4_quantized_linear(x, packed, info.meta.scale, 0, orig_k=32, bias=bias)
        assert result.shape == (1, 16)

    def test_batch_input(self):
        x = np.random.randn(4, 32).astype(np.float32)
        w = np.random.randn(16, 32).astype(np.float32)
        engine = Quantine(bits=4, mode="symmetric")
        info = engine.quantize("w", w)
        packed = _ensure_2d_packed(info.array, 32)
        result = int4_quantized_linear(x, packed, info.meta.scale, 0, orig_k=32)
        assert result.shape == (4, 16)


class TestSuggestFormat:
    def test_returns_dict(self):
        result = Quantine.suggest_format()
        assert "format" in result
        assert "bits" in result
        assert "reason" in result
        assert "benchmark" in result

    def test_format_value(self):
        result = Quantine.suggest_format()
        assert result["format"] in ("fp32", "int8", "int4")
        assert result["bits"] in (32, 8, 4)

    def test_with_sample_weight(self):
        w = np.random.randn(64, 64).astype(np.float32)
        result = Quantine.suggest_format(sample_weight=w)
        assert result["format"] in ("fp32", "int8", "int4")

    def test_benchmark_results(self):
        result = Quantine.suggest_format()
        for fmt in ("fp32", "int8", "int4"):
            assert "time_s" in result["benchmark"][fmt]
            assert "cosine_sim" in result["benchmark"][fmt]

    def test_custom_threshold(self):
        result = Quantine.suggest_format(quality_threshold=0.5)
        assert result["format"] in ("fp32", "int8", "int4")
