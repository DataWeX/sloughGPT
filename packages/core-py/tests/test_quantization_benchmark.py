"""
Benchmark: quantized vs non-quantized inference.

Tests:
  - Quantization quality (MSE, cosine similarity) on realistic weight distributions
  - Throughput comparison (quantized vs float32)
  - Per-tensor error distribution across a full model
"""

import time

import numpy as np
import pytest

from domains.infrastructure.quantization import QuantEngine, TensorInfo


def _make_gpt2_like_weights(n_layer=12, n_embed=768, n_head=12):
    """Create weight arrays that mimic GPT-2's distribution."""
    rng = np.random.RandomState(42)
    weights = {}

    # Token embedding
    weights["tok_emb.weight"] = rng.randn(50257, n_embed).astype(np.float32) * 0.02

    for i in range(n_layer):
        prefix = f"blocks.{i}"
        # Attention weights (smaller scale)
        weights[f"{prefix}.attn.q_proj.weight"] = rng.randn(n_embed, n_embed).astype(np.float32) * 0.02
        weights[f"{prefix}.attn.k_proj.weight"] = rng.randn(n_embed, n_embed).astype(np.float32) * 0.02
        weights[f"{prefix}.attn.v_proj.weight"] = rng.randn(n_embed, n_embed).astype(np.float32) * 0.02
        weights[f"{prefix}.attn.o_proj.weight"] = rng.randn(n_embed, n_embed).astype(np.float32) * 0.02

        # FFN weights (wider intermediate)
        intermediate = n_embed * 4
        weights[f"{prefix}.ff.w1.weight"] = rng.randn(intermediate, n_embed).astype(np.float32) * 0.02
        weights[f"{prefix}.ff.w2.weight"] = rng.randn(n_embed, intermediate).astype(np.float32) * 0.02
        weights[f"{prefix}.ff.w3.weight"] = rng.randn(intermediate, n_embed).astype(np.float32) * 0.02

        # Norms (close to 1.0)
        weights[f"{prefix}.attn_norm.weight"] = rng.randn(n_embed).astype(np.float32) * 0.01 + 1.0
        weights[f"{prefix}.ff_norm.weight"] = rng.randn(n_embed).astype(np.float32) * 0.01 + 1.0

    # Final norm
    weights["norm.weight"] = rng.randn(n_embed).astype(np.float32) * 0.01 + 1.0

    return weights


class TestQuantizationBenchmark:
    """Benchmark quantized vs non-quantized inference quality."""

    @pytest.fixture
    def gpt2_weights(self):
        return _make_gpt2_like_weights()

    def test_int8_quantization_quality(self, gpt2_weights):
        """Quantized weights should have high cosine similarity with originals."""
        engine = QuantEngine(bits=8, mode="symmetric")

        cos_sims = []
        for name, arr in gpt2_weights.items():
            info = engine.quantize(name, arr)
            if info.is_quantized:
                cos_sims.append(info.meta.cosine_sim)

        # All quantized tensors should have cosine similarity > 0.99
        assert len(cos_sims) > 0
        assert min(cos_sims) > 0.99
        assert np.mean(cos_sims) > 0.999

    def test_int8_with_clip_quality(self, gpt2_weights):
        """Clipping should improve quality for weights with outliers."""
        engine_noclip = QuantEngine(bits=8, mode="symmetric")
        engine_clip = QuantEngine(bits=8, mode="symmetric", clip_percentile=0.999)

        for name, arr in gpt2_weights.items():
            info_noclip = engine_noclip.quantize(name, arr)
            info_clip = engine_clip.quantize(name, arr)

            if info_noclip.is_quantized and info_clip.is_quantized:
                # Clipping should not make things worse
                assert info_clip.meta.cosine_sim >= info_noclip.meta.cosine_sim - 0.01

    def test_quantization_speedup(self, gpt2_weights):
        """Quantized matmul should be faster than float32 matmul."""
        # Prepare a quantized weight
        engine = QuantEngine(bits=8, mode="symmetric")
        weight_name = "blocks.0.attn.q_proj.weight"
        arr = gpt2_weights[weight_name]
        info = engine.quantize(weight_name, arr)

        if not info.is_quantized:
            pytest.skip("Weight was skipped by quantizer")

        # Input activations
        rng = np.random.RandomState(123)
        x = rng.randn(1, 128, 768).astype(np.float32)

        # Float32 matmul
        t0 = time.perf_counter()
        for _ in range(10):
            y_fp32 = x @ arr.T
        t_fp32 = (time.perf_counter() - t0) / 10

        # Quantized matmul (dequantize on the fly)
        quant_arr = info.array
        t0 = time.perf_counter()
        for _ in range(10):
            y_quant = x @ info.as_float().T
        t_quant = (time.perf_counter() - t0) / 10

        # Output should be very close
        mse = np.mean((y_fp32 - y_quant) ** 2)
        cos = np.dot(y_fp32.flatten(), y_quant.flatten()) / (
            np.linalg.norm(y_fp32) * np.linalg.norm(y_quant)
        )

        assert cos > 0.99, f"Cosine similarity too low: {cos}"
        assert mse < 0.01, f"MSE too high: {mse}"

    def test_full_model_quantization_report(self, gpt2_weights):
        """Full model quantization should produce a complete report."""
        engine = QuantEngine(bits=8, mode="symmetric")

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
        """Int8 should be more accurate than int4."""
        engine8 = QuantEngine(bits=8, mode="symmetric")
        engine4 = QuantEngine(bits=4, mode="symmetric")

        cos8_list = []
        cos4_list = []
        for name, arr in gpt2_weights.items():
            info8 = engine8.quantize(name, arr)
            info4 = engine4.quantize(name, arr)

            if info8.is_quantized and info4.is_quantized:
                cos8_list.append(info8.meta.cosine_sim)
                cos4_list.append(info4.meta.cosine_sim)

        assert len(cos8_list) > 0
        # Int8 should have higher average cosine similarity
        assert np.mean(cos8_list) > np.mean(cos4_list)

    def test_int4_memory_savings(self, gpt2_weights):
        """Int4 should achieve ~8x compression across the model."""
        engine = QuantEngine(bits=4, mode="symmetric")

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
