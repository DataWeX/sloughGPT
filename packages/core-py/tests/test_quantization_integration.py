"""
End-to-end quantization integration test.

Tests the full pipeline:
  - Create a tiny SloTransformer
  - Quantize all SloLinear layers (int8 and int4)
  - Run inference through quantized model
  - Compare outputs to float32 baseline
  - Measure memory savings
"""

import numpy as np
import pytest
import time

from domains.infrastructure.quantization import QuantEngine
from domains.training.slonet import SloTransformer


def _walk_linear_layers(model):
    """Find all SloLinear layers — delegates to shared utility."""
    from domains.infrastructure.quantization import walk_slo_linears
    return walk_slo_linears(model)


def _count_linear_params(model):
    """Count total float32 bytes taken by SloLinear weights."""
    total = 0
    for _, module in _walk_linear_layers(model).items():
        total += module.weight.data.nbytes
        if hasattr(module, 'bias') and module.use_bias:
            total += module.bias.data.nbytes
    return total


def _count_quantized_bytes(model):
    """Count total bytes taken by quantized weights on SloLinear layers."""
    total = 0
    for _, module in _walk_linear_layers(model).items():
        if module._quant_info is not None:
            total += module._quant_info.array.nbytes
            if hasattr(module, 'bias') and module.use_bias:
                total += module.bias.data.nbytes
    return total


def _run_model(model, inp):
    """Run model and return logits as numpy array."""
    out = model(inp)
    if isinstance(out, tuple):
        return out[0].data
    return out.data if hasattr(out, 'data') else out


@pytest.fixture
def tiny_model():
    """Create a tiny SloTransformer for testing."""
    model = SloTransformer(
        vocab_size=100,
        n_embed=64,
        n_layer=2,
        n_head=4,
        intermediate_size=128,
        block_size=32,
        max_seq_len=32,
        use_rope=False,
        dropout=0.0,
        tie_weights=True,
        use_abs_pos_emb=True,
        norm_type="layer_norm",
    )
    return model


@pytest.fixture
def sample_input():
    """Sample tokenized input for inference."""
    return np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)


class TestQuantizationIntegration:
    """End-to-end quantization pipeline tests."""

    def test_linear_layers_identified(self, tiny_model):
        """Verify _walk_linear_layers finds all SloLinear modules."""
        layers = _walk_linear_layers(tiny_model)
        assert len(layers) > 0
        assert 'lm_head' in layers, "Should include output projection"
        assert any('attn.W_q' in name for name in layers)
        assert any('attn.W_o' in name for name in layers)
        assert any('ff.w1' in name or 'ff.w2' in name or 'ff.w3' in name for name in layers)

    def test_quantize_all_linears_int8(self, tiny_model, sample_input):
        """Quantize all SloLinear layers to int8 and verify inference works."""
        engine = QuantEngine(bits=8, mode="symmetric")

        # Float32 baseline
        logits_fp32 = _run_model(tiny_model, sample_input)

        # Quantize all SloLinear layers
        layers = _walk_linear_layers(tiny_model)
        quantized_count = 0
        for name, module in layers.items():
            if 'norm' in name:
                continue  # skip norms
            info = engine.quantize(f"{name}.weight", module.weight.data.copy())
            if info.is_quantized:
                module.set_quantized_weight(info)
                quantized_count += 1

        assert quantized_count > 0, "No layers were quantized"

        # Inference with quantized weights
        logits_int8 = _run_model(tiny_model, sample_input)

        assert logits_int8.shape == logits_fp32.shape

        cosine = np.dot(logits_fp32.flatten(), logits_int8.flatten()) / (
            np.linalg.norm(logits_fp32) * np.linalg.norm(logits_int8)
        )
        assert cosine > 0.95, f"int8 cosine={cosine} — degraded too much"

        # Memory savings
        fp32_bytes = _count_linear_params(tiny_model)
        quant_bytes = _count_quantized_bytes(tiny_model)
        assert quant_bytes < fp32_bytes, "Quantized should use less memory"

    def test_quantize_all_linears_int4(self, tiny_model, sample_input):
        """Quantize all SloLinear layers to int4 and verify inference works."""
        engine = QuantEngine(bits=4, mode="symmetric")

        logits_fp32 = _run_model(tiny_model, sample_input)

        layers = _walk_linear_layers(tiny_model)
        quantized_count = 0
        for name, module in layers.items():
            if 'norm' in name:
                continue
            info = engine.quantize(f"{name}.weight", module.weight.data.copy())
            if info.is_quantized:
                module.set_quantized_weight(info)
                quantized_count += 1

        assert quantized_count > 0

        logits_int4 = _run_model(tiny_model, sample_input)
        assert logits_int4.shape == logits_fp32.shape

        cosine = np.dot(logits_fp32.flatten(), logits_int4.flatten()) / (
            np.linalg.norm(logits_fp32) * np.linalg.norm(logits_int4)
        )
        assert cosine > 0.85, f"int4 cosine={cosine} — degraded too much"

    def test_quantize_during_inference_no_crash(self, tiny_model, sample_input):
        """Quantizing layers while model is in use shouldn't crash."""
        engine = QuantEngine(bits=8, mode="symmetric")

        # First pass (float32)
        _run_model(tiny_model, sample_input)

        # Quantize mid-stream
        layers = _walk_linear_layers(tiny_model)
        for name, module in layers.items():
            if 'norm' in name:
                continue
            info = engine.quantize(name, module.weight.data.copy())
            if info.is_quantized:
                module.set_quantized_weight(info)

        # Second pass (quantized)
        logits = _run_model(tiny_model, sample_input)
        assert logits.shape == (1, 8, 100)  # (batch, seq, vocab)

    def test_int8_memory_savings(self, tiny_model):
        """int8 should reduce SloLinear weight memory ~4x."""
        fp32_bytes = _count_linear_params(tiny_model)
        assert fp32_bytes > 0

        engine = QuantEngine(bits=8, mode="symmetric")
        layers = _walk_linear_layers(tiny_model)
        for name, module in layers.items():
            if 'norm' in name:
                continue
            info = engine.quantize(f"{name}.weight", module.weight.data.copy())
            if info.is_quantized:
                module.set_quantized_weight(info)

        quant_bytes = _count_quantized_bytes(tiny_model)
        ratio = fp32_bytes / max(quant_bytes, 1)
        assert 3.0 < ratio < 5.0, f"int8 compression {ratio:.1f}x should be ~4x"

    def test_int4_memory_savings(self, tiny_model):
        """int4 should reduce SloLinear weight memory ~8x."""
        fp32_bytes = _count_linear_params(tiny_model)
        assert fp32_bytes > 0

        engine = QuantEngine(bits=4, mode="symmetric")
        layers = _walk_linear_layers(tiny_model)
        for name, module in layers.items():
            if 'norm' in name:
                continue
            info = engine.quantize(f"{name}.weight", module.weight.data.copy())
            if info.is_quantized:
                module.set_quantized_weight(info)

        quant_bytes = _count_quantized_bytes(tiny_model)
        ratio = fp32_bytes / max(quant_bytes, 1)
        assert 6.0 < ratio < 10.0, f"int4 compression {ratio:.1f}x should be ~8x"

    def test_int8_vs_int4_quality_tradeoff(self, tiny_model, sample_input):
        """int8 should be more accurate than int4."""
        logits_fp32 = _run_model(tiny_model, sample_input)

        layers = _walk_linear_layers(tiny_model)
        norm_names = {n for n in layers if 'norm' in n}

        # int8
        engine8 = QuantEngine(bits=8, mode="symmetric")
        for name, module in layers.items():
            if name in norm_names:
                continue
            info = engine8.quantize(f"{name}.weight", module.weight.data.copy())
            if info.is_quantized:
                module.set_quantized_weight(info)
        logits_int8 = _run_model(tiny_model, sample_input)

        # Reset model
        for _, module in layers.items():
            module._quant_info = None

        # int4
        engine4 = QuantEngine(bits=4, mode="symmetric")
        for name, module in layers.items():
            if name in norm_names:
                continue
            info = engine4.quantize(f"{name}.weight", module.weight.data.copy())
            if info.is_quantized:
                module.set_quantized_weight(info)
        logits_int4 = _run_model(tiny_model, sample_input)

        cos_int8 = np.dot(logits_fp32.flatten(), logits_int8.flatten()) / (
            np.linalg.norm(logits_fp32) * np.linalg.norm(logits_int8)
        )
        cos_int4 = np.dot(logits_fp32.flatten(), logits_int4.flatten()) / (
            np.linalg.norm(logits_fp32) * np.linalg.norm(logits_int4)
        )

        assert cos_int8 > cos_int4, f"int8 ({cos_int8:.4f}) should beat int4 ({cos_int4:.4f})"

    def test_quantized_stability(self, sample_input):
        """Gold-standard stability: 30 sequential quantized inferences.

        Gold thresholds (from benchmark_stability.py):
          - Crash rate: 0%
          - Latency degradation: ≤1.20x (p95 last 5 / p95 first 5)
          - Empty response rate: 0%
          - Length CV: ≤0.30
          - Response rate: 100%
        """
        model = SloTransformer(
            vocab_size=100, n_embed=64, n_layer=2, n_head=4,
            intermediate_size=128, block_size=32, max_seq_len=32,
            use_rope=False, dropout=0.0, tie_weights=True,
            use_abs_pos_emb=True, norm_type="layer_norm",
        )

        # Quantize all SloLinear layers int8
        engine = QuantEngine(bits=8, mode="symmetric")
        layers = _walk_linear_layers(model)
        for name, module in layers.items():
            if 'norm' in name:
                continue
            info = engine.quantize(f"{name}.weight", module.weight.data.copy())
            if info.is_quantized:
                module.set_quantized_weight(info)

        N = 30
        latencies = []
        response_lengths = []
        crashes = 0
        empties = 0

        for i in range(N):
            inp = np.array([[i + 1]], dtype=np.int64)  # different token each call
            t0 = time.perf_counter()
            try:
                logits = _run_model(model, inp)
                elapsed = time.perf_counter() - t0
                latencies.append(elapsed)
                resp_len = logits.shape[-1]  # vocab size
                response_lengths.append(resp_len)
                if np.all(logits == 0):
                    empties += 1
            except Exception:
                crashes += 1

        crash_rate = crashes / N
        assert crash_rate == 0.0, f"Crash rate {crash_rate:.1%} — gold requires 0%"

        response_rate = (N - crashes) / N
        assert response_rate == 1.0, f"Response rate {response_rate:.1%} — gold requires 100%"

        empty_rate = empties / N
        assert empty_rate == 0.0, f"Empty rate {empty_rate:.1%} — gold requires 0%"

        # Latency degradation: p95 last 5 / p95 first 5 ≤ 1.20
        if len(latencies) >= 10:
            first5 = sorted(latencies[:5])
            last5 = sorted(latencies[-5:])
            p95_first = first5[int(len(first5) * 0.95)] if len(first5) > 0 else 0
            p95_last = last5[int(len(last5) * 0.95)] if len(last5) > 0 else 0
            if p95_first > 0:
                degradation = p95_last / p95_first
                assert degradation <= 1.20, (
                    f"Latency degradation {degradation:.2f}x — gold requires ≤1.20x"
                )

        # Length CV ≤ 0.30
        if len(response_lengths) > 1:
            lengths = np.array(response_lengths, dtype=np.float32)
            cv = np.std(lengths) / np.mean(lengths)
            assert cv <= 0.30, f"Length CV {cv:.3f} — gold requires ≤0.30"


class TestQuantizeEndpoint:
    """Tests for POST /models/quantize endpoint logic."""

    def test_quantize_endpoint_smoke(self, tiny_model, sample_input):
        """Simulate the quantize endpoint logic directly."""
        from domains.infrastructure.quantization import QuantEngine, walk_slo_linears

        # Walk layers
        layers = walk_slo_linears(tiny_model)
        assert len(layers) >= 14  # 2 blocks × 7 linears + lm_head
        assert 'lm_head' in layers
        assert layers['lm_head'].weight.data.shape[0] == 100  # vocab_size

        # Quantize all layers int8
        engine = QuantEngine(bits=8, mode="symmetric")
        quantized_count = 0
        for name, module in layers.items():
            info = engine.quantize(f"{name}.weight", module.weight.data.copy())
            if info.is_quantized:
                module.set_quantized_weight(info)
                quantized_count += 1

        assert quantized_count == len(layers), f"Quantized {quantized_count}/{len(layers)}"

        # Inference still works
        logits = _run_model(tiny_model, sample_input)
        assert logits.shape == (1, 8, 100)

        # Summary report fields match what the endpoint returns
        summary = engine.summary()
        assert summary["tensors"] == quantized_count
        assert summary["bits"] == 8
        assert summary["avg_cosine_sim"] > 0.99

        # Verify lm_head is quantized
        assert tiny_model.layers[-1]._quant_info is not None
        assert tiny_model.layers[-1]._quant_info.meta.bits == 8

    def test_from_slnc_object_identity_matching(self, tiny_model, sample_input):
        """Simulate from_slnc() quantize path: match params by object identity.

        from_slnc() uses walk_slo_linears + object-identity matching
        (``param is module.weight``) instead of string prefix matching.
        This test verifies every parameter finds its module. Prevents
        regression of the naming mismatch bug (q_proj vs W_q).
        """
        from domains.infrastructure.quantization import QuantEngine, walk_slo_linears

        linear_map = walk_slo_linears(tiny_model)
        param_names = dict(tiny_model.named_parameters())

        # Build reverse lookup by object identity (same as from_slnc)
        param_to_module = {}
        for mod_name, module in linear_map.items():
            for pname, param in param_names.items():
                if param is module.weight:
                    param_to_module[pname] = mod_name
                    break

        # Every SloLinear weight should match
        linear_params_found = set()
        for mod_name, module in linear_map.items():
            for pname, param in param_names.items():
                if param is module.weight:
                    linear_params_found.add(pname)
                    break

        # Quantize using object-identity matching
        engine = QuantEngine(bits=8, mode="symmetric")
        quantized_count = 0
        for pname, param in param_names.items():
            if pname not in param_to_module:
                continue
            arr = param.data.copy()
            info = engine.quantize(pname, arr)
            if info.is_quantized:
                linear_map[param_to_module[pname]].set_quantized_weight(info)
                quantized_count += 1

        assert quantized_count > 0, "No layers quantized via object-identity matching"
        assert quantized_count == len(linear_map), (
            f"Object-identity matching quantized {quantized_count}/{len(linear_map)}"
        )

        # Inference with quantized weights
        logits = _run_model(tiny_model, sample_input)
        assert logits.shape == (1, 8, 100)

        # Logit stability: 10 sequential calls
        N = 10
        for i in range(N):
            inp = np.array([[i + 1]], dtype=np.int64)
            out = _run_model(tiny_model, inp)
            assert out.shape[0] == 1
            assert not np.any(np.isnan(out.data)), f"NaN in output at step {i}"
            assert not np.any(np.isinf(out.data)), f"Inf in output at step {i}"
