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

from domains.infrastructure.quantization import QuantEngine
from domains.training.slonet import SloTransformer


def _walk_linear_layers(model):
    """Walk SloLinear layers in a SloTransformer.

    SloTransformer stores blocks as a plain Python list, so named_modules()
    doesn't recurse. We walk manually.
    """
    layers = {}

    # Transformer blocks
    if hasattr(model, 'blocks'):
        for i, block in enumerate(model.blocks):
            if hasattr(block, 'attn'):
                for proj_name, proj_attr in [('W_q', 'W_q'), ('W_k', 'W_k'), ('W_v', 'W_v'), ('W_o', 'W_o')]:
                    p = getattr(block.attn, proj_attr, None)
                    if p is not None and hasattr(p, 'forward_numpy'):
                        layers[f'blocks.{i}.attn.{proj_name}'] = p
            if hasattr(block, 'ff'):
                for proj_name, proj_attr in [('w1', 'w1'), ('w2', 'w2'), ('w3', 'w3')]:
                    p = getattr(block.ff, proj_attr, None)
                    if p is not None and hasattr(p, 'forward_numpy'):
                        layers[f'blocks.{i}.ff.{proj_name}'] = p

    return layers


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
        assert cosine > 0.90, f"int4 cosine={cosine} — degraded too much"

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
