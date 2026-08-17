"""Tests for LoRA numpy inference bridge and HF LoRA fine-tuning."""

import os
import tempfile
import numpy as np
import pytest
from pathlib import Path

from domains.training.slonet import (
    SloTransformer, SloTransformerBlock, SloMultiHeadAttention,
    SloFeedForward, SloLinear, SloEmbedding, SloRMSNorm,
    Tensor, cross_entropy,
)
from domains.training.lora import (
    LoRALinear, LoRAEmbedding, LoRAConfig, LoRAType,
    apply_lora_to_model, get_lora_parameters, count_lora_parameters,
)


# ============================================================================
# Helpers
# ============================================================================


def _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2):
    """Create a tiny SloTransformer for testing."""
    model = SloTransformer(
        vocab_size=vocab_size,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        block_size=128,
    )
    return model


def _make_tiny_data(n_tokens=500, vocab_size=64):
    """Create random token data for testing."""
    return np.random.randint(0, vocab_size, size=(1, n_tokens), dtype=np.int64)


# ============================================================================
# LoRALinear.forward_numpy tests
# ============================================================================


class TestLoRALinearForwardNumpy:
    """Tests for LoRALinear.forward_numpy()."""

    def test_forward_numpy_matches_forward(self):
        """forward_numpy() and forward() should produce same output."""
        linear = SloLinear(32, 16, bias=True, name="test_linear")
        linear.weight.data[:] = np.random.randn(16, 32).astype(np.float32) * 0.1
        linear.bias.data[:] = np.zeros(16, dtype=np.float32)

        lora = LoRALinear(
            in_features=32, out_features=16, bias=True,
            rank=4, alpha=8.0, original_weight=linear.weight.data,
            original_bias=linear.bias.data,
        )
        lora.eval()

        x_np = np.random.randn(1, 32).astype(np.float32)
        out_np = lora.forward_numpy(x_np)

        x_t = Tensor(x_np, requires_grad=False)
        out_t = lora.forward(x_t)

        np.testing.assert_allclose(out_np, out_t.data, rtol=1e-5, atol=1e-5)

    def test_forward_numpy_zero_lora(self):
        """At init (B=0), LoRA output should equal base weight output."""
        w = np.random.randn(16, 32).astype(np.float32) * 0.1
        b = np.zeros(16, dtype=np.float32)

        lora = LoRALinear(
            in_features=32, out_features=16, bias=True,
            rank=4, alpha=8.0, original_weight=w, original_bias=b,
        )

        x = np.random.randn(1, 32).astype(np.float32)
        out_lora = lora.forward_numpy(x)
        out_base = x @ w.T + b

        np.testing.assert_allclose(out_lora, out_base, rtol=1e-5, atol=1e-5)

    def test_forward_numpy_with_lora_trained(self):
        """After modifying A/B, output should differ from base."""
        w = np.random.randn(16, 32).astype(np.float32) * 0.1
        b = np.zeros(16, dtype=np.float32)

        lora = LoRALinear(
            in_features=32, out_features=16, bias=True,
            rank=4, alpha=8.0, original_weight=w, original_bias=b,
        )

        # Modify LoRA params
        lora.lora_A.data[:] = np.random.randn(4, 32).astype(np.float32) * 0.1
        lora.lora_B.data[:] = np.random.randn(16, 4).astype(np.float32) * 0.1

        x = np.random.randn(1, 32).astype(np.float32)
        out_lora = lora.forward_numpy(x)
        out_base = x @ w.T + b

        # Should differ
        assert not np.allclose(out_lora, out_base, rtol=1e-3)

    def test_forward_numpy_ia3(self):
        """IA3 mode should apply element-wise scaling."""
        w = np.random.randn(16, 32).astype(np.float32) * 0.1

        lora = LoRALinear(
            in_features=32, out_features=16, bias=False,
            rank=4, alpha=8.0, lora_type=LoRAType.IA3,
            original_weight=w,
        )
        lora.lora_s.data[:] = 2.0  # double the output

        x = np.random.randn(1, 32).astype(np.float32)
        out = lora.forward_numpy(x)
        expected = (x @ w.T) * 2.0

        np.testing.assert_allclose(out, expected, rtol=1e-5)


# ============================================================================
# LoRAEmbedding.forward_numpy tests
# ============================================================================


class TestLoRAEmbeddingForwardNumpy:
    """Tests for LoRAEmbedding.forward_numpy()."""

    def test_forward_numpy_basic(self):
        """Embedding LoRA should produce valid output."""
        emb = LoRAEmbedding(
            num_embeddings=64, embedding_dim=32, rank=4, alpha=8.0,
        )

        x = np.array([0, 1, 2, 3], dtype=np.int64)
        out = emb.forward_numpy(x)

        assert out.shape == (4, 32)
        assert np.isfinite(out).all()

    def test_forward_numpy_zero_lora(self):
        """At init (B=0), embedding LoRA should equal base embedding."""
        emb = LoRAEmbedding(
            num_embeddings=64, embedding_dim=32, rank=4, alpha=8.0,
        )

        x = np.array([0, 1, 2, 3], dtype=np.int64)
        out_lora = emb.forward_numpy(x)
        out_base = emb.weight.weight.data[x]

        np.testing.assert_allclose(out_lora, out_base, rtol=1e-5, atol=1e-5)


# ============================================================================
# _has_lora flag tests
# ============================================================================


class TestHasLoraFlag:
    """Tests for the _has_lora flag on models."""

    def test_apply_lora_sets_flag(self):
        """apply_lora_to_model should set _has_lora=True on the model."""
        model = _make_tiny_model()
        assert not getattr(model, '_has_lora', False)

        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)

        assert getattr(model, '_has_lora', False) is True

    def test_model_without_lora_no_flag(self):
        """Model without LoRA should not have _has_lora flag."""
        model = _make_tiny_model()
        assert not getattr(model, '_has_lora', False)


# ============================================================================
# _generate_numpy_lora tests
# ============================================================================


class TestGenerateNumpyLora:
    """Tests for LoRA-aware generation path."""

    def test_generate_numpy_lora_basic(self):
        """generate_numpy should work with LoRA active."""
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)

        input_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        output = model.generate_numpy(input_ids, max_new_tokens=10, temperature=0.0)

        assert output.shape[0] == 1
        assert output.shape[1] == 15  # 5 prompt + 10 generated
        assert np.all(output[:, :5] == input_ids)

    def test_generate_numpy_lora_deterministic(self):
        """LoRA generation should be deterministic with greedy sampling."""
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)

        input_ids = np.array([[1, 2, 3]], dtype=np.int64)
        out1 = model.generate_numpy(input_ids, max_new_tokens=5, temperature=0.0)
        out2 = model.generate_numpy(input_ids, max_new_tokens=5, temperature=0.0)

        np.testing.assert_array_equal(out1, out2)


# ============================================================================
# LoRA parameter extraction tests
# ============================================================================


class TestLoRAParameters:
    """Tests for LoRA parameter extraction."""

    def test_get_lora_parameters(self):
        """get_lora_parameters should return only LoRA params."""
        model = _make_tiny_model()
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)

        params = get_lora_parameters(model)
        assert len(params) > 0
        for name in params:
            assert "lora_" in name

    def test_count_lora_parameters(self):
        """count_lora_parameters should return a positive number."""
        model = _make_tiny_model()
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)

        count = count_lora_parameters(model)
        assert count > 0


# ============================================================================
# Merge weights tests
# ============================================================================


class TestMergeWeights:
    """Tests for LoRA weight merging."""

    def test_merge_lora_into_base(self):
        """After merge, base weight should include LoRA contribution."""
        w = np.random.randn(16, 32).astype(np.float32) * 0.1
        lora = LoRALinear(
            in_features=32, out_features=16, bias=False,
            rank=4, alpha=8.0, original_weight=w,
        )

        lora.lora_A.data[:] = np.random.randn(4, 32).astype(np.float32) * 0.1
        lora.lora_B.data[:] = np.random.randn(16, 4).astype(np.float32) * 0.1

        # Compute expected merged weight
        lora_w = lora.lora_B.data @ lora.lora_A.data * (lora.alpha / lora.rank)
        expected_w = w + lora_w

        lora.merge_weights()

        np.testing.assert_allclose(lora.weight.data, expected_w, rtol=1e-5)
        np.testing.assert_allclose(lora.lora_A.data, 0.0, atol=1e-7)
        np.testing.assert_allclose(lora.lora_B.data, 0.0, atol=1e-7)


# ============================================================================
# Save/load adapter tests
# ============================================================================


class TestSaveLoadAdapter:
    """Tests for LoRA adapter save/load."""

    def test_save_load_roundtrip(self):
        """Save and load adapter should preserve weights."""
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)

        # Modify some LoRA weights
        for name, param in get_lora_parameters(model).items():
            if hasattr(param, 'data'):
                param.data[:] = np.random.randn(*param.data.shape).astype(np.float32) * 0.1

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "test_adapter.npz"
            adapter_dict = {}
            for name, param in get_lora_parameters(model).items():
                if hasattr(param, 'data'):
                    adapter_dict[name] = param.data
            np.savez_compressed(str(adapter_path), **adapter_dict)

            # Load into fresh model
            model2 = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
            model2 = apply_lora_to_model(model2, config)
            adapter = np.load(str(adapter_path))
            for name, param in get_lora_parameters(model2).items():
                if name in adapter and hasattr(param, 'data'):
                    param.data[:] = adapter[name]

            # Verify weights match
            for name in get_lora_parameters(model):
                p1 = get_lora_parameters(model)[name]
                p2 = get_lora_parameters(model2)[name]
                np.testing.assert_allclose(p1.data, p2.data, rtol=1e-6)


# ============================================================================
# Forward pass through full model with LoRA
# ============================================================================


class TestFullModelForward:
    """Tests for full model forward pass with LoRA."""

    def test_forward_pass_with_lora(self):
        """Full model forward should work with LoRA layers."""
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)

        x = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        logits, _ = model.forward(Tensor(x))
        logits = logits.data

        assert logits.shape == (1, 5, 64)
        assert np.isfinite(logits).all()

    def test_forward_numpy_path_with_lora(self):
        """Non-inlined forward_numpy should work with LoRA."""
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)

        # Run through the non-inlined path
        x = np.random.randn(1, 5, 32).astype(np.float32)
        block = model.layers[2]  # first SloTransformerBlock
        out, (k, v) = block.forward_numpy(x)

        assert out.shape == (1, 5, 32)
        assert np.isfinite(out).all()
