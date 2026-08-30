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
    _walk_slo_tree, _set_nested,
)
from domains.training.hf_lora_finetune import (
    HFLoraConfig, HFLoraTrainer, _LoRADataset,
    load_lora_adapter, merge_lora_adapter,
)
from domains.training.trainer_protocol import TrainResult


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
# LoRAConfig / LoRAType tests
# ============================================================================


class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_default_config(self):
        config = LoRAConfig()
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.dropout == 0.05
        assert config.lora_type == LoRAType.LORA
        assert config.bias == "none"

    def test_default_target_modules(self):
        config = LoRAConfig(target_modules=None)
        assert isinstance(config.target_modules, list)
        assert len(config.target_modules) > 0

    def test_custom_config(self):
        config = LoRAConfig(rank=4, alpha=8.0, dropout=0.1)
        assert config.rank == 4
        assert config.alpha == 8.0
        assert config.dropout == 0.1

    def test_ia3_type(self):
        config = LoRAConfig(lora_type=LoRAType.IA3)
        assert config.lora_type == LoRAType.IA3

    def test_lora_type_enum_values(self):
        assert LoRAType.LORA.value == "lora"
        assert LoRAType.LORA_PLUS.value == "lora_plus"
        assert LoRAType.IA3.value == "ia3"


# ============================================================================
# _LoRADataset tests
# ============================================================================


class TestLoRADataset:
    """Tests for the _LoRADataset class."""

    def test_length(self):
        data = np.arange(100, dtype=np.int64)
        ds = _LoRADataset(data, block_size=32)
        assert len(ds) == 68  # 100 - 32

    def test_length_shorter_than_block(self):
        data = np.arange(10, dtype=np.int64)
        ds = _LoRADataset(data, block_size=32)
        assert len(ds) == 0

    def test_length_equal_to_block(self):
        data = np.arange(32, dtype=np.int64)
        ds = _LoRADataset(data, block_size=32)
        assert len(ds) == 0

    def test_getitem_shapes(self):
        data = np.arange(100, dtype=np.int64)
        ds = _LoRADataset(data, block_size=32)
        x, y = ds[0]
        assert x.shape == (32,)
        assert y.shape == (32,)

    def test_getitem_x_y_shifted(self):
        data = np.arange(100, dtype=np.int64)
        ds = _LoRADataset(data, block_size=32)
        x, y = ds[0]
        np.testing.assert_array_equal(x, data[:32])
        np.testing.assert_array_equal(y, data[1:33])

    def test_getitem_different_indices(self):
        data = np.arange(100, dtype=np.int64)
        ds = _LoRADataset(data, block_size=10)
        x0, y0 = ds[0]
        x1, y1 = ds[1]
        assert not np.array_equal(x0, x1)

    def test_getitem_last_valid_index(self):
        data = np.arange(50, dtype=np.int64)
        ds = _LoRADataset(data, block_size=10)
        x, y = ds[39]  # last valid: 50 - 10 = 40, index 39
        assert x.shape == (10,)

    def test_input_converted_to_ndarray(self):
        data = [1, 2, 3, 4, 5]
        ds = _LoRADataset(data, block_size=3)
        assert isinstance(ds.data, np.ndarray)

    def test_empty_data(self):
        ds = _LoRADataset(np.array([], dtype=np.int64), block_size=10)
        assert len(ds) == 0


# ============================================================================
# HFLoraConfig tests
# ============================================================================


class TestHFLoraConfig:
    """Tests for HFLoraConfig dataclass."""

    def test_default_config(self):
        config = HFLoraConfig()
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.epochs == 3
        assert config.batch_size == 8
        assert config.block_size == 128
        assert config.learning_rate == 1e-4

    def test_adapter_name_auto_generated(self):
        config = HFLoraConfig(model_path="models/gpt2.slnc")
        assert config.adapter_name == "gpt2_lora_r8"

    def test_adapter_name_custom(self):
        config = HFLoraConfig(model_path="models/gpt2.slnc", adapter_name="my_adapter")
        assert config.adapter_name == "my_adapter"

    def test_adapter_name_with_rank(self):
        config = HFLoraConfig(model_path="models/llama.slnc", rank=16)
        assert "r16" in config.adapter_name

    def test_default_target_modules(self):
        config = HFLoraConfig()
        assert "W_q" in config.target_modules
        assert "W_k" in config.target_modules
        assert "W_v" in config.target_modules
        assert "W_o" in config.target_modules

    def test_weight_decay(self):
        config = HFLoraConfig(weight_decay=0.05)
        assert config.weight_decay == 0.05

    def test_grad_clip(self):
        config = HFLoraConfig(grad_clip=0.5)
        assert config.grad_clip == 0.5

    def test_warmup_steps(self):
        config = HFLoraConfig(warmup_steps=100)
        assert config.warmup_steps == 100

    def test_log_interval(self):
        config = HFLoraConfig(log_interval=5)
        assert config.log_interval == 5

    def test_progress_callback_none(self):
        config = HFLoraConfig()
        assert config.progress_callback is None

    def test_grad_accumulation_steps(self):
        config = HFLoraConfig(grad_accumulation_steps=4)
        assert config.grad_accumulation_steps == 4


# ============================================================================
# HFLoraTrainer tests (without load_model / train — no external files)
# ============================================================================


class TestHFLoraTrainer:
    """Tests for HFLoraTrainer that don't require model/data files."""

    def test_init(self):
        config = HFLoraConfig(model_path="m.slnc")
        trainer = HFLoraTrainer(config)
        assert trainer.config is config
        assert trainer.model is None
        assert trainer.lora_params == {}

    def test_is_training_default(self):
        config = HFLoraConfig(model_path="m.slnc")
        trainer = HFLoraTrainer(config)
        assert not trainer.is_training

    def test_stop(self):
        config = HFLoraConfig(model_path="m.slnc")
        trainer = HFLoraTrainer(config)
        trainer.stop()
        assert not trainer.is_training

    def test_apply_lora_without_model_raises(self):
        config = HFLoraConfig(model_path="m.slnc")
        trainer = HFLoraTrainer(config)
        with pytest.raises(RuntimeError, match="Model not loaded"):
            trainer.apply_lora()

    def test_load_model_nonexistent_raises(self):
        config = HFLoraConfig(model_path="/nonexistent/model.slnc")
        trainer = HFLoraTrainer(config)
        with pytest.raises(FileNotFoundError):
            trainer.load_model()

    def test_cancel_event_none(self):
        config = HFLoraConfig()
        trainer = HFLoraTrainer(config)
        assert trainer._training_thread is None

    def test_stop_without_cancel_event(self):
        config = HFLoraConfig()
        trainer = HFLoraTrainer(config)
        trainer.stop()
        assert not trainer.is_training


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

    def test_forward_numpy_no_bias(self):
        """Forward without bias should not add bias term."""
        w = np.random.randn(16, 32).astype(np.float32) * 0.1
        lora = LoRALinear(
            in_features=32, out_features=16, bias=False,
            rank=4, alpha=8.0, original_weight=w,
        )
        x = np.random.randn(1, 32).astype(np.float32)
        out = lora.forward_numpy(x)
        expected = x @ w.T
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    def test_forward_numpy_batch(self):
        """Forward should handle batch size > 1."""
        w = np.random.randn(16, 32).astype(np.float32) * 0.1
        lora = LoRALinear(
            in_features=32, out_features=16, bias=True,
            rank=4, alpha=8.0, original_weight=w,
        )
        x = np.random.randn(8, 32).astype(np.float32)
        out = lora.forward_numpy(x)
        assert out.shape == (8, 16)

    def test_forward_numpy_preserves_shape(self):
        """Output shape should match (batch, out_features)."""
        w = np.random.randn(32, 64).astype(np.float32) * 0.1
        lora = LoRALinear(
            in_features=64, out_features=32, bias=True,
            rank=2, alpha=4.0, original_weight=w,
        )
        x = np.random.randn(4, 64).astype(np.float32)
        out = lora.forward_numpy(x)
        assert out.shape == (4, 32)

    def test_forward_numpy_ia3_with_bias(self):
        """IA3 with bias should scale the biased output."""
        w = np.random.randn(16, 32).astype(np.float32) * 0.1
        b = np.ones(16, dtype=np.float32) * 0.5
        lora = LoRALinear(
            in_features=32, out_features=16, bias=True,
            rank=4, alpha=8.0, lora_type=LoRAType.IA3,
            original_weight=w, original_bias=b,
        )
        lora.lora_s.data[:] = 3.0
        x = np.random.randn(1, 32).astype(np.float32)
        out = lora.forward_numpy(x)
        expected = (x @ w.T + b) * 3.0
        np.testing.assert_allclose(out, expected, rtol=1e-5)


# ============================================================================
# LoRALinear.forward / train / eval tests
# ============================================================================


class TestLoRALinearTrainEval:
    """Tests for LoRALinear training mode switching."""

    def test_train_mode(self):
        lora = LoRALinear(32, 16, bias=True, rank=4, alpha=8.0)
        lora.train()
        assert lora.training is True

    def test_eval_mode(self):
        lora = LoRALinear(32, 16, bias=True, rank=4, alpha=8.0)
        lora.train()
        lora.eval()
        assert lora.training is False

    def test_default_training_mode(self):
        lora = LoRALinear(32, 16, bias=True, rank=4, alpha=8.0)
        assert lora.training is True

    def test_get_trainable_parameters(self):
        lora = LoRALinear(32, 16, bias=True, rank=4, alpha=8.0)
        params = lora.get_trainable_parameters()
        assert len(params) == 2  # lora_A, lora_B

    def test_get_trainable_parameters_ia3(self):
        lora = LoRALinear(32, 16, bias=True, rank=4, alpha=8.0, lora_type=LoRAType.IA3)
        params = lora.get_trainable_parameters()
        assert len(params) == 1  # lora_s

    def test_parameters_matches_get_trainable(self):
        lora = LoRALinear(32, 16, bias=True, rank=4, alpha=8.0)
        assert lora.parameters() == lora.get_trainable_parameters()

    def test_named_parameters(self):
        lora = LoRALinear(32, 16, bias=True, rank=4, alpha=8.0)
        named = lora.named_parameters(prefix="block.")
        names = [n for n, _ in named]
        assert "block.lora_A" in names
        assert "block.lora_B" in names


# ============================================================================
# LoRAEmbedding tests
# ============================================================================


class TestLoRAEmbedding:
    """Tests for LoRAEmbedding."""

    def test_forward_numpy_basic(self):
        emb = LoRAEmbedding(num_embeddings=64, embedding_dim=32, rank=4, alpha=8.0)
        x = np.array([0, 1, 2, 3], dtype=np.int64)
        out = emb.forward_numpy(x)
        assert out.shape == (4, 32)
        assert np.isfinite(out).all()

    def test_forward_numpy_zero_lora(self):
        emb = LoRAEmbedding(num_embeddings=64, embedding_dim=32, rank=4, alpha=8.0)
        x = np.array([0, 1, 2, 3], dtype=np.int64)
        out_lora = emb.forward_numpy(x)
        out_base = emb.weight.weight.data[x]
        np.testing.assert_allclose(out_lora, out_base, rtol=1e-5, atol=1e-5)

    def test_forward_numpy_single_index(self):
        emb = LoRAEmbedding(num_embeddings=64, embedding_dim=32, rank=4, alpha=8.0)
        x = np.array([5], dtype=np.int64)
        out = emb.forward_numpy(x)
        assert out.shape == (1, 32)

    def test_get_trainable_parameters(self):
        emb = LoRAEmbedding(num_embeddings=64, embedding_dim=32, rank=4, alpha=8.0)
        params = emb.get_trainable_parameters()
        assert len(params) == 2  # lora_A, lora_B

    def test_merge_weights_updates_embedding(self):
        emb = LoRAEmbedding(num_embeddings=64, embedding_dim=64, rank=4, alpha=8.0)
        emb.lora_A.data[:] = np.random.randn(4, 64).astype(np.float32) * 0.01
        emb.lora_B.data[:] = np.random.randn(64, 4).astype(np.float32) * 0.01
        before = emb.weight.weight.data[0, :].copy()
        emb.merge_weights()
        after = emb.weight.weight.data[0, :]
        assert not np.allclose(before, after, atol=1e-6)

    def test_forward_matches_looked_up_base(self):
        emb = LoRAEmbedding(num_embeddings=64, embedding_dim=32, rank=4, alpha=8.0)
        x = np.array([0, 1, 2, 3], dtype=np.int64)
        out = emb.forward_numpy(x)
        base = emb.weight.weight.data[x]
        # At init, B=0 so LoRA contribution is zero
        np.testing.assert_allclose(out, base, rtol=1e-5, atol=1e-5)


# ============================================================================
# _has_lora flag tests
# ============================================================================


class TestHasLoraFlag:
    """Tests for the _has_lora flag on models."""

    def test_apply_lora_sets_flag(self):
        model = _make_tiny_model()
        assert not getattr(model, '_has_lora', False)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        assert getattr(model, '_has_lora', False) is True

    def test_model_without_lora_no_flag(self):
        model = _make_tiny_model()
        assert not getattr(model, '_has_lora', False)

    def test_apply_lora_with_no_targets(self):
        model = _make_tiny_model()
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["nonexistent"])
        model = apply_lora_to_model(model, config)
        assert getattr(model, '_has_lora', False) is False


# ============================================================================
# _generate_numpy_lora tests
# ============================================================================


class TestGenerateNumpyLora:
    """Tests for LoRA-aware generation path."""

    def test_generate_numpy_lora_basic(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        input_ids = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        output = model.generate_numpy(input_ids, max_new_tokens=10, temperature=0.0)
        assert output.shape[0] == 1
        assert output.shape[1] == 15
        assert np.all(output[:, :5] == input_ids)

    def test_generate_numpy_lora_deterministic(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        input_ids = np.array([[1, 2, 3]], dtype=np.int64)
        out1 = model.generate_numpy(input_ids, max_new_tokens=5, temperature=0.0)
        out2 = model.generate_numpy(input_ids, max_new_tokens=5, temperature=0.0)
        np.testing.assert_array_equal(out1, out2)

    def test_generate_numpy_lora_with_temperature(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        input_ids = np.array([[1, 2, 3]], dtype=np.int64)
        output = model.generate_numpy(input_ids, max_new_tokens=5, temperature=0.8)
        assert output.shape == (1, 8)
        assert np.all(output[:, :3] == input_ids)
        assert np.isfinite(output).all()

    def test_generate_numpy_lora_with_top_k(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        input_ids = np.array([[1, 2, 3]], dtype=np.int64)
        output = model.generate_numpy(input_ids, max_new_tokens=5, temperature=0.5, top_k=10)
        assert output.shape == (1, 8)
        assert np.all(output[:, :3] == input_ids)
        assert np.isfinite(output).all()

    def test_generate_numpy_lora_with_top_p(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        input_ids = np.array([[1, 2, 3]], dtype=np.int64)
        output = model.generate_numpy(input_ids, max_new_tokens=5, temperature=0.5, top_p=0.9)
        assert output.shape == (1, 8)
        assert np.all(output[:, :3] == input_ids)
        assert np.isfinite(output).all()

    def test_generate_numpy_lora_with_repetition_penalty(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        input_ids = np.array([[1, 2, 3]], dtype=np.int64)
        output = model.generate_numpy(
            input_ids, max_new_tokens=5, temperature=0.5, repetition_penalty=1.2,
        )
        assert output.shape == (1, 8)
        assert np.all(output[:, :3] == input_ids)
        assert np.isfinite(output).all()

    def test_generate_numpy_lora_small_generation(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        input_ids = np.array([[1, 2, 3]], dtype=np.int64)
        output = model.generate_numpy(input_ids, max_new_tokens=2, temperature=0.0)
        assert output.shape[0] == 1
        assert output.shape[1] >= 3
        np.testing.assert_array_equal(output[:, :3], input_ids)


# ============================================================================
# LoRA parameter extraction tests
# ============================================================================


class TestLoRAParameters:
    """Tests for LoRA parameter extraction."""

    def test_get_lora_parameters(self):
        model = _make_tiny_model()
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        params = get_lora_parameters(model)
        assert len(params) > 0
        for name in params:
            assert "lora_" in name

    def test_count_lora_parameters(self):
        model = _make_tiny_model()
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        count = count_lora_parameters(model)
        assert count > 0

    def test_count_lora_without_lora(self):
        model = _make_tiny_model()
        count = count_lora_parameters(model)
        # Without LoRA, only base parameters that require_grad
        assert count >= 0

    def test_get_lora_parameters_ia3(self):
        model = _make_tiny_model()
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"], lora_type=LoRAType.IA3)
        model = apply_lora_to_model(model, config)
        params = get_lora_parameters(model)
        for name in params:
            assert "lora_s" in name

    def test_get_lora_parameters_multiple_targets(self):
        model = _make_tiny_model()
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_k", "W_v", "W_o"])
        model = apply_lora_to_model(model, config)
        params = get_lora_parameters(model)
        # Should have lora_A and lora_B for each target module
        assert len(params) >= 8  # 4 targets * 2 params each

    def test_lora_param_shapes(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q"])
        model = apply_lora_to_model(model, config)
        params = get_lora_parameters(model)
        for name, param in params.items():
            if "lora_A" in name:
                assert param.data.shape[0] == 4  # rank
            elif "lora_B" in name:
                assert param.data.shape[1] == 4  # rank


# ============================================================================
# Merge weights tests
# ============================================================================


class TestMergeWeights:
    """Tests for LoRA weight merging."""

    def test_merge_lora_into_base(self):
        w = np.random.randn(16, 32).astype(np.float32) * 0.1
        lora = LoRALinear(
            in_features=32, out_features=16, bias=False,
            rank=4, alpha=8.0, original_weight=w,
        )
        lora.lora_A.data[:] = np.random.randn(4, 32).astype(np.float32) * 0.1
        lora.lora_B.data[:] = np.random.randn(16, 4).astype(np.float32) * 0.1
        lora_w = lora.lora_B.data @ lora.lora_A.data * (lora.alpha / lora.rank)
        expected_w = w + lora_w
        lora.merge_weights()
        np.testing.assert_allclose(lora.weight.data, expected_w, rtol=1e-5)
        np.testing.assert_allclose(lora.lora_A.data, 0.0, atol=1e-7)
        np.testing.assert_allclose(lora.lora_B.data, 0.0, atol=1e-7)

    def test_merge_ia3_into_base(self):
        w = np.random.randn(16, 32).astype(np.float32) * 0.1
        lora = LoRALinear(
            in_features=32, out_features=16, bias=False,
            rank=4, alpha=8.0, lora_type=LoRAType.IA3,
            original_weight=w,
        )
        lora.lora_s.data[:] = 3.0
        lora.merge_weights()
        expected_w = w * 3.0
        np.testing.assert_allclose(lora.weight.data, expected_w, rtol=1e-5)
        np.testing.assert_allclose(lora.lora_s.data, 1.0, atol=1e-7)

    def test_merge_preserves_output(self):
        w = np.random.randn(16, 32).astype(np.float32) * 0.1
        lora = LoRALinear(
            in_features=32, out_features=16, bias=False,
            rank=4, alpha=8.0, original_weight=w,
        )
        lora.lora_A.data[:] = np.random.randn(4, 32).astype(np.float32) * 0.1
        lora.lora_B.data[:] = np.random.randn(16, 4).astype(np.float32) * 0.1
        x = np.random.randn(1, 32).astype(np.float32)
        out_before = lora.forward_numpy(x)
        lora.merge_weights()
        out_after = lora.forward_numpy(x)
        np.testing.assert_allclose(out_before, out_after, rtol=1e-5)

    def test_merge_zero_lora_no_change(self):
        w = np.random.randn(16, 32).astype(np.float32) * 0.1
        lora = LoRALinear(
            in_features=32, out_features=16, bias=False,
            rank=4, alpha=8.0, original_weight=w,
        )
        w_before = lora.weight.data.copy()
        lora.merge_weights()
        np.testing.assert_allclose(lora.weight.data, w_before, rtol=1e-5)


# ============================================================================
# Save/load adapter tests
# ============================================================================


class TestSaveLoadAdapter:
    """Tests for LoRA adapter save/load."""

    def test_save_load_roundtrip(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        for name, param in get_lora_parameters(model).items():
            if hasattr(param, 'data'):
                param.data[:] = np.random.randn(*param.data.shape).astype(np.float32) * 0.1

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "test_adapter.npz"
            adapter_dict = {}
            for name, param in get_lora_parameters(model).items():
                if hasattr(param, 'data'):
                    adapter_dict[name] = param.data
            np.savez_compressed(str(adapter_path), **adapter_dict)

            model2 = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
            model2 = apply_lora_to_model(model2, config)
            adapter = np.load(str(adapter_path))
            for name, param in get_lora_parameters(model2).items():
                if name in adapter and hasattr(param, 'data'):
                    param.data[:] = adapter[name]

            for name in get_lora_parameters(model):
                p1 = get_lora_parameters(model)[name]
                p2 = get_lora_parameters(model2)[name]
                np.testing.assert_allclose(p1.data, p2.data, rtol=1e-6)

    def test_load_lora_adapter_function(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        for name, param in get_lora_parameters(model).items():
            if hasattr(param, 'data'):
                param.data[:] = np.ones_like(param.data) * 0.5

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "adapter.npz"
            adapter_dict = {}
            for name, param in get_lora_parameters(model).items():
                if hasattr(param, 'data'):
                    adapter_dict[name] = param.data
            np.savez_compressed(str(adapter_path), **adapter_dict)

            model2 = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
            model2 = apply_lora_to_model(model2, config)
            loaded = load_lora_adapter(model2, str(adapter_path))
            for name in get_lora_parameters(model):
                np.testing.assert_allclose(
                    get_lora_parameters(model)[name].data,
                    get_lora_parameters(loaded)[name].data,
                    rtol=1e-6,
                )


# ============================================================================
# merge_lora_adapter function tests
# ============================================================================


class TestMergeLoraAdapterFunction:
    """Tests for the merge_lora_adapter function."""

    def test_merge_removes_lora_layers(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        assert model._has_lora is True
        merged = merge_lora_adapter(model)
        assert merged._has_lora is False

    def test_merge_preserves_output_finite(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        x = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        merged = merge_lora_adapter(model)
        logits, _ = merged.forward(Tensor(x))
        assert logits.data.shape == (1, 5, 64)
        assert np.isfinite(logits.data).all()

    def test_merge_replaces_with_slo_linear(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        merged = merge_lora_adapter(model)
        # Walk the tree and check no LoRALinear remains
        for path, module in _walk_slo_tree(merged, []):
            assert not isinstance(module, LoRALinear), f"LoRALinear found at {path}"


# ============================================================================
# Full model forward with LoRA
# ============================================================================


class TestFullModelForward:
    """Tests for full model forward pass with LoRA."""

    def test_forward_pass_with_lora(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        x = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        logits, _ = model.forward(Tensor(x))
        logits = logits.data
        assert logits.shape == (1, 5, 64)
        assert np.isfinite(logits).all()

    def test_forward_numpy_path_with_lora(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        x = np.random.randn(1, 5, 32).astype(np.float32)
        block = model.layers[2]
        out, (k, v) = block.forward_numpy(x)
        assert out.shape == (1, 5, 32)
        assert np.isfinite(out).all()

    def test_forward_with_targets(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        x = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
        targets = np.array([[2, 3, 4, 5, 6]], dtype=np.int64)
        logits, loss = model.forward(Tensor(x), targets=Tensor(targets))
        assert logits.data.shape == (1, 5, 64)
        assert loss is not None
        assert np.isfinite(loss.data)


# ============================================================================
# _walk_slo_tree tests
# ============================================================================


class TestWalkSloTree:
    """Tests for _walk_slo_tree helper."""

    def test_walk_finds_linear_layers(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        found = []
        for path, module in _walk_slo_tree(model, []):
            if isinstance(module, SloLinear):
                found.append(path)
        assert len(found) > 0

    def test_walk_finds_embedding(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        found = []
        for path, module in _walk_slo_tree(model, []):
            if isinstance(module, SloEmbedding):
                found.append(path)
        assert len(found) > 0

    def test_walk_paths_are_dotted(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        for path, module in _walk_slo_tree(model, []):
            assert "." in path or "layers" in path

    def test_walk_finds_lora_layers(self):
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        found = []
        for path, module in _walk_slo_tree(model, []):
            if isinstance(module, LoRALinear):
                found.append(path)
        assert len(found) > 0


# ============================================================================
# _set_nested tests
# ============================================================================


class TestSetNested:
    """Tests for _set_nested helper."""

    def test_set_simple_attr(self):
        class Obj:
            pass
        obj = Obj()
        _set_nested(obj, ["x"], 42)
        assert obj.x == 42

    def test_set_nested_attr(self):
        class Inner:
            pass
        class Outer:
            pass
        outer = Outer()
        outer.inner = Inner()
        _set_nested(outer, ["inner", "x"], 42)
        assert outer.inner.x == 42

    def test_set_list_index(self):
        class Obj:
            pass
        obj = Obj()
        obj.items = [10, 20, 30]
        _set_nested(obj, ["items[1]"], 99)
        assert obj.items[1] == 99

    def test_set_deeply_nested(self):
        class A:
            pass
        class B:
            pass
        class C:
            pass
        a = A()
        a.b = B()
        a.b.c = C()
        _set_nested(a, ["b", "c", "val"], "hello")
        assert a.b.c.val == "hello"


# ============================================================================
# TrainResult protocol tests
# ============================================================================


class TestTrainResult:
    """Tests for TrainResult dataclass."""

    def test_defaults(self):
        r = TrainResult()
        assert r.success is True
        assert r.status == "completed"
        assert r.final_loss is None
        assert r.error is None

    def test_get_backward_compat(self):
        r = TrainResult(final_loss=0.5)
        assert r.get("final_loss") == 0.5
        assert r.get("nonexistent", "default") == "default"

    def test_getitem_backward_compat(self):
        r = TrainResult(final_loss=0.5)
        assert r["final_loss"] == 0.5

    def test_getitem_missing_key(self):
        r = TrainResult()
        with pytest.raises(KeyError):
            _ = r["nonexistent"]

    def test_contains(self):
        r = TrainResult(final_loss=0.5)
        assert "final_loss" in r
        assert "nonexistent" not in r

    def test_to_dict(self):
        r = TrainResult(final_loss=0.5, status="completed")
        d = r.to_dict()
        assert isinstance(d, dict)
        assert d["final_loss"] == 0.5
        assert d["status"] == "completed"

    def test_checkpoint_alias(self):
        r = TrainResult(checkpoint_name="my_ckpt")
        assert r.checkpoint == "my_ckpt"

    def test_failure_result(self):
        r = TrainResult(success=False, status="failed", error="boom")
        assert not r.success
        assert r.error == "boom"


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    """Edge case tests for LoRA components."""

    def test_lora_rank_1(self):
        lora = LoRALinear(32, 16, bias=True, rank=1, alpha=2.0)
        x = np.random.randn(1, 32).astype(np.float32)
        out = lora.forward_numpy(x)
        assert out.shape == (1, 16)

    def test_lora_large_rank(self):
        lora = LoRALinear(32, 16, bias=True, rank=16, alpha=32.0)
        x = np.random.randn(1, 32).astype(np.float32)
        out = lora.forward_numpy(x)
        assert out.shape == (1, 16)

    def test_lora_alpha_zero(self):
        w = np.random.randn(16, 32).astype(np.float32) * 0.1
        lora = LoRALinear(32, 16, bias=False, rank=4, alpha=0.0, original_weight=w)
        x = np.random.randn(1, 32).astype(np.float32)
        out = lora.forward_numpy(x)
        expected = x @ w.T
        np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)

    def test_lora_dropout_on_forward(self):
        lora = LoRALinear(32, 16, bias=True, rank=4, alpha=8.0, dropout=0.5)
        lora.train()
        x = Tensor(np.random.randn(1, 32).astype(np.float32))
        out = lora.forward(x)
        assert out.data.shape == (1, 16)

    def test_lora_no_original_weight(self):
        lora = LoRALinear(32, 16, bias=True, rank=4, alpha=8.0)
        assert lora.weight.data.shape == (16, 32)

    def test_lora_no_original_bias(self):
        lora = LoRALinear(32, 16, bias=False, rank=4, alpha=8.0)
        assert lora.bias is None

    def test_lora_embedding_large_vocab(self):
        emb = LoRAEmbedding(num_embeddings=10000, embedding_dim=64, rank=4, alpha=8.0)
        x = np.array([0, 9999, 5000], dtype=np.int64)
        out = emb.forward_numpy(x)
        assert out.shape == (3, 64)

    def test_adapter_npz_roundtrip(self):
        """Save and load adapter should preserve config metadata."""
        model = _make_tiny_model(vocab_size=64, n_embed=32, n_layer=2, n_head=2)
        config = LoRAConfig(rank=4, alpha=8.0, target_modules=["W_q", "W_v"])
        model = apply_lora_to_model(model, config)
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = Path(tmpdir) / "adapter.npz"
            adapter_dict = {}
            for name, param in get_lora_parameters(model).items():
                if hasattr(param, 'data'):
                    adapter_dict[name] = param.data
            adapter_dict["_config/rank"] = np.array([config.rank])
            adapter_dict["_config/alpha"] = np.array([config.alpha])
            np.savez_compressed(str(adapter_path), **adapter_dict)
            loaded = np.load(str(adapter_path))
            assert loaded["_config/rank"][0] == 4
            assert loaded["_config/alpha"][0] == 8.0
