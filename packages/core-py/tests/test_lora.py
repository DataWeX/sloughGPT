"""Tests for LoRA module — config, LoRALinear, LoRAEmbedding, apply, count."""

import math
import pytest
import numpy as np
from domains.training.lora import (
    LoRAType, LoRAConfig, LoRALinear, LoRAEmbedding,
    apply_lora_to_model, get_lora_parameters, count_lora_parameters,
    _to_np, _to_tensor,
)
from domains.training.slonet import Tensor


# ── Helpers ────────────────────────────────────────────────────────────────

class TestToNp:

    def test_from_tensor(self):
        t = Tensor(np.array([1.0, 2.0, 3.0]))
        result = _to_np(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_from_ndarray(self):
        arr = np.array([4.0, 5.0])
        result = _to_np(arr)
        np.testing.assert_array_equal(result, arr)

    def test_from_list(self):
        result = _to_np([1.0, 2.0])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0])


class TestToTensor:

    def test_from_ndarray(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        t = _to_tensor(arr)
        assert isinstance(t, Tensor)
        np.testing.assert_array_equal(t.data, arr)
        assert t.requires_grad is True

    def test_from_list(self):
        t = _to_tensor([3.0, 4.0])
        assert isinstance(t, Tensor)

    def test_no_grad(self):
        t = _to_tensor(np.array([1.0]), requires_grad=False)
        assert t.requires_grad is False


# ── LoRAConfig ─────────────────────────────────────────────────────────────

class TestLoRAConfig:

    def test_defaults(self):
        cfg = LoRAConfig()
        assert cfg.rank == 8
        assert cfg.alpha == 16.0
        assert cfg.dropout == 0.05
        assert cfg.lora_type == LoRAType.LORA
        assert cfg.bias == "none"
        assert cfg.task_type == "CAUSAL_LM"

    def test_default_target_modules(self):
        cfg = LoRAConfig()
        assert "q_proj" in cfg.target_modules
        assert "v_proj" in cfg.target_modules
        assert "k_proj" in cfg.target_modules
        assert "o_proj" in cfg.target_modules

    def test_custom_config(self):
        cfg = LoRAConfig(rank=4, alpha=8.0, dropout=0.1, lora_type=LoRAType.IA3)
        assert cfg.rank == 4
        assert cfg.alpha == 8.0
        assert cfg.dropout == 0.1
        assert cfg.lora_type == LoRAType.IA3

    def test_custom_target_modules(self):
        cfg = LoRAConfig(target_modules=["attn.q_proj"])
        assert cfg.target_modules == ["attn.q_proj"]


# ── LoRAType ───────────────────────────────────────────────────────────────

class TestLoRAType:

    def test_values(self):
        assert LoRAType.LORA.value == "lora"
        assert LoRAType.LORA_PLUS.value == "lora_plus"
        assert LoRAType.IA3.value == "ia3"


# ── LoRALinear ─────────────────────────────────────────────────────────────

class TestLoRALinear:

    def test_init_default(self):
        layer = LoRALinear(128, 64)
        assert layer.in_features == 128
        assert layer.out_features == 64
        assert layer.rank == 8
        assert layer.alpha == 16.0
        assert layer.weight.data.shape == (64, 128)
        assert layer.lora_A.data.shape == (8, 128)
        assert layer.lora_B.data.shape == (64, 8)
        assert layer.bias is not None

    def test_init_no_bias(self):
        layer = LoRALinear(64, 32, bias=False)
        assert layer.bias is None

    def test_init_with_original_weight(self):
        w = np.ones((32, 64), dtype=np.float32)
        layer = LoRALinear(64, 32, original_weight=w)
        np.testing.assert_array_equal(layer.weight.data, w)

    def test_init_ia3(self):
        layer = LoRALinear(64, 32, lora_type=LoRAType.IA3)
        assert hasattr(layer, "lora_s")
        assert not hasattr(layer, "lora_A")

    def test_forward_lora(self):
        layer = LoRALinear(64, 32, rank=4)
        x = Tensor(np.random.randn(1, 64).astype(np.float32), requires_grad=True)
        out = layer.forward(x)
        assert out.data.shape == (1, 32)

    def test_forward_ia3(self):
        layer = LoRALinear(64, 32, lora_type=LoRAType.IA3)
        x = Tensor(np.random.randn(1, 64).astype(np.float32), requires_grad=True)
        out = layer.forward(x)
        assert out.data.shape == (1, 32)

    def test_train_eval(self):
        layer = LoRALinear(64, 32, dropout=0.1)
        layer.eval()
        assert layer.training is False
        layer.train()
        assert layer.training is True

    def test_merge_weights_lora_resets_adapters(self):
        layer = LoRALinear(64, 32)
        layer.merge_weights()
        assert np.allclose(layer.lora_A.data, 0.0)
        assert np.allclose(layer.lora_B.data, 0.0)

    def test_merge_weights_ia3_resets_lora_s(self):
        layer = LoRALinear(64, 32, lora_type=LoRAType.IA3)
        layer.lora_s.data[:] = 2.0
        w_before = layer.weight.data.copy()
        layer.merge_weights()
        assert np.allclose(layer.lora_s.data, 1.0)

    def test_merge_weights_with_trained_lora(self):
        layer = LoRALinear(64, 32, rank=4)
        layer.lora_A.data = np.random.randn(4, 64).astype(np.float32) * 0.1
        layer.lora_B.data = np.random.randn(32, 4).astype(np.float32) * 0.1
        w_before = layer.weight.data.copy()
        layer.merge_weights()
        assert not np.allclose(layer.weight.data, w_before)

    def test_get_trainable_parameters_lora(self):
        layer = LoRALinear(64, 32)
        params = layer.get_trainable_parameters()
        assert len(params) == 2

    def test_get_trainable_parameters_ia3(self):
        layer = LoRALinear(64, 32, lora_type=LoRAType.IA3)
        params = layer.get_trainable_parameters()
        assert len(params) == 1

    def test_parameters(self):
        layer = LoRALinear(64, 32)
        params = layer.parameters()
        assert len(params) == 2

    def test_named_parameters_lora(self):
        layer = LoRALinear(64, 32, lora_type=LoRAType.LORA)
        named = layer.named_parameters()
        assert len(named) == 2
        names = [n for n, _ in named]
        assert "lora_A" in names
        assert "lora_B" in names

    def test_named_parameters_ia3(self):
        layer = LoRALinear(64, 32, lora_type=LoRAType.IA3)
        named = layer.named_parameters()
        assert len(named) == 1
        names = [n for n, _ in named]
        assert "lora_s" in names


# ── LoRAEmbedding ──────────────────────────────────────────────────────────

class TestLoRAEmbedding:

    def test_init(self):
        emb = LoRAEmbedding(100, 32, rank=4)
        assert emb.num_embeddings == 100
        assert emb.embedding_dim == 32
        assert emb.rank == 4
        assert emb.lora_A.data.shape == (4, 32)
        assert emb.lora_B.data.shape == (32, 4)

    def test_forward(self):
        emb = LoRAEmbedding(100, 32, rank=4)
        x = Tensor(np.array([[0, 1, 2]]), requires_grad=True)
        out = emb.forward(x)
        assert out.data.shape[-1] == 32

    def test_merge_weights(self):
        emb = LoRAEmbedding(32, 32, rank=4)
        w_before = emb.weight.weight.data.copy()
        emb.merge_weights()
        assert emb.lora_A.data.shape == (4, 32)
        assert emb.lora_B.data.shape == (32, 4)

    def test_get_trainable_parameters(self):
        emb = LoRAEmbedding(100, 32)
        params = emb.get_trainable_parameters()
        assert len(params) == 2


# ── apply_lora_to_model ────────────────────────────────────────────────────

class TestApplyLoRA:

    def _make_simple_model(self):
        class SimpleLayer:
            def __init__(self):
                self.weight = Tensor(np.ones((32, 64), dtype=np.float32), requires_grad=False)
                self.bias = Tensor(np.zeros(32, dtype=np.float32), requires_grad=True)
                self.in_features = 64
                self.out_features = 32

        class SimpleModel:
            def __init__(self):
                self.linear = SimpleLayer()

            def named_modules(self):
                yield "", self
                yield "linear", self.linear

        return SimpleModel()

    def test_apply_to_matching_module(self):
        model = self._make_simple_model()
        result = apply_lora_to_model(model, target_modules=["linear"])
        assert isinstance(result.linear, LoRALinear)

    def test_apply_no_match(self):
        model = self._make_simple_model()
        result = apply_lora_to_model(model, target_modules=["nonexistent"])
        assert not isinstance(result.linear, LoRALinear)

    def test_apply_with_config(self):
        model = self._make_simple_model()
        cfg = LoRAConfig(rank=4, alpha=8.0, target_modules=["linear"])
        result = apply_lora_to_model(model, config=cfg)
        assert isinstance(result.linear, LoRALinear)
        assert result.linear.rank == 4


# ── count_lora_parameters ──────────────────────────────────────────────────

class TestCountLoRAParameters:

    def test_count(self):
        layer = LoRALinear(64, 32)
        count = count_lora_parameters(layer)
        assert count > 0

    def test_count_ia3(self):
        layer = LoRALinear(64, 32, lora_type=LoRAType.IA3)
        count = count_lora_parameters(layer)
        assert count > 0


# ── get_lora_parameters ────────────────────────────────────────────────────

class TestGetLoRAParameters:

    def test_empty_model_no_params(self):
        class NoParams:
            def named_parameters(self):
                return []

        result = get_lora_parameters(NoParams())
        assert result == {}

    def test_model_with_lora(self):
        layer = LoRALinear(64, 32)

        class SimpleModel:
            def named_parameters(self):
                yield "lora_A", layer.lora_A
                yield "lora_B", layer.lora_B
                yield "weight", layer.weight

        result = get_lora_parameters(SimpleModel())
        assert "lora_A" in result
        assert "lora_B" in result
        assert "weight" not in result
