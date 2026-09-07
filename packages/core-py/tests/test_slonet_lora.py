"""Tests for training.lora — LoRA adapter classes."""

from __future__ import annotations

import pytest
import numpy as np
from domains.training.lora import (
    LoRALinear,
    LoRAEmbedding,
    LoRAConfig,
    LoRAType,
    _to_np,
    _to_tensor,
)
from domains.training.slonet import Tensor, SloLayer


# ── _to_np / _to_tensor helpers ────────────────────────────────────────────


class TestHelperFunctions:

    def test_to_np_tensor(self):
        t = Tensor(np.array([1.0, 2.0]))
        result = _to_np(t)
        assert isinstance(result, np.ndarray)

    def test_to_np_array(self):
        arr = np.array([1.0, 2.0])
        result = _to_np(arr)
        assert isinstance(result, np.ndarray)

    def test_to_np_list(self):
        result = _to_np([1.0, 2.0])
        assert isinstance(result, np.ndarray)

    def test_to_tensor_array(self):
        arr = np.array([1.0, 2.0])
        t = _to_tensor(arr)
        assert isinstance(t, Tensor)
        assert t.requires_grad is True

    def test_to_tensor_no_grad(self):
        arr = np.array([1.0, 2.0])
        t = _to_tensor(arr, requires_grad=False)
        assert t.requires_grad is False


# ── LoRAType / LoRAConfig ──────────────────────────────────────────────────


class TestLoRAConfig:

    def test_default(self):
        config = LoRAConfig()
        assert config.rank == 8
        assert config.alpha == 16.0
        assert config.lora_type == LoRAType.LORA

    def test_custom(self):
        config = LoRAConfig(rank=4, alpha=8.0, lora_type=LoRAType.IA3)
        assert config.rank == 4
        assert config.alpha == 8.0
        assert config.lora_type == LoRAType.IA3

    def test_target_modules_default(self):
        config = LoRAConfig()
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules


# ── LoRALinear ──────────────────────────────────────────────────────────────


class TestLoRALinear:

    def test_init(self):
        layer = LoRALinear(10, 5)
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.rank == 8

    def test_init_custom(self):
        layer = LoRALinear(10, 5, rank=4, alpha=8.0)
        assert layer.rank == 4
        assert layer.alpha == 8.0

    def test_no_bias(self):
        layer = LoRALinear(10, 5, bias=False)
        assert layer.bias is None

    def test_with_bias(self):
        layer = LoRALinear(10, 5, bias=True)
        assert layer.bias is not None
        assert layer.bias.shape == (5,)

    def test_ia3_type(self):
        layer = LoRALinear(10, 5, lora_type=LoRAType.IA3)
        assert hasattr(layer, "lora_s")
        assert not hasattr(layer, "lora_A")

    def test_lora_type(self):
        layer = LoRALinear(10, 5, lora_type=LoRAType.LORA)
        assert hasattr(layer, "lora_A")
        assert hasattr(layer, "lora_B")

    def test_forward_numpy(self):
        layer = LoRALinear(10, 5, bias=True)
        x = np.random.randn(1, 10).astype(np.float32)
        out = layer.forward_numpy(x)
        assert out.shape == (1, 5)

    def test_forward_numpy_no_bias(self):
        layer = LoRALinear(10, 5, bias=False)
        x = np.random.randn(1, 10).astype(np.float32)
        out = layer.forward_numpy(x)
        assert out.shape == (1, 5)

    def test_train_eval(self):
        layer = LoRALinear(10, 5)
        layer.train()
        assert layer.training is True
        layer.eval()
        assert layer.training is False

    def test_isinstance_slope_layer(self):
        layer = LoRALinear(10, 5)
        assert isinstance(layer, SloLayer)

    def test_gradient(self):
        layer = LoRALinear(10, 5, rank=4, alpha=8.0)
        x = np.random.randn(1, 10).astype(np.float32)
        out = layer.forward_numpy(x)
        assert out.shape == (1, 5)


# ── LoRAEmbedding ──────────────────────────────────────────────────────────


class TestLoRAEmbedding:

    def test_init(self):
        emb = LoRAEmbedding(100, 32)
        assert emb.num_embeddings == 100
        assert emb.embedding_dim == 32

    def test_init_custom_rank(self):
        emb = LoRAEmbedding(100, 32, rank=4)
        assert emb.rank == 4

    def test_forward(self):
        emb = LoRAEmbedding(100, 32)
        x = np.array([0, 1, 2])
        out = emb.forward_numpy(x)
        assert out.shape == (3, 32)

    def test_isinstance_slope_layer(self):
        emb = LoRAEmbedding(100, 32)
        assert isinstance(emb, SloLayer)

    def test_train_eval(self):
        emb = LoRAEmbedding(100, 32)
        assert hasattr(emb, 'lora_A')


# ── LoRA merge/unmerge ─────────────────────────────────────────────────────


class TestLoRAMerge:

    def test_lora_a_b(self):
        layer = LoRALinear(10, 5, rank=4, alpha=8.0)
        assert layer.lora_A.shape == (4, 10)
        assert layer.lora_B.shape == (5, 4)

    def test_ia3_s(self):
        layer = LoRALinear(10, 5, lora_type=LoRAType.IA3)
        assert layer.lora_s.shape == (5,)
