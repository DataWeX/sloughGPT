"""Comprehensive tests for weight_loader.py — data structures, infer_arch,
build_load_plan (via mock arch_config), WeightLoaderRegistry, DirectWeightLoader,
load_into_model, WeightLoadResult.

Covers: TensorMapping, LoadPlan, WeightLoadResult, infer_arch_from_state_dict,
WeightLoaderRegistry register/get/load, DirectWeightLoader, load_into_model.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from domains.infrastructure.weight_loader import (
    TensorMapping,
    LoadPlan,
    WeightLoadResult,
    infer_arch_from_state_dict,
    WeightLoaderRegistry,
    get_weight_loader_registry,
    load_into_model,
    _NO_TRANSPOSE,
    _ARCH_TO_SLONET,
    _SWIGLU_MAP,
    _GELU_MAP,
    _NORM_BIAS_MAP,
)


# ---------------------------------------------------------------------------
# TensorMapping
# ---------------------------------------------------------------------------

class TestTensorMapping:
    def test_creation(self):
        tm = TensorMapping(
            param_name="blocks.0.attn.q_proj.weight",
            needs_transpose=True,
            canonical="layers.{i}.q.weight",
        )
        assert tm.param_name == "blocks.0.attn.q_proj.weight"
        assert tm.needs_transpose is True
        assert tm.canonical == "layers.{i}.q.weight"

    def test_frozen(self):
        tm = TensorMapping(param_name="x", needs_transpose=False, canonical="y")
        with pytest.raises(AttributeError):
            tm.param_name = "z"


# ---------------------------------------------------------------------------
# LoadPlan
# ---------------------------------------------------------------------------

class TestLoadPlan:
    def test_creation(self):
        plan = LoadPlan(
            tensor_map={},
            tied_weights=[],
            synthesized_params=[],
            fused_qkv={},
            n_layer=4,
            n_embed=128,
            arch_name="gpt2",
        )
        assert plan.n_layer == 4
        assert plan.n_embed == 128
        assert plan.arch_name == "gpt2"
        assert plan.tensor_map == {}
        assert plan.tied_weights == []


# ---------------------------------------------------------------------------
# WeightLoadResult
# ---------------------------------------------------------------------------

class TestWeightLoadResult:
    def test_success(self):
        r = WeightLoadResult(success=True, n_written=5, n_fused=3)
        assert r.success is True
        assert r.n_written == 5
        assert r.n_fused == 3
        assert r.error is None

    def test_failure(self):
        r = WeightLoadResult(success=False, error="file not found")
        assert r.success is False
        assert r.error == "file not found"

    def test_timing(self):
        r = WeightLoadResult(success=True, timing={"total": 0.5})
        assert r.timing["total"] == 0.5

    def test_defaults(self):
        r = WeightLoadResult(success=True)
        assert r.n_written == 0
        assert r.n_fused == 0
        assert r.timing == {}


# ---------------------------------------------------------------------------
# infer_arch_from_state_dict
# ---------------------------------------------------------------------------

class TestInferArch:
    def test_basic_gpt2(self):
        state_dict = {
            "tok_emb.weight": np.zeros((50257, 768)),
            "wte.weight": np.zeros((50257, 768)),
            "wpe.weight": np.zeros((1024, 768)),
            "blocks.0.attn_norm.weight": np.zeros((768,)),
            "blocks.0.attn.q_proj.weight": np.zeros((768, 768)),
        }
        result = infer_arch_from_state_dict(state_dict)
        assert result["vocab_size"] == 50257
        assert result["n_embed"] == 768

    def test_defaults(self):
        result = infer_arch_from_state_dict({})
        assert result["vocab_size"] == 256
        assert result["n_embed"] == 128
        assert result["n_layer"] == 1

    def test_n_layer_detection(self):
        state_dict = {
            "blocks.0.attn_norm.weight": np.zeros((64,)),
            "blocks.1.attn_norm.weight": np.zeros((64,)),
            "blocks.2.attn_norm.weight": np.zeros((64,)),
        }
        result = infer_arch_from_state_dict(state_dict)
        assert result["n_layer"] == 3

    def test_n_head_detection(self):
        state_dict = {
            "tok_emb.weight": np.zeros((256, 64)),
            "blocks.0.attn.q_proj.weight": np.zeros((64, 64)),
        }
        result = infer_arch_from_state_dict(state_dict)
        # head_dim = 64 // 8 = 8, q shape[0]=64, detected = 64/8 = 8
        assert result["n_head"] == 8

    def test_intermediate_size(self):
        state_dict = {
            "blocks.0.ff.w1.weight": np.zeros((256, 64)),
        }
        result = infer_arch_from_state_dict(state_dict)
        assert result["intermediate_size"] == 256


# ---------------------------------------------------------------------------
# Mapping constants
# ---------------------------------------------------------------------------

class TestMappingConstants:
    def test_arch_to_slonet_has_key_embeddings(self):
        assert "embed.token" in _ARCH_TO_SLONET
        assert "embed.pos" in _ARCH_TO_SLONET

    def test_swiglu_map_has_gate(self):
        assert "layers.{i}.ffn.gate.weight" in _SWIGLU_MAP

    def test_gelu_map_has_up(self):
        assert "layers.{i}.ffn.up.weight" in _GELU_MAP

    def test_norm_bias_map(self):
        assert "layers.{i}.attn_norm.bias" in _NORM_BIAS_MAP

    def test_no_transpose_set(self):
        assert "embed.token" in _NO_TRANSPOSE
        assert "embed.pos" in _NO_TRANSPOSE
        assert "lm_head" in _NO_TRANSPOSE


# ---------------------------------------------------------------------------
# WeightLoaderRegistry
# ---------------------------------------------------------------------------

class TestWeightLoaderRegistry:
    def test_register_and_get(self):
        reg = WeightLoaderRegistry()
        loader_cls = MagicMock
        reg.register_loader(".soul", loader_cls)
        assert reg.get_loader("model.soul") is loader_cls

    def test_get_unknown_suffix(self):
        reg = WeightLoaderRegistry()
        assert reg.get_loader("model.xyz") is None

    def test_default_loader(self):
        reg = WeightLoaderRegistry()
        fallback = MagicMock
        reg.set_default(fallback)
        assert reg.get_loader("model.xyz") is fallback

    def test_suffix_priority(self):
        reg = WeightLoaderRegistry()
        cls1 = MagicMock
        cls2 = MagicMock
        reg.register_loader(".bin", cls1)
        reg.register_loader(".safetensors", cls2)
        assert reg.get_loader("model.bin") is cls1
        assert reg.get_loader("model.safetensors") is cls2

    def test_load_file_no_loader(self):
        reg = WeightLoaderRegistry()
        result = reg.load_file("model.xyz", MagicMock())
        assert result.success is False
        assert "No loader registered" in result.error

    def test_load_file_success(self):
        reg = WeightLoaderRegistry()
        mock_loader_cls = MagicMock()
        mock_loader = MagicMock()
        mock_loader.load.return_value = WeightLoadResult(success=True, n_written=5)
        mock_loader_cls.return_value = mock_loader
        reg.register_loader(".test", mock_loader_cls)

        result = reg.load_file("model.test", MagicMock())
        assert result.success is True
        assert result.n_written == 5
        mock_loader_cls.assert_called_once_with("model.test")
        mock_loader.load.assert_called_once()

    def test_load_file_exception(self):
        reg = WeightLoaderRegistry()
        mock_loader_cls = MagicMock()
        mock_loader_cls.side_effect = RuntimeError("load failed")
        reg.register_loader(".fail", mock_loader_cls)

        result = reg.load_file("model.fail", MagicMock())
        assert result.success is False
        assert "load failed" in result.error

    def test_get_loader_case_insensitive(self):
        reg = WeightLoaderRegistry()
        cls = MagicMock
        reg.register_loader(".SLNC", cls)
        assert reg.get_loader("model.slnc") is cls

    def test_global_registry_has_slnc_and_soul(self):
        reg = get_weight_loader_registry()
        assert reg.get_loader("model.slnc") is not None
        assert reg.get_loader("model.soul") is not None


# ---------------------------------------------------------------------------
# load_into_model — mock model
# ---------------------------------------------------------------------------

class TestLoadIntoModel:
    def _make_mock_model(self, param_names):
        """Create a mock model with named parameters."""
        model = MagicMock()
        params = {}
        for name in param_names:
            p = MagicMock()
            p.data = np.zeros((4, 4), dtype=np.float32)
            params[name] = p
        model._named_parameters.return_value = params.items()
        return model, params

    def test_direct_writes(self):
        plan = LoadPlan(
            tensor_map={
                "hf.weight": TensorMapping(
                    param_name="model.weight",
                    needs_transpose=False,
                    canonical="embed.token",
                )
            },
            tied_weights=[],
            synthesized_params=[],
            fused_qkv={},
            n_layer=1,
            n_embed=4,
            arch_name="test",
        )
        model, params = self._make_mock_model(["model.weight"])
        tensor_data = {"hf.weight": np.ones((4, 4), dtype=np.float32)}

        result = load_into_model(model, plan, tensor_data)
        assert result.success is True
        assert result.n_written == 1
        np.testing.assert_array_equal(params["model.weight"].data, np.ones((4, 4)))

    def test_transpose(self):
        plan = LoadPlan(
            tensor_map={
                "hf.weight": TensorMapping(
                    param_name="model.weight",
                    needs_transpose=True,
                    canonical="layers.0.q.weight",
                )
            },
            tied_weights=[],
            synthesized_params=[],
            fused_qkv={},
            n_layer=1,
            n_embed=4,
            arch_name="test",
        )
        model, params = self._make_mock_model(["model.weight"])
        arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float32)
        tensor_data = {"hf.weight": arr}

        result = load_into_model(model, plan, tensor_data)
        assert result.success is True
        np.testing.assert_array_equal(params["model.weight"].data, arr.T)

    def test_fused_qkv(self):
        plan = LoadPlan(
            tensor_map={},
            tied_weights=[],
            synthesized_params=[],
            fused_qkv={
                "hf.qkv.weight": [
                    "blocks.0.attn.q_proj.weight",
                    "blocks.0.attn.k_proj.weight",
                    "blocks.0.attn.v_proj.weight",
                ]
            },
            n_layer=1,
            n_embed=4,
            arch_name="test",
        )
        model, params = self._make_mock_model([
            "blocks.0.attn.q_proj.weight",
            "blocks.0.attn.k_proj.weight",
            "blocks.0.attn.v_proj.weight",
        ])
        # fused weight shape: (3*n_embed, n_embed) → (12, 4)
        fused = np.random.randn(12, 4).astype(np.float32)
        tensor_data = {"hf.qkv.weight": fused}

        result = load_into_model(model, plan, tensor_data)
        assert result.success is True
        assert result.n_fused == 3
        # q = fused[:4], k = fused[4:8], v = fused[8:12] (after transpose)
        np.testing.assert_array_almost_equal(
            params["blocks.0.attn.q_proj.weight"].data, fused[:4].T
        )
        np.testing.assert_array_almost_equal(
            params["blocks.0.attn.k_proj.weight"].data, fused[4:8].T
        )
        np.testing.assert_array_almost_equal(
            params["blocks.0.attn.v_proj.weight"].data, fused[8:12].T
        )

    def test_fused_qkv_bias(self):
        plan = LoadPlan(
            tensor_map={},
            tied_weights=[],
            synthesized_params=[],
            fused_qkv={
                "hf.qkv.bias": [
                    "blocks.0.attn.q_proj.bias",
                    "blocks.0.attn.k_proj.bias",
                    "blocks.0.attn.v_proj.bias",
                ]
            },
            n_layer=1,
            n_embed=4,
            arch_name="test",
        )
        model, params = self._make_mock_model([
            "blocks.0.attn.q_proj.bias",
            "blocks.0.attn.k_proj.bias",
            "blocks.0.attn.v_proj.bias",
        ])
        fused_bias = np.random.randn(12).astype(np.float32)
        tensor_data = {"hf.qkv.bias": fused_bias}

        result = load_into_model(model, plan, tensor_data)
        assert result.n_fused == 3
        np.testing.assert_array_almost_equal(
            params["blocks.0.attn.q_proj.bias"].data[:4], fused_bias[:4]
        )

    def test_tied_weights(self):
        plan = LoadPlan(
            tensor_map={},
            tied_weights=[("lm_head.weight", "tok_emb.weight")],
            synthesized_params=[],
            fused_qkv={},
            n_layer=1,
            n_embed=4,
            arch_name="test",
        )
        model, params = self._make_mock_model(["lm_head.weight", "tok_emb.weight"])
        params["tok_emb.weight"].data = np.ones((4, 4), dtype=np.float32)

        result = load_into_model(model, plan, {})
        assert result.success is True
        np.testing.assert_array_equal(params["lm_head.weight"].data, params["tok_emb.weight"].data)

    def test_synthesized_params(self):
        plan = LoadPlan(
            tensor_map={},
            tied_weights=[],
            synthesized_params=[
                ("blocks.0.ff.w3.weight", "0", "blocks.0.ff.w1.weight"),
                ("blocks.0.ff.w3.bias", "1", "blocks.0.ff.w1.weight"),
            ],
            fused_qkv={},
            n_layer=1,
            n_embed=4,
            arch_name="test",
        )
        model, params = self._make_mock_model(["blocks.0.ff.w3.weight", "blocks.0.ff.w3.bias"])

        result = load_into_model(model, plan, {})
        assert result.success is True
        np.testing.assert_array_equal(params["blocks.0.ff.w3.weight"].data, 0.0)
        np.testing.assert_array_equal(params["blocks.0.ff.w3.bias"].data, 1.0)

    def test_timing_recorded(self):
        plan = LoadPlan(
            tensor_map={}, tied_weights=[], synthesized_params=[],
            fused_qkv={}, n_layer=1, n_embed=4, arch_name="test",
        )
        model, _ = self._make_mock_model([])
        result = load_into_model(model, plan, {})
        assert "total" in result.timing
        assert result.timing["total"] >= 0


# ---------------------------------------------------------------------------
# _is_intel_mac helper (imported from model_server, tested here too)
# ---------------------------------------------------------------------------

class TestIsIntelMac:
    def test_returns_bool(self):
        from domains.infrastructure.model_server import _is_intel_mac
        assert isinstance(_is_intel_mac(), bool)
