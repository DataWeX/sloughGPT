"""Tests for weight_mapper — SLNC tensor dict to flat C weight array."""

import numpy as np
import pytest
from domains.inference.native.weight_mapper import map_slnc_to_native


def _make_tensors(D, NH, NKV, HD, FF, V, n_layers, prefix="model.layers"):
    tensors = {}
    tensors["model.embed_tokens.weight"] = np.ones((V, D), dtype=np.float32)
    tensors["model.norm.weight"] = np.ones(D, dtype=np.float32)
    tensors["model.lm_head.weight"] = np.ones((V, D), dtype=np.float32)
    for i in range(n_layers):
        p = f"{prefix}.{i}"
        tensors[f"{p}.input_layernorm.weight"] = np.ones(D, dtype=np.float32)
        tensors[f"{p}.self_attn.q_proj.weight"] = np.ones((D, NH * HD), dtype=np.float32)
        tensors[f"{p}.self_attn.q_proj.bias"] = np.ones(NH * HD, dtype=np.float32)
        tensors[f"{p}.self_attn.k_proj.weight"] = np.ones((D, NKV * HD), dtype=np.float32)
        tensors[f"{p}.self_attn.k_proj.bias"] = np.ones(NKV * HD, dtype=np.float32)
        tensors[f"{p}.self_attn.v_proj.weight"] = np.ones((D, NKV * HD), dtype=np.float32)
        tensors[f"{p}.self_attn.v_proj.bias"] = np.ones(NKV * HD, dtype=np.float32)
        tensors[f"{p}.self_attn.o_proj.weight"] = np.ones((NH * HD, D), dtype=np.float32)
        tensors[f"{p}.self_attn.o_proj.bias"] = np.ones(D, dtype=np.float32)
        tensors[f"{p}.post_attention_layernorm.weight"] = np.ones(D, dtype=np.float32)
        tensors[f"{p}.mlp.gate_proj.weight"] = np.ones((D, FF), dtype=np.float32)
        tensors[f"{p}.mlp.gate_proj.bias"] = np.ones(FF, dtype=np.float32)
        tensors[f"{p}.mlp.up_proj.weight"] = np.ones((D, FF), dtype=np.float32)
        tensors[f"{p}.mlp.up_proj.bias"] = np.ones(FF, dtype=np.float32)
        tensors[f"{p}.mlp.down_proj.weight"] = np.ones((FF, D), dtype=np.float32)
        tensors[f"{p}.mlp.down_proj.bias"] = np.ones(D, dtype=np.float32)
    return tensors


class TestMapSlncToNative:
    def test_output_finite(self):
        D, NH, NKV, HD, FF, V = 32, 4, 2, 8, 64, 100
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, n_layers=2)
        flat, info = map_slnc_to_native(tensors, 2, D, NH, NKV, HD, FF, V)
        assert np.all(np.isfinite(flat))
        assert flat.dtype == np.float32

    def test_info_has_layers(self):
        D, NH, NKV, HD, FF, V = 16, 2, 2, 8, 32, 50
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, n_layers=3)
        flat, info = map_slnc_to_native(tensors, 3, D, NH, NKV, HD, FF, V)
        assert "layer_size" in info
        assert "total_floats" in info
        assert len(info["layers"]) == 3

    def test_total_floats_matches_array(self):
        D, NH, NKV, HD, FF, V = 16, 2, 1, 8, 32, 50
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, n_layers=1)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert info["total_floats"] == len(flat)

    def test_missing_tensors_fallback_to_zeros(self):
        D, NH, NKV, HD, FF, V = 16, 2, 1, 8, 32, 50
        flat, info = map_slnc_to_native({}, 1, D, NH, NKV, HD, FF, V)
        assert np.all(flat == 0.0)
        assert info["total_floats"] > 0

    def test_qwen_style_prefix(self):
        D, NH, NKV, HD, FF, V = 16, 2, 1, 8, 32, 50
        t1 = _make_tensors(D, NH, NKV, HD, FF, V, 1, prefix="model.layers")
        flat1, _ = map_slnc_to_native(t1, 1, D, NH, NKV, HD, FF, V, is_qwen_style=True)
        t2 = _make_tensors(D, NH, NKV, HD, FF, V, 1, prefix="h")
        flat2, _ = map_slnc_to_native(t2, 1, D, NH, NKV, HD, FF, V, is_qwen_style=False)
        assert np.allclose(flat1, flat2)

    def test_biasless_model(self):
        D, NH, NKV, HD, FF, V = 16, 2, 1, 8, 32, 50
        tensors = {}
        tensors["model.embed_tokens.weight"] = np.ones((V, D), dtype=np.float32)
        tensors["model.norm.weight"] = np.ones(D, dtype=np.float32)
        tensors["model.lm_head.weight"] = np.ones((V, D), dtype=np.float32)
        p = "model.layers.0"
        tensors[f"{p}.input_layernorm.weight"] = np.ones(D, dtype=np.float32)
        tensors[f"{p}.self_attn.q_proj.weight"] = np.ones((D, NH * HD), dtype=np.float32)
        tensors[f"{p}.self_attn.k_proj.weight"] = np.ones((D, NKV * HD), dtype=np.float32)
        tensors[f"{p}.self_attn.v_proj.weight"] = np.ones((D, NKV * HD), dtype=np.float32)
        tensors[f"{p}.self_attn.o_proj.weight"] = np.ones((NH * HD, D), dtype=np.float32)
        tensors[f"{p}.post_attention_layernorm.weight"] = np.ones(D, dtype=np.float32)
        tensors[f"{p}.mlp.gate_proj.weight"] = np.ones((D, FF), dtype=np.float32)
        tensors[f"{p}.mlp.up_proj.weight"] = np.ones((D, FF), dtype=np.float32)
        tensors[f"{p}.mlp.down_proj.weight"] = np.ones((FF, D), dtype=np.float32)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert np.all(np.isfinite(flat))

    def test_lm_head_fallback_to_embed(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 20
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        del tensors["model.lm_head.weight"]
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert np.all(np.isfinite(flat))

    def test_empty_tensors_minimal(self):
        flat, info = map_slnc_to_native({}, 0, 8, 2, 1, 4, 16, 10)
        assert info["total_floats"] > 0
        assert np.all(np.isfinite(flat))
