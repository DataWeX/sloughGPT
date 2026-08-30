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


class TestLayerLayout:
    def test_layer_size_formula(self):
        D, NH, NKV, HD, FF = 16, 2, 1, 8, 32
        expected = (D + D*(NH*HD) + NH*HD + D*(NKV*HD) + NKV*HD
                    + D*(NKV*HD) + NKV*HD + NH*HD*D + D
                    + D + D*FF + FF + D*FF + FF + FF*D + D)
        D2, NH2, NKV2, HD2, FF2, V = D, NH, NKV, HD, FF, 10
        tensors = _make_tensors(D2, NH2, NKV2, HD2, FF2, V, 1)
        _, info = map_slnc_to_native(tensors, 1, D2, NH2, NKV2, HD2, FF2, V)
        assert info["layer_size"] == expected

    def test_total_floats_calculation(self):
        D, NH, NKV, HD, FF, V = 16, 2, 1, 8, 32, 50
        n_layers = 2
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, n_layers)
        flat, info = map_slnc_to_native(tensors, n_layers, D, NH, NKV, HD, FF, V)
        layer_size = info["layer_size"]
        expected = V * D + layer_size * n_layers + D + V * D
        assert info["total_floats"] == expected

    def test_single_layer_offsets(self):
        D, NH, NKV, HD, FF, V = 16, 2, 1, 8, 32, 50
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert 0 in info["layers"]
        assert info["layers"][0]["offset"] == V * D

    def test_two_layer_offsets(self):
        D, NH, NKV, HD, FF, V = 16, 2, 1, 8, 32, 50
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 2)
        flat, info = map_slnc_to_native(tensors, 2, D, NH, NKV, HD, FF, V)
        assert info["layers"][0]["offset"] == V * D
        assert info["layers"][1]["offset"] == V * D + info["layer_size"]

    def test_no_layers(self):
        D, NH, NKV, HD, FF, V = 16, 2, 1, 8, 32, 50
        flat, info = map_slnc_to_native({}, 0, D, NH, NKV, HD, FF, V)
        expected = V * D + D + V * D
        assert info["total_floats"] == expected


class TestTensorContent:
    def test_embed_section(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 10
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        embed = flat[:V*D]
        assert np.all(embed == 1.0)

    def test_layer_norm_section(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 10
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        start = info["layers"][0]["offset"]
        ln = flat[start:start+D]
        assert np.all(ln == 1.0)

    def test_all_ones_tensor(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 10
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert np.all(flat == 1.0)

    def test_zeros_for_missing(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 10
        flat, info = map_slnc_to_native({}, 1, D, NH, NKV, HD, FF, V)
        assert np.all(flat == 0.0)

    def test_partial_tensors_fills_zeros(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 10
        tensors = {"model.embed_tokens.weight": np.ones((V, D), dtype=np.float32)}
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        embed = flat[:V*D]
        assert np.all(embed == 1.0)
        assert info["total_floats"] > V * D


class TestDifferentConfigs:
    def test_small_model(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 10
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert np.all(np.isfinite(flat))

    def test_larger_model(self):
        D, NH, NKV, HD, FF, V = 128, 8, 4, 16, 512, 1000
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 4)
        flat, info = map_slnc_to_native(tensors, 4, D, NH, NKV, HD, FF, V)
        assert np.all(np.isfinite(flat))
        assert info["total_floats"] == len(flat)

    def test_gqa_config(self):
        D, NH, NKV, HD, FF, V = 32, 8, 2, 4, 64, 100
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 2)
        flat, info = map_slnc_to_native(tensors, 2, D, NH, NKV, HD, FF, V)
        assert np.all(np.isfinite(flat))

    def test_mqa_config(self):
        D, NH, NKV, HD, FF, V = 32, 8, 1, 4, 64, 100
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 2)
        flat, info = map_slnc_to_native(tensors, 2, D, NH, NKV, HD, FF, V)
        assert np.all(np.isfinite(flat))

    def test_many_layers(self):
        D, NH, NKV, HD, FF, V = 16, 2, 1, 8, 32, 20
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 8)
        flat, info = map_slnc_to_native(tensors, 8, D, NH, NKV, HD, FF, V)
        assert len(info["layers"]) == 8
        assert np.all(np.isfinite(flat))


class TestAlternativeNaming:
    def test_wte_embed_fallback(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 20
        tensors = {}
        tensors["wte.weight"] = np.ones((V, D), dtype=np.float32)
        tensors["model.norm.weight"] = np.ones(D, dtype=np.float32)
        tensors["model.lm_head.weight"] = np.ones((V, D), dtype=np.float32)
        flat, info = map_slnc_to_native(tensors, 0, D, NH, NKV, HD, FF, V)
        assert np.all(np.isfinite(flat))

    def test_ln_f_norm_fallback(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 20
        tensors = {}
        tensors["model.embed_tokens.weight"] = np.ones((V, D), dtype=np.float32)
        tensors["ln_f.weight"] = np.ones(D, dtype=np.float32)
        tensors["model.lm_head.weight"] = np.ones((V, D), dtype=np.float32)
        flat, info = map_slnc_to_native(tensors, 0, D, NH, NKV, HD, FF, V)
        assert np.all(np.isfinite(flat))

    def test_o_proj_bias_fallback(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 20
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        del tensors["model.layers.0.self_attn.o_proj.bias"]
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert np.all(np.isfinite(flat))

    def test_wte_lm_head_tied(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 20
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        del tensors["model.lm_head.weight"]
        tensors["wte.weight"] = tensors["model.embed_tokens.weight"].copy()
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert np.all(np.isfinite(flat))


class TestInfoDict:
    def test_info_keys(self):
        D, NH, NKV, HD, FF, V = 16, 2, 1, 8, 32, 50
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 2)
        _, info = map_slnc_to_native(tensors, 2, D, NH, NKV, HD, FF, V)
        assert "total_floats" in info
        assert "layer_size" in info
        assert "layers" in info

    def test_layer_info_keys(self):
        D, NH, NKV, HD, FF, V = 16, 2, 1, 8, 32, 50
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        _, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert "offset" in info["layers"][0]
        assert "size" in info["layers"][0]

    def test_layer_info_sizes_match(self):
        D, NH, NKV, HD, FF, V = 16, 2, 1, 8, 32, 50
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 3)
        _, info = map_slnc_to_native(tensors, 3, D, NH, NKV, HD, FF, V)
        for layer in info["layers"].values():
            assert layer["size"] == info["layer_size"]

    def test_info_total_matches_offset(self):
        D, NH, NKV, HD, FF, V = 16, 2, 1, 8, 32, 50
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 2)
        _, info = map_slnc_to_native(tensors, 2, D, NH, NKV, HD, FF, V)
        last_layer = info["layers"][1]
        end = last_layer["offset"] + last_layer["size"] + D + V * D
        assert info["total_floats"] == end


class TestTensorShapes:
    def test_embed_tokens_shape(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 20
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert tensors["model.embed_tokens.weight"].shape == (V, D)

    def test_q_proj_shape(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 20
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        q = tensors["model.layers.0.self_attn.q_proj.weight"]
        assert q.shape == (D, NH * HD)

    def test_k_proj_shape(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 20
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        k = tensors["model.layers.0.self_attn.k_proj.weight"]
        assert k.shape == (D, NKV * HD)

    def test_gate_proj_shape(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 20
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        g = tensors["model.layers.0.mlp.gate_proj.weight"]
        assert g.shape == (D, FF)

    def test_down_proj_shape(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 20
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        d = tensors["model.layers.0.mlp.down_proj.weight"]
        assert d.shape == (FF, D)

    def test_bias_shapes(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 20
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        assert tensors["model.layers.0.self_attn.q_proj.bias"].shape == (NH * HD,)
        assert tensors["model.layers.0.self_attn.k_proj.bias"].shape == (NKV * HD,)
        assert tensors["model.layers.0.self_attn.v_proj.bias"].shape == (NKV * HD,)
        assert tensors["model.layers.0.self_attn.o_proj.bias"].shape == (D,)
        assert tensors["model.layers.0.mlp.gate_proj.bias"].shape == (FF,)
        assert tensors["model.layers.0.mlp.up_proj.bias"].shape == (FF,)
        assert tensors["model.layers.0.mlp.down_proj.bias"].shape == (D,)


class TestNumericalStability:
    def test_large_values(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 10
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        for k in tensors:
            tensors[k] = np.full_like(tensors[k], 1e6)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert np.all(np.isfinite(flat))

    def test_small_values(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 10
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        for k in tensors:
            tensors[k] = np.full_like(tensors[k], 1e-6)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert np.all(np.isfinite(flat))

    def test_negative_values(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 10
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        for k in tensors:
            tensors[k] = np.full_like(tensors[k], -1.0)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert np.all(flat == -1.0)

    def test_mixed_values(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 10
        tensors = {}
        tensors["model.embed_tokens.weight"] = np.random.randn(V, D).astype(np.float32)
        tensors["model.norm.weight"] = np.random.randn(D).astype(np.float32)
        tensors["model.lm_head.weight"] = np.random.randn(V, D).astype(np.float32)
        flat, info = map_slnc_to_native(tensors, 0, D, NH, NKV, HD, FF, V)
        assert np.all(np.isfinite(flat))

    def test_zeros_tensor(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 10
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        for k in tensors:
            tensors[k] = np.zeros_like(tensors[k])
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert np.all(flat == 0.0)

    def test_nan_not_in_output(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 10
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert not np.any(np.isnan(flat))

    def test_inf_not_in_output(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 10
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert not np.any(np.isinf(flat))


class TestRavelBehavior:
    def test_2d_weight_raveled(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 10
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        tensors["model.layers.0.self_attn.q_proj.weight"] = np.arange(D * NH * HD, dtype=np.float32).reshape(D, NH * HD)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        start = info["layers"][0]["offset"] + D
        q_section = flat[start:start + D * NH * HD]
        expected = np.arange(D * NH * HD, dtype=np.float32)
        np.testing.assert_array_equal(q_section, expected)

    def test_1d_bias_not_raveled(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 10
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        tensors["model.layers.0.self_attn.q_proj.bias"] = np.arange(NH * HD, dtype=np.float32)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        start = info["layers"][0]["offset"] + D + D * NH * HD
        bias_section = flat[start:start + NH * HD]
        expected = np.arange(NH * HD, dtype=np.float32)
        np.testing.assert_array_equal(bias_section, expected)

    def test_truncation_if_oversized(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 10
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        tensors["model.layers.0.self_attn.q_proj.weight"] = np.ones((D + 5, NH * HD), dtype=np.float32)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert np.all(np.isfinite(flat))


class TestEdgeCases:
    def test_zero_layers_embed_only(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 10
        tensors = {}
        tensors["model.embed_tokens.weight"] = np.ones((V, D), dtype=np.float32)
        tensors["model.norm.weight"] = np.ones(D, dtype=np.float32)
        tensors["model.lm_head.weight"] = np.ones((V, D), dtype=np.float32)
        flat, info = map_slnc_to_native(tensors, 0, D, NH, NKV, HD, FF, V)
        expected = V * D + D + V * D
        assert info["total_floats"] == expected
        assert len(flat) == expected

    def test_vocab_size_one(self):
        D, NH, NKV, HD, FF, V = 8, 2, 1, 4, 16, 1
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert np.all(np.isfinite(flat))
        assert info["total_floats"] > 0

    def test_hidden_dim_one(self):
        D, NH, NKV, HD, FF, V = 1, 2, 1, 1, 8, 10
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert np.all(np.isfinite(flat))

    def test_head_dim_one(self):
        D, NH, NKV, HD, FF, V = 16, 4, 2, 1, 32, 20
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert np.all(np.isfinite(flat))

    def test_gqa_heads_equal(self):
        D, NH, NKV, HD, FF, V = 16, 4, 4, 4, 32, 20
        tensors = _make_tensors(D, NH, NKV, HD, FF, V, 1)
        flat, info = map_slnc_to_native(tensors, 1, D, NH, NKV, HD, FF, V)
        assert np.all(np.isfinite(flat))

