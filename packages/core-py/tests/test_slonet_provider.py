"""Tests for the universal HF→SloNet weight converter."""

import json
import struct
import tempfile

import numpy as np
import pytest

from domains.inference.slonet_provider import convert_hf_to_slonet
from domains.training.slonet import SloTransformer


def _fake_gpt2_state_dict():
    """Create a minimal GPT-2 style state dict (fused QKV, GELU MLP)."""
    n_embed, n_layer = 64, 2
    sd = {}
    sd["wte.weight"] = np.random.randn(1000, n_embed).astype(np.float32)
    sd["wpe.weight"] = np.random.randn(512, n_embed).astype(np.float32)
    sd["ln_f.weight"] = np.ones(n_embed, dtype=np.float32)
    sd["ln_f.bias"] = np.zeros(n_embed, dtype=np.float32)
    for i in range(n_layer):
        sd[f"h.{i}.ln_1.weight"] = np.ones(n_embed, dtype=np.float32)
        sd[f"h.{i}.ln_1.bias"] = np.zeros(n_embed, dtype=np.float32)
        sd[f"h.{i}.attn.c_attn.weight"] = np.random.randn(n_embed, 3 * n_embed).astype(np.float32)
        sd[f"h.{i}.attn.c_attn.bias"] = np.random.randn(3 * n_embed).astype(np.float32)
        sd[f"h.{i}.attn.c_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
        sd[f"h.{i}.attn.c_proj.bias"] = np.zeros(n_embed, dtype=np.float32)
        sd[f"h.{i}.ln_2.weight"] = np.ones(n_embed, dtype=np.float32)
        sd[f"h.{i}.ln_2.bias"] = np.zeros(n_embed, dtype=np.float32)
        sd[f"h.{i}.mlp.c_fc.weight"] = np.random.randn(n_embed, 4 * n_embed).astype(np.float32)
        sd[f"h.{i}.mlp.c_fc.bias"] = np.zeros(4 * n_embed, dtype=np.float32)
        sd[f"h.{i}.mlp.c_proj.weight"] = np.random.randn(4 * n_embed, n_embed).astype(np.float32)
        sd[f"h.{i}.mlp.c_proj.bias"] = np.zeros(n_embed, dtype=np.float32)
    return sd


def test_universal_converter_gpt2():
    """GPT-2 style: fused QKV, GELU MLP, no positional embeddings in SloNet."""
    sd = _fake_gpt2_state_dict()
    result = convert_hf_to_slonet(sd, n_layer=2)

    # Should have: tok_emb, lm_head, 2x (attn_norm, q/k/v, o_proj, ff_norm, w1/w2/w3), final norm
    # Check key groups exist
    assert "tok_emb.weight" in result
    assert "lm_head.weight" in result
    for i in range(2):
        assert f"blocks.{i}.attn_norm.weight" in result
        assert f"blocks.{i}.attn.q_proj.weight" in result
        assert f"blocks.{i}.attn.k_proj.weight" in result
        assert f"blocks.{i}.attn.v_proj.weight" in result
        assert f"blocks.{i}.attn.o_proj.weight" in result
        assert f"blocks.{i}.ff_norm.weight" in result
        assert f"blocks.{i}.ff.w1.weight" in result
        assert f"blocks.{i}.ff.w2.weight" in result
        assert f"blocks.{i}.ff.w3.weight" in result  # synthesized identity

    # QKV split: fused is (64, 192) → transpose → (192, 64) → split into 3 × (64, 64)
    q = result["blocks.0.attn.q_proj.weight"]
    k = result["blocks.0.attn.k_proj.weight"]
    v = result["blocks.0.attn.v_proj.weight"]
    assert q.shape == (64, 64)
    assert k.shape == q.shape
    assert v.shape == q.shape

    # FF: w1 is the first linear (from c_fc), w2 is the second (from c_proj), w3 is identity
    w1 = result["blocks.0.ff.w1.weight"]
    w3 = result["blocks.0.ff.w3.weight"]
    assert w1.shape == (4 * 64, 64)  # transposed from (64, 256)
    assert w3.shape == w1.shape
    # w3 should be zeros (identity when multiplied with sigmoid-like activation)
    assert np.allclose(w3, 0.0)


def test_universal_converter_llama_style():
    """LLaMA-style: split QKV, SwiGLU MLP."""
    n_embed, n_layer = 64, 2
    sd = {}
    sd["model.embed_tokens.weight"] = np.random.randn(1000, n_embed).astype(np.float32)
    sd["model.norm.weight"] = np.ones(n_embed, dtype=np.float32)
    for i in range(n_layer):
        sd[f"model.layers.{i}.input_layernorm.weight"] = np.ones(n_embed, dtype=np.float32)
        sd[f"model.layers.{i}.self_attn.q_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
        sd[f"model.layers.{i}.self_attn.k_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
        sd[f"model.layers.{i}.self_attn.v_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
        sd[f"model.layers.{i}.self_attn.o_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
        sd[f"model.layers.{i}.post_attention_layernorm.weight"] = np.ones(n_embed, dtype=np.float32)
        sd[f"model.layers.{i}.mlp.gate_proj.weight"] = np.random.randn(4 * n_embed, n_embed).astype(np.float32)
        sd[f"model.layers.{i}.mlp.up_proj.weight"] = np.random.randn(4 * n_embed, n_embed).astype(np.float32)
        sd[f"model.layers.{i}.mlp.down_proj.weight"] = np.random.randn(n_embed, 4 * n_embed).astype(np.float32)

    result = convert_hf_to_slonet(sd, n_layer=2)

    assert "tok_emb.weight" in result
    assert "lm_head.weight" in result
    for i in range(2):
        assert f"blocks.{i}.ff.w1.weight" in result  # gate
        assert f"blocks.{i}.ff.w2.weight" in result  # down
        assert f"blocks.{i}.ff.w3.weight" in result  # up
        # No synthesized zeros — these should be real weights
        assert not np.allclose(result[f"blocks.{i}.ff.w3.weight"], 0.0)


def test_to_server_builds_guard_backed_server():
    """to_server() wraps the provider's model/tokenizer in a SloNetServer."""
    from unittest.mock import MagicMock

    from domains.inference.slonet_provider import SloNetChatProvider
    from domains.infrastructure.slonet_server import SloNetServer

    provider = SloNetChatProvider.__new__(SloNetChatProvider)
    provider._model = MagicMock()
    provider._tokenizer = MagicMock()
    provider._model_id = "test-slo"

    guard = MagicMock()
    server = provider.to_server(process_guard=guard)

    assert isinstance(server, SloNetServer)
    assert server._process_guard is guard
    assert server._model is provider._model
    assert server._tokenizer is provider._tokenizer
    assert server._model_id == "test-slo"

    no_guard = provider.to_server()
    assert no_guard._process_guard is None


class TestConvertHFToSloNet:
    def test_returns_dict(self):
        sd = _fake_gpt2_state_dict()
        result = convert_hf_to_slonet(sd, n_layer=2)
        assert isinstance(result, dict)

    def test_all_values_are_numpy(self):
        sd = _fake_gpt2_state_dict()
        result = convert_hf_to_slonet(sd, n_layer=2)
        for k, v in result.items():
            assert isinstance(v, np.ndarray), f"{k} is not ndarray"

    def test_final_norm_exists(self):
        sd = _fake_gpt2_state_dict()
        result = convert_hf_to_slonet(sd, n_layer=2)
        assert "norm.weight" in result

    def test_attn_bias_converted(self):
        sd = _fake_gpt2_state_dict()
        result = convert_hf_to_slonet(sd, n_layer=2)
        for i in range(2):
            assert f"blocks.{i}.attn.q_proj.bias" in result
            assert f"blocks.{i}.attn.k_proj.bias" in result
            assert f"blocks.{i}.attn.v_proj.bias" in result

    def test_ff_bias_converted(self):
        sd = _fake_gpt2_state_dict()
        result = convert_hf_to_slonet(sd, n_layer=2)
        for i in range(2):
            assert f"blocks.{i}.ff.w1.bias" in result
            assert f"blocks.{i}.ff.w2.bias" in result

    def test_single_layer(self):
        sd = _fake_gpt2_state_dict()
        result = convert_hf_to_slonet(sd, n_layer=1)
        assert "blocks.0.attn.q_proj.weight" in result
        assert "blocks.1.attn.q_proj.weight" not in result

    def test_large_n_embed(self):
        n_embed, n_layer = 256, 2
        sd = {}
        sd["wte.weight"] = np.random.randn(1000, n_embed).astype(np.float32)
        sd["wpe.weight"] = np.random.randn(512, n_embed).astype(np.float32)
        sd["ln_f.weight"] = np.ones(n_embed, dtype=np.float32)
        sd["ln_f.bias"] = np.zeros(n_embed, dtype=np.float32)
        for i in range(n_layer):
            sd[f"h.{i}.ln_1.weight"] = np.ones(n_embed, dtype=np.float32)
            sd[f"h.{i}.ln_1.bias"] = np.zeros(n_embed, dtype=np.float32)
            sd[f"h.{i}.attn.c_attn.weight"] = np.random.randn(n_embed, 3 * n_embed).astype(np.float32)
            sd[f"h.{i}.attn.c_attn.bias"] = np.random.randn(3 * n_embed).astype(np.float32)
            sd[f"h.{i}.attn.c_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
            sd[f"h.{i}.attn.c_proj.bias"] = np.zeros(n_embed, dtype=np.float32)
            sd[f"h.{i}.ln_2.weight"] = np.ones(n_embed, dtype=np.float32)
            sd[f"h.{i}.ln_2.bias"] = np.zeros(n_embed, dtype=np.float32)
            sd[f"h.{i}.mlp.c_fc.weight"] = np.random.randn(n_embed, 4 * n_embed).astype(np.float32)
            sd[f"h.{i}.mlp.c_fc.bias"] = np.zeros(4 * n_embed, dtype=np.float32)
            sd[f"h.{i}.mlp.c_proj.weight"] = np.random.randn(4 * n_embed, n_embed).astype(np.float32)
            sd[f"h.{i}.mlp.c_proj.bias"] = np.zeros(n_embed, dtype=np.float32)
        result = convert_hf_to_slonet(sd, n_layer=2)
        assert result["blocks.0.attn.q_proj.weight"].shape == (n_embed, n_embed)


class TestFusedConvertLoad:
    """Verify fused convert+load produces identical weights to separate convert+load."""

    def _make_model(self, n_embed=64, n_layer=2):
        return SloTransformer(
            vocab_size=1000,
            n_embed=n_embed,
            n_layer=n_layer,
            n_head=4,
            intermediate_size=4 * n_embed,
            block_size=512,
            max_seq_len=512,
            use_rope=False,
            dropout=0.0,
            tie_weights=True,
            use_abs_pos_emb=True,
            norm_type="layer_norm",
            activation="gelu",
            _lazy=True,
        )

    def _fake_gpt2_sd(self, n_embed=64, n_layer=2):
        sd = {}
        sd["wte.weight"] = np.random.randn(1000, n_embed).astype(np.float32)
        sd["wpe.weight"] = np.random.randn(512, n_embed).astype(np.float32)
        sd["ln_f.weight"] = np.ones(n_embed, dtype=np.float32)
        sd["ln_f.bias"] = np.zeros(n_embed, dtype=np.float32)
        for i in range(n_layer):
            sd[f"h.{i}.ln_1.weight"] = np.ones(n_embed, dtype=np.float32)
            sd[f"h.{i}.ln_1.bias"] = np.zeros(n_embed, dtype=np.float32)
            sd[f"h.{i}.attn.c_attn.weight"] = np.random.randn(n_embed, 3 * n_embed).astype(np.float32)
            sd[f"h.{i}.attn.c_attn.bias"] = np.random.randn(3 * n_embed).astype(np.float32)
            sd[f"h.{i}.attn.c_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
            sd[f"h.{i}.attn.c_proj.bias"] = np.zeros(n_embed, dtype=np.float32)
            sd[f"h.{i}.ln_2.weight"] = np.ones(n_embed, dtype=np.float32)
            sd[f"h.{i}.ln_2.bias"] = np.zeros(n_embed, dtype=np.float32)
            sd[f"h.{i}.mlp.c_fc.weight"] = np.random.randn(n_embed, 4 * n_embed).astype(np.float32)
            sd[f"h.{i}.mlp.c_fc.bias"] = np.zeros(4 * n_embed, dtype=np.float32)
            sd[f"h.{i}.mlp.c_proj.weight"] = np.random.randn(4 * n_embed, n_embed).astype(np.float32)
            sd[f"h.{i}.mlp.c_proj.bias"] = np.zeros(n_embed, dtype=np.float32)
        return sd

    def test_fused_matches_separate_gpt2(self):
        """Fused path produces identical weights to separate convert+load for GPT-2."""
        sd = self._fake_gpt2_sd()

        # Separate path: convert then load
        model_ref = self._make_model()
        mapped = convert_hf_to_slonet(sd, n_layer=2)
        model_ref.load_state_dict(mapped)
        ref_params = {k: v.data.copy() for k, v in dict(model_ref._named_parameters()).items()}

        # Fused path: convert with param_map
        model_fused = self._make_model()
        param_map = dict(model_fused._named_parameters())
        convert_hf_to_slonet(sd, n_layer=2, param_map=param_map)

        # Verify all parameters match
        for key in ref_params:
            assert key in param_map, f"Missing key in fused path: {key}"
            np.testing.assert_array_equal(
                ref_params[key], param_map[key].data,
                err_msg=f"Mismatch for {key}",
            )

    def test_fused_matches_separate_llama(self):
        """Fused path produces identical weights for LLaMA-style (SwiGLU, split QKV)."""
        n_embed, n_layer = 64, 2
        sd = {}
        sd["model.embed_tokens.weight"] = np.random.randn(1000, n_embed).astype(np.float32)
        sd["model.norm.weight"] = np.ones(n_embed, dtype=np.float32)
        for i in range(n_layer):
            sd[f"model.layers.{i}.input_layernorm.weight"] = np.ones(n_embed, dtype=np.float32)
            sd[f"model.layers.{i}.self_attn.q_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
            sd[f"model.layers.{i}.self_attn.k_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
            sd[f"model.layers.{i}.self_attn.v_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
            sd[f"model.layers.{i}.self_attn.o_proj.weight"] = np.random.randn(n_embed, n_embed).astype(np.float32)
            sd[f"model.layers.{i}.post_attention_layernorm.weight"] = np.ones(n_embed, dtype=np.float32)
            sd[f"model.layers.{i}.mlp.gate_proj.weight"] = np.random.randn(4 * n_embed, n_embed).astype(np.float32)
            sd[f"model.layers.{i}.mlp.up_proj.weight"] = np.random.randn(4 * n_embed, n_embed).astype(np.float32)
            sd[f"model.layers.{i}.mlp.down_proj.weight"] = np.random.randn(n_embed, 4 * n_embed).astype(np.float32)

        model_ref = self._make_model()
        mapped = convert_hf_to_slonet(sd, n_layer=2)
        model_ref.load_state_dict(mapped)
        ref_params = {k: v.data.copy() for k, v in dict(model_ref._named_parameters()).items()}

        model_fused = self._make_model()
        param_map = dict(model_fused._named_parameters())
        convert_hf_to_slonet(sd, n_layer=2, param_map=param_map)

        for key in ref_params:
            assert key in param_map, f"Missing key in fused path: {key}"
            np.testing.assert_array_equal(
                ref_params[key], param_map[key].data,
                err_msg=f"Mismatch for {key}",
            )


def _build_slnc_file(tensors, config, n_layer, n_embd, n_head):
    """Build a valid .slnc binary file for testing."""
    from domains.infrastructure.slnc.spec import (
        ALIGNMENT, DTYPE_FLOAT32, MAGIC, VERSION,
        compute_header_size, compute_tensor_entry_size,
        dtype_to_code, _align,
    )

    json_bytes = json.dumps(config, sort_keys=True).encode()
    header_size = compute_header_size(json_bytes)

    table_size = 0
    for t in tensors:
        name_bytes = t["name"].encode()
        ndim = t["data"].ndim
        table_size += compute_tensor_entry_size(ndim, len(name_bytes))

    data_start = _align(header_size + table_size)

    data_offsets = []
    current = data_start
    for t in tensors:
        data_offsets.append(current)
        current += t["data"].nbytes

    tensor_table = bytearray()
    for t, data_off in zip(tensors, data_offsets):
        name = t["name"]
        data = t["data"]
        name_bytes = name.encode()
        ndim = data.ndim
        tensor_table += struct.pack("<I", len(name_bytes))
        tensor_table += name_bytes
        tensor_table += struct.pack("<Q", data_off)
        tensor_table += struct.pack("<I", data.nbytes)
        tensor_table += struct.pack("<I", ndim)
        for dim in data.shape:
            tensor_table += struct.pack("<I", dim)
        tensor_table += struct.pack("<I", dtype_to_code(data.dtype))
        tensor_table += struct.pack("<I", 0)

    header = bytearray()
    header += MAGIC
    header += struct.pack("<I", VERSION)
    header += struct.pack("<I", 0)
    header += struct.pack("<I", n_layer)
    header += struct.pack("<I", n_embd)
    header += struct.pack("<I", n_head)
    header += struct.pack("<I", n_embd * 4)
    header += struct.pack("<I", config.get("vocab_size", 1000))
    header += struct.pack("<I", config.get("n_positions", 512))
    header += struct.pack("<I", n_layer)
    header += struct.pack("<I", 128)
    header += struct.pack("<I", len(tensors))
    header += struct.pack("<I", data_start)
    header += b"\x00" * 24
    header += struct.pack("<I", len(json_bytes))
    header += json_bytes

    while len(header) % ALIGNMENT != 0:
        header += b"\x00"

    pre_data = len(header) + len(tensor_table)
    padding = b"\x00" * (data_start - pre_data)

    tensor_data = bytearray()
    for t in tensors:
        tensor_data += t["data"].tobytes()

    return bytes(header) + bytes(tensor_table) + padding + bytes(tensor_data)


class TestFromSlncPipeline:
    """Integration test: full from_slnc pipeline with realistic .slnc file."""

    def test_gpt2_style_fused_pipeline(self, tmp_path):
        """Load GPT-2 style .slnc file through full from_slnc with fused path."""
        from unittest.mock import MagicMock, patch
        from domains.inference.slonet_provider import SloNetChatProvider

        n_embed, n_layer, n_head = 64, 2, 4
        vocab_size = 1000

        config = {
            "model": "test-gpt2",
            "vocab_size": vocab_size,
            "n_positions": 512,
            "n_embd": n_embed,
            "n_head": n_head,
            "n_layer": n_layer,
            "n_inner": n_embed * 4,
        }

        tensors = []
        tensors.append({"name": "wte.weight", "data": np.random.randn(vocab_size, n_embed).astype(np.float32)})
        tensors.append({"name": "ln_f.weight", "data": np.ones(n_embed, dtype=np.float32)})
        tensors.append({"name": "ln_f.bias", "data": np.zeros(n_embed, dtype=np.float32)})
        for i in range(n_layer):
            tensors.append({"name": f"h.{i}.ln_1.weight", "data": np.ones(n_embed, dtype=np.float32)})
            tensors.append({"name": f"h.{i}.ln_1.bias", "data": np.zeros(n_embed, dtype=np.float32)})
            tensors.append({"name": f"h.{i}.attn.c_attn.weight", "data": np.random.randn(n_embed, 3 * n_embed).astype(np.float32)})
            tensors.append({"name": f"h.{i}.attn.c_attn.bias", "data": np.random.randn(3 * n_embed).astype(np.float32)})
            tensors.append({"name": f"h.{i}.attn.c_proj.weight", "data": np.random.randn(n_embed, n_embed).astype(np.float32)})
            tensors.append({"name": f"h.{i}.attn.c_proj.bias", "data": np.zeros(n_embed, dtype=np.float32)})
            tensors.append({"name": f"h.{i}.ln_2.weight", "data": np.ones(n_embed, dtype=np.float32)})
            tensors.append({"name": f"h.{i}.ln_2.bias", "data": np.zeros(n_embed, dtype=np.float32)})
            tensors.append({"name": f"h.{i}.mlp.c_fc.weight", "data": np.random.randn(n_embed, 4 * n_embed).astype(np.float32)})
            tensors.append({"name": f"h.{i}.mlp.c_fc.bias", "data": np.zeros(4 * n_embed, dtype=np.float32)})
            tensors.append({"name": f"h.{i}.mlp.c_proj.weight", "data": np.random.randn(4 * n_embed, n_embed).astype(np.float32)})
            tensors.append({"name": f"h.{i}.mlp.c_proj.bias", "data": np.zeros(n_embed, dtype=np.float32)})

        path = tmp_path / "test_gpt2.slnc"
        path.write_bytes(_build_slnc_file(tensors, config, n_layer, n_embed, n_head))

        mock_tokenizer = MagicMock()
        with patch.object(SloNetChatProvider, "_load_tokenizer", return_value=mock_tokenizer):
            provider = SloNetChatProvider.from_slnc(str(path), model_id="gpt2")

        # Verify model loaded correctly
        assert provider._model is not None
        params = dict(provider._model._named_parameters())
        assert "tok_emb.weight" in params
        assert params["tok_emb.weight"].data.shape == (vocab_size, n_embed)
        assert params["blocks.0.attn.q_proj.weight"].data.shape == (n_embed, n_embed)

        # Verify lm_head tied to tok_emb
        np.testing.assert_array_equal(
            params["tok_emb.weight"].data,
            params["lm_head.weight"].data,
        )

        # Verify w3 bias is ones (GELU synthesis) — shape matches intermediate_size
        np.testing.assert_array_equal(
            params["blocks.0.ff.w3.bias"].data,
            np.ones(n_embed * 4, dtype=np.float32),
        )
