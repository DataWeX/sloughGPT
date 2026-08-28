"""Tests for the weight loading infrastructure."""

import json
import struct

import numpy as np
import pytest

from domains.infrastructure.slnc.spec import (
    ALIGNMENT, DTYPE_FLOAT32, MAGIC, VERSION,
    compute_header_size, compute_tensor_entry_size,
    dtype_to_code, _align,
)
from domains.infrastructure.slnc.parser import SLNCParser
from domains.infrastructure.weight_loader import (
    DirectWeightLoader, build_load_plan, LoadPlan, TensorMapping,
    load_into_model, WeightLoadResult,
)
from domains.training.slonet import SloTransformer


def _build_slnc_file(tensors, config, n_layer, n_embd, n_head):
    """Build a valid .slnc binary file."""
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


def _make_gpt2_tensors(n_embed=64, n_layer=2, vocab_size=1000):
    """Create GPT-2 style tensors for testing."""
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
    return tensors


def _make_llama_tensors(n_embed=64, n_layer=2, vocab_size=1000):
    """Create LLaMA style tensors for testing."""
    tensors = []
    tensors.append({"name": "model.embed_tokens.weight", "data": np.random.randn(vocab_size, n_embed).astype(np.float32)})
    tensors.append({"name": "model.norm.weight", "data": np.ones(n_embed, dtype=np.float32)})
    for i in range(n_layer):
        tensors.append({"name": f"model.layers.{i}.input_layernorm.weight", "data": np.ones(n_embed, dtype=np.float32)})
        tensors.append({"name": f"model.layers.{i}.self_attn.q_proj.weight", "data": np.random.randn(n_embed, n_embed).astype(np.float32)})
        tensors.append({"name": f"model.layers.{i}.self_attn.k_proj.weight", "data": np.random.randn(n_embed, n_embed).astype(np.float32)})
        tensors.append({"name": f"model.layers.{i}.self_attn.v_proj.weight", "data": np.random.randn(n_embed, n_embed).astype(np.float32)})
        tensors.append({"name": f"model.layers.{i}.self_attn.o_proj.weight", "data": np.random.randn(n_embed, n_embed).astype(np.float32)})
        tensors.append({"name": f"model.layers.{i}.post_attention_layernorm.weight", "data": np.ones(n_embed, dtype=np.float32)})
        tensors.append({"name": f"model.layers.{i}.mlp.gate_proj.weight", "data": np.random.randn(4 * n_embed, n_embed).astype(np.float32)})
        tensors.append({"name": f"model.layers.{i}.mlp.up_proj.weight", "data": np.random.randn(4 * n_embed, n_embed).astype(np.float32)})
        tensors.append({"name": f"model.layers.{i}.mlp.down_proj.weight", "data": np.random.randn(n_embed, 4 * n_embed).astype(np.float32)})
    return tensors


def _make_model(n_embed=64, n_layer=2, vocab_size=1000, activation="gelu"):
    """Create a SloTransformer model for testing."""
    return SloTransformer(
        vocab_size=vocab_size,
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
        activation=activation,
        _lazy=True,
    )


class TestBuildLoadPlan:
    """Test load plan construction."""

    def test_gpt2_plan(self):
        n_embed, n_layer = 64, 2
        sd = {}
        sd["wte.weight"] = np.random.randn(1000, n_embed).astype(np.float32)
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

        config = {"architectures": ["GPT2LMHeadModel"], "vocab_size": 1000, "n_positions": 512,
                  "n_embd": n_embed, "n_head": 4, "n_layer": n_layer, "n_inner": n_embed * 4}
        plan = build_load_plan(sd, n_layer, config)

        assert plan.arch_name == "GPT2LMHeadModel"
        assert plan.n_layer == n_layer
        assert len(plan.tied_weights) == 1  # lm_head tied to tok_emb
        assert len(plan.synthesized_params) > 0  # w3 synthesis

    def test_llama_plan(self):
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

        config = {"architectures": ["LlamaForCausalLM"], "vocab_size": 1000, "n_positions": 512,
                  "n_embd": n_embed, "n_head": 4, "n_layer": n_layer, "n_inner": n_embed * 4,
                  "hidden_act": "silu", "rms_norm_eps": 1e-5}
        plan = build_load_plan(sd, n_layer, config)

        assert "llama" in plan.arch_name.lower() or "LlamaForCausalLM" in plan.arch_name
        assert len(plan.synthesized_params) == 0  # SwiGLU — no w3 synthesis


class TestDirectWeightLoader:
    """Test the direct weight loader end-to-end."""

    def test_gpt2_direct_load_matches_fused(self, tmp_path):
        """DirectWeightLoader produces identical weights to fused convert_hf_to_slonet."""
        from domains.inference.slonet_provider import convert_hf_to_slonet

        n_embed, n_layer = 64, 2
        sd = {}
        sd["wte.weight"] = np.random.randn(1000, n_embed).astype(np.float32)
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

        config = {"architectures": ["GPT2LMHeadModel"], "vocab_size": 1000, "n_positions": 512,
                  "n_embd": n_embed, "n_head": 4, "n_layer": n_layer, "n_inner": n_embed * 4}
        tensors = [{"name": k, "data": v} for k, v in sd.items()]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors, config, n_layer, n_embed, 4))

        model_ref = _make_model(n_embed, n_layer)
        param_map_ref = dict(model_ref._named_parameters())
        convert_hf_to_slonet(sd, n_layer=n_layer, param_map=param_map_ref)
        ref_params = {k: v.data.copy() for k, v in param_map_ref.items()}

        parser = SLNCParser(str(path))
        model_direct = _make_model(n_embed, n_layer)
        loader = DirectWeightLoader(parser, sd, config)
        loader.load(model_direct)
        direct_params = {k: v.data.copy() for k, v in dict(model_direct._named_parameters()).items()}

        for key in ref_params:
            assert key in direct_params, f"Missing key in direct path: {key}"
            np.testing.assert_array_equal(
                ref_params[key], direct_params[key],
                err_msg=f"Mismatch for {key}",
            )

        parser.close()

    def test_llama_direct_load_matches_fused(self, tmp_path):
        """DirectWeightLoader produces identical weights for LLaMA (SwiGLU)."""
        from domains.inference.slonet_provider import convert_hf_to_slonet

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

        config = {"architectures": ["LlamaForCausalLM"], "vocab_size": 1000, "n_positions": 512,
                  "n_embd": n_embed, "n_head": 4, "n_layer": n_layer, "n_inner": n_embed * 4,
                  "hidden_act": "silu", "rms_norm_eps": 1e-5}
        tensors = [{"name": k, "data": v} for k, v in sd.items()]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors, config, n_layer, n_embed, 4))

        model_ref = _make_model(n_embed, n_layer, activation="silu")
        param_map_ref = dict(model_ref._named_parameters())
        convert_hf_to_slonet(sd, n_layer=n_layer, param_map=param_map_ref)
        ref_params = {k: v.data.copy() for k, v in param_map_ref.items()}

        parser = SLNCParser(str(path))
        model_direct = _make_model(n_embed, n_layer, activation="silu")
        loader = DirectWeightLoader(parser, sd, config)
        loader.load(model_direct)
        direct_params = {k: v.data.copy() for k, v in dict(model_direct._named_parameters()).items()}

        for key in ref_params:
            assert key in direct_params, f"Missing key in direct path: {key}"
            np.testing.assert_array_equal(
                ref_params[key], direct_params[key],
                err_msg=f"Mismatch for {key}",
            )

        parser.close()


class TestWeightLoaderRegistry:
    """Test format auto-detection and dispatch."""

    def test_register_and_get_loader(self):
        from domains.infrastructure.weight_loader import WeightLoaderRegistry

        class FakeLoader:
            pass

        reg = WeightLoaderRegistry()
        reg.register_loader(".slnc", FakeLoader)

        assert reg.get_loader("model.slnc") is FakeLoader
        assert reg.get_loader("path/to/model.slnc") is FakeLoader
        assert reg.get_loader("model.bin") is None

    def test_default_loader(self):
        from domains.infrastructure.weight_loader import WeightLoaderRegistry

        class Fallback:
            pass

        reg = WeightLoaderRegistry()
        reg.set_default(Fallback)

        assert reg.get_loader("model.xyz") is Fallback
        assert reg.get_loader("model.slnc") is Fallback

    def test_specific_overrides_default(self):
        from domains.infrastructure.weight_loader import WeightLoaderRegistry

        class Specific:
            pass
        class Fallback:
            pass

        reg = WeightLoaderRegistry()
        reg.set_default(Fallback)
        reg.register_loader(".slnc", Specific)

        assert reg.get_loader("model.slnc") is Specific
        assert reg.get_loader("model.xyz") is Fallback

    def test_load_file_no_loader(self):
        from domains.infrastructure.weight_loader import WeightLoaderRegistry

        reg = WeightLoaderRegistry()
        result = reg.load_file("model.xyz", None)

        assert result.success is False
        assert "No loader" in result.error

    def test_load_file_calls_loader(self, tmp_path):
        from domains.infrastructure.weight_loader import WeightLoaderRegistry, WeightLoadResult

        loaded = {}

        class TestLoader:
            def __init__(self, path, extra_kw=None):
                loaded["path"] = path
                loaded["extra"] = extra_kw
            def load(self, model):
                return WeightLoadResult(success=True, n_written=42)

        reg = WeightLoaderRegistry()
        reg.register_loader(".test", TestLoader)

        result = reg.load_file(str(tmp_path / "model.test"), "fake_model", extra_kw="hello")

        assert result.success is True
        assert result.n_written == 42
        assert loaded["path"] == str(tmp_path / "model.test")
        assert loaded["extra"] == "hello"


class TestSoulWeightLoader:
    """Test .soul checkpoint loading."""

    def test_soul_loader_loads_weights(self, tmp_path):
        """SoulWeightLoader loads state_dict into model parameters."""
        from domains.infrastructure.weight_loader import SoulWeightLoader
        from domains.training.slonet import SloTransformer
        from domains.inference.slo_format import save_soul

        n_embed, n_layer = 32, 2
        model = SloTransformer(
            vocab_size=100, n_embed=n_embed, n_layer=n_layer,
            n_head=4, block_size=64, dropout=0.0, _lazy=True,
        )
        # Save checkpoint
        soul_path = str(tmp_path / "test.soul")
        save_soul(model, soul_path)

        # Load into fresh model
        model2 = SloTransformer(
            vocab_size=100, n_embed=n_embed, n_layer=n_layer,
            n_head=4, block_size=64, dropout=0.0, _lazy=True,
        )
        loader = SoulWeightLoader(soul_path)
        result = loader.load(model2)

        assert result.success is True
        assert result.n_written > 0
        assert result.timing["total"] > 0

    def test_registry_auto_registers_soul(self):
        """Registry auto-registers .soul suffix on first access."""
        from domains.infrastructure.weight_loader import get_weight_loader_registry, SoulWeightLoader

        reg = get_weight_loader_registry()
        assert reg.get_loader("model.soul") is SoulWeightLoader

    def test_registry_auto_registers_slnc(self):
        """Registry auto-registers .slnc suffix on first access."""
        from domains.infrastructure.weight_loader import get_weight_loader_registry, DirectWeightLoader

        reg = get_weight_loader_registry()
        assert reg.get_loader("model.slnc") is DirectWeightLoader


class TestInferArch:
    """Test architecture inference from state dict."""

    def test_gpt2_arch(self):
        from domains.infrastructure.weight_loader import infer_arch_from_state_dict
        import numpy as np

        sd = {}
        sd["tok_emb.weight"] = np.zeros((1000, 768), dtype=np.float32)
        for i in range(12):
            sd[f"blocks.{i}.attn_norm.weight"] = np.ones(768, dtype=np.float32)
            sd[f"blocks.{i}.attn.q_proj.weight"] = np.zeros((768, 768), dtype=np.float32)
            sd[f"blocks.{i}.ff.mlp.w1.weight"] = np.zeros((3072, 768), dtype=np.float32)

        arch = infer_arch_from_state_dict(sd)
        assert arch["vocab_size"] == 1000
        assert arch["n_embed"] == 768
        assert arch["n_layer"] == 12
        assert arch["intermediate_size"] == 3072

    def test_llama_arch(self):
        from domains.infrastructure.weight_loader import infer_arch_from_state_dict
        import numpy as np

        sd = {}
        sd["tok_emb.weight"] = np.zeros((32000, 4096), dtype=np.float32)
        for i in range(32):
            sd[f"blocks.{i}.attn_norm.weight"] = np.ones(4096, dtype=np.float32)
            sd[f"blocks.{i}.attn.q_proj.weight"] = np.zeros((4096, 4096), dtype=np.float32)
            sd[f"blocks.{i}.ff.mlp.w1.weight"] = np.zeros((11008, 4096), dtype=np.float32)

        arch = infer_arch_from_state_dict(sd)
        assert arch["vocab_size"] == 32000
        assert arch["n_embed"] == 4096
        assert arch["n_layer"] == 32
        assert arch["intermediate_size"] == 11008

    def test_empty_state_dict(self):
        from domains.infrastructure.weight_loader import infer_arch_from_state_dict

        arch = infer_arch_from_state_dict({})
        assert arch["vocab_size"] == 256
        assert arch["n_embed"] == 128
        assert arch["n_layer"] == 1
        assert arch["n_head"] == 8

    def test_single_layer(self):
        from domains.infrastructure.weight_loader import infer_arch_from_state_dict
        import numpy as np

        sd = {}
        sd["tok_emb.weight"] = np.zeros((100, 64), dtype=np.float32)
        sd["blocks.0.attn_norm.weight"] = np.ones(64, dtype=np.float32)
        sd["blocks.0.attn.q_proj.weight"] = np.zeros((64, 64), dtype=np.float32)

        arch = infer_arch_from_state_dict(sd)
        assert arch["n_layer"] == 1
        assert arch["vocab_size"] == 100
