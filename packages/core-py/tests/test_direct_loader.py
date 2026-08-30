"""Tests for the weight loading infrastructure."""

import json
import struct
import threading

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
    infer_arch_from_state_dict, build_model_from_config,
    WeightLoaderRegistry, get_weight_loader_registry,
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


# ── TensorMapping ─────────────────────────────────────────────────────

class TestTensorMapping:
    def test_creation(self):
        tm = TensorMapping(param_name="weight", needs_transpose=True, canonical="layers.0.weight")
        assert tm.param_name == "weight"
        assert tm.needs_transpose is True
        assert tm.canonical == "layers.0.weight"

    def test_frozen(self):
        tm = TensorMapping(param_name="w", needs_transpose=False, canonical="c")
        with pytest.raises(AttributeError):
            tm.param_name = "new"

    def test_equality(self):
        a = TensorMapping("w", False, "c")
        b = TensorMapping("w", False, "c")
        assert a == b

    def test_repr(self):
        tm = TensorMapping("w", True, "c")
        assert "w" in repr(tm)


# ── WeightLoadResult ──────────────────────────────────────────────────

class TestWeightLoadResult:
    def test_defaults(self):
        r = WeightLoadResult(success=True)
        assert r.success is True
        assert r.n_written == 0
        assert r.n_fused == 0
        assert r.timing == {}
        assert r.error is None

    def test_custom(self):
        r = WeightLoadResult(success=False, n_written=5, n_fused=3, timing={"t": 1.0}, error="bad")
        assert r.success is False
        assert r.n_written == 5
        assert r.n_fused == 3
        assert r.timing == {"t": 1.0}
        assert r.error == "bad"

    def test_success_with_timing(self):
        r = WeightLoadResult(success=True, timing={"total": 0.5, "direct": 0.3})
        assert r.success is True
        assert r.timing["total"] == 0.5

    def test_result_mutable(self):
        r = WeightLoadResult(success=True)
        r.n_written = 10
        assert r.n_written == 10

    def test_error_none_by_default(self):
        r = WeightLoadResult(success=True)
        assert r.error is None


# ── LoadPlan ──────────────────────────────────────────────────────────

class TestLoadPlan:
    def test_creation(self):
        plan = LoadPlan(
            tensor_map={}, tied_weights=[], synthesized_params=[],
            fused_qkv={}, n_layer=2, n_embed=64, arch_name="test"
        )
        assert plan.n_layer == 2
        assert plan.n_embed == 64
        assert plan.arch_name == "test"
        assert plan.tensor_map == {}

    def test_with_data(self):
        tm = TensorMapping("w", False, "c")
        plan = LoadPlan(
            tensor_map={"file.w": tm},
            tied_weights=[("lm_head", "tok_emb")],
            synthesized_params=[("w3", "0", "w1")],
            fused_qkv={"qkv": ["q", "k", "v"]},
            n_layer=4, n_embed=256, arch_name="GPT2"
        )
        assert len(plan.tensor_map) == 1
        assert len(plan.tied_weights) == 1
        assert len(plan.synthesized_params) == 1

    def test_empty_plan(self):
        plan = LoadPlan(
            tensor_map={}, tied_weights=[], synthesized_params=[],
            fused_qkv={}, n_layer=0, n_embed=0, arch_name=""
        )
        assert plan.n_layer == 0
        assert plan.n_embed == 0


# ── SLNC Spec helpers ─────────────────────────────────────────────────

class TestSLNCSpec:
    def test_compute_header_size(self):
        jb = b'{"key": "value"}'
        size = compute_header_size(jb)
        assert size % ALIGNMENT == 0
        assert size >= len(jb)

    def test_compute_header_size_empty(self):
        size = compute_header_size(b"")
        assert size % ALIGNMENT == 0

    def test_compute_header_size_large_json(self):
        jb = b'{"key": "' + b"x" * 1000 + b'"}'
        size = compute_header_size(jb)
        assert size % ALIGNMENT == 0

    def test_compute_tensor_entry_size_1d(self):
        size = compute_tensor_entry_size(1, 10)
        assert size == 32 + 4 + 10

    def test_compute_tensor_entry_size_2d(self):
        size = compute_tensor_entry_size(2, 5)
        assert size == 32 + 8 + 5

    def test_compute_tensor_entry_size_3d(self):
        size = compute_tensor_entry_size(3, 20)
        assert size == 32 + 12 + 20

    def test_align(self):
        assert _align(0) == 0
        assert _align(1) == ALIGNMENT
        assert _align(ALIGNMENT) == ALIGNMENT
        assert _align(ALIGNMENT + 1) == 2 * ALIGNMENT

    def test_align_exact_multiple(self):
        assert _align(64) == 64
        assert _align(128) == 128

    def test_dtype_to_code_float32(self):
        assert dtype_to_code(np.float32) == DTYPE_FLOAT32

    def test_dtype_to_code_float16(self):
        from domains.infrastructure.slnc.spec import DTYPE_FLOAT16
        assert dtype_to_code(np.float16) == DTYPE_FLOAT16

    def test_dtype_to_code_int32(self):
        from domains.infrastructure.slnc.spec import DTYPE_INT32
        assert dtype_to_code(np.int32) == DTYPE_INT32

    def test_dtype_to_code_int64(self):
        from domains.infrastructure.slnc.spec import DTYPE_INT64
        assert dtype_to_code(np.int64) == DTYPE_INT64

    def test_dtype_to_code_uint8(self):
        from domains.infrastructure.slnc.spec import DTYPE_UINT8
        assert dtype_to_code(np.uint8) == DTYPE_UINT8

    def test_dtype_to_code_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            dtype_to_code(np.float64)

    def test_code_to_dtype_roundtrip(self):
        from domains.infrastructure.slnc.spec import code_to_dtype
        assert code_to_dtype(DTYPE_FLOAT32) == np.float32
        assert code_to_dtype(0) == np.float32


# ── build_load_plan ───────────────────────────────────────────────────

class TestBuildLoadPlan:
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
        assert len(plan.tied_weights) == 1
        assert len(plan.synthesized_params) > 0

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
        assert len(plan.synthesized_params) == 0

    def test_single_layer(self):
        sd = {"wte.weight": np.zeros((100, 64), dtype=np.float32)}
        config = {"architectures": ["GPT2LMHeadModel"], "vocab_size": 100, "n_embd": 64, "n_layer": 1, "n_head": 4}
        plan = build_load_plan(sd, 1, config)
        assert plan.n_layer == 1

    def test_empty_state_dict(self):
        config = {"architectures": ["unknown"], "vocab_size": 256, "n_embd": 128, "n_layer": 1, "n_head": 8}
        plan = build_load_plan({}, 1, config)
        assert len(plan.tensor_map) == 0

    def test_plan_has_fused_qkv(self):
        n_embed, n_layer = 32, 1
        sd = {
            "wte.weight": np.zeros((100, n_embed), dtype=np.float32),
            "h.0.attn.c_attn.weight": np.random.randn(n_embed, 3 * n_embed).astype(np.float32),
        }
        config = {"architectures": ["GPT2LMHeadModel"], "vocab_size": 100, "n_embd": n_embed, "n_layer": 1, "n_head": 4}
        plan = build_load_plan(sd, 1, config)
        assert len(plan.fused_qkv) > 0

    def test_plan_tensor_map_contains_mapping(self):
        n_embed, n_layer = 32, 1
        sd = {
            "wte.weight": np.zeros((100, n_embed), dtype=np.float32),
            "h.0.ln_1.weight": np.ones(n_embed, dtype=np.float32),
        }
        config = {"architectures": ["GPT2LMHeadModel"], "vocab_size": 100, "n_embd": n_embed, "n_layer": 1, "n_head": 4}
        plan = build_load_plan(sd, 1, config)
        assert "wte.weight" in plan.tensor_map
        assert plan.tensor_map["wte.weight"].param_name == "tok_emb.weight"


# ── DirectWeightLoader ────────────────────────────────────────────────

class TestDirectWeightLoader:
    def test_gpt2_direct_load_matches_fused(self, tmp_path):
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

    def test_from_plan(self, tmp_path):
        """Test _from_plan classmethod constructs loader correctly."""
        n_embed, n_layer = 32, 1
        sd = {"wte.weight": np.zeros((100, n_embed), dtype=np.float32)}
        config = {"architectures": ["GPT2LMHeadModel"], "vocab_size": 100, "n_embd": n_embed, "n_layer": 1, "n_head": 4}
        tensors = [{"name": k, "data": v} for k, v in sd.items()]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors, config, 1, n_embed, 4))
        parser = SLNCParser(str(path))
        plan = build_load_plan(sd, 1, config)
        loader = DirectWeightLoader._from_plan(parser, plan, sd)
        assert loader.plan is plan
        parser.close()

    def test_plan_property(self, tmp_path):
        n_embed, n_layer = 32, 1
        sd = {"wte.weight": np.zeros((100, n_embed), dtype=np.float32)}
        config = {"architectures": ["GPT2LMHeadModel"], "vocab_size": 100, "n_embd": n_embed, "n_layer": 1, "n_head": 4}
        tensors = [{"name": k, "data": v} for k, v in sd.items()]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors, config, 1, n_embed, 4))
        parser = SLNCParser(str(path))
        loader = DirectWeightLoader(parser, sd, config)
        assert isinstance(loader.plan, LoadPlan)
        parser.close()

    def test_load_result_timing(self, tmp_path):
        n_embed, n_layer = 32, 1
        sd = {"wte.weight": np.random.randn(100, n_embed).astype(np.float32)}
        config = {"architectures": ["GPT2LMHeadModel"], "vocab_size": 100, "n_embd": n_embed, "n_layer": 1, "n_head": 4}
        tensors = [{"name": k, "data": v} for k, v in sd.items()]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors, config, 1, n_embed, 4))
        parser = SLNCParser(str(path))
        model = _make_model(n_embed, n_layer, vocab_size=100)
        loader = DirectWeightLoader(parser, sd, config)
        result = loader.load(model)
        assert "direct" in result.timing
        assert "fused_qkv" in result.timing
        assert "tied_synth" in result.timing
        assert "total" in result.timing
        parser.close()


# ── load_into_model (generic loader) ──────────────────────────────────

class TestLoadIntoModel:
    def test_basic_load(self):
        """Test loading GPT-2 state dict into model via generic loader."""
        n_embed, n_layer = 32, 1
        model = _make_model(n_embed, n_layer, vocab_size=100)
        sd = {
            "wte.weight": np.random.randn(100, n_embed).astype(np.float32),
            "h.0.ln_1.weight": np.ones(n_embed, dtype=np.float32),
            "ln_f.weight": np.ones(n_embed, dtype=np.float32),
        }
        config = {"architectures": ["GPT2LMHeadModel"], "vocab_size": 100, "n_embd": n_embed, "n_layer": 1, "n_head": 4}
        plan = build_load_plan(sd, 1, config)
        result = load_into_model(model, plan, sd)
        assert result.success is True
        assert result.n_written > 0
        assert result.timing["total"] > 0

    def test_missing_tensor_data(self):
        n_embed, n_layer = 32, 1
        model = _make_model(n_embed, n_layer, vocab_size=100)
        sd = {"wte.weight": np.zeros((100, n_embed), dtype=np.float32)}
        config = {"architectures": ["GPT2LMHeadModel"], "vocab_size": 100, "n_embd": n_embed, "n_layer": 1, "n_head": 4}
        plan = build_load_plan(sd, 1, config)
        result = load_into_model(model, plan, {})
        assert result.success is True
        assert result.n_written == 0

    def test_tied_weights_copied(self):
        n_embed, n_layer = 32, 1
        model = _make_model(n_embed, n_layer, vocab_size=100)
        sd = {"wte.weight": np.ones((100, n_embed), dtype=np.float32)}
        config = {"architectures": ["GPT2LMHeadModel"], "vocab_size": 100, "n_embd": n_embed, "n_layer": 1, "n_head": 4}
        plan = build_load_plan(sd, 1, config)
        load_into_model(model, plan, sd)
        param_map = dict(model._named_parameters())
        if "lm_head.weight" in param_map and "tok_emb.weight" in param_map:
            np.testing.assert_array_equal(
                param_map["lm_head.weight"].data, param_map["tok_emb.weight"].data
            )

    def test_synthesized_params_zero_fill(self):
        n_embed, n_layer = 32, 1
        model = _make_model(n_embed, n_layer, vocab_size=100)
        sd = {"wte.weight": np.zeros((100, n_embed), dtype=np.float32)}
        config = {"architectures": ["GPT2LMHeadModel"], "vocab_size": 100, "n_embd": n_embed, "n_layer": 1, "n_head": 4}
        plan = build_load_plan(sd, 1, config)
        if plan.synthesized_params:
            load_into_model(model, plan, sd)
            param_map = dict(model._named_parameters())
            for pname, fill_val, _ in plan.synthesized_params:
                if pname in param_map:
                    expected = 0 if fill_val == "0" else 1
                    np.testing.assert_array_equal(
                        param_map[pname].data, np.full_like(param_map[pname].data, expected)
                    )

    def test_empty_plan(self):
        n_embed, n_layer = 32, 1
        model = _make_model(n_embed, n_layer, vocab_size=100)
        plan = LoadPlan(tensor_map={}, tied_weights=[], synthesized_params=[], fused_qkv={}, n_layer=1, n_embed=64, arch_name="test")
        result = load_into_model(model, plan, {})
        assert result.success is True
        assert result.n_written == 0
        assert result.n_fused == 0

    def test_result_timing_fields(self):
        n_embed, n_layer = 32, 1
        model = _make_model(n_embed, n_layer, vocab_size=100)
        sd = {"wte.weight": np.random.randn(100, n_embed).astype(np.float32)}
        config = {"architectures": ["GPT2LMHeadModel"], "vocab_size": 100, "n_embd": n_embed, "n_layer": 1, "n_head": 4}
        plan = build_load_plan(sd, 1, config)
        result = load_into_model(model, plan, sd)
        assert "direct" in result.timing
        assert "fused_qkv" in result.timing
        assert "tied_synth" in result.timing


# ── WeightLoaderRegistry ──────────────────────────────────────────────

class TestWeightLoaderRegistry:
    def test_register_and_get_loader(self):
        class FakeLoader:
            pass

        reg = WeightLoaderRegistry()
        reg.register_loader(".slnc", FakeLoader)

        assert reg.get_loader("model.slnc") is FakeLoader
        assert reg.get_loader("path/to/model.slnc") is FakeLoader
        assert reg.get_loader("model.bin") is None

    def test_default_loader(self):
        class Fallback:
            pass

        reg = WeightLoaderRegistry()
        reg.set_default(Fallback)

        assert reg.get_loader("model.xyz") is Fallback
        assert reg.get_loader("model.slnc") is Fallback

    def test_specific_overrides_default(self):
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
        reg = WeightLoaderRegistry()
        result = reg.load_file("model.xyz", None)

        assert result.success is False
        assert "No loader" in result.error

    def test_load_file_calls_loader(self, tmp_path):
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

    def test_load_file_exception(self):
        class BadLoader:
            def __init__(self, path, **kw):
                pass
            def load(self, model):
                raise RuntimeError("corrupt file")

        reg = WeightLoaderRegistry()
        reg.register_loader(".bad", BadLoader)
        result = reg.load_file("model.bad", None)
        assert result.success is False
        assert "corrupt file" in result.error

    def test_suffix_case_insensitive(self):
        class MyLoader:
            pass

        reg = WeightLoaderRegistry()
        reg.register_loader(".slnc", MyLoader)
        assert reg.get_loader("model.SLNC") is MyLoader

    def test_multiple_suffixes(self):
        class LoaderA:
            pass
        class LoaderB:
            pass

        reg = WeightLoaderRegistry()
        reg.register_loader(".a", LoaderA)
        reg.register_loader(".b", LoaderB)
        assert reg.get_loader("model.a") is LoaderA
        assert reg.get_loader("model.b") is LoaderB

    def test_get_loader_empty_path(self):
        reg = WeightLoaderRegistry()
        assert reg.get_loader("") is None

    def test_get_loader_no_suffix(self):
        reg = WeightLoaderRegistry()
        assert reg.get_loader("model") is None

    def test_register_overwrites(self):
        class LoaderV1:
            pass
        class LoaderV2:
            pass

        reg = WeightLoaderRegistry()
        reg.register_loader(".x", LoaderV1)
        reg.register_loader(".x", LoaderV2)
        assert reg.get_loader("model.x") is LoaderV2

    def test_load_file_timing(self, tmp_path):
        class SlowLoader:
            def __init__(self, path, **kw):
                pass
            def load(self, model):
                return WeightLoadResult(success=True, n_written=5)

        reg = WeightLoaderRegistry()
        reg.register_loader(".slow", SlowLoader)
        result = reg.load_file("model.slow", None)
        assert "total" in result.timing


# ── SoulWeightLoader ──────────────────────────────────────────────────

class TestSoulWeightLoader:
    def test_soul_loader_loads_weights(self, tmp_path):
        from domains.infrastructure.weight_loader import SoulWeightLoader
        from domains.inference.slo_format import save_soul

        n_embed, n_layer = 32, 2
        model = SloTransformer(
            vocab_size=100, n_embed=n_embed, n_layer=n_layer,
            n_head=4, block_size=64, dropout=0.0, _lazy=True,
        )
        soul_path = str(tmp_path / "test.soul")
        save_soul(model, soul_path)

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
        from domains.infrastructure.weight_loader import get_weight_loader_registry, SoulWeightLoader
        reg = get_weight_loader_registry()
        assert reg.get_loader("model.soul") is SoulWeightLoader

    def test_registry_auto_registers_slnc(self):
        from domains.infrastructure.weight_loader import get_weight_loader_registry, DirectWeightLoader
        reg = get_weight_loader_registry()
        assert reg.get_loader("model.slnc") is DirectWeightLoader

    def test_soul_load_metadata(self, tmp_path):
        from domains.infrastructure.weight_loader import SoulWeightLoader
        from domains.inference.slo_format import save_soul

        n_embed, n_layer = 32, 2
        model = SloTransformer(
            vocab_size=100, n_embed=n_embed, n_layer=n_layer,
            n_head=4, block_size=64, dropout=0.0, _lazy=True,
        )
        soul_path = str(tmp_path / "test.soul")
        save_soul(model, soul_path)
        loader = SoulWeightLoader(soul_path)
        meta = loader.load_metadata()
        assert "vocab_size" in meta
        assert "n_embed" in meta
        assert "soul" in meta
        assert meta["vocab_size"] == 100

    def test_soul_load_timing_keys(self, tmp_path):
        from domains.infrastructure.weight_loader import SoulWeightLoader
        from domains.inference.slo_format import save_soul

        n_embed, n_layer = 32, 1
        model = SloTransformer(
            vocab_size=50, n_embed=n_embed, n_layer=n_layer,
            n_head=4, block_size=32, dropout=0.0, _lazy=True,
        )
        soul_path = str(tmp_path / "timing.soul")
        save_soul(model, soul_path)
        loader = SoulWeightLoader(soul_path)
        model2 = SloTransformer(vocab_size=50, n_embed=n_embed, n_layer=n_layer, n_head=4, block_size=32, dropout=0.0, _lazy=True)
        result = loader.load(model2)
        assert "load_soul" in result.timing
        assert "apply" in result.timing


# ── infer_arch_from_state_dict ────────────────────────────────────────

class TestInferArch:
    def test_gpt2_arch(self):
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
        arch = infer_arch_from_state_dict({})
        assert arch["vocab_size"] == 256
        assert arch["n_embed"] == 128
        assert arch["n_layer"] == 1
        assert arch["n_head"] == 8

    def test_single_layer(self):
        sd = {}
        sd["tok_emb.weight"] = np.zeros((100, 64), dtype=np.float32)
        sd["blocks.0.attn_norm.weight"] = np.ones(64, dtype=np.float32)
        sd["blocks.0.attn.q_proj.weight"] = np.zeros((64, 64), dtype=np.float32)

        arch = infer_arch_from_state_dict(sd)
        assert arch["n_layer"] == 1
        assert arch["vocab_size"] == 100

    def test_q_proj_alternate_key(self):
        sd = {
            "tok_emb.weight": np.zeros((50, 128), dtype=np.float32),
            "blocks.0.attn_norm.weight": np.ones(128, dtype=np.float32),
            "blocks.0.q_proj.weight": np.zeros((256, 128), dtype=np.float32),
        }
        arch = infer_arch_from_state_dict(sd)
        assert arch["n_head"] >= 1

    def test_gate_proj_key(self):
        sd = {
            "tok_emb.weight": np.zeros((50, 128), dtype=np.float32),
            "blocks.0.attn_norm.weight": np.ones(128, dtype=np.float32),
            "blocks.0.ff.mlp.gate_proj.weight": np.zeros((256, 128), dtype=np.float32),
        }
        arch = infer_arch_from_state_dict(sd)
        assert arch["intermediate_size"] == 256

    def test_corrupt_block_index(self):
        sd = {
            "tok_emb.weight": np.zeros((50, 128), dtype=np.float32),
            "blocks.abc.attn_norm.weight": np.ones(128, dtype=np.float32),
        }
        arch = infer_arch_from_state_dict(sd)
        assert arch["n_layer"] == 1

    def test_multi_layer_count(self):
        sd = {}
        sd["tok_emb.weight"] = np.zeros((50, 64), dtype=np.float32)
        for i in range(5):
            sd[f"blocks.{i}.attn_norm.weight"] = np.ones(64, dtype=np.float32)
        arch = infer_arch_from_state_dict(sd)
        assert arch["n_layer"] == 5

    def test_no_tok_emb(self):
        sd = {"blocks.0.attn_norm.weight": np.ones(64, dtype=np.float32)}
        arch = infer_arch_from_state_dict(sd)
        assert arch["vocab_size"] == 256

    def test_tok_emb_1d_ignored(self):
        sd = {"tok_emb.weight": np.ones(128, dtype=np.float32)}
        arch = infer_arch_from_state_dict(sd)
        assert arch["vocab_size"] == 256


# ── build_model_from_config ──────────────────────────────────────────

class TestBuildModelFromConfig:
    def test_gpt2_config(self):
        config = {
            "architectures": ["GPT2LMHeadModel"],
            "vocab_size": 1000, "n_embd": 128, "n_layer": 2, "n_head": 4,
        }
        model = build_model_from_config(config)
        assert model.vocab_size == 1000
        assert model.n_embed == 128
        assert model.n_layer == 2

    def test_llama_config(self):
        config = {
            "architectures": ["LlamaForCausalLM"],
            "vocab_size": 32000, "hidden_size": 256, "num_hidden_layers": 4,
            "num_attention_heads": 8, "hidden_act": "silu", "rms_norm_eps": 1e-5,
        }
        model = build_model_from_config(config)
        assert model.vocab_size == 32000
        assert model.n_embed == 256
        assert model.n_layer == 4

    def test_rope_config(self):
        config = {
            "vocab_size": 100, "n_embd": 64, "n_layer": 1, "n_head": 4,
            "rope_theta": 10000.0,
        }
        model = build_model_from_config(config)
        # Model should be constructable; verify it has expected properties
        assert model.vocab_size == 100
        assert model.n_embed == 64

    def test_explicit_norm_type(self):
        config = {
            "vocab_size": 100, "n_embd": 64, "n_layer": 1, "n_head": 4,
            "layer_norm_type": "rms_norm",
        }
        model = build_model_from_config(config)
        # Verify model constructed successfully
        assert model.vocab_size == 100

    def test_lazy_true(self):
        config = {"vocab_size": 100, "n_embd": 64, "n_layer": 1, "n_head": 4}
        model = build_model_from_config(config, _lazy=True)
        assert model is not None

    def test_defaults(self):
        config = {}
        model = build_model_from_config(config)
        assert model.vocab_size == 50257
        assert model.n_embed == 768

    def test_max_pos_embeddings(self):
        config = {"vocab_size": 100, "n_embd": 64, "n_layer": 1, "n_head": 4, "max_position_embeddings": 256}
        model = build_model_from_config(config)
        assert model.block_size == 256

    def test_intermediate_size_key(self):
        config = {"vocab_size": 100, "n_embd": 64, "n_layer": 1, "n_head": 4, "intermediate_size": 128}
        model = build_model_from_config(config)
        assert model is not None


# ── Thread safety ─────────────────────────────────────────────────────

class TestRegistryThreadSafety:
    def test_concurrent_get_loader(self):
        from domains.infrastructure.weight_loader import WeightLoaderRegistry
        reg = WeightLoaderRegistry()
        results = []

        def worker():
            results.append(reg.get_loader("test.slnc"))

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(results) == 10

    def test_global_registry_singleton(self):
        a = get_weight_loader_registry()
        b = get_weight_loader_registry()
        assert a is b
