"""Tests for domains.infrastructure.slnc.compiler — helper functions and SLNCCompiler."""

import json
import struct
import numpy as np
from pathlib import Path
import tempfile
from domains.infrastructure.slnc.compiler import (
    _crc32, _xxhash64, SLNCCompiler, GPT2_BLOCK_LAYOUT, GPT2_NON_BLOCK_LAYOUT,
    LLAMA_BLOCK_LAYOUT, LLAMA_NON_BLOCK_LAYOUT, _ARCH_LAYOUTS,
)
from domains.infrastructure.slnc.spec import (
    MAGIC, VERSION, FLAGS_DEFAULT, ALIGNMENT, DTYPE_FLOAT32,
    compute_header_size, compute_tensor_entry_size, compute_tensor_table_size,
    dtype_to_code, _align, DTYPE_MAP,
)


class TestCrc32:
    def test_deterministic(self):
        assert _crc32(b"hello") == _crc32(b"hello")

    def test_different_inputs(self):
        assert _crc32(b"hello") != _crc32(b"world")

    def test_returns_int(self):
        result = _crc32(b"test")
        assert isinstance(result, int)
        assert 0 <= result <= 0xFFFFFFFF

    def test_empty_bytes(self):
        result = _crc32(b"")
        assert isinstance(result, int)

    def test_long_bytes(self):
        data = b"x" * 10000
        result = _crc32(data)
        assert isinstance(result, int)
        assert 0 <= result <= 0xFFFFFFFF

    def test_same_data_same_hash(self):
        data = np.random.bytes(1024)
        assert _crc32(data) == _crc32(data)

    def test_one_bit_difference(self):
        a = b"\x00\x00\x00\x00"
        b = b"\x00\x00\x00\x01"
        assert _crc32(a) != _crc32(b)

    def test_all_zeros(self):
        result = _crc32(b"\x00" * 100)
        assert isinstance(result, int)

    def test_all_ones(self):
        result = _crc32(b"\xff" * 100)
        assert isinstance(result, int)

    def test_single_byte(self):
        results = [_crc32(bytes([i])) for i in range(256)]
        assert len(set(results)) == 256


class TestXxhash64:
    def test_deterministic(self):
        assert _xxhash64(b"hello") == _xxhash64(b"hello")

    def test_different_inputs(self):
        assert _xxhash64(b"hello") != _xxhash64(b"world")

    def test_returns_int(self):
        result = _xxhash64(b"test")
        assert isinstance(result, int)

    def test_empty_bytes(self):
        result = _xxhash64(b"")
        assert isinstance(result, int)

    def test_consistent_across_calls(self):
        data = b"consistency check"
        h1 = _xxhash64(data)
        h2 = _xxhash64(data)
        assert h1 == h2

    def test_large_data(self):
        data = b"x" * 100000
        result = _xxhash64(data)
        assert isinstance(result, int)

    def test_different_from_crc32(self):
        data = b"test data"
        assert _xxhash64(data) != _crc32(data)


class TestSLNCCompiler:
    def test_init(self):
        comp = SLNCCompiler()
        assert comp is not None

    def test_compute_block_size_gpt2(self):
        comp = SLNCCompiler()
        config = {"n_embd": 768, "n_inner": 3072}
        size = comp._compute_block_size(config)
        assert size > 0

    def test_compute_block_size_llama(self):
        comp = SLNCCompiler()
        config = {"hidden_size": 768, "intermediate_size": 2048, "rope_theta": 10000.0}
        size = comp._compute_block_size(config)
        assert size > 0

    def test_order_tensors(self):
        comp = SLNCCompiler()
        config = {"n_layer": 2}
        weights = {
            "h.0.attn.c_attn.weight": np.zeros((3, 3)),
            "h.1.attn.c_attn.weight": np.zeros((3, 3)),
            "wte.weight": np.zeros((10, 3)),
        }
        ordered = comp._order_tensors(config, weights)
        assert isinstance(ordered, list)
        assert len(ordered) == 3

    def test_order_tensors_llama(self):
        comp = SLNCCompiler()
        config = {"num_hidden_layers": 1}
        weights = {
            "model.embed_tokens.weight": np.zeros((10, 16)),
            "model.layers.0.self_attn.q_proj.weight": np.zeros((16, 16)),
            "model.layers.0.self_attn.k_proj.weight": np.zeros((16, 16)),
            "model.layers.0.self_attn.v_proj.weight": np.zeros((16, 16)),
            "model.layers.0.self_attn.o_proj.weight": np.zeros((16, 16)),
            "model.layers.0.mlp.gate_proj.weight": np.zeros((16, 32)),
            "model.layers.0.mlp.up_proj.weight": np.zeros((16, 32)),
            "model.layers.0.mlp.down_proj.weight": np.zeros((32, 16)),
            "model.norm.weight": np.zeros((16,)),
            "model.lm_head.weight": np.zeros((10, 16)),
        }
        ordered = comp._order_tensors(config, weights)
        assert len(ordered) > 0

    def test_order_tensors_gpt2_explicit(self):
        comp = SLNCCompiler()
        config = {"n_layer": 1}
        weights = {
            "h.0.ln_1.weight": np.zeros(16),
            "h.0.ln_1.bias": np.zeros(16),
            "h.0.attn.c_attn.weight": np.zeros((16, 48)),
            "h.0.attn.c_attn.bias": np.zeros(48),
            "h.0.attn.c_proj.weight": np.zeros((16, 16)),
            "h.0.attn.c_proj.bias": np.zeros(16),
            "h.0.ln_2.weight": np.zeros(16),
            "h.0.ln_2.bias": np.zeros(16),
            "h.0.mlp.c_fc.weight": np.zeros((16, 64)),
            "h.0.mlp.c_fc.bias": np.zeros(64),
            "h.0.mlp.c_proj.weight": np.zeros((64, 16)),
            "h.0.mlp.c_proj.bias": np.zeros(16),
            "ln_f.weight": np.zeros(16),
            "ln_f.bias": np.zeros(16),
            "wte.weight": np.zeros((100, 16)),
            "wpe.weight": np.zeros((512, 16)),
        }
        ordered = comp._order_tensors(config, weights)
        assert len(ordered) == len(weights)

    def test_compile_from_dict(self):
        comp = SLNCCompiler()
        config = {"n_layer": 1, "n_embd": 16, "n_head": 2, "n_inner": 32,
                  "vocab_size": 100, "n_positions": 128}
        weights = {
            "h.0.ln_1.weight": np.ones(16, dtype=np.float32),
            "h.0.ln_1.bias": np.zeros(16, dtype=np.float32),
            "h.0.attn.c_attn.weight": np.random.randn(16, 48).astype(np.float32),
            "h.0.attn.c_attn.bias": np.zeros(48, dtype=np.float32),
            "h.0.attn.c_proj.weight": np.random.randn(16, 16).astype(np.float32),
            "h.0.attn.c_proj.bias": np.zeros(16, dtype=np.float32),
            "h.0.ln_2.weight": np.ones(16, dtype=np.float32),
            "h.0.ln_2.bias": np.zeros(16, dtype=np.float32),
            "h.0.mlp.c_fc.weight": np.random.randn(16, 32).astype(np.float32),
            "h.0.mlp.c_fc.bias": np.zeros(32, dtype=np.float32),
            "h.0.mlp.c_proj.weight": np.random.randn(32, 16).astype(np.float32),
            "h.0.mlp.c_proj.bias": np.zeros(16, dtype=np.float32),
            "ln_f.weight": np.ones(16, dtype=np.float32),
            "ln_f.bias": np.zeros(16, dtype=np.float32),
            "wte.weight": np.random.randn(100, 16).astype(np.float32),
            "wpe.weight": np.random.randn(128, 16).astype(np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix=".slnc", delete=False) as f:
            output = f.name

        result = comp.compile_from_dict(config, weights, output)
        assert result == output
        assert Path(output).exists()

        with open(output, "rb") as f:
            magic = f.read(4)
        assert magic == MAGIC

    def test_compile_from_dict_tensor_count(self):
        comp = SLNCCompiler()
        config = {"n_layer": 1, "n_embd": 8, "n_head": 2, "n_inner": 16,
                  "vocab_size": 10, "n_positions": 32}
        weights = {
            "h.0.ln_1.weight": np.ones(8, dtype=np.float32),
            "h.0.ln_1.bias": np.zeros(8, dtype=np.float32),
            "h.0.attn.c_attn.weight": np.random.randn(8, 24).astype(np.float32),
            "h.0.attn.c_attn.bias": np.zeros(24, dtype=np.float32),
            "h.0.attn.c_proj.weight": np.random.randn(8, 8).astype(np.float32),
            "h.0.attn.c_proj.bias": np.zeros(8, dtype=np.float32),
            "h.0.ln_2.weight": np.ones(8, dtype=np.float32),
            "h.0.ln_2.bias": np.zeros(8, dtype=np.float32),
            "h.0.mlp.c_fc.weight": np.random.randn(8, 16).astype(np.float32),
            "h.0.mlp.c_fc.bias": np.zeros(16, dtype=np.float32),
            "h.0.mlp.c_proj.weight": np.random.randn(16, 8).astype(np.float32),
            "h.0.mlp.c_proj.bias": np.zeros(8, dtype=np.float32),
            "ln_f.weight": np.ones(8, dtype=np.float32),
            "ln_f.bias": np.zeros(8, dtype=np.float32),
            "wte.weight": np.random.randn(10, 8).astype(np.float32),
            "wpe.weight": np.random.randn(32, 8).astype(np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix=".slnc", delete=False) as f:
            output = f.name

        comp.compile_from_dict(config, weights, output)

        with open(output, "rb") as f:
            f.seek(12)
            n_layer = struct.unpack("<I", f.read(4))[0]
            n_embd = struct.unpack("<I", f.read(4))[0]
            n_head = struct.unpack("<I", f.read(4))[0]
            n_inner = struct.unpack("<I", f.read(4))[0]
            vocab_size = struct.unpack("<I", f.read(4))[0]

        assert n_layer == 1
        assert n_embd == 8
        assert n_head == 2
        assert n_inner == 16
        assert vocab_size == 10

    def test_multiple_layers(self):
        comp = SLNCCompiler()
        config = {"n_layer": 3, "n_embd": 8, "n_head": 2, "n_inner": 16,
                  "vocab_size": 10, "n_positions": 32}
        weights = {}
        for i in range(3):
            weights[f"h.{i}.ln_1.weight"] = np.ones(8, dtype=np.float32)
            weights[f"h.{i}.ln_1.bias"] = np.zeros(8, dtype=np.float32)
            weights[f"h.{i}.attn.c_attn.weight"] = np.random.randn(8, 24).astype(np.float32)
            weights[f"h.{i}.attn.c_attn.bias"] = np.zeros(24, dtype=np.float32)
            weights[f"h.{i}.attn.c_proj.weight"] = np.random.randn(8, 8).astype(np.float32)
            weights[f"h.{i}.attn.c_proj.bias"] = np.zeros(8, dtype=np.float32)
            weights[f"h.{i}.ln_2.weight"] = np.ones(8, dtype=np.float32)
            weights[f"h.{i}.ln_2.bias"] = np.zeros(8, dtype=np.float32)
            weights[f"h.{i}.mlp.c_fc.weight"] = np.random.randn(8, 16).astype(np.float32)
            weights[f"h.{i}.mlp.c_fc.bias"] = np.zeros(16, dtype=np.float32)
            weights[f"h.{i}.mlp.c_proj.weight"] = np.random.randn(16, 8).astype(np.float32)
            weights[f"h.{i}.mlp.c_proj.bias"] = np.zeros(8, dtype=np.float32)
        weights["ln_f.weight"] = np.ones(8, dtype=np.float32)
        weights["ln_f.bias"] = np.zeros(8, dtype=np.float32)
        weights["wte.weight"] = np.random.randn(10, 8).astype(np.float32)
        weights["wpe.weight"] = np.random.randn(32, 8).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".slnc", delete=False) as f:
            output = f.name

        result = comp.compile_from_dict(config, weights, output)
        assert Path(output).exists()
        size = Path(output).stat().st_size
        assert size > 0

    def test_compile_from_dict_file_size(self):
        comp = SLNCCompiler()
        config = {"n_layer": 1, "n_embd": 8, "n_head": 2, "n_inner": 16,
                  "vocab_size": 10, "n_positions": 32}
        weights = {
            "h.0.ln_1.weight": np.ones(8, dtype=np.float32),
            "h.0.ln_1.bias": np.zeros(8, dtype=np.float32),
            "h.0.attn.c_attn.weight": np.random.randn(8, 24).astype(np.float32),
            "h.0.attn.c_attn.bias": np.zeros(24, dtype=np.float32),
            "h.0.attn.c_proj.weight": np.random.randn(8, 8).astype(np.float32),
            "h.0.attn.c_proj.bias": np.zeros(8, dtype=np.float32),
            "h.0.ln_2.weight": np.ones(8, dtype=np.float32),
            "h.0.ln_2.bias": np.zeros(8, dtype=np.float32),
            "h.0.mlp.c_fc.weight": np.random.randn(8, 16).astype(np.float32),
            "h.0.mlp.c_fc.bias": np.zeros(16, dtype=np.float32),
            "h.0.mlp.c_proj.weight": np.random.randn(16, 8).astype(np.float32),
            "h.0.mlp.c_proj.bias": np.zeros(8, dtype=np.float32),
            "ln_f.weight": np.ones(8, dtype=np.float32),
            "ln_f.bias": np.zeros(8, dtype=np.float32),
            "wte.weight": np.random.randn(10, 8).astype(np.float32),
            "wpe.weight": np.random.randn(32, 8).astype(np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix=".slnc", delete=False) as f:
            output = f.name

        comp.compile_from_dict(config, weights, output)
        file_size = Path(output).stat().st_size
        assert file_size > 100

    def test_compile_llama_from_dict(self):
        comp = SLNCCompiler()
        config = {"num_hidden_layers": 1, "hidden_size": 16, "num_attention_heads": 2,
                  "intermediate_size": 32, "vocab_size": 100, "max_position_embeddings": 128,
                  "rope_theta": 10000.0}
        weights = {
            "model.embed_tokens.weight": np.random.randn(100, 16).astype(np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.random.randn(16, 16).astype(np.float32),
            "model.layers.0.self_attn.k_proj.weight": np.random.randn(16, 16).astype(np.float32),
            "model.layers.0.self_attn.v_proj.weight": np.random.randn(16, 16).astype(np.float32),
            "model.layers.0.self_attn.o_proj.weight": np.random.randn(16, 16).astype(np.float32),
            "model.layers.0.input_layernorm.weight": np.ones(16, dtype=np.float32),
            "model.layers.0.post_attention_layernorm.weight": np.ones(16, dtype=np.float32),
            "model.layers.0.mlp.gate_proj.weight": np.random.randn(16, 32).astype(np.float32),
            "model.layers.0.mlp.up_proj.weight": np.random.randn(16, 32).astype(np.float32),
            "model.layers.0.mlp.down_proj.weight": np.random.randn(32, 16).astype(np.float32),
            "model.norm.weight": np.ones(16, dtype=np.float32),
            "model.lm_head.weight": np.random.randn(100, 16).astype(np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix=".slnc", delete=False) as f:
            output = f.name

        result = comp.compile_from_dict(config, weights, output)
        assert Path(output).exists()

    def test_compile_preserves_weight_data(self):
        comp = SLNCCompiler()
        config = {"n_layer": 1, "n_embd": 8, "n_head": 2, "n_inner": 16,
                  "vocab_size": 10, "n_positions": 32}
        weight_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        weights = {
            "h.0.ln_1.weight": weight_val.copy(),
            "h.0.ln_1.bias": np.zeros(8, dtype=np.float32),
            "h.0.attn.c_attn.weight": np.random.randn(8, 24).astype(np.float32),
            "h.0.attn.c_attn.bias": np.zeros(24, dtype=np.float32),
            "h.0.attn.c_proj.weight": np.random.randn(8, 8).astype(np.float32),
            "h.0.attn.c_proj.bias": np.zeros(8, dtype=np.float32),
            "h.0.ln_2.weight": np.ones(8, dtype=np.float32),
            "h.0.ln_2.bias": np.zeros(8, dtype=np.float32),
            "h.0.mlp.c_fc.weight": np.random.randn(8, 16).astype(np.float32),
            "h.0.mlp.c_fc.bias": np.zeros(16, dtype=np.float32),
            "h.0.mlp.c_proj.weight": np.random.randn(16, 8).astype(np.float32),
            "h.0.mlp.c_proj.bias": np.zeros(8, dtype=np.float32),
            "ln_f.weight": np.ones(8, dtype=np.float32),
            "ln_f.bias": np.zeros(8, dtype=np.float32),
            "wte.weight": np.random.randn(10, 8).astype(np.float32),
            "wpe.weight": np.random.randn(32, 8).astype(np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix=".slnc", delete=False) as f:
            output = f.name

        comp.compile_from_dict(config, weights, output)
        assert Path(output).exists()


class TestSpecHelpers:
    def test_compute_header_size(self):
        config_json = b'{"n_layer": 1}'
        size = compute_header_size(config_json)
        assert size >= ALIGNMENT
        assert size % ALIGNMENT == 0

    def test_compute_tensor_entry_size(self):
        size = compute_tensor_entry_size(ndim=2, name_len=10)
        assert size > 0

    def test_compute_tensor_table_size(self):
        entries = [("weight", 0, None, 2, None, 0), ("bias", 0, None, 1, None, 0)]
        size = compute_tensor_table_size(entries)
        assert size > 0

    def test_align(self):
        assert _align(0) == 0
        assert _align(1) == ALIGNMENT
        assert _align(ALIGNMENT) == ALIGNMENT
        assert _align(ALIGNMENT + 1) == 2 * ALIGNMENT

    def test_dtype_to_code(self):
        assert dtype_to_code(np.float32) == DTYPE_FLOAT32
        assert dtype_to_code(np.float16) == 1
        assert dtype_to_code(np.int32) == 3
        assert dtype_to_code(np.int64) == 4
        assert dtype_to_code(np.uint8) == 5

    def test_dtype_to_code_unsupported(self):
        import pytest
        with pytest.raises(ValueError):
            dtype_to_code(np.float64)

    def test_code_to_dtype(self):
        from domains.infrastructure.slnc.spec import code_to_dtype
        assert code_to_dtype(0) == np.float32
        assert code_to_dtype(1) == np.float16
        assert code_to_dtype(3) == np.int32
        assert code_to_dtype(4) == np.int64
        assert code_to_dtype(5) == np.uint8

    def test_code_to_dtype_invalid(self):
        import pytest
        from domains.infrastructure.slnc.spec import code_to_dtype
        with pytest.raises(ValueError):
            code_to_dtype(999)

    def test_magic_bytes(self):
        assert MAGIC == b"SLNC"
        assert len(MAGIC) == 4

    def test_version(self):
        assert VERSION == 1

    def test_alignment(self):
        assert ALIGNMENT == 64

    def test_dtype_map_coverage(self):
        for code, name in DTYPE_MAP.items():
            if name == "bfloat16":
                continue
            assert hasattr(np, name)

    def test_compute_header_size_empty(self):
        size = compute_header_size(b"")
        assert size >= ALIGNMENT

    def test_compute_header_size_large(self):
        large_json = b'{"key": "' + b"x" * 10000 + b'"}'
        size = compute_header_size(large_json)
        assert size >= ALIGNMENT
        assert size % ALIGNMENT == 0

    def test_compute_tensor_entry_size_1d(self):
        size = compute_tensor_entry_size(ndim=1, name_len=5)
        assert size > 0

    def test_compute_tensor_entry_size_3d(self):
        size = compute_tensor_entry_size(ndim=3, name_len=20)
        assert size > 0

    def test_align_already_aligned(self):
        assert _align(64) == 64
        assert _align(128) == 128

    def test_align_need_padding(self):
        assert _align(1) == 64
        assert _align(63) == 64
        assert _align(65) == 128


class TestGPT2BlockLayout:
    def test_layout_not_empty(self):
        assert len(GPT2_BLOCK_LAYOUT) > 0

    def test_non_block_layout(self):
        assert len(GPT2_NON_BLOCK_LAYOUT) > 0
        assert "ln_f.weight" in GPT2_NON_BLOCK_LAYOUT
        assert "wte.weight" in GPT2_NON_BLOCK_LAYOUT

    def test_has_attention(self):
        assert any("attn" in t for t in GPT2_BLOCK_LAYOUT)

    def test_has_mlp(self):
        assert any("mlp" in t for t in GPT2_BLOCK_LAYOUT)

    def test_has_norms(self):
        assert any("ln_" in t for t in GPT2_BLOCK_LAYOUT)


class TestLLAMABlockLayout:
    def test_layout_not_empty(self):
        assert len(LLAMA_BLOCK_LAYOUT) > 0

    def test_non_block_layout(self):
        assert len(LLAMA_NON_BLOCK_LAYOUT) > 0
        assert "model.norm.weight" in LLAMA_NON_BLOCK_LAYOUT
        assert "model.embed_tokens.weight" in LLAMA_NON_BLOCK_LAYOUT

    def test_has_attention(self):
        assert any("attn" in t or "proj" in t for t in LLAMA_BLOCK_LAYOUT)

    def test_has_mlp(self):
        assert any("mlp" in t for t in LLAMA_BLOCK_LAYOUT)

    def test_has_norms(self):
        assert any("norm" in t for t in LLAMA_BLOCK_LAYOUT)


class TestArchLayouts:
    def test_gpt2_layout(self):
        assert "gpt2" in _ARCH_LAYOUTS
        block, non_block, prefix = _ARCH_LAYOUTS["gpt2"]
        assert prefix == "h.{i}."

    def test_llama_layout(self):
        assert "llama" in _ARCH_LAYOUTS
        block, non_block, prefix = _ARCH_LAYOUTS["llama"]
        assert prefix == "model.layers.{i}."

    def test_gpt2_block_has_bias(self):
        block, _, _ = _ARCH_LAYOUTS["gpt2"]
        assert any(".bias" in t for t in block)

    def test_llama_block_no_bias(self):
        block, _, _ = _ARCH_LAYOUTS["llama"]
        assert not any(".bias" in t for t in block)
