"""Tests for slnc.compiler — .slnc binary compiler and round-trip via SLNCParser."""

import json
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from domains.infrastructure.slnc import spec
from domains.infrastructure.slnc.compiler import (
    GPT2_BLOCK_LAYOUT,
    LLAMA_BLOCK_LAYOUT,
    SLNCCompiler,
    _crc32,
    _xxhash64,
)
from domains.infrastructure.slnc.parser import SLNCParser

CFG = {
    "n_layer": 1,
    "n_embd": 4,
    "n_head": 1,
    "n_inner": 8,
    "vocab_size": 8,
    "n_positions": 6,
    "model_type": "gpt2",
}


def _gpt2_weights():
    """Synthetic GPT-2 weights matching CFG, safetensors naming (h.{i}. prefix)."""
    n_embd, n_inner, n_layer = CFG["n_embd"], CFG["n_inner"], CFG["n_layer"]
    weights = {}
    for l in range(n_layer):
        weights.update({
            f"h.{l}.ln_1.weight": np.ones(n_embd, dtype=np.float32),
            f"h.{l}.ln_1.bias": np.zeros(n_embd, dtype=np.float32),
            f"h.{l}.attn.c_attn.weight": np.arange(3 * n_embd * n_embd, dtype=np.float32).reshape(n_embd, 3 * n_embd),
            f"h.{l}.attn.c_attn.bias": np.arange(3 * n_embd, dtype=np.float32),
            f"h.{l}.attn.c_proj.weight": np.ones((n_embd, n_embd), dtype=np.float32),
            f"h.{l}.attn.c_proj.bias": np.zeros(n_embd, dtype=np.float32),
            f"h.{l}.ln_2.weight": np.ones(n_embd, dtype=np.float32),
            f"h.{l}.ln_2.bias": np.zeros(n_embd, dtype=np.float32),
            f"h.{l}.mlp.c_fc.weight": np.arange(n_embd * n_inner, dtype=np.float32).reshape(n_embd, n_inner),
            f"h.{l}.mlp.c_fc.bias": np.arange(n_inner, dtype=np.float32),
            f"h.{l}.mlp.c_proj.weight": np.arange(n_inner * n_embd, dtype=np.float32).reshape(n_inner, n_embd),
            f"h.{l}.mlp.c_proj.bias": np.zeros(n_embd, dtype=np.float32),
        })
    weights["ln_f.weight"] = np.ones(n_embd, dtype=np.float32)
    weights["ln_f.bias"] = np.zeros(n_embd, dtype=np.float32)
    weights["wte.weight"] = np.arange(CFG["vocab_size"] * n_embd, dtype=np.float32).reshape(CFG["vocab_size"], n_embd)
    weights["wpe.weight"] = np.arange(CFG["n_positions"] * n_embd, dtype=np.float32).reshape(CFG["n_positions"], n_embd)
    return weights


def _compile(compiler, output, config=CFG, weights=None):
    if weights is None:
        weights = _gpt2_weights()
    return compiler.compile_from_dict(config, weights, str(output))


def _write_safetensors(path, weights, dtype="F32"):
    """Write a minimal .safetensors file with per-tensor dtype strings."""
    header = {"__metadata__": {}}
    data = b""
    for name, arr in weights.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(arr.shape),
            "data_offsets": [len(data), len(data) + arr.nbytes],
        }
        data += arr.tobytes()
    header_json = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(header_json)) + header_json + data)


class TestCompileFromDict:
    def test_returns_output_path(self, tmp_path):
        out = tmp_path / "m.slnc"
        assert _compile(SLNCCompiler(), output=out) == str(out)

    def test_writes_valid_magic_and_version(self, tmp_path):
        out = tmp_path / "m.slnc"
        _compile(SLNCCompiler(), output=out)
        with open(out, "rb") as f:
            assert f.read(4) == b"SLNC"
            assert struct.unpack("<I", f.read(4))[0] == 1

    def test_header_metadata_fields(self, tmp_path):
        out = tmp_path / "m.slnc"
        compiler = SLNCCompiler()
        _compile(compiler, output=out)
        with open(out, "rb") as f:
            f.seek(12)
            fields = struct.unpack("<10I", f.read(40))
        n_layer, n_embd, n_head, n_inner, vocab, n_pos, block_count, block_size, tensor_count, data_offset = fields
        assert n_layer == 1
        assert n_embd == 4
        assert n_head == 1
        assert n_inner == 8
        assert vocab == 8
        assert n_pos == 6
        assert block_count == 1
        assert tensor_count == 16
        assert data_offset % 64 == 0
        assert block_size == compiler._compute_block_size(CFG)

    def test_llama_fallback_config_keys(self, tmp_path):
        llama_cfg = {
            "num_hidden_layers": 1,
            "hidden_size": 4,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "vocab_size": 16,
            "max_position_embeddings": 8,
        }
        out = tmp_path / "llama.slnc"
        compiler = SLNCCompiler()
        _compile(compiler, config=llama_cfg, weights={}, output=out)
        parser = SLNCParser(str(out))
        assert parser.config["hidden_size"] == 4
        assert parser.tensor_count == 0  # no weights provided — empty table


class TestRoundTrip:
    def test_round_trip_all_tensors_match(self, tmp_path):
        out = tmp_path / "m.slnc"
        weights = _gpt2_weights()
        _compile(SLNCCompiler(), output=out)
        parser = SLNCParser(str(out))
        assert parser.tensor_count == len(weights)
        for name, expected in weights.items():
            np.testing.assert_array_equal(parser.get_tensor(name), expected)

    def test_get_block_returns_short_names(self, tmp_path):
        out = tmp_path / "m.slnc"
        _compile(SLNCCompiler(), output=out)
        parser = SLNCParser(str(out))
        block = parser.get_block(0)
        assert set(block) == set(GPT2_BLOCK_LAYOUT)
        np.testing.assert_array_equal(block["attn.c_attn.weight"], _gpt2_weights()["h.0.attn.c_attn.weight"])

    def test_unknown_tensor_raises_key_error(self, tmp_path):
        out = tmp_path / "m.slnc"
        _compile(SLNCCompiler(), output=out)
        with pytest.raises(KeyError):
            SLNCParser(str(out)).get_tensor("h.0.bogus.weight")


class TestBinaryLayout:
    def test_data_start_is_64_aligned(self, tmp_path):
        out = tmp_path / "m.slnc"
        _compile(SLNCCompiler(), output=out)
        with open(out, "rb") as f:
            f.seek(48)
            data_offset = struct.unpack("<I", f.read(4))[0]
        assert data_offset % 64 == 0

    def test_tensor_offsets_monotonic_and_contiguous(self, tmp_path):
        out = tmp_path / "m.slnc"
        _compile(SLNCCompiler(), output=out)
        parser = SLNCParser(str(out))
        offsets = []
        for name in parser._tensor_map:
            offset, shape, dtype, _ = parser._tensor_map[name]
            offsets.append(offset)
        assert offsets == sorted(offsets)
        for i in range(len(offsets) - 1):
            offset, shape, dtype, _ = parser._tensor_map[list(parser._tensor_map)[i]]
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            assert offset + nbytes == offsets[i + 1]

    def test_entry_fields_match_writer(self, tmp_path):
        out = tmp_path / "m.slnc"
        weights = _gpt2_weights()
        _compile(SLNCCompiler(), output=out)
        data = open(out, "rb").read()
        json_bytes = json.dumps(CFG, sort_keys=True).encode()
        header_size = spec.compute_header_size(json_bytes)
        pos = header_size
        expected = {k: v for k, v in weights.items()}
        for _ in range(len(expected)):
            name_len = struct.unpack("<I", data[pos:pos + 4])[0]
            pos += 4
            name = data[pos:pos + name_len].decode()
            pos += name_len
            offset, size, ndim = struct.unpack("<QII", data[pos:pos + 16])
            pos += 16
            shape = struct.unpack("<%dI" % ndim, data[pos:pos + 4 * ndim])
            pos += 4 * ndim
            dtype_code, crc = struct.unpack("<II", data[pos:pos + 8])
            pos += 8
            tensor = expected[name]
            assert size == tensor.nbytes
            assert shape == tensor.shape
            assert dtype_code == spec.dtype_to_code(tensor.dtype)
            assert crc == zlib.crc32(tensor.tobytes()) & 0xFFFFFFFF

    def test_tensor_data_written_in_computation_order(self, tmp_path):
        out = tmp_path / "m.slnc"
        _compile(SLNCCompiler(), output=out)
        parser = SLNCParser(str(out))
        names = list(parser._tensor_map.keys())
        assert names[0] == "h.0.ln_1.weight"
        assert names[1] == "h.0.ln_1.bias"
        assert "h.0.mlp.c_proj.weight" in names
        assert names.index("ln_f.weight") > names.index("h.0.mlp.c_proj.bias")
        assert names.index("wte.weight") > names.index("ln_f.bias")


class TestChecksums:
    def test_crc32_matches_zlib(self):
        data = b"some tensor bytes \x00\xff"
        assert _crc32(data) == zlib.crc32(data) & 0xFFFFFFFF

    def test_xxhash64_is_deterministic_int(self):
        h1 = _xxhash64(b"h.0.attn.c_attn.weight")
        h2 = _xxhash64(b"h.0.attn.c_attn.weight")
        assert isinstance(h1, int)
        assert h1 == h2

    def test_verify_all_true_after_compile(self, tmp_path):
        out = tmp_path / "m.slnc"
        _compile(SLNCCompiler(), output=out)
        assert SLNCParser(str(out)).verify_all() is True

    def test_corrupt_byte_detected(self, tmp_path):
        out = tmp_path / "m.slnc"
        _compile(SLNCCompiler(), output=out)
        data = bytearray(open(out, "rb").read())
        data[-1] ^= 0xFF
        corrupt = tmp_path / "bad.slnc"
        corrupt.write_bytes(bytes(data))
        parser = SLNCParser(str(corrupt), verify_checksums=True)
        assert parser.verify_all() is False

    def test_checksum_mismatch_raises_on_access(self, tmp_path):
        out = tmp_path / "m.slnc"
        _compile(SLNCCompiler(), output=out)
        data = bytearray(open(out, "rb").read())
        data[-1] ^= 0xFF
        corrupt = tmp_path / "bad.slnc"
        corrupt.write_bytes(bytes(data))
        parser = SLNCParser(str(corrupt), verify_checksums=True)
        last_name = list(parser._tensor_map)[-1]
        with pytest.raises(ValueError):
            parser.get_tensor(last_name)

    def test_invalid_magic_rejected(self, tmp_path):
        out = tmp_path / "m.slnc"
        _compile(SLNCCompiler(), output=out)
        data = bytearray(open(out, "rb").read())
        data[0:4] = b"BOG"
        bad = tmp_path / "bogus.slnc"
        bad.write_bytes(bytes(data))
        with pytest.raises(ValueError):
            SLNCParser(str(bad))


class TestOrderTensors:
    def test_bias_appended_after_weight(self):
        weights = _gpt2_weights()
        ordered = SLNCCompiler()._order_tensors(CFG, weights)
        names = [n for n, _ in ordered]
        assert names.index("h.0.attn.c_attn.bias") == names.index("h.0.attn.c_attn.weight") + 1

    def test_missing_biases_skipped(self, tmp_path):
        weights = {
            "h.0.ln_1.weight": np.ones(4, dtype=np.float32),
            "h.0.attn.c_attn.weight": np.zeros((4, 12), dtype=np.float32),
            "h.0.attn.c_proj.weight": np.zeros((4, 4), dtype=np.float32),
            "h.0.ln_2.weight": np.ones(4, dtype=np.float32),
            "h.0.mlp.c_fc.weight": np.zeros((4, 8), dtype=np.float32),
            "h.0.mlp.c_proj.weight": np.zeros((8, 4), dtype=np.float32),
            "ln_f.weight": np.ones(4, dtype=np.float32),
            "ln_f.bias": np.zeros(4, dtype=np.float32),
            "wte.weight": np.zeros((8, 4), dtype=np.float32),
            "wpe.weight": np.zeros((6, 4), dtype=np.float32),
        }
        out = tmp_path / "m.slnc"
        _compile(SLNCCompiler(), weights=weights, output=out)
        parser = SLNCParser(str(out))
        assert parser.tensor_count == len(weights)

    def test_llama_arch_detection_and_weight_tying(self, tmp_path):
        llama_cfg = {
            "n_layer": 1,
            "hidden_size": 4,
            "num_attention_heads": 1,
            "intermediate_size": 8,
            "vocab_size": 8,
            "max_position_embeddings": 6,
            "rope_theta": 10000.0,
            "model_type": "llama",
        }
        weights = {
            "model.embed_tokens.weight": np.arange(32, dtype=np.float32).reshape(8, 4),
            "model.layers.0.input_layernorm.weight": np.ones(4, dtype=np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.arange(16, dtype=np.float32).reshape(4, 4),
            "model.layers.0.self_attn.k_proj.weight": np.zeros((4, 4), dtype=np.float32),
            "model.layers.0.self_attn.v_proj.weight": np.zeros((4, 4), dtype=np.float32),
            "model.layers.0.self_attn.o_proj.weight": np.zeros((4, 4), dtype=np.float32),
            "model.layers.0.post_attention_layernorm.weight": np.ones(4, dtype=np.float32),
            "model.layers.0.mlp.gate_proj.weight": np.zeros((4, 8), dtype=np.float32),
            "model.layers.0.mlp.up_proj.weight": np.zeros((4, 8), dtype=np.float32),
            "model.layers.0.mlp.down_proj.weight": np.zeros((8, 4), dtype=np.float32),
            "model.norm.weight": np.ones(4, dtype=np.float32),
        }
        compiler = SLNCCompiler()
        ordered = compiler._order_tensors(llama_cfg, weights)
        names = [n for n, _ in ordered]
        assert names[0] == "model.layers.0.input_layernorm.weight"
        assert names.index("model.lm_head.weight") == len(names) - 1
        lm_head = dict(ordered)["model.lm_head.weight"]
        np.testing.assert_array_equal(lm_head, weights["model.embed_tokens.weight"])

    def test_llama_compile_round_trip(self, tmp_path):
        llama_cfg = {
            "n_layer": 1,
            "hidden_size": 4,
            "num_attention_heads": 1,
            "intermediate_size": 8,
            "vocab_size": 8,
            "max_position_embeddings": 6,
            "rope_theta": 10000.0,
            "model_type": "llama",
        }
        weights = {
            "model.embed_tokens.weight": np.arange(32, dtype=np.float32).reshape(8, 4),
            "model.layers.0.self_attn.q_proj.weight": np.arange(16, dtype=np.float32).reshape(4, 4),
            "model.norm.weight": np.ones(4, dtype=np.float32),
        }
        out = tmp_path / "llama.slnc"
        compiler = SLNCCompiler()
        _compile(compiler, config=llama_cfg, weights=weights, output=out)
        parser = SLNCParser(str(out))
        assert parser.tensor_count == 4  # 3 weights + tied lm_head
        assert parser.verify_all() is True
        np.testing.assert_array_equal(parser.get_tensor("model.embed_tokens.weight"), weights["model.embed_tokens.weight"])


class TestBlockSize:
    def test_gpt2_block_size(self):
        compiler = SLNCCompiler()
        assert compiler._compute_block_size(CFG) == 172 * 4

    def test_llama_block_size_includes_swiglu(self):
        llama_cfg = {"n_embd": 4, "n_inner": 8, "rope_theta": 10000.0}
        compiler = SLNCCompiler()
        # 9 weight matrices (q/k/v/o 4x4, gate/up/down 4x8/8x4, 2 layernorms 4)
        assert compiler._compute_block_size(llama_cfg) == 168 * 4

    def test_rope_detection_without_rope_theta(self):
        cfg = {"n_embd": 4, "n_inner": 8}
        compiler = SLNCCompiler()
        assert compiler._compute_block_size(cfg) == 172 * 4


class TestCompileHf:
    def _write_safetensors(self, path, weights):
        return _write_safetensors(path, weights)

    def test_compile_reads_safetensors_and_writes(self, tmp_path, monkeypatch):
        import domains.infrastructure.safetensors_loader as stl

        weights = {
            "h.0.ln_1.weight": np.ones(4, dtype=np.float32),
            "wte.weight": np.arange(32, dtype=np.float32).reshape(8, 4),
        }
        st_path = tmp_path / "model.safetensors"
        self._write_safetensors(st_path, weights)
        monkeypatch.setattr(stl, "_get_model_dir", lambda model_id: tmp_path)
        monkeypatch.setattr(stl, "_find_safetensors", lambda model_dir: st_path)
        monkeypatch.setattr(stl, "load_model_config", lambda model_id: CFG)

        out = tmp_path / "compiled.slnc"
        result = SLNCCompiler().compile("gpt2", output=str(out))
        assert result == str(out)
        parser = SLNCParser(str(out))
        assert parser.tensor_count == len(weights)
        assert parser.verify_all() is True
        np.testing.assert_array_equal(parser.get_tensor("h.0.ln_1.weight"), weights["h.0.ln_1.weight"])

    def test_compile_missing_safetensors_raises(self, tmp_path, monkeypatch):
        import domains.infrastructure.safetensors_loader as stl

        monkeypatch.setattr(stl, "_get_model_dir", lambda model_id: tmp_path)
        monkeypatch.setattr(stl, "_find_safetensors", lambda model_dir: None)
        monkeypatch.setattr(stl, "load_model_config", lambda model_id: CFG)
        with pytest.raises(FileNotFoundError):
            SLNCCompiler().compile("gpt2", output=str(tmp_path / "x.slnc"))

    def test_compile_bf16_safetensors_converts_to_f32(self, tmp_path, monkeypatch):
        import domains.infrastructure.safetensors_loader as stl

        raw = np.array([0x3F80, 0xC000], dtype=np.uint16)  # 1.0, -2.0 as bfloat16 bits
        st_path = tmp_path / "model.safetensors"
        _write_safetensors(st_path, {"h.0.ln_1.weight": raw}, dtype="BF16")
        monkeypatch.setattr(stl, "_get_model_dir", lambda model_id: tmp_path)
        monkeypatch.setattr(stl, "_find_safetensors", lambda model_dir: st_path)
        monkeypatch.setattr(stl, "load_model_config", lambda model_id: CFG)

        out = tmp_path / "bf16.slnc"
        SLNCCompiler().compile("gpt2", output=str(out))
        parser = SLNCParser(str(out))
        got = parser.get_tensor("h.0.ln_1.weight")
        expected = (raw.astype(np.uint32) << 16).view(np.float32)
        np.testing.assert_array_equal(got, expected)

    def test_compile_f16_safetensors_converts_to_f32(self, tmp_path, monkeypatch):
        import domains.infrastructure.safetensors_loader as stl

        f16 = np.array([1.0, -2.5, 0.25], dtype=np.float16)
        st_path = tmp_path / "model.safetensors"
        _write_safetensors(st_path, {"h.0.ln_1.weight": f16}, dtype="F16")
        monkeypatch.setattr(stl, "_get_model_dir", lambda model_id: tmp_path)
        monkeypatch.setattr(stl, "_find_safetensors", lambda model_dir: st_path)
        monkeypatch.setattr(stl, "load_model_config", lambda model_id: CFG)

        out = tmp_path / "f16.slnc"
        SLNCCompiler().compile("gpt2", output=str(out))
        parser = SLNCParser(str(out))
        np.testing.assert_array_equal(
            parser.get_tensor("h.0.ln_1.weight"), f16.astype(np.float32)
        )

    def test_compile_unknown_safetensors_dtype_reads_as_f32(self, tmp_path, monkeypatch):
        import domains.infrastructure.safetensors_loader as stl

        raw = np.arange(8, dtype=np.float32)
        st_path = tmp_path / "model.safetensors"
        _write_safetensors(st_path, {"h.0.ln_1.weight": raw}, dtype="I8")
        monkeypatch.setattr(stl, "_get_model_dir", lambda model_id: tmp_path)
        monkeypatch.setattr(stl, "_find_safetensors", lambda model_dir: st_path)
        monkeypatch.setattr(stl, "load_model_config", lambda model_id: CFG)

        out = tmp_path / "unknown.slnc"
        SLNCCompiler().compile("gpt2", output=str(out))
        parser = SLNCParser(str(out))
        np.testing.assert_array_equal(parser.get_tensor("h.0.ln_1.weight"), raw)


class TestCompileOutputAndProtect:
    def test_default_output_path(self, tmp_path, monkeypatch):
        import os

        import domains.infrastructure.model_protector as mp
        import domains.infrastructure.safetensors_loader as stl
        import domains.infrastructure.slnc.compiler as compiler_mod

        monkeypatch.setattr(mp, "protect_model", lambda *a, **k: None)
        weights = {"h.0.ln_1.weight": np.ones(4, dtype=np.float32)}
        st_path = tmp_path / "model.safetensors"
        _write_safetensors(st_path, weights)
        monkeypatch.setattr(stl, "_get_model_dir", lambda model_id: tmp_path)
        monkeypatch.setattr(stl, "_find_safetensors", lambda model_dir: st_path)
        monkeypatch.setattr(stl, "load_model_config", lambda model_id: CFG)

        compiler = SLNCCompiler()
        models_dir = (
            Path(compiler_mod.__file__).resolve().parents[4] / "models"
        )
        expected = models_dir / "gpt2.slnc"
        if expected.exists():
            os.chmod(expected, 0o644)
            os.remove(expected)
        result = None
        try:
            result = compiler.compile("gpt2")
            assert result == str(expected)
            assert Path(result).exists()
        finally:
            if result and Path(result).exists():
                os.remove(result)
            if models_dir.exists() and not any(models_dir.iterdir()):
                os.rmdir(models_dir)

    def test_protect_failure_is_silent(self, tmp_path, monkeypatch):
        import domains.infrastructure.safetensors_loader as stl

        weights = {"h.0.ln_1.weight": np.ones(4, dtype=np.float32)}
        st_path = tmp_path / "model.safetensors"
        _write_safetensors(st_path, weights)
        monkeypatch.setattr(stl, "_get_model_dir", lambda model_id: tmp_path)
        monkeypatch.setattr(stl, "_find_safetensors", lambda model_dir: st_path)
        monkeypatch.setattr(stl, "load_model_config", lambda model_id: CFG)

        def boom(model_id, files):
            raise RuntimeError("protection unavailable")

        monkeypatch.setattr(
            "domains.infrastructure.model_protector.protect_model", boom
        )
        out = tmp_path / "unprotected.slnc"
        result = SLNCCompiler().compile("gpt2", output=str(out))
        assert result == str(out)
        assert SLNCParser(str(out)).tensor_count == len(weights)


class TestParserEdges:
    def _compiled(self, tmp_path):
        out = tmp_path / "m.slnc"
        _compile(SLNCCompiler(), output=out)
        return out

    def test_unsupported_version_rejected(self, tmp_path):
        out = self._compiled(tmp_path)
        data = bytearray(open(out, "rb").read())
        struct.pack_into("<I", data, 4, 99)
        bad = tmp_path / "v99.slnc"
        bad.write_bytes(bytes(data))
        with pytest.raises(ValueError, match="Unsupported version"):
            SLNCParser(str(bad))

    def test_file_size_property(self, tmp_path):
        import os

        out = self._compiled(tmp_path)
        parser = SLNCParser(str(out))
        assert parser.file_size == os.path.getsize(out)

    def test_del_is_exception_safe(self, tmp_path):
        import types

        out = self._compiled(tmp_path)
        parser = SLNCParser(str(out))
        parser._mm = types.SimpleNamespace(
            close=lambda: (_ for _ in ()).throw(OSError("closed"))
        )
        parser.__del__()  # must not raise

    def test_repr(self, tmp_path):
        out = self._compiled(tmp_path)
        parser = SLNCParser(str(out))
        r = repr(parser)
        assert "SLNCParser" in r
        assert "1 layers" in r
        assert "tensors" in r

    def test_get_weights_dict_round_trip(self, tmp_path):
        out = self._compiled(tmp_path)
        parser = SLNCParser(str(out))
        weights = parser.get_weights_dict()
        assert set(weights) == set(_gpt2_weights())


class TestLlamaAutoBias:
    def test_bias_appended_when_not_in_layout(self, tmp_path):
        llama_cfg = {
            "n_layer": 1,
            "hidden_size": 4,
            "num_attention_heads": 1,
            "intermediate_size": 8,
            "vocab_size": 8,
            "max_position_embeddings": 6,
            "rope_theta": 10000.0,
            "model_type": "llama",
        }
        weights = {
            "model.embed_tokens.weight": np.zeros((8, 4), dtype=np.float32),
            "model.layers.0.input_layernorm.weight": np.ones(4, dtype=np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.zeros((4, 4), dtype=np.float32),
            "model.layers.0.self_attn.q_proj.bias": np.arange(4, dtype=np.float32),
            "model.norm.weight": np.ones(4, dtype=np.float32),
        }
        ordered = SLNCCompiler()._order_tensors(llama_cfg, weights)
        names = [n for n, _ in ordered]
        assert "model.layers.0.self_attn.q_proj.bias" in names
        assert names.index("model.layers.0.self_attn.q_proj.bias") == (
            names.index("model.layers.0.self_attn.q_proj.weight") + 1
        )


class TestXxhash:
    def test_uses_xxhash_when_installed(self, monkeypatch):
        import sys
        import types

        calls = []
        fake = types.ModuleType("xxhash")

        class _H:
            def __init__(self, data):
                calls.append(data)

            def intdigest(self):
                return 12345

        fake.xxh64 = lambda data: _H(data)
        monkeypatch.setitem(sys.modules, "xxhash", fake)
        assert _xxhash64(b"h.0.attn.c_attn.weight") == 12345
        assert calls == [b"h.0.attn.c_attn.weight"]

    def test_fallback_when_xxhash_missing(self, monkeypatch):
        import sys

        monkeypatch.delitem(sys.modules, "xxhash", raising=False)
        h = _xxhash64(b"h.0.attn.c_attn.weight")
        assert isinstance(h, int)
        assert h >= 0
