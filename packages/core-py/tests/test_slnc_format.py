"""Tests for slnc_format — .slnc neural cache layout, writer, header reader."""

import json
import struct
import sys
import types

import numpy as np
import pytest

from domains.infrastructure import slnc_format as sf
from domains.infrastructure.slnc_loader import SLNCLoader

CFG = {
    "n_layer": 1,
    "n_embd": 4,
    "n_head": 1,
    "n_inner": 8,
    "vocab_size": 8,
    "n_positions": 6,
    "model_type": "gpt2",
}


def _write_slnc(path, config=CFG):
    """Write a .slnc file in convert_to_slnc's exact format."""
    layout = sf.compute_layout(config)
    with open(path, "wb") as out:
        out.write(sf.SLNC_MAGIC)
        for v in (
            sf.SLNC_VERSION,
            config["n_layer"],
            config["n_embd"],
            config.get("n_head", 12),
            config.get("n_inner", config["n_embd"] * 4),
            config["vocab_size"],
            config.get("n_positions", 1024),
            config["n_layer"],
            layout["block_size"],
            layout["header_size"] + layout["block_size"] * config["n_layer"],
            sum(s for _, _, s in layout["non_block_tensors"]),
        ):
            out.write(struct.pack("<I", v))
        tensor_count = config["n_layer"] * len(sf.GPT2_BLOCK_TENSORS) + len(
            layout["non_block_tensors"]
        )
        out.write(struct.pack("<I", tensor_count))
        json_bytes = json.dumps(config).encode()
        out.write(struct.pack("<I", len(json_bytes)))
        out.write(json_bytes)
        out.write(b"\x00" * (layout["header_size"] - out.tell()))
        for layer_idx in range(config["n_layer"]):
            for name, _, size in layout["block_tensors"]:
                shape = sf._get_tensor_shape(name, config)
                out.write((np.arange(size // 4) + layer_idx * 1000).astype(np.float32).reshape(shape).tobytes())
        for name, _, size in layout["non_block_tensors"]:
            shape = sf._get_tensor_shape(name, config)
            out.write((np.arange(size // 4) + 50000).astype(np.float32).reshape(shape).tobytes())
    return path


def _weights_from_config(config):
    """Build a synthetic safetensors-style weight dict matching config."""
    weights = {}
    for layer in range(config["n_layer"]):
        for name, _, _ in sf.compute_layout(config)["block_tensors"]:
            shape = sf._get_tensor_shape(name, config)
            weights[f"h.{layer}.{name}"] = (
                np.arange(int(np.prod(shape))) + layer * 1000
            ).astype(np.float32).reshape(shape)
    for name in ["wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"]:
        shape = sf._get_tensor_shape(name, config)
        weights[name] = (np.arange(int(np.prod(shape))) + 50000).astype(np.float32).reshape(shape)
    return weights


class _FakeSafeOpen:
    def __init__(self, weights):
        self._weights = weights

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def keys(self):
        return list(self._weights)

    def get_tensor(self, key):
        return self._weights[key]


@pytest.fixture
def fake_hf(monkeypatch, tmp_path):
    """Stub HF/safetensors plumbing so convert_to_slnc runs without a model."""
    import domains.infrastructure.safetensors_loader as stl

    config = dict(CFG)
    weights = _weights_from_config(config)
    monkeypatch.setattr(stl, "_get_model_dir", lambda model_id: tmp_path)
    monkeypatch.setattr(stl, "_find_safetensors", lambda model_dir: model_dir / "w.safetensors")
    monkeypatch.setattr(stl, "load_model_config", lambda model_id: config)
    fake_mod = types.ModuleType("safetensors")
    fake_mod.safe_open = lambda path, framework: _FakeSafeOpen(weights)
    monkeypatch.setitem(sys.modules, "safetensors", fake_mod)
    return config, weights


class TestConstants:
    def test_magic(self):
        assert sf.SLNC_MAGIC == b"SLNC"

    def test_version(self):
        assert sf.SLNC_VERSION == 1


class TestTensorShape:
    def test_ln_weight(self):
        assert sf._get_tensor_shape("ln_1.weight", CFG) == (4,)

    def test_ln_bias(self):
        assert sf._get_tensor_shape("ln_2.bias", CFG) == (4,)

    def test_c_attn_weight_fused_qkv(self):
        assert sf._get_tensor_shape("attn.c_attn.weight", CFG) == (4, 12)

    def test_c_attn_bias(self):
        assert sf._get_tensor_shape("attn.c_attn.bias", CFG) == (12,)

    def test_c_proj_weight(self):
        assert sf._get_tensor_shape("attn.c_proj.weight", CFG) == (4, 4)

    def test_mlp_fc_weight(self):
        assert sf._get_tensor_shape("mlp.c_fc.weight", CFG) == (4, 8)

    def test_mlp_fc_bias(self):
        assert sf._get_tensor_shape("mlp.c_fc.bias", CFG) == (8,)

    def test_mlp_proj_weight(self):
        assert sf._get_tensor_shape("mlp.c_proj.weight", CFG) == (8, 4)

    def test_wte(self):
        assert sf._get_tensor_shape("wte.weight", CFG) == (8, 4)

    def test_wpe(self):
        assert sf._get_tensor_shape("wpe.weight", CFG) == (6, 4)

    def test_ln_f(self):
        assert sf._get_tensor_shape("ln_f.weight", CFG) == (4,)

    def test_unknown_tensor_raises(self):
        with pytest.raises(ValueError):
            sf._get_tensor_shape("bogus.weight", CFG)

    def test_default_n_inner_is_4x(self):
        cfg = {k: v for k, v in CFG.items() if k != "n_inner"}
        assert sf._get_tensor_shape("mlp.c_fc.weight", cfg) == (4, 16)

    def test_default_n_positions(self):
        cfg = {k: v for k, v in CFG.items() if k != "n_positions"}
        assert sf._get_tensor_shape("wpe.weight", cfg) == (1024, 4)


class TestTensorSize:
    def test_ln_size(self):
        assert sf._compute_tensor_size("ln_1.weight", CFG) == 16

    def test_c_attn_size(self):
        assert sf._compute_tensor_size("attn.c_attn.weight", CFG) == 192

    def test_wte_size(self):
        assert sf._compute_tensor_size("wte.weight", CFG) == 128

    def test_size_matches_shape_product(self):
        for name, _ in sf.GPT2_BLOCK_TENSORS:
            assert sf._compute_tensor_size(name, CFG) == int(np.prod(sf._get_tensor_shape(name, CFG))) * 4


class TestComputeLayout:
    def test_block_tensor_offsets_contiguous(self):
        layout = sf.compute_layout(CFG)
        offset = 0
        for _, off, size in layout["block_tensors"]:
            assert off == offset
            offset += size
        assert layout["block_size"] == offset

    def test_block_tensors_in_computation_order(self):
        layout = sf.compute_layout(CFG)
        names = [n for n, _, _ in layout["block_tensors"]]
        assert names == [n for n, _ in sf.GPT2_BLOCK_TENSORS]

    def test_header_size_64_aligned(self):
        layout = sf.compute_layout(CFG)
        assert layout["header_size"] % 64 == 0

    def test_non_block_offset_after_blocks(self):
        layout = sf.compute_layout(CFG)
        non_block_offset = layout["header_size"] + layout["block_size"] * CFG["n_layer"]
        assert layout["total_size"] == non_block_offset + sum(
            s for _, _, s in layout["non_block_tensors"]
        )

    def test_non_block_tensors_order(self):
        layout = sf.compute_layout(CFG)
        assert [n for n, _, _ in layout["non_block_tensors"]] == [
            "wte.weight",
            "wpe.weight",
            "ln_f.weight",
            "ln_f.bias",
        ]

    def test_total_size_ends_after_non_block(self):
        layout = sf.compute_layout(CFG)
        last = layout["non_block_tensors"][-1]
        assert layout["total_size"] == last[1] + last[2]

    def test_multi_layer_blocks(self):
        cfg = {**CFG, "n_layer": 3}
        layout = sf.compute_layout(cfg)
        non_block_offset = layout["header_size"] + layout["block_size"] * 3
        assert layout["total_size"] == non_block_offset + sum(
            s for _, _, s in layout["non_block_tensors"]
        )
        assert len(layout["block_tensors"]) == len(sf.GPT2_BLOCK_TENSORS)


class TestConvertToSlnc:
    def test_roundtrip_values(self, fake_hf, tmp_path):
        out = str(tmp_path / "m.slnc")
        created = sf.convert_to_slnc("gpt2", output_path=out)
        assert created == out
        loader = SLNCLoader(out)
        for key, source in _weights_from_config(CFG).items():
            loader_key = "blocks." + key[2:] if key.startswith("h.") else key
            np.testing.assert_array_equal(loader.get_tensor(loader_key), source)

    def test_roundtrip_header(self, fake_hf, tmp_path):
        out = str(tmp_path / "m.slnc")
        sf.convert_to_slnc("gpt2", output_path=out)
        h = sf.read_slnc_header(out)
        assert h["version"] == sf.SLNC_VERSION
        assert h["n_layer"] == CFG["n_layer"]
        assert h["n_embd"] == CFG["n_embd"]
        assert h["n_head"] == CFG["n_head"]
        assert h["n_inner"] == CFG["n_inner"]
        assert h["vocab_size"] == CFG["vocab_size"]
        assert h["n_positions"] == CFG["n_positions"]
        assert h["config"] == CFG
        assert h["tensor_count"] == CFG["n_layer"] * len(sf.GPT2_BLOCK_TENSORS) + 4

    def test_roundtrip_loader_and_reader_agree(self, fake_hf, tmp_path):
        out = str(tmp_path / "m.slnc")
        sf.convert_to_slnc("gpt2", output_path=out)
        h = sf.read_slnc_header(out)
        loader = SLNCLoader(out)
        assert h["block_size"] == loader._block_size
        assert h["non_block_offset"] == loader._non_block_offset

    def test_shape_mismatch_raises(self, fake_hf, tmp_path):
        config, weights = fake_hf
        bad_key = "h.0.attn.c_attn.weight"
        weights[bad_key] = weights[bad_key][:-1]
        with pytest.raises(ValueError, match="Shape mismatch"):
            sf.convert_to_slnc("gpt2", output_path=str(tmp_path / "m.slnc"))

    def test_missing_tensor_raises(self, fake_hf, tmp_path):
        config, weights = fake_hf
        del weights["h.0.mlp.c_proj.weight"]
        with pytest.raises(KeyError, match="mlp.c_proj.weight"):
            sf.convert_to_slnc("gpt2", output_path=str(tmp_path / "m.slnc"))

    def test_roundtrip_tensor_count(self, fake_hf, tmp_path):
        out = str(tmp_path / "m.slnc")
        sf.convert_to_slnc("gpt2", output_path=out)
        loader = SLNCLoader(out)
        h = sf.read_slnc_header(out)
        assert h["block_count"] == 1
        assert len(loader._tensor_map) == len(sf.GPT2_BLOCK_TENSORS) + 4


class TestReadSlncHeader:
    def test_valid_file(self, tmp_path):
        p = _write_slnc(tmp_path / "m.slnc")
        h = sf.read_slnc_header(str(p))
        assert h["config"]["n_embd"] == 4
        assert h["n_layer"] == 1

    def test_invalid_magic(self, tmp_path):
        p = tmp_path / "bad.slnc"
        p.write_bytes(b"XXXX" + b"\x00" * 60)
        with pytest.raises(ValueError, match="Invalid magic"):
            sf.read_slnc_header(str(p))

    def test_unsupported_version(self, tmp_path):
        p = tmp_path / "v.slnc"
        p.write_bytes(sf.SLNC_MAGIC + struct.pack("<I", 99) + b"\x00" * 56)
        with pytest.raises(ValueError, match="Unsupported version"):
            sf.read_slnc_header(str(p))
