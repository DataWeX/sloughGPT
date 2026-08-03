"""Tests for slnc_loader — memory-mapped .slnc reader."""

import json
import os
import struct

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


@pytest.fixture
def loader(tmp_path):
    return SLNCLoader(str(_write_slnc(tmp_path / "m.slnc")))


class TestLoad:
    def test_config_roundtrip(self, loader):
        assert loader.config == CFG

    def test_file_size(self, loader, tmp_path):
        assert loader.file_size == os.path.getsize(tmp_path / "m.slnc")

    def test_repr(self, loader):
        r = repr(loader)
        assert "m.slnc" in r
        assert "1 layers" in r

    def test_invalid_magic(self, tmp_path):
        p = tmp_path / "bad.slnc"
        p.write_bytes(b"XXXX" + b"\x00" * 60)
        with pytest.raises(ValueError, match="Invalid magic"):
            SLNCLoader(str(p))

    def test_unsupported_version(self, tmp_path):
        p = tmp_path / "v.slnc"
        p.write_bytes(sf.SLNC_MAGIC + struct.pack("<I", 99) + b"\x00" * 56)
        with pytest.raises(ValueError, match="Unsupported version"):
            SLNCLoader(str(p))


class TestGetTensor:
    def test_block_tensor_values(self, loader):
        expected = np.arange(4).astype(np.float32)
        np.testing.assert_array_equal(loader.get_tensor("blocks.0.ln_1.weight"), expected)

    def test_mlp_weight_shape(self, loader):
        assert loader.get_tensor("blocks.0.mlp.c_fc.weight").shape == (4, 8)

    def test_c_attn_weight(self, loader):
        assert loader.get_tensor("blocks.0.attn.c_attn.weight").shape == (4, 12)

    def test_non_block_tensor_values(self, loader):
        expected = (np.arange(8 * 4) + 50000).astype(np.float32).reshape(8, 4)
        np.testing.assert_array_equal(loader.get_tensor("wte.weight"), expected)

    def test_unknown_tensor_raises(self, loader):
        with pytest.raises(KeyError):
            loader.get_tensor("blocks.0.bogus.weight")

    def test_view_is_read_only(self, loader):
        arr = loader.get_tensor("blocks.0.ln_1.weight")
        assert not arr.flags.writeable

    def test_returns_float32(self, loader):
        assert loader.get_tensor("wte.weight").dtype == np.float32


class TestGetBlock:
    def test_returns_all_block_tensors(self, loader):
        block = loader.get_block(0)
        assert set(block) == {n for n, _ in sf.GPT2_BLOCK_TENSORS}

    def test_block_values(self, loader):
        block = loader.get_block(0)
        np.testing.assert_array_equal(block["ln_1.weight"], np.arange(4).astype(np.float32))

    def test_layer_two_values(self, tmp_path):
        cfg = {**CFG, "n_layer": 2}
        ld = SLNCLoader(str(_write_slnc(tmp_path / "m2.slnc", cfg)))
        layer0 = np.arange(4).astype(np.float32)
        layer1 = (np.arange(4) + 1000).astype(np.float32)
        np.testing.assert_array_equal(ld.get_tensor("blocks.0.ln_1.weight"), layer0)
        np.testing.assert_array_equal(ld.get_tensor("blocks.1.ln_1.weight"), layer1)


class TestGetWeightsDict:
    def test_includes_block_and_non_block(self, loader):
        weights = loader.get_weights_dict()
        assert "blocks.0.attn.c_attn.weight" in weights
        assert "wte.weight" in weights
        assert "ln_f.bias" in weights

    def test_key_count(self, loader):
        assert len(loader.get_weights_dict()) == len(sf.GPT2_BLOCK_TENSORS) + 4

    def test_non_block_offsets_sequential(self, tmp_path):
        cfg = {**CFG, "n_layer": 2}
        ld = SLNCLoader(str(_write_slnc(tmp_path / "m2.slnc", cfg)))
        names = ["wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"]
        offs = [ld._tensor_map[n][0] for n in names]
        sizes = [int(np.prod(ld._tensor_map[n][1])) * 4 for n in names]
        for i in range(len(offs) - 1):
            assert offs[i + 1] == offs[i] + sizes[i]

    def test_del_tolerates_closed_fd(self, tmp_path):
        loader = SLNCLoader(str(_write_slnc(tmp_path / "m.slnc")))
        os.close(loader._fd)
        del loader
