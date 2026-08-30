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

    def test_repr_contains_size_mb(self, loader):
        r = repr(loader)
        assert "MB" in r

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

    def test_corrupt_json_header(self, tmp_path):
        """json_len points beyond file bounds → ValueError."""
        p = tmp_path / "corrupt.slnc"
        with open(p, "wb") as f:
            f.write(sf.SLNC_MAGIC)
            f.write(struct.pack("<I", sf.SLNC_VERSION))
            # Write 11 header fields (n_layer through tensor_count)
            for _ in range(11):
                f.write(struct.pack("<I", 0))
            # Set json_len to a huge value exceeding file bounds
            f.write(struct.pack("<I", 999999))
        with pytest.raises(ValueError, match="Corrupt SLNC header"):
            SLNCLoader(str(p))

    def test_corrupt_json_content(self, tmp_path):
        """Valid json_len but invalid JSON bytes → ValueError."""
        p = tmp_path / "bad_json.slnc"
        with open(p, "wb") as f:
            f.write(sf.SLNC_MAGIC)
            f.write(struct.pack("<I", sf.SLNC_VERSION))
            for _ in range(11):
                f.write(struct.pack("<I", 0))
            bad_json = b"{not valid json"
            f.write(struct.pack("<I", len(bad_json)))
            f.write(bad_json)
        with pytest.raises(ValueError, match="Corrupt SLNC config JSON"):
            SLNCLoader(str(p))

    def test_config_property_returns_dict(self, loader):
        cfg = loader.config
        assert isinstance(cfg, dict)
        assert "n_embd" in cfg

    def test_path_stored(self, loader, tmp_path):
        assert loader._path == str(tmp_path / "m.slnc")


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

    def test_ln_1_bias_shape(self, loader):
        assert loader.get_tensor("blocks.0.ln_1.bias").shape == (4,)

    def test_ln_2_weight_shape(self, loader):
        assert loader.get_tensor("blocks.0.ln_2.weight").shape == (4,)

    def test_ln_2_bias_shape(self, loader):
        assert loader.get_tensor("blocks.0.ln_2.bias").shape == (4,)

    def test_attn_c_attn_bias_shape(self, loader):
        assert loader.get_tensor("blocks.0.attn.c_attn.bias").shape == (12,)

    def test_attn_c_proj_weight_shape(self, loader):
        assert loader.get_tensor("blocks.0.attn.c_proj.weight").shape == (4, 4)

    def test_attn_c_proj_bias_shape(self, loader):
        assert loader.get_tensor("blocks.0.attn.c_proj.bias").shape == (4,)

    def test_mlp_c_fc_bias_shape(self, loader):
        assert loader.get_tensor("blocks.0.mlp.c_fc.bias").shape == (8,)

    def test_mlp_c_proj_weight_shape(self, loader):
        assert loader.get_tensor("blocks.0.mlp.c_proj.weight").shape == (8, 4)

    def test_mlp_c_proj_bias_shape(self, loader):
        assert loader.get_tensor("blocks.0.mlp.c_proj.bias").shape == (4,)

    def test_wpe_weight_shape(self, loader):
        assert loader.get_tensor("wpe.weight").shape == (6, 4)

    def test_ln_f_weight_shape(self, loader):
        assert loader.get_tensor("ln_f.weight").shape == (4,)

    def test_ln_f_bias_shape(self, loader):
        assert loader.get_tensor("ln_f.bias").shape == (4,)

    def test_wpe_weight_values(self, loader):
        """Each non-block tensor starts at arange(size//4) + 50000."""
        expected = (np.arange(6 * 4) + 50000).astype(np.float32).reshape(6, 4)
        np.testing.assert_array_equal(loader.get_tensor("wpe.weight"), expected)

    def test_ln_f_weight_values(self, loader):
        expected = (np.arange(4) + 50000).astype(np.float32)
        np.testing.assert_array_equal(loader.get_tensor("ln_f.weight"), expected)

    def test_ln_f_bias_values(self, loader):
        expected = (np.arange(4) + 50000).astype(np.float32)
        np.testing.assert_array_equal(loader.get_tensor("ln_f.bias"), expected)


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

    def test_block_returns_dict(self, loader):
        block = loader.get_block(0)
        assert isinstance(block, dict)

    def test_block_all_values_are_float32(self, loader):
        block = loader.get_block(0)
        for name, arr in block.items():
            assert arr.dtype == np.float32, f"{name} is not float32"

    def test_block_mlp_c_proj_values(self, loader):
        """Each block tensor starts at arange(size//4) + layer_idx*1000."""
        block = loader.get_block(0)
        expected = (np.arange(32) + 0).astype(np.float32).reshape(8, 4)
        np.testing.assert_array_equal(block["mlp.c_proj.weight"], expected)

    def test_two_layer_block_count(self, tmp_path):
        cfg = {**CFG, "n_layer": 2}
        ld = SLNCLoader(str(_write_slnc(tmp_path / "m2.slnc", cfg)))
        block0 = ld.get_block(0)
        block1 = ld.get_block(1)
        assert set(block0.keys()) == set(block1.keys())


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

    def test_all_keys_are_strings(self, loader):
        weights = loader.get_weights_dict()
        for key in weights:
            assert isinstance(key, str)

    def test_all_values_are_ndarrays(self, loader):
        weights = loader.get_weights_dict()
        for val in weights.values():
            assert isinstance(val, np.ndarray)

    def test_two_layer_weight_dict_count(self, tmp_path):
        cfg = {**CFG, "n_layer": 2}
        ld = SLNCLoader(str(_write_slnc(tmp_path / "m2.slnc", cfg)))
        # 2 layers * 12 block tensors + 4 non-block = 28
        assert len(ld.get_weights_dict()) == 2 * len(sf.GPT2_BLOCK_TENSORS) + 4

    def test_tensor_map_consistency(self, loader):
        """tensor_map has entries for all block and non-block tensors."""
        block_count = CFG["n_layer"] * len(sf.GPT2_BLOCK_TENSORS)
        non_block_count = 4
        assert len(loader._tensor_map) == block_count + non_block_count


class TestMultiLayer:
    @pytest.fixture
    def loader_3layer(self, tmp_path):
        cfg = {**CFG, "n_layer": 3}
        return SLNCLoader(str(_write_slnc(tmp_path / "m3.slnc", cfg)))

    def test_three_layers_accessible(self, loader_3layer):
        for layer in range(3):
            arr = loader_3layer.get_tensor(f"blocks.{layer}.ln_1.weight")
            assert arr.shape == (4,)

    def test_three_layer_values_differ(self, loader_3layer):
        vals = []
        for layer in range(3):
            arr = loader_3layer.get_tensor(f"blocks.{layer}.ln_1.weight")
            vals.append(arr.copy())
        assert not np.array_equal(vals[0], vals[1])
        assert not np.array_equal(vals[1], vals[2])

    def test_three_layer_key_count(self, loader_3layer):
        assert len(loader_3layer.get_weights_dict()) == 3 * len(sf.GPT2_BLOCK_TENSORS) + 4

    def test_three_layer_repr(self, loader_3layer):
        r = repr(loader_3layer)
        assert "3 layers" in r


class TestEdgeCases:
    def test_get_tensor_before_blocks(self, loader):
        """non-block tensors start after all blocks."""
        wte_off = loader._tensor_map["wte.weight"][0]
        block_end = (loader._non_block_offset
                     - loader._block_size * loader._n_layer
                     + loader._block_size * loader._n_layer)
        assert wte_off == block_end

    def test_file_size_positive(self, loader):
        assert loader.file_size > 0

    def test_tensor_count_matches(self, loader):
        expected = CFG["n_layer"] * len(sf.GPT2_BLOCK_TENSORS) + 4
        assert len(loader._tensor_map) == expected

    def test_block_size_positive(self, loader):
        assert loader._block_size > 0

    def test_n_embd_from_config(self, loader):
        assert loader._n_embd == CFG["n_embd"]

    def test_n_layer_from_config(self, loader):
        assert loader._n_layer == CFG["n_layer"]

    def test_n_head_from_config(self, loader):
        assert loader._n_head == CFG["n_head"]

    def test_vocab_size_from_config(self, loader):
        assert loader._vocab_size == CFG["vocab_size"]

    def test_n_positions_from_config(self, loader):
        assert loader._n_positions == CFG["n_positions"]
