"""Tests for SLNC parser and spec modules."""

import json
import struct
import zlib

import numpy as np
import pytest

from domains.infrastructure.slnc.spec import (
    ALIGNMENT,
    DTYPE_FLOAT32,
    DTYPE_FLOAT16,
    DTYPE_INT32,
    DTYPE_INT64,
    DTYPE_UINT8,
    MAGIC,
    VERSION,
    _align,
    code_to_dtype,
    compute_header_size,
    compute_tensor_entry_size,
    dtype_to_code,
)
from domains.infrastructure.slnc.parser import SLNCParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_slnc_file(tensors, n_layer=2, n_embd=64, n_head=4, config=None):
    """Build a valid .slnc binary file from a list of tensor dicts.

    Each tensor dict: {"name": str, "data": np.ndarray}
    Returns bytes of the complete .slnc file.
    """
    if config is None:
        config = {"model": "test"}

    json_bytes = json.dumps(config, sort_keys=True).encode()
    header_size = compute_header_size(json_bytes)

    # Compute tensor table size (variable-length entries)
    table_size = 0
    for t in tensors:
        name_bytes = t["name"].encode()
        ndim = t["data"].ndim
        table_size += compute_tensor_entry_size(ndim, len(name_bytes))

    # Data starts after header + tensor table, aligned to ALIGNMENT
    data_start = _align(header_size + table_size)

    # Compute data offsets for each tensor (sequential, packed)
    data_offsets = []
    current = data_start
    for t in tensors:
        data_offsets.append(current)
        current += t["data"].nbytes

    # Build tensor table
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
        tensor_table += struct.pack("<I", 0)  # crc32 placeholder

    # Build header
    header = bytearray()
    header += MAGIC
    header += struct.pack("<I", VERSION)
    header += struct.pack("<I", 0)  # flags
    header += struct.pack("<I", n_layer)
    header += struct.pack("<I", n_embd)
    header += struct.pack("<I", n_head)
    header += struct.pack("<I", n_embd * 4)  # n_inner
    header += struct.pack("<I", 1000)  # vocab_size
    header += struct.pack("<I", 512)  # n_positions
    header += struct.pack("<I", n_layer)  # block_count
    header += struct.pack("<I", 128)  # block_size
    header += struct.pack("<I", len(tensors))  # tensor_count
    header += struct.pack("<I", data_start)  # data_offset
    header += b"\x00" * 24  # reserved
    header += struct.pack("<I", len(json_bytes))
    header += json_bytes

    # Pad to alignment before tensor table
    while len(header) % ALIGNMENT != 0:
        header += b"\x00"

    # Pad after tensor table to data_start alignment
    pre_data = len(header) + len(tensor_table)
    padding = b"\x00" * (data_start - pre_data)

    # Build tensor data section
    tensor_data = bytearray()
    for t in tensors:
        tensor_data += t["data"].tobytes()

    # Compute CRC32 for each tensor and patch tensor table
    pos = 0
    for t in tensors:
        name_bytes = t["name"].encode()
        ndim = t["data"].ndim
        name_len = struct.unpack_from("<I", tensor_table, pos)[0]
        pos += 4 + name_len + 8 + 4 + 4 + ndim * 4 + 4
        actual_crc = zlib.crc32(t["data"].tobytes()) & 0xFFFFFFFF
        struct.pack_into("<I", tensor_table, pos, actual_crc)
        pos += 4

    return bytes(header) + bytes(tensor_table) + padding + bytes(tensor_data)


def _make_tensor(name, shape, dtype=np.float32, fill=1.0):
    """Create a tensor dict with given fill value."""
    return {"name": name, "data": np.full(shape, fill, dtype=dtype)}


# ---------------------------------------------------------------------------
# spec.py tests
# ---------------------------------------------------------------------------

class TestSpecFunctions:
    def test_align_already_aligned(self):
        assert _align(64) == 64
        assert _align(128) == 128

    def test_align_needs_padding(self):
        assert _align(1) == 64
        assert _align(65) == 128
        assert _align(100) == 128

    def test_align_zero(self):
        assert _align(0) == 0

    def test_compute_header_size(self):
        json_bytes = b'{"model":"test"}'
        size = compute_header_size(json_bytes)
        raw = 4 + 4 + 4 + 64 + 4 + len(json_bytes)
        assert size == _align(raw)
        assert size % ALIGNMENT == 0

    def test_compute_header_size_empty_json(self):
        size = compute_header_size(b"{}")
        assert size % ALIGNMENT == 0

    def test_compute_tensor_entry_size(self):
        # ndim=2, name_len=5: 32 + 2*4 + 5 = 45
        assert compute_tensor_entry_size(2, 5) == 45

    def test_compute_tensor_entry_size_1d(self):
        assert compute_tensor_entry_size(1, 10) == 46

    def test_dtype_to_code_float32(self):
        assert dtype_to_code(np.float32) == DTYPE_FLOAT32

    def test_dtype_to_code_float16(self):
        assert dtype_to_code(np.float16) == DTYPE_FLOAT16

    def test_dtype_to_code_int32(self):
        assert dtype_to_code(np.int32) == DTYPE_INT32

    def test_dtype_to_code_int64(self):
        assert dtype_to_code(np.int64) == DTYPE_INT64

    def test_dtype_to_code_uint8(self):
        assert dtype_to_code(np.uint8) == DTYPE_UINT8

    def test_dtype_to_code_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            dtype_to_code(np.float64)

    def test_code_to_dtype_float32(self):
        assert code_to_dtype(DTYPE_FLOAT32) == np.float32

    def test_code_to_dtype_float16(self):
        assert code_to_dtype(DTYPE_FLOAT16) == np.float16

    def test_code_to_dtype_int32(self):
        assert code_to_dtype(DTYPE_INT32) == np.int32

    def test_code_to_dtype_int64(self):
        assert code_to_dtype(DTYPE_INT64) == np.int64

    def test_code_to_dtype_uint8(self):
        assert code_to_dtype(DTYPE_UINT8) == np.uint8

    def test_code_to_dtype_unknown(self):
        with pytest.raises(ValueError, match="Unknown dtype code"):
            code_to_dtype(999)


# ---------------------------------------------------------------------------
# SLNCParser — valid file tests
# ---------------------------------------------------------------------------

class TestSLNCParserValid:
    def test_open_valid_file(self, tmp_path):
        tensors = [_make_tensor("wte", (100, 32), fill=0.5)]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        assert parser.tensor_count == 1
        parser.close()

    def test_tensor_count(self, tmp_path):
        tensors = [
            _make_tensor("wte", (100, 32)),
            _make_tensor("wpe", (512, 32)),
        ]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        assert parser.tensor_count == 2
        parser.close()

    def test_get_tensor_returns_correct_values(self, tmp_path):
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        tensors = [_make_tensor("test_tensor", (4,), fill=0.0)]
        tensors[0]["data"] = data
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        result = parser.get_tensor("test_tensor")
        np.testing.assert_array_equal(result, data)
        parser.close()

    def test_get_tensor_shape(self, tmp_path):
        data = np.arange(24, dtype=np.float32).reshape(4, 6)
        tensors = [{"name": "matrix", "data": data}]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        result = parser.get_tensor("matrix")
        assert result.shape == (4, 6)
        np.testing.assert_array_equal(result, data)
        parser.close()

    def test_get_weights_dict(self, tmp_path):
        d1 = np.array([1.0, 2.0], dtype=np.float32)
        d2 = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        tensors = [
            {"name": "a", "data": d1},
            {"name": "b", "data": d2},
        ]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        weights = parser.get_weights_dict()
        assert set(weights.keys()) == {"a", "b"}
        np.testing.assert_array_equal(weights["a"], d1)
        np.testing.assert_array_equal(weights["b"], d2)
        parser.close()

    def test_get_weights_dict_parallel(self, tmp_path):
        d1 = np.array([1.0, 2.0], dtype=np.float32)
        d2 = np.array([3.0, 4.0, 5.0], dtype=np.float32)
        d3 = np.arange(12, dtype=np.float32).reshape(3, 4)
        tensors = [
            {"name": "a", "data": d1},
            {"name": "b", "data": d2},
            {"name": "c", "data": d3},
        ]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        weights = parser.get_weights_dict_parallel()
        assert set(weights.keys()) == {"a", "b", "c"}
        np.testing.assert_array_equal(weights["a"], d1)
        np.testing.assert_array_equal(weights["b"], d2)
        np.testing.assert_array_equal(weights["c"], d3)
        parser.close()

    def test_get_weights_dict_parallel_matches_sequential(self, tmp_path):
        tensors = [_make_tensor(f"t{i}", (10, 8), fill=float(i)) for i in range(20)]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        seq = parser.get_weights_dict()
        par = parser.get_weights_dict_parallel()
        assert set(seq.keys()) == set(par.keys())
        for key in seq:
            np.testing.assert_array_equal(seq[key], par[key])
        parser.close()

    def test_properties_tensor_count(self, tmp_path):
        tensors = [_make_tensor(f"t{i}", (2,)) for i in range(5)]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        assert parser.tensor_count == 5
        parser.close()

    def test_properties_param_count(self, tmp_path):
        t1 = np.zeros((10, 20), dtype=np.float32)  # 200 params
        t2 = np.zeros((5,), dtype=np.float32)  # 5 params
        tensors = [
            {"name": "t1", "data": t1},
            {"name": "t2", "data": t2},
        ]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        assert parser.param_count == 205
        parser.close()

    def test_properties_n_layer(self, tmp_path):
        tensors = [_make_tensor("wte", (10, 8))]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors, n_layer=4))
        parser = SLNCParser(str(path))
        assert parser.n_layer == 4
        parser.close()

    def test_properties_n_embd(self, tmp_path):
        tensors = [_make_tensor("wte", (10, 8))]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors, n_embd=128))
        parser = SLNCParser(str(path))
        assert parser.n_embd == 128
        parser.close()

    def test_properties_n_head(self, tmp_path):
        tensors = [_make_tensor("wte", (10, 8))]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors, n_head=8))
        parser = SLNCParser(str(path))
        assert parser.n_head == 8
        parser.close()

    def test_properties_vocab_size(self, tmp_path):
        tensors = [_make_tensor("wte", (10, 8))]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        assert parser.vocab_size == 1000
        parser.close()

    def test_properties_n_positions(self, tmp_path):
        tensors = [_make_tensor("wte", (10, 8))]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        assert parser.n_positions == 512
        parser.close()

    def test_config_property(self, tmp_path):
        config = {"architectures": ["GPT2LMHeadModel"], "n_layer": 6}
        tensors = [_make_tensor("wte", (10, 8))]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors, config=config))
        parser = SLNCParser(str(path))
        assert parser.config == config
        parser.close()

    def test_file_size_property(self, tmp_path):
        tensors = [_make_tensor("wte", (10, 8))]
        path = tmp_path / "test.slnc"
        data = _build_slnc_file(tensors)
        path.write_bytes(data)
        parser = SLNCParser(str(path))
        assert parser.file_size == len(data)
        parser.close()

    def test_verify_all_returns_true(self, tmp_path):
        t1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        t2 = np.array([4.0, 5.0], dtype=np.float32)
        tensors = [
            {"name": "a", "data": t1},
            {"name": "b", "data": t2},
        ]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        assert parser.verify_all() is True
        parser.close()

    def test_release_file_pages_returns_true(self, tmp_path):
        tensors = [_make_tensor("wte", (10, 8))]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        # May return True or False depending on OS madvise support
        result = parser.release_file_pages()
        assert isinstance(result, bool)
        parser.close()

    def test_close_idempotent(self, tmp_path):
        tensors = [_make_tensor("wte", (10, 8))]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        parser.close()
        parser.close()
        parser.close()

    def test_repr_returns_string(self, tmp_path):
        tensors = [_make_tensor("wte", (10, 8))]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        r = repr(parser)
        assert isinstance(r, str)
        assert "SLNCParser" in r
        assert "layers" in r
        assert "1 tensors" in r
        parser.close()


# ---------------------------------------------------------------------------
# SLNCParser — get_block tests
# ---------------------------------------------------------------------------

class TestSLNCParserGetBlock:
    def _build_block_file(self, tmp_path, n_layer=1):
        """Build a .slnc file with all 12 block tensors for each layer."""
        tensors = []
        block_tensor_names = [
            "ln_1.weight", "ln_1.bias",
            "attn.c_attn.weight", "attn.c_attn.bias",
            "attn.c_proj.weight", "attn.c_proj.bias",
            "ln_2.weight", "ln_2.bias",
            "mlp.c_fc.weight", "mlp.c_fc.bias",
            "mlp.c_proj.weight", "mlp.c_proj.bias",
        ]
        for layer in range(n_layer):
            for i, name in enumerate(block_tensor_names):
                tensors.append({"name": f"h.{layer}.{name}", "data": np.full((4,), float(i), dtype=np.float32)})
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors, n_layer=n_layer))
        return str(path)

    def test_get_block_returns_dict_with_expected_keys(self, tmp_path):
        path = self._build_block_file(tmp_path, n_layer=2)
        parser = SLNCParser(path)
        block = parser.get_block(0)
        expected_keys = [
            "ln_1.weight", "ln_1.bias",
            "attn.c_attn.weight", "attn.c_attn.bias",
            "attn.c_proj.weight", "attn.c_proj.bias",
            "ln_2.weight", "ln_2.bias",
            "mlp.c_fc.weight", "mlp.c_fc.bias",
            "mlp.c_proj.weight", "mlp.c_proj.bias",
        ]
        assert set(block.keys()) == set(expected_keys)
        parser.close()

    def test_get_block_values_are_ndarrays(self, tmp_path):
        path = self._build_block_file(tmp_path, n_layer=1)
        parser = SLNCParser(path)
        block = parser.get_block(0)
        for key, val in block.items():
            assert isinstance(val, np.ndarray), f"{key} is not ndarray"
        parser.close()

    def test_get_block_layer1(self, tmp_path):
        path = self._build_block_file(tmp_path, n_layer=3)
        parser = SLNCParser(path)
        block = parser.get_block(1)
        assert "attn.c_attn.weight" in block
        np.testing.assert_array_equal(
            block["attn.c_attn.weight"],
            np.full((4,), 2.0, dtype=np.float32),
        )
        parser.close()


# ---------------------------------------------------------------------------
# SLNCParser — error paths
# ---------------------------------------------------------------------------

class TestSLNCParserErrors:
    def test_invalid_magic(self, tmp_path):
        data = bytearray(b"NOPE" + b"\x00" * 200)
        path = tmp_path / "bad_magic.slnc"
        path.write_bytes(bytes(data))
        with pytest.raises(ValueError, match="Invalid magic"):
            SLNCParser(str(path))

    def test_wrong_version(self, tmp_path):
        data = bytearray(MAGIC)
        data += struct.pack("<I", 999)  # wrong version
        data += b"\x00" * 300
        path = tmp_path / "bad_version.slnc"
        path.write_bytes(bytes(data))
        with pytest.raises(ValueError, match="Unsupported version"):
            SLNCParser(str(path))

    def test_unknown_tensor_name(self, tmp_path):
        tensors = [_make_tensor("wte", (10, 8))]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        with pytest.raises(KeyError, match="Unknown tensor"):
            parser.get_tensor("nonexistent_tensor")
        parser.close()


# ---------------------------------------------------------------------------
# SLNCParser — dtype variations
# ---------------------------------------------------------------------------

class TestSLNCDtypes:
    def test_float16_tensor(self, tmp_path):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float16)
        tensors = [{"name": "f16", "data": data}]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        result = parser.get_tensor("f16")
        assert result.dtype == np.float16
        np.testing.assert_array_equal(result, data)
        parser.close()

    def test_int32_tensor(self, tmp_path):
        data = np.array([10, 20, 30], dtype=np.int32)
        tensors = [{"name": "i32", "data": data}]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        result = parser.get_tensor("i32")
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result, data)
        parser.close()

    def test_int64_tensor(self, tmp_path):
        data = np.array([100, 200], dtype=np.int64)
        tensors = [{"name": "i64", "data": data}]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        result = parser.get_tensor("i64")
        assert result.dtype == np.int64
        np.testing.assert_array_equal(result, data)
        parser.close()

    def test_uint8_tensor(self, tmp_path):
        data = np.array([0, 127, 255], dtype=np.uint8)
        tensors = [{"name": "u8", "data": data}]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        result = parser.get_tensor("u8")
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, data)
        parser.close()


# ---------------------------------------------------------------------------
# SLNCParser — multiple tensors and larger data
# ---------------------------------------------------------------------------

class TestSLNCParserMultipleTensors:
    def test_many_tensors(self, tmp_path):
        tensors = [_make_tensor(f"t{i}", (3, 4), fill=float(i)) for i in range(20)]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        assert parser.tensor_count == 20
        for i in range(20):
            result = parser.get_tensor(f"t{i}")
            np.testing.assert_allclose(result, np.full((3, 4), float(i), dtype=np.float32))
        parser.close()

    def test_multidimensional_tensor(self, tmp_path):
        data = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        tensors = [{"name": "rank4", "data": data}]
        path = tmp_path / "test.slnc"
        path.write_bytes(_build_slnc_file(tensors))
        parser = SLNCParser(str(path))
        result = parser.get_tensor("rank4")
        assert result.shape == (2, 3, 4, 5)
        np.testing.assert_array_equal(result, data)
        parser.close()
