"""_safetensors reader tests — the format built byte-by-byte per spec.

Ground truth is the FORMAT, not our code: every fixture assembles the
8-byte length prefix, the JSON header, and the raw buffer by hand, so a
reader bug cannot hide behind a matching writer bug.
"""

from __future__ import annotations

import importlib.util
import json
import struct
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "_safetensors", Path(__file__).resolve().parents[1] / "scripts" / "_safetensors.py"
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
read_safetensors = _mod.read_safetensors


def _build(
    tmp_path: Path,
    tensors: dict[str, tuple[str, list[int], np.ndarray]],
    metadata: dict | None = None,
) -> Path:
    """Assemble a .safetensors file exactly per the spec."""
    header: dict = {}
    if metadata:
        header["__metadata__"] = metadata
    buffer = b""
    for name, (dtype, shape, array) in tensors.items():
        raw = array.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(buffer), len(buffer) + len(raw)],
        }
        buffer += raw
    head = json.dumps(header).encode()
    path = tmp_path / "fixture.safetensors"
    path.write_bytes(struct.pack("<Q", len(head)) + head + buffer)
    return path


class TestReadSafetensors:
    def test_multi_tensor_round_trip_with_metadata(self, tmp_path):
        w = np.arange(6, dtype="<f4").reshape(2, 3)
        b = np.array([-7, 42], dtype="<i8")
        path = _build(
            tmp_path,
            {
                "layers.0.weight": ("F32", [2, 3], w),
                "layers.0.bias": ("I64", [2], b),
            },
            metadata={"format": "pt"},
        )
        tensors = read_safetensors(path)
        assert set(tensors) == {"layers.0.weight", "layers.0.bias"}
        np.testing.assert_array_equal(tensors["layers.0.weight"], w)
        assert tensors["layers.0.weight"].dtype == np.float32
        np.testing.assert_array_equal(tensors["layers.0.bias"], b)
        assert tensors["layers.0.bias"].dtype == np.int64

    def test_f16_and_bool(self, tmp_path):
        f16 = np.array([1.5, -0.25], dtype="<f2")
        flags = np.array([True, False, True])
        path = _build(
            tmp_path,
            {"a": ("F16", [2], f16), "flag": ("BOOL", [3], flags)},
        )
        tensors = read_safetensors(path)
        np.testing.assert_array_equal(tensors["a"], f16)
        assert tensors["a"].dtype == np.float16
        np.testing.assert_array_equal(tensors["flag"], flags)

    def test_bf16_widening_preserves_representable_values(self, tmp_path):
        """bfloat16 keeps the float32 top 16 bits — exactly representable
        values must come back bit-identical."""
        original = np.array([1.0, 2.5, -0.25, 1024.0], dtype="<f4")
        bf16_raw = (original.view("<u4") >> 16).astype("<u2").tobytes()
        header = json.dumps(
            {"w": {"dtype": "BF16", "shape": [4], "data_offsets": [0, len(bf16_raw)]}}
        ).encode()
        path = tmp_path / "bf16.safetensors"
        path.write_bytes(struct.pack("<Q", len(header)) + header + bf16_raw)

        tensors = read_safetensors(path)
        np.testing.assert_array_equal(tensors["w"], original)
        assert tensors["w"].dtype == np.float32

    def test_scalar_tensor(self, tmp_path):
        scalar = np.array(3.5, dtype="<f4")
        path = _build(tmp_path, {"loss": ("F32", [], scalar)})
        tensors = read_safetensors(path)
        assert float(tensors["loss"]) == 3.5

    def test_unknown_dtype_raises_honestly(self, tmp_path):
        path = _build(tmp_path, {"x": ("F8M5", [2], np.zeros(2, dtype="<f4"))})
        with pytest.raises(ValueError, match="F8M5"):
            read_safetensors(path)

    def test_too_short_file_raises(self, tmp_path):
        path = tmp_path / "tiny.safetensors"
        path.write_bytes(b"\x01\x02")
        with pytest.raises(ValueError, match="too short"):
            read_safetensors(path)

    def test_offsets_slice_the_buffer_exactly(self, tmp_path):
        """The second tensor must start exactly where the first ends —
        offsets, not adjacency, define the layout."""
        t1 = np.array([1, 2, 3], dtype="<u1")
        t2 = np.array([4, 5], dtype="<u1")
        path = _build(tmp_path, {"first": ("U8", [3], t1), "second": ("U8", [2], t2)})
        tensors = read_safetensors(path)
        np.testing.assert_array_equal(tensors["first"], t1)
        np.testing.assert_array_equal(tensors["second"], t2)
