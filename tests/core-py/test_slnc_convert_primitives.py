"""Conversion primitives: shared safetensors reader + integer preservation.

Covers the consolidation of ``model_loader._try_convert_to_slnc`` onto
``SLNCCompiler.read_weights`` / ``compile_from_dict`` and the I64/I32 paths
added to the shared reader (previously fell through to float32).
"""

from __future__ import annotations

import json
import struct

import numpy as np

from domain.infrastructure._internal.safetensors_loader import load_model_weights
from domain.infrastructure._internal.slnc.compiler import SLNCCompiler
from domain.infrastructure._internal.slnc.parser import SLNCParser

BF16_VALS = np.array([0.5, 1.0, 2.5], dtype=np.float32)


def _buffers() -> dict[str, tuple[np.ndarray, str]]:
    """Name -> (source array with intended shape, safetensors dtype string)."""
    return {
        "wte.weight": (np.arange(6, dtype=np.float32).reshape(2, 3), "F32"),
        "h.0.ln_1.weight": (np.array([1.5, -2.5, 3.0], dtype=np.float32), "F16"),
        "h.0.ln_1.bias": (np.array([1, -1, 7], dtype=np.int64), "I64"),
        "h.0.attn.c_attn.weight": (BF16_VALS, "BF16"),
        "h.0.attn.c_attn.bias": (np.array([42, -42], dtype=np.int32), "I32"),
    }


def _encode(arr: np.ndarray, dtype_str: str) -> bytes:
    if dtype_str == "F32":
        return arr.astype(np.float32).tobytes()
    if dtype_str == "F16":
        return arr.astype(np.float16).tobytes()
    if dtype_str == "BF16":
        return (arr.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes()
    if dtype_str == "I64":
        return arr.astype(np.int64).tobytes()
    if dtype_str == "I32":
        return arr.astype(np.int32).tobytes()
    raise AssertionError(dtype_str)


def _expected() -> dict[str, np.ndarray]:
    """Post-read expectations: BF16/F16/float32 widened to float32, ints kept."""
    out = {}
    for name, (arr, dtype_str) in _buffers().items():
        target = np.float32 if dtype_str in ("F32", "F16", "BF16") else _decode_dtype(dtype_str)
        out[name] = arr.astype(target)
    return out


def _decode_dtype(dtype_str: str) -> np.dtype:
    return {"I64": np.dtype(np.int64), "I32": np.dtype(np.int32)}[dtype_str]


def _write_safetensors(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"__metadata__": {"format": "pt"}}
    data = b""
    offset = 0
    for name, (arr, dtype_str) in _buffers().items():
        raw = _encode(arr, dtype_str)
        meta[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        data += raw
        offset += len(raw)
    header = json.dumps(meta).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(data)


def _assert_values_equal(actual: dict, expected: dict) -> None:
    assert set(actual) == set(expected)
    for name, exp in expected.items():
        assert actual[name].dtype == exp.dtype, (name, actual[name].dtype)
        np.testing.assert_allclose(actual[name], exp, atol=1e-3)


class TestReadWeights:
    def test_all_dtypes_preserved(self, tmp_path):
        st = tmp_path / "model.safetensors"
        _write_safetensors(st)
        _assert_values_equal(SLNCCompiler().read_weights(st), _expected())

    def test_progress_callback(self, tmp_path):
        st = tmp_path / "model.safetensors"
        _write_safetensors(st)
        seen: list[tuple[str, int, int]] = []
        SLNCCompiler().read_weights(st, on_tensor=lambda n, i, t: seen.append((n, i, t)))
        assert len(seen) == 5
        assert [i for _, i, _ in seen] == [1, 2, 3, 4, 5]
        assert all(t == 5 for _, _, t in seen)


class TestCompileRoundtrip:
    def test_roundtrip_keeps_int_arrays(self, tmp_path):
        st = tmp_path / "model.safetensors"
        _write_safetensors(st)
        compiler = SLNCCompiler()
        slnc = tmp_path / "model.slnc"
        compiler.compile_from_dict({"n_layer": 1}, compiler.read_weights(st), str(slnc))
        loaded = SLNCParser(str(slnc)).get_weights_dict_parallel()
        _assert_values_equal(loaded, _expected())


class TestEndToEndLoader:
    def test_load_model_weights_converts_and_loads(self, tmp_path, monkeypatch):
        hf_home = tmp_path / "hf"
        snap = hf_home / "hub" / "models--con-pri-test" / "snapshots" / "abc123"
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "config.json").write_text(
            json.dumps({"architectures": ["GPT2LMHeadModel"], "n_layer": 1})
        )
        _write_safetensors(snap / "model.safetensors")
        monkeypatch.setenv("HF_HOME", str(hf_home))
        weights = load_model_weights("con-pri-test")
        exp = _expected()
        assert set(weights) == set(exp)
        for name, arr in exp.items():
            assert weights[name].dtype == np.float32, name
            np.testing.assert_allclose(weights[name], arr, atol=1e-3)
