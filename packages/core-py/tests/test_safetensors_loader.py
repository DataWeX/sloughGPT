"""Tests for safetensors_loader — torch-free weight loading."""

import json
import struct

import numpy as np
import pytest
from domains.infrastructure.safetensors_loader import (
    load_model_weights,
    load_model_config,
    list_cached_models,
    _get_model_dir,
    _find_safetensors,
    _load_weights_raw,
)
from pathlib import Path

QWEN2_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _is_cached(model_id: str) -> bool:
    return _find_safetensors(_get_model_dir(model_id)) is not None


@pytest.fixture(scope="session")
def qwen_weights():
    """Load Qwen weights once for entire test session."""
    return load_model_weights(QWEN2_ID)


@pytest.fixture(scope="session")
def qwen_config():
    """Load Qwen config once for entire test session."""
    return load_model_config(QWEN2_ID)


class TestGetModelDir:
    """Path resolution tests."""

    def test_namespaced_model(self):
        d = _get_model_dir("Qwen/Qwen2.5-0.5B-Instruct")
        assert d.name == "models--Qwen--Qwen2.5-0.5B-Instruct"

    def test_namespaced_model_unrelated(self):
        d = _get_model_dir("Qwen/Qwen2-0.5B-Instruct")
        assert d.name == "models--Qwen--Qwen2-0.5B-Instruct"


class TestLoadModelConfig:
    """Config loading tests."""

    @pytest.mark.skipif(not _is_cached(QWEN2_ID), reason=f"{QWEN2_ID} not cached locally")
    def test_qwen_config(self, qwen_config):
        assert qwen_config["model_type"] == "qwen2"
        assert qwen_config["vocab_size"] == 151936
        assert qwen_config["num_hidden_layers"] == 24
        assert qwen_config["num_attention_heads"] == 14
        assert qwen_config["hidden_size"] == 896

    def test_unknown_model_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model_config("nonexistent/model-xyz")


class TestLoadModelWeights:
    """Weight loading tests."""

    @pytest.mark.skipif(not _is_cached(QWEN2_ID), reason=f"{QWEN2_ID} not cached locally")
    def test_qwen_weights(self, qwen_weights):
        assert isinstance(qwen_weights, dict)
        assert len(qwen_weights) > 0
        assert all(isinstance(v, np.ndarray) for v in qwen_weights.values())

    @pytest.mark.skipif(not _is_cached(QWEN2_ID), reason=f"{QWEN2_ID} not cached locally")
    def test_qwen_has_embed(self, qwen_weights):
        assert "model.embed_tokens.weight" in qwen_weights
        assert qwen_weights["model.embed_tokens.weight"].shape == (151936, 896)

    @pytest.mark.skipif(not _is_cached(QWEN2_ID), reason=f"{QWEN2_ID} not cached locally")
    def test_qwen_has_attn(self, qwen_weights):
        attn_keys = [k for k in qwen_weights if "attn" in k]
        assert len(attn_keys) > 0

    def test_unknown_model_raises(self):
        with pytest.raises(FileNotFoundError):
            load_model_weights("nonexistent/model-xyz")

    @pytest.mark.skipif(not _is_cached(QWEN2_ID), reason=f"{QWEN2_ID} not cached locally")
    def test_weights_are_float32(self, qwen_weights):
        for k, v in list(qwen_weights.items())[:5]:
            assert v.dtype == np.float32, f"{k} has dtype {v.dtype}"


class TestRawParser:
    """Tests for _load_weights_raw — the built-in torch-free .safetensors parser."""

    @staticmethod
    def _build_safetensors(tensors):
        """Build a .safetensors binary from a list of (name, numpy array, dtype_str)."""
        header = {}
        data = bytearray()
        for name, arr, dtype_str in tensors:
            start = len(data)
            data.extend(arr.tobytes())
            header[name] = {
                "dtype": dtype_str,
                "shape": list(arr.shape),
                "data_offsets": [start, len(data)],
            }
        header["__metadata__"] = {"format": "pt"}
        encoded = json.dumps(header).encode()
        return struct.pack("<Q", len(encoded)) + encoded + bytes(data)

    @staticmethod
    def _bf16_from_f32(f32: np.ndarray) -> np.ndarray:
        """Encode float32 values as BF16 (top 16 bits of the float32 pattern)."""
        u32 = f32.astype(np.float32).view(np.uint32)
        return (u32 >> 16).astype(np.uint16)

    def test_f32_roundtrip(self, tmp_path):
        arr = np.arange(6, dtype=np.float32).reshape(2, 3)
        p = tmp_path / "f32.safetensors"
        p.write_bytes(self._build_safetensors([("w", arr, "F32")]))
        out = _load_weights_raw(p, np.float32)
        assert set(out) == {"w"}
        np.testing.assert_array_equal(out["w"], arr)
        assert out["w"].dtype == np.float32

    def test_f16_roundtrip(self, tmp_path):
        arr = np.array([1.0, -2.0, 3.5], dtype=np.float16)
        p = tmp_path / "f16.safetensors"
        p.write_bytes(self._build_safetensors([("w", arr, "F16")]))
        out = _load_weights_raw(p, np.float32)
        np.testing.assert_allclose(out["w"], arr.astype(np.float32))
        assert out["w"].dtype == np.float32

    def test_bf16_roundtrip(self, tmp_path):
        f32 = np.array([1.0, 0.5, -1.25], dtype=np.float32)
        arr = self._bf16_from_f32(f32)
        p = tmp_path / "bf16.safetensors"
        p.write_bytes(self._build_safetensors([("w", arr, "BF16")]))
        out = _load_weights_raw(p, np.float32)
        np.testing.assert_allclose(out["w"], f32, rtol=0.01)
        assert out["w"].dtype == np.float32

    def test_unknown_dtype_reads_as_f32(self, tmp_path):
        arr = np.array([1, 2, 3, 4], dtype=np.float32)
        p = tmp_path / "unknown.safetensors"
        p.write_bytes(self._build_safetensors([("w", arr, "I8")]))
        out = _load_weights_raw(p, np.float32)
        np.testing.assert_array_equal(out["w"], arr)
        assert out["w"].dtype == np.float32

    def test_multiple_tensors_and_metadata_skipped(self, tmp_path):
        a = np.ones(4, dtype=np.float32)
        b = np.arange(9, dtype=np.float32).reshape(3, 3)
        p = tmp_path / "multi.safetensors"
        p.write_bytes(self._build_safetensors([("a", a, "F32"), ("b", b, "F32")]))
        out = _load_weights_raw(p, np.float32)
        assert set(out) == {"a", "b"}
        np.testing.assert_array_equal(out["a"], a)
        np.testing.assert_array_equal(out["b"], b)

    def test_target_dtype_applied(self, tmp_path):
        arr = np.array([1.5, 2.5], dtype=np.float32)
        p = tmp_path / "cast.safetensors"
        p.write_bytes(self._build_safetensors([("w", arr, "F32")]))
        out = _load_weights_raw(p, np.float64)
        assert out["w"].dtype == np.float64
        np.testing.assert_array_equal(out["w"], arr.astype(np.float64))


class TestListCachedModels:
    """Cached model listing tests."""

    def test_returns_list(self):
        models = list_cached_models()
        assert isinstance(models, list)

    @pytest.mark.skipif(not _is_cached(QWEN2_ID), reason=f"{QWEN2_ID} not cached locally")
    def test_qwen_in_list(self):
        models = list_cached_models()
        ids = [m["id"] for m in models]
        assert "Qwen/Qwen2.5-0.5B-Instruct" in ids

    def test_each_model_has_fields(self):
        models = list_cached_models()
        for m in models:
            assert "id" in m
            assert "path" in m
            assert "size_mb" in m
            assert m["size_mb"] > 0

    def test_sorted_by_id(self):
        models = list_cached_models()
        ids = [m["id"] for m in models]
        assert ids == sorted(ids)
