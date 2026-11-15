"""Tests for the surviving safetensors_loader API — the SLNC orchestrator.

Model-resolution utilities now live in model_resolver.py (covered by
test_model_resolver.py) and .slnc parsing in slnc/parser.py (covered by
test_slnc_parser.py). This module only orchestrates resolve → auto-convert
→ load via SLNC, so only that flow is tested here.
"""

import json
import struct

import numpy as np
import pytest

from domain.infrastructure._internal.model_resolver import find_safetensors, get_model_dir
from domain.infrastructure._internal.safetensors_loader import load_model_weights

QWEN2_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def _is_cached(model_id: str) -> bool:
    return find_safetensors(get_model_dir(model_id)) is not None


@pytest.fixture(scope="session")
def qwen_weights():
    """Load Qwen weights once for entire test session."""
    return load_model_weights(QWEN2_ID)


def _write_fake_safetensors(model_dir, names=("wte.weight", "w2")):
    """Write a minimal .safetensors file into model_dir."""
    model_dir.mkdir(parents=True, exist_ok=True)
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    data = bytearray()
    header = {}
    for name in names:
        start = len(data)
        data.extend(arr.tobytes())
        header[name] = {
            "dtype": "F32",
            "shape": list(arr.shape),
            "data_offsets": [start, len(data)],
        }
    header["__metadata__"] = {"format": "pt"}
    encoded = json.dumps(header).encode()
    (model_dir / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + bytes(data)
    )
    return model_dir


class TestLoadModelWeights:
    """Weight loading via the SLNC orchestrator."""

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


class TestLoadModelWeightsFlow:
    """Orchestration branches of load_model_weights."""

    def test_converts_then_loads(self, tmp_path, monkeypatch):
        model_dir = _write_fake_safetensors(tmp_path / "model")
        monkeypatch.setattr(
            "domain.infrastructure._internal.safetensors_loader.get_model_dir",
            lambda model_id: model_dir,
        )

        converted = {}
        monkeypatch.setattr(
            "domain.infrastructure._internal.safetensors_loader._auto_convert",
            lambda st_path, slnc_path, model_id: converted.update(slnc=str(slnc_path)),
        )
        fake_weights = {
            "w": np.array([1.0, 2.0], dtype=np.float32),
            "b": np.array([0.5], dtype=np.float32),
        }
        monkeypatch.setattr(
            "domain.infrastructure._internal.safetensors_loader._load_from_slnc",
            lambda slnc_path, dtype: fake_weights,
        )

        weights = load_model_weights("fake/model", dtype=np.float32)
        assert weights == fake_weights
        assert converted, "expected auto-convert to run before load"

    def test_skips_conversion_when_slnc_exists(self, tmp_path, monkeypatch):
        model_dir = _write_fake_safetensors(tmp_path / "model")
        (tmp_path / "model" / "model.slnc").write_bytes(b"slnc-ready")
        monkeypatch.setattr(
            "domain.infrastructure._internal.safetensors_loader.get_model_dir",
            lambda model_id: model_dir,
        )
        auto_convert = pytest.fail  # must not be called

        monkeypatch.setattr(
            "domain.infrastructure._internal.safetensors_loader._auto_convert",
            auto_convert,
        )
        monkeypatch.setattr(
            "domain.infrastructure._internal.safetensors_loader._load_from_slnc",
            lambda slnc_path, dtype: {},
        )

        assert load_model_weights("fake/model") == {}

    def test_raises_when_no_safetensors(self, tmp_path, monkeypatch):
        model_dir = tmp_path / "empty"
        model_dir.mkdir()
        monkeypatch.setattr(
            "domain.infrastructure._internal.safetensors_loader.get_model_dir",
            lambda model_id: model_dir,
        )
        with pytest.raises(FileNotFoundError):
            load_model_weights("fake/model")
