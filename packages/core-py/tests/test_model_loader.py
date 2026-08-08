"""
Tests for SloNet ModelLoader.

Tests the torch-free model loading via .slnc format, including detection,
conversion, verification, and the singleton accessor.
"""

import json
import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from domains.infrastructure.model_loader import (
    LoadResult,
    ModelLoader,
    get_model_loader,
    load_model,
)


def _write_safetensors(path: Path, tensors: dict, metadata: dict = None) -> None:
    """Write a minimal valid .safetensors file.

    Args:
        path: Destination path.
        tensors: Mapping of tensor name -> (dtype_str, np.ndarray).
        metadata: Optional dict stored under the __metadata__ header key.
    """
    header = {}
    offset = 0
    blob = b""
    for name, (dtype_str, arr) in tensors.items():
        raw = arr.tobytes()
        header[name] = {
            "dtype": dtype_str,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blob += raw
        offset += len(raw)
    if metadata is not None:
        header["__metadata__"] = metadata
    header_bytes = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(blob)


class TestLoadResult:
    """Test LoadResult dataclass."""

    def test_success_result(self):
        result = LoadResult(
            success=True,
            model_id="gpt2",
            model_type="slonet",
            provider=MagicMock(),
            model=MagicMock(),
        )
        assert result.success
        assert result.model_id == "gpt2"
        assert result.model_type == "slonet"
        assert result.error is None
        assert result.metrics == {}

    def test_failure_result(self):
        result = LoadResult(
            success=False,
            model_id="missing",
            model_type="slonet",
            error="Model not found",
        )
        assert not result.success
        assert result.error == "Model not found"
        assert result.provider is None

    def test_metrics_field(self):
        result = LoadResult(
            success=True,
            model_id="test",
            model_type="slonet",
            metrics={"quantized": True, "quant_bits": 8},
        )
        assert result.metrics["quantized"] is True
        assert result.metrics["quant_bits"] == 8


class TestModelLoader:
    """Test ModelLoader detection and routing."""

    def test_init_default(self):
        loader = ModelLoader()
        from domains.infrastructure.model_loader import _REPO_ROOT
        assert loader.models_dir == _REPO_ROOT / "models"

    def test_init_custom_dir(self, tmp_path):
        loader = ModelLoader(models_dir=tmp_path)
        assert loader.models_dir == tmp_path

    def test_try_load_slnc_returns_none_when_no_slnc(self, tmp_path):
        loader = ModelLoader(models_dir=tmp_path)
        result = loader._try_load_slnc("nonexistent", "cpu", False, 8, "symmetric")
        assert result is None

    def test_load_returns_error_when_no_slnc(self, tmp_path):
        """load() returns error result when no .slnc file found."""
        loader = ModelLoader(models_dir=tmp_path)
        result = loader.load("nonexistent-model")
        assert not result.success
        assert result.model_type == "slonet"
        assert "No .slnc or .soul file" in result.error

    def test_verify_model_passes_for_valid_model(self):
        """_verify_model returns True for a model that forward passes."""
        mock_model = MagicMock()
        mock_model.forward.return_value = np.zeros((1, 10))

        result = LoadResult(
            success=True,
            model_id="test",
            model_type="slonet",
            model=mock_model,
        )
        loader = ModelLoader()
        assert loader._verify_model(result) is True
        assert result.metrics["verified"] is True

    def test_verify_model_fails_for_none_output(self):
        """_verify_model returns False when model returns None."""
        mock_model = MagicMock()
        mock_model.forward.return_value = None

        result = LoadResult(
            success=True,
            model_id="test",
            model_type="slonet",
            model=mock_model,
        )
        loader = ModelLoader()
        assert loader._verify_model(result) is False
        assert result.success is False
        assert "empty output" in result.error

    def test_verify_model_fails_on_exception(self):
        """_verify_model returns False when model raises."""
        mock_model = MagicMock()
        mock_model.forward.side_effect = RuntimeError("boom")

        result = LoadResult(
            success=True,
            model_id="test",
            model_type="slonet",
            model=mock_model,
        )
        loader = ModelLoader()
        assert loader._verify_model(result) is False
        assert result.metrics["verified"] is False


class TestModelLoaderSlncLoad:
    """Test load()/_try_load_slnc() success and failure paths."""

    def test_load_detects_existing_slnc(self, tmp_path, monkeypatch):
        """has_slnc True when model.slnc exists; no tracker.fail on failure."""
        (tmp_path / "model.slnc").write_bytes(b"SLNC")
        monkeypatch.setattr(
            "domains.infrastructure.safetensors_loader._get_model_dir",
            lambda model_id: tmp_path,
        )
        monkeypatch.setattr(
            "domains.inference.slonet_provider.SloNetChatProvider.from_slnc",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("corrupt")),
        )
        loader = ModelLoader(models_dir=tmp_path)
        result = loader.load("gpt2")
        assert not result.success
        assert "corrupt" in result.error

    def test_load_slnc_check_raises(self, tmp_path, monkeypatch):
        """_get_model_dir raising is tolerated (has_slnc=False)."""

        def _raise(model_id):
            raise OSError("boom")

        monkeypatch.setattr("domains.infrastructure.safetensors_loader._get_model_dir", _raise)
        loader = ModelLoader(models_dir=tmp_path)
        result = loader.load("gpt2")
        assert not result.success
        assert "No .slnc or .soul file" in result.error

    def test_load_success_path_with_verify(self, tmp_path, monkeypatch):
        """Full load() success: slnc exists, provider mocked, verify runs."""
        (tmp_path / "model.slnc").write_bytes(b"SLNC")
        monkeypatch.setattr(
            "domains.infrastructure.safetensors_loader._get_model_dir",
            lambda model_id: tmp_path,
        )
        fake_model = MagicMock()
        fake_model.forward.return_value = np.zeros((1, 3))
        fake_provider = MagicMock()
        fake_provider._model = fake_model
        fake_provider._tokenizer = MagicMock()
        monkeypatch.setattr(
            "domains.inference.slonet_provider.SloNetChatProvider.from_slnc",
            lambda *a, **k: fake_provider,
        )
        loader = ModelLoader(models_dir=tmp_path)
        result = loader.load("gpt2", quantize=True, quant_bits=4)
        assert result.success
        assert result.provider is fake_provider
        assert result.model is fake_model
        assert result.tokenizer is fake_provider._tokenizer
        assert result.metrics["quantized"] is True
        assert result.metrics["quant_bits"] == 4
        assert result.metrics["verified"] is True

    def test_try_load_slnc_returns_none_when_convert_fails(self, tmp_path, monkeypatch):
        """Missing slnc + failed conversion returns None."""
        monkeypatch.setattr(
            "domains.infrastructure.safetensors_loader._get_model_dir",
            lambda model_id: tmp_path,
        )
        loader = ModelLoader(models_dir=tmp_path)
        monkeypatch.setattr(loader, "_try_convert_to_slnc", lambda cache_dir, model_id: None)
        assert loader._try_load_slnc("gpt2", "cpu", False, 8, "symmetric") is None

    def test_try_load_slnc_converts_when_missing(self, tmp_path, monkeypatch):
        """Missing slnc + successful conversion loads the new file."""
        monkeypatch.setattr(
            "domains.infrastructure.safetensors_loader._get_model_dir",
            lambda model_id: tmp_path,
        )
        converted = tmp_path / "model.slnc"
        fake_provider = MagicMock()
        fake_provider._model = MagicMock()
        fake_provider._tokenizer = MagicMock()
        monkeypatch.setattr(
            "domains.inference.slonet_provider.SloNetChatProvider.from_slnc",
            lambda *a, **k: fake_provider,
        )
        loader = ModelLoader(models_dir=tmp_path)
        monkeypatch.setattr(
            loader, "_try_convert_to_slnc", lambda cache_dir, model_id: converted
        )
        result = loader._try_load_slnc("gpt2", "cpu", True, 8, "symmetric")
        assert result is not None
        assert result.success
        assert result.metrics["slnc_path"] == str(converted)

    def test_try_load_slnc_returns_error_on_provider_exception(self, tmp_path, monkeypatch):
        """from_slnc raising produces an error LoadResult, not None."""
        (tmp_path / "model.slnc").write_bytes(b"SLNC")
        monkeypatch.setattr(
            "domains.infrastructure.safetensors_loader._get_model_dir",
            lambda model_id: tmp_path,
        )
        monkeypatch.setattr(
            "domains.inference.slonet_provider.SloNetChatProvider.from_slnc",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("corrupt slnc")),
        )
        loader = ModelLoader(models_dir=tmp_path)
        result = loader._try_load_slnc("gpt2", "cpu", False, 8, "symmetric")
        assert not result.success
        assert "corrupt slnc" in result.error


class TestModelLoaderConversion:
    """Test _try_convert_to_slnc (safetensors -> .slnc)."""

    def test_try_convert_returns_none_when_no_safetensors(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "domains.infrastructure.safetensors_loader._find_safetensors",
            lambda cache_dir: None,
        )
        loader = ModelLoader()
        assert loader._try_convert_to_slnc(tmp_path, "gpt2") is None

    def test_try_convert_success_all_dtypes(self, tmp_path, monkeypatch):
        """Full conversion walks a real safetensors file with mixed dtypes."""
        st_path = tmp_path / "model.safetensors"
        _write_safetensors(
            st_path,
            {
                "w_f32": ("F32", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
                "w_f16": ("F16", np.array([1.0, 0.5], dtype=np.float16)),
                "w_bf16": ("BF16", np.array([0x3F80, 0x4000], dtype=np.uint16)),
                "w_i64": ("I64", np.array([1, 2], dtype=np.int64)),
                "w_i32": ("I32", np.array([1, 2], dtype=np.int32)),
                "w_other": ("U8", np.array([1, 2], dtype=np.uint32)),
            },
            metadata={"format": "pt"},
        )
        monkeypatch.setattr(
            "domains.infrastructure.safetensors_loader._find_safetensors",
            lambda cache_dir: st_path,
        )
        monkeypatch.setattr(
            "domains.infrastructure.safetensors_loader.load_model_config",
            lambda model_id: {"model_type": "gpt2"},
        )
        compiled = {}

        def _compile(compiler, config, weights, path):
            compiled["config"] = config
            compiled["weights"] = weights
            compiled["path"] = path

        monkeypatch.setattr(
            "domains.infrastructure.slnc.compiler.SLNCCompiler.compile_from_dict", _compile
        )
        monkeypatch.setattr(
            "domains.infrastructure.model_protector.protect_model",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("protect failed")),
        )
        loader = ModelLoader()
        result = loader._try_convert_to_slnc(tmp_path, "gpt2")
        assert result == tmp_path / "model.slnc"
        assert compiled["config"] == {"model_type": "gpt2"}
        assert compiled["path"] == str(tmp_path / "model.slnc")
        assert compiled["weights"]["w_f32"].dtype == np.float32
        assert compiled["weights"]["w_f16"].dtype == np.float32
        assert compiled["weights"]["w_bf16"].dtype == np.float32
        np.testing.assert_allclose(compiled["weights"]["w_bf16"], [1.0, 2.0], rtol=1e-3)
        assert compiled["weights"]["w_i64"].dtype == np.int64
        assert compiled["weights"]["w_i32"].dtype == np.int32
        assert compiled["weights"]["w_other"].dtype == np.float32

    def test_try_convert_returns_none_on_corrupt_file(self, tmp_path, monkeypatch):
        bad = tmp_path / "model.safetensors"
        bad.write_bytes(b"\x00\x00\x00\x00\x00\x00\x00\x00")
        monkeypatch.setattr(
            "domains.infrastructure.safetensors_loader._find_safetensors",
            lambda cache_dir: bad,
        )
        loader = ModelLoader()
        assert loader._try_convert_to_slnc(tmp_path, "gpt2") is None


class TestModelLoaderSingleton:
    """Test the singleton accessor."""

    def test_get_model_loader_returns_same_instance(self):
        loader1 = get_model_loader()
        loader2 = get_model_loader()
        assert loader1 is loader2

    def test_load_model_convenience_function(self):
        """Test the convenience load_model function."""
        with patch.object(ModelLoader, "load") as mock_load:
            mock_load.return_value = LoadResult(
                success=True,
                model_id="test",
                model_type="slonet",
            )
            result = load_model("test", device="cpu")
            assert result.success
            mock_load.assert_called_once_with(
                model_id="test",
                device="cpu",
                quantize=False,
                quant_bits=8,
                quant_mode="symmetric",
                verify=True,
            )
