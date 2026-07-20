"""
Tests for SloNet ModelLoader.

Tests the torch-free model loading via .slnc format, including detection,
conversion, verification, and the singleton accessor.
"""

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
        assert loader.models_dir == Path("models")

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
        assert "No .slnc file" in result.error

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
