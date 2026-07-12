"""
Tests for unified ModelLoader.

Tests the common interface for loading both SloNet and HuggingFace models
through a single loader, including detection, verification, and quantization.
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
            model_type="huggingface",
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

    def test_try_load_hf_returns_error_when_import_fails(self, tmp_path):
        loader = ModelLoader(models_dir=tmp_path)
        # HF loading requires torch/transformers which may not be installed
        result = loader._try_load_hf("nonexistent-model", "cpu", False, 8, "symmetric")
        # Should return error result, not crash
        assert not result.success
        assert result.error is not None

    def test_resolve_device_returns_cpu_when_no_torch(self, tmp_path):
        loader = ModelLoader(models_dir=tmp_path)
        with patch.dict("sys.modules", {"torch": None}):
            device = loader._resolve_device()
            assert device == "cpu"

    def test_walk_hf_linears_finds_linear_layers(self, tmp_path):
        """Test walk_hf_linears finds all nn.Linear layers."""
        try:
            import torch
            import torch.nn as nn

            # Create a simple model with Linear layers
            class SimpleModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embed = nn.Embedding(100, 32)
                    self.attn = nn.Linear(32, 32)
                    self.ff = nn.Linear(32, 100)

                def forward(self, x):
                    return self.ff(self.attn(self.embed(x)))

            model = SimpleModel()
            loader = ModelLoader(models_dir=tmp_path)
            layers = loader.walk_hf_linears(model)

            assert "attn" in layers
            assert "ff" in layers
            assert "embed" not in layers  # Embedding is not Linear
            assert layers["attn"].weight.shape == (32, 32)
            assert layers["ff"].weight.shape == (100, 32)
        except ImportError:
            pytest.skip("torch not installed")

    def test_walk_hf_linears_handles_nested_modules(self, tmp_path):
        """Test walk_hf_linears finds Linear layers in nested modules."""
        try:
            import torch
            import torch.nn as nn

            class NestedModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layer1 = nn.ModuleDict({
                        "attn": nn.Linear(16, 16),
                        "ff": nn.Linear(16, 16),
                    })
                    self.layer2 = nn.Linear(16, 16)

                def forward(self, x):
                    return self.layer2(self.layer1["ff"](self.layer1["attn"](x)))

            model = NestedModel()
            loader = ModelLoader(models_dir=tmp_path)
            layers = loader.walk_hf_linears(model)

            assert "layer1.attn" in layers
            assert "layer1.ff" in layers
            assert "layer2" in layers
            assert len(layers) == 3
        except ImportError:
            pytest.skip("torch not installed")


class TestWalkHfLinearsFunction:
    """Test the standalone walk_hf_linears function in quantization module."""

    def test_walk_hf_linears_importable(self):
        from domains.infrastructure.quantization import walk_hf_linears
        assert callable(walk_hf_linears)

    def test_walk_hf_linears_finds_layers(self):
        try:
            import torch
            import torch.nn as nn
            from domains.infrastructure.quantization import walk_hf_linears

            class TestModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.q_proj = nn.Linear(32, 32)
                    self.k_proj = nn.Linear(32, 32)
                    self.v_proj = nn.Linear(32, 32)

                def forward(self, x):
                    return self.v_proj(self.k_proj(self.q_proj(x)))

            model = TestModel()
            layers = walk_hf_linears(model)

            assert "q_proj" in layers
            assert "k_proj" in layers
            assert "v_proj" in layers
            assert len(layers) == 3
        except ImportError:
            pytest.skip("torch not installed")


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
