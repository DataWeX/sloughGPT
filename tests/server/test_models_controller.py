"""Tests for ModelsController."""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'apps', 'api', 'server'))

from controllers.models import ModelsController


@pytest.fixture
def ctrl(tmp_path):
    return ModelsController(tmp_path)


class TestInit:
    def test_creates_instance(self, ctrl):
        assert ctrl is not None

    def test_no_model_loaded(self, ctrl):
        assert ctrl._current_model is None

    def test_inference_count_zero(self, ctrl):
        assert ctrl._inference_count == 0


class TestFindModelPath:
    def test_find_nonexistent(self, ctrl):
        assert ctrl._find_model_path("no-such-model") is None

    def test_find_pt_file(self, ctrl):
        ctrl.models_dir.mkdir(parents=True, exist_ok=True)
        (ctrl.models_dir / "test.pt").touch()
        result = ctrl._find_model_path("test")
        assert result is not None
        assert result.name == "test.pt"


class TestResolveDevice:
    def test_auto_resolves(self, ctrl):
        result = ctrl._resolve_device("auto")
        assert result in ("cpu", "mps", "cuda")

    def test_explicit_device(self, ctrl):
        result = ctrl._resolve_device("cpu")
        assert result == "cpu"

    def test_none_resolves(self, ctrl):
        result = ctrl._resolve_device(None)
        assert result in ("cpu", "mps", "cuda")

    def test_explicit_cpu_stays_cpu(self, ctrl):
        result = ctrl._resolve_device("cpu")
        assert result == "cpu"

    def test_explicit_cuda_unavailable_falls_back_to_cpu(self, ctrl):
        """Regression: requesting cuda on a GPU-less machine must not report
        cuda as the active device — resolve to cpu instead."""
        with patch("domains.infrastructure.ml_types._cuda_available", return_value=False):
            result = ctrl._resolve_device("cuda")
        assert result == "cpu"

    def test_explicit_cuda_available_stays_cuda(self, ctrl):
        with patch("domains.infrastructure.ml_types._cuda_available", return_value=True):
            result = ctrl._resolve_device("cuda")
        assert result == "cuda"

    def test_explicit_mps_unavailable_falls_back_to_cpu(self, ctrl):
        with patch("domains.infrastructure.ml_types._mps_available", return_value=False):
            result = ctrl._resolve_device("mps")
        assert result == "cpu"

    def test_explicit_mps_available_stays_mps(self, ctrl):
        with patch("domains.infrastructure.ml_types._mps_available", return_value=True):
            result = ctrl._resolve_device("mps")
        assert result == "mps"


class TestGetCurrentModel:
    def test_no_model_loaded(self, ctrl):
        result = ctrl.get_current_model()
        assert result is None

    def test_has_fields(self, ctrl):
        result = ctrl.get_current_model()
        assert result is None  # No model loaded

    def test_model_without_device_returns_none(self, ctrl):
        """Regression: a model id without a resolved device must not be
        reported as loaded — the old code emitted device=None and the router's
        ModelInfo schema rejected it with a 422 on /models."""
        ctrl._current_model = "gpt2"
        ctrl._current_device = None
        assert ctrl.get_current_model() is None

    def test_device_without_model_returns_none(self, ctrl):
        ctrl._current_device = "cpu"
        assert ctrl.get_current_model() is None

    def test_model_and_device_returns_dict(self, ctrl):
        ctrl._current_model = "gpt2"
        ctrl._current_device = "cpu"
        result = ctrl.get_current_model()
        assert result is not None
        assert result["model_id"] == "gpt2"
        assert result["device"] == "cpu"
        assert result["status"] == "loaded"

    def test_missing_loaded_at_returns_none_field(self, ctrl):
        ctrl._current_model = "gpt2"
        ctrl._current_device = "cpu"
        result = ctrl.get_current_model()
        assert result["loaded_at"] is None


class TestGetInferenceStats:
    def test_stats_has_fields(self, ctrl):
        result = ctrl.get_inference_stats()
        assert "inference_count" in result
        assert result["inference_count"] == 0

    def test_stats_has_tokens(self, ctrl):
        result = ctrl.get_inference_stats()
        assert "total_tokens_generated" in result


class TestProcessGuard:
    def test_get_process_guard_status_has_fields(self, ctrl):
        result = ctrl.get_process_guard_status()
        assert "enabled" in result
