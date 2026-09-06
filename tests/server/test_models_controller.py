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

    def test_find_gguf_file(self, ctrl):
        ctrl.models_dir.mkdir(parents=True, exist_ok=True)
        (ctrl.models_dir / "test.gguf").touch()
        result = ctrl._find_model_path("test")
        assert result is not None
        assert result.name == "test.gguf"


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

    def test_guard_status_active_when_alive(self, ctrl):
        guard = MagicMock()
        guard.alive = True
        guard.health.return_value = {"alive": True}
        ctrl._process_guard = guard
        result = ctrl.get_process_guard_status()
        assert result["active"] is True
        assert result["health"] == {"alive": True}

    def test_guard_status_health_failure_returns_fallback(self, ctrl):
        guard = MagicMock()
        guard.alive = True
        guard.health.side_effect = RuntimeError("dead")
        ctrl._process_guard = guard
        result = ctrl.get_process_guard_status()
        assert result["health"] == {"alive": False}

    def test_set_guard_enabled_false_stops_guard(self, ctrl):
        guard = MagicMock()
        guard.alive = True
        ctrl._process_guard = guard
        with patch("config.set_process_guard_enabled") as mock_set:
            with patch("config.get_process_guard_enabled", return_value=False):
                result = ctrl.set_process_guard_enabled(False)
        mock_set.assert_called_once_with(False)
        guard.stop.assert_called_once()
        assert ctrl._process_guard is None
        assert result["enabled"] is False

    def test_adopt_process_guard_sets_model(self, ctrl):
        guard = MagicMock()
        guard.device = "cpu"
        guard.worker_id = "w1"
        with patch("domains.models.provider.attach_process_guard_to_provider"):
            ctrl.adopt_process_guard(guard, model_id="gpt2")
        assert ctrl._process_guard is guard
        assert ctrl._current_model == "gpt2"
        assert ctrl._current_device == "cpu"


class TestInferConfig:
    def test_infers_vocab_and_embed(self, ctrl):
        import numpy as np
        state_dict = {"tok_emb.weight": np.zeros((1000, 64), dtype=np.float32)}
        config = ctrl._infer_config(state_dict)
        assert config["vocab_size"] == 1000
        assert config["n_embed"] == 64
        assert config["block_size"] == 64

    def test_infers_layer_count(self, ctrl):
        import numpy as np
        state_dict = {
            "tok_emb.weight": np.zeros((1000, 64), dtype=np.float32),
            "blocks.0.attn_norm.weight": np.zeros((64,)),
            "blocks.1.attn_norm.weight": np.zeros((64,)),
            "blocks.2.attn_norm.weight": np.zeros((64,)),
            "blocks.0.attn.q_proj.weight": np.zeros((64, 64)),
        }
        config = ctrl._infer_config(state_dict)
        assert config["n_embed"] == 64
        assert config["n_layer"] == 3

    def test_empty_state_dict_returns_defaults(self, ctrl):
        config = ctrl._infer_config({})
        assert config["vocab_size"] == 256
        assert config["n_embed"] == 128
        assert config["n_layer"] == 1
        assert config["block_size"] == 128


class TestListAvailableModels:
    def test_lists_gguf_models(self, ctrl):
        ctrl.models_dir.mkdir(parents=True, exist_ok=True)
        (ctrl.models_dir / "a.gguf").write_bytes(b"x" * 2048)
        (ctrl.models_dir / "b.gguf").write_bytes(b"y" * 1024)
        models = ctrl.list_available_models()
        ids = {m["model_id"]: m for m in models}
        assert ids["a"]["type"] == "gguf"
        assert ids["b"]["type"] == "gguf"
        assert round(ids["a"]["size_mb"], 3) == round(2048 / (1024 * 1024), 3)

    def test_no_models_dir_returns_empty(self, ctrl):
        assert ctrl.list_available_models() == []


class TestLoadModel:
    def test_load_success(self, ctrl):
        with patch.object(ctrl, "_load_hf_model", return_value={"model_id": "gpt2"}) as mock_load:
            result = ctrl.load_model("gpt2", device="cpu")
        mock_load.assert_called_once_with("gpt2", "cpu")
        assert result["status"] == "loaded"
        assert result["model_id"] == "gpt2"
        assert result["device"] == "cpu"
        assert ctrl._current_model == "gpt2"

    def test_load_error_returns_error_dict(self, ctrl):
        with patch.object(ctrl, "_load_hf_model", side_effect=RuntimeError("no weights")):
            result = ctrl.load_model("gpt2")
        assert result["status"] == "error"
        assert "no weights" in result["error"]

    def test_load_gguf_path(self, ctrl):
        with patch.object(ctrl, "_load_gguf_model", return_value={"model_id": "m.gguf", "type": "gguf"}) as mock_gguf:
            result = ctrl.load_model("my-model.gguf", device="cpu")
        mock_gguf.assert_called_once_with("my-model.gguf", "cpu")
        assert result["status"] == "loaded"

    def test_load_path_not_a_directory(self, ctrl, tmp_path):
        file_path = tmp_path / "file.txt"
        file_path.write_text("x")
        result = ctrl.load_model_path(str(file_path))
        assert result["status"] == "error"
        assert "Not a directory" in result["error"]


class TestInferenceRecording:
    def test_record_start_sets_flags(self, ctrl):
        ctrl.record_inference_start()
        assert ctrl._is_inferencing is True
        assert ctrl._inference_count == 1
        assert ctrl._last_inference_time is not None

    def test_record_end_accumulates_tokens(self, ctrl):
        ctrl.record_inference_start()
        ctrl.record_inference_end(tokens_generated=42)
        assert ctrl._is_inferencing is False
        assert ctrl._total_tokens_generated == 42
        assert ctrl._inference_count == 1

    def test_stats_reflect_recordings(self, ctrl):
        ctrl.record_inference_start()
        ctrl.record_inference_end(tokens_generated=10)
        stats = ctrl.get_inference_stats()
        assert stats["inference_count"] == 1
        assert stats["total_tokens_generated"] == 10


class TestResolveActiveModelId:
    def test_controller_state_wins(self, ctrl):
        ctrl._current_model = "ctrl-model"
        assert ctrl._resolve_active_model_id() == "ctrl-model"

    def test_registry_default_fallback(self, ctrl):
        registry = MagicMock()
        registry.default_id = "reg-model"
        with patch("domains.infrastructure.model_registry.get_model_registry", return_value=registry):
            assert ctrl._resolve_active_model_id() == "reg-model"

    def test_server_state_fallback(self, ctrl):
        registry = MagicMock()
        registry.default_id = None
        with patch("domains.infrastructure.model_registry.get_model_registry", return_value=registry), \
             patch("state.model_type", "state-model", create=True):
            assert ctrl._resolve_active_model_id() == "state-model"

    def test_none_when_nothing_loaded(self, ctrl):
        registry = MagicMock()
        registry.default_id = None
        with patch("domains.infrastructure.model_registry.get_model_registry", return_value=registry), \
             patch("state.model_type", None, create=True):
            assert ctrl._resolve_active_model_id() is None


class TestListHFModels:
    def test_curated_fallback_when_api_unreachable(self, ctrl):
        with patch("requests.get", side_effect=Exception("offline")):
            models = ctrl.list_hf_models(q="gpt2")
        assert any(m["model_id"] == "gpt2" for m in models)

    def test_query_filters_curated(self, ctrl):
        with patch("requests.get", side_effect=Exception("offline")):
            models = ctrl.list_hf_models(q="qwen")
        assert len(models) > 0
        assert all("qwen" in m["model_id"].lower() for m in models)

    def test_api_returns_models(self, ctrl):
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = [
            {"id": "gpt2", "pipeline_tag": "text-generation", "num_parameters": 124000000, "config": {"vocab_size": 50257}},
        ]
        with patch("requests.get", return_value=fake_resp):
            models = ctrl.list_hf_models(q="gpt2")
        assert models[0]["model_id"] == "gpt2"
        assert models[0]["parameters"] == 124000000
        assert models[0]["vocab_size"] == 50257

    def test_api_skips_non_text_generation(self, ctrl):
        fake_resp = MagicMock()
        fake_resp.status_code = 200
        fake_resp.json.return_value = [
            {"id": "img-model", "pipeline_tag": "image-classification"},
        ]
        with patch("requests.get", return_value=fake_resp):
            models = ctrl.list_hf_models(q="img")
        assert models == []


class TestEstimateParams:
    @pytest.mark.parametrize("mid,expected", [
        ("llama-13b", 13000000000),
        ("falcon-7b", 7000000000),
        ("gemma-3b", 3000000000),
        ("bloom-1b", 1000000000),
        ("qwen-0.5b", 500000000),
        ("tiny-125m", 125000000),
        ("unknown", 0),
    ])
    def test_estimates(self, ctrl, mid, expected):
        assert ctrl._estimate_params(mid) == expected
