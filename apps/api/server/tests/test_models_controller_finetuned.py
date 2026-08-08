"""
Tests for ``ModelsController.load_model_path`` — loading a local fine-tuned
model directory into chat via SloNet (compile-to-.slnc then register provider).

The heavy steps (SLNC compilation, provider registration) are mocked; this suite
verifies the orchestration contract: directory validation, conditional compile,
base-model-id resolution, and provider wiring.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import state
from controllers.models import ModelsController

CFG = {"_name_or_path": "gpt2", "model_type": "gpt2", "n_layer": 1}


@pytest.fixture(autouse=True)
def _disable_process_guard(monkeypatch):
    """Keep unit tests hermetic: no real subprocess workers are spawned for the
    fake .slnc files. The runtime toggle is patched off because
    ``_build_process_guard_for_path`` consults it (not ``cfg.enable_process_guard``)."""
    monkeypatch.setattr("config.get_process_guard_enabled", lambda: False)


class _FakeCompiler:
    """Deterministic stand-in for SLNCCompiler that records its calls."""

    calls = []

    def __init__(self):
        pass

    def compile_from_directory(self, model_dir, output, config_path=None):
        _FakeCompiler.calls.append((model_dir, output))
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).touch()
        return str(output)


def _make_finetuned_dir(tmp_path):
    d = tmp_path / "gpt2__dataset_1"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(CFG))
    (d / "tokenizer.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\x00" * 8)
    return d


def _controller(tmp_path):
    return ModelsController(repo_root=Path(tmp_path))


def test_load_model_path_non_directory_returns_error(tmp_path):
    ctrl = _controller(tmp_path)
    result = ctrl.load_model_path(str(tmp_path / "missing-dir"), "cpu")
    assert result["status"] == "error"
    assert "Not a directory" in result["error"]


def test_load_model_path_compiles_and_registers(tmp_path):
    d = _make_finetuned_dir(tmp_path)
    cfg_model = MagicMock()
    cfg_model.quantize_slonet = False
    cfg_model.quant_bits = 8
    cfg_model.quant_mode = "symmetric"
    cfg_model.enable_process_guard = False
    ctrl = _controller(tmp_path)
    _FakeCompiler.calls.clear()
    with patch("domains.infrastructure.slnc.compiler.SLNCCompiler", _FakeCompiler), \
         patch("config.ServerConfig.from_env", return_value=cfg_model), \
         patch("domains.models.provider.setup_providers") as setup:
        result = ctrl.load_model_path(str(d), "cpu")
    assert result["status"] == "loaded"
    assert result["model_id"] == "gpt2"
    assert result["model_path"] == str(d)
    assert result["slnc_path"] == str(d / "model.slnc")
    assert (d / "model.slnc").exists()
    assert _FakeCompiler.calls == [(str(d), str(d / "model.slnc"))]
    _, kwargs = setup.call_args
    assert kwargs["slonet_hf_id"] == "gpt2"
    assert kwargs["slonet_path"] == str(d / "model.slnc")


def test_load_model_path_reuses_existing_slnc(tmp_path):
    d = _make_finetuned_dir(tmp_path)
    (d / "model.slnc").write_bytes(b"\x00" * 8)
    cfg_model = MagicMock()
    cfg_model.quantize_slonet = False
    cfg_model.quant_bits = 8
    cfg_model.quant_mode = "symmetric"
    cfg_model.enable_process_guard = False
    ctrl = _controller(tmp_path)
    _FakeCompiler.calls.clear()
    with patch("domains.infrastructure.slnc.compiler.SLNCCompiler", _FakeCompiler), \
         patch("config.ServerConfig.from_env", return_value=cfg_model), \
         patch("domains.models.provider.setup_providers") as setup:
        result = ctrl.load_model_path(str(d), "cpu")
    assert result["status"] == "loaded"
    assert _FakeCompiler.calls == []  # existing .slnc skipped compilation
    _, kwargs = setup.call_args
    assert kwargs["slonet_path"] == str(d / "model.slnc")


def test_load_model_path_falls_back_to_dir_name_without_base(tmp_path):
    d = tmp_path / "no-name-or-path"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "tokenizer.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\x00")
    cfg_model = MagicMock()
    cfg_model.quantize_slonet = False
    cfg_model.quant_bits = 8
    cfg_model.quant_mode = "symmetric"
    cfg_model.enable_process_guard = False
    ctrl = _controller(tmp_path)
    with patch("domains.infrastructure.slnc.compiler.SLNCCompiler", _FakeCompiler), \
         patch("config.ServerConfig.from_env", return_value=cfg_model), \
         patch("domains.models.provider.setup_providers") as setup:
        result = ctrl.load_model_path(str(d), "cpu")
    assert result["status"] == "loaded"
    _, kwargs = setup.call_args
    assert kwargs["slonet_hf_id"] == "no-name-or-path"


def test_load_model_path_unregisters_stale_registry_default(tmp_path):
    """Loading a fine-tuned model must drop any previously registered HF model
    from the registry so /health (registry-first) reflects the new model."""
    d = _make_finetuned_dir(tmp_path)
    cfg_model = MagicMock()
    cfg_model.quantize_slonet = False
    cfg_model.quant_bits = 8
    cfg_model.quant_mode = "symmetric"
    cfg_model.enable_process_guard = False
    ctrl = _controller(tmp_path)
    registry = MagicMock()
    registry.default_id = "Qwen/Qwen2.5-0.5B-Instruct"
    with patch("domains.infrastructure.slnc.compiler.SLNCCompiler", _FakeCompiler), \
         patch("config.ServerConfig.from_env", return_value=cfg_model), \
         patch("domains.models.provider.setup_providers"), \
         patch("domains.infrastructure.model_registry.get_model_registry", return_value=registry):
        result = ctrl.load_model_path(str(d), "cpu")
    assert result["status"] == "loaded"
    assert result["model_id"] == "gpt2"
    registry.unregister.assert_called_once_with("Qwen/Qwen2.5-0.5B-Instruct")


def test_load_model_path_keeps_registry_when_default_matches(tmp_path):
    """Loading the model already registered in the registry should not
    unregister it (avoid self-destructive unregister)."""
    d = _make_finetuned_dir(tmp_path)
    cfg_model = MagicMock()
    cfg_model.quantize_slonet = False
    cfg_model.quant_bits = 8
    cfg_model.quant_mode = "symmetric"
    cfg_model.enable_process_guard = False
    ctrl = _controller(tmp_path)
    registry = MagicMock()
    registry.default_id = "gpt2"
    with patch("domains.infrastructure.slnc.compiler.SLNCCompiler", _FakeCompiler), \
         patch("config.ServerConfig.from_env", return_value=cfg_model), \
         patch("domains.models.provider.setup_providers"), \
         patch("domains.infrastructure.model_registry.get_model_registry", return_value=registry):
        result = ctrl.load_model_path(str(d), "cpu")
    assert result["status"] == "loaded"
    registry.unregister.assert_not_called()


def test_unload_model_exists_and_clears_state(tmp_path):
    ctrl = _controller(tmp_path)
    ctrl._current_model = "gpt2"
    ctrl._current_device = "cpu"
    ctrl._loaded_at = __import__("datetime").datetime.now()
    with patch("domains.infrastructure.model_registry.get_model_registry") as registry:
        result = ctrl.unload_model()
    assert result["status"] == "unloaded"
    assert result["model_id"] == "gpt2"
    assert ctrl._current_model is None
    assert ctrl._current_device is None
    assert ctrl._loaded_at is None
    registry.return_value.unregister.assert_called_once_with("gpt2")


def test_unload_model_uses_registry_default_when_controller_never_loaded(tmp_path):
    """Autoload registers the model directly with the registry, bypassing the
    controller. Unload must resolve the active model from ``registry.default_id``,
    unregister it, clear providers, and reset server state."""
    ctrl = _controller(tmp_path)
    registry = MagicMock()
    registry.default_id = "Qwen/Qwen2.5-0.5B-Instruct"
    with patch("domains.infrastructure.model_registry.get_model_registry", return_value=registry), \
         patch("domains.models.provider.clear_providers") as clear, \
         patch("state.model", new=MagicMock()), \
         patch("state.tokenizer", new=MagicMock()):
        result = ctrl.unload_model()
    assert result["status"] == "unloaded"
    assert result["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
    registry.unregister.assert_called_once_with("Qwen/Qwen2.5-0.5B-Instruct")
    clear.assert_called_once()
    assert ctrl._current_model is None
    assert ctrl._current_device is None
    assert ctrl._loaded_at is None


def test_load_model_path_provider_failure_returns_error(tmp_path):
    d = _make_finetuned_dir(tmp_path)
    cfg_model = MagicMock()
    cfg_model.quantize_slonet = False
    cfg_model.quant_bits = 8
    cfg_model.quant_mode = "symmetric"
    cfg_model.enable_process_guard = False
    ctrl = _controller(tmp_path)
    with patch("domains.infrastructure.slnc.compiler.SLNCCompiler", _FakeCompiler), \
         patch("config.ServerConfig.from_env", return_value=cfg_model), \
         patch("domains.models.provider.setup_providers", side_effect=RuntimeError("boom")):
        result = ctrl.load_model_path(str(d), "cpu")
    assert result["status"] == "error"
    assert "boom" in result["error"]


def test_load_hf_model_publishes_to_state_and_server_state():
    """After setup_providers, the provider must be published to BOTH the
    ``state`` module (inference guards) and the core ``ServerState`` singleton
    (health_score). Regression for Bugs C/D — the two model slots diverged."""
    ctrl = ModelsController(repo_root=Path("/tmp"))
    provider = MagicMock()
    provider._model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    cfg = MagicMock()
    cfg.quantize_slonet = False
    cfg.quant_bits = 8
    cfg.quant_mode = "symmetric"
    with patch("config.ServerConfig.from_env", return_value=cfg), \
         patch.object(ModelsController, "_build_process_guard", return_value=None), \
         patch("domains.models.provider.setup_providers"), \
         patch("domains.models.provider.get_provider", return_value=provider), \
         patch("domains.infrastructure.server_state.get_server_state") as core, \
         patch("state.model", new=MagicMock()), \
         patch("state.provider", new=MagicMock()), \
         patch("state.model_type", new=MagicMock()):
        result = ctrl._load_hf_model("Qwen/Qwen2.5-0.5B-Instruct", "cpu")
        assert result["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"
        assert state.model is provider
        assert state.provider is provider
        assert state.model_type == "Qwen/Qwen2.5-0.5B-Instruct"
        core.return_value.model.set.assert_called_once_with(provider)
        core.return_value.model_type.set.assert_called_once_with("Qwen/Qwen2.5-0.5B-Instruct")


def test_load_hf_model_provider_failure_does_not_publish():
    """If setup_providers raises, the provider is never published — no partial
    state where inference guards pass but no model is actually loaded."""
    ctrl = ModelsController(repo_root=Path("/tmp"))
    cfg = MagicMock()
    cfg.quantize_slonet = False
    cfg.quant_bits = 8
    cfg.quant_mode = "symmetric"
    with patch("config.ServerConfig.from_env", return_value=cfg), \
         patch.object(ModelsController, "_build_process_guard", return_value=None), \
         patch("domains.models.provider.setup_providers", side_effect=RuntimeError("boom")), \
         patch("domains.infrastructure.server_state.get_server_state") as core, \
         patch("state.model_type", new=MagicMock()):
        try:
            ctrl._load_hf_model("gpt2", "cpu")
            raise AssertionError("expected _load_hf_model to raise")
        except RuntimeError:
            pass
        core.return_value.model.set.assert_not_called()
        core.return_value.model_type.set.assert_not_called()


def test_load_hf_model_stale_provider_not_published():
    """setup_providers() logs-and-continues when a model fails to load (e.g.
    missing .slnc), leaving a STALE provider registered from a previous load.
    A provider whose ``_model_id`` does not match the requested model must not
    be published to server_state — otherwise health reports the model loaded
    while inference serves the wrong (or no) weights."""
    ctrl = ModelsController(repo_root=Path("/tmp"))
    stale = MagicMock()
    stale._model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    cfg = MagicMock()
    cfg.quantize_slonet = False
    cfg.quant_bits = 8
    cfg.quant_mode = "symmetric"
    with patch("config.ServerConfig.from_env", return_value=cfg), \
         patch.object(ModelsController, "_build_process_guard", return_value=None), \
         patch("domains.models.provider.setup_providers"), \
         patch("domains.models.provider.get_provider", return_value=stale), \
         patch("domains.infrastructure.server_state.get_server_state") as core, \
         patch("state.model", new=MagicMock()), \
         patch("state.provider", new=MagicMock()), \
         patch("state.model_type", new=MagicMock()):
        try:
            ctrl._load_hf_model("gpt2", "cpu")
            raise AssertionError("expected _load_hf_model to raise")
        except RuntimeError as e:
            assert "gpt2" in str(e)
            assert "Qwen/Qwen2.5-0.5B-Instruct" in str(e)
        assert state.model is not stale
        assert state.model_type != "gpt2"
        core.return_value.model.set.assert_not_called()
        core.return_value.model_type.set.assert_not_called()