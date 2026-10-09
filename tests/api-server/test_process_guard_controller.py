"""
Tests for ``ModelsController`` ProcessGuard wiring — adoption of guards
created on the autoload path, status resolution across load paths, and the
runtime enable/disable toggle.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from controllers.models import ModelsController


def _controller(tmp_path):
    return ModelsController(repo_root=Path(tmp_path))


class _FakeGuard:
    """Minimal stand-in for a started ProcessGuard."""

    def __init__(self, alive=True):
        self.alive = alive
        self.worker_id = "slo-fake"
        self.stopped = False
        self.start_called = False

    def start(self):
        self.start_called = True

    def stop(self):
        self.stopped = True

    def health(self):
        return {
            "alive": self.alive,
            "requests_served": 0,
            "restart_count": 0,
            "max_restarts": 3,
            "exhausted": False,
            "memory_mb": 128,
            "memory_limit_mb": 4096,
            "over_limit": False,
        }


def test_adopt_process_guard_sets_current_model(tmp_path):
    ctrl = _controller(tmp_path)
    guard = _FakeGuard()
    ctrl.adopt_process_guard(guard, "Qwen/Qwen2.5-0.5B-Instruct")
    assert ctrl._process_guard is guard
    assert ctrl._current_model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert ctrl._current_device == "cpu"
    assert ctrl._loaded_at is not None


def test_adopt_process_guard_reads_guard_device(tmp_path):
    """The adopted guard's device must be recorded so /models reports a real
    device instead of null (regression: null device caused pydantic 422)."""
    ctrl = _controller(tmp_path)
    guard = _FakeGuard()
    guard.device = "mps"
    ctrl.adopt_process_guard(guard, "gpt2")
    assert ctrl._current_device == "mps"


def test_adopt_process_guard_defaults_device_to_cpu(tmp_path):
    ctrl = _controller(tmp_path)
    ctrl.adopt_process_guard(_FakeGuard(), "gpt2")
    assert ctrl._current_device == "cpu"


def test_adopt_process_guard_sets_loaded_at_once(tmp_path):
    """Re-adopting must not overwrite an existing loaded_at timestamp."""
    ctrl = _controller(tmp_path)
    ctrl.adopt_process_guard(_FakeGuard(), "gpt2")
    first = ctrl._loaded_at
    ctrl.adopt_process_guard(_FakeGuard(), "gpt2")
    assert ctrl._loaded_at == first


def test_adopt_process_guard_stops_previous(tmp_path):
    ctrl = _controller(tmp_path)
    old = _FakeGuard()
    new = _FakeGuard()
    ctrl.adopt_process_guard(old, "gpt2")
    ctrl.adopt_process_guard(new, "gpt2")
    assert old.stopped is True
    assert ctrl._process_guard is new


def test_adopt_process_guard_keeps_existing_current_model(tmp_path):
    ctrl = _controller(tmp_path)
    ctrl._current_model = "manual-model"
    guard = _FakeGuard()
    ctrl.adopt_process_guard(guard, "autoload-model")
    assert ctrl._current_model == "manual-model"


def test_status_active_when_guard_alive(tmp_path):
    ctrl = _controller(tmp_path)
    ctrl.adopt_process_guard(_FakeGuard(alive=True), "gpt2")
    with patch("config.get_process_guard_enabled", return_value=True):
        status = ctrl.get_process_guard_status()
    assert status["active"] is True
    assert status["model_id"] == "gpt2"
    assert status["health"]["alive"] is True


def test_status_model_id_falls_back_to_registry_default(tmp_path):
    ctrl = _controller(tmp_path)
    registry = MagicMock()
    registry.default_id = "Qwen/Qwen2.5-0.5B-Instruct"
    with (
        patch("config.get_process_guard_enabled", return_value=True),
        patch("domain.infrastructure.model_registry.get_model_registry", return_value=registry),
    ):
        status = ctrl.get_process_guard_status()
    assert status["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_status_model_id_falls_back_to_server_state(tmp_path):
    ctrl = _controller(tmp_path)
    with (
        patch("config.get_process_guard_enabled", return_value=True),
        patch("state.model_type", "native-soul"),
    ):
        status = ctrl.get_process_guard_status()
    assert status["model_id"] == "native-soul"


def test_disable_stops_adopted_guard(tmp_path):
    ctrl = _controller(tmp_path)
    guard = _FakeGuard(alive=True)
    ctrl.adopt_process_guard(guard, "gpt2")
    with (
        patch("config.set_process_guard_enabled") as set_enabled,
        patch("config.get_process_guard_enabled", return_value=False),
    ):
        status = ctrl.set_process_guard_enabled(False)
    set_enabled.assert_called_once_with(False)
    assert guard.stopped is True
    assert status["enabled"] is False


def test_enable_builds_guard_for_autoloaded_model(tmp_path):
    """Enabling must start a guard even when the model was autoloaded via the
    registry (controller ``_current_model`` is None)."""
    ctrl = _controller(tmp_path)
    registry = MagicMock()
    registry.default_id = "Qwen/Qwen2.5-0.5B-Instruct"
    fake_guard = _FakeGuard(alive=True)
    built_for = []

    def _fake_build(self, model_id):
        self._process_guard = fake_guard
        built_for.append(model_id)
        return fake_guard

    with (
        patch("config.set_process_guard_enabled"),
        patch("config.get_process_guard_enabled", return_value=True),
        patch("domain.infrastructure.model_registry.get_model_registry", return_value=registry),
        patch.object(ModelsController, "_build_process_guard", new=_fake_build),
    ):
        status = ctrl.set_process_guard_enabled(True)
    assert built_for == ["Qwen/Qwen2.5-0.5B-Instruct"]
    assert status["active"] is True


def test_build_process_guard_uses_config_not_undefined_var(tmp_path):
    """Regression: ``_build_process_guard`` referenced an undefined ``server_cfg``,
    raising NameError (swallowed) so the guard never started. It must read quant
    settings from ``ServerConfig`` and start the guard when a .slnc exists."""
    ctrl = _controller(tmp_path)
    cfg = MagicMock()
    cfg.quantize_slonet = False
    cfg.quant_bits = 8
    cfg.quant_mode = "symmetric"
    cfg.quant_clip = 0.999
    cfg.process_guard_memory_limit_mb = 0.0
    fake_guard = _FakeGuard(alive=True)
    slnc_dir = Path("/tmp") / "models--Fake--Model" / "model.slnc"
    with (
        patch("config.get_process_guard_enabled", return_value=True),
        patch("config.ServerConfig.from_env", return_value=cfg),
        patch(
            "domain.infrastructure.process_guard.ProcessGuard", return_value=fake_guard
        ) as pg_cls,
        patch("domain.infrastructure.model_resolver.get_model_dir", return_value=slnc_dir.parent),
        patch("os.path.exists", return_value=True),
    ):
        guard = ctrl._build_process_guard("Fake/Model")
    assert guard is fake_guard
    _, kwargs = pg_cls.call_args
    assert kwargs["slnc_path"] == str(slnc_dir)
    assert kwargs["quantize"] is False
    assert kwargs["memory_limit_mb"] is None
    assert fake_guard.start_called is True


def test_build_process_guard_propagates_guard_to_provider_server(tmp_path):
    """Regression: a guard rebuilt at runtime (enable toggle) must replace the
    server's old guard reference. Otherwise generation falls back in-process
    because ``_use_guard()`` sees the stopped guard, leaving
    ``requests_served`` at 0."""
    from domain.models._internal.provider import clear_providers, register_provider

    class _FakeServer:
        def __init__(self):
            self.guard = None

        def set_process_guard(self, guard):
            self.guard = guard

    class _FakeProvider:
        def __init__(self, server):
            self._server = server

        def get_server(self):
            return self._server

    server = _FakeServer()
    register_provider("slonet-native", _FakeProvider(server))

    ctrl = _controller(tmp_path)
    cfg = MagicMock()
    cfg.quantize_slonet = False
    cfg.quant_bits = 8
    cfg.quant_mode = "symmetric"
    cfg.quant_clip = 0.999
    cfg.process_guard_memory_limit_mb = 0.0
    rebuilt_guard = _FakeGuard(alive=True)
    try:
        with (
            patch("config.get_process_guard_enabled", return_value=True),
            patch("config.ServerConfig.from_env", return_value=cfg),
            patch("domain.infrastructure.process_guard.ProcessGuard", return_value=rebuilt_guard),
            patch(
                "domain.infrastructure.model_resolver.get_model_dir",
                return_value=Path("/tmp/models--Fake--Model"),
            ),
            patch("os.path.exists", return_value=True),
        ):
            guard = ctrl._build_process_guard("Fake/Model")
        assert guard is rebuilt_guard
        assert server.guard is rebuilt_guard
    finally:
        clear_providers()


def test_server_config_parses_memory_limit_env():
    """SLO_PROCESS_GUARD_MEMORY_LIMIT_MB must feed ServerConfig so the
    operator can override the auto-sized guard memory limit."""
    import os

    from config import ServerConfig

    with patch.dict(os.environ, {"SLO_PROCESS_GUARD_MEMORY_LIMIT_MB": "15000"}):
        cfg = ServerConfig.from_env()
    assert cfg.process_guard_memory_limit_mb == 15000.0


# ── Why the guard gave up (surfaced in the user-facing load error) ────────
# _build_process_guard returns None for four different reasons; only the
# caller knows that None is fatal on the lazy path, so the reason has to
# travel with it — otherwise the banner says "could not be started" and the
# operator is left scrolling logs (2026-10-08 Qwen load incident).


def test_build_process_guard_records_disabled_reason(tmp_path):
    ctrl = _controller(tmp_path)
    with patch("config.get_process_guard_enabled", return_value=False):
        assert ctrl._build_process_guard("Fake/Model") is None
    assert ctrl._guard_failure == "ProcessGuard is disabled"


def test_build_process_guard_records_missing_slnc_reason(tmp_path):
    ctrl = _controller(tmp_path)
    with (
        patch("config.get_process_guard_enabled", return_value=True),
        patch("domain.infrastructure.model_resolver.get_model_dir", return_value=tmp_path),
        patch("os.path.exists", return_value=False),
    ):
        assert ctrl._build_process_guard("Fake/Model") is None
    assert ctrl._guard_failure is not None
    assert "no compiled model" in ctrl._guard_failure
    assert str(tmp_path) in ctrl._guard_failure


def test_build_process_guard_records_start_failure_reason(tmp_path):
    """A child that dies during start must reach the caller verbatim — that
    string is what the model-load banner shows."""
    ctrl = _controller(tmp_path)
    cfg = MagicMock()
    cfg.quantize_slonet = False
    cfg.quant_bits = 8
    cfg.quant_mode = "symmetric"
    cfg.quant_clip = 0.999
    cfg.process_guard_memory_limit_mb = 0.0
    cause = "child exited after 1.2s: Model load failed: boom"
    with (
        patch("config.get_process_guard_enabled", return_value=True),
        patch("config.ServerConfig.from_env", return_value=cfg),
        patch("domain.infrastructure.model_resolver.get_model_dir", return_value=tmp_path),
        patch("os.path.exists", return_value=True),
        patch(
            "domain.infrastructure.process_guard.ProcessGuard",
            side_effect=RuntimeError(cause),
        ),
    ):
        assert ctrl._build_process_guard("Fake/Model") is None
    assert ctrl._guard_failure == cause


def test_build_process_guard_clears_reason_after_success(tmp_path):
    ctrl = _controller(tmp_path)
    ctrl._guard_failure = "previous failure"
    cfg = MagicMock()
    cfg.quantize_slonet = False
    cfg.quant_bits = 8
    cfg.quant_mode = "symmetric"
    cfg.quant_clip = 0.999
    cfg.process_guard_memory_limit_mb = 0.0
    fake_guard = _FakeGuard(alive=True)
    with (
        patch("config.get_process_guard_enabled", return_value=True),
        patch("config.ServerConfig.from_env", return_value=cfg),
        patch("domain.infrastructure.process_guard.ProcessGuard", return_value=fake_guard),
        patch(
            "domain.infrastructure.model_resolver.get_model_dir",
            return_value=Path("/tmp/models--Fake--Model"),
        ),
        patch("os.path.exists", return_value=True),
    ):
        guard = ctrl._build_process_guard("Fake/Model")
    assert guard is fake_guard
    assert ctrl._guard_failure is None


def test_lazy_load_error_carries_the_guard_reason(tmp_path):
    """End-to-end: the RuntimeError behind the banner must name the cause.

    Reaches the real lazy branch of ``_load_hf_model`` (a compiled .slnc is
    present, guard enabled) and fails the guard at construction, so the only
    difference from production is the injected failure itself.
    """
    ctrl = ModelsController(repo_root=Path(tmp_path))
    model_dir = Path(tmp_path) / "models--Fake--Model"
    model_dir.mkdir(parents=True)
    (model_dir / "model.slnc").write_bytes(b"x")

    cfg = MagicMock()
    cfg.lazy_guard_autoload = True
    cfg.quantize_slonet = False
    cfg.quant_bits = 8
    cfg.quant_mode = "symmetric"
    cfg.quant_clip = 0.999
    cfg.process_guard_memory_limit_mb = 0.0
    cause = "child exited after 1.2s: Model load failed: boom"
    with (
        patch("config.ServerConfig.from_env", return_value=cfg),
        patch("config.get_process_guard_enabled", return_value=True),
        patch("domain.infrastructure.model_resolver.get_model_dir", return_value=model_dir),
        patch("domain.infrastructure.process_guard.ProcessGuard", side_effect=RuntimeError(cause)),
        pytest.raises(RuntimeError) as excinfo,
    ):
        ctrl._load_hf_model("Fake/Model", "cpu")

    message = str(excinfo.value)
    assert "ProcessGuard could not be started" in message
    assert cause in message
