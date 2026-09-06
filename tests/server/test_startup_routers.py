"""
Regression tests for the startup router-registration race fix.

The intermittent startup failure (``OSError: [Errno 9] Bad file descriptor``
or a partial ``ImportError``) happened when two threads performed first-time
module imports concurrently — the background model-load thread and the main
thread's router registration. Two layers guard against it:

1. ``_preload_model_imports`` imports the model-load thread's full dependency
   graph up front (main thread, before the background task is created) so its
   later imports are ``sys.modules`` cache hits and perform zero file reads.
2. ``_phase6_routers`` retries the whole registration pass once when the
   first attempt fails with a transient import error (EBADF errno 9 or a
   partial ``ImportError``).

The tests here pin both behaviors: every module in the prewarm list must be
importable, and the retry must recover a transient failure to a fully
registered state instead of degrading to the 5 pre-registered routes.
"""

import asyncio
import os
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import APIRouter, FastAPI

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = REPO_ROOT / "apps/api/server"
CORE_DIR = REPO_ROOT / "packages/core-py"
for _p in (SERVER_DIR, CORE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from apps.api.server.config import ServerConfig  # noqa: E402
from apps.api.server.infrastructure.startup import (  # noqa: E402
    _PREWARM_MODEL_LOAD_IMPORTS,
    _preload_model_imports,
    _build_guard_for_model,
    StartupOrchestrator,
)


def test_preload_populates_sys_modules_for_all_declared_modules():
    """Every module in the prewarm list must actually import.

    A module that is unreachable (renamed, moved, or deleted) would be
    silently swallowed by ``_preload_model_imports`` and re-open the
    concurrent-import race, so this asserts the full graph lands in
    ``sys.modules``.
    """
    _preload_model_imports()
    missing = [m for m in _PREWARM_MODEL_LOAD_IMPORTS if m not in sys.modules]
    assert not missing, f"prewarm failed to import: {missing}"


def test_preload_is_idempotent():
    """Running the prewarm twice must not raise and must be a no-op."""
    _preload_model_imports()
    _preload_model_imports()


def _make_orchestrator():
    return StartupOrchestrator(FastAPI(), ServerConfig())


def _stub_routers(get_all_routers):
    """Fake ``routers`` module whose ``get_all_routers`` is controllable.

    ``_phase6_routers`` does ``from routers import get_all_routers`` inside
    the function body; inserting this fake into ``sys.modules`` lets the
    test drive the race it is guarding against.
    """
    mod = types.ModuleType("routers")
    mod.get_all_routers = get_all_routers
    return mod


def _stub_training_router():
    """Empty ``training.router`` so the optional include is a no-op."""
    mod = types.ModuleType("training.router")
    mod.router = APIRouter()
    return mod


def test_phase6_routers_retries_once_on_transient_ebadf():
    """An ``OSError`` errno 9 on the first pass must be retried and recover."""
    get_all = MagicMock(side_effect=[OSError(9, "Bad file descriptor"), []])
    orch = _make_orchestrator()
    with patch.dict(
        sys.modules,
        {"routers": _stub_routers(get_all), "training.router": _stub_training_router()},
    ), patch("faulthandler.dump_traceback"), patch("traceback.print_exc"):
        asyncio.run(orch._phase6_routers())
    assert get_all.call_count == 2
    assert orch._routers_registered is True


def test_phase6_routers_retries_once_on_transient_partial_import():
    """A partial ``ImportError`` on the first pass must also be retried."""
    get_all = MagicMock(side_effect=[ImportError("cannot import name 'ProcessGuard'"), []])
    orch = _make_orchestrator()
    with patch.dict(
        sys.modules,
        {"routers": _stub_routers(get_all), "training.router": _stub_training_router()},
    ), patch("faulthandler.dump_traceback"), patch("traceback.print_exc"):
        asyncio.run(orch._phase6_routers())
    assert get_all.call_count == 2
    assert orch._routers_registered is True


def test_phase6_routers_raises_on_nontransient_error_without_retry():
    """A non-transient failure must not be masked by a retry."""
    get_all = MagicMock(side_effect=RuntimeError("boom"))
    orch = _make_orchestrator()
    with patch.dict(
        sys.modules,
        {"routers": _stub_routers(get_all), "training.router": _stub_training_router()},
    ), patch("faulthandler.dump_traceback"), patch("traceback.print_exc"):
        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(orch._phase6_routers())
    assert get_all.call_count == 1
    assert orch._routers_registered is False


def test_phase6_routers_skips_already_registered_prefixes():
    """Routers whose prefix is already on the app must be skipped."""
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    included = APIRouter(prefix="/foo")

    @included.get("/bar")
    def bar():
        return {}

    skipped = types.SimpleNamespace(prefix="/health")

    get_all = MagicMock(return_value=[skipped, included])
    orch = StartupOrchestrator(app, ServerConfig())
    with patch.dict(
        sys.modules,
        {"routers": _stub_routers(get_all), "training.router": _stub_training_router()},
    ):
        asyncio.run(orch._phase6_routers())
    assert orch._routers_registered is True
    assert get_all.call_count == 1
    paths = app.openapi()["paths"]
    assert "/foo/bar" in paths
    assert not any(p.startswith("/health") for p in paths if p != "/health")


# ── Phase 2: model load ──────────────────────────────────────────────────────


class TestPhase2ModelLoad:
    """_phase2_model_load — autoload disabled / lazy-guard routing."""

    async def _run_phase_with(self, orch, cfg):
        await orch._phase2_model_load()
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        if pending:
            await asyncio.gather(*pending)

    def test_autoload_disabled_returns_without_task(self):
        cfg = ServerConfig(autoload_model="false")
        orch = StartupOrchestrator(FastAPI(), cfg)
        orch._model_load_task = None
        with patch("apps.api.server.infrastructure.startup._preload_model_imports") as mock_preload:
            asyncio.run(self._run_phase_with(orch, cfg))
        mock_preload.assert_called_once()
        assert orch._model_load_task is None

    def test_autoload_disabled_variants(self):
        for disabled in ("0", "none", "no", "off", "disable", "None", "FALSE"):
            cfg = ServerConfig(autoload_model=disabled)
            orch = StartupOrchestrator(FastAPI(), cfg)
            with patch("apps.api.server.infrastructure.startup._preload_model_imports"):
                asyncio.run(self._run_phase_with(orch, cfg))
            assert orch._model_load_task is None, disabled

    def test_lazy_guard_autoload_path(self):
        cfg = ServerConfig(autoload_model="gpt2")
        orch = StartupOrchestrator(FastAPI(), cfg)
        with patch("apps.api.server.infrastructure.startup._preload_model_imports") as mock_preload, \
             patch("apps.api.server.infrastructure.startup._try_lazy_guard_autoload",
                   return_value=True) as mock_lazy, \
             patch("apps.api.server.infrastructure.startup._sync_soul_traits") as mock_sync:
            asyncio.run(self._run_phase_with(orch, cfg))
        mock_preload.assert_called_once()
        mock_lazy.assert_called_once_with(cfg)
        mock_sync.assert_called_once()
        assert orch._model_load_task is None


# ── Phase 5: model registry ──────────────────────────────────────────────────


class TestPhase5ModelRegistry:
    """_phase5_model_registry — success + failure isolation."""

    def test_registry_initialized(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        with patch("domains.infrastructure.model_registry.get_model_registry",
                   return_value=object()) as mock_get:
            asyncio.run(orch._phase5_model_registry())
        mock_get.assert_called_once()
        assert orch._registry is not None

    def test_registry_failure_does_not_raise(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        with patch("domains.infrastructure.model_registry.get_model_registry",
                   side_effect=RuntimeError("boom")):
            asyncio.run(orch._phase5_model_registry())
        assert orch._registry is None


# ── Phase: task queue ────────────────────────────────────────────────────────


class TestPhaseTaskQueue:
    """_phase_task_queue — queue + training/memory handler registration."""

    def test_initializes_queue_and_handlers(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        q = object()
        with patch("domains.infrastructure.task_queue.get_task_queue", return_value=q) as mock_q, \
             patch("domains.infrastructure.training_queue.register_training_handlers") as mock_reg, \
             patch("domains.memory.register_memory_handlers") as mock_mem, \
             patch("domains.memory.maintenance.start_memory_maintenance") as mock_maint:
            asyncio.run(orch._phase_task_queue())
        mock_q.assert_called_once()
        mock_reg.assert_called_once()
        mock_mem.assert_called_once()
        mock_maint.assert_called_once()
        assert orch._task_queue is q

    def test_queue_failure_swallowed(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        with patch("domains.infrastructure.task_queue.get_task_queue",
                   side_effect=RuntimeError("boom")), \
             patch("domains.infrastructure.training_queue.register_training_handlers") as mock_reg, \
             patch("domains.memory.register_memory_handlers") as mock_mem, \
             patch("domains.memory.maintenance.start_memory_maintenance") as mock_maint:
            asyncio.run(orch._phase_task_queue())
        mock_reg.assert_called_once()
        mock_mem.assert_called_once()


# ── Phase: config + ready ────────────────────────────────────────────────────


class TestPhaseConfigReady:
    """_phase_config and _phase_ready."""

    def test_config_validated(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        with patch("domains.infrastructure.config.get_config") as mock_cfg, \
             patch("domains.infrastructure.resource_manager.get_resource_manager") as mock_rm:
            rm = mock_rm.return_value
            rm.apply_blas_env.return_value = None
            rm.apply_compute_limits.return_value = None
            rm.apply_environment.return_value = None
            rm.mode = "cpu"
            rm.summary.return_value = ""
            asyncio.run(orch._phase_config())
        mock_cfg.assert_called_once()
        assert rm.apply_blas_env.call_count == 1
        assert rm.apply_compute_limits.call_count == 1
        assert rm.apply_environment.call_count == 1

    def test_config_failure_swallowed(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        with patch("domains.infrastructure.config.get_config",
                   side_effect=RuntimeError("boom")), \
             patch("domains.infrastructure.resource_manager.get_resource_manager",
                   side_effect=RuntimeError("boom2")):
            asyncio.run(orch._phase_config())

    def test_ready_updates_phase(self):
        from startup_progress import STARTUP_PHASE
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        asyncio.run(orch._phase_ready())
        assert STARTUP_PHASE["phase"] == "ready"


# ── Shutdown hooks ───────────────────────────────────────────────────────────


class TestShutdownHooks:
    """Each shutdown hook swallows errors and runs cleanly."""

    def test_shutdown_task_queue_with_instance(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        q = MagicMock()
        q.stop = AsyncMock()
        orch._task_queue = q
        asyncio.run(orch._shutdown_task_queue())
        q.stop.assert_awaited_once()

    def test_shutdown_task_queue_none(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        orch._task_queue = None
        asyncio.run(orch._shutdown_task_queue())

    def test_shutdown_jobs_marks_running_crashed(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        with patch("training.job_store.get_job_store") as mock_store:
            store = mock_store.return_value
            store.list.return_value = [{"id": "j1"}, {"id": "j2"}]
            asyncio.run(orch._shutdown_jobs())
        store.list.assert_called_once_with(status="running")
        assert store.mark_crashed.call_count == 2

    def test_shutdown_jobs_failure_swallowed(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        with patch("training.job_store.get_job_store",
                   side_effect=RuntimeError("boom")):
            asyncio.run(orch._shutdown_jobs())

    def test_shutdown_wandb_cancels_task(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())

        async def _scenario():
            task = asyncio.ensure_future(asyncio.sleep(999))
            orch._wandb_task = task
            await orch._shutdown_wandb()
            return task

        task = asyncio.run(_scenario())
        assert task.cancelled()

    def test_shutdown_registry_resets_metrics(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        with patch("domains.infrastructure.model_registry.get_model_registry") as mock_get:
            asyncio.run(orch._shutdown_registry())
        mock_get.return_value.reset_metrics.assert_called_once()

    def test_shutdown_executor_shuts_down_instance(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        with patch("domains.training.executor._instance") as mock_inst:
            asyncio.run(orch._shutdown_executor())
        mock_inst.shutdown.assert_called_once_with(wait=True)

    def test_shutdown_executor_no_instance(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        with patch("domains.training.executor._instance", None):
            asyncio.run(orch._shutdown_executor())


# ── is_model_loaded / _build_guard_for_model ─────────────────────────────────


class TestModelLoadedGuard:
    """_is_model_loaded gate and _build_guard_for_model."""

    def test_is_model_loaded_true(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        with patch("state.model", object()):
            assert orch._is_model_loaded() is True

    def test_is_model_loaded_false(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        with patch("state.model", None):
            assert orch._is_model_loaded() is False

    def test_build_guard_disabled_returns_none(self):
        cfg = ServerConfig()
        with patch("config.get_process_guard_enabled", return_value=False):
            assert _build_guard_for_model(cfg, "gpt2") is None

    def test_build_guard_missing_slnc_returns_none(self):
        cfg = ServerConfig()
        with patch("config.get_process_guard_enabled", return_value=True), \
             patch("domains.infrastructure.model_resolver.get_model_dir",
                   return_value=Path("/nonexistent-dir")) as mock_dir, \
             patch("os.path.exists", return_value=False):
            assert _build_guard_for_model(cfg, "gpt2") is None
        mock_dir.assert_called_with("gpt2")

    def test_build_guard_returns_started_guard(self):
        cfg = ServerConfig()
        guard = MagicMock()
        with patch("config.get_process_guard_enabled", return_value=True), \
             patch("domains.infrastructure.process_guard.ProcessGuard",
                   return_value=guard) as mock_pg, \
             patch("domains.infrastructure.model_resolver.get_model_dir",
                   return_value=Path("/fake/slnc-dir")), \
             patch("os.path.exists", return_value=True), \
             patch("domains.infrastructure.process_guard.resolve_memory_limit_mb",
                   return_value=512.0), \
             patch("controllers.models.get_models_controller") as mock_ctrl:
            result = _build_guard_for_model(cfg, "gpt2")
        mock_pg.assert_called_once()
        guard.start.assert_called_once()
        mock_ctrl.return_value.adopt_process_guard.assert_called_once_with(guard, "gpt2")
        assert result is guard

    def test_build_guard_failure_returns_none(self):
        cfg = ServerConfig()
        with patch("config.get_process_guard_enabled", return_value=True), \
             patch("domains.infrastructure.process_guard.ProcessGuard",
                   side_effect=RuntimeError("boom")):
            assert _build_guard_for_model(cfg, "gpt2") is None


# ── _init_lifecycle ───────────────────────────────────────────────────────────


class TestInitLifecycle:
    """_init_lifecycle — hook/gate registration + profile resolution."""

    def test_registers_hooks_and_gates(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig(), profile="full")
        lifecycle = MagicMock()
        lifecycle._startup_hooks = []
        lifecycle._shutdown_hooks = []
        with patch("domains.infrastructure.event_bus.EventBus") as mock_bus, \
             patch("domains.infrastructure.lifecycle.get_lifecycle_manager",
                   return_value=lifecycle):
            asyncio.run(orch._init_lifecycle())
        assert orch._lifecycle is lifecycle
        assert lifecycle.register_startup_hook.call_count == 7
        assert lifecycle.register_shutdown_hook.call_count == 7
        assert lifecycle.register_gate.call_count == 2
        mock_bus.assert_called_once()
        assert orch._profile_enum is not None

    def test_skips_when_already_initialized(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        orch._lifecycle = object()
        with patch("domains.infrastructure.lifecycle.get_lifecycle_manager") as mock_get:
            asyncio.run(orch._init_lifecycle())
        mock_get.assert_not_called()

    def test_invalid_profile_falls_back_to_full(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig(), profile="bogus")
        with patch("domains.infrastructure.event_bus.EventBus"), \
             patch("domains.infrastructure.lifecycle.get_lifecycle_manager"):
            asyncio.run(orch._init_lifecycle())
        assert orch._profile_enum.value == "full"


# ── run() orchestration ───────────────────────────────────────────────────────


class TestRun:
    """run() — lifecycle start + fallback direct path."""

    def test_runs_lifecycle_and_ready(self):
        from domains.infrastructure.lifecycle import StartupProfile
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        orch._profile_enum = StartupProfile.FULL
        lifecycle = MagicMock()
        lifecycle.start = AsyncMock(return_value=True)
        orch._lifecycle = lifecycle
        with patch.object(StartupOrchestrator, "_init_lifecycle", new=AsyncMock()) as mock_init, \
             patch.object(StartupOrchestrator, "_phase_ready", new=AsyncMock()) as mock_ready:
            asyncio.run(orch.run())
        mock_init.assert_awaited_once()
        lifecycle.start.assert_awaited_once()
        mock_ready.assert_awaited_once()
        assert orch._profile_enum.value == "full"

    def test_lifecycle_start_false_still_ready(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        lifecycle = MagicMock()
        lifecycle.start = AsyncMock(return_value=False)
        orch._lifecycle = lifecycle
        with patch.object(StartupOrchestrator, "_init_lifecycle", new=AsyncMock()), \
             patch.object(StartupOrchestrator, "_phase_ready", new=AsyncMock()) as mock_ready:
            asyncio.run(orch.run())
        mock_ready.assert_awaited_once()


# ── W&B + multimodal phases ───────────────────────────────────────────────────


class TestPhase3Wandb:
    """_phase3_wandb — disabled by default, enabled with env var."""

    def test_disabled_by_default(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SLO_WANDB", None)
            asyncio.run(orch._phase3_wandb())
        assert orch._wandb_task is None


# ── _autoload_model → native soul path ────────────────────────────────────────


class TestAutoloadNativeSoul:
    """_autoload_model — explicit native .soul path short-circuits HF."""

    def _soul_file(self, tmp_path):
        p = tmp_path / "my.soul"
        p.write_bytes(b"\x00\x01")
        return str(p)

    def test_native_soul_loaded(self, tmp_path):
        cfg = ServerConfig(autoload_model="gpt2", native_soul_path=self._soul_file(tmp_path))
        provider = MagicMock()
        provider._model = MagicMock()
        state = MagicMock()
        state.model = None
        state.model_type = None
        state.provider = None
        with patch.dict("sys.modules", {"state": state}), \
             patch("domains.inference.slonet_provider.SloNetChatProvider") as mock_provider:
            mock_provider.from_soul.return_value = provider
            from apps.api.server.infrastructure import startup as startup_mod
            startup_mod._autoload_model(cfg)
        mock_provider.from_soul.assert_called_once()
        assert state.model is provider._model
        assert state.provider is provider

    def test_native_soul_missing_falls_through_to_loader(self, tmp_path):
        cfg = ServerConfig(native_soul_path=str(tmp_path / "missing.soul"))
        state = MagicMock()
        state.model = None
        result = MagicMock()
        result.success = True
        result.model = MagicMock()
        result.model_id = "gpt2"
        result.tokenizer = None
        result.provider = None
        with patch.dict("sys.modules", {"state": state}), \
             patch("domains.infrastructure.model_loader.ModelLoader") as mock_loader:
            mock_loader.return_value.load.return_value = result
            from apps.api.server.infrastructure import startup as startup_mod
            startup_mod._autoload_model(cfg)
        assert state.model is result.model
        assert startup_mod._autoload_model.__module__ == "apps.api.server.infrastructure.startup"


# ── shutdown() orchestration ──────────────────────────────────────────────────


class TestShutdownOrchestrator:
    """shutdown() — lifecycle drain path + direct fallback."""

    def test_delegates_to_lifecycle(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        lifecycle = MagicMock()
        lifecycle.shutdown = AsyncMock()
        orch._lifecycle = lifecycle
        with patch.object(StartupOrchestrator, "_shutdown_task_queue", new=AsyncMock()) as mock_tq:
            asyncio.run(orch.shutdown())
        lifecycle.shutdown.assert_awaited_once()
        mock_tq.assert_not_called()

    def test_lifecycle_error_falls_back_to_direct(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        lifecycle = MagicMock()
        lifecycle.shutdown = AsyncMock(side_effect=RuntimeError("drain failed"))
        orch._lifecycle = lifecycle
        with patch.object(StartupOrchestrator, "_shutdown_task_queue", new=AsyncMock()) as mock_tq, \
             patch.object(StartupOrchestrator, "_shutdown_jobs", new=AsyncMock()) as mock_jobs, \
             patch.object(StartupOrchestrator, "_shutdown_wandb", new=AsyncMock()) as mock_wandb, \
             patch.object(StartupOrchestrator, "_shutdown_registry", new=AsyncMock()) as mock_reg, \
             patch.object(StartupOrchestrator, "_shutdown_pool", new=AsyncMock()) as mock_pool, \
             patch.object(StartupOrchestrator, "_shutdown_executor", new=AsyncMock()) as mock_exec, \
             patch.object(StartupOrchestrator, "_shutdown_process_guard", new=AsyncMock()) as mock_pg:
            asyncio.run(orch.shutdown())
        mock_tq.assert_awaited_once()
        mock_jobs.assert_awaited_once()
        mock_wandb.assert_awaited_once()
        mock_reg.assert_awaited_once()
        mock_pool.assert_awaited_once()
        mock_exec.assert_awaited_once()
        mock_pg.assert_awaited_once()

    def test_fallback_runs_when_no_lifecycle(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        orch._lifecycle = None
        with patch.object(StartupOrchestrator, "_shutdown_task_queue", new=AsyncMock()) as mock_tq, \
             patch.object(StartupOrchestrator, "_shutdown_jobs", new=AsyncMock()) as mock_jobs, \
             patch.object(StartupOrchestrator, "_shutdown_wandb", new=AsyncMock()) as mock_wandb, \
             patch.object(StartupOrchestrator, "_shutdown_registry", new=AsyncMock()) as mock_reg, \
             patch.object(StartupOrchestrator, "_shutdown_pool", new=AsyncMock()) as mock_pool, \
             patch.object(StartupOrchestrator, "_shutdown_executor", new=AsyncMock()) as mock_exec, \
             patch.object(StartupOrchestrator, "_shutdown_process_guard", new=AsyncMock()) as mock_pg:
            asyncio.run(orch.shutdown())
        mock_tq.assert_awaited_once()
        mock_jobs.assert_awaited_once()
        mock_wandb.assert_awaited_once()
        mock_reg.assert_awaited_once()
        mock_pool.assert_awaited_once()
        mock_exec.assert_awaited_once()
        mock_pg.assert_awaited_once()


# ── _try_lazy_guard_autoload ──────────────────────────────────────────────────


class TestTryLazyGuardAutoload:
    """_try_lazy_guard_autoload — every decision branch."""

    def _import(self):
        from apps.api.server.infrastructure import startup as startup_mod
        return startup_mod

    def test_model_already_loaded_returns_false(self):
        mod = self._import()
        with patch.dict("sys.modules", {"state": MagicMock(model=object())}):
            assert mod._try_lazy_guard_autoload(ServerConfig()) is False

    def test_lazy_guard_disabled_returns_false(self):
        mod = self._import()
        cfg = types.SimpleNamespace(
            lazy_guard_autoload=False,
            autoload_model="gpt2",
            quantize_slonet=False, quant_bits=8, quant_mode="sym",
            quant_clip=0.9, process_guard_memory_limit_mb=0.0,
        )
        with patch.dict("sys.modules", {"state": MagicMock(model=None)}):
            assert mod._try_lazy_guard_autoload(cfg) is False

    def test_process_guard_disabled_returns_false(self):
        mod = self._import()
        with patch.dict("sys.modules", {"state": MagicMock(model=None)}), \
             patch("config.get_process_guard_enabled", return_value=False):
            assert mod._try_lazy_guard_autoload(ServerConfig()) is False

    def test_no_slnc_returns_false(self):
        mod = self._import()
        with patch.dict("sys.modules", {"state": MagicMock(model=None)}), \
             patch("config.get_process_guard_enabled", return_value=True), \
             patch("domains.infrastructure.model_resolver.get_model_dir",
                   return_value=Path("/missing")), \
             patch("os.path.exists", return_value=False):
            assert mod._try_lazy_guard_autoload(ServerConfig(autoload_model="gpt2")) is False

    def test_provider_creation_failure_returns_false(self):
        mod = self._import()
        with patch.dict("sys.modules", {"state": MagicMock(model=None)}), \
             patch("config.get_process_guard_enabled", return_value=True), \
             patch("domains.infrastructure.model_resolver.get_model_dir",
                   return_value=Path("/fake")), \
             patch("os.path.exists", return_value=True), \
             patch("domains.inference.slonet_provider.SloNetChatProvider",
                   side_effect=RuntimeError("boom")):
            assert mod._try_lazy_guard_autoload(ServerConfig(autoload_model="gpt2")) is False

    def test_guard_creation_failure_returns_false(self):
        mod = self._import()
        with patch.dict("sys.modules", {"state": MagicMock(model=None)}), \
             patch("config.get_process_guard_enabled", return_value=True), \
             patch("domains.infrastructure.model_resolver.get_model_dir",
                   return_value=Path("/fake")), \
             patch("os.path.exists", return_value=True), \
             patch("domains.inference.slonet_provider.SloNetChatProvider") as mock_provider, \
             patch("domains.infrastructure.process_guard.ProcessGuard",
                   side_effect=RuntimeError("boom")):
            mock_provider.lazy_from_slnc.return_value = MagicMock()
            assert mod._try_lazy_guard_autoload(ServerConfig(autoload_model="gpt2")) is False

    def test_success_returns_true(self):
        mod = self._import()
        state = MagicMock(model=None)
        guard = MagicMock()
        provider = MagicMock()
        with patch.dict("sys.modules", {"state": state}), \
             patch("config.get_process_guard_enabled", return_value=True), \
             patch("domains.infrastructure.model_resolver.get_model_dir",
                   return_value=Path("/fake")), \
             patch("os.path.exists", return_value=True), \
             patch("domains.inference.slonet_provider.SloNetChatProvider",
                   lazy_from_slnc=MagicMock(return_value=provider)), \
             patch("domains.infrastructure.process_guard.ProcessGuard",
                   return_value=guard) as mock_pg, \
             patch("domains.infrastructure.process_guard.resolve_memory_limit_mb",
                   return_value=64.0), \
             patch("domains.infrastructure.model_registry.get_model_registry",
                   return_value=MagicMock()), \
             patch("domains.models.provider.setup_providers") as mock_setup, \
             patch("controllers.models.get_models_controller") as mock_ctrl, \
             patch("domains.infrastructure.server_state.get_server_state") as mock_core:
            core = mock_core.return_value
            result = mod._try_lazy_guard_autoload(ServerConfig(autoload_model="gpt2"))
        assert result is True
        assert mock_pg.return_value is guard
        guard.start.assert_called_once()
        mock_setup.assert_called_once()
        mock_ctrl.return_value.adopt_process_guard.assert_called_once_with(guard, "gpt2")


# ── run() direct fallback (no lifecycle) ──────────────────────────────────────


class TestRunDirectFallback:
    """run() — profile_enum None + lifecycle None → direct phase calls."""

    def test_direct_fallback_runs_registry_and_routers(self):
        orch = StartupOrchestrator(FastAPI(), ServerConfig())
        orch._lifecycle = None
        orch._profile_enum = None
        with patch.object(StartupOrchestrator, "_init_lifecycle", new=AsyncMock()), \
             patch.object(StartupOrchestrator, "_phase5_model_registry",
                          new=AsyncMock()) as mock_reg, \
             patch.object(StartupOrchestrator, "_phase6_routers",
                          new=AsyncMock()) as mock_routers, \
             patch.object(StartupOrchestrator, "_phase_ready", new=AsyncMock()) as mock_ready:
            asyncio.run(orch.run())
        assert mock_reg.call_count == 1
        assert mock_routers.call_count == 1
        mock_ready.assert_awaited()
