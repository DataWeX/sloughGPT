"""
Startup orchestrator — phased initialization of all server subsystems.

Manages the multi-phase startup with proper error isolation so one
subsystem's failure never crashes the entire server.

Integrates with LifecycleManager for phase state machine, health gates,
graceful drain, EventBus integration, and startup profiles.

Hooks are scoped to startup profiles:
* ``full`` — all components (AI model, W&B, multimodal, TaskQueue, Config)
* ``quick`` — skip AI model load (for dev/testing)
* ``minimal`` — core infrastructure only (logging, registry, routers)
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import sys
from typing import Any, Optional

from fastapi import FastAPI

from config import ServerConfig
from startup_progress import STARTUP_PHASE

logger = logging.getLogger("slo.startup")

# Timeout constants for startup/shutdown hooks (seconds)
_TIMEOUT_TASK_QUEUE = 10.0
_TIMEOUT_CONFIG = 5.0
_TIMEOUT_MODEL_LOAD = 120.0
_TIMEOUT_WANDB = 30.0
_TIMEOUT_MULTIMODAL = 30.0
_TIMEOUT_MODEL_REGISTRY = 10.0
_TIMEOUT_ROUTERS = 30.0
_TIMEOUT_STARTUP_TOTAL = 120.0
_TIMEOUT_SHUTDOWN = 30.0

# Modules the background model-load thread imports while the main thread is
# registering routers. First-time imports from two threads concurrently are
# the root cause of the intermittent startup EBADF (importlib.get_data →
# OSError: [Errno 9] Bad file descriptor) and partial ImportError races.
# Pre-importing this full graph up front (main thread, before the task is
# created) turns every later ``from X import Y`` in the thread into a cached
# sys.modules lookup — no file reads, no race.
_PREWARM_MODEL_LOAD_IMPORTS = [
    "state",
    "config",
    "domains.infrastructure.safetensors_loader",
    "domains.inference.slonet_provider",
    "domains.infrastructure.process_guard",
    "domains.infrastructure.server_state",
    "controllers.models",
    "domains.infrastructure.model_registry",
    "domains.models.provider",
    "domains.inference.slo_manager",
    "domains.slolib.gpu",
    "domains.infrastructure.model_catalog",
    "domains.infrastructure.task_queue",
    "domains.infrastructure.training_queue",
    "domains.api.sse_envelope",
    "pydantic.v1",
]
_TIMEOUT_REGISTER_GENERATE = 120.0


class StartupProfileSelector:
    """Helper to resolve the active startup profile from config or env."""

    @staticmethod
    def resolve(config: ServerConfig) -> str:
        """Return profile name: env var > config attribute > default."""
        raw = os.environ.get("SLO_STARTUP_PROFILE", "")
        if raw:
            return raw.strip().lower()
        if hasattr(config, "startup_profile"):
            return config.startup_profile
        return "full"


def _preload_model_imports() -> None:
    """Import the background model-load dependency graph in the main thread.

    Called once at the start of ``_phase2_model_load``, before the background
    task is created. Without this, the model-load thread and the
    router-registration code path both perform first-time module imports
    concurrently, which intermittently fails in the import machinery
    (``importlib`` ``get_data`` → ``OSError: [Errno 9] Bad file descriptor``
    or a partial ``ImportError``). After preloading, every ``from X import Y``
    in the thread hits ``sys.modules`` and performs zero file reads.

    Side effects:
        - Populates ``sys.modules`` with the model-load dependency graph.
    """
    for mod in _PREWARM_MODEL_LOAD_IMPORTS:
        try:
            __import__(mod)
        except Exception as e:
            logger.warning("Preload import failed for %s: %s", mod, e, extra={"tag": "START"})


class StartupOrchestrator:
    """Phased server initialization with LifecycleManager integration.

    Each phase is registered as a startup hook on the lifecycle manager.
    Background phases (model load, W&B, multimodal) are non-critical —
    a failure there doesn't block the server from starting.

    Supports startup profiles:
    * ``full`` — all components including AI model, W&B, multimodal
    * ``quick`` — skip AI model load (for dev/testing)
    * ``minimal`` — core infrastructure only (logging, registry, routers)

    On shutdown, the lifecycle manager drains in-flight requests before
    running cleanup hooks.
    """

    def __init__(self, app: FastAPI, config: ServerConfig, profile: str | None = None):
        self._app = app
        self._config = config
        self._profile = profile or StartupProfileSelector.resolve(config)
        self._wandb_task: Optional[asyncio.Task] = None
        self._registry: Any = None
        self._task_queue: Any = None
        self._lifecycle = None
        self._routers_registered = False
        self._model_load_task: Optional[asyncio.Task] = None

    async def _init_lifecycle(self):
        """Lazy-init lifecycle manager with EventBus."""
        if self._lifecycle is not None:
            return
        try:
            from domains.infrastructure.event_bus import EventBus
            from domains.infrastructure.lifecycle import (
                ALL_PROFILES,
                StartupHook,
                StartupProfile,
                ShutdownHook,
                get_lifecycle_manager,
            )

            # Resolve profile enum
            try:
                self._profile_enum = StartupProfile(self._profile)
            except ValueError:
                self._profile_enum = StartupProfile.FULL
            profile_enum = self._profile_enum

            full_only: frozenset[StartupProfile] = frozenset({StartupProfile.FULL})
            quick_plus: frozenset[StartupProfile] = frozenset(
                {StartupProfile.FULL, StartupProfile.QUICK}
            )
            all_profiles: frozenset[StartupProfile] = ALL_PROFILES

            bus = EventBus(max_history=200)
            self._lifecycle = get_lifecycle_manager(event_bus=bus)

            # Register startup hooks with profile scoping
            self._lifecycle.register_startup_hook(
                StartupHook(
                    "task_queue", self._phase_task_queue,
                    depends_on=[], timeout=_TIMEOUT_TASK_QUEUE, critical=False,
                    profiles=quick_plus,
                ),
            )
            self._lifecycle.register_startup_hook(
                StartupHook(
                    "config", self._phase_config,
                    depends_on=[], timeout=_TIMEOUT_CONFIG, critical=False,
                    profiles=quick_plus,
                ),
            )
            self._lifecycle.register_startup_hook(
                StartupHook(
                    "model_load", self._phase2_model_load,
                    depends_on=["config"], timeout=_TIMEOUT_MODEL_LOAD, critical=False,
                    profiles=full_only,
                ),
            )
            self._lifecycle.register_startup_hook(
                StartupHook(
                    "wandb", self._phase3_wandb,
                    depends_on=[], timeout=_TIMEOUT_WANDB, critical=False,
                    profiles=full_only,
                ),
            )
            self._lifecycle.register_startup_hook(
                StartupHook(
                    "multimodal", self._phase4_multimodal,
                    depends_on=[], timeout=_TIMEOUT_MULTIMODAL, critical=False,
                    profiles=full_only,
                ),
            )
            self._lifecycle.register_startup_hook(
                StartupHook(
                    "model_registry", self._phase5_model_registry,
                    depends_on=["task_queue", "config"],
                    timeout=_TIMEOUT_MODEL_REGISTRY, critical=False,
                    profiles=quick_plus,
                ),
            )
            self._lifecycle.register_startup_hook(
                StartupHook(
                    "routers", self._phase6_routers,
                    depends_on=["model_registry"], timeout=_TIMEOUT_ROUTERS, critical=True,
                    profiles=all_profiles,
                ),
            )

            # Register shutdown hooks
            self._lifecycle.register_shutdown_hook(
                ShutdownHook("job_cleanup", self._shutdown_jobs, depends_on=[], timeout=_TIMEOUT_TASK_QUEUE),
            )
            self._lifecycle.register_shutdown_hook(
                ShutdownHook("wandb_cancel", self._shutdown_wandb, depends_on=[], timeout=_TIMEOUT_CONFIG),
            )
            self._lifecycle.register_shutdown_hook(
                ShutdownHook("registry_cleanup", self._shutdown_registry, depends_on=[], timeout=_TIMEOUT_CONFIG),
            )
            self._lifecycle.register_shutdown_hook(
                ShutdownHook("task_queue_shutdown", self._shutdown_task_queue, depends_on=[], timeout=_TIMEOUT_TASK_QUEUE),
            )
            self._lifecycle.register_shutdown_hook(
                ShutdownHook("pool_shutdown", self._shutdown_pool, depends_on=[], timeout=10.0),
            )
            self._lifecycle.register_shutdown_hook(
                ShutdownHook("executor_shutdown", self._shutdown_executor, depends_on=[], timeout=10.0),
            )
            self._lifecycle.register_shutdown_hook(
                ShutdownHook("process_guard_shutdown", self._shutdown_process_guard, depends_on=[], timeout=_TIMEOUT_CONFIG),
            )

            # Health gates
            self._lifecycle.register_gate("model_loaded", lambda: self._is_model_loaded())
            self._lifecycle.register_gate("routers_registered", lambda: self._routers_registered)

            logger.info(
                "LifecycleManager initialized (profile=%s, %d startup hooks, %d shutdown hooks)",
                profile_enum.value,
                len(self._lifecycle._startup_hooks),
                len(self._lifecycle._shutdown_hooks),
                extra={"tag": "START"},
            )
        except Exception as exc:
            logger.warning("LifecycleManager init skipped: %s", exc, extra={"tag": "START"})

    @property
    def lifecycle(self):
        return self._lifecycle

    def _is_model_loaded(self) -> bool:
        """Check if a model is loaded (either via autoload or manually)."""
        import state as server_state
        return server_state.model is not None

    async def run(self):
        """Execute all startup phases via the lifecycle manager."""
        # Initialize lifecycle manager
        await self._init_lifecycle()

        # Reuse profile enum resolved in _init_lifecycle
        profile_enum = getattr(self, '_profile_enum', None)
        if profile_enum is None:
            try:
                from domains.infrastructure.lifecycle import StartupProfile
                profile_enum = StartupProfile.FULL
            except Exception:
                profile_enum = None

        # Run sequential phases via lifecycle manager
        if self._lifecycle is not None:
            ok = await self._lifecycle.start(timeout=180.0, profile=profile_enum)
            if not ok:
                logger.warning("Lifecycle startup incomplete — continuing anyway", extra={"tag": "START"})
        else:
            # Fallback: run phases directly
            self._phase5_model_registry()
            self._phase6_routers()

        await _restore_training_runtime()
        await self._phase_ready()

    async def _phase2_model_load(self):
        """Start model load as a background task (non-blocking).

        The model load takes ~45s. Previously this blocked the entire startup
        lifecycle, preventing routers from registering. Now we fire-and-forget
        so routers register immediately and the server accepts requests.
        Requests that need the model get a "model still loading" response.
        """
        cfg = self._config

        # Serialize first-time imports: the background load thread and the
        # main thread (router registration, phase 6) must never import fresh
        # modules at the same instant. Pre-import the thread's full dependency
        # graph here so its later imports are cache hits.
        _preload_model_imports()

        raw = cfg.autoload_model
        if not raw or raw.lower() in ("false", "0", "none", "no", "off", "disable"):
            logger.info("Phase: autoload disabled (%r)", raw, extra={"tag": "START"})
            return

        STARTUP_PHASE.update(phase="loading_model", step=4, total=9, message="Loading model weights...")
        logger.info("Phase 4: loading model %s (background)", raw, extra={"tag": "START"})

        def _load_and_register():
            """Load model then register with registry/providers."""
            import state as server_state

            # Lazy-guard fast path (default): with a ProcessGuard + .slnc
            # available, defer the parent weight load entirely. The guard's
            # worker process owns the weights and serves inference; the parent
            # stays near-idle and materializes weights lazily only if the
            # guard dies.
            if _try_lazy_guard_autoload(cfg):
                _sync_soul_traits()
                logger.info(
                    "Lazy-guard autoload ready: %s (parent weights deferred)",
                    cfg.autoload_model, extra={"tag": "START"},
                )
                return

            try:
                _autoload_model(cfg)
            except Exception as e:
                logger.error("Model load failed: %s", e, exc_info=True, extra={"tag": "START"})
                return

            # After model is loaded, register with registry + providers
            try:
                process_guard = _build_guard_for_model(cfg, server_state.model_type)
                _register_loaded(cfg, process_guard)
            except Exception as e:
                logger.error("Post-load registration failed: %s", e, exc_info=True, extra={"tag": "START"})

        # Fire-and-forget: model loads in background while routers register
        task = asyncio.create_task(asyncio.to_thread(_load_and_register))
        task.add_done_callback(self._on_model_load_done)

    def _on_model_load_done(self, task: asyncio.Task):
        try:
            task.result()
        except asyncio.CancelledError:
            logger.debug("Model load task cancelled (server shutting down)", extra={"tag": "START"})
            return
        except Exception as e:
            logger.error("Model load task failed: %s", e, exc_info=True, extra={"tag": "START"})

        # Sync to persistent model catalog
        try:
            import state as server_state
            from domains.infrastructure.model_catalog import get_model_catalog
            catalog = get_model_catalog()
            catalog.sync_from_disk()
            if hasattr(server_state, "model_type") and server_state.model_type:
                catalog.mark_loaded(
                    server_state.model_type,
                    device=getattr(server_state, "device", "cpu"),
                )
        except Exception as e:
            logger.debug("Model catalog sync failed: %s", e, extra={"tag": "START"})

    async def _phase3_wandb(self):
        """Start W&B metrics server (disabled by default to save RAM).

        Enable with SLO_WANDB=1 environment variable.
        """
        STARTUP_PHASE.update(phase="wandb_server", step=5, total=9, message="W&B: disabled by default")
        enabled = os.environ.get("SLO_WANDB", "").lower() in ("1", "true", "yes")
        if not enabled:
            logger.info("Phase: W&B skipped (enable with SLO_WANDB=1)", extra={"tag": "START"})
            return
        try:
            from domains.ops.wandb_server import start_wandb_server_background

            async def _start():
                try:
                    from host_metrics import sample_host_metrics_sync

                    def _extra_metrics():
                        h = sample_host_metrics_sync()
                        out = {
                            "host/cpu_percent": float(h["cpu_percent"]),
                            "host/memory_percent": float(h["memory_percent"]),
                        }
                        rss = h.get("process_rss_bytes")
                        if isinstance(rss, int) and rss >= 0:
                            out["server/process_rss_bytes"] = float(rss)
                        return out

                    class _NoopMetrics:
                        @staticmethod
                        def wandb_aggregate():
                            return {}

                    self._wandb_task = await start_wandb_server_background(
                        _NoopMetrics(), extra_metrics=_extra_metrics,
                    )
                    logger.info("Phase: W&B metrics server started", extra={"tag": "START"})
                except Exception as e:
                    logger.warning("Phase: W&B server skipped: %s", e, extra={"tag": "START"})

            self._wandb_task = asyncio.create_task(_start())
        except Exception as e:
            logger.warning("Phase: W&B unavailable: %s", e, extra={"tag": "START"})

    async def _phase4_multimodal(self):
        """Initialize multimodal engine (lazy — skipped at startup to save RAM).

        The multimodal engine (VisionCNN + models) is loaded on first use
        via the /multimodal/* endpoints, not at server startup.
        """
        STARTUP_PHASE.update(phase="multimodal", step=6, total=9, message="Multimodal: lazy-load enabled")
        logger.info("Phase 4/6: multimodal deferred to first use (saves ~200MB RAM)", extra={"tag": "START"})

    async def _phase5_model_registry(self):
        """Initialize model registry."""
        STARTUP_PHASE.update(phase="model_registry", step=7, total=9, message="Initializing model registry...")
        try:
            from domains.infrastructure.model_registry import get_model_registry
            self._registry = get_model_registry()
            logger.info("Phase: model registry initialized", extra={"tag": "START"})
        except Exception as e:
            logger.warning("Phase: model registry failed: %s", e, extra={"tag": "START"})

    async def _phase6_routers(self):
        """Register all feature routers.

        Skips any routers already registered before lifespan (e.g.
        health/status which are needed during model load).

        A first-time import can transiently fail with ``OSError`` errno 9
        (EBADF) or a partial ``ImportError`` if another thread is importing
        concurrently; in that case the whole pass is retried once. Failed
        imports are rolled back out of ``sys.modules`` by importlib, so a
        retry re-imports cleanly and skips any routers already included.
        """
        STARTUP_PHASE.update(phase="registering_routers", step=8, total=9, message="Registering routes...")
        last_exc: Optional[BaseException] = None
        for attempt in range(3):
            try:
                from routers import get_all_routers
                # Collect prefixes already registered (health/status are
                # registered pre-lifespan in main.py).
                existing = set()
                for route in self._app.routes:
                    if hasattr(route, "path") and hasattr(route, "methods"):
                        existing.add(route.path)
                for r in get_all_routers():
                    # Skip if this router's prefix already has routes
                    prefix = getattr(r, "prefix", "")
                    if prefix and any(p.startswith(prefix) for p in existing):
                        logger.debug("Skipping already-registered router: %s", prefix)
                        continue
                    self._app.include_router(r)
                    if prefix:
                        existing.add(prefix)
                try:
                    from training.router import router as training_router
                    self._app.include_router(training_router)
                except Exception as exc:
                    logger.warning("Phase: training router failed: %s", exc, extra={"tag": "START"})
                logger.info(
                    "Phase: all routers registered (%d routes)",
                    len(self._app.routes),
                    extra={"tag": "START"},
                )
                self._routers_registered = True
                return
            except Exception as e:
                last_exc = e
                transient = isinstance(e, ImportError) or (
                    isinstance(e, OSError) and getattr(e, "errno", None) == 9
                )
                if transient and attempt < 2:
                    logger.warning(
                        "Phase: router registration transient import failure (%s) — retrying (attempt %d/3)",
                        e, attempt + 1, extra={"tag": "START"},
                    )
                    import traceback as _tb
                    import faulthandler as _fth
                    try:
                        _fth.dump_traceback(all_threads=True, file=sys.stderr)
                    except Exception:
                        pass
                    _tb.print_exc(file=sys.stderr)
                    # Clear import caches (not sys.modules entries) so the
                    # next import attempt re-reads from disk instead of
                    # using stale/broken file descriptors.
                    importlib.invalidate_caches()
                    await asyncio.sleep(0.5 * (attempt + 1))
                    continue
                self._routers_registered = False
                import traceback as _tb
                import faulthandler as _fth
                try:
                    _fth.dump_traceback(all_threads=True, file=sys.stderr)
                except Exception:
                    pass
                _tb.print_exc(file=sys.stderr)
                logger.error("Phase: router registration failed: %s", last_exc, exc_info=True, extra={"tag": "START"})
                raise

    async def _phase_task_queue(self):
        """Initialize the background task queue and register training handlers."""
        STARTUP_PHASE.update(phase="task_queue", step=2, total=9, message="Initializing task queue...")
        try:
            from domains.infrastructure.task_queue import get_task_queue
            self._task_queue = get_task_queue()
            await self._task_queue.start()
            logger.info("Task queue initialized and started", extra={"tag": "START"})
        except Exception as e:
            logger.warning("Task queue init failed: %s", e, extra={"tag": "START"})
        try:
            from domains.infrastructure.training_queue import register_training_handlers
            register_training_handlers()
            logger.info("Training handlers registered with task queue", extra={"tag": "START"})
        except Exception as e:
            logger.warning("Training handler registration failed: %s", e, extra={"tag": "START"})
        try:
            from domains.memory import register_memory_handlers
            register_memory_handlers()
            logger.info("Memory handlers registered with task queue", extra={"tag": "START"})
        except Exception as e:
            logger.warning("Memory handler registration failed: %s", e, extra={"tag": "START"})
        try:
            from domains.memory.maintenance import start_memory_maintenance
            start_memory_maintenance()
            logger.info("Memory maintenance scheduler started", extra={"tag": "START"})
        except Exception as e:
            logger.warning("Memory maintenance scheduler start failed: %s", e, extra={"tag": "START"})

    async def _phase_config(self):
        """Validate and warm the config system + init ResourceManager."""
        STARTUP_PHASE.update(phase="config", step=3, total=9, message="Validating config...")
        try:
            from domains.infrastructure.config import get_config
            cfg = get_config()
            _ = cfg.model.name
            logger.info("Config system validated", extra={"tag": "START"})
        except Exception as e:
            logger.warning("Config system init: %s", e, extra={"tag": "START"})
        # Init ResourceManager — applies BLAS env vars before numpy loads
        try:
            from domains.infrastructure.resource_manager import get_resource_manager
            rm = get_resource_manager()
            rm.apply_blas_env()
            rm.apply_compute_limits()
            rm.apply_environment()
            logger.info("ResourceManager initialised: mode=%s %s", rm.mode, rm.summary(),
                extra={"tag": "START"})
        except Exception as e:
            logger.warning("ResourceManager init: %s", e, extra={"tag": "START"})

    async def _phase_ready(self):
        """Mark server as ready — happens after all synchronous phases complete."""
        STARTUP_PHASE.update(phase="ready", step=9, total=9, message="Server ready")
        logger.info("Startup complete — server ready for requests", extra={"tag": "START"})

    # ── Shutdown hooks ──

    async def _shutdown_training_runtime(self):
        """Gracefully stop tracked training jobs and persist final state.

        Runs before the task queue stops so cooperative trainers get their
        drain budget to save a final checkpoint. Anything still running when
        the budget expires is marked ``interrupted`` and stays recoverable.
        """
        try:
            from training.runtime import get_training_runtime
            await asyncio.to_thread(get_training_runtime().shutdown)
        except Exception as e:
            logger.warning("Training runtime shutdown: %s", e, extra={"tag": "START"})

    async def _shutdown_task_queue(self):
        """Gracefully stop the background task queue."""
        if self._task_queue is not None:
            try:
                from domains.memory.maintenance import stop_memory_maintenance
                await stop_memory_maintenance()
            except Exception as e:
                logger.warning("Memory maintenance shutdown: %s", e, extra={"tag": "START"})
            try:
                await self._task_queue.stop()
                logger.info("Task queue stopped", extra={"tag": "START"})
            except Exception as e:
                logger.warning("Task queue shutdown: %s", e, extra={"tag": "START"})

    async def _shutdown_jobs(self):
        """Mark running training jobs as crashed on shutdown."""
        try:
            from training.job_store import get_job_store
            store = get_job_store()
            for job in store.list(status="running"):
                store.mark_crashed(job["id"])
                logger.info("Marked job %s as interrupted on shutdown", job["id"], extra={"tag": "START"})
        except Exception as e:
            logger.warning("Shutdown job cleanup: %s", e, extra={"tag": "START"})

    async def _shutdown_wandb(self):
        """Cancel W&B server task."""
        if self._wandb_task is not None:
            self._wandb_task.cancel()
            try:
                await self._wandb_task
            except asyncio.CancelledError:
                pass

    async def _shutdown_registry(self):
        """Reset model registry metrics."""
        try:
            from domains.infrastructure.model_registry import get_model_registry
            get_model_registry().reset_metrics()
        except Exception as e:
            logger.debug("Registry reset failed during shutdown: %s", e)

    async def _shutdown_process_guard(self):
        """Stop any active ProcessGuard so worker subprocesses exit cleanly."""
        try:
            from controllers.models import get_models_controller
            ctrl = get_models_controller()
            if hasattr(ctrl, "_stop_process_guard"):
                ctrl._stop_process_guard()
        except Exception as e:
            logger.debug("ProcessGuard shutdown failed: %s", e)

    async def _shutdown_pool(self):
        """Shut down inference pool."""
        try:
            from infrastructure.inference_pool import InferencePool
            pool = await InferencePool.get_instance()
            await pool.shutdown()
        except Exception as e:
            logger.debug("Pool shutdown failed: %s", e)

    async def _shutdown_executor(self):
        """Gracefully shut down the TrainingExecutor thread pool."""
        try:
            from domains.training.executor import _instance
            if _instance is not None:
                _instance.shutdown(wait=True)
                logger.info("TrainingExecutor shut down", extra={"tag": "START"})
        except Exception as e:
                logger.warning("TrainingExecutor shutdown: %s", e, extra={"tag": "START"})

    async def shutdown(self):
        """Clean up on server shutdown — uses lifecycle drain if available."""
        if self._lifecycle is not None:
            try:
                await self._lifecycle.shutdown(timeout=30.0)
                return
            except Exception as e:
                logger.warning("Lifecycle shutdown error: %s", e, extra={"tag": "START"})

        # Fallback: direct cleanup
        await self._shutdown_training_runtime()
        await self._shutdown_task_queue()
        await self._shutdown_jobs()
        await self._shutdown_wandb()
        await self._shutdown_registry()
        await self._shutdown_pool()
        await self._shutdown_executor()
        await self._shutdown_process_guard()


async def _restore_training_runtime():
    """Restore persisted training jobs into the runtime + job registry."""
    try:
        from training.runtime import get_training_runtime
        get_training_runtime().restore()
    except Exception as e:
        logger.warning("Training runtime restore failed: %s", e, extra={"tag": "START"})


def _sync_soul_traits():
    """Sync current soul traits to the PersonalityProcessor (best-effort)."""
    try:
        from domains.inference.slo_manager import get_slo_manager
        from domains.models.provider import update_personality_traits
        mgr = get_slo_manager()
        current = mgr.get_current_soul()
        if current and hasattr(current, "personality") and current.personality:
            update_personality_traits(current.personality)
    except Exception as e:
        logger.debug("Failed to sync soul traits to personality processor: %s", e, extra={"tag": "START"})


def _try_lazy_guard_autoload(cfg) -> bool:
    """Defer the parent weight load when a ProcessGuard + .slnc are available.

    Creates a header-only lazy ``SloNetChatProvider`` (no weight pages faulted
    into the parent) and a ``ProcessGuard`` whose worker process loads the
    weights. The parent process stays near-idle; inference is served by the
    guard worker; if the guard dies, the parent materializes weights on demand
    via the lazy provider's ``_get_model()``.

    Returns:
        True when the lazy fast path was applied, False to fall back to the
        eager autoload.
    """
    import state as server_state

    if server_state.model is not None:
        return False
    if not cfg.lazy_guard_autoload:
        return False
    try:
        from config import get_process_guard_enabled
        if not get_process_guard_enabled():
            return False
    except Exception:
        return False

    model_type = cfg.autoload_model
    if not model_type:
        return False
    try:
        from domains.infrastructure.safetensors_loader import _get_model_dir
        slnc_path = str(_get_model_dir(model_type) / "model.slnc")
        if not os.path.exists(slnc_path):
            logger.info("Lazy-guard autoload skipped: no .slnc at %s", slnc_path, extra={"tag": "START"})
            return False
    except Exception as e:
        logger.warning("Lazy-guard autoload: slnc resolution failed (%s)", e, extra={"tag": "START"})
        return False

    try:
        from domains.inference.slonet_provider import SloNetChatProvider
        provider = SloNetChatProvider.lazy_from_slnc(
            slnc_path,
            model_id=model_type,
            quantize=cfg.quantize_slonet,
            quant_bits=cfg.quant_bits,
            quant_mode=cfg.quant_mode,
            quant_clip=cfg.quant_clip,
        )
    except Exception as e:
        logger.warning("Lazy-guard autoload: lazy provider creation failed (%s)", e, extra={"tag": "START"})
        return False

    try:
        from domains.infrastructure.process_guard import ProcessGuard, resolve_memory_limit_mb
        process_guard = ProcessGuard(
            slnc_path=slnc_path,
            model_id=model_type,
            worker_id=f"slo-{model_type.split('/')[-1]}",
            max_restarts=3,
            restart_delay=2.0,
            generate_timeout=120.0,
            memory_limit_mb=resolve_memory_limit_mb(slnc_path, cfg.process_guard_memory_limit_mb),
            quantize=cfg.quantize_slonet,
            quant_bits=cfg.quant_bits,
            quant_mode=cfg.quant_mode,
            quant_clip=cfg.quant_clip,
        )
        process_guard.start()
    except Exception as e:
        logger.warning(
            "Lazy-guard autoload: guard start failed (%s) — falling back to eager load",
            e, extra={"tag": "START"},
        )
        return False

    server_state.model_type = model_type
    server_state.model = None
    server_state.provider = provider
    server_state.tokenizer = getattr(provider, "_tokenizer", None)

    # Mirror to the core ServerState singleton — the source for
    # get_health_score() — so /health/detailed health_score reports a
    # loaded model (provider-backed) instead of "No model loaded".
    # Mirrors the manual-load path in controllers/models.py.
    try:
        from domains.infrastructure.server_state import get_server_state
        core = get_server_state()
        core.model.set(provider)
        core.model_type.set(model_type)
    except Exception as e:
        logger.debug("Core ServerState mirror failed: %s", e, extra={"tag": "START"})

    # Track the guard so /models/process-guard status and the runtime toggle
    # can manage it (autoload path bypasses the controller).
    try:
        from controllers.models import get_models_controller
        get_models_controller().adopt_process_guard(process_guard, model_type)
    except Exception as e:
        logger.debug("ProcessGuard adoption into controller failed: %s", e, extra={"tag": "START"})

    try:
        from domains.infrastructure.model_registry import get_model_registry
        from domains.models.provider import setup_providers
        setup_providers(
            slonet_provider=provider,
            model_registry=get_model_registry(),
            process_guard=process_guard,
            quantize=cfg.quantize_slonet,
            quant_bits=cfg.quant_bits,
            quant_mode=cfg.quant_mode,
        )
    except Exception as e:
        logger.error("Lazy-guard autoload: registration failed (%s)", e, exc_info=True, extra={"tag": "START"})
        return False

    logger.info(
        "Lazy-guard autoload active for %s (worker: %s) — parent weights deferred",
        model_type, process_guard.worker_id, extra={"tag": "START"},
    )
    return True


def _build_guard_for_model(cfg, model_type: str):
    """Create and start a ProcessGuard for a loaded model's .slnc (or None).

    Args:
        cfg: ServerConfig with process-guard + quantization settings
        model_type: Model id (used for the .slnc path and worker naming)

    Returns:
        A started ProcessGuard, or None when disabled / no .slnc / failure.
    """
    try:
        from config import get_process_guard_enabled
        if not get_process_guard_enabled():
            return None
        from domains.infrastructure.process_guard import ProcessGuard, resolve_memory_limit_mb
        from domains.infrastructure.safetensors_loader import _get_model_dir

        slnc_path = str(_get_model_dir(model_type) / "model.slnc")
        if not os.path.exists(slnc_path):
            logger.info("ProcessGuard skipped: no .slnc file at %s", slnc_path, extra={"tag": "START"})
            return None
        process_guard = ProcessGuard(
            slnc_path=slnc_path,
            model_id=model_type,
            worker_id=f"slo-{model_type.split('/')[-1]}",
            max_restarts=3,
            restart_delay=2.0,
            generate_timeout=120.0,
            memory_limit_mb=resolve_memory_limit_mb(slnc_path, cfg.process_guard_memory_limit_mb),
            quantize=cfg.quantize_slonet,
            quant_bits=cfg.quant_bits,
            quant_mode=cfg.quant_mode,
            quant_clip=cfg.quant_clip,
        )
        process_guard.start()
        logger.info("ProcessGuard started for %s", model_type, extra={"tag": "START"})
        # Track the guard so /models/process-guard status and the runtime toggle
        # can manage it (autoload path bypasses the controller).
        try:
            from controllers.models import get_models_controller
            get_models_controller().adopt_process_guard(process_guard, model_type)
        except Exception as e:
            logger.debug("ProcessGuard adoption into controller failed: %s", e, extra={"tag": "START"})
        return process_guard
    except Exception as e:
        logger.warning("ProcessGuard creation failed: %s", e, extra={"tag": "START"})
        return None


def _register_loaded(cfg, process_guard) -> None:
    """Register a fully-loaded (eager) model with registry + providers."""
    import state as server_state
    from domains.infrastructure.model_registry import get_model_registry
    from domains.infrastructure.safetensors_loader import _get_model_dir
    from domains.models.provider import setup_providers

    registry = get_model_registry()

    if server_state.model is not None and server_state.tokenizer is not None:
        server = registry.register(
            server_state.model_type, server_state.model, server_state.tokenizer,
            make_default=True, generate_timeout=120.0,
            process_guard=process_guard,
            idle_timeout_s=cfg.idle_timeout_seconds,
        )
        # Store reload parameters for idle-reload capability
        if cfg.idle_timeout_seconds > 0 and server_state.model_type:
            _slnc = str(_get_model_dir(server_state.model_type) / "model.slnc")
            server.set_hf_model_id(
                server_state.model_type,
                slnc_path=_slnc if os.path.exists(_slnc) else None,
                quantize=cfg.quantize_slonet,
                quant_bits=cfg.quant_bits,
                quant_mode=cfg.quant_mode,
            )

    # Pass pre-loaded provider to avoid duplicate SLNC load (~6s)
    preloaded = getattr(server_state, 'provider', None)
    setup_providers(
        slonet_hf_id=server_state.model_type,
        slonet_provider=preloaded,
        model_registry=registry,
        process_guard=process_guard,
        quantize=cfg.quantize_slonet,
        quant_bits=cfg.quant_bits,
        quant_mode=cfg.quant_mode,
    )

    # Auto-select precision on GPU (fp16 benchmark)
    try:
        from domains.slolib.gpu import set_accelerator_precision
        active = set_accelerator_precision("auto")
        if active == "fp16":
            logger.info("GPU precision set to fp16 (auto-selected via benchmark)", extra={"tag": "START"})
    except Exception:
        pass
    _sync_soul_traits()
    logger.info("Model loaded + providers registered: %s", server_state.model_type, extra={"tag": "START"})


def _autoload_model(cfg: ServerConfig):
    """Load model weights into server_state. Registration handled by caller."""
    import state as server_state

    if server_state.model is not None:
        return

    # 0) Explicit native .soul path — skip HuggingFace entirely
    if cfg.native_soul_path:
        from pathlib import Path
        soul_path = Path(cfg.native_soul_path)
        if soul_path.exists():
            try:
                from domains.inference.slonet_provider import SloNetChatProvider
                provider = SloNetChatProvider.from_soul(str(soul_path), model_id="native-soul")
                server_state.model = provider._model
                server_state.model_type = "native-soul"
                server_state.provider = provider
                logger.info("Autoload native soul: %s", soul_path, extra={"tag": "START"})
                return
            except Exception as e:
                logger.warning("Native soul load failed (%s), falling back to standard autoload", e, extra={"tag": "START"})
        else:
            logger.warning("Native soul path %s not found, falling back to standard autoload", soul_path, extra={"tag": "START"})

    model_id = cfg.autoload_model

    # 1) Try local .slnc / safetensors via ModelLoader
    from domains.infrastructure.model_loader import ModelLoader

    loader = ModelLoader()
    result = loader.load(
        model_id=model_id,
        device=cfg.autoload_device,
        quantize=cfg.quantize_slonet,
        quant_bits=cfg.quant_bits,
        quant_mode=cfg.quant_mode,
        verify=True,
    )

    if result.success:
        server_state.model = result.model
        server_state.model_type = result.model_id
        if result.tokenizer is not None:
            server_state.tokenizer = result.tokenizer
        if result.provider is not None:
            server_state.provider = result.provider
        logger.info("Autoload ok: %s (%s)", model_id, result.model_type, extra={"tag": "START"})
        return

    # 2) No local file — download from HuggingFace via downcraft (resume-aware),
    #    then load via ModelLoader
    logger.info("No local .slnc/safetensors for %s — downloading from HuggingFace", model_id, extra={"tag": "START"})
    try:
        from domains.infrastructure.hf_hub import download_hf_model
        from domains.infrastructure.safetensors_loader import _get_model_dir

        cache_dir = _get_model_dir(model_id)
        logger.info("Downloading %s to %s ...", model_id, cache_dir, extra={"tag": "START"})
        download_hf_model(model_id)
        logger.info("Download complete: %s", model_id, extra={"tag": "START"})
    except Exception as e:
        logger.warning("HuggingFace download failed: %s", e, extra={"tag": "START"})
        return

    # 3) Retry load now that safetensors are cached
    result = loader.load(
        model_id=model_id,
        device=cfg.autoload_device,
        quantize=cfg.quantize_slonet,
        quant_bits=cfg.quant_bits,
        quant_mode=cfg.quant_mode,
        verify=True,
    )
    if result.success:
        server_state.model = result.model
        server_state.model_type = result.model_id
        if result.tokenizer is not None:
            server_state.tokenizer = result.tokenizer
        if result.provider is not None:
            server_state.provider = result.provider
        logger.info("Autoload ok (after download): %s (%s)", model_id, result.model_type, extra={"tag": "START"})
    else:
        logger.warning("Autoload failed after download: %s", result.error, extra={"tag": "START"})
