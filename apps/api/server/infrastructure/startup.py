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
import logging
import os
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
                    depends_on=[], timeout=_TIMEOUT_MODEL_LOAD, critical=False,
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
        profile_enum = getattr(self, '_profile_enum', StartupProfile.FULL)

        # Run sequential phases via lifecycle manager
        if self._lifecycle is not None:
            ok = await self._lifecycle.start(timeout=180.0, profile=profile_enum)
            if not ok:
                logger.warning("Lifecycle startup incomplete — continuing anyway", extra={"tag": "START"})
        else:
            # Fallback: run phases directly
            self._phase5_model_registry()
            self._phase6_routers()

        await self._phase_ready()

    async def _phase2_model_load(self):
        """Start model load as a background task (non-blocking).

        The model load takes ~45s. Previously this blocked the entire startup
        lifecycle, preventing routers from registering. Now we fire-and-forget
        so routers register immediately and the server accepts requests.
        Requests that need the model get a "model still loading" response.
        """
        cfg = self._config

        raw = cfg.autoload_model
        if not raw or raw.lower() in ("false", "0", "none", "no", "off", "disable"):
            logger.info("Phase: autoload disabled (%r)", raw, extra={"tag": "START"})
            return

        STARTUP_PHASE.update(phase="loading_model", step=4, total=9, message="Loading model weights...")
        logger.info("Phase 4: loading model %s (background)", raw, extra={"tag": "START"})

        def _load_and_register():
            """Load model then register with registry/providers."""
            try:
                _autoload_model(cfg)
            except Exception as e:
                logger.error("Model load failed: %s", e, exc_info=True, extra={"tag": "START"})
                return

            # After model is loaded, register with registry + providers
            try:
                import state as server_state
                from domains.infrastructure.model_registry import get_model_registry
                from domains.models.provider import setup_providers

                registry = get_model_registry()

                # Create process guard if enabled
                process_guard = None
                if cfg.enable_process_guard:
                    try:
                        from domains.infrastructure.process_guard import ProcessGuard
                        from domains.infrastructure.safetensors_loader import _get_model_dir

                        slnc_path = str(_get_model_dir(server_state.model_type) / "model.slnc")
                        if os.path.exists(slnc_path):
                            process_guard = ProcessGuard(
                                slnc_path=slnc_path,
                                model_id=server_state.model_type,
                                worker_id=f"slo-{server_state.model_type.split('/')[-1]}",
                                max_restarts=3,
                                restart_delay=2.0,
                                generate_timeout=120.0,
                                quantize=cfg.quantize_slonet,
                                quant_bits=cfg.quant_bits,
                                quant_mode=cfg.quant_mode,
                                quant_clip=cfg.quant_clip,
                            )
                            process_guard.start()
                            logger.info("ProcessGuard started for %s", server_state.model_type,
                                extra={"tag": "START"})
                        else:
                            logger.info("ProcessGuard skipped: no .slnc file at %s", slnc_path,
                                extra={"tag": "START"})
                    except Exception as e:
                        logger.warning("ProcessGuard creation failed: %s", e, extra={"tag": "START"})

                if server_state.model is not None and server_state.tokenizer is not None:
                    registry.register(
                        server_state.model_type, server_state.model, server_state.tokenizer,
                        make_default=True, generate_timeout=120.0,
                        process_guard=process_guard,
                    )

                slonet_id = cfg.autoload_model if cfg.autoload_model else None
                # Pass pre-loaded provider to avoid duplicate SLNC load (~6s)
                preloaded = getattr(server_state, 'provider', None)
                setup_providers(
                    slonet_hf_id=server_state.model_type,
                    slonet_provider=preloaded,
                    model_registry=registry,
                    quantize=cfg.quantize_slonet,
                    quant_bits=cfg.quant_bits,
                    quant_mode=cfg.quant_mode,
                )
                # Sync current soul traits to PersonalityProcessor
                try:
                    from domains.inference.slo_manager import get_slo_manager
                    from domains.models.provider import update_personality_traits
                    mgr = get_slo_manager()
                    current = mgr.get_current_soul()
                    if current and hasattr(current, "personality") and current.personality:
                        update_personality_traits(current.personality)
                except Exception as e:
                    logger.debug("Failed to sync soul traits to personality processor: %s", e, extra={"tag": "START"})
                logger.info("Model loaded + providers registered: %s", server_state.model_type, extra={"tag": "START"})
            except Exception as e:
                logger.error("Post-load registration failed: %s", e, exc_info=True, extra={"tag": "START"})

        # Fire-and-forget: model loads in background while routers register
        task = asyncio.create_task(asyncio.to_thread(_load_and_register))
        task.add_done_callback(self._on_model_load_done)

    def _on_model_load_done(self, task: asyncio.Task):
        try:
            task.result()
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
        """
        STARTUP_PHASE.update(phase="registering_routers", step=8, total=9, message="Registering routes...")
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
        except Exception as e:
            self._routers_registered = False
            logger.error("Phase: router registration failed: %s", e, extra={"tag": "START"})
            raise

    async def _phase_task_queue(self):
        """Initialize the background task queue."""
        STARTUP_PHASE.update(phase="task_queue", step=2, total=9, message="Initializing task queue...")
        try:
            from domains.infrastructure.task_queue import get_task_queue
            self._task_queue = get_task_queue()
            logger.info("Task queue initialized", extra={"tag": "START"})
        except Exception as e:
            logger.warning("Task queue init failed: %s", e, extra={"tag": "START"})

    async def _phase_config(self):
        """Validate and warm the config system."""
        STARTUP_PHASE.update(phase="config", step=3, total=9, message="Validating config...")
        try:
            from domains.infrastructure.config import get_config
            cfg = get_config()
            _ = cfg.model.name
            logger.info("Config system validated", extra={"tag": "START"})
        except Exception as e:
            logger.warning("Config system init: %s", e, extra={"tag": "START"})

    async def _phase_ready(self):
        """Mark server as ready — happens after all synchronous phases complete."""
        STARTUP_PHASE.update(phase="ready", step=9, total=9, message="Server ready")
        logger.info("Startup complete — server ready for requests", extra={"tag": "START"})

    # ── Shutdown hooks ──

    async def _shutdown_task_queue(self):
        """Gracefully stop the background task queue."""
        if self._task_queue is not None:
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
        await self._shutdown_task_queue()
        await self._shutdown_jobs()
        await self._shutdown_wandb()
        await self._shutdown_registry()
        await self._shutdown_pool()
        await self._shutdown_executor()


def _autoload_model(cfg: ServerConfig):
    """Load model weights into server_state. Registration handled by caller."""
    import state as server_state

    if server_state.model is not None:
        return

    from domains.infrastructure.model_loader import ModelLoader

    loader = ModelLoader()
    result = loader.load(
        model_id=cfg.autoload_model,
        device=cfg.autoload_device,
        quantize=cfg.quantize_slonet,
        quant_bits=cfg.quant_bits,
        quant_mode=cfg.quant_mode,
        verify=True,
    )

    if not result.success:
        logger.warning("Autoload failed: %s", result.error, extra={"tag": "START"})
        return

    # Store model references in server state
    server_state.model = result.model
    server_state.model_type = result.model_id
    if result.tokenizer is not None:
        server_state.tokenizer = result.tokenizer
    # Store the provider to avoid re-loading SLNC in setup_providers
    if result.provider is not None:
        server_state.provider = result.provider

    logger.info("Autoload ok: %s (%s)", cfg.autoload_model, result.model_type, extra={"tag": "START"})
