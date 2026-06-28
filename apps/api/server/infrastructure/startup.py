"""
Startup orchestrator — phased initialization of all server subsystems.

Manages the 6-phase startup with proper error isolation so one
subsystem's failure never crashes the entire server.

Integrates with LifecycleManager for phase state machine, health gates,
graceful drain, and EventBus integration.

Phases:
  1. Logging setup (async, sequential)
  2. Model load (background)
  3. W&B metrics server (background)
  4. Multimodal engine (background)
  5. Model registry (async, sequential)
  6. Router registration (async, sequential)
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from fastapi import FastAPI

from config import ServerConfig
from startup_progress import STARTUP_PHASE

logger = logging.getLogger("man.startup")


class StartupOrchestrator:
    """Phased server initialization with LifecycleManager integration.

    Each phase is registered as a startup hook on the lifecycle manager.
    Background phases (model load, W&B, multimodal) are non-critical —
    a failure there doesn't block the server from starting.

    On shutdown, the lifecycle manager drains in-flight requests before
    running cleanup hooks.
    """

    def __init__(self, app: FastAPI, config: ServerConfig):
        self._app = app
        self._config = config
        self._wandb_task: Optional[asyncio.Task] = None
        self._registry: Any = None
        self._lifecycle = None

    async def _init_lifecycle(self):
        """Lazy-init lifecycle manager with EventBus."""
        if self._lifecycle is not None:
            return
        try:
            from domains.infrastructure.event_bus import EventBus, EventPriority
            from domains.infrastructure.lifecycle import (
                LifecycleManager,
                StartupHook,
                ShutdownHook,
                get_lifecycle_manager,
            )

            bus = EventBus(max_history=200)
            self._lifecycle = get_lifecycle_manager(event_bus=bus)

            # Register startup hooks for sequential phases
            self._lifecycle.register_startup_hook(
                StartupHook("logging", self._phase1_logging, depends_on=[], timeout=5.0, critical=False),
            )
            self._lifecycle.register_startup_hook(
                StartupHook("model_registry", self._phase5_model_registry, depends_on=["logging"], timeout=10.0, critical=False),
            )
            self._lifecycle.register_startup_hook(
                StartupHook("routers", self._phase6_routers, depends_on=["model_registry"], timeout=30.0, critical=True),
            )

            # Register shutdown hooks
            self._lifecycle.register_shutdown_hook(
                ShutdownHook("job_cleanup", self._shutdown_jobs, depends_on=[], timeout=10.0),
            )
            self._lifecycle.register_shutdown_hook(
                ShutdownHook("wandb_cancel", self._shutdown_wandb, depends_on=[], timeout=5.0),
            )
            self._lifecycle.register_shutdown_hook(
                ShutdownHook("registry_cleanup", self._shutdown_registry, depends_on=[], timeout=5.0),
            )
            self._lifecycle.register_shutdown_hook(
                ShutdownHook("pool_shutdown", self._shutdown_pool, depends_on=[], timeout=10.0),
            )

            # Health gates
            self._lifecycle.register_gate("model_loaded", lambda: self._is_model_loaded())
            self._lifecycle.register_gate("routers_registered", lambda: self._routers_registered)

            logger.info("LifecycleManager initialized with event bus")
        except Exception as exc:
            logger.warning("LifecycleManager init skipped: %s", exc)

    @property
    def lifecycle(self):
        return self._lifecycle

    def _is_model_loaded(self) -> bool:
        """Check if a model is loaded (either via autoload or manually)."""
        import state as server_state
        return server_state.model is not None

    async def run(self):
        """Execute all startup phases via the lifecycle manager."""
        try:
            from domains.infrastructure.event_bus import EventBus
            bus = EventBus(max_history=200)
        except Exception:
            bus = None

        # Initialize lifecycle manager
        await self._init_lifecycle()

        # Start background phases (model load, W&B, multimodal)
        self._phase2_model_load()
        self._phase3_wandb()
        self._phase4_multimodal()

        # Register health gate for background model load
        import state as server_state
        if self._lifecycle is not None:
            self._lifecycle.register_gate(
                "model_loaded",
                lambda: self._is_model_loaded()
            )

        # Run sequential phases via lifecycle manager
        if self._lifecycle is not None:
            ok = await self._lifecycle.start(timeout=120.0)
            if not ok:
                logger.warning("Lifecycle startup incomplete — continuing anyway")
        else:
            # Fallback: run phases directly
            await self._phase1_logging()
            self._phase5_model_registry()
            self._phase6_routers()

        await self._phase_ready()

    async def _phase1_logging(self):
        STARTUP_PHASE.update(phase="initializing", step=1, total=6, message="Starting up...")
        logger.info("Startup phase 1/6: logging initialized")

    def _phase2_model_load(self):
        """Start background model load."""
        import asyncio
        from config import ServerConfig

        cfg = ServerConfig.from_env()

        raw = cfg.autoload_model
        if not raw or raw.lower() in ("false", "0", "none", "no", "off", "disable"):
            logger.info("Phase 2/6: autoload disabled (%r)", raw)
            return

        STARTUP_PHASE.update(phase="loading_model", step=2, message="Loading model weights...")
        logger.info("Phase 2/6: loading model %s in background", raw)
        self._model_load_task = asyncio.create_task(asyncio.to_thread(_autoload_model, cfg))
        self._model_load_task.add_done_callback(self._on_model_load_done)

    def _on_model_load_done(self, task: asyncio.Task):
        try:
            task.result()
        except Exception as e:
            logger.error("Model load task failed: %s", e, exc_info=True)

    def _phase3_wandb(self):
        """Start W&B metrics server (if available)."""
        STARTUP_PHASE.update(phase="wandb_server", step=3, message="Starting W&B metrics server...")
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
                    logger.info("Phase 3/6: W&B metrics server started")
                except Exception as e:
                    logger.warning("Phase 3/6: W&B server skipped: %s", e)

            asyncio.create_task(_start())
        except Exception as e:
            logger.warning("Phase 3/6: W&B unavailable: %s", e)

    def _phase4_multimodal(self):
        """Initialize multimodal engine (if available)."""
        STARTUP_PHASE.update(phase="multimodal", step=4, message="Initializing multimodal engine...")
        try:

            def _init():
                try:
                    speech = os.environ.get("SPEECH_SERVER", "").lower() in ("1", "true", "yes")
                    if speech:
                        from domains.multimodal import initialize_multimodal
                        initialize_multimodal(speech_server=True, vision_model="slonet")
                    else:
                        from domains.multimodal import get_multimodal_manager
                        get_multimodal_manager().initialize(vision_model="slonet")
                    logger.info("Phase 4/6: multimodal initialized")
                except Exception as e:
                    logger.warning("Phase 4/6: multimodal skipped: %s", e)

            asyncio.create_task(asyncio.to_thread(_init))
        except Exception as e:
            logger.warning("Phase 4/6: multimodal init failed: %s", e)

    def _phase5_model_registry(self):
        """Initialize model registry."""
        STARTUP_PHASE.update(phase="model_registry", step=5, message="Initializing model registry...")
        try:
            from domains.infrastructure.model_registry import get_model_registry
            self._registry = get_model_registry()
            logger.info("Phase 5/6: model registry initialized")
        except Exception as e:
            logger.warning("Phase 5/6: model registry failed: %s", e)

    def _phase6_routers(self):
        """Register all feature routers."""
        STARTUP_PHASE.update(phase="registering_routers", step=6, message="Registering routes...")
        try:
            from routers import get_all_routers
            for r in get_all_routers():
                self._app.include_router(r)
            try:
                from training.router import router as training_router
                self._app.include_router(training_router)
            except Exception as exc:
                logger.warning("Phase 6/6: training router failed: %s", exc)
            logger.info(
                "Phase 6/6: all routers registered (%d routes)",
                len(self._app.routes),
            )
            self._routers_registered = True
        except Exception as e:
            self._routers_registered = False
            logger.error("Phase 6/6: router registration failed: %s", e)
            raise

    async def _phase_ready(self):
        """Mark server as ready — happens after all synchronous phases complete."""
        STARTUP_PHASE.update(phase="ready", step=7, message="Server ready")
        logger.info("Startup complete — server ready for requests")

    # ── Shutdown hooks ──

    async def _shutdown_jobs(self):
        """Mark running training jobs as crashed on shutdown."""
        try:
            from training.job_store import get_job_store
            store = get_job_store()
            for job in store.list(status="running"):
                store.mark_crashed(job["id"])
                logger.info("Marked job %s as interrupted on shutdown", job["id"])
        except Exception as e:
            logger.warning("Shutdown job cleanup: %s", e)

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
        except Exception:
            pass

    async def _shutdown_pool(self):
        """Shut down inference pool."""
        try:
            from infrastructure.inference_pool import InferencePool
            pool = await InferencePool.get_instance()
            await pool.shutdown()
        except Exception:
            pass

    async def shutdown(self):
        """Clean up on server shutdown — uses lifecycle drain if available."""
        if self._lifecycle is not None:
            try:
                await self._lifecycle.shutdown(timeout=30.0)
                return
            except Exception as e:
                logger.warning("Lifecycle shutdown error: %s", e)

        # Fallback: direct cleanup
        await self._shutdown_jobs()
        await self._shutdown_wandb()
        await self._shutdown_registry()
        await self._shutdown_pool()


def _autoload_model(cfg: ServerConfig):
    """Background model loader — delegates to controller + registry."""
    import state as server_state

    if server_state.model is not None:
        return

    class _LoadRequest:
        model_id = cfg.autoload_model
        mode = "local"
        device = cfg.autoload_device

    from controllers.models import get_models_controller
    ctrl = get_models_controller()
    result = ctrl.load_model(cfg.autoload_model, cfg.autoload_device, use_slonet=cfg.use_slonet)

    if result.get("status") == "error":
        logger.warning("Autoload failed: %s", result.get("error"))
        return

    if cfg.use_slonet:
        server_state.model_type = cfg.autoload_model
        return

    model = getattr(ctrl, "_hf_model", None)
    tokenizer = getattr(ctrl, "_tokenizer", None)
    if model is None or tokenizer is None:
        logger.warning("Autoload: model loaded but refs unavailable")
        return

    server_state.model = model
    server_state.tokenizer = tokenizer
    server_state.model_type = cfg.autoload_model

    # Register with ModelRegistry for lifecycle management
    try:
        from domains.infrastructure.model_registry import get_model_registry
        registry = get_model_registry()
        registry.register(
            model_id=cfg.autoload_model,
            model=model,
            tokenizer=tokenizer,
            make_default=True,
            max_concurrent=1,
            generate_timeout=120.0,
        )
        from domains.models.provider import register_provider, HFModelProvider, ProviderRouter, VisionProcessor
        model_server = registry.get(cfg.autoload_model)
        provider = HFModelProvider(model, tokenizer, model_id_str=cfg.autoload_model, model_server=model_server)
        register_provider("hf-default", provider)

        router = ProviderRouter()
        router.add_processor(VisionProcessor("multimodal"))
        router.set_text_provider("hf-default")
        register_provider("default", router)
        logger.info("Autoload: registered with ModelRegistry + provider + default router")
    except Exception as e:
        logger.warning("Autoload: registry registration failed: %s", e)

    logger.info("Autoload ok: %s", cfg.autoload_model)
