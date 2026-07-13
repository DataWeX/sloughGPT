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

logger = logging.getLogger("man.startup")


class StartupProfileSelector:
    """Helper to resolve the active startup profile from config or env."""

    @staticmethod
    def resolve(config: ServerConfig) -> str:
        """Return profile name: env var > config attribute > default."""
        raw = os.environ.get("MAN_STARTUP_PROFILE", "")
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
            from domains.infrastructure.event_bus import EventBus, EventPriority
            from domains.infrastructure.lifecycle import (
                ALL_PROFILES,
                LifecycleManager,
                StartupHook,
                StartupProfile,
                ShutdownHook,
                get_lifecycle_manager,
            )

            # Resolve profile enum
            try:
                profile_enum = StartupProfile(self._profile)
            except ValueError:
                profile_enum = StartupProfile.FULL

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
                    depends_on=[], timeout=10.0, critical=False,
                    profiles=quick_plus,
                ),
            )
            self._lifecycle.register_startup_hook(
                StartupHook(
                    "config", self._phase_config,
                    depends_on=[], timeout=5.0, critical=False,
                    profiles=quick_plus,
                ),
            )
            self._lifecycle.register_startup_hook(
                StartupHook(
                    "model_load", self._phase2_model_load,
                    depends_on=[], timeout=120.0, critical=False,
                    profiles=full_only,
                ),
            )
            self._lifecycle.register_startup_hook(
                StartupHook(
                    "wandb", self._phase3_wandb,
                    depends_on=[], timeout=30.0, critical=False,
                    profiles=full_only,
                ),
            )
            self._lifecycle.register_startup_hook(
                StartupHook(
                    "multimodal", self._phase4_multimodal,
                    depends_on=[], timeout=30.0, critical=False,
                    profiles=full_only,
                ),
            )
            self._lifecycle.register_startup_hook(
                StartupHook(
                    "model_registry", self._phase5_model_registry,
                    depends_on=["task_queue", "config"],
                    timeout=10.0, critical=False,
                    profiles=quick_plus,
                ),
            )
            self._lifecycle.register_startup_hook(
                StartupHook(
                    "routers", self._phase6_routers,
                    depends_on=["model_registry"], timeout=30.0, critical=True,
                    profiles=all_profiles,
                ),
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
                ShutdownHook("task_queue_shutdown", self._shutdown_task_queue, depends_on=[], timeout=10.0),
            )
            self._lifecycle.register_shutdown_hook(
                ShutdownHook("pool_shutdown", self._shutdown_pool, depends_on=[], timeout=10.0),
            )

            # Health gates
            self._lifecycle.register_gate("model_loaded", lambda: self._is_model_loaded())
            self._lifecycle.register_gate("routers_registered", lambda: self._routers_registered)

            logger.info(
                "LifecycleManager initialized (profile=%s, %d startup hooks, %d shutdown hooks)",
                profile_enum.value,
                len(self._lifecycle._startup_hooks),
                len(self._lifecycle._shutdown_hooks),
            )
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

        from domains.infrastructure.lifecycle import StartupProfile

        try:
            profile_enum = StartupProfile(self._profile)
        except ValueError:
            profile_enum = StartupProfile.FULL

        # Register health gate for background model load
        import state as server_state
        if self._lifecycle is not None:
            self._lifecycle.register_gate(
                "model_loaded",
                lambda: self._is_model_loaded()
            )

        # Run sequential phases via lifecycle manager
        if self._lifecycle is not None:
            ok = await self._lifecycle.start(timeout=120.0, profile=profile_enum)
            if not ok:
                logger.warning("Lifecycle startup incomplete — continuing anyway")
        else:
            # Fallback: run phases directly
            self._phase5_model_registry()
            self._phase6_routers()

        await self._phase_ready()

    async def _phase2_model_load(self):
        """Start background model load."""
        import asyncio
        from config import ServerConfig

        cfg = ServerConfig.from_env()

        raw = cfg.autoload_model
        if not raw or raw.lower() in ("false", "0", "none", "no", "off", "disable"):
            logger.info("Phase: autoload disabled (%r)", raw)
            return

        STARTUP_PHASE.update(phase="loading_model", step=4, total=8, message="Loading model weights...")
        logger.info("Phase 4: loading model %s", raw)
        # Run model load synchronously so provider is registered before serving requests
        try:
            await asyncio.to_thread(_autoload_model, cfg)
        except Exception as e:
            logger.error("Model load failed: %s", e, exc_info=True)

    def _on_model_load_done(self, task: asyncio.Task):
        try:
            task.result()
        except Exception as e:
            logger.error("Model load task failed: %s", e, exc_info=True)

    async def _phase3_wandb(self):
        """Start W&B metrics server (if available)."""
        STARTUP_PHASE.update(phase="wandb_server", step=5, total=8, message="Starting W&B metrics server...")
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
                    logger.info("Phase: W&B metrics server started")
                except Exception as e:
                    logger.warning("Phase: W&B server skipped: %s", e)

            asyncio.create_task(_start())
        except Exception as e:
            logger.warning("Phase: W&B unavailable: %s", e)

    async def _phase4_multimodal(self):
        """Initialize multimodal engine (if available)."""
        STARTUP_PHASE.update(phase="multimodal", step=6, total=8, message="Initializing multimodal engine...")
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

    async def _phase5_model_registry(self):
        """Initialize model registry."""
        STARTUP_PHASE.update(phase="model_registry", step=7, total=8, message="Initializing model registry...")
        try:
            from domains.infrastructure.model_registry import get_model_registry
            self._registry = get_model_registry()
            logger.info("Phase: model registry initialized")
        except Exception as e:
            logger.warning("Phase: model registry failed: %s", e)

    async def _phase6_routers(self):
        """Register all feature routers."""
        STARTUP_PHASE.update(phase="registering_routers", step=8, total=8, message="Registering routes...")
        try:
            from routers import get_all_routers
            for r in get_all_routers():
                self._app.include_router(r)
            try:
                from training.router import router as training_router
                self._app.include_router(training_router)
            except Exception as exc:
                logger.warning("Phase: training router failed: %s", exc)
            logger.info(
                "Phase: all routers registered (%d routes)",
                len(self._app.routes),
            )
            self._routers_registered = True
        except Exception as e:
            self._routers_registered = False
            logger.error("Phase: router registration failed: %s", e)
            raise

    async def _phase_task_queue(self):
        """Initialize the background task queue."""
        STARTUP_PHASE.update(phase="task_queue", step=2, total=8, message="Initializing task queue...")
        try:
            from domains.infrastructure.task_queue import TaskQueue, get_task_queue
            self._task_queue = get_task_queue()
            logger.info("Task queue initialized")
        except Exception as e:
            logger.warning("Task queue init failed: %s", e)

    async def _phase_config(self):
        """Validate and warm the config system."""
        STARTUP_PHASE.update(phase="config", step=3, total=8, message="Validating config...")
        try:
            from domains.infrastructure.config import Config
            cfg = Config.get_instance()
            _ = cfg.get("app.name", "sloughgpt")
            logger.info("Config system validated")
        except Exception as e:
            logger.warning("Config system init: %s", e)

    async def _phase_ready(self):
        """Mark server as ready — happens after all synchronous phases complete."""
        STARTUP_PHASE.update(phase="ready", step=9, total=8, message="Server ready")
        logger.info("Startup complete — server ready for requests")

    # ── Shutdown hooks ──

    async def _shutdown_task_queue(self):
        """Gracefully stop the background task queue."""
        if self._task_queue is not None:
            try:
                await self._task_queue.stop()
                logger.info("Task queue stopped")
            except Exception as e:
                logger.warning("Task queue shutdown: %s", e)

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

    async def _shutdown_executor(self):
        """Gracefully shut down the TrainingExecutor thread pool."""
        try:
            from domains.training.executor import _instance
            if _instance is not None:
                _instance.shutdown(wait=True)
                logger.info("TrainingExecutor shut down")
        except Exception as e:
            logger.warning("TrainingExecutor shutdown: %s", e)

    async def shutdown(self):
        """Clean up on server shutdown — uses lifecycle drain if available."""
        if self._lifecycle is not None:
            try:
                await self._lifecycle.shutdown(timeout=30.0)
                return
            except Exception as e:
                logger.warning("Lifecycle shutdown error: %s", e)

        # Fallback: direct cleanup
        await self._shutdown_task_queue()
        await self._shutdown_jobs()
        await self._shutdown_wandb()
        await self._shutdown_registry()
        await self._shutdown_pool()
        await self._shutdown_executor()


def _autoload_model(cfg: ServerConfig):
    """Background model loader — uses unified ModelLoader."""
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
        logger.warning("Autoload failed: %s", result.error)
        return

    # Store model references in server state
    server_state.model = result.model
    server_state.model_type = result.model_id
    if result.tokenizer is not None:
        server_state.tokenizer = result.tokenizer

    # Register with ModelRegistry (creates ModelServer with SessionKVCache)
    from domains.infrastructure.model_registry import get_model_registry
    registry = get_model_registry()
    if result.model is not None and result.tokenizer is not None:
        registry.register(
            result.model_id, result.model, result.tokenizer,
            make_default=True, generate_timeout=120.0,
        )

    # Create InferenceEngine for KV-cache-powered generation
    inference_engine = None
    if result.model is not None and result.tokenizer is not None and result.model_type != "slonet":
        try:
            from domains.inference.engine import InferenceEngine
            inference_engine = InferenceEngine(
                model=result.model, tokenizer=result.tokenizer, device="cpu",
            )
        except Exception as e:
            logger.warning("Failed to create InferenceEngine: %s", e)

    # Register providers via setup_providers (handles hf-default + inference-engine + default router)
    from domains.models.provider import setup_providers

    # GGUF quantized inference — enabled via MAN_GGUF_MODEL env var
    gguf_model = os.environ.get("MAN_GGUF_MODEL") or None

    setup_providers(
        hf_model=result.model,
        hf_tokenizer=result.tokenizer,
        hf_model_id=result.model_id,
        inference_engine=inference_engine,
        model_registry=registry,
        gguf_model=gguf_model,
    )

    if gguf_model:
        logger.info("GGUF quantized inference enabled: %s", gguf_model)
    else:
        logger.info("Autoload ok: %s (%s) — KV cache active (ModelServer + InferenceEngine)",
                    cfg.autoload_model, result.model_type)
