"""
Startup orchestrator — phased initialization of all server subsystems.

Manages the 6-phase startup with proper error isolation so one
subsystem's failure never crashes the entire server.

Phases:
  1. Logging setup
  2. Model load (background)
  3. W&B metrics server (background)
  4. Multimodal engine (background)
  5. Model registry
  6. Router registration
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from fastapi import FastAPI

from config import ServerConfig

logger = logging.getLogger("man.startup")


class StartupOrchestrator:
    """Phased server initialization with error isolation per subsystem."""

    def __init__(self, app: FastAPI, config: ServerConfig):
        self._app = app
        self._config = config
        self._wandb_task: Optional[asyncio.Task] = None
        self._registry: Any = None

    async def run(self):
        """Execute all startup phases."""
        await self._phase1_logging()
        self._phase2_model_load()
        self._phase3_wandb()
        self._phase4_multimodal()
        self._phase5_model_registry()
        self._phase6_routers()

    async def _phase1_logging(self):
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

        logger.info("Phase 2/6: loading model %s in background", raw)
        asyncio.create_task(asyncio.to_thread(_autoload_model, cfg))

    def _phase3_wandb(self):
        """Start W&B metrics server (if available)."""
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
        try:
            from domains.infrastructure.model_registry import get_model_registry
            self._registry = get_model_registry()
            logger.info("Phase 5/6: model registry initialized")
        except Exception as e:
            logger.warning("Phase 5/6: model registry failed: %s", e)

    def _phase6_routers(self):
        """Register all feature routers."""
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
        except Exception as e:
            logger.error("Phase 6/6: router registration failed: %s", e)
            raise

    async def shutdown(self):
        """Clean up on server shutdown."""
        try:
            from training.job_store import get_job_store
            store = get_job_store()
            for job in store.list(status="running"):
                store.mark_crashed(job["id"])
                logger.info("Marked job %s as interrupted on shutdown", job["id"])
        except Exception as e:
            logger.warning("Shutdown job cleanup: %s", e)

        if self._wandb_task is not None:
            self._wandb_task.cancel()
            try:
                await self._wandb_task
            except asyncio.CancelledError:
                pass

        try:
            from domains.infrastructure.model_registry import get_model_registry
            get_model_registry().reset_metrics()
        except Exception:
            pass

        try:
            from infrastructure.inference_pool import InferencePool
            pool = await InferencePool.get_instance()
            await pool.shutdown()
        except Exception:
            pass


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
        from domains.models.provider import register_provider, HFModelProvider
        model_server = registry.get(cfg.autoload_model)
        provider = HFModelProvider(model, tokenizer, model_id_str=cfg.autoload_model, model_server=model_server)
        register_provider("hf-default", provider)
        logger.info("Autoload: registered with ModelRegistry + provider")
    except Exception as e:
        logger.warning("Autoload: registry registration failed: %s", e)

    logger.info("Autoload ok: %s", cfg.autoload_model)
