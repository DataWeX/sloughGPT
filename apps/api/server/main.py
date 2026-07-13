#!/usr/bin/env python3
"""
SloughGPT Model Server — thin entry point (~240 lines).

Startup phases, middleware, exception handlers, and auth have been
extracted into the ``server/infrastructure/`` package.  This file
only handles:

  1. Python path bootstrapping (``sys.path``, ``HF_HOME``)
  2. Structured logging initialisation
  3. CORS middleware
  4. Lifespan delegating to ``StartupOrchestrator``
  5. The ``__main__`` entry point (argument parsing, uvicorn launch)

Backward-compatible re-exports are provided so existing consumers
(``from main import app``, ``from main import audit_logger`` etc.)
continue to work unchanged.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import torch
except ImportError:
    torch = None
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ── Path bootstrapping (must happen before any domain imports) ────────
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SERVER_ROOT = Path(__file__).resolve().parent
_CORE_PY_ROOT = _REPO_ROOT / "packages" / "core-py"
_SGLOADER_ROOT = _REPO_ROOT / "packages" / "downcraft"

_HF_CACHE = _REPO_ROOT / "models" / "hf-cache"
_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))

for _p in (_SERVER_ROOT, _CORE_PY_ROOT, _SGLOADER_ROOT, _REPO_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from domains.torch_runtime import apply_api_process_torch_env

apply_api_process_torch_env()

import state as server_state  # noqa: E402

# ── Warning suppression ──────────────────────────────────────────────
warnings.filterwarnings("ignore", category=DeprecationWarning, module="urllib3")
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

# ── Structured logging ───────────────────────────────────────────────
from domains.logging import ConsoleLogger, BridgeHandler, set_global, LogLevel  # noqa: E402

_log_level_name = os.environ.get("MAN_LOG_LEVEL", "INFO").upper()
_log_level = getattr(LogLevel, _log_level_name, LogLevel.INFO)

_console_logger = ConsoleLogger("man", level=_log_level)
set_global(_console_logger)

_bridge = BridgeHandler(_console_logger)
_bridge.setLevel(getattr(logging, _log_level_name, logging.INFO))
logging.root.addHandler(_bridge)
logging.root.setLevel(getattr(logging, _log_level_name, logging.INFO))

# Log bridge → output buffer (for SSE streaming via /system/stream)
try:
    from domains.infrastructure.output_buffer import install_log_bridge
    install_log_bridge()
except Exception:
    pass

logger = logging.getLogger("man")


# ── Config ──────────────────────────────────────────────────────────
from config import GenerationConfig, ServerConfig  # noqa: E402

gen_config = GenerationConfig.from_env()
server_state.gen_config = gen_config

cfg = ServerConfig.from_env()

# Wire new typed config system alongside existing config for migration
try:
    from domains.infrastructure.config import get_config, AppConfig
    _new_cfg: AppConfig = get_config()
    logger.info(
        "Config: %s @ %s:%d (features=%s)",
        _new_cfg.model.name,
        _new_cfg.server.host,
        _new_cfg.server.port,
        {k: v for k, v in _new_cfg.features.model_dump().items() if not k.startswith("_")},
    )
except Exception as exc:
    logger.warning("New config system unavailable: %s", exc)


# ── Lifespan ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app_inst: FastAPI):
    """Delegate startup phases to ``StartupOrchestrator``."""
    try:
        import os
        from infrastructure.startup import StartupOrchestrator

        profile = os.environ.get("MAN_STARTUP_PROFILE", "full")
        orch = StartupOrchestrator(app_inst, cfg, profile=profile)
        await orch.run()

        # Start auto-trainer if MAN_AUTO_TRAIN=1
        try:
            from domains.training.auto_trainer import start_auto_trainer_if_enabled
            start_auto_trainer_if_enabled()
        except Exception as e:
            logger.warning("AutoTrainer startup failed (non-fatal): %s", e)

        yield

        # Stop auto-trainer
        try:
            from domains.training.auto_trainer import stop_auto_trainer
            stop_auto_trainer()
        except Exception:
            pass

        await orch.shutdown()
    except Exception as exc:
        logger.critical("Startup failed: %s", exc, exc_info=True)
        yield
        raise


# ── FastAPI application ─────────────────────────────────────────────
app = FastAPI(
    title="SloughGPT API",
    description="SloughGPT Model Inference API with HuggingFace models",
    version="1.0.0",
    docs_url="/docs",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register structured middleware from the infrastructure package.
from infrastructure.middleware import register_all_middleware  # noqa: E402
register_all_middleware(app, request_timeout=cfg.request_timeout_seconds)

# Register health/status routes IMMEDIATELY — before lifespan runs.
# During the 25-40s model loading phase, the frontend must still get
# real responses from /health, /health/startup-progress, /health/summary
# instead of connection errors.  These lightweight routers have zero
# heavy imports (no torch, no transformers).
from routers.health import router as _health_router
from routers.status import router as _status_router
app.include_router(_health_router)
app.include_router(_status_router)
# Health/status routes are now registered pre-lifespan.
# _phase6_routers() skips them by checking existing route prefixes.

# Feature routers are registered by StartupOrchestrator._phase6_routers()
# during the lifespan context.  Tests that need routes without lifespan
# should import and register only the specific router they test.

# Register exception handlers.
from infrastructure.exception_handlers import register_all_handlers  # noqa: E402
register_all_handlers(app)


# ── Legacy helpers (preserved for existing callers) ─────────────────
def _first_trainable_device(module: Any) -> torch.device:
    """Device for placing tokenized inputs beside a loaded HF ``model``."""
    try:
        return next(module.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        return torch.device("cpu")


def _inputs_to_model_device(inputs: Any, model: Any) -> Any:
    """Move tokenizer outputs (``BatchEncoding`` or ``dict``) to the module device."""
    dev = _first_trainable_device(model)
    if dev.type == "meta":
        return inputs
    if hasattr(inputs, "to"):
        return inputs.to(dev)
    if isinstance(inputs, dict):
        return {k: v.to(dev) for k, v in inputs.items()}
    return inputs


def _inference_engine_device_str(module: Any) -> str:
    """String device for ``InferenceEngine`` — match where ``module`` parameters live."""
    if module is None:
        return "cpu"
    try:
        dev = _first_trainable_device(module)
    except Exception:
        return "cpu"
    if dev.type == "meta":
        return "cpu"
    return str(dev.type)


def load_model(model_path: Optional[str] = None):
    """Load the actual sloughgpt model. If model_path provided, load that specific checkpoint."""
    try:
        import torch

        if model_path:
            if not Path(model_path).is_absolute():
                model_path = str((_REPO_ROOT / model_path).resolve())
        else:
            default_paths = [
                "models/sloughgpt_finetuned.pt",
                "models/sloughgpt_lora.pt",
                "models/sloughgpt_variant.pt",
            ]
            model_path = None
            for path in default_paths:
                if (_REPO_ROOT / path).exists():
                    model_path = str((_REPO_ROOT / path).resolve())
                    break

        if model_path is None or not Path(model_path).exists():
            server_state.model = None
            server_state.model_type = "demo"
            logger.info("No model found, demo mode active")
            return

        logger.info("Loading model", extra={"context": {"model_path": model_path}})
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict):
            server_state.model = ckpt
            server_state.model_type = "sloughgpt_finetuned"
            if "chars" in ckpt and "stoi" in ckpt and "itos" in ckpt:
                server_state.tokenizer = {
                    "chars": ckpt["chars"],
                    "stoi": ckpt["stoi"],
                    "itos": ckpt["itos"],
                    "vocab_size": len(ckpt["chars"]),
                }
                logger.info("Tokenizer loaded", extra={"context": {"char_count": len(ckpt['chars'])}})
            else:
                server_state.tokenizer = None
            logger.info("Model loaded", extra={"context": {"param_count": len(ckpt.get('model', {}))}})
        else:
            server_state.model = ckpt
            server_state.model_type = "sloughgpt_finetuned"
            server_state.tokenizer = None
            logger.info("Model loaded successfully", extra={"context": {"model_path": model_path}})

    except Exception as e:
        logger.warning("Failed to load model", extra={"context": {"error": str(e)}})
        import traceback
        traceback.print_exc()
        server_state.model = None
        server_state.tokenizer = None
        server_state.model_type = "demo"


# ── Meta-weight manager ─────────────────────────────────────────────
_meta_weight_manager = None


def get_meta_weight_manager():
    """Lazy load meta-weight manager to avoid import issues."""
    global _meta_weight_manager
    if _meta_weight_manager is None:
        try:
            from domains.feedback import get_meta_weight_manager as _get_manager
            _meta_weight_manager = _get_manager()
        except ImportError:
            return None
    return _meta_weight_manager


# ── Backward-compatible re-exports ─────────────────────────────────
# Consumers still import from main.py (e.g. ``from main import audit_logger``).
# These symbols used to be defined inline but now live in the ``server/infrastructure/``
# package.  The re-exports below ensure zero-changes for existing callers.

from config import gen_config as gen_config_reexport  # noqa: E402, F401
gen_config = gen_config_reexport
server_state.gen_config = gen_config

from infrastructure.auth import JWTAuth, AuditLogger, get_jwt_auth, get_audit_logger  # noqa: E402
jwt_auth = get_jwt_auth()
audit_logger = get_audit_logger()

# Security settings (kept as module-level globals for legacy imports)
from settings import get_security_settings  # noqa: E402
_sec = get_security_settings()
JWT_SECRET = _sec.jwt_secret
JWT_ALGORITHM = _sec.jwt_algorithm
JWT_EXPIRATION_HOURS = _sec.jwt_expiration_hours
VALID_API_KEYS = _sec.valid_api_keys

from pydantic import BaseModel  # noqa: E402


class LoadModelRequest(BaseModel):
    """Request schema for ``POST /models/load`` — kept here for import compatibility."""

    model_id: str
    mode: Optional[str] = "local"
    device: Optional[str] = "auto"


def _load_hf_model_core(request: LoadModelRequest, use_slonet: bool = False) -> Dict[str, Any]:
    """
    Load HuggingFace model via ModelsController (extracted into
    ``routers/models.py`` endpoint).  Preserved here for startup import.
    """
    try:
        from controllers.models import get_models_controller

        ctrl = get_models_controller()
        result = ctrl.load_model(request.model_id, request.device or "auto", use_slonet=use_slonet)

        if result.get("status") == "error":
            logger.warning("Failed to load model %s: %s", request.model_id, result.get("error"))
            return result

        if use_slonet:
            server_state.model_type = request.model_id
            return result

        model = getattr(ctrl, "_hf_model", None)
        tokenizer = getattr(ctrl, "_tokenizer", None)

        if model is None or tokenizer is None:
            return {"status": "error", "error": "Model loaded but model/tokenizer not available"}

        server_state.model = model
        server_state.tokenizer = tokenizer
        server_state.model_type = request.model_id

        # Auto-load knowledge adapter
        try:
            from domains.infrastructure.knowledge_weight_integrator import load_knowledge_adapter, get_adapter_status

            status = get_adapter_status()
            if status.get("adapter_exists"):
                model = load_knowledge_adapter(model, device="cpu", merge=True)
                server_state.model = model
                logger.info("Knowledge adapter merged (%d facts)", status.get("fact_count", 0))
        except Exception as e:
            logger.warning("Knowledge adapter load skipped: %s", e)

        # Optionally wrap in ProcessGuard for crash isolation
        process_guard = None
        if os.environ.get("MAN_ENABLE_PROCESS_GUARD", "").lower() in ("1", "true", "yes"):
            try:
                from domains.infrastructure.process_guard import create_model_guard
                process_guard = create_model_guard(
                    model_id=request.model_id,
                    device=request.device or "cpu",
                    max_restarts=3,
                    restart_delay=2.0,
                    memory_limit_mb=4096,
                )
                logger.info("_load_hf_model_core: ProcessGuard started for %s", request.model_id)
            except Exception as e:
                logger.warning("_load_hf_model_core: ProcessGuard init failed (without): %s", e)

        # Register with ModelRegistry
        try:
            from domains.infrastructure.model_registry import get_model_registry

            registry = get_model_registry()
            registry.register(
                model_id=request.model_id,
                model=model,
                tokenizer=tokenizer,
                make_default=True,
                max_concurrent=1,
                generate_timeout=120.0,
                process_guard=process_guard,
            )
            from domains.models.provider import register_provider, HFModelProvider, ProviderRouter, VisionProcessor

            model_server = registry.get(request.model_id)
            provider = HFModelProvider(
                model, tokenizer,
                model_id_str=request.model_id,
                model_server=model_server,
            )
            register_provider("hf-default", provider)
            logger.info("hf-default provider re-registered with ModelServer: %s", request.model_id)

            # Wire default provider router — but don't override SloNet if already active.
            # SloNet is registered by auto_train as "default"; replacing it would
            # cause chat to silently fall back to HF and produce empty responses.
            from domains.models.provider import get_provider as _gp
            existing = _gp("default")
            _is_slonet = existing is not None and type(existing).__name__ in ("SloTransformerProvider", "SloNetChatProvider")
            if not _is_slonet:
                router = ProviderRouter()
                router.add_processor(VisionProcessor("multimodal"))
                router.set_text_provider("hf-default")
                register_provider("default", router)
                logger.info("Default provider router registered with VisionProcessor")
            else:
                logger.info("SloNet provider active — keeping as default (skipping HF override)")
        except Exception as e:
            logger.warning("Failed to register with ModelRegistry: %s", e)

        effective = _inference_engine_device_str(model)
        return {
            "status": "loaded",
            "model": request.model_id,
            "mode": request.mode or "local",
            "device": result.get("device", request.device),
            "effective_device": effective,
            "model_type": server_state.model_type or request.model_id,
        }
    except Exception as e:
        logger.error("_load_hf_model_core failed: %s", e, exc_info=True)
        return {"status": "error", "error": str(e)}


def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port (DEPRECATED - use domains.shared.find_available_port)."""
    from domains.shared import find_available_port as _find_available_port
    return _find_available_port(host="", start_port=start_port, max_attempts=max_attempts)


# ── Background daemons (callable from __main__) ─────────────────────
def _start_feedback_workflow() -> None:
    """Start the automated feedback workflow at server startup."""
    try:
        from domains.feedback import get_feedback_workflow

        auto_start = os.environ.get("MAN_AUTO_WORKFLOW", "true").lower() == "true"
        if not auto_start:
            logger.info("MAN_AUTO_WORKFLOW is false; skipping workflow startup")
            return

        workflow = get_feedback_workflow()
        if not workflow._running:
            workflow.start()
            logger.info("Feedback workflow started automatically")
    except Exception as e:
        logger.warning("Failed to start feedback workflow", extra={"context": {"error": str(e)}})


def _start_health_monitor() -> None:
    """Start the model health monitor background thread at server startup."""
    try:
        from domains.feedback.model_health import get_health_monitor

        enabled = os.environ.get("MAN_HEALTH_MONITOR", "true").lower() == "true"
        if not enabled:
            logger.info("MAN_HEALTH_MONITOR is false; skipping health monitor startup")
            return

        interval = int(os.environ.get("MAN_HEALTH_INTERVAL", "300"))
        monitor = get_health_monitor()
        thread = monitor.start_auto_monitoring(interval_seconds=interval)
        thread.name = "health-monitor"
        logger.info(
            "Model health monitor started",
            extra={"context": {"interval_seconds": interval}},
        )
    except Exception as e:
        logger.warning("Failed to start health monitor", extra={"context": {"error": str(e)}})


def _start_watchdog() -> None:
    """Start the health watchdog that auto-recovers from server crashes."""
    try:
        from domains.infrastructure.watchdog import get_watchdog

        enabled = os.environ.get("MAN_WATCHDOG", "true").lower() == "true"
        if not enabled:
            logger.info("MAN_WATCHDOG is false; skipping watchdog startup")
            return

        import time

        _startup_time = time.time()
        _WATCHDOG_GRACE_SECS = 120

        watchdog = get_watchdog()

        def _check_health() -> bool:
            """Quick health check — model loaded and providers registered."""
            try:
                if time.time() - _startup_time < _WATCHDOG_GRACE_SECS:
                    return True
                if server_state.training_active:
                    return True
                if server_state.model is None:
                    from domains.models.provider import get_provider

                    default = get_provider("default")
                    if default is not None:
                        return True
                    if os.environ.get("MAN_AUTOLOAD_MODEL", ""):
                        return False
                    return True
                from domains.models.provider import get_provider

                router = get_provider("default")
                return router is not None
            except Exception:
                return False

        def _recover() -> bool:
            """Log-only recovery — model reload from a watchdog thread is dangerous
            (memory pressure, provider state corruption, 45s+ blocking on GIL/I/O).
            If recovery is truly needed, call /models/load manually.
            """
            logger.warning(
                "Watchdog detected %d consecutive health failures. "
                "Manual recovery required — model inference may be degraded.",
                watchdog._consecutive_failures,
            )
            return False

        watchdog.set_health_check_fn(_check_health)
        watchdog.set_recovery_fn(_recover)
        watchdog.start(poll_interval=15, max_failures=3)
        logger.info("Health watchdog started (poll=15s, max_failures=3, recovery=log-only)")
    except Exception as e:
        logger.warning("Failed to start watchdog: %s", e)


# ── Entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import atexit
    import signal
    import subprocess
    import uvicorn

    # Ignore SIGHUP so server survives shell session close
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    def _handle_uncaught_exception(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, (KeyboardInterrupt, SystemExit)):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.critical("Unhandled exception: %s", exc_value, exc_info=(exc_type, exc_value, exc_tb))

    sys.excepthook = _handle_uncaught_exception

    parser = argparse.ArgumentParser(description="SloughGPT API Server")
    parser.add_argument(
        "--reload",
        action="store_true",
        default=cfg.reload,
        help="Enable auto-reload on file changes",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind (default: from config or 8000)",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        default=cfg.enable_web,
        help="Serve web frontend alongside API",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        default=False,
        help="Run as daemon (detach from terminal)",
    )
    args = parser.parse_args()

    # Daemonize if requested — use subprocess to avoid fork() issues
    if args.daemon:
        import subprocess as sp
        cmd = [sys.executable, __file__]
        if args.port:
            cmd += ["--port", str(args.port)]
        proc = sp.Popen(
            cmd,
            stdin=sp.DEVNULL,
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
            start_new_session=True,
        )
        print(f"Server started as daemon (PID {proc.pid})")
        sys.exit(0)

    # Kill orphan processes on target port to avoid port conflicts
    bind_port = args.port or cfg.port
    try:
        orphans = subprocess.check_output(["lsof", "-ti", f":{bind_port}"], timeout=5).decode().strip().split()
        for pid in orphans:
            if pid and pid != str(os.getpid()):
                os.kill(int(pid), 9)
                logger.warning("Killed orphan process %s on port %d", pid, bind_port)
    except Exception:
        pass

    _start_feedback_workflow()
    _start_health_monitor()
    _start_watchdog()
    logger.info("Starting SloughGPT server", extra={"context": {"port": bind_port, "reload": args.reload}})

    # Optional web frontend
    web_proc = None
    if args.web:
        web_root = _REPO_ROOT / "apps" / "web"
        standalone_dir = web_root / ".next" / "standalone"
        web_port = find_available_port(3000)
        web_env = {**os.environ, "PORT": str(web_port)}

        if standalone_dir.is_dir() and (standalone_dir / "server.js").is_file():
            web_proc = subprocess.Popen(
                ["node", "server.js"],
                cwd=str(standalone_dir),
                env=web_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        else:
            web_proc = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(web_root),
                env=web_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

    if web_proc:
        atexit.register(lambda p=web_proc: (p.terminate(), p.wait(timeout=5)) if p.poll() is None else None)

    uvicorn_kw: dict = dict(
        app=app,
        host=cfg.host,
        port=bind_port,
        log_level=cfg.log_level.lower(),
    )
    if args.reload:
        uvicorn_kw["reload"] = True
        uvicorn_kw["app"] = "main:app"
        uvicorn_kw["reload_includes"] = [
            "apps/api/server/**/*.py",
            "packages/core-py/domains/**/*.py",
        ]
        uvicorn_kw["reload_excludes"] = [
            ".*/**", "node_modules/**", "__pycache__/**", "*.pyc",
            ".git/**", ".venv/**", "venv/**", "env/**",
            "build/**", "dist/**", ".next/**", "data/**", "datasets/**", "models/**",
            "tests/**", "logs/**", "checkpoints/**",
            "apps/web/**", "apps/cli/**",
        ]

    try:
        uvicorn.run(**uvicorn_kw)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.critical("Server crashed: %s", e, exc_info=True)
        sys.exit(1)
