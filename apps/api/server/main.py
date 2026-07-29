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

import state as server_state  # noqa: E402

# ── Warning suppression ──────────────────────────────────────────────
warnings.filterwarnings("ignore", category=DeprecationWarning, module="urllib3")
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

# ── Structured logging ───────────────────────────────────────────────
from domains.logging import ConsoleLogger, BridgeHandler, set_global, LogLevel  # noqa: E402

_log_level_name = os.environ.get("SLO_LOG_LEVEL", "INFO").upper()
_log_level = getattr(LogLevel, _log_level_name, LogLevel.INFO)
_log_format = os.environ.get("SLO_LOG_FORMAT", "human").lower()  # "human" or "json"

_console_logger = ConsoleLogger("slo", level=_log_level, format=_log_format)
set_global(_console_logger)

_bridge = BridgeHandler(_console_logger)
_bridge.setLevel(getattr(logging, _log_level_name, logging.INFO))
logging.root.addHandler(_bridge)
logging.root.setLevel(getattr(logging, _log_level_name, logging.INFO))

# ── Suppress noisy third-party loggers ────────────────────────────────
for _noisy in ("httpx", "httpcore", "uvicorn.access", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger("slo")

# Log bridge → output buffer (for SSE streaming via /system/stream)
try:
    from domains.infrastructure.output_buffer import install_log_bridge, install_stdio_bridge
    _buf_handler = install_log_bridge()
    install_stdio_bridge()
    logger.info("Output buffer bridge installed (handler=%s)", _buf_handler, extra={"tag": "START"})
except Exception as exc:
    logger.warning("Output buffer bridge install failed: %s", exc, extra={"tag": "START"})

# ── Filter client-side extension errors ────────────────────────────────
class _ClientExtensionFilter(logging.Filter):
    """Suppress noisy client errors from browser extensions (crypto wallets, etc.)."""
    _PATTERNS = ("CLIENT ERROR", "0 0", "chrome-extension://", "moz-extension://")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in self._PATTERNS)

logging.root.addFilter(_ClientExtensionFilter())


# ── Config ──────────────────────────────────────────────────────────
from config import GenerationConfig, ServerConfig  # noqa: E402

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
        extra={"tag": "START"},
    )
except Exception as exc:
    logger.warning("New config system unavailable: %s", exc, extra={"tag": "START"})


# ── Lifespan ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app_inst: FastAPI):
    """Delegate startup phases to ``StartupOrchestrator``."""
    try:
        from infrastructure.startup import StartupOrchestrator

        profile = os.environ.get("SLO_STARTUP_PROFILE", "full")
        orch = StartupOrchestrator(app_inst, cfg, profile=profile)
        await orch.run()

        # Start auto-trainer if SLO_AUTO_TRAIN=1
        try:
            from domains.training.auto_trainer import start_auto_trainer_if_enabled
            start_auto_trainer_if_enabled()
        except Exception as e:
            logger.warning("AutoTrainer startup failed (non-fatal): %s", e, extra={"tag": "START"})

        # Start background daemons (moved from pre-uvicorn to post-startup)
        _start_feedback_workflow()
        _start_health_monitor()
        _start_watchdog()

        yield

        # Stop auto-trainer
        try:
            from domains.training.auto_trainer import stop_auto_trainer
            stop_auto_trainer()
        except (ImportError, AttributeError) as e:
            logger.debug("Auto-trainer shutdown skipped: %s", e)

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
    allow_origins=os.environ.get("SLO_CORS_ORIGINS", "http://localhost:3000,http://localhost:8000").split(","),
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

from config import gen_config as gen_config_reexport  # noqa: E402
server_state.gen_config = gen_config_reexport

from infrastructure.auth import get_jwt_auth, get_audit_logger  # noqa: E402
jwt_auth = get_jwt_auth()
audit_logger = get_audit_logger()

# Security settings (kept as module-level globals for legacy imports)
from settings import get_security_settings  # noqa: E402
_sec = get_security_settings()
JWT_SECRET = _sec.jwt_secret
JWT_ALGORITHM = _sec.jwt_algorithm
JWT_EXPIRATION_HOURS = _sec.jwt_expiration_hours
VALID_API_KEYS = _sec.valid_api_keys



# ── Background daemons (callable from __main__) ─────────────────────
def _start_feedback_workflow() -> None:
    """Start the automated feedback workflow at server startup."""
    try:
        from domains.feedback import get_feedback_workflow

        auto_start = os.environ.get("SLO_AUTO_WORKFLOW", "true").lower() == "true"
        if not auto_start:
            logger.info("SLO_AUTO_WORKFLOW is false; skipping workflow startup", extra={"tag": "START"})
            return

        workflow = get_feedback_workflow()
        if not workflow._running:
            workflow.start()
            logger.info("Feedback workflow started automatically", extra={"tag": "START"})
    except Exception as e:
        logger.warning("Failed to start feedback workflow", extra={"context": {"error": str(e)}, "tag": "START"})


def _start_health_monitor() -> None:
    """Start the model health monitor background thread at server startup."""
    try:
        from domains.feedback.model_health import get_health_monitor

        enabled = os.environ.get("SLO_HEALTH_MONITOR", "true").lower() == "true"
        if not enabled:
            logger.info("SLO_HEALTH_MONITOR is false; skipping health monitor startup", extra={"tag": "START"})
            return

        interval = int(os.environ.get("SLO_HEALTH_INTERVAL", "300"))
        monitor = get_health_monitor()
        thread = monitor.start_auto_monitoring(interval_seconds=interval)
        thread.name = "health-monitor"
        logger.info(
            "Model health monitor started",
            extra={"context": {"interval_seconds": interval}, "tag": "START"},
        )
    except Exception as e:
        logger.warning("Failed to start health monitor", extra={"context": {"error": str(e)}, "tag": "START"})


def _start_watchdog() -> None:
    """Start the health watchdog that auto-recovers from server crashes."""
    try:
        from domains.infrastructure.watchdog import get_watchdog

        enabled = os.environ.get("SLO_WATCHDOG", "true").lower() == "true"
        if not enabled:
            logger.info("SLO_WATCHDOG is false; skipping watchdog startup", extra={"tag": "START"})
            return

        import time

        _startup_time = time.time()
        _WATCHDOG_GRACE_SECS = 120

        watchdog = get_watchdog()

        def _check_health() -> bool:
            """Quick health check — model loaded, provider registered, circuit breaker healthy."""
            try:
                if time.time() - _startup_time < _WATCHDOG_GRACE_SECS:
                    return True
                if server_state.training_active:
                    return True
                from domains.models.provider import get_provider
                router = get_provider("default")
                if router is None:
                    if os.environ.get("SLO_AUTOLOAD_MODEL", ""):
                        return False
                    return True
                # Check if the underlying ModelServer's circuit breaker is open
                server = getattr(router, '_server', None)
                if server is not None:
                    cb = getattr(server, '_circuit_breaker', None)
                    if cb is not None and cb.state.value == "open":
                        return False
                    status = getattr(server, '_status', None)
                    if status is not None and status.value == "error":
                        return False
                return True
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
                extra={"tag": "START"},
            )
            return False

        watchdog.set_health_check_fn(_check_health)
        watchdog.set_recovery_fn(_recover)
        watchdog.start(poll_interval=15, max_failures=3)
        logger.info("Health watchdog started (poll=15s, max_failures=3, recovery=log-only)", extra={"tag": "START"})
    except Exception as e:
        logger.warning("Failed to start watchdog: %s", e, extra={"tag": "START"})


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
        logger.info("Server started as daemon (PID %s)", proc.pid)
        sys.exit(0)

    # Kill orphan processes on target port to avoid port conflicts
    bind_port = args.port or cfg.port
    try:
        orphans = subprocess.check_output(["lsof", "-ti", f":{bind_port}"], timeout=5).decode().strip().split()
        for pid in orphans:
            if pid and pid != str(os.getpid()):
                os.kill(int(pid), 9)
                logger.warning("Killed orphan process %s on port %d", pid, bind_port, extra={"tag": "START"})
    except Exception:
        pass

    # Background daemons start AFTER uvicorn binds (moved from pre-uvicorn)
    # They are now started in the lifespan context below.
    logger.info("Starting SloughGPT server", extra={"context": {"port": bind_port, "reload": args.reload}, "tag": "START"})

    # Optional web frontend
    web_proc = None
    if args.web:
        web_root = _REPO_ROOT / "apps" / "web"
        standalone_dir = web_root / ".next" / "standalone"
        from domains.shared import find_available_port as _find_available_port
        web_port = _find_available_port(host="", start_port=3000)
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
        logger.info("Interrupted by user", extra={"tag": "START"})
    except Exception as e:
        logger.critical("Server crashed: %s", e, exc_info=True)
        sys.exit(1)
