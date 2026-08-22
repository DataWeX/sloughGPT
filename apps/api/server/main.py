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

import faulthandler
import logging
import os
import sys
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from domains.shared import find_repo_root

# ── Path bootstrapping (must happen before any domain imports) ────────
_REPO_ROOT = find_repo_root(Path(__file__).resolve())
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

# ── Structured logging (centralized) ─────────────────────────────────
from domains.logging.config import setup_logging  # noqa: E402

_log_setup = setup_logging()
logger = logging.getLogger("slo")
logger.info(
    "Logging: level=%s format=%s log_dir=%s",
    _log_setup["level"], _log_setup["format"], _log_setup["log_dir"],
    extra={"tag": "START"},
)


# ── Config ──────────────────────────────────────────────────────────
from config import ServerConfig  # noqa: E402

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
_pgq_engine = None  # PGQ core infra engine instance

@asynccontextmanager
async def lifespan(app_inst: FastAPI):
    """Delegate startup phases to ``StartupOrchestrator``."""
    global _pgq_engine
    _install_stack_dump_timer()
    try:
        from infrastructure.startup import StartupOrchestrator

        profile = os.environ.get("SLO_STARTUP_PROFILE", "full")
        orch = StartupOrchestrator(app_inst, cfg, profile=profile)
        await orch.run()

        # Start PGQ core infra engine (background thread)
        try:
            from domains.infrastructure.pugqeep import PGQ
            _pgq_engine = PGQ("sloughgpt")
            logger.info("PGQ core engine created", extra={"tag": "START"})
        except Exception as e:
            logger.warning("PGQ engine creation failed (non-fatal): %s", e, extra={"tag": "START"})

        # Start auto-trainer if SLO_AUTO_TRAIN=1
        try:
            from domains.training.auto_trainer import start_auto_trainer_if_enabled
            start_auto_trainer_if_enabled()
        except Exception as e:
            logger.warning("AutoTrainer startup failed (non-fatal): %s", e, extra={"tag": "START"})

        # Auto-ingest repo docs into production RAG (if empty)
        # Moved to background thread: importing rag_service chains through
        # rag.py → HybridRetriever and blocks the lifespan.
        import threading
        def _rag_init_and_ingest():
            try:
                from domains.cognitive.rag_service import get_rag_service
                _rag = get_rag_service()
                if _rag.stats().get("total_chunks", 0) == 0:
                    try:
                        _rag.auto_ingest_directory(str(find_repo_root(Path(__file__).resolve())), max_files=150)
                    except Exception as e:
                        logger.debug("RAG auto-ingest failed: %s", e)
            except Exception as e:
                logger.debug("RAG init skipped: %s", e)
        threading.Thread(target=_rag_init_and_ingest, daemon=True, name="rag-init").start()

        # Start background daemons (moved from pre-uvicorn to post-startup)
        _start_feedback_workflow()
        _start_health_monitor()
        _start_watchdog()

        # Start idle manager if configured
        if cfg.idle_timeout_seconds > 0:
            try:
                from domains.infrastructure.model_server import get_idle_manager
                idle_mgr = get_idle_manager()
                idle_mgr._idle_timeout_s = cfg.idle_timeout_seconds
                logger.info(
                    "Idle manager active: timeout=%ss", cfg.idle_timeout_seconds,
                    extra={"tag": "IDLE"},
                )
            except Exception as e:
                logger.warning("Idle manager startup failed (non-fatal): %s", e)

        yield

        # Stop idle manager
        try:
            from domains.infrastructure.model_server import get_idle_manager
            get_idle_manager().shutdown()
        except Exception:
            pass

        # Stop auto-trainer
        try:
            from domains.training.auto_trainer import stop_auto_trainer
            stop_auto_trainer()
        except (ImportError, AttributeError) as e:
            logger.debug("Auto-trainer shutdown skipped: %s", e)

        # Stop PGQ engine
        if _pgq_engine is not None:
            try:
                _pgq_engine.stop()
                logger.info("PGQ engine stopped", extra={"tag": "START"})
            except Exception as e:
                logger.debug("PGQ engine shutdown skipped: %s", e)

        await orch.shutdown()
    except Exception as exc:
        logger.critical("Startup failed: %s", exc, exc_info=True)
        yield
        raise
    finally:
        _cancel_stack_dump_timer()


def _install_stack_dump_timer() -> None:
    """Periodically dump all thread stacks to stderr when ``SLO_DUMP_STACKS=1``.

    Hang diagnosis aid: when a request stalls, the periodic dump shows which
    thread and source line is blocked (e.g. a sync call hogging the event
    loop). Off by default; interval configurable via ``SLO_DUMP_STACKS_INTERVAL``
    seconds (default 30).

    Side effects:
        - registers a faulthandler repeating timer dumping to stderr
    """
    if os.environ.get("SLO_DUMP_STACKS", "0").lower() not in ("1", "true", "yes"):
        return
    try:
        interval = float(os.environ.get("SLO_DUMP_STACKS_INTERVAL", "30"))
        faulthandler.dump_traceback_later(interval, repeat=True)
        logger.info("faulthandler stack dump active every %ss (SLO_DUMP_STACKS=1)", interval, extra={"tag": "START"})
    except Exception as exc:
        logger.warning("faulthandler stack dump not installed: %s", exc, extra={"tag": "START"})


def _cancel_stack_dump_timer() -> None:
    """Cancel the faulthandler dump timer installed by ``_install_stack_dump_timer``."""
    try:
        faulthandler.cancel_dump_traceback_later()
    except Exception:
        pass


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

# GZip omitted — Starlette GZipMiddleware buffers responses, which kills SSE streaming.
# /chat/stream and /inference/generate/stream send chunked text/event-stream that must
# not be buffered. Non-streaming responses (health, models, etc.) are <5KB — compression
# overhead exceeds bandwidth savings at that size. Add back only if large payload
# endpoints (/datasets/export, /training/export-text) need it, using per-route config.

# Register structured middleware from the infrastructure package.
from infrastructure.middleware import register_all_middleware  # noqa: E402
register_all_middleware(app, request_timeout=cfg.request_timeout_seconds)

# Register health/status routes IMMEDIATELY — before lifespan runs.
# During the 25-40s model loading phase, the frontend must still get
# real responses from /health, /health/startup-progress, /health/summary
# instead of connection errors.  These lightweight routers have zero
# heavy imports.
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

        # Wire the active server model so background training has a model to
        # incrementally fine-tune on feedback (falls back to auto-train student
        # if present). No-op if no model is loaded yet.
        try:
            import state as server_state
            model = getattr(server_state, "model", None)
            tokenizer = getattr(server_state, "tokenizer", None)
            if model is not None and tokenizer is not None:
                workflow.set_model(model, tokenizer)
        except Exception as e:
            logger.debug("Feedback workflow model wiring skipped: %s", e)
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
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Kill existing process on port and start fresh (default: connect to existing)",
    )
    args = parser.parse_args()

    # Daemonize if requested — use subprocess to avoid fork() issues
    if args.daemon:
        import subprocess as sp
        cmd = [sys.executable, __file__]
        if args.port:
            cmd += ["--port", str(args.port)]
        if args.force:
            cmd += ["--force"]
        proc = sp.Popen(
            cmd,
            stdin=sp.DEVNULL,
            stdout=sp.DEVNULL,
            stderr=sp.DEVNULL,
            start_new_session=True,
        )
        logger.info("Server started as daemon (PID %s)", proc.pid)
        sys.exit(0)

    bind_port = args.port or cfg.port

    # ── Singleton gate ────────────────────────────────────────────────
    # Detect whether the port is in use.  If it is, determine whether
    # the occupant is a healthy SloughGPT server or something else.
    # Without --force: healthy server -> exit 0 (reuse); non-server -> exit 1.
    # With --force: kill everything on the port and proceed.
    import socket as _sock
    _port_open = True
    try:
        with _sock.create_connection(("127.0.0.1", bind_port), timeout=1.0):
            _port_open = False
    except (ConnectionRefusedError, OSError, TimeoutError):
        pass

    if not _port_open:
        # Something is listening -- is it a SloughGPT server?
        import urllib.request as _urllib_request
        _is_server = False
        try:
            req = _urllib_request.Request(f"http://127.0.0.1:{bind_port}/health", method="GET")
            with _urllib_request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    _is_server = True
        except Exception:
            pass

        if _is_server and not args.force:
            logger.info(
                "Server already running on port %d -- exiting (use --force to replace)",
                bind_port, extra={"tag": "START"},
            )
            sys.exit(0)

        if args.force:
            try:
                pids = subprocess.check_output(
                    ["lsof", "-ti", f":{bind_port}"], timeout=5,
                ).decode().strip().split()
                for pid in pids:
                    if pid and pid != str(os.getpid()):
                        os.kill(int(pid), 9)
                        logger.warning("Killed process %s on port %d (--force)", pid, bind_port, extra={"tag": "START"})
            except Exception:
                pass
        elif not _is_server:
            # Port occupied by a non-server process
            try:
                pids = subprocess.check_output(
                    ["lsof", "-ti", f":{bind_port}"], timeout=5,
                ).decode().strip().split()
                pids = [p for p in pids if p and p != str(os.getpid())]
                if pids:
                    logger.error(
                        "Port %d occupied by process %s (not a SloughGPT server). "
                        "Use --force to kill it.",
                        bind_port, ",".join(pids), extra={"tag": "START"},
                    )
                    sys.exit(1)
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
