#!/usr/bin/env python3
"""
SloughGPT Model Server
FastAPI server for model inference with HuggingFace fallback.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Import path must exist before domains (see ``domains.torch_runtime``).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SERVER_ROOT = Path(__file__).resolve().parent
_CORE_PY_ROOT = _REPO_ROOT / "packages" / "core-py"
_SGLOADER_ROOT = _REPO_ROOT / "packages" / "downcraft"


# Project-local HuggingFace cache (models/hf-cache/ instead of ~/.cache/huggingface/)
_HF_CACHE = _REPO_ROOT / "models" / "hf-cache"
_HF_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_HF_CACHE))

for _p in (_SERVER_ROOT, _CORE_PY_ROOT, _SGLOADER_ROOT, _REPO_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from domains.torch_runtime import apply_api_process_torch_env

apply_api_process_torch_env()

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from starlette.requests import Request
from starlette.responses import JSONResponse
from datetime import datetime, timedelta
import hashlib
import uuid

from fastapi import (
    FastAPI,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from settings import get_security_settings

# MVC routers
import state as server_state
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dataclasses import dataclass
import json
import asyncio
import time
import logging
from domains.errors import SloughGPTDomainError

# Note: model_registry imported in lifespan to avoid 14s torch import at module level

# ── Structured logging setup ────────────────────────────────────────────
from domains.logging import ConsoleLogger, BridgeHandler, set_global, LogLevel

_log_level_name = os.environ.get("MAN_LOG_LEVEL", "INFO").upper()
_log_level = getattr(LogLevel, _log_level_name, LogLevel.INFO)

_console_logger = ConsoleLogger("man", level=_log_level)
set_global(_console_logger)

# Bridge standard logging.getLogger("man.xxx") through our ConsoleLogger
_bridge = BridgeHandler(_console_logger)
_bridge.setLevel(getattr(logging, _log_level_name, logging.INFO))
logging.root.addHandler(_bridge)
logging.root.setLevel(getattr(logging, _log_level_name, logging.INFO))

# Suppress noisy urllib3 NotOpenSSLWarning (LibreSSL on macOS is fine for dev)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="urllib3")
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

logger = logging.getLogger("man")

_PROCESS_START_MONOTONIC = time.monotonic()

from startup_progress import STARTUP_PHASE

# Suppress noisy urllib3 NotOpenSSLWarning (LibreSSL on macOS is fine for dev)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="urllib3")
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")


# ============ Production Configuration ============
@dataclass
class GenerationConfig:
    """Production configuration for text generation. Set via environment variables."""

    temperature: float = float(os.getenv("SLOUGHGT_TEMPERATURE", "0.8"))
    top_p: float = float(os.getenv("SLOUGHGT_TOP_P", "0.9"))
    top_k: int = int(os.getenv("SLOUGHGT_TOP_K", "50"))
    repetition_penalty: float = float(os.getenv("SLOUGHGT_REPETITION_PENALTY", "1.2"))
    max_new_tokens: int = int(os.getenv("SLOUGHGT_MAX_NEW_TOKENS", "200"))
    max_context_length: int = int(os.getenv("SLOUGHGT_MAX_CONTEXT_LENGTH", "1024"))

    @classmethod
    def from_env(cls) -> "GenerationConfig":
        """Create config from environment variables."""
        return cls(
            temperature=float(os.getenv("SLOUGHGT_TEMPERATURE", "0.8")),
            top_p=float(os.getenv("SLOUGHGT_TOP_P", "0.9")),
            top_k=int(os.getenv("SLOUGHGT_TOP_K", "50")),
            repetition_penalty=float(os.getenv("SLOUGHGT_REPETITION_PENALTY", "1.2")),
            max_new_tokens=int(os.getenv("SLOUGHGT_MAX_NEW_TOKENS", "200")),
            max_context_length=int(os.getenv("SLOUGHGT_MAX_CONTEXT_LENGTH", "1024")),
        )


gen_config = GenerationConfig.from_env()
server_state.gen_config = gen_config
logger.info(
    f"Generation config: temp={gen_config.temperature}, top_p={gen_config.top_p}, "
    f"top_k={gen_config.top_k}, rep_penalty={gen_config.repetition_penalty}"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load default HF weights in background when ``MAN_AUTOLOAD_MODEL`` is set (default: Qwen2.5-0.5B-Instruct)."""
    STARTUP_PHASE.update(phase="loading_model", step=1, message="Loading model weights...")
    logger.info("Startup phase 1/6: loading model")
    model_load_task = asyncio.create_task(asyncio.to_thread(_autoload_hf_model_at_startup))
    STARTUP_PHASE.update(phase="wandb_server", step=2, message="Starting W&B metrics server...")
    wandb_server_task: Optional[asyncio.Task] = None
    try:
        from domains.ops.wandb_server import start_wandb_server_background

        def _wandb_server_extra_metrics() -> Dict[str, Any]:
            from host_metrics import sample_host_metrics_sync
            h = sample_host_metrics_sync()
            out: Dict[str, Any] = {
                "host/cpu_percent": float(h["cpu_percent"]),
                "host/memory_percent": float(h["memory_percent"]),
            }
            rss = h.get("process_rss_bytes")
            if isinstance(rss, int) and rss >= 0:
                out["server/process_rss_bytes"] = float(rss)
            return out

        class _NoopHttpMetrics:
            @staticmethod
            def wandb_aggregate() -> dict:
                return {}
        wandb_server_task = await start_wandb_server_background(
            _NoopHttpMetrics(),
            extra_metrics=_wandb_server_extra_metrics,
        )
    except Exception as e:
        logger.warning("W&B server background task did not start: %s", e)
    STARTUP_PHASE.update(phase="multimodal", step=3, message="Initializing multimodal engine...")
    def _init_multimodal():
        try:
            speech_server = os.environ.get("SPEECH_SERVER", "").lower() in ("1", "true", "yes")
            if speech_server:
                from domains.multimodal import initialize_multimodal
                initialize_multimodal(speech_server=True, vision_model="slonet")
                logger.info("Multimodal initialized (server-side ASR enabled)")
            else:
                from domains.multimodal import get_multimodal_manager
                get_multimodal_manager().initialize(vision_model="slonet")
                logger.info("Multimodal initialized (browser ASR only)")
        except Exception as e:
            logger.warning("Multimodal initialization failed: %s", e)
    asyncio.create_task(asyncio.to_thread(_init_multimodal))
    STARTUP_PHASE.update(phase="model_registry", step=5, message="Initializing model registry...")
    from domains.infrastructure.model_registry import get_model_registry
    registry = get_model_registry()
    logger.info("Model registry initialized")

    STARTUP_PHASE.update(phase="ready", step=6, message="Server ready")
    logger.info("Startup complete")

    # Register feature routers (synchronous — all heavy deps now lazy)
    from routers import get_all_routers
    for r in get_all_routers():
        app.include_router(r)
    try:
        from training.router import router as training_router
        app.include_router(training_router)
    except Exception as exc:
        logger.warning("Failed to register training router: %s", exc)
    logger.info("All routers registered (%d routes)", len(app.routes))

    yield
    # Shutdown: unregister all models
    registry.reset_metrics()
    logger.info("Model registry reset on shutdown")
    try:
        from training.job_store import get_job_store
        store = get_job_store()
        running_jobs = store.list(status="running")
        for job in running_jobs:
            logger.info(f"Marking job {job['id']} as interrupted on shutdown")
            store.mark_crashed(job["id"])
    except Exception as e:
        logger.warning("Failed to mark running jobs as interrupted: %s", e)
    if wandb_server_task is not None:
        wandb_server_task.cancel()
        try:
            await wandb_server_task
        except asyncio.CancelledError:
            pass


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

# Note: Feature routers are registered in the lifespan (below) to avoid
# blocking module-level imports for 40+s (JAX/sentence_transformers transitive deps).


@app.middleware("http")
async def request_timeout_middleware(request: Request, call_next):
    """
    Enforce a maximum request duration to prevent hung requests from
    blocking the event loop and making the server appear offline.

    Streaming endpoints (chat, auto-train) are excluded since they
    hold the connection open intentionally.
    """
    import asyncio

    # Skip timeout for streaming endpoints
    path = request.url.path
    if any(path.startswith(p) for p in ["/chat/stream", "/auto-train/stream", "/session/", "/generate/stream", "/models/load", "/inference/generate", "/chat"]):
        return await call_next(request)

    # 60-second timeout for regular requests
    try:
        return await asyncio.wait_for(call_next(request), timeout=60.0)
    except asyncio.TimeoutError:
        logger.warning("Request timed out: %s %s", request.method, request.url.path)
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=504,
            content={"error": "Request timed out", "path": request.url.path},
        )


@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    """Generate or preserve ``X-Request-ID`` for every request.

    Also records request timing in ServerState for debug/monitoring.
    """
    import time as _time
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    start = _time.monotonic()
    try:
        from domains.infrastructure.server_state import get_server_state
        get_server_state().record_request()
    except Exception:
        pass
    response = await call_next(request)
    try:
        elapsed_ms = (_time.monotonic() - start) * 1000
        from domains.infrastructure.server_state import get_server_state
        ss = get_server_state()
        ss.record_request_latency(
            path=str(request.url.path),
            method=request.method,
            status=response.status_code,
            elapsed_ms=elapsed_ms,
        )
        ss.record_path_latency(
            path=str(request.url.path),
            elapsed_ms=elapsed_ms,
        )
        # Snapshot health score every 30 requests for trend tracking
        if ss.request_count % 30 == 0:
            ss.record_health_snapshot()
            ss.record_memory_snapshot()
        # Rate limit check for inference endpoints (10/sec)
        if str(request.url.path).startswith(("/chat", "/inference")):
            if not ss.check_rate_limit(str(request.url.path), max_per_second=10):
                logging.getLogger("man.middleware").debug(
                    "Rate limit exceeded: %s", request.url.path
                )
    except Exception:
        pass
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with verbose context (method, path, detail, traceback in debug)."""
    client_ip = request.client.host if request.client else "unknown"
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    try:
        from domains.infrastructure.server_state import get_server_state
        get_server_state().record_error_detail(
            path=str(request.url.path),
            method=request.method,
            status=exc.status_code,
            message=detail,
            error_type="HTTPException",
        )
    except Exception:
        pass
    extra = {
        "method": request.method,
        "path": str(request.url.path),
        "detail": detail,
    }
    if os.environ.get("MAN_DEBUG", "").lower() in ("1", "true"):
        import traceback
        extra["traceback"] = traceback.format_exc()
    audit_logger.log(
        "http_error",
        client_ip,
        resource=str(request.url.path),
        action=str(exc.status_code),
        status="failure",
        details=extra,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=extra,
    )


@app.exception_handler(Exception)
async def root_exception_handler(request: Request, exc: Exception):
    """Catch-all that prevents any unhandled exception from crashing the process.

    Returns a JSON 500 with error detail instead of raising an internal server error.
    Also records the failure in the ModelRegistry circuit breaker if applicable.
    """
    import traceback
    tb = traceback.format_exc()
    logger.error("Unhandled %s on %s %s:\n%s", type(exc).__name__, request.method, request.url.path, tb)
    try:
        from domains.infrastructure.server_state import get_server_state
        get_server_state().record_error_detail(
            path=str(request.url.path),
            method=request.method,
            status=500,
            message=f"{type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
        )
    except Exception:
        pass
    try:
        from domains.infrastructure.model_registry import get_model_registry
        registry = get_model_registry()
        health = registry.health_summary()
        degraded_models = [m["model_id"] for m in health.get("models", []) if m.get("status") == "degraded"]
        if degraded_models:
            logger.warning("Degraded models: %s", degraded_models)
    except Exception:
        pass
    return JSONResponse(
        status_code=500,
        content={
            "error": f"Internal error: {type(exc).__name__}",
            "error_type": type(exc).__name__,
            "detail": str(exc),
            "path": request.url.path,
        },
    )


@app.exception_handler(SloughGPTDomainError)
async def domain_exception_handler(request: Request, exc: SloughGPTDomainError):
    """Map core ``domains`` errors to JSON with verbose context."""
    client_ip = request.client.host if request.client else "unknown"
    extra = {
        "method": request.method,
        "path": str(request.url.path),
        "error": str(exc),
        "code": exc.code,
    }
    try:
        from domains.infrastructure.server_state import get_server_state
        get_server_state().record_error_detail(
            path=str(request.url.path),
            method=request.method,
            status=422,
            message=str(exc),
            error_type="SloughGPTDomainError",
        )
    except Exception:
        pass
    if os.environ.get("MAN_DEBUG", "").lower() in ("1", "true"):
        import traceback
        extra["traceback"] = traceback.format_exc()
    audit_logger.log(
        "domain_error",
        client_ip,
        resource=str(request.url.path),
        action="domain_error",
        status="failure",
        details=extra,
    )
    return JSONResponse(
        status_code=422,
        content=extra,
    )




# Model globals live in server_state (state.py)


# ============ Security Configuration ============
_sec = get_security_settings()
JWT_SECRET = _sec.jwt_secret
JWT_ALGORITHM = _sec.jwt_algorithm
JWT_EXPIRATION_HOURS = _sec.jwt_expiration_hours
VALID_API_KEYS = _sec.valid_api_keys


# ============ JWT Authentication ============
class JWTAuth:
    """Simple JWT implementation."""

    def __init__(self):
        self.secret = JWT_SECRET
        self.algorithm = JWT_ALGORITHM
        self.expiration_hours = JWT_EXPIRATION_HOURS

    def create_token(self, subject: str, **extra_claims) -> str:
        """Create a JWT token."""
        import base64
        import json

        now = datetime.utcnow()
        payload = {
            "sub": subject,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=self.expiration_hours)).timestamp()),
            **extra_claims,
        }

        header = {"alg": self.algorithm, "typ": "JWT"}
        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")

        import hmac

        signature = hmac.new(
            self.secret.encode(), f"{header_b64}.{payload_b64}".encode(), hashlib.sha256
        )
        signature_b64 = base64.urlsafe_b64encode(signature.digest()).decode().rstrip("=")

        return f"{header_b64}.{payload_b64}.{signature_b64}"

    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode a JWT token."""
        import base64
        import json
        import hmac

        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None

            header_b64, payload_b64, signature_b64 = parts

            # Verify signature
            expected_sig = hmac.new(
                self.secret.encode(), f"{header_b64}.{payload_b64}".encode(), hashlib.sha256
            )
            expected_sig_b64 = base64.urlsafe_b64encode(expected_sig.digest()).decode().rstrip("=")

            if not hmac.compare_digest(signature_b64, expected_sig_b64):
                return None

            # Decode payload
            payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=="))

            # Check expiration
            if payload.get("exp", 0) < datetime.utcnow().timestamp():
                return None

            return payload
        except Exception:
            return None

    def refresh_token(self, token: str) -> Optional[str]:
        """Refresh a JWT token."""
        payload = self.verify_token(token)
        if payload:
            return self.create_token(
                payload["sub"], **{k: v for k, v in payload.items() if k != "sub"}
            )
        return None


jwt_auth = JWTAuth()


# ============ Audit Logger ============
class AuditLogger:
    """Audit logging for security events."""

    def __init__(self):
        self.logs: List[Dict] = []
        self.max_logs = 10000

    def log(
        self,
        event_type: str,
        client_ip: str,
        user_id: Optional[str] = None,
        resource: str = "",
        action: str = "",
        status: str = "success",
        details: Optional[Dict] = None,
    ):
        """Log an audit event."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "client_ip": client_ip,
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "status": status,
            "details": details or {},
        }
        self.logs.append(entry)
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs :]

        # Log to standard logger
        log_level = logging.INFO if status == "success" else logging.WARNING
        logger.log(log_level, f"AUDIT: {event_type} - {client_ip} - {action} - {status}")

    def get_logs(self, limit: int = 100, event_type: Optional[str] = None) -> List[Dict]:
        """Get audit logs."""
        logs = self.logs[-limit:]
        if event_type:
            logs = [l for l in logs if l["event_type"] == event_type]
        return logs


audit_logger = AuditLogger()


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

        logger.info("Loading model from %s...", model_path)
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
                logger.info("Tokenizer loaded: %d characters", len(ckpt['chars']))
            else:
                server_state.tokenizer = None
            logger.info("Model loaded: %d parameters", len(ckpt.get('model', {})))
        else:
            server_state.model = ckpt
            server_state.model_type = "sloughgpt_finetuned"
            server_state.tokenizer = None
            logger.info("Model loaded successfully from %s", model_path)

    except Exception as e:
        logger.warning("Failed to load model: %s", e)
        import traceback
        traceback.print_exc()
        server_state.model = None
        server_state.tokenizer = None
        server_state.model_type = "demo"


# ============ Meta-Weight Learning Endpoints ============

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


@app.post("/session/{session_id}/regenerate", tags=["session"])
async def regenerate_response(session_id: str, req: Request):
    """Stream a regenerated response for the last user message in the conversation."""
    from fastapi.responses import StreamingResponse
    import json
    from typing import AsyncIterator

    try:
        body = await req.json()
    except Exception:
        body = {}

    messages = body.get("messages", [])
    if not messages:
        from domains.infrastructure.session_core import SessionCore
        messages = SessionCore.get_messages(session_id)
        if not messages:
            return StreamingResponse(
                iter([f"data: " + json.dumps({"stream": "chat", "status": "error", "phase": "STREAMING", "data": {"error": "No session context found"}}) + "\n\n"]),
                media_type="text/event-stream"
            )

    user_msg = None
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_msg = msg.get("content", "")
            break

    if user_msg is None:
        return StreamingResponse(
            iter([f"data: " + json.dumps({"stream": "chat", "status": "error", "phase": "STREAMING", "data": {"error": "No user message found"}}) + "\n\n"]),
            media_type="text/event-stream"
        )

    from controllers.models import get_models_controller
    _mc = get_models_controller()
    _model = _mc._hf_model if _mc._hf_model is not None else server_state.model
    _tokenizer = _mc._tokenizer if _mc._tokenizer is not None else server_state.tokenizer

    if _model is None or _tokenizer is None:
        return StreamingResponse(
            iter([f"data: " + json.dumps({"stream": "chat", "status": "error", "phase": "STREAMING", "data": {"error": "Model not loaded"}}) + "\n\n"]),
            media_type="text/event-stream"
        )

    async def generate() -> AsyncIterator[str]:
        try:
            from transformers import TextIteratorStreamer
            from threading import Thread
            user_ids = _tokenizer(user_msg, return_tensors="pt")
            user_ids = _inputs_to_model_device(user_ids, _model)

            streamer = TextIteratorStreamer(_tokenizer, skip_prompt=True, skip_special_tokens=True)

            def run_gen():
                _model.generate(
                    **user_ids,
                    max_new_tokens=body.get("max_new_tokens", 256),
                    temperature=body.get("temperature", 0.8),
                    do_sample=True,
                    pad_token_id=_tokenizer.eos_token_id,
                    streamer=streamer,
                )

            thread = Thread(target=run_gen)
            thread.start()

            for token in streamer:
                if token:
                    yield f"data: " + json.dumps({"stream": "chat", "phase": "STREAMING", "status": "working", "data": {"token": token}}) + "\n\n"

            thread.join()
            yield f"data: " + json.dumps({"stream": "chat", "phase": "STREAMING", "status": "complete", "data": {"token": ""}}) + "\n\n"

        except Exception as e:
            yield f"data: " + json.dumps({"stream": "chat", "status": "error", "phase": "STREAMING", "data": {"error": str(e)}}) + "\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


class LoadModelRequest(BaseModel):
    model_id: str
    mode: Optional[str] = "local"
    device: Optional[str] = "auto"


def _load_hf_model_core(request: LoadModelRequest, use_slonet: bool = False) -> Dict[str, Any]:
    """
    Load HuggingFace model via ModelsController.

    Notes:
      - ``_load_hf_model()`` inside the controller already creates the InferenceEngine,
        calls ``setup_providers()``, and injects into ChatDomain.
      - This function only adds ModelRegistry registration + server_state updates.
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

        # Register with ModelRegistry (the only step _load_hf_model does NOT do)
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
            )
            # Re-register HF provider so it uses the ModelServer for
            # lifecycle-managed generation (semaphore, timeout, circuit breaker).
            from domains.models.provider import register_provider, HFModelProvider
            model_server = registry.get(request.model_id)
            provider = HFModelProvider(
                model, tokenizer,
                model_id_str=request.model_id,
                model_server=model_server,
            )
            register_provider("hf-default", provider)
            logger.info("hf-default provider re-registered with ModelServer: %s", request.model_id)
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


def _autoload_hf_model_at_startup() -> None:
    """
    Load default inference weights without a manual ``POST /models/load``.

    - ``MAN_AUTOLOAD_MODEL``: HuggingFace model id (default: ``gpt2``). Set to empty to skip.
    - ``MAN_AUTOLOAD_DEVICE``: passed through to the loader (default: ``auto``).
    - ``MAN_USE_SLONET``: If set to ``1`` or ``true``, loads into SloTransformer
      (pure NumPy) instead of PyTorch. Recommended for stability.
    """
    raw = os.environ.get("MAN_AUTOLOAD_MODEL", "Qwen/Qwen2.5-0.5B-Instruct").strip()
    if not raw or raw.lower() in ("false", "0", "none", "no", "off", "disable"):
        logger.info("MAN_AUTOLOAD_MODEL=%r — skipping startup autoload", raw)
        server_state.autoload_skipped = True
        return
    if server_state.model is not None:
        return
    device = (os.environ.get("MAN_AUTOLOAD_DEVICE") or "auto").strip() or "auto"
    use_slonet = os.environ.get("MAN_USE_SLONET", "0").strip().lower() in ("1", "true", "yes")
    req = LoadModelRequest(model_id=raw, mode="local", device=device)
    result = _load_hf_model_core(req, use_slonet=use_slonet)
    if result.get("status") == "error":
        logger.warning("Startup autoload failed for %s: %s", raw, result.get("error"))
    else:
        logger.info(
            "Startup autoload ok: model_id=%s effective_device=%s mode=%s",
            raw,
            result.get("effective_device"),
            "slonet" if use_slonet else "pytorch",
        )


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
        logger.warning("Failed to start feedback workflow: %s", e)


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
            "Model health monitor started (interval=%ds)",
            interval,
        )
    except Exception as e:
        logger.warning("Failed to start health monitor: %s", e)


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
                # If no model loaded, check if any provider exists (SloNet, HF, etc.)
                if server_state.model is None:
                    from domains.models.provider import get_provider
                    default = get_provider("default")
                    if default is not None:
                        return True
                    # No model and no provider — only unhealthy if we expected one
                    # (MAN_AUTOLOAD_MODEL was set but failed, or model was unloaded)
                    import os
                    if os.environ.get("MAN_AUTOLOAD_MODEL", ""):
                        return False
                    # Training-only mode (no autoload configured) — always healthy
                    return True
                from domains.models.provider import get_provider
                router = get_provider("default")
                return router is not None
            except Exception:
                return False

        def _recover() -> bool:
            """Attempt to recover by reloading the autoload model."""
            try:
                import gc
                import torch
                # Clear any stale state
                gc.collect()
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                # Reload model — _autoload_hf_model_at_startup reads MAN_AUTOLOAD_MODEL
                # from environment; if unset it defaults to gpt2
                _autoload_hf_model_at_startup()
                import state as srv_state
                return srv_state.model is not None
            except Exception as e:
                logger.error("Recovery failed: %s", e)
                return False

        watchdog.set_health_check_fn(_check_health)
        watchdog.set_recovery_fn(_recover)
        watchdog.start(poll_interval=15, max_failures=3)
        logger.info("Health watchdog started (poll=15s, max_failures=3)")
    except Exception as e:
        logger.warning("Failed to start watchdog: %s", e)


if __name__ == "__main__":
    import argparse
    import atexit
    import signal
    import traceback

    # ── Global exception handler — log and exit cleanly ──
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
        default=os.environ.get("MAN_RELOAD", "").lower() in ("1", "true", "yes"),
        help="Enable auto-reload on file changes (default: $MAN_RELOAD or false)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind (default: 8000, falls back to next available)",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        default=os.environ.get("MAN_WEB", "").lower() in ("1", "true", "yes"),
        help="Serve web frontend alongside API (default: $MAN_WEB or false)",
    )
    args, _ = parser.parse_known_args()

    # Kill orphan processes on port 8000 to avoid port conflicts
    import subprocess
    try:
        orphans = subprocess.check_output(["lsof", "-ti", ":8000"], timeout=5).decode().strip().split()
        for pid in orphans:
            if pid and pid != str(os.getpid()):
                os.kill(int(pid), 9)
                logger.warning(f"Killed orphan process {pid} on port 8000")
    except Exception:
        pass

    import uvicorn

    raw_port = os.environ.get("MAN_API_PORT", "").strip()
    if args.port:
        port = args.port
    elif raw_port:
        port = int(raw_port)
    else:
        port = find_available_port(8000)

    _start_feedback_workflow()
    _start_health_monitor()
    _start_watchdog()
    logger.info("Starting SloughGPT server on port %d... (reload=%s)", port, args.reload)

    # ── Web frontend (optional) ────────────────────────────────
    web_proc = None
    if args.web:
        web_root = _REPO_ROOT / "apps" / "web"
        standalone_dir = web_root / ".next" / "standalone"
        web_port = find_available_port(3000)
        web_env = {**os.environ, "PORT": str(web_port)}

        if standalone_dir.is_dir() and (standalone_dir / "server.js").is_file():
            # Run standalone Next.js server (needs Node.js runtime for SSR)
            logger.info("Starting built web frontend on http://localhost:%d", web_port)
            web_proc = subprocess.Popen(
                ["node", "server.js"],
                cwd=str(standalone_dir),
                env=web_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        else:
            # Spawn Next.js dev server
            logger.info("No standalone build found — spawning Next.js dev server on http://localhost:%d", web_port)
            web_proc = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=str(web_root),
                env=web_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

    # Clean up web subprocess on exit
    if web_proc:
        atexit.register(lambda p=web_proc: (p.terminate(), p.wait(timeout=5)) if p.poll() is None else None)

    # ── Graceful shutdown on SIGTERM/SIGINT ──
    def _shutdown_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info("Received %s — shutting down gracefully", sig_name)
        # Let uvicorn's lifespan handle cleanup; just exit cleanly
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    uvicorn_kw = dict(
        app=app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
    if args.reload:
        uvicorn_kw["reload"] = True
        # Pass as import string so reload spawns a real child process
        uvicorn_kw["app"] = "main:app"
        # Only watch Python files and exclude large third-party directories
        uvicorn_kw["reload_includes"] = ["*.py"]
        uvicorn_kw["reload_excludes"] = [
            ".*/**",
            "node_modules/**",
            "__pycache__/**",
            "*.pyc",
            ".git/**",
            ".venv/**",
            "venv/**",
            "env/**",
            "build/**",
            "dist/**",
            ".next/**",
            "data/**",
            "datasets/**",
            "models/**",
        ]

    try:
        uvicorn.run(**uvicorn_kw)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.critical("Server crashed: %s", e, exc_info=True)
        sys.exit(1)
