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


for _p in (_SERVER_ROOT, _CORE_PY_ROOT, _REPO_ROOT):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from domains.torch_runtime import apply_api_process_torch_env

apply_api_process_torch_env()

# Pre-import transformers core classes to work around a uvicorn import-ordering
# issue where ``from transformers import AutoModelForCausalLM`` fails with
# ``cannot import name`` when called later in `lifespan`.
# This eager import at module level ensures the class is available.
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    import transformers as _t
    _t  # silence unused
except Exception:
    pass

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from starlette.requests import Request
from starlette.responses import JSONResponse
from datetime import datetime, timedelta
import hashlib

from fastapi import (
    FastAPI,
    HTTPException,
)
from fastapi.middleware.cors import CORSMiddleware
from settings import get_security_settings

# MVC routers
from routers import get_all_routers
import state as server_state
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dataclasses import dataclass
import json
import asyncio
import time
import logging
from domains.errors import SloughGPTDomainError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sloughgpt")

_PROCESS_START_MONOTONIC = time.monotonic()


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
    """Load default HF weights in-process when ``SLOUGHGPT_AUTOLOAD_MODEL`` is set (default: ``gpt2``)."""
    try:
        await asyncio.to_thread(_autoload_hf_model_at_startup)
    except Exception as e:
        logger.warning("Startup model autoload failed: %s", e, exc_info=True)
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
    yield
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


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with verbose context (method, path, detail, traceback in debug)."""
    client_ip = request.client.host if request.client else "unknown"
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    extra = {
        "method": request.method,
        "path": str(request.url.path),
        "detail": detail,
    }
    if os.environ.get("SLOUGHGPT_DEBUG", "").lower() in ("1", "true"):
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
    if os.environ.get("SLOUGHGPT_DEBUG", "").lower() in ("1", "true"):
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


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Catch-all for unhandled 500s — include request context and traceback."""
    client_ip = request.client.host if request.client else "unknown"
    import traceback
    tb = traceback.format_exc()
    extra = {
        "error": f"Internal server error: {exc}",
        "method": request.method,
        "path": str(request.url.path),
        "status_code": 500,
    }
    if os.environ.get("SLOUGHGPT_DEBUG", "").lower() in ("1", "true"):
        extra["traceback"] = tb
    logger.error("Unhandled exception on %s %s:\n%s", request.method, request.url.path, tb)
    audit_logger.log(
        "unhandled_error",
        client_ip,
        resource=str(request.url.path),
        action="500",
        status="failure",
        details=extra,
    )
    return JSONResponse(
        status_code=500,
        content=extra,
    )


# Register all feature routers
for _router in get_all_routers():
    app.include_router(_router)

# Training router (defined in training/ subdirectory, not in routers/)
try:
    from training.router import router as training_router
    app.include_router(training_router)
except Exception as exc:
    logger.warning("Failed to register training router: %s", exc, exc_info=True)


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


def _load_hf_model_core(request: LoadModelRequest) -> Dict[str, Any]:
    """Load HuggingFace weights using ModelsController's safe loader (shared by autoload and inline load)."""

    try:
        from controllers.models import get_models_controller
        ctrl = get_models_controller()
        result = ctrl.load_model(request.model_id, request.device or "auto")

        model = getattr(ctrl, "_hf_model", None)
        tokenizer = getattr(ctrl, "_tokenizer", None)

        if model is not None and tokenizer is not None:
            server_state.model = model
            server_state.tokenizer = tokenizer
            server_state.model_type = request.model_id

            inference_engine = None
            try:
                from domains.inference.engine import InferenceEngine
                inference_engine = InferenceEngine(
                    model=model,
                    tokenizer=tokenizer,
                    device=model.device if hasattr(model, 'device') else "cpu",
                )
            except Exception as e:
                logger.warning("Failed to create InferenceEngine: %s", e)

            try:
                from domains.models.provider import setup_providers
                setup_providers(model, tokenizer, hf_model_id=request.model_id, inference_engine=inference_engine)
            except Exception as e:
                logger.warning("Failed to set up model providers: %s", e)

            try:
                from domains.chat.domain import get_chat_domain
                cd = get_chat_domain()
                if inference_engine is not None:
                    cd.set_engine(inference_engine)
                else:
                    cd.set_engine(None)
            except Exception as e:
                logger.warning("Failed to inject engine into ChatDomain: %s", e)
        else:
            return {"status": "error", "error": "Model loaded but model/tokenizer not available"}

        effective = _inference_engine_device_str(model) if model is not None else None

        return {
            "status": "loaded",
            "model": request.model_id,
            "mode": request.mode or "local",
            "device": request.device,
            "effective_device": effective,
            "model_type": server_state.model_type or "gpt2",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def find_available_port(start_port: int = 8000, max_attempts: int = 10) -> int:
    """Find an available port starting from start_port (DEPRECATED - use domains.shared.find_available_port)."""
    from domains.shared import find_available_port as _find_available_port

    return _find_available_port(host="", start_port=start_port, max_attempts=max_attempts)


def _autoload_hf_model_at_startup() -> None:
    """
    Load default inference weights without a manual ``POST /models/load``.

    - ``SLOUGHGPT_AUTOLOAD_MODEL``: HuggingFace model id (default: ``gpt2``). Set to empty to skip.
    - ``SLOUGHGPT_AUTOLOAD_DEVICE``: passed through to the loader (default: ``auto``).
    """
    raw = os.environ.get("SLOUGHGPT_AUTOLOAD_MODEL", "gpt2-medium").strip()
    if not raw:
        logger.info("SLOUGHGPT_AUTOLOAD_MODEL is empty; skipping startup autoload")
        return
    if server_state.model is not None:
        return
    device = (os.environ.get("SLOUGHGPT_AUTOLOAD_DEVICE") or "auto").strip() or "auto"
    req = LoadModelRequest(model_id=raw, mode="local", device=device)
    result = _load_hf_model_core(req)
    if result.get("status") == "error":
        logger.warning("Startup autoload failed for %s: %s", raw, result.get("error"))
    else:
        logger.info(
            "Startup autoload ok: model_id=%s effective_device=%s",
            raw,
            result.get("effective_device"),
        )


def _start_feedback_workflow() -> None:
    """Start the automated feedback workflow at server startup."""
    try:
        from domains.feedback import get_feedback_workflow

        auto_start = os.environ.get("SLOUGHGPT_AUTO_WORKFLOW", "true").lower() == "true"
        if not auto_start:
            logger.info("SLOUGHGPT_AUTO_WORKFLOW is false; skipping workflow startup")
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

        enabled = os.environ.get("SLOUGHGPT_HEALTH_MONITOR", "true").lower() == "true"
        if not enabled:
            logger.info("SLOUGHGPT_HEALTH_MONITOR is false; skipping health monitor startup")
            return

        interval = int(os.environ.get("SLOUGHGPT_HEALTH_INTERVAL", "300"))
        monitor = get_health_monitor()
        thread = monitor.start_auto_monitoring(interval_seconds=interval)
        thread.name = "health-monitor"
        logger.info(
            "Model health monitor started (interval=%ds)",
            interval,
        )
    except Exception as e:
        logger.warning("Failed to start health monitor: %s", e)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SloughGPT API Server")
    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.environ.get("SLOUGHGPT_RELOAD", "").lower() in ("1", "true", "yes"),
        help="Enable auto-reload on file changes (default: $SLOUGHGPT_RELOAD or false)",
    )
    args, _ = parser.parse_known_args()

    import uvicorn

    raw_port = os.environ.get("SLOUGHGPT_API_PORT", "").strip()
    if raw_port:
        port = int(raw_port)
    else:
        port = find_available_port(8000)

    _start_feedback_workflow()
    _start_health_monitor()
    logger.info("Starting SloughGPT server on port %d... (reload=%s)", port, args.reload)

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
    uvicorn.run(**uvicorn_kw)
