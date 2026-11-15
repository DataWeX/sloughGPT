"""
Models Router - MVC View layer
Uses ModelsController for business logic
"""

import asyncio
import datetime
import json
import logging
import os
import re
import time
import time as _time
from collections.abc import AsyncGenerator, AsyncIterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from controllers.models import get_models_controller
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel, Field
from schemas.common import (
    classify_and_raise,
    endpoint,
    raise_error,
    safe_audit_log,
    success_response,
    wrap_controller_result,
)
from schemas.models import LoadModelRequest, ModelInfo, ModelStatus

from config import ServerConfig
from domain.infrastructure import AppError

logger = logging.getLogger(__name__)

from domain.infrastructure.compute_backend import get_backend
from domain.infrastructure.model_size import compute_model_size_gb, format_size_gb, is_model_cached

# Module-level so tests can patch ``routers.models._hf_cache_dir``; resolved at call time.
_hf_cache_dir = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"


class ExportRequest(BaseModel):
    output_path: str = Field(default="models/exported", max_length=500)
    format: str = Field(default="sou", pattern=r"^(sou|safetensors|onnx|gguf)$")
    include_tokenizer: bool = True


class DownloadRequest(BaseModel):
    model_id: str = Field(..., min_length=1, max_length=200)
    total_bytes_hint: int = Field(default=0, ge=0)


class QuantizeRequest(BaseModel):
    """Request body for POST /models/quantize."""

    bits: int = Field(default=8)
    mode: str = Field(default="symmetric")


class PrecisionRequest(BaseModel):
    """Request body for POST /models/precision."""

    mode: str = Field(default="auto", pattern=r"^(auto|fp32|fp16)$")


class ProcessGuardRequest(BaseModel):
    enabled: bool


class ExternalServerRequest(BaseModel):
    """Request body for POST /models/external/servers."""

    name: str = Field(..., min_length=1, max_length=100)
    url: str = Field(..., min_length=1, max_length=500)
    compressed: bool = Field(default=True)


class ExternalDownloadRequest(BaseModel):
    """Request body for POST /models/external/download."""

    server: str = Field(..., min_length=1, max_length=100)
    model_id: str = Field(..., min_length=1, max_length=200)


@dataclass(frozen=True)
class ModelProfile:
    """Structured model description — the prose in ``ModelInfo.description`` is a projection of this.

    Follows the entity convention (see auth User / ServingProfile): data holds
    structure, ``to_dict()`` serializes it, and rendered text is derived — never
    the other way round.
    """

    size_tier: str | None  # "small" | "medium" | "large" | None (parameters unknown)
    capability: str  # "conversations" | "code" | "text_generation"
    speed_hint: str | None  # "fast_cpu" | None
    loaded: bool

    _SIZE_LABELS: ClassVar[dict[str | None, str]] = {
        "small": "Small, fast model",
        "medium": "Medium-sized model",
        "large": "Large model",
        None: "Model",
    }
    _CAPABILITY_LABELS: ClassVar[dict[str, str]] = {
        "conversations": "good for conversations",
        "code": "good for code",
        "text_generation": "good for text generation",
    }
    _SPEED_LABELS: ClassVar[dict[str, str]] = {"fast_cpu": "runs fast on CPU"}

    def as_description(self) -> str:
        """Legacy-compatible prose projection (byte-identical to the old builder)."""
        parts = [self._SIZE_LABELS[self.size_tier], self._CAPABILITY_LABELS[self.capability]]
        if self.speed_hint:
            parts.append(self._SPEED_LABELS[self.speed_hint])
        if self.loaded:
            parts.append("(currently loaded)")
        return " — ".join(parts[:2]) + (f". {parts[2]}" if len(parts) > 2 else "") + "."

    def to_dict(self) -> dict[str, Any]:
        return {
            "size_tier": self.size_tier,
            "capability": self.capability,
            "speed_hint": self.speed_hint,
            "loaded": self.loaded,
        }


def build_model_profile(model_id: str, parameters: int, loaded: bool) -> ModelProfile:
    """Pure heuristic profile of a model from its id and parameter count."""
    try:
        params = int(parameters or 0)
    except (TypeError, ValueError):
        params = 0

    mid = model_id.lower()
    if any(kw in mid for kw in ("instruct", "chat", "qwen")):
        capability = "conversations"
    elif any(kw in mid for kw in ("code", "starcoder")):
        capability = "code"
    else:
        capability = "text_generation"

    if not params:
        size_tier = None
    elif params < 150_000_000:
        size_tier = "small"
    elif params < 1_000_000_000:
        size_tier = "medium"
    else:
        size_tier = "large"

    speed_hint = "fast_cpu" if params and params < 500_000_000 else None
    return ModelProfile(
        size_tier=size_tier, capability=capability, speed_hint=speed_hint, loaded=loaded
    )


class ModelsRouter:
    """Models Router - MVC View layer."""

    def __init__(self):
        self.router = APIRouter(prefix="/models", tags=["models"])
        from mogdb.cache import QueryCache

        self._cache = QueryCache(ttl_seconds=5.0, max_entries=16)
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(path="", endpoint=self.list_models, methods=["GET"])
        self.router.add_api_route(path="/load", endpoint=self.load_model, methods=["POST"])
        self.router.add_api_route(path="/unload", endpoint=self.unload_model, methods=["POST"])
        self.router.add_api_route(path="/current", endpoint=self.current_model, methods=["GET"])
        self.router.add_api_route(path="/hf", endpoint=self.list_hf_models, methods=["GET"])
        self.router.add_api_route(path="/logs", endpoint=self.get_model_logs, methods=["GET"])
        self.router.add_api_route(
            path="/export", endpoint=self.export_model, methods=["POST"], tags=["models"]
        )
        self.router.add_api_route(
            path="/export/formats",
            endpoint=self.get_export_formats,
            methods=["GET"],
            tags=["models"],
        )
        self.router.add_api_route(path="/download", endpoint=self.start_download, methods=["POST"])
        self.router.add_api_route(
            path="/download/{model_id:path}", endpoint=self.get_download_status, methods=["GET"]
        )
        self.router.add_api_route(path="/downloads", endpoint=self.list_downloads, methods=["GET"])
        self.router.add_api_route(
            path="/download/{model_id:path}/cancel", endpoint=self.cancel_download, methods=["POST"]
        )
        self.router.add_api_route(
            path="/download/{model_id:path}/pause", endpoint=self.pause_download, methods=["POST"]
        )
        self.router.add_api_route(
            path="/download/{model_id:path}/resume", endpoint=self.resume_download, methods=["POST"]
        )
        self.router.add_api_route(
            path="/download/{model_id:path}/verify", endpoint=self.verify_download, methods=["POST"]
        )
        self.router.add_api_route(
            path="/download/{model_id:path}/retry", endpoint=self.retry_download, methods=["POST"]
        )
        self.router.add_api_route(path="/cache-usage", endpoint=self.cache_usage, methods=["GET"])
        self.router.add_api_route(
            path="/download/qwen-gguf", endpoint=self.download_qwen_gguf, methods=["GET"]
        )
        self.router.add_api_route(
            path="/file/{model_id:path}/{file_path:path}",
            endpoint=self.serve_model_file,
            methods=["GET"],
        )
        self.router.add_api_route(
            path="/external/servers",
            endpoint=self.list_external_servers,
            methods=["GET"],
        )
        self.router.add_api_route(
            path="/external/servers",
            endpoint=self.register_external_server,
            methods=["POST"],
        )
        self.router.add_api_route(
            path="/external/servers/{name}",
            endpoint=self.remove_external_server,
            methods=["DELETE"],
        )
        self.router.add_api_route(
            path="/external/models",
            endpoint=self.list_external_models,
            methods=["GET"],
        )
        self.router.add_api_route(
            path="/external/download",
            endpoint=self.download_external_model,
            methods=["POST"],
        )
        self.router.add_api_route(
            path="/history",
            endpoint=self.download_history,
            methods=["GET"],
        )
        self.router.add_api_route(
            path="/backends",
            endpoint=self.list_backends,
            methods=["GET"],
        )
        self.router.add_api_route(
            path="/backends/active",
            endpoint=self.get_active_backend,
            methods=["GET"],
        )
        self.router.add_api_route(
            path="/backends/active",
            endpoint=self.set_active_backend,
            methods=["POST"],
        )
        self.router.add_api_route(
            path="/visual-load", endpoint=self.visual_model_load, methods=["POST"]
        )
        self.router.add_api_route(path="/quantize", endpoint=self.quantize_model, methods=["POST"])
        self.router.add_api_route(
            path="/dequantize", endpoint=self.dequantize_model, methods=["POST"]
        )
        self.router.add_api_route(path="/precision", endpoint=self.set_precision, methods=["POST"])
        self.router.add_api_route(path="/catalog", endpoint=self.get_catalog, methods=["GET"])
        self.router.add_api_route(
            path="/catalog/stats", endpoint=self.get_catalog_stats, methods=["GET"]
        )
        self.router.add_api_route(
            path="/conversion-status", endpoint=self.get_conversion_status, methods=["GET"]
        )
        self.router.add_api_route(
            path="/process-guard", endpoint=self.get_process_guard, methods=["GET"]
        )
        self.router.add_api_route(
            path="/process-guard", endpoint=self.set_process_guard, methods=["POST"]
        )
        self.router.add_api_route(
            path="/engine/status", endpoint=self.get_engine_status, methods=["GET"]
        )
        self.router.add_api_route(
            path="/engine/reload", endpoint=self.reload_engine, methods=["POST"]
        )
        self.router.add_api_route(
            path="/debug/providers", endpoint=self.debug_providers, methods=["GET"]
        )
        self.router.add_api_route(
            path="/memory-cleanup", endpoint=self.memory_cleanup, methods=["POST"]
        )
        self.router.add_api_route(
            path="/memory-pressure", endpoint=self.memory_pressure, methods=["GET"]
        )
        # Peer-to-peer serving: registered LAST on purpose. Both use a greedy
        # `{model_id:path}` prefix, so they must be tried only after every
        # static route above — otherwise `/models/file/{id}/{path}` and
        # `/models/download/...` would be swallowed by the catch-all shapes.
        # These two are what ExternalDownloadBackend actually requests.
        self.router.add_api_route(
            path="/{model_id:path}/manifest.json",
            endpoint=self.serve_model_manifest,
            methods=["GET"],
        )
        self.router.add_api_route(
            path="/{model_id:path}/file/{file_path:path}",
            endpoint=self.serve_model_file_nested,
            methods=["GET"],
        )

    @staticmethod
    def _audit_model_id(provider) -> str:
        """Best-effort resolve the current model id from a provider for audit events."""
        if provider is None:
            return "unknown"
        return (
            getattr(provider, "_model_id", None)
            or getattr(provider, "_model_id_str", None)
            or "unknown"
        )

    def _describe_model(self, model_id: str, parameters: int, loaded: bool) -> str:
        """Generate a plain-language description of a model.

        A projection of :func:`build_model_profile` — the structured profile is
        the source of truth, this string is derived from it.
        """
        return build_model_profile(model_id, parameters, loaded).as_description()

    def _model_display_name(self, model_id: str) -> str:
        """Generate a human-friendly display name from a HuggingFace model ID.

        Fully algorithmic — no hardcoded lookup tables.

        Examples:
            "Qwen/Qwen2.5-0.5B-Instruct" → "Qwen 2.5 0.5B Instruct"
            "gpt2" → "GPT 2"
            "gpt2-medium" → "GPT 2 Medium"
            "gpt2-xl" → "GPT 2 XL"
            "microsoft/Phi-3.5-mini-instruct" → "Phi 3.5 Mini Instruct"
            "meta-llama/Llama-3-8B" → "Llama 3 8B"
        """
        name = model_id.split("/")[-1] if "/" in model_id else model_id

        # Strip cache prefix: "models--Qwen--Qwen2.5..." → "Qwen2.5..."
        if name.startswith("models--"):
            after = name[len("models--") :]
            # "models--org--model" → take everything after second "--"
            idx = after.find("--")
            if idx >= 0:
                name = after[idx + 2 :]

        # Split on common separators
        parts = re.split(r"[/\-_]", name)

        result = []
        for part in parts:
            if not part:
                continue
            # Short all-lowercase abbreviations (xl, bp, etc.) → uppercase all
            if len(part) <= 3 and part.isalpha() and part.islower():
                result.append(part.upper())
            # All lowercase letters followed by digits: "gpt2", "llama3"
            elif re.match(r"^[a-z]+\d+$", part):
                match = re.match(r"^([a-z]+)(\d+)$", part)
                if match:
                    result.append(match.group(1).upper())
                    result.append(match.group(2))
                else:
                    result.append(part.upper())
            # Number with size suffix: "0.5B", "3B", "8B"
            elif re.match(r"^\d+\.?\d*[a-zA-Z]$", part):
                result.append(part)
            else:
                # Normal mixed case: split at letter→digit and digit→letter boundaries
                # But keep short digit-letter patterns together (e.g., "4e1t")
                sub = re.sub(r"([a-zA-Z]{2,})(\d)", r"\1 \2", part)
                sub = re.sub(r"(\d)([a-zA-Z]{2,})", r"\1 \2", sub)
                # Capitalize first letter of each sub-word
                words = sub.split()
                for w in words:
                    if re.match(r"^[\d.]+$", w):
                        result.append(w)
                    else:
                        result.append(w[0].upper() + w[1:])
        return " ".join(result)

    @endpoint("models.list_models")
    async def list_models(self) -> dict:
        """List available/loaded models with plain-language descriptions."""

        def compute():
            ctrl = get_models_controller()
            current = ctrl.get_current_model()
            models = []
            if current:
                params = int(current.get("parameters", 0) or 0)
                vocab = int(current.get("vocab_size", 0) or 0)
                prof = build_model_profile(current["model_id"], params, loaded=True)
                models.append(
                    ModelInfo(
                        model_id=current["model_id"],
                        status=ModelStatus.LOADED,
                        device=current["device"],
                        parameters=params,
                        vocab_size=vocab,
                        loaded_at=current.get("loaded_at"),
                        description=prof.as_description(),
                        profile=prof.to_dict(),
                    )
                )
            loaded_ids = {m.model_id for m in models}
            hf_models = ctrl.list_hf_models()
            for entry in hf_models:
                model_id = entry["model_id"]
                if model_id not in loaded_ids:
                    params = int(entry.get("parameters", 0) or 0)
                    vocab = int(entry.get("vocab_size", 0) or 0)
                    prof = build_model_profile(model_id, params, loaded=False)
                    models.append(
                        ModelInfo(
                            model_id=model_id,
                            status=ModelStatus.AVAILABLE,
                            device="cpu",
                            parameters=params,
                            vocab_size=vocab,
                            loaded_at=None,
                            description=prof.as_description(),
                            profile=prof.to_dict(),
                        )
                    )
            return [m.model_dump() for m in models]

        try:
            data = self._cache.get_or_set("models:list", compute)
            return success_response(data=data)
        except Exception as e:
            classify_and_raise(e, source="models.list")

    @endpoint("models.load_model")
    async def load_model(
        self,
        req: LoadModelRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Load a model"""
        try:
            ctrl = get_models_controller()
            _t0 = time.monotonic()
            result = ctrl.load_model(req.model_id, req.device.value, req.quantize)
            _elapsed_ms = (time.monotonic() - _t0) * 1000
            try:
                from domain.infrastructure.server_state import get_server_state

                ss = get_server_state()
                if result.get("status") == "loaded":
                    resolved_device = result.get("device") or req.device.value
                    ss.record_model_event("load", req.model_id, f"device={resolved_device}")
                else:
                    ss.record_model_event("error", req.model_id, result.get("error", "unknown"))
            except Exception as e:
                logger.warning("Failed to record model load event: %s", e)
            logger.info("Model loaded: %s (%.0fms)", req.model_id, _elapsed_ms)
            safe_audit_log(
                "model.load",
                resource=req.model_id,
                detail=f"{result.get('status', 'unknown')} ({_elapsed_ms:.0f}ms)",
                device=req.device.value,
                quantize=req.quantize,
            )
            return wrap_controller_result(result)
        except Exception as e:
            classify_and_raise(e, source="models.load")

    @endpoint("models.unload_model")
    async def unload_model(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Unload current model"""
        ctrl = get_models_controller()
        model_id = ctrl.active_model_id()
        result = ctrl.unload_model()
        try:
            from domain.infrastructure.server_state import get_server_state

            ss = get_server_state()
            ss.record_model_event("unload", model_id or "unknown")
        except Exception as e:
            logger.warning("Failed to record model unload event: %s", e)
        safe_audit_log(
            "model.unload",
            resource=model_id or "unknown",
            detail=result.get("status", "unknown"),
        )
        return wrap_controller_result(result)

    @endpoint("models.current_model")
    async def current_model(self) -> dict:
        """Get current model info"""
        ctrl = get_models_controller()
        model = ctrl.get_current_model()
        if not model:
            raise_error("No model loaded", "E_NOT_FOUND")
        return success_response(data=model)

    @endpoint("models.list_hf_models")
    async def list_hf_models(
        self,
        q: str | None = None,
        limit: int = 50,
        offset: int = 0,
        shallow: bool = False,
    ) -> dict:
        """List HuggingFace available models with actual sizes and cache status.

        ``shallow=true`` is the first-paint mode used by the ``/bootstrap``
        fan-out: sizes resolve from the local HF cache only and never fall
        through to the Hub API, so a cold size cache costs zero network
        round-trips (measured 4.3s -> <50ms warm, was ~19.5s cold under
        the boot fan-out). Payload shape is unchanged; size fields are
        ``null`` when the local cache cannot answer.
        """
        try:
            ctrl = get_models_controller()

            def _build_list():
                model_ids = ctrl.list_hf_models(q)

                def _is_cached(model_id: str) -> bool:
                    try:
                        return is_model_cached(model_id)
                    except Exception as exc:
                        logger.error(
                            "is_model_cached(%s) failed: %s", model_id, exc, extra={"tag": "MODEL"}
                        )
                        return False

                def _cache_model_id(cache_dir_name: str) -> str | None:
                    if not cache_dir_name.startswith("models--"):
                        return None
                    return cache_dir_name[len("models--") :].replace("--", "/")

                models_out = []
                seen_ids = set()

                all_model_ids = []
                for m in model_ids:
                    mid = m["model_id"] if isinstance(m, dict) else m
                    if mid not in seen_ids:
                        seen_ids.add(mid)
                        all_model_ids.append(mid)

                if not q and _hf_cache_dir.exists():
                    try:
                        for entry in _hf_cache_dir.iterdir():
                            if not entry.name.startswith("models--") or not entry.is_dir():
                                continue
                            cached_id = _cache_model_id(entry.name)
                            if cached_id and cached_id not in seen_ids:
                                seen_ids.add(cached_id)
                                all_model_ids.append(cached_id)
                    except Exception:
                        logger.debug("Failed to scan HF cache", exc_info=True)

                total = len(all_model_ids)
                page_ids = all_model_ids[offset : offset + limit]

                size_results: dict[str, float | None] = {}
                cached_results: dict[str, bool] = {}

                def _compute_one(mid: str):
                    return mid, compute_model_size_gb(mid, offline=shallow), _is_cached(mid)

                from domain.infrastructure.resource_manager import get_resource_manager

                rm = get_resource_manager()
                max_workers = min(max(len(page_ids), rm.inference_pool_size * 2), 16)
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {pool.submit(_compute_one, mid): mid for mid in page_ids}
                    for future in as_completed(futures):
                        try:
                            mid, size_gb, cached = future.result()
                            size_results[mid] = size_gb
                            cached_results[mid] = cached
                        except Exception as exc:
                            mid = futures[future]
                            import logging

                            logging.getLogger("slo.models").debug(
                                "Size computation failed for %s: %s", mid, exc
                            )
                            size_results[mid] = None
                            cached_results[mid] = False

                for mid in page_ids:
                    size_gb = size_results.get(mid)
                    models_out.append(
                        {
                            "id": mid,
                            "name": self._model_display_name(mid),
                            "hf_model_id": mid,
                            "source": "huggingface",
                            "size_mb": size_gb * 1024 if size_gb is not None else None,
                            "size_gb": size_gb,
                            "cached": cached_results.get(mid, False),
                        }
                    )

                return models_out, total

            models, total = await asyncio.to_thread(_build_list)
            return success_response(
                data=models,
                meta={
                    "q": q,
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "shallow": shallow,
                },
            )
        except Exception as e:
            classify_and_raise(e, source="models.hf_list")

    @endpoint("models.get_model_logs")
    async def get_model_logs(self, limit: int = 50, model_filter: str | None = None) -> dict:
        """Get model request logs (for debugging/monitoring)."""
        from state import model_request_logger as _logger

        if _logger:
            return success_response(
                data=_logger.get_logs(limit=limit, model=model_filter), meta=_logger.get_stats()
            )
        return success_response(data=[], meta={})

    @endpoint("models.export_model")
    async def export_model(
        self, request: ExportRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Export current model to file."""
        import time

        import state as server_state

        if server_state.model is None:
            raise_error("No model loaded", code="E_NOT_FOUND")
        try:
            from domain.training import ExportConfig
            from domain.training import export_model as do_export

            config = ExportConfig(
                input_path="current",
                output_path=request.output_path,
                format=request.format,
                include_tokenizer=request.include_tokenizer,
                metadata={
                    "model_type": server_state.model_type,
                    "exported_at": str(time.time()),
                },
            )
            _t0 = time.monotonic()
            results = do_export(config, server_state.model, server_state.tokenizer)
            _elapsed_ms = (time.monotonic() - _t0) * 1000
            safe_audit_log(
                "model.export",
                resource=request.output_path or "unknown",
                detail=f"format={request.format} elapsed={_elapsed_ms:.0f}ms",
            )
            return success_response(
                data={
                    "format": request.format,
                    "files": results,
                    "elapsed_ms": round(_elapsed_ms, 1),
                },
                message="exported",
            )
        except Exception as e:
            logger.warning("Export model failed: %s", e)
            classify_and_raise(e, source="export_model")

    @endpoint("models.get_export_formats")
    async def get_export_formats(self) -> dict:
        """Get list of supported export formats."""
        from domain.training import list_export_formats

        return success_response(data=list_export_formats())

    @endpoint("models.start_download")
    async def start_download(
        self, req: DownloadRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict[str, Any]:
        """Start downloading a model from HuggingFace Hub with progress tracking."""
        try:
            from domain.infrastructure.download_manager import get_download_manager

            mgr = get_download_manager()

            if mgr.is_cached(req.model_id):
                return success_response(data={"model_id": req.model_id}, message="already_cached")

            if mgr.is_downloading(req.model_id):
                return success_response(
                    data={"model_id": req.model_id}, message="already_downloading"
                )

            asyncio.create_task(self._run_download(req.model_id, req.total_bytes_hint))
            safe_audit_log(
                "model.download",
                resource=req.model_id,
                detail="started",
                total_bytes_hint=req.total_bytes_hint,
            )
            return success_response(data={"model_id": req.model_id}, message="started")
        except Exception as e:
            classify_and_raise(e, source="models.download_start")

    async def _run_download(self, model_id: str, total_bytes_hint: int):
        """Background task that runs the actual download."""
        import time as _time

        from controllers.models import get_models_controller

        from domain.infrastructure.cancel_manager import OpType, get_cancel_manager
        from domain.infrastructure.download_manager import get_download_manager

        mgr = get_download_manager()
        cm_op: str | None = None
        _download_t0 = _time.monotonic()
        try:
            cm = get_cancel_manager()
            cm_op = cm.register(
                op_type=OpType.DOWNLOAD,
                label=f"download:{model_id}",
                cancel_fn=lambda: mgr.cancel(model_id),
            )
            cm.start(cm_op)
        except Exception as e:
            logger.warning(
                "CancelManager registration failed for download %s: %s",
                model_id,
                e,
                extra={"tag": "MODEL"},
            )

        try:
            result = await mgr.download(model_id, total_bytes_hint)
            _download_elapsed_ms = (_time.monotonic() - _download_t0) * 1000

            if result.get("status") == "complete":
                safe_audit_log(
                    "model.download.complete",
                    resource=model_id,
                    detail=f"elapsed={_download_elapsed_ms:.0f}ms size={result.get('size', 0)}",
                )
                ctrl = get_models_controller()
                try:
                    ctrl.load_model(model_id)
                except Exception as e:
                    logger.warning(
                        "Auto-load after download failed for %s: %s",
                        model_id,
                        e,
                        extra={"tag": "MODEL"},
                    )
                if cm_op:
                    try:
                        get_cancel_manager().finish(cm_op)
                    except Exception as exc:
                        logger.warning(
                            "CancelManager.finish failed for download %s: %s", model_id, exc
                        )
            elif result.get("status") == "cancelled":
                if cm_op:
                    try:
                        get_cancel_manager().finish(cm_op, error="cancelled")
                    except Exception as exc:
                        logger.warning(
                            "CancelManager.finish failed for cancelled download %s: %s",
                            model_id,
                            exc,
                        )
            else:
                safe_audit_log(
                    "model.download.failed",
                    resource=model_id,
                    detail=f"elapsed={_download_elapsed_ms:.0f}ms error={result.get('error', 'unknown')}",
                )
                if cm_op:
                    try:
                        get_cancel_manager().finish(
                            cm_op, error=result.get("error", "download failed")
                        )
                    except Exception as exc:
                        logger.warning(
                            "CancelManager.finish failed for failed download %s: %s", model_id, exc
                        )
        except Exception as e:
            _download_elapsed_ms = (_time.monotonic() - _download_t0) * 1000
            safe_audit_log(
                "model.download.failed",
                resource=model_id,
                detail=f"elapsed={_download_elapsed_ms:.0f}ms error={e}",
            )
            if cm_op:
                try:
                    get_cancel_manager().finish(cm_op, error=str(e))
                except Exception as exc:
                    logger.warning(
                        "CancelManager.finish failed for download exception %s: %s", model_id, exc
                    )

    @endpoint("models.get_download_status")
    async def get_download_status(self, model_id: str) -> dict[str, Any]:
        """Get download progress for a specific model."""
        from domain.infrastructure.download_manager import get_download_manager

        mgr = get_download_manager()
        progress = mgr.get_progress(model_id)
        if progress is None:
            cached = mgr.is_cached(model_id)
            return success_response(
                data={"model_id": model_id, "cached": cached}, message="not_found"
            )
        return success_response(data=progress)

    @endpoint("models.list_downloads")
    async def list_downloads(self) -> dict[str, Any]:
        """List all active and recent downloads."""
        from domain.infrastructure.download_manager import get_download_manager

        mgr = get_download_manager()
        mgr.cleanup_stale()
        return success_response(data=mgr.list_downloads())

    @endpoint("models.download_history")
    async def download_history(
        self,
        limit: int = 50,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict[str, Any]:
        """Get download history (completed, failed, cancelled).

        Args:
            limit: Maximum number of entries to return (default 50).
        """
        from domain.infrastructure.event_buffer import get_event_buffer

        event_buffer = get_event_buffer()
        downloads = []

        for event in event_buffer.recent(limit * 2):  # Get extra to filter
            if event.get("category") == "DOWNLOAD":
                detail = event.get("message", "")
                entry = {
                    "detail": detail,
                    "timestamp": event.get("ts"),
                }
                # Parse status from detail
                if "complete" in detail.lower():
                    entry["status"] = "completed"
                elif "failed" in detail.lower():
                    entry["status"] = "failed"
                elif "cancelled" in detail.lower():
                    entry["status"] = "cancelled"
                else:
                    entry["status"] = "unknown"
                downloads.append(entry)

        # Sort by timestamp descending, limit results
        downloads.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        downloads = downloads[:limit]

        return success_response(
            data={
                "history": downloads,
                "total": len(downloads),
            }
        )

    @endpoint("models.cancel_download")
    async def cancel_download(
        self, model_id: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict[str, Any]:
        """Cancel an in-progress download."""
        try:
            from domain.infrastructure.download_manager import get_download_manager

            mgr = get_download_manager()
            if mgr.cancel(model_id):
                safe_audit_log("model.cancel", resource=model_id, detail="cancelled")
                return success_response(data={"model_id": model_id}, message="cancelled")
            return success_response(data={"model_id": model_id}, message="not_found")
        except Exception as e:
            classify_and_raise(e, source="models.download_cancel")

    @endpoint("models.pause_download")
    async def pause_download(
        self, model_id: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict[str, Any]:
        """Pause an in-progress download."""
        try:
            from domain.infrastructure.download_manager import get_download_manager

            mgr = get_download_manager()
            if mgr.pause(model_id):
                safe_audit_log("model.pause", resource=model_id, detail="paused")
                return success_response(data={"model_id": model_id}, message="paused")
            return success_response(data={"model_id": model_id}, message="not_found")
        except Exception as e:
            classify_and_raise(e, source="models.download_pause")

    @endpoint("models.resume_download")
    async def resume_download(
        self, model_id: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict[str, Any]:
        """Resume a paused download."""
        try:
            from domain.infrastructure.download_manager import get_download_manager

            mgr = get_download_manager()
            if mgr.resume(model_id):
                safe_audit_log("model.resume", resource=model_id, detail="resumed")
                return success_response(data={"model_id": model_id}, message="resumed")
            return success_response(data={"model_id": model_id}, message="not_found")
        except Exception as e:
            classify_and_raise(e, source="models.download_resume")

    @endpoint("models.verify_download")
    async def verify_download(
        self, model_id: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict[str, Any]:
        """Verify a downloaded model's weight files against Hub SHA-256 checksums.
        Returns verification result and on-disk size."""
        from domain.infrastructure.hf_hub import (
            get_cache_dir,
            list_missing_files,
            verify_model,
        )

        try:

            def _verify():
                cache_dir = get_cache_dir(model_id)
                refs_main = cache_dir / "refs" / "main"
                if not refs_main.exists():
                    return {"status": "not_cached", "model_id": model_id}
                ok = verify_model(model_id)
                missing = list_missing_files(model_id)
                size_str = format_size_gb(compute_model_size_gb(model_id)) or "—"
                return {
                    "status": "verified" if ok else "corrupt",
                    "model_id": model_id,
                    "verified": ok,
                    "missing_files_count": len(missing),
                    "missing_files": missing,
                    "size_on_disk": size_str,
                }

            result = await asyncio.to_thread(_verify)
            return success_response(data=result)
        except Exception as e:
            logger.warning("Verify download failed (model=%s): %s", model_id, e)
            classify_and_raise(e, source="verify_download")

    @endpoint("models.retry_download")
    async def retry_download(
        self, model_id: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict[str, Any]:
        try:
            """Redownload a cached model (cleanup + fresh download)."""
            from domain.infrastructure.download_manager import (
                cleanup_incomplete,
                get_download_manager,
                is_download_complete,
            )

            if is_download_complete(model_id):
                cleanup_incomplete(model_id)
                safe_audit_log("model.retry_download", resource=model_id, detail="cleanup+restart")

            mgr = get_download_manager()
            if mgr.is_downloading(model_id):
                return success_response(data={"model_id": model_id}, message="already_downloading")

            asyncio.create_task(self._run_download(model_id, 0))
            return success_response(data={"model_id": model_id}, message="started")

        except Exception as e:
            classify_and_raise(e, source="models.retry_download")

    @endpoint("models.cache_usage")
    async def cache_usage(self) -> dict[str, Any]:
        try:
            """Total disk usage of the HuggingFace model cache (fast — walks blobs/ only)."""
            cache = _hf_cache_dir

            def _compute():
                if not cache.exists():
                    return 0, 0
                total = 0
                count = 0
                for entry in cache.iterdir():
                    if entry.name.startswith("models--") and entry.is_dir():
                        blobs = entry / "blobs"
                        if blobs.is_dir():
                            for f in blobs.iterdir():
                                if f.is_file():
                                    try:
                                        total += f.stat().st_size
                                    except OSError:
                                        pass  # Expected: file deleted between is_file() and stat()
                        count += 1
                return total, count

            total, count = await asyncio.to_thread(_compute)
            if total == 0 and count == 0:
                return success_response(
                    data={
                        "total_bytes": 0,
                        "total_gb": 0,
                        "model_count": 0,
                        "cache_dir": str(cache),
                    }
                )
            return success_response(
                data={
                    "total_bytes": total,
                    "total_gb": round(total / (1024**3), 2),
                    "model_count": count,
                    "cache_dir": str(cache),
                }
            )

        except Exception as e:
            classify_and_raise(e, source="models.cache_usage")

    @endpoint("models.download_qwen_gguf")
    async def download_qwen_gguf(self) -> dict:
        """Download Qwen2.5-0.5B-Instruct GGUF (Q4_K_M) from HuggingFace Hub.

        Returns the GGUF file as a streaming download.  The mobile app calls this
        to get the model for llama.rn inference.
        """
        from downcraft.download.http import download_file

        repo_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
        filename = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

        cache_dir = Path.home() / ".cache" / "sloughgpt" / "gguf"
        cached_path = cache_dir / filename

        def _check_cache():
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cached_path.exists()

        if await asyncio.to_thread(_check_cache):
            logger.info("Serving cached GGUF: %s", cached_path, extra={"tag": "MODEL"})
            return FileResponse(
                str(cached_path), media_type="application/octet-stream", filename=filename
            )

        logger.info(
            "Downloading Qwen GGUF from HuggingFace Hub (this may take a while)...",
            extra={"tag": "MODEL"},
        )
        try:
            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            _t0 = time.monotonic()

            def _download():
                download_file(url=url, dest=cached_path)

            await asyncio.to_thread(_download)
            _elapsed_ms = (time.monotonic() - _t0) * 1000
            file_size = await asyncio.to_thread(lambda: cached_path.stat().st_size)
            safe_audit_log(
                "model.download_gguf",
                resource=repo_id,
                detail=f"elapsed={_elapsed_ms:.0f}ms size={file_size}",
            )
            logger.info(
                "GGUF downloaded and cached: %s (%.0fms)",
                cached_path,
                _elapsed_ms,
                extra={"tag": "MODEL"},
            )
            return FileResponse(
                str(cached_path), media_type="application/octet-stream", filename=filename
            )
        except Exception as e:
            logger.warning("Download GGUF failed: %s", e)
            classify_and_raise(e, source="download_gguf")

    @endpoint("models.serve_model_file")
    async def serve_model_file(
        self,
        model_id: str,
        file_path: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ):
        """Serve a cached model file with optional SGZ1 compression.

        External clients can download model files from this endpoint.
        When the client sends ``Accept-Encoding: gzip``, the file is
        served compressed via SGZ1 format.
        """
        from starlette.responses import StreamingResponse

        backend = get_backend()
        if not backend.supports_compressed_serve():
            raise_error("Compressed serving not supported", status_code=501)

        result = backend.serve_compressed(model_id, file_path)
        if result is None:
            raise_error(f"File not found: {model_id}/{file_path}", status_code=404)

        headers = dict(result.get("headers", {}))
        if result.get("size"):
            headers["Content-Length"] = str(result["size"])

        return StreamingResponse(
            result["iterator"],
            media_type="application/octet-stream",
            headers=headers,
        )

    @endpoint("models.serve_model_manifest")
    async def serve_model_manifest(
        self,
        model_id: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ):
        """Serve a cached model's file manifest so a peer can download it.

        ``ExternalDownloadBackend`` fetches this route *first* and returns
        ``{"status": "error"}`` on 404, so without it a server running this
        codebase cannot serve models to ``POST /models/external/download``
        clients — the client seam existed but the server half did not.

        Deliberately **not** wrapped in ``success_response()``: the peer does
        ``json.loads(resp.read())`` and reads ``manifest["files"]`` at the top
        level. The standard ``{"status": ..., "data": ...}`` envelope would make
        it see zero files and fail with "Manifest contains no files".
        """
        from domain.infrastructure import build_model_manifest

        manifest = build_model_manifest(model_id)
        if manifest is None:
            raise_error(f"Model not cached: {model_id}", status_code=404)

        # Manifests describe an immutable snapshot; let peers revalidate cheaply.
        return JSONResponse(
            content=manifest,
            headers={"Cache-Control": "public, max-age=300"},
        )

    @endpoint("models.serve_model_file")
    async def serve_model_file_nested(
        self,
        model_id: str,
        file_path: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ):
        """Peer-shaped alias of :meth:`serve_model_file`.

        ``ExternalDownloadBackend`` builds ``/models/{id}/file/{path}``, while
        the pre-existing route serves ``/models/file/{id}/{path}`` — the same
        bytes under a different shape. Serving both keeps the existing contract
        untouched and is what actually lets a peer finish the download the
        manifest starts.

        Serves straight from the cache via ``resolve_cached_path`` (the same
        resolver the compressed path uses) instead of delegating: the existing
        route calls ``get_backend()`` with no argument, which raises
        ``TypeError`` against ``get_backend(name: str)``. A plain ``FileResponse``
        is a valid answer because ``CompressedDownloader`` passes bytes through
        unchanged when no ``lz4`` ``Content-Encoding`` is advertised.
        """
        from domain.infrastructure import resolve_cached_path

        resolved = resolve_cached_path(model_id, file_path)
        if resolved is None or not resolved.is_file():
            raise_error(f"File not found: {model_id}/{file_path}", status_code=404)
        return FileResponse(resolved, media_type="application/octet-stream")

    # ── External server management ───────────────────────────────────────

    _external_servers: dict[str, dict] = {}

    @endpoint("models.list_external_servers")
    async def list_external_servers(
        self,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """List registered external model servers."""
        return success_response(data=self._external_servers)

    @endpoint("models.register_external_server")
    async def register_external_server(
        self,
        req: ExternalServerRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Register an external model server for downloads."""
        self._external_servers[req.name] = {
            "url": req.url,
            "compressed": req.compressed,
        }
        logger.info("Registered external server: %s at %s", req.name, req.url)
        return success_response(
            data={"name": req.name, "url": req.url, "compressed": req.compressed},
            message="registered",
        )

    @endpoint("models.remove_external_server")
    async def remove_external_server(
        self,
        name: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Remove a registered external server."""
        if name not in self._external_servers:
            raise_error(f"Server not found: {name}", status_code=404)
        del self._external_servers[name]
        logger.info("Removed external server: %s", name)
        return success_response(data={"name": name}, message="removed")

    @endpoint("models.list_external_models")
    async def list_external_models(
        self,
        server: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """List available models from an external server."""
        if server not in self._external_servers:
            raise_error(f"Server not found: {server}", status_code=404)

        from domain.infrastructure.external_download import ExternalDownloadBackend

        cfg = self._external_servers[server]
        ExternalDownloadBackend(cfg["url"], compressed=cfg.get("compressed", True))

        import urllib.error
        import urllib.request

        def _fetch() -> list:
            url = f"{cfg['url']}/models"
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())

        try:
            # Off-loop: a slow/hung upstream must not stall the event loop
            # for every other route/stream (sync-core/async-host seam — same
            # pattern as routers/tokenizer.py _fetch).
            models = await asyncio.to_thread(_fetch)
        except Exception as e:
            raise_error(f"Failed to list models from {server}: {e}", status_code=502)

        return success_response(data={"server": server, "models": models})

    @endpoint("models.download_external")
    async def download_external_model(
        self,
        req: ExternalDownloadRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Download a model from an external server."""
        if req.server not in self._external_servers:
            raise_error(f"Server not found: {req.server}", status_code=404)

        from domain.infrastructure.external_download import ExternalDownloadBackend

        cfg = self._external_servers[req.server]
        backend = ExternalDownloadBackend(cfg["url"], compressed=cfg.get("compressed", True))

        from domain.infrastructure.download_manager import get_download_manager

        mgr = get_download_manager()
        if mgr.is_downloading(req.model_id):
            return success_response(data={"model_id": req.model_id}, message="already_downloading")

        asyncio.create_task(self._run_external_download(backend, req.model_id))
        return success_response(
            data={"model_id": req.model_id, "server": req.server},
            message="download_started",
        )

    async def _run_external_download(self, backend, model_id: str):
        """Background task that runs the external download."""
        from domain.infrastructure.download_manager import get_download_manager

        mgr = get_download_manager()
        try:
            result = await mgr.download_with(model_id, backend)
            if result.get("status") == "complete":
                logger.info("External download completed: %s", model_id)
        except Exception as e:
            logger.warning("External download failed for %s: %s", model_id, e)

    # ── Backend management ───────────────────────────────────────────────

    @endpoint("models.list_backends")
    async def list_backends(
        self,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """List available download backends with capabilities."""
        backends = {}

        # HuggingFace backend
        hf_info = {
            "name": "hf",
            "description": "HuggingFace Hub (with SGZ1 compression)",
            "active": True,
            "capabilities": {
                "compression": True,
                "compressed_serve": True,
                "cancel": True,
                "progress": True,
            },
        }
        try:
            from domain.infrastructure.hf_hub import HFDownloadBackend  # noqa: F401

            hf_info["available"] = True
        except ImportError:
            hf_info["available"] = False
        backends["hf"] = hf_info

        # External backend
        ext_info = {
            "name": "external",
            "description": "External download servers",
            "capabilities": {
                "compression": True,
                "compressed_serve": True,
                "cancel": True,
                "progress": True,
            },
        }
        try:
            from domain.infrastructure.external_download import (
                ExternalDownloadBackend,  # noqa: F401
            )

            ext_info["available"] = True
        except ImportError:
            ext_info["available"] = False
        backends["external"] = ext_info

        # Git backend
        git_info = {
            "name": "git",
            "description": "Git repositories (GitHub, GitLab, etc.)",
            "capabilities": {
                "compression": False,
                "compressed_serve": False,
                "cancel": True,
                "progress": False,
            },
        }
        try:
            from domain.infrastructure.git_download import GitBackend  # noqa: F401

            git_info["available"] = True
        except ImportError:
            git_info["available"] = False
        backends["git"] = git_info

        # Local backend
        local_info = {
            "name": "local",
            "description": "Local file copies",
            "capabilities": {
                "compression": False,
                "compressed_serve": False,
                "cancel": True,
                "progress": True,
            },
        }
        try:
            from domain.infrastructure.local_download import LocalFileBackend  # noqa: F401

            local_info["available"] = True
        except ImportError:
            local_info["available"] = False
        backends["local"] = local_info

        return success_response(data=backends)

    @endpoint("models.get_active_backend")
    async def get_active_backend(
        self,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Get the currently active download backend."""
        backend_type = "hf"

        try:
            from domain.infrastructure.external_download import (
                ExternalDownloadBackend,  # noqa: F401
            )

            # If external backend is configured, prefer it
            backend_type = "external"
        except ImportError:
            pass

        return success_response(
            data={
                "type": backend_type,
                "class": f"{backend_type.title()}Backend",
            }
        )

    @endpoint("models.set_active_backend")
    async def set_active_backend(
        self,
        backend_name: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Switch the active download backend."""
        if backend_name == "hf":
            return success_response(data={"type": "hf"}, message="switched_to_hf")
        elif backend_name == "external":
            raise_error(
                "Use /models/external/download for external server downloads",
                status_code=400,
            )
        else:
            raise_error(f"Unknown backend: {backend_name}", status_code=404)

    @endpoint("models.visual_model_load")
    async def visual_model_load(
        self,
        model_dir: str = "",
        model_id: str = "",
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        try:
            """Load a vision / multimodal model from a local directory.

            ``model_dir`` — path to the model directory on disk.
            ``model_id``  — HuggingFace model identifier (used when `model_dir` is empty).
            """
            logger.info(
                "Visual model load requested: dir=%s, id=%s",
                model_dir,
                model_id,
                extra={"tag": "MODEL"},
            )
            ctrl = get_models_controller()
            if model_dir:
                from pathlib import Path as _P

                resolved = _P(model_dir).resolve()
                allowed_parents = [_P("models").resolve(), _P("data").resolve(), _P.home()]
                if not any(str(resolved).startswith(str(p)) for p in allowed_parents):
                    raise_error("model_dir must be under models/, data/, or home", "E_BAD_REQUEST")
                result = ctrl.load_model_path(model_dir)
            elif model_id:
                result = ctrl.load_model(model_id)
            else:
                raise_error("Either model_dir or model_id required", "E_BAD_REQUEST")
            return wrap_controller_result(result)

        except Exception as e:
            classify_and_raise(e, source="models.visual_model_load")

    @endpoint("models.quantize_model")
    async def quantize_model(
        self, req: QuantizeRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        try:
            """Apply int8/int4 quantization to the currently loaded model.

            Works with both SloNet and HuggingFace models. Quantizes all
            linear layers in-place — no model reload required. The quantization
            state is reflected in the health endpoint's ``quantization`` field.

            Args:
                bits: 4 or 8 (default 8)
                mode: ``symmetric`` (default) or ``asymmetric``

            Returns:
                Quantization report with per-tensor error metrics and aggregate summary.
            """
            try:
                from domain.infrastructure.quantization import Quantine
            except ImportError:
                raise_error(
                    "Quantization not available — domain.infrastructure.quantization missing",
                    "E_BAD_REQUEST",
                )
            try:
                from domain.infrastructure.quant_core.wrapper import HAS_AVX2
            except ImportError:
                HAS_AVX2 = False
            import numpy as np

            bits = req.bits
            mode = req.mode

            if bits not in (4, 8):
                raise_error(f"bits must be 4 or 8, got {bits}", "E_BAD_REQUEST")
            if mode not in ("symmetric", "asymmetric"):
                raise_error(f"mode must be symmetric or asymmetric, got {mode}", "E_BAD_REQUEST")

            # Find the active provider (try SloNet first, then HuggingFace)
            from domain.models import get_provider

            provider = get_provider("slonet")
            model_type = "slonet"

            if provider is None:
                provider = get_provider("hf-default")
                model_type = "huggingface"

            if provider is None:
                raise_error("No model loaded", "E_BAD_REQUEST")

            model = getattr(provider, "_model", None)
            if model is None:
                raise_error("Provider has no model", "E_BAD_REQUEST")

            # Walk linear layers using the appropriate walker
            if model_type == "slonet":
                from domain.infrastructure.quantization import walk_slo_linears

                layers = walk_slo_linears(model)
            else:
                from domain.infrastructure.quantization import walk_hf_linears

                layers = walk_hf_linears(model)

            engine = Quantine(bits=bits, mode=mode)
            quantized_count = 0
            tensor_infos = {}
            for name, module in layers.items():
                weight = module.weight.data
                # Convert torch tensor to numpy if needed
                if hasattr(weight, "cpu"):
                    weight = weight.cpu().numpy().astype(np.float32).copy()
                else:
                    weight = np.asarray(weight, dtype=np.float32).copy()
                info = engine.quantize(f"{name}.weight", weight)
                if info.is_quantized:
                    if model_type == "slonet":
                        module.set_quantized_weight(info)
                    else:
                        # For HuggingFace models: monkey-patch forward with quantized path
                        from domain.infrastructure.quantization import QuantizedLinear

                        module._quant_info = info
                        ql = QuantizedLinear.from_linear(module, info)
                        module._ql = ql
                        module._orig_forward = module.forward
                        module.forward = ql.make_torch_forward()
                    tensor_infos[name] = info
                    quantized_count += 1

            # Persist quantized weights to disk for fast future loads
            if quantized_count > 0 and hasattr(provider, "_model_path"):
                model_path = provider._model_path
                if model_path:
                    from pathlib import Path

                    p = Path(str(model_path))
                    quant_npz = p.with_suffix(p.suffix + ".quant.npz")
                    engine.save_weights(str(quant_npz), tensor_infos)

            # Store the engine on the provider for health endpoint access
            provider._quant_engine = engine

            report = {
                "quantized": True,
                "bits": bits,
                "mode": mode,
                "model_type": model_type,
                "layers_quantized": quantized_count,
                "total_layers": len(layers),
                "summary": engine.summary(),
                "per_tensor": engine.error_report(),
                "avx2_enabled": False,
            }

            # Check if AVX2 extension is available
            try:
                from domain.infrastructure.quant_core.wrapper import HAS_AVX2

                report["avx2_enabled"] = bool(HAS_AVX2)
            except Exception as exc:
                logger.debug("AVX2 detection failed: %s", exc)
                report["avx2_enabled"] = False

            safe_audit_log(
                "model.quantize",
                resource=self._audit_model_id(provider),
                detail=f"bits={bits} mode={mode}",
                bits=bits,
                mode=mode,
                layers_quantized=quantized_count,
                model_type=model_type,
            )

            return success_response(data=report)

        except Exception as e:
            classify_and_raise(e, source="models.quantize_model")

    @endpoint("models.dequantize_model")
    async def dequantize_model(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Reset quantized model back to float32 weights.

        Clears quantization state from all linear layers. The model
        returns to its original float32 precision.

        Returns:
            Status report with number of layers reset.
        """
        from domain.models import get_provider

        provider = get_provider("slonet")
        model_type = "slonet"

        if provider is None:
            provider = get_provider("hf-default")
            model_type = "huggingface"

        if provider is None:
            raise_error("No model loaded", "E_BAD_REQUEST")

        model = getattr(provider, "_model", None)
        if model is None:
            raise_error("Provider has no model", "E_BAD_REQUEST")

        # Clear quantization state
        if model_type == "slonet":
            from domain.infrastructure.quantization import walk_slo_linears

            layers = walk_slo_linears(model)
            for _name, module in layers.items():
                module._quant_info = None
        else:
            from domain.infrastructure.quantization import walk_hf_linears

            layers = walk_hf_linears(model)
            for _name, module in layers.items():
                if hasattr(module, "_quant_info"):
                    module._quant_info = None
                    # Restore original forward if we patched it
                    if hasattr(module, "_orig_forward"):
                        module.forward = module._orig_forward
                        del module._orig_forward
                    if hasattr(module, "_ql"):
                        del module._ql

        # Clear the quantization engine
        provider._quant_engine = None

        safe_audit_log(
            "model.dequantize",
            resource=self._audit_model_id(provider),
            detail=f"model_type={model_type}",
            layers_reset=len(layers),
        )

        return success_response(
            data={
                "dequantized": True,
                "model_type": model_type,
                "layers_reset": len(layers),
            }
        )

    @endpoint("models.set_precision")
    async def set_precision(
        self, req: PrecisionRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        try:
            """Switch compute precision on-the-fly without model reload.

            Works on both GPU (fp16 via accelerator) and CPU (fp32/int8/int4).
            Already-loaded models switch immediately — no restart needed.

            Args:
                mode: ``"auto"`` (benchmark and pick fastest), ``"fp32"``, or
                      ``"fp16"``. On GPU without fp16 support, silently falls
                      back to ``"fp32"``.

            Returns:
                Active precision mode, benchmark results (if ``mode="auto"``),
                and per-format timing/quality.
            """
            import numpy as np

            from domain.slolib import get_accelerator, set_accelerator_precision

            acc = get_accelerator()
            acc_mode = req.mode

            result = {
                "accelerator": acc.name,
                "device_type": acc.device_type,
            }

            if acc.name == "cpu":
                # CPU path: use Quantine to select best format
                from domain.infrastructure.quantization import Quantine

                suggestion = Quantine.suggest_format()
                result["precision"] = suggestion["format"]
                result["bits"] = suggestion["bits"]
                result["reason"] = suggestion["reason"]
                result["benchmark"] = suggestion["benchmark"]
                result["fp16_mode"] = False

                # If int8/int4 selected, apply quantization
                if suggestion["format"] in ("int8", "int4"):
                    from domain.models import get_provider

                    provider = get_provider("slonet") or get_provider("hf-default")
                    if provider is not None:
                        model = getattr(provider, "_model", None)
                        if model is not None:
                            # Re-use existing quantize logic
                            from domain.infrastructure.quantization import (
                                Quantine,
                                walk_hf_linears,
                                walk_slo_linears,
                            )

                            engine = Quantine(bits=suggestion["bits"], mode="symmetric")
                            if hasattr(model, "layers"):
                                layers = walk_slo_linears(model)
                            else:
                                layers = walk_hf_linears(model)
                            quantized = 0
                            for name, module in layers.items():
                                weight = module.weight.data
                                if hasattr(weight, "cpu"):
                                    weight = weight.cpu().numpy().astype(np.float32).copy()
                                else:
                                    weight = np.asarray(weight, dtype=np.float32).copy()
                                info = engine.quantize(f"{name}.weight", weight)
                                if info.is_quantized:
                                    module._quant_info = info
                                    quantized += 1
                            result["layers_quantized"] = quantized
                            result["total_layers"] = len(layers)
            else:
                # GPU path: use accelerator's set_precision
                active = set_accelerator_precision(acc_mode)
                result["precision"] = active
                result["fp16_mode"] = acc._fp16_mode
                result["reason"] = f"Accelerator {acc.name} set to {active}"

            safe_audit_log(
                "model.precision",
                resource=acc.name,
                detail=str(result.get("precision", "unknown")),
                mode=acc_mode,
            )

            return wrap_controller_result(result)

        except Exception as e:
            classify_and_raise(e, source="models.set_precision")

    @endpoint("models.get_catalog")
    async def get_catalog(self) -> dict:
        """Get the persistent model catalog."""
        from domain.infrastructure.model_catalog import get_model_catalog

        catalog = get_model_catalog()
        return success_response(data=catalog.list_all())

    @endpoint("models.get_catalog_stats")
    async def get_catalog_stats(self) -> dict:
        """Get catalog statistics."""
        from domain.infrastructure.model_catalog import get_model_catalog

        catalog = get_model_catalog()
        return success_response(data=catalog.stats())

    @endpoint("models.get_conversion_status")
    async def get_conversion_status(self, model_id: str | None = None) -> dict:
        """Get model conversion/download status.

        Without model_id: returns all active conversions.
        With model_id: returns status for that specific model.
        """
        from domain.infrastructure.conversion_tracker import get_tracker

        tracker = get_tracker()

        if model_id:
            status = tracker.get(model_id)
            if not status:
                return success_response(data={"model_id": model_id, "stage": "idle", "progress": 0})
            return success_response(data=status)

        return success_response(data=tracker.get_active())

    @endpoint("models.get_process_guard")
    async def get_process_guard(self) -> dict:
        """Get ProcessGuard status.

        Returns enabled state, whether a guard is actively running, the
        guarded model id, and the guard health snapshot.
        """
        ctrl = get_models_controller()
        return success_response(data=ctrl.get_process_guard_status())

    @endpoint("models.set_process_guard")
    async def set_process_guard(
        self, req: ProcessGuardRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        try:
            """Enable or disable ProcessGuard at runtime.

            Body: ``{"enabled": true}`` or ``{"enabled": false}``

            Disabling stops any active guard. Enabling starts the guard for the
            currently loaded model if a .slnc file exists.
            """
            ctrl = get_models_controller()
            result = ctrl.set_process_guard_enabled(req.enabled)
            safe_audit_log(
                "model.process_guard", resource="process_guard", detail=f"enabled={req.enabled}"
            )
            return success_response(data=result)

        except Exception as e:
            classify_and_raise(e, source="models.set_process_guard")

    @endpoint("models.get_engine_status")
    async def get_engine_status(self) -> dict:
        """Get standalone inference engine status.

        Returns whether the engine subprocess is enabled, its PID, whether
        the client is connected, the model id, and the last 20 lines of stderr.
        """
        import state as server_state

        proc = getattr(server_state, "_inference_engine_proc", None)
        provider = getattr(server_state, "provider", None)
        stderr_tail = list(getattr(server_state, "_inference_engine_stderr", []))
        from domain.infrastructure.inference_client import InferenceClient

        is_client = isinstance(provider, InferenceClient)
        pid = proc.pid if proc is not None else None
        alive = proc.poll() is None if proc is not None else False
        health = {}
        metrics = {}
        if is_client and alive:
            try:
                health = await asyncio.to_thread(provider.health)
                metrics = health.get("metrics", {})
            except Exception as exc:
                import logging

                logging.getLogger("slo.models").warning("Provider health check failed: %s", exc)
                health = {"type": "error", "error": str(exc)}
        return success_response(
            data={
                "enabled": is_client,
                "pid": pid,
                "alive": alive,
                "model_id": getattr(provider, "model_id", None) if is_client else None,
                "health": health,
                "metrics": metrics,
                "stderr_tail": stderr_tail[-20:],
            }
        )

    @endpoint("models.reload_engine")
    async def reload_engine(
        self, req: dict[str, Any], auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        try:
            """Hot-reload the inference engine model.

            Body: ``{"model_id": "Qwen/Qwen2.5-0.5B-Instruct", "slnc_path": "/path/to/model.slnc"}``

            Swaps the model in the engine subprocess without restarting.
            """
            import state as server_state

            from domain.infrastructure.inference_client import InferenceClient

            provider = getattr(server_state, "provider", None)
            if not isinstance(provider, InferenceClient):
                return success_response(data={"error": "Inference engine not active"})
            model_id = req.get("model_id", server_state.model_type)
            slnc_path = req.get("slnc_path")
            result = await asyncio.to_thread(provider.reload, model_id, slnc_path)
            if result.get("type") == "reload_ok":
                server_state.model_type = model_id
            safe_audit_log(
                "model.reload_engine",
                resource=model_id or "unknown",
                detail=f"result={result.get('type', 'unknown')}",
            )
            return success_response(data=result)
        except Exception as e:
            classify_and_raise(e, source="models.reload_engine")

    @endpoint("models.debug_providers")
    async def debug_providers(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Diagnostic endpoint: full provider chain status.

        Returns the registered providers, their types, text provider wiring,
        and model state — useful for debugging "no provider" or text generation
        failures.
        """
        import state as server_state

        from domain.models import get_provider, list_providers

        providers = {}
        for name in list_providers():
            p = get_provider(name)
            info: dict[str, Any] = {
                "type": type(p).__name__,
                "module": type(p).__module__,
            }
            # ProviderRouter-specific: show text provider name
            if hasattr(p, "_text_name"):
                info["text_provider"] = p._text_name
            if hasattr(p, "_processors"):
                info["processors"] = [type(proc).__name__ for proc in p._processors]
            # SloNetChatProvider-specific
            if hasattr(p, "_model_id"):
                info["model_id"] = p._model_id
            if hasattr(p, "_server"):
                srv = p._server
                info["server"] = {
                    "type": type(srv).__name__,
                    "has_circuit_breaker": hasattr(srv, "_circuit_breaker"),
                }
            providers[name] = info

        model_state = {
            "model": type(server_state.model).__name__ if server_state.model is not None else None,
            "model_type": getattr(server_state, "model_type", None),
            "tokenizer": type(server_state.tokenizer).__name__
            if getattr(server_state, "tokenizer", None) is not None
            else None,
            "provider": type(server_state.provider).__name__
            if getattr(server_state, "provider", None) is not None
            else None,
        }

        from startup_progress import STARTUP_PHASE

        return success_response(
            data={
                "providers": providers,
                "default_provider": list_providers()[0] if "default" in list_providers() else None,
                "model_state": model_state,
                "startup_phase": STARTUP_PHASE.get("phase", "unknown"),
            }
        )

    @staticmethod
    @endpoint("models.memory_cleanup")
    async def memory_cleanup(auth_user: dict = Depends(require_auth_if_enabled)):
        """Force an immediate memory cleanup cycle.

        Triggers critical-level cleanup: drops KV caches, forces GC,
        releases idle model weights, runs malloc_trim, and stops guard
        subprocesses. Returns the cleanup result and current memory stats.
        """
        from domain.infrastructure.memory_pressure import get_memory_pressure_monitor

        monitor = get_memory_pressure_monitor()
        result = monitor.force_cleanup()
        stats = monitor.stats()
        return success_response(data={"cleanup": result, "stats": stats})

    @staticmethod
    @endpoint("models.memory_pressure")
    async def memory_pressure(auth_user: dict = Depends(require_auth_if_enabled)):
        """Return current memory pressure stats (level, thresholds, counters)."""
        from domain.infrastructure.memory_pressure import get_memory_pressure_monitor

        monitor = get_memory_pressure_monitor()
        return success_response(data=monitor.stats())


# ── Verbatim move: infer.py (cfg, schemas, InferRouter) — card 20261101_154054.
# Dropped: infer's own `logger` line (a second module-level assignment would rebind
# the shared global and route every record through slo.infer) and the module tail
# `router = InferRouter().router` (replaced by the aggregate facade below).
cfg = ServerConfig.from_env()


class InferRequest(BaseModel):
    """Text generation request."""

    prompt: str = Field(..., min_length=1, max_length=50000)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.85, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=500)
    repetition_penalty: float = Field(default=1.15, ge=0.5, le=2.0)
    model: str | None = None
    response_format: str | None = Field(default=None, description='"text" or "json"')


class InferResponse(BaseModel):
    """Generation result."""

    text: str
    model: str
    tokens_generated: int = 0
    elapsed_ms: float = 0


class EmbedRequest(BaseModel):
    """Embedding request."""

    text: str = Field(..., min_length=1, max_length=50000)
    model: str | None = None


class EmbedResponse(BaseModel):
    """Embedding result."""

    embedding: list[float]
    dimensions: int
    model: str


class TokenizeRequest(BaseModel):
    """Tokenization request."""

    text: str = Field(..., min_length=1, max_length=50000)
    model: str | None = None


class TokenizeResponse(BaseModel):
    """Tokenization result."""

    tokens: list[str]
    ids: list[int]
    count: int


class DetokenizeRequest(BaseModel):
    """Detokenization request."""

    ids: list[int] = Field(..., max_length=10000)
    model: str | None = None


class DetokenizeResponse(BaseModel):
    """Detokenization result."""

    text: str
    count: int


class InferHealthResponse(BaseModel):
    """Engine health status."""

    status: str
    model_loaded: bool
    model_id: str | None = None
    engine_type: str | None = None
    has_streaming: bool = True
    has_embedding: bool = False


class InferInfoResponse(BaseModel):
    """Loaded model metadata."""

    model_id: str
    model_type: str
    num_parameters: int = 0
    vocab_size: int = 0
    max_context: int = 0
    num_layers: int = 0
    has_tokenizer: bool = False
    has_streaming: bool = True
    has_embedding: bool = False
    extra: dict = Field(default_factory=dict)


class InferRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/infer", tags=["infer"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("", self.infer, methods=["POST"], response_model=InferResponse)
        self.router.add_api_route(
            "/stream", self.infer_stream, methods=["POST"], response_model=None
        )
        self.router.add_api_route(
            "/embed", self.infer_embed, methods=["POST"], response_model=EmbedResponse
        )
        self.router.add_api_route(
            "/tokenize", self.infer_tokenize, methods=["POST"], response_model=TokenizeResponse
        )
        self.router.add_api_route(
            "/detokenize",
            self.infer_detokenize,
            methods=["POST"],
            response_model=DetokenizeResponse,
        )
        self.router.add_api_route(
            "/health", self.infer_health, methods=["GET"], response_model=InferHealthResponse
        )
        self.router.add_api_route(
            "/info", self.infer_info, methods=["GET"], response_model=InferInfoResponse
        )

    # --- Helpers ---

    def _get_model(self):
        """Get the currently loaded model from server state."""
        try:
            import state as _state

            return _state.model
        except ImportError:
            return None

    def _get_model_interface(self):
        """Get the loaded model wrapped as ModelInterface."""
        model = self._get_model()
        if model is None:
            return None
        from domain.models import ModelInterface

        if isinstance(model, ModelInterface):
            return model
        return None

    def _sse_event(self, stream, phase, status, data=None, meta=None, message=""):
        """SSE event helper."""
        import json

        return (
            "data: "
            + json.dumps(
                {
                    "stream": stream,
                    "phase": phase,
                    "status": status,
                    "data": data or {},
                    "meta": meta or {},
                    "message": message,
                }
            )
            + "\n\n"
        )

    # --- Endpoints ---

    @endpoint("infer.infer")
    async def infer(
        self, req: InferRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> InferResponse:
        """Generate text from a prompt.

        Single non-streaming generation endpoint. Delegates to the active provider.
        """
        # Block inference under memory pressure
        try:
            from domain.infrastructure.memory_pressure import (
                PressureLevel,
                get_memory_pressure_monitor,
            )

            if get_memory_pressure_monitor().check() == PressureLevel.EMERGENCY:
                raise_error(
                    "System memory too low for inference — try again later",
                    "E_MEMORY_PRESSURE",
                    status_code=503,
                )
        except ImportError as exc:
            logger.debug("Memory pressure guard unavailable: %s", exc)

        from domain.models import get_provider
        from routers.inference import _get_model_status

        ms = _get_model_status()
        if not ms["ready"]:
            raise_error(ms["reason"], ms["code"], status_code=ms["status"])

        provider = get_provider("default")
        if provider is None:
            raise_error(
                "No provider available — load a model first", "E_INFRA_REGISTRY", status_code=503
            )

        prompt_text = req.prompt
        if req.response_format == "json":
            prompt_text = (
                f"{req.prompt}\n\n"
                "Respond ONLY with valid JSON. No markdown, no explanation, no code fences."
            )
        provider_messages = [{"role": "user", "content": prompt_text}]
        start = datetime.datetime.now()
        try:
            result = await provider.chat(
                provider_messages,
                max_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                repetition_penalty=req.repetition_penalty,
            )
            elapsed_ms = (datetime.datetime.now() - start).total_seconds() * 1000
            tokens = len(result.split())
            try:
                from domain.infrastructure.server_state import get_server_state

                get_server_state().record_inference(
                    tokens=tokens, elapsed_ms=elapsed_ms, model=req.model
                )
            except Exception as e:
                logger.warning("Failed to record inference metrics: %s", e)
            model = self._get_model()
            model_name = req.model or (
                getattr(model, "model_id", None) or type(model).__name__ if model else "unknown"
            )
            safe_audit_log(
                "infer.generate",
                resource=model_name,
                detail=f"elapsed={elapsed_ms:.0f}ms tokens={tokens}",
            )
            return InferResponse(
                text=result,
                model=model_name,
                tokens_generated=tokens,
                elapsed_ms=round(elapsed_ms, 1),
            )
        except AppError as e:
            classify_and_raise(e, source="infer.infer")
        except Exception as e:
            logger.warning("Inference failed: %s", e)
            classify_and_raise(e, source="infer")

    @endpoint("infer.infer_stream")
    async def infer_stream(
        self,
        req: InferRequest,
        request: Request,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> AsyncGenerator[str, None]:
        """Stream generated tokens as SSE.

        Yields standard envelope: {stream, phase, status, data: {token}, meta}.
        """
        # Block streaming inference under memory pressure
        try:
            from domain.infrastructure.memory_pressure import (
                PressureLevel,
                get_memory_pressure_monitor,
            )

            if get_memory_pressure_monitor().check() == PressureLevel.EMERGENCY:

                async def error_stream() -> AsyncIterator[str]:
                    """error_stream."""
                    yield self._sse_error(
                        "infer",
                        "IDLE",
                        "System memory too low for inference — try again later",
                        code="E_MEMORY_PRESSURE",
                        http_status=503,
                    )

                return StreamingResponse(error_stream(), media_type="text/event-stream")
        except ImportError as exc:
            logger.debug("Memory pressure guard unavailable: %s", exc)

        try:
            from routers.inference import _get_model_status

            ms = _get_model_status()
            if not ms["ready"]:

                async def error_stream() -> AsyncIterator[str]:
                    """error_stream."""
                    yield self._sse_error(
                        "infer",
                        "IDLE",
                        ms["reason"],
                        code=ms["code"],
                        http_status=ms["status"],
                    )

                return StreamingResponse(error_stream(), media_type="text/event-stream")

            async def generate() -> AsyncIterator[str]:
                """generate."""
                from domain.models import get_provider

                provider = get_provider("default")
                if provider is None:
                    yield self._sse_error(
                        "infer",
                        "IDLE",
                        "No provider available — load a model first",
                        code="E_INFRA_REGISTRY",
                        http_status=503,
                    )
                    return

                prompt_text = req.prompt
                if req.response_format == "json":
                    prompt_text = (
                        f"{req.prompt}\n\n"
                        "Respond ONLY with valid JSON. No markdown, no explanation, no code fences."
                    )
                provider_messages = [{"role": "user", "content": prompt_text}]
                start = datetime.datetime.now()
                token_count = 0
                _token_gen_start = _time.time()
                _max_token_wait_s = getattr(cfg, "generate_timeout", 60)
                _heartbeat_interval_s = 10.0
                _last_heartbeat = _time.time()
                try:
                    async for token in provider.chat_stream(
                        provider_messages,
                        max_tokens=req.max_new_tokens,
                        temperature=req.temperature,
                        top_p=req.top_p,
                        top_k=req.top_k,
                        repetition_penalty=req.repetition_penalty,
                    ):
                        if await request.is_disconnected():
                            return
                        if token:
                            _token_gen_start = _time.time()
                            token_count += 1
                            yield self._sse_token("infer", token)
                        else:
                            now = _time.time()
                            if now - _last_heartbeat >= _heartbeat_interval_s:
                                yield ": heartbeat\n\n"
                                _last_heartbeat = now
                        elapsed_since_token = _time.time() - _token_gen_start
                        if elapsed_since_token > _max_token_wait_s:
                            logger.warning(
                                "Infer stream stalled for %.1fs (limit=%.1fs)",
                                elapsed_since_token,
                                _max_token_wait_s,
                                extra={"tag": "INF"},
                            )
                            yield self._sse_error(
                                "infer",
                                "TIMEOUT",
                                f"Generation stalled for {elapsed_since_token:.0f}s",
                                code="MODEL_TIMEOUT",
                                http_status=504,
                            )
                            return
                except Exception as e:
                    from domain.infrastructure import (
                        classify_exception,
                        emit_error_event,
                    )

                    err = classify_exception(e)
                    emit_error_event(err, source="infer_stream")
                    yield self._sse_error("infer", err.code, err.user_message)
                    return
                elapsed_ms = (datetime.datetime.now() - start).total_seconds() * 1000
                try:
                    from domain.infrastructure.server_state import get_server_state

                    get_server_state().record_inference(
                        tokens=token_count, elapsed_ms=elapsed_ms, model=req.model
                    )
                except Exception as e:
                    logger.warning("Failed to record inference metrics: %s", e)
                safe_audit_log(
                    "infer.stream",
                    resource=req.model or "default",
                    detail=f"elapsed={elapsed_ms:.0f}ms tokens={token_count}",
                )
                import json

                yield (
                    "data: "
                    + json.dumps(
                        {
                            "stream": "infer",
                            "phase": "STREAMING",
                            "status": "complete",
                            "data": {},
                            "meta": {"tokens": token_count, "elapsed_ms": round(elapsed_ms, 1)},
                        }
                    )
                    + "\n\n"
                )

            return StreamingResponse(generate(), media_type="text/event-stream")

        except Exception as e:
            classify_and_raise(e, source="infer.infer_stream")

    @endpoint("infer.infer_embed")
    async def infer_embed(
        self, req: EmbedRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> EmbedResponse:
        try:
            """Return embedding vector for text.

            Uses the loaded model's embed() method if available, otherwise falls back
            to the n-gram TF-IDF embedder.
            """
            _t0 = _time.monotonic()
            model = self._get_model_interface()
            if model is not None and hasattr(model, "embed"):
                model_name = req.model or getattr(model, "model_id", "unknown")
                try:
                    import numpy as np

                    vec = model.embed(req.text)
                    if isinstance(vec, np.ndarray):
                        vec = vec.tolist()
                    _elapsed_ms = (_time.monotonic() - _t0) * 1000
                    safe_audit_log(
                        "infer.embed",
                        resource=model_name,
                        detail=f"elapsed={_elapsed_ms:.0f}ms dims={len(vec)}",
                    )
                    return EmbedResponse(embedding=vec, dimensions=len(vec), model=model_name)
                except NotImplementedError:
                    logger.debug(
                        "Model %s does not support embed, falling back to n-gram", model_name
                    )
                except Exception as e:
                    logger.debug("Model embed failed, falling back to n-gram: %s", e)

            # Fallback: n-gram TF-IDF embedder
            try:
                import numpy as np

                from domain.inference import _ngram_embed

                vec = _ngram_embed(req.text)
                if isinstance(vec, np.ndarray):
                    vec = vec.tolist()
                model_name = req.model or "ngram-tfidf"
                _elapsed_ms = (_time.monotonic() - _t0) * 1000
                safe_audit_log(
                    "infer.embed",
                    resource=model_name,
                    detail=f"elapsed={_elapsed_ms:.0f}ms dims={len(vec)} fallback=ngram",
                )
                return EmbedResponse(embedding=vec, dimensions=len(vec), model=model_name)
            except ImportError:
                raise_error("No embedding backend available", "E_BAD_REQUEST", status_code=503)

        except Exception as e:
            classify_and_raise(e, source="infer.infer_embed")

    @endpoint("infer.infer_tokenize")
    async def infer_tokenize(
        self, req: TokenizeRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> TokenizeResponse:
        try:
            """Tokenize text into token IDs and strings.

            Uses the loaded model's tokenizer if available, otherwise falls back to
            simple byte-level tokenization.
            """
            _t0 = _time.monotonic()
            model = self._get_model_interface()
            if model is not None:
                try:
                    tokenizer = getattr(model, "_tokenizer", None) or getattr(
                        model, "tokenizer", None
                    )
                    if tokenizer is not None and hasattr(tokenizer, "encode"):
                        ids = tokenizer.encode(req.text)
                        tokens = [getattr(tokenizer, "itos", {}).get(i, f"<{i}>") for i in ids]
                        model_name = req.model or "model-tokenizer"
                        _elapsed_ms = (_time.monotonic() - _t0) * 1000
                        safe_audit_log(
                            "infer.tokenize",
                            resource=model_name,
                            detail=f"elapsed={_elapsed_ms:.0f}ms tokens={len(ids)}",
                        )
                        return TokenizeResponse(tokens=tokens, ids=ids, count=len(ids))
                except Exception as e:
                    logger.debug("Model tokenize failed, falling back to byte-level: %s", e)

            # Fallback: byte-level tokenization (always works, no dependencies)
            ids = list(req.text.encode("utf-8"))
            tokens = [f"b{b}" for b in ids]
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            safe_audit_log(
                "infer.tokenize",
                resource="byte-level",
                detail=f"elapsed={_elapsed_ms:.0f}ms tokens={len(ids)}",
            )
            return TokenizeResponse(tokens=tokens, ids=ids, count=len(ids))

        except Exception as e:
            classify_and_raise(e, source="infer.infer_tokenize")

    @endpoint("infer.infer_detokenize")
    async def infer_detokenize(
        self, req: DetokenizeRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> DetokenizeResponse:
        try:
            """Convert token IDs back to text."""
            _t0 = _time.monotonic()
            model = self._get_model_interface()
            if model is not None:
                try:
                    tokenizer = getattr(model, "_tokenizer", None) or getattr(
                        model, "tokenizer", None
                    )
                    if tokenizer is not None and hasattr(tokenizer, "decode"):
                        text = tokenizer.decode(req.ids)
                        _elapsed_ms = (_time.monotonic() - _t0) * 1000
                        safe_audit_log(
                            "infer.detokenize",
                            resource="model",
                            detail=f"elapsed={_elapsed_ms:.0f}ms ids={len(req.ids)}",
                        )
                        return DetokenizeResponse(text=text, count=len(req.ids))
                except Exception as exc:
                    logger.debug("Tokenizer decode failed: %s", exc)

            # Fallback: interpret IDs as byte values
            try:
                text = bytes(req.ids).decode("utf-8", errors="replace")
                _elapsed_ms = (_time.monotonic() - _t0) * 1000
                safe_audit_log(
                    "infer.detokenize",
                    resource="byte-level",
                    detail=f"elapsed={_elapsed_ms:.0f}ms ids={len(req.ids)}",
                )
                return DetokenizeResponse(text=text, count=len(req.ids))
            except Exception as exc:
                import logging

                logging.getLogger("slo.infer").warning("Detokenize failed: %s", exc)
                return DetokenizeResponse(text="", count=len(req.ids))

        except Exception as e:
            classify_and_raise(e, source="infer.infer_detokenize")

    @endpoint("infer.infer_health")
    async def infer_health(self) -> InferHealthResponse:
        """Engine health — whether a model is loaded and what capabilities it has."""
        model = self._get_model_interface()
        if model is None:
            return InferHealthResponse(
                status="no_model",
                model_loaded=False,
                has_streaming=False,
                has_embedding=False,
            )

        info = model.info() if hasattr(model, "info") else None
        return InferHealthResponse(
            status="ready",
            model_loaded=True,
            model_id=info.model_id if info else "unknown",
            engine_type=info.model_type if info else type(model).__name__,
            has_streaming=info.has_streaming if info else hasattr(model, "generate_stream"),
            has_embedding=info.has_embedding if info else hasattr(model, "embed"),
        )

    @endpoint("infer.infer_info")
    async def infer_info(self) -> InferInfoResponse:
        """Metadata about the currently loaded model."""
        from routers.inference import _get_model_status

        ms = _get_model_status()
        if not ms["ready"]:
            raise_error(ms["reason"], ms["code"], status_code=ms["status"])

        model = self._get_model_interface()

        info = model.info() if hasattr(model, "info") else None
        if info is None:
            return InferInfoResponse(
                model_id="unknown",
                model_type=type(model).__name__,
                num_parameters=model.num_parameters() if hasattr(model, "num_parameters") else 0,
            )

        return InferInfoResponse(
            model_id=info.model_id,
            model_type=info.model_type,
            num_parameters=info.num_parameters,
            vocab_size=info.vocab_size,
            max_context=info.max_context,
            num_layers=info.num_layers,
            has_tokenizer=info.has_tokenizer,
            has_streaming=info.has_streaming,
            has_embedding=info.has_embedding,
            extra=info.extra,
        )

        # --- SSE helpers (local, avoid import issues) ---

    def _sse_token(
        self,
        stream: str,
        token: str,
        done: bool = False,
        meta: dict = None,
        elapsed_ms: float = None,
    ) -> str:
        phase = "STREAMING"
        status = "complete" if done else "working"
        m = dict(meta) if meta else {}
        if done and elapsed_ms is not None:
            m["elapsed_ms"] = round(elapsed_ms, 1)
        return self._sse_event(stream, phase, status, {"token": token}, m, "")

    def _sse_error(
        self,
        stream: str,
        phase: str,
        error: str,
        meta: dict = None,
        code: str = None,
        http_status: int = None,
    ) -> str:
        data = {"error": error}
        if code is not None:
            data["code"] = code
        if http_status is not None:
            data["http_status"] = http_status
        return self._sse_event(stream, phase, "error", data, meta or {}, f"Error: {error}")


# ── Verbatim move: registry.py (RegistryRouter) — card 20261101_154054. ──
# Module tail `router = RegistryRouter().router` replaced by the aggregate facade.
class RegistryRouter:
    """Registry Router - Proxies to the real ModelRegistry."""

    def __init__(self):
        self.router = APIRouter(prefix="/registry", tags=["registry"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(path="/models", endpoint=self.list_models, methods=["GET"])
        self.router.add_api_route(
            path="/models/{model_id}", endpoint=self.get_model, methods=["GET"]
        )
        self.router.add_api_route(path="/best", endpoint=self.get_best_model, methods=["GET"])
        self.router.add_api_route(path="/stats", endpoint=self.get_registry_stats, methods=["GET"])
        self.router.add_api_route(path="/artifacts", endpoint=self.list_artifacts, methods=["GET"])
        self.router.add_api_route(path="/verify", endpoint=self.verify_artifacts, methods=["GET"])

    def _get_registry(self):
        from domain.infrastructure.model_registry import get_model_registry

        return get_model_registry()

    @endpoint("registry.list")
    async def list_models(self) -> dict:
        """List registered models from the live registry."""
        _t0 = _time.monotonic()
        reg = self._get_registry()
        models = reg.list_models()
        _elapsed_ms = (_time.monotonic() - _t0) * 1000
        safe_audit_log(
            "registry.list",
            resource="models",
            detail=f"elapsed={_elapsed_ms:.0f}ms count={len(models)}",
        )
        return success_response(data={"models": models, "count": len(models)})

    @endpoint("registry.get")
    async def get_model(self, model_id: str) -> dict:
        """Get model details from the live registry."""
        _t0 = _time.monotonic()
        reg = self._get_registry()
        models = reg.list_models()
        found = next((m for m in models if m.get("model_id") == model_id), None)
        if not found:
            raise_error("Model not found", "E_NOT_FOUND", status_code=404)
        _elapsed_ms = (_time.monotonic() - _t0) * 1000
        safe_audit_log("registry.get", resource=model_id, detail=f"elapsed={_elapsed_ms:.0f}ms")
        return success_response(data=found)

    @endpoint("registry.best")
    async def get_best_model(self) -> dict:
        """Get best performing model by metrics."""
        _t0 = _time.monotonic()
        reg = self._get_registry()
        health = reg.health_summary()
        _elapsed_ms = (_time.monotonic() - _t0) * 1000
        safe_audit_log("registry.best", resource="health", detail=f"elapsed={_elapsed_ms:.0f}ms")
        return success_response(data=health)

    @endpoint("registry.stats")
    async def get_registry_stats(self) -> dict:
        """Retrieve aggregate statistics for the model registry."""
        _t0 = _time.monotonic()
        reg = self._get_registry()
        health = reg.health_summary()
        _elapsed_ms = (_time.monotonic() - _t0) * 1000
        safe_audit_log("registry.stats", resource="health", detail=f"elapsed={_elapsed_ms:.0f}ms")
        return success_response(data=health)

    @endpoint("registry.artifacts")
    async def list_artifacts(
        self,
        kind: str | None = Query(None, description="Filter by kind"),
        q: str | None = Query(None, description="Substring search on id/name"),
        lineage: bool = Query(
            False,
            description="Attach lineage parents per artifact (sidecar + metadata union)",
        ),
    ) -> dict:
        """List artifacts across all kinds (Stage 1: read-only index)."""
        import asyncio as _aio

        from domain.infrastructure import artifact_registry as _ar

        if kind is not None and kind not in _ar.ARTIFACT_KINDS:
            raise_error(
                f"Unknown kind {kind!r} (expected one of {', '.join(_ar.ARTIFACT_KINDS)})",
                "E_BAD_REQUEST",
                status_code=400,
            )
        _t0 = _time.monotonic()
        artifacts = await _aio.to_thread(_ar.list_artifacts, kind, q, with_provenance=lineage)
        _elapsed_ms = (_time.monotonic() - _t0) * 1000
        safe_audit_log(
            "registry.artifacts",
            resource=kind or "all",
            detail=f"elapsed={_elapsed_ms:.0f}ms count={len(artifacts)}",
        )
        return success_response(data={"artifacts": artifacts, "count": len(artifacts)})

    @endpoint("registry.verify")
    async def verify_artifacts(
        self,
        kind: str | None = Query(None, description="Filter by kind"),
        q: str | None = Query(None, description="Substring search on id/name"),
    ) -> dict:
        """Hash/size integrity check over artifacts (read-only, hashed in thread)."""
        import asyncio as _aio

        from domain.infrastructure import artifact_registry as _ar

        if kind is not None and kind not in _ar.ARTIFACT_KINDS:
            raise_error(
                f"Unknown kind {kind!r} (expected one of {', '.join(_ar.ARTIFACT_KINDS)})",
                "E_BAD_REQUEST",
                status_code=400,
            )
        _t0 = _time.monotonic()
        report = await _aio.to_thread(_ar.verify, kind, q)
        _elapsed_ms = (_time.monotonic() - _t0) * 1000
        safe_audit_log(
            "registry.verify",
            resource=kind or "all",
            detail=f"elapsed={_elapsed_ms:.0f}ms summary={report['summary']}",
        )
        return success_response(data=report)


# Aggregate facade (route-map card 3/7): ONE module-level router serving
# /models/*, /infer/* and /registry/* — sub-router prefixes bake at
# add-time, so every path stays byte-identical to the three old modules.
_instance = ModelsRouter()
_infer_instance = InferRouter()
_registry_instance = RegistryRouter()

router = APIRouter()
router.include_router(_instance.router)
router.include_router(_infer_instance.router)
router.include_router(_registry_instance.router)
