"""
Models Router - MVC View layer
Uses ModelsController for business logic
"""
import asyncio
import os
import logging
import re
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path

from schemas.models import ModelInfo, LoadModelRequest, LoadModelResponse, ModelStatus
from schemas.common import StandardResponse, success_response, error_response, wrap_controller_result
from controllers.models import get_models_controller
from infrastructure.auth import require_auth_if_enabled

logger = logging.getLogger(__name__)

from domains.infrastructure.model_size import compute_model_size_gb, format_size_gb, is_model_cached

# Module-level so tests can patch ``routers.models._hf_cache_dir``; resolved at call time.
_hf_cache_dir = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"


class ExportRequest(BaseModel):
    output_path: str = "models/exported"
    format: str = "sou"
    include_tokenizer: bool = True


class DownloadRequest(BaseModel):
    model_id: str
    total_bytes_hint: int = 0


class QuantizeRequest(BaseModel):
    """Request body for POST /models/quantize."""
    bits: int = 8
    mode: str = "symmetric"


class PrecisionRequest(BaseModel):
    """Request body for POST /models/precision."""
    mode: str = "auto"


class ModelsRouter:
    """Models Router - MVC View layer."""

    def __init__(self):
        self.router = APIRouter(prefix="/models", tags=["models"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(path="", endpoint=self.list_models, methods=["GET"], response_model=StandardResponse[List[ModelInfo]])
        self.router.add_api_route(path="/load", endpoint=self.load_model, methods=["POST"])
        self.router.add_api_route(path="/unload", endpoint=self.unload_model, methods=["POST"])
        self.router.add_api_route(path="/current", endpoint=self.current_model, methods=["GET"])
        self.router.add_api_route(path="/hf", endpoint=self.list_hf_models, methods=["GET"])
        self.router.add_api_route(path="/logs", endpoint=self.get_model_logs, methods=["GET"])
        self.router.add_api_route(path="/export", endpoint=self.export_model, methods=["POST"], tags=["models"])
        self.router.add_api_route(path="/export/formats", endpoint=self.get_export_formats, methods=["GET"], tags=["models"])
        self.router.add_api_route(path="/download", endpoint=self.start_download, methods=["POST"])
        self.router.add_api_route(path="/download/{model_id:path}", endpoint=self.get_download_status, methods=["GET"])
        self.router.add_api_route(path="/downloads", endpoint=self.list_downloads, methods=["GET"])
        self.router.add_api_route(path="/download/{model_id:path}/cancel", endpoint=self.cancel_download, methods=["POST"])
        self.router.add_api_route(path="/download/{model_id:path}/verify", endpoint=self.verify_download, methods=["POST"])
        self.router.add_api_route(path="/download/{model_id:path}/retry", endpoint=self.retry_download, methods=["POST"])
        self.router.add_api_route(path="/cache-usage", endpoint=self.cache_usage, methods=["GET"])
        self.router.add_api_route(path="/download/qwen-gguf", endpoint=self.download_qwen_gguf, methods=["GET"])
        self.router.add_api_route(path="/visual-load", endpoint=self.visual_model_load, methods=["POST"])
        self.router.add_api_route(path="/quantize", endpoint=self.quantize_model, methods=["POST"])
        self.router.add_api_route(path="/dequantize", endpoint=self.dequantize_model, methods=["POST"])
        self.router.add_api_route(path="/precision", endpoint=self.set_precision, methods=["POST"])
        self.router.add_api_route(path="/catalog", endpoint=self.get_catalog, methods=["GET"])
        self.router.add_api_route(path="/catalog/stats", endpoint=self.get_catalog_stats, methods=["GET"])
        self.router.add_api_route(path="/conversion-status", endpoint=self.get_conversion_status, methods=["GET"])
        self.router.add_api_route(path="/process-guard", endpoint=self.get_process_guard, methods=["GET"])
        self.router.add_api_route(path="/process-guard", endpoint=self.set_process_guard, methods=["POST"])

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
        """Generate a plain-language description of a model."""
        parts = []
        name = model_id.split("/")[-1] if "/" in model_id else model_id

        # Size description
        if parameters:
            if parameters < 150_000_000:
                parts.append("Small, fast model")
            elif parameters < 1_000_000_000:
                parts.append("Medium-sized model")
            else:
                parts.append("Large model")
        else:
            parts.append("Model")

        # Chat capability
        if any(kw in model_id.lower() for kw in ["instruct", "chat", "qwen"]):
            parts.append("good for conversations")
        elif any(kw in model_id.lower() for kw in ["code", "starcoder"]):
            parts.append("good for code")
        else:
            parts.append("good for text generation")

        # Speed hint
        if parameters and parameters < 500_000_000:
            parts.append("runs fast on CPU")

        if loaded:
            parts.append("(currently loaded)")

        return " — ".join(parts[:2]) + (f". {parts[2]}" if len(parts) > 2 else "") + "."

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
            after = name[len("models--"):]
            # "models--org--model" → take everything after second "--"
            idx = after.find("--")
            if idx >= 0:
                name = after[idx + 2:]

        # Split on common separators
        parts = re.split(r'[/\-_]', name)

        result = []
        for part in parts:
            if not part:
                continue
            # Short all-lowercase abbreviations (xl, bp, etc.) → uppercase all
            if len(part) <= 3 and part.isalpha() and part.islower():
                result.append(part.upper())
            # All lowercase letters followed by digits: "gpt2", "llama3"
            elif re.match(r'^[a-z]+\d+$', part):
                match = re.match(r'^([a-z]+)(\d+)$', part)
                if match:
                    result.append(match.group(1).upper())
                    result.append(match.group(2))
                else:
                    result.append(part.upper())
            # Number with size suffix: "0.5B", "3B", "8B"
            elif re.match(r'^\d+\.?\d*[a-zA-Z]$', part):
                result.append(part)
            else:
                # Normal mixed case: split at letter→digit and digit→letter boundaries
                # But keep short digit-letter patterns together (e.g., "4e1t")
                sub = re.sub(r'([a-zA-Z]{2,})(\d)', r'\1 \2', part)
                sub = re.sub(r'(\d)([a-zA-Z]{2,})', r'\1 \2', sub)
                # Capitalize first letter of each sub-word
                words = sub.split()
                for w in words:
                    if re.match(r'^[\d.]+$', w):
                        result.append(w)
                    else:
                        result.append(w[0].upper() + w[1:])
        return " ".join(result)

    async def list_models(self):
        """List available/loaded models with plain-language descriptions."""
        ctrl = get_models_controller()

        # Get current model info
        current = ctrl.get_current_model()

        models = []

        # Add currently loaded model
        if current:
            params = int(current.get("parameters", 0) or 0)
            vocab = int(current.get("vocab_size", 0) or 0)
            models.append(ModelInfo(
                model_id=current["model_id"],
                status=ModelStatus.LOADED,
                device=current["device"],
                parameters=params,
                vocab_size=vocab,
                loaded_at=current.get("loaded_at"),
                description=self._describe_model(current["model_id"], params, loaded=True),
            ))

        # Add available HuggingFace models (skip if already listed as loaded)
        loaded_ids = {m.model_id for m in models}
        hf_models = ctrl.list_hf_models()
        for entry in hf_models:
            model_id = entry["model_id"]
            if model_id not in loaded_ids:
                params = int(entry.get("parameters", 0) or 0)
                vocab = int(entry.get("vocab_size", 0) or 0)
                models.append(ModelInfo(
                    model_id=model_id,
                    status=ModelStatus.AVAILABLE,
                    device="cpu",
                    parameters=params,
                    vocab_size=vocab,
                    loaded_at=None,
                    description=self._describe_model(model_id, params, loaded=False),
                ))

        return success_response(data=[m.model_dump() for m in models])

    async def load_model(
        self,
        req: LoadModelRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ):
        """Load a model"""
        ctrl = get_models_controller()
        result = ctrl.load_model(req.model_id, req.device.value, req.quantize)
        try:
            from domains.infrastructure.server_state import get_server_state
            ss = get_server_state()
            if result.get("status") == "loaded":
                resolved_device = result.get("device") or req.device.value
                ss.record_model_event("load", req.model_id, f"device={resolved_device}")
            else:
                ss.record_model_event("error", req.model_id, result.get("error", "unknown"))
        except Exception as e:
            logger.debug("Failed to record model load event: %s", e)
        try:
            from infrastructure.auth import get_audit_logger, audit_user
            get_audit_logger().log(
                "model.load",
                user=audit_user(auth_user),
                resource=req.model_id,
                detail=result.get("status", ""),
                extra={"device": req.device.value, "quantize": req.quantize},
            )
        except Exception:
            pass
        return wrap_controller_result(result)

    async def unload_model(self, auth_user: dict = Depends(require_auth_if_enabled)):
        """Unload current model"""
        ctrl = get_models_controller()
        model_id = ctrl._current_model
        if not model_id:
            try:
                from domains.infrastructure.model_registry import get_model_registry
                model_id = get_model_registry().default_id
            except Exception:
                pass
        result = ctrl.unload_model()
        try:
            from domains.infrastructure.server_state import get_server_state
            ss = get_server_state()
            ss.record_model_event("unload", model_id or "unknown")
        except Exception as e:
            logger.debug("Failed to record model unload event: %s", e)
        try:
            from infrastructure.auth import get_audit_logger, audit_user
            get_audit_logger().log(
                "model.unload",
                user=audit_user(auth_user),
                resource=model_id or "unknown",
                detail=result.get("status", ""),
            )
        except Exception:
            pass
        return wrap_controller_result(result)

    async def current_model(self):
        """Get current model info"""
        ctrl = get_models_controller()
        model = ctrl.get_current_model()
        if not model:
            raise HTTPException(status_code=404, detail="No model loaded")
        return success_response(data=model)

    async def list_hf_models(self, q: Optional[str] = None):
        """List HuggingFace available models with actual sizes and cache status.

        Models come from two sources:
        1. HuggingFace Hub API (top 50 text-generation by downloads, or curated fallback)
        2. Local HF cache — any model that has safetensors files on disk

        Sizes are computed in parallel to avoid sequential blocking on Hub API calls.
        """
        ctrl = get_models_controller()
        model_ids = ctrl.list_hf_models(q)

        def _is_cached(model_id: str) -> bool:
            try:
                return is_model_cached(model_id)
            except Exception as exc:
                logger.error("is_model_cached(%s) failed: %s", model_id, exc, extra={"tag": "MODEL"})
                return False

        def _cache_model_id(cache_dir_name: str) -> Optional[str]:
            """Convert a HF cache directory name like 'models--Qwen--Qwen2.5-0.5B-Instruct'
            back to a model ID like 'Qwen/Qwen2.5-0.5B-Instruct'."""
            if not cache_dir_name.startswith("models--"):
                return None
            return cache_dir_name[len("models--"):].replace("--", "/")

        models_out = []
        seen_ids = set()

        # Collect all model IDs to process (hub list + local cache extras)
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
                pass

        # Compute sizes in parallel (each call may hit HF Hub API)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        size_results: dict[str, Optional[float]] = {}
        cached_results: dict[str, bool] = {}

        def _compute_one(mid: str):
            return mid, compute_model_size_gb(mid), _is_cached(mid)

        from domains.infrastructure.resource_manager import get_resource_manager
        rm = get_resource_manager()
        max_workers = min(max(len(all_model_ids), rm.inference_pool_size * 2), 16)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_compute_one, mid): mid for mid in all_model_ids}
            for future in as_completed(futures):
                try:
                    mid, size_gb, cached = future.result()
                    size_results[mid] = size_gb
                    cached_results[mid] = cached
                except Exception:
                    mid = futures[future]
                    size_results[mid] = None
                    cached_results[mid] = False

        # Build output in original order
        for mid in all_model_ids:
            size_gb = size_results.get(mid)
            models_out.append({
                "id": mid,
                "name": self._model_display_name(mid),
                "hf_model_id": mid,
                "source": "huggingface",
                "size_mb": size_gb * 1024 if size_gb is not None else None,
                "size_gb": size_gb,
                "cached": cached_results.get(mid, False),
            })

        return success_response(data=models_out, meta={"q": q})

    async def get_model_logs(self, limit: int = 50, model_filter: Optional[str] = None):
        """Get model request logs (for debugging/monitoring)."""
        try:
            from state import model_request_logger as _logger
            if _logger:
                return success_response(data=_logger.get_logs(limit=limit, model=model_filter), meta=_logger.get_stats())
            return success_response(data=[], meta={})
        except ImportError:
            return success_response(data=[], meta={})

    async def export_model(self, request: ExportRequest):
        """Export current model to file."""
        import state as server_state
        import time
        if server_state.model is None:
            return error_response("No model loaded", code="E_NOT_FOUND")
        try:
            from domains.training.export import export_model as do_export, ExportConfig
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
            results = do_export(config, server_state.model, server_state.tokenizer)
            return success_response(data={"format": request.format, "files": results}, message="exported")
        except Exception as e:
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="export_model")
            return error_response(err.user_message, code=err.code)

    async def get_export_formats(self):
        """Get list of supported export formats."""
        from domains.training.export import list_export_formats
        return success_response(data=list_export_formats())

    async def start_download(self, req: DownloadRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> Dict[str, Any]:
        """
        Start downloading a model from HuggingFace Hub with progress tracking.

        Returns immediately with download status. Poll `/models/download/{model_id}`
        for progress updates.
        """
        from domains.infrastructure.download_manager import get_download_manager

        mgr = get_download_manager()

        if mgr.is_cached(req.model_id):
            return success_response(data={"model_id": req.model_id}, message="already_cached")

        if mgr.is_downloading(req.model_id):
            return success_response(data={"model_id": req.model_id}, message="already_downloading")

        asyncio.create_task(self._run_download(req.model_id, req.total_bytes_hint))
        try:
            from infrastructure.auth import get_audit_logger, audit_user
            get_audit_logger().log(
                "model.download",
                user=audit_user(auth_user),
                resource=req.model_id,
                detail="started",
                extra={"total_bytes_hint": req.total_bytes_hint},
            )
        except Exception:
            pass
        return success_response(data={"model_id": req.model_id}, message="started")

    async def _run_download(self, model_id: str, total_bytes_hint: int):
        """Background task that runs the actual download."""
        from domains.infrastructure.download_manager import get_download_manager
        from controllers.models import get_models_controller

        mgr = get_download_manager()
        result = await mgr.download(model_id, total_bytes_hint)

        if result.get("status") == "complete":
            ctrl = get_models_controller()
            try:
                ctrl.load_model(model_id)
            except Exception as e:
                logger.warning("Auto-load after download failed for %s: %s", model_id, e, extra={"tag": "MODEL"})

    async def get_download_status(self, model_id: str) -> Dict[str, Any]:
        """Get download progress for a specific model."""
        from domains.infrastructure.download_manager import get_download_manager

        mgr = get_download_manager()
        progress = mgr.get_progress(model_id)
        if progress is None:
            cached = mgr.is_cached(model_id)
            return success_response(data={"model_id": model_id, "cached": cached}, message="not_found")
        return success_response(data=progress)

    async def list_downloads(self) -> Dict[str, Any]:
        """List all active and recent downloads."""
        from domains.infrastructure.download_manager import get_download_manager

        mgr = get_download_manager()
        mgr.cleanup_stale()
        return success_response(data=mgr.list_downloads())

    async def cancel_download(self, model_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> Dict[str, Any]:
        """Cancel an in-progress download."""
        from domains.infrastructure.download_manager import get_download_manager

        mgr = get_download_manager()
        if mgr.cancel(model_id):
            try:
                from infrastructure.auth import get_audit_logger, audit_user
                get_audit_logger().log(
                    "model.cancel",
                    user=audit_user(auth_user),
                    resource=model_id,
                    detail="cancelled",
                )
            except Exception:
                pass
            return success_response(data={"model_id": model_id}, message="cancelled")
        return success_response(data={"model_id": model_id}, message="not_found")

    async def verify_download(self, model_id: str) -> Dict[str, Any]:
        """Verify a downloaded model's weight files against Hub SHA-256 checksums.
        Returns verification result and on-disk size."""
        from domains.infrastructure.hf_hub import (
            get_cache_dir,
            verify_model,
            list_missing_files,
        )

        try:
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
        except Exception as e:
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="verify_download")
            return error_response(err.user_message, code=err.code, details={"model_id": model_id})

    async def retry_download(self, model_id: str) -> Dict[str, Any]:
        """Redownload a cached model (cleanup + fresh download)."""
        from domains.infrastructure.download_manager import (
            cleanup_incomplete,
            get_download_manager,
            is_download_complete,
        )

        if is_download_complete(model_id):
            cleanup_incomplete(model_id)

        mgr = get_download_manager()
        if mgr.is_downloading(model_id):
            return success_response(data={"model_id": model_id}, message="already_downloading")

        asyncio.create_task(self._run_download(model_id, 0))
        return success_response(data={"model_id": model_id}, message="started")

    async def cache_usage(self) -> Dict[str, Any]:
        """Total disk usage of the HuggingFace model cache (fast — walks blobs/ only)."""
        cache = _hf_cache_dir
        if not cache.exists():
            return success_response(data={"total_bytes": 0, "total_gb": 0, "model_count": 0, "cache_dir": str(cache)})
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
                                pass
                count += 1
        return success_response(data={
            "total_bytes": total,
            "total_gb": round(total / (1024**3), 2),
            "model_count": count,
            "cache_dir": str(cache),
        })

    async def download_qwen_gguf(self):
        """Download Qwen2.5-0.5B-Instruct GGUF (Q4_K_M) from HuggingFace Hub.

        Returns the GGUF file as a streaming download.  The mobile app calls this
        to get the model for llama.rn inference.
        """
        from downcraft.downloader import download_file

        repo_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
        filename = "qwen2.5-0.5b-instruct-q4_k_m.gguf"

        cache_dir = Path.home() / ".cache" / "sloughgpt" / "gguf"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_path = cache_dir / filename

        if cached_path.exists():
            logger.info("Serving cached GGUF: %s", cached_path, extra={"tag": "MODEL"})
            return FileResponse(str(cached_path), media_type="application/octet-stream", filename=filename)

        logger.info("Downloading Qwen GGUF from HuggingFace Hub (this may take a while)...", extra={"tag": "MODEL"})
        try:
            url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
            download_file(url=url, dest=cached_path)
            logger.info("GGUF downloaded and cached: %s", cached_path, extra={"tag": "MODEL"})
            return FileResponse(str(cached_path), media_type="application/octet-stream", filename=filename)
        except Exception as e:
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="download_gguf")
            raise HTTPException(status_code=err.http_status, detail=err.user_message)

    async def visual_model_load(self, model_dir: str = "", model_id: str = ""):
        """Load a vision / multimodal model from a local directory.

        ``model_dir`` — path to the model directory on disk.
        ``model_id``  — HuggingFace model identifier (used when `model_dir` is empty).
        """
        logger.info("Visual model load requested: dir=%s, id=%s", model_dir, model_id, extra={"tag": "MODEL"})
        ctrl = get_models_controller()
        if model_dir:
            result = ctrl.load_model_path(model_dir)
        elif model_id:
            result = ctrl.load_model(model_id)
        else:
            raise HTTPException(status_code=400, detail="Either model_dir or model_id required")
        return wrap_controller_result(result)

    async def quantize_model(self, req: QuantizeRequest, auth_user: dict = Depends(require_auth_if_enabled)):
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
        from domains.infrastructure.quantization import Quantine
        from domains.infrastructure.quant_core.wrapper import HAS_AVX2
        import numpy as np

        bits = req.bits
        mode = req.mode

        if bits not in (4, 8):
            raise HTTPException(status_code=400, detail=f"bits must be 4 or 8, got {bits}")
        if mode not in ("symmetric", "asymmetric"):
            raise HTTPException(status_code=400, detail=f"mode must be symmetric or asymmetric, got {mode}")

        # Find the active provider (try SloNet first, then HuggingFace)
        from domains.models.provider import get_provider

        provider = get_provider("slonet")
        model_type = "slonet"

        if provider is None:
            provider = get_provider("hf-default")
            model_type = "huggingface"

        if provider is None:
            raise HTTPException(status_code=400, detail="No model loaded")

        model = getattr(provider, "_model", None)
        if model is None:
            raise HTTPException(status_code=400, detail="Provider has no model")

        # Walk linear layers using the appropriate walker
        if model_type == "slonet":
            from domains.infrastructure.quantization import walk_slo_linears
            layers = walk_slo_linears(model)
        else:
            from domains.infrastructure.quantization import walk_hf_linears
            layers = walk_hf_linears(model)

        engine = Quantine(bits=bits, mode=mode)
        quantized_count = 0
        tensor_infos = {}
        for name, module in layers.items():
            weight = module.weight.data
            # Convert torch tensor to numpy if needed
            if hasattr(weight, 'cpu'):
                weight = weight.cpu().numpy().astype(np.float32).copy()
            else:
                weight = np.asarray(weight, dtype=np.float32).copy()
            info = engine.quantize(f"{name}.weight", weight)
            if info.is_quantized:
                if model_type == "slonet":
                    module.set_quantized_weight(info)
                else:
                    # For HuggingFace models: monkey-patch forward with quantized path
                    from domains.infrastructure.quantization import QuantizedLinear
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
            from domains.infrastructure.quant_core.wrapper import HAS_AVX2
            report["avx2_enabled"] = bool(HAS_AVX2)
        except Exception:
            report["avx2_enabled"] = False

        try:
            from infrastructure.auth import get_audit_logger, audit_user
            get_audit_logger().log(
                "model.quantize",
                user=audit_user(auth_user),
                resource=self._audit_model_id(provider),
                detail=f"bits={bits} mode={mode}",
                extra={"bits": bits, "mode": mode, "layers_quantized": quantized_count, "model_type": model_type},
            )
        except Exception:
            pass

        return success_response(data=report)

    async def dequantize_model(self, auth_user: dict = Depends(require_auth_if_enabled)):
        """Reset quantized model back to float32 weights.

        Clears quantization state from all linear layers. The model
        returns to its original float32 precision.

        Returns:
            Status report with number of layers reset.
        """
        from domains.models.provider import get_provider

        provider = get_provider("slonet")
        model_type = "slonet"

        if provider is None:
            provider = get_provider("hf-default")
            model_type = "huggingface"

        if provider is None:
            raise HTTPException(status_code=400, detail="No model loaded")

        model = getattr(provider, "_model", None)
        if model is None:
            raise HTTPException(status_code=400, detail="Provider has no model")

        # Clear quantization state
        if model_type == "slonet":
            from domains.infrastructure.quantization import walk_slo_linears
            layers = walk_slo_linears(model)
            for name, module in layers.items():
                module._quant_info = None
        else:
            from domains.infrastructure.quantization import walk_hf_linears
            layers = walk_hf_linears(model)
            for name, module in layers.items():
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

        try:
            from infrastructure.auth import get_audit_logger, audit_user
            get_audit_logger().log(
                "model.dequantize",
                user=audit_user(auth_user),
                resource=self._audit_model_id(provider),
                detail=f"model_type={model_type}",
                extra={"layers_reset": len(layers)},
            )
        except Exception:
            pass

        return success_response(data={
            "dequantized": True,
            "model_type": model_type,
            "layers_reset": len(layers),
        })

    async def set_precision(self, req: PrecisionRequest, auth_user: dict = Depends(require_auth_if_enabled)):
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
        from domains.slolib.gpu import get_accelerator, set_accelerator_precision
        import numpy as np

        acc = get_accelerator()
        acc_mode = req.mode

        result = {
            "accelerator": acc.name,
            "device_type": acc.device_type,
        }

        if acc.name == "cpu":
            # CPU path: use Quantine to select best format
            from domains.infrastructure.quantization import Quantine
            suggestion = Quantine.suggest_format()
            result["precision"] = suggestion["format"]
            result["bits"] = suggestion["bits"]
            result["reason"] = suggestion["reason"]
            result["benchmark"] = suggestion["benchmark"]
            result["fp16_mode"] = False

            # If int8/int4 selected, apply quantization
            if suggestion["format"] in ("int8", "int4"):
                from domains.models.provider import get_provider
                provider = get_provider("slonet") or get_provider("hf-default")
                if provider is not None:
                    model = getattr(provider, "_model", None)
                    if model is not None:
                        # Re-use existing quantize logic
                        from domains.infrastructure.quantization import Quantine, walk_slo_linears, walk_hf_linears
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

        try:
            from infrastructure.auth import get_audit_logger, audit_user
            get_audit_logger().log(
                "model.precision",
                user=audit_user(auth_user),
                resource=acc.name,
                detail=str(result.get("precision", "")),
                extra={"mode": acc_mode},
            )
        except Exception:
            pass

        return wrap_controller_result(result)

    async def get_catalog(self):
        """Get the persistent model catalog."""
        from domains.infrastructure.model_catalog import get_model_catalog
        catalog = get_model_catalog()
        return success_response(data=catalog.list_all())

    async def get_catalog_stats(self):
        """Get catalog statistics."""
        from domains.infrastructure.model_catalog import get_model_catalog
        catalog = get_model_catalog()
        return success_response(data=catalog.stats())

    async def get_conversion_status(self, model_id: Optional[str] = None):
        """Get model conversion/download status.

        Without model_id: returns all active conversions.
        With model_id: returns status for that specific model.
        """
        from domains.infrastructure.conversion_tracker import get_tracker
        tracker = get_tracker()

        if model_id:
            status = tracker.get(model_id)
            if not status:
                return success_response(data={"model_id": model_id, "stage": "idle", "progress": 0})
            return success_response(data=status)

        return success_response(data=tracker.get_active())

    async def get_process_guard(self):
        """Get ProcessGuard status.

        Returns enabled state, whether a guard is actively running, the
        guarded model id, and the guard health snapshot.
        """
        ctrl = get_models_controller()
        return success_response(data=ctrl.get_process_guard_status())

    async def set_process_guard(self, request: Dict[str, Any]):
        """Enable or disable ProcessGuard at runtime.

        Body: ``{"enabled": true}`` or ``{"enabled": false}``

        Disabling stops any active guard. Enabling starts the guard for the
        currently loaded model if a .slnc file exists.
        """
        enabled = request.get("enabled")
        if not isinstance(enabled, bool):
            raise HTTPException(status_code=422, detail="`enabled` must be a boolean")
        ctrl = get_models_controller()
        return success_response(data=ctrl.set_process_guard_enabled(enabled))


router = ModelsRouter().router
