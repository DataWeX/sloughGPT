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
from schemas.common import StandardResponse, success_response, error_response
from controllers.models import get_models_controller
from infrastructure.auth import require_auth_if_enabled

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])

from domains.infrastructure.model_size import compute_model_size_gb, format_size_gb, is_model_cached

_hf_cache_dir = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))) / "hub"


@router.get("", response_model=StandardResponse[List[ModelInfo]])
async def list_models():
    """List available/loaded models with plain-language descriptions."""
    ctrl = get_models_controller()

    # Get current model info
    current = ctrl.get_current_model()

    models = []

    # Add currently loaded model
    if current:
        models.append(ModelInfo(
            model_id=current["model_id"],
            status=ModelStatus.LOADED,
            device=current["device"],
            parameters=current.get("parameters", 0),
            vocab_size=current.get("vocab_size", 0),
            loaded_at=current.get("loaded_at"),
            description=_describe_model(current["model_id"], current.get("parameters", 0), loaded=True),
        ))

    # Add available HuggingFace models (skip if already listed as loaded)
    loaded_ids = {m.model_id for m in models}
    hf_models = ctrl.list_hf_models()
    for entry in hf_models:
        model_id = entry["model_id"]
        if model_id not in loaded_ids:
            models.append(ModelInfo(
                model_id=model_id,
                status=ModelStatus.AVAILABLE,
                device="cpu",
                parameters=entry.get("parameters", 0),
                vocab_size=entry.get("vocab_size", 0),
                loaded_at=None,
                description=_describe_model(model_id, entry.get("parameters", 0), loaded=False),
            ))

    return success_response(data=[m.model_dump() for m in models])


def _describe_model(model_id: str, parameters: int, loaded: bool) -> str:
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


def _model_display_name(model_id: str) -> str:
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
    import re
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


@router.post("/load", response_model=LoadModelResponse)
async def load_model(
    req: LoadModelRequest,
    auth_user: dict = Depends(require_auth_if_enabled),
):
    """Load a model"""
    ctrl = get_models_controller()
    result = ctrl.load_model(req.model_id, req.device.value, req.quantize)
    try:
        from domains.infrastructure.server_state import get_server_state
        ss = get_server_state()
        if result.get("success"):
            ss.record_model_event("load", req.model_id, f"device={req.device.value}")
        else:
            ss.record_model_event("error", req.model_id, result.get("error", "unknown"))
    except Exception as e:
        logger.debug("Failed to record model load event: %s", e)
    return success_response(data=result)


@router.post("/unload")
async def unload_model(auth_user: dict = Depends(require_auth_if_enabled)):
    """Unload current model"""
    ctrl = get_models_controller()
    result = ctrl.unload_model()
    try:
        from domains.infrastructure.server_state import get_server_state
        ss = get_server_state()
        ss.record_model_event("unload", ctrl._current_model or "unknown")
    except Exception as e:
        logger.debug("Failed to record model unload event: %s", e)
    return success_response(data=result)


@router.get("/current")
async def current_model():
    """Get current model info"""
    ctrl = get_models_controller()
    model = ctrl.get_current_model()
    if not model:
        raise HTTPException(status_code=404, detail="No model loaded")
    return success_response(data=model)


@router.get("/hf")
async def list_hf_models(q: Optional[str] = None):
    """List HuggingFace available models with actual sizes and cache status.

    Models come from two sources:
    1. HuggingFace Hub API (top 50 text-generation by downloads, or curated fallback)
    2. Local HF cache — any model that has safetensors files on disk
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

    for m in model_ids:
        mid = m["model_id"] if isinstance(m, dict) else m
        seen_ids.add(mid)
        size_gb = compute_model_size_gb(mid)
        models_out.append({
            "id": mid,
            "name": _model_display_name(mid),
            "hf_model_id": mid,
            "source": "huggingface",
            "size_mb": size_gb * 1024 if size_gb is not None else None,
            "size_gb": size_gb,
            "cached": _is_cached(mid),
        })

    # Scan local HF cache for models not in the Hub list
    if not q and _hf_cache_dir.exists():
        try:
            for entry in _hf_cache_dir.iterdir():
                if not entry.name.startswith("models--") or not entry.is_dir():
                    continue
                cached_id = _cache_model_id(entry.name)
                if cached_id and cached_id not in seen_ids:
                    seen_ids.add(cached_id)
                    size_gb = compute_model_size_gb(cached_id)
                    models_out.append({
                        "id": cached_id,
                        "name": _model_display_name(cached_id),
                        "hf_model_id": cached_id,
                        "source": "huggingface",
                        "size_mb": size_gb * 1024 if size_gb is not None else None,
                        "size_gb": size_gb,
                        "cached": _is_cached(cached_id),
                    })
        except Exception:
            pass

    return success_response(data=models_out, meta={"q": q})


@router.get("/logs")
async def get_model_logs(limit: int = 50, model_filter: Optional[str] = None):
    """Get model request logs (for debugging/monitoring)."""
    try:
        from state import model_request_logger as _logger
        if _logger:
            return success_response(data=_logger.get_logs(limit=limit, model=model_filter), meta=_logger.get_stats())
        return success_response(data=[], meta={})
    except ImportError:
        return success_response(data=[], meta={})


class ExportRequest(BaseModel):
    output_path: str = "models/exported"
    format: str = "sou"
    include_tokenizer: bool = True


@router.post("/export", tags=["models"])
async def export_model(request: ExportRequest):
    """Export current model to file."""
    import state as server_state
    import time
    if server_state.model is None:
        return error_response(message="No model loaded")
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
        return error_response(message=str(e))


@router.get("/export/formats", tags=["models"])
async def get_export_formats():
    """Get list of supported export formats."""
    from domains.training.export import list_export_formats
    return success_response(data=list_export_formats())


class DownloadRequest(BaseModel):
    model_id: str
    total_bytes_hint: int = 0


@router.post("/download")
async def start_download(req: DownloadRequest) -> Dict[str, Any]:
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

    asyncio.create_task(_run_download(req.model_id, req.total_bytes_hint))
    return success_response(data={"model_id": req.model_id}, message="started")


async def _run_download(model_id: str, total_bytes_hint: int):
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


@router.get("/download/{model_id:path}")
async def get_download_status(model_id: str) -> Dict[str, Any]:
    """Get download progress for a specific model."""
    from domains.infrastructure.download_manager import get_download_manager

    mgr = get_download_manager()
    progress = mgr.get_progress(model_id)
    if progress is None:
        cached = mgr.is_cached(model_id)
        return success_response(data={"model_id": model_id, "cached": cached}, message="not_found")
    return success_response(data=progress)


@router.get("/downloads")
async def list_downloads() -> Dict[str, Any]:
    """List all active and recent downloads."""
    from domains.infrastructure.download_manager import get_download_manager

    mgr = get_download_manager()
    mgr.cleanup_stale()
    return success_response(data=mgr.list_downloads())


@router.post("/download/{model_id:path}/cancel")
async def cancel_download(model_id: str) -> Dict[str, Any]:
    """Cancel an in-progress download."""
    from domains.infrastructure.download_manager import get_download_manager

    mgr = get_download_manager()
    if mgr.cancel(model_id):
        return success_response(data={"model_id": model_id}, message="cancelled")
    return success_response(data={"model_id": model_id}, message="not_found")


@router.post("/download/{model_id:path}/verify")
async def verify_download(model_id: str) -> Dict[str, Any]:
    """Verify a downloaded model's weight files against Hub SHA-256 checksums.
    Returns verification result and on-disk size."""
    from downcraft import verify as sg_verify
    from downcraft.hf_hub import get_cache_dir

    try:
        cache_dir = get_cache_dir(model_id)
        refs_main = cache_dir / "refs" / "main"
        if not refs_main.exists():
            return {"status": "not_cached", "model_id": model_id}
        ok = sg_verify.verify_model(model_id)
        missing = sg_verify.list_missing_files(model_id)
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
        return {"status": "error", "model_id": model_id, "error": str(e)}


@router.post("/download/{model_id:path}/retry")
async def retry_download(model_id: str) -> Dict[str, Any]:
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

    asyncio.create_task(_run_download(model_id, 0))
    return success_response(data={"model_id": model_id}, message="started")


@router.get("/cache-usage")
async def cache_usage() -> Dict[str, Any]:
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


@router.get("/download/qwen-gguf")
async def download_qwen_gguf():
    """Download Qwen2.5-0.5B-Instruct GGUF (Q4_K_M) from HuggingFace Hub.

    Returns the GGUF file as a streaming download.  The mobile app calls this
    to get the model for llama.rn inference.
    """
    from huggingface_hub import hf_hub_download
    import shutil

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
        downloaded = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=str(cache_dir))
        shutil.copy2(downloaded, cached_path)
        logger.info("GGUF downloaded and cached: %s", cached_path, extra={"tag": "MODEL"})
        return FileResponse(str(cached_path), media_type="application/octet-stream", filename=filename)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to download GGUF model: {e}")


@router.post("/visual-load")
async def visual_model_load(model_dir: str = "", model_id: str = ""):
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
    return success_response(data=result, message=result.get("status", "ok"))


class QuantizeRequest(BaseModel):
    """Request body for POST /models/quantize."""
    bits: int = 8
    mode: str = "symmetric"


@router.post("/quantize")
async def quantize_model(req: QuantizeRequest):
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
    from domains.infrastructure.quantization import QuantEngine
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

    engine = QuantEngine(bits=bits, mode=mode)
    quantized_count = 0
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
            quantized_count += 1

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

    return success_response(data=report)


@router.post("/dequantize")
async def dequantize_model(auth_user: dict = Depends(require_auth_if_enabled)):
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

    return success_response(data={
        "dequantized": True,
        "model_type": model_type,
        "layers_reset": len(layers),
    })


@router.get("/catalog")
async def get_catalog():
    """Get the persistent model catalog."""
    from domains.infrastructure.model_catalog import get_model_catalog
    catalog = get_model_catalog()
    return success_response(data=catalog.list_all())


@router.get("/catalog/stats")
async def get_catalog_stats():
    """Get catalog statistics."""
    from domains.infrastructure.model_catalog import get_model_catalog
    catalog = get_model_catalog()
    return success_response(data=catalog.stats())


@router.get("/conversion-status")
async def get_conversion_status(model_id: Optional[str] = None):
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
