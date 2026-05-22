"""
Models Router - MVC View layer
Uses ModelsController for business logic
"""
import asyncio
import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path

from schemas.models import ModelInfo, LoadModelRequest, LoadModelResponse, ModelStatus
from controllers.models import get_models_controller

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/models", tags=["models"])

_hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
HF_CACHE = _hf_home / "hub"


def _get_hf_model_size_gb(model_id: str) -> Optional[float]:
    """Get actual model size from HuggingFace Hub API or local cache.

    Priority:
    1. HuggingFace Hub API — safetensors total bytes (always accurate)
    2. Local HF cache — only if download is complete (verified by safetensors presence)
    3. Known models dict — verified sizes for common models
    """
    # 1. Always try Hub API first — gives the true intended size
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(model_id)
        if info and info.safetensors and info.safetensors.total:
            return round(info.safetensors.total / (1024 ** 3), 2)
    except Exception:
        pass

    # 2. Check HF cache — but only if download looks complete
    cache_dir = HF_CACHE / f"models--{model_id.replace('/', '--')}"
    if cache_dir.exists():
        # Check for actual safetensors files (not just metadata/partial downloads)
        safetensors_files = list(cache_dir.rglob("*.safetensors"))
        if safetensors_files:
            total_bytes = sum(f.stat().st_size for f in safetensors_files)
            if total_bytes > 0:
                return round(total_bytes / (1024 ** 3), 2)

    # 3. Known verified sizes (safetensors on disk)
    known_sizes_gb = {
        "gpt2": 0.52,
        "gpt2-medium": 1.48,
        "gpt2-large": 3.15,
        "gpt2-xl": 6.18,
        "distilgpt2": 0.34,
        "EleutherAI/gpt-neo-125M": 0.52,
        "EleutherAI/gpt-neo-1.3B": 5.4,
        "EleutherAI/gpt-j-6B": 24.6,
        "microsoft/phi-2": 5.4,
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 4.4,
        "microsoft/Phi-3-mini-128k-instruct": 7.6,
        "Qwen/Qwen2-0.5B-Instruct": 1.2,
        "Qwen/Qwen2.5-0.5B-Instruct": 1.2,
        "Qwen/Qwen2.5-1.5B-Instruct": 3.4,
        "Qwen/Qwen2.5-3B-Instruct": 6.8,
        "Qwen/Qwen2.5-7B-Instruct": 15.2,
        "meta-llama/Llama-3.2-1B-Instruct": 2.5,
        "meta-llama/Llama-3.2-3B-Instruct": 6.8,
        "meta-llama/Llama-3.1-8B-Instruct": 16.1,
        "google/gemma-2b": 4.8,
        "google/gemma-7b": 16.8,
    }
    return known_sizes_gb.get(model_id)


@router.get("", response_model=List[ModelInfo])
async def list_models():
    """List available/loaded models"""
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
            parameters=124439808,
            vocab_size=50257,
            loaded_at=current.get("loaded_at"),
        ))
    
    # Add available HuggingFace models (skip if already listed as loaded)
    loaded_ids = {m.model_id for m in models}
    hf_models = ctrl.list_hf_models()
    for model_id in hf_models:
        if model_id not in loaded_ids:
            models.append(ModelInfo(
                model_id=model_id,
                status=ModelStatus.AVAILABLE,
                device="cpu",
                parameters=124439808,
                vocab_size=50257,
                loaded_at=None,
            ))
    
    return models


@router.post("/load", response_model=LoadModelResponse)
async def load_model(req: LoadModelRequest):
    """Load a model"""
    ctrl = get_models_controller()
    result = ctrl.load_model(req.model_id, req.device.value, req.quantize)
    return LoadModelResponse(**result)


@router.post("/unload")
async def unload_model():
    """Unload current model"""
    ctrl = get_models_controller()
    return ctrl.unload_model()


@router.get("/current")
async def current_model():
    """Get current model info"""
    ctrl = get_models_controller()
    model = ctrl.get_current_model()
    if not model:
        raise HTTPException(status_code=404, detail="No model loaded")
    return model


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
        cache_dir = HF_CACHE / f"models--{model_id.replace('/', '--')}"
        return cache_dir.exists()

    def _cache_model_id(cache_dir_name: str) -> Optional[str]:
        """Convert a HF cache directory name like 'models--Qwen--Qwen2.5-0.5B-Instruct'
        back to a model ID like 'Qwen/Qwen2.5-0.5B-Instruct'."""
        if not cache_dir_name.startswith("models--"):
            return None
        return cache_dir_name[len("models--"):].replace("--", "/")

    models_out = []
    seen_ids = set()

    for m in model_ids:
        seen_ids.add(m)
        size_gb = _get_hf_model_size_gb(m)
        models_out.append({
            "id": m,
            "name": m,
            "hf_model_id": m,
            "source": "huggingface",
            "size_mb": size_gb * 1024 if size_gb is not None else None,
            "size_gb": size_gb,
            "cached": _is_cached(m),
        })

    # Scan local HF cache for models not in the Hub list
    if not q and HF_CACHE.exists():
        try:
            for entry in HF_CACHE.iterdir():
                if not entry.name.startswith("models--") or not entry.is_dir():
                    continue
                cached_id = _cache_model_id(entry.name)
                if cached_id and cached_id not in seen_ids:
                    seen_ids.add(cached_id)
                    size_gb = _get_hf_model_size_gb(cached_id)
                    models_out.append({
                        "id": cached_id,
                        "name": cached_id,
                        "hf_model_id": cached_id,
                        "source": "huggingface",
                        "size_mb": size_gb * 1024 if size_gb is not None else None,
                        "size_gb": size_gb,
                        "cached": True,
                    })
        except Exception:
            pass

    return {
        "models": models_out,
        "q": q,
    }


@router.get("/logs")
async def get_model_logs(limit: int = 50, model_filter: Optional[str] = None):
    """Get model request logs (for debugging/monitoring)."""
    try:
        from state import model_request_logger as _logger
        if _logger:
            return {
                "logs": _logger.get_logs(limit=limit, model=model_filter),
                "stats": _logger.get_stats(),
            }
        return {"logs": [], "stats": {}}
    except ImportError:
        return {"logs": [], "stats": {}}


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
        return {"error": "No model loaded"}
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
        return {"status": "exported", "format": request.format, "files": results}
    except Exception as e:
        return {"error": str(e)}


@router.get("/export/formats", tags=["models"])
async def get_export_formats():
    """Get list of supported export formats."""
    from domains.training.export import list_export_formats
    return {"formats": list_export_formats()}


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
        return {"status": "already_cached", "model_id": req.model_id}

    if mgr.is_downloading(req.model_id):
        return {"status": "already_downloading", "model_id": req.model_id}

    asyncio.create_task(_run_download(req.model_id, req.total_bytes_hint))
    return {"status": "started", "model_id": req.model_id}


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
            logger.warning("Auto-load after download failed for %s: %s", model_id, e)


@router.get("/download/{model_id:path}")
async def get_download_status(model_id: str) -> Dict[str, Any]:
    """Get download progress for a specific model."""
    from domains.infrastructure.download_manager import get_download_manager

    mgr = get_download_manager()
    progress = mgr.get_progress(model_id)
    if progress is None:
        cached = mgr.is_cached(model_id)
        return {"model_id": model_id, "status": "not_found", "cached": cached}
    return progress


@router.get("/downloads")
async def list_downloads() -> Dict[str, Any]:
    """List all active and recent downloads."""
    from domains.infrastructure.download_manager import get_download_manager

    mgr = get_download_manager()
    mgr.cleanup_stale()
    return {"downloads": mgr.list_downloads()}


@router.post("/download/{model_id:path}/cancel")
async def cancel_download(model_id: str) -> Dict[str, Any]:
    """Cancel an in-progress download."""
    from domains.infrastructure.download_manager import get_download_manager

    mgr = get_download_manager()
    if mgr.cancel(model_id):
        return {"status": "cancelled", "model_id": model_id}
    return {"status": "not_found", "model_id": model_id}