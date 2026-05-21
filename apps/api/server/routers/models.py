"""
Models Router - MVC View layer
Uses ModelsController for business logic
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path

from schemas.models import ModelInfo, LoadModelRequest, LoadModelResponse, ModelStatus
from controllers.models import get_models_controller

router = APIRouter(prefix="/models", tags=["models"])

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def _get_hf_model_size_gb(model_id: str) -> Optional[float]:
    """Get actual model size from HuggingFace cache, or estimate from params."""
    # Check HF cache for downloaded models
    cache_dir = HF_CACHE / f"models--{model_id.replace('/', '--')}"
    if cache_dir.exists():
        total_bytes = 0
        for f in cache_dir.rglob("*"):
            if f.is_file() and not f.name.startswith("."):
                total_bytes += f.stat().st_size
        if total_bytes > 0:
            return round(total_bytes / (1024 ** 3), 2)

    # Estimate from known parameter counts (better than blind guess)
    param_estimates = {
        "gpt2": 124_000_000,
        "gpt2-medium": 355_000_000,
        "gpt2-large": 774_000_000,
        "gpt2-xl": 1_500_000_000,
        "distilgpt2": 82_000_000,
        "EleutherAI/gpt-neo-125M": 125_000_000,
        "microsoft/phi-2": 2_700_000_000,
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 1_100_000_000,
        "microsoft/Phi-3-mini-128k-instruct": 3_800_000_000,
        "Qwen/Qwen2-0.5B-Instruct": 500_000_000,
    }
    params = param_estimates.get(model_id)
    if params is None:
        return None

    # fp32: 4 bytes per param, plus ~10% overhead
    estimated_gb = (params * 4 * 1.1) / (1024 ** 3)
    return round(estimated_gb, 2)


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
    """List HuggingFace available models with actual sizes"""
    ctrl = get_models_controller()
    model_ids = ctrl.list_hf_models(q)

    return {
        "models": [
            {
                "id": m,
                "name": m,
                "hf_model_id": m,
                "source": "huggingface",
                "size_mb": _get_hf_model_size_gb(m) * 1024 if _get_hf_model_size_gb(m) else None,
                "size_gb": _get_hf_model_size_gb(m),
            }
            for m in model_ids
        ],
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