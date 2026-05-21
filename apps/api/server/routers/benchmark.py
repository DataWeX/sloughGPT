"""
Benchmark Router - Model benchmarking endpoints

Includes quality evaluation:
- Coherence scoring
- Repetition detection
- Real model metrics
"""
from fastapi import APIRouter, HTTPException
from pathlib import Path
from typing import Optional, List, Dict, Any

router = APIRouter(prefix="/benchmark", tags=["benchmark"])


def _get_model_metrics(model: str) -> Dict[str, Any]:
    """Get real model metrics from the model controller."""
    try:
        from controllers.models import get_models_controller
        
        ctrl = get_models_controller()
        
        if not ctrl._hf_model:
            return {"model": model, "model_loaded": False}
        
        inference_time = time.time() - ctrl._last_inference_time if ctrl._last_inference_time else 0
        total_tokens = ctrl._total_tokens_generated
        total_inferences = ctrl._inference_count
        
        # Calculate throughput
        tokens_per_sec = 0
        if inference_time > 0 and total_tokens > 0:
            tokens_per_sec = total_tokens / (inference_time * total_inferences) if total_inferences > 0 else 0
        
        # Memory estimate
        memory_mb = 0
        if ctrl._hf_model:
            try:
                import torch
                memory_mb = torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
            except Exception:
                memory_mb = 500  # Rough estimate
        
        return {
            "model": model,
            "model_loaded": True,
            "inference_count": total_inferences,
            "total_tokens": total_tokens,
            "tokens_per_second": round(tokens_per_sec, 1),
            "memory_mb": round(memory_mb, 1),
        }
    except Exception as e:
        return {"model": model, "error": str(e)}


import time


@router.post("/run")
async def run_benchmark(model: str = "gpt2"):
    """Run model benchmark - returns real metrics"""
    return _get_model_metrics(model)


@router.get("/metrics")
async def get_model_metrics(model: str = "gpt2"):
    """Get real-time model metrics"""
    return _get_model_metrics(model)


@router.post("/perplexity")
async def calculate_perplexity(text: str = "Sample text for evaluation"):
    """Calculate perplexity on text"""
    try:
        import torch
        from controllers.models import get_models_controller
        
        ctrl = get_models_controller()
        if not ctrl._tokenizer or not ctrl._hf_model:
            raise HTTPException(status_code=400, detail="Model not loaded")
        
        # Tokenize
        inputs = ctrl._tokenizer(text, return_tensors="pt")
        if ctrl._hf_model.device.type == "cuda":
            inputs = {k: v.to(ctrl._hf_model.device) for k, v in inputs.items()}
        
        # Get loss
        with torch.no_grad():
            outputs = ctrl._hf_model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            perplexity = torch.exp(torch.tensor(loss)).item()
        
        return {
            "text": text[:30],
            "perplexity": round(perplexity, 2),
            "loss": round(loss, 4),
            "tokens": inputs["input_ids"].shape[1],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/compare")
async def compare_benchmarks():
    """Compare all available models with real/estimated metrics"""
    from controllers.models import get_models_controller

    ctrl = get_models_controller()
    models = ctrl.list_hf_models()

    # Real stats for the currently loaded model
    current_id = ctrl._current_model
    inference_time = time.time() - ctrl._last_inference_time if ctrl._last_inference_time else 1
    total_tokens = ctrl._total_tokens_generated
    total_inferences = max(ctrl._inference_count, 1)
    loaded_tps = round(total_tokens / (inference_time * total_inferences), 1) if inference_time > 0 and total_tokens > 0 else 0

    # Predefined model metadata — params, size, family
    MODEL_META = {
        "gpt2":                 {"params": "124M", "family": "gpt"},
        "distilgpt2":           {"params": "82M",  "family": "gpt"},
        "gpt2-medium":          {"params": "355M", "family": "gpt"},
        "gpt2-large":           {"params": "774M", "family": "gpt"},
        "EleutherAI/gpt-neo-125M": {"params": "125M", "family": "gpt"},
        "microsoft/phi-2":      {"params": "2.7B", "family": "phi"},
        "microsoft/Phi-3-mini-128k-instruct": {"params": "3.8B", "family": "phi"},
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": {"params": "1.1B", "family": "llama"},
        "Qwen/Qwen2-0.5B-Instruct": {"params": "0.5B", "family": "qwen"},
    }

    # Rough tokens/sec estimates based on model size (empirical for M1 Mac)
    EST_TPS = {"tiny": 60, "small": 35, "medium": 15, "large": 6, "xl": 2}
    _tps_for = lambda p: EST_TPS["tiny"] if "82" in p or "124" in p or "125" in p else \
                         EST_TPS["small"] if "355" in p or "0.5" in p else \
                         EST_TPS["medium"] if "774" in p or "1.1" in p else \
                         EST_TPS["large"] if "2.7" in p or "3.8" in p else \
                         EST_TPS["xl"]

    result = {"models": [], "current_model": current_id}
    for m in models:
        meta = MODEL_META.get(m, {"params": "—", "family": "other"})
        is_loaded = m == current_id

        result["models"].append({
            "name": m,
            "family": meta["family"],
            "params": meta["params"],
            "loaded": is_loaded,
            "tokens_per_sec": loaded_tps if is_loaded else _tps_for(meta["params"]),
            "memory_mb": _get_hf_model_size_gb(m) * 1024,
            "inference_count": ctrl._inference_count if is_loaded else 0,
        })

    return result


def _get_hf_model_size_gb(model_id: str) -> float:
    """Estimate model size in GB from param count."""
    from controllers.models import get_models_controller
    size_map = {
        "distilgpt2": 0.3,
        "gpt2": 0.5,
        "gpt2-medium": 1.5,
        "gpt2-large": 3.0,
        "EleutherAI/gpt-neo-125M": 0.5,
        "microsoft/phi-2": 5.0,
        "microsoft/Phi-3-mini-128k-instruct": 7.0,
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 2.2,
        "Qwen/Qwen2-0.5B-Instruct": 1.0,
    }
    if model_id in size_map:
        return size_map[model_id]
    # Fallback: cache check
    import hashlib, json
    try:
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        safe_id = "models--" + model_id.replace("/", "--")
        snapshots = cache_dir / safe_id / "snapshots"
        if snapshots.exists():
            refs = cache_dir / safe_id / "refs"
            main = (refs / "main").read_text().strip() if (refs / "main").exists() else None
            if main:
                for blob in (snapshots / main).glob("**/*"):
                    if blob.is_file() and blob.stat().st_size > 1_000_000:
                        return blob.stat().st_size / (1024 ** 3)
    except Exception:
        pass
    return 0.5


@router.get("/quality")
async def get_quality_metrics(
    limit: int = 50,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get response quality metrics from logged responses.
    
    Returns coherence_score, quality_score, repetition_rate, etc.
    Uses BenchmarkDomain for clean architecture.
    """
    try:
        from domains import get_benchmark_domain
        
        bench = get_benchmark_domain()
        return bench.evaluate_latest(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/responses")
async def get_logged_responses(
    limit: int = 20,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Get recent logged responses for review."""
    try:
        from domains.feedback.response_tracker import get_response_tracker
        
        tracker = get_response_tracker()
        responses = tracker.get_responses(limit=limit, model=model)
        
        return {
            "responses": [
                {
                    "timestamp": r.timestamp,
                    "user_message": r.user_message[:100],
                    "assistant_response": r.assistant_response[:200],
                    "model": r.model,
                    "tokens_generated": r.tokens_generated,
                    "duration_ms": r.duration_ms,
                }
                for r in responses
            ],
            "count": len(responses),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_tracker_stats() -> Dict[str, Any]:
    """Get response tracker statistics - uses BenchmarkDomain."""
    try:
        from domains import get_benchmark_domain
        bench = get_benchmark_domain()
        return bench.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


__all__ = ["router"]