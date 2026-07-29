"""
Benchmark Router - Model benchmarking endpoints

Includes quality evaluation:
- Coherence scoring
- Repetition detection
- Real model metrics
"""
import time
from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any

from schemas.common import success_response


class BenchmarkRouter:
    """Router for model benchmarking and quality evaluation."""

    def __init__(self):
        self.router = APIRouter(prefix="/benchmark", tags=["benchmark"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/run", self.run_benchmark, methods=["POST"])
        self.router.add_api_route("/metrics", self.get_model_metrics, methods=["GET"])
        self.router.add_api_route("/perplexity", self.calculate_perplexity, methods=["POST"])
        self.router.add_api_route("/quality", self.get_quality_metrics, methods=["GET"])
        self.router.add_api_route("/responses", self.get_logged_responses, methods=["GET"])
        self.router.add_api_route("/stats", self.get_tracker_stats, methods=["GET"])
        self.router.add_api_route("/history/clear", self.clear_history, methods=["POST"])

    def _get_model_metrics(self, model: str) -> Dict[str, Any]:
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

    async def run_benchmark(self, model: str = "gpt2"):
        """Run model benchmark - returns real metrics"""
        return success_response(data=self._get_model_metrics(model))

    async def get_model_metrics(self, model: str = "gpt2"):
        """Get real-time model metrics"""
        return success_response(data=self._get_model_metrics(model))

    async def calculate_perplexity(self, text: str = "Sample text for evaluation"):
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

            return success_response(data={
                "text": text[:30],
                "perplexity": round(perplexity, 2),
                "loss": round(loss, 4),
                "tokens": inputs["input_ids"].shape[1],
            })
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_quality_metrics(
        self,
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
            return success_response(data=bench.evaluate_latest(limit=limit))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_logged_responses(
        self,
        limit: int = 20,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get recent logged responses for review."""
        try:
            from domains.feedback.response_tracker import get_response_tracker

            tracker = get_response_tracker()
            responses = tracker.get_responses(limit=limit, model=model)

            return success_response(data={
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
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_tracker_stats(self) -> Dict[str, Any]:
        """Get response tracker statistics - uses BenchmarkDomain."""
        try:
            from domains import get_benchmark_domain

            bench = get_benchmark_domain()
            return success_response(data=bench.get_stats())
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def clear_history(self) -> dict:
        """Clear benchmark history and logged responses."""
        try:
            from domains import get_benchmark_domain

            bench = get_benchmark_domain()
            bench.clear_history()
            return success_response(data={"status": "ok", "cleared": True})
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


router = BenchmarkRouter().router

__all__ = ["router"]
