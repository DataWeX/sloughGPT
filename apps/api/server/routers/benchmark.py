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

from schemas.common import success_response, error_response


def _numpy_perplexity(model, ids):
    """Compute next-token perplexity with a single causal forward pass.

    With a causal mask, logits at position i see tokens 0..i and are used to
    score the successor token ids[i+1]; NLL is averaged over positions
    0..N-2. The final token has no successor and is excluded.

    Args:
        model: SloNet model exposing ``forward(input_ids, targets)`` and
            returning ``(logits, loss)`` where logits is a numpy-backed
            tensor of shape (1, seq_len, vocab).
        ids: List of token ids.

    Returns:
        (perplexity, mean negative log-likelihood) as floats.
    """
    import numpy as np

    input_ids = np.array(ids, dtype=np.int64).reshape(1, -1)
    logits_t, _ = model.forward(input_ids, None)
    logits = np.asarray(logits_t.data, dtype=np.float64)[0]  # (N, vocab)
    targets = input_ids[0, 1:]  # next-token labels for positions 0..N-2
    head = logits[: targets.shape[0]]  # only positions with a successor
    head = head - head.max(axis=-1, keepdims=True)
    log_probs = head - np.log(np.sum(np.exp(head), axis=-1, keepdims=True))
    nll = float(-log_probs[np.arange(targets.shape[0]), targets].mean())
    return float(np.exp(nll)), nll


def _process_memory_mb() -> float:
    """Resident memory of this process in MiB (torch-free best effort).

    Prefers psutil when installed; falls back to ``/proc/self/statm``
    (Linux) or ``resource.getrusage`` (POSIX).

    Returns:
        RSS in MiB, or 0.0 when undeterminable.
    """
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        pass
    try:
        import os
        with open("/proc/self/statm") as fh:
            resident_pages = int(fh.read().split()[1])
        page_size_kb = os.sysconf("SC_PAGE_SIZE") / 1024
        return resident_pages * page_size_kb / 1024
    except Exception:
        pass
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        pass
    return 0.0


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
        """Get real model metrics from the active SloNet provider.

        Reads the provider that ModelsController publishes into the core
        ServerState (pure NumPy; no torch involved). Inference counters come
        from the controller; memory is this process's resident set.

        Args:
            model: Requested model id (echoed back in the response).

        Returns:
            Metrics dict, or ``{"model": ..., "model_loaded": False}`` when
            no provider is resident.

        Side effects:
            - Reads the ServerState singleton.
        """
        try:
            from controllers.models import get_models_controller
            from domains.infrastructure.server_state import get_server_state

            ctrl = get_models_controller()
            provider = get_server_state().model.get()
            if provider is None:
                return {"model": model, "model_loaded": False}

            inference_time = time.time() - ctrl._last_inference_time if ctrl._last_inference_time else 0
            total_tokens = ctrl._total_tokens_generated
            total_inferences = ctrl._inference_count

            # Calculate throughput
            tokens_per_sec = 0
            if inference_time > 0 and total_tokens > 0 and total_inferences > 0:
                tokens_per_sec = total_tokens / (inference_time * total_inferences)

            return {
                "model": model,
                "model_loaded": True,
                "model_id": getattr(provider, "model_id", model),
                "inference_count": total_inferences,
                "total_tokens": total_tokens,
                "tokens_per_second": round(tokens_per_sec, 1),
                "memory_mb": round(_process_memory_mb(), 1),
                "num_parameters": getattr(provider, "num_parameters", None),
            }
        except Exception as e:
            return error_response(str(e), "E_DOMAIN", details={"model": model})

    async def run_benchmark(self, model: str = "gpt2"):
        """Run model benchmark - returns real metrics"""
        return success_response(data=self._get_model_metrics(model))

    async def get_model_metrics(self, model: str = "gpt2"):
        """Get real-time model metrics"""
        return success_response(data=self._get_model_metrics(model))

    async def calculate_perplexity(self, text: str = "Sample text for evaluation"):
        """Calculate next-token perplexity on text using the active SloNet model.

        Pure NumPy: one causal forward pass, then the negative log-likelihood
        of each token's successor averaged over positions 0..N-2. No torch.

        Args:
            text: Text to evaluate.

        Returns:
            Success envelope with perplexity, loss, and token count.

        Side effects:
            - May materialize a lazy guard-backed model in the parent process
              for the duration of the forward pass.
        """
        try:
            from domains.infrastructure.server_state import get_server_state

            provider = get_server_state().model.get()
            if provider is None:
                raise HTTPException(status_code=400, detail="Model not loaded")
            model = getattr(provider, "_get_model", lambda: None)()
            if model is None:
                raise HTTPException(status_code=400, detail="Model not loaded")

            ids = provider.tokenize(text)
            if len(ids) < 2:
                raise HTTPException(status_code=400, detail="Input produced fewer than 2 tokens")

            perplexity, loss = _numpy_perplexity(model, ids)

            return success_response(data={
                "text": text[:30],
                "perplexity": round(perplexity, 2),
                "loss": round(loss, 4),
                "tokens": len(ids),
            })
        except HTTPException:
            raise
        except Exception as e:
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="benchmark")
            raise HTTPException(status_code=err.http_status, detail=err.user_message)

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
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="benchmark")
            raise HTTPException(status_code=err.http_status, detail=err.user_message)

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
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="benchmark")
            raise HTTPException(status_code=err.http_status, detail=err.user_message)

    async def get_tracker_stats(self) -> Dict[str, Any]:
        """Get response tracker statistics - uses BenchmarkDomain."""
        try:
            from domains import get_benchmark_domain

            bench = get_benchmark_domain()
            return success_response(data=bench.get_stats())
        except Exception as e:
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="benchmark")
            raise HTTPException(status_code=err.http_status, detail=err.user_message)

    async def clear_history(self) -> dict:
        """Clear benchmark history and logged responses."""
        try:
            from domains import get_benchmark_domain

            bench = get_benchmark_domain()
            bench.clear_history()
            return success_response(data={"status": "ok", "cleared": True})
        except Exception as e:
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="benchmark")
            raise HTTPException(status_code=err.http_status, detail=err.user_message)


router = BenchmarkRouter().router

__all__ = ["router"]
