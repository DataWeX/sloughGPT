"""
LoRA Evaluation Router - Trigger adapter quality evaluation.
"""
import asyncio
import logging
import re
from pathlib import Path
from fastapi import APIRouter, Query

from schemas.common import raise_error, success_response, classify_and_raise, safe_audit_log

_VALID_ADAPTER_PATH = re.compile(r'^[\w\-/]+\.npz$')
_ADAPTER_BASE = Path("data/user_adapters").resolve()


class LoraEvalRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/lora-eval", tags=["lora-eval"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/run", self.run_eval, methods=["GET"])
        self.router.add_api_route("/history", self.get_eval_history, methods=["GET"])
        self.router.add_api_route("/aggregate", self.trigger_aggregation, methods=["POST"])

    async def run_eval(
        self,
        adapter_path: str = "data/user_adapters/best_aggregated.npz",
        soul: str = "assistant",
    ) -> dict:
        """
        Run LoRA quality evaluation — baseline vs with-adapter comparison.

        Args:
            adapter_path: path to adapter .npz file
            soul: soul name for evaluation

        Returns:
            dict with baseline/with_adapter metrics and delta comparison

        Side effects:
            - calls LoRAEvaluator.run() with and without adapter
            - saves results to eval history
        """
        if not _VALID_ADAPTER_PATH.match(adapter_path):
            raise_error(f"Invalid adapter path: {adapter_path!r}", code="E_VAL_REQUEST", details={"adapter_path": adapter_path})
        resolved = Path(adapter_path).resolve()
        if not str(resolved).startswith(str(_ADAPTER_BASE)):
            raise_error(f"Adapter path must be under data/user_adapters/: {adapter_path!r}", code="E_VAL_REQUEST", details={"adapter_path": adapter_path})
        try:
            import time as _time
            from domains.feedback.lora_eval import get_lora_evaluator

            evaluator = get_lora_evaluator()
            adapter_file = adapter_path

            _t0 = _time.monotonic()
            baseline = evaluator.run(adapter_path=None, soul_name=soul, save=True)

            try:
                if await asyncio.to_thread(Path(adapter_file).exists):
                    with_adapter = evaluator.run(adapter_path=adapter_file, soul_name=soul, save=True)
                    delta = evaluator.compare(baseline, with_adapter)
                    _elapsed_ms = (_time.monotonic() - _t0) * 1000
                    safe_audit_log("lora_eval.run_eval", resource=soul or "default", detail=f"elapsed={_elapsed_ms:.0f}ms")
                    return success_response(data={
                        "status": "compared",
                        "baseline": baseline.to_dict(),
                        "with_adapter": with_adapter.to_dict(),
                        "delta": delta,
                        "report": evaluator.compare_with_report(baseline, with_adapter),
                        "elapsed_ms": round(_elapsed_ms, 1),
                    })
            except Exception as e:
                logging.getLogger("slo.lora_eval").warning("Adapter comparison failed: %s", e)

            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            safe_audit_log("lora_eval.run_eval", resource=soul or "default", detail=f"baseline_only elapsed={_elapsed_ms:.0f}ms")
            return success_response(data={
                "status": "baseline_only",
                "baseline": baseline.to_dict(),
                "note": "No adapter found — run aggregate first",
            })
        except Exception as e:
            logging.getLogger("slo.lora_eval").warning("LoRA eval run failed: %s", e)
            classify_and_raise(e, source="lora_eval_run")

    async def get_eval_history(
        self,
        limit: int = Query(default=20, ge=1, le=100),
    ) -> dict:
        """
        Retrieve recent evaluation history.

        Args:
            limit: max number of results to return (default 20)

        Returns:
            dict with list of eval result dicts

        Side effects:
            - reads from LoRAEvaluator history
        """
        try:
            from domains.feedback.lora_eval import get_lora_evaluator
            evaluator = get_lora_evaluator()
            results = evaluator.get_history(limit=limit)
            return success_response(data={"results": [r.to_dict() for r in results]})
        except Exception as e:
            raise_error(str(e), "E_INFRA_STARTUP", status_code=500)

    async def trigger_aggregation(
        self,
        top_k: int = Query(default=10, ge=1, le=50),
        min_feedback: int = Query(default=5, ge=1),
        output_name: str = Query(default="best_aggregated"),
        run_eval: bool = Query(default=True),
    ) -> dict:
        """
        Trigger aggregation of top-k best user adapters.

        Calls PerUserLoRAStore.aggregate_best_adapters() which merges top adapters,
        then runs LoRAEvaluator before/after to measure quality delta.
        """
        try:
            from domains.feedback.per_user_lora import get_per_user_lora

            store = get_per_user_lora()
            result = store.aggregate_best_adapters(
                top_k=top_k,
                min_feedback_count=min_feedback,
                output_name=output_name,
                run_eval=run_eval,
            )

            safe_audit_log("adapter.eval.aggregate", resource=output_name, user_count=result.get("user_count", 0), total_feedback=result.get("total_feedback", 0))

            if "error" in result:
                return success_response(data={"status": "no_adapters", "message": result["error"]})

            eval_result = result.get("eval", {})
            if "error" not in eval_result:
                return success_response(data={
                    "status": "aggregated_with_eval",
                    "output_path": result["output_path"],
                    "user_count": result["user_count"],
                    "total_feedback": result["total_feedback"],
                    "eval": {
                        "verdict": eval_result.get("delta", {}).get("verdict", "unknown"),
                        "perplexity_delta": eval_result.get("delta", {}).get("perplexity_delta"),
                        "bleu_delta": eval_result.get("delta", {}).get("bleu_delta"),
                        "throughput_delta": eval_result.get("delta", {}).get("throughput_delta"),
                        "report": eval_result.get("report", ""),
                    },
                })

            return success_response(data={
                "status": "aggregated_no_eval",
                "output_path": result["output_path"],
                "user_count": result["user_count"],
                "total_feedback": result["total_feedback"],
            })
        except Exception as e:
            logging.getLogger("slo.lora_eval").warning("LoRA eval aggregate failed: %s", e)
            classify_and_raise(e, source="lora_eval_aggregate")


router = LoraEvalRouter().router
