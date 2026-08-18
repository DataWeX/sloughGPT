"""
LoRA Evaluation Router - Trigger adapter quality evaluation.
"""
from fastapi import APIRouter, HTTPException, Query

from schemas.common import success_response, classify_and_raise, safe_audit_log


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
    ):
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
        try:
            from domains.feedback.lora_eval import get_lora_evaluator

            evaluator = get_lora_evaluator()
            adapter_file = adapter_path

            baseline = evaluator.run(adapter_path=None, soul_name=soul, save=True)

            try:
                from pathlib import Path
                if Path(adapter_file).exists():
                    with_adapter = evaluator.run(adapter_path=adapter_file, soul_name=soul, save=True)
                    delta = evaluator.compare(baseline, with_adapter)
                    return success_response(data={
                        "status": "compared",
                        "baseline": baseline.to_dict(),
                        "with_adapter": with_adapter.to_dict(),
                        "delta": delta,
                        "report": evaluator.compare_with_report(baseline, with_adapter),
                    })
            except Exception as e:
                import logging
                logging.getLogger("slo.lora_eval").warning("Adapter comparison failed: %s", e)

            return success_response(data={
                "status": "baseline_only",
                "baseline": baseline.to_dict(),
                "note": "No adapter found — run aggregate first",
            })
        except Exception as e:
            classify_and_raise(e, source="lora_eval_run")
            raise HTTPException(status_code=err.http_status, detail=err.user_message)

    async def get_eval_history(
        self,
        limit: int = Query(default=20, ge=1, le=100),
    ):
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
            raise HTTPException(status_code=500, detail=str(e))

    async def trigger_aggregation(
        self,
        top_k: int = Query(default=10, ge=1, le=50),
        min_feedback: int = Query(default=5, ge=1),
        output_name: str = Query(default="best_aggregated"),
        run_eval: bool = Query(default=True),
    ):
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
            classify_and_raise(e, source="lora_eval_aggregate")
            raise HTTPException(status_code=err.http_status, detail=err.user_message)


router = LoraEvalRouter().router
