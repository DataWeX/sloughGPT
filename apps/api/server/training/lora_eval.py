"""
LoRA Evaluation Router - Trigger adapter quality evaluation.
"""

import asyncio
import logging
import re
from pathlib import Path
from typing import ClassVar

from fastapi import APIRouter, Depends
from infrastructure.auth import require_auth_if_enabled
from infrastructure.contract import RouteSpec, create_router
from schemas.common import (
    classify_and_raise,
    endpoint,
    raise_error,
    safe_audit_log,
    success_response,
)

from domain.agents import ToolParam, ToolSpec

_VALID_ADAPTER_PATH = re.compile(r"^[\w\-/]+\.npz$")
_ADAPTER_BASE = Path("data/user_adapters").resolve()


class LoraEvalRouter:
    # One projection per process (contract registry: same identity
    # re-registering = double projection). Later instances — test fixtures —
    # share the router the first instance registered.
    _shared_router: ClassVar[APIRouter | None] = None

    def __init__(self):
        if LoraEvalRouter._shared_router is not None:
            self.router = LoraEvalRouter._shared_router
            return
        # Descriptor projection (card 20261101_154058): full paths in the
        # RouteSpecs — no prefix, byte-identical paths on direct mount.
        self.router = APIRouter()
        self._register_routes()
        LoraEvalRouter._shared_router = self.router

    def _register_routes(self):
        # Both POSTs are query-carried (params=... in tests and the hand-built
        # FastAPI twin's plain defaults) — input="query" keeps that wire.
        # envelope=False: handlers keep their success_response wraps.
        self.router.include_router(
            create_router(
                ToolSpec(
                    name="training.lora_eval.run",
                    description=(
                        "Run LoRA quality evaluation — baseline vs with-adapter comparison.\n"
                        "\n"
                        "Args:\n"
                        "    adapter_path: path to adapter .npz file\n"
                        "    soul: soul name for evaluation\n"
                        "\n"
                        "Returns:\n"
                        "    dict with baseline/with_adapter metrics and delta comparison\n"
                        "\n"
                        "Side effects:\n"
                        "    - calls LoRAEvaluator.run() with and without adapter\n"
                        "    - saves results to eval history"
                    ),
                    parameters=[
                        ToolParam("adapter_path", "string", "", False),
                        ToolParam("soul", "string", "", False),
                    ],
                    execute=self.run_eval,
                    auth_scope="public",
                    idempotent=False,
                    version="1",
                ),
                RouteSpec(
                    path="/lora-eval/run",
                    input="query",
                    envelope=False,
                    summary="Run LoRA Evaluation",
                    tags=["lora-eval"],
                ),
                self.run_eval,
            )
        )
        self.router.include_router(
            create_router(
                ToolSpec(
                    name="training.lora_eval.history",
                    description=(
                        "Retrieve recent evaluation history.\n"
                        "\n"
                        "Args:\n"
                        "    limit: max number of results to return (default 20)\n"
                        "\n"
                        "Returns:\n"
                        "    dict with list of eval result dicts\n"
                        "\n"
                        "Side effects:\n"
                        "    - reads from LoRAEvaluator history"
                    ),
                    parameters=[ToolParam("limit", "integer", "", False)],
                    execute=self.get_eval_history,
                    auth_scope="public",
                    idempotent=True,
                    version="1",
                ),
                RouteSpec(
                    path="/lora-eval/history",
                    envelope=False,
                    summary="Get Evaluation History",
                    tags=["lora-eval"],
                ),
                self.get_eval_history,
            )
        )
        self.router.include_router(
            create_router(
                ToolSpec(
                    name="training.lora_eval.aggregate",
                    description=(
                        "Trigger aggregation of top-k best user adapters.\n"
                        "\n"
                        "Calls PerUserLoRAStore.aggregate_best_adapters() which merges top adapters,\n"
                        "then runs LoRAEvaluator before/after to measure quality delta."
                    ),
                    parameters=[
                        ToolParam("top_k", "integer", "", False),
                        ToolParam("min_feedback", "integer", "", False),
                        ToolParam("output_name", "string", "", False),
                        ToolParam("run_eval", "boolean", "", False),
                    ],
                    execute=self.trigger_aggregation,
                    auth_scope="authenticated",
                    idempotent=False,
                    version="1",
                ),
                RouteSpec(
                    path="/lora-eval/aggregate",
                    input="query",
                    envelope=False,
                    summary="Aggregate Best Adapters",
                    tags=["lora-eval"],
                ),
                self.trigger_aggregation,
            )
        )

    @endpoint("lora_eval.run")
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
            raise_error(
                f"Invalid adapter path: {adapter_path!r}",
                code="E_VAL_REQUEST",
                details={"adapter_path": adapter_path},
            )
        resolved = Path(adapter_path).resolve()
        if not str(resolved).startswith(str(_ADAPTER_BASE)):
            raise_error(
                f"Adapter path must be under data/user_adapters/: {adapter_path!r}",
                code="E_VAL_REQUEST",
                details={"adapter_path": adapter_path},
            )
        try:
            import time as _time

            from domain.feedback import get_lora_evaluator

            evaluator = get_lora_evaluator()
            adapter_file = adapter_path

            _t0 = _time.monotonic()
            baseline = evaluator.run(adapter_path=None, soul_name=soul, save=True)

            try:
                if await asyncio.to_thread(Path(adapter_file).exists):
                    with_adapter = evaluator.run(
                        adapter_path=adapter_file, soul_name=soul, save=True
                    )
                    delta = evaluator.compare(baseline, with_adapter)
                    _elapsed_ms = (_time.monotonic() - _t0) * 1000
                    safe_audit_log(
                        "lora_eval.run_eval",
                        resource=soul or "default",
                        detail=f"elapsed={_elapsed_ms:.0f}ms",
                    )
                    return success_response(
                        data={
                            "status": "compared",
                            "baseline": baseline.to_dict(),
                            "with_adapter": with_adapter.to_dict(),
                            "delta": delta,
                            "report": evaluator.compare_with_report(baseline, with_adapter),
                            "elapsed_ms": round(_elapsed_ms, 1),
                        }
                    )
            except Exception as e:
                logging.getLogger("slo.lora_eval").warning("Adapter comparison failed: %s", e)

            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            safe_audit_log(
                "lora_eval.run_eval",
                resource=soul or "default",
                detail=f"baseline_only elapsed={_elapsed_ms:.0f}ms",
            )
            return success_response(
                data={
                    "status": "baseline_only",
                    "baseline": baseline.to_dict(),
                    "note": "No adapter found — run aggregate first",
                }
            )
        except Exception as e:
            logging.getLogger("slo.lora_eval").warning("LoRA eval run failed: %s", e)
            classify_and_raise(e, source="lora_eval_run")

    @endpoint("lora_eval.history")
    async def get_eval_history(
        self,
        limit: int = 20,  # plain default: the projection omits absent query keys
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
        # Was FastAPI Query(ge=1, le=100); the projection's ToolParam has no
        # range vocabulary, so the bound lives here — same 422, same inputs.
        if not 1 <= limit <= 100:
            raise_error(
                f"limit must be between 1 and 100, got {limit}",
                "E_VAL_REQUEST",
                status_code=422,
            )
        try:
            from domain.feedback import get_lora_evaluator

            evaluator = get_lora_evaluator()
            results = evaluator.get_history(limit=limit)
            return success_response(data={"results": [r.to_dict() for r in results]})
        except Exception as e:
            raise_error(str(e), "E_INFRA_STARTUP", status_code=500)

    @endpoint("lora_eval.aggregate")
    async def trigger_aggregation(
        self,
        top_k: int = 10,  # plain defaults: the projection omits absent query keys
        min_feedback: int = 5,
        output_name: str = "best_aggregated",
        run_eval: bool = True,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """
        Trigger aggregation of top-k best user adapters.

        Calls PerUserLoRAStore.aggregate_best_adapters() which merges top adapters,
        then runs LoRAEvaluator before/after to measure quality delta.
        """
        # Was FastAPI Query(ge/le); range bound moved here — same 422, same inputs.
        if not 1 <= top_k <= 50:
            raise_error(
                f"top_k must be between 1 and 50, got {top_k}",
                "E_VAL_REQUEST",
                status_code=422,
            )
        if min_feedback < 1:
            raise_error(
                f"min_feedback must be at least 1, got {min_feedback}",
                "E_VAL_REQUEST",
                status_code=422,
            )
        try:
            from domain.feedback import get_per_user_lora

            store = get_per_user_lora()
            result = store.aggregate_best_adapters(
                top_k=top_k,
                min_feedback_count=min_feedback,
                output_name=output_name,
                run_eval=run_eval,
            )

            safe_audit_log(
                "adapter.eval.aggregate",
                resource=output_name,
                user_count=result.get("user_count", 0),
                total_feedback=result.get("total_feedback", 0),
            )

            if "error" in result:
                return success_response(data={"status": "no_adapters", "message": result["error"]})

            eval_result = result.get("eval", {})
            if "error" not in eval_result:
                return success_response(
                    data={
                        "status": "aggregated_with_eval",
                        "output_path": result["output_path"],
                        "user_count": result["user_count"],
                        "total_feedback": result["total_feedback"],
                        "eval": {
                            "verdict": eval_result.get("delta", {}).get("verdict", "unknown"),
                            "perplexity_delta": eval_result.get("delta", {}).get(
                                "perplexity_delta"
                            ),
                            "bleu_delta": eval_result.get("delta", {}).get("bleu_delta"),
                            "throughput_delta": eval_result.get("delta", {}).get(
                                "throughput_delta"
                            ),
                            "report": eval_result.get("report", ""),
                        },
                    }
                )

            return success_response(
                data={
                    "status": "aggregated_no_eval",
                    "output_path": result["output_path"],
                    "user_count": result["user_count"],
                    "total_feedback": result["total_feedback"],
                }
            )
        except Exception as e:
            logging.getLogger("slo.lora_eval").warning("LoRA eval aggregate failed: %s", e)
            classify_and_raise(e, source="lora_eval_aggregate")


router = LoraEvalRouter().router
