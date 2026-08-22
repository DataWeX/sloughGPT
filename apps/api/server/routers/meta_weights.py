"""
Meta Weights Router - Feedback-driven weight adaptation.
"""
import logging
import time as _time
from fastapi import APIRouter, Request
from pydantic import BaseModel

from schemas.common import raise_error, success_response
from domains.infrastructure.logging import safe_audit_log

logger = logging.getLogger("slo.routers.meta_weights")


class GetMetaWeightsRequest(BaseModel):
    user_message: str
    k: int = 5
    user_id: str = "default"


class MetaWeightResponse(BaseModel):
    temperature: float = 0.8
    repetition_penalty: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    style_bias: float = 0.0
    confidence_boost: float = 0.0
    based_on_samples: int = 0


class MetaWeightsRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/meta-weights", tags=["meta-weights"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/ping", self.ping, methods=["GET"])
        self.router.add_api_route("/get", self.get_meta_weights, methods=["POST"], response_model=MetaWeightResponse)
        self.router.add_api_route("/stats", self.get_meta_weight_stats, methods=["GET"])

    async def get_meta_weights(self, request: GetMetaWeightsRequest, req: Request) -> dict:
        """Get meta-weight adjustments based on similar past feedback."""
        _t0 = _time.monotonic()
        from domains.feedback import get_meta_weight_manager as _get_manager
        manager = _get_manager()
        if manager is None:
            raise_error("Meta-weight system not available", "E_BAD_REQUEST", status_code=503)
        weights = manager.get_adjustment(
            user_message=request.user_message, k=request.k or 5, user_id=request.user_id or "default"
        )
        _elapsed_ms = (_time.monotonic() - _t0) * 1000
        logger.info("Meta-weights computed in %.1fms (samples=%d)", _elapsed_ms, len(manager._weight_history))
        safe_audit_log("meta_weights.get", resource=request.user_message[:80], detail=f"elapsed={_elapsed_ms:.0f}ms samples={len(manager._weight_history)}")
        return MetaWeightResponse(
            temperature=weights.temperature,
            repetition_penalty=weights.repetition_penalty,
            top_p=weights.top_p,
            top_k=weights.top_k,
            style_bias=weights.style_bias,
            confidence_boost=weights.confidence_boost,
            based_on_samples=len(manager._weight_history),
        )

    async def get_meta_weight_stats(self, req: Request) -> dict:
        """Get meta-weight system statistics.

        Returns db stats, quality trend, current average weights,
        and history length.

        Side effects:
            - reads from feedback database
        """
        from domains.feedback import get_meta_weight_manager as _get_manager
        manager = _get_manager()
        if manager is None:
            raise_error("Meta-weight system not available", "E_BAD_REQUEST", status_code=503)
        return success_response(data=manager.get_stats())

    async def ping(self) -> dict:
        """
        Health probe for the meta-weights system.

        Returns:
            dict: ``{"status": "ok"}``

        Side effects:
            - none
        """
        return success_response(data={"status": "ok"})


router = MetaWeightsRouter().router
