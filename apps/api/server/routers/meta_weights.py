"""
Meta Weights Router - Feedback-driven weight adaptation
Migration target for inline /meta-weights/* endpoints from main.py.
"""
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel

from schemas.common import success_response


class GetMetaWeightsRequest(BaseModel):
    user_message: str
    k: int = 5
    user_id: str = "default"


class MetaWeightResponse(BaseModel):
    temperature: float = 0.8
    repetition_penalty: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    based_on_samples: int = 0


class MetaWeightsRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/meta-weights", tags=["meta-weights"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/get", self.get_meta_weights, methods=["POST"], response_model=MetaWeightResponse)
        self.router.add_api_route("/stats", self.get_meta_weight_stats, methods=["GET"])

    async def get_meta_weights(self, request: GetMetaWeightsRequest, req: Request):
        """Get meta-weight adjustments based on similar past feedback."""
        from domains.feedback import get_meta_weight_manager as _get_manager
        manager = _get_manager()
        if manager is None:
            raise HTTPException(status_code=503, detail="Meta-weight system not available")
        weights = manager.get_adjustment(
            user_message=request.user_message, k=request.k or 5, user_id=request.user_id or "default"
        )
        return MetaWeightResponse(
            temperature=weights.temperature,
            repetition_penalty=weights.repetition_penalty,
            top_p=weights.top_p,
            top_k=weights.top_k,
            based_on_samples=len(manager._weight_history),
        )

    async def get_meta_weight_stats(self, req: Request):
        """Get meta-weight system statistics."""
        from domains.feedback import get_meta_weight_manager as _get_manager
        manager = _get_manager()
        if manager is None:
            raise HTTPException(status_code=503, detail="Meta-weight system not available")
        return success_response(data=manager.get_stats())


router = MetaWeightsRouter().router
