"""
Consciousness Router — API endpoints for the consciousness system.

Provides status, reflection, training control, and level configuration
for the self-awareness subsystem.
"""

import logging

from fastapi import APIRouter, Depends, Query
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel, Field
from schemas.common import endpoint, raise_error, success_response

logger = logging.getLogger("slo.routers.consciousness")


class ConsciousnessLevelRequest(BaseModel):
    level: int = Field(..., ge=0, le=3, description="Consciousness level (0=off, 1=basic, 2=full, 3=deep)")


class ConsciousnessTrainRequest(BaseModel):
    model_path: str = Field(default="", description="Base model path for LoRA training")


class ConsciousnessRouter:
    """API endpoints for the consciousness system."""

    def __init__(self):
        self.router = APIRouter(prefix="/consciousness", tags=["consciousness"])
        self._engine = None
        self._trainer = None
        self._register_routes()

    def _get_engine(self):
        """Lazy-load the consciousness engine."""
        if self._engine is None:
            from domains.consciousness import get_consciousness
            self._engine = get_consciousness()
        return self._engine

    def _get_trainer(self):
        """Lazy-load the consciousness trainer."""
        if self._trainer is None:
            from domains.consciousness.training import ConsciousnessTrainer, TrainingConfig
            engine = self._get_engine()
            config = TrainingConfig(
                model_path="",
                rank=engine.config.lora_rank,
                alpha=float(engine.config.lora_alpha),
                min_pairs_for_training=10,
            )
            self._trainer = ConsciousnessTrainer(config)
        return self._trainer

    def _register_routes(self):
        self.router.add_api_route("/status", self.get_status, methods=["GET"])
        self.router.add_api_route("/self-model", self.get_self_model, methods=["GET"])
        self.router.add_api_route("/qualia", self.get_qualia, methods=["GET"])
        self.router.add_api_route("/reflect", self.reflect, methods=["POST"])
        self.router.add_api_route("/config", self.update_config, methods=["PATCH"])
        self.router.add_api_route("/train/status", self.train_status, methods=["GET"])
        self.router.add_api_route("/train/start", self.train_start, methods=["POST"])
        self.router.add_api_route("/evaluate", self.evaluate, methods=["GET"])

    @endpoint("consciousness.status")
    async def get_status(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Get current consciousness system status."""
        engine = self._get_engine()
        status = engine.get_status()
        trainer = self._get_trainer()
        status["training"] = trainer.get_status()
        return success_response(data=status)

    @endpoint("consciousness.self_model")
    async def get_self_model(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Get the self-model state."""
        engine = self._get_engine()
        sm = engine.self_model
        return success_response(data={
            "identity": {
                "name": sm.identity.name,
                "capabilities": sm.identity.capabilities,
                "limitations": sm.identity.limitations,
                "values": sm.identity.values,
            },
            "beliefs": dict(sm.self_beliefs),
            "doubts": sm.self_doubts,
            "episode_count": len(sm.episodes),
        })

    @endpoint("consciousness.qualia")
    async def get_qualia(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Get current qualia state and narrative."""
        engine = self._get_engine()
        qualia = engine.qualia
        return success_response(data={
            "current": qualia.current.to_dict(),
            "narrative": qualia.get_narrative(),
            "history_count": len(qualia.history),
        })

    @endpoint("consciousness.reflect")
    async def reflect(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Trigger a self-reflection."""
        engine = self._get_engine()
        reflection = engine.reflect()
        return success_response(data={"reflection": reflection})

    @endpoint("consciousness.update_config")
    async def update_config(
        self,
        req: ConsciousnessLevelRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Update consciousness level (0-3)."""
        engine = self._get_engine()
        engine.config.level = req.level
        engine.config.save()
        return success_response(data={
            "level": engine.config.level,
            "enabled": engine.config.is_enabled(),
        })

    @endpoint("consciousness.train_status")
    async def train_status(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Get consciousness training status."""
        trainer = self._get_trainer()
        return success_response(data=trainer.get_status())

    @endpoint("consciousness.train_start")
    async def train_start(
        self,
        req: ConsciousnessTrainRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Start consciousness LoRA training."""
        trainer = self._get_trainer()
        if trainer.is_training:
            raise_error("Training already in progress", "E_BUSY", status_code=409)
        if not trainer.should_train():
            raise_error(
                f"Need at least {trainer.config.min_pairs_for_training} training pairs "
                f"(have {len(trainer._pairs)})",
                "E_INSUFFICIENT_DATA",
                status_code=400,
            )
        if req.model_path:
            trainer.config.model_path = req.model_path

        import asyncio
        result = await asyncio.to_thread(trainer.train)
        return success_response(data=result.to_dict())

    @endpoint("consciousness.evaluate")
    async def evaluate(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Evaluate consciousness system quality."""
        from domains.consciousness.evaluation import ConsciousnessEvaluator

        engine = self._get_engine()
        evaluator = ConsciousnessEvaluator()

        episodes = [
            {
                "self_insight": e.self_insight,
                "growth_delta": e.growth_delta,
                "qualia": e.qualia,
            }
            for e in engine.self_model.episodes
        ]
        beliefs = dict(engine.self_model.self_beliefs)
        qualia_history = [
            q[1].to_dict() for q in engine.qualia.history
        ]

        report = evaluator.evaluate(episodes, beliefs, qualia_history)
        return success_response(data=report.to_dict())


router = ConsciousnessRouter().router
