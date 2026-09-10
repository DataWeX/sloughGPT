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


class FeedbackRequest(BaseModel):
    episode_index: int = Field(..., ge=0, description="Index of the episode to rate")
    rating: int = Field(..., ge=1, le=5, description="Rating 1-5")


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
        self.router.add_api_route("/history/episodes", self.get_episode_history, methods=["GET"])
        self.router.add_api_route("/history/qualia", self.get_qualia_history, methods=["GET"])
        self.router.add_api_route("/history/beliefs", self.get_beliefs_history, methods=["GET"])
        self.router.add_api_route("/feedback", self.submit_feedback, methods=["POST"])
        self.router.add_api_route("/seed", self.seed_data, methods=["POST"])

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

    @endpoint("consciousness.episode_history")
    async def get_episode_history(
        self,
        limit: int = 50,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Get episode history with timestamps."""
        engine = self._get_engine()
        episodes = engine.self_model.episodes[-limit:]
        return success_response(data={
            "episodes": [
                {
                    "timestamp": e.timestamp,
                    "input_text": e.input_text[:200],
                    "self_insight": e.self_insight,
                    "growth_delta": round(e.growth_delta, 4),
                    "qualia": e.qualia,
                }
                for e in episodes
            ],
            "total": len(engine.self_model.episodes),
        })

    @endpoint("consciousness.qualia_history")
    async def get_qualia_history(
        self,
        limit: int = 100,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Get qualia history as time series."""
        engine = self._get_engine()
        history = engine.qualia.history[-limit:]
        return success_response(data={
            "history": [
                {
                    "timestamp": ts,
                    **state.to_dict(),
                }
                for ts, state in history
            ],
            "total": len(engine.qualia.history),
        })

    @endpoint("consciousness.beliefs_history")
    async def get_beliefs_history(
        self,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Get beliefs evolution derived from episodes."""
        engine = self._get_engine()
        episodes = engine.self_model.episodes
        if not episodes:
            return success_response(data={"beliefs": [], "labels": []})

        # Reconstruct beliefs evolution: start from defaults, replay episodes
        defaults = {
            "competence": 0.7,
            "helpfulness": 0.8,
            "creativity": 0.5,
            "accuracy": 0.6,
            "empathy": 0.4,
        }
        beliefs = dict(defaults)
        evolution = []
        step = max(1, len(episodes) // 50)  # Cap at 50 data points

        for i, ep in enumerate(episodes):
            # Apply the same deltas as self_model._update_beliefs_from_episode
            if ep.growth_delta > 0.02:
                beliefs["competence"] = min(1.0, beliefs["competence"] + 0.03)
                beliefs["helpfulness"] = min(1.0, beliefs["helpfulness"] + 0.02)
                beliefs["accuracy"] = min(1.0, beliefs["accuracy"] + 0.01)
            elif ep.growth_delta < -0.02:
                beliefs["competence"] = max(0.0, beliefs["competence"] - 0.03)
                beliefs["helpfulness"] = max(0.0, beliefs["helpfulness"] - 0.01)

            novelty = ep.qualia.get("novelty", 0.5)
            if novelty > 0.7:
                beliefs["creativity"] = min(1.0, beliefs["creativity"] + 0.03)

            valence = ep.qualia.get("valence", 0.0)
            if valence > 0.3:
                beliefs["empathy"] = min(1.0, beliefs["empathy"] + 0.02)
            elif valence < -0.3:
                beliefs["empathy"] = max(0.0, beliefs["empathy"] - 0.01)

            if i % step == 0 or i == len(episodes) - 1:
                evolution.append({
                    "timestamp": ep.timestamp,
                    "step": i,
                    **{k: round(v, 4) for k, v in beliefs.items()},
                })

        return success_response(data={
            "beliefs": evolution,
            "labels": list(defaults.keys()),
        })

    @endpoint("consciousness.feedback")
    async def submit_feedback(
        self,
        body: FeedbackRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Submit feedback rating for an episode, triggering belief updates."""
        engine = self._get_engine()
        episodes = engine.self_model.episodes

        if body.episode_index >= len(episodes):
            raise_error("Episode not found", "E_NOT_FOUND", status_code=404)

        episode = episodes[body.episode_index]
        # Recompute growth with feedback
        new_growth = engine.self_model._compute_growth(
            episode.input_text, episode.response, body.rating
        )
        episode.growth_delta = new_growth
        engine.self_model._update_beliefs_from_episode(episode)

        return success_response(data={
            "episode_index": body.episode_index,
            "new_growth_delta": round(new_growth, 4),
            "beliefs": {k: round(v, 4) for k, v in engine.self_model.self_beliefs.items()},
        })

    @endpoint("consciousness.seed")
    async def seed_data(
        self,
        count: int = Query(default=30, ge=5, le=200),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Seed consciousness with synthetic episodes for dashboard visualization."""
        import random
        import time as _time

        engine = self._get_engine()

        topics = [
            "What is machine learning?",
            "Explain quantum computing",
            "How does consciousness work?",
            "What are neural networks?",
            "Tell me about recursion",
            "How does encryption work?",
            "What is biodiversity?",
            "Explain the Big Bang",
            "How do vaccines work?",
            "What is dark matter?",
            "How does the brain process memory?",
            "What is the meaning of life?",
            "Explain climate change",
            "How does evolution work?",
            "What is string theory?",
        ]

        responses = [
            "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement.",
            "Consciousness remains one of the hardest problems in science and philosophy.",
            "Neural networks are computing systems inspired by biological neural networks in the brain.",
            "Recursion is a process where a function calls itself as a subroutine.",
            "Encryption is the process of converting information into a code to prevent unauthorized access.",
            "Biodiversity is the variety of life in the world or in a particular habitat.",
            "The Big Bang theory describes the origin of the universe from an extremely hot, dense state.",
            "Vaccines work by training the immune system to recognize and fight specific pathogens.",
            "Dark matter is a hypothetical form of matter that does not interact with electromagnetic radiation.",
        ]

        now = _time.time()
        for i in range(count):
            topic = random.choice(topics)
            resp = random.choice(responses)
            qualia = engine.qualia.experience(topic)
            growth = random.uniform(-0.08, 0.12)

            from domains.consciousness.self_model import SelfEpisode
            episode = SelfEpisode(
                timestamp=now - (count - i) * 120,  # 2 min apart
                input_text=topic,
                response=resp,
                qualia=qualia.to_dict(),
                self_insight=random.choice([
                    "I learned something new here.",
                    "This felt familiar but I refined my understanding.",
                    "An interesting challenge to my existing beliefs.",
                    "I see a pattern emerging in how I process these queries.",
                    "This pushed the boundaries of my knowledge.",
                ]),
                growth_delta=growth,
            )
            engine.self_model.episodes.append(episode)
            engine.self_model._update_beliefs_from_episode(episode)

        return success_response(data={
            "seeded": count,
            "total_episodes": len(engine.self_model.episodes),
        })


router = ConsciousnessRouter().router
