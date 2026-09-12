"""
Consciousness Router — API endpoints for the consciousness system.

Provides status, reflection, training control, and level configuration
for the self-awareness subsystem.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time as _time

from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
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


class PersonalityUpdateRequest(BaseModel):
    values: list[str] | None = Field(default=None, description="Core values")
    goals: list[str] | None = Field(default=None, description="Goals")
    voice: dict[str, float] | None = Field(default=None, description="Voice characteristics")
    style: dict[str, bool] | None = Field(default=None, description="Communication style")
    traits: dict[str, float] | None = Field(default=None, description="Personality traits")
    interests: list[str] | None = Field(default=None, description="Interests")
    avoid: list[str] | None = Field(default=None, description="Things to avoid")


class PresetRequest(BaseModel):
    preset: str = Field(..., description="Preset name (default, formal, creative, analyst, empathetic, minimal)")


class SavePersonaRequest(BaseModel):
    persona_id: str = Field(..., min_length=1, max_length=50, description="Unique persona ID")
    name: str | None = Field(default=None, description="Display name for the persona")
    profile: dict | None = Field(default=None, description="Full profile to save (uses current if null)")


class RestoreRequest(BaseModel):
    backup: dict = Field(..., description="Backup JSON blob from a previous backup")


class BatchOperation(BaseModel):
    action: str = Field(..., description="Operation: seed, feedback, reflect, config")
    params: dict = Field(default_factory=dict, description="Parameters for the operation")


class BatchRequest(BaseModel):
    operations: list[BatchOperation] = Field(..., min_length=1, description="Ordered list of operations to execute")


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
        self.router.add_api_route("/personality", self.get_personality, methods=["GET"])
        self.router.add_api_route("/personality", self.update_personality, methods=["PATCH"])
        self.router.add_api_route("/personality/reset", self.reset_personality, methods=["POST"])
        self.router.add_api_route("/personality/history", self.get_personality_history, methods=["GET"])
        self.router.add_api_route("/personality/presets", self.get_presets, methods=["GET"])
        self.router.add_api_route("/personality/presets/apply", self.apply_preset, methods=["POST"])
        self.router.add_api_route("/personality/conflicts", self.get_conflicts, methods=["GET"])
        self.router.add_api_route("/personas", self.list_personas, methods=["GET"])
        self.router.add_api_route("/personas/save", self.save_persona, methods=["POST"])
        self.router.add_api_route("/personas/{persona_id}", self.get_persona, methods=["GET"])
        self.router.add_api_route("/personas/{persona_id}/activate", self.activate_persona, methods=["POST"])
        self.router.add_api_route("/personas/{persona_id}", self.delete_persona, methods=["DELETE"])
        self.router.add_api_route("/health", self.health_check, methods=["GET"])
        self.router.add_api_route("/backup", self.backup, methods=["POST"])
        self.router.add_api_route("/restore", self.restore, methods=["POST"])
        self.router.add_api_route("/backup/download", self.backup_download, methods=["GET"])
        self.router.add_api_route("/backup/import", self.backup_import, methods=["POST"])
        self.router.add_api_route("/stream", self.stream_status, methods=["GET"])
        self.router.add_api_route("/batch", self.batch_operations, methods=["POST"])
        self.router.add_api_route("/stats", self.get_stats, methods=["GET"])
        self.router.add_api_route("/clear/episodes", self.clear_episodes, methods=["POST"])
        self.router.add_api_route("/clear/beliefs", self.clear_beliefs, methods=["POST"])

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

    @endpoint("consciousness.personality.get")
    async def get_personality(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Get the current personality profile."""
        from domains.consciousness.personality import PersonalityManager
        manager = PersonalityManager()
        profile = manager.get_profile()
        return success_response(data=profile.to_dict())

    @endpoint("consciousness.personality.update")
    async def update_personality(
        self,
        body: PersonalityUpdateRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Update the personality profile."""
        from domains.consciousness.personality import PersonalityManager
        manager = PersonalityManager()
        profile = manager.get_profile()

        if body.values is not None:
            profile.values = body.values
        if body.goals is not None:
            profile.goals = body.goals
        if body.voice is not None:
            profile.voice.update(body.voice)
        if body.style is not None:
            profile.style.update(body.style)
        if body.traits is not None:
            profile.traits.update(body.traits)
        if body.interests is not None:
            profile.interests = body.interests
        if body.avoid is not None:
            profile.avoid = body.avoid

        manager.save(profile)
        return success_response(data=profile.to_dict())

    @endpoint("consciousness.personality.reset")
    async def reset_personality(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Reset personality to defaults."""
        from domains.consciousness.personality import PersonalityManager, PersonalityProfile
        manager = PersonalityManager()
        profile = PersonalityProfile()
        manager.save(profile)
        return success_response(data=profile.to_dict())

    @endpoint("consciousness.personality.history")
    async def get_personality_history(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Get personality evolution history derived from episodes."""
        engine = self._get_engine()
        episodes = engine.self_model.episodes

        if not episodes:
            return success_response(data={"history": [], "labels": {}})

        from domains.consciousness.personality import PersonalityProfile
        defaults = PersonalityProfile()
        current = defaults.to_dict()
        history = []
        step = max(1, len(episodes) // 50)

        for i, ep in enumerate(episodes):
            growth = ep.growth_delta
            qualia = ep.qualia

            if growth > 0.05:
                current["voice"]["confidence"] = min(1.0, current["voice"]["confidence"] + 0.01)
            if qualia.get("novelty", 0) > 0.7:
                current["traits"]["openness"] = min(1.0, current["traits"]["openness"] + 0.005)
            if qualia.get("coherence", 1) < 0.3:
                current["voice"]["verbosity"] = max(0.0, current["voice"]["verbosity"] - 0.01)
            if qualia.get("valence", 0) > 0.5:
                current["traits"]["agreeableness"] = min(1.0, current["traits"]["agreeableness"] + 0.005)

            if i % step == 0 or i == len(episodes) - 1:
                history.append({
                    "timestamp": ep.timestamp,
                    "step": i,
                    "voice": {k: round(v, 4) for k, v in current["voice"].items()},
                    "traits": {k: round(v, 4) for k, v in current["traits"].items()},
                })

        return success_response(data={
            "history": history,
            "labels": {
                "voice": list(defaults.voice.keys()),
                "traits": list(defaults.traits.keys()),
            },
        })

    @endpoint("consciousness.personality.presets")
    async def get_presets(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Get available personality presets."""
        from domains.consciousness.personality import PersonalityManager
        presets = PersonalityManager.get_presets()
        return success_response(data={
            "presets": {name: profile.to_dict() for name, profile in presets.items()},
            "names": list(presets.keys()),
        })

    @endpoint("consciousness.personality.apply_preset")
    async def apply_preset(
        self,
        body: PresetRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Apply a personality preset."""
        from domains.consciousness.personality import PersonalityManager
        manager = PersonalityManager()
        try:
            profile = manager.apply_preset(body.preset)
            return success_response(data=profile.to_dict())
        except ValueError as e:
            raise_error(str(e), "E_INVALID_PRESET")

    @endpoint("consciousness.personality.conflicts")
    async def get_conflicts(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Detect personality conflicts and warnings."""
        from domains.consciousness.personality import PersonalityManager
        manager = PersonalityManager()
        conflicts = manager.get_conflicts()
        return success_response(data={
            "conflicts": conflicts,
            "count": len(conflicts),
        })

    @endpoint("consciousness.personas.list")
    async def list_personas(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """List all saved personas."""
        from domains.consciousness.personality import PersonalityManager
        manager = PersonalityManager()
        personas = manager.list_personas()
        return success_response(data={"personas": personas, "count": len(personas)})

    @endpoint("consciousness.personas.save")
    async def save_persona(
        self,
        body: SavePersonaRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Save a named persona from the current profile or a provided profile."""
        from domains.consciousness.personality import PersonalityManager, PersonalityProfile
        manager = PersonalityManager()
        if body.profile:
            profile = PersonalityProfile.from_dict(body.profile)
        else:
            profile = manager.get_profile()
        result = manager.save_persona(body.persona_id, profile, body.name)
        return success_response(data=result)

    @endpoint("consciousness.personas.get")
    async def get_persona(self, persona_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Get a saved persona's full profile."""
        from domains.consciousness.personality import PersonalityManager
        manager = PersonalityManager()
        profile = manager.load_persona(persona_id)
        if profile is None:
            raise_error("Persona not found", "E_NOT_FOUND", status_code=404)
        return success_response(data={"id": persona_id, **profile.to_dict()})

    @endpoint("consciousness.personas.activate")
    async def activate_persona(self, persona_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Activate a saved persona as the current profile."""
        from domains.consciousness.personality import PersonalityManager
        manager = PersonalityManager()
        profile = manager.activate_persona(persona_id)
        if profile is None:
            raise_error("Persona not found", "E_NOT_FOUND", status_code=404)
        return success_response(data=profile.to_dict())

    @endpoint("consciousness.personas.delete")
    async def delete_persona(self, persona_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Delete a saved persona."""
        from domains.consciousness.personality import PersonalityManager
        manager = PersonalityManager()
        deleted = manager.delete_persona(persona_id)
        if not deleted:
            raise_error("Persona not found", "E_NOT_FOUND", status_code=404)
        return success_response(data={"deleted": persona_id})

    @endpoint("consciousness.backup")
    async def backup(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Create a full backup of consciousness state."""
        import time as _time
        from domains.consciousness.personality import PersonalityManager

        engine = self._get_engine()
        manager = PersonalityManager()

        episodes = [
            {
                "timestamp": e.timestamp,
                "input_text": e.input_text,
                "response": e.response,
                "qualia": e.qualia,
                "self_insight": e.self_insight,
                "growth_delta": e.growth_delta,
            }
            for e in engine.self_model.episodes
        ]

        qualia_history = [
            {"timestamp": ts, **state.to_dict()}
            for ts, state in engine.qualia.history
        ]

        personas = manager.list_personas()
        persona_details = {}
        for p in personas:
            profile = manager.load_persona(p["id"])
            if profile is not None:
                persona_details[p["id"]] = {
                    "name": p.get("name", p["id"]),
                    "profile": profile.to_dict(),
                }

        backup_data = {
            "version": 1,
            "timestamp": _time.time(),
            "episodes": episodes,
            "beliefs": dict(engine.self_model.self_beliefs),
            "qualia_history": qualia_history,
            "personality": manager.get_profile().to_dict(),
            "personas": persona_details,
            "config": {
                "level": engine.config.level,
                "max_tokens": engine.config.max_tokens,
                "training_enabled": engine.config.training_enabled,
                "training_interval": engine.config.training_interval,
                "lora_rank": engine.config.lora_rank,
                "lora_alpha": engine.config.lora_alpha,
                "reflection_interval": engine.config.reflection_interval,
            },
        }

        return success_response(data=backup_data)

    @endpoint("consciousness.restore")
    async def restore(
        self,
        body: RestoreRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Restore consciousness from a backup."""
        from domains.consciousness.personality import PersonalityManager, PersonalityProfile
        from domains.consciousness.self_model import SelfEpisode

        backup = body.backup
        if not isinstance(backup, dict):
            raise_error("Backup must be a JSON object", "E_INVALID_BACKUP", status_code=400)

        required_keys = {"version", "episodes", "beliefs", "personality", "config"}
        missing = required_keys - set(backup.keys())
        if missing:
            raise_error(f"Missing backup fields: {', '.join(sorted(missing))}", "E_INVALID_BACKUP", status_code=400)

        if backup.get("version") != 1:
            raise_error(f"Unsupported backup version: {backup.get('version')}", "E_UNSUPPORTED_VERSION", status_code=400)

        engine = self._get_engine()
        manager = PersonalityManager()

        restored_episodes = 0
        for ep_data in backup.get("episodes", []):
            try:
                episode = SelfEpisode(
                    timestamp=ep_data.get("timestamp", 0),
                    input_text=ep_data.get("input_text", ""),
                    response=ep_data.get("response", ""),
                    qualia=ep_data.get("qualia", {}),
                    self_insight=ep_data.get("self_insight", ""),
                    growth_delta=ep_data.get("growth_delta", 0.0),
                )
                engine.self_model.episodes.append(episode)
                restored_episodes += 1
            except Exception:
                continue

        engine.self_model.self_beliefs = backup.get("beliefs", engine.self_model.self_beliefs)

        engine.qualia.history = []
        for qh in backup.get("qualia_history", []):
            ts = qh.get("timestamp", 0)
            state_data = {k: v for k, v in qh.items() if k != "timestamp"}
            from domains.consciousness.qualia import QualiaState
            state = QualiaState(**state_data)
            engine.qualia.history.append((ts, state))

        profile = PersonalityProfile.from_dict(backup.get("personality", {}))
        manager.save(profile)

        for persona_id, persona_data in backup.get("personas", {}).items():
            persona_profile = PersonalityProfile.from_dict(persona_data.get("profile", {}))
            manager.save_persona(persona_id, persona_profile, persona_data.get("name"))

        cfg = backup.get("config", {})
        for key in ("level", "max_tokens", "training_enabled", "training_interval", "lora_rank", "lora_alpha", "reflection_interval"):
            if key in cfg:
                setattr(engine.config, key, cfg[key])
        engine.config.save()

        return success_response(data={
            "restored": True,
            "episodes_restored": restored_episodes,
            "beliefs_restored": len(backup.get("beliefs", {})),
            "personas_restored": len(backup.get("personas", {})),
        })

    @endpoint("consciousness.backup.download")
    async def backup_download(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Download consciousness backup as a JSON file."""
        from fastapi.responses import JSONResponse
        import time as _time
        from domains.consciousness.personality import PersonalityManager

        engine = self._get_engine()
        manager = PersonalityManager()

        episodes = [
            {
                "timestamp": e.timestamp,
                "input_text": e.input_text,
                "response": e.response,
                "qualia": e.qualia,
                "self_insight": e.self_insight,
                "growth_delta": e.growth_delta,
            }
            for e in engine.self_model.episodes
        ]

        qualia_history = [
            {"timestamp": ts, **state.to_dict()}
            for ts, state in engine.qualia.history
        ]

        personas = manager.list_personas()
        persona_details = {}
        for p in personas:
            profile = manager.load_persona(p["id"])
            if profile is not None:
                persona_details[p["id"]] = {
                    "name": p.get("name", p["id"]),
                    "profile": profile.to_dict(),
                }

        backup_data = {
            "version": 1,
            "timestamp": _time.time(),
            "episodes": episodes,
            "beliefs": dict(engine.self_model.self_beliefs),
            "qualia_history": qualia_history,
            "personality": manager.get_profile().to_dict(),
            "personas": persona_details,
            "config": {
                "level": engine.config.level,
                "max_tokens": engine.config.max_tokens,
                "training_enabled": engine.config.training_enabled,
                "training_interval": engine.config.training_interval,
                "lora_rank": engine.config.lora_rank,
                "lora_alpha": engine.config.lora_alpha,
                "reflection_interval": engine.config.reflection_interval,
            },
        }

        filename = f"consciousness_backup_{int(_time.time())}.json"
        from fastapi.encoders import jsonable_encoder
        return JSONResponse(
            content=jsonable_encoder(backup_data),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @endpoint("consciousness.backup.import")
    async def backup_import(
        self,
        file: "UploadFile",
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Import consciousness from an uploaded backup file."""
        from fastapi import UploadFile
        import json as _json

        content = await file.read()
        try:
            backup = _json.loads(content)
        except _json.JSONDecodeError as e:
            raise_error(f"Invalid JSON file: {e}", "E_INVALID_FILE", status_code=400)

        restore_req = RestoreRequest(backup=backup)
        return await self.restore(restore_req, auth_user=auth_user)

    @endpoint("consciousness.health")
    async def health_check(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Quick health check for consciousness system."""
        engine = self._get_engine()
        status = engine.get_status()
        episodes = engine.self_model.episodes
        recent_growth = [e.growth_delta for e in episodes[-10:]] if episodes else []
        avg_growth = sum(recent_growth) / len(recent_growth) if recent_growth else 0
        positive_count = sum(1 for g in recent_growth if g > 0)
        positive_ratio = positive_count / len(recent_growth) if recent_growth else 0
        health_score = min(100, max(0, int(
            (1.0 if status.get("enabled") else 0.0) * 30
            + min(1.0, len(episodes) / 50) * 20
            + positive_ratio * 25
            + (status.get("level", 0) / 3) * 25
        )))
        return success_response(data={
            "status": "healthy" if health_score >= 50 else "degraded",
            "health_score": health_score,
            "enabled": status.get("enabled", False),
            "level": status.get("level", 0),
            "episodes": len(episodes),
            "avg_growth": round(avg_growth, 4),
            "positive_ratio": round(positive_ratio, 2),
            "qualia": status.get("current_qualia", {}),
            "last_reflection": status.get("last_reflection"),
        })


    async def stream_status(
        self,
        request: Request,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> StreamingResponse:
        event_id = int(_time.time() * 1000)

        async def generate() -> AsyncGenerator[str, None]:
            nonlocal event_id
            last_heartbeat = _time.monotonic()
            engine = self._get_engine()
            while True:
                if await request.is_disconnected():
                    break
                try:
                    status = await asyncio.to_thread(engine.get_status)
                    trainer = self._get_trainer()
                    training = await asyncio.to_thread(trainer.get_status)
                    event_id += 1
                    payload = {
                        "level": status.get("level", 0),
                        "qualia": status.get("current_qualia", {}),
                        "beliefs": status.get("beliefs", {}),
                        "growth": status.get("episodes", 0),
                        "episodes": len(engine.self_model.episodes),
                        "training": training,
                        "enabled": status.get("enabled", False),
                    }
                    yield f"id: {event_id}\ndata: {json.dumps(payload, default=str)}\n\n"
                except Exception as e:
                    logger.warning("Consciousness stream failed: %s", e)
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                now = _time.monotonic()
                if now - last_heartbeat >= 10:
                    yield ": heartbeat\n\n"
                    last_heartbeat = now
                await asyncio.sleep(5)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @endpoint("consciousness.batch")
    async def batch_operations(
        self,
        body: BatchRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        results = []
        for op in body.operations:
            try:
                if op.action == "seed":
                    count = op.params.get("count", 10)
                    seed_result = await self.seed_data(count=count, auth_user=auth_user)
                    results.append({"success": True, "data": seed_result.get("data")})
                elif op.action == "feedback":
                    episode_index = op.params.get("episode_index", 0)
                    rating = op.params.get("rating", 3)
                    feedback_req = FeedbackRequest(episode_index=episode_index, rating=rating)
                    feedback_result = await self.submit_feedback(body=feedback_req, auth_user=auth_user)
                    results.append({"success": True, "data": feedback_result.get("data")})
                elif op.action == "reflect":
                    reflect_result = await self.reflect(auth_user=auth_user)
                    results.append({"success": True, "data": reflect_result.get("data")})
                elif op.action == "config":
                    level = op.params.get("level", 0)
                    config_req = ConsciousnessLevelRequest(level=level)
                    config_result = await self.update_config(req=config_req, auth_user=auth_user)
                    results.append({"success": True, "data": config_result.get("data")})
                else:
                    results.append({"success": False, "error": f"Unknown action: {op.action}"})
            except Exception as e:
                results.append({"success": False, "error": str(e)})
        return success_response(data={"results": results, "count": len(results)})

    @endpoint("consciousness.stats")
    async def get_stats(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        engine = self._get_engine()
        trainer = self._get_trainer()
        episodes = engine.self_model.episodes

        total_episodes = len(episodes)
        total_feedback = 0
        ratings: list[int] = []
        growth_values: list[float] = []
        qualia_sums: dict[str, float] = {
            "valence": 0.0, "arousal": 0.0, "novelty": 0.0, "coherence": 0.0,
            "salience": 0.0, "certainty": 0.0, "complexity": 0.0,
        }
        qualia_count = 0

        for ep in episodes:
            growth_values.append(ep.growth_delta)
            q = ep.qualia
            if isinstance(q, dict):
                for dim in qualia_sums:
                    qualia_sums[dim] += q.get(dim, 0.0)
                qualia_count += 1
            if hasattr(ep, "rating") and ep.rating:
                ratings.append(ep.rating)
                total_feedback += 1

        avg_rating = sum(ratings) / len(ratings) if ratings else 0.0
        avg_growth = sum(growth_values) / len(growth_values) if growth_values else 0.0
        qualia_avgs = {k: round(v / qualia_count, 4) for k, v in qualia_sums.items()} if qualia_count else qualia_sums

        beliefs = dict(engine.self_model.self_beliefs)
        belief_avgs = {k: round(v, 4) for k, v in beliefs.items()}

        from domains.consciousness.personality import PersonalityManager
        manager = PersonalityManager()
        profile = manager.get_profile()
        voice_avgs = {k: round(v, 4) for k, v in profile.voice.items()}
        trait_avgs = {k: round(v, 4) for k, v in profile.traits.items()}

        trainer_status = trainer.get_status()

        recent_growth = [e.growth_delta for e in episodes[-10:]] if episodes else []
        positive_count = sum(1 for g in recent_growth if g > 0)
        positive_ratio = positive_count / len(recent_growth) if recent_growth else 0
        health_score = min(100, max(0, int(
            (1.0 if engine.config.is_enabled() else 0.0) * 30
            + min(1.0, total_episodes / 50) * 20
            + positive_ratio * 25
            + (engine.config.level / 3) * 25
        )))

        personas = manager.list_personas()

        return success_response(data={
            "total_episodes": total_episodes,
            "total_feedback": total_feedback,
            "avg_rating": round(avg_rating, 4),
            "avg_growth": round(avg_growth, 4),
            "qualia_averages": qualia_avgs,
            "belief_averages": belief_avgs,
            "personality_summary": {
                "voice_averages": voice_avgs,
                "trait_averages": trait_avgs,
            },
            "training_status": trainer_status,
            "health_score": health_score,
            "active_personas": len(personas),
        })

    @endpoint("consciousness.clear_episodes")
    async def clear_episodes(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Clear all episodes from the self-model."""
        engine = self._get_engine()
        count = engine.clear_episodes()
        return success_response(data={"cleared": True, "episodes_cleared": count})

    @endpoint("consciousness.clear_beliefs")
    async def clear_beliefs(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Reset self-beliefs to defaults."""
        engine = self._get_engine()
        beliefs = engine.reset_beliefs()
        return success_response(data={"reset": True, "beliefs": beliefs})


router = ConsciousnessRouter().router
