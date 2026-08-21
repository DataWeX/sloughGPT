"""
Companion Router - AI Companion endpoints

Endpoints to manage and chat with the AI companion.
"""
import logging
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, List

from schemas.common import success_response, safe_audit_log

logger = logging.getLogger("slo.routers.companion")


class SetPersonalityRequest(BaseModel):
    """Set companion personality (full replacement)."""
    name: str = Field(default="Friend", max_length=100)
    warmth: float = Field(default=0.7, ge=0.0, le=1.0)
    curiosity: float = Field(default=0.6, ge=0.0, le=1.0)
    creativity: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    humor: float = Field(default=0.4, ge=0.0, le=1.0)


class PatchPersonalityRequest(BaseModel):
    """Partial update to companion personality (only provided fields are updated)."""
    name: Optional[str] = Field(default=None, max_length=100)
    warmth: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    curiosity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    creativity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    humor: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class PresetRequest(BaseModel):
    """Use a preset personality."""
    name: str = Field(default="Friend", max_length=100)
    preset: str = Field(default="warm", max_length=50)


class ChatRequest(BaseModel):
    """Chat with companion."""
    message: str = Field(max_length=10000)
    user_name: Optional[str] = Field(default=None, max_length=100)
    user_mood: Optional[str] = Field(default=None, max_length=100)
    include_system_prompt: bool = True
    max_tokens: int = Field(default=256, ge=1, le=4096, description="Max tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")


class ChatResponse(BaseModel):
    """Companion response."""
    response: str
    system_prompt: str


class CompanionRouter:
    """Router for AI companion management and chat."""

    def __init__(self):
        self.router = APIRouter(prefix="/companion", tags=["companion"])
        self._companion = None
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/", self.get_companion_info, methods=["GET"])
        self.router.add_api_route("/", self.reset_companion, methods=["DELETE"])
        self.router.add_api_route("/personality", self.set_personality, methods=["POST"])
        self.router.add_api_route("/personality", self.patch_personality, methods=["PATCH"])
        self.router.add_api_route("/preset", self.use_preset, methods=["POST"])
        self.router.add_api_route("/prompt", self.get_prompt, methods=["GET"])
        self.router.add_api_route("/chat", self.chat, methods=["POST"], response_model=ChatResponse)
        self.router.add_api_route("/presets", self.list_presets, methods=["GET"])

    def _get_companion(self):
        """Get or create companion."""
        if self._companion is None:
            from domains.companion import get_companion
            self._companion = get_companion()
        return self._companion

    async def get_companion_info(self) -> dict:
        """Return the current companion's full state as a dictionary.

        Returns:
            Success envelope containing the companion's traits (name,
            warmth, curiosity, creativity, confidence, humor) and
            other configuration.

        Side effects:
            Lazily instantiates the CompanionSystem singleton on first
            call via get_companion().
        """
        comp = self._get_companion()
        return success_response(data=comp.to_dict())

    async def set_personality(self, req: SetPersonalityRequest) -> dict:
        """Set companion personality (full replacement)."""
        comp = self._get_companion()
        comp.set_personality(
            name=req.name,
            warmth=req.warmth,
            curiosity=req.curiosity,
            creativity=req.creativity,
            confidence=req.confidence,
            humor=req.humor,
        )
        return success_response(data={"status": "ok", "traits": comp.to_dict()["traits"]})

    async def patch_personality(self, req: PatchPersonalityRequest) -> dict:
        """Partial update to companion personality (only provided fields are changed)."""
        comp = self._get_companion()
        current = comp.to_dict()["traits"]
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        merged = {**current, **updates}
        comp.set_personality(
            name=merged.get("name", current.get("name", "Friend")),
            warmth=merged.get("warmth", current.get("warmth", 0.7)),
            curiosity=merged.get("curiosity", current.get("curiosity", 0.6)),
            creativity=merged.get("creativity", current.get("creativity", 0.5)),
            confidence=merged.get("confidence", current.get("confidence", 0.5)),
            humor=merged.get("humor", current.get("humor", 0.4)),
        )
        return success_response(data={"status": "ok", "traits": comp.to_dict()["traits"]})

    async def reset_companion(self) -> dict:
        """Reset companion to default personality."""
        from domains.companion import create_companion
        self._companion = create_companion()
        safe_audit_log("companion.reset")
        return success_response(data={"status": "ok", "traits": self._companion.to_dict()["traits"]})

    async def use_preset(self, req: PresetRequest) -> dict:
        """Replace the current companion with a preset personality.

        Args:
            req: PresetRequest with name (used as the companion's display
                name) and preset (one of: warm, curious, playful, balanced).

        Returns:
            Success envelope containing the preset ID and the new traits
            dictionary.

        Side effects:
            Replaces the internal CompanionSystem instance with a new
            one configured to the chosen preset.
        """
        from domains.companion import create_companion

        self._companion = create_companion(name=req.name, personality=req.preset)

        return success_response(data={
            "status": "ok",
            "preset": req.preset,
            "traits": self._companion.to_dict()["traits"],
        })

    async def get_prompt(self) -> dict:
        """Return the system prompt currently used by the companion.

        Returns:
            Success envelope with a system_prompt string containing the
            full system prompt derived from the companion's personality
            traits and configuration.

        Side effects:
            Lazily instantiates the CompanionSystem singleton on first
            call via get_companion().
        """
        comp = self._get_companion()
        return success_response(data={"system_prompt": comp.get_system_prompt()})

    async def chat(self, req: ChatRequest) -> dict:
        """Chat with companion — generates a response using the active model."""
        comp = self._get_companion()

        # Adjust for mood
        if req.user_mood:
            comp.adjust_for_mood(req.user_mood)

        # Get system prompt
        system_prompt = comp.get_system_prompt() if req.include_system_prompt else ""

        # Build messages for the provider
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": req.message})

        response_text = ""
        error_msg = None
        try:
            from domains.models.provider import get_provider
            provider = get_provider("default")
            if provider is not None:
                response_text = await provider.chat(
                    messages,
                    max_tokens=req.max_tokens,
                    temperature=req.temperature,
                )
            else:
                error_msg = "No model loaded"
        except Exception as e:
            error_msg = str(e)
            logger.warning("Companion chat failed: %s", e, extra={"tag": "MODEL", "context": {"error": str(e)}})

        if not response_text and error_msg:
            response_text = f"[Error: {error_msg}]"

        return ChatResponse(
            response=response_text,
            system_prompt=system_prompt,
        )

    async def list_presets(self) -> dict:
        """Return the hardcoded list of available companion presets.

        Returns:
            Success envelope containing a presets array. Each preset
            has id (warm/curious/playful/balanced), name, description,
            and a traits dictionary with warmth, curiosity, humor values.

        Side effects:
            None. The preset list is static and does not read from
            any external store.
        """
        presets = [
            {"id": "warm", "name": "Warm Friend", "description": "Caring and supportive", "traits": {"warmth": 0.9, "curiosity": 0.6, "humor": 0.3}},
            {"id": "curious", "name": "Curious Friend", "description": "Interested in everything", "traits": {"warmth": 0.6, "curiosity": 0.9, "humor": 0.3}},
            {"id": "playful", "name": "Playful Friend", "description": "Fun and humorous", "traits": {"warmth": 0.7, "curiosity": 0.5, "humor": 0.8}},
            {"id": "balanced", "name": "Balanced Friend", "description": "Well-rounded", "traits": {"warmth": 0.7, "curiosity": 0.6, "humor": 0.5}},
        ]
        return success_response(data={"presets": presets})


_companion_router = CompanionRouter()
router = _companion_router.router


def _get_companion():
    return _companion_router._get_companion()


__all__ = ["router"]
