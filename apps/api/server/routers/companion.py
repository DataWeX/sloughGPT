"""
Companion Router - AI Companion endpoints

Endpoints to manage and chat with the AI companion.
"""
import logging
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List

from schemas.common import success_response

logger = logging.getLogger("slo.routers.companion")


class SetPersonalityRequest(BaseModel):
    """Set companion personality."""
    name: str = "Friend"
    warmth: float = 0.7
    curiosity: float = 0.6
    creativity: float = 0.5
    confidence: float = 0.5
    humor: float = 0.4


class PresetRequest(BaseModel):
    """Use a preset personality."""
    name: str = "Friend"
    preset: str = "warm"  # warm, curious, playful, balanced


class ChatRequest(BaseModel):
    """Chat with companion."""
    message: str
    user_name: Optional[str] = None
    user_mood: Optional[str] = None
    include_system_prompt: bool = True


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
        self.router.add_api_route("/personality", self.set_personality, methods=["POST"])
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

    async def get_companion_info(self):
        """Get companion info."""
        comp = self._get_companion()
        return success_response(data=comp.to_dict())

    async def set_personality(self, req: SetPersonalityRequest):
        """Set companion personality."""
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

    async def use_preset(self, req: PresetRequest):
        """Use preset personality."""
        from domains.companion import create_companion

        self._companion = create_companion(name=req.name, personality=req.preset)

        return success_response(data={
            "status": "ok",
            "preset": req.preset,
            "traits": self._companion.to_dict()["traits"],
        })

    async def get_prompt(self):
        """Get current system prompt."""
        comp = self._get_companion()
        return success_response(data={"system_prompt": comp.get_system_prompt()})

    async def chat(self, req: ChatRequest):
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
                    max_tokens=256,
                    temperature=0.7,
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

    async def list_presets(self):
        """List available presets."""
        return success_response(data={
            "presets": [
                {"id": "warm", "name": "Warm Friend", "description": "Caring and supportive"},
                {"id": "curious", "name": "Curious Friend", "description": "Interested in everything"},
                {"id": "playful", "name": "Playful Friend", "description": "Fun and humorous"},
                {"id": "balanced", "name": "Balanced Friend", "description": "Well-rounded"},
            ]
        })


_companion_router = CompanionRouter()
router = _companion_router.router


def _get_companion():
    return _companion_router._get_companion()


__all__ = ["router"]
