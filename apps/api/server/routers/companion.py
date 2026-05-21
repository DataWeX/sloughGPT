"""
Companion Router - AI Companion endpoints

Endpoints to manage and chat with the AI companion.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter(prefix="/companion", tags=["companion"])

# Global companion instance
_companion = None


def _get_companion():
    """Get or create companion."""
    global _companion
    if _companion is None:
        from domains.companion import get_companion
        _companion = get_companion()
    return _companion


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


@router.get("/")
async def get_companion_info():
    """Get companion info."""
    comp = _get_companion()
    return comp.to_dict()


@router.post("/personality")
async def set_personality(req: SetPersonalityRequest):
    """Set companion personality."""
    comp = _get_companion()
    comp.set_personality(
        name=req.name,
        warmth=req.warmth,
        curiosity=req.curiosity,
        creativity=req.creativity,
        confidence=req.confidence,
        humor=req.humor,
    )
    return {"status": "ok", "traits": comp.to_dict()["traits"]}


@router.post("/preset")
async def use_preset(req: PresetRequest):
    """Use preset personality."""
    from domains.companion import create_companion
    
    global _companion
    _companion = create_companion(name=req.name, personality=req.preset)
    
    return {
        "status": "ok", 
        "preset": req.preset,
        "traits": _companion.to_dict()["traits"],
    }


@router.get("/prompt")
async def get_prompt():
    """Get current system prompt."""
    comp = _get_companion()
    return {"system_prompt": comp.get_system_prompt()}


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Chat with companion."""
    comp = _get_companion()
    
    # Adjust for mood
    if req.user_mood:
        comp.adjust_for_mood(req.user_mood)
    
    # Get system prompt
    system_prompt = comp.get_system_prompt() if req.include_system_prompt else ""
    
    return ChatResponse(
        response="",  # Model will generate this
        system_prompt=system_prompt,
    )


@router.get("/presets")
async def list_presets():
    """List available presets."""
    return {
        "presets": [
            {"id": "warm", "name": "Warm Friend", "description": "Caring and supportive"},
            {"id": "curious", "name": "Curious Friend", "description": "Interested in everything"},
            {"id": "playful", "name": "Playful Friend", "description": "Fun and humorous"},
            {"id": "balanced", "name": "Balanced Friend", "description": "Well-rounded"},
        ]
    }


__all__ = ["router"]