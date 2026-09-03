"""
Companion Router - AI Companion endpoints

Uses MogDB as the storage engine with automatic JSON sync.
Presets are stored in MogDB and synced to JSON for human readability.
"""
import logging
import os
import time as _time
from pathlib import Path
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import Optional, List

from schemas.common import success_response, raise_error, classify_and_raise, safe_audit_log
from infrastructure.auth import require_auth_if_enabled

logger = logging.getLogger("slo.routers.companion")


def _get_db():
    from mogdb import MogDB
    repo_root = Path(__file__).resolve().parents[4]
    db_path = os.path.join(repo_root, "data", "companion_mogdb")
    sync_path = os.path.join(repo_root, "data", "companion_json")
    return MogDB(db_path, sync_dir=sync_path)


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


class PresetInfo(BaseModel):
    id: str
    name: str
    description: str
    traits: dict
    system_prompt: str = ""


class PresetCreateRequest(BaseModel):
    id: str = Field(..., max_length=50, pattern=r"^[a-z0-9_-]+$")
    name: str = Field(..., max_length=100)
    description: str = Field(default="", max_length=500)
    traits: dict = Field(default_factory=dict)
    system_prompt: str = Field(default="", max_length=2000)


def _load_presets() -> List[dict]:
    """Load presets from MogDB."""
    db = _get_db()
    col = db.collection("presets")
    return col.find()


def _seed_default_presets() -> None:
    """Seed default presets if collection is empty."""
    db = _get_db()
    col = db.collection("presets")
    if col.count() > 0:
        return
    defaults = [
        {"id": "warm", "name": "Warm Friend", "description": "Caring and supportive", "traits": {"warmth": 0.9, "curiosity": 0.6, "humor": 0.3}, "system_prompt": "You are a warm, caring friend."},
        {"id": "curious", "name": "Curious Friend", "description": "Interested in everything", "traits": {"warmth": 0.6, "curiosity": 0.9, "humor": 0.3}, "system_prompt": "You are a deeply curious friend."},
        {"id": "playful", "name": "Playful Friend", "description": "Fun and humorous", "traits": {"warmth": 0.7, "curiosity": 0.5, "humor": 0.8}, "system_prompt": "You are a playful, fun-loving friend."},
        {"id": "balanced", "name": "Balanced Friend", "description": "Well-rounded", "traits": {"warmth": 0.7, "curiosity": 0.6, "humor": 0.5}, "system_prompt": "You are a balanced, well-rounded friend."},
    ]
    for p in defaults:
        col.insert_one(p)


class ChatRequest(BaseModel):
    """Chat with companion."""
    message: str = Field(..., min_length=1, max_length=10000)
    user_name: Optional[str] = Field(default=None, max_length=100)
    user_mood: Optional[str] = Field(default=None, max_length=100)
    include_system_prompt: bool = True
    max_tokens: int = Field(default=256, ge=1, le=4096, description="Max tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")


class ChatResponse(BaseModel):
    """Companion response."""
    response: str
    system_prompt: str
    elapsed_ms: float = 0.0


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
        self.router.add_api_route("/presets", self.create_preset, methods=["POST"])
        self.router.add_api_route("/presets/{preset_id}", self.delete_preset, methods=["DELETE"])

    def _get_companion(self):
        """Get or create companion."""
        if self._companion is None:
            from domains.companion import get_companion
            self._companion = get_companion()
        return self._companion

    async def get_companion_info(self) -> dict:
        """Return the current companion's full state as a dictionary."""
        try:
            companion = self._get_companion()
            return success_response(data=companion.to_dict())
        except Exception as e:
            classify_and_raise(e, source="companion.get_info")

    async def reset_companion(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Reset companion to default state."""
        try:
            self._companion = None
            from domains.companion import reset_companion
            reset_companion()
            safe_audit_log("companion.reset")
            return success_response(data={"reset": True})
        except Exception as e:
            classify_and_raise(e, source="companion.reset")

    async def set_personality(self, req: SetPersonalityRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Set companion personality (full replacement)."""
        try:
            companion = self._get_companion()
            companion.name = req.name
            companion.warmth = req.warmth
            companion.curiosity = req.curiosity
            companion.creativity = req.creativity
            companion.confidence = req.confidence
            companion.humor = req.humor
            safe_audit_log("companion.personality.set", detail=f"name={req.name}")
            return success_response(data=companion.to_dict())
        except Exception as e:
            classify_and_raise(e, source="companion.set_personality")

    async def patch_personality(self, req: PatchPersonalityRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Partial update to companion personality."""
        try:
            companion = self._get_companion()
            if req.name is not None:
                companion.name = req.name
            if req.warmth is not None:
                companion.warmth = req.warmth
            if req.curiosity is not None:
                companion.curiosity = req.curiosity
            if req.creativity is not None:
                companion.creativity = req.creativity
            if req.confidence is not None:
                companion.confidence = req.confidence
            if req.humor is not None:
                companion.humor = req.humor
            safe_audit_log("companion.personality.patch")
            return success_response(data=companion.to_dict())
        except Exception as e:
            classify_and_raise(e, source="companion.patch_personality")

    async def use_preset(self, preset_id: str = Field(...), auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Apply a preset personality."""
        try:
            db = _get_db()
            col = db.collection("presets")
            preset = col.find_one({"id": preset_id})
            if not preset:
                raise_error(f"Preset '{preset_id}' not found", "E_NOT_FOUND", status_code=404)
            companion = self._get_companion()
            traits = preset.get("traits", {})
            for k, v in traits.items():
                if hasattr(companion, k):
                    setattr(companion, k, v)
            safe_audit_log("companion.preset.use", resource=preset_id)
            return success_response(data=companion.to_dict())
        except Exception as e:
            classify_and_raise(e, source="companion.use_preset")

    async def get_prompt(self) -> dict:
        """Get the current system prompt."""
        try:
            companion = self._get_companion()
            return success_response(data={"system_prompt": companion.build_system_prompt()})
        except Exception as e:
            classify_and_raise(e, source="companion.get_prompt")

    async def chat(self, req: ChatRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> ChatResponse:
        """Chat with the companion."""
        try:
            companion = self._get_companion()
            system_prompt = companion.build_system_prompt() if req.include_system_prompt else ""
            _chat_start = _time.monotonic()
            response_text = await companion.generate(
                user_message=req.message,
                system_prompt=system_prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
            )
            _chat_elapsed_ms = (_time.monotonic() - _chat_start) * 1000
            safe_audit_log("companion.chat", detail=f"elapsed={_chat_elapsed_ms:.0f}ms tokens={len(response_text.split())}")

            return ChatResponse(
                response=response_text,
                system_prompt=system_prompt,
                elapsed_ms=round(_chat_elapsed_ms, 1),
            )
        except Exception as e:
            classify_and_raise(e, source="companion.chat")

    async def list_presets(self) -> dict:
        """Return the list of available companion presets."""
        try:
            _seed_default_presets()
            presets = _load_presets()
            return success_response(data={"presets": presets})
        except Exception as e:
            classify_and_raise(e, source="companion.presets")

    async def create_preset(self, req: PresetCreateRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Create a new companion preset."""
        try:
            db = _get_db()
            col = db.collection("presets")
            existing = col.find_one({"id": req.id})
            if existing:
                raise_error(f"Preset '{req.id}' already exists", "E_CONFLICT", status_code=409)
            preset = {
                "id": req.id,
                "name": req.name,
                "description": req.description,
                "traits": req.traits,
                "system_prompt": req.system_prompt,
            }
            col.insert_one(preset)
            safe_audit_log("companion.preset.create", resource=req.id)
            return success_response(data={"preset": preset})
        except Exception as e:
            classify_and_raise(e, source="companion.create_preset")

    async def delete_preset(self, preset_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Delete a companion preset."""
        try:
            db = _get_db()
            col = db.collection("presets")
            deleted = col.delete_one({"id": preset_id})
            if not deleted:
                raise_error(f"Preset '{preset_id}' not found", "E_NOT_FOUND", status_code=404)
            safe_audit_log("companion.preset.delete", resource=preset_id)
            return success_response(data={"deleted": preset_id})
        except Exception as e:
            classify_and_raise(e, source="companion.delete_preset")


_companion_router = CompanionRouter()
router = _companion_router.router


def _get_companion():
    return _companion_router._get_companion()


__all__ = ["router"]
