"""
Settings Router — persistent user configuration that survives restarts.

Replaces the in-memory config controller with a persistent backend.
"""

import logging

from fastapi import APIRouter, Depends
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel, Field
from schemas.common import classify_and_raise, endpoint, safe_audit_log, success_response

logger = logging.getLogger("slo.routers.settings")


# ── Schemas ────────────────────────────────────────────────────────


class GenerationSettingsUpdate(BaseModel):
    temperature: float | None = Field(default=None, ge=0.1, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1, le=200)
    repetition_penalty: float | None = Field(default=None, ge=1.0, le=2.0)
    max_new_tokens: int | None = Field(default=None, ge=1, le=4096)
    max_context_length: int | None = Field(default=None, ge=64, le=8192)


class TrainingSettingsUpdate(BaseModel):
    preferred_model: str | None = None
    auto_train: bool | None = None
    auto_train_threshold: int | None = Field(default=None, ge=10, le=10000)
    preferred_method: str | None = None
    max_checkpoints: int | None = Field(default=None, ge=1, le=50)
    enable_tracking: bool | None = None


class AdaptiveSettingsUpdate(BaseModel):
    enabled: bool | None = None
    exploration_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    learning_enabled: bool | None = None


class VoiceSettingsUpdate(BaseModel):
    noise_gate_db: float | None = Field(default=None, ge=-80.0, le=0.0)
    target_level_db: float | None = Field(default=None, ge=-60.0, le=0.0)
    vad_enabled: bool | None = None
    vad_min_speech_ms: int | None = Field(default=None, ge=50, le=2000)
    agc_enabled: bool | None = None


class UISettingsUpdate(BaseModel):
    theme: str | None = None
    language: str | None = None
    show_confidence: bool | None = None
    compact_mode: bool | None = None


# ── Router ─────────────────────────────────────────────────────────


class SettingsRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/settings", tags=["settings"])
        self._register_routes()

    def _register_routes(self):
        # Get all settings
        self.router.add_api_route("", self.get_settings, methods=["GET"])

        # Section-specific endpoints
        self.router.add_api_route(
            "/generation", self.get_generation, methods=["GET"]
        )
        self.router.add_api_route(
            "/generation", self.update_generation, methods=["PATCH"],
            operation_id="update_settings_generation",
        )
        self.router.add_api_route(
            "/training", self.get_training, methods=["GET"]
        )
        self.router.add_api_route(
            "/training", self.update_training, methods=["PATCH"],
            operation_id="update_settings_training",
        )
        self.router.add_api_route(
            "/adaptive", self.get_adaptive, methods=["GET"]
        )
        self.router.add_api_route(
            "/adaptive", self.update_adaptive, methods=["PATCH"],
            operation_id="update_settings_adaptive",
        )
        self.router.add_api_route(
            "/voice", self.get_voice, methods=["GET"]
        )
        self.router.add_api_route(
            "/voice", self.update_voice, methods=["PATCH"],
            operation_id="update_settings_voice",
        )
        self.router.add_api_route(
            "/ui", self.get_ui, methods=["GET"]
        )
        self.router.add_api_route(
            "/ui", self.update_ui, methods=["PATCH"],
            operation_id="update_settings_ui",
        )

        # Reset
        self.router.add_api_route(
            "/reset", self.reset_settings, methods=["POST"],
        )

        # Adaptive insights
        self.router.add_api_route(
            "/adaptive/insights", self.get_adaptive_insights, methods=["GET"],
        )

    # ── Handlers ────────────────────────────────────────────────────

    @endpoint("settings.get_all")
    async def get_settings(self) -> dict:
        """Get all user settings."""
        from domains.settings.persistent import get_settings as _get
        ps = _get()
        return success_response(data=ps.to_dict())

    @endpoint("settings.get_generation")
    async def get_generation(self) -> dict:
        """Get generation settings."""
        from domains.settings.persistent import get_settings as _get
        s = _get().settings.generation
        from dataclasses import asdict
        return success_response(data=asdict(s))

    @endpoint("settings.update_generation")
    async def update_generation(
        self, req: GenerationSettingsUpdate, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Update generation settings."""
        from domains.settings.persistent import get_settings as _get
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        if not updates:
            return success_response(data={"message": "No changes"})
        ps = _get()
        ps.update("generation", **updates)
        safe_audit_log("settings.update", resource="generation", detail=str(updates))
        from dataclasses import asdict
        return success_response(data=asdict(ps.settings.generation))

    @endpoint("settings.get_training")
    async def get_training(self) -> dict:
        """Get training settings."""
        from domains.settings.persistent import get_settings as _get
        s = _get().settings.training
        from dataclasses import asdict
        return success_response(data=asdict(s))

    @endpoint("settings.update_training")
    async def update_training(
        self, req: TrainingSettingsUpdate, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Update training settings."""
        from domains.settings.persistent import get_settings as _get
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        if not updates:
            return success_response(data={"message": "No changes"})
        ps = _get()
        ps.update("training", **updates)
        safe_audit_log("settings.update", resource="training", detail=str(updates))
        from dataclasses import asdict
        return success_response(data=asdict(ps.settings.training))

    @endpoint("settings.get_adaptive")
    async def get_adaptive(self) -> dict:
        """Get adaptive settings."""
        from domains.settings.persistent import get_settings as _get
        s = _get().settings.adaptive
        from dataclasses import asdict
        return success_response(data=asdict(s))

    @endpoint("settings.update_adaptive")
    async def update_adaptive(
        self, req: AdaptiveSettingsUpdate, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Update adaptive settings."""
        from domains.settings.persistent import get_settings as _get
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        if not updates:
            return success_response(data={"message": "No changes"})
        ps = _get()
        ps.update("adaptive", **updates)
        safe_audit_log("settings.update", resource="adaptive", detail=str(updates))
        from dataclasses import asdict
        return success_response(data=asdict(ps.settings.adaptive))

    @endpoint("settings.get_voice")
    async def get_voice(self) -> dict:
        """Get voice settings."""
        from domains.settings.persistent import get_settings as _get
        s = _get().settings.voice
        from dataclasses import asdict
        return success_response(data=asdict(s))

    @endpoint("settings.update_voice")
    async def update_voice(
        self, req: VoiceSettingsUpdate, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Update voice settings."""
        from domains.settings.persistent import get_settings as _get
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        if not updates:
            return success_response(data={"message": "No changes"})
        ps = _get()
        ps.update("voice", **updates)
        safe_audit_log("settings.update", resource="voice", detail=str(updates))
        from dataclasses import asdict
        return success_response(data=asdict(ps.settings.voice))

    @endpoint("settings.get_ui")
    async def get_ui(self) -> dict:
        """Get UI settings."""
        from domains.settings.persistent import get_settings as _get
        s = _get().settings.ui
        from dataclasses import asdict
        return success_response(data=asdict(s))

    @endpoint("settings.update_ui")
    async def update_ui(
        self, req: UISettingsUpdate, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Update UI settings."""
        from domains.settings.persistent import get_settings as _get
        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        if not updates:
            return success_response(data={"message": "No changes"})
        ps = _get()
        ps.update("ui", **updates)
        safe_audit_log("settings.update", resource="ui", detail=str(updates))
        from dataclasses import asdict
        return success_response(data=asdict(ps.settings.ui))

    @endpoint("settings.reset")
    async def reset_settings(
        self, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Reset all settings to defaults."""
        from domains.settings.persistent import get_settings as _get
        ps = _get()
        ps.reset()
        safe_audit_log("settings.reset", resource="all")
        return success_response(data={"status": "reset", "message": "All settings restored to defaults"})

    @endpoint("settings.adaptive_insights")
    async def get_adaptive_insights(self) -> dict:
        """Get adaptive training insights from history."""
        from domains.training.adaptive_config import AdaptiveConfigEngine
        from domains.settings.persistent import get_settings as _get
        ps = _get()
        model = ps.settings.training.preferred_model or "gpt2"
        engine = AdaptiveConfigEngine()
        insights = engine.get_insights(model=model)
        return success_response(data=insights)


router = SettingsRouter().router
