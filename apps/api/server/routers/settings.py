"""
Settings Router — persistent user configuration that survives restarts.

Replaces the in-memory config controller with a persistent backend.
"""

import asyncio
import hashlib
import logging
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Query
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel, Field
from schemas.common import endpoint, raise_error, safe_audit_log, success_response

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


class ProviderApiSettingsUpdate(BaseModel):
    enabled: bool | None = None
    api_url: str | None = Field(default=None, pattern=r"^https?://.+")
    api_key: str | None = None
    model: str | None = None
    timeout: float | None = Field(default=None, ge=1.0, le=600.0)
    max_retries: int | None = Field(default=None, ge=0, le=10)


def _provider_public(s) -> dict:
    """Public view of provider settings — never leaks the raw API key."""
    return {
        "enabled": s.enabled,
        "api_url": s.api_url,
        "model": s.model,
        "timeout": s.timeout,
        "max_retries": s.max_retries,
        "api_key_set": bool(s.api_key),
    }


# ── Router ─────────────────────────────────────────────────────────


class SettingsRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/settings", tags=["settings"])
        self._register_routes()

    def _register_routes(self):
        # Get all settings
        self.router.add_api_route("", self.get_settings, methods=["GET"])

        # Section-specific endpoints
        self.router.add_api_route("/generation", self.get_generation, methods=["GET"])
        self.router.add_api_route(
            "/generation",
            self.update_generation,
            methods=["PATCH"],
            operation_id="update_settings_generation",
        )
        self.router.add_api_route("/training", self.get_training, methods=["GET"])
        self.router.add_api_route(
            "/training",
            self.update_training,
            methods=["PATCH"],
            operation_id="update_settings_training",
        )
        self.router.add_api_route("/adaptive", self.get_adaptive, methods=["GET"])
        self.router.add_api_route(
            "/adaptive",
            self.update_adaptive,
            methods=["PATCH"],
            operation_id="update_settings_adaptive",
        )
        self.router.add_api_route("/voice", self.get_voice, methods=["GET"])
        self.router.add_api_route(
            "/voice",
            self.update_voice,
            methods=["PATCH"],
            operation_id="update_settings_voice",
        )
        self.router.add_api_route("/ui", self.get_ui, methods=["GET"])
        self.router.add_api_route(
            "/ui",
            self.update_ui,
            methods=["PATCH"],
            operation_id="update_settings_ui",
        )

        # External API provider (OpenRouter / OpenAI-compatible)
        self.router.add_api_route("/providers/api", self.get_providers_api, methods=["GET"])
        self.router.add_api_route(
            "/providers/api",
            self.update_providers_api,
            methods=["PATCH"],
            operation_id="update_settings_providers_api",
        )

        # Reset
        self.router.add_api_route(
            "/reset",
            self.reset_settings,
            methods=["POST"],
        )

        # Adaptive insights
        self.router.add_api_route(
            "/adaptive/insights",
            self.get_adaptive_insights,
            methods=["GET"],
        )

        # Batch training status
        self.router.add_api_route(
            "/training/batch-status",
            self.get_batch_training_status,
            methods=["GET"],
        )

        # Training presets
        self.router.add_api_route(
            "/training/presets",
            self.list_training_presets,
            methods=["GET"],
        )
        self.router.add_api_route(
            "/training/presets/{preset_name}",
            self.get_training_preset,
            methods=["GET"],
        )
        self.router.add_api_route(
            "/training/presets/{preset_name}/apply",
            self.apply_training_preset,
            methods=["POST"],
        )

        # Training history export
        self.router.add_api_route(
            "/training/history/export",
            self.export_training_history,
            methods=["GET"],
        )

        # Model card generation
        self.router.add_api_route(
            "/model-card",
            self.generate_model_card,
            methods=["POST"],
        )

        # Training run comparison
        self.router.add_api_route(
            "/training/compare",
            self.compare_training_runs,
            methods=["GET"],
        )

        # Training run management
        self.router.add_api_route(
            "/training/runs",
            self.filter_training_runs,
            methods=["GET"],
        )
        # Bulk operations (MUST be before /{run_id} routes to avoid path conflicts)
        self.router.add_api_route(
            "/training/runs/bulk/delete",
            self.bulk_delete_runs,
            methods=["POST"],
        )
        self.router.add_api_route(
            "/training/runs/bulk/tag",
            self.bulk_add_tag,
            methods=["POST"],
        )
        self.router.add_api_route(
            "/training/runs/bulk/bookmark",
            self.bulk_bookmark,
            methods=["POST"],
        )
        self.router.add_api_route(
            "/training/runs/{run_id}",
            self.get_training_run,
            methods=["GET"],
        )
        self.router.add_api_route(
            "/training/runs/{run_id}",
            self.delete_training_run,
            methods=["DELETE"],
        )
        self.router.add_api_route(
            "/training/history/clear",
            self.clear_training_history,
            methods=["POST"],
        )

        # Training run tags and notes
        self.router.add_api_route(
            "/training/runs/{run_id}/tags",
            self.add_run_tag,
            methods=["POST"],
        )
        self.router.add_api_route(
            "/training/runs/{run_id}/tags/{tag}",
            self.remove_run_tag,
            methods=["DELETE"],
        )
        self.router.add_api_route(
            "/training/runs/{run_id}/notes",
            self.set_run_notes,
            methods=["PUT"],
        )
        self.router.add_api_route(
            "/training/tags",
            self.get_all_tags,
            methods=["GET"],
        )
        self.router.add_api_route(
            "/training/tags/{tag}",
            self.get_runs_by_tag,
            methods=["GET"],
        )
        self.router.add_api_route(
            "/training/runs/{run_id}/export",
            self.export_training_run,
            methods=["GET"],
        )

        # Training run bookmarks
        self.router.add_api_route(
            "/training/runs/{run_id}/bookmark",
            self.toggle_bookmark,
            methods=["POST"],
        )
        self.router.add_api_route(
            "/training/bookmarks",
            self.get_bookmarked_runs,
            methods=["GET"],
        )

        # Training run duplicate
        self.router.add_api_route(
            "/training/runs/{run_id}/duplicate",
            self.duplicate_training_run,
            methods=["POST"],
        )

        # Auto-train status + config
        self.router.add_api_route(
            "/training/auto-train/status",
            self.auto_train_status,
            methods=["GET"],
        )
        self.router.add_api_route(
            "/training/auto-train/config",
            self.auto_train_config,
            methods=["PATCH"],
        )

        # Training analytics
        self.router.add_api_route(
            "/training/analytics",
            self.get_training_analytics,
            methods=["GET"],
        )

    # ── Handlers ────────────────────────────────────────────────────

    @endpoint("settings.get_all")
    async def get_settings(self) -> dict:
        """Get all user settings."""
        from domain.settings import get_settings as _get

        ps = _get()
        return success_response(data=ps.to_dict())

    @endpoint("settings.get_generation")
    async def get_generation(self) -> dict:
        """Get generation settings."""
        from domain.settings import get_settings as _get

        s = _get().settings.generation
        from dataclasses import asdict

        return success_response(data=asdict(s))

    @endpoint("settings.update_generation")
    async def update_generation(
        self, req: GenerationSettingsUpdate, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Update generation settings."""
        from domain.settings import get_settings as _get

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
        from domain.settings import get_settings as _get

        s = _get().settings.training
        from dataclasses import asdict

        return success_response(data=asdict(s))

    @endpoint("settings.update_training")
    async def update_training(
        self, req: TrainingSettingsUpdate, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Update training settings."""
        from domain.settings import get_settings as _get

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
        from domain.settings import get_settings as _get

        s = _get().settings.adaptive
        from dataclasses import asdict

        return success_response(data=asdict(s))

    @endpoint("settings.update_adaptive")
    async def update_adaptive(
        self, req: AdaptiveSettingsUpdate, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Update adaptive settings."""
        from domain.settings import get_settings as _get

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
        from domain.settings import get_settings as _get

        s = _get().settings.voice
        from dataclasses import asdict

        return success_response(data=asdict(s))

    @endpoint("settings.update_voice")
    async def update_voice(
        self, req: VoiceSettingsUpdate, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Update voice settings."""
        from domain.settings import get_settings as _get

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
        from domain.settings import get_settings as _get

        s = _get().settings.ui
        from dataclasses import asdict

        return success_response(data=asdict(s))

    @endpoint("settings.update_ui")
    async def update_ui(
        self, req: UISettingsUpdate, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Update UI settings."""
        from domain.settings import get_settings as _get

        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        if not updates:
            return success_response(data={"message": "No changes"})
        ps = _get()
        ps.update("ui", **updates)
        safe_audit_log("settings.update", resource="ui", detail=str(updates))
        from dataclasses import asdict

        return success_response(data=asdict(ps.settings.ui))

    @endpoint("settings.get_providers_api")
    async def get_providers_api(self) -> dict:
        """Get external API provider settings (key never returned)."""
        from domain.settings import get_settings as _get

        s = _get().settings.providers
        return success_response(data=_provider_public(s))

    @endpoint("settings.update_providers_api")
    async def update_providers_api(
        self,
        req: ProviderApiSettingsUpdate,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Update external API provider settings and apply at runtime."""
        from dataclasses import asdict

        from domain.inference import configure_api_provider
        from domain.settings import get_settings as _get

        updates = {k: v for k, v in req.model_dump().items() if v is not None}
        if not updates:
            return success_response(data={"message": "No changes"})
        ps = _get()
        merged = {**asdict(ps.settings.providers), **updates}
        ps.update("providers", **merged)
        outcome = configure_api_provider(
            bool(ps.settings.providers.enabled),
            api_url=ps.settings.providers.api_url,
            api_key=ps.settings.providers.api_key,
            model=ps.settings.providers.model,
            timeout=ps.settings.providers.timeout,
            max_retries=ps.settings.providers.max_retries,
        )
        safe_audit_log("settings.update", resource="providers.api", detail=str(updates))
        return success_response(
            data={**_provider_public(ps.settings.providers), "registered": outcome["registered"]}
        )

    @endpoint("settings.reset")
    async def reset_settings(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Reset all settings to defaults."""
        from domain.settings import get_settings as _get

        ps = _get()
        ps.reset()
        safe_audit_log("settings.reset", resource="all")
        return success_response(
            data={"status": "reset", "message": "All settings restored to defaults"}
        )

    @endpoint("settings.adaptive_insights")
    async def get_adaptive_insights(self) -> dict:
        """Get adaptive training insights from history."""
        from domain.settings import get_settings as _get
        from domain.training import AdaptiveConfigEngine

        ps = _get()
        model = ps.settings.training.preferred_model or "gpt2"
        engine = AdaptiveConfigEngine()
        insights = engine.get_insights(model=model)
        return success_response(data=insights)

    @endpoint("settings.get_batch_training_status")
    async def get_batch_training_status(self) -> dict:
        """Get status of all training jobs (running, queued, completed, failed)."""
        try:
            from training.jobs import training_jobs
        except ImportError:
            return success_response(
                data={
                    "jobs": [],
                    "summary": {"total": 0, "running": 0, "queued": 0, "completed": 0, "failed": 0},
                }
            )

        jobs = []
        summary = {"total": 0, "running": 0, "queued": 0, "completed": 0, "failed": 0}
        for job_id, job in training_jobs.items():
            status = job.get("status", "unknown")
            summary["total"] += 1
            if status in ("running", "starting"):
                summary["running"] += 1
            elif status == "queued":
                summary["queued"] += 1
            elif status == "completed":
                summary["completed"] += 1
            elif status == "failed":
                summary["failed"] += 1
            jobs.append(
                {
                    "job_id": job_id,
                    "name": job.get("name", job_id[:8]),
                    "model": job.get("model", ""),
                    "dataset": job.get("dataset", ""),
                    "status": status,
                    "progress": job.get("progress", 0),
                    "current_step": job.get("current_step", ""),
                    "total_steps": job.get("total_steps", ""),
                    "created_at": job.get("created_at", 0),
                }
            )
        return success_response(data={"jobs": jobs, "summary": summary})

    @endpoint("settings.export_training_history")
    async def export_training_history(
        self,
        format: str = "json",
        limit: int = 0,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Export training history as JSON or CSV."""
        from domain.training import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        if format == "csv":
            csv_data = tracker.export_csv(limit=limit)
            return success_response(
                data={"format": "csv", "content": csv_data, "count": len(csv_data.splitlines()) - 1}
            )
        return success_response(
            data={
                "format": "json",
                "outcomes": tracker.export_json(limit=limit),
                "count": len(tracker.export_json(limit=limit)),
            }
        )

    @endpoint("settings.generate_model_card")
    async def generate_model_card(
        self,
        name: str = "model",
        base_model: str = "",
        description: str = "",
        dataset: str = "",
        dataset_size: int = 0,
        epochs: int = 0,
        learning_rate: float = 0.0,
        batch_size: int = 0,
        final_loss: float = 0.0,
        perplexity: float = 0.0,
        quality_score: float = 0.0,
        training_time_s: float = 0.0,
        training_method: str = "",
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Generate a model card from training metadata."""
        from domain.training import generate_model_card as _generate

        card = _generate(
            name=name,
            base_model=base_model,
            description=description,
            dataset=dataset,
            dataset_size=dataset_size,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            final_loss=final_loss,
            perplexity=perplexity,
            quality_score=quality_score,
            training_time_s=training_time_s,
            training_method=training_method,
        )
        return success_response(data={"card": card.to_dict(), "markdown": card.to_markdown()})

    @endpoint("settings.compare_training_runs")
    async def compare_training_runs(
        self,
        run_a: str,
        run_b: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Compare two training runs side by side."""
        from domain.training import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        result = tracker.compare(run_a, run_b)
        if result is None:
            return success_response(data={"error": "One or both run IDs not found"})
        return success_response(data=result)

    @endpoint("settings.list_training_presets")
    async def list_training_presets(self) -> dict:
        """List all available training presets."""
        from domain.training import list_presets

        return success_response(data={"presets": list_presets()})

    @endpoint("settings.get_training_preset")
    async def get_training_preset(self, preset_name: str) -> dict:
        """Get a specific training preset."""
        from domain.training import get_preset

        preset = get_preset(preset_name)
        if not preset:
            return success_response(data={"error": f"Preset '{preset_name}' not found"})
        return success_response(data=preset)

    @endpoint("settings.apply_training_preset")
    async def apply_training_preset(
        self,
        preset_name: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Apply a training preset to current settings."""
        from domain.training import apply_preset

        config = apply_preset(preset_name)
        if not config:
            return success_response(data={"error": f"Preset '{preset_name}' not found"})
        from domain.settings import get_settings as _get

        ps = _get()
        ps.update("training", **config)
        safe_audit_log("settings.apply_preset", resource="training", detail=preset_name)
        from dataclasses import asdict

        return success_response(
            data={
                "preset": preset_name,
                "applied": config,
                "settings": asdict(ps.settings.training),
            }
        )

    @endpoint("settings.get_training_run")
    async def get_training_run(self, run_id: str) -> dict:
        """Get a single training run by ID."""
        from domain.training import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        outcomes = tracker.load_outcomes()
        run = next((o for o in outcomes if o.run_id == run_id), None)
        if not run:
            return success_response(data={"error": f"Run '{run_id}' not found"})
        return success_response(data=run.to_dict())

    @endpoint("settings.delete_training_run")
    async def delete_training_run(
        self,
        run_id: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Delete a specific training run by ID."""
        import json

        from domain.training import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        outcomes = tracker.load_outcomes()
        filtered = [o for o in outcomes if o.run_id != run_id]
        if len(filtered) == len(outcomes):
            return success_response(data={"error": f"Run '{run_id}' not found", "deleted": False})
        # Rewrite history without the deleted run
        tracker.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tracker.history_path, "w") as f:
            for o in filtered:
                f.write(json.dumps(o.to_dict()) + "\n")
        safe_audit_log("settings.delete_run", resource="training", detail=run_id)
        return success_response(data={"deleted": True, "run_id": run_id})

    @endpoint("settings.filter_training_runs")
    async def filter_training_runs(
        self,
        model: str | None = None,
        method: str | None = None,
        converged: bool | None = None,
        min_quality: float | None = None,
        limit: int = 50,
    ) -> dict:
        """Filter training runs by model, method, convergence, or quality."""
        from domain.training import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        outcomes = tracker.load_outcomes()
        if model:
            outcomes = [o for o in outcomes if o.model == model]
        if method:
            outcomes = [o for o in outcomes if o.method == method]
        if converged is not None:
            outcomes = [o for o in outcomes if o.converged == converged]
        if min_quality is not None:
            outcomes = [o for o in outcomes if o.quality_score >= min_quality]
        outcomes = outcomes[-limit:]
        return success_response(
            data={
                "runs": [o.to_dict() for o in outcomes],
                "count": len(outcomes),
            }
        )

    @endpoint("settings.clear_training_history")
    async def clear_training_history(
        self,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Clear all training history."""
        from domain.training import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        count = tracker.clear()
        safe_audit_log(
            "settings.clear_history", resource="training", detail=f"{count} runs removed"
        )
        return success_response(data={"cleared": True, "removed_count": count})

    @endpoint("settings.add_run_tag")
    async def add_run_tag(
        self,
        run_id: str,
        tag: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Add a tag to a training run."""
        from domain.training import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        run = tracker.add_tag(run_id, tag)
        if not run:
            return success_response(data={"error": f"Run '{run_id}' not found"})
        safe_audit_log("settings.add_tag", resource="training", detail=f"{run_id}: {tag}")
        return success_response(data=run.to_dict())

    @endpoint("settings.remove_run_tag")
    async def remove_run_tag(
        self,
        run_id: str,
        tag: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Remove a tag from a training run."""
        from domain.training import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        run = tracker.remove_tag(run_id, tag)
        if not run:
            return success_response(data={"error": f"Run '{run_id}' not found"})
        safe_audit_log("settings.remove_tag", resource="training", detail=f"{run_id}: {tag}")
        return success_response(data=run.to_dict())

    @endpoint("settings.set_run_notes")
    async def set_run_notes(
        self,
        run_id: str,
        notes: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Set notes on a training run."""
        from domain.training import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        run = tracker.set_notes(run_id, notes)
        if not run:
            return success_response(data={"error": f"Run '{run_id}' not found"})
        safe_audit_log("settings.set_notes", resource="training", detail=run_id)
        return success_response(data=run.to_dict())

    @endpoint("settings.get_all_tags")
    async def get_all_tags(self) -> dict:
        """Get all unique tags across all training runs."""
        from domain.training import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        tags = tracker.get_all_tags()
        return success_response(data={"tags": tags})

    @endpoint("settings.get_runs_by_tag")
    async def get_runs_by_tag(self, tag: str) -> dict:
        """Get all training runs with a specific tag."""
        from domain.training import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        runs = tracker.get_outcomes_by_tag(tag)
        return success_response(
            data={
                "runs": [o.to_dict() for o in runs],
                "count": len(runs),
                "tag": tag,
            }
        )

    @endpoint("settings.export_training_run")
    async def export_training_run(self, run_id: str, format: str = "json") -> dict:
        """Export a single training run as JSON or YAML."""
        from domain.training import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        outcomes = tracker.load_outcomes()
        run = next((o for o in outcomes if o.run_id == run_id), None)
        if not run:
            return success_response(data={"error": f"Run '{run_id}' not found"})
        run_dict = run.to_dict()
        if format == "yaml":
            try:
                import yaml

                content = yaml.dump(run_dict, default_flow_style=False)
            except ImportError:
                content = str(run_dict)
        else:
            import json

            content = json.dumps(run_dict, indent=2)
        return success_response(data={"run_id": run_id, "format": format, "content": content})

    @endpoint("settings.toggle_bookmark")
    async def toggle_bookmark(
        self,
        run_id: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Toggle bookmark status on a training run."""
        from domain.training import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        run = tracker.toggle_bookmark(run_id)
        if not run:
            return success_response(data={"error": f"Run '{run_id}' not found"})
        safe_audit_log("settings.toggle_bookmark", resource="training", detail=run_id)
        return success_response(data=run.to_dict())

    @endpoint("settings.get_bookmarked_runs")
    async def get_bookmarked_runs(self) -> dict:
        """Get all bookmarked training runs."""
        from domain.training import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        runs = tracker.get_bookmarked()
        return success_response(
            data={
                "runs": [o.to_dict() for o in runs],
                "count": len(runs),
            }
        )

    @endpoint("settings.duplicate_training_run")
    async def duplicate_training_run(
        self,
        run_id: str,
        new_run_id: str = "",
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Duplicate a training run with a new ID."""
        from domain.training import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        new_run = tracker.duplicate_run(run_id, new_run_id)
        if not new_run:
            return success_response(data={"error": f"Run '{run_id}' not found"})
        safe_audit_log(
            "settings.duplicate_run", resource="training", detail=f"{run_id} -> {new_run.run_id}"
        )
        return success_response(data=new_run.to_dict())

    @endpoint("settings.bulk_delete_runs")
    async def bulk_delete_runs(
        self,
        run_ids: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Delete multiple training runs. run_ids is a comma-separated list."""
        from domain.training import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        ids = [x.strip() for x in run_ids.split(",") if x.strip()]
        count = tracker.bulk_delete(ids)
        safe_audit_log("settings.bulk_delete", resource="training", detail=f"{count} runs deleted")
        return success_response(data={"deleted_count": count, "requested": len(ids)})

    @endpoint("settings.bulk_add_tag")
    async def bulk_add_tag(
        self,
        run_ids: str,
        tag: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Add a tag to multiple training runs. run_ids is a comma-separated list."""
        from domain.training import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        ids = [x.strip() for x in run_ids.split(",") if x.strip()]
        count = tracker.bulk_add_tag(ids, tag)
        safe_audit_log(
            "settings.bulk_add_tag", resource="training", detail=f"{count} runs tagged '{tag}'"
        )
        return success_response(data={"updated_count": count, "tag": tag})

    @endpoint("settings.bulk_bookmark")
    async def bulk_bookmark(
        self,
        run_ids: str,
        bookmarked: bool = True,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Set bookmark status on multiple training runs."""
        from domain.training import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        ids = [x.strip() for x in run_ids.split(",") if x.strip()]
        count = tracker.bulk_bookmark(ids, bookmarked)
        safe_audit_log(
            "settings.bulk_bookmark",
            resource="training",
            detail=f"{count} runs bookmark={bookmarked}",
        )
        return success_response(data={"updated_count": count, "bookmarked": bookmarked})

    # ── Auto-train ───────────────────────────────────────────────────

    @endpoint("settings.auto_train_status")
    async def auto_train_status(self) -> dict:
        """Get auto-trainer status and configuration."""
        from domain.training import get_auto_trainer

        trainer = get_auto_trainer()
        return success_response(data=trainer.status())

    @endpoint("settings.auto_train_config")
    async def auto_train_config(
        self,
        threshold: int | None = Query(None, ge=1, le=100),
        interval_s: int | None = Query(None, ge=30, le=3600),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Update auto-trainer configuration at runtime."""
        from domain.training import get_auto_trainer

        trainer = get_auto_trainer()
        if threshold is not None:
            trainer.threshold = threshold
        if interval_s is not None:
            trainer.interval_s = interval_s
        safe_audit_log(
            "settings.auto_train_config",
            resource="training",
            detail=f"threshold={threshold} interval={interval_s}",
        )
        return success_response(data=trainer.status())

    # ── Training Analytics ──────────────────────────────────────────

    @endpoint("settings.training_analytics")
    async def get_training_analytics(self) -> dict:
        """Get aggregated training analytics for charts and summaries."""
        from domain.training import TrainingOutcomeTracker

        tracker = TrainingOutcomeTracker()
        outcomes = tracker.load_outcomes()

        if not outcomes:
            return success_response(
                data={
                    "total_runs": 0,
                    "quality_trend": [],
                    "method_distribution": {},
                    "model_distribution": {},
                    "convergence_rate": 0.0,
                    "avg_quality": 0.0,
                    "best_run": None,
                    "recent_runs": [],
                    "tag_cloud": {},
                }
            )

        total = len(outcomes)
        qualities = [
            o.get("quality_score", 0) for o in outcomes if o.get("quality_score") is not None
        ]
        avg_quality = sum(qualities) / len(qualities) if qualities else 0.0
        converged = sum(1 for o in outcomes if o.get("converged"))
        convergence_rate = converged / total if total else 0.0

        best = max(outcomes, key=lambda o: o.get("quality_score", 0)) if outcomes else None

        method_dist: dict[str, int] = {}
        model_dist: dict[str, int] = {}
        tag_cloud: dict[str, int] = {}
        for o in outcomes:
            m = o.get("method", "unknown")
            method_dist[m] = method_dist.get(m, 0) + 1
            mdl = o.get("model", "unknown")
            model_dist[mdl] = model_dist.get(mdl, 0) + 1
            for t in o.get("tags", []):
                tag_cloud[t] = tag_cloud.get(t, 0) + 1

        quality_trend = [
            {
                "run_id": o.get("run_id", ""),
                "quality": o.get("quality_score", 0),
                "timestamp": o.get("timestamp", ""),
            }
            for o in outcomes[-50:]
        ]

        recent = [
            {
                "run_id": o.get("run_id", ""),
                "model": o.get("model", ""),
                "method": o.get("method", ""),
                "quality_score": o.get("quality_score", 0),
                "converged": o.get("converged", False),
                "timestamp": o.get("timestamp", ""),
                "tags": o.get("tags", []),
            }
            for o in outcomes[-10:]
        ]

        return success_response(
            data={
                "total_runs": total,
                "quality_trend": quality_trend,
                "method_distribution": method_dist,
                "model_distribution": model_dist,
                "convergence_rate": round(convergence_rate, 3),
                "avg_quality": round(avg_quality, 3),
                "best_run": {
                    "run_id": best.get("run_id", ""),
                    "model": best.get("model", ""),
                    "quality_score": best.get("quality_score", 0),
                    "method": best.get("method", ""),
                }
                if best
                else None,
                "recent_runs": recent,
                "tag_cloud": tag_cloud,
            }
        )


# ── Moved verbatim from routers/api_keys.py (route-map card 5/7, card 20261101_154056) ────
# ApiKeyManager is live storage (auth/workspaces/inference import it); ApiKeysRouter
# stays deliberately UNMOUNTED exactly as before (layer-conformance allowlist card
# b22e878a) — its 6 duplicate /security/keys declarations ride along, total flat.


class CreateKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    scopes: list[str] = Field(default=["*"])
    expires_at: int | None = Field(default=None, description="Unix timestamp for key expiration")


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class ApiKeyManager:
    """Manages API keys with MogDB storage and optional JSON sync."""

    def __init__(
        self, db=None, db_path: str | Path | None = None, sync_dir: str | Path | None = None
    ):
        if db is not None:
            self._db = db
        else:
            from mogdb import MogDB

            if db_path is None:
                repo_root = Path(__file__).parent.parent.parent.parent
                db_path = repo_root / "data" / "api_keys_mogdb"
            if sync_dir is None:
                repo_root = Path(__file__).parent.parent.parent.parent
                sync_dir = repo_root / "data" / "api_keys_json"
            self._db = MogDB(str(db_path), sync_dir=str(sync_dir))
        self._collection = self._db.collection("api_keys")

    def create(
        self,
        name: str,
        scopes: list[str] | None = None,
        expires_at: int | None = None,
        workspace_id: str = "",
        user_id: str = "",
    ) -> dict[str, Any]:
        key = f"slo_{secrets.token_urlsafe(32)}"
        doc = {
            "name": name,
            "key": key,
            "key_hash": _hash_key(key),
            "scopes": scopes or ["*"],
            "created_at": int(time.time()),
            "workspace_id": workspace_id,
            "user_id": user_id,
        }
        if expires_at is not None:
            doc["expires_at"] = expires_at
        doc_id = self._collection.insert_one(doc)
        doc["id"] = doc_id
        doc["revoked"] = False
        logger.info("Created API key '%s' (id=%s, ws=%s)", name, doc_id, workspace_id or "global")
        return doc

    def list(self, workspace_id: str = "", user_id: str = "") -> list[dict[str, Any]]:
        query: dict[str, Any] = {}
        if workspace_id:
            query["workspace_id"] = workspace_id
        if user_id:
            query["user_id"] = user_id
        docs = self._collection.find(query)
        result = []
        for d in docs:
            entry = {
                "id": d["_id"],
                "name": d["name"],
                "key_hash": d["key_hash"],
                "scopes": d.get("scopes", ["*"]),
                "created_at": d["created_at"],
                "revoked": d.get("revoked", False),
                "workspace_id": d.get("workspace_id", ""),
                "user_id": d.get("user_id", ""),
            }
            if "expires_at" in d:
                entry["expires_at"] = d["expires_at"]
            result.append(entry)
        return result

    def get(self, key_id: str) -> dict[str, Any] | None:
        doc = self._collection.find_one({"_id": key_id})
        if doc is None:
            return None
        return {
            "id": doc["_id"],
            "name": doc["name"],
            "key": doc["key"],
            "key_hash": doc["key_hash"],
            "scopes": doc.get("scopes", ["*"]),
            "created_at": doc["created_at"],
            "expires_at": doc.get("expires_at"),
            "revoked": doc.get("revoked", False),
            "workspace_id": doc.get("workspace_id", ""),
            "user_id": doc.get("user_id", ""),
        }

    def revoke(self, key_id: str) -> None:
        doc = self._collection.find_one({"_id": key_id})
        if doc is None:
            raise ValueError(f"API key not found: {key_id}")
        self._collection.update_one({"_id": key_id}, {"$set": {"revoked": True}})
        logger.info("Revoked API key '%s' (id=%s)", doc.get("name"), key_id)

    def rotate(self, key_id: str) -> dict[str, Any]:
        old = self._collection.find_one({"_id": key_id})
        if old is None:
            raise ValueError(f"API key not found: {key_id}")
        self.revoke(key_id)
        return self.create(
            old["name"],
            scopes=old.get("scopes", ["*"]),
            expires_at=old.get("expires_at"),
            workspace_id=old.get("workspace_id", ""),
            user_id=old.get("user_id", ""),
        )

    def validate(self, key: str) -> bool:
        key_hash = _hash_key(key)
        docs = self._collection.find({"key_hash": key_hash})
        for doc in docs:
            if doc.get("revoked", False):
                return False
            if "expires_at" in doc and doc["expires_at"] < time.time():
                return False
            return True
        return False


class ApiKeysRouter:
    """FastAPI router for API key CRUD operations."""

    def __init__(self, key_manager: ApiKeyManager | None = None):
        self._manager = key_manager or ApiKeyManager()
        self.router = APIRouter(prefix="/security", tags=["security"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(path="/keys", endpoint=self.create_key, methods=["POST"])
        self.router.add_api_route(path="/keys", endpoint=self.list_keys, methods=["GET"])
        self.router.add_api_route(
            path="/keys/validate", endpoint=self.validate_key, methods=["POST"]
        )
        self.router.add_api_route(path="/keys/{key_id}", endpoint=self.get_key, methods=["GET"])
        self.router.add_api_route(
            path="/keys/{key_id}", endpoint=self.delete_key, methods=["DELETE"]
        )
        self.router.add_api_route(
            path="/keys/{key_id}/rotate", endpoint=self.rotate_key, methods=["POST"]
        )

    def _get_workspace_user(self, auth_user: dict) -> tuple[str, str]:
        """Extract workspace_id and user_id from auth_user."""
        user_id = auth_user.get("sub", "") if auth_user else ""
        workspace_id = auth_user.get("workspace_id", "") if auth_user else ""
        return workspace_id, user_id

    @endpoint("api_keys.create")
    async def create_key(
        self, req: CreateKeyRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        workspace_id, user_id = self._get_workspace_user(auth_user)
        key = self._manager.create(
            req.name,
            scopes=req.scopes,
            expires_at=req.expires_at,
            workspace_id=workspace_id,
            user_id=user_id,
        )
        return success_response(data=key)

    @endpoint("api_keys.list")
    async def list_keys(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        workspace_id, user_id = self._get_workspace_user(auth_user)
        keys = self._manager.list(workspace_id=workspace_id, user_id=user_id)
        return success_response(data={"keys": keys, "count": len(keys)})

    @endpoint("api_keys.get")
    async def get_key(self, key_id: str) -> dict:
        key = self._manager.get(key_id)
        if key is None:
            raise_error("API key not found", "E_NOT_FOUND", status_code=404)
        return success_response(data=key)

    @endpoint("api_keys.delete")
    async def delete_key(self, key_id: str) -> dict:
        try:
            self._manager.revoke(key_id)
        except ValueError as e:
            raise_error(str(e), "E_NOT_FOUND", status_code=404)
        return success_response(data={"revoked": True})

    @endpoint("api_keys.rotate")
    async def rotate_key(self, key_id: str) -> dict:
        try:
            new_key = self._manager.rotate(key_id)
        except ValueError as e:
            raise_error(str(e), "E_NOT_FOUND", status_code=404)
        return success_response(data=new_key)

    @endpoint("api_keys.validate")
    async def validate_key(self, body: dict) -> dict:
        key = body.get("key", "")
        valid = self._manager.validate(key)
        return success_response(data={"valid": valid})


_api_keys_router: ApiKeysRouter | None = None


def get_api_keys_router() -> ApiKeysRouter:
    global _api_keys_router
    if _api_keys_router is None:
        _api_keys_router = ApiKeysRouter()
    return _api_keys_router


# ── Moved verbatim from routers/security.py (route-map card 5/7) ─────────────


def _get_key_manager():
    # Manager moved with the same card (20261101_154056) — it now lives in
    # this module, so no cross-module import is needed.
    return ApiKeyManager()


class SecurityRouter:
    """Security Router - Audit logs and API key management."""

    def __init__(self):
        self.router = APIRouter(prefix="/security", tags=["security"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(path="/audit", endpoint=self.get_audit_logs, methods=["GET"])
        self.router.add_api_route(path="/keys", endpoint=self.list_keys, methods=["GET"])
        self.router.add_api_route(path="/keys", endpoint=self.create_key, methods=["POST"])
        self.router.add_api_route(
            path="/keys/validate", endpoint=self.validate_key, methods=["POST"]
        )
        self.router.add_api_route(path="/keys/{key_id}", endpoint=self.get_key, methods=["GET"])
        self.router.add_api_route(
            path="/keys/{key_id}", endpoint=self.delete_key, methods=["DELETE"]
        )
        self.router.add_api_route(
            path="/keys/{key_id}/rotate", endpoint=self.rotate_key, methods=["POST"]
        )

    # ── Audit logs ──

    @staticmethod
    @endpoint("security.audit_logs")
    async def get_audit_logs(
        limit: int = Query(
            default=100, ge=1, le=10000, description="Maximum number of log entries to return"
        ),
        event_type: str | None = Query(default=None, description="Filter by event type"),
        history: bool = Query(default=False, description="Read from persisted audit.log file"),
        before: str | None = Query(default=None, description="ISO-8601 cursor for pagination"),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        from infrastructure.auth import get_audit_logger

        if auth_user and auth_user.get("role") not in ("owner", "admin"):
            raise_error("Admin access required", code="auth/forbidden", status_code=403)

        audit_logger = get_audit_logger()

        workspace_id = ""
        if auth_user and auth_user.get("sub"):
            workspace_id = auth_user.get("workspace_id", "")

        if history:
            logs = await asyncio.to_thread(
                audit_logger.file_query,
                limit=limit,
                event_type=event_type,
                before=before,
                workspace_id=workspace_id,
            )
        else:
            logs = audit_logger.logs[-limit:]
            if event_type:
                logs = [l for l in logs if l.get("event_type") == event_type]
            if workspace_id:
                logs = [l for l in logs if l.get("workspace_id", "") == workspace_id]
        return success_response(data={"logs": logs, "count": len(logs)})

    # ── API key management ──

    @staticmethod
    @endpoint("security.create_key")
    async def create_key(
        req: CreateKeyRequest, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        mgr = _get_key_manager()
        key = mgr.create(req.name, scopes=req.scopes, expires_at=req.expires_at)
        return success_response(data=key)

    @staticmethod
    @endpoint("security.list_keys")
    async def list_keys(auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        mgr = _get_key_manager()
        keys = mgr.list()
        return success_response(
            data={"keys": keys, "count": len(keys), "configured": len(keys) > 0}
        )

    @staticmethod
    @endpoint("security.get_key")
    async def get_key(key_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        mgr = _get_key_manager()
        key = mgr.get(key_id)
        if key is None:
            raise_error("API key not found", "E_NOT_FOUND", status_code=404)
        if (
            auth_user is not None
            and auth_user.get("sub") != key.get("user_id")
            and auth_user.get("role") != "admin"
        ):
            raise_error("Not authorized to manage this key", code="auth/forbidden", status_code=403)
        return success_response(data=key)

    @staticmethod
    @endpoint("security.delete_key")
    async def delete_key(key_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            mgr = _get_key_manager()
            key = mgr.get(key_id)
            if key is None:
                raise_error("API key not found", "E_NOT_FOUND", status_code=404)
            if (
                auth_user is not None
                and auth_user.get("sub") != key.get("user_id")
                and auth_user.get("role") != "admin"
            ):
                raise_error(
                    "Not authorized to manage this key", code="auth/forbidden", status_code=403
                )
            mgr.revoke(key_id)
            return success_response(data={"revoked": True})
        except ValueError as e:
            raise_error(str(e), "E_NOT_FOUND", status_code=404)

    @staticmethod
    @endpoint("security.rotate_key")
    async def rotate_key(key_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            mgr = _get_key_manager()
            key = mgr.get(key_id)
            if key is None:
                raise_error("API key not found", "E_NOT_FOUND", status_code=404)
            if (
                auth_user is not None
                and auth_user.get("sub") != key.get("user_id")
                and auth_user.get("role") != "admin"
            ):
                raise_error(
                    "Not authorized to manage this key", code="auth/forbidden", status_code=403
                )
            new_key = mgr.rotate(key_id)
            return success_response(data=new_key)
        except ValueError as e:
            raise_error(str(e), "E_NOT_FOUND", status_code=404)

    @staticmethod
    @endpoint("security.validate_key")
    async def validate_key(body: dict, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        mgr = _get_key_manager()
        key = body.get("key", "")
        valid = mgr.validate(key)
        return success_response(data={"valid": valid})


# Aggregate facade (route-map card 5/7): ONE module-level router serving
# /settings/* and /security/* — sub-router prefixes bake at add-time, so every
# path stays byte-identical to the two old modules.
_instance = SettingsRouter()
_security_router = SecurityRouter()

router = APIRouter()
router.include_router(_instance.router)
router.include_router(_security_router.router)
