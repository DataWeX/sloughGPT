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

        # Batch training status
        self.router.add_api_route(
            "/training/batch-status", self.get_batch_training_status, methods=["GET"],
        )

        # Training presets
        self.router.add_api_route(
            "/training/presets", self.list_training_presets, methods=["GET"],
        )
        self.router.add_api_route(
            "/training/presets/{preset_name}", self.get_training_preset, methods=["GET"],
        )
        self.router.add_api_route(
            "/training/presets/{preset_name}/apply", self.apply_training_preset, methods=["POST"],
        )

        # Training history export
        self.router.add_api_route(
            "/training/history/export", self.export_training_history, methods=["GET"],
        )

        # Model card generation
        self.router.add_api_route(
            "/model-card", self.generate_model_card, methods=["POST"],
        )

        # Training run comparison
        self.router.add_api_route(
            "/training/compare", self.compare_training_runs, methods=["GET"],
        )

        # Training run management
        self.router.add_api_route(
            "/training/runs", self.filter_training_runs, methods=["GET"],
        )
        self.router.add_api_route(
            "/training/runs/{run_id}", self.get_training_run, methods=["GET"],
        )
        self.router.add_api_route(
            "/training/runs/{run_id}", self.delete_training_run, methods=["DELETE"],
        )
        self.router.add_api_route(
            "/training/history/clear", self.clear_training_history, methods=["POST"],
        )

        # Training run tags and notes
        self.router.add_api_route(
            "/training/runs/{run_id}/tags", self.add_run_tag, methods=["POST"],
        )
        self.router.add_api_route(
            "/training/runs/{run_id}/tags/{tag}", self.remove_run_tag, methods=["DELETE"],
        )
        self.router.add_api_route(
            "/training/runs/{run_id}/notes", self.set_run_notes, methods=["PUT"],
        )
        self.router.add_api_route(
            "/training/tags", self.get_all_tags, methods=["GET"],
        )
        self.router.add_api_route(
            "/training/tags/{tag}", self.get_runs_by_tag, methods=["GET"],
        )
        self.router.add_api_route(
            "/training/runs/{run_id}/export", self.export_training_run, methods=["GET"],
        )

        # Training run bookmarks
        self.router.add_api_route(
            "/training/runs/{run_id}/bookmark", self.toggle_bookmark, methods=["POST"],
        )
        self.router.add_api_route(
            "/training/bookmarks", self.get_bookmarked_runs, methods=["GET"],
        )

        # Training run duplicate
        self.router.add_api_route(
            "/training/runs/{run_id}/duplicate", self.duplicate_training_run, methods=["POST"],
        )

        # Bulk operations
        self.router.add_api_route(
            "/training/runs/bulk/delete", self.bulk_delete_runs, methods=["POST"],
        )
        self.router.add_api_route(
            "/training/runs/bulk/tag", self.bulk_add_tag, methods=["POST"],
        )
        self.router.add_api_route(
            "/training/runs/bulk/bookmark", self.bulk_bookmark, methods=["POST"],
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

    @endpoint("settings.get_batch_training_status")
    async def get_batch_training_status(self) -> dict:
        """Get status of all training jobs (running, queued, completed, failed)."""
        try:
            from training.jobs import training_jobs
        except ImportError:
            return success_response(data={"jobs": [], "summary": {"total": 0, "running": 0, "queued": 0, "completed": 0, "failed": 0}})

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
            jobs.append({
                "job_id": job_id,
                "name": job.get("name", job_id[:8]),
                "model": job.get("model", ""),
                "dataset": job.get("dataset", ""),
                "status": status,
                "progress": job.get("progress", 0),
                "current_step": job.get("current_step", ""),
                "total_steps": job.get("total_steps", ""),
                "created_at": job.get("created_at", 0),
            })
        return success_response(data={"jobs": jobs, "summary": summary})

    @endpoint("settings.export_training_history")
    async def export_training_history(
        self,
        format: str = "json",
        limit: int = 0,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Export training history as JSON or CSV."""
        from domains.training.outcome_tracker import TrainingOutcomeTracker
        tracker = TrainingOutcomeTracker()
        if format == "csv":
            csv_data = tracker.export_csv(limit=limit)
            return success_response(data={"format": "csv", "content": csv_data, "count": len(csv_data.splitlines()) - 1})
        return success_response(data={"format": "json", "outcomes": tracker.export_json(limit=limit), "count": len(tracker.export_json(limit=limit))})

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
        from domains.training.model_card import generate_model_card as _generate
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
        from domains.training.outcome_tracker import TrainingOutcomeTracker
        tracker = TrainingOutcomeTracker()
        result = tracker.compare(run_a, run_b)
        if result is None:
            return success_response(data={"error": "One or both run IDs not found"})
        return success_response(data=result)

    @endpoint("settings.list_training_presets")
    async def list_training_presets(self) -> dict:
        """List all available training presets."""
        from domains.training.presets import list_presets
        return success_response(data={"presets": list_presets()})

    @endpoint("settings.get_training_preset")
    async def get_training_preset(self, preset_name: str) -> dict:
        """Get a specific training preset."""
        from domains.training.presets import get_preset
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
        from domains.training.presets import apply_preset
        config = apply_preset(preset_name)
        if not config:
            return success_response(data={"error": f"Preset '{preset_name}' not found"})
        from domains.settings.persistent import get_settings as _get
        ps = _get()
        ps.update("training", **config)
        safe_audit_log("settings.apply_preset", resource="training", detail=preset_name)
        from dataclasses import asdict
        return success_response(data={"preset": preset_name, "applied": config, "settings": asdict(ps.settings.training)})

    @endpoint("settings.get_training_run")
    async def get_training_run(self, run_id: str) -> dict:
        """Get a single training run by ID."""
        from domains.training.outcome_tracker import TrainingOutcomeTracker
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
        from domains.training.outcome_tracker import TrainingOutcomeTracker
        import json
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
        from domains.training.outcome_tracker import TrainingOutcomeTracker
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
        return success_response(data={
            "runs": [o.to_dict() for o in outcomes],
            "count": len(outcomes),
        })

    @endpoint("settings.clear_training_history")
    async def clear_training_history(
        self,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Clear all training history."""
        from domains.training.outcome_tracker import TrainingOutcomeTracker
        tracker = TrainingOutcomeTracker()
        count = tracker.clear()
        safe_audit_log("settings.clear_history", resource="training", detail=f"{count} runs removed")
        return success_response(data={"cleared": True, "removed_count": count})

    @endpoint("settings.add_run_tag")
    async def add_run_tag(
        self,
        run_id: str,
        tag: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Add a tag to a training run."""
        from domains.training.outcome_tracker import TrainingOutcomeTracker
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
        from domains.training.outcome_tracker import TrainingOutcomeTracker
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
        from domains.training.outcome_tracker import TrainingOutcomeTracker
        tracker = TrainingOutcomeTracker()
        run = tracker.set_notes(run_id, notes)
        if not run:
            return success_response(data={"error": f"Run '{run_id}' not found"})
        safe_audit_log("settings.set_notes", resource="training", detail=run_id)
        return success_response(data=run.to_dict())

    @endpoint("settings.get_all_tags")
    async def get_all_tags(self) -> dict:
        """Get all unique tags across all training runs."""
        from domains.training.outcome_tracker import TrainingOutcomeTracker
        tracker = TrainingOutcomeTracker()
        tags = tracker.get_all_tags()
        return success_response(data={"tags": tags})

    @endpoint("settings.get_runs_by_tag")
    async def get_runs_by_tag(self, tag: str) -> dict:
        """Get all training runs with a specific tag."""
        from domains.training.outcome_tracker import TrainingOutcomeTracker
        tracker = TrainingOutcomeTracker()
        runs = tracker.get_outcomes_by_tag(tag)
        return success_response(data={
            "runs": [o.to_dict() for o in runs],
            "count": len(runs),
            "tag": tag,
        })

    @endpoint("settings.export_training_run")
    async def export_training_run(self, run_id: str, format: str = "json") -> dict:
        """Export a single training run as JSON or YAML."""
        from domains.training.outcome_tracker import TrainingOutcomeTracker
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
        from domains.training.outcome_tracker import TrainingOutcomeTracker
        tracker = TrainingOutcomeTracker()
        run = tracker.toggle_bookmark(run_id)
        if not run:
            return success_response(data={"error": f"Run '{run_id}' not found"})
        safe_audit_log("settings.toggle_bookmark", resource="training", detail=run_id)
        return success_response(data=run.to_dict())

    @endpoint("settings.get_bookmarked_runs")
    async def get_bookmarked_runs(self) -> dict:
        """Get all bookmarked training runs."""
        from domains.training.outcome_tracker import TrainingOutcomeTracker
        tracker = TrainingOutcomeTracker()
        runs = tracker.get_bookmarked()
        return success_response(data={
            "runs": [o.to_dict() for o in runs],
            "count": len(runs),
        })

    @endpoint("settings.duplicate_training_run")
    async def duplicate_training_run(
        self,
        run_id: str,
        new_run_id: str = "",
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Duplicate a training run with a new ID."""
        from domains.training.outcome_tracker import TrainingOutcomeTracker
        tracker = TrainingOutcomeTracker()
        new_run = tracker.duplicate_run(run_id, new_run_id)
        if not new_run:
            return success_response(data={"error": f"Run '{run_id}' not found"})
        safe_audit_log("settings.duplicate_run", resource="training", detail=f"{run_id} -> {new_run.run_id}")
        return success_response(data=new_run.to_dict())

    @endpoint("settings.bulk_delete_runs")
    async def bulk_delete_runs(
        self,
        run_ids: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Delete multiple training runs. run_ids is a comma-separated list."""
        from domains.training.outcome_tracker import TrainingOutcomeTracker
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
        from domains.training.outcome_tracker import TrainingOutcomeTracker
        tracker = TrainingOutcomeTracker()
        ids = [x.strip() for x in run_ids.split(",") if x.strip()]
        count = tracker.bulk_add_tag(ids, tag)
        safe_audit_log("settings.bulk_add_tag", resource="training", detail=f"{count} runs tagged '{tag}'")
        return success_response(data={"updated_count": count, "tag": tag})

    @endpoint("settings.bulk_bookmark")
    async def bulk_bookmark(
        self,
        run_ids: str,
        bookmarked: bool = True,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Set bookmark status on multiple training runs."""
        from domains.training.outcome_tracker import TrainingOutcomeTracker
        tracker = TrainingOutcomeTracker()
        ids = [x.strip() for x in run_ids.split(",") if x.strip()]
        count = tracker.bulk_bookmark(ids, bookmarked)
        safe_audit_log("settings.bulk_bookmark", resource="training", detail=f"{count} runs bookmark={bookmarked}")
        return success_response(data={"updated_count": count, "bookmarked": bookmarked})


router = SettingsRouter().router
