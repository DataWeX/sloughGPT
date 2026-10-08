"""TrainingEngine — unified feature facade for training capabilities.

Wraps DatasetManager, TrainingPipeline, ModelManager, and checkpoint
operations into a single cohesive API. Routers should use this, not
import from domain.training._internal directly.

Usage:
    from domain.training.engine import TrainingEngine

    engine = TrainingEngine()
    datasets = engine.list_datasets()
    job = engine.start_training(config)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("slo.training.engine")


@dataclass
class TrainingResult:
    """Result from a training operation."""

    success: bool
    data: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainingEngine:
    """Unified training feature API.

    Provides a single entry point for all training operations:
    - Dataset management (list, import, stats)
    - Training execution (start, stop, status)
    - Checkpoint management (list, load, delete)
    - Model operations (load, compare)
    """

    def __init__(self) -> None:
        self._dataset_manager: Any = None
        self._model_manager: Any = None

    def _get_dataset_manager(self) -> Any:
        if self._dataset_manager is None:
            from domain.training import DatasetManager

            self._dataset_manager = DatasetManager()
        return self._dataset_manager

    def _get_model_manager(self) -> Any:
        if self._model_manager is None:
            from domain.training import ModelManager

            self._model_manager = ModelManager()
        return self._model_manager

    # ── Dataset operations ──

    def list_datasets(self) -> TrainingResult:
        """List all available datasets.

        Returns:
            TrainingResult with list of dataset info dicts.
        """
        try:
            mgr = self._get_dataset_manager()
            datasets = mgr.list_datasets() if hasattr(mgr, "list_datasets") else []
            return TrainingResult(success=True, data=datasets)
        except Exception as e:
            logger.error("List datasets failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def get_dataset(self, name: str) -> TrainingResult:
        """Get dataset details by name.

        Args:
            name: Dataset name.

        Returns:
            TrainingResult with dataset info.
        """
        try:
            mgr = self._get_dataset_manager()
            ds = mgr.get_dataset(name) if hasattr(mgr, "get_dataset") else None
            if ds is None:
                return TrainingResult(success=False, error=f"Dataset not found: {name}")
            return TrainingResult(success=True, data=ds)
        except Exception as e:
            logger.error("Get dataset failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def import_dataset(self, source: str, name: str | None = None, **kwargs: Any) -> TrainingResult:
        """Import a dataset from a source.

        Args:
            source: Import source (URL, path, HuggingFace ID, etc.).
            name: Optional name for the dataset.

        Returns:
            TrainingResult with import status.
        """
        try:
            mgr = self._get_dataset_manager()
            result = (
                mgr.import_dataset(source, name=name, **kwargs)
                if hasattr(mgr, "import_dataset")
                else None
            )
            return TrainingResult(
                success=True,
                data=result,
                metadata={"source": source, "name": name},
            )
        except Exception as e:
            logger.error("Import dataset failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    # ── Training execution ──

    def start_training(self, config: dict[str, Any]) -> TrainingResult:
        """Start a training job.

        Args:
            config: Training configuration (dataset, epochs, hyperparams).

        Returns:
            TrainingResult with job info.
        """
        try:
            from domain.training import PipelineConfig, TrainingPipeline

            pipeline_config = (
                PipelineConfig(**config) if hasattr(PipelineConfig, "__init__") else config
            )
            pipeline = TrainingPipeline(pipeline_config)
            job = pipeline.start() if hasattr(pipeline, "start") else {"status": "started"}
            return TrainingResult(
                success=True,
                data=job,
                metadata={"config": config},
            )
        except Exception as e:
            logger.error("Start training failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def get_status(self) -> TrainingResult:
        """Get current training status.

        Returns:
            TrainingResult with status dict.
        """
        try:
            return TrainingResult(
                success=True,
                data={"status": "idle", "running": False},
            )
        except Exception as e:
            logger.error("Get training status failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    # ── Checkpoint operations ──

    async def list_checkpoints(self) -> list[dict]:
        """List all saved checkpoints (delegates to the checkpoint service scan).

        Returns:
            List of checkpoint info dicts (async — service scans in a thread).
        """
        from domain.training._internal.service import list_checkpoints

        return await list_checkpoints()

    def load_checkpoint(self, name: str) -> TrainingResult:
        """Load a checkpoint by name.

        Args:
            name: Checkpoint name or path.

        Returns:
            TrainingResult with checkpoint data.
        """
        try:
            from domain.training import load_soul

            soul = load_soul(name)
            return TrainingResult(
                success=True,
                data=soul,
                metadata={"name": name},
            )
        except Exception as e:
            logger.error("Load checkpoint failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    # ── Model operations ──

    def list_models(self) -> TrainingResult:
        """List available models.

        Returns:
            TrainingResult with list of model info.
        """
        try:
            mgr = self._get_model_manager()
            models = mgr.list_models() if hasattr(mgr, "list_models") else []
            return TrainingResult(success=True, data=models)
        except Exception as e:
            logger.error("List models failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def get_model(self, name: str) -> TrainingResult:
        """Get model details by name.

        Args:
            name: Model name.

        Returns:
            TrainingResult with model info.
        """
        try:
            mgr = self._get_model_manager()
            model = mgr.get_model(name) if hasattr(mgr, "get_model") else None
            if model is None:
                return TrainingResult(success=False, error=f"Model not found: {name}")
            return TrainingResult(success=True, data=model)
        except Exception as e:
            logger.error("Get model failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def status(self) -> dict[str, Any]:
        """Get training engine status.

        Returns:
            Dict with capability status.
        """
        return {
            "dataset_manager": self._dataset_manager is not None,
            "model_manager": self._model_manager is not None,
            "capabilities": ["datasets", "training", "checkpoints", "models", "imports", "search"],
        }

    # ── Import operations ──

    def import_from_url(self, url: str, name: str | None = None, **kwargs: Any) -> TrainingResult:
        """Import a dataset from a URL."""
        try:
            from domain.training import URLImporter

            importer = URLImporter()
            result = importer.import_url(url, name=name, **kwargs)
            return TrainingResult(
                success=True,
                data=result,
                metadata={"url": url, "name": name},
            )
        except Exception as e:
            logger.error("URL import failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def import_from_github(
        self, repo: str, path: str | None = None, **kwargs: Any
    ) -> TrainingResult:
        """Import a dataset from a GitHub repo."""
        try:
            from domain.training import RepoImporter

            importer = RepoImporter()
            result = importer.import_repo(repo, path=path, **kwargs)
            return TrainingResult(
                success=True,
                data=result,
                metadata={"repo": repo, "path": path},
            )
        except Exception as e:
            logger.error("GitHub import failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def import_from_huggingface(self, dataset_id: str, **kwargs: Any) -> TrainingResult:
        """Import a dataset from HuggingFace."""
        try:
            from domain.training import HuggingFaceImporter

            importer = HuggingFaceImporter()
            result = importer.import_dataset(dataset_id, **kwargs)
            return TrainingResult(
                success=True,
                data=result,
                metadata={"dataset_id": dataset_id},
            )
        except Exception as e:
            logger.error("HuggingFace import failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def search_books(self, query: str, **kwargs: Any) -> TrainingResult:
        """Search for books as training data."""
        try:
            from domain.training import BooksSearch

            searcher = BooksSearch()
            results = searcher.search(query, **kwargs)
            return TrainingResult(success=True, data=results)
        except Exception as e:
            logger.error("Book search failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def search_github(self, query: str, **kwargs: Any) -> TrainingResult:
        """Search GitHub for datasets."""
        try:
            from domain.training import GitHubSearch

            searcher = GitHubSearch()
            results = searcher.search(query, **kwargs)
            return TrainingResult(success=True, data=results)
        except Exception as e:
            logger.error("GitHub search failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def import_isbn(self, isbn: str, name: str | None = None, **kwargs: Any) -> TrainingResult:
        """Import a book by ISBN.

        Args:
            isbn: ISBN-10 or ISBN-13.
            name: Optional dataset name (defaults to book_{isbn}).

        Returns:
            TrainingResult with import status.
        """
        try:
            from domain.training import ISBNImporter

            importer = ISBNImporter()
            result = importer.import_from_isbn(isbn, name=name or f"book_{isbn}")
            return TrainingResult(
                success=result.success,
                data={
                    "name": result.name,
                    "files_imported": result.files_imported,
                    "total_chars": result.total_chars,
                    "output_path": result.output_path,
                },
                error=result.error,
                metadata={"isbn": isbn},
            )
        except Exception as e:
            logger.error("ISBN import failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def score_quality(self, data: list[dict]) -> TrainingResult:
        """Score data quality for a dataset."""
        try:
            from domain.training._internal.quality_scorer import compute_data_quality

            scores = compute_data_quality(data)
            return TrainingResult(success=True, data=scores)
        except Exception as e:
            logger.error("Quality scoring failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    # ── Mobile Training Pair CRUD ────────────────────────────────────────

    def get_training_store(self) -> Any:
        """Get the mobile training store singleton."""
        from domain.training._internal.mobile_training_store import get_training_store

        return get_training_store()

    def add_training_pairs(self, pairs: list[dict], topic: str = "mobile") -> TrainingResult:
        """Add training pairs to the mobile store."""
        try:
            store = self.get_training_store()
            count = store.add_batch(pairs, topic=topic)
            return TrainingResult(success=True, data={"added": count})
        except Exception as e:
            logger.error("Add training pairs failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def list_training_pairs(self, limit: int = 100, offset: int = 0, **kwargs) -> TrainingResult:
        """List training pairs from the mobile store."""
        try:
            store = self.get_training_store()
            pairs = store.list_pairs(limit=limit, offset=offset, **kwargs)
            count = store.count()
            return TrainingResult(success=True, data={"pairs": pairs, "total": count})
        except Exception as e:
            logger.error("List training pairs failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def get_pending_pairs(self, limit: int = 50) -> TrainingResult:
        """Get pending training pairs."""
        try:
            store = self.get_training_store()
            pairs = store.get_pending_pairs(limit=limit)
            return TrainingResult(success=True, data=pairs)
        except Exception as e:
            logger.error("Get pending pairs failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def get_session_pairs(self, session_id: str) -> TrainingResult:
        """Get training pairs for a session."""
        try:
            store = self.get_training_store()
            pairs = store.get_pairs_by_session(session_id)
            return TrainingResult(success=True, data=pairs)
        except Exception as e:
            logger.error("Get session pairs failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def update_pair_quality(self, pair_id: str, quality: float) -> TrainingResult:
        """Update quality score for a training pair."""
        try:
            store = self.get_training_store()
            store.update_quality(pair_id, quality)
            return TrainingResult(success=True, data={"updated": True})
        except Exception as e:
            logger.error("Update pair quality failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def delete_pair(self, pair_id: str) -> TrainingResult:
        """Delete a training pair."""
        try:
            store = self.get_training_store()
            store.delete_pair(pair_id)
            return TrainingResult(success=True, data={"deleted": True})
        except Exception as e:
            logger.error("Delete pair failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def delete_synced_pairs(self) -> TrainingResult:
        """Delete all synced training pairs."""
        try:
            store = self.get_training_store()
            count = store.delete_synced()
            return TrainingResult(success=True, data={"deleted": count})
        except Exception as e:
            logger.error("Delete synced pairs failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def compact_training_store(self) -> TrainingResult:
        """Compact the training store."""
        try:
            store = self.get_training_store()
            result = store.compact()
            return TrainingResult(success=True, data=result)
        except Exception as e:
            logger.error("Compact training store failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def training_store_stats(self) -> TrainingResult:
        """Get training store statistics."""
        try:
            store = self.get_training_store()
            stats = store.stats()
            breakdown = store.quality_breakdown()
            return TrainingResult(success=True, data={**stats, "quality_breakdown": breakdown})
        except Exception as e:
            logger.error("Training store stats failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def extract_pairs_from_sessions(
        self,
        limit: int = 50,
        min_length: int = 5,
        session_ids: list[str] | None = None,
    ) -> TrainingResult:
        """Extract training pairs from session data."""
        try:
            from domain.training._internal.pair_extractor import extract_pairs_from_sessions

            pairs = extract_pairs_from_sessions(
                limit=limit, min_length=min_length, session_ids=session_ids
            )
            return TrainingResult(success=True, data=pairs)
        except Exception as e:
            logger.error("Extract pairs from sessions failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def extract_pairs_from_logs(
        self,
        limit: int = 100,
        min_length: int = 5,
        model: str | None = None,
    ) -> TrainingResult:
        """Extract training pairs from log data."""
        try:
            from domain.training._internal.pair_extractor import extract_pairs_from_logs

            pairs = extract_pairs_from_logs(limit=limit, min_length=min_length, model=model)
            return TrainingResult(success=True, data=pairs)
        except Exception as e:
            logger.error("Extract pairs from logs failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def score_pairs(self, pairs: list[dict]) -> TrainingResult:
        """Score a batch of training pairs for quality."""
        try:
            from domain.training._internal.quality_scorer import score_batch

            scores = score_batch(pairs)
            return TrainingResult(success=True, data=scores)
        except Exception as e:
            logger.error("Score pairs failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def get_auto_trainer_status(self) -> TrainingResult:
        """Get auto-trainer status."""
        try:
            from domain.training._internal.auto_trainer import get_auto_trainer

            trainer = get_auto_trainer()
            return TrainingResult(success=True, data=trainer.status())
        except Exception as e:
            logger.error("Auto-trainer status failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def update_auto_trainer_config(self, **kwargs) -> TrainingResult:
        """Update auto-trainer configuration."""
        try:
            from domain.training._internal.auto_trainer import get_auto_trainer

            trainer = get_auto_trainer()
            for key, val in kwargs.items():
                if hasattr(trainer, key):
                    setattr(trainer, key, val)
            return TrainingResult(success=True, data=trainer.status())
        except Exception as e:
            logger.error("Auto-trainer config update failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    # ── Cloud training jobs ──

    @staticmethod
    def _get_cloud_provider(provider: str = "local") -> Any:
        from domain.training._internal.cloud import get_provider

        return get_provider(provider)

    @staticmethod
    def _job_to_dict(j: Any) -> dict[str, Any]:
        return {
            "job_id": j.job_id,
            "provider": j.provider,
            "status": j.status,
            "progress": j.progress,
            "error": j.error,
        }

    def list_cloud_jobs(self, limit: int = 10) -> TrainingResult:
        """List recent cloud training jobs."""
        try:
            jobs = self._get_cloud_provider("local").list_jobs(limit)
            return TrainingResult(success=True, data={"jobs": [self._job_to_dict(j) for j in jobs]})
        except Exception as e:
            logger.error("List cloud jobs failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def submit_cloud_job(self, provider: str = "local", dataset_id: str = "") -> TrainingResult:
        """Submit a cloud training job."""
        try:
            from domain.training._internal.cloud import CloudTrainingConfig

            p = self._get_cloud_provider(provider)
            config = CloudTrainingConfig(provider=provider)
            job_id = p.submit_job(config, dataset_id, "train.py", {})
            return TrainingResult(
                success=True,
                data={"job_id": job_id, "provider": provider, "status": "submitted"},
            )
        except Exception as e:
            logger.error("Submit cloud job failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def cloud_job_status(self, job_id: str) -> TrainingResult:
        """Get status of a cloud training job."""
        try:
            status = self._get_cloud_provider("local").get_status(job_id)
            return TrainingResult(success=True, data=self._job_to_dict(status))
        except Exception as e:
            logger.error("Cloud job status failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def cancel_cloud_job(self, job_id: str) -> TrainingResult:
        """Cancel a cloud training job."""
        try:
            cancelled = self._get_cloud_provider("local").cancel_job(job_id)
            return TrainingResult(success=True, data={"job_id": job_id, "cancelled": cancelled})
        except Exception as e:
            logger.error("Cancel cloud job failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    # ── Executor (job pool) ─────────────────────────────────────────────
    # Thin delegations: unexpected exceptions propagate so router-level
    # classify_and_raise keeps mapping error types to HTTP codes.

    @staticmethod
    def _get_executor() -> Any:
        """Return the initialized executor singleton, or None (never creates)."""
        from domain.training._internal import executor as executor_mod

        return executor_mod._instance

    def executor_status(self) -> TrainingResult:
        """Executor pool status; always succeeds (initialized=False when absent)."""
        inst = self._get_executor()
        if inst is None:
            return TrainingResult(
                success=True,
                data={
                    "initialized": False,
                    "active_jobs": 0,
                    "max_workers": 0,
                    "total_tracked": 0,
                    "jobs": [],
                },
            )
        return TrainingResult(
            success=True,
            data={
                "initialized": True,
                "active_jobs": inst.active_count(),
                "max_workers": inst.max_workers,
                "total_tracked": inst.job_count,
                "jobs": inst.list_jobs(),
            },
        )

    def executor_job(self, job_id: str) -> TrainingResult:
        """Metadata for a single training job by ID."""
        inst = self._get_executor()
        if inst is None:
            return TrainingResult(
                success=False,
                error="executor not initialized",
                metadata={"code": "E_INFRA_STARTUP"},
            )
        status = inst.status(job_id)
        if status is None:
            return TrainingResult(
                success=False,
                error=f"job {job_id} not found",
                metadata={"code": "E_NOT_FOUND"},
            )
        return TrainingResult(success=True, data=status)

    def executor_job_result(self, job_id: str) -> TrainingResult:
        """Shape/dtype summary for a completed job's trained weights."""
        inst = self._get_executor()
        if inst is None:
            return TrainingResult(
                success=False,
                error="executor not initialized",
                metadata={"code": "E_INFRA_STARTUP"},
            )
        summary = inst.result_summary(job_id)
        if summary is None:
            if inst.status(job_id) is None:
                return TrainingResult(
                    success=False,
                    error=f"job {job_id} not found",
                    metadata={"code": "E_NOT_FOUND"},
                )
            return TrainingResult(
                success=False,
                error="job not completed or has no weight result",
                metadata={"code": "E_DOMAIN"},
            )
        return TrainingResult(success=True, data=summary)

    def purge_executor_jobs(self, max_age_s: float = 3600.0) -> TrainingResult:
        """Remove completed/failed/cancelled jobs older than max_age_s."""
        inst = self._get_executor()
        if inst is None:
            return TrainingResult(success=True, data={"purged": 0})
        purged = inst.purge_completed(max_age_s=max_age_s)
        return TrainingResult(success=True, data={"purged": purged})

    def cancel_executor_job(self, job_id: str) -> TrainingResult:
        """Request cancellation for a training job."""
        inst = self._get_executor()
        if inst is None:
            return TrainingResult(
                success=True,
                data={"cancelled": False, "reason": "executor not initialized"},
            )
        cancelled = inst.cancel(job_id)
        return TrainingResult(success=True, data={"cancelled": cancelled})

    # ── Model export ────────────────────────────────────────────────────

    def export_model(
        self,
        *,
        model: Any,
        tokenizer: Any,
        output_path: str | None = None,
        format: str = "gguf",
        include_tokenizer: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """Export a model to file. Exceptions propagate to the caller."""
        from domain.training._internal.export import ExportConfig
        from domain.training._internal.export import export_model as do_export

        config = ExportConfig(
            input_path="current",
            output_path=output_path,
            format=format,
            include_tokenizer=include_tokenizer,
            metadata=metadata or {},
        )
        files = do_export(config, model, tokenizer)
        return TrainingResult(success=True, data=files)

    def list_export_formats(self) -> TrainingResult:
        """List supported export formats."""
        from domain.training._internal.export import list_export_formats

        return TrainingResult(success=True, data=list_export_formats())

    # ── Auto-train turbo / outcomes ─────────────────────────────────────

    def turbo_state(self) -> TrainingResult:
        """Snapshot of the auto-train turbo state (lock held while copying)."""
        from domain.training._internal.service import get_turbo_lock, get_turbo_state

        with get_turbo_lock():
            return TrainingResult(success=True, data=dict(get_turbo_state()))

    def outcome_stats(self) -> TrainingResult:
        """Aggregate training-outcome statistics."""
        from domain.training._internal.outcome_tracker import TrainingOutcomeTracker

        return TrainingResult(success=True, data=TrainingOutcomeTracker().get_stats())

    # ── Checkpoint service / advisor / monitor (router delegation) ──
    #
    # Raw-passthrough delegates for training/router.py endpoints: the
    # underlying service raises on failure and the router/global handler
    # classifies — exceptions bubble by design (no TrainingResult envelope).

    async def get_log(self) -> list[str]:
        """Training log lines."""
        from domain.training._internal.service import get_log

        return await get_log()

    async def delete_checkpoint(self, name: str, path: str | None = None) -> list[str]:
        """Delete one checkpoint: by the row's `path` when the caller has it."""
        from domain.training._internal.service import delete_checkpoint

        return await delete_checkpoint(name, path=path)

    def is_valid_checkpoint_name(self, name: str) -> bool:
        """True if the checkpoint name matches the canonical validator."""
        from domain.training._internal.state import VALID_CKPT_NAME

        return bool(VALID_CKPT_NAME.match(name))

    async def checkpoint_load(self, name: str, path: str | None = None) -> Any:
        """Load and register a checkpoint for serving."""
        from domain.training._internal.service import load_checkpoint

        return await load_checkpoint(name, path=path)

    async def checkpoint_download_path(self, name: str, path: str | None = None) -> Any:
        """Filesystem path for downloading a checkpoint (None if missing)."""
        from domain.training._internal.service import download_checkpoint_path

        return await download_checkpoint_path(name, path=path)

    async def checkpoint_info(self, name: str, path: str | None = None) -> Any:
        """Describe a single checkpoint."""
        from domain.training._internal.service import checkpoint_info

        return await checkpoint_info(name, path=path)

    async def checkpoint_compare(
        self,
        a: str,
        b: str,
        prompt: str,
        max_new_tokens: int,
        path_a: str | None = None,
        path_b: str | None = None,
    ) -> Any:
        """Run the same prompt against two checkpoints."""
        from domain.training._internal.service import compare_checkpoints

        return await compare_checkpoints(a, b, prompt, max_new_tokens, path_a, path_b)

    async def checkpoint_all_data(self) -> Any:
        """Full metrics payload for every checkpoint (export)."""
        from domain.training._internal.service import get_all_checkpoint_data

        return await get_all_checkpoint_data()

    def training_state(self) -> Any:
        """Current training state (config, phase, progress)."""
        from domain.training._internal.state import get_state

        return get_state()

    def cache_root(self) -> Any:
        """Dataset cache root directory (path-like)."""
        from domain.training._internal.cache_tags import get_cache_root

        return get_cache_root()

    def find_corpus_file(self, path: Any) -> Any:
        """Locate the primary corpus file inside a dataset directory."""
        from domain.training._internal.cache_tags import find_corpus_file

        return find_corpus_file(path)

    def training_tips(self, dataset_size: int = 0) -> list[str]:
        """Plain-language training tips for a dataset size."""
        from domain.training._internal.training_advisor import get_training_tips

        return get_training_tips(dataset_size=dataset_size)

    def recommend_training_config(
        self,
        *,
        dataset_size: int = 0,
        method: str = "distill",
        avg_quality: float | None = None,
    ) -> Any:
        """Recommended training configuration for a dataset."""
        from domain.training._internal.training_advisor import recommend_training_config

        return recommend_training_config(
            dataset_size=dataset_size,
            method=method,
            avg_quality=avg_quality,
        )

    def list_outcomes(self) -> list[Any]:
        """Raw training-outcome records (unfiltered, unsorted)."""
        from domain.training._internal.outcome_tracker import TrainingOutcomeTracker

        return TrainingOutcomeTracker().load_outcomes()

    # ── Training monitor ──

    def monitor_status(self) -> dict[str, Any]:
        """Monitor status (health flags for loss/resource issues)."""
        from domain.training._internal.monitor import get_training_monitor

        return get_training_monitor().get_status()

    def monitor_alerts(self, severity: str | None = None, limit: int = 50) -> dict:
        """Alerts filtered by severity string (info/warning/error/critical)."""
        from domain.training._internal.monitor import AlertSeverity, get_training_monitor

        severity_filter = AlertSeverity(severity) if severity else None
        alerts = get_training_monitor().get_alerts(severity=severity_filter, limit=limit)
        return {"alerts": [a.to_dict() for a in alerts], "total": len(alerts)}

    def monitor_metrics(self, limit: int = 100) -> dict:
        """Recent metric snapshots."""
        from domain.training._internal.monitor import get_training_monitor

        metrics = get_training_monitor().get_metrics_history(limit=limit)
        return {"metrics": [m.to_dict() for m in metrics], "total": len(metrics)}

    def monitor_reset(self) -> dict[str, str]:
        """Clear monitor alerts/metrics."""
        from domain.training._internal.monitor import get_training_monitor

        get_training_monitor().reset()
        return {"message": "Monitor reset"}

    def monitor_resources(self) -> dict[str, Any]:
        """CPU/memory/GPU usage snapshot with threshold alerts."""
        from domain.training._internal.monitor import get_training_monitor

        return get_training_monitor().check_resources()

    def resolve_resume_checkpoint(
        self, checkpoint_dir: str, checkpoint_path: str, stem_prefixes: list[str]
    ) -> tuple[str | None, Any]:
        """Resolve (path, bundle) for a recovery run.

        Recorded path: validated and loaded exactly once; missing, unsupported,
        or unreadable raise ValueError with the user-facing message (router
        maps it to 422). No recorded path: scan only the job's own stems for
        the newest loadable checkpoint — a shared dir must never silently
        adopt another job's weights.
        """
        from domain.training._internal.train_pipeline import CheckpointManager

        manager = CheckpointManager(checkpoint_dir)
        if checkpoint_path:
            if not CheckpointManager.is_resumable(checkpoint_path):
                raise ValueError(
                    f"Cannot resume from '{checkpoint_path}': checkpoint missing or "
                    "unsupported (use a .soul or .npz file)"
                )
            try:
                bundle = CheckpointManager.load_from_path(checkpoint_path)
            except Exception as exc:
                raise ValueError(
                    f"Cannot resume from '{checkpoint_path}': checkpoint is unreadable ({exc})"
                )
            if bundle is None:
                raise ValueError(
                    f"Cannot resume from '{checkpoint_path}': checkpoint missing or "
                    "unsupported (use a .soul or .npz file)"
                )
            return checkpoint_path, bundle
        for stem in stem_prefixes:
            path, bundle = manager.load_latest_with_path(stem_prefix=stem)
            if bundle is not None:
                return path, bundle
        return None, None

    def run_recovery_training(
        self,
        trainer_config: dict[str, Any],
        *,
        on_progress: Any,
        resume_checkpoint: Any,
        cancel_event: Any,
        pause_event: Any,
    ) -> Any:
        """Build the trainer from the original job's config and run it resumed.

        Returns the trainer so the caller can read the produced checkpoint
        path; on_progress/cancel/pause stay caller-owned (job-record state).
        """
        from domain.training._internal.train_pipeline import SloughGPTTrainer

        trainer = SloughGPTTrainer(**trainer_config)
        trainer.train(
            on_progress=on_progress,
            resume=True,
            resume_checkpoint=resume_checkpoint,
            cancel_event=cancel_event,
            pause_event=pause_event,
        )
        return trainer

    def executor_submit(self, fn: Any, *args: Any) -> None:
        """Submit a background run to the training executor."""
        from domain.training._internal import executor as executor_mod

        executor_mod.get_training_executor().submit(fn, *args)


_engine: TrainingEngine | None = None


def get_training_engine() -> TrainingEngine:
    """Singleton accessor for TrainingEngine."""
    global _engine
    if _engine is None:
        _engine = TrainingEngine()
    return _engine
