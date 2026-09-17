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

    def list_checkpoints(self) -> TrainingResult:
        """List all saved checkpoints.

        Returns:
            TrainingResult with list of checkpoint info.
        """
        try:
            from domain.training import find_checkpoint

            checkpoints = find_checkpoint("*") if callable(find_checkpoint) else []
            return TrainingResult(success=True, data=checkpoints)
        except Exception as e:
            logger.error("List checkpoints failed: %s", e)
            return TrainingResult(success=False, error=str(e))

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

    def extract_pairs_from_sessions(self, sessions: list[dict]) -> TrainingResult:
        """Extract training pairs from session data."""
        try:
            from domain.training._internal.pair_extractor import extract_pairs_from_sessions

            pairs = extract_pairs_from_sessions(sessions)
            return TrainingResult(success=True, data=pairs)
        except Exception as e:
            logger.error("Extract pairs from sessions failed: %s", e)
            return TrainingResult(success=False, error=str(e))

    def extract_pairs_from_logs(self, logs: list[str]) -> TrainingResult:
        """Extract training pairs from log data."""
        try:
            from domain.training._internal.pair_extractor import extract_pairs_from_logs

            pairs = extract_pairs_from_logs(logs)
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


_engine: TrainingEngine | None = None


def get_training_engine() -> TrainingEngine:
    """Singleton accessor for TrainingEngine."""
    global _engine
    if _engine is None:
        _engine = TrainingEngine()
    return _engine
