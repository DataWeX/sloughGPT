"""
Training Data Pipeline — conversation-to-training conversion.

Design principles:
1. Separate raw data from processed training data
2. Version control for training datasets
3. Quality scoring and filtering
4. Audit trail for reproducibility
5. Incremental updates for continuous learning

Storage: MogDB (``data/training_pipeline.db``) with three collections —
``conversations`` (raw), ``training_pairs`` (processed), ``training_runs``
(history). Legacy JSON ``conversations.db`` / ``training_pairs.db`` /
``training_runs.db`` files are migrated automatically on init and removed.

Feedback ratings use the feedback-domain vocabulary ``"thumbs_up"`` /
``"thumbs_down"`` / ``"neutral"`` (None = unrated), matching
``domains/feedback``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mogdb import MogDB

logger = logging.getLogger("slo.infrastructure.training_pipeline")

FEEDBACK_UP = "thumbs_up"
FEEDBACK_DOWN = "thumbs_down"
FEEDBACK_NEUTRAL = "neutral"
NEUTRAL_QUALITY = 0.5
GOOD_QUALITY = 0.8
BAD_QUALITY = 0.3


@dataclass
class Conversation:
    """Raw conversation record."""

    id: str
    session_id: str
    user_message: str
    assistant_message: str
    model: str
    timestamp: str
    tokens: Optional[int] = None
    feedback: Optional[str] = None  # FEEDBACK_UP, FEEDBACK_DOWN, FEEDBACK_NEUTRAL, or None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingPair:
    """Processed training pair ready for fine-tuning."""

    id: str
    conversation_id: str
    prompt: str
    response: str
    quality_score: float  # 0.0 to 1.0
    feedback: Optional[str]
    created_at: str
    used_in_training: bool = False
    training_run_id: Optional[str] = None


@dataclass
class TrainingRun:
    """Record of a training run."""

    id: str
    created_at: str
    dataset_version: str
    pairs_count: int
    model_used: str
    status: str  # "pending", "running", "completed", "failed"
    metrics: Dict[str, Any] = field(default_factory=dict)


class TrainingDataPipeline:
    """
    Production-grade training data pipeline.

    Architecture:
        /data
        /training_pipeline.db   <- MogDB store
            /conversations.journal.jsonl    <- Raw conversation storage
            /training_pairs.journal.jsonl   <- Processed training pairs
            /training_runs.journal.jsonl    <- Training run history
        /exports/              <- Exported datasets
            /v1.0.jsonl
            /v1.1.jsonl
        /backups/              <- DB backups

    Legacy JSON ``conversations.db`` / ``training_pairs.db`` /
    ``training_runs.db`` siblings are migrated into MogDB on first init.
    """

    VERSION = "1.0"

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "training_pipeline.db"
        self.exports_dir = self.data_dir / "exports"
        self.backups_dir = self.data_dir / "backups"

        self._lock = threading.RLock()

        self._ensure_directories()
        self._init_dbs()

    def _ensure_directories(self):
        """Create the exports and backups directories."""
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)

    def _init_dbs(self):
        """Open the MogDB store and migrate any legacy JSON databases."""
        self._db = MogDB(str(self.db_path))
        self._conversations = self._db.collection("conversations")
        self._training_pairs = self._db.collection("training_pairs")
        self._training_runs = self._db.collection("training_runs")
        self._migrate_from_json()

    def _read_json_db(self, path: Path) -> List[Dict[str, Any]]:
        """Read records from a legacy JSON database file.

        Args:
            path: Path to the legacy ``*.db`` JSON file.

        Returns:
            The ``records`` list; empty list for missing or corrupt files.
        """
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return []
        records = data.get("records", []) if isinstance(data, dict) else []
        return records if isinstance(records, list) else []

    def _migrate_from_json(self):
        """Migrate legacy JSON ``*.db`` files into MogDB collections.

        Each record is inserted with ``_id`` set to its ``id``; records
        already present are skipped. The legacy file is removed afterwards.

        Side effects:
            - writes migrated records into the MogDB collections
            - deletes the legacy ``*.db`` files after migration
        """
        for name, col in (
            ("conversations.db", self._conversations),
            ("training_pairs.db", self._training_pairs),
            ("training_runs.db", self._training_runs),
        ):
            path = self.data_dir / name
            if not path.is_file():
                continue
            records = self._read_json_db(path)
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                rec_id = rec.get("id")
                if not rec_id or col.find_one({"_id": rec_id}):
                    continue
                col.insert_one({**rec, "_id": rec_id})
            path.unlink(missing_ok=True)
            if records:
                logger.info(
                    "Migrated %d legacy records from %s into MogDB", len(records), name
                )

    @staticmethod
    def _to_model(cls, doc: Dict[str, Any]):
        """Reconstruct a dataclass from a MogDB document.

        Fields absent from the document are omitted so the dataclass default
        applies; required fields missing from legacy documents become ``None``.
        Meta keys (``_id``, ``_created``, ``_updated``) are ignored.
        """
        kwargs: Dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name in doc:
                kwargs[f.name] = doc[f.name]
            elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                kwargs[f.name] = None
        return cls(**kwargs)

    @staticmethod
    def _validate_feedback(feedback: Optional[str]):
        """Validate a feedback rating at the input boundary.

        Accepts the feedback-domain vocabulary (``thumbs_up``,
        ``thumbs_down``, ``neutral``) or None. Anything else raises
        ValueError so a mistyped rating fails loudly instead of being
        silently treated as neutral quality.

        Args:
            feedback: The rating to validate.

        Raises:
            ValueError: If feedback is not a recognized rating.
        """
        if feedback not in (None, FEEDBACK_UP, FEEDBACK_DOWN, FEEDBACK_NEUTRAL):
            raise ValueError(
                f"Invalid feedback rating {feedback!r}; expected "
                f"{FEEDBACK_UP!r}, {FEEDBACK_DOWN!r}, {FEEDBACK_NEUTRAL!r}, or None"
            )

    @staticmethod
    def _quality_for_feedback(feedback: Optional[str]) -> float:
        """Map a feedback rating to a quality score in 0.0-1.0."""
        if feedback == FEEDBACK_UP:
            return 1.0
        if feedback == FEEDBACK_DOWN:
            return 0.0
        return NEUTRAL_QUALITY

    @staticmethod
    def _score_quality(feedback: Optional[str], response: Any) -> float:
        """Derive the quality score for a pair from feedback and response.

        Empty responses are never useful for training, so they score 0.0
        regardless of feedback. Otherwise the rating maps to its quality.

        Args:
            feedback: The conversation's rating (or None).
            response: The assistant response text.

        Returns:
            Quality score in 0.0-1.0.
        """
        if not str(response or "").strip():
            return 0.0
        return TrainingDataPipeline._quality_for_feedback(feedback)

    # ============ Conversations ============

    def add_conversation(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        model: str,
        tokens: Optional[int] = None,
        metadata: Optional[Dict] = None,
        feedback: Optional[str] = None,
    ) -> Conversation:
        """Add a raw conversation and create its training pair.

        Args:
            session_id: Identifier for the chat session.
            user_message: The user's prompt text.
            assistant_message: The assistant's response text.
            model: Model identifier that produced the response.
            tokens: Token count of the response, if known.
            metadata: Optional extra context.
            feedback: Optional rating (FEEDBACK_UP, FEEDBACK_DOWN,
                FEEDBACK_NEUTRAL, or None).

        Returns:
            The stored Conversation.

        Raises:
            ValueError: If feedback is not a recognized rating.
            TypeError: If metadata is not a dict or None.

        Side effects:
            - inserts the conversation into the ``conversations`` collection
            - creates a corresponding TrainingPair with derived quality score
        """
        self._validate_feedback(feedback)
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict or None")
        with self._lock:
            conv_id = f"conv_{self._conversations.count()}_{int(datetime.now().timestamp() * 1000)}"
            conv = {
                "id": conv_id,
                "session_id": session_id,
                "user_message": user_message,
                "assistant_message": assistant_message,
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "tokens": tokens,
                "feedback": feedback,
                "metadata": metadata or {},
            }
            self._conversations.insert_one({**conv, "_id": conv_id})
            self._create_training_pair(conv)
            return Conversation(**conv)

    def add_feedback(self, conversation_id: str, feedback: Optional[str]) -> bool:
        """Add or update feedback on a conversation.

        Updates the conversation's ``feedback`` field and the quality score
        of its corresponding training pair.

        Args:
            conversation_id: Id of the conversation to rate.
            feedback: FEEDBACK_UP, FEEDBACK_DOWN, FEEDBACK_NEUTRAL, or None.

        Returns:
            True if the conversation was found, False otherwise.

        Raises:
            ValueError: If feedback is not a recognized rating.

        Side effects:
            - updates the conversation document and its training pair
        """
        self._validate_feedback(feedback)
        with self._lock:
            conv = self._conversations.find_one({"_id": conversation_id})
            if not conv:
                return False
            self._conversations.update_one(
                {"_id": conversation_id}, {"$set": {"feedback": feedback}}
            )
            self._update_pair_quality(conv, feedback)
            return True

    def get_conversations(
        self,
        session_id: Optional[str] = None,
        feedback: Optional[str] = None,
        limit: int = 100,
    ) -> List[Conversation]:
        """Get conversations with optional filters, newest appended last.

        Args:
            session_id: Filter to a single session when provided.
            feedback: Filter to a rating (FEEDBACK_UP/FEEDBACK_DOWN/
                FEEDBACK_NEUTRAL) when provided.
            limit: Maximum number of most-recent conversations to return.

        Returns:
            List of Conversation models in insertion order.
        """
        with self._lock:
            query: Dict[str, Any] = {}
            if session_id:
                query["session_id"] = session_id
            if feedback:
                query["feedback"] = feedback
            docs = self._conversations.find(query, sort=[("_created", 1)])
            if limit is not None:
                docs = docs[-limit:]
            return [self._to_model(Conversation, d) for d in docs]

    # ============ Training Pairs ============

    def _create_training_pair(self, conversation: Dict):
        """Create a training pair from a conversation with quality scoring.

        Args:
            conversation: The stored conversation document.

        Side effects:
            - inserts a TrainingPair into the ``training_pairs`` collection
        """
        quality = self._score_quality(
            conversation.get("feedback"), conversation.get("assistant_message")
        )

        pair_id = f"pair_{self._training_pairs.count()}_{int(datetime.now().timestamp() * 1000)}"
        pair = {
            "id": pair_id,
            "conversation_id": conversation["id"],
            "prompt": conversation["user_message"],
            "response": conversation["assistant_message"],
            "quality_score": quality,
            "feedback": conversation.get("feedback"),
            "created_at": datetime.now().isoformat(),
            "used_in_training": False,
            "training_run_id": None,
        }
        self._training_pairs.insert_one({**pair, "_id": pair_id})

    def _update_pair_quality(self, conversation: Dict, feedback: Optional[str]):
        """Update a pair's feedback and quality score.

        Uses the same scoring rules as pair creation, so an empty response
        stays at 0.0 even when rated.

        Args:
            conversation: The conversation document being rated.
            feedback: The new rating (FEEDBACK_UP, FEEDBACK_DOWN, or None).

        Side effects:
            - updates the matching training pair document
        """
        quality = self._score_quality(feedback, conversation.get("assistant_message"))
        self._training_pairs.update_one(
            {"conversation_id": conversation["id"]},
            {"$set": {"feedback": feedback, "quality_score": quality}},
        )

    def get_training_pairs(
        self,
        min_quality: float = 0.0,
        include_used: bool = True,
        limit: Optional[int] = None,
    ) -> List[TrainingPair]:
        """Get training pairs with quality filtering, newest appended last.

        Args:
            min_quality: Minimum quality score (0.0-1.0) to include.
            include_used: When False, exclude pairs already used in training.
            limit: Maximum number of most-recent pairs to return.

        Returns:
            List of TrainingPair models in insertion order.
        """
        with self._lock:
            docs = self._training_pairs.find(sort=[("_created", 1)])
            docs = [d for d in docs if d.get("quality_score", 0.0) >= min_quality]
            if not include_used:
                docs = [d for d in docs if not d.get("used_in_training", False)]
            if limit is not None:
                docs = docs[-limit:]
            return [self._to_model(TrainingPair, d) for d in docs]

    def mark_pairs_used(self, pair_ids: List[str], training_run_id: str):
        """Mark pairs as used in a training run.

        Args:
            pair_ids: Ids of the pairs to mark as used.
            training_run_id: Id of the training run that consumed them.

        Side effects:
            - sets ``used_in_training`` and ``training_run_id`` on each pair
        """
        with self._lock:
            for pair_id in pair_ids:
                self._training_pairs.update_one(
                    {"_id": pair_id},
                    {"$set": {"used_in_training": True, "training_run_id": training_run_id}},
                )

    # ============ Training Runs ============

    def create_training_run(
        self,
        dataset_version: str,
        pairs_count: int,
        model_used: str,
    ) -> TrainingRun:
        """Create a new training run record.

        Args:
            dataset_version: Version tag for the exported dataset.
            pairs_count: Number of pairs in the run.
            model_used: Model identifier the run trained.

        Returns:
            The stored TrainingRun (status ``"pending"``).

        Side effects:
            - inserts the run into the ``training_runs`` collection
        """
        with self._lock:
            run_id = f"run_{self._training_runs.count()}_{int(datetime.now().timestamp() * 1000)}"
            run = {
                "id": run_id,
                "created_at": datetime.now().isoformat(),
                "dataset_version": dataset_version,
                "pairs_count": pairs_count,
                "model_used": model_used,
                "status": "pending",
                "metrics": {},
            }
            self._training_runs.insert_one({**run, "_id": run_id})
            return TrainingRun(**run)

    def update_training_run(
        self, run_id: str, status: str, metrics: Optional[Dict] = None
    ):
        """Update a training run's status and metrics.

        Metrics are merged into existing run metrics, not replaced.

        Args:
            run_id: Id of the training run to update.
            status: New status ("pending", "running", "completed", "failed").
            metrics: Optional metric updates to merge.

        Side effects:
            - updates the training run document
        """
        with self._lock:
            doc = self._training_runs.find_one({"_id": run_id})
            if not doc:
                return
            update: Dict[str, Any] = {"status": status}
            if metrics:
                merged = dict(doc.get("metrics", {}))
                merged.update(metrics)
                update["metrics"] = merged
            self._training_runs.update_one({"_id": run_id}, {"$set": update})

    def get_training_runs(self, limit: Optional[int] = 10) -> List[TrainingRun]:
        """Get recent training runs, newest appended last.

        Args:
            limit: Maximum number of most-recent runs to return.

        Returns:
            List of TrainingRun models in insertion order.
        """
        with self._lock:
            docs = self._training_runs.find(sort=[("_created", 1)])
            if limit is not None:
                docs = docs[-limit:]
            return [self._to_model(TrainingRun, d) for d in docs]

    # ============ Export ============

    def export_training_data(
        self,
        min_quality: float = 0.5,
        format: str = "jsonl",
        version: Optional[str] = None,
    ) -> str:
        """Export unused training pairs to a dataset file.

        Exported pairs are marked as used and recorded in a training run.

        Args:
            min_quality: Minimum pair quality to export.
            format: Output format, "jsonl" or "json".
            version: Dataset version tag; defaults to a timestamp.

        Returns:
            Path to the exported file.

        Raises:
            ValueError: If no pairs match, or the format is unknown.

        Side effects:
            - writes the dataset file
            - writes a ``latest.jsonl`` copy (jsonl format only)
            - marks exported pairs as used
            - creates a completed training run record
        """
        with self._lock:
            pairs = self.get_training_pairs(min_quality=min_quality, include_used=False)

            if not pairs:
                raise ValueError("No training pairs to export")

            if version is None:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")

            if format == "jsonl":
                filename = f"training_v{version}.jsonl"
                filepath = self.exports_dir / filename
                tmp_path = self.exports_dir / f".{filename}.tmp"
                try:
                    with open(tmp_path, "w") as f:
                        for pair in pairs:
                            f.write(
                                json.dumps(
                                    {
                                        "prompt": pair.prompt,
                                        "response": pair.response,
                                        "quality": pair.quality_score,
                                    }
                                )
                                + "\n"
                            )
                    os.replace(tmp_path, filepath)
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()

                # "latest" copy for consumers that expect a stable path
                latest_link = self.exports_dir / "latest.jsonl"
                if latest_link.exists():
                    latest_link.unlink()
                shutil.copy2(filepath, latest_link)

            elif format == "json":
                filename = f"training_v{version}.json"
                filepath = self.exports_dir / filename
                tmp_path = self.exports_dir / f".{filename}.tmp"
                try:
                    with open(tmp_path, "w") as f:
                        json.dump(
                            [
                                {
                                    "prompt": p.prompt,
                                    "response": p.response,
                                    "quality": p.quality_score,
                                }
                                for p in pairs
                            ],
                            f,
                            indent=2,
                        )
                    os.replace(tmp_path, filepath)
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()

            else:
                raise ValueError(f"Unknown format: {format}")

            pair_ids = [p.id for p in pairs]
            run = self.create_training_run(
                dataset_version=version,
                pairs_count=len(pairs),
                model_used="export",
            )
            self.mark_pairs_used(pair_ids, run.id)
            self.update_training_run(run.id, "completed")

            return str(filepath)

    # ============ Stats ============

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics.

        Returns:
            Dict of counts for conversations, pairs, runs, and exports.
        """
        with self._lock:
            convs = self._conversations.find()
            all_pairs = self._training_pairs.find()
            good_pairs = len([p for p in all_pairs if p.get("quality_score", 0.0) >= GOOD_QUALITY])
            bad_pairs = len([p for p in all_pairs if p.get("quality_score", 0.0) < BAD_QUALITY])
            unused_pairs = len([p for p in all_pairs if not p.get("used_in_training", False)])

            return {
                "conversations_total": len(convs),
                "conversations_with_feedback": len(
                    [c for c in convs if c.get("feedback")]
                ),
                "training_pairs_total": len(all_pairs),
                "training_pairs_good": good_pairs,
                "training_pairs_bad": bad_pairs,
                "training_pairs_unused": unused_pairs,
                "training_runs": self._training_runs.count(),
                "exports_count": len(
                    [
                        p
                        for p in self.exports_dir.glob("*.jsonl")
                        if p.name != "latest.jsonl"
                    ]
                ),
            }

    # ============ Backup ============

    def create_backup(self) -> str:
        """Create a full backup of all data (MogDB snapshots + latest export).

        The store is compacted first so the copied snapshot is consistent
        (no torn journal lines), then written under the pipeline lock.

        Returns:
            Path to the created backup directory.

        Side effects:
            - creates a timestamped directory under ``backups/``
            - compacts the MogDB store (prunes journals)
            - copies the compacted store files and latest export into it
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"pipeline_{timestamp}"
        backup_path = self.backups_dir / backup_name
        backup_path.mkdir()

        with self._lock:
            self._db.compact_all()
            for src in self.db_path.glob("*"):
                if src.is_file():
                    shutil.copy2(src, backup_path / src.name)

        # Include the latest export when present (best-effort)
        try:
            latest_export = self.exports_dir / "latest.jsonl"
            if latest_export.exists():
                shutil.copy2(latest_export, backup_path / "latest.jsonl")
        except Exception as exc:
            logger.debug("Export backup copy failed: %s", exc)

        return str(backup_path)


# Singleton
_pipeline: Optional[TrainingDataPipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline(data_dir: str = "data") -> TrainingDataPipeline:
    """Get or create the shared training data pipeline.

    The pipeline is created once per process; requesting a different
    ``data_dir`` afterwards is an error rather than a silent no-op.

    Args:
        data_dir: Directory holding the pipeline store and exports.

    Returns:
        The process-wide TrainingDataPipeline singleton.

    Raises:
        ValueError: If a pipeline already exists for a different ``data_dir``.

    Side effects:
        - constructs the pipeline on first call
    """
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = TrainingDataPipeline(data_dir)
        elif Path(_pipeline.data_dir).resolve() != Path(data_dir).resolve():
            raise ValueError(
                f"training pipeline already bound to {_pipeline.data_dir}; "
                f"cannot rebind to {data_dir}"
            )
        return _pipeline


__all__ = [
    "Conversation",
    "TrainingPair",
    "TrainingRun",
    "TrainingDataPipeline",
    "get_pipeline",
]
