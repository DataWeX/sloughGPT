"""
Training Data Pipeline — best practices for conversation-to-training conversion.

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
"""

import dataclasses
import json
import logging
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from mogdb import MogDB

logger = logging.getLogger("slo.infrastructure.training_pipeline")


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
    feedback: Optional[str] = None  # "up", "down", None
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

        self._lock = threading.Lock()

        self._ensure_directories()
        self._init_dbs()

    def _ensure_directories(self):
        """Create necessary directories."""
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        self.backups_dir.mkdir(parents=True, exist_ok=True)

    def _init_dbs(self):
        """Initialize the MogDB store and migrate any legacy JSON databases."""
        self._db = MogDB(str(self.db_path))
        self._conversations = self._db.collection("conversations")
        self._training_pairs = self._db.collection("training_pairs")
        self._training_runs = self._db.collection("training_runs")
        self._migrate_from_json()

    def _read_json_db(self, path: Path) -> List[Dict[str, Any]]:
        """Read records from a legacy JSON database file.

        Returns an empty list for missing or corrupt files.
        """
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError, ValueError):
            return []
        records = data.get("records", []) if isinstance(data, dict) else []
        return records if isinstance(records, list) else []

    def _migrate_from_json(self):
        """Migrate legacy JSON ``*.db`` files into MogDB collections.

        Each record is inserted with ``_id`` set to its ``id``; records
        already present are skipped. The legacy file is removed afterwards.
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
        """Reconstruct a dataclass from a MogDB document, ignoring meta keys."""
        return cls(**{f.name: doc.get(f.name) for f in dataclasses.fields(cls)})

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
        """Add a raw conversation.

        A training pair is created automatically with quality derived from
        ``feedback`` when provided.

        Args:
            session_id: Identifier for the chat session.
            user_message: The user's prompt text.
            assistant_message: The assistant's response text.
            model: Model identifier that produced the response.
            tokens: Token count of the response, if known.
            metadata: Optional extra context.
            feedback: Optional rating (``"up"``, ``"down"``, or None).

        Returns:
            The stored Conversation.
        """
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

    def add_feedback(self, conversation_id: str, feedback: str) -> bool:
        """Add or update feedback on a conversation.

        Updates the conversation's ``feedback`` field and the quality score
        of its corresponding training pair.

        Returns:
            True if the conversation was found, False otherwise.
        """
        with self._lock:
            updated = self._conversations.update_one(
                {"_id": conversation_id}, {"$set": {"feedback": feedback}}
            )
            if updated:
                self._update_pair_quality(conversation_id, feedback)
            return bool(updated)

    def get_conversations(
        self,
        session_id: Optional[str] = None,
        feedback: Optional[str] = None,
        limit: int = 100,
    ) -> List[Conversation]:
        """Get conversations with optional filters, newest appended last."""
        query: Dict[str, Any] = {}
        if session_id:
            query["session_id"] = session_id
        if feedback:
            query["feedback"] = feedback
        docs = self._conversations.find(query, sort=[("_created", 1)])
        docs = docs[-limit:]
        return [self._to_model(Conversation, d) for d in docs]

    # ============ Training Pairs ============

    def _create_training_pair(self, conversation: Dict):
        """Create training pair from conversation with quality scoring."""
        feedback = conversation.get("feedback")
        if feedback == "up":
            quality = 1.0
        elif feedback == "down":
            quality = 0.0
        else:
            quality = 0.5  # Neutral

        # Check for empty responses
        if not str(conversation.get("assistant_message", "") or "").strip():
            quality = 0.0

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

    def _update_pair_quality(self, conversation_id: str, feedback: str):
        """Update quality score when feedback is added."""
        update: Dict[str, Any] = {"feedback": feedback}
        if feedback == "up":
            update["quality_score"] = 1.0
        elif feedback == "down":
            update["quality_score"] = 0.0
        self._training_pairs.update_one(
            {"conversation_id": conversation_id}, {"$set": update}
        )

    def get_training_pairs(
        self,
        min_quality: float = 0.0,
        include_used: bool = True,
        limit: Optional[int] = None,
    ) -> List[TrainingPair]:
        """Get training pairs with quality filtering, newest appended last."""
        docs = self._training_pairs.find(sort=[("_created", 1)])
        docs = [d for d in docs if d.get("quality_score", 0.0) >= min_quality]
        if not include_used:
            docs = [d for d in docs if not d.get("used_in_training", False)]
        if limit:
            docs = docs[-limit:]
        return [self._to_model(TrainingPair, d) for d in docs]

    def mark_pairs_used(self, pair_ids: List[str], training_run_id: str):
        """Mark pairs as used in a training run."""
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
        """Create a new training run record."""
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
        """Update training run status and metrics (merged, not replaced)."""
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

    def get_training_runs(self, limit: int = 10) -> List[TrainingRun]:
        """Get recent training runs, newest appended last."""
        docs = self._training_runs.find(sort=[("_created", 1)])
        docs = docs[-limit:]
        return [self._to_model(TrainingRun, d) for d in docs]

    # ============ Export ============

    def export_training_data(
        self,
        min_quality: float = 0.5,
        format: str = "jsonl",
        version: Optional[str] = None,
    ) -> str:
        """
        Export training data to file.

        Returns path to exported file.
        """
        pairs = self.get_training_pairs(min_quality=min_quality, include_used=False)

        if not pairs:
            raise ValueError("No training pairs to export")

        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "jsonl":
            filename = f"training_v{version}.jsonl"
            filepath = self.exports_dir / filename

            with open(filepath, "w") as f:
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

            # Also create "latest" copy
            latest_link = self.exports_dir / "latest.jsonl"
            if latest_link.exists():
                latest_link.unlink()
            # Note: symlink might not work on all systems, use copy instead
            shutil.copy2(filepath, latest_link)

        elif format == "json":
            filename = f"training_v{version}.json"
            filepath = self.exports_dir / filename

            with open(filepath, "w") as f:
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

        else:
            raise ValueError(f"Unknown format: {format}")

        # Mark pairs as used
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
        """Get pipeline statistics."""
        convs = self._conversations.find()
        all_pairs = self._training_pairs.find()
        good_pairs = len([p for p in all_pairs if p.get("quality_score", 0.0) >= 0.8])
        bad_pairs = len([p for p in all_pairs if p.get("quality_score", 0.0) < 0.3])
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
            "exports_count": len(list(self.exports_dir.glob("*.jsonl"))),
        }

    # ============ Backup ============

    def create_backup(self) -> str:
        """Create full backup of all data (MogDB journals + latest export)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"pipeline_{timestamp}"
        backup_path = self.backups_dir / backup_name
        backup_path.mkdir()

        for src in self.db_path.glob("*"):
            if src.is_file():
                shutil.copy2(src, backup_path / src.name)

        # Export latest training data to backup
        try:
            latest_export = self.exports_dir / "latest.jsonl"
            if latest_export.exists():
                shutil.copy2(latest_export, backup_path / "latest.jsonl")
        except Exception:
            pass

        return str(backup_path)


# Singleton
_pipeline: Optional[TrainingDataPipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline(data_dir: str = "data") -> TrainingDataPipeline:
    """Get or create the training data pipeline."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = TrainingDataPipeline(data_dir)
        return _pipeline


__all__ = [
    "Conversation",
    "TrainingPair",
    "TrainingRun",
    "TrainingDataPipeline",
    "get_pipeline",
]
