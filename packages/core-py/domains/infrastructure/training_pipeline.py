"""
Training Data Pipeline - Best practices for conversation-to-training conversion.

Design principles:
1. Separate raw data from processed training data
2. Version control for training datasets
3. Quality scoring and filtering
4. Audit trail for reproducibility
5. Incremental updates for continuous learning
"""

import json
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
import threading
import hashlib


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
        /conversations.db      <- Raw conversation storage
        /training_pairs.db     <- Processed training pairs
        /training_runs.db      <- Training run history
        /exports/              <- Exported datasets
            /v1.0.jsonl
            /v1.1.jsonl
        /backups/              <- DB backups
    """

    VERSION = "1.0"

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.conversations_path = self.data_dir / "conversations.db"
        self.pairs_path = self.data_dir / "training_pairs.db"
        self.runs_path = self.data_dir / "training_runs.db"
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
        """Initialize databases if they don't exist."""
        for path in [self.conversations_path, self.pairs_path, self.runs_path]:
            if not path.exists():
                self._write_db(path, {"version": self.VERSION, "records": []})

    def _read_db(self, path: Path) -> Dict:
        """Read a database file."""
        if not path.exists():
            return {"version": self.VERSION, "records": []}
        try:
            content = path.read_text()
            return (
                json.loads(content) if content.strip() else {"version": self.VERSION, "records": []}
            )
        except json.JSONDecodeError:
            return {"version": self.VERSION, "records": []}

    def _write_db(self, path: Path, data: Dict):
        """Write to a database file."""
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ============ Conversations ============

    def add_conversation(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        model: str,
        tokens: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> Conversation:
        """Add a raw conversation."""
        db = self._read_db(self.conversations_path)

        conv_id = f"conv_{len(db['records'])}_{int(datetime.now().timestamp() * 1000)}"
        conv = {
            "id": conv_id,
            "session_id": session_id,
            "user_message": user_message,
            "assistant_message": assistant_message,
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "tokens": tokens,
            "feedback": None,
            "metadata": metadata or {},
        }

        db["records"].append(conv)
        self._write_db(self.conversations_path, db)

        # Auto-create training pair
        self._create_training_pair(conv)

        return Conversation(**conv)

    def add_feedback(self, conversation_id: str, feedback: str) -> bool:
        """Add feedback to a conversation."""
        db = self._read_db(self.conversations_path)

        for conv in db["records"]:
            if conv["id"] == conversation_id:
                conv["feedback"] = feedback
                self._write_db(self.conversations_path, db)

                # Update quality score of corresponding training pair
                self._update_pair_quality(conversation_id, feedback)
                return True

        return False

    def get_conversations(
        self,
        session_id: Optional[str] = None,
        feedback: Optional[str] = None,
        limit: int = 100,
    ) -> List[Conversation]:
        """Get conversations with optional filters."""
        db = self._read_db(self.conversations_path)
        records = db["records"]

        if session_id:
            records = [r for r in records if r["session_id"] == session_id]
        if feedback:
            records = [r for r in records if r.get("feedback") == feedback]

        records = records[-limit:]
        return [Conversation(**r) for r in records]

    # ============ Training Pairs ============

    def _create_training_pair(self, conversation: Dict):
        """Create training pair from conversation with quality scoring."""
        pairs_db = self._read_db(self.pairs_path)

        # Calculate quality score based on feedback
        feedback = conversation.get("feedback")
        if feedback == "up":
            quality = 1.0
        elif feedback == "down":
            quality = 0.0
        else:
            quality = 0.5  # Neutral

        # Check for empty responses
        if not conversation.get("assistant_message", "").strip():
            quality = 0.0

        pair_id = f"pair_{len(pairs_db['records'])}_{int(datetime.now().timestamp() * 1000)}"
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

        pairs_db["records"].append(pair)
        self._write_db(self.pairs_path, pairs_db)

    def _update_pair_quality(self, conversation_id: str, feedback: str):
        """Update quality score when feedback is added."""
        pairs_db = self._read_db(self.pairs_path)

        for pair in pairs_db["records"]:
            if pair["conversation_id"] == conversation_id:
                if feedback == "up":
                    pair["quality_score"] = 1.0
                elif feedback == "down":
                    pair["quality_score"] = 0.0
                pair["feedback"] = feedback
                break

        self._write_db(self.pairs_path, pairs_db)

    def get_training_pairs(
        self,
        min_quality: float = 0.0,
        include_used: bool = True,
        limit: Optional[int] = None,
    ) -> List[TrainingPair]:
        """Get training pairs with quality filtering."""
        db = self._read_db(self.pairs_path)
        records = db["records"]

        # Filter by quality
        records = [r for r in records if r["quality_score"] >= min_quality]

        # Filter by used status
        if not include_used:
            records = [r for r in records if not r.get("used_in_training", False)]

        if limit:
            records = records[-limit:]

        return [TrainingPair(**r) for r in records]

    def mark_pairs_used(self, pair_ids: List[str], training_run_id: str):
        """Mark pairs as used in a training run."""
        pairs_db = self._read_db(self.pairs_path)

        for pair in pairs_db["records"]:
            if pair["id"] in pair_ids:
                pair["used_in_training"] = True
                pair["training_run_id"] = training_run_id

        self._write_db(self.pairs_path, pairs_db)

    # ============ Training Runs ============

    def create_training_run(
        self,
        dataset_version: str,
        pairs_count: int,
        model_used: str,
    ) -> TrainingRun:
        """Create a new training run record."""
        db = self._read_db(self.runs_path)

        run_id = f"run_{len(db['records'])}_{int(datetime.now().timestamp() * 1000)}"
        run = {
            "id": run_id,
            "created_at": datetime.now().isoformat(),
            "dataset_version": dataset_version,
            "pairs_count": pairs_count,
            "model_used": model_used,
            "status": "pending",
            "metrics": {},
        }

        db["records"].append(run)
        self._write_db(self.runs_path, db)

        return TrainingRun(**run)

    def update_training_run(self, run_id: str, status: str, metrics: Optional[Dict] = None):
        """Update training run status and metrics."""
        db = self._read_db(self.runs_path)

        for run in db["records"]:
            if run["id"] == run_id:
                run["status"] = status
                if metrics:
                    run["metrics"].update(metrics)
                break

        self._write_db(self.runs_path, db)

    def get_training_runs(self, limit: int = 10) -> List[TrainingRun]:
        """Get recent training runs."""
        db = self._read_db(self.runs_path)
        records = db["records"][-limit:]
        return [TrainingRun(**r) for r in records]

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

            # Also create "latest" symlink
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
        conv_db = self._read_db(self.conversations_path)
        pairs_db = self._read_db(self.pairs_path)
        runs_db = self._read_db(self.runs_path)

        all_pairs = pairs_db["records"]
        good_pairs = len([p for p in all_pairs if p["quality_score"] >= 0.8])
        bad_pairs = len([p for p in all_pairs if p["quality_score"] < 0.3])
        unused_pairs = len([p for p in all_pairs if not p.get("used_in_training", False)])

        return {
            "conversations_total": len(conv_db["records"]),
            "conversations_with_feedback": len(
                [c for c in conv_db["records"] if c.get("feedback")]
            ),
            "training_pairs_total": len(all_pairs),
            "training_pairs_good": good_pairs,
            "training_pairs_bad": bad_pairs,
            "training_pairs_unused": unused_pairs,
            "training_runs": len(runs_db["records"]),
            "exports_count": len(list(self.exports_dir.glob("*.jsonl"))),
        }

    # ============ Backup ============

    def create_backup(self) -> str:
        """Create full backup of all data."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"pipeline_{timestamp}"
        backup_path = self.backups_dir / backup_name
        backup_path.mkdir()

        for src in [self.conversations_path, self.pairs_path, self.runs_path]:
            if src.exists():
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
