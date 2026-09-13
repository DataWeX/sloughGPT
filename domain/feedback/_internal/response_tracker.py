"""
Response Tracker

Logs and tracks chat responses for benchmarking quality evaluation.
Stores input/output pairs for offline analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from domains.shared import find_repo_root

logger = logging.getLogger("slo.response_tracker")


@dataclass
class ResponseLog:
    """Single response log entry."""
    timestamp: str
    user_message: str
    assistant_response: str
    model: str
    temperature: float
    max_tokens: int
    session_id: str
    user_id: str
    tokens_generated: int
    duration_ms: float
    has_images: bool = False
    context_tokens: int = 0
    eval_scores: dict[str, float] | None = None


class ResponseTracker:
    """
    Track chat responses for quality evaluation.

    Usage:
        tracker = ResponseTracker()

        # Log a response
        tracker.log(
            user_message="What is Python?",
            assistant_response="Python is a programming language...",
            model="gpt2",
            config={"temperature": 0.8, "max_tokens": 256},
            session_id="abc123",
            user_id="user_1",
            tokens_generated=45,
            duration_ms=1200,
        )

        # Get logged responses
        responses = tracker.get_responses(limit=100)

        # Export for benchmarking
        tracker.export_jsonl("data/response_logs.jsonl")
    """

    def __init__(self, log_dir: str = "data/response_logs"):
        # Navigate from domains/feedback to repo root
        repo_root = find_repo_root(Path(__file__).resolve())
        self.log_dir = repo_root / log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_file = self.log_dir / f"responses_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        self._buffer: list[ResponseLog] = []
        self._buffer_size = 1
        # MogDB persistence
        self._mogdb = None
        self._coll = None
        self._init_mogdb(repo_root)

    def _init_mogdb(self, repo_root: Path) -> None:
        """Initialize MogDB for response log persistence with TTL index."""
        try:
            from mogdb import MogDB
            db_path = str(repo_root / "data" / "response_mogdb")
            self._mogdb = MogDB(db_path)
            self._coll = self._mogdb.collection("responses")
            # Create TTL index on expires_at (numeric epoch) for auto-expiry
            self._coll.create_ttl_index("expires_at", expire_after_seconds=30 * 24 * 3600)
        except Exception as e:
            logger.warning("Failed to initialize MogDB for response logs: %s", e)

    def log(
        self,
        user_message: str,
        assistant_response: str,
        model: str,
        config: dict[str, Any],
        session_id: str,
        user_id: str,
        tokens_generated: int,
        duration_ms: float,
        has_images: bool = False,
        context_tokens: int = 0,
    ) -> ResponseLog:
        """Log a response."""
        entry = ResponseLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_message=user_message,
            assistant_response=assistant_response,
            model=model,
            temperature=config.get("temperature", 0.8),
            max_tokens=config.get("max_tokens", 256),
            session_id=session_id,
            user_id=user_id,
            tokens_generated=tokens_generated,
            duration_ms=duration_ms,
            has_images=has_images,
            context_tokens=context_tokens,
        )

        self._buffer.append(entry)

        # Flush to file
        if len(self._buffer) >= self._buffer_size:
            self._flush()

        return entry

    def _flush(self):
        """Flush buffer to file and MogDB."""
        if not self._buffer:
            return

        # Write to JSONL file
        with open(self.current_file, "a") as f:
            for entry in self._buffer:
                f.write(json.dumps({
                    "timestamp": entry.timestamp,
                    "user_message": entry.user_message,
                    "assistant_response": entry.assistant_response,
                    "model": entry.model,
                    "temperature": entry.temperature,
                    "max_tokens": entry.max_tokens,
                    "session_id": entry.session_id,
                    "user_id": entry.user_id,
                    "tokens_generated": entry.tokens_generated,
                    "duration_ms": entry.duration_ms,
                    "has_images": entry.has_images,
                    "context_tokens": entry.context_tokens,
                }) + "\n")

        # Write to MogDB
        if self._coll is not None:
            try:
                import time
                now = time.time()
                for entry in self._buffer:
                    self._coll.insert_one({
                        "timestamp": entry.timestamp,
                        "expires_at": now,  # Numeric epoch for TTL
                        "user_message": entry.user_message,
                        "assistant_response": entry.assistant_response,
                        "model": entry.model,
                        "temperature": entry.temperature,
                        "max_tokens": entry.max_tokens,
                        "session_id": entry.session_id,
                        "user_id": entry.user_id,
                        "tokens_generated": entry.tokens_generated,
                        "duration_ms": entry.duration_ms,
                        "has_images": entry.has_images,
                        "context_tokens": entry.context_tokens,
                    })
            except Exception as e:
                logger.warning("Failed to write to MogDB: %s", e)

        count = len(self._buffer)
        self._buffer.clear()
        logger.info("Flushed %s responses to %s", count, self.current_file, extra={"tag": "INFRA"})

    def get_responses(
        self,
        limit: int = 100,
        model: str | None = None,
        since: str | None = None,
    ) -> list[ResponseLog]:
        """Get recent responses."""
        responses = []

        # Read from today's file
        if self.current_file.exists():
            with open(self.current_file) as f:
                for line in f:
                    try:
                        data = json.loads(line)

                        if model and data.get("model") != model:
                            continue
                        if since and data.get("timestamp") < since:
                            continue

                        responses.append(ResponseLog(**data))
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        logger.debug("Skipping malformed response log line: %s", e)
                        continue

        return responses[-limit:]

    def export_jsonl(self, path: str):
        """Export all responses to JSONL."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for entry in self.get_responses(limit=10000):
                f.write(json.dumps({
                    "timestamp": entry.timestamp,
                    "user_message": entry.user_message,
                    "assistant_response": entry.assistant_response,
                    "model": entry.model,
                    "session_id": entry.session_id,
                }) + "\n")

        logger.info("Exported to %s", output_path, extra={"tag": "INFRA"})
        return str(output_path)

    def get_stats(self) -> dict[str, Any]:
        """Get aggregated stats."""
        responses = self.get_responses(limit=1000)

        if not responses:
            return {"total": 0}

        total = len(responses)
        avg_tokens = sum(r.tokens_generated for r in responses) / total
        avg_duration = sum(r.duration_ms for r in responses) / total
        models = {r.model for r in responses}

        return {
            "total": total,
            "avg_tokens": avg_tokens,
            "avg_duration_ms": avg_duration,
            "unique_models": list(models),
            "latest_file": str(self.current_file),
        }


# Global tracker instance
_response_tracker: ResponseTracker | None = None


def get_response_tracker() -> ResponseTracker:
    """Get global response tracker."""
    global _response_tracker
    if _response_tracker is None:
        _response_tracker = ResponseTracker()
    return _response_tracker


__all__ = [
    "ResponseLog",
    "ResponseTracker",
    "get_response_tracker",
]
