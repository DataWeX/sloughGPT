"""
Response Tracker

Logs and tracks chat responses for benchmarking quality evaluation.
Stores input/output pairs for offline analysis.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger("man.response_tracker")


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
    eval_scores: Optional[Dict[str, float]] = None


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
        # Use absolute path from package location
        import os
        # Navigate from domains/feedback to repo root
        repo_root = Path(__file__).resolve().parents[4]
        self.log_dir = repo_root / log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_file = self.log_dir / f"responses_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self._buffer: List[ResponseLog] = []
        self._buffer_size = 1

    def log(
        self,
        user_message: str,
        assistant_response: str,
        model: str,
        config: Dict[str, Any],
        session_id: str,
        user_id: str,
        tokens_generated: int,
        duration_ms: float,
        has_images: bool = False,
        context_tokens: int = 0,
    ) -> ResponseLog:
        """Log a response."""
        entry = ResponseLog(
            timestamp=datetime.now().isoformat(),
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
        """Flush buffer to file."""
        if not self._buffer:
            return

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

        self._buffer.clear()
        logger.info(f"Flushed {len(self._buffer)} responses to {self.current_file}")

    def get_responses(
        self,
        limit: int = 100,
        model: Optional[str] = None,
        since: Optional[str] = None,
    ) -> List[ResponseLog]:
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
                    except (json.JSONDecodeError, ValueError, KeyError):
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

        logger.info(f"Exported to {output_path}")
        return str(output_path)

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated stats."""
        responses = self.get_responses(limit=1000)

        if not responses:
            return {"total": 0}

        total = len(responses)
        avg_tokens = sum(r.tokens_generated for r in responses) / total
        avg_duration = sum(r.duration_ms for r in responses) / total
        models = set(r.model for r in responses)

        return {
            "total": total,
            "avg_tokens": avg_tokens,
            "avg_duration_ms": avg_duration,
            "unique_models": list(models),
            "latest_file": str(self.current_file),
        }


# Global tracker instance
_response_tracker: Optional[ResponseTracker] = None


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
