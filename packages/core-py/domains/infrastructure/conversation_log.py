"""
Conversation logger — persists API request/response pairs for reuse as training data.

Key classes and functions:
- ``ConversationLogger`` — appends one prompt/response exchange to a dataset in
  messages format (``corpus.jsonl``) and dialogue text format (``input.txt``).
- ``get_conversation_logger()`` — module-level singleton accessor.

Both output files live under ``data/api_conversations/`` so the dataset
manager (``GET /datasets``) lists it and the training pipelines
(character-level ``train_pipeline.py`` and HF fine-tune) can consume it.
Capture is opt-out via ``MAN_CAPTURE_CONVERSATIONS=0``.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from domains.shared import find_repo_root

logger = logging.getLogger("slo.infrastructure.conversation_log")

_REPO_ROOT = find_repo_root(Path(__file__).resolve())
_DEFAULT_DIR = _REPO_ROOT / "data" / "api_conversations"


class ConversationLogger:
    """Append API exchanges to a training dataset (thread-safe)."""

    def __init__(self, data_dir: Path = _DEFAULT_DIR) -> None:
        """
        Initialize the logger.

        Args:
            data_dir: dataset directory holding ``corpus.jsonl`` + ``input.txt``

        Side effects:
            - creates the data directory if missing
        """
        self.data_dir = data_dir
        self.corpus_path = data_dir / "corpus.jsonl"
        self.text_path = data_dir / "input.txt"
        self._lock = threading.Lock()
        data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self) -> bool:
        """True when capture is enabled (default on, disable via env var)."""
        return os.environ.get("MAN_CAPTURE_CONVERSATIONS", "1") != "0"

    def record(
        self,
        prompt: str,
        response: str,
        model: str = "unknown",
        tokens_generated: int = 0,
        elapsed_ms: float = 0.0,
        temperature: Optional[float] = None,
        meta: Optional[Dict] = None,
    ) -> Optional[int]:
        """
        Append one exchange to the dataset.

        Args:
            prompt: user prompt text
            response: assistant response text
            model: model id used for generation
            tokens_generated: token count of the response
            elapsed_ms: generation duration in milliseconds
            temperature: sampling temperature used (if known)
            meta: extra metadata dict to embed in the corpus row

        Returns:
            number of bytes appended to ``corpus.jsonl``, or None when disabled

        Side effects:
            - appends one line to ``corpus.jsonl`` (messages format)
            - appends dialogue text to ``input.txt`` (plain text format)
        """
        if not self.enabled:
            return None
        prompt = (prompt or "").strip()
        response = (response or "").strip()
        if not prompt or not response:
            return None

        row = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "meta": {
                "model": model,
                "tokens_generated": tokens_generated,
                "elapsed_ms": round(elapsed_ms, 1),
                "temperature": temperature,
                "captured_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        if meta:
            row["meta"].update(meta)

        dialogue = f"User: {prompt}\nAssistant: {response}\n\n"

        with self._lock:
            written = 0
            with open(self.corpus_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
            with open(self.text_path, "a", encoding="utf-8") as f:
                f.write(dialogue)
        return written


_logger: Optional[ConversationLogger] = None
_logger_lock = threading.Lock()


def get_conversation_logger() -> ConversationLogger:
    """Return the shared ConversationLogger singleton (lazily created)."""
    global _logger
    if _logger is None:
        with _logger_lock:
            if _logger is None:
                _logger = ConversationLogger()
    return _logger


def reset_conversation_logger() -> None:
    """Reset the singleton (for testing)."""
    global _logger
    with _logger_lock:
        _logger = None


def capture(
    prompt: str,
    response: str,
    model: str = "unknown",
    tokens_generated: int = 0,
    elapsed_ms: float = 0.0,
    temperature: Optional[float] = None,
    meta: Optional[Dict] = None,
) -> bool:
    """Record an exchange without ever raising (safe for request handlers).

    Args:
        prompt: user prompt text
        response: assistant response text
        model: model id used for generation
        tokens_generated: token count of the response
        elapsed_ms: generation duration in milliseconds
        temperature: sampling temperature used (if known)
        meta: extra metadata dict to embed in the corpus row

    Returns:
        True when the exchange was written, False when disabled/empty/error

    Side effects:
        - appends to the conversation dataset when enabled
    """
    try:
        result = get_conversation_logger().record(
            prompt,
            response,
            model=model,
            tokens_generated=tokens_generated,
            elapsed_ms=elapsed_ms,
            temperature=temperature,
            meta=meta,
        )
        return result is not None
    except Exception as exc:
        logger.warning("Failed to log conversation turn: %s", exc)
        return False
