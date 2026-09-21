"""Experience adapter — owned data (14): train on the model's own experience.

The model generates experience through:
  - chat sessions (conversation_logger → datasets/api_conversations/corpus.jsonl)
  - tool outcomes (tool call results + success/failure)
  - feedback (ratings, corrections)

This adapter aggregates that experience into a BatchSampler for TrainingLoop,
so training is on lived data, not just third-party text.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .training_handler import BatchSampler

logger = logging.getLogger("slo.training.experience")


def _load_chat_corpus(path: Path | None = None) -> list[dict[str, str]]:
    """Load chat pairs from the experience corpus."""
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    # default locations (project-local + user cache)
    candidates.extend(
        [
            Path("datasets/api_conversations/corpus.jsonl"),
            Path("data/api_conversations/corpus.jsonl"),
            Path.home() / ".cache" / "sloughgpt" / "experience" / "corpus.jsonl",
        ]
    )
    for p in candidates:
        if p.exists():
            out = []
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    # support both {user_msg,assistant_msg} and {prompt,completion}
                    u = rec.get("user_msg") or rec.get("prompt") or rec.get("user")
                    a = rec.get("assistant_msg") or rec.get("completion") or rec.get("assistant")
                    if u and a:
                        out.append({"user_msg": str(u), "assistant_msg": str(a)})
            except Exception as e:
                logger.debug("Failed to read corpus %s: %s", p, e)
                continue
            if out:
                logger.info(
                    "Loaded %d experience pairs from %s", len(out), p, extra={"tag": "TRAIN"}
                )
                return out
    return []


def _load_feedback_pairs(path: Path | None = None) -> list[dict[str, str]]:
    """Load feedback-corrected pairs (user corrected assistant)."""
    candidates = []
    if path is not None:
        candidates.append(Path(path))
    candidates.extend(
        [
            Path("datasets/feedback/corrections.jsonl"),
            Path("data/feedback.jsonl"),
        ]
    )
    for p in candidates:
        if p.exists():
            out = []
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    # feedback record: {user_message, assistant_response, corrected_response, rating}
                    # only use positively-rated or corrected
                    if rec.get("rating", 0) >= 4 or rec.get("corrected_response"):
                        u = rec.get("user_message") or rec.get("user_msg")
                        a = rec.get("corrected_response") or rec.get("assistant_response")
                        if u and a:
                            out.append({"user_msg": str(u), "assistant_msg": str(a)})
            except Exception as e:
                logger.debug("Failed to read feedback %s: %s", p, e)
                continue
            if out:
                logger.info("Loaded %d feedback pairs from %s", len(out), p, extra={"tag": "TRAIN"})
                return out
    return []


class ExperienceSampler(BatchSampler):
    """BatchSampler that concatenates owned experience into token blocks.

    Wraps ChatPairSampler-style text but sources from live experience
    (chat corpus + feedback). Falls back to synthetic if no experience yet.

    Args:
        stoi: token -> id map (char-level or BPE)
        block_size: context length
        corpus_path: override path to corpus.jsonl
        feedback_path: override path to feedback jsonl
        seed: RNG seed
    """

    def __init__(
        self,
        stoi: dict[str, int],
        block_size: int,
        corpus_path: Path | str | None = None,
        feedback_path: Path | str | None = None,
        seed: int = 42,
    ) -> None:
        self.stoi = stoi
        self.block_size = block_size
        self._rng = np.random.default_rng(seed)

        pairs = []
        pairs.extend(_load_chat_corpus(Path(corpus_path) if corpus_path else None))
        pairs.extend(_load_feedback_pairs(Path(feedback_path) if feedback_path else None))

        if not pairs:
            # synthetic fallback so training never crashes on fresh install
            pairs = [
                {"user_msg": "hello", "assistant_msg": "hi there"},
                {"user_msg": "how are you", "assistant_msg": "doing well, how can I help?"},
            ]
            logger.info("No owned experience found — using synthetic seed", extra={"tag": "TRAIN"})

        # build flat ids same as ChatPairSampler but from owned pairs
        text = "".join(f"User: {p['user_msg']}\nAssistant: {p['assistant_msg']}\n\n" for p in pairs)
        self.ids = np.array([stoi.get(c, 0) for c in text], dtype=np.int32)
        self.n_samples = max(1, len(self.ids) - block_size - 1)
        self._pairs = pairs

        logger.info(
            "ExperienceSampler: %d pairs → %d tokens → %d samples (block=%d)",
            len(pairs),
            len(self.ids),
            self.n_samples,
            block_size,
            extra={"tag": "TRAIN"},
        )

    def __len__(self) -> int:
        return self.n_samples

    def get_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        indices = self._rng.integers(0, self.n_samples, size=batch_size)
        offsets = np.arange(self.block_size)
        pos = indices[:, None] + offsets  # (B, block)
        x = self.ids[pos]
        y = self.ids[pos + 1]
        return x.astype(np.int32), y.astype(np.int32)

    def stats(self) -> dict[str, Any]:
        return {
            "pairs": len(self._pairs),
            "tokens": int(len(self.ids)),
            "samples": int(self.n_samples),
            "block_size": int(self.block_size),
        }
