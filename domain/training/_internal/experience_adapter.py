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


def _build_experience_text(
    pairs: list[dict[str, str]],
    block_size: int | None = None,
    use_prompt_engine: bool = True,
) -> str:
    """Build well-structured training text from owned pairs.

    Structure is the single most important factor for owned-data training.
    Requirements:
      - Unambiguous role boundaries (model must never confuse user vs assistant)
      - Train ≡ serve (same template as inference, so learned delimiters transfer)
      - No collision with user content (delimiters are special tokens, not plain "User:")
      - Explicit turn boundaries (each pair is a complete turn, not ad-hoc lines)

    Format (Qwen-style, also used by NativeEngine/SloNetChatProvider via PromptEngine):
      <|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>\n

    When a tokenizer with apply_chat_template is available and use_prompt_engine=True,
    we delegate to PromptEngine for 100% train/serve identity. Otherwise we use
    the explicit fallback above, which is still unambiguous and mirrors native/engine.

    Why not "User: {u}\\nAssistant: {a}\\n\\n":
      - "User:" and "Assistant:" appear verbatim inside user messages ("User: ...")
      - No explicit end marker — model can't tell where assistant ends and next user begins
      - Bare newlines are weak boundaries; the model must guess turn structure

    The explicit <|im_start|>/<|im_end|> tokens are multi-char but atomic at the
    char-level vocab — they tokenize deterministically and become strong learned
    boundaries. For BPE vocab they map to single tokens; for char vocab they are
    still unique sequences the model can latch onto.

    Args:
        pairs: list of {user_msg, assistant_msg}
        block_size: unused for text building, kept for API compat
        use_prompt_engine: try PromptEngine first if True

    Returns:
        Single concatenated string ready for char/BPE tokenization.
    """
    # Try PromptEngine for 100% train/serve parity when available
    if use_prompt_engine:
        try:
            from domain.inference._internal.prompt_engine import render_prompt

            # Build one text per pair via the same engine that serves inference.
            # render_prompt expects list[dict] messages; we render each pair as
            # a 2-turn conversation and strip the trailing generation prompt.
            parts: list[str] = []
            for p in pairs:
                msgs = [
                    {"role": "user", "content": p["user_msg"]},
                    {"role": "assistant", "content": p["assistant_msg"]},
                ]
                rendered = render_prompt(msgs, model_type="qwen2")
                # Qwen fallback appends a trailing generation prompt
                # "<|im_start|>assistant\n" after the completed assistant turn —
                # for training we want the *closed* turn only, so strip it.
                trail = "<|im_start|>assistant\n"
                if rendered.endswith(trail):
                    rendered = rendered[: -len(trail)]
                elif rendered.endswith("<|im_start|>assistant"):
                    rendered = rendered[: -len("<|im_start|>assistant")]
                # Ensure each turn ends with <|im_end|>\n so boundaries are explicit
                if not rendered.endswith("\n"):
                    rendered += "\n"
                parts.append(rendered)
            # Join turns — each already ends with a delimiter, no extra separator needed
            return "".join(parts)
        except Exception as e:
            logger.debug("PromptEngine render failed, using explicit fallback: %s", e)

    # Explicit fallback — well-structured, unambiguous, never collides
    IMS, IME = "<|im_start|>", "<|im_end|>"
    out: list[str] = []
    for p in pairs:
        u = p["user_msg"].strip()
        a = p["assistant_msg"].strip()
        out.append(f"{IMS}user\n{u}{IME}\n{IMS}assistant\n{a}{IME}\n")
    return "".join(out)


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

        # Build flat ids — STRUCTURE IS CRITICAL for owned-data quality.
        # Use the same chat template as inference (PromptEngine) so train ≡ serve.
        # Fallback is explicit role tags that never collide with user text.
        # Format per pair (well-structured, unambiguous, role-isolated):
        #   <|im_start|>user\n{user}\n<|im_end|>\n<|im_start|>assistant\n{assistant}\n<|im_end|>\n
        # For char-level vocab without special tokens, this still tokenizes
        # deterministically and the model learns the delimiter as a boundary.
        # We use PromptEngine when a tokenizer is available; else the explicit
        # fallback above (mirrors native/engine format_chat for Qwen).
        text = _build_experience_text(pairs, block_size=block_size)
        self.ids = np.array([stoi.get(c, 0) for c in text], dtype=np.int32)
        self._text = text  # keep for debugging / structure inspection
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
