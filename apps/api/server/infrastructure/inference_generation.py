"""Shared provider-backed text generation for inference API routes."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationResult:
    text: str
    tokens_generated: int
    elapsed_ms: float


def count_generated_tokens(text: str, tokenizer: Any = None) -> int:
    """Count output tokens consistently, falling back to whitespace words."""
    try:
        if tokenizer is not None and hasattr(tokenizer, "encode"):
            return len(tokenizer.encode(text))
    except Exception:
        pass
    return len(text.split())


async def generate_text(
    provider: Any,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    generation_params: Mapping[str, Any],
    tokenizer: Any = None,
) -> GenerationResult:
    """Run one provider generation and return shared text/token/timing data."""
    started = time.monotonic()
    text = await provider.chat(messages, max_tokens=max_tokens, **generation_params)
    elapsed_ms = (time.monotonic() - started) * 1000
    return GenerationResult(
        text=text,
        tokens_generated=count_generated_tokens(text, tokenizer),
        elapsed_ms=elapsed_ms,
    )
