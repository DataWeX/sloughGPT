"""
PromptEngine — single chat-template renderer.

Wraps MorphTokenizer.apply_chat_template as primary; a tokenizer bound
without a chat template falls back to raw last-message content, and the
legacy format_chat (Qwen/LLaMA/GPT2) shape is reserved for when no
tokenizer is bound at all.
All call sites (NativeEngine, SloNetChatProvider, ChatDomain) delegate
here so prompt strings stay bit-identical across the stack.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("slo.inference.prompt_engine")


def _fallback_format_chat(
    messages: list[dict[str, str]], model_type: str = "qwen2", system: str = ""
) -> str:
    """Minimal fallback when no tokenizer is available (mirrors native/engine format_chat)."""
    # Import here to avoid cycle.
    try:
        from .native.engine import format_chat as _fmt  # type: ignore

        return _fmt(messages, model_type, system)
    except Exception:
        # Ultimate fallback: System/User/Assistant lines (ChatDomain legacy).
        parts: list[str] = []
        if system:
            parts.append(f"System: {system}")
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)


def render_prompt(
    messages: Any,
    *,
    tokenizer: Any | None = None,
    model_type: str = "qwen2",
    system: str = "",
) -> str:
    """
    Render a chat prompt from messages.

    Args:
        messages: list[dict] | list[str] | str | None — normalized to list[dict].
        tokenizer: MorphTokenizer with apply_chat_template (preferred).
        model_type: fallback model family for format_chat.
        system: extra system prompt prepended as a system message.

    Returns:
        Prompt string ready for tokenization.
    """
    if not messages:
        return ""

    # Preserve legacy shortcuts exactly as SloNetChatProvider did.
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list) and messages and isinstance(messages[0], str):
        return messages[-1]

    # Normalize list[dict] shapes.
    if isinstance(messages, list):
        # Ensure dicts have role/content.
        norm: list[dict[str, str]] = []
        for m in messages:
            if isinstance(m, dict):
                norm.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        messages = norm

    # Prepend system if given.
    if system:
        messages = [{"role": "system", "content": system}] + list(messages)  # type: ignore

    # Primary: tokenizer chat template.
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        try:
            rendered = tokenizer.apply_chat_template(messages)  # type: ignore[attr-defined]
            if rendered:
                return rendered
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "PromptEngine: apply_chat_template failed, falling back: %s",
                exc,
                extra={"tag": "MODEL"},
            )

    # Tokenizer bound but with no apply_chat_template: emit the last message
    # content raw, so the prompt text stays aligned with what the tokenizer
    # can actually encode (never invent a template the tokenizer lacks).
    if tokenizer is not None:
        for m in reversed(messages):  # type: ignore[arg-type]
            content = m.get("content", "")
            if content:
                return content
        return ""

    return _fallback_format_chat(messages, model_type, system="")


# Back-compat alias for call sites that import PromptEngine class.
class PromptEngine:
    """Thin class wrapper around render_prompt for DI."""

    def __init__(self, tokenizer: Any | None = None, model_type: str = "qwen2"):
        self._tokenizer = tokenizer
        self._model_type = model_type

    def render(self, messages: Any, system: str = "") -> str:
        return render_prompt(
            messages, tokenizer=self._tokenizer, model_type=self._model_type, system=system
        )
