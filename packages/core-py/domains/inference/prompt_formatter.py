"""
PromptFormatter — model-agnostic message↔prompt conversion.

Handles two directions:
  User input (messages dicts) → Model prompt (string)
  Model output (tokens/chunks) → Clean user-facing text

Model families supported:
  - Chat-tuned (Qwen, TinyLlama, etc.) via tokenizer.apply_chat_template()
  - Base (GPT-2, etc.) via User:/Assistant: formatting
  - Instruct (LLaMA 3, Mistral) via <|start_header_id|>…<|end_header_id|> templates
  - Custom via injected format_fn
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Optional, Protocol


# ── Protocol ──────────────────────────────────────────────────────────────────

class PromptFormatterProtocol(Protocol):
    """Minimal interface a formatter must satisfy."""

    def messages_to_prompt(self, messages: List[dict]) -> str: ...
    def clean_chunk(self, chunk: str, *, first: bool = False) -> str: ...


# ── Artifact patterns ─────────────────────────────────────────────────────────

_STRIP_LEADING_NL = re.compile(r"^\n+")
_STRIP_ASSISTANT = re.compile(r"^[\s\n]*Assistant:\s*", re.IGNORECASE)
_STRIP_USER = re.compile(r"^[\s\n]*User:\s*", re.IGNORECASE)
_STRIP_SPECIAL_TOKENS = re.compile(
    r"<\|im_start\|>\s*(user|assistant|system)\s*|"
    r"<\|start_header_id\|>\s*(user|assistant|system)\s*<\|end_header_id\|>\s*\n*|"
    r"<\|eot_id\|>|<\|im_end\|>"
)


# ── Formatter ─────────────────────────────────────────────────────────────────


class PromptFormatter:
    """
    Formats chat messages into model-appropriate prompt strings and
    cleans model output chunks back into natural text.

    Args:
        tokenizer: Optional tokenizer with ``apply_chat_template`` and
                   ``chat_template`` attributes.
        user_prefix: Label used for user messages (default ``"User"``).
        assistant_prefix: Label used for assistant messages / generation
                         prompt suffix (default ``"Assistant"``).

    Usage::

        fmt = PromptFormatter(tokenizer=tokenizer)
        prompt = fmt.messages_to_prompt(messages)
        for chunk in stream:
            yield fmt.clean_chunk(chunk, first=first)
            first = False
    """

    def __init__(
        self,
        tokenizer: Any = None,
        user_prefix: str = "User",
        assistant_prefix: str = "Assistant",
        format_fn: Optional[Callable[[List[dict]], str]] = None,
    ):
        self._tokenizer = tokenizer
        self._user_prefix = user_prefix
        self._assistant_prefix = assistant_prefix
        self._format_fn = format_fn

    # ── Prompt building ──────────────────────────────────────────────────────

    def messages_to_prompt(self, messages: List[dict]) -> str:
        """
        Convert a list of ``{role, content}`` dicts to a prompt string.

        Resolution order:
          1. Custom ``format_fn`` (injected at init).
          2. ``tokenizer.apply_chat_template()`` (chat-tuned models).
          3. ``User:/Assistant:`` prefix format (base models).
        """
        if self._format_fn is not None:
            return self._format_fn(messages)

        if self._has_chat_template():
            try:
                return self._tokenizer.apply_chat_template(  # type: ignore
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        return self._base_format(messages)

    # ── Chunk cleaning ───────────────────────────────────────────────────────

    def clean_chunk(self, chunk: str, *, first: bool = False) -> str:
        """
        Strip model-specific artifacts from a generated chunk.

        When ``first=True`` (first chunk of a generation) applies
        aggressive cleaning — strips leading whitespace, ``Assistant:``
        prefix, special tokens, and ``User:`` echoes.
        """

        # 1) Strip special tokens regardless
        cleaned = _STRIP_SPECIAL_TOKENS.sub("", chunk)

        if not first:
            return cleaned

        # 2) First chunk: aggressive cleaning
        cleaned = _STRIP_LEADING_NL.sub("", cleaned)
        cleaned = _STRIP_ASSISTANT.sub("", cleaned)
        cleaned = _STRIP_USER.sub("", cleaned)
        return cleaned

    def clean_response(self, text: str) -> str:
        """
        Strip artifacts from a full (non-streamed) response string.
        """
        text = _STRIP_SPECIAL_TOKENS.sub("", text)
        text = _STRIP_ASSISTANT.sub("", text)
        return text

    # ── Internals ────────────────────────────────────────────────────────────

    def _has_chat_template(self) -> bool:
        return (
            self._tokenizer is not None
            and hasattr(self._tokenizer, "chat_template")
            and self._tokenizer.chat_template is not None
        )

    def _base_format(self, messages: List[dict]) -> str:
        """Base model prompt: ``User: ...\\n\\nAssistant: ...\\n\\nAssistant:``"""
        parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(content)
            elif role == "user":
                parts.append(f"{self._user_prefix}: {content}")
            elif role == "assistant":
                parts.append(f"{self._assistant_prefix}: {content}")
        parts.append(f"{self._assistant_prefix}:")
        return "\n\n".join(parts)
