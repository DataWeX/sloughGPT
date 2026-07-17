"""
PromptFormatter — model-agnostic message→prompt conversion + output cleaning.

Handles two directions:
  User input (messages dicts) → Model prompt (string)
  Model output (tokens/chunks) → Clean user-facing text

Model families supported:
  - Chat-tuned (Qwen, TinyLlama, etc.) via tokenizer.apply_chat_template()
  - Base (GPT-2, etc.) via User:/Assistant: formatting
  - Custom via injected format_fn
"""

from __future__ import annotations

import re
from typing import Any, Callable, List, Optional, Protocol


class PromptFormatterProtocol(Protocol):
    def messages_to_prompt(self, messages: List[dict]) -> str: ...
    def clean_chunk(self, chunk: str, *, first: bool = False) -> str: ...


# Patterns for stripping model artifacts
_STRIP_SPECIAL_TOKENS = re.compile(
    r"<\|im_start\|>\s*(user|assistant|system)\s*|"
    r"<\|start_header_id\|>\s*(user|assistant|system)\s*<\|end_header_id\|>\s*\n*|"
    r"<\|eot_id\|>|<\|im_end\|>"
)
_STRIP_THINKING = re.compile(
    r"<think>.*?</think>",
    re.DOTALL,
)
_STRIP_UNTERMINATED_THINK = re.compile(
    r"<think>.*",
    re.DOTALL,
)
_STRIP_ORPHAN_THINK_CLOSE = re.compile(
    r"</think>",
)
_STRIP_TOOL_CALLS = re.compile(
    r"<tool_call>.*?</tool_call>",
    re.DOTALL,
)
_STRIP_UNTERMINATED_TOOL = re.compile(
    r"<tool_call>.*",
    re.DOTALL,
)
_STRIP_ORPHAN_TOOL_CLOSE = re.compile(
    r"</tool_call>",
)
_STRIP_LEADING_NL = re.compile(r"^\n+")
_STRIP_ASSISTANT_PREFIX = re.compile(r"^(?:\s*\n)*Assistant:\s*")
_STRIP_USER_PREFIX = re.compile(r"^(?:\s*\n)*User:\s*")
_STRIP_INSTRUCTIONS = re.compile(
    r"\[PERSONALITY INSTRUCTIONS\].*?(?=Assistant:|$)", re.DOTALL
)
_STRIP_KNOWLEDGE = re.compile(
    r"\[KNOWLEDGE\].*?\[/KNOWLEDGE\]\s*", re.DOTALL
)


class PromptFormatter:
    """
    Formats chat messages into model-appropriate prompt strings and
    cleans model output chunks back into natural text.

    Args:
        tokenizer: Optional tokenizer with ``apply_chat_template``.
        user_prefix: Label for user messages (default ``"User"``).
        assistant_prefix: Label for assistant messages (default ``"Assistant"``).
        format_fn: Custom format function (overrides all built-in logic).
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
        # Streaming state: tracks partial tags across chunks
        self._in_think = False
        self._in_tool_call = False

    def messages_to_prompt(self, messages: List[dict]) -> str:
        """Convert messages to a prompt string.

        Resolution: format_fn → apply_chat_template → User:/Assistant: base format.
        """
        if self._format_fn is not None:
            return self._format_fn(messages)

        if self._has_chat_template():
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass

        return self._base_format(messages)

    def clean_chunk(self, chunk: str, *, first: bool = False) -> str:
        """Clean a single token chunk. Stateful across chunks for thinking/tool_call blocks."""
        # Strip structural tokens (stateless)
        cleaned = _STRIP_SPECIAL_TOKENS.sub("", chunk)
        cleaned = _STRIP_INSTRUCTIONS.sub("", cleaned)
        cleaned = _STRIP_KNOWLEDGE.sub("", cleaned)

        # Stateful thinking/tool_call stripping across chunks
        result = []
        i = 0
        n = len(cleaned)
        while i < n:
            if self._in_think:
                end = cleaned.find("</think>", i)
                if end != -1:
                    self._in_think = False
                    i = end + len("</think>")
                else:
                    break  # rest of chunk is inside think block
            elif self._in_tool_call:
                end = cleaned.find("</tool_call>", i)
                if end != -1:
                    self._in_tool_call = False
                    i = end + len("</tool_call>")
                else:
                    break  # rest of chunk is inside tool block
            else:
                think_start = cleaned.find("<think>", i)
                tool_start = cleaned.find("<tool_call>", i)

                # Find which comes first (or neither)
                next_start = n
                if think_start != -1:
                    next_start = min(next_start, think_start)
                if tool_start != -1:
                    next_start = min(next_start, tool_start)

                if next_start == n:
                    # No tags found -- emit rest
                    result.append(cleaned[i:])
                    break
                else:
                    # Emit text before the tag
                    result.append(cleaned[i:next_start])
                    if next_start == think_start:
                        self._in_think = True
                        i = next_start + len("<thought>")
                    else:
                        self._in_tool_call = True
                        i = next_start + len("<tool_call>")

        out = "".join(result)

        if first:
            out = _STRIP_LEADING_NL.sub("", out)
            out = _STRIP_ASSISTANT_PREFIX.sub("", out)
            out = _STRIP_USER_PREFIX.sub("", out)

        return out

    def clean_response(self, text: str) -> str:
        """Clean a full response (non-streamed)."""
        text = _STRIP_SPECIAL_TOKENS.sub("", text)
        text = _STRIP_INSTRUCTIONS.sub("", text)
        text = _STRIP_KNOWLEDGE.sub("", text)
        # Strip completed and unterminated thinking blocks
        text = _STRIP_THINKING.sub("", text)
        text = _STRIP_UNTERMINATED_THINK.sub("", text)
        text = _STRIP_ORPHAN_THINK_CLOSE.sub("", text)
        # Strip completed and unterminated tool_call blocks
        text = _STRIP_TOOL_CALLS.sub("", text)
        text = _STRIP_UNTERMINATED_TOOL.sub("", text)
        text = _STRIP_ORPHAN_TOOL_CLOSE.sub("", text)
        # Strip any trailing role markers
        text = re.sub(r"\n*\s*(?:User|Assistant):\s*$", "", text)
        return text.strip()

    def _has_chat_template(self) -> bool:
        return (
            self._tokenizer is not None
            and hasattr(self._tokenizer, "chat_template")
            and self._tokenizer.chat_template is not None
        )

    def _base_format(self, messages: List[dict]) -> str:
        """Base model prompt: ``User: ...\\n\\nAssistant:``

        System messages are skipped for base models.
        """
        parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                pass  # Skip for base models
            elif role == "user":
                parts.append(f"{self._user_prefix}: {content}")
            elif role == "assistant":
                parts.append(f"{self._assistant_prefix}: {content}")
        parts.append(f"{self._assistant_prefix}:")
        return "\n\n".join(parts)
