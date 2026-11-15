"""Agent tool-call protocol — one parser, two codecs.

The canonical wire format is a fenced JSON block the model emits in its text:

    ```tool_call
    {"name": "calculator", "args": {"expression": "2+2"}}
    ```

    or multiple calls in one turn:

    ```tool_call
    {"tool_calls": [
        {"name": "calculator", "args": {"expression": "2+2"}},
        {"name": "current_time", "args": {}}
    ]}
    ```

The legacy ``[[TOOL: name]] arg`` marker (single non-space token as args —
``domain/models/_internal/provider/processors/tool_use.py``) is kept as one
codec so existing provider loops migrate without a flag day. It cannot carry
structured args; a JSON object token is accepted, anything else is wrapped
under ``_raw``.

Usage:
    calls, residual = decode(text)          # raises ProtocolError on bad JSON
    block = encode_tool_block(calls)        # canonical block, for tests/fakes
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

CANONICAL_RE = re.compile(r"```tool_call\s*\n(.*?)\n\s*```", re.DOTALL)
LEGACY_TOOL_RE = re.compile(r"\[\[TOOL:\s*(\w+)\s*\]\]\s*([^\n]+)")


class ProtocolError(ValueError):
    """Strict decode failure — never keyword-fallback guess."""

    def __init__(self, message: str, position: int | None = None) -> None:
        super().__init__(message)
        self.position = position


@dataclass(frozen=True)
class ToolCall:
    """Canonical in-memory tool invocation (mirrors ToolSpec.parameters)."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)


def _parse_canonical_payload(payload: Any, position: int) -> list[ToolCall]:
    if isinstance(payload, dict) and "tool_calls" in payload:
        items = payload["tool_calls"]
        if not isinstance(items, list):
            raise ProtocolError("'tool_calls' must be a list", position=position)
        return [_parse_call_item(item, position) for item in items]
    return [_parse_call_item(payload, position)]


def _parse_call_item(item: Any, position: int) -> ToolCall:
    if not isinstance(item, dict):
        raise ProtocolError("tool call must be a JSON object", position=position)
    name = item.get("name") or item.get("tool")
    if not isinstance(name, str) or not name:
        raise ProtocolError("tool call missing 'name'", position=position)
    args = item.get("args", item.get("arguments", {}))
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ProtocolError(f"args for '{name}' must be an object", position=position)
    return ToolCall(name=name, args=dict(args))


def _decode_canonical(text: str, calls: list[ToolCall]) -> str:
    def _sub(m: re.Match[str]) -> str:
        raw = m.group(1).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ProtocolError(f"invalid tool_call JSON: {e}", position=m.start()) from e
        calls.extend(_parse_canonical_payload(payload, m.start()))
        return ""

    return CANONICAL_RE.sub(_sub, text)


def _decode_legacy(text: str, calls: list[ToolCall]) -> str:
    def _sub(m: re.Match[str]) -> str:
        name, rest = m.group(1), m.group(2).strip()
        if not rest:
            raise ProtocolError(
                f"legacy marker [[TOOL: {name}]] missing argument", position=m.start()
            )
        try:
            parsed = json.loads(rest)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            calls.append(ToolCall(name=name, args=parsed))
            return ""
        # Single-token legacy arg: first token is the argument; the remainder
        # stays as prose (matches the original \S+ marker semantics).
        parts = rest.split(None, 1)
        token = parts[0]
        leftover = (" " + parts[1]) if len(parts) > 1 else ""
        try:
            tok_parsed = json.loads(token)
        except json.JSONDecodeError:
            tok_parsed = None
        args: dict[str, Any] = (
            tok_parsed
            if isinstance(tok_parsed, dict)
            else {"_raw": tok_parsed if tok_parsed is not None else token}
        )
        calls.append(ToolCall(name=name, args=args))
        return leftover

    return LEGACY_TOOL_RE.sub(_sub, text)


def decode(text: str) -> tuple[list[ToolCall], str]:
    """Extract tool calls from model text.

    Returns ``(calls, residual)`` where ``residual`` is the text with both
    codec blocks stripped (what the user should see as prose). Raises
    :class:`ProtocolError` on a canonical block that does not parse.
    """
    calls: list[ToolCall] = []
    residual = _decode_canonical(text, calls)
    residual = _decode_legacy(residual, calls)
    return calls, residual.strip()


def encode_tool_block(calls: list[ToolCall]) -> str:
    """Render the canonical fenced block (symmetry helper for tests/fakes)."""
    if len(calls) == 1:
        payload: Any = {"name": calls[0].name, "args": calls[0].args}
    else:
        payload = {"tool_calls": [{"name": c.name, "args": c.args} for c in calls]}
    return "```tool_call\n" + json.dumps(payload) + "\n```"
