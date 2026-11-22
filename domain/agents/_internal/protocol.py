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
from functools import lru_cache
from typing import Any

CANONICAL_RE = re.compile(r"```(tool_call|json)\s*\n(.*?)\n\s*```", re.DOTALL)
# An UNCLOSED fence: the model hits finish_reason=stop (or max_tokens)
# before emitting the closing backticks — live-observed on the Phase 2
# journey walk (card 20261116_090332): "```tool_call\n{...}" with no
# close. Runs to EOF — an unclosed fence swallows the rest by definition.
UNCLOSED_RE = re.compile(r"```(tool_call|json)\s*\n(.*)\Z", re.DOTALL)
LEGACY_TOOL_RE = re.compile(r"\[\[TOOL:\s*(\w+)\s*\]\]\s*([^\n]+)")
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
# One ``key: value`` line of the informal body walk_img10 carried
# (``prompt: "A blue circle"`` — YAML-ish, not JSON).
_KV_LINE_RE = re.compile(r"^\s*([A-Za-z_][\w-]*)\s*:\s*(.*?)\s*$")


@lru_cache(maxsize=64)
def _toolnamed_re(tool_names: frozenset[str], unclosed: bool) -> re.Pattern[str]:
    """Fence pattern for tags naming a REGISTERED tool (walk_img10:
    `` ```image_gen ``). Built per registry — a tag only counts as wire
    format when it names a tool this loop can actually run, so `` ```python
    `` never matches here. The closer tolerates a mangled tail word
    (`` ```image `` — the live walk's missing ``_gen``) so it does not
    leak into the residual. Longest tag first so a tool named ``image``
    cannot shadow ``image_gen``."""
    tags = "|".join(sorted((re.escape(t) for t in tool_names), key=len, reverse=True))
    if unclosed:
        return re.compile(rf"```({tags})\s*\n(.*)\Z", re.DOTALL)
    return re.compile(rf"```({tags})\s*\n(.*?)\n\s*```\w*", re.DOTALL)


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
    if "args" in item or "arguments" in item:
        args = item.get("args", item.get("arguments"))
        if args is None:
            args = {}
    else:
        # Flat shape (live-observed): {"name": ..., "query": "..."} — the
        # model puts parameters as siblings of name. Carry them instead of
        # silently dropping to {}: validate_args can then name the mismatch
        # ("unknown parameter 'query'") and round-2 feedback corrects it.
        args = {k: v for k, v in item.items() if k not in ("name", "tool")}
    if not isinstance(args, dict):
        raise ProtocolError(f"args for '{name}' must be an object", position=position)
    return ToolCall(name=name, args=dict(args))


def _repair_trailing_commas(raw: str) -> str:
    """Strip trailing commas before ``}``/``]`` — the 0.5B model's most
    common JSON sloppiness (live-observed on the Phase 2 journey walk,
    card 20261116_090332)."""
    return _TRAILING_COMMA_RE.sub(r"\1", raw)


def _repair_unbalanced(raw: str) -> str:
    """Append the missing ``}``/``]`` closers — walk_img7 bytes: the
    payload was missing its outer brace (``{"tool_calls": [{...}]``).
    Scans outside strings; an unterminated string is NOT repairable
    (truncation mid-string — the closers would land inside the string),
    so the raw text is returned and the strict floor still applies."""
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in raw:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
    if in_str or not stack:
        return raw
    return raw + "".join(reversed(stack))


def _parse_repairing(raw: str) -> Any | None:
    """Parse with the repair ladder: strict → trailing-comma → balance
    (comma+balance combined). None = unrepairable (no keyword guessing)."""
    for candidate in (raw, _repair_trailing_commas(raw)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    balanced = _repair_unbalanced(_repair_trailing_commas(raw))
    if balanced != raw:
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            pass
    return None


def _looks_like_tool_call(payload: Any) -> bool:
    """Tool-shaped = a call name, or a ``tool_calls`` list (json tag gate)."""
    if not isinstance(payload, dict):
        return False
    if "tool_calls" in payload:
        return isinstance(payload["tool_calls"], list)
    return bool(payload.get("name") or payload.get("tool"))


def _parse_kv_body(raw: str) -> dict[str, Any] | None:
    """Parse an informal ``key: value`` body (walk_img10's YAML-ish block).

    Every non-empty line must be a pair — prose that merely starts with a
    word returns None (no keyword guessing: the block stays code, never a
    call). Values are JSON when they parse (quoted strings, nested objects
    like the walk's hallucinated ``output: {...}`` line), verbatim text
    otherwise (``style: cyberpunk``).
    """
    args: dict[str, Any] = {}
    for line in raw.splitlines():
        if not line.strip():
            continue
        m = _KV_LINE_RE.match(line)
        if not m:
            return None
        key, val = m.group(1), m.group(2)
        if not val:
            return None
        parsed = _parse_repairing(val)
        args[key] = parsed if parsed is not None else val
    return args or None


def _parse_tool_named_payload(tag: str, payload: Any) -> list[ToolCall] | None:
    """Turn a parsed body from a tool-named fence into calls.

    ``None`` = not call-shaped (stays prose). The tag names the tool, so a
    flat body is that tool's args; an explicit ``name``/``tool_calls``
    still wins where present.
    """
    if isinstance(payload, dict) and "tool_calls" in payload:
        items = payload["tool_calls"]
        if not isinstance(items, list):
            return None
        out: list[ToolCall] = []
        for item in items:
            if not isinstance(item, dict):
                return None
            name = item.get("name") or item.get("tool") or tag
            if not isinstance(name, str) or not name:
                return None
            args = item.get("args", item.get("arguments"))
            if args is None:
                args = {k: v for k, v in item.items() if k not in ("name", "tool")}
            if not isinstance(args, dict):
                return None
            out.append(ToolCall(name=name, args=dict(args)))
        return out
    if not isinstance(payload, dict):
        return None
    name = payload.get("name") or payload.get("tool") or tag
    if not isinstance(name, str) or not name:
        return None
    if "args" in payload or "arguments" in payload:
        args = payload.get("args", payload.get("arguments"))
        if args is None:
            args = {}
    else:
        args = {k: v for k, v in payload.items() if k not in ("name", "tool")}
    if not isinstance(args, dict):
        return None
    return [ToolCall(name=name, args=dict(args))]


def _decode_tool_named(text: str, calls: list[ToolCall], tool_names: frozenset[str]) -> str:
    """Decode fences whose tag names a registered tool (walk_img10:
    `` ```image_gen `` + a key: value body + a mangled `` ```image ``
    closer). Body is JSON-first, then key: value lines; anything
    unparseable stays prose — an informal tag never raises."""
    if not tool_names:
        return text

    def _handle(tag: str, raw: str) -> list[ToolCall] | None:
        if not raw:
            return None  # bare header — prose
        payload = _parse_repairing(raw)
        if payload is not None:
            return _parse_tool_named_payload(tag, payload)
        kv = _parse_kv_body(raw)
        if kv is None:
            return None
        return [ToolCall(name=tag, args=kv)]

    def _sub(m: re.Match[str]) -> str:
        parsed = _handle(m.group(1), m.group(2).strip())
        if parsed is None:
            return m.group(0)
        calls.extend(parsed)
        return ""

    residual = _toolnamed_re(tool_names, unclosed=False).sub(_sub, text)

    def _sub_unclosed(m: re.Match[str]) -> str:
        parsed = _handle(m.group(1), m.group(2).strip())
        if parsed is None:
            return m.group(0)
        calls.extend(parsed)
        return ""

    return _toolnamed_re(tool_names, unclosed=True).sub(_sub_unclosed, residual)


def _decode_canonical(text: str, calls: list[ToolCall]) -> str:
    def _consume(m: re.Match[str], payload: Any) -> str:
        if m.group(1) == "json" and not _looks_like_tool_call(payload):
            return m.group(0)  # a plain JSON snippet — code, not a call
        calls.extend(_parse_canonical_payload(payload, m.start()))
        return ""

    def _sub(m: re.Match[str]) -> str:
        tag, raw = m.group(1), m.group(2).strip()
        payload = _parse_repairing(raw)
        if payload is None:
            if tag == "tool_call":
                # Strictness floor: UNREPAIRABLE canonical JSON raises —
                # repairable near-misses (trailing comma, missing brace)
                # execute instead of degrading to a raw leak.
                raise ProtocolError(
                    "invalid tool_call JSON: unrepairable payload", position=m.start()
                )
            # Informal ``json`` tag carries no contract: the block is
            # just code — prose, never an error.
            return m.group(0)
        return _consume(m, payload)

    residual = CANONICAL_RE.sub(_sub, text)

    # Unclosed fence (model stopped before the closing backticks): same
    # tag rules, block runs to EOF. Closed fences were already consumed
    # above, so any remaining header is genuinely unterminated.
    def _sub_unclosed(m: re.Match[str]) -> str:
        tag, raw = m.group(1), m.group(2).strip()
        if not raw:
            return m.group(0)  # bare header, nothing to parse — prose
        payload = _parse_repairing(raw)
        if payload is None:
            if tag == "tool_call":
                raise ProtocolError(
                    "invalid tool_call JSON: unrepairable payload", position=m.start()
                )
            return m.group(0)
        return _consume(m, payload)

    return UNCLOSED_RE.sub(_sub_unclosed, residual)


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


def decode(
    text: str, tool_names: set[str] | frozenset[str] | None = None
) -> tuple[list[ToolCall], str]:
    """Extract tool calls from model text.

    Returns ``(calls, residual)`` where ``residual`` is the text with both
    codec blocks stripped (what the user should see as prose). Raises
    :class:`ProtocolError` on a canonical block that does not parse.

    ``tool_names`` — the registry this loop resolves against — additionally
    accepts fences whose TAG names a registered tool (the model inventing
    `` ```image_gen `` instead of `` ```tool_call ``, live walk #4, card
    20261116_090332). Without it, only the two protocol tags are calls.
    """
    calls: list[ToolCall] = []
    residual = _decode_canonical(text, calls)
    residual = _decode_tool_named(residual, calls, frozenset(tool_names or ()))
    residual = _decode_legacy(residual, calls)
    return calls, residual.strip()


def encode_tool_block(calls: list[ToolCall]) -> str:
    """Render the canonical fenced block (symmetry helper for tests/fakes)."""
    if len(calls) == 1:
        payload: Any = {"name": calls[0].name, "args": calls[0].args}
    else:
        payload = {"tool_calls": [{"name": c.name, "args": c.args} for c in calls]}
    return "```tool_call\n" + json.dumps(payload) + "\n```"
