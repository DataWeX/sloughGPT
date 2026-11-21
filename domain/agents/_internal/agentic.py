"""Agentic chat turn — AgentLoop behind the engine port, streaming.

The domain seam ``stream_turn`` wraps to give chat model-driven intent
routing (card 20261116_090332): the canonical :class:`AgentLoop` decides via
the fenced ```tool_call protocol (never a pattern table), while this adapter
keeps the STREAM contract the chat surface depends on:

* **Tokens forward live** — round text is pushed to the caller as it is
  generated, exactly like today's plain path.
* **Fence tails hold** — once buffered text is (or may become) the
  ```tool_call header, those bytes are NOT forwarded: the model's raw wire
  format must never reach the user. A diverging tail (a code block) flushes
  immediately; a confirming tail becomes tool traffic the loop executes.
* **Degradation is lossless** — at loop end, the final round's held/dropped
  tail is flushed, so a malformed fence still yields today's plain answer
  (preamble was streamed; the tail arrives whole).
* **Attachments stay in-process** — a tool's ``attachment`` payload is
  captured by wrapping ``tools.execute`` and appended after the final text
  as inline markdown; events never carry it (SSE stays small).

Yields ``str`` tokens and ``{"kind": <event kind>, "data": <payload>}``
dicts (tool_call / tool_result) — edges project the dicts (chat maps them to
``phase: TOOL`` frames) and pass the strings through as tokens.

Usage:
    async for item in agentic_stream(messages, tools=registry, stream_fn=...):
        ...
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from domain.agents._internal.loop import AgentLoop
from domain.agents._internal.provider_infer import _prepare_messages, render_tool_contract
from domain.agents._internal.tools import ToolRegistry

FENCE = "```tool_call"
# Both headers the model uses for a tool block: the canonical contract tag
# and the informal ``json`` variant it emits spontaneously (live-observed on
# the Phase 2 journey walk, card 20261116_090332). Held identically so
# ``decode`` gets a chance; a json block that decodes to nothing flushes as
# prose at round end.
FENCE_HEADERS = (FENCE, "```json")


def _fence_head(text: str) -> str:
    """The known fence header ``text`` matches — or "" for a divergence.

    Asks the shortening question against both headers: a partial header
    (`` ``` `` … `` ```js ``) holds, a complete one confirms, anything else
    is a code block / stray backticks and passes through."""
    return next((h for h in FENCE_HEADERS if text.startswith(h)), "")


StreamFn = Callable[[list[dict[str, Any]]], AsyncIterator[str]]


@dataclass
class _GuardState:
    """Per-turn streaming guard. Reset per round: ``held``, ``confirmed``,
    ``round_dropped``, ``round_had_calls`` (set by the driver on tool events,
    judged once at loop end — the FINAL round decides what must flush)."""

    held: str = ""
    confirmed: bool = False
    round_dropped: str = ""
    round_had_calls: bool = False


def _guard(state: _GuardState, token: str) -> str | None:
    """Route one token: forward now (str), or hold/drop (None)."""
    if state.confirmed:
        state.round_dropped += token
        return None
    if state.held:
        text = state.held + token
        state.held = ""
        head = ""
    elif "```" in token:
        i = token.index("```")
        head, text = token[:i], token[i:]
    else:
        return token
    matched = _fence_head(text)
    if matched:
        state.confirmed = True
        state.round_dropped += text
        return head or None
    if any(h.startswith(text) for h in FENCE_HEADERS):
        state.held = text
        return head or None
    # Divergence — not (yet) the fence header: a code block or stray backticks.
    return head + text


class _AttachmentTap:
    """Wraps a ToolRegistry to capture ``attachment`` payloads out-of-band.

    Duck-typed to the three members AgentLoop touches (get / list_tools /
    execute) — the loop and its events stay byte-identical; the bytes the
    user will render never enter the model's context or an SSE frame.
    """

    def __init__(self, tools: ToolRegistry) -> None:
        self._tools = tools
        self.attachments: list[dict[str, Any]] = []

    def get(self, name: str):
        return self._tools.get(name)

    def list_tools(self):
        return self._tools.list_tools()

    async def execute(self, name, args, approve=None):
        result = await self._tools.execute(name, args, approve=approve)
        att = result.metadata.get("attachment")
        if isinstance(att, dict):
            self.attachments.append(att)
        return result


def _task_text(messages: list[dict[str, Any]]) -> str:
    """The run's display name: the latest user message (plan event only)."""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
            return ""
    return ""


async def agentic_stream(
    messages: list[dict[str, Any]],
    *,
    tools: ToolRegistry,
    stream_fn: StreamFn,
    max_iters: int = 3,
) -> AsyncIterator[str | dict[str, Any]]:
    """Drive one agentic chat turn: loop events + live tokens, one stream.

    ``stream_fn(prepared_messages)`` runs ONE provider round (params bound by
    the caller — the engine port's own cancellation/sampling live there); the
    adapter injects the tool contract and re-attributes ``tool``-role results
    per round via the canonical provider helpers.
    """
    state = _GuardState()
    tap = _AttachmentTap(tools)
    q: asyncio.Queue = asyncio.Queue()
    seed = [dict(m) for m in messages]

    async def infer(msgs: list[dict[str, Any]], specs: list) -> str:
        state.held = ""
        state.confirmed = False
        state.round_dropped = ""
        state.round_had_calls = False
        prepared = _prepare_messages(msgs, render_tool_contract(specs))
        parts: list[str] = []
        async for token in stream_fn(prepared):
            if not token:
                continue
            parts.append(token)
            out = _guard(state, token)
            if out:
                await q.put(("tok", out))
        return "".join(parts)

    async def drive() -> None:
        try:
            async for ev in AgentLoop(max_iters=max_iters).run(
                _task_text(seed), infer, tap, messages=seed
            ):
                if ev.kind in ("tool_call", "tool_result"):
                    state.round_had_calls = True
                    await q.put(("evt", {"kind": ev.kind, "data": ev.payload}))
                elif ev.kind == "final":
                    await q.put(("end", None))
                    return
            await q.put(("end", None))
        except BaseException as e:  # surfaced by the consumer, not swallowed
            await q.put(("fatal", e))

    driver = asyncio.create_task(drive())
    try:
        while True:
            kind, val = await q.get()
            if kind == "end":
                break
            if kind == "fatal":
                raise val
            if kind == "evt":
                yield val
            else:
                yield val
        # Terminal flush: the FINAL round's round had no executed tool calls
        # (decode found none / failed), so whatever the guard held or dropped
        # there is plain answer text — flush it. Earlier rounds' fences were
        # consumed by tools and stay hidden.
        if not state.round_had_calls:
            pending = state.round_dropped + state.held
            if pending:
                yield pending
        for att in tap.attachments:
            alt = att.get("alt") or "generated image"
            yield f"\n\n![{alt}](data:{att['mime']};base64,{att['b64']})"
    finally:
        driver.cancel()
        try:
            await driver
        except (asyncio.CancelledError, Exception):
            pass
