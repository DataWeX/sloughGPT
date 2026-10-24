"""Canonical agent loop — ToolSpec-driven, event-emitting, headless.

One loop to replace the four copies (domain ``Agent`` one-shot plan, shell
``AgentLoop``, ``ProviderRouter`` marker rounds, orchestrator single tool
attempt). Sync core: no I/O of its own — every external touch is a caller-
supplied callable (``infer``) or the injected :class:`ToolRegistry`. The only
``async`` is the tool seam, because ``ToolSpec.execute`` is a coroutine by
contract; consumers bridge with ``asyncio.run`` (CLI/REPL) or iterate natively
(SSE adapter).

Usage:
    async for event in AgentLoop().run(task, infer=my_infer, tools=registry):
        ...

Events: ``plan`` (once, start of run), ``tool_call``, ``tool_result``,
``final`` (terminal — includes the truncated-on-max-iters case), ``error``
(decode/validation failures; the run still terminates with ``final``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from domain.agents._internal.protocol import ProtocolError, ToolCall, decode
from domain.agents._internal.tools import ToolRegistry, ToolSpec

InferFn = Callable[[list[dict[str, Any]], list[ToolSpec]], Awaitable[str]]
ApproveFn = Callable[[ToolCall], bool]

#: param.type -> acceptable python types for the basic pre-exec check
_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "number": (int, float),
    "integer": (int,),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


@dataclass(frozen=True)
class Event:
    kind: str  # plan | tool_call | tool_result | final | error
    payload: dict[str, Any] = field(default_factory=dict)


def validate_args(spec: ToolSpec, args: dict[str, Any]) -> list[str]:
    """Validate ``args`` against the spec's stored ``parameters`` contract.

    Returns a list of human-readable problems (empty = valid). Mirrors the
    HTTP-route validation that ``create_router`` derives from the same
    descriptor — but at the loop seam, where it was missing.
    """
    problems: list[str] = []
    declared = {p.name: p for p in spec.parameters}
    for p in spec.parameters:
        if p.required and p.name not in args:
            problems.append(f"missing required parameter '{p.name}'")
    for key, value in args.items():
        p = declared.get(key)
        if p is None:
            problems.append(f"unknown parameter '{key}'")
            continue
        expected = _TYPE_MAP.get(p.type)
        if expected and not isinstance(value, expected):
            problems.append(f"parameter '{key}' must be {p.type}")
    return problems


class AgentLoop:
    """Drive a model + tool registry to a final answer, yielding events."""

    def __init__(self, max_iters: int = 10) -> None:
        if max_iters < 1:
            raise ValueError("max_iters must be >= 1")
        self.max_iters = max_iters

    async def run(
        self,
        task: str,
        infer: InferFn,
        tools: ToolRegistry,
        *,
        system_prompt: str | None = None,
        approve: ApproveFn | None = None,
    ) -> AsyncIterator[Event]:
        specs = [tools.get(t["name"]) for t in tools.list_tools()]
        specs = [s for s in specs if s is not None]
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": task})
        yield Event(
            "plan", {"task": task, "tools": [s.name for s in specs], "max_iters": self.max_iters}
        )

        last_text = ""
        for iteration in range(1, self.max_iters + 1):
            text = await infer(messages, specs)
            last_text = text
            try:
                calls, residual = decode(text)
            except ProtocolError as e:
                yield Event("error", {"stage": "decode", "iteration": iteration, "message": str(e)})
                yield Event("final", {"text": text, "iteration": iteration, "truncated": False})
                return

            if not calls:
                yield Event(
                    "final", {"text": residual or text, "iteration": iteration, "truncated": False}
                )
                return

            messages.append({"role": "assistant", "content": text})
            for call in calls:
                yield Event(
                    "tool_call", {"name": call.name, "args": call.args, "iteration": iteration}
                )
                result_text = await self._run_call(call, tools, approve)
                yield Event(
                    "tool_result",
                    {
                        "name": call.name,
                        "output": result_text["output"],
                        "error": result_text["error"],
                        "success": result_text["error"] is None,
                        "iteration": iteration,
                    },
                )
                messages.append(
                    {
                        "role": "tool",
                        "name": call.name,
                        "content": result_text["output"] or result_text["error"] or "",
                    }
                )

        yield Event("final", {"text": last_text, "iteration": self.max_iters, "truncated": True})

    async def _run_call(
        self, call: ToolCall, tools: ToolRegistry, approve: ApproveFn | None
    ) -> dict[str, Any]:
        spec = tools.get(call.name)
        if spec is None:
            return {"output": "", "error": f"Unknown tool: {call.name}"}
        problems = validate_args(spec, call.args)
        if problems:
            return {"output": "", "error": f"invalid args for '{call.name}': {'; '.join(problems)}"}
        if spec.requires_approval:
            if approve is None:
                return {
                    "output": "",
                    "error": f"tool '{call.name}' requires approval (no approver supplied)",
                }
            if not approve(call):
                return {"output": "", "error": f"approval denied for tool '{call.name}'"}
        result = await tools.execute(call.name, call.args)
        return {"output": result.output, "error": result.error}
