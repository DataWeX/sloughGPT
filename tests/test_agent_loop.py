"""Tests for the canonical agent tool-call protocol and AgentLoop (card 3344013d)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from domain.agents._internal.loop import AgentLoop, Event, validate_args
from domain.agents._internal.protocol import ProtocolError, ToolCall, decode, encode_tool_block
from domain.agents._internal.tools import ToolParam, ToolRegistry, ToolSpec

# ── protocol ─────────────────────────────────────────────────────────


def test_decode_canonical_single_call():
    text = 'I will calc.\n```tool_call\n{"name": "calculator", "args": {"expression": "2+2"}}\n```'
    calls, residual = decode(text)
    assert calls == [ToolCall(name="calculator", args={"expression": "2+2"})]
    assert "I will calc." in residual
    assert "tool_call" not in residual


def test_decode_canonical_multi_call():
    payload = {
        "tool_calls": [
            {"name": "calculator", "args": {"expression": "1+1"}},
            {"name": "current_time", "args": {}},
        ]
    }
    text = "```tool_call\n" + json.dumps(payload) + "\n```"
    calls, residual = decode(text)
    assert [c.name for c in calls] == ["calculator", "current_time"]
    assert residual == ""


def test_decode_canonical_accepts_tool_and_arguments_aliases():
    text = '```tool_call\n{"tool": "web_search", "arguments": {"query": "two words"}}\n```'
    calls, _ = decode(text)
    assert calls == [ToolCall(name="web_search", args={"query": "two words"})]


def test_decode_invalid_json_raises_protocol_error():
    text = "```tool_call\n{not json}\n```"
    with pytest.raises(ProtocolError):
        decode(text)


def test_decode_missing_name_raises():
    text = '```tool_call\n{"args": {}}\n```'
    with pytest.raises(ProtocolError):
        decode(text)


def test_decode_non_object_args_raises():
    text = '```tool_call\n{"name": "x", "args": "oops"}\n```'
    with pytest.raises(ProtocolError):
        decode(text)


def test_decode_legacy_marker_plain_token_wraps_raw():
    calls, residual = decode("run [[TOOL: file_read]] /tmp/a.txt please")
    assert calls == [ToolCall(name="file_read", args={"_raw": "/tmp/a.txt"})]
    assert "TOOL" not in residual
    assert "run" in residual and "please" in residual


def test_decode_legacy_marker_json_dict_args():
    calls, _ = decode('[[TOOL: web_search]] {"query": "two words"}')
    assert calls == [ToolCall(name="web_search", args={"query": "two words"})]


def test_decode_both_codecs_in_one_text():
    text = (
        '```tool_call\n{"name": "calculator", "args": {"expression": "2+2"}}\n```\n'
        "then [[TOOL: current_time]] now"
    )
    calls, residual = decode(text)
    assert [c.name for c in calls] == ["calculator", "current_time"]
    assert "then" in residual


def test_encode_roundtrip_single_and_multi():
    single = [ToolCall(name="a", args={"x": 1})]
    calls, _ = decode(encode_tool_block(single))
    assert calls == single
    multi = [ToolCall(name="a", args={}), ToolCall(name="b", args={"y": 2})]
    calls, _ = decode(encode_tool_block(multi))
    assert calls == multi


# ── loop helpers ─────────────────────────────────────────────────────


def _scripted_infer(script: list[str], sink: list[Any] | None = None):
    """Fake infer: pops the next scripted model response; records messages."""
    state = {"i": 0}

    async def infer(messages, specs):
        if sink is not None:
            sink.append([dict(m) for m in messages])
        if state["i"] >= len(script):
            raise AssertionError("infer called more times than scripted")
        resp = script[state["i"]]
        state["i"] += 1
        return resp

    return infer


def _kinds(events: list[Event]) -> list[str]:
    return [e.kind for e in events]


async def _run(script, *, tools=None, max_iters=10, approve=None, system_prompt=None):
    registry = tools or ToolRegistry()
    events = []
    loop = AgentLoop(max_iters=max_iters)
    async for ev in loop.run(
        "do the task",
        infer=_scripted_infer(script),
        tools=registry,
        approve=approve,
        system_prompt=system_prompt,
    ):
        events.append(ev)
    return events


# ── loop ─────────────────────────────────────────────────────────────


async def test_loop_final_only_path():
    events = await _run(["just answering, no tools"])
    assert _kinds(events) == ["plan", "final"]
    assert events[-1].payload["text"].startswith("just answering")
    assert events[-1].payload["truncated"] is False


async def test_loop_happy_multi_turn_tool_use():
    answer = "The answer is 4"
    script = [
        encode_tool_block([ToolCall(name="calculator", args={"expression": "2+2"})]),
        answer,
    ]
    events = await _run(script)
    assert _kinds(events) == ["plan", "tool_call", "tool_result", "final"]
    call = events[1]
    assert call.payload == {"name": "calculator", "args": {"expression": "2+2"}, "iteration": 1}
    result = events[2]
    assert result.payload["success"] is True
    assert result.payload["output"].strip() == "4"
    assert events[-1].payload["text"] == answer


async def test_loop_feeds_tool_result_back_into_messages():
    script = [
        encode_tool_block([ToolCall(name="calculator", args={"expression": "2+2"})]),
        "done",
    ]
    # second infer call must have seen the assistant tool block + tool result
    sink: list[Any] = []
    registry = ToolRegistry()
    loop = AgentLoop()
    state = {"i": 0}
    script2 = script

    async def infer(messages, specs):
        sink.append([dict(m) for m in messages])
        r = script2[state["i"]]
        state["i"] += 1
        return r

    async for _ in loop.run("do the task", infer=infer, tools=registry):
        pass
    assert len(sink) == 2
    roles_second = [m["role"] for m in sink[1]]
    assert roles_second == ["user", "assistant", "tool"]
    assert sink[1][-1]["content"].strip() == "4"


async def test_loop_unknown_tool_yields_error_result_then_model_recovers():
    script = [
        encode_tool_block([ToolCall(name="no_such_tool", args={})]),
        "recovered",
    ]
    events = await _run(script)
    assert _kinds(events) == ["plan", "tool_call", "tool_result", "final"]
    result = events[2]
    assert result.payload["success"] is False
    assert "Unknown tool" in result.payload["error"]


async def test_loop_missing_required_arg_is_rejected_before_execute():
    script = [
        encode_tool_block([ToolCall(name="calculator", args={})]),
        "ok",
    ]
    events = await _run(script)
    result = events[2]
    assert result.payload["success"] is False
    assert "missing required parameter 'expression'" in result.payload["error"]


async def test_loop_unknown_and_mistyped_args_rejected():
    registry = ToolRegistry()

    async def _echo(text: str, n: int = 1) -> dict[str, Any]:
        return {"output": text * n}

    registry.register(
        ToolSpec(
            name="echo",
            description="echo",
            parameters=[
                ToolParam("text", "string", "text", required=True),
                ToolParam("n", "integer", "count", required=False),
            ],
            execute=_echo,
        )
    )
    script = [
        encode_tool_block([ToolCall(name="echo", args={"text": "hi", "nope": 1, "n": "two"})]),
        "ok",
    ]
    events = await _run(script, tools=registry)
    err = events[2].payload["error"]
    assert "unknown parameter 'nope'" in err
    assert "parameter 'n' must be integer" in err


async def test_loop_max_iters_truncates():
    forever = encode_tool_block([ToolCall(name="calculator", args={"expression": "2+2"})])
    events = await _run([forever, forever, forever], max_iters=2)
    assert _kinds(events) == [
        "plan",
        "tool_call",
        "tool_result",
        "tool_call",
        "tool_result",
        "final",
    ]
    assert events[-1].payload["truncated"] is True


async def test_loop_decode_error_yields_error_then_final():
    events = await _run(["```tool_call\n{broken\n```"])
    assert _kinds(events) == ["plan", "error", "final"]
    assert events[1].payload["stage"] == "decode"
    assert events[-1].payload["truncated"] is False


async def test_loop_requires_approval_without_approver_fails_closed():
    registry = ToolRegistry()

    async def _danger() -> dict[str, Any]:
        return {"output": "boom"}

    registry.register(
        ToolSpec(
            name="danger",
            description="danger",
            parameters=[],
            execute=_danger,
            requires_approval=True,
        )
    )
    script = [encode_tool_block([ToolCall(name="danger", args={})]), "ok"]
    events = await _run(script, tools=registry)
    assert events[2].payload["success"] is False
    assert "requires approval" in events[2].payload["error"]


async def test_loop_approval_gate_allow_and_deny():
    registry = ToolRegistry()
    ran = {"n": 0}

    async def _danger() -> dict[str, Any]:
        ran["n"] += 1
        return {"output": "boom"}

    registry.register(
        ToolSpec(
            name="danger",
            description="danger",
            parameters=[],
            execute=_danger,
            requires_approval=True,
        )
    )
    call = ToolCall(name="danger", args={})
    block = encode_tool_block([call])

    # deny
    events = await _run([block, "ok"], tools=registry, approve=lambda c: False)
    assert events[2].payload["success"] is False
    assert "denied" in events[2].payload["error"]
    assert ran["n"] == 0

    # allow
    events = await _run([block, "ok"], tools=registry, approve=lambda c: c.name == "danger")
    assert events[2].payload["success"] is True
    assert ran["n"] == 1


async def test_loop_system_prompt_precedes_task():
    sink: list[Any] = []
    registry = ToolRegistry()
    loop = AgentLoop()
    state = {"i": 0}
    script = ["final answer"]

    async def infer(messages, specs):
        sink.append([dict(m) for m in messages])
        r = script[state["i"]]
        state["i"] += 1
        return r

    async for _ in loop.run("task", infer=infer, tools=registry, system_prompt="be brief"):
        pass
    assert sink[0][0] == {"role": "system", "content": "be brief"}
    assert sink[0][1] == {"role": "user", "content": "task"}


def test_validate_args_type_map_and_required():
    async def _noop(**kwargs):
        return {}

    spec = ToolSpec(
        name="t",
        description="t",
        parameters=[
            ToolParam("a", "string", "a", required=True),
            ToolParam("b", "number", "b", required=False),
            ToolParam("c", "array", "c", required=False),
        ],
        execute=_noop,
    )
    assert validate_args(spec, {"a": "x", "b": 1.5, "c": [1]}) == []
    problems = validate_args(spec, {"b": "x", "c": {}, "z": 1})
    assert "missing required parameter 'a'" in problems
    assert "parameter 'b' must be number" in problems
    assert "parameter 'c' must be array" in problems
    assert "unknown parameter 'z'" in problems


def test_agent_loop_rejects_bad_max_iters():
    with pytest.raises(ValueError):
        AgentLoop(max_iters=0)
