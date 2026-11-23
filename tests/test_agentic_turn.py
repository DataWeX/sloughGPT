"""Agentic chat turn: AgentLoop behind the engine port (card 20261116_090332).

``agentic_stream`` is the domain seam stream_turn wraps: it drives the
canonical AgentLoop while forwarding round tokens live, HOLDING any tail that
is forming a ```tool_call fence (the model's raw wire format must never reach
the user), and flushing held/dropped content only on degradation so a broken
fence still yields today's plain answer.

Also covers the two port-side pieces it needs: ``AgentLoop.run(messages=)``
(full chat history instead of task-only) and ``TurnEvent`` (the out-of-band
event type edges project as SSE TOOL frames; ``fold`` ignores it safely).
"""

from __future__ import annotations

import pytest

from domain.agents._internal.agentic import agentic_stream
from domain.agents._internal.tools import ToolParam, ToolRegistry, ToolSpec
from domain.inference.engine import TurnEvent, TurnResult, fold


def _registry_with_echo() -> ToolRegistry:
    async def _echo(text: str) -> dict:
        return {"output": text.upper()}

    reg = ToolRegistry(register_defaults=False)
    reg.register(
        ToolSpec(
            name="echo",
            description="Echo text back uppercased",
            parameters=[ToolParam("text", "string", "text to echo", required=True)],
            execute=_echo,
        )
    )
    return reg


def _registry_with_image_gen() -> ToolRegistry:
    async def _gen(prompt: str, style: str = "default") -> dict:
        return {"output": f"generated: {prompt} ({style})"}

    reg = ToolRegistry(register_defaults=False)
    reg.register(
        ToolSpec(
            name="image_gen",
            description="Generate an image from a prompt",
            parameters=[
                ToolParam("prompt", "string", "image description", required=True),
                ToolParam("style", "string", "art style", required=False),
            ],
            execute=_gen,
        )
    )
    return reg


class _Recorder:
    """Fake provider round-stream: scripted rounds, captures prepared msgs."""

    def __init__(self, rounds: list[list[str]]):
        self.rounds = rounds
        self.calls: list[list[dict]] = []

    def __call__(self, messages):
        self.calls.append([dict(m) for m in messages])
        idx = len(self.calls) - 1
        tokens = self.rounds[min(idx, len(self.rounds) - 1)]

        async def _gen():
            for t in tokens:
                yield t

        return _gen()


async def _collect(stream):
    out = []
    async for item in stream:
        out.append(item)
    return out


def _text(items) -> str:
    return "".join(i for i in items if isinstance(i, str))


def _events(items) -> list[dict]:
    return [i for i in items if isinstance(i, dict)]


class TestPlainTurn:
    async def test_no_fence_tokens_forwarded_verbatim(self) -> None:
        rec = _Recorder([["Hello", ", ", "world!"]])
        items = await _collect(
            agentic_stream(
                [{"role": "user", "content": "hi"}],
                tools=ToolRegistry(register_defaults=False),
                stream_fn=rec,
            )
        )
        assert _text(items) == "Hello, world!"
        assert _events(items) == []
        assert len(rec.calls) == 1  # single round — same as today's plain path

    async def test_tool_contract_injected_into_system_message(self) -> None:
        rec = _Recorder([["ok"]])
        await _collect(
            agentic_stream(
                [{"role": "user", "content": "hi"}],
                tools=_registry_with_echo(),
                stream_fn=rec,
            )
        )
        system = rec.calls[0][0]
        assert system["role"] == "system"
        assert "tool_call" in system["content"]
        assert "- echo:" in system["content"]


class TestFenceTurn:
    async def test_fence_held_tool_runs_events_flow_answer_streams(self) -> None:
        fence = "```tool_call\n" + '{"name": "echo", "args": {"text": "hi"}}' + "\n```"
        rec = _Recorder(
            [
                ["Sure, ", "doing it: ", fence],
                ["All ", "done!"],
            ]
        )
        items = await _collect(
            agentic_stream(
                [{"role": "user", "content": "echo hi"}],
                tools=_registry_with_echo(),
                stream_fn=rec,
            )
        )

        text = _text(items)
        # raw fence never reaches the user; preamble + round-2 answer do
        assert "```" not in text
        assert text == "Sure, doing it: All done!"

        evs = _events(items)
        kinds = [e["kind"] for e in evs]
        assert kinds == ["tool_call", "tool_result"]
        assert evs[0]["data"]["name"] == "echo"
        assert evs[1]["data"]["output"] == "HI"
        assert evs[1]["data"]["success"] is True

        # round 2 saw the tool result, re-attributed for the provider
        round2 = rec.calls[1]
        toolish = [m for m in round2 if "HI" in str(m.get("content"))]
        assert toolish, "round-2 messages must carry the tool result"

    async def test_second_round_still_gets_contract(self) -> None:
        rec = _Recorder(
            [
                ["```tool_call\n" + '{"name": "echo", "args": {"text": "x"}}' + "\n```"],
                ["answer"],
            ]
        )
        await _collect(
            agentic_stream(
                [{"role": "user", "content": "go"}],
                tools=_registry_with_echo(),
                stream_fn=rec,
            )
        )
        for prepared in rec.calls:
            assert "tool_call" in prepared[0]["content"]


class TestCodeBlockNotAte:
    async def test_code_fence_flushes_and_streams_normally(self) -> None:
        tokens = ["Here: ", "```py", "thon\nprint(1)", "\n```", " done"]
        rec = _Recorder([tokens])
        items = await _collect(
            agentic_stream(
                [{"role": "user", "content": "code"}],
                tools=ToolRegistry(register_defaults=False),
                stream_fn=rec,
            )
        )
        assert _text(items) == "".join(tokens)
        assert _events(items) == []


class TestJsonTaggedFence:
    """The model tags its tool block ``json`` as often as ``tool_call``
    (live-observed, card 20261116_090332): the guard must hold it like the
    canonical fence so decode gets a chance — and the raw block must never
    reach the user when it turns out to be a call."""

    async def test_json_fence_executes_and_stays_hidden(self) -> None:
        fence = '```json\n{"name": "echo", "args": {"text": "hi"}}\n```'
        rec = _Recorder([["Sure, ", fence], ["All done!"]])
        items = await _collect(
            agentic_stream(
                [{"role": "user", "content": "echo hi"}],
                tools=_registry_with_echo(),
                stream_fn=rec,
            )
        )
        text = _text(items)
        assert "```" not in text, "json-tagged wire format must not reach the user"
        assert text == "Sure, All done!"
        kinds = [e["kind"] for e in _events(items)]
        assert kinds == ["tool_call", "tool_result"]
        assert _events(items)[0]["data"]["name"] == "echo"

    async def test_non_tool_json_block_flushed_as_prose(self) -> None:
        # A plain JSON snippet is code: held like any fence, then flushed
        # whole at round end because no call was decoded (degradation —
        # the user still sees the block, as text).
        block = '```json\n{"answer": 42}\n```'
        rec = _Recorder([["Result: ", block]])
        items = await _collect(
            agentic_stream(
                [{"role": "user", "content": "show json"}],
                tools=ToolRegistry(register_defaults=False),
                stream_fn=rec,
            )
        )
        assert _text(items) == "Result: " + block
        assert _events(items) == []


class TestUnclosedFence:
    """The model hits finish_reason=stop before the closing backticks
    (live walk, card 20261116_090332: `` ```tool_call\\n{...}`` with no
    close). The block must still execute — and the raw wire format must
    never reach the user."""

    async def test_unclosed_fence_executes_and_stays_hidden(self) -> None:
        # Exact live shape: no closing fence, flat query sibling.
        fence = '```tool_call\n{"name": "echo", "args": {"text": "hi"}}'
        rec = _Recorder([["Sure, ", fence], ["All done!"]])
        items = await _collect(
            agentic_stream(
                [{"role": "user", "content": "echo hi"}],
                tools=_registry_with_echo(),
                stream_fn=rec,
            )
        )
        text = _text(items)
        assert "```" not in text, "the unclosed wire block must not reach the user"
        assert text == "Sure, All done!"
        kinds = [e["kind"] for e in _events(items)]
        assert kinds == ["tool_call", "tool_result"]
        assert _events(items)[0]["data"]["name"] == "echo"

    async def test_missing_outer_brace_executes_and_stays_hidden(self) -> None:
        # walk_img7 exact bytes: closed fence, payload missing the
        # outer "}" — must repair, execute, and never reach the user.
        fence = '```tool_call\n{"tool_calls": [{"name": "echo", "query": "hi"}]\n```'
        rec = _Recorder([["Sure, ", fence], ["All done!"]])
        items = await _collect(
            agentic_stream(
                [{"role": "user", "content": "echo hi"}],
                tools=_registry_with_echo(),
                stream_fn=rec,
            )
        )
        text = _text(items)
        assert "```" not in text, "the near-miss wire block must not reach the user"
        assert text == "Sure, All done!"
        evs = _events(items)
        assert [e["kind"] for e in evs] == ["tool_call", "tool_result"]
        assert evs[0]["data"]["name"] == "echo"


class TestToolNamedFence:
    """The model invents its own fence tag naming a registered tool
    (live walk #4, card 20261116_090332: ```image_gen with a key: value
    body and a mangled ```image closer). The guard must HOLD that block —
    a tag naming a registered tool is wire format — and decode must
    execute it; prose tags (python, js) keep streaming through."""

    async def test_tool_named_fence_executes_and_stays_hidden(self) -> None:
        # Exact live shape: tool-named tag, key: value body, mangled closer.
        fence = (
            "```image_gen\n"
            'prompt: "A blue circle"\n'
            'style: "cyberpunk"\n'
            'output: {"name": "image_gen", "args": {"filename": "", "url": ""}}\n'
            "```image"
        )
        rec = _Recorder([["Sure. ", fence], ["Done!"]])
        items = await _collect(
            agentic_stream(
                [{"role": "user", "content": "generate a blue circle"}],
                tools=_registry_with_image_gen(),
                stream_fn=rec,
            )
        )
        text = _text(items)
        assert "```" not in text, "the wire block must not reach the user"
        assert text == "Sure. Done!"
        calls = [e for e in _events(items) if e["kind"] == "tool_call"]
        assert len(calls) == 1
        assert calls[0]["data"]["name"] == "image_gen"
        assert calls[0]["data"]["args"]["prompt"] == "A blue circle"

    async def test_prose_tag_streams_through(self) -> None:
        # A code block is not a call: ```python diverges and streams live
        # (pinned by TestCodeBlockNotAte; this guards the extended header
        # set does not swallow ordinary prose fences).
        rec = _Recorder([["Here:", "```python\nprint(1)\n```", "!"]])
        items = await _collect(
            agentic_stream(
                [{"role": "user", "content": "show python"}],
                tools=_registry_with_image_gen(),
                stream_fn=rec,
            )
        )
        text = _text(items)
        assert "```python" in text
        assert _events(items) == []


class TestDegradation:
    async def test_protocol_error_flushes_dropped_fence_as_plain_answer(self) -> None:
        bad = "```tool_call\n{not valid json}\n```"
        rec = _Recorder([["Sure ", bad]])
        items = await _collect(
            agentic_stream(
                [{"role": "user", "content": "x"}],
                tools=_registry_with_echo(),
                stream_fn=rec,
            )
        )
        # degraded but complete: preamble (forwarded) + dropped tail (flushed)
        assert _text(items) == "Sure " + bad

    async def test_trailing_partial_fence_flushed_at_end(self) -> None:
        rec = _Recorder([["answer ```tool_c"]])
        items = await _collect(
            agentic_stream(
                [{"role": "user", "content": "x"}],
                tools=_registry_with_echo(),
                stream_fn=rec,
            )
        )
        assert _text(items) == "answer ```tool_c"


class TestAttachmentAppend:
    async def test_image_attachment_appended_as_final_markdown(self) -> None:
        async def _gen(prompt: str, style: str = "realistic") -> dict:
            return {
                "output": "Image attached to the reply.",
                "attachment": {
                    "type": "image",
                    "mime": "image/png",
                    "path": "/data/gallery/generated_deadbeef.png",
                    "alt": "a blue sky",
                    "b64": "QUJD",
                },
            }

        reg = ToolRegistry(register_defaults=False)
        reg.register(
            ToolSpec(
                name="image_gen",
                description="Generate an image",
                parameters=[ToolParam("prompt", "string", "what to draw", required=True)],
                execute=_gen,
            )
        )
        rec = _Recorder(
            [
                [
                    "```tool_call\n"
                    + '{"name": "image_gen", "args": {"prompt": "a blue sky"}}'
                    + "\n```"
                ],
                ["Here you go."],
            ]
        )
        items = await _collect(
            agentic_stream(
                [{"role": "user", "content": "draw a blue sky"}],
                tools=reg,
                stream_fn=rec,
            )
        )
        assert _text(items) == "Here you go.\n\n![a blue sky](data:image/png;base64,QUJD)"
        # attachment must NOT leak into any event payload (SSE stays small)
        for ev in _events(items):
            assert "b64" not in str(ev)


class TestLoopMessagesKwarg:
    async def test_run_uses_provided_history_instead_of_task_only(self) -> None:
        seen: list[list[dict]] = []

        async def infer(messages, specs):
            seen.append([dict(m) for m in messages])
            return "done"

        history = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier answer"},
            {"role": "user", "content": "latest question"},
        ]
        from domain.agents._internal.loop import AgentLoop

        async for _ in AgentLoop(max_iters=1).run(
            "latest question", infer, ToolRegistry(register_defaults=False), messages=history
        ):
            pass
        assert seen[0] == history


class TestTurnEvent:
    async def test_fold_ignores_turn_events_and_returns_result(self) -> None:
        async def gen():
            yield "tok"
            yield TurnEvent(kind="tool", data={"name": "image_gen", "status": "executing"})
            yield TurnResult(text="hi", session_id="s1")

        turn = await fold(gen())
        assert turn.text == "hi"
        assert turn.session_id == "s1"

    def test_defaults(self) -> None:
        ev = TurnEvent()
        assert ev.kind == "tool"
        assert ev.data == {}


@pytest.mark.parametrize("kind", ["tool"])
def test_turn_event_kinds_are_plain_strings(kind: str) -> None:
    assert TurnEvent(kind=kind, data={"x": 1}).kind == kind
