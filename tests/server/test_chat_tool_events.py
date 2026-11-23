"""Chat Phase 2 PR-C: tool events flow engine → SSE (card 20261116_090332).

``stream_turn`` wraps ``agentic_stream`` (the AgentLoop adapter landed in
PR-B): the model routes intent by emitting the fenced ```tool_call protocol,
the loop executes tools, and the engine pushes the loop's
tool_call/tool_result events out-of-band as ``TurnEvent`` BETWEEN tokens.
POST /chat projects those events as ``phase: TOOL`` SSE frames — the exact
contract ``apps/web/lib/stream-chat-response.ts`` already reads (status
working → executing, complete → success, error → error; ``data.tool``
required).

Invariants pinned here:

1. **Wire format never reaches the user** — the raw ```tool_call fence is
   consumed by the loop; the token stream carries only prose, and exactly
   ONE final ``TurnResult`` closes the stream (events ride between tokens,
   they are not tokens).
2. **Contract injected every turn, no wire flag** — round 1's system message
   teaches the protocol; round 2 sees the re-attributed ``[tool result: …]``
   message the adapter injects.
3. **Attachment split** — the kernel-appended image markdown rides the token
   stream and ``TurnResult.text`` (so cache/history reload keeps the image),
   while ``remember_facts_async`` gets the clean pre-append text and
   ``_run_post_gen_tasks`` the augmented one.
4. **POST /chat projection** — ``TurnEvent`` → TOOL frames with the
   frontend's exact data shape; STREAMING frames and the done close are
   byte-unchanged (fence-less turns stay identical — covered by
   test_inference_router's TestChat).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.exception_handlers import register_all_handlers

from apps.api.server.routers.chat import router as _chat_router
from apps.api.server.routers.inference import _instance as _inference_router
from domain.agents import ToolParam, ToolSpec
from domain.agents._internal.tools import ToolRegistry
from domain.inference.engine import TurnEvent, TurnRequest, TurnResult

# Readiness-gate stand-ins for stream_turn's GATES block (same doubles
# test_inference_router / test_consciousness_pipeline use): _model_ready
# reads ``state`` via function-local import, so a sys.modules patch flips it
# without loading any weights.
MOCK_STATE = MagicMock()
MOCK_STATE.model = MagicMock()
MOCK_STATE.model_type = "gpt2"
MOCK_STATE.tokenizer = None
MOCK_STATE.model_request_logger = None
MOCK_STATE.checkpoint = None
MOCK_STATE.soul_engine = None
MOCK_STATE.current_soul = None

MOCK_STARTUP = {"phase": "ready", "step": 8, "total": 8, "message": "Ready"}

# The canonical single-call fence, byte-identical to encode_tool_block's
# output — what the model (or a fake) emits to invoke a tool.
FENCE_ECHO = '```tool_call\n{"name": "echo", "args": {"text": "hi"}}\n```'
FENCE_DRAW = '```tool_call\n{"name": "draw", "args": {"prompt": "sky"}}\n```'


@pytest.fixture
def app():
    _app = FastAPI()
    register_all_handlers(_app)
    # Same mounted surface as the deployed app: inference kernel + chat
    # router (POST /chat owner) — mirrors test_inference_router's fixture.
    _app.include_router(_inference_router.router)
    _app.include_router(_chat_router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def _fake_provider(rounds: list[list[str]]):
    """Provider double: one ``chat_stream`` call per loop round.

    Round detection mirrors ``_prepare_messages``' re-attribution: round 2+
    sees the ``[tool result: …]`` user message the adapter injects, round 1
    does not.
    """
    calls: list[list[dict]] = []

    async def chat_stream(messages, **kw):
        calls.append([dict(m) for m in messages])
        followup = any((m.get("content") or "").startswith("[tool result:") for m in messages)
        for tok in rounds[min(1 if followup else 0, len(rounds) - 1)]:
            yield tok

    provider = MagicMock()
    provider.chat_stream = chat_stream
    return provider, calls


def _echo_registry() -> ToolRegistry:
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


def _draw_registry() -> ToolRegistry:
    """A tool whose result carries an attachment (the image-gen shape)."""

    async def _draw(prompt: str) -> dict:
        return {
            "output": "Image attached to the reply.",
            "attachment": {
                "type": "image",
                "mime": "image/png",
                "alt": "a blue sky",
                "b64": "QUJD",
            },
        }

    reg = ToolRegistry(register_defaults=False)
    reg.register(
        ToolSpec(
            name="draw",
            description="Draw an image",
            parameters=[ToolParam("prompt", "string", "what to draw", required=True)],
            execute=_draw,
        )
    )
    return reg


@patch.dict(
    "sys.modules",
    {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
)
async def test_stream_turn_yields_tool_events_and_hides_fence():
    """The agentic path: events between tokens, fence consumed, one close."""
    provider, calls = _fake_provider([["Let me echo ", FENCE_ECHO], ["Done."]])

    with (
        patch("apps.api.server.routers.inference.get_provider", return_value=provider),
        patch(
            "apps.api.server.routers.inference.get_tool_registry",
            return_value=_echo_registry(),
        ),
    ):
        req = TurnRequest(
            messages=[{"role": "user", "content": "echo hi"}],
            session_id="sess-agentic-wire",
        )
        items = [i async for i in _inference_router.stream_turn(req)]

    finals = [i for i in items if isinstance(i, TurnResult)]
    events = [i for i in items if isinstance(i, TurnEvent)]
    tokens = [i for i in items if isinstance(i, str)]

    assert len(finals) == 1, "exactly one TurnResult must close the stream"
    assert items[-1] is finals[0], "TurnResult must be the final yield"
    assert [e.kind for e in events] == ["tool_call", "tool_result"], (
        "loop events must ride out-of-band as TurnEvent, in loop order"
    )
    call, result = events
    assert call.data["name"] == "echo"
    assert call.data["args"] == {"text": "hi"}
    assert result.data["name"] == "echo"
    assert result.data["output"] == "HI"
    assert result.data["success"] is True

    text = "".join(tokens)
    assert "```tool_call" not in text, "raw wire format must never reach the token stream"
    assert text == "Let me echo Done.", "fence-less tokens must pass through byte-identical"
    assert finals[0].text == "Let me echo Done."

    # Contract injected EVERY turn (no wire flag): round 1's system message
    # teaches the protocol, and the loop runs exactly two provider rounds —
    # round 2 sees the re-attributed tool result.
    assert len(calls) == 2, "loop must run one round before + one after the tool call"
    assert any("Available tools:" in (m.get("content") or "") for m in calls[0]), (
        "tool contract must be injected into round 1's system message"
    )
    assert any((m.get("content") or "").startswith("[tool result: echo]") for m in calls[1]), (
        "round 2 must carry the re-attributed tool result"
    )


@patch.dict(
    "sys.modules",
    {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
)
async def test_stream_turn_attachment_markdown_rides_stream_but_not_memory():
    """Attachment split: augmented for stream/capture, clean for memory."""
    provider, _ = _fake_provider([["Drawing now ", FENCE_DRAW], ["Here you go."]])

    mem_calls: list[str] = []
    postgen_texts: list[str] = []

    async def _remember(user_msg, text, *a, **kw):
        mem_calls.append(text)
        return []

    def _post_gen(text, *a, **kw):
        postgen_texts.append(text)
        return {}

    memory_service = SimpleNamespace(remember_facts_async=_remember)

    with (
        patch("apps.api.server.routers.inference.get_provider", return_value=provider),
        patch(
            "apps.api.server.routers.inference.get_tool_registry",
            return_value=_draw_registry(),
        ),
        patch(
            "apps.api.server.routers.inference.get_memory_service",
            return_value=memory_service,
        ),
        patch(
            "apps.api.server.routers.inference._run_post_gen_tasks",
            side_effect=_post_gen,
        ),
    ):
        req = TurnRequest(
            messages=[{"role": "user", "content": "draw a sky"}],
            session_id="sess-agentic-att",
        )
        items = [i async for i in _inference_router.stream_turn(req)]

    md = "![a blue sky](data:image/png;base64,QUJD)"
    streamed = "".join(i for i in items if isinstance(i, str))
    finals = [i for i in items if isinstance(i, TurnResult)]
    assert len(finals) == 1

    assert md in streamed, (
        "attachment markdown must ride the token stream — the chat cache and "
        "history are built from these tokens, so a reload keeps the image"
    )
    assert finals[0].text == "Drawing now Here you go.\n\n" + md, (
        "TurnResult.text is what history stores — it must carry the image"
    )

    assert mem_calls, "memory step must run for this turn"
    assert all("base64" not in t for t in mem_calls), (
        "remember_facts_async must get the clean pre-append text — never the base64 blob"
    )
    assert all("Here you go." in t for t in mem_calls)
    assert postgen_texts, "_run_post_gen_tasks must run for this turn"
    assert md in postgen_texts[0], "post-gen (capture) gets the augmented text"


def test_post_chat_projects_turn_events_as_tool_frames(client):
    """POST /chat projects engine TurnEvents as the frontend's TOOL frames."""
    turn = TurnResult(
        text="the answer",
        session_id="sess-tool-proj",
        done=True,
        model="qwen",
        usage={"completion_tokens": 5, "total_tokens": 5},
        elapsed_ms=12.0,
        finish_reason="stop",
    )
    items = [
        "Thinking ",
        TurnEvent(
            kind="tool_call", data={"name": "image_gen", "args": {"prompt": "cat"}, "iteration": 1}
        ),
        TurnEvent(
            kind="tool_result",
            data={
                "name": "image_gen",
                "output": "ok",
                "error": None,
                "success": True,
                "iteration": 1,
            },
        ),
        TurnEvent(
            kind="tool_result",
            data={
                "name": "file_qa",
                "output": "",
                "error": "boom",
                "success": False,
                "iteration": 1,
            },
        ),
        turn,
    ]

    async def _stream(req):
        for item in items:
            yield item

    fake_engine = SimpleNamespace(stream=_stream)

    with patch("apps.api.server.routers.chat.get_chat_engine", return_value=fake_engine):
        resp = client.post(
            "/chat",
            json={
                "session_id": "sess-tool-proj",
                "messages": [{"role": "user", "content": "draw a cat"}],
            },
        )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    frames = [
        json.loads(line[len("data: ") :])
        for line in resp.text.splitlines()
        if line.startswith("data: ")
    ]

    tool_frames = [f for f in frames if f.get("phase") == "TOOL"]
    assert len(tool_frames) == 3, "each TurnEvent projects exactly one TOOL frame"
    call_f, ok_f, err_f = tool_frames

    # tool_call → status working (frontend maps it to 'executing'), data.tool
    # required + args optional — the shape stream-chat-response.ts reads.
    assert call_f["status"] == "working"
    assert call_f["data"] == {"tool": "image_gen", "args": {"prompt": "cat"}}

    # tool_result success → complete ('success' client-side)
    assert ok_f["status"] == "complete"
    assert ok_f["data"] == {"tool": "image_gen", "output": "ok", "error": "", "success": True}

    # tool_result failure → error
    assert err_f["status"] == "error"
    assert err_f["data"] == {"tool": "file_qa", "output": "", "error": "boom", "success": False}

    # Tokens and the close are unchanged: prose rides STREAMING frames, the
    # done frame (empty token, status complete) still ends the stream.
    tok_frames = [f for f in frames if f.get("phase") == "STREAMING"]
    assert any(f["data"].get("token") == "Thinking " for f in tok_frames)
    assert tok_frames[-1]["status"] == "complete"
    assert tok_frames[-1]["data"].get("token") == ""
    assert frames[-1] is tok_frames[-1], "done frame must remain the last frame"


@patch.dict(
    "sys.modules",
    {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
)
async def test_stream_turn_regenerate_skips_counting_side_effects():
    """A re-run must not double-count (card 20261120_150317).

    ``TurnRequest.regenerate=True`` keeps every generation-side stage —
    readiness gates, context/RAG enrichment, the agentic loop, the token
    stream itself — but skips the counting-side work: conversation capture,
    response tracking and metrics (``_run_post_gen_tasks``) and the
    auto-memory ingest. The ORIGINAL turn already recorded those; recording
    them again would count one conversation event twice (audit gap G2's
    rule applied to the re-run).
    """
    provider, _ = _fake_provider([["Plain answer."]])

    postgen_texts: list[str] = []
    mem_calls: list[str] = []

    async def _remember(user_msg, text, *a, **kw):
        mem_calls.append(text)
        return []

    def _post_gen(text, *a, **kw):
        postgen_texts.append(text)
        return {}

    memory_service = SimpleNamespace(remember_facts_async=_remember)

    with (
        patch("apps.api.server.routers.inference.get_provider", return_value=provider),
        patch(
            "apps.api.server.routers.inference.get_memory_service",
            return_value=memory_service,
        ),
        patch(
            "apps.api.server.routers.inference._run_post_gen_tasks",
            side_effect=_post_gen,
        ) as mock_pg,
    ):
        rerun = TurnRequest(
            messages=[{"role": "user", "content": "hi"}],
            session_id="sess-regen-rerun",
            regenerate=True,
        )
        items = [i async for i in _inference_router.stream_turn(rerun)]

    finals = [i for i in items if isinstance(i, TurnResult)]
    assert len(finals) == 1, "a re-run still streams a complete turn"
    assert "".join(i for i in items if isinstance(i, str)) == "Plain answer."
    assert finals[0].memory is None, "a re-run must not surface a MEMORY sideband"
    assert finals[0].consciousness is None, "no reflection sideband on a re-run"
    assert mock_pg.call_count == 0, "a re-run must not re-run post-gen (capture/tracker/metrics)"
    assert mem_calls == [], "a re-run must not re-ingest auto-memory"

    # Contrast in the same harness: a fresh turn still counts — the guard
    # is the flag, not a broken default path.
    with (
        patch("apps.api.server.routers.inference.get_provider", return_value=provider),
        patch(
            "apps.api.server.routers.inference.get_memory_service",
            return_value=memory_service,
        ),
        patch(
            "apps.api.server.routers.inference._run_post_gen_tasks",
            side_effect=_post_gen,
        ) as mock_pg2,
    ):
        fresh = TurnRequest(
            messages=[{"role": "user", "content": "hi"}],
            session_id="sess-regen-fresh",
        )
        [i async for i in _inference_router.stream_turn(fresh)]

    assert mock_pg2.call_count == 1, "a fresh turn runs post-gen exactly once"
    assert mem_calls, "a fresh turn still runs auto-memory"
