"""POST /chat must own chat's state rather than delegate it away.

The delegation to inference's ``handle_chat`` quietly made chat's main endpoint
stateless: inference's ``chat()`` takes ``req.messages`` from the client and
never persists, so no history was written and no op was ever registered for
cancellation — while ``regenerate``/``cancel``/``active_sessions`` kept running
on ``domain/chat``'s stateful kernel. These tests pin the composed behaviour at
the seam: generation in inference, state in chat.

``ChatManager`` declares ``__slots__``, so its collaborators are patched on the
CLASS, never on an instance.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from apps.api.server.routers.chat import ChatRouter
from domain.chat._internal.manager import ChatManager
from domain.generation.engine import TurnResult


@pytest.fixture
def manager():
    """A real ChatManager so the tracking/store control flow actually runs."""
    return ChatManager()


@pytest.fixture
def client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from apps.api.server.routers.chat import router as _chat_router

    app = FastAPI()
    app.include_router(_chat_router)
    return TestClient(app, raise_server_exceptions=True)


def _respond(manager, **kw):
    return asyncio.run(manager.respond(messages=[{"role": "user", "content": "hi"}], **kw))


def _engine_yielding(*items):
    """A fake generation engine whose ``stream`` yields the given items.

    Streaming is the engine's one method, so the fake mirrors that shape: an
    async generator producing raw tokens followed by the final ``TurnResult``.
    """

    async def stream(req):
        for item in items:
            yield item

    return SimpleNamespace(stream=stream)


def test_respond_untracks_on_success(manager):
    """Regression: `else: _untrack` never ran, because `return` skips `else`.

    Every successful reply used to leak its op_id into ``_session_ops``, which
    ``cancel_session`` walks. Untracking must happen on the success path too.
    """
    recorded: list[tuple[str, str | None]] = []

    with (
        patch.object(ChatManager, "_track", return_value=("op-1", None)),
        patch.object(ChatManager, "_store"),
        patch.object(
            ChatManager,
            "_untrack",
            side_effect=lambda sid, op_id, error=None: recorded.append((op_id, error)),
        ),
        # _domain is an instance slot (set in __init__), so it patches there,
        # unlike the _track/_store/_untrack methods which are class attributes.
        patch.object(
            manager,
            "_domain",
            SimpleNamespace(respond=AsyncMock(return_value=SimpleNamespace(text="ok"))),
        ),
    ):
        _respond(manager)

    assert recorded == [("op-1", None)], "success path must untrack, with no error"


def test_respond_untracks_on_failure(manager):
    recorded: list[tuple[str, str | None]] = []

    with (
        patch.object(ChatManager, "_track", return_value=("op-2", None)),
        patch.object(
            ChatManager,
            "_untrack",
            side_effect=lambda sid, op_id, error=None: recorded.append((op_id, error)),
        ),
        patch.object(
            manager,
            "_domain",
            SimpleNamespace(respond=AsyncMock(side_effect=RuntimeError("boom"))),
        ),
    ):
        with pytest.raises(RuntimeError):
            _respond(manager)

    assert recorded == [("op-2", "boom")], "failure path must untrack with the error"


def test_post_chat_persists_history(client, manager):
    """POST /chat must store the exchange, exactly as domain/chat's own path did."""
    fake = TurnResult(text="hi back", session_id="s1")
    stored: list[tuple[str, str]] = []
    fake_engine = _engine_yielding(fake)

    with (
        patch("domain.chat.get_chat_manager", return_value=manager),
        patch("apps.api.server.routers.chat.get_chat_engine", return_value=fake_engine),
        patch.object(
            ChatManager,
            "_store",
            side_effect=lambda sid, msgs, reply: stored.append((sid, reply)),
        ),
        patch.object(ChatManager, "_track", return_value=("op-3", None)),
        patch.object(ChatManager, "_untrack"),
    ):
        resp = client.post(
            "/chat",
            json={
                "session_id": "s1",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert resp.status_code == 200
    assert resp.json()["message"] == "hi back"
    assert stored == [("s1", "hi back")], "history must be persisted for POST /chat"


def test_post_chat_passes_session_id_to_manager(client, manager):
    """The wire contract's session_id must reach chat's state layer."""
    fake = TurnResult(text="ok", session_id="sess-42")
    fake_engine = _engine_yielding(fake)
    tracked: list[str] = []

    def _track(session_id, label):
        tracked.append(session_id)
        return "op", None

    with (
        patch("domain.chat.get_chat_manager", return_value=manager),
        patch("apps.api.server.routers.chat.get_chat_engine", return_value=fake_engine),
        patch.object(ChatManager, "_store"),
        patch.object(ChatManager, "_track", side_effect=_track),
        patch.object(ChatManager, "_untrack"),
    ):
        resp = client.post(
            "/chat",
            json={
                "session_id": "sess-42",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert resp.status_code == 200
    assert tracked == ["sess-42"]


def test_generate_tracked_returns_backend_object_untouched(manager):
    """The seam must pass the backend's response through, storing only its text."""
    sentinel = TurnResult(text="full reply", session_id="s")
    stored: list[str] = []

    async def _gen():
        return sentinel

    with (
        patch.object(ChatManager, "_track", return_value=("op-4", None)),
        patch.object(ChatManager, "_untrack"),
        patch.object(
            ChatManager,
            "_store",
            side_effect=lambda sid, msgs, reply: stored.append(reply),
        ),
    ):
        result = asyncio.run(
            manager.generate_tracked(
                "s",
                [{"role": "user", "content": "hi"}],
                _gen,
                text_of=lambda r: r.text,
            )
        )

    assert result is sentinel
    assert stored == ["full reply"]


def test_router_exposes_single_generation_seam():
    """Both endpoints must funnel through one streaming seam, per the merge contract."""
    assert hasattr(ChatRouter, "_stream_tokens")
    assert hasattr(ChatRouter, "respond")
    assert hasattr(ChatRouter, "stream")


def test_post_chat_surfaces_engine_details(client, manager):
    """Chat must read the engine's details off the port and shape the response.

    The engine (inference) computes model/usage/finish/timing; chat is the
    response handler, so it reads those off the returned TurnResult and puts
    them on the wire for the frontend — the details the old handler discarded.
    """
    turn = TurnResult(
        text="answer",
        session_id="s1",
        model="qwen2.5-0.5b-instruct",
        usage={"completion_tokens": 7, "total_tokens": 7},
        elapsed_ms=12.5,
        finish_reason="stop",
    )
    fake_engine = _engine_yielding(turn)

    with (
        patch("domain.chat.get_chat_manager", return_value=manager),
        patch("apps.api.server.routers.chat.get_chat_engine", return_value=fake_engine),
        patch.object(ChatManager, "_store"),
        patch.object(ChatManager, "_track", return_value=("op-5", None)),
        patch.object(ChatManager, "_untrack"),
    ):
        resp = client.post(
            "/chat",
            json={"session_id": "s1", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["message"] == "answer"
    assert body["model"] == "qwen2.5-0.5b-instruct"
    assert body["usage"]["completion_tokens"] == 7
    assert body["elapsed_ms"] == 12.5
    assert body["finish_reason"] == "stop"


def test_chat_router_does_not_import_inference_router():
    """Layering pin: the chat surface must read the engine via its port, never
    by importing the sibling inference router (routers must not import routers)."""
    import inspect

    import apps.api.server.routers.chat as chat_mod

    src = inspect.getsource(chat_mod)
    assert "from routers.inference" not in src, (
        "chat.py must not import routers.inference — use domain.generation.engine's port"
    )
    assert "from apps.api.server.routers.inference" not in src
