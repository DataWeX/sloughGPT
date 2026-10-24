"""POST /chat must own chat's state rather than delegate it away.

The delegation away from chat's own handler quietly made the main endpoint
stateless: the old adapter took ``req.messages`` from the client and
never persisted, so no history was written and no op was ever registered for
cancellation — while ``regenerate``/``cancel``/``active_sessions`` kept running
on ``domain/chat``'s stateful kernel. These tests pin the composed behaviour at
the seam: the model in the engine, state in chat.

``ChatManager`` declares ``__slots__``, so its collaborators are patched on the
CLASS, never on an instance.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apps.api.server.routers.chat import ChatRouter
from domain.chat._internal.manager import ChatManager
from domain.inference.engine import TurnResult


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
    """A fake engine whose ``stream`` yields the given items.

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
    # Token first, then the final result — the shape the engine actually emits.
    fake_engine = _engine_yielding("hi back", fake)

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
    assert resp.headers["content-type"].startswith("text/event-stream"), (
        "POST /chat streams — it is the endpoint's only state"
    )
    assert "hi back" in resp.text, "the engine's token must reach the wire"
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

    async def _gen(_stop):
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


def test_generate_tracked_delivers_the_registered_stop_event(manager):
    """The stop event handed to the callable must be the one the registry sets.

    This is the whole cancel channel: ``_track`` mints the ``threading.Event``
    and registers ``stop.set`` as the cancel callback, so the ONLY way a cancel
    reaches the token loop is if that same object is passed into the callable.
    A discarded ``_stop`` leaves ``CancelManager.cancel()`` firing into a flag
    nobody reads — the API reports ``cancelled: true`` while generation runs to
    completion. Both tracked entry points must deliver it.
    """
    import threading

    stop = threading.Event()
    seen: dict[str, object] = {}

    async def _gen(received):
        seen["stop"] = received
        return TurnResult(text="ok", session_id="s")

    async def _stream(received):
        seen["stream_stop"] = received
        if False:  # pragma: no cover - makes this an async generator
            yield None

    async def _drain():
        async for _ in manager.generate_stream_tracked(
            "s",
            [{"role": "user", "content": "hi"}],
            _stream,
            text_of=lambda r: r.text,
        ):
            pass

    with (
        patch.object(ChatManager, "_track", return_value=("op-9", stop)),
        patch.object(ChatManager, "_untrack"),
        patch.object(ChatManager, "_store"),
    ):
        asyncio.run(
            manager.generate_tracked(
                "s", [{"role": "user", "content": "hi"}], _gen, text_of=lambda r: r.text
            )
        )
        asyncio.run(_drain())

    assert seen["stop"] is stop, "generate_tracked must deliver the registered stop event"
    assert seen["stream_stop"] is stop, (
        "generate_stream_tracked must deliver the registered stop event"
    )


def test_router_exposes_single_engine_seam():
    """One chat entry point: POST /chat, and it streams.

    Pinned so a well-meaning "add a non-streaming variant" fails loudly. A
    second route would re-advertise streaming as a choice, which is the exact
    coupling the engine port removed; a caller wanting a complete reply folds
    the stream at its own edge.
    """
    assert hasattr(ChatRouter, "_stream_tokens")
    assert hasattr(ChatRouter, "stream")
    assert not hasattr(ChatRouter, "respond"), (
        "POST /chat is the single entry point — the fold is not a separate route"
    )

    paths = {
        (route.path, method) for route in ChatRouter().router.routes for method in route.methods
    }
    assert ("/chat", "POST") in paths, "POST /chat must be registered"
    assert ("/stream", "POST") not in paths, (
        "/chat/stream must not return — streaming is POST /chat's only state"
    )


def test_post_chat_surfaces_engine_details(client, manager):
    """Chat must read the engine's details off the port and put them on the wire.

    The engine computes usage/timing/finish on the final ``TurnResult``; chat is
    the response handler, so it surfaces those on the closing SSE frame instead
    of discarding them the way the old handler did.
    """
    turn = TurnResult(
        text="answer",
        session_id="s1",
        model="qwen2.5-0.5b-instruct",
        usage={"completion_tokens": 7, "total_tokens": 7},
        elapsed_ms=12.5,
        finish_reason="stop",
    )
    # Token first, then the final result carrying the details.
    fake_engine = _engine_yielding("answer", turn)

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
    body = resp.text
    assert "answer" in body
    # Details ride the closing frame, so a folding client loses nothing.
    for key in ("completion_tokens", "total_tokens", "model", "finish_reason", "elapsed_ms"):
        assert key in body, f"closing frame must carry {key} off the TurnResult"


def test_chat_router_does_not_import_inference_router():
    """Layering pin: the chat surface must read the engine via its port, never
    by importing the sibling inference router (routers must not import routers)."""
    import inspect

    import apps.api.server.routers.chat as chat_mod

    src = inspect.getsource(chat_mod)
    assert "from routers.inference" not in src, (
        "chat.py must not import routers.inference — use domain.inference.engine's port"
    )
    assert "from apps.api.server.routers.inference" not in src


# --- Reconnect replay (card 8125c86e) ---------------------------------------
# POST /chat stores partials on disconnect and completions after the closing
# frame; a retry that proves itself with Last-Event-ID replays the cached
# turn's *unseen tail* without ever calling the engine again.


def test_complete_turn_populates_reconnect_cache(client, manager):
    """DoD 1 (completion half): a finished turn lands in the reconnect cache."""
    from infrastructure.chat_response_cache import get_chat_response_cache

    sid = "s-cache-complete"
    fake_engine = _engine_yielding("hi back", TurnResult(text="hi back", session_id=sid))

    with (
        patch("domain.chat.get_chat_manager", return_value=manager),
        patch("apps.api.server.routers.chat.get_chat_engine", return_value=fake_engine),
        patch.object(ChatManager, "_store"),
        patch.object(ChatManager, "_track", return_value=("op-c1", None)),
        patch.object(ChatManager, "_untrack"),
    ):
        resp = client.post(
            "/chat",
            json={"session_id": sid, "messages": [{"role": "user", "content": "hi"}]},
        )

    assert resp.status_code == 200
    cached = get_chat_response_cache(sid)
    assert cached is not None, "a completed turn must be cached for reconnection"
    assert cached["complete"] is True
    assert cached["tokens"] == ["hi back"]
    assert int(cached["event_counter"]) >= 2, "counter must cover the token frame + done frame"


def test_replay_serves_unseen_tail_without_calling_engine(client, manager):
    """DoD 2 + 3: complete cache + Last-Event-ID replays the tail; engine never runs."""
    from infrastructure.chat_response_cache import set_chat_response_cache

    sid = "s-cache-replay"
    # 3 token frames (ids 1..3) + done (id 4); the client folded ids 1..2.
    set_chat_response_cache(
        sid,
        {"tokens": ["Hel", "lo ", "world"], "complete": True, "event_counter": 4},
    )
    fake_engine = MagicMock()

    with (
        patch("domain.chat.get_chat_manager", return_value=manager),
        patch("apps.api.server.routers.chat.get_chat_engine", return_value=fake_engine),
    ):
        resp = client.post(
            "/chat",
            json={"session_id": sid, "messages": [{"role": "user", "content": "hi"}]},
            headers={"Last-Event-ID": "2"},
        )

    assert resp.status_code == 200
    body = resp.text
    assert "world" in body, "the unseen tail must reach the client"
    assert "Hel" not in body and "lo " not in body, (
        "already-folded tokens must not be re-emitted — the frontend appends "
        "across retries without resetting its buffer"
    )
    assert '"status": "complete"' in body, "the closing frame must follow the tail"
    fake_engine.stream.assert_not_called()


def test_replay_sends_only_closing_frame_when_client_missed_done(client, manager):
    """Client folded every token but never saw the done frame → done-only replay."""
    from infrastructure.chat_response_cache import set_chat_response_cache

    sid = "s-cache-doneonly"
    set_chat_response_cache(
        sid,
        {"tokens": ["a", "b"], "complete": True, "event_counter": 3},
    )
    fake_engine = MagicMock()

    with (
        patch("domain.chat.get_chat_manager", return_value=manager),
        patch("apps.api.server.routers.chat.get_chat_engine", return_value=fake_engine),
    ):
        resp = client.post(
            "/chat",
            json={"session_id": sid, "messages": [{"role": "user", "content": "hi"}]},
            headers={"Last-Event-ID": "2"},
        )

    assert resp.status_code == 200
    assert '"status": "complete"' in resp.text
    assert '"token": "a"' not in resp.text, "full re-emit would duplicate the client's buffer"
    fake_engine.stream.assert_not_called()


def test_fresh_request_without_header_regenerates_over_stale_cache(client, manager):
    """A new user message has no Last-Event-ID — it must generate, never replay.

    Pins the strict counter check: a stale complete cache left by the previous
    turn must not be served as the answer to the next question.
    """
    from infrastructure.chat_response_cache import (
        get_chat_response_cache,
        set_chat_response_cache,
    )

    sid = "s-cache-fresh"
    set_chat_response_cache(
        sid,
        {"tokens": ["stale answer"], "complete": True, "event_counter": 2},
    )
    fake_engine = _engine_yielding("fresh reply", TurnResult(text="fresh reply", session_id=sid))

    with (
        patch("domain.chat.get_chat_manager", return_value=manager),
        patch("apps.api.server.routers.chat.get_chat_engine", return_value=fake_engine),
        patch.object(ChatManager, "_store"),
        patch.object(ChatManager, "_track", return_value=("op-c2", None)),
        patch.object(ChatManager, "_untrack"),
    ):
        resp = client.post(
            "/chat",
            json={"session_id": sid, "messages": [{"role": "user", "content": "second question"}]},
        )

    assert resp.status_code == 200
    assert "fresh reply" in resp.text
    assert "stale answer" not in resp.text, "replay must never answer a NEW message"
    cached = get_chat_response_cache(sid)
    assert cached["tokens"] == ["fresh reply"], "the new turn overwrites the cache"


def test_disconnect_after_first_token_caches_partial(manager):
    """DoD 1 (disconnect half): a mid-stream disconnect stores the partial text."""
    from infrastructure.chat_response_cache import get_chat_response_cache

    from apps.api.server.schemas.chat import ChatRequest

    sid = "s-cache-partial"
    fake_engine = _engine_yielding("Hel", TurnResult(text="Hel", session_id=sid))
    http_req = SimpleNamespace(
        headers={},
        # First item streams; the client is gone by the second.
        is_disconnected=AsyncMock(side_effect=[False, True]),
    )

    with (
        patch("domain.chat.get_chat_manager", return_value=manager),
        patch("apps.api.server.routers.chat.get_chat_engine", return_value=fake_engine),
        patch.object(ChatManager, "_store"),
        patch.object(ChatManager, "_track", return_value=("op-p1", None)),
        patch.object(ChatManager, "_untrack"),
    ):
        request = ChatRequest(messages=[{"role": "user", "content": "hi"}], session_id=sid)
        resp = asyncio.run(ChatRouter().stream(request, http_req, auth_user=None))

        async def _drain() -> None:
            async for _ in resp.body_iterator:  # pragma: no cover - loop consumed
                pass

        asyncio.run(_drain())

    cached = get_chat_response_cache(sid)
    assert cached is not None, "disconnect must cache what was already generated"
    assert cached["complete"] is False
    assert cached["tokens"] == ["Hel"]
