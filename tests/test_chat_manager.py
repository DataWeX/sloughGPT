"""Tests for ChatManager facade (domain/chat/_internal/manager.py)."""

import uuid

import pytest

import domain.models._internal.provider as provmod
from domain.chat import (
    ChatManager,
    get_chat_manager,
    reset_chat_manager,
)
from domain.chat._internal.domain import ChatDomain


class FakeProvider:
    """Minimal provider honoring cancel_event + session_id like the real ones."""

    model_id = "fake-test"

    def __init__(self, tokens=("he", "llo", " world")):
        self._tokens = list(tokens)
        self.seen = []
        self.last_usage = None

    async def chat(self, messages, max_tokens=512, temperature=0.7, **kwargs):
        self.seen.append((list(messages), kwargs.get("session_id")))
        return "hello"

    async def chat_stream(
        self,
        messages,
        max_tokens=512,
        temperature=0.7,
        cancel_event=None,
        session_id=None,
        **kwargs,
    ):
        self.seen.append((list(messages), session_id))
        for tok in self._tokens:
            if cancel_event is not None and cancel_event.is_set():
                break
            yield tok


@pytest.fixture
def fake_provider():
    prev = dict(provmod._providers)
    fake = FakeProvider()
    provmod._providers["default"] = fake
    yield fake
    provmod._providers.clear()
    provmod._providers.update(prev)


def test_last_usage_reports_provider_usage(manager, fake_provider):
    fake_provider.last_usage = {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
    assert manager.last_usage() == {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}


def test_last_usage_none_without_provider():
    prev = dict(provmod._providers)
    provmod._providers.clear()
    try:
        mgr = ChatManager(domain=ChatDomain())
        assert mgr.last_usage() is None
    finally:
        provmod._providers.clear()
        provmod._providers.update(prev)


@pytest.fixture
def manager(fake_provider):  # noqa: ARG001
    reset_chat_manager()
    mgr = ChatManager(domain=ChatDomain())
    yield mgr
    reset_chat_manager()


def _sid():
    return f"cm-test-{uuid.uuid4().hex[:8]}"


async def test_respond_returns_text_and_stores_history(manager):
    sid = _sid()
    resp = await manager.respond([{"role": "user", "content": "hi"}], session_id=sid)
    assert resp.text == "hello"
    assert resp.session_id == sid
    history = manager.history(sid)
    assert history[0] == {"role": "user", "content": "hi"}
    assert history[-1]["role"] == "assistant"


async def test_respond_no_provider_returns_error():
    prev = dict(provmod._providers)
    provmod._providers.clear()
    try:
        mgr = ChatManager(domain=ChatDomain())
        resp = await mgr.respond([{"role": "user", "content": "hi"}], session_id=_sid())
        assert resp.text.startswith("[Error: No provider")
    finally:
        provmod._providers.clear()
        provmod._providers.update(prev)


async def test_stream_yields_tokens_in_order(manager):
    tokens = [
        t async for t in manager.stream([{"role": "user", "content": "hi"}], session_id=_sid())
    ]
    assert tokens == ["he", "llo", " world"]


async def test_stream_cancel_stops_early(manager):
    sid = _sid()
    gen = manager.stream([{"role": "user", "content": "hi"}], session_id=sid)
    first = await gen.__anext__()
    assert first == "he"
    assert manager.cancel_session(sid) >= 1
    rest = [t async for t in gen]
    assert len(rest) < 2  # stopped early, never buffered the tail


async def test_regenerate_uses_stored_history(manager):
    sid = _sid()
    await manager.respond([{"role": "user", "content": "hi"}], session_id=sid)
    resp = await manager.regenerate(sid)
    assert resp.text == "hello"


async def test_regenerate_empty_history(manager):
    resp = await manager.regenerate(_sid())
    assert resp.text == "[no history]"


def test_cancel_unknown_session_returns_zero(manager):
    assert manager.cancel_session("nope-" + uuid.uuid4().hex[:8]) == 0


def test_health_reports_availability(manager):
    info = manager.health()
    assert info["available"] is True
    assert info["model_id"] == "fake-test"


def test_health_unavailable_when_no_provider():
    prev = dict(provmod._providers)
    provmod._providers.clear()
    try:
        assert ChatManager().health()["available"] is False
    finally:
        provmod._providers.clear()
        provmod._providers.update(prev)


def test_singleton():
    reset_chat_manager()
    try:
        assert get_chat_manager() is get_chat_manager()
    finally:
        reset_chat_manager()


def test_addons_lists_processors_and_capabilities(manager):
    from unittest.mock import patch

    with patch.dict(provmod._processors, {"ToolUse": object()}):
        info = manager.addons()
    assert "ToolUse" in info["processors"]
    assert info["processors"] == sorted(info["processors"])
    assert isinstance(info["capabilities"], dict)


def test_addons_no_provider_reports_empty():
    prev = dict(provmod._providers)
    provmod._providers.clear()
    try:
        info = ChatManager().addons()
        assert info["capabilities"] == {}
        assert isinstance(info["processors"], list)
    finally:
        provmod._providers.clear()
        provmod._providers.update(prev)


def test_attach_adapter_unsupported_provider(manager):
    out = manager.attach_adapter("/tmp/nope.npz")
    assert out == {"attached": False, "error": "Provider does not support adapters"}


def test_attach_adapter_delegates_to_provider():
    class AdapterProvider(FakeProvider):
        def __init__(self):
            super().__init__()
            self.calls = []

        def apply_adapter(self, path, merge=False):
            self.calls.append((path, merge))
            return {"rank": 8, "merged": merge}

    prev = dict(provmod._providers)
    fake = AdapterProvider()
    provmod._providers["default"] = fake
    try:
        out = ChatManager().attach_adapter("/tmp/a.npz", merge=True)
        assert out["attached"] is True
        assert out["rank"] == 8 and out["merged"] is True
        assert fake.calls == [("/tmp/a.npz", True)]
    finally:
        provmod._providers.clear()
        provmod._providers.update(prev)


def test_attach_adapter_no_provider():
    prev = dict(provmod._providers)
    provmod._providers.clear()
    try:
        out = ChatManager().attach_adapter("/tmp/a.npz")
        assert out == {"attached": False, "error": "No provider available"}
    finally:
        provmod._providers.clear()
        provmod._providers.update(prev)
