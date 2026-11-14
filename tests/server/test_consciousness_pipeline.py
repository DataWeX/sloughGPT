"""CONSCIOUSNESS_PROC pipeline step (card 243016c3) — reflection injection path.

The chat pipeline pulls the consciousness engine's propose-only reflection
narrative (reflect(apply=False) — never mutates beliefs) and prepends it as a
system message. Reflection is ADVISORY: a disabled engine, an empty narrative,
or any failure skips injection and the chat proceeds with unchanged messages
(try/except skip pattern, unlike KNOWLEDGE_PROC which errors the stream).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from routers.inference import _apply_consciousness_reflection

# Readiness-gate stand-ins for stream_turn's GATES block: _model_ready reads
# ``state`` via function-local import, so a sys.modules patch flips it without
# loading any weights (same doubles test_inference_router uses).
MOCK_STATE = MagicMock()
MOCK_STATE.model = MagicMock()
MOCK_STATE.model_type = "gpt2"
MOCK_STATE.tokenizer = None
MOCK_STATE.model_request_logger = None
MOCK_STATE.checkpoint = None
MOCK_STATE.soul_engine = None
MOCK_STATE.current_soul = None

MOCK_STARTUP = {"phase": "ready", "step": 8, "total": 8, "message": "Ready"}


class _FakeEngine:
    def __init__(self, narrative: str = "I am curious about compilers.", enabled: bool = True):
        self._narrative = narrative
        self.config = SimpleNamespace(is_enabled=lambda: enabled)

    def reflect(self, apply: bool = True):
        assert apply is False, "prompt injection must be propose-only"
        return SimpleNamespace(narrative=self._narrative)


_MSGS = [{"role": "user", "content": "hi"}]


def _patch_engine(monkeypatch, engine) -> None:
    import domain.cognition as cognition

    monkeypatch.setattr(cognition, "get_consciousness", lambda *a, **k: engine)


class TestConsciousnessInjection:
    async def test_injects_reflection_narrative(self, monkeypatch):
        _patch_engine(monkeypatch, _FakeEngine())
        out = await _apply_consciousness_reflection(list(_MSGS), "corr")
        assert out[0]["role"] == "system"
        assert "curious about compilers" in out[0]["content"]
        assert out[1:] == _MSGS

    async def test_disabled_engine_is_noop(self, monkeypatch):
        _patch_engine(monkeypatch, _FakeEngine(enabled=False))
        assert await _apply_consciousness_reflection(list(_MSGS), "corr") == _MSGS

    async def test_empty_narrative_is_noop(self, monkeypatch):
        _patch_engine(monkeypatch, _FakeEngine(""))
        assert await _apply_consciousness_reflection(list(_MSGS), "corr") == _MSGS

    async def test_engine_failure_skips_injection(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("consciousness unavailable")

        import domain.cognition as cognition

        monkeypatch.setattr(cognition, "get_consciousness", boom)
        assert await _apply_consciousness_reflection(list(_MSGS), "corr") == _MSGS

    async def test_reflect_error_skips_injection(self, monkeypatch):
        class _BoomEngine:
            config = SimpleNamespace(is_enabled=lambda: True)

            def reflect(self, apply: bool = True):
                raise ValueError("self model empty")

        _patch_engine(monkeypatch, _BoomEngine())
        assert await _apply_consciousness_reflection(list(_MSGS), "corr") == _MSGS


class TestStreamTurnConsciousnessGate:
    """``use_consciousness=False`` keeps the bare probe bare (quick-test card).

    The home page's Quick test POSTs "Hello!" as a health check: with the flag
    off, ``stream_turn`` must not touch the consciousness engine at all (no
    reflection narrative prepended, no episode ingest in post-gen). The
    DEFAULT is now OFF (card 20261114_092948 — the narrative-in-system-head
    hijacked ordinary chat replies, probe-proven live 2026-11-14): omitting
    the flag must SKIP reflection, and clients opt in explicitly per turn.

    The probe is ``domain.cognition.get_consciousness`` rather than a router
    attribute: the router module exists under two import aliases
    (``routers.inference`` / ``apps.api.server.routers.inference``) and import
    order decides which one registers as the chat engine, while the call sites
    resolve ``from domain.core import get_consciousness`` lazily at call time —
    one patch point that cannot miss.
    """

    @pytest.fixture
    def chat_client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from infrastructure.exception_handlers import register_all_handlers

        from apps.api.server.routers.chat import router as chat_router
        from apps.api.server.routers.inference import _instance as inference_router

        app = FastAPI()
        register_all_handlers(app)
        # Same mounted surface as the deployed app (both routers own /chat
        # halves — see test_inference_router's app fixture).
        app.include_router(inference_router.router)
        app.include_router(chat_router)
        return TestClient(app, raise_server_exceptions=False)

    @patch.dict(
        "sys.modules",
        {"state": MOCK_STATE, "startup_progress": MagicMock(STARTUP_PHASE=MOCK_STARTUP)},
    )
    def test_flag_off_never_touches_consciousness_default_now_skips(self, chat_client, monkeypatch):
        import routers.inference as inf_top

        import domain.cognition as cognition
        from apps.api.server.routers import inference as inf_apps

        # Both router aliases patched: either may be the registered engine.
        touches: list[str] = []

        def _probe(*_a, **_k):
            touches.append("touched")
            return _FakeEngine()

        provider = MagicMock()

        async def _chat_stream(messages, **kwargs):
            yield "ok"

        provider.chat_stream = _chat_stream

        for mod in (inf_top, inf_apps):
            monkeypatch.setattr(mod, "get_provider", lambda *a, **k: provider)
            # Post-gen side effects stubbed: the real learner/tracker pollute
            # the process for later stream tests (see
            # apps/api/server/tests/test_consciousness_post_process.py).
            monkeypatch.setattr(mod, "capture", lambda *a, **k: None)
            monkeypatch.setattr(mod, "extract_and_store", lambda *a, **k: None)
            monkeypatch.setattr(mod, "get_learner", lambda: MagicMock())
            monkeypatch.setattr(mod, "get_response_tracker", lambda: MagicMock())
            monkeypatch.setattr(mod, "get_server_state", lambda: MagicMock())
        monkeypatch.setattr(cognition, "get_consciousness", _probe)

        # Bare probe: flag off → the engine is never touched.
        resp = chat_client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "Hello!"}],
                "session_id": "quick-test",
                "use_consciousness": False,
                "use_context_core": False,
                "use_rag": False,
            },
        )
        assert resp.status_code == 200
        assert touches == [], "use_consciousness=False must skip reflection AND ingest"

        # Same request without the flag → reflection is SKIPPED (default OFF
        # since card 20261114_092948: narrative-in-system-head hijacked chat).
        resp = chat_client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "hi there"}],
                "use_context_core": False,
                "use_rag": False,
            },
        )
        assert resp.status_code == 200
        assert touches == [], "omitting use_consciousness must default to OFF"

        # Explicit opt-in still runs the reflection. Distinct message: the
        # session-keyed replay cache (chat.py turn_hash) would otherwise
        # replay request 2's completed response and never reach stream_turn.
        resp = chat_client.post(
            "/chat",
            json={
                "messages": [{"role": "user", "content": "reflect on this"}],
                "use_context_core": False,
                "use_rag": False,
                "use_consciousness": True,
            },
        )
        assert resp.status_code == 200
        assert touches, "use_consciousness=True must still run reflection"
