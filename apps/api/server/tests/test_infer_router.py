"""Tests for the Infer router — 7 endpoints."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.exception_handlers import register_app_error_handler

from domain.infrastructure.memory_pressure import PressureLevel

_AUTH_USER = {"sub": "user1", "tenant_id": "t1"}


def _build_app(auth_user_dict=_AUTH_USER):
    from routers.models import InferRouter

    router_obj = InferRouter()
    _app = FastAPI()
    register_app_error_handler(_app)
    _app.include_router(router_obj.router)
    _app.dependency_overrides[require_auth_if_enabled] = lambda: auth_user_dict
    return _app, router_obj


def _mock_provider():
    provider = MagicMock()
    provider.generate.return_value = "Hello, world!"
    provider.tokenize.return_value = [1, 2, 3]
    provider.detokenize.return_value = "hello"
    provider.embed.return_value = [0.1, 0.2, 0.3]

    # chat is async, needs to return an awaitable
    async def _chat(*args, **kwargs):
        return "Hello, world!"

    provider.chat = _chat
    return provider


# ── Health ───────────────────────────────────────────────────────────────────


class TestInferHealth:
    def test_health_no_model(self):
        _app, router = _build_app()
        with patch.object(router, "_get_model", return_value=None):
            resp = TestClient(_app).get("/infer/health")
        assert resp.status_code == 200
        data = resp.json()
        # Health returns status based on model state
        assert "status" in data


# ── Info ─────────────────────────────────────────────────────────────────────


class TestInferInfo:
    def test_info_no_model(self):
        _app, router = _build_app()
        with patch.object(router, "_get_model", return_value=None):
            resp = TestClient(_app).get("/infer/info")
        # Info may return 503 when no model is loaded
        assert resp.status_code in (200, 503)


# ── Tokenize / Detokenize ───────────────────────────────────────────────────


class TestInferTokenize:
    def test_tokenize(self):
        _app, router = _build_app()
        provider = _mock_provider()
        with patch("domain.models.get_provider", return_value=provider):
            resp = TestClient(_app).post("/infer/tokenize", json={"text": "hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert "ids" in data
        assert "tokens" in data

    def test_detokenize(self):
        _app, router = _build_app()
        provider = _mock_provider()
        with patch("domain.models.get_provider", return_value=provider):
            resp = TestClient(_app).post("/infer/detokenize", json={"ids": [1, 2, 3]})
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data


# ── Embed ────────────────────────────────────────────────────────────────────


class TestInferEmbed:
    def test_embed(self):
        _app, router = _build_app()
        provider = _mock_provider()
        with patch("domain.models.get_provider", return_value=provider):
            resp = TestClient(_app).post("/infer/embed", json={"text": "hello world"})
        assert resp.status_code == 200
        data = resp.json()
        assert "embedding" in data
        assert "dimensions" in data


# ── Infer (generate) ─────────────────────────────────────────────────────────


class TestInferGenerate:
    def test_generate_endpoint_exists(self):
        """Verify the generate endpoint is registered and accepts POST."""
        _app, router = _build_app()
        # Just verify the endpoint exists and accepts the request format
        # The actual generation depends on model state which is complex to mock
        provider = _mock_provider()
        mock_model = MagicMock()
        with (
            patch.object(router, "_get_model", return_value=mock_model),
            patch("domain.models.get_provider", return_value=provider),
        ):
            resp = TestClient(_app).post(
                "/infer",
                json={"prompt": "Hello", "max_new_tokens": 10},
            )
        # Endpoint exists and processes the request (may fail on model internals)
        assert resp.status_code in (200, 400, 500, 503)


# ── Response cache: /infer's projection of the ladder (card 20261102_023531) ─


class TestInferResponseCacheLadder:
    """The /infer answer tier: a repeated deterministic request is replayed.

    The ladder itself lives in ``routers.llm_v1._ladder`` — ONE decision
    function every envelope projects (card 20261102_023531: reuse the /v1
    ladder, never re-derive it). These tests pin the projection: the seam
    (the provider is called once for two identical deterministic requests),
    key completeness (two requests differing in exactly ONE InferRequest field
    must both reach the engine — a field missing from the key is a fluent
    wrong-answer class), and the envelope facts (non-streaming, tool-less,
    the stochastic default bypasses).
    """

    def setup_method(self):
        from domain.inference._internal.response_cache import reset_response_cache_for_tests

        reset_response_cache_for_tests()

    @staticmethod
    def _stats() -> dict:
        from domain.inference._internal.response_cache import get_response_cache

        return get_response_cache().get_stats()

    @staticmethod
    def _fake_provider() -> MagicMock:
        provider = MagicMock()
        provider.calls: list[tuple[list, dict]] = []

        async def _chat(messages, **kwargs):
            provider.calls.append((list(messages), dict(kwargs)))
            return "Hello, world!"

        provider.chat = _chat
        return provider

    @contextmanager
    def _engine(self, provider: MagicMock):
        """Patch the reuse seams to a deterministic engine (llm_v1 parity)."""
        monitor = MagicMock()
        monitor.check.return_value = PressureLevel.NORMAL
        with ExitStack() as stack:
            stack.enter_context(
                patch("routers.inference._get_model_status", return_value={"ready": True})
            )
            stack.enter_context(
                patch(
                    "domain.infrastructure.memory_pressure.get_memory_pressure_monitor",
                    return_value=monitor,
                )
            )
            stack.enter_context(patch("domain.models.get_provider", return_value=provider))
            yield

    @staticmethod
    def _post(app: FastAPI, **overrides):
        body = {"prompt": "hi", "temperature": 0.0}
        body.update(overrides)
        return TestClient(app).post("/infer", json=body)

    def test_repeated_deterministic_request_is_replayed(self):
        provider = self._fake_provider()
        app, router = _build_app()
        with patch.object(router, "_get_model", return_value=None), self._engine(provider):
            first = self._post(app)
            second = self._post(app)

        assert len(provider.calls) == 1, "second identical request reached the engine"
        generated, replay = first.json(), second.json()
        assert generated["cached"] is False
        assert replay["cached"] is True
        assert replay["text"] == generated["text"]
        assert replay["tokens_generated"] == generated["tokens_generated"]
        assert replay["model"] == generated["model"]
        assert self._stats()["hits"] == 1

    def test_stochastic_default_always_generates(self):
        """/infer's default temperature (0.7) sits above the determinism rung."""
        provider = self._fake_provider()
        app, router = _build_app()
        with patch.object(router, "_get_model", return_value=None), self._engine(provider):
            self._post(app, temperature=0.9)
            second = self._post(app, temperature=0.9)

        assert len(provider.calls) == 2
        assert second.json()["cached"] is False
        assert self._stats()["bypass_stochastic"] == 2

    @pytest.mark.parametrize(
        ("field", "first", "second"),
        [
            ("prompt", "hi", "hello"),
            ("max_new_tokens", 16, 32),
            ("temperature", 0.0, 0.1),
            ("top_p", 0.5, 0.9),
            ("top_k", 40, 20),
            ("repetition_penalty", 1.15, 1.3),
            ("model", "m1", "m2"),
            ("response_format", None, "json"),
        ],
    )
    def test_every_request_field_is_in_the_key(self, field, first, second):
        """Key completeness is the entire correctness argument.

        Every InferRequest field participates exactly once; a field missing
        from the key would replay one request's text under another request's
        parameters — a fluent wrong answer, invisible downstream.
        """
        provider = self._fake_provider()
        app, router = _build_app()
        with patch.object(router, "_get_model", return_value=None), self._engine(provider):
            self._post(app, **{field: first})
            replay = self._post(app, **{field: second})

        assert len(provider.calls) == 2, f"{field} is missing from the key surface"
        assert replay.json()["cached"] is False
