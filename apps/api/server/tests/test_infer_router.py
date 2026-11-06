"""Tests for the Infer router — 7 endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.exception_handlers import register_app_error_handler

_AUTH_USER = {"sub": "user1", "tenant_id": "t1"}


def _build_app(auth_user_dict=_AUTH_USER):
    from routers.infer import InferRouter
    from routers.inference import router as inference_router

    router_obj = InferRouter()
    _app = FastAPI()
    register_app_error_handler(_app)
    _app.include_router(router_obj.router)
    _app.include_router(inference_router)
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
    def test_infer_uses_shared_text_generation_logic(self):
        _app, router = _build_app()
        from infrastructure.inference_generation import GenerationResult

        provider = MagicMock()
        generate = AsyncMock(
            return_value=GenerationResult(
                text="Hello, world!", tokens_generated=2, elapsed_ms=8.5
            )
        )
        server_state = MagicMock()
        server_state.model_type.get.return_value = "test-model"
        server_state.tokenizer.get.return_value = None
        payload = {
            "prompt": "Hello",
            "max_new_tokens": 10,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 5,
            "repetition_penalty": 1.0,
        }
        with (
            patch("routers.inference._get_model_status", return_value={"ready": True}),
            patch("routers.inference._check_memory_pressure", return_value=None),
            patch("routers.inference.get_provider", return_value=provider),
            patch("domain.infrastructure.server_state.get_server_state", return_value=server_state),
            patch("routers.inference.generate_text", new=generate),
        ):
            response = TestClient(_app).post("/infer", json=payload)

        assert response.status_code == 200
        assert response.json() == {
            "text": "Hello, world!",
            "model": "test-model",
            "tokens_generated": 2,
            "elapsed_ms": 8.5,
        }
        generate.assert_awaited_once_with(
            provider,
            [{"role": "user", "content": "Hello"}],
            max_tokens=10,
            generation_params={
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 5,
                "repetition_penalty": 1.0,
            },
            tokenizer=None,
        )
