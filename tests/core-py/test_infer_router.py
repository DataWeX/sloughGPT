"""
Tests for the unified /infer router.

Verifies all 7 endpoints: generate, stream, embed, tokenize, detokenize, health, info.
Uses mocked providers and models — no real inference.
"""

import pytest

pytest.importorskip("fastapi")

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

pytestmark = pytest.mark.slow


# ── Fake model ───────────────────────────────────────────────────────


class FakeModel:
    """Minimal ModelInterface mock for testing."""

    def __init__(self):
        self._device = "cpu"

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, **kwargs):
        import numpy as np

        return np.array([[1, 2, 3, 4, 5]])

    def generate_stream(self, prompt, max_new_tokens=100, temperature=0.8, **kwargs):
        yield "Hello"
        yield " "
        yield "world"

    def embed(self, text):
        import numpy as np

        return np.array([0.1, 0.2, 0.3, 0.4])

    def info(self):
        from types import SimpleNamespace

        return SimpleNamespace(
            model_id="test-model",
            model_type="MockModel",
            num_parameters=1000,
            vocab_size=256,
            max_context=128,
            num_layers=0,
            has_tokenizer=True,
            has_streaming=True,
            has_embedding=True,
            extra={},
        )

    def forward(self, input_ids, targets=None, **kwargs):
        import numpy as np

        return np.random.randn(1, 256), None

    def state_dict(self):
        return {}

    def load_state_dict(self, sd, **kwargs):
        pass

    def num_parameters(self):
        return 1000

    def config(self):
        return {"vocab_size": 256, "model_id": "test-model"}

    def to(self, device):
        return self

    def eval(self):
        return self

    def train_mode(self):
        return self


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def app():
    """Create FastAPI app with the infer router."""
    from fastapi import FastAPI
    from infrastructure.exception_handlers import register_app_error_handler

    from apps.api.server.routers.infer import router

    app = FastAPI()
    app.include_router(router)
    # Without this the bare app has no AppError handler, so raise_error() in the
    # router re-raises out of TestClient instead of becoming the 503 response the
    # assertions expect. register_app_error_handler is the "minimal handler for
    # test clients" variant — it adds only AppError, leaving FastAPI defaults.
    register_app_error_handler(app)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


# ── Helper to mock state + startup ──────────────────────────────────


def _patch_state(model=None, phase="ready"):
    """Return list of context managers that mock the router's model accessors.

    Two gates stand between the router and a 200, and the original helper opened
    neither:

    1. ``_get_model_status()`` checks ``_model_ready()``, which reads **global
       ``state``** (``state.model`` / ``state.provider`` / ``ServerState``) — not
       ``InferRouter._get_model``. Patching the accessors alone left it False, so
       the endpoint answered 503 "Model still loading" with a FakeModel in hand.
       Bound here to "ready iff a model was supplied", which keeps the no-model
       tests on their 503 path.
    2. ``STARTUP_PHASE`` starts at ``initializing`` (step 0/8), so a second gate
       answered E_STARTING. ``phase`` was accepted but never applied.
    """
    import routers.inference as inference_mod
    import startup_progress

    import apps.api.server.routers.infer as infer_mod

    return [
        patch.object(infer_mod.InferRouter, "_get_model", return_value=model),
        patch.object(infer_mod.InferRouter, "_get_model_interface", return_value=model),
        # _model_ready lives in routers.inference — where _get_model_status's
        # __globals__ resolve it. `routers.inference` and
        # `apps.api.server.routers.inference` are TWO DISTINCT module objects
        # (verified: id() differs) for the same file, so patching the long-form
        # name silently does nothing. Patch the binding it is read from.
        patch.object(inference_mod, "_model_ready", return_value=model is not None),
        patch.dict(
            startup_progress.STARTUP_PHASE,
            {"phase": phase, "step": 8, "message": "Ready"},
        ),
    ]


# ── Health endpoint ──────────────────────────────────────────────────


class TestInferHealth:
    def test_health_no_model(self, client):
        """Health returns no_model when no model is loaded."""
        patches = _patch_state(model=None)
        for p in patches:
            p.start()
        try:
            resp = client.get("/infer/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "no_model"
            assert data["model_loaded"] is False
        finally:
            for p in patches:
                p.stop()

    def test_health_with_model(self, client):
        """Health returns ready when model is loaded."""
        fake = FakeModel()
        patches = _patch_state(model=fake)
        for p in patches:
            p.start()
        try:
            resp = client.get("/infer/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ready"
            assert data["model_loaded"] is True
            assert data["model_id"] == "test-model"
            assert data["engine_type"] == "MockModel"
        finally:
            for p in patches:
                p.stop()


# ── Info endpoint ────────────────────────────────────────────────────


class TestInferInfo:
    def test_info_no_model_returns_503(self, client):
        """Info returns 503 when no model loaded."""
        patches = _patch_state(model=None)
        for p in patches:
            p.start()
        try:
            resp = client.get("/infer/info")
            assert resp.status_code == 503
        finally:
            for p in patches:
                p.stop()

    def test_info_with_model(self, client):
        """Info returns model metadata."""
        fake = FakeModel()
        patches = _patch_state(model=fake)
        for p in patches:
            p.start()
        try:
            resp = client.get("/infer/info")
            assert resp.status_code == 200
            data = resp.json()
            assert data["model_id"] == "test-model"
            assert data["model_type"] == "MockModel"
            assert data["num_parameters"] == 1000
        finally:
            for p in patches:
                p.stop()


# ── Tokenize endpoint ───────────────────────────────────────────────


class TestInferTokenize:
    def test_tokenize_falls_back_to_morph(self, client):
        """Tokenize falls back to MorphTokenizer when no model tokenizer."""
        patches = _patch_state(model=None)
        for p in patches:
            p.start()
        try:
            resp = client.post("/infer/tokenize", json={"text": "hello"})
            assert resp.status_code == 200
            data = resp.json()
            assert "ids" in data
            assert "tokens" in data
            assert data["count"] > 0
        finally:
            for p in patches:
                p.stop()


# ── Embed endpoint ──────────────────────────────────────────────────


class TestInferEmbed:
    def test_embed_falls_back_to_ngram(self, client):
        """Embed falls back to n-gram when no model embed."""
        patches = _patch_state(model=None)
        for p in patches:
            p.start()
        try:
            resp = client.post("/infer/embed", json={"text": "hello world"})
            assert resp.status_code == 200
            data = resp.json()
            assert "embedding" in data
            assert len(data["embedding"]) > 0
            assert data["dimensions"] == len(data["embedding"])
        finally:
            for p in patches:
                p.stop()

    def test_embed_with_model(self, client):
        """Embed uses model.embed() when available."""
        fake = FakeModel()
        patches = _patch_state(model=fake)
        for p in patches:
            p.start()
        try:
            resp = client.post("/infer/embed", json={"text": "hello"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["embedding"] == [0.1, 0.2, 0.3, 0.4]
            assert data["dimensions"] == 4
        finally:
            for p in patches:
                p.stop()


# ── Generate endpoint ───────────────────────────────────────────────


class TestInferGenerate:
    def test_generate_no_model_returns_503(self, client):
        """Generate returns 503 when model not ready."""
        patches = _patch_state(model=None)
        for p in patches:
            p.start()
        try:
            resp = client.post("/infer", json={"prompt": "hello"})
            assert resp.status_code == 503
        finally:
            for p in patches:
                p.stop()

    def test_generate_with_provider(self, client):
        """Generate returns text when provider is available."""
        fake = FakeModel()
        mock_provider = AsyncMock()
        mock_provider.chat = AsyncMock(return_value="Hello world!")

        # _patch_state also opens the startup gate — without it the endpoint
        # answers E_STARTING (503) before the provider is ever consulted.
        #
        # Provider target: infer.py does `from domain.models import get_provider`
        # (function-level), so the binding actually read is
        # domain.models.get_provider. Patching
        # domain.models._internal.provider.get_provider moves a different object
        # and the endpoint still answers "No provider available" (503).
        patches = _patch_state(model=fake) + [
            patch("domain.models.get_provider", return_value=mock_provider),
        ]
        for p in patches:
            p.start()
        try:
            resp = client.post("/infer", json={"prompt": "hello"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["text"] == "Hello world!"
            assert data["tokens_generated"] > 0
        finally:
            for p in patches:
                p.stop()


# ── Schema validation ───────────────────────────────────────────────


class TestInferSchemas:
    def test_generate_rejects_empty_prompt(self, client):
        """Generate rejects request with no prompt."""
        resp = client.post("/infer", json={})
        assert resp.status_code == 422

    def test_generate_rejects_large_max_tokens(self, client):
        """Generate rejects max_new_tokens > 2048."""
        resp = client.post("/infer", json={"prompt": "hi", "max_new_tokens": 9999})
        assert resp.status_code == 422

    def test_embed_rejects_empty_text(self, client):
        """Embed rejects request with no text."""
        resp = client.post("/infer/embed", json={})
        assert resp.status_code == 422

    def test_tokenize_rejects_empty_text(self, client):
        """Tokenize rejects request with no text."""
        resp = client.post("/infer/tokenize", json={})
        assert resp.status_code == 422

    def test_detokenize_rejects_empty_ids(self, client):
        """Detokenize rejects request with no ids."""
        resp = client.post("/infer/detokenize", json={})
        assert resp.status_code == 422
