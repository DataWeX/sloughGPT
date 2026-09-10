"""Tests for the Tokenizer router — all 11 endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.exception_handlers import register_app_error_handler


_AUTH_USER = {"sub": "user1", "tenant_id": "t1"}


def _build_app(auth_user_dict=_AUTH_USER):
    from routers.tokenizer import TokenizerRouter

    router_obj = TokenizerRouter()
    _app = FastAPI()
    register_app_error_handler(_app)
    _app.include_router(router_obj.router)
    _app.dependency_overrides[require_auth_if_enabled] = lambda: auth_user_dict
    return _app, router_obj


def _mock_manager(trained=True):
    mgr = MagicMock()
    mgr.is_trained.return_value = trained
    mgr.stats.return_value = {
        "vocab_size": 256,
        "base_chars": 50,
        "merged_subwords": 200,
        "special_tokens": 4,
        "total_merges": 200,
        "trained": trained,
    }
    mgr.tokenize.return_value = [1, 2, 3]
    mgr.detokenize.return_value = "hello world"
    mgr.show_pretokenization.return_value = {"original": "test", "parts": ["test"]}
    mgr.decompose_token.return_value = {"token": "test", "parts": ["t", "est"]}
    mgr.analyze_corpus.return_value = {"total_tokens": 100, "unique_tokens": 50}
    tok = MagicMock()
    tok.vocab_size = 256
    tok.vocab = {i: f"token_{i}" for i in range(256)}
    tok.itos = {i: f"token_{i}" for i in range(256)}
    tok.SPECIAL_TOKENS = {"<pad>", "<unk>", "<s>", "</s>"}
    tok.merges = [("a", "b"), ("c", "d")]
    tok.encode.return_value = [1, 2, 3]
    mgr.get_tokenizer.return_value = tok
    return mgr


# ── Stats ────────────────────────────────────────────────────────────────────


class TestTokenizerStats:
    def test_get_stats(self):
        _app, router = _build_app()
        mgr = _mock_manager()
        with patch("routers.tokenizer.get_tokenizer_manager", return_value=mgr):
            resp = TestClient(_app).get("/tokenizer/stats")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "vocab_size" in data
        assert data["vocab_size"] == 256


# ── Tokenize / Detokenize ───────────────────────────────────────────────────


class TestTokenize:
    def test_tokenize(self):
        _app, router = _build_app()
        mgr = _mock_manager()
        with patch("routers.tokenizer.get_tokenizer_manager", return_value=mgr):
            resp = TestClient(_app).post("/tokenizer/tokenize", json={"text": "hello"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "ids" in data
        assert "tokens" in data

    def test_detokenize(self):
        _app, router = _build_app()
        mgr = _mock_manager()
        with patch("routers.tokenizer.get_tokenizer_manager", return_value=mgr):
            resp = TestClient(_app).post("/tokenizer/detokenize", json={"ids": [1, 2, 3]})
        assert resp.status_code == 200
        assert resp.json()["data"]["text"] == "hello world"


# ── Pre-tokenize / Decompose / Analyze ───────────────────────────────────────


class TestPreTokenize:
    def test_pretokenize(self):
        _app, router = _build_app()
        mgr = _mock_manager()
        with patch("routers.tokenizer.get_tokenizer_manager", return_value=mgr):
            resp = TestClient(_app).post("/tokenizer/pretokenize", json={"text": "hello"})
        assert resp.status_code == 200

    def test_decompose(self):
        _app, router = _build_app()
        mgr = _mock_manager()
        with patch("routers.tokenizer.get_tokenizer_manager", return_value=mgr):
            resp = TestClient(_app).post("/tokenizer/decompose", json={"text": "test"})
        assert resp.status_code == 200

    def test_analyze(self):
        _app, router = _build_app()
        mgr = _mock_manager()
        with patch("routers.tokenizer.get_tokenizer_manager", return_value=mgr):
            resp = TestClient(_app).post("/tokenizer/analyze", json={"texts": ["hello", "world"]})
        assert resp.status_code == 200


# ── Vocab / Merges ──────────────────────────────────────────────────────────


class TestVocabMerges:
    def test_get_vocab(self):
        _app, router = _build_app()
        mgr = _mock_manager()
        with patch("routers.tokenizer.get_tokenizer_manager", return_value=mgr):
            resp = TestClient(_app).get("/tokenizer/vocab?limit=10")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "entries" in data
        assert len(data["entries"]) == 10

    def test_get_merges(self):
        _app, router = _build_app()
        mgr = _mock_manager()
        with patch("routers.tokenizer.get_tokenizer_manager", return_value=mgr):
            resp = TestClient(_app).get("/tokenizer/merges?limit=5")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "merges" in data


# ── Train ────────────────────────────────────────────────────────────────────


class TestTrain:
    def test_train_with_texts(self):
        _app, router = _build_app()
        mgr = _mock_manager()
        with patch("routers.tokenizer.get_tokenizer_manager", return_value=mgr):
            resp = TestClient(_app).post(
                "/tokenizer/train",
                json={"vocab_size": 256, "texts": ["hello world", "test"]},
            )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] == "trained"


# ── Sample ───────────────────────────────────────────────────────────────────


class TestSample:
    def test_get_sample(self):
        _app, router = _build_app()
        mgr = _mock_manager()
        with patch("routers.tokenizer.get_tokenizer_manager", return_value=mgr):
            resp = TestClient(_app).get("/tokenizer/sample")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "samples" in data
