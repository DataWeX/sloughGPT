"""
Tests for the Phoneme router endpoints (apps/api/server/routers/phoneme.py).

Covers POST /phoneme/encode, /decode, /visualize, /score, /batch-encode,
/detect-language, /synthesize and GET /phoneme/languages, /status — happy path
+ 404/405/422 + auth edge. The phoneme engine is faked for determinism; these
tests exercise the HTTP contract (envelope, validation, auth).
"""

from types import SimpleNamespace

import pytest
from routers.phoneme import PhonemeRouter
from test_support import _data, get_test_client

client = get_test_client()


class _FakeEngine:
    """Deterministic stand-in for domain.voice.get_phoneme_engine()."""

    def encode(self, text, language=None):
        return SimpleNamespace(success=True, data={"text": text, "ids": [1, 2, 3]}, error=None)

    def decode(self, ids):
        return SimpleNamespace(success=True, data={"text": "ok", "ids": list(ids)}, error=None)

    def visualize(self, text, language=None):
        return SimpleNamespace(success=True, data={"text": text}, error=None)

    def score(self, target, spoken, language=None):
        return SimpleNamespace(success=True, data={"score": 0.9}, error=None)

    def batch_encode(self, texts, language=None):
        return SimpleNamespace(
            success=True,
            data={"results": [{"text": t, "ids": [1]} for t in texts]},
            error=None,
        )

    def detect_language(self, text):
        return SimpleNamespace(success=True, data={"language": "en"}, error=None)

    def synthesize(self, text):
        return SimpleNamespace(success=True, data={"audio": "base64==", "text": text}, error=None)

    def supported_languages(self):
        return SimpleNamespace(success=True, data={"languages": ["en"]}, error=None)

    def status(self):
        return {"ready": True, "engine": "fake"}


@pytest.fixture(autouse=True)
def _fake_engine(monkeypatch):
    monkeypatch.setattr(PhonemeRouter, "_get_engine", lambda self: _FakeEngine())


class TestPhonemeEncodeDecode:
    """POST /phoneme/encode and /decode."""

    def test_encode_happy_path(self):
        resp = client.post("/phoneme/encode", json={"text": "hello"})
        assert resp.status_code == 200
        assert _data(resp)["ids"] == [1, 2, 3]

    def test_encode_empty_text_is_422(self):
        resp = client.post("/phoneme/encode", json={"text": ""})
        assert resp.status_code == 422

    def test_encode_missing_text_is_422(self):
        resp = client.post("/phoneme/encode", json={})
        assert resp.status_code == 422

    def test_decode_happy_path(self):
        resp = client.post("/phoneme/decode", json=[1, 2, 3])
        assert resp.status_code == 200
        assert _data(resp)["ids"] == [1, 2, 3]


class TestPhonemeAnalysis:
    """POST /phoneme/visualize, /score, /batch-encode, /detect-language, /synthesize."""

    def test_visualize(self):
        resp = client.post("/phoneme/visualize", json={"text": "hi"})
        assert resp.status_code == 200
        assert _data(resp)["text"] == "hi"

    def test_score(self):
        resp = client.post(
            "/phoneme/score",
            json={"target": "hello", "spoken": "hallo"},
        )
        assert resp.status_code == 200
        assert _data(resp)["score"] == 0.9

    def test_score_missing_fields_is_422(self):
        resp = client.post("/phoneme/score", json={"target": "x"})
        assert resp.status_code == 422

    def test_batch_encode(self):
        resp = client.post("/phoneme/batch-encode", json={"texts": ["a", "b"]})
        assert resp.status_code == 200
        assert len(_data(resp)["results"]) == 2

    def test_batch_encode_empty_list_is_422(self):
        resp = client.post("/phoneme/batch-encode", json={"texts": []})
        assert resp.status_code == 422

    def test_detect_language(self):
        resp = client.post("/phoneme/detect-language", json={"text": "bonjour"})
        assert resp.status_code == 200
        assert _data(resp)["language"] == "en"

    def test_synthesize(self):
        resp = client.post("/phoneme/synthesize", json={"text": "hi"})
        assert resp.status_code == 200
        assert _data(resp)["audio"] == "base64=="


class TestPhonemeReadModels:
    """GET /phoneme/languages and /phoneme/status."""

    def test_languages(self):
        resp = client.get("/phoneme/languages")
        assert resp.status_code == 200
        assert _data(resp)["languages"] == ["en"]

    def test_status(self):
        resp = client.get("/phoneme/status")
        assert resp.status_code == 200
        assert _data(resp)["ready"] is True

    def test_unknown_path_is_404(self):
        resp = client.get("/phoneme/nope")
        assert resp.status_code == 404

    def test_wrong_method_is_405(self):
        # /phoneme/languages exists as GET only
        resp = client.post("/phoneme/languages")
        assert resp.status_code == 405


class TestPhonemeAuth:
    """Auth edge: enforced only when SLO_AUTH_REQUIRED=true."""

    def test_auth_disabled_allows_anonymous(self, monkeypatch):
        monkeypatch.setenv("SLO_AUTH_REQUIRED", "false")
        resp = client.get("/phoneme/status")
        assert resp.status_code == 200

    def test_auth_enabled_rejects_missing_token(self, monkeypatch):
        monkeypatch.setenv("SLO_AUTH_REQUIRED", "true")
        resp = client.post("/phoneme/encode", json={"text": "hello"})
        assert resp.status_code == 401
