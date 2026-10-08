"""Tests for the phoneme API router (routers/phoneme.py).

Covers: encode, decode, visualize, score, batch_encode, detect_language, synthesize, languages, status.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_server_dir = str(Path(__file__).resolve().parents[2] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _mock_engine():
    engine = MagicMock()
    engine.encode.return_value = MagicMock(success=True, data={"ids": [1, 2, 3]})
    engine.decode.return_value = MagicMock(success=True, data={"text": "hello"})
    engine.visualize.return_value = MagicMock(
        success=True, data={"phonemes": ["h", "eh", "l", "ow"]}
    )
    engine.score.return_value = MagicMock(success=True, data={"score": 0.85, "details": {}})
    engine.batch_encode.return_value = MagicMock(
        success=True, data={"results": [{"ids": [1]}, {"ids": [2]}]}
    )
    engine.detect_language.return_value = MagicMock(
        success=True, data={"language": "en", "confidence": 0.95}
    )
    engine.synthesize.return_value = MagicMock(success=True, data={"audio": "base64data"})
    engine.supported_languages.return_value = MagicMock(
        success=True, data={"languages": ["en", "es", "fr"]}
    )
    engine.status.return_value = {"loaded": True, "model": "test"}
    return engine


def _app(engine=None):
    from routers.phoneme import PhonemeRouter

    router_inst = PhonemeRouter()
    if engine is not None:
        router_inst._engine = engine

    app = FastAPI()
    app.include_router(router_inst.router)
    from infrastructure.exception_handlers import register_all_handlers

    register_all_handlers(app)
    return app


class TestPhonemeEncode:
    def test_encode(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post("/phoneme/encode", json={"text": "hello"})
        assert resp.status_code == 200
        assert resp.json()["data"]["ids"] == [1, 2, 3]

    def test_encode_with_language(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post("/phoneme/encode", json={"text": "hola", "language": "es"})
        assert resp.status_code == 200

    def test_encode_empty_text(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post("/phoneme/encode", json={"text": ""})
        assert resp.status_code == 422


class TestPhonemeDecode:
    def test_decode(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post("/phoneme/decode", json=[1, 2, 3])
        assert resp.status_code == 200
        assert resp.json()["data"]["text"] == "hello"

    def test_decode_error(self):
        engine = _mock_engine()
        engine.decode.return_value = MagicMock(success=False, error="bad ids")
        client = TestClient(_app(engine=engine))
        resp = client.post("/phoneme/decode", json=[999])
        assert resp.status_code == 500


class TestPhonemeVisualize:
    def test_visualize(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post("/phoneme/visualize", json={"text": "hello"})
        assert resp.status_code == 200
        assert "phonemes" in resp.json()["data"]


class TestPhonemeScore:
    def test_score(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post(
            "/phoneme/score",
            json={"target": "hello", "spoken": "hello"},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["score"] == 0.85


class TestPhonemeBatchEncode:
    def test_batch_encode(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post(
            "/phoneme/batch-encode",
            json={"texts": ["hello", "world"]},
        )
        assert resp.status_code == 200
        assert len(resp.json()["data"]["results"]) == 2

    def test_batch_encode_empty(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post("/phoneme/batch-encode", json={"texts": []})
        assert resp.status_code == 422


class TestPhonemeDetectLanguage:
    def test_detect_language(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post("/phoneme/detect-language", json={"text": "hello"})
        assert resp.status_code == 200
        assert resp.json()["data"]["language"] == "en"


class TestPhonemeSynthesize:
    def test_synthesize(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.post("/phoneme/synthesize", json={"text": "hello world"})
        assert resp.status_code == 200
        assert "audio" in resp.json()["data"]

    def test_synthesize_error(self):
        engine = _mock_engine()
        engine.synthesize.return_value = MagicMock(success=False, error="no voice")
        client = TestClient(_app(engine=engine))
        resp = client.post("/phoneme/synthesize", json={"text": "fail"})
        assert resp.status_code == 500


class TestPhonemeLanguages:
    def test_languages(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.get("/phoneme/languages")
        assert resp.status_code == 200
        assert "en" in resp.json()["data"]["languages"]


class TestPhonemeStatus:
    def test_status(self):
        client = TestClient(_app(engine=_mock_engine()))
        resp = client.get("/phoneme/status")
        assert resp.status_code == 200
        assert resp.json()["data"]["loaded"] is True
