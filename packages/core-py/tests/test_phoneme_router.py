"""Tests for the phoneme API router (routers/phoneme.py).

Covers: encode, decode, visualize, score, batch_encode, detect_language, synthesize, languages, status.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
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


class TestPhonemeLanguagesRealEngine:
    """The same routes, driven by the real engine instead of a mock.

     Every test above injects `_mock_engine()`, and a bare `MagicMock` accepts
     any attribute access and any call. That is how `GET /phoneme/languages`
     shipped returning 500 in production while its test stayed green: the
     handler called `encoder.supported_languages()`, which is a `@property`
     returning `list[str]`, and the mock's happy little `()` returned another
     mock instead of raising `'list' object is not callable`.

     The mock also encoded the wrong *shape* — it returned
     `data={"languages": [...]}` where the real engine returns the list itself
    — so the old assertion was checking a payload no client has ever received.

     These tests therefore assert two things at once: the status code, and that
     `data` is a plain list of language codes. Both are properties of the real
     object, which is the only thing that was ever in doubt.
    """

    def test_languages_returns_200_with_the_real_engine(self):
        client = TestClient(_app())  # no engine injected -> the real one
        resp = client.get("/phoneme/languages")
        assert resp.status_code == 200, resp.text

    def test_languages_data_is_a_list_of_codes(self):
        client = TestClient(_app())
        data = client.get("/phoneme/languages").json()["data"]
        assert isinstance(data, list), (
            f"expected the bare language list, got {type(data).__name__} — the "
            f"mock's {data!r} shape is not what the engine returns"
        )
        assert "en" in data

    def test_status_still_works_on_the_real_engine(self):
        client = TestClient(_app())
        resp = client.get("/phoneme/status")
        assert resp.status_code == 200, resp.text


class TestPhonemeSynthesizeWav:
    """`/phoneme/synthesize` must return a JSON-ready payload, not an ndarray.

    The voice engine hands back a raw ``np.ndarray`` waveform. Returning it
    untouched made this one route 500 with ``E_SERIALIZATION`` — the FastAPI
    response model cannot encode an ndarray — while the other eight phoneme
    routes returned 200. The mock above hid it: ``engine.synthesize.return_value
    = MagicMock(data={"audio": ...})`` already returns a dict, so the test never
    exercised the real ndarray path. These tests drive the engine method with a
    stubbed voice that returns a genuine numpy waveform — the shape that
    actually broke — and assert the result serialises.
    """

    def test_wav_data_uri_encodes_a_playable_wav(self):
        import base64
        import io
        import wave

        import numpy as np

        from domain.voice._internal.phoneme import _SAMPLE_RATE, _wav_data_uri

        waveform = np.zeros(_SAMPLE_RATE, dtype=np.float32)  # 1s of silence
        uri = _wav_data_uri(waveform)
        assert uri.startswith("data:audio/wav;base64,"), uri
        raw = base64.b64decode(uri.split(",", 1)[1])
        with wave.open(io.BytesIO(raw), "rb") as wf:
            assert wf.getframerate() == _SAMPLE_RATE
            assert wf.getnchannels() == 1
            assert wf.getnframes() == _SAMPLE_RATE

    def test_synthesize_returns_json_ready_payload(self):
        import json

        import numpy as np

        from domain.voice._internal.phoneme import _SAMPLE_RATE, PhonemeEngine

        engine = PhonemeEngine()
        waveform = np.linspace(-0.5, 0.5, _SAMPLE_RATE // 2, dtype=np.float32)
        fake_voice = MagicMock()
        fake_voice.synthesize.return_value = MagicMock(data=waveform)
        engine._voice = fake_voice  # _get_voice returns the cached _voice

        result = engine.synthesize("hello world")
        assert result.success, result.error
        # The exact failure this guards: the payload must survive JSON encoding.
        json.dumps(result.data)  # would raise TypeError on a raw ndarray
        assert result.data["audio"].startswith("data:audio/wav;base64,")
        assert result.data["text"] == "hello world"
        assert abs(result.data["duration_sec"] - 0.5) < 1e-6
