"""Router-level tests for the phoneme routes (parity gate: w/o tests: 2 → 1).

Nine routes, previously untested — absent coverage is what let
GET /phoneme/languages 500 for months (`encoder.supported_languages` is a
list attribute, not a method; fixed in e56d383fe). The final test in this
file pins THAT defect chain at the encoder boundary so it cannot regress:
real PhonemeEngine + a stub encoder whose ``supported_languages`` is a list.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.phoneme import PhonemeRouter
from domain.voice._internal.phoneme import PhonemeEngine, PhonemeResult


class FakeEngine:
    """Stands in for PhonemeEngine at the ROUTER boundary (result envelope)."""

    def __init__(self, ok: bool = True):
        self._ok = ok

    def _res(self, data=None, error=None) -> PhonemeResult:
        if self._ok:
            return PhonemeResult(success=True, data=data)
        return PhonemeResult(success=False, error=error or "boom")

    def encode(self, text, language=None):
        return self._res({"ids": [1, 2, 3]})

    def decode(self, ids):
        return self._res({"text": "hello"})

    def visualize(self, text, language=None):
        return self._res({"visual": "h@l0"})

    def score(self, target, spoken, language=None):
        return self._res({"score": 0.9})

    def batch_encode(self, texts, language=None):
        return self._res({"batch": [{"ids": [1]} for _ in texts]})

    def detect_language(self, text):
        return self._res({"language": "en"})

    def synthesize(self, text):
        return self._res({"audio": "data:audio/wav;base64,AAAA"})

    def supported_languages(self):
        return self._res(["en", "es"])

    def status(self):
        return {"ok": True}


def _client(engine) -> TestClient:
    pho = PhonemeRouter()
    pho._engine = engine
    app = FastAPI()
    from apps.api.server.infrastructure.exception_handlers import register_all_handlers

    register_all_handlers(app)
    app.include_router(pho.router)
    return TestClient(app)


class TestPhonemeRoutes:
    def test_encode(self):
        r = _client(FakeEngine()).post("/phoneme/encode", json={"text": "hello"})
        assert r.status_code == 200
        assert r.json()["data"]["ids"] == [1, 2, 3]

    def test_decode(self):
        r = _client(FakeEngine()).post("/phoneme/decode", json=[1, 2, 3])
        assert r.status_code == 200
        assert r.json()["data"]["text"] == "hello"

    def test_visualize(self):
        r = _client(FakeEngine()).post("/phoneme/visualize", json={"text": "hello"})
        assert r.status_code == 200
        assert r.json()["data"]["visual"] == "h@l0"

    def test_score(self):
        r = _client(FakeEngine()).post(
            "/phoneme/score", json={"target": "helloc", "spoken": "hello"}
        )
        assert r.status_code == 200
        assert r.json()["data"]["score"] == 0.9

    def test_batch_encode(self):
        r = _client(FakeEngine()).post("/phoneme/batch-encode", json={"texts": ["a", "b"]})
        assert r.status_code == 200
        assert len(r.json()["data"]["batch"]) == 2

    def test_detect_language(self):
        r = _client(FakeEngine()).post("/phoneme/detect-language", json={"text": "hola"})
        assert r.status_code == 200
        assert r.json()["data"]["language"] == "en"

    def test_synthesize(self):
        r = _client(FakeEngine()).post("/phoneme/synthesize", json={"text": "hello"})
        assert r.status_code == 200
        assert r.json()["data"]["audio"].startswith("data:audio/wav")

    def test_languages(self):
        r = _client(FakeEngine()).get("/phoneme/languages")
        assert r.status_code == 200
        assert r.json()["data"] == ["en", "es"]

    def test_status(self):
        r = _client(FakeEngine()).get("/phoneme/status")
        assert r.status_code == 200
        assert r.json()["data"] == {"ok": True}


class TestPhonemeErrorMapping:
    def test_engine_failure_maps_to_500_with_route_code(self):
        r = _client(FakeEngine(ok=False)).post("/phoneme/encode", json={"text": "x"})
        assert r.status_code == 500
        body = r.json()
        assert body["code"] == "E_ENCODE_FAILED"
        assert body["error"] == "boom"

    def test_languages_failure_maps_to_500_with_route_code(self):
        r = _client(FakeEngine(ok=False)).get("/phoneme/languages")
        assert r.status_code == 500
        assert r.json()["code"] == "E_LANGUAGES_FAILED"


class TestLanguagesRegression:
    """GET /phoneme/languages 500 (card 3bd79e78 BUG 1, fixed e56d383fe).

    The real encoder exposes ``supported_languages`` as a LIST ATTRIBUTE.
    The service must read it without call parens all the way from the route
    down to the encoder — this test drives the actual route through the real
    PhonemeEngine with only the encoder stubbed, so a reintroduced `()`
    anywhere in that chain turns this into a 500 again.
    """

    def test_route_200_with_encoder_list_attribute(self):
        engine = PhonemeEngine()
        stub = type("StubEncoder", (), {})()
        stub.supported_languages = ["en", "es"]  # list attribute, not callable
        engine._encoder = stub

        r = _client(engine).get("/phoneme/languages")
        assert r.status_code == 200
        assert r.json()["data"] == ["en", "es"]
