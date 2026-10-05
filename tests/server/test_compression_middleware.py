"""Tests for SelectiveGZipMiddleware (infra review 13, kanban 223001e4).

Pins the wire invariants: gzip when appropriate, identity otherwise, Vary on
every governed response, SSE never buffered, and ``gzip;q=0`` treated as a
refusal rather than an offer.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import StreamingResponse

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = REPO_ROOT / "apps/api/server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from infrastructure.compression import SelectiveGZipMiddleware, _accepts_gzip  # noqa: E402

BIG = "x" * 5000


def make_client(**middleware_kwargs) -> TestClient:
    app = FastAPI()

    @app.get("/json")
    def big_json():
        return {"data": BIG}

    @app.get("/small")
    def small():
        return {"ok": True}

    @app.get("/sse")
    async def sse():
        async def gen():
            for i in range(3):
                yield f"data: {i}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    app.add_middleware(SelectiveGZipMiddleware, **middleware_kwargs)
    return TestClient(app)


def _decode(response) -> bytes:
    raw = response.content
    if response.headers.get("content-encoding") == "gzip":
        try:
            raw = gzip.decompress(raw)
        except OSError:
            pass  # client already transparently decoded
    return raw


# ── Accept-Encoding parsing (the q=0 fix) ────────────────────────────


def test_accepts_gzip_parsing() -> None:
    assert _accepts_gzip("gzip")
    assert _accepts_gzip("br, gzip")
    assert _accepts_gzip("gzip;q=0.5")
    assert not _accepts_gzip("")  # absent = no support
    assert not _accepts_gzip("deflate, br")
    assert not _accepts_gzip("gzip;q=0")  # explicit refusal, not an offer
    assert not _accepts_gzip("gzip;q=0.0, br")


# ── Wire behavior ────────────────────────────────────────────────────


def test_json_is_gzipped_and_varies() -> None:
    client = make_client()
    resp = client.get("/json", headers={"accept-encoding": "gzip"})
    assert resp.headers["content-encoding"] == "gzip"
    assert "accept-encoding" in resp.headers["vary"].lower()
    assert json.loads(_decode(resp))["data"] == BIG
    # compressed must actually be smaller than identity
    assert int(resp.headers["content-length"]) < len(BIG)


def test_small_response_is_identity_but_still_varies() -> None:
    client = make_client()
    resp = client.get("/small", headers={"accept-encoding": "gzip"})
    assert "content-encoding" not in resp.headers
    assert "accept-encoding" in resp.headers["vary"].lower()
    assert resp.json() == {"ok": True}


def test_sse_is_never_buffered_or_compressed() -> None:
    client = make_client()
    resp = client.get("/sse", headers={"accept-encoding": "gzip"})
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert "content-encoding" not in resp.headers
    assert resp.text == "data: 0\n\ndata: 1\n\ndata: 2\n\n"


def test_q_zero_refusal_gets_identity() -> None:
    client = make_client()
    resp = client.get("/json", headers={"accept-encoding": "gzip;q=0"})
    assert "content-encoding" not in resp.headers
    assert json.loads(_decode(resp))["data"] == BIG


def test_no_accept_encoding_header_gets_identity() -> None:
    client = make_client()
    resp = client.get("/json", headers={"accept-encoding": "identity"})
    assert "content-encoding" not in resp.headers


def test_oversized_response_serves_identity() -> None:
    """Above the cap: identity by design (OOM avoidance), Vary still set."""
    client = make_client(maximum_size=1024)
    resp = client.get("/json", headers={"accept-encoding": "gzip"})
    assert "content-encoding" not in resp.headers
    assert "accept-encoding" in resp.headers["vary"].lower()
    assert json.loads(_decode(resp))["data"] == BIG


def test_incompressible_body_falls_back_to_identity() -> None:
    """If gzip would not shrink the wire, serve the original bytes."""
    import os

    noise = os.urandom(2048)
    client = make_client(minimum_size=10)
    app = client.app

    @app.get("/noise-json")
    def noise_json():
        from fastapi import Response

        return Response(content=noise, media_type="application/json")

    resp = client.get("/noise-json", headers={"accept-encoding": "gzip"})
    # Random bytes do not gzip smaller — the size guard must serve identity:
    # wire bytes are never worse than identity (the invariant from review 13).
    assert "content-encoding" not in resp.headers
    assert _decode(resp) == noise
