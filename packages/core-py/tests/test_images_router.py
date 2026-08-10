"""Tests for the images API router (routers/images.py).

Covers: generate_image, list_styles, list_gallery.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.images import ImagesRouter  # noqa: E402


def _app(ir: ImagesRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(ir.router)
    return app


class TestListStyles:
    def test_styles(self):
        ir = ImagesRouter()
        client = TestClient(_app(ir))
        resp = client.get("/images/styles")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["styles"][0][0] == "realistic"
        assert data["styles"][1][0] == "cartoon"
        assert len(data["styles"]) == 5


class TestListGallery:
    def test_gallery_returns_list(self):
        ir = ImagesRouter()
        client = TestClient(_app(ir))
        resp = client.get("/images/gallery")
        assert resp.status_code == 200
        assert "images" in resp.json()["data"]
        assert isinstance(resp.json()["data"]["images"], list)


class TestGenerateImage:
    def test_generate_default_style(self):
        ir = ImagesRouter()
        client = TestClient(_app(ir))
        resp = client.post("/images/generate", json={"prompt": "a cat"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["style"] == "realistic"
        assert data["prompt"] == "a cat"
        assert data["image"].startswith("data:image")

    def test_generate_custom_style(self):
        ir = ImagesRouter()
        client = TestClient(_app(ir))
        resp = client.post("/images/generate", json={"prompt": "a dog", "style": "cartoon"})
        assert resp.status_code == 200
        assert resp.json()["style"] == "cartoon"
