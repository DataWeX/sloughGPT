"""Tests for the images API router (routers/images.py).

Covers: generate_image, list_styles, list_gallery, edge cases.
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
    from infrastructure.exception_handlers import register_all_handlers
    register_all_handlers(app)
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

    def test_styles_have_descriptions(self):
        ir = ImagesRouter()
        client = TestClient(_app(ir))
        resp = client.get("/images/styles")
        styles = resp.json()["data"]["styles"]
        for name, desc in styles:
            assert isinstance(name, str)
            assert isinstance(desc, str)
            assert len(name) > 0
            assert len(desc) > 0

    def test_styles_unique_names(self):
        ir = ImagesRouter()
        client = TestClient(_app(ir))
        resp = client.get("/images/styles")
        names = [s[0] for s in resp.json()["data"]["styles"]]
        assert len(names) == len(set(names))


class TestListGallery:
    def test_gallery_returns_list(self):
        ir = ImagesRouter()
        client = TestClient(_app(ir))
        resp = client.get("/images/gallery")
        assert resp.status_code == 200
        assert "images" in resp.json()["data"]
        assert isinstance(resp.json()["data"]["images"], list)

    def test_gallery_images_have_required_fields(self):
        ir = ImagesRouter()
        client = TestClient(_app(ir))
        resp = client.get("/images/gallery")
        for img in resp.json()["data"]["images"]:
            assert "id" in img
            assert "path" in img
            assert "created" in img


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

    def test_generate_empty_prompt(self):
        ir = ImagesRouter()
        client = TestClient(_app(ir))
        resp = client.post("/images/generate", json={"prompt": ""})
        assert resp.status_code == 200
        assert resp.json()["prompt"] == ""

    def test_generate_long_prompt(self):
        ir = ImagesRouter()
        client = TestClient(_app(ir))
        long_prompt = "a beautiful landscape " * 100
        resp = client.post("/images/generate", json={"prompt": long_prompt})
        assert resp.status_code == 200
        assert resp.json()["prompt"] == long_prompt

    def test_generate_returns_data_uri(self):
        ir = ImagesRouter()
        client = TestClient(_app(ir))
        resp = client.post("/images/generate", json={"prompt": "test"})
        img = resp.json()["image"]
        assert img.startswith("data:image/png;base64,")

    def test_generate_all_styles(self):
        ir = ImagesRouter()
        client = TestClient(_app(ir))
        styles_resp = client.get("/images/styles")
        style_names = [s[0] for s in styles_resp.json()["data"]["styles"]]
        for style in style_names:
            resp = client.post("/images/generate", json={"prompt": "test", "style": style})
            assert resp.status_code == 200
            assert resp.json()["style"] == style

    def test_generate_missing_prompt(self):
        ir = ImagesRouter()
        client = TestClient(_app(ir))
        resp = client.post("/images/generate", json={})
        assert resp.status_code == 422  # validation error
