"""
Tests for the images router — generate, gallery, styles.
"""

import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.images import ImagesRouter, router


@pytest.fixture
def app():
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestListStyles:
    """GET /images/styles"""

    def test_returns_all_styles(self, client):
        resp = client.get("/images/styles")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "styles" in data
        assert len(data["styles"]) == 5


class TestListGallery:
    """GET /images/gallery"""

    def test_returns_image_list(self, client):
        resp = client.get("/images/gallery")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "images" in data
        assert isinstance(data["images"], list)


class TestGenerateImage:
    """POST /images/generate"""

    @patch.object(ImagesRouter, "_generate_image", return_value=b"fake_png_bytes")
    @patch.object(ImagesRouter, "_save_image", return_value="/data/gallery/test.png")
    def test_generates_image(self, mock_save, mock_gen, client):
        resp = client.post("/images/generate", json={
            "prompt": "a sunset",
            "style": "realistic",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["style"] == "realistic"
        assert body["prompt"] == "a sunset"
        assert body["image"].startswith("data:image/png;base64,")

    def test_validates_style_enum(self, client):
        resp = client.post("/images/generate", json={
            "prompt": "test",
            "style": "invalid_style",
        })
        assert resp.status_code == 422

    def test_requires_prompt(self, client):
        resp = client.post("/images/generate", json={"style": "realistic"})
        assert resp.status_code == 422
