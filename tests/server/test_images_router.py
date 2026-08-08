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

    def test_styles_are_key_value_pairs(self, client):
        resp = client.get("/images/styles")
        styles = resp.json()["data"]["styles"]
        for key, label in styles:
            assert isinstance(key, str)
            assert isinstance(label, str)

    def test_expected_style_keys(self, client):
        resp = client.get("/images/styles")
        keys = [k for k, _ in resp.json()["data"]["styles"]]
        assert set(keys) == {"realistic", "cartoon", "watercolor", "sketch", "fantasy"}


class TestListGallery:
    """GET /images/gallery"""

    def test_returns_image_list(self, client):
        resp = client.get("/images/gallery")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "images" in data
        assert isinstance(data["images"], list)

    def test_empty_gallery(self, client):
        resp = client.get("/images/gallery")
        assert resp.status_code == 200
        assert isinstance(resp.json()["data"]["images"], list)

    def test_success_status(self, client):
        resp = client.get("/images/gallery")
        assert resp.json()["status"] == "success"


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

    @patch.object(ImagesRouter, "_generate_image", return_value=b"fake_png")
    @patch.object(ImagesRouter, "_save_image", return_value="/data/gallery/abc.png")
    def test_cartoon_style(self, mock_save, mock_gen, client):
        resp = client.post("/images/generate", json={
            "prompt": "a cat", "style": "cartoon",
        })
        assert resp.status_code == 200
        assert resp.json()["style"] == "cartoon"

    @patch.object(ImagesRouter, "_generate_image", return_value=b"fake_png")
    @patch.object(ImagesRouter, "_save_image", return_value="/data/gallery/abc.png")
    def test_watercolor_style(self, mock_save, mock_gen, client):
        resp = client.post("/images/generate", json={
            "prompt": "flowers", "style": "watercolor",
        })
        assert resp.status_code == 200
        assert resp.json()["style"] == "watercolor"

    @patch.object(ImagesRouter, "_generate_image", return_value=b"fake_png")
    @patch.object(ImagesRouter, "_save_image", return_value="/data/gallery/abc.png")
    def test_sketch_style(self, mock_save, mock_gen, client):
        resp = client.post("/images/generate", json={
            "prompt": "a house", "style": "sketch",
        })
        assert resp.status_code == 200
        assert resp.json()["style"] == "sketch"

    @patch.object(ImagesRouter, "_generate_image", return_value=b"fake_png")
    @patch.object(ImagesRouter, "_save_image", return_value="/data/gallery/abc.png")
    def test_fantasy_style(self, mock_save, mock_gen, client):
        resp = client.post("/images/generate", json={
            "prompt": "a dragon", "style": "fantasy",
        })
        assert resp.status_code == 200
        assert resp.json()["style"] == "fantasy"

    @patch.object(ImagesRouter, "_generate_image", return_value=b"fake_png")
    @patch.object(ImagesRouter, "_save_image", return_value="/data/gallery/abc.png")
    def test_default_style_is_realistic(self, mock_save, mock_gen, client):
        resp = client.post("/images/generate", json={"prompt": "anything"})
        assert resp.status_code == 200
        assert resp.json()["style"] == "realistic"

    @patch.object(ImagesRouter, "_generate_image", return_value=b"fake_png")
    @patch.object(ImagesRouter, "_save_image", return_value="/data/gallery/abc.png")
    def test_response_has_id_field(self, mock_save, mock_gen, client):
        resp = client.post("/images/generate", json={"prompt": "test"})
        assert "id" in resp.json()

    @patch.object(ImagesRouter, "_generate_image")
    @patch.object(ImagesRouter, "_save_image", return_value="/data/gallery/abc.png")
    def test_generation_failure_returns_500(self, mock_save, mock_gen, client):
        mock_gen.side_effect = RuntimeError("Pillow not installed")
        resp = client.post("/images/generate", json={"prompt": "test"})
        assert resp.status_code == 500

    @patch.object(ImagesRouter, "_generate_image", return_value=b"fake_png")
    @patch.object(ImagesRouter, "_save_image", return_value="/data/gallery/abc.png")
    def test_empty_prompt_accepted(self, mock_save, mock_gen, client):
        resp = client.post("/images/generate", json={"prompt": ""})
        assert resp.status_code == 200

    def test_empty_body_rejected(self, client):
        resp = client.post("/images/generate", json={})
        assert resp.status_code == 422

    @patch.object(ImagesRouter, "_generate_image", return_value=b"fake_png")
    @patch.object(ImagesRouter, "_save_image", return_value="/data/gallery/abc.png")
    def test_extra_fields_ignored(self, mock_save, mock_gen, client):
        resp = client.post("/images/generate", json={
            "prompt": "test", "style": "realistic", "unknown_field": "ignored",
        })
        assert resp.status_code == 200


class TestHexToRgb:
    """Module-level hex_to_rgb helper"""

    def test_converts_hex(self):
        from apps.api.server.routers.images import hex_to_rgb
        assert hex_to_rgb("#ff0000") == (255, 0, 0)

    def test_converts_without_hash(self):
        from apps.api.server.routers.images import hex_to_rgb
        assert hex_to_rgb("00ff00") == (0, 255, 0)
