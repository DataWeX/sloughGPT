"""Tests for images router endpoints."""
import pytest

from tests.test_support import get_test_client

client = get_test_client()


class TestListStyles:
    def test_list_styles(self):
        resp = client.get("/images/styles")
        assert resp.status_code == 200
        body = resp.json()
        styles = body["data"]["styles"]
        assert isinstance(styles, list)
        assert len(styles) >= 5

    def test_styles_contain_expected_names(self):
        resp = client.get("/images/styles")
        style_names = [s[0] for s in resp.json()["data"]["styles"]]
        assert "realistic" in style_names
        assert "cartoon" in style_names
        assert "watercolor" in style_names
        assert "sketch" in style_names
        assert "fantasy" in style_names


class TestListGallery:
    def test_gallery_empty(self):
        resp = client.get("/images/gallery")
        assert resp.status_code == 200
        body = resp.json()
        assert isinstance(body["data"]["images"], list)

    def test_gallery_has_images_field(self):
        resp = client.get("/images/gallery")
        assert isinstance(resp.json()["data"]["images"], list)


class TestGenerateImage:
    def test_generate_default_style(self):
        resp = client.post("/images/generate", json={"prompt": "a blue sky"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["image"].startswith("data:image/png;base64,")
        assert body["style"] == "realistic"
        assert body["prompt"] == "a blue sky"

    def test_generate_cartoon_style(self):
        resp = client.post("/images/generate", json={
            "prompt": "a cat",
            "style": "cartoon",
        })
        assert resp.status_code == 200
        assert resp.json()["style"] == "cartoon"

    def test_generate_watercolor(self):
        resp = client.post("/images/generate", json={
            "prompt": "mountain lake",
            "style": "watercolor",
        })
        assert resp.status_code == 200

    def test_generate_sketch(self):
        resp = client.post("/images/generate", json={
            "prompt": "portrait",
            "style": "sketch",
        })
        assert resp.status_code == 200

    def test_generate_fantasy(self):
        resp = client.post("/images/generate", json={
            "prompt": "magical forest",
            "style": "fantasy",
        })
        assert resp.status_code == 200

    def test_generate_has_id(self):
        resp = client.post("/images/generate", json={"prompt": "test"})
        assert "id" in resp.json()

    def test_generate_missing_prompt(self):
        resp = client.post("/images/generate", json={})
        assert resp.status_code == 422

    def test_generate_invalid_style(self):
        resp = client.post("/images/generate", json={
            "prompt": "test",
            "style": "nonexistent",
        })
        assert resp.status_code == 422

    def test_gallery_after_generation(self):
        client.post("/images/generate", json={"prompt": "gallery test"})
        resp = client.get("/images/gallery")
        body = resp.json()
        assert len(body["data"]["images"]) >= 1


class TestHexToRgb:
    def test_hex_conversion(self):
        from routers.images import hex_to_rgb
        assert hex_to_rgb("#ff0000") == (255, 0, 0)
        assert hex_to_rgb("#00ff00") == (0, 255, 0)
        assert hex_to_rgb("#0000ff") == (0, 0, 255)
        assert hex_to_rgb("ffffff") == (255, 255, 255)
