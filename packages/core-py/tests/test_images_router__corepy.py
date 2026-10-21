"""Tests for the images API router (routers/images.py).

Covers: generate_image, list_styles, list_gallery, edge cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.images import ImagesRouter  # noqa: E402


@pytest.fixture(autouse=True)
def _tmp_asset_store(tmp_path, monkeypatch):
    """Point the router's asset store at tmp_path so tests never write the real
    data/assets CAS or adopt the real gallery into it."""
    from routers import images as images_mod

    from domain.infrastructure.asset_store import AssetStore

    monkeypatch.setattr(images_mod, "get_asset_store", lambda: AssetStore(tmp_path / "assets"))


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
        # 2ca971c55 added prompt length validation (min_length=1): an empty
        # prompt is now a request-validation error, same as the missing-key
        # case below. The old echo-back contract predates the validator.
        assert resp.status_code == 422

    def test_generate_long_prompt(self):
        ir = ImagesRouter()
        client = TestClient(_app(ir))
        long_prompt = "a beautiful landscape " * 100
        resp = client.post("/images/generate", json={"prompt": long_prompt})
        # 2100 chars > max_length=2000 (2ca971c55) -> validation error; the
        # echo-back contract predates the validator.
        assert resp.status_code == 422

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


class TestGetGalleryImage:
    """GET /images/gallery/{filename} — the <img> src contract (J6)."""

    def test_serves_png_bytes(self):
        ir = ImagesRouter()
        client = TestClient(_app(ir))
        gallery_dir = Path(__file__).resolve().parents[3] / "data" / "gallery"
        gallery_dir.mkdir(parents=True, exist_ok=True)
        probe = gallery_dir / "generated_ztestprobe.png"
        probe.write_bytes(b"\x89PNG\r\n\x1a\nprobe-bytes")
        try:
            resp = client.get("/images/gallery/generated_ztestprobe.png")
            assert resp.status_code == 200
            assert resp.headers["content-type"].startswith("image/png")
            assert resp.content.startswith(b"\x89PNG")
        finally:
            probe.unlink(missing_ok=True)

    def test_missing_image_is_404(self):
        ir = ImagesRouter()
        client = TestClient(_app(ir))
        resp = client.get("/images/gallery/generated_zdoesnotexist.png")
        assert resp.status_code == 404

    def test_rejects_path_shaped_names(self):
        ir = ImagesRouter()
        client = TestClient(_app(ir))
        # Layered defense: URL layers normalize `..` away (404) and the
        # handler's bare-name guard rejects anything that still reaches it
        # (400). Either way no traversal ever serves bytes.
        for url in (
            "/images/gallery/..%2F..%2Fetc%2Fpasswd",
            "/images/gallery/%2E%2E%2F%2E%2E%2Fetc%2Fpasswd",
            "/images/gallery/%2e%2e%2fsecret.png",
        ):
            resp = client.get(url)
            assert resp.status_code in (400, 404), (url, resp.status_code)
            assert b"root:" not in resp.content

    def test_rejects_non_generated_names(self):
        ir = ImagesRouter()
        client = TestClient(_app(ir))
        resp = client.get("/images/gallery/secrets.png")
        assert resp.status_code == 400

    def test_generate_then_gallery_path_resolves(self):
        """J6 round trip: generate -> the returned id serves bytes from the URL."""
        ir = ImagesRouter()
        client = TestClient(_app(ir))
        gen = client.post("/images/generate", json={"prompt": "a lighthouse", "style": "sketch"})
        assert gen.status_code == 200
        image_id = gen.json()["id"]
        listing = client.get("/images/gallery").json()["data"]["images"]
        ours = [i for i in listing if i["id"] == image_id]
        assert ours, f"{image_id} missing from gallery listing"
        served = client.get(ours[0]["path"])
        assert served.status_code == 200
        assert served.content.startswith(b"\x89PNG")
