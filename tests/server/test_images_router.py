"""
Tests for the images router — generate, gallery, styles.
"""

import pytest
from unittest.mock import MagicMock, patch
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

    def test_converts_uppercase(self):
        from apps.api.server.routers.images import hex_to_rgb
        assert hex_to_rgb("#AABBCC") == (170, 187, 204)

    def test_invalid_length_raises(self):
        from apps.api.server.routers.images import hex_to_rgb
        with pytest.raises(ValueError):
            hex_to_rgb("#ff00")

    def test_static_method_equivalent(self):
        from apps.api.server.routers.images import ImagesRouter
        assert ImagesRouter.hex_to_rgb("#123456") == (18, 52, 86)


class _FakeGalleryPath:
    """Chainable Path stand-in for list_gallery."""

    def __init__(self, files=None, exists=True):
        self.files = files if files is not None else []
        self._exists = exists

    def resolve(self):
        return self

    @property
    def parents(self):
        return [self] * 10

    def __truediv__(self, other):
        return self

    def exists(self):
        return self._exists

    def glob(self, pattern):
        return self.files


def _fake_file(stem, mtime):
    f = MagicMock()
    f.stem = stem
    f.name = f"{stem}.png"
    f.stat.return_value.st_mtime = mtime
    return f


class TestListGalleryWithFiles:
    """GET /images/gallery with a populated gallery dir."""

    @patch("apps.api.server.routers.images.Path")
    def test_lists_images_sorted_newest_first(self, mock_path, client):
        mock_path.return_value = _FakeGalleryPath(
            files=[_fake_file("generated_aaa", 100), _fake_file("generated_bbb", 300), _fake_file("generated_ccc", 200)],
        )
        resp = client.get("/images/gallery")
        assert resp.status_code == 200
        images = resp.json()["data"]["images"]
        assert [i["id"] for i in images] == ["generated_bbb", "generated_ccc", "generated_aaa"]

    @patch("apps.api.server.routers.images.Path")
    def test_image_entries_have_expected_fields(self, mock_path, client):
        mock_path.return_value = _FakeGalleryPath(files=[_fake_file("generated_abc", 123)])
        resp = client.get("/images/gallery")
        entry = resp.json()["data"]["images"][0]
        assert entry["id"] == "generated_abc"
        assert entry["path"] == "/data/gallery/generated_abc.png"
        assert entry["created"] == 123

    @patch("apps.api.server.routers.images.Path")
    def test_caps_at_fifty_images(self, mock_path, client):
        files = [_fake_file(f"generated_{i:04d}", i) for i in range(60)]
        mock_path.return_value = _FakeGalleryPath(files=files)
        resp = client.get("/images/gallery")
        assert len(resp.json()["data"]["images"]) == 50

    @patch("apps.api.server.routers.images.Path")
    def test_missing_dir_returns_empty(self, mock_path, client):
        mock_path.return_value = _FakeGalleryPath(exists=False)
        resp = client.get("/images/gallery")
        assert resp.status_code == 200
        assert resp.json()["data"]["images"] == []


class TestMethodMismatch:
    """Wrong HTTP methods on images routes."""

    def test_generate_get_405(self, client):
        resp = client.get("/images/generate")
        assert resp.status_code == 405

    def test_styles_post_405(self, client):
        resp = client.post("/images/styles")
        assert resp.status_code == 405

    def test_gallery_post_405(self, client):
        resp = client.post("/images/gallery")
        assert resp.status_code == 405


class TestGradientGenerator:
    """_generate_gradient_image keyword palette branches (real Pillow)."""

    def _gen(self, prompt):
        img = ImagesRouter()
        return img._generate_gradient_image(prompt, width=24, height=24)

    def _first_color_pixel(self, png_bytes):
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        return img.getpixel((15, 0))

    def test_returns_valid_png(self):
        data = self._gen("hello world")
        assert data.startswith(b"\x89PNG")

    def test_nature_keywords_green_palette(self):
        assert self._first_color_pixel(self._gen("a cabin in the forest")) == (45, 90, 39)

    def test_sunset_keywords_orange_palette(self):
        assert self._first_color_pixel(self._gen("a sunset sky")) == (211, 84, 0)

    def test_ocean_keywords_blue_palette(self):
        assert self._first_color_pixel(self._gen("the ocean water")) == (30, 55, 153)

    def test_mountain_keywords_light_palette(self):
        assert self._first_color_pixel(self._gen("bright mountain")) == (58, 134, 255)

    def test_fire_keywords_red_palette(self):
        assert self._first_color_pixel(self._gen("fire burning heat")) == (181, 23, 0)

    def test_default_palette(self):
        assert self._first_color_pixel(self._gen("something unknown")) == (106, 17, 203)


class TestStyleGenerators:
    """Each procedural style generator produces a valid PNG."""

    @pytest.fixture(scope="class")
    def img(self):
        return ImagesRouter()

    def _valid_png(self, data):
        assert data.startswith(b"\x89PNG")

    def test_cartoon(self, img):
        self._valid_png(img._generate_cartoon_image("a cat", width=16, height=16))

    def test_watercolor(self, img):
        self._valid_png(img._generate_watercolor_image("flowers", width=16, height=16))

    def test_sketch(self, img):
        self._valid_png(img._generate_sketch_image("house", width=16, height=16))

    def test_fantasy(self, img):
        self._valid_png(img._generate_fantasy_image("dragon", width=16, height=16))


class TestDispatch:
    """_generate_image style dispatch."""

    def test_known_style_uses_its_generator(self):
        img = ImagesRouter()
        with patch.object(img, "_generate_cartoon_image", return_value=b"cartoon_bytes") as mock_c:
            out = img._generate_image("x", "cartoon")
            assert out == b"cartoon_bytes"
            mock_c.assert_called_once_with("x")

    def test_unknown_style_falls_back_to_gradient(self):
        img = ImagesRouter()
        with patch.object(img, "_generate_gradient_image", return_value=b"gradient_bytes") as mock_g:
            out = img._generate_image("x", "bogus_style")
            assert out == b"gradient_bytes"
            mock_g.assert_called_once_with("x")


class TestGenerateEdgeCases:
    """Additional generate_image endpoint behavior."""

    @patch.object(ImagesRouter, "_generate_image", return_value=b"fake_png_bytes")
    @patch.object(ImagesRouter, "_save_image", return_value="/data/gallery/abc.png")
    def test_id_derived_from_saved_path(self, mock_save, mock_gen, client):
        resp = client.post("/images/generate", json={"prompt": "test"})
        assert resp.json()["id"] == "abc"

    @patch.object(ImagesRouter, "_generate_image", return_value=b"fake_png_bytes")
    @patch.object(ImagesRouter, "_save_image", side_effect=OSError("disk full"))
    def test_save_failure_returns_500(self, mock_save, mock_gen, client):
        resp = client.post("/images/generate", json={"prompt": "test"})
        assert resp.status_code == 500
