"""Procedural gallery generators, extracted from ImagesRouter into domain.

One implementation, two callers (Surface Rule one-identity move, card
20261116_090332): the ``POST /images/generate`` endpoint and the chat
``image_gen`` tool both draw from ``domain.multimodal``'s generators, so the
journey image and the API image are byte-for-byte the same art.
"""

from __future__ import annotations

from domain.multimodal import IMAGE_STYLES, generate_image, hex_to_rgb, save_image

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class TestStyles:
    def test_five_styles_with_display_names(self) -> None:
        assert set(IMAGE_STYLES) == {"realistic", "cartoon", "watercolor", "sketch", "fantasy"}
        assert all(isinstance(v, str) and v for v in IMAGE_STYLES.values())

    def test_hex_to_rgb(self) -> None:
        assert hex_to_rgb("#ff0000") == (255, 0, 0)
        assert hex_to_rgb("ffffff") == (255, 255, 255)


class TestGenerateImage:
    def test_every_style_returns_png_bytes(self) -> None:
        for style in IMAGE_STYLES:
            data = generate_image("a blue sky", style)
            assert isinstance(data, bytes), style
            assert data[:8] == PNG_MAGIC, style
            assert len(data) > 1000, style

    def test_unknown_style_falls_back_to_gradient(self) -> None:
        # Parity with the old dispatcher: generators.get(style, gradient).
        data = generate_image("a blue sky", "no-such-style")
        assert data[:8] == PNG_MAGIC

    def test_gradient_prompt_keywords_change_the_art(self) -> None:
        # The realistic/gradient generator is deterministic — prompt keywords
        # pick the palette, so different keywords must give different bytes.
        forest = generate_image("green forest cabin", "realistic")
        sunset = generate_image("orange sunset", "realistic")
        assert forest != sunset


class TestSaveImage:
    def test_writes_generated_file_and_returns_gallery_path(self, tmp_path) -> None:
        png = generate_image("a blue sky", "realistic")
        rel = save_image(png, "realistic", gallery_dir=tmp_path)
        name = rel.rsplit("/", 1)[-1]
        assert rel == f"/data/gallery/{name}"
        assert name.startswith("generated_") and name.endswith(".png")
        assert (tmp_path / name).read_bytes() == png

    def test_filenames_are_unique(self, tmp_path) -> None:
        png = generate_image("x", "sketch")
        names = {save_image(png, "sketch", gallery_dir=tmp_path) for _ in range(3)}
        assert len(names) == 3
        for rel in names:
            pid = rel.rsplit("/", 1)[-1].removeprefix("generated_").removesuffix(".png")
            assert len(pid) == 8 and all(c in "0123456789abcdef" for c in pid), rel
