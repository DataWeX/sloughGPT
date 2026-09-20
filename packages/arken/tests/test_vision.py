"""Vision tests — hashes, diffs, graceful missing-dep behavior."""

import io

import pytest
from arken.interact.primitives import BoundingBox
from arken.vision import ImageAnalyzer, MatchMethod, VisualMatch


class TestHash:
    def test_stable(self):
        a = ImageAnalyzer()
        assert a.compute_hash(b"abc") == a.compute_hash(b"abc")

    def test_differs(self):
        a = ImageAnalyzer()
        assert a.compute_hash(b"abc") != a.compute_hash(b"abd")


class TestDiff:
    def test_identical_is_zero(self):
        assert ImageAnalyzer().pixel_diff_bytes(b"x", b"x") == 0.0

    def test_different_is_nonzero(self):
        assert ImageAnalyzer().pixel_diff_bytes(b"x", b"y") > 0.0

    def test_real_images_compare(self):
        Image = pytest.importorskip("PIL.Image", reason="pillow not installed")

        def png(color):
            buf = io.BytesIO()
            Image.new("RGB", (8, 8), color).save(buf, format="PNG")
            return buf.getvalue()

        a = ImageAnalyzer()
        assert a.pixel_diff_bytes(png("red"), png("red")) == 0.0
        assert a.pixel_diff_bytes(png("red"), png("blue")) > 0.9


class TestGraceful:
    def test_ocr_missing_dep_empty(self):
        import sys

        if "pytesseract" in sys.modules:
            pytest.skip("pytesseract installed")
        try:
            __import__("pytesseract")
            pytest.skip("pytesseract installed")
        except ImportError:
            assert ImageAnalyzer().ocr_text(b"nope") == []

    def test_template_missing_dep_empty(self):
        try:
            __import__("cv2")
            pytest.skip("opencv installed")
        except ImportError:
            assert ImageAnalyzer().find_template(b"a", b"b") == []


class TestMatch:
    def test_to_dict(self):
        m = VisualMatch(
            bbox=BoundingBox(x=1, y=2, width=3, height=4),
            confidence=0.9,
            method=MatchMethod.TEXT_REGION,
            label="Start",
        )
        d = m.to_dict()
        assert d["method"] == "text_region"
        assert d["bbox"] == {"x": 1, "y": 2, "width": 3, "height": 4}
