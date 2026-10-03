"""Vision tests — hashes, diffs, graceful missing-dep behavior."""

import io
import sys

import pytest
from avion.interact.primitives import BoundingBox
from avion.vision import (
    Capabilities,
    Capability,
    ImageAnalyzer,
    MatchMethod,
    VisualMatch,
)


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


class TestCapabilities:
    def test_shape_matches_the_contract(self):
        caps = ImageAnalyzer().capabilities()
        assert isinstance(caps, Capabilities)
        for c in caps.all:
            assert isinstance(c, Capability)
            assert isinstance(c.available, bool)
            assert isinstance(c.reason, str)
        d = caps.to_dict()
        assert set(d) >= {"ocr", "template", "reason"}

    def test_reason_agrees_with_flags(self):
        caps = ImageAnalyzer().capabilities()
        broken = [c for c in caps.all if not c.available]
        if not broken:
            assert caps.reason == ""
            assert sorted(caps.available) == ["ocr", "pixel", "template"]
        else:
            assert caps.reason != ""
            for c in broken:
                assert c.reason and c.name in caps.reason

    def test_truthiness_reflects_availability(self):
        caps = ImageAnalyzer().capabilities()
        assert bool(caps.pixel) is caps.pixel.available
        assert bool(Capability("x", False)) is False
        assert bool(Capability("x", True)) is True

    def test_detection_does_not_import_the_optional_deps(self):
        """capabilities() is a cheap probe, not an import side effect."""
        before = set(sys.modules)
        ImageAnalyzer().capabilities()
        for name in ("cv2", "pytesseract", "PIL"):
            if name not in before:
                assert name not in sys.modules

    def test_stubbed_out_module_counts_as_missing(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cv2", None)
        caps = ImageAnalyzer().capabilities()
        assert caps.template.available is False
        assert "cv2" in caps.template.reason

    def test_ocr_degraded_path_reports_why(self):
        caps = ImageAnalyzer().capabilities()
        if caps.ocr.available:
            pytest.skip("tesseract installed — degraded path unreachable here")
        assert "tesseract" in caps.ocr.reason
        assert ImageAnalyzer().ocr_text(b"nope") == []

    def test_template_degraded_path_reports_why(self):
        caps = ImageAnalyzer().capabilities()
        if caps.template.available:
            pytest.skip("opencv installed — degraded path unreachable here")
        assert "cv2" in caps.template.reason
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
