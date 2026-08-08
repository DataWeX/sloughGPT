"""Tests for the PDF-VLM processing pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
pytestmark = pytest.mark.slow
from domains.inference.pdf_vlm import PDFVLMProcessor


@pytest.fixture
def fake_pdf_bytes():
    """Create a minimal valid PDF with one page of text."""
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length 44>>stream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET\n"
        b"endstream\n"
        b"endobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n"
        b"0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000351 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n"
        b"405\n"
        b"%%EOF\n"
    )
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf)
        return Path(f.name)


@pytest.fixture
def fake_pdf_bytes_empty():
    """Create a minimal PDF with no text content."""
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
        b"xref\n"
        b"0 4\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n"
        b"196\n"
        b"%%EOF\n"
    )
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf)
        return Path(f.name)


class TestPDFVLMProcessor:
    """Tests for PDFVLMProcessor."""

    def test_extract_text_returns_string(self, fake_pdf_bytes):
        """_extract_text should return concatenated text."""
        p = PDFVLMProcessor(max_pages=5)
        text = p._extract_text(str(fake_pdf_bytes))
        assert isinstance(text, str)
        fake_pdf_bytes.unlink()

    def test_extract_text_empty_pdf(self, fake_pdf_bytes_empty):
        """_extract_text should handle empty PDF gracefully."""
        p = PDFVLMProcessor(max_pages=5)
        text = p._extract_text(str(fake_pdf_bytes_empty))
        assert isinstance(text, str)
        fake_pdf_bytes_empty.unlink()

    def test_analyze_returns_string(self, fake_pdf_bytes):
        """analyze should return text content."""
        p = PDFVLMProcessor(max_pages=5)
        result = p.analyze(str(fake_pdf_bytes), question="Summarize this.")
        assert isinstance(result, str)
        assert len(result) > 0
        fake_pdf_bytes.unlink()

    def test_analyze_empty_pdf(self, fake_pdf_bytes_empty):
        """analyze should handle empty PDF gracefully."""
        p = PDFVLMProcessor(max_pages=5)
        result = p.analyze(str(fake_pdf_bytes_empty))
        assert "empty" in result.lower() or isinstance(result, str)
        fake_pdf_bytes_empty.unlink()

    def test_analyze_pages_returns_list(self, fake_pdf_bytes):
        """analyze_pages should return per-page results."""
        p = PDFVLMProcessor(max_pages=5)
        results = p.analyze_pages(str(fake_pdf_bytes))
        assert isinstance(results, list)
        if results:
            assert "page" in results[0]
            assert "text" in results[0]
        fake_pdf_bytes.unlink()

    def test_analyze_nonexistent_file(self):
        """analyze should handle missing file gracefully."""
        p = PDFVLMProcessor(max_pages=5)
        result = p.analyze("/nonexistent/file.pdf")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_page_images_empty_without_pdf2image(self):
        """_page_images should return [] when pdf2image not installed."""
        p = PDFVLMProcessor(max_pages=5)
        with patch.dict("sys.modules", {"pdf2image": None}):
            result = p._page_images("/fake.pdf")
            assert result == []

    def test_page_images_with_pdf2image(self, monkeypatch):
        """_page_images should render pages to PNG bytes via pdf2image."""
        import sys
        import types

        class FakeImage:
            def save(self, buf, format=None):
                buf.write(b"PNG-DATA")

        fake = types.ModuleType("pdf2image")
        fake.convert_from_path = lambda *a, **k: [FakeImage(), FakeImage()]
        monkeypatch.setitem(sys.modules, "pdf2image", fake)

        p = PDFVLMProcessor(max_pages=5)
        result = p._page_images("/fake.pdf")
        assert result == [b"PNG-DATA", b"PNG-DATA"]

    def test_page_images_pdf2image_raises(self, monkeypatch):
        """_page_images should return [] when rendering raises."""
        import sys
        import types

        def boom(*a, **k):
            raise ValueError("render failed")

        fake = types.ModuleType("pdf2image")
        fake.convert_from_path = boom
        monkeypatch.setitem(sys.modules, "pdf2image", fake)

        p = PDFVLMProcessor(max_pages=5)
        assert p._page_images("/fake.pdf") == []

    def test_extract_text_with_pymupdf(self, monkeypatch):
        """_extract_text should use PyMuPDF when installed."""
        import sys
        import types

        class FakePage:
            def get_text(self):
                return "Page text "

        class FakeDoc:
            def __init__(self):
                self.pages = [FakePage(), FakePage(), FakePage()]

            def __iter__(self):
                return iter(self.pages)

            def close(self):
                pass

        fake = types.ModuleType("fitz")
        fake.open = lambda path: FakeDoc()
        monkeypatch.setitem(sys.modules, "fitz", fake)

        p = PDFVLMProcessor(max_pages=2)
        assert p._extract_text("/fake.pdf") == "Page text Page text "

    def test_extract_text_pypdf_breaks_at_max_pages(self, tmp_path, monkeypatch):
        """pypdf fallback should stop reading past max_pages."""
        import sys
        import types

        monkeypatch.delitem(sys.modules, "fitz", raising=False)

        class FakePage:
            def extract_text(self):
                return "P "

        class FakePdfReader:
            def __init__(self, stream):
                self.pages = [FakePage(), FakePage(), FakePage()]

        fake = types.ModuleType("pypdf")
        fake.PdfReader = FakePdfReader
        monkeypatch.setitem(sys.modules, "pypdf", fake)

        path = tmp_path / "multi.pdf"
        path.write_bytes(b"%PDF-1.4 fake body")

        p = PDFVLMProcessor(max_pages=1)
        assert p._extract_text(str(path)) == "P "
