"""Tests for the PDF-VLM processing pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

@pytest.fixture
def fake_pdf_bytes():
    """Create a minimal valid PDF with one page of text."""
    # Minimal PDF that renders one page with "Hello World"
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
    """Tests for PDFVLMProcessor (with mock VLM)."""

    pytestmark = pytest.mark.slow

    def test_extract_pages_returns_pages(self, fake_pdf_bytes):
        """extract_pages should return one PDFPage per page."""
        from domains.inference.pdf_vlm import PDFVLMProcessor

        with patch.object(PDFVLMProcessor, "__init__", return_value=None):
            p = PDFVLMProcessor()
            p.max_pages = 5

            pages = p.extract_pages(str(fake_pdf_bytes))
            assert len(pages) >= 1
            assert pages[0].page_num == 1
            assert pages[0].image is not None

        fake_pdf_bytes.unlink()

    def test_extract_pages_nonexistent(self):
        """extract_pages should raise on missing file."""
        from domains.inference.pdf_vlm import PDFVLMProcessor

        p = PDFVLMProcessor.__new__(PDFVLMProcessor)
        with pytest.raises(FileNotFoundError):
            p.extract_pages("/nonexistent/file.pdf")

    def test_extract_text_returns_string(self, fake_pdf_bytes):
        """extract_text should return concatenated text."""
        from domains.inference.pdf_vlm import PDFVLMProcessor

        with patch.object(PDFVLMProcessor, "__init__", return_value=None):
            p = PDFVLMProcessor()
            text = p.extract_text(str(fake_pdf_bytes))
            assert isinstance(text, str)

        fake_pdf_bytes.unlink()

    @patch("domains.inference.pdf_vlm.PDFVLMProcessor.extract_pages")
    def test_analyze_returns_string(self, mock_extract):
        """analyze should return VLM-generated text."""
        from PIL import Image
        from domains.inference.pdf_vlm import PDFVLMProcessor, PDFPage

        mock_page = PDFPage(page_num=1, text="test content", image=Image.new("RGB", (224, 224)))
        mock_extract.return_value = [mock_page]

        p = PDFVLMProcessor.__new__(PDFVLMProcessor)
        p.max_pages = 5

        with patch.object(p, "vlm") as mock_vlm:
            mock_vlm.generate.return_value = "This is a test summary."

            result = p.analyze("/fake.pdf", question="Summarize this.")
            assert result == "This is a test summary."
            mock_vlm.generate.assert_called_once()

    @patch("domains.inference.pdf_vlm.PDFVLMProcessor.extract_pages")
    def test_analyze_empty_pdf(self, mock_extract):
        """analyze should handle empty PDF gracefully."""
        from domains.inference.pdf_vlm import PDFVLMProcessor

        mock_extract.return_value = []
        p = PDFVLMProcessor.__new__(PDFVLMProcessor)

        result = p.analyze("/empty.pdf")
        assert "empty" in result.lower()

    @patch("domains.inference.pdf_vlm.PDFVLMProcessor.extract_pages")
    def test_analyze_pages_returns_list(self, mock_extract):
        """analyze_pages should return per-page results."""
        from PIL import Image
        from domains.inference.pdf_vlm import PDFVLMProcessor, PDFPage

        mock_pages = [
            PDFPage(page_num=1, text="page one", image=Image.new("RGB", (224, 224))),
            PDFPage(page_num=2, text="page two", image=Image.new("RGB", (224, 224))),
        ]
        mock_extract.return_value = mock_pages

        p = PDFVLMProcessor.__new__(PDFVLMProcessor)
        p.max_pages = 5

        with patch.object(p, "vlm") as mock_vlm:
            mock_vlm.generate.side_effect = ["Response 1", "Response 2"]
            results = p.analyze_pages("/fake.pdf")

            assert len(results) == 2
            assert results[0]["page_num"] == 1
            assert results[0]["response"] == "Response 1"
            assert results[1]["page_num"] == 2
            assert results[1]["response"] == "Response 2"
            assert mock_vlm.generate.call_count == 2

    def test_summarize_calls_analyze(self):
        """summarize should delegate to analyze."""
        from domains.inference.pdf_vlm import PDFVLMProcessor

        p = PDFVLMProcessor.__new__(PDFVLMProcessor)
        with patch.object(p, "analyze", return_value="summary text") as mock_analyze:
            result = p.summarize("/fake.pdf")
            assert result == "summary text"
            mock_analyze.assert_called_once()

    def test_pdf_page_dataclass(self):
        """PDFPage should store page_num, text, image."""
        from PIL import Image
        from domains.inference.pdf_vlm import PDFPage

        img = Image.new("RGB", (100, 100))
        page = PDFPage(page_num=3, text="hello", image=img)
        assert page.page_num == 3
        assert page.text == "hello"
        assert page.image is img



