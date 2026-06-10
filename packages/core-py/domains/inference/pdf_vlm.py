"""
pdf_vlm — PDF ingestion and analysis via VLM.

Extracts text and renders page images from PDFs using PyMuPDF,
then feeds both into the VLM inference engine for visual + textual analysis.

Usage:
    processor = PDFVLMProcessor("models/vlm-finetuned")
    result = processor.analyze("path/to/doc.pdf", question="Summarize this document")

    # Per-page
    pages = processor.analyze_pages("path/to/doc.pdf")
"""

from __future__ import annotations

import logging
import io
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image

logger = logging.getLogger("man.pdf_vlm")


class PDFPage:
    """Represents a single PDF page with its text and rendered image."""

    def __init__(self, page_num: int, text: str, image: Image.Image):
        self.page_num = page_num
        self.text = text
        self.image = image


class PDFVLMProcessor:
    """Process PDFs through the VLM engine.

    Uses PyMuPDF to extract text and render page images,
    then feeds them into ``VLMInference`` for analysis.
    """

    def __init__(
        self,
        model_dir: str = "models/vlm-finetuned",
        max_pages: int = 5,
        device: Optional[str] = None,
    ):
        self.max_pages = max_pages
        self.model_dir = model_dir
        self.device = device

        from domains.inference.vlm_inference import VLMInference
        self.vlm = VLMInference(model_dir=model_dir, device=device)

    # ── PDF Processing ────────────────────────────────────────────

    def extract_pages(self, pdf_path: str) -> list[PDFPage]:
        """Extract text and render images from a PDF.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of ``PDFPage`` objects (text + PIL Image per page).
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(str(path))
        pages: list[PDFPage] = []

        for i, page_num in enumerate(range(len(doc))):
            if i >= self.max_pages:
                break

            page = doc[page_num]
            text = page.get_text().strip()

            # Render page to PIL Image at 150 DPI (RGB)
            mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")

            pages.append(PDFPage(page_num + 1, text, img))

        doc.close()
        logger.info("Extracted %d pages from %s", len(pages), pdf_path)
        return pages

    def extract_text(self, pdf_path: str) -> str:
        """Extract all text from a PDF (no rendering)."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(str(path))
        texts: list[str] = []
        for page in doc:
            t = page.get_text().strip()
            if t:
                texts.append(t)
        doc.close()
        return "\n\n".join(texts)

    # ── VLM Analysis ──────────────────────────────────────────────

    def _build_multi_page_prompt(self, pages: list[PDFPage], question: str) -> str:
        """Build a text prompt that includes extracted text from all pages."""
        context_parts = []
        for p in pages:
            header = f"--- Page {p.page_num} ---"
            text_content = p.text if p.text else "(no text on this page)"
            context_parts.append(f"{header}\n{text_content}")

        context = "\n\n".join(context_parts)
        prompt = (
            f"I am showing you a PDF document. Here is the extracted text:\n\n"
            f"{context}\n\n"
            f"Based on the document above and the page images shown to you, "
            f"{question}"
        )
        return prompt

    def analyze(
        self,
        pdf_path: str,
        question: str = "Summarize this document.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Analyze a PDF using VLM (first page image + all extracted text).

        Args:
            pdf_path: Path to the PDF file.
            question: Question or instruction about the document.
            max_new_tokens: Maximum tokens for the response.
            temperature: Sampling temperature.

        Returns:
            VLM-generated analysis text.
        """
        pages = self.extract_pages(pdf_path)
        if not pages:
            return "The PDF appears to be empty or could not be read."

        # Use the first page as the visual image
        first_page_image = pages[0].image

        # Build prompt with all extracted text + question
        prompt = self._build_multi_page_prompt(pages, question)

        return self.vlm.generate(
            first_page_image,
            text=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def analyze_pages(
        self,
        pdf_path: str,
        question: str = "Describe what is shown on this page.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> list[dict]:
        """Analyze each page of a PDF individually.

        Args:
            pdf_path: Path to the PDF file.
            question: Question per page.
            max_new_tokens: Max tokens per page response.
            temperature: Sampling temperature.

        Returns:
            List of dicts with keys: page_num, text, response.
        """
        pages = self.extract_pages(pdf_path)
        results: list[dict] = []

        for p in pages:
            page_text = p.text if p.text else "(no text)"
            prompt = (
                f"This is page {p.page_num} of a document. "
                f"Extracted text:\n{page_text}\n\n"
                f"{question}"
            )
            response = self.vlm.generate(
                p.image,
                text=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            results.append({
                "page_num": p.page_num,
                "text_snippet": page_text[:500],
                "response": response,
            })

        return results

    def summarize(
        self,
        pdf_path: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Quick summary of a document."""
        return self.analyze(
            pdf_path,
            question="provide a concise summary of this document, covering the main topics and key points.",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
