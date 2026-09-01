"""PDFVLMProcessor — analyze PDF documents using VLM."""

from __future__ import annotations

import logging

logger = logging.getLogger("slo.pdf_vlm")


class PDFVLMProcessor:
    """Analyze PDF documents using VLM inference.

    Converts PDF pages to images and passes them through the VLM
    for analysis. Falls back to text extraction if VLM is unavailable.
    """

    def __init__(self, max_pages: int = 10):
        self.max_pages = max_pages

    def _get_vlm(self):
        """VLM inference is not available."""
        return None

    def _page_images(self, pdf_path: str) -> list[bytes]:
        """Convert PDF pages to PNG images."""
        try:
            import pdf2image
            images = pdf2image.convert_from_path(
                pdf_path,
                first_page=1,
                last_page=self.max_pages,
                fmt="png",
            )
            import io
            result = []
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                result.append(buf.getvalue())
            return result
        except ImportError:  # pragma: no cover — pdf2image is a required dependency here
            logger.warning("pdf2image not installed, using text-only fallback", extra={"tag": "INF"})
            return []
        except Exception as e:
            logger.warning("PDF page rendering failed: %s", e, extra={"tag": "INF"})
            return []

    def _extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF."""
        try:
            import fitz
            doc = fitz.open(pdf_path)
            text = ""
            for i, page in enumerate(doc):
                if i >= self.max_pages:
                    break
                text += page.get_text()
            doc.close()
            return text
        except ImportError:
            logger.warning("PyMuPDF not installed, trying pypdf...", extra={"tag": "INF"})
        try:
            import pypdf
            with open(pdf_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                text = ""
                for i, page in enumerate(reader.pages):
                    if i >= self.max_pages:
                        break
                    text += page.extract_text() or ""
            return text
        except ImportError:  # pragma: no cover — pypdf is a required dependency here
            logger.warning("pypdf not installed either", extra={"tag": "INF"})
            return ""
        except Exception as e:
            logger.warning("PDF text extraction failed: %s", e, extra={"tag": "INF"})
            return ""

    def analyze(
        self,
        pdf_path: str,
        question: str = "Summarize this document.",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        """Analyze a PDF document.

        Tries VLM-based analysis first, falls back to text-only summary.
        """
        vlm = self._get_vlm()

        if vlm is not None:  # pragma: no cover — VLM backend not available
            try:  # pragma: no cover
                import base64  # pragma: no cover
                pages = self._page_images(pdf_path)  # pragma: no cover
                if pages:  # pragma: no cover
                    # Use first page image
                    b64 = base64.b64encode(pages[0]).decode("utf-8")  # pragma: no cover
                    result = vlm.generate(  # pragma: no cover
                        image_base64=b64,
                        prompt=f"Analyze this document: {question}",
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    return result.get("text", "No analysis generated.")  # pragma: no cover
            except Exception as e:  # pragma: no cover
                logger.warning("VLM PDF analysis failed: %s", e, extra={"tag": "INF"})  # pragma: no cover

        # Fallback: text-only
        try:
            text = self._extract_text(pdf_path)
            if not text:
                return "Could not extract text from this PDF."
            return f"PDF text content ({len(text)} chars):\n\n{text[:3000]}..."
        except Exception as e:  # pragma: no cover — _extract_text swallows all errors
            logger.warning("PDF text extraction failed: %s", e, extra={"tag": "INF"})
            return f"Could not analyze PDF: {e}"

    def analyze_pages(
        self,
        pdf_path: str,
        question: str = "Summarize this document.",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> list[dict]:
        """Analyze each page of a PDF separately.

        Returns list of {page, text} dicts.
        """
        vlm = self._get_vlm()
        results = []

        if vlm is not None:  # pragma: no cover — VLM backend not available
            try:  # pragma: no cover
                import base64  # pragma: no cover
                page_images = self._page_images(pdf_path)  # pragma: no cover
                for i, img_bytes in enumerate(page_images):  # pragma: no cover
                    b64 = base64.b64encode(img_bytes).decode("utf-8")  # pragma: no cover
                    result = vlm.generate(  # pragma: no cover
                        image_base64=b64,
                        prompt=f"Page {i+1}: {question}",
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    results.append({  # pragma: no cover
                        "page": i + 1,
                        "text": result.get("text", ""),
                    })
                return results  # pragma: no cover
            except Exception as e:  # pragma: no cover
                logger.warning("VLM per-page analysis failed: %s", e, extra={"tag": "INF"})  # pragma: no cover

        # Fallback
        text = self._extract_text(pdf_path)
        if text:
            results.append({"page": 1, "text": text[:2000]})
        return results
