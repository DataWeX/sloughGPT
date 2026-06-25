"""PDFVLMProcessor — analyze PDF documents using VLM."""

import logging
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger("man.pdf_vlm")


class PDFVLMProcessor:
    """Analyze PDF documents using VLM inference.

    Converts PDF pages to images and passes them through the VLM
    for analysis. Falls back to text extraction if VLM is unavailable.
    """

    def __init__(self, max_pages: int = 10):
        self.max_pages = max_pages

    def _get_vlm(self):
        """Try to get loaded VLM inference engine."""
        try:
            from apps.api.server.routers.visual import _vlm_inference
            return _vlm_inference
        except ImportError:
            return None
        except Exception:
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
        except ImportError:
            logger.warning("pdf2image not installed, using text-only fallback")
            return []
        except Exception as e:
            logger.warning("PDF page rendering failed: %s", e)
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
            logger.warning("PyMuPDF not installed, trying pypdf...")
        try:
            import PyPDF2
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for i, page in enumerate(reader.pages):
                    if i >= self.max_pages:
                        break
                    text += page.extract_text()
            return text
        except ImportError:
            logger.warning("PyPDF2 not installed either")
            return ""
        except Exception as e:
            logger.warning("PDF text extraction failed: %s", e)
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

        if vlm is not None:
            try:
                import base64
                pages = self._page_images(pdf_path)
                if pages:
                    # Use first page image
                    b64 = base64.b64encode(pages[0]).decode("utf-8")
                    result = vlm.generate(
                        image_base64=b64,
                        prompt=f"Analyze this document: {question}",
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    return result.get("text", "No analysis generated.")
            except Exception as e:
                logger.warning("VLM PDF analysis failed: %s", e)

        # Fallback: text-only
        text = self._extract_text(pdf_path)
        if not text:
            return "Could not extract text from this PDF."

        return f"PDF text content ({len(text)} chars):\n\n{text[:3000]}..."

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

        if vlm is not None:
            try:
                import base64
                page_images = self._page_images(pdf_path)
                for i, img_bytes in enumerate(page_images):
                    b64 = base64.b64encode(img_bytes).decode("utf-8")
                    result = vlm.generate(
                        image_base64=b64,
                        prompt=f"Page {i+1}: {question}",
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                    results.append({
                        "page": i + 1,
                        "text": result.get("text", ""),
                    })
                return results
            except Exception as e:
                logger.warning("VLM per-page analysis failed: %s", e)

        # Fallback
        text = self._extract_text(pdf_path)
        if text:
            results.append({"page": 1, "text": text[:2000]})
        return results
