"""File Reading Router - extract text from PDF/DOCX/TXT files."""

import io
import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("man.routers.files")
router = APIRouter(prefix="/files", tags=["files"])


SUPPORTED_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
}


class ExtractResponse(BaseModel):
    text: str
    filename: str
    pages: int
    chars: int
    extension: str


@router.post("/extract", response_model=ExtractResponse)
async def extract_file(
    file: UploadFile = File(...),
):
    """Extract text from an uploaded file.

    Supports PDF (.pdf), Word (.docx), text files (.txt, .md, .csv, .json).

    Returns extracted text, page count (or 1 for non-PDF), character count,
    and original filename metadata.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Get extension
    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS.keys()))}",
        )

    contents = await file.read()
    text = ""
    pages = 1

    try:
        if ext == ".pdf":
            text, pages = _extract_pdf(contents)
        elif ext == ".docx":
            text = _extract_docx(contents)
        else:
            # Plain text files
            text = contents.decode("utf-8", errors="replace")
    except Exception as e:
        logger.error(f"File extraction failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")

    if not text.strip():
        text = f"[File '{file.filename}' appears to be empty or contains no extractable text.]"

    return ExtractResponse(
        text=text.strip(),
        filename=file.filename,
        pages=pages,
        chars=len(text),
        extension=ext,
    )


def _extract_pdf(content: bytes) -> tuple:
    """Extract text from a PDF file using PyMuPDF."""
    import fitz  # PyMuPDF
    doc = fitz.open(stream=content, filetype="pdf")
    pages = len(doc)
    texts = []
    for page in doc:
        texts.append(page.get_text())
    doc.close()
    return "\n\n".join(texts), pages


def _extract_docx(content: bytes) -> str:
    """Extract text from a Word document."""
    import docx
    doc = docx.Document(io.BytesIO(content))
    return "\n".join(p.text for p in doc.paragraphs)
