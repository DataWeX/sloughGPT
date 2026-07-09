"""File Management Router - upload, list, search, delete files with metadata."""

import io
import json
import logging
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel

from schemas.common import success_response

logger = logging.getLogger("man.routers.files")
router = APIRouter(prefix="/files", tags=["files"])

UPLOADS_DIR = Path(__file__).resolve().parents[4] / "data" / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

METADATA_FILE = UPLOADS_DIR / "_metadata.json"

SUPPORTED_EXTENSIONS = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/typescript",
    ".html": "text/html",
    ".css": "text/css",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".xml": "text/xml",
    ".log": "text/plain",
}


class FileMetadata(BaseModel):
    id: str
    filename: str
    extension: str
    size_bytes: int
    chars: int
    pages: int
    uploaded_at: float
    tags: list[str] = []


class FileItem(BaseModel):
    id: str
    filename: str
    extension: str
    size_bytes: int
    uploaded_at: float
    tags: list[str] = []


class FileDetail(BaseModel):
    id: str
    filename: str
    extension: str
    size_bytes: int
    chars: int
    pages: int
    uploaded_at: float
    tags: list[str]
    text: str


class UploadResponse(BaseModel):
    id: str
    filename: str
    chars: int
    pages: int
    size_bytes: int


class FileListResponse(BaseModel):
    files: list[FileItem]
    total: int


class IngestResponse(BaseModel):
    id: str
    filename: str
    chars: int
    facts_stored: int


# ── Metadata persistence ──


def _load_metadata() -> dict[str, dict]:
    if METADATA_FILE.exists():
        try:
            return json.loads(METADATA_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_metadata(meta: dict[str, dict]) -> None:
    METADATA_FILE.write_text(json.dumps(meta, indent=2))


def _file_id(filename: str) -> str:
    return f"{int(time.time())}_{filename}"


# ── Endpoints ──


@router.get("", response_model=FileListResponse)
async def list_files(
    sort: str = Query("uploaded_at", description="Sort field"),
    order: str = Query("desc", description="asc or desc"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
):
    """List all uploaded files with metadata."""
    meta = _load_metadata()
    items = []
    for fid, m in meta.items():
        file_path = UPLOADS_DIR / m["filename"]
        if not file_path.exists():
            continue
        if tag and tag not in m.get("tags", []):
            continue
        items.append(FileItem(
            id=fid,
            filename=m["filename"],
            extension=m.get("extension", ""),
            size_bytes=m.get("size_bytes", 0),
            uploaded_at=m.get("uploaded_at", 0.0),
            tags=m.get("tags", []),
        ))
    reverse = order.lower() != "asc"
    items.sort(key=lambda x: getattr(x, sort, 0), reverse=reverse)
    return FileListResponse(files=items, total=len(items))


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    tags: str = Form("[]"),
):
    """Upload a file and save it to the server."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if not ext:
        raise HTTPException(status_code=400, detail="File must have an extension")

    contents = await file.read()
    fid = _file_id(file.filename)
    file_path = UPLOADS_DIR / f"{fid}{ext}"

    with open(file_path, "wb") as f:
        f.write(contents)

    try:
        tag_list = json.loads(tags) if isinstance(tags, str) else (tags or [])
    except (json.JSONDecodeError, TypeError):
        tag_list = []

    meta = _load_metadata()
    meta[fid] = {
        "filename": f"{fid}{ext}",
        "original_name": file.filename,
        "extension": ext,
        "size_bytes": len(contents),
        "chars": 0,
        "pages": 1,
        "uploaded_at": time.time(),
        "tags": tag_list,
    }
    _save_metadata(meta)

    return UploadResponse(
        id=fid,
        filename=file.filename,
        chars=0,
        pages=1,
        size_bytes=len(contents),
    )


@router.get("/{file_id}", response_model=FileDetail)
async def get_file(file_id: str):
    """Get file metadata and extracted text."""
    meta = _load_metadata()
    m = meta.get(file_id)
    if not m:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = UPLOADS_DIR / m["filename"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    content = file_path.read_bytes()
    text, pages = _extract_text(content, m.get("extension", ""))
    chars = len(text)

    # Update cached metadata
    m["chars"] = chars
    m["pages"] = pages
    _save_metadata(meta)

    return FileDetail(
        id=file_id,
        filename=m.get("original_name", m["filename"]),
        extension=m.get("extension", ""),
        size_bytes=m.get("size_bytes", 0),
        chars=chars,
        pages=pages,
        uploaded_at=m.get("uploaded_at", 0.0),
        tags=m.get("tags", []),
        text=text,
    )


@router.delete("/{file_id}")
async def delete_file(file_id: str):
    """Delete a file and its metadata."""
    meta = _load_metadata()
    m = meta.pop(file_id, None)
    if not m:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = UPLOADS_DIR / m["filename"]
    if file_path.exists():
        file_path.unlink()
    _save_metadata(meta)
    return success_response(data={"status": "deleted", "file_id": file_id})


@router.get("/search", response_model=FileListResponse)
async def search_files(
    q: str = Query(..., min_length=1, description="Search query"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
):
    """Search files by name."""
    query = q.lower()
    meta = _load_metadata()
    items = []
    for fid, m in meta.items():
        file_path = UPLOADS_DIR / m["filename"]
        if not file_path.exists():
            continue
        name = m.get("original_name", m["filename"]).lower()
        if query not in name:
            continue
        if tag and tag not in m.get("tags", []):
            continue
        items.append(FileItem(
            id=fid,
            filename=m.get("original_name", m["filename"]),
            extension=m.get("extension", ""),
            size_bytes=m.get("size_bytes", 0),
            uploaded_at=m.get("uploaded_at", 0.0),
            tags=m.get("tags", []),
        ))
    items.sort(key=lambda x: x.uploaded_at, reverse=True)
    return FileListResponse(files=items, total=len(items))


@router.post("/{file_id}/ingest", response_model=IngestResponse)
async def ingest_file(file_id: str):
    """Extract text from a file and store it in the knowledge base."""
    meta = _load_metadata()
    m = meta.get(file_id)
    if not m:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = UPLOADS_DIR / m["filename"]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on disk")

    content = file_path.read_bytes()
    text, pages = _extract_text(content, m.get("extension", ""))
    if not text.strip():
        return IngestResponse(id=file_id, filename=m.get("original_name", m["filename"]), chars=0, facts_stored=0)

    try:
        from domains.learner.knowledge import get_knowledge_memory, KnowledgeFact

        mem = get_knowledge_memory()
        stored = 0
        chunks = _chunk_text(text, max_chars=500)
        for chunk in chunks:
            if len(chunk) > 20:
                fact = KnowledgeFact(
                    content=chunk.strip(),
                    topic=m.get("original_name", m["filename"]),
                    source="upload",
                    timestamp=time.time(),
                    importance=0.6,
                )
                if mem.add_fact(fact):
                    stored += 1
        return IngestResponse(
            id=file_id,
            filename=m.get("original_name", m["filename"]),
            chars=len(text),
            facts_stored=stored,
        )
    except ImportError:
        raise HTTPException(status_code=501, detail="Knowledge base not available")


# ── Helpers ──


def _extract_text(content: bytes, ext: str) -> tuple[str, int]:
    """Extract text from file content based on extension."""
    ext = ext.lower()
    try:
        if ext == ".pdf":
            import fitz
            doc = fitz.open(stream=content, filetype="pdf")
            pages = len(doc)
            texts = [page.get_text() for page in doc]
            doc.close()
            return "\n\n".join(texts), pages
        elif ext == ".docx":
            import docx
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs)
            return text, 1
        else:
            return content.decode("utf-8", errors="replace"), 1
    except ImportError as e:
        logger.warning("Text extraction import failed: %s", e)
        return content.decode("utf-8", errors="replace"), 1


def _chunk_text(text: str, max_chars: int = 500) -> list[str]:
    """Split text into chunks at sentence boundaries."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    for s in sentences:
        if len(current) + len(s) > max_chars and current:
            chunks.append(current)
            current = s
        else:
            current = (current + " " + s).strip()
    if current:
        chunks.append(current)
    return chunks or [text]
