"""File Management Router - upload, list, search, delete files with metadata."""

import io
import json
import logging
import time
from pathlib import Path
from typing import Optional
import re

from fastapi import APIRouter, UploadFile, File, Form, Query
from pydantic import BaseModel

from schemas.common import raise_error, success_response

logger = logging.getLogger("slo.routers.files")


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


class FilesRouter:
    """OOP router for file management endpoints."""

    def __init__(self):
        self.router = APIRouter(prefix="/files", tags=["files"])
        self.UPLOADS_DIR = Path(__file__).resolve().parents[4] / "data" / "uploads"
        self.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        self.METADATA_FILE = self.UPLOADS_DIR / "_metadata.json"
        self.SUPPORTED_EXTENSIONS = {
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
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(
            "", self.list_files, methods=["GET"], response_model=FileListResponse
        )
        self.router.add_api_route(
            "/upload", self.upload_file, methods=["POST"], response_model=UploadResponse
        )
        self.router.add_api_route(
            "/search", self.search_files, methods=["GET"], response_model=FileListResponse
        )
        self.router.add_api_route(
            "/{file_id}", self.get_file, methods=["GET"], response_model=FileDetail
        )
        self.router.add_api_route(
            "/{file_id}", self.delete_file, methods=["DELETE"]
        )
        self.router.add_api_route(
            "/{file_id}/ingest", self.ingest_file, methods=["POST"], response_model=IngestResponse
        )

    # ── Metadata persistence ──

    def _load_metadata(self) -> dict[str, dict]:
        if self.METADATA_FILE.exists():
            try:
                return json.loads(self.METADATA_FILE.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_metadata(self, meta: dict[str, dict]) -> None:
        self.METADATA_FILE.write_text(json.dumps(meta, indent=2))

    def _file_id(self, filename: str) -> str:
        return f"{int(time.time())}_{filename}"

    # ── Endpoints ──

    async def list_files(
        self,
        sort: str = Query("uploaded_at", description="Sort field"),
        order: str = Query("desc", description="asc or desc"),
        tag: Optional[str] = Query(None, description="Filter by tag"),
    ) -> dict:
        """List all uploaded files with metadata."""
        meta = self._load_metadata()
        items = []
        for fid, m in meta.items():
            file_path = self.UPLOADS_DIR / m["filename"]
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

    async def upload_file(
        self,
        file: UploadFile = File(...),
        tags: str = Form("[]"),
    ) -> dict:
        """Upload a file and save it to the server."""
        if not file.filename:
            raise_error("No filename provided", "E_BAD_REQUEST", status_code=400)

        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if not ext:
            raise_error("File must have an extension", "E_BAD_REQUEST", status_code=400)

        contents = await file.read()
        fid = self._file_id(file.filename)
        file_path = self.UPLOADS_DIR / f"{fid}{ext}"

        with open(file_path, "wb") as f:
            f.write(contents)

        try:
            tag_list = json.loads(tags) if isinstance(tags, str) else (tags or [])
        except (json.JSONDecodeError, TypeError):
            tag_list = []

        meta = self._load_metadata()
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
        self._save_metadata(meta)

        return UploadResponse(
            id=fid,
            filename=file.filename,
            chars=0,
            pages=1,
            size_bytes=len(contents),
        )

    async def search_files(
        self,
        q: str = Query(..., min_length=1, description="Search query"),
        tag: Optional[str] = Query(None, description="Filter by tag"),
    ) -> dict:
        """Search uploaded files by name substring match.

        Performs case-insensitive substring matching against the original
        filename of all uploaded files. Optionally filters by tag.

        Args:
            q: Search query to match against file names (min 1 character).
            tag: Optional tag to filter results by.

        Returns:
            FileListResponse with matching files sorted by upload time descending.
        """
        query = q.lower()
        meta = self._load_metadata()
        items = []
        for fid, m in meta.items():
            file_path = self.UPLOADS_DIR / m["filename"]
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

    async def get_file(self, file_id: str) -> dict:
        """Get file metadata and extracted text."""
        meta = self._load_metadata()
        m = meta.get(file_id)
        if not m:
            raise_error("File not found", "E_NOT_FOUND", status_code=404)

        file_path = self.UPLOADS_DIR / m["filename"]
        if not file_path.exists():
            raise_error("File not found on disk", "E_NOT_FOUND", status_code=404)

        content = file_path.read_bytes()
        text, pages = self._extract_text(content, m.get("extension", ""))
        chars = len(text)

        m["chars"] = chars
        m["pages"] = pages
        self._save_metadata(meta)

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

    async def delete_file(self, file_id: str) -> dict:
        """Delete a file and its metadata."""
        meta = self._load_metadata()
        m = meta.pop(file_id, None)
        if not m:
            raise_error("File not found", "E_NOT_FOUND", status_code=404)

        file_path = self.UPLOADS_DIR / m["filename"]
        if file_path.exists():
            file_path.unlink()
        self._save_metadata(meta)
        return success_response(data={"status": "deleted", "file_id": file_id})

    async def ingest_file(self, file_id: str) -> dict:
        """Extract text from a file and store it in the knowledge base."""
        meta = self._load_metadata()
        m = meta.get(file_id)
        if not m:
            raise_error("File not found", "E_NOT_FOUND", status_code=404)

        file_path = self.UPLOADS_DIR / m["filename"]
        if not file_path.exists():
            raise_error("File not found on disk", "E_NOT_FOUND", status_code=404)

        content = file_path.read_bytes()
        text, pages = self._extract_text(content, m.get("extension", ""))
        if not text.strip():
            return IngestResponse(id=file_id, filename=m.get("original_name", m["filename"]), chars=0, facts_stored=0)

        try:
            from domains.learner.knowledge import get_knowledge_memory, KnowledgeFact

            mem = get_knowledge_memory()
            stored = 0
            chunks = self._chunk_text(text, max_chars=500)
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
            raise_error("Knowledge base not available", "E_INFRA_STARTUP", status_code=501)

    # ── Helpers ──

    def _extract_text(self, content: bytes, ext: str) -> tuple[str, int]:
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
            logger.warning("Text extraction import failed: %s", e, extra={"tag": "INFRA"})
            return content.decode("utf-8", errors="replace"), 1

    def _chunk_text(self, text: str, max_chars: int = 500) -> list[str]:
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


# ── Module-level singleton + re-exports for backward compatibility ──

_files_instance = FilesRouter()
router = _files_instance.router
UPLOADS_DIR = _files_instance.UPLOADS_DIR
METADATA_FILE = _files_instance.METADATA_FILE


def _load_metadata():
    return _files_instance._load_metadata()


def _save_metadata(meta):
    return _files_instance._save_metadata(meta)
