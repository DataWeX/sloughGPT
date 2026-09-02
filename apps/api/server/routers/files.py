"""File Management Router - upload, list, search, delete files with metadata.

Uses MogDB as the storage engine with automatic JSON sync.
File metadata is stored in MogDB and synced to JSON for human readability.
"""

import asyncio
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional
import re

from fastapi import APIRouter, UploadFile, File, Form, Query, Depends
from pydantic import BaseModel

from infrastructure.auth import require_auth_if_enabled
from schemas.common import raise_error, success_response, classify_and_raise, safe_audit_log

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


def _get_db():
    from mogdb import MogDB
    repo_root = Path(__file__).resolve().parents[4]
    db_path = os.path.join(repo_root, "data", "uploads_mogdb")
    sync_path = os.path.join(repo_root, "data", "uploads_json")
    return MogDB(db_path, sync_dir=sync_path)


class FilesRouter:
    """OOP router for file management endpoints."""

    def __init__(self):
        self.router = APIRouter(prefix="/files", tags=["files"])
        self.UPLOADS_DIR = Path(__file__).resolve().parents[4] / "data" / "uploads"
        self.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
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

    # ── Metadata persistence via MogDB ──

    def _load_metadata(self) -> dict[str, dict]:
        """Load file metadata from MogDB."""
        try:
            db = _get_db()
            col = db.collection("files")
            docs = col.find()
            meta = {}
            for doc in docs:
                fid = doc.get("file_id", "")
                if fid:
                    meta[fid] = {
                        "filename": doc.get("filename", ""),
                        "original_name": doc.get("original_name", ""),
                        "extension": doc.get("extension", ""),
                        "size_bytes": doc.get("size_bytes", 0),
                        "chars": doc.get("chars", 0),
                        "pages": doc.get("pages", 1),
                        "uploaded_at": doc.get("uploaded_at", 0.0),
                        "tags": doc.get("tags", []),
                    }
            return meta
        except Exception as e:
            logger.warning("Failed to load metadata from MogDB: %s", e)
            return {}

    def _save_metadata(self, meta: dict[str, dict]) -> None:
        """Save file metadata to MogDB (replaces all entries)."""
        try:
            db = _get_db()
            col = db.collection("files")
            # Clear and rewrite
            col.delete_many({})
            for fid, m in meta.items():
                col.insert_one({
                    "file_id": fid,
                    "filename": m.get("filename", ""),
                    "original_name": m.get("original_name", ""),
                    "extension": m.get("extension", ""),
                    "size_bytes": m.get("size_bytes", 0),
                    "chars": m.get("chars", 0),
                    "pages": m.get("pages", 1),
                    "uploaded_at": m.get("uploaded_at", 0.0),
                    "tags": m.get("tags", []),
                })
        except Exception as e:
            logger.warning("Failed to save metadata to MogDB: %s", e)

    async def _async_load_metadata(self) -> dict[str, dict]:
        return await asyncio.to_thread(self._load_metadata)

    async def _async_save_metadata(self, meta: dict[str, dict]) -> None:
        await asyncio.to_thread(self._save_metadata, meta)

    def _file_id(self, filename: str) -> str:
        import re
        safe_name = re.sub(r'[^\w\-]', '_', filename)
        return f"{int(time.time())}_{safe_name}"

    # ── Endpoints ──

    async def list_files(
        self,
        sort: str = Query("uploaded_at", description="Sort field"),
        order: str = Query("desc", description="asc or desc"),
        tag: Optional[str] = Query(None, description="Filter by tag"),
    ) -> dict:
        """List all uploaded files with metadata."""
        meta = await self._async_load_metadata()
        items = []
        for fid, m in meta.items():
            file_path = self.UPLOADS_DIR / m["filename"]
            try:
                file_path.stat()
            except FileNotFoundError:
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
        auth_user: dict = Depends(require_auth_if_enabled),
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

        def _write_file():
            with open(file_path, "wb") as f:
                f.write(contents)
        await asyncio.to_thread(_write_file)

        try:
            tag_list = json.loads(tags) if isinstance(tags, str) else (tags or [])
        except (json.JSONDecodeError, TypeError):
            tag_list = []

        meta = await self._async_load_metadata()
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
        await self._async_save_metadata(meta)

        safe_audit_log("file.upload", resource=fid, detail=f"filename={file.filename}, size={len(contents)}")
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
        """Search uploaded files by name substring match."""
        query = q.lower()
        meta = await self._async_load_metadata()
        items = []
        for fid, m in meta.items():
            file_path = self.UPLOADS_DIR / m["filename"]
            try:
                file_path.stat()
            except FileNotFoundError:
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
        """Get file details including content."""
        meta = await self._async_load_metadata()
        if file_id not in meta:
            raise_error("File not found", "E_NOT_FOUND", status_code=404)
        m = meta[file_id]
        file_path = self.UPLOADS_DIR / m["filename"]
        try:
            text = file_path.read_text(errors="replace")
        except FileNotFoundError:
            raise_error("File not found on disk", "E_NOT_FOUND", status_code=404)
        return FileDetail(
            id=file_id,
            filename=m.get("original_name", m["filename"]),
            extension=m.get("extension", ""),
            size_bytes=m.get("size_bytes", 0),
            chars=m.get("chars", 0),
            pages=m.get("pages", 1),
            uploaded_at=m.get("uploaded_at", 0.0),
            tags=m.get("tags", []),
            text=text,
        )

    async def delete_file(self, file_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Delete a file and its metadata."""
        meta = await self._async_load_metadata()
        if file_id not in meta:
            raise_error("File not found", "E_NOT_FOUND", status_code=404)
        m = meta[file_id]
        file_path = self.UPLOADS_DIR / m["filename"]
        try:
            file_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Failed to delete file %s: %s", file_path, e)
        del meta[file_id]
        await self._async_save_metadata(meta)
        safe_audit_log("file.delete", resource=file_id)
        return success_response(data={"deleted": file_id})

    async def ingest_file(self, file_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Ingest file content into the RAG/knowledge store."""
        meta = await self._async_load_metadata()
        if file_id not in meta:
            raise_error("File not found", "E_NOT_FOUND", status_code=404)
        m = meta[file_id]
        file_path = self.UPLOADS_DIR / m["filename"]
        try:
            text = file_path.read_text(errors="replace")
        except FileNotFoundError:
            raise_error("File not found on disk", "E_NOT_FOUND", status_code=404)
        # Update chars count
        m["chars"] = len(text)
        meta[file_id] = m
        await self._async_save_metadata(meta)
        # TODO: integrate with RAG service
        safe_audit_log("file.ingest", resource=file_id, detail=f"chars={len(text)}")
        return IngestResponse(
            id=file_id,
            filename=m.get("original_name", m["filename"]),
            chars=len(text),
            facts_stored=0,
        )


router = FilesRouter().router
