"""File Management Router — upload, list, search, delete, download files.

Business logic lives in the ``domain.files.FileStore`` core (sync,
standalone-runnable); this module is the HTTP projection only. The store
applies the proven codec policy (card fba1ce5e): text/code files are
zstd-compressed at rest (~7x — the stored frame doubles as the wire format
via ``Content-Encoding: zstd``), model-weight artifacts are stored raw
(proven incompressible), unknown classes get a zstd-if-smaller probe.

Grandfathering: same ``data/uploads`` directory, same MogDB ``files``
collection, same response shapes. Legacy entries (no ``codec`` field)
are read back exactly as before.
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel
from schemas.common import endpoint, raise_error, safe_audit_log, success_response

logger = logging.getLogger("slo.routers.files")

MAX_UPLOAD_SIZE = int(os.environ.get("SLO_MAX_UPLOAD_MB", "2048")) * 1024 * 1024


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
    from infrastructure.db_pool import get_db

    return get_db("uploads_mogdb")


class FilesRouter:
    """OOP router for file management endpoints."""

    def __init__(self):
        self.router = APIRouter(prefix="/files", tags=["files"])
        from domain.files import FileStore

        self.store = FileStore(db=_get_db())
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
            # model-weight artifacts (stored raw — proven incompressible)
            ".gguf": "application/octet-stream",
            ".safetensors": "application/octet-stream",
            ".pt": "application/octet-stream",
            ".pth": "application/octet-stream",
            ".ckpt": "application/octet-stream",
            ".onnx": "application/octet-stream",
            ".bin": "application/octet-stream",
            ".npz": "application/octet-stream",
        }
        self._register_routes()

    # ``UPLOADS_DIR`` stays an attribute for legacy callers/tests, but is
    # bound to the store root so overrides redirect blob storage too.
    @property
    def UPLOADS_DIR(self) -> Path:
        return self.store.root

    @UPLOADS_DIR.setter
    def UPLOADS_DIR(self, value) -> None:
        self.store.root = Path(value)
        self.store.root.mkdir(parents=True, exist_ok=True)

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
        self.router.add_api_route("/{file_id}", self.delete_file, methods=["DELETE"])
        self.router.add_api_route(
            "/{file_id}/download", self.download_file, methods=["GET"]
        )
        self.router.add_api_route(
            "/{file_id}/ingest", self.ingest_file, methods=["POST"], response_model=IngestResponse
        )

    def _meta_or_404(self, file_id: str) -> dict:
        try:
            return self.store.get_meta(file_id)
        except FileNotFoundError:
            raise_error("File not found", "E_NOT_FOUND", status_code=404)

    # ── Endpoints ──────────────────────────────────────────────────

    @endpoint("files.list_files")
    async def list_files(
        self,
        sort: str = Query("uploaded_at", description="Sort field"),
        order: str = Query("desc", description="asc or desc"),
        tag: str | None = Query(None, description="Filter by tag"),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """List all uploaded files with metadata."""
        items = [
            FileItem(
                id=m["file_id"],
                filename=m["filename"],
                extension=m.get("extension", ""),
                size_bytes=m.get("size_bytes", 0),
                uploaded_at=m.get("uploaded_at", 0.0),
                tags=m.get("tags", []),
            )
            for m in await asyncio.to_thread(self.store.list, tag)
        ]
        reverse = order.lower() != "asc"
        items.sort(key=lambda x: getattr(x, sort, 0), reverse=reverse)
        return FileListResponse(files=items, total=len(items))

    @endpoint("files.upload_file")
    async def upload_file(
        self,
        file: UploadFile = File(...),
        tags: str = Form("[]"),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Upload a file; stored via the FileStore codec policy (chunked)."""
        if not file.filename:
            raise_error("No filename provided", "E_BAD_REQUEST", status_code=400)

        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if not ext:
            raise_error("File must have an extension", "E_BAD_REQUEST", status_code=400)

        try:
            tag_list = json.loads(tags) if isinstance(tags, str) else (tags or [])
        except (json.JSONDecodeError, TypeError):
            tag_list = []

        # Chunked through the store — the payload never lands whole in memory.
        try:
            meta = await asyncio.to_thread(
                self.store.put_stream,
                file.file,
                file.filename,
                tag_list,
                MAX_UPLOAD_SIZE,
            )
        except ValueError as e:
            status = 413 if "too large" in str(e).lower() else 400
            raise_error(str(e), "E_BAD_REQUEST", status_code=status)

        safe_audit_log(
            "file.upload",
            resource=meta["file_id"],
            detail=(
                f"filename={file.filename}, size={meta['size_bytes']}, "
                f"stored={meta['stored_size']}, codec={meta['codec']}"
            ),
        )
        return UploadResponse(
            id=meta["file_id"],
            filename=file.filename,
            chars=0,
            pages=1,
            size_bytes=meta["size_bytes"],
        )

    @endpoint("files.search_files")
    async def search_files(
        self,
        q: str = Query(..., min_length=1, description="Search query"),
        tag: str | None = Query(None, description="Filter by tag"),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Search uploaded files by name substring match."""
        query = q.lower()
        items = []
        for m in await asyncio.to_thread(self.store.list, tag):
            name = m.get("original_name", m["filename"]).lower()
            if query not in name:
                continue
            items.append(
                FileItem(
                    id=m["file_id"],
                    filename=m.get("original_name", m["filename"]),
                    extension=m.get("extension", ""),
                    size_bytes=m.get("size_bytes", 0),
                    uploaded_at=m.get("uploaded_at", 0.0),
                    tags=m.get("tags", []),
                )
            )
        items.sort(key=lambda x: x.uploaded_at, reverse=True)
        return FileListResponse(files=items, total=len(items))

    @endpoint("files.get_file")
    async def get_file(
        self,
        file_id: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Get file details including content (decompressed text)."""
        m = self._meta_or_404(file_id)
        try:
            data = await asyncio.to_thread(lambda: self.store.get(file_id)[0])
        except FileNotFoundError:
            raise_error("File not found on disk", "E_NOT_FOUND", status_code=404)
        except ValueError as e:
            raise_error(str(e), "E_CORRUPT", status_code=500)
        return FileDetail(
            id=file_id,
            filename=m.get("original_name", m["filename"]),
            extension=m.get("extension", ""),
            size_bytes=m.get("size_bytes", 0),
            chars=m.get("chars", 0),
            pages=m.get("pages", 1),
            uploaded_at=m.get("uploaded_at", 0.0),
            tags=m.get("tags", []),
            text=data.decode("utf-8", errors="replace"),
        )

    @endpoint("files.download_file")
    async def download_file(
        self,
        file_id: str,
        request: Request,
        auth_user: dict = Depends(require_auth_if_enabled),
    ):
        """Download raw file bytes.

        Text/code files are stored zstd-compressed; when the client sends
        ``Accept-Encoding: zstd`` the stored frame is served untouched (zero
        transcoding — the at-rest format *is* the wire format). Otherwise the
        stream is decompressed on the way out.
        """
        from starlette.responses import StreamingResponse
        m = self._meta_or_404(file_id)
        accept = request.headers.get("accept-encoding", "")
        common = {
            "Vary": "Accept-Encoding",
            "X-Slo-Raw-Size": str(m.get("size_bytes", 0)),
            "X-Slo-Codec": str(m.get("codec", "raw")),
        }
        if m.get("codec") == "zstd" and "zstd" in accept:
            stored, _ = self.store.open_stored(file_id)
            return FileResponse(
                str(stored),
                media_type="application/octet-stream",
                filename=m.get("original_name", file_id),
                headers={**common, "Content-Encoding": "zstd"},
            )
        headers = {**common, "Content-Length": str(m.get("size_bytes", 0))}
        return StreamingResponse(
            self.store.iter_chunks(file_id),
            media_type="application/octet-stream",
            headers=headers,
        )

    @endpoint("files.delete_file")
    async def delete_file(
        self, file_id: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Delete a file and its metadata."""
        self._meta_or_404(file_id)
        try:
            await asyncio.to_thread(self.store.delete, file_id)
        except FileNotFoundError:
            raise_error("File not found", "E_NOT_FOUND", status_code=404)
        safe_audit_log("file.delete", resource=file_id)
        return success_response(data={"deleted": file_id})

    @endpoint("files.ingest_file")
    async def ingest_file(
        self, file_id: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Ingest file content into the RAG/knowledge store."""
        m = self._meta_or_404(file_id)
        try:
            data = await asyncio.to_thread(lambda: self.store.get(file_id)[0])
        except FileNotFoundError:
            raise_error("File not found on disk", "E_NOT_FOUND", status_code=404)
        except ValueError as e:
            raise_error(str(e), "E_CORRUPT", status_code=500)
        text = data.decode("utf-8", errors="replace")
        # Update chars count
        m["chars"] = len(text)
        await asyncio.to_thread(self.store.update_meta, m)
        # Integrate with RAG service
        facts_stored = 0
        try:
            from domain.cognition import get_rag_service

            rag = get_rag_service()
            chunk_ids = rag.add_document(
                content=text,
                metadata={
                    "source": "file",
                    "file_id": file_id,
                    "filename": m.get("original_name", m["filename"]),
                },
            )
            facts_stored = len(chunk_ids)
        except Exception as e:
            logger.warning("RAG ingest failed for file %s: %s", file_id, e)
        safe_audit_log(
            "file.ingest", resource=file_id, detail=f"chars={len(text)} facts={facts_stored}"
        )
        return IngestResponse(
            id=file_id,
            filename=m.get("original_name", m["filename"]),
            chars=len(text),
            facts_stored=facts_stored,
        )


router = FilesRouter().router
