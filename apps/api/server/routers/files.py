"""File Management Router - upload, list, search, delete files with metadata.

Uses MogDB as the storage engine with automatic JSON sync.
File metadata is stored in MogDB and synced to JSON for human readability.
"""

import asyncio
import json
import logging
import re
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, Request, Response, UploadFile
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel
from schemas.common import endpoint, raise_error, safe_audit_log, success_response

from domain.infrastructure.asset_store import get_asset_store

logger = logging.getLogger("slo.routers.files")

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB


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


class AssetEvent(BaseModel):
    ts: float
    action: str
    asset_id: str = ""
    sha256: str = ""
    original_name: str = ""
    origin: str = ""
    detail: dict = {}


class HistoryResponse(BaseModel):
    events: list[AssetEvent]


def _get_db():
    from infrastructure.db_pool import get_db

    return get_db("uploads_mogdb")


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
        self._store = get_asset_store()
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
        # Registered before "/{file_id}" so the literal path wins the match.
        self.router.add_api_route(
            "/history", self.list_history, methods=["GET"], response_model=HistoryResponse
        )
        self.router.add_api_route(
            "/{file_id}", self.get_file, methods=["GET"], response_model=FileDetail
        )
        self.router.add_api_route("/{file_id}", self.delete_file, methods=["DELETE"])
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
                        "asset_id": doc.get("asset_id", ""),
                        "sha256": doc.get("sha256", ""),
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
                col.insert_one(
                    {
                        "file_id": fid,
                        "filename": m.get("filename", ""),
                        "original_name": m.get("original_name", ""),
                        "extension": m.get("extension", ""),
                        "size_bytes": m.get("size_bytes", 0),
                        "chars": m.get("chars", 0),
                        "pages": m.get("pages", 1),
                        "uploaded_at": m.get("uploaded_at", 0.0),
                        "tags": m.get("tags", []),
                        "asset_id": m.get("asset_id", ""),
                        "sha256": m.get("sha256", ""),
                    }
                )
        except Exception as e:
            logger.warning("Failed to save metadata to MogDB: %s", e)

    def _upsert_file_metadata(self, file_id: str, meta: dict) -> None:
        """Insert or update a single file's metadata in MogDB."""
        try:
            db = _get_db()
            col = db.collection("files")
            doc = {
                "file_id": file_id,
                "filename": meta.get("filename", ""),
                "original_name": meta.get("original_name", ""),
                "extension": meta.get("extension", ""),
                "size_bytes": meta.get("size_bytes", 0),
                "chars": meta.get("chars", 0),
                "pages": meta.get("pages", 1),
                "uploaded_at": meta.get("uploaded_at", 0.0),
                "tags": meta.get("tags", []),
                "asset_id": meta.get("asset_id", ""),
                "sha256": meta.get("sha256", ""),
            }
            existing = col.find_one({"file_id": file_id})
            if existing:
                col.update_one({"file_id": file_id}, {"$set": doc})
            else:
                col.insert_one(doc)
        except Exception as e:
            logger.warning("Failed to upsert file metadata: %s", e)

    def _delete_file_metadata(self, file_id: str) -> None:
        """Delete a single file's metadata from MogDB."""
        try:
            db = _get_db()
            col = db.collection("files")
            col.delete_one({"file_id": file_id})
        except Exception as e:
            logger.warning("Failed to delete file metadata: %s", e)

    async def _async_load_metadata(self) -> dict[str, dict]:
        return await asyncio.to_thread(self._load_metadata)

    async def _async_save_metadata(self, meta: dict[str, dict]) -> None:
        await asyncio.to_thread(self._save_metadata, meta)

    def _file_id(self, filename: str) -> str:
        safe_name = re.sub(r"[^\w\-]", "_", filename)
        return f"{int(time.time())}_{safe_name}"

    def _ensure_migrated(self) -> None:
        """One-shot adoption of legacy ``data/uploads`` files into the asset store.

        Bytes are COPIED into the CAS — originals stay on disk (file-safety) —
        and each adopted row gains ``asset_id``/``sha256`` so later reads
        resolve through the store. Idempotent (rows already carrying an
        ``asset_id``, and paths the store recognises via ``legacy_path``, are
        skipped), so concurrent first requests may both run it harmlessly.
        """
        if getattr(self, "_migrated", False):
            return
        self._migrated = True
        try:
            meta = self._load_metadata()
            entries: list[tuple[str, Path, dict]] = []
            for fid, m in meta.items():
                if m.get("asset_id"):
                    continue
                path = self.UPLOADS_DIR / m.get("filename", "")
                if not path.exists():
                    continue
                entries.append((fid, path, m))
            if not entries:
                return
            stats = self._store.migrate_uploads(
                [
                    (
                        path,
                        {
                            "filename": m.get("filename", ""),
                            "original_name": m.get("original_name", ""),
                            "tags": m.get("tags", []),
                        },
                    )
                    for _fid, path, m in entries
                ]
            )
            by_legacy = {a.legacy_path: a for a in self._store.list(origin="upload")}
            for fid, path, m in entries:
                asset = by_legacy.get(str(path))
                if asset is None:
                    continue
                row = dict(m)
                row["asset_id"] = asset.id
                row["sha256"] = asset.sha256
                self._upsert_file_metadata(fid, row)
            logger.info("asset store migration: %s", stats)
        except Exception as e:
            logger.warning("asset store migration failed: %s", e)

    def _read_text(self, m: dict) -> str | None:
        """Read an asset's text: store-first, legacy disk fallback. None if absent."""
        asset_id = m.get("asset_id", "")
        if asset_id:
            data = self._store.get_bytes(asset_id)
            if data is not None:
                return data.decode("utf-8", errors="replace")
        try:
            return (self.UPLOADS_DIR / m.get("filename", "")).read_text(errors="replace")
        except FileNotFoundError:
            return None

    # ── Endpoints ──

    @endpoint("files.list_files")
    async def list_files(
        self,
        sort: str = Query("uploaded_at", description="Sort field"),
        order: str = Query("desc", description="asc or desc"),
        tag: str | None = Query(None, description="Filter by tag"),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """List all uploaded files with metadata."""
        await asyncio.to_thread(self._ensure_migrated)
        meta = await self._async_load_metadata()
        items = []
        for fid, m in meta.items():
            if not m.get("asset_id"):
                file_path = self.UPLOADS_DIR / m["filename"]
                try:
                    file_path.stat()
                except FileNotFoundError:
                    continue
            if tag and tag not in m.get("tags", []):
                continue
            items.append(
                FileItem(
                    id=fid,
                    filename=m["filename"],
                    extension=m.get("extension", ""),
                    size_bytes=m.get("size_bytes", 0),
                    uploaded_at=m.get("uploaded_at", 0.0),
                    tags=m.get("tags", []),
                )
            )
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
        """Upload a file and save it to the server."""
        if not file.filename:
            raise_error("No filename provided", "E_BAD_REQUEST", status_code=400)

        ext = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if not ext:
            raise_error("File must have an extension", "E_BAD_REQUEST", status_code=400)

        contents = await file.read()
        if len(contents) > MAX_UPLOAD_SIZE:
            raise_error(
                f"File too large ({len(contents)} bytes, max {MAX_UPLOAD_SIZE})",
                "E_BAD_REQUEST",
                status_code=413,
            )
        fid = self._file_id(file.filename)

        try:
            tag_list = json.loads(tags) if isinstance(tags, str) else (tags or [])
        except (json.JSONDecodeError, TypeError):
            tag_list = []

        await asyncio.to_thread(self._ensure_migrated)
        asset = await asyncio.to_thread(
            self._store.put,
            contents,
            original_name=file.filename,
            origin="upload",
            media_type=self.SUPPORTED_EXTENSIONS.get(ext),
            tags=tag_list,
        )

        file_meta = {
            "filename": f"{fid}{ext}",
            "original_name": file.filename,
            "extension": ext,
            "size_bytes": len(contents),
            "chars": 0,
            "pages": 1,
            "uploaded_at": time.time(),
            "tags": tag_list,
            "asset_id": asset.id,
            "sha256": asset.sha256,
        }
        await asyncio.to_thread(self._upsert_file_metadata, fid, file_meta)

        safe_audit_log(
            "file.upload", resource=fid, detail=f"filename={file.filename}, size={len(contents)}"
        )
        return UploadResponse(
            id=fid,
            filename=file.filename,
            chars=0,
            pages=1,
            size_bytes=len(contents),
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
        await asyncio.to_thread(self._ensure_migrated)
        meta = await self._async_load_metadata()
        items = []
        for fid, m in meta.items():
            if not m.get("asset_id"):
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
            items.append(
                FileItem(
                    id=fid,
                    filename=m.get("original_name", m["filename"]),
                    extension=m.get("extension", ""),
                    size_bytes=m.get("size_bytes", 0),
                    uploaded_at=m.get("uploaded_at", 0.0),
                    tags=m.get("tags", []),
                )
            )
        items.sort(key=lambda x: x.uploaded_at, reverse=True)
        return FileListResponse(files=items, total=len(items))

    @endpoint("files.list_history")
    async def list_history(
        self,
        asset_id: str | None = Query(None, description="Filter by asset id"),
        action: str | None = Query(
            None, description="Filter by action (put/download/delete/migrate)"
        ),
        limit: int = Query(100, ge=1, le=500, description="Max events returned"),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Asset event history from the store collector (newest first)."""
        events = await asyncio.to_thread(
            self._store.history, asset_id=asset_id or "", action=action or "", limit=limit
        )
        return HistoryResponse(events=[AssetEvent(**e) for e in events])

    @endpoint("files.get_file")
    async def get_file(
        self,
        file_id: str,
        request: Request,
        response: Response,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Get file details including content.

        Serves a strong ETag (the content sha256 when known) so preview
        refetches can come back as 304; a 304 is not a download and records
        no event.
        """
        await asyncio.to_thread(self._ensure_migrated)
        meta = await self._async_load_metadata()
        if file_id not in meta:
            raise_error("File not found", "E_NOT_FOUND", status_code=404)
        m = meta[file_id]
        sha = m.get("sha256", "")
        if sha:
            etag = f'"{sha}"'
        else:
            etag = f'"{file_id}-{int(m.get("uploaded_at", 0))}"'
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers={"ETag": etag})
        text = await asyncio.to_thread(self._read_text, m)
        if text is None:
            raise_error("File not found on disk", "E_NOT_FOUND", status_code=404)
        if m.get("asset_id"):
            await asyncio.to_thread(
                self._store.record_event,
                "download",
                asset_id=m["asset_id"],
                original_name=m.get("original_name", ""),
                origin="upload",
            )
        response.headers["ETag"] = etag
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

    @endpoint("files.delete_file")
    async def delete_file(
        self, file_id: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Delete a file and its metadata."""
        await asyncio.to_thread(self._ensure_migrated)
        meta = await self._async_load_metadata()
        if file_id not in meta:
            raise_error("File not found", "E_NOT_FOUND", status_code=404)
        m = meta[file_id]
        if m.get("asset_id"):
            await asyncio.to_thread(self._store.delete, m["asset_id"], purge_blob=True)
        file_path = self.UPLOADS_DIR / m["filename"]
        try:
            file_path.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Failed to delete file %s: %s", file_path, e)
        await asyncio.to_thread(self._delete_file_metadata, file_id)
        safe_audit_log("file.delete", resource=file_id)
        return success_response(data={"deleted": file_id})

    @endpoint("files.ingest_file")
    async def ingest_file(
        self, file_id: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
        """Ingest file content into the RAG/knowledge store."""
        await asyncio.to_thread(self._ensure_migrated)
        meta = await self._async_load_metadata()
        if file_id not in meta:
            raise_error("File not found", "E_NOT_FOUND", status_code=404)
        m = meta[file_id]
        text = await asyncio.to_thread(self._read_text, m)
        if text is None:
            raise_error("File not found on disk", "E_NOT_FOUND", status_code=404)
        # Update chars count
        m["chars"] = len(text)
        await asyncio.to_thread(self._upsert_file_metadata, file_id, m)
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
