"""File Management Router - upload, list, search, delete files with metadata.

Also serves the image-generation surface (generate/gallery/styles), merged
from routers/images.py in card 20261101_154058 — one /files facade.

Uses MogDB as the storage engine with automatic JSON sync.
File metadata is stored in MogDB and synced to JSON for human readability.
"""

import asyncio
import base64
import io
import ipaddress
import json
import logging
import os
import re
import socket
import threading
import time
from pathlib import Path
from typing import Literal
from urllib.parse import unquote, urlparse
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, Query, UploadFile
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel, Field
from schemas.common import endpoint, raise_error, safe_audit_log, success_response

logger = logging.getLogger("slo.routers.files")

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_IMPORT_SIZE = 50 * 1024 * 1024  # 50 MB — same cap as uploads
_IMPORT_JOB_KEEP = 20  # terminal jobs retained for status reads


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


class ImportRequest(BaseModel):
    url: str


class ImportStartResponse(BaseModel):
    job_id: str


class ImportStatusResponse(BaseModel):
    job_id: str
    state: str  # running | done | error
    filename: str
    done: int  # decoded bytes on disk
    total: int  # decoded total; 0 = unknown (client shows indeterminate)
    pct: float
    wire_bytes: int | None = None
    saved_bytes: int | None = None
    content_encoding: str = "identity"
    error: str | None = None


def _get_db():
    from infrastructure.db_pool import get_db

    return get_db("uploads_mogdb")


# In-memory import-job registry: written by worker threads, read by status
# handlers. Terminal jobs are kept (bounded by _IMPORT_JOB_KEEP) so the UI
# can read the final wire/savings receipt after the poll ends.
_IMPORT_JOBS: dict[str, dict] = {}
_IMPORT_JOBS_LOCK = threading.Lock()
_IMPORT_TASKS: set = set()  # hold asyncio task refs so GC cannot cancel them


class _ImportSizeCap(Exception):
    """Raised from on_chunk to abort an import over MAX_IMPORT_SIZE."""


def _validate_import_url(url: str) -> str:
    """SSRF guard for user-supplied import URLs; returns the URL's filename.

    Blocks non-http(s) schemes and hosts resolving to non-global addresses
    (loopback / private / link-local — including cloud metadata
    169.254.169.254) unless ``SLO_FILES_ALLOW_PRIVATE=1`` (tests/dev).

    The check runs once at request time; downcraft re-resolves on connect,
    so a DNS-rebinding window remains — accepted for this local tool (this
    blocks the direct-hit class, which is the one that matters here).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise_error("Only http(s) URLs can be imported", "E_BAD_REQUEST", status_code=400)
    host = parsed.hostname
    if not host:
        raise_error("URL has no host", "E_BAD_REQUEST", status_code=400)
    if os.environ.get("SLO_FILES_ALLOW_PRIVATE") != "1":
        try:
            infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        except socket.gaierror:
            raise_error(f"Could not resolve host {host}", "E_BAD_REQUEST", status_code=400)
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if ip.version == 6 and ip.ipv4_mapped is not None:
                ip = ip.ipv4_mapped
            if not ip.is_global:
                raise_error(
                    "Import blocked: URL resolves to a private/internal address",
                    "E_FORBIDDEN",
                    status_code=403,
                )
    filename = unquote(Path(parsed.path).name).strip()
    if "." not in filename:
        raise_error("URL must point to a file with an extension", "E_BAD_REQUEST", status_code=400)
    return filename


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
            "/import", self.download_url, methods=["POST"], response_model=ImportStartResponse
        )
        self.router.add_api_route(
            "/import/{job_id}",
            self.import_status,
            methods=["GET"],
            response_model=ImportStatusResponse,
        )
        self.router.add_api_route(
            "/search", self.search_files, methods=["GET"], response_model=FileListResponse
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
        file_path = self.UPLOADS_DIR / f"{fid}{ext}"

        def _write_file():
            with open(file_path, "wb") as f:
                f.write(contents)

        await asyncio.to_thread(_write_file)

        try:
            tag_list = json.loads(tags) if isinstance(tags, str) else (tags or [])
        except (json.JSONDecodeError, TypeError):
            tag_list = []

        file_meta = {
            "filename": f"{fid}{ext}",
            "original_name": file.filename,
            "extension": ext,
            "size_bytes": len(contents),
            "chars": 0,
            "pages": 1,
            "uploaded_at": time.time(),
            "tags": tag_list,
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

    # ── Import from URL (one-request, compressed, honest progress) ─────────

    @staticmethod
    def _job_put(job_id: str, job: dict) -> None:
        with _IMPORT_JOBS_LOCK:
            _IMPORT_JOBS[job_id] = job
            terminal = [k for k, v in _IMPORT_JOBS.items() if v["state"] != "running"]
            overflow = len(_IMPORT_JOBS) - _IMPORT_JOB_KEEP
            for k in sorted(terminal)[: max(0, overflow)]:
                _IMPORT_JOBS.pop(k, None)

    def _run_import(self, job_id: str, url: str, filename: str) -> None:
        """Worker thread: downcraft fetch → uploads dir → metadata upsert.

        Progress stays in disk byte-space: on_chunk reports decoded bytes on
        disk vs the decoded total (0 → client shows indeterminate). Wire bytes
        and savings come from WireStats and are reported only at the end —
        never mixed into the bar.
        """
        from downcraft.download.http import WireStats, download_file

        job = _IMPORT_JOBS.get(job_id)
        if job is None:
            return
        job.update(state="running", filename=filename, done=0, total=0)
        ext = filename[filename.rfind(".") :]
        fid = self._file_id(filename)
        dest = self.UPLOADS_DIR / f"{fid}{ext}"
        stats = WireStats()

        def on_chunk(done: int, total: int) -> None:
            if done > MAX_IMPORT_SIZE:
                raise _ImportSizeCap(f"import exceeds {MAX_IMPORT_SIZE} bytes")
            j = _IMPORT_JOBS.get(job_id)
            if j is not None:
                j["done"], j["total"] = done, total

        def _fetch() -> Path:
            from downcraft.download.http import _part_path

            try:
                return download_file(
                    url,
                    dest,
                    compressed=True,
                    skip_if_exists=False,
                    stats=stats,
                    on_chunk=on_chunk,
                )
            except BaseException:
                # Never leave a partial body or part file behind.
                _part_path(dest).unlink(missing_ok=True)
                dest.unlink(missing_ok=True)
                raise

        try:
            # _run_import runs inside a worker thread (no running loop) — blocking call.
            path = _fetch()
            size = path.stat().st_size

            def _upsert() -> None:
                try:
                    tag_list = json.loads(os.environ.get("SLO_IMPORT_TAGS", "[]"))
                except (TypeError, ValueError):
                    tag_list = []
                self._upsert_file_metadata(
                    fid,
                    {
                        "filename": path.name,
                        "original_name": filename,
                        "extension": ext.lstrip(".").lower(),
                        "size_bytes": size,
                        "chars": 0,
                        "pages": 1,
                        "uploaded_at": time.time(),
                        "tags": tag_list,
                    },
                )

            _upsert()
            safe_audit_log(
                "file.import",
                resource=fid,
                detail=(
                    f"url={url}, size={size}, wire={stats.wire_bytes}, "
                    f"saved={stats.saved_bytes}, enc={stats.content_encoding}"
                ),
            )
            job.update(
                state="done",
                done=size,
                total=size,
                file_id=fid,
                wire_bytes=stats.wire_bytes,
                saved_bytes=stats.saved_bytes,
                content_encoding=stats.content_encoding or "identity",
                error=None,
            )
        except _ImportSizeCap as exc:
            safe_audit_log("file.import_start", resource=job_id, detail=f"aborted: {exc}")
            job.update(state="error", error=str(exc))
        except Exception as exc:  # noqa: BLE001 — surfaced to the status reader
            logger.warning("Import failed for %s: %s", url, exc, extra={"tag": "FILES"})
            job.update(state="error", error=str(exc) or exc.__class__.__name__)

    @endpoint("files.download_url")
    async def download_url(
        self,
        req: ImportRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> ImportStartResponse:
        """Start a background import of a remote URL into the file manager."""
        url = req.url.strip()
        if not url:
            raise_error("No URL provided", "E_BAD_REQUEST", status_code=400)
        filename = _validate_import_url(url)
        job_id = uuid4().hex[:12]
        job = {
            "job_id": job_id,
            "state": "running",
            "filename": filename,
            "done": 0,
            "total": 0,
            "file_id": None,
            "wire_bytes": None,
            "saved_bytes": None,
            "content_encoding": "identity",
            "error": None,
        }
        self._job_put(job_id, job)
        safe_audit_log("file.import_start", resource=job_id, detail=f"url={url}")
        task = asyncio.create_task(asyncio.to_thread(self._run_import, job_id, url, filename))
        _IMPORT_TASKS.add(task)
        task.add_done_callback(_IMPORT_TASKS.discard)
        return ImportStartResponse(job_id=job_id)

    @endpoint("files.import_status")
    async def import_status(
        self,
        job_id: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> ImportStatusResponse:
        """Read an import job's progress (decoded/decoded pct) and receipt."""
        job = _IMPORT_JOBS.get(job_id)
        if job is None:
            raise_error("Import job not found", "E_NOT_FOUND", status_code=404)
        done, total, state = job["done"], job["total"], job["state"]
        if state == "done":
            pct = 100.0
        elif total > done >= 0 and total > 0:
            pct = round(100.0 * done / total, 1)
        else:
            pct = 0.0
        return ImportStatusResponse(
            job_id=job_id,
            state=state,
            filename=job["filename"],
            done=done,
            total=total,
            pct=pct,
            wire_bytes=job["wire_bytes"],
            saved_bytes=job["saved_bytes"],
            content_encoding=job["content_encoding"],
            error=job["error"],
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

    @endpoint("files.get_file")
    async def get_file(
        self,
        file_id: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
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

    @endpoint("files.delete_file")
    async def delete_file(
        self, file_id: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
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
        await asyncio.to_thread(self._delete_file_metadata, file_id)
        safe_audit_log("file.delete", resource=file_id)
        return success_response(data={"deleted": file_id})

    @endpoint("files.ingest_file")
    async def ingest_file(
        self, file_id: str, auth_user: dict = Depends(require_auth_if_enabled)
    ) -> dict:
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


# ── Image generation (merged from routers/images.py) ─────────────────────
# Schemas first: ImagesRouter's method annotations resolve them at class-
# definition time. Kept verbatim so every wire path and response shape is
# byte-identical to the old module.

# ── Schema ────────────────────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    style: Literal["realistic", "cartoon", "watercolor", "sketch", "fantasy"] = "realistic"


class GenerateResponse(BaseModel):
    image: str  # base64 data URL
    style: str
    prompt: str
    id: str


class ImagesRouter:
    """OOP-based router for image generation endpoints."""

    def __init__(self):
        self.router = APIRouter(prefix="/images", tags=["images"])
        self.STYLES = {
            "realistic": "Realistic",
            "cartoon": "Cartoon",
            "watercolor": "Watercolor",
            "sketch": "Sketch",
            "fantasy": "Fantasy",
        }
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(
            "/generate",
            self.generate_image,
            methods=["POST"],
            response_model=GenerateResponse,
        )
        self.router.add_api_route(
            "/gallery",
            self.list_gallery,
            methods=["GET"],
        )
        self.router.add_api_route(
            "/styles",
            self.list_styles,
            methods=["GET"],
        )

    # ── Procedural Image Generators (Pillow-based, no external deps) ──────

    def _generate_gradient_image(self, prompt: str, width: int = 512, height: int = 512) -> bytes:
        """Generate a gradient-based image based on prompt keywords."""
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            raise_error(
                "Pillow library required for image generation", "E_INFRA_STARTUP", status_code=500
            )

        prompt_lower = prompt.lower()
        colors = []

        if any(k in prompt_lower for k in ["cabin", "nature", "forest", "tree", "green"]):
            colors = ["#2d5a27", "#4a7c3e", "#6b9e5a"]
        elif any(k in prompt_lower for k in ["sunset", "evening", "sky", "orange", "red"]):
            colors = ["#d35400", "#e67e22", "#f39c12", "#f1c40f"]
        elif any(k in prompt_lower for k in ["ocean", "water", "blue", "sea"]):
            colors = ["#1e3799", "#3a86ff", "#83b6ed", "#caf0f8"]
        elif any(k in prompt_lower for k in ["mountain", "sky", "cloud", "white", "bright"]):
            colors = ["#3a86ff", "#83b6ed", "#caf0f8", "#ffffff"]
        elif any(k in prompt_lower for k in ["fire", "flame", "heat", "red", "burn"]):
            colors = ["#b51700", "#e85d04", "#faa330", "#ffdd59"]
        else:
            colors = ["#6a11cb", "#2575fc"]

        def _c(c):
            return int(c.lstrip("#")[:2], 16)

        rgb_colors = [(_c(c), _c(c), _c(c)) for c in colors]
        rgb_colors = []
        for c in colors:
            h = c.lstrip("#")
            rgb_colors.append((int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)))

        img = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(img)

        for y in range(height):
            t = y / height
            idx = t * (len(rgb_colors) - 1)
            i = int(idx)
            f = idx - i
            if i >= len(rgb_colors) - 1:
                r, g, b = rgb_colors[-1]
            else:
                r = int(rgb_colors[i][0] * (1 - f) + rgb_colors[i + 1][0] * f)
                g = int(rgb_colors[i][1] * (1 - f) + rgb_colors[i + 1][1] * f)
                b = int(rgb_colors[i][2] * (1 - f) + rgb_colors[i + 1][2] * f)
            draw.line([(0, y), (width, y)], fill=(r, g, b))

        draw2 = ImageDraw.Draw(img)
        for i in range(0, width, 20):
            for j in range(0, height, 20):
                alpha = 30 if (i + j) % 40 == 0 else 15
                draw2.rectangle([i, j, i + 10, j + 10], fill=(255, 255, 255, alpha))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()

    def _generate_cartoon_image(self, prompt: str, width: int = 512, height: int = 512) -> bytes:
        """Generate a cartoon-style image with bold colors and outlines."""
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError:
            raise_error("Pillow library required", "E_INFRA_STARTUP", status_code=500)

        img = Image.new("RGB", (width, height), color="#f0f0f0")
        draw = ImageDraw.Draw(img)

        rand_x = (width // 4) + (hash(prompt) % (width // 2))
        rand_y = (height // 4) + (hash(prompt + "y") % (height // 2))
        size = min(width, height) // 3

        draw.ellipse(
            [rand_x, rand_y, rand_x + size, rand_y + size], fill="#ff6b6b", outline="#333", width=4
        )
        draw.ellipse(
            [rand_x + size // 2, rand_y + size // 2, rand_x + size * 1.5, rand_y + size * 1.5],
            fill="#4ecdc4",
            outline="#333",
            width=4,
        )

        try:
            font = ImageFont.load_default()
        except Exception as exc:
            logger.debug("Font loading failed: %s", exc)
            font = None
        draw.text((width // 2 - 50, height - 50), prompt[:20], fill="#333", font=font)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()

    def _generate_watercolor_image(self, prompt: str, width: int = 512, height: int = 512) -> bytes:
        """Generate a watercolor-style image with soft, blended colors."""
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            raise_error("Pillow library required", "E_INFRA_STARTUP", status_code=500)

        img = Image.new("RGBA", (width, height), color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        import random

        random.seed(hash(prompt))

        colors = [
            (255, 200, 200, 80),
            (200, 255, 200, 80),
            (200, 200, 255, 80),
            (255, 255, 200, 80),
            (255, 200, 255, 80),
        ]

        for _ in range(5):
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(50, 200)
            color = random.choice(colors)
            draw.ellipse([x, y, x + size, y + size], fill=color)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()

    def _generate_sketch_image(self, prompt: str, width: int = 512, height: int = 512) -> bytes:
        """Generate a sketch-style image with hand-drawn appearance."""
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            raise_error("Pillow library required", "E_INFRA_STARTUP", status_code=500)

        img = Image.new("RGB", (width, height), color="#fdfbf7")
        draw = ImageDraw.Draw(img)

        import random

        random.seed(hash(prompt + "sketch"))

        color = (50, 50, 50)

        for i in range(0, width, 10):
            y_start = random.randint(0, height // 4)
            draw.line(
                [i, y_start, i, y_start + random.randint(50, 200)],
                fill=color,
                width=random.randint(1, 2),
            )

        for j in range(0, height, 10):
            x_start = random.randint(0, width // 4)
            draw.line(
                [x_start, j, x_start + random.randint(50, 200), j],
                fill=color,
                width=random.randint(1, 2),
            )

        cx, cy = width // 2, height // 2
        draw.arc([cx - 80, cy - 80, cx + 80, cy + 80], 0, 270, fill=color, width=3)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()

    def _generate_fantasy_image(self, prompt: str, width: int = 512, height: int = 512) -> bytes:
        """Generate a fantasy-style image with magical colors and glow."""
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            raise_error("Pillow library required", "E_INFRA_STARTUP", status_code=500)

        img = Image.new("RGB", (width, height), color="#1a1a2e")
        draw = ImageDraw.Draw(img)

        import random

        random.seed(hash(prompt + "fantasy"))

        for _ in range(50):
            x = random.randint(0, width)
            y = random.randint(0, height)
            size = random.randint(1, 3)
            draw.ellipse([x, y, x + size, y + size], fill="#fffff0")

        colors = ["#ff00ff", "#00ffff", "#ff00ff", "#ffff00"]
        for i, color in enumerate(colors):
            cx = width // 2 + random.randint(-100, 100)
            cy = height // 2 + random.randint(-100, 100)
            size = 100 - i * 15
            alpha = 60 - i * 10
            draw.ellipse(
                [cx - size, cy - size, cx + size, cy + size], fill=(*self.hex_to_rgb(color), alpha)
            ) if False else None
            draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=self.hex_to_rgb(color))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()

    @staticmethod
    def hex_to_rgb(hex_color: str) -> tuple:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

    # ── Generation Functions ───────────────────────────────────────────────

    def _generate_image(self, prompt: str, style: str) -> bytes:
        """Generate an image based on prompt and style using procedural methods."""
        generators = {
            "realistic": self._generate_gradient_image,
            "cartoon": self._generate_cartoon_image,
            "watercolor": self._generate_watercolor_image,
            "sketch": self._generate_sketch_image,
            "fantasy": self._generate_fantasy_image,
        }

        generator = generators.get(style, self._generate_gradient_image)
        return generator(prompt)

    def _save_image(self, image_bytes: bytes, style: str) -> str:
        """Save image to gallery and return relative path."""
        gallery_dir = Path(__file__).resolve().parents[4] / "data" / "gallery"
        gallery_dir.mkdir(parents=True, exist_ok=True)

        import uuid

        filename = f"generated_{uuid.uuid4().hex[:8]}.png"
        filepath = gallery_dir / filename

        filepath.write_bytes(image_bytes)

        return f"/data/gallery/{filename}"

    # ── Endpoints ─────────────────────────────────────────────────────────

    @endpoint("images.generate_image")
    async def generate_image(
        self,
        request: GenerateRequest,
        background_tasks: BackgroundTasks = None,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Generate an image from text description."""
        import time as _time

        _t0 = _time.monotonic()
        image_bytes = self._generate_image(request.prompt, request.style)

        image_path = self._save_image(image_bytes, request.style)

        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{base64_image}"
        _elapsed_ms = (_time.monotonic() - _t0) * 1000
        safe_audit_log(
            "images.generate",
            resource=request.prompt[:80],
            detail=f"style={request.style} elapsed={_elapsed_ms:.0f}ms",
        )

        return GenerateResponse(
            image=data_url,
            style=request.style,
            prompt=request.prompt,
            id=image_path.split("/")[-1].replace(".png", ""),
        )

    @endpoint("images.list_gallery")
    async def list_gallery(self) -> dict:
        """List all generated images in the gallery."""
        gallery_dir = Path(__file__).resolve().parents[4] / "data" / "gallery"

        def _scan_gallery():
            if not gallery_dir.exists():
                return []
            images = []
            for filepath in sorted(
                gallery_dir.glob("generated_*.png"),
                key=lambda x: x.stat().st_mtime,
                reverse=True,
            ):
                images.append(
                    {
                        "id": filepath.stem,
                        "path": f"/data/gallery/{filepath.name}",
                        "created": int(filepath.stat().st_mtime),
                    }
                )
            return images[:50]

        images = await asyncio.to_thread(_scan_gallery)
        return success_response(data={"images": images})

    @endpoint("images.list_styles")
    async def list_styles(self) -> dict:
        """List available image generation styles."""
        return success_response(data={"styles": list(self.STYLES.items())})


def hex_to_rgb(hex_color: str) -> tuple:
    """Convert hex color to RGB tuple. Module-level wrapper for ImagesRouter.hex_to_rgb."""
    return ImagesRouter.hex_to_rgb(hex_color)


# ── Aggregate facade ──────────────────────────────────────────────────────
# Boot contract: one module-level router per module. Sub-router prefixes
# bake at add-time, so /files/* and /images/* paths are unchanged.

router = APIRouter()
router.include_router(FilesRouter().router)
router.include_router(ImagesRouter().router)
