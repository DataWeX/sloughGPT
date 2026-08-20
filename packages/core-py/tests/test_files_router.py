"""Tests for the files API router (routers/files.py).

Covers: list_files, upload_file, search_files, get_file, delete_file, ingest_file.
File I/O is done via tmp_path; only HTTP-level behavior is tested.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.files import FilesRouter  # noqa: E402


def _app(fr: FilesRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(fr.router)
    from infrastructure.exception_handlers import register_all_handlers
    register_all_handlers(app)
    return app


def _make_router(tmp_path: Path) -> FilesRouter:
    """Create a FilesRouter with uploads/metadata pointing to tmp_path."""
    from fastapi import APIRouter
    fr = FilesRouter.__new__(FilesRouter)
    fr.router = APIRouter(prefix="/files", tags=["files"])
    fr.UPLOADS_DIR = tmp_path
    fr.METADATA_FILE = tmp_path / "_metadata.json"
    fr.SUPPORTED_EXTENSIONS = {".txt": "text/plain", ".pdf": "application/pdf"}
    fr._register_routes()
    return fr


class TestUpload:
    def test_upload_txt_file(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        resp = client.post("/files/upload", files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")})
        assert resp.status_code == 200
        data = resp.json()
        assert data["size_bytes"] == 5
        assert data["filename"].endswith(".txt")

    def test_upload_no_extension(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        resp = client.post("/files/upload", files={"file": ("noext", io.BytesIO(b"data"), "text/plain")})
        assert resp.status_code == 400

    def test_upload_with_tags(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        resp = client.post("/files/upload",
            files={"file": ("tagged.txt", io.BytesIO(b"content"), "text/plain")},
            data={"tags": '["important"]'})
        assert resp.status_code == 200


class TestList:
    def test_list_empty(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        resp = client.get("/files")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_list_after_upload(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        client.post("/files/upload", files={"file": ("a.txt", io.BytesIO(b"a"), "text/plain")})
        resp = client.get("/files")
        assert resp.json()["total"] == 1


class TestSearch:
    def test_search_no_match(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        client.post("/files/upload", files={"file": ("hello.txt", io.BytesIO(b"hello world"), "text/plain")})
        resp = client.get("/files/search?q=xyz")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_search_match(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        client.post("/files/upload", files={"file": ("hello.txt", io.BytesIO(b"hello world"), "text/plain")})
        resp = client.get("/files/search?q=hello")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1


class TestGetFile:
    def test_get_existing(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        r = client.post("/files/upload", files={"file": ("doc.txt", io.BytesIO(b"test content"), "text/plain")})
        fid = r.json()["id"]
        resp = client.get(f"/files/{fid}")
        assert resp.status_code == 200
        assert resp.json()["text"] == "test content"

    def test_get_nonexistent(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        resp = client.get("/files/nonexistent")
        assert resp.status_code == 404


class TestDelete:
    def test_delete_existing(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        r = client.post("/files/upload", files={"file": ("del.txt", io.BytesIO(b"bye"), "text/plain")})
        fid = r.json()["id"]
        resp = client.delete(f"/files/{fid}")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "deleted"

    def test_delete_nonexistent(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        resp = client.delete("/files/nonexistent")
        assert resp.status_code == 404


class TestIngest:
    def test_ingest_file(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        long_text = b"This is a knowledge text chunk that is definitely longer than twenty characters."
        r = client.post("/files/upload", files={"file": ("ingest.txt", io.BytesIO(long_text), "text/plain")})
        fid = r.json()["id"]
        with patch("domains.learner.knowledge.get_knowledge_memory") as mock_km:
            mock_km.return_value.add_fact.return_value = True
            resp = client.post(f"/files/{fid}/ingest")
        assert resp.status_code == 200
        assert resp.json()["facts_stored"] == 1
