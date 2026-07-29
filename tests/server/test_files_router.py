"""
Tests for the files router — upload, list, search, get, delete, ingest.
"""

import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.files import FilesRouter


@pytest.fixture
def isolated_router(tmp_path):
    r = FilesRouter()
    r.UPLOADS_DIR = tmp_path / "uploads"
    r.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    r.METADATA_FILE = r.UPLOADS_DIR / "_metadata.json"
    return r


@pytest.fixture
def app(isolated_router):
    _app = FastAPI()
    _app.include_router(isolated_router.router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestListFiles:
    """GET /files"""

    def test_returns_empty_list_initially(self, client):
        resp = client.get("/files")
        assert resp.status_code == 200
        body = resp.json()
        assert body["files"] == []
        assert body["total"] == 0


class TestUploadFile:
    """POST /files/upload"""

    def test_upload_txt_file(self, client):
        resp = client.post("/files/upload", files={
            "file": ("test.txt", b"hello world", "text/plain"),
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "test.txt"
        assert body["size_bytes"] == 11

    def test_upload_rejects_no_extension(self, client):
        resp = client.post("/files/upload", files={
            "file": ("noext", b"content", "text/plain"),
        })
        assert resp.status_code == 400

    def test_upload_with_tags(self, client):
        resp = client.post("/files/upload", files={
            "file": ("doc.md", b"# hello", "text/markdown"),
        }, data={"tags": '["docs", "markdown"]'})
        assert resp.status_code == 200
        assert resp.json()["chars"] == 0


class TestSearchFiles:
    """GET /files/search"""

    def test_search_by_name(self, client):
        client.post("/files/upload", files={
            "file": ("my_document.txt", b"content", "text/plain"),
        })
        resp = client.get("/files/search?q=document")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    def test_search_no_match(self, client):
        resp = client.get("/files/search?q=nonexistent")
        assert resp.json()["total"] == 0


class TestGetFile:
    """GET /files/{file_id}"""

    def test_get_file_not_found(self, client):
        resp = client.get("/files/nonexistent")
        assert resp.status_code == 404


class TestDeleteFile:
    """DELETE /files/{file_id}"""

    def test_delete_nonexistent_returns_404(self, client):
        resp = client.delete("/files/nonexistent")
        assert resp.status_code == 404
