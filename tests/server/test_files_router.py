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

    def test_returns_uploaded_files(self, client):
        client.post("/files/upload", files={"file": ("a.txt", b"content", "text/plain")})
        resp = client.get("/files")
        assert resp.json()["total"] >= 1


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

    def test_upload_json_file(self, client):
        resp = client.post("/files/upload", files={
            "file": ("data.json", b'{"key": "value"}', "application/json"),
        })
        assert resp.status_code == 200
        assert resp.json()["filename"] == "data.json"

    def test_upload_python_file(self, client):
        resp = client.post("/files/upload", files={
            "file": ("script.py", b"print('hello')", "text/x-python"),
        })
        assert resp.status_code == 200

    def test_upload_csv_file(self, client):
        resp = client.post("/files/upload", files={
            "file": ("data.csv", b"a,b,c\n1,2,3", "text/csv"),
        })
        assert resp.status_code == 200


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

    def test_get_existing_file(self, client):
        upload = client.post("/files/upload", files={
            "file": ("readable.txt", b"hello", "text/plain"),
        })
        file_id = upload.json()["id"]
        resp = client.get(f"/files/{file_id}")
        assert resp.status_code == 200
        assert resp.json()["filename"] == "readable.txt"


class TestDeleteFile:
    """DELETE /files/{file_id}"""

    def test_delete_nonexistent_returns_404(self, client):
        resp = client.delete("/files/nonexistent")
        assert resp.status_code == 404

    def test_delete_existing_file(self, client):
        upload = client.post("/files/upload", files={
            "file": ("to_delete.txt", b"delete me", "text/plain"),
        })
        file_id = upload.json()["id"]
        resp = client.delete(f"/files/{file_id}")
        assert resp.status_code == 200

    def test_list_after_upload(self, client):
        client.post("/files/upload", files={
            "file": ("listed.txt", b"content", "text/plain"),
        })
        resp = client.get("/files")
        assert resp.json()["total"] >= 1

    def test_get_after_delete(self, client):
        upload = client.post("/files/upload", files={
            "file": ("temp.txt", b"temp", "text/plain"),
        })
        file_id = upload.json()["id"]
        client.delete(f"/files/{file_id}")
        resp = client.get(f"/files/{file_id}")
        assert resp.status_code == 404


class TestIngestFile:
    """POST /files/{file_id}/ingest"""

    def test_ingest_nonexistent_file(self, client):
        resp = client.post("/files/nonexistent/ingest")
        assert resp.status_code == 404

    @patch("domains.learner.knowledge.get_knowledge_memory")
    def test_ingest_existing_file(self, mock_get_mem, client):
        mem = mock_get_mem.return_value
        mem.add_fact.return_value = True
        upload = client.post("/files/upload", files={
            "file": ("ingest.txt", b"some content for knowledge base", "text/plain"),
        })
        file_id = upload.json()["id"]
        resp = client.post(f"/files/{file_id}/ingest")
        assert resp.status_code == 200
        assert resp.json()["facts_stored"] >= 0


class TestSupportedExtensions:
    def test_router_has_extensions(self, isolated_router):
        assert ".txt" in isolated_router.SUPPORTED_EXTENSIONS
        assert ".pdf" in isolated_router.SUPPORTED_EXTENSIONS
        assert ".py" in isolated_router.SUPPORTED_EXTENSIONS
