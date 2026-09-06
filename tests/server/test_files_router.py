"""
Tests for the files router — upload, list, search, get, delete, ingest.
"""

import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.infrastructure.exception_handlers import register_all_handlers
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
    register_all_handlers(_app)
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

    @patch("domains.cognitive.rag_service.get_rag_service")
    def test_ingest_existing_file(self, mock_get_rag, client):
        rag = mock_get_rag.return_value
        rag.add_document.return_value = ["chunk1"]
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


class TestUploadEdgeCases:
    def test_upload_missing_file_field_422(self, client):
        resp = client.post("/files/upload")
        assert resp.status_code == 422

    def test_upload_empty_filename_422(self, client):
        resp = client.post("/files/upload", files={
            "file": ("", b"content", "text/plain"),
        })
        assert resp.status_code == 422

    def test_upload_dotfile_has_extension(self, client):
        resp = client.post("/files/upload", files={
            "file": (".env", b"SECRET=1", "text/plain"),
        })
        assert resp.status_code == 200
        assert resp.json()["filename"] == ".env"

    def test_upload_uppercase_extension(self, client):
        resp = client.post("/files/upload", files={
            "file": ("DATA.TXT", b"hello", "text/plain"),
        })
        assert resp.status_code == 200
        assert resp.json()["filename"] == "DATA.TXT"

    def test_upload_invalid_tags_ignored(self, client):
        resp = client.post("/files/upload", files={
            "file": ("tagged.txt", b"hello", "text/plain"),
        }, data={"tags": "not-json"})
        assert resp.status_code == 200

    def test_upload_stores_actual_content(self, client):
        payload = b"x" * 100
        resp = client.post("/files/upload", files={
            "file": ("big.txt", payload, "text/plain"),
        })
        assert resp.json()["size_bytes"] == 100


class TestListSorting:
    def test_list_tag_filter(self, client):
        client.post("/files/upload", files={
            "file": ("filtered.txt", b"c", "text/plain"),
        }, data={"tags": '["keep"]'})
        resp = client.get("/files?tag=keep")
        assert resp.json()["total"] >= 1
        resp2 = client.get("/files?tag=absent")
        assert resp2.json()["total"] == 0

    def test_list_asc_order(self, client):
        client.post("/files/upload", files={
            "file": ("asc.txt", b"c", "text/plain"),
        })
        resp = client.get("/files?order=asc")
        assert resp.status_code == 200


class TestSearchEdgeCases:
    def test_search_empty_query_422(self, client):
        resp = client.get("/files/search")
        assert resp.status_code == 422

    def test_search_case_insensitive(self, client):
        client.post("/files/upload", files={
            "file": ("MyReport.TXT", b"content", "text/plain"),
        })
        resp = client.get("/files/search?q=myreport")
        assert resp.json()["total"] >= 1

    def test_search_by_tag(self, client):
        client.post("/files/upload", files={
            "file": ("tagged_search.txt", b"content", "text/plain"),
        }, data={"tags": '["searchable"]'})
        resp = client.get("/files/search?q=tagged&tag=searchable")
        assert resp.json()["total"] >= 1
        resp2 = client.get("/files/search?q=tagged&tag=nope")
        assert resp2.json()["total"] == 0


class TestGetFileDetail:
    def test_get_returns_text_content(self, client):
        upload = client.post("/files/upload", files={
            "file": ("content.txt", b"Hello World", "text/plain"),
        })
        file_id = upload.json()["id"]
        resp = client.get(f"/files/{file_id}")
        assert resp.json()["text"] == "Hello World"

    def test_get_returns_tags(self, client):
        upload = client.post("/files/upload", files={
            "file": ("tagged2.txt", b"c", "text/plain"),
        }, data={"tags": '["alpha", "beta"]'})
        file_id = upload.json()["id"]
        resp = client.get(f"/files/{file_id}")
        assert resp.json()["tags"] == ["alpha", "beta"]


class TestIngestChunking:
    @patch("domains.cognitive.rag_service.get_rag_service")
    def test_ingest_chunks_long_text(self, mock_get_rag, client):
        rag = mock_get_rag.return_value
        rag.add_document.return_value = ["chunk1", "chunk2", "chunk3"]
        long_text = ("This is a long sentence that continues. "
                     "And another one follows it. And a third. ") * 20
        upload = client.post("/files/upload", files={
            "file": ("long_ingest.txt", long_text.encode(), "text/plain"),
        })
        file_id = upload.json()["id"]
        resp = client.post(f"/files/{file_id}/ingest")
        assert resp.status_code == 200
        assert resp.json()["facts_stored"] >= 1

    @patch("domains.cognitive.rag_service.get_rag_service")
    def test_ingest_empty_text_zero_facts(self, mock_get_rag, client):
        rag = mock_get_rag.return_value
        rag.add_document.return_value = []
        upload = client.post("/files/upload", files={
            "file": ("empty_ingest.txt", b"", "text/plain"),
        })
        file_id = upload.json()["id"]
        resp = client.post(f"/files/{file_id}/ingest")
        assert resp.status_code == 200
        assert resp.json()["facts_stored"] == 0


class TestMethodCoverage:
    def test_upload_get_captured_by_file_id(self, client):
        resp = client.get("/files/upload")
        assert resp.status_code == 404

    def test_search_wrong_method_405(self, client):
        resp = client.post("/files/search")
        assert resp.status_code == 405

    def test_ingest_wrong_method_405(self, client):
        resp = client.get("/files/x/ingest")
        assert resp.status_code == 405
