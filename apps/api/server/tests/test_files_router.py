"""Tests for files router endpoints."""
import json
import os
import time
import pytest
from io import BytesIO
from pathlib import Path

from tests.test_support import get_test_client
from routers.files import UPLOADS_DIR, METADATA_FILE, _save_metadata, _load_metadata

client = get_test_client()


@pytest.fixture(autouse=True)
def _cleanup_uploads(tmp_path):
    """Save and restore metadata around tests to avoid pollution."""
    orig_meta = {}
    if METADATA_FILE.exists():
        try:
            orig_meta = json.loads(METADATA_FILE.read_text())
        except Exception:
            pass
    yield
    _save_metadata(orig_meta)


class TestListFiles:
    def test_list_empty(self):
        resp = client.get("/files")
        assert resp.status_code == 200
        body = resp.json()
        assert body["files"] == []
        assert body["total"] == 0

    def test_list_has_files_field(self):
        resp = client.get("/files")
        body = resp.json()
        assert "files" in body
        assert "total" in body


class TestUploadFile:
    def test_upload_txt_file(self):
        content = b"Hello, world!"
        resp = client.post(
            "/files/upload",
            files={"file": ("test.txt", BytesIO(content), "text/plain")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "test.txt"
        assert body["size_bytes"] == len(content)
        assert "id" in body

    def test_upload_no_extension_fails(self):
        resp = client.post(
            "/files/upload",
            files={"file": ("noext", BytesIO(b"data"), "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_upload_empty_filename_fails(self):
        resp = client.post(
            "/files/upload",
            files={"file": ("", BytesIO(b"data"), "text/plain")},
        )
        assert resp.status_code in (400, 422)


class TestGetFile:
    def test_upload_then_get(self):
        upload_resp = client.post(
            "/files/upload",
            files={"file": ("gettest.txt", BytesIO(b"content here"), "text/plain")},
        )
        fid = upload_resp.json()["id"]
        resp = client.get(f"/files/{fid}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["text"] == "content here"
        assert body["filename"] == "gettest.txt"

    def test_get_nonexistent_file(self):
        resp = client.get("/files/nonexistent_id_12345")
        assert resp.status_code == 404


class TestDeleteFile:
    def test_upload_then_delete(self):
        upload_resp = client.post(
            "/files/upload",
            files={"file": ("del.txt", BytesIO(b"delete me"), "text/plain")},
        )
        fid = upload_resp.json()["id"]
        resp = client.delete(f"/files/{fid}")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "deleted"

        get_resp = client.get(f"/files/{fid}")
        assert get_resp.status_code == 404

    def test_delete_nonexistent(self):
        resp = client.delete("/files/nonexistent_999")
        assert resp.status_code == 404


class TestSearchFiles:
    def test_search_no_results(self):
        resp = client.get("/files/search?q=nonexistent")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_search_after_upload(self):
        client.post(
            "/files/upload",
            files={"file": ("report.pdf", BytesIO(b"data"), "application/pdf")},
        )
        resp = client.get("/files/search?q=report")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 1

    def test_search_requires_query(self):
        resp = client.get("/files/search")
        assert resp.status_code == 422


class TestListFilesAfterUpload:
    def test_list_shows_uploaded(self):
        client.post(
            "/files/upload",
            files={"file": ("listed.txt", BytesIO(b"listed"), "text/plain")},
        )
        resp = client.get("/files")
        body = resp.json()
        assert body["total"] >= 1

    def test_list_sort_order(self):
        client.post(
            "/files/upload",
            files={"file": ("a.txt", BytesIO(b"a"), "text/plain")},
        )
        client.post(
            "/files/upload",
            files={"file": ("b.txt", BytesIO(b"b"), "text/plain")},
        )
        resp = client.get("/files?sort=uploaded_at&order=desc")
        body = resp.json()
        assert body["total"] >= 2
        times = [f["uploaded_at"] for f in body["files"]]
        assert times == sorted(times, reverse=True)
