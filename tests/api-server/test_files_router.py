"""Tests for files router endpoints."""

from io import BytesIO

import pytest
import routers.files as files_mod
from routers.files import FilesRouter
from test_support import get_test_client

client = get_test_client()


def _app_files_router() -> FilesRouter:
    """The live FilesRouter whose routes the app serves.

    ``routers.files`` builds it at import (``router = FilesRouter().router``)
    and keeps only the APIRouter, so the instance is reached through its
    routes' bound endpoints.
    """
    for route in getattr(files_mod.router, "routes", []):
        instance = getattr(getattr(route, "endpoint", None), "__self__", None)
        if isinstance(instance, FilesRouter):
            return instance
    raise AssertionError("app-side FilesRouter instance not found — isolation would be silent")


@pytest.fixture(autouse=True)
def _isolated_uploads(tmp_path, monkeypatch):
    """Point every FilesRouter at a per-test tmp uploads dir (card c130b2aa).

    The old fixture captured the router's real ``<repo>/data/uploads`` path at
    module import — before any isolation applies — and unlinked its contents on
    teardown, so any run of this suite wiped live top-level upload files, while
    the tests themselves wrote bytes into the real dir through the app-side
    instance. Both halves go to ``tmp_path`` now: the metadata store was
    already isolated by the root conftest's ``SLO_DATA_DIR``, and teardown
    touches nothing (tmp cleanup is pytest's). This mirrors the instance-patch
    isolation tests/server and tests/core-py already use.
    """
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(_app_files_router(), "UPLOADS_DIR", upload_dir)
    yield


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
        assert resp.json()["data"]["deleted"] == fid

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
