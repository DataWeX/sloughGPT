"""Tests for the files API router (routers/files.py).

Covers: list_files, upload_file, search_files, get_file, delete_file, ingest_file.
File I/O is done via tmp_path; only HTTP-level behavior is tested.
"""

from __future__ import annotations

import io
import sys
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

# Bind domain.cognition into the attr chain: py3.12 mock.patch resolves targets via
# pkgutil.resolve_name (getattr-only), so the lazy package chain must be materialised.
import domain.cognition  # noqa: E402,F401
from domain.infrastructure.asset_store import AssetStore  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_files_collection():
    """Empty the worktree-local files collection before each test.

    Rows persist in the file-backed store across tests within a session (and a
    fresh worktree adopts main's data/uploads_mogdb on first use), so the
    list/total assertions need a clean slate. Only THIS repo's store is
    touched: db_pool resolves data/ from this tree; the legacy main-checkout
    dir is read-only there (copied in, never written).
    """
    from infrastructure.db_pool import get_db

    try:
        get_db("uploads_mogdb").collection("files").delete_many({})
    except Exception:
        pass
    yield


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
    fr._store = AssetStore(tmp_path / "assets")
    fr._register_routes()
    return fr


class TestUpload:
    def test_upload_txt_file(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        resp = client.post(
            "/files/upload", files={"file": ("test.txt", io.BytesIO(b"hello"), "text/plain")}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["size_bytes"] == 5
        assert data["filename"].endswith(".txt")

    def test_upload_no_extension(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        resp = client.post(
            "/files/upload", files={"file": ("noext", io.BytesIO(b"data"), "text/plain")}
        )
        assert resp.status_code == 400

    def test_upload_with_tags(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        resp = client.post(
            "/files/upload",
            files={"file": ("tagged.txt", io.BytesIO(b"content"), "text/plain")},
            data={"tags": '["important"]'},
        )
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
        client.post(
            "/files/upload", files={"file": ("hello.txt", io.BytesIO(b"hello world"), "text/plain")}
        )
        resp = client.get("/files/search?q=xyz")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_search_match(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        client.post(
            "/files/upload", files={"file": ("hello.txt", io.BytesIO(b"hello world"), "text/plain")}
        )
        resp = client.get("/files/search?q=hello")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1


class TestGetFile:
    def test_get_existing(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        r = client.post(
            "/files/upload", files={"file": ("doc.txt", io.BytesIO(b"test content"), "text/plain")}
        )
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
        r = client.post(
            "/files/upload", files={"file": ("del.txt", io.BytesIO(b"bye"), "text/plain")}
        )
        fid = r.json()["id"]
        resp = client.delete(f"/files/{fid}")
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] == fid

    def test_delete_nonexistent(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        resp = client.delete("/files/nonexistent")
        assert resp.status_code == 404


class TestIngest:
    def test_ingest_file(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        long_text = (
            b"This is a knowledge text chunk that is definitely longer than twenty characters."
        )
        r = client.post(
            "/files/upload", files={"file": ("ingest.txt", io.BytesIO(long_text), "text/plain")}
        )
        fid = r.json()["id"]
        with patch("domain.cognition.get_rag_service") as mock_rag:
            mock_rag.return_value.add_document.return_value = ["c1"]
            resp = client.post(f"/files/{fid}/ingest")
        assert resp.status_code == 200
        assert resp.json()["facts_stored"] == 1


class TestAssetStoreBacking:
    """Slice 2: uploads land in the content-addressed store; legacy rows adopted."""

    def test_upload_goes_to_store_not_uploads_dir(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        r = client.post(
            "/files/upload", files={"file": ("cas.txt", io.BytesIO(b"cas payload"), "text/plain")}
        )
        assert r.status_code == 200
        fid = r.json()["id"]
        # Bytes live in the CAS (one blob, sha256 layout)...
        blobs = [p for p in (tmp_path / "assets" / "blobs").rglob("*") if p.is_file()]
        assert len(blobs) == 1
        # ...nothing was written to the legacy uploads layout...
        assert not any(p.is_file() and p.suffix == ".txt" for p in tmp_path.iterdir())
        # ...and content resolves through the store.
        detail = client.get(f"/files/{fid}")
        assert detail.status_code == 200
        assert detail.json()["text"] == "cas payload"

    def test_legacy_row_adopted_and_backfilled(self, tmp_path):
        fr = _make_router(tmp_path)
        # Legacy layout: bytes on disk, metadata row WITHOUT asset_id.
        (tmp_path / "1717_legacy.txt").write_text("legacy bytes")
        fr._upsert_file_metadata(
            "1717_legacy",
            {
                "filename": "1717_legacy.txt",
                "original_name": "legacy.txt",
                "extension": ".txt",
                "size_bytes": 11,
                "chars": 0,
                "pages": 1,
                "uploaded_at": 1717.0,
                "tags": ["old"],
            },
        )
        client = TestClient(_app(fr))
        listed = client.get("/files")
        assert listed.status_code == 200
        assert listed.json()["total"] == 1
        # Row backfilled with store ids; original untouched (copy-not-move).
        row = fr._load_metadata()["1717_legacy"]
        assert row["asset_id"]
        assert row["sha256"]
        assert (tmp_path / "1717_legacy.txt").read_text() == "legacy bytes"
        # Content resolves through the store; download lands in history.
        detail = client.get("/files/1717_legacy")
        assert detail.status_code == 200
        assert detail.json()["text"] == "legacy bytes"
        actions = {e["action"] for e in fr._store.history(asset_id=row["asset_id"])}
        assert {"put", "migrate", "download"} <= actions
        # Cleanup the row (polite to any later consumer of the shared collection).
        assert client.delete("/files/1717_legacy").status_code == 200

    def test_history_endpoint(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        r = client.post(
            "/files/upload", files={"file": ("hist.txt", io.BytesIO(b"hist"), "text/plain")}
        )
        fid = r.json()["id"]
        client.get(f"/files/{fid}")  # records a download event
        resp = client.get("/files/history")
        assert resp.status_code == 200
        actions = [e["action"] for e in resp.json()["events"]]
        assert "put" in actions and "download" in actions
        assert actions.index("download") < actions.index("put")  # newest first
        # Action filter narrows the stream.
        only_dl = client.get("/files/history?action=download").json()["events"]
        assert only_dl and all(e["action"] == "download" for e in only_dl)
        # "/files/history" is a literal route, not a {file_id} match.
        assert all("text" not in e for e in only_dl)

    def test_get_file_etag_304(self, tmp_path):
        fr = _make_router(tmp_path)
        client = TestClient(_app(fr))
        r = client.post(
            "/files/upload",
            files={"file": ("etag.txt", io.BytesIO(b"etag me"), "text/plain")},
        )
        fid = r.json()["id"]
        first = client.get(f"/files/{fid}")
        assert first.status_code == 200
        etag = first.headers.get("etag")
        assert etag  # strong ETag from the content sha256
        # Conditional repeat: 304 with no body, and no download event recorded.
        before = len(fr._store.history(asset_id=fr._load_metadata()[fid]["asset_id"]))
        second = client.get(f"/files/{fid}", headers={"If-None-Match": etag})
        assert second.status_code == 304
        assert not second.content
        after = len(fr._store.history(asset_id=fr._load_metadata()[fid]["asset_id"]))
        assert after == before
        # A different validator -> normal 200.
        third = client.get(f"/files/{fid}", headers={"If-None-Match": '"stale"'})
        assert third.status_code == 200
        assert third.json()["text"] == "etag me"
