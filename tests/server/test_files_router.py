"""
Tests for the files router — upload, list, search, get, delete, ingest.
"""

import time
from unittest.mock import patch

import pytest
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
        resp = client.post(
            "/files/upload",
            files={
                "file": ("test.txt", b"hello world", "text/plain"),
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["filename"] == "test.txt"
        assert body["size_bytes"] == 11

    def test_upload_rejects_no_extension(self, client):
        resp = client.post(
            "/files/upload",
            files={
                "file": ("noext", b"content", "text/plain"),
            },
        )
        assert resp.status_code == 400

    def test_upload_with_tags(self, client):
        resp = client.post(
            "/files/upload",
            files={
                "file": ("doc.md", b"# hello", "text/markdown"),
            },
            data={"tags": '["docs", "markdown"]'},
        )
        assert resp.status_code == 200
        assert resp.json()["chars"] == 0

    def test_upload_json_file(self, client):
        resp = client.post(
            "/files/upload",
            files={
                "file": ("data.json", b'{"key": "value"}', "application/json"),
            },
        )
        assert resp.status_code == 200
        assert resp.json()["filename"] == "data.json"

    def test_upload_python_file(self, client):
        resp = client.post(
            "/files/upload",
            files={
                "file": ("script.py", b"print('hello')", "text/x-python"),
            },
        )
        assert resp.status_code == 200

    def test_upload_csv_file(self, client):
        resp = client.post(
            "/files/upload",
            files={
                "file": ("data.csv", b"a,b,c\n1,2,3", "text/csv"),
            },
        )
        assert resp.status_code == 200


class TestSearchFiles:
    """GET /files/search"""

    def test_search_by_name(self, client):
        client.post(
            "/files/upload",
            files={
                "file": ("my_document.txt", b"content", "text/plain"),
            },
        )
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
        upload = client.post(
            "/files/upload",
            files={
                "file": ("readable.txt", b"hello", "text/plain"),
            },
        )
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
        upload = client.post(
            "/files/upload",
            files={
                "file": ("to_delete.txt", b"delete me", "text/plain"),
            },
        )
        file_id = upload.json()["id"]
        resp = client.delete(f"/files/{file_id}")
        assert resp.status_code == 200

    def test_list_after_upload(self, client):
        client.post(
            "/files/upload",
            files={
                "file": ("listed.txt", b"content", "text/plain"),
            },
        )
        resp = client.get("/files")
        assert resp.json()["total"] >= 1

    def test_get_after_delete(self, client):
        upload = client.post(
            "/files/upload",
            files={
                "file": ("temp.txt", b"temp", "text/plain"),
            },
        )
        file_id = upload.json()["id"]
        client.delete(f"/files/{file_id}")
        resp = client.get(f"/files/{file_id}")
        assert resp.status_code == 404


class TestIngestFile:
    """POST /files/{file_id}/ingest"""

    def test_ingest_nonexistent_file(self, client):
        resp = client.post("/files/nonexistent/ingest")
        assert resp.status_code == 404

    @patch("domain.cognition._internal.rag_service.get_rag_service")
    def test_ingest_existing_file(self, mock_get_rag, client):
        rag = mock_get_rag.return_value
        rag.add_document.return_value = ["chunk1"]
        upload = client.post(
            "/files/upload",
            files={
                "file": ("ingest.txt", b"some content for knowledge base", "text/plain"),
            },
        )
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
        resp = client.post(
            "/files/upload",
            files={
                "file": ("", b"content", "text/plain"),
            },
        )
        assert resp.status_code == 422

    def test_upload_dotfile_has_extension(self, client):
        resp = client.post(
            "/files/upload",
            files={
                "file": (".env", b"SECRET=1", "text/plain"),
            },
        )
        assert resp.status_code == 200
        assert resp.json()["filename"] == ".env"

    def test_upload_uppercase_extension(self, client):
        resp = client.post(
            "/files/upload",
            files={
                "file": ("DATA.TXT", b"hello", "text/plain"),
            },
        )
        assert resp.status_code == 200
        assert resp.json()["filename"] == "DATA.TXT"

    def test_upload_invalid_tags_ignored(self, client):
        resp = client.post(
            "/files/upload",
            files={
                "file": ("tagged.txt", b"hello", "text/plain"),
            },
            data={"tags": "not-json"},
        )
        assert resp.status_code == 200

    def test_upload_stores_actual_content(self, client):
        payload = b"x" * 100
        resp = client.post(
            "/files/upload",
            files={
                "file": ("big.txt", payload, "text/plain"),
            },
        )
        assert resp.json()["size_bytes"] == 100


class TestListSorting:
    def test_list_tag_filter(self, client):
        client.post(
            "/files/upload",
            files={
                "file": ("filtered.txt", b"c", "text/plain"),
            },
            data={"tags": '["keep"]'},
        )
        resp = client.get("/files?tag=keep")
        assert resp.json()["total"] >= 1
        resp2 = client.get("/files?tag=absent")
        assert resp2.json()["total"] == 0

    def test_list_asc_order(self, client):
        client.post(
            "/files/upload",
            files={
                "file": ("asc.txt", b"c", "text/plain"),
            },
        )
        resp = client.get("/files?order=asc")
        assert resp.status_code == 200


class TestSearchEdgeCases:
    def test_search_empty_query_422(self, client):
        resp = client.get("/files/search")
        assert resp.status_code == 422

    def test_search_case_insensitive(self, client):
        client.post(
            "/files/upload",
            files={
                "file": ("MyReport.TXT", b"content", "text/plain"),
            },
        )
        resp = client.get("/files/search?q=myreport")
        assert resp.json()["total"] >= 1

    def test_search_by_tag(self, client):
        client.post(
            "/files/upload",
            files={
                "file": ("tagged_search.txt", b"content", "text/plain"),
            },
            data={"tags": '["searchable"]'},
        )
        resp = client.get("/files/search?q=tagged&tag=searchable")
        assert resp.json()["total"] >= 1
        resp2 = client.get("/files/search?q=tagged&tag=nope")
        assert resp2.json()["total"] == 0


class TestGetFileDetail:
    def test_get_returns_text_content(self, client):
        upload = client.post(
            "/files/upload",
            files={
                "file": ("content.txt", b"Hello World", "text/plain"),
            },
        )
        file_id = upload.json()["id"]
        resp = client.get(f"/files/{file_id}")
        assert resp.json()["text"] == "Hello World"

    def test_get_returns_tags(self, client):
        upload = client.post(
            "/files/upload",
            files={
                "file": ("tagged2.txt", b"c", "text/plain"),
            },
            data={"tags": '["alpha", "beta"]'},
        )
        file_id = upload.json()["id"]
        resp = client.get(f"/files/{file_id}")
        assert resp.json()["tags"] == ["alpha", "beta"]


class TestIngestChunking:
    @patch("domain.cognition._internal.rag_service.get_rag_service")
    def test_ingest_chunks_long_text(self, mock_get_rag, client):
        rag = mock_get_rag.return_value
        rag.add_document.return_value = ["chunk1", "chunk2", "chunk3"]
        long_text = (
            "This is a long sentence that continues. And another one follows it. And a third. "
        ) * 20
        upload = client.post(
            "/files/upload",
            files={
                "file": ("long_ingest.txt", long_text.encode(), "text/plain"),
            },
        )
        file_id = upload.json()["id"]
        resp = client.post(f"/files/{file_id}/ingest")
        assert resp.status_code == 200
        assert resp.json()["facts_stored"] >= 1

    @patch("domain.cognition._internal.rag_service.get_rag_service")
    def test_ingest_empty_text_zero_facts(self, mock_get_rag, client):
        rag = mock_get_rag.return_value
        rag.add_document.return_value = []
        upload = client.post(
            "/files/upload",
            files={
                "file": ("empty_ingest.txt", b"", "text/plain"),
            },
        )
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


# ── Import from URL ────────────────────────────────────────────────────────


class _Origin:
    """Tiny local HTTP origin: counts GETs, optional gzip + start delay."""

    def __init__(self):
        self.count = 0
        self.body = b"hello"
        self.encoding = None
        self.delay = 0.0


@pytest.fixture
def origin():
    import gzip as gzip_mod
    import threading
    import time as time_mod
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    state = _Origin()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state.count += 1
            if state.delay:
                time_mod.sleep(state.delay)
            body = state.body
            if state.encoding == "gzip":
                body = gzip_mod.compress(body)
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            if state.encoding:
                self.send_header("Content-Encoding", state.encoding)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # silence test output
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    state.base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield state
    srv.shutdown()
    srv.server_close()


def _wait_job(client, job_id, timeout=10.0):
    deadline = time.time() + timeout
    while True:
        resp = client.get(f"/files/import/{job_id}")
        assert resp.status_code == 200, resp.text
        job = resp.json()
        if job["state"] != "running":
            return job
        assert time.time() < deadline, f"import never finished: {job}"
        time.sleep(0.05)


class TestImportUrl:
    """POST /files/import + GET /files/import/{job_id} — one-request import."""

    def test_identity_single_request_and_decoded_pct(self, client, origin, monkeypatch):
        """One GET, no HEAD probe; pct lives in decoded (disk) byte-space."""
        monkeypatch.setenv("SLO_FILES_ALLOW_PRIVATE", "1")
        origin.body = b"x" * 50_000
        with client:
            resp = client.post("/files/import", json={"url": f"{origin.base}/data.txt"})
            assert resp.status_code == 200, resp.text
            job = _wait_job(client, resp.json()["job_id"])

        assert job["state"] == "done", job
        assert job["error"] is None
        # decoded/decoded: final total equals the file on disk, not a wire figure
        assert job["done"] == job["total"] == 50_000
        assert job["pct"] == 100.0
        # identity: wire == decoded, nothing saved, single request
        assert origin.count == 1
        assert job["wire_bytes"] == 50_000
        assert job["saved_bytes"] == 0
        # the file landed in the manager (list reports the stored filename)
        listing = client.get("/files/").json()
        assert any(
            f["filename"].endswith(".txt") and f["size_bytes"] == 50_000 for f in listing["files"]
        ), listing

    def test_gzip_origin_saves_wire_and_stays_decoded(
        self, client, origin, monkeypatch, isolated_router
    ):
        """gzip origin: bar counts decoded bytes; savings reported separately."""
        monkeypatch.setenv("SLO_FILES_ALLOW_PRIVATE", "1")
        original = b"compress me " * 4000  # ~48 KB, compresses well
        origin.body = original
        origin.encoding = "gzip"
        with client:
            resp = client.post("/files/import", json={"url": f"{origin.base}/doc.txt"})
            job = _wait_job(client, resp.json()["job_id"])

        assert job["state"] == "done", job
        assert job["done"] == len(original)  # decoded bytes on disk
        assert job["pct"] == 100.0
        assert job["wire_bytes"] < len(original)
        assert job["saved_bytes"] > 0
        assert job["content_encoding"] == "gzip"
        assert origin.count == 1
        # stored file is the decoded body
        stored = [p for p in isolated_router.UPLOADS_DIR.iterdir() if p.name.endswith(".txt")]
        assert stored and stored[0].read_bytes() == original

    def test_running_progress_never_reports_wire_total(self, client, origin, monkeypatch):
        """Mid-run samples: total is decoded or unknown — never the wire size."""
        monkeypatch.setenv("SLO_FILES_ALLOW_PRIVATE", "1")
        original = b"payload " * 6000  # 48 KB
        origin.body = original
        origin.encoding = "gzip"
        origin.delay = 0.2  # hold the response open so we can sample mid-run
        with client:
            resp = client.post("/files/import", json={"url": f"{origin.base}/slow.txt"})
            job_id = resp.json()["job_id"]
            samples = []
            deadline = time.time() + 3.0
            while time.time() < deadline:
                j = client.get(f"/files/import/{job_id}").json()
                samples.append(j)
                if j["state"] != "running":
                    break
                time.sleep(0.05)
            final = _wait_job(client, job_id)

        wire_size = None
        for s in samples:
            # decoded total (or unknown=0) — a wire-space total would equal
            # the gzip length, which is neither 0 nor the decoded length.
            if s["total"] not in (0, len(original)):
                wire_size = s["total"]
            assert 0.0 <= s["pct"] <= 100.0
        assert wire_size is None, f"wire total leaked into progress: {wire_size}"
        assert final["state"] == "done"

    def test_rejects_non_http_scheme(self, client):
        resp = client.post("/files/import", json={"url": "ftp://host/x.txt"})
        assert resp.status_code == 400

    def test_rejects_private_address(self, client, monkeypatch):
        monkeypatch.delenv("SLO_FILES_ALLOW_PRIVATE", raising=False)
        resp = client.post("/files/import", json={"url": "http://127.0.0.1:1/x.txt"})
        assert resp.status_code == 403

    def test_rejects_url_without_extension(self, client, monkeypatch):
        monkeypatch.setenv("SLO_FILES_ALLOW_PRIVATE", "1")
        resp = client.post("/files/import", json={"url": "http://127.0.0.1:1/noext"})
        assert resp.status_code == 400

    def test_rejects_empty_url(self, client):
        resp = client.post("/files/import", json={"url": "  "})
        assert resp.status_code == 400

    def test_size_cap_aborts_and_cleans_up(self, client, origin, monkeypatch, isolated_router):
        monkeypatch.setenv("SLO_FILES_ALLOW_PRIVATE", "1")
        monkeypatch.setattr("apps.api.server.routers.files.MAX_IMPORT_SIZE", 1024)
        origin.body = b"y" * 64_000
        with client:
            resp = client.post("/files/import", json={"url": f"{origin.base}/big.txt"})
            job = _wait_job(client, resp.json()["job_id"])

        assert job["state"] == "error", job
        assert "exceeds" in (job["error"] or "")
        # no partial body, no .sgpart left behind
        leftovers = [p for p in isolated_router.UPLOADS_DIR.iterdir() if p.name != "_metadata.json"]
        assert leftovers == [], leftovers

    def test_unknown_job_is_404(self, client):
        resp = client.get("/files/import/nope")
        assert resp.status_code == 404

    def test_import_route_registered_before_file_id_catchall(self, client):
        # GET /files/import has no GET route → falls to /{file_id} → 404, not 500
        resp = client.get("/files/import")
        assert resp.status_code == 404

    def test_job_registry_stays_bounded(self):
        """_job_put prunes terminal jobs past _IMPORT_JOB_KEEP; running stay."""
        from apps.api.server.routers import files as files_mod

        saved = dict(files_mod._IMPORT_JOBS)
        try:
            files_mod._IMPORT_JOBS.clear()

            def _job(jid: str, state: str) -> dict:
                return {
                    "job_id": jid,
                    "state": state,
                    "filename": "x.txt",
                    "done": 0,
                    "total": 0,
                    "file_id": None,
                    "wire_bytes": None,
                    "saved_bytes": None,
                    "content_encoding": "identity",
                    "error": None,
                }

            running_id = "keep-running"
            files_mod._IMPORT_JOBS[running_id] = _job(running_id, "running")
            for i in range(files_mod._IMPORT_JOB_KEEP + 10):
                FilesRouter._job_put(f"bench-{i}", _job(f"bench-{i}", "done"))

            assert len(files_mod._IMPORT_JOBS) <= files_mod._IMPORT_JOB_KEEP
            assert running_id in files_mod._IMPORT_JOBS  # running jobs never evicted
        finally:
            files_mod._IMPORT_JOBS.clear()
            files_mod._IMPORT_JOBS.update(saved)
