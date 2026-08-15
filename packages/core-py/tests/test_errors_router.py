"""Tests for the errors API router (routers/errors.py).

Covers: log, recent, grouped, unread, clear, export.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, _server_dir)
from routers.errors import ErrorsRouter  # noqa: E402


def _app(er: ErrorsRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(er.router)
    return app


def _make_error(msg: str = "test error", source: str = "web") -> dict:
    return {"message": msg, "source": source, "url": "http://localhost", "line": 1, "col": 1}


class TestLogErrors:
    def test_log_single(self):
        er = ErrorsRouter()
        client = TestClient(_app(er))
        resp = client.post("/errors/log", json={"errors": [_make_error()]})
        assert resp.status_code == 200
        assert resp.json()["data"]["logged"] == 1

    def test_log_multiple(self):
        er = ErrorsRouter()
        client = TestClient(_app(er))
        resp = client.post("/errors/log", json={"errors": [_make_error("e1"), _make_error("e2")]})
        assert resp.status_code == 200
        assert resp.json()["data"]["logged"] == 2


class TestRecentErrors:
    def test_empty(self):
        er = ErrorsRouter()
        er._error_buffer.clear()
        er._error_count_since_clear = 0
        client = TestClient(_app(er))
        resp = client.get("/errors/recent")
        assert resp.status_code == 200
        assert resp.json()["data"]["errors"] == []
        assert resp.json()["data"]["total"] == 0

    def test_after_logging(self):
        er = ErrorsRouter()
        er._error_buffer.clear()
        er._error_count_since_clear = 0
        client = TestClient(_app(er))
        client.post("/errors/log", json={"errors": [_make_error("boom")]})
        resp = client.get("/errors/recent")
        assert resp.json()["data"]["total"] == 1
        assert resp.json()["data"]["errors"][0]["message"] == "boom"


class TestGroupedErrors:
    def test_empty(self):
        er = ErrorsRouter()
        er._error_buffer.clear()
        er._error_count_since_clear = 0
        client = TestClient(_app(er))
        resp = client.get("/errors/grouped")
        assert resp.status_code == 200
        assert resp.json()["data"]["groups"] == []

    def test_groups_same_message(self):
        er = ErrorsRouter()
        er._error_buffer.clear()
        er._error_count_since_clear = 0
        client = TestClient(_app(er))
        client.post("/errors/log", json={"errors": [_make_error("same"), _make_error("same")]})
        resp = client.get("/errors/grouped")
        groups = resp.json()["data"]["groups"]
        assert len(groups) == 1
        assert groups[0]["count"] == 2


class TestUnreadCount:
    def test_unread(self):
        er = ErrorsRouter()
        er._error_buffer.clear()
        er._error_count_since_clear = 0
        client = TestClient(_app(er))
        client.post("/errors/log", json={"errors": [_make_error()]})
        resp = client.get("/errors/unread")
        assert resp.status_code == 200
        assert resp.json()["data"]["unread_count"] == 1


class TestClearErrors:
    def test_clear(self):
        er = ErrorsRouter()
        er._error_buffer.clear()
        er._error_count_since_clear = 0
        client = TestClient(_app(er))
        client.post("/errors/log", json={"errors": [_make_error()]})
        resp = client.delete("/errors/clear")
        assert resp.status_code == 200
        resp2 = client.get("/errors/recent")
        assert resp2.json()["data"]["total"] == 0


class TestExportErrors:
    def test_export_empty(self):
        er = ErrorsRouter()
        er._error_buffer.clear()
        er._error_count_since_clear = 0
        client = TestClient(_app(er))
        resp = client.get("/errors/export")
        assert resp.status_code == 200
        assert "errors" in resp.json()
        assert resp.json()["total"] == 0


class TestFingerprint:
    def test_same_message_same_fingerprint(self):
        er = ErrorsRouter()
        assert er._fingerprint("error occurred") == er._fingerprint("error occurred")

    def test_numbers_normalized(self):
        er = ErrorsRouter()
        assert er._fingerprint("error 123 happened") == er._fingerprint("error 456 happened")

    def test_hex_ids_normalized(self):
        er = ErrorsRouter()
        assert er._fingerprint("id abcdef12 failed") == er._fingerprint("id 09876543 failed")

    def test_different_messages_different_fingerprints(self):
        er = ErrorsRouter()
        assert er._fingerprint("error A") != er._fingerprint("error B")


class TestDedup:
    def test_same_message_within_window_bumps_count(self):
        er = ErrorsRouter()
        er._error_buffer.clear()
        er._dedup_map.clear()
        client = TestClient(_app(er))

        client.post("/errors/log", json={"errors": [_make_error("dup")]})
        client.post("/errors/log", json={"errors": [_make_error("dup")]})

        resp = client.get("/errors/recent")
        errors = resp.json()["data"]["errors"]
        assert len(errors) == 1
        assert errors[0]["count"] == 2

    def test_same_message_after_window_creates_new_record(self):
        er = ErrorsRouter()
        er._error_buffer.clear()
        er._dedup_map.clear()
        client = TestClient(_app(er))

        client.post("/errors/log", json={"errors": [_make_error("dup")]})
        er._dedup_map["dup"] = er._dedup_map.get("dup", 0) - 20
        client.post("/errors/log", json={"errors": [_make_error("dup")]})

        resp = client.get("/errors/recent")
        errors = resp.json()["data"]["errors"]
        assert len(errors) == 2

    def test_different_messages_never_deduped(self):
        er = ErrorsRouter()
        er._error_buffer.clear()
        er._dedup_map.clear()
        client = TestClient(_app(er))

        client.post("/errors/log", json={"errors": [_make_error("msg-a"), _make_error("msg-b")]})

        resp = client.get("/errors/recent")
        assert resp.json()["data"]["total"] == 2

    def test_same_message_in_same_batch_not_deduped(self):
        er = ErrorsRouter()
        er._error_buffer.clear()
        er._dedup_map.clear()
        client = TestClient(_app(er))

        resp = client.post("/errors/log", json={
            "errors": [_make_error("dup"), _make_error("dup"), _make_error("dup")]
        })
        assert resp.json()["data"]["logged"] == 3

    def test_clear_resets_dedup_map(self):
        er = ErrorsRouter()
        er._error_buffer.clear()
        er._dedup_map.clear()
        client = TestClient(_app(er))

        client.post("/errors/log", json={"errors": [_make_error("dup")]})
        client.delete("/errors/clear")
        client.post("/errors/log", json={"errors": [_make_error("dup")]})

        resp = client.get("/errors/recent")
        assert resp.json()["data"]["total"] == 1


class TestDedupPruning:
    def test_prunes_old_entries_when_map_exceeds_500(self):
        er = ErrorsRouter()
        er._dedup_map.clear()

        for i in range(501):
            er._dedup_map[f"old-{i}"] = 0.0

        er._dedup_map["fresh"] = 9999999999.0
        er._fingerprint("fresh")

        for i in range(501):
            msg = f"trigger-{i}"
            er._dedup_map[msg] = 0.0

        client = TestClient(_app(er))
        er._error_buffer.clear()
        client.post("/errors/log", json={"errors": [_make_error("fresh")]})
        resp = client.get("/errors/recent")
        assert resp.json()["data"]["total"] >= 1


class TestIngestLogs:
    def test_ingest(self):
        er = ErrorsRouter()
        client = TestClient(_app(er))
        resp = client.post("/errors/logs/ingest", json={
            "logs": [{"level": "info", "logger": "test", "message": "hello"}]
        })
        assert resp.status_code == 200
        assert resp.json()["data"]["ingested"] == 1
