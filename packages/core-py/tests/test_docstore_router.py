"""Tests for the docstore API router (routers/docstore.py).

Covers: list_docs, get_doc, put_doc, patch_doc, delete_doc, clear_collection, bulk_put.
MogDB is mocked to avoid filesystem side effects.
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

from routers.docstore import DocStoreRouter, _strip_meta, COLLECTIONS


def _mock_collection():
    coll = MagicMock()
    coll.find.return_value = []
    coll.find_one.return_value = None
    coll.insert_one.return_value = None
    coll.delete_one.return_value = True
    coll.update_one.return_value = 0
    coll.drop.return_value = None
    return coll


def _app() -> tuple[FastAPI, MagicMock]:
    coll = _mock_collection()
    mock_db = MagicMock()
    mock_db.collection.return_value = coll

    dsr = DocStoreRouter()
    app = FastAPI()
    app.include_router(dsr.router)
    from infrastructure.exception_handlers import register_all_handlers
    register_all_handlers(app)
    return app, coll


class TestStripMeta:
    def test_strips_internal_fields(self):
        doc = {"_id": "x", "_created": 1, "_updated": 2, "name": "hello"}
        assert _strip_meta(doc) == {"name": "hello"}

    def test_empty_doc(self):
        assert _strip_meta({}) == {}

    def test_no_meta_fields(self):
        doc = {"a": 1, "b": 2}
        assert _strip_meta(doc) == {"a": 1, "b": 2}


class TestCollections:
    def test_all_expected_collections(self):
        expected = {"sessions", "pendingMessages", "knowledge", "bookmarks",
                    "prompts", "drafts", "kv", "errors"}
        assert COLLECTIONS == expected


class TestListDocs:
    @patch("routers.docstore._get_db")
    def test_list_empty(self, mock_get_db):
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_coll.find.return_value = []
        mock_db.collection.return_value = mock_coll
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        resp = client.get("/docstore/sessions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["data"] == []

    @patch("routers.docstore._get_db")
    def test_list_with_docs(self, mock_get_db):
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_coll.find.return_value = [
            {"_id": "1", "name": "a", "_created": 1},
            {"_id": "2", "name": "b", "_created": 2},
        ]
        mock_db.collection.return_value = mock_coll
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        resp = client.get("/docstore/sessions")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert len(data) == 2
        assert data[0]["name"] == "a"
        assert "_id" not in data[0]

    @patch("routers.docstore._get_db")
    def test_list_unknown_collection(self, mock_get_db):
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        resp = client.get("/docstore/nonexistent")
        assert resp.status_code in (400, 404, 422)

    @patch("routers.docstore._get_db")
    def test_list_with_sort(self, mock_get_db):
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_coll.find.return_value = [{"_id": "1"}]
        mock_db.collection.return_value = mock_coll
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        resp = client.get("/docstore/sessions?sort=name&dir=1&limit=5")
        assert resp.status_code == 200
        mock_coll.find.assert_called_with(sort=[("name", 1)], limit=5)


class TestGetDoc:
    @patch("routers.docstore._get_db")
    def test_get_existing(self, mock_get_db):
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_coll.find_one.return_value = {"_id": "d1", "text": "hello", "_created": 1}
        mock_db.collection.return_value = mock_coll
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        resp = client.get("/docstore/sessions/d1")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["text"] == "hello"
        assert "_id" not in data

    @patch("routers.docstore._get_db")
    def test_get_missing(self, mock_get_db):
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_coll.find_one.return_value = None
        mock_db.collection.return_value = mock_coll
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        resp = client.get("/docstore/sessions/nope")
        assert resp.status_code == 200
        assert resp.json()["data"] is None


class TestPutDoc:
    @patch("routers.docstore._get_db")
    def test_put_creates_new(self, mock_get_db):
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_coll.find_one.return_value = None
        mock_db.collection.return_value = mock_coll
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        resp = client.put("/docstore/sessions/s1", json={"name": "session1"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["id"] == "s1"
        assert data["created"] is True
        mock_coll.insert_one.assert_called_once()

    @patch("routers.docstore._get_db")
    def test_put_replaces_existing(self, mock_get_db):
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_coll.find_one.return_value = {"_id": "s1"}
        mock_db.collection.return_value = mock_coll
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        resp = client.put("/docstore/sessions/s1", json={"name": "updated"})
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["created"] is False
        mock_coll.delete_one.assert_called_with({"_id": "s1"})
        mock_coll.insert_one.assert_called_once()


class TestPatchDoc:
    @patch("routers.docstore._get_db")
    def test_patch_updates_fields(self, mock_get_db):
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_coll.update_one.return_value = 1
        mock_db.collection.return_value = mock_coll
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        resp = client.patch("/docstore/sessions/s1", json={"name": "new"})
        assert resp.status_code == 200
        assert resp.json()["data"]["modified"] == 1
        mock_coll.update_one.assert_called_with({"_id": "s1"}, {"$set": {"name": "new"}})

    @patch("routers.docstore._get_db")
    def test_patch_strips_id_from_body(self, mock_get_db):
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_coll.update_one.return_value = 0
        mock_db.collection.return_value = mock_coll
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        resp = client.patch("/docstore/sessions/s1", json={"_id": "ignore", "x": 1})
        assert resp.status_code == 200
        mock_coll.update_one.assert_called_with({"_id": "s1"}, {"$set": {"x": 1}})

    @patch("routers.docstore._get_db")
    def test_patch_empty_body_returns_zero(self, mock_get_db):
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        resp = client.patch("/docstore/sessions/s1", json={"_id": "only"})
        assert resp.status_code == 200
        assert resp.json()["data"]["modified"] == 0


class TestDeleteDoc:
    @patch("routers.docstore._get_db")
    def test_delete_existing(self, mock_get_db):
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_coll.delete_one.return_value = True
        mock_db.collection.return_value = mock_coll
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        resp = client.delete("/docstore/sessions/s1")
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] is True

    @patch("routers.docstore._get_db")
    def test_delete_nonexistent(self, mock_get_db):
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_coll.delete_one.return_value = False
        mock_db.collection.return_value = mock_coll
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        resp = client.delete("/docstore/sessions/nope")
        assert resp.status_code == 200
        assert resp.json()["data"]["deleted"] is False


class TestClearCollection:
    @patch("routers.docstore._get_db")
    def test_clear(self, mock_get_db):
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_db.collection.return_value = mock_coll
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        resp = client.delete("/docstore/sessions")
        assert resp.status_code == 200
        assert resp.json()["data"]["cleared"] is True
        mock_coll.drop.assert_called_once()


class TestBulkPut:
    @patch("routers.docstore._get_db")
    def test_bulk_put_inserts(self, mock_get_db):
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_coll.find_one.return_value = None
        mock_db.collection.return_value = mock_coll
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        docs = [{"id": "b1", "val": 1}, {"id": "b2", "val": 2}]
        resp = client.post("/docstore/sessions/bulk", json={"docs": docs})
        assert resp.status_code == 200
        assert resp.json()["data"]["imported"] == 2
        assert mock_coll.insert_one.call_count == 2

    @patch("routers.docstore._get_db")
    def test_bulk_put_replaces_existing(self, mock_get_db):
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_coll.find_one.return_value = {"_id": "b1"}
        mock_db.collection.return_value = mock_coll
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        resp = client.post("/docstore/sessions/bulk", json={"docs": [{"id": "b1", "x": 1}]})
        assert resp.status_code == 200
        assert resp.json()["data"]["imported"] == 1
        mock_coll.delete_one.assert_called_with({"_id": "b1"})
        mock_coll.insert_one.assert_called_once()

    @patch("routers.docstore._get_db")
    def test_bulk_put_skips_no_id(self, mock_get_db):
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_db.collection.return_value = mock_coll
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        resp = client.post("/docstore/sessions/bulk", json={"docs": [{"no_id": True}]})
        assert resp.status_code == 200
        assert resp.json()["data"]["imported"] == 0

    @patch("routers.docstore._get_db")
    def test_bulk_put_skips_non_dict(self, mock_get_db):
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_db.collection.return_value = mock_coll
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        resp = client.post("/docstore/sessions/bulk", json={"docs": ["bad", 42]})
        assert resp.status_code == 200
        assert resp.json()["data"]["imported"] == 0

    @patch("routers.docstore._get_db")
    def test_bulk_put_empty_docs(self, mock_get_db):
        mock_db = MagicMock()
        mock_coll = MagicMock()
        mock_db.collection.return_value = mock_coll
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        resp = client.post("/docstore/sessions/bulk", json={"docs": []})
        assert resp.status_code == 200
        assert resp.json()["data"]["imported"] == 0

    @patch("routers.docstore._get_db")
    def test_bulk_put_missing_docs_key(self, mock_get_db):
        mock_db = MagicMock()
        mock_get_db.return_value = mock_db

        app, _ = _app()
        client = TestClient(app)
        resp = client.post("/docstore/sessions/bulk", json={"wrong": []})
        assert resp.status_code in (400, 422)
