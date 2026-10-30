"""Tests for the descriptor-projected crawl ingest route — ``datasets.import_crawl``.

``POST /datasets/import/crawl`` (card: server crawl ingest — server well first).
Covers the wire contract (envelope, error codes), the durable store layout
(``raw/crawl_<utc>.jsonl`` + appended ``corpus.jsonl``), per-name import locks
and the ContractRegistry row. Domain-level store behaviour lives in
``tests/test_crawl_store.py``.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.contract import REGISTRY
from infrastructure.exception_handlers import register_app_error_handler
from routers import datasets_crawl


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets_crawl, "DATASETS_DIR", tmp_path)
    monkeypatch.setattr(datasets_crawl, "_import_locks", {})
    _app = FastAPI()
    register_app_error_handler(_app)
    _app.include_router(datasets_crawl.router)
    return TestClient(_app, raise_server_exceptions=False)


def _page(url="https://example.com/a", text="hello world", **extra):
    base = {"url": url, "text": text}
    base.update(extra)
    return base


class TestImportCrawl:
    def test_route_is_registered(self, client):
        paths = client.app.openapi()["paths"]
        assert "/datasets/import/crawl" in paths
        assert "post" in paths["/datasets/import/crawl"]

    def test_stores_valid_pages(self, client, tmp_path):
        resp = client.post(
            "/datasets/import/crawl",
            json={"name": "site_a", "pages": [_page(), _page(url="https://b.com")]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        data = body["data"]
        assert data["dataset_id"] == "site_a"
        assert data["pages_received"] == 2
        assert data["pages_stored"] == 2
        assert data["pages_dropped"] == 0
        assert (tmp_path / "site_a" / "raw").is_dir()
        corpus = (tmp_path / "site_a" / "corpus.jsonl").read_text(encoding="utf-8")
        assert json.loads(corpus.splitlines()[0])["content"] == "hello world"

    def test_jsonl_body_accepted(self, client, tmp_path):
        raw = json.dumps(_page()) + "\n" + json.dumps(_page(url="https://b.com")) + "\n"
        resp = client.post("/datasets/import/crawl", json={"name": "site_a", "jsonl": raw})
        assert resp.status_code == 200
        assert resp.json()["data"]["pages_stored"] == 2

    def test_pages_and_jsonl_merge(self, client):
        raw = json.dumps(_page(url="https://j.com")) + "\n"
        resp = client.post(
            "/datasets/import/crawl",
            json={"name": "site_a", "pages": [_page()], "jsonl": raw},
        )
        assert resp.status_code == 200
        assert resp.json()["data"]["pages_received"] == 2

    def test_invalid_pages_counted_not_fatal(self, client):
        resp = client.post(
            "/datasets/import/crawl",
            json={"name": "site_a", "pages": [_page(), {"text": "no url"}, "junk"]},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["pages_stored"] == 1
        assert data["pages_dropped"] == 2

    def test_all_invalid_is_400(self, client):
        resp = client.post(
            "/datasets/import/crawl", json={"name": "site_a", "pages": [{"text": "x"}]}
        )
        assert resp.status_code == 400
        assert "no usable pages" in resp.json()["error"]

    def test_empty_request_is_400(self, client):
        resp = client.post("/datasets/import/crawl", json={"name": "site_a"})
        assert resp.status_code == 400

    def test_empty_pages_list_is_400(self, client):
        resp = client.post("/datasets/import/crawl", json={"name": "site_a", "pages": []})
        assert resp.status_code == 400

    def test_invalid_name_is_400(self, client):
        resp = client.post("/datasets/import/crawl", json={"name": "bad/name", "pages": [_page()]})
        assert resp.status_code == 400
        assert "invalid dataset name" in resp.json()["error"]

    def test_bad_jsonl_is_400(self, client):
        resp = client.post(
            "/datasets/import/crawl", json={"name": "site_a", "jsonl": "{not json\n"}
        )
        assert resp.status_code == 400
        assert "line 1" in resp.json()["error"]

    def test_missing_name_is_422(self, client):
        resp = client.post("/datasets/import/crawl", json={"pages": [_page()]})
        assert resp.status_code == 422

    def test_lock_contention_is_409(self, client, monkeypatch):
        class _HeldLock:
            def locked(self) -> bool:
                return True

            async def __aenter__(self):
                raise AssertionError("must not enter a held lock")

            async def __aexit__(self, *exc):
                return False

        async def _held(name: str):
            return _HeldLock()

        monkeypatch.setattr(datasets_crawl, "_get_import_lock", _held)
        resp = client.post("/datasets/import/crawl", json={"name": "site_a", "pages": [_page()]})
        assert resp.status_code == 409
        assert "in progress" in resp.json()["error"]

    def test_contract_registered(self):
        routes = REGISTRY.routes(prefix="/datasets/import/crawl")
        assert len(routes) == 1
        entry = routes[0]
        assert entry["name"] == "datasets.import_crawl"
        assert entry["method"] == "POST"
        assert entry["path"] == "/datasets/import/crawl"
        assert entry["idempotent"] is False
        assert entry["auth_scope"] == "authenticated"
