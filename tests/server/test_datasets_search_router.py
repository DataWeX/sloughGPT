"""
Tests for the federated dataset search router — descriptor projection,
partial-failure semantics, validation, and the sync/async seam.
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.infrastructure.exception_handlers import register_all_handlers
from apps.api.server.routers.datasets_search import SPEC, router

_PROVIDERS = [
    {"source": "github", "ok": True, "count": 1, "error": None, "elapsed_ms": 50},
    {
        "source": "kaggle",
        "ok": False,
        "count": 0,
        "error": "Kaggle CLI not installed",
        "elapsed_ms": 2,
    },
]

_RESULTS = [
    {"source": "github", "id": "u/r", "title": "u/r", "description": None, "stats": {}},
]


@pytest.fixture
def client():
    _app = FastAPI()
    register_all_handlers(_app)
    _app.include_router(router)
    return TestClient(_app, raise_server_exceptions=False)


class TestFederatedSearchRoute:
    def test_serves_merged_results_and_provider_status(self, client):
        with patch(
            "apps.api.server.routers.datasets_search.FederatedSearch.search",
            return_value={"results": _RESULTS, "providers": _PROVIDERS},
        ):
            resp = client.get("/datasets/search/all", params={"q": "shakespeare"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["results"] == _RESULTS
        assert body["data"]["providers"] == _PROVIDERS

    def test_passes_sources_subset_through(self, client):
        with patch(
            "apps.api.server.routers.datasets_search.FederatedSearch.search",
            return_value={"results": [], "providers": []},
        ) as mock_search:
            resp = client.get(
                "/datasets/search/all",
                params={"q": "x", "sources": "github, huggingface"},
            )
        assert resp.status_code == 200
        # hook calls searcher(q, limit, sources) through to_thread — positional args
        assert mock_search.call_args.args[2] == ["github", "huggingface"]

    def test_missing_q_is_422(self, client):
        resp = client.get("/datasets/search/all")
        assert resp.status_code == 422

    def test_limit_must_be_integer(self, client):
        resp = client.get("/datasets/search/all", params={"q": "x", "limit": "abc"})
        assert resp.status_code == 422

    def test_provider_failure_is_partial_not_fatal(self, client):
        """A failing provider must surface as ok=False, never a 500."""
        with patch(
            "apps.api.server.routers.datasets_search.FederatedSearch.search",
            return_value={
                "results": [],
                "providers": [
                    {"source": "github", "ok": False, "count": 0, "error": "boom", "elapsed_ms": 1},
                ],
            },
        ):
            resp = client.get("/datasets/search/all", params={"q": "x"})
        assert resp.status_code == 200
        assert resp.json()["data"]["providers"][0]["ok"] is False


class TestFederatedSearchSpec:
    def test_descriptor_is_idempotent_get(self):
        """Discovery is a read — the derived verb must be GET."""
        from apps.api.server.infrastructure.contract import derive_method

        assert SPEC.idempotent is True
        assert derive_method(SPEC) == "GET"

    def test_query_is_required_in_descriptor(self):
        required = [p.name for p in SPEC.parameters if p.required]
        assert required == ["q"]

    def test_projected_route_is_registered(self):
        served = {(route.path, next(iter(route.methods))) for route in router.routes}
        assert ("/datasets/search/all", "GET") in served
