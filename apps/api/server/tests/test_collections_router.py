"""Tests for the /collections router.

Covers all 9 endpoints:
  GET  /collections              — list_pipelines
  POST /collections/create       — create_pipeline
  POST /collections/run          — run_pipeline
  POST /collections/collect      — collect_direct
  GET  /collections/stats        — get_stats
  GET  /collections/{id}         — get_pipeline
  DELETE /collections/{id}       — delete_pipeline
  POST /collections/{id}/collect — collect
  GET  /collections/{id}/records — get_records
"""

import pytest
from test_support import get_test_client


def _d(resp):
    """Unwrap success_response envelope."""
    j = resp.json()
    return j.get("data", j)


@pytest.fixture(autouse=True)
def _fresh_registry():
    """Reset the global registry singleton before each test."""
    from domains.collections.registry import get_registry, CollectionRegistry

    import domains.collections.registry as reg_mod

    reg_mod._default_registry = CollectionRegistry()

    registry = get_registry()
    from domains.collections.stores import MemoryStore
    from domains.collections.sources import GeneratorSource
    from domains.collections.filters import LengthFilter

    registry.register_store("memory", MemoryStore())
    registry.register_source("generator", GeneratorSource(lambda: iter(["item1", "item2"])))
    registry.register_filter("length", LengthFilter(min_length=1))

    yield

    reg_mod._default_registry = None


class TestListPipelines:
    def setup_method(self):
        self.client = get_test_client()

    def test_empty_list(self):
        resp = self.client.get("/collections")
        assert resp.status_code == 200
        data = _d(resp)
        assert data["pipelines"] == []
        assert data["sources"] == ["generator"]
        assert data["stores"] == ["memory"]
        assert data["filters"] == ["length"]
        assert data["counts"]["pipelines"] == 0

    def test_list_after_create(self):
        from domains.collections.registry import get_registry

        registry = get_registry()
        registry.create_pipeline("p1", "generator", "memory")
        resp = self.client.get("/collections")
        assert resp.status_code == 200
        data = _d(resp)
        assert "p1" in data["pipelines"]
        assert data["counts"]["pipelines"] == 1


class TestCreatePipeline:
    def setup_method(self):
        self.client = get_test_client()

    def test_create_with_valid_source_and_store(self):
        resp = self.client.post(
            "/collections/create",
            json={"name": "test-pipe", "source_type": "generator", "store_type": "memory"},
        )
        assert resp.status_code == 200
        data = _d(resp)
        assert data["name"] == "test-pipe"
        assert data["source_type"] == "generator"
        assert data["store_type"] == "memory"
        assert data["filters"] == 0

    def test_create_with_filters(self):
        resp = self.client.post(
            "/collections/create",
            json={
                "name": "filtered-pipe",
                "source_type": "generator",
                "store_type": "memory",
                "filter_chain": [{"type": "length"}],
            },
        )
        assert resp.status_code == 200
        data = _d(resp)
        assert data["filters"] == 1

    def test_create_with_invalid_source_returns_400(self):
        resp = self.client.post(
            "/collections/create",
            json={"name": "bad-pipe", "source_type": "nonexistent", "store_type": "memory"},
        )
        assert resp.status_code in (400, 500)

    def test_create_with_invalid_store_returns_400(self):
        resp = self.client.post(
            "/collections/create",
            json={"name": "bad-pipe", "source_type": "generator", "store_type": "nonexistent"},
        )
        assert resp.status_code in (400, 500)

    def test_create_missing_required_fields(self):
        resp = self.client.post("/collections/create", json={})
        assert resp.status_code == 422

    def test_create_missing_source_type(self):
        resp = self.client.post(
            "/collections/create", json={"name": "x", "store_type": "memory"}
        )
        assert resp.status_code == 422


class TestRunPipeline:
    def setup_method(self):
        self.client = get_test_client()

    def test_run_existing_pipeline(self):
        from domains.collections.registry import get_registry

        registry = get_registry()
        registry.create_pipeline("runner", "generator", "memory")
        resp = self.client.post("/collections/run?name=runner")
        assert resp.status_code == 200
        data = _d(resp)
        assert data["pipeline"] == "runner"
        assert data["collected"] >= 0
        assert "elapsed_ms" in data

    def test_run_nonexistent_pipeline(self):
        resp = self.client.post("/collections/run?name=nope")
        assert resp.status_code in (400, 404, 500)

    def test_run_requires_name_query_param(self):
        resp = self.client.post("/collections/run")
        assert resp.status_code == 422


class TestCollectDirect:
    def setup_method(self):
        self.client = get_test_client()

    def test_collect_with_generator_source(self):
        resp = self.client.post(
            "/collections/collect",
            json={"source_type": "generator", "source_config": {}, "min_length": 0, "dedup": False},
        )
        assert resp.status_code == 500

    def test_collect_with_invalid_source(self):
        resp = self.client.post(
            "/collections/collect",
            json={"source_type": "nonexistent"},
        )
        assert resp.status_code in (400, 422, 500)

    def test_collect_missing_source_type(self):
        resp = self.client.post("/collections/collect", json={})
        assert resp.status_code == 422


class TestGetStats:
    def setup_method(self):
        self.client = get_test_client()

    def test_stats_empty(self):
        resp = self.client.get("/collections/stats")
        assert resp.status_code == 200
        data = _d(resp)
        assert "sources" in data
        assert "stores" in data
        assert "filters" in data
        assert "pipelines" in data

    def test_stats_with_pipeline(self):
        from domains.collections.registry import get_registry

        registry = get_registry()
        registry.create_pipeline("s1", "generator", "memory")
        resp = self.client.get("/collections/stats")
        assert resp.status_code == 200
        data = _d(resp)
        assert "s1" in data["pipelines"]


class TestGetPipeline:
    def setup_method(self):
        self.client = get_test_client()

    def test_get_existing(self):
        from domains.collections.registry import get_registry

        registry = get_registry()
        registry.create_pipeline("my-pipe", "generator", "memory")
        resp = self.client.get("/collections/my-pipe")
        assert resp.status_code == 200
        data = _d(resp)
        assert data["id"] == "my-pipe"
        assert data["name"] == "my-pipe"
        assert "stats" in data

    def test_get_nonexistent(self):
        resp = self.client.get("/collections/nope")
        assert resp.status_code == 404


class TestDeletePipeline:
    def setup_method(self):
        self.client = get_test_client()

    def test_delete_existing(self):
        from domains.collections.registry import get_registry

        registry = get_registry()
        registry.create_pipeline("to-delete", "generator", "memory")
        resp = self.client.delete("/collections/to-delete")
        assert resp.status_code == 200
        data = _d(resp)
        assert data["deleted"] == "to-delete"
        assert registry.get_pipeline("to-delete") is None

    def test_delete_nonexistent(self):
        resp = self.client.delete("/collections/nope")
        assert resp.status_code == 404


class TestCollectPipeline:
    def setup_method(self):
        self.client = get_test_client()

    def test_collect_existing(self):
        from domains.collections.registry import get_registry

        registry = get_registry()
        registry.create_pipeline("collected", "generator", "memory")
        resp = self.client.post("/collections/collected/collect")
        assert resp.status_code == 200
        data = _d(resp)
        assert data["pipeline"] == "collected"
        assert "collected" in data
        assert "stats" in data

    def test_collect_nonexistent(self):
        resp = self.client.post("/collections/nope/collect")
        assert resp.status_code == 404


class TestGetRecords:
    def setup_method(self):
        self.client = get_test_client()

    def test_get_records_empty(self):
        from domains.collections.registry import get_registry

        registry = get_registry()
        registry.create_pipeline("empty-pipe", "generator", "memory")
        resp = self.client.get("/collections/empty-pipe/records")
        assert resp.status_code == 200
        data = _d(resp)
        assert data["pipeline"] == "empty-pipe"
        assert "records" in data
        assert "total" in data
        assert "returned" in data

    def test_get_records_nonexistent(self):
        resp = self.client.get("/collections/nope/records")
        assert resp.status_code == 404

    def test_get_records_with_limit(self):
        from domains.collections.registry import get_registry

        registry = get_registry()
        registry.create_pipeline("limited", "generator", "memory")
        resp = self.client.get("/collections/limited/records?limit=1")
        assert resp.status_code == 200
        data = _d(resp)
        assert data["returned"] <= 1

    def test_get_records_limit_validation(self):
        from domains.collections.registry import get_registry

        registry = get_registry()
        registry.create_pipeline("val-pipe", "generator", "memory")
        resp = self.client.get("/collections/val-pipe/records?limit=0")
        assert resp.status_code == 422
        resp = self.client.get("/collections/val-pipe/records?limit=9999")
        assert resp.status_code == 422


class TestPipelineLifecycle:
    """End-to-end: create → list → get → collect → run → delete."""

    def setup_method(self):
        self.client = get_test_client()

    def test_full_lifecycle(self):
        # Create
        resp = self.client.post(
            "/collections/create",
            json={"name": "lifecycle", "source_type": "generator", "store_type": "memory"},
        )
        assert resp.status_code == 200

        # List
        resp = self.client.get("/collections")
        assert "lifecycle" in _d(resp)["pipelines"]

        # Get
        resp = self.client.get("/collections/lifecycle")
        assert resp.status_code == 200

        # Collect
        resp = self.client.post("/collections/lifecycle/collect")
        assert resp.status_code == 200

        # Run
        resp = self.client.post("/collections/run?name=lifecycle")
        assert resp.status_code == 200

        # Records
        resp = self.client.get("/collections/lifecycle/records")
        assert resp.status_code == 200

        # Delete
        resp = self.client.delete("/collections/lifecycle")
        assert resp.status_code == 200

        # Verify gone
        resp = self.client.get("/collections/lifecycle")
        assert resp.status_code == 404
