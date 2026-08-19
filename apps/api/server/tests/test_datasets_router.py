from infrastructure.exception_handlers import register_app_error_handler
"""
Tests for datasets router — CRUD, search, stats, data, preview, export, versioning.

Only registers the datasets router to avoid pulling in heavy dependencies.
"""

import tempfile
from pathlib import Path
import json
import asyncio
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI, HTTPException

from routers.datasets import router as datasets_router, DatasetsRouter
from controllers.datasets import DatasetsController

app = FastAPI()
register_app_error_handler(app)
app.include_router(datasets_router)
client = TestClient(app)


def _data(resp):
    """Unwrap the success_response() envelope."""
    body = resp.json()
    if isinstance(body, list):
        return body
    return body.get("data", body)


# ── Fixtures ───────────────────────────────────────────────────────────────

SAMPLE_DATASETS = [
    {
        "id": "shakespeare",
        "name": "Shakespeare",
        "path": "/tmp/datasets/shakespeare",
        "type": "corpus",
        "size_bytes": 27851,
        "size_formatted": "27.2 KB",
        "num_samples": 100,
    },
    {
        "id": "poetry",
        "name": "Poetry",
        "path": "/tmp/datasets/poetry",
        "type": "text",
        "size_bytes": 1024,
        "size_formatted": "1.0 KB",
        "num_samples": 50,
    },
]


@pytest.fixture(autouse=True)
def mock_controller():
    """Mock the datasets controller so tests don't touch the real filesystem."""
    ctrl = MagicMock()

    ctrl.list_datasets.return_value = SAMPLE_DATASETS
    ctrl.get_dataset.return_value = SAMPLE_DATASETS[0]
    ctrl.create_dataset.return_value = {
        "id": "new-dataset",
        "name": "New Dataset",
        "path": "/tmp/datasets/new-dataset",
        "type": "corpus",
        "size_bytes": 0,
        "size_formatted": "Empty",
        "num_samples": 0,
    }
    ctrl.update_dataset.return_value = {
        "id": "shakespeare",
        "name": "Updated",
        "path": "/tmp/datasets/shakespeare",
        "type": "corpus",
        "size_bytes": 27851,
        "size_formatted": "27.2 KB",
        "num_samples": 100,
    }
    ctrl.get_dataset_stats.return_value = {
        "format": "jsonl",
        "samples": 100,
        "chars": 27851,
        "avg_length": 278.5,
        "has_messages": False,
        "sample_preview": ["line1", "line2"],
        "lines": 100,
        "suggested_method": "finetune",
        "file_type": "jsonl",
    }
    ctrl.dataset_exists.return_value = True
    ctrl.search_datasets.return_value = [
        {
            "id": "shakespeare",
            "name": "Shakespeare",
            "path": "/tmp/datasets/shakespeare",
            "type": "corpus",
            "size_bytes": 27851,
            "size_formatted": "27.2 KB",
            "num_samples": 100,
        }
    ]
    ctrl.add_data.return_value = 3
    ctrl.preview_dataset.return_value = {
        "rows": [{"text": "hello"}, {"text": "world"}],
        "columns": ["text"],
        "total": 2,
    }
    ctrl.export_dataset.return_value = None  # real path set per test
    ctrl.create_version_snapshot.return_value = "2025-01-01T00:00:00"
    ctrl.list_versions.return_value = ["v1", "v2"]
    ctrl.restore_version.return_value = True

    with patch("routers.datasets.get_datasets_controller", return_value=ctrl):
        yield ctrl


# ── GET /datasets ─────────────────────────────────────────────────────────

class TestListDatasets:

    def test_list_returns_datasets(self, mock_controller):
        resp = client.get("/datasets")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert data["datasets"][0]["id"] == "shakespeare"

    def test_list_with_query(self, mock_controller):
        client.get("/datasets?q=shake")
        mock_controller.list_datasets.assert_called_with("shake", None)

    def test_list_with_type_filter(self, mock_controller):
        client.get("/datasets?type=text")
        mock_controller.list_datasets.assert_called_with(None, "text")


# ── GET /datasets/search ──────────────────────────────────────────────────

class TestSearch:

    def test_search_returns_results(self, mock_controller):
        resp = client.get("/datasets/search?q=shake")
        assert resp.status_code == 200
        data = _data(resp)
        assert data["count"] == 1

    def test_search_delegates_to_controller(self, mock_controller):
        client.get("/datasets/search?q=poem")
        mock_controller.search_datasets.assert_called_with("poem")


# ── Controller: search_datasets returns full summaries ───────────────────

class TestControllerSearch:
    """Direct controller test — the router mocks the controller, so a shape
    regression (names instead of full summaries) can only be caught here."""

    def _make_controller(self, tmp_path: Path) -> DatasetsController:
        datasets_dir = tmp_path / "datasets"
        (datasets_dir / "shakespeare").mkdir(parents=True)
        (datasets_dir / "shakespeare" / "input.txt").write_text("To be or not to be.\n")
        (datasets_dir / "poetry").mkdir(parents=True)
        (datasets_dir / "poetry" / "input.txt").write_text("Roses are red.\n")
        return DatasetsController(repo_root=tmp_path)

    def test_search_returns_full_dataset_summaries(self, tmp_path):
        ctrl = self._make_controller(tmp_path)
        results = ctrl.search_datasets("shake")

        assert len(results) == 1
        entry = results[0]
        assert isinstance(entry, dict)
        assert entry["id"] == "shakespeare"
        assert entry["name"] == "Shakespeare"
        assert entry["size"] == entry["size_bytes"]
        assert entry["samples"] == entry["num_samples"]
        assert "path" in entry and "type" in entry

    def test_search_matches_case_insensitively(self, tmp_path):
        ctrl = self._make_controller(tmp_path)
        assert len(ctrl.search_datasets("SHAKESPEARE")) == 1
        assert len(ctrl.search_datasets("SHAKE")) == 1

    def test_search_no_match_returns_empty(self, tmp_path):
        ctrl = self._make_controller(tmp_path)
        assert ctrl.search_datasets("zzz-nothing") == []

    def test_search_no_datasets_dir_returns_empty(self, tmp_path):
        ctrl = DatasetsController(repo_root=tmp_path)
        assert ctrl.search_datasets("anything") == []


# ── GET /datasets/{id} ───────────────────────────────────────────────────

class TestGetDataset:

    def test_get_existing(self, mock_controller):
        resp = client.get("/datasets/shakespeare")
        assert resp.status_code == 200
        assert resp.json()["id"] == "shakespeare"

    def test_get_nonexistent(self, mock_controller):
        mock_controller.get_dataset.return_value = None
        resp = client.get("/datasets/nonexistent")
        assert resp.status_code == 404


# ── POST /datasets ───────────────────────────────────────────────────────

class TestCreate:

    def test_create(self, mock_controller):
        resp = client.post("/datasets", json={"name": "New Dataset"})
        assert resp.status_code == 200
        assert resp.json()["id"] == "new-dataset"

    def test_create_calls_controller(self, mock_controller):
        client.post("/datasets", json={"name": "My Set", "description": "test"})
        mock_controller.create_dataset.assert_called_with("My Set", "test")


# ── PATCH /datasets/{id} ────────────────────────────────────────────────

class TestUpdate:

    def test_update_existing(self, mock_controller):
        resp = client.patch("/datasets/shakespeare", json={"name": "Updated"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated"

    def test_update_nonexistent(self, mock_controller):
        mock_controller.update_dataset.return_value = None
        resp = client.patch("/datasets/nonexistent", json={"name": "Nope"})
        assert resp.status_code == 404


# ── DELETE /datasets/{id} ───────────────────────────────────────────────

class TestDelete:

    def test_delete_existing(self, mock_controller):
        mock_controller.delete_dataset.return_value = True
        resp = client.delete("/datasets/shakespeare")
        assert resp.status_code == 200
        assert _data(resp)["status"] == "deleted"

    def test_delete_nonexistent(self, mock_controller):
        mock_controller.delete_dataset.return_value = False
        resp = client.delete("/datasets/nonexistent")
        assert resp.status_code == 404


# ── GET /datasets/{id}/stats ────────────────────────────────────────────

class TestStats:

    def test_stats_existing(self, mock_controller):
        resp = client.get("/datasets/shakespeare/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "format" in data
        assert "samples" in data
        assert "lines" in data

    def test_stats_nonexistent(self, mock_controller):
        mock_controller.get_dataset_stats.return_value = None
        resp = client.get("/datasets/nonexistent/stats")
        assert resp.status_code == 404


# ── POST /datasets/{id}/data ────────────────────────────────────────────

class TestAppendData:

    def test_append_data(self, mock_controller):
        resp = client.post("/datasets/shakespeare/data", json={"data": ["a", "b", "c"]})
        assert resp.status_code == 200
        assert _data(resp)["rows_added"] == 3

    def test_append_nonexistent(self, mock_controller):
        mock_controller.add_data.return_value = None
        resp = client.post("/datasets/nonexistent/data", json={"data": ["x"]})
        assert resp.status_code == 404


# ── GET /datasets/{id}/preview ──────────────────────────────────────────

class TestPreview:

    def test_preview(self, mock_controller):
        resp = client.get("/datasets/shakespeare/preview?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()["rows"]) == 2

    def test_preview_nonexistent(self, mock_controller):
        mock_controller.preview_dataset.return_value = None
        resp = client.get("/datasets/nonexistent/preview")
        assert resp.status_code == 404


# ── POST /datasets/{id}/export ──────────────────────────────────────────

class TestExport:

    def test_export_jsonl(self, mock_controller):
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        tmp.write(b"{}")
        tmp.close()
        mock_controller.export_dataset.return_value = tmp.name

        resp = client.post(
            "/datasets/shakespeare/export",
            json={"format": "jsonl"},
        )
        assert resp.status_code == 200
        assert "shakespeare" in resp.headers.get("content-disposition", "")

        Path(tmp.name).unlink(missing_ok=True)

    def test_export_defaults_to_jsonl(self, mock_controller):
        tmp = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        tmp.write(b"{}")
        tmp.close()
        mock_controller.export_dataset.return_value = tmp.name

        resp = client.post("/datasets/shakespeare/export", json={})
        assert resp.status_code == 200
        mock_controller.export_dataset.assert_called_with("shakespeare", "jsonl")

        Path(tmp.name).unlink(missing_ok=True)

    def test_export_nonexistent(self, mock_controller):
        mock_controller.export_dataset.return_value = None
        resp = client.post("/datasets/nonexistent/export", json={})
        assert resp.status_code == 404


# ── Versioning ──────────────────────────────────────────────────────────

class TestVersions:

    def test_create_version(self, mock_controller):
        resp = client.post("/datasets/shakespeare/versions")
        assert resp.status_code == 200
        assert resp.json()["timestamp"] == "2025-01-01T00:00:00"

    def test_create_version_nonexistent(self, mock_controller):
        mock_controller.create_version_snapshot.return_value = None
        resp = client.post("/datasets/nonexistent/versions")
        assert resp.status_code == 404

    def test_list_versions(self, mock_controller):
        resp = client.get("/datasets/shakespeare/versions")
        assert resp.status_code == 200
        assert resp.json()["count"] == 2
        assert resp.json()["versions"] == ["v1", "v2"]

    def test_list_versions_nonexistent(self, mock_controller):
        mock_controller.list_versions.return_value = []
        resp = client.get("/datasets/nonexistent/versions")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_restore_version(self, mock_controller):
        resp = client.post("/datasets/shakespeare/versions/v1")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_restore_version_nonexistent(self, mock_controller):
        mock_controller.restore_version.return_value = False
        resp = client.post("/datasets/nonexistent/versions/v2")
        assert resp.status_code == 404


# ── Convert to messages ──────────────────────────────────────────────────
# The convert endpoint reads input.jsonl directly from the datasets dir and
# creates the converted dataset via the controller. A fresh router instance
# with a temp _DATASETS_DIR keeps it off the real filesystem.

def _make_convert_router(tmp_path, source_id, source_name="Corpus"):
    router = DatasetsRouter()
    router._DATASETS_DIR = tmp_path / "datasets"
    router._DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    ctrl = MagicMock()
    ctrl.list_datasets.return_value = [{"id": source_id, "name": source_name}]
    ctrl.create_dataset.return_value = {
        "id": f"{source_id}-messages",
        "name": f"{source_name}-messages",
    }
    return router, ctrl


def _write_jsonl(router, ds_id, lines):
    d = router._DATASETS_DIR / ds_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "input.jsonl").write_text("\n".join(json.dumps(row) for row in lines) + "\n")


def _read_rows(router, ds_id):
    out = router._DATASETS_DIR / ds_id / "input.jsonl"
    return [json.loads(x) for x in out.read_text().splitlines() if x.strip()]


class TestConvertToMessages:

    def test_convert_text_rows_wraps_in_messages(self, tmp_path):
        router, ctrl = _make_convert_router(tmp_path, "corpus")
        _write_jsonl(router, "corpus", [{"text": "hello"}, {"text": "world"}])
        with patch("routers.datasets.get_datasets_controller", return_value=ctrl):
            res = asyncio.run(router.convert_to_messages("corpus"))
        assert res["status"] == "converted"
        assert res["new_dataset_id"] == "corpus-messages"
        assert res["total_conversations"] == 2
        rows = _read_rows(router, "corpus-messages")
        assert rows[0]["messages"][0]["role"] == "system"
        assert rows[0]["messages"][1]["content"] == "hello"
        ctrl.create_dataset.assert_called_once()

    def test_convert_applies_custom_system_prompt(self, tmp_path):
        router, ctrl = _make_convert_router(tmp_path, "corpus")
        _write_jsonl(router, "corpus", [{"text": "hi"}])
        with patch("routers.datasets.get_datasets_controller", return_value=ctrl):
            res = asyncio.run(router.convert_to_messages("corpus", "You are a poet."))
        rows = _read_rows(router, "corpus-messages")
        assert rows[0]["messages"][0] == {"role": "system", "content": "You are a poet."}

    def test_convert_keeps_existing_system_message(self, tmp_path):
        router, ctrl = _make_convert_router(tmp_path, "chat")
        _write_jsonl(router, "chat", [{"messages": [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "hi"},
        ]}])
        with patch("routers.datasets.get_datasets_controller", return_value=ctrl):
            res = asyncio.run(router.convert_to_messages("chat"))
        rows = _read_rows(router, "chat-messages")
        assert rows[0]["messages"][0]["content"] == "You are a bot."
        assert res["total_conversations"] == 1

    def test_convert_missing_dataset_raises_404(self, tmp_path):
        router, ctrl = _make_convert_router(tmp_path, "corpus")
        ctrl.list_datasets.return_value = []
        with patch("routers.datasets.get_datasets_controller", return_value=ctrl):
            with pytest.raises(HTTPException) as ei:
                asyncio.run(router.convert_to_messages("nope"))
        assert ei.value.status_code == 404

    def test_convert_missing_jsonl_raises_404(self, tmp_path):
        router, ctrl = _make_convert_router(tmp_path, "corpus")
        (router._DATASETS_DIR / "corpus").mkdir(parents=True)
        with patch("routers.datasets.get_datasets_controller", return_value=ctrl):
            with pytest.raises(HTTPException) as ei:
                asyncio.run(router.convert_to_messages("corpus"))
        assert ei.value.status_code == 404
