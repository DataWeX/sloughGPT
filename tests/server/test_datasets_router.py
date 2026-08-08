"""
Tests for the datasets router — list, create, get, update, delete, import, export, versions.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.datasets import DatasetsRouter


@pytest.fixture
def router(tmp_path):
    r = DatasetsRouter()
    r._DATASETS_DIR = tmp_path / "datasets"
    r._DATASETS_DIR.mkdir(exist_ok=True)
    return r


@pytest.fixture
def app(router):
    _app = FastAPI()
    _app.include_router(router.router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestListDatasets:
    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_returns_list(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.list_datasets.return_value = []
        resp = client.get("/datasets")
        assert resp.status_code == 200


_DATASET_FIXTURE = {"id": "ds1", "name": "test", "path": "/datasets/ds1"}


class TestCreateDataset:
    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_creates_dataset(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.create_dataset.return_value = dict(_DATASET_FIXTURE)
        resp = client.post("/datasets", json={"name": "test"})
        assert resp.status_code == 200


class TestGetDataset:
    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_gets_dataset(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.get_dataset.return_value = dict(_DATASET_FIXTURE)
        resp = client.get("/datasets/valid_id")
        assert resp.status_code == 200

    def test_returns_404_for_bad_id(self, client):
        resp = client.get("/datasets/bad_id")
        assert resp.status_code == 404


class TestSearchDatasets:
    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_searches(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.search_datasets.return_value = []
        resp = client.get("/datasets/search?q=test")
        assert resp.status_code == 200


class TestDatasetVersions:
    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_create_version(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.create_version_snapshot.return_value = "20260801120000"
        resp = client.post("/datasets/ds1/versions")
        assert resp.status_code == 200
        assert resp.json()["timestamp"] == "20260801120000"

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_create_version_404_when_dataset_missing(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.create_version_snapshot.return_value = None
        resp = client.post("/datasets/ds1/versions")
        assert resp.status_code == 404

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_list_versions(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.list_versions.return_value = ["20260801120000", "20260801110000"]
        resp = client.get("/datasets/ds1/versions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 2
        assert body["versions"][0] == "20260801120000"

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_restore_version(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.restore_version.return_value = True
        resp = client.post("/datasets/ds1/versions/20260801120000")
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_restore_version_404_when_version_missing(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.restore_version.return_value = False
        resp = client.post("/datasets/ds1/versions/20260801120000")
        assert resp.status_code == 404


class TestUpdateDataset:
    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_updates_dataset(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.update_dataset.return_value = dict(_DATASET_FIXTURE)
        resp = client.patch("/datasets/ds1", json={"name": "renamed"})
        assert resp.status_code == 200
        ctrl.update_dataset.assert_called_once()
        assert resp.json()["name"] == "test"

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_update_404_when_missing(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.update_dataset.return_value = None
        resp = client.patch("/datasets/missing", json={"description": "x"})
        assert resp.status_code == 404


class TestDeleteDataset:
    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_deletes_dataset(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.delete_dataset.return_value = True
        resp = client.delete("/datasets/ds1")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "deleted"

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_delete_404_when_missing(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.delete_dataset.return_value = False
        resp = client.delete("/datasets/ds1")
        assert resp.status_code == 404


class TestDatasetValidation:
    def test_rejects_invalid_id(self, client):
        resp = client.get("/datasets/bad.id")
        assert resp.status_code == 422

    def test_rejects_id_with_slashes(self, client):
        resp = client.get("/datasets/a/b")
        assert resp.status_code in (404, 422)


class TestDatasetStats:
    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_get_stats(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.get_dataset_stats.return_value = {
            "format": "text", "samples": 5, "chars": 100, "avg_length": 20.0,
        }
        resp = client.get("/datasets/ds1/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["samples"] == 5
        assert body["chars"] == 100

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_stats_404_when_missing(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.get_dataset_stats.return_value = None
        resp = client.get("/datasets/ds1/stats")
        assert resp.status_code == 404


class TestAddDatasetData:
    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_appends_data(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.add_data.return_value = 2
        resp = client.post("/datasets/ds1/data", json={"data": ["line1", "line2"]})
        assert resp.status_code == 200
        assert resp.json()["data"]["rows_added"] == 2

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_add_data_404_when_missing(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.add_data.return_value = None
        resp = client.post("/datasets/ds1/data", json={"data": ["x"]})
        assert resp.status_code == 404


class TestPreviewDataset:
    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_preview_returns_rows(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.preview_dataset.return_value = [{"text": "hello"}, {"text": "world"}]
        resp = client.get("/datasets/ds1/preview")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_preview_404_when_empty(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.preview_dataset.return_value = None
        resp = client.get("/datasets/ds1/preview")
        assert resp.status_code == 404


class TestExportDataset:
    def test_export_invalid_format(self, client):
        resp = client.post("/datasets/ds1/export?format=xml")
        assert resp.status_code == 422

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_export_returns_file(self, mock_get_ctrl, router, tmp_path):
        export_file = tmp_path / "ds1.jsonl"
        export_file.write_text('{"text": "hi"}\n')
        ctrl = mock_get_ctrl.return_value
        ctrl.export_dataset.return_value = export_file
        _app = FastAPI()
        _app.include_router(router.router)
        client = TestClient(_app, raise_server_exceptions=False)
        resp = client.post("/datasets/ds1/export?format=jsonl")
        assert resp.status_code == 200
        assert "jsonl" in resp.headers.get("content-disposition", "")

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_export_404_when_empty(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.export_dataset.return_value = None
        resp = client.post("/datasets/ds1/export?format=jsonl")
        assert resp.status_code == 404


class TestCreateFromChat:
    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_creates_from_messages(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.create_dataset.return_value = dict(_DATASET_FIXTURE)
        resp = client.post("/datasets/from-chat", json={
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "created"
        assert body["messages_exported"] == 2

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_creator_skips_empty_messages(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.create_dataset.return_value = dict(_DATASET_FIXTURE)
        resp = client.post("/datasets/from-chat", json={
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "system", "content": "ignored"},
                {"role": "user", "content": ""},
            ],
        })
        assert resp.json()["messages_exported"] == 1


class TestImportFromLocal:
    def test_rejects_outside_allowed_paths(self, client):
        resp = client.post("/datasets/import/local", json={
            "path": "/etc/passwd", "name": "bad",
        })
        assert resp.status_code == 403

    @patch("apps.api.server.routers.datasets.DatasetsRouter._get_data_importer")
    def test_imports_local_dir(self, mock_importer, router):
        result = MagicMock()
        result.success = True
        result.files_imported = 3
        result.total_chars = 500
        result.output_path = "/tmp/x"
        result.error = None
        mock_importer.return_value.import_from_local.return_value = result
        _app = FastAPI()
        _app.include_router(router.router)
        client = TestClient(_app, raise_server_exceptions=False)
        resp = client.post("/datasets/import/local", json={
            "path": str(router._DATASETS_DIR), "name": "mine",
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert resp.json()["output_path"] == "/tmp/x"
        assert "3 files" in resp.json()["message"]

    @patch("apps.api.server.routers.datasets.DatasetsRouter._get_data_importer")
    def test_import_failure_400(self, mock_importer, router):
        result = MagicMock()
        result.success = False
        result.error = "nothing found"
        result.files_imported = 0
        result.total_chars = 0
        result.output_path = None
        mock_importer.return_value.import_from_local.return_value = result
        _app = FastAPI()
        _app.include_router(router.router)
        client = TestClient(_app, raise_server_exceptions=False)
        r = client.post("/datasets/import/local", json={
            "path": str(router._DATASETS_DIR), "name": "mine",
        })
        assert r.status_code == 400
