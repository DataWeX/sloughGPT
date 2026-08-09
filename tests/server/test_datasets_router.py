"""
Tests for the datasets router — list, create, get, update, delete, import, export, versions.
"""

import json

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


class TestListDatasetDetails:
    """GET /datasets — query passthrough and count."""

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_passes_query_and_type(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.list_datasets.return_value = []
        client.get("/datasets?q=shakespeare&type=text")
        ctrl.list_datasets.assert_called_once_with("shakespeare", "text")

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_list_reports_count(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.list_datasets.return_value = [
            {"id": "ds1", "name": "a", "path": "/p1"},
            {"id": "ds2", "name": "b", "path": "/p2"},
        ]
        resp = client.get("/datasets")
        body = resp.json()
        assert body["count"] == 2
        assert body["datasets"][0]["id"] == "ds1"


class TestCreateDatasetValidation:
    """POST /datasets — required field and response shape."""

    def test_missing_name_422(self, client):
        resp = client.post("/datasets", json={})
        assert resp.status_code == 422

    def test_empty_name_422(self, client):
        resp = client.post("/datasets", json={"name": ""})
        assert resp.status_code == 422

    def test_name_wrong_type_422(self, client):
        resp = client.post("/datasets", json={"name": 42})
        assert resp.status_code == 422

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_create_returns_full_fixture(self, mock_get_ctrl, client):
        ctrl = mock_get_ctrl.return_value
        ctrl.create_dataset.return_value = {
            "id": "ds1", "name": "test", "path": "/datasets/ds1",
            "type": "text", "num_samples": 10,
        }
        resp = client.post("/datasets", json={"name": "test"})
        body = resp.json()
        assert body["id"] == "ds1"
        assert body["path"] == "/datasets/ds1"
        assert body["num_samples"] == 10


class TestImportGithub:
    """POST /datasets/import/github"""

    @patch("domains.training.data_import.RepoImporter")
    def test_success(self, mock_cls, client):
        result = MagicMock()
        result.success = True
        result.name = None
        result.files_imported = 4
        result.total_chars = 500
        result.output_path = "/tmp/repo"
        result.error = None
        mock_cls.return_value.import_from_github.return_value = result
        resp = client.post("/datasets/import/github", json={
            "url": "https://github.com/org/repo", "name": "repo",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["dataset_id"] == "repo"
        assert "4 files" in body["message"]

    @patch("domains.training.data_import.RepoImporter")
    def test_failure_400(self, mock_cls, client):
        result = MagicMock()
        result.success = False
        result.error = "clone timeout"
        result.files_imported = 0
        result.total_chars = 0
        result.output_path = None
        mock_cls.return_value.import_from_github.return_value = result
        resp = client.post("/datasets/import/github", json={
            "url": "https://github.com/org/repo", "name": "repo",
        })
        assert resp.status_code == 400
        assert resp.json()["detail"] == "clone timeout"

    def test_missing_url_422(self, client):
        resp = client.post("/datasets/import/github", json={"name": "repo"})
        assert resp.status_code == 422

    def test_missing_name_422(self, client):
        resp = client.post("/datasets/import/github", json={"url": "https://x"})
        assert resp.status_code == 422


class TestImportHuggingface:
    """POST /datasets/import/huggingface"""

    @patch("domains.training.data_import.HuggingFaceImporter")
    def test_success_default_name(self, mock_cls, client):
        result = MagicMock()
        result.success = True
        result.files_imported = 2
        result.total_chars = 100
        result.output_path = "/tmp/hf"
        result.error = None
        mock_cls.return_value.download_dataset.return_value = result
        resp = client.post("/datasets/import/huggingface", json={"dataset_id": "org/myds"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["dataset_id"] == "myds"
        assert "2 splits" in body["message"]

    @patch("domains.training.data_import.HuggingFaceImporter")
    def test_failure_400(self, mock_cls, client):
        result = MagicMock()
        result.success = False
        result.error = "download failed"
        result.files_imported = 0
        result.total_chars = 0
        mock_cls.return_value.download_dataset.return_value = result
        resp = client.post("/datasets/import/huggingface", json={"dataset_id": "org/myds"})
        assert resp.status_code == 400
        assert resp.json()["detail"] == "download failed"

    def test_missing_dataset_id_422(self, client):
        resp = client.post("/datasets/import/huggingface", json={})
        assert resp.status_code == 422


class TestImportUrl:
    """POST /datasets/import/url"""

    @patch("domains.training.data_import.URLImporter")
    def test_success(self, mock_cls, client):
        result = MagicMock()
        result.success = True
        result.files_imported = 1
        result.total_chars = 50
        result.output_path = "/tmp/u"
        result.error = None
        mock_cls.return_value.import_from_url.return_value = result
        resp = client.post("/datasets/import/url", json={
            "url": "https://x.example/data.txt", "name": "u",
        })
        assert resp.status_code == 200
        assert resp.json()["success"] is True
        assert "Downloaded 50 chars" in resp.json()["message"]

    @patch("domains.training.data_import.URLImporter")
    def test_failure_400(self, mock_cls, client):
        result = MagicMock()
        result.success = False
        result.error = "404 not found"
        result.files_imported = 0
        result.total_chars = 0
        mock_cls.return_value.import_from_url.return_value = result
        resp = client.post("/datasets/import/url", json={
            "url": "https://x.example/missing.txt", "name": "u",
        })
        assert resp.status_code == 400
        assert resp.json()["detail"] == "404 not found"

    def test_missing_url_422(self, client):
        resp = client.post("/datasets/import/url", json={"name": "u"})
        assert resp.status_code == 422


class TestImportISBN:
    """POST /datasets/import/isbn"""

    @patch("domains.training.data_import.ISBNImporter")
    def test_success_unnamed(self, mock_cls, client):
        result = MagicMock()
        result.success = True
        result.name = None
        result.files_imported = 1
        result.total_chars = 300
        result.output_path = "/tmp/book"
        result.error = None
        mock_cls.return_value.import_from_isbn.return_value = result
        resp = client.post("/datasets/import/isbn", json={"isbn": "1234567890123"})
        assert resp.status_code == 200
        assert resp.json()["dataset_id"] == "book_1234567890123"

    @patch("domains.training.data_import.ISBNImporter")
    def test_metadata_only_when_no_files(self, mock_cls, client):
        result = MagicMock()
        result.success = True
        result.name = "bk"
        result.files_imported = 0
        result.total_chars = 0
        result.output_path = "/tmp/bk"
        result.error = None
        mock_cls.return_value.import_from_isbn.return_value = result
        resp = client.post("/datasets/import/isbn", json={"isbn": "1234567890123"})
        assert resp.status_code == 200
        assert resp.json()["message"] == "Book metadata saved"

    def test_isbn_too_short_422(self, client):
        resp = client.post("/datasets/import/isbn", json={"isbn": "123"})
        assert resp.status_code == 422

    def test_missing_isbn_422(self, client):
        resp = client.post("/datasets/import/isbn", json={})
        assert resp.status_code == 422


class TestImportMiscValidation:
    """Other import endpoints — schema validation."""

    def test_kaggle_missing_dataset_422(self, client):
        resp = client.post("/datasets/import/kaggle", json={})
        assert resp.status_code == 422

    def test_csv_missing_url_422(self, client):
        resp = client.post("/datasets/import/csv", json={"name": "c"})
        assert resp.status_code == 422

    def test_csv_missing_name_422(self, client):
        resp = client.post("/datasets/import/csv", json={"url": "https://x.csv"})
        assert resp.status_code == 422


class TestBatchImport:
    """POST /datasets/import/batch"""

    @patch("domains.training.data_import.URLImporter")
    def test_mixed_sources_reports_errors(self, mock_url, client):
        ok = MagicMock()
        ok.success = True
        ok.files_imported = 1
        ok.total_chars = 10
        mock_url.return_value.import_from_url.return_value = ok
        resp = client.post("/datasets/import/batch", json={
            "sources": [
                {"type": "url", "name": "u1", "url": "http://x"},
                {"type": "bogus", "name": "b1"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["imported"] == 1
        assert len(data["errors"]) == 1
        assert "Unsupported source type" in data["errors"][0]["error"]

    @patch("apps.api.server.routers.datasets.DatasetsRouter._get_data_importer")
    def test_local_source_success(self, mock_imp, client):
        local = MagicMock()
        local.success = True
        local.files_imported = 5
        local.total_chars = 200
        mock_imp.return_value.import_from_local.return_value = local
        resp = client.post("/datasets/import/batch", json={
            "sources": [{"type": "local", "name": "l1", "path": "/somewhere"}],
        })
        assert resp.json()["data"]["imported"] == 1

    @patch("domains.training.data_import.RepoImporter")
    def test_source_failure_recorded(self, mock_repo, client):
        bad = MagicMock()
        bad.success = False
        bad.error = "boom"
        bad.files_imported = 0
        bad.total_chars = 0
        mock_repo.return_value.import_from_github.return_value = bad
        resp = client.post("/datasets/import/batch", json={
            "sources": [{"type": "github", "name": "g1", "url": "http://x"}],
        })
        data = resp.json()["data"]
        assert data["imported"] == 0
        assert data["errors"][0]["error"] == "boom"

    def test_empty_sources(self, client):
        resp = client.post("/datasets/import/batch", json={"sources": []})
        assert resp.status_code == 200
        assert resp.json()["data"]["imported"] == 0


class TestSearchValidation:
    """Search endpoints — required params and bounds."""

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_search_missing_q_422(self, mock_get_ctrl, client):
        resp = client.get("/datasets/search")
        assert resp.status_code == 422

    def test_search_empty_q_422(self, client):
        resp = client.get("/datasets/search?q=")
        assert resp.status_code == 422

    def test_search_q_too_long_422(self, client):
        resp = client.get("/datasets/search?q=" + "a" * 501)
        assert resp.status_code == 422

    def test_books_missing_q_422(self, client):
        resp = client.get("/datasets/search/books")
        assert resp.status_code == 422

    def test_books_limit_zero_422(self, client):
        resp = client.get("/datasets/search/books?q=x&limit=0")
        assert resp.status_code == 422

    def test_books_limit_above_max_422(self, client):
        resp = client.get("/datasets/search/books?q=x&limit=51")
        assert resp.status_code == 422

    def test_github_limit_zero_422(self, client):
        resp = client.get("/datasets/search/github?q=x&limit=0")
        assert resp.status_code == 422


class TestExportValidation:
    """POST /datasets/{id}/export bounds."""

    def test_invalid_format_422(self, client):
        resp = client.post("/datasets/ds1/export?format=xml")
        assert resp.status_code == 422

    def test_format_case_sensitive_422(self, client):
        resp = client.post("/datasets/ds1/export?format=CSV")
        assert resp.status_code == 422

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_csv_format_supported(self, mock_get_ctrl, router, tmp_path):
        export_file = tmp_path / "ds1.csv"
        export_file.write_text("a,b\n1,2\n")
        mock_get_ctrl.return_value.export_dataset.return_value = export_file
        _app = FastAPI()
        _app.include_router(router.router)
        client = TestClient(_app, raise_server_exceptions=False)
        resp = client.post("/datasets/ds1/export?format=csv")
        assert resp.status_code == 200
        assert "csv" in resp.headers.get("content-disposition", "")


class TestPreviewValidation:
    """GET /datasets/{id}/preview bounds."""

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_limit_zero_422(self, mock_get_ctrl, client):
        resp = client.get("/datasets/ds1/preview?limit=0")
        assert resp.status_code == 422

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_limit_above_max_422(self, mock_get_ctrl, client):
        resp = client.get("/datasets/ds1/preview?limit=1001")
        assert resp.status_code == 422


class TestFromChatValidation:
    """POST /datasets/from-chat"""

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_empty_messages_400(self, mock_get_ctrl, client):
        resp = client.post("/datasets/from-chat", json={"messages": []})
        assert resp.status_code == 400
        assert "No messages provided" in resp.json()["detail"]

    def test_missing_messages_422(self, client):
        resp = client.post("/datasets/from-chat", json={})
        assert resp.status_code == 422

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_invalid_role_422(self, mock_get_ctrl, client):
        resp = client.post("/datasets/from-chat", json={
            "messages": [{"role": "owner", "content": "hello"}],
        })
        assert resp.status_code == 422

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_name_too_long_422(self, mock_get_ctrl, client):
        resp = client.post("/datasets/from-chat", json={
            "messages": [{"role": "user", "content": "hi"}],
            "name": "x" * 101,
        })
        assert resp.status_code == 422


class TestConvertToMessages:
    """POST /datasets/convert-to-messages"""

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_not_found_404(self, mock_get_ctrl, client):
        mock_get_ctrl.return_value.list_datasets.return_value = []
        resp = client.post("/datasets/convert-to-messages?dataset_id=ds1")
        assert resp.status_code == 404

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_missing_input_jsonl_404(self, mock_get_ctrl, router):
        mock_get_ctrl.return_value.list_datasets.return_value = [{"id": "ds1", "name": "dia"}]
        _app = FastAPI()
        _app.include_router(router.router)
        client = TestClient(_app, raise_server_exceptions=False)
        resp = client.post("/datasets/convert-to-messages?dataset_id=ds1")
        assert resp.status_code == 404
        assert "input.jsonl" in resp.json()["detail"]

    @patch("apps.api.server.routers.datasets.get_datasets_controller")
    def test_converts_text_and_messages(self, mock_get_ctrl, router):
        ctrl = mock_get_ctrl.return_value
        ctrl.list_datasets.return_value = [{"id": "ds1", "name": "dia"}]
        ctrl.create_dataset.return_value = {"id": "out", "name": "dia-messages", "path": "/tmp/out"}
        ds = router._DATASETS_DIR / "ds1"
        ds.mkdir(parents=True, exist_ok=True)
        (ds / "input.jsonl").write_text(
            '{"text": "hello"}\n'
            '{"messages": [{"role": "user", "content": "hi"}]}\n'
        )
        _app = FastAPI()
        _app.include_router(router.router)
        client = TestClient(_app, raise_server_exceptions=False)
        resp = client.post("/datasets/convert-to-messages?dataset_id=ds1&system_prompt=BE_STRICT")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "converted"
        assert body["total_conversations"] == 2
        out = router._DATASETS_DIR / "out" / "input.jsonl"
        assert out.exists()
        lines = [json.loads(l) for l in out.read_text().splitlines()]
        assert len(lines) == 2
        system_contents = [
            m["content"] for msgs in lines for m in msgs["messages"] if m["role"] == "system"
        ]
        assert system_contents == ["BE_STRICT", "BE_STRICT"]
