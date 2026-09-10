"""Tests for the Python SDK client methods."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the SDK package is importable
_sdk_dir = str(Path(__file__).resolve().parent.parent)
if _sdk_dir not in sys.path:
    sys.path.insert(0, _sdk_dir)

from sloughgpt_sdk.client import SloughGPTClient


@pytest.fixture
def client():
    """Create a test client with mocked HTTP."""
    with patch("sloughgpt_sdk.client.SloughGPTClient._request") as mock_req:
        c = SloughGPTClient(base_url="http://localhost:8000")
        c._mock_request = mock_req
        yield c


class TestExportTrainingHistory:
    def test_json_export(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"format": "json", "outcomes": [], "count": 0})
        result = client.export_training_history("json", 10)
        client._mock_request.assert_called_once_with("GET", "/settings/training/history/export?format=json&limit=10")
        assert result["format"] == "json"

    def test_csv_export(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"format": "csv", "content": "a,b\n1,2", "count": 1})
        result = client.export_training_history("csv")
        client._mock_request.assert_called_once_with("GET", "/settings/training/history/export?format=csv&limit=0")
        assert result["format"] == "csv"


class TestGenerateModelCard:
    def test_generate(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"card": {"model_name": "test"}, "markdown": "# test"})
        result = client.generate_model_card("test", base_model="gpt2")
        client._mock_request.assert_called_once()
        assert result["card"]["model_name"] == "test"


class TestGetDashboardSummary:
    def test_get_summary(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"health": {"model_loaded": True}})
        result = client.get_dashboard_summary()
        client._mock_request.assert_called_once_with("GET", "/dashboard/summary")
        assert result["health"]["model_loaded"] is True


class TestCompareTrainingRuns:
    def test_compare(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"run_a": {}, "run_b": {}, "differences": {}})
        result = client.compare_training_runs("run-1", "run-2")
        client._mock_request.assert_called_once_with("GET", "/settings/training/compare?run_a=run-1&run_b=run-2")
        assert "run_a" in result


class TestGetBatchTrainingStatus:
    def test_get_status(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"jobs": [], "summary": {"total": 0}})
        result = client.get_batch_training_status()
        client._mock_request.assert_called_once_with("GET", "/settings/training/batch-status")
        assert "jobs" in result


class TestListTrainingPresets:
    def test_list_presets(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"presets": [{"name": "quick-finetune"}]})
        result = client.list_training_presets()
        client._mock_request.assert_called_once_with("GET", "/settings/training/presets")
        assert len(result["presets"]) == 1


class TestGetTrainingPreset:
    def test_get_preset(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"name": "Quick Fine-Tune", "model": "gpt2"})
        result = client.get_training_preset("quick-finetune")
        client._mock_request.assert_called_once_with("GET", "/settings/training/presets/quick-finetune")
        assert result["name"] == "Quick Fine-Tune"


class TestApplyTrainingPreset:
    def test_apply_preset(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"preset": "quick-finetune", "applied": {}})
        result = client.apply_training_preset("quick-finetune")
        client._mock_request.assert_called_once_with("POST", "/settings/training/presets/quick-finetune/apply")
        assert result["preset"] == "quick-finetune"


class TestGetTrainingRun:
    def test_get_run(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"run_id": "run-1", "model": "gpt2"})
        result = client.get_training_run("run-1")
        client._mock_request.assert_called_once_with("GET", "/settings/training/runs/run-1")
        assert result["run_id"] == "run-1"


class TestDeleteTrainingRun:
    def test_delete_run(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"deleted": True, "run_id": "run-1"})
        result = client.delete_training_run("run-1")
        client._mock_request.assert_called_once_with("DELETE", "/settings/training/runs/run-1")
        assert result["deleted"] is True


class TestFilterTrainingRuns:
    def test_filter_runs(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"runs": [], "count": 0})
        result = client.filter_training_runs(model="gpt2", limit=10)
        client._mock_request.assert_called_once()
        assert "model=gpt2" in client._mock_request.call_args[0][1]
        assert "limit=10" in client._mock_request.call_args[0][1]


class TestClearTrainingHistory:
    def test_clear_history(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"cleared": True, "removed_count": 5})
        result = client.clear_training_history()
        client._mock_request.assert_called_once_with("POST", "/settings/training/history/clear")
        assert result["cleared"] is True


class TestAddRunTag:
    def test_add_tag(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"run_id": "run-1", "tags": ["best"]})
        result = client.add_run_tag("run-1", "best")
        client._mock_request.assert_called_once()
        assert result["run_id"] == "run-1"


class TestRemoveRunTag:
    def test_remove_tag(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"run_id": "run-1", "tags": []})
        result = client.remove_run_tag("run-1", "best")
        client._mock_request.assert_called_once()
        assert result["run_id"] == "run-1"


class TestSetRunNotes:
    def test_set_notes(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"run_id": "run-1", "notes": "test note"})
        result = client.set_run_notes("run-1", "test note")
        client._mock_request.assert_called_once()
        assert result["notes"] == "test note"


class TestGetAllTags:
    def test_get_tags(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"tags": ["best", "production"]})
        result = client.get_all_tags()
        client._mock_request.assert_called_once_with("GET", "/settings/training/tags")
        assert result["tags"] == ["best", "production"]


class TestGetRunsByTag:
    def test_get_runs_by_tag(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"runs": [], "count": 0, "tag": "best"})
        result = client.get_runs_by_tag("best")
        client._mock_request.assert_called_once_with("GET", "/settings/training/tags/best")
        assert result["tag"] == "best"


class TestExportTrainingRun:
    def test_export_run(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"run_id": "run-1", "format": "json", "content": "{}"})
        result = client.export_training_run("run-1", "json")
        client._mock_request.assert_called_once_with("GET", "/settings/training/runs/run-1/export?format=json")
        assert result["run_id"] == "run-1"


class TestToggleBookmark:
    def test_toggle_bookmark(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"run_id": "run-1", "bookmarked": True})
        result = client.toggle_bookmark("run-1")
        client._mock_request.assert_called_once_with("POST", "/settings/training/runs/run-1/bookmark")
        assert result["run_id"] == "run-1"


class TestGetBookmarkedRuns:
    def test_get_bookmarked(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"runs": [], "count": 0})
        result = client.get_bookmarked_runs()
        client._mock_request.assert_called_once_with("GET", "/settings/training/bookmarks")
        assert "runs" in result


class TestDuplicateTrainingRun:
    def test_duplicate_run(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"run_id": "new-run", "model": "gpt2"})
        result = client.duplicate_training_run("run-1", "new-run")
        client._mock_request.assert_called_once()
        assert result["run_id"] == "new-run"


class TestBulkDeleteRuns:
    def test_bulk_delete(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"deleted_count": 2, "requested": 2})
        result = client.bulk_delete_runs(["run-1", "run-2"])
        client._mock_request.assert_called_once()
        assert result["deleted_count"] == 2


class TestBulkAddTag:
    def test_bulk_add_tag(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"updated_count": 2, "tag": "best"})
        result = client.bulk_add_tag(["run-1", "run-2"], "best")
        client._mock_request.assert_called_once()
        assert result["tag"] == "best"


class TestBulkBookmark:
    def test_bulk_bookmark(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"updated_count": 2, "bookmarked": True})
        result = client.bulk_bookmark(["run-1", "run-2"], True)
        client._mock_request.assert_called_once()
        assert result["bookmarked"] is True
