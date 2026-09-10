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


class TestAutoTrainStatus:
    def test_get_auto_train_status(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"enabled": True, "threshold": 10})
        result = client.get_auto_train_status()
        client._mock_request.assert_called_once()
        assert result["enabled"] is True
        assert result["threshold"] == 10


class TestAutoTrainConfig:
    def test_update_auto_train_config(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"enabled": True, "threshold": 20})
        result = client.update_auto_train_config(threshold=20)
        client._mock_request.assert_called_once()
        assert result["threshold"] == 20


class TestSecurityKeys:
    def test_list_keys(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"keys": [{"id": "k1"}], "count": 1})
        result = client.get_security_keys()
        client._mock_request.assert_called_once_with("GET", "/security/keys")
        assert len(result) == 1

    def test_create_key(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"id": "k2", "name": "test"})
        result = client.create_security_key("test", scopes=["read"], expires_in_days=30)
        client._mock_request.assert_called_once()
        assert result["name"] == "test"

    def test_get_key(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"id": "k1"})
        result = client.get_security_key("k1")
        client._mock_request.assert_called_once_with("GET", "/security/keys/k1")
        assert result["id"] == "k1"

    def test_delete_key(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"deleted": True})
        result = client.delete_security_key("k1")
        client._mock_request.assert_called_once_with("DELETE", "/security/keys/k1")
        assert result["deleted"] is True

    def test_rotate_key(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"id": "k1", "new_secret": "s3cret"})
        result = client.rotate_security_key("k1")
        client._mock_request.assert_called_once_with("POST", "/security/keys/k1/rotate")
        assert "new_secret" in result

    def test_validate_key(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"valid": True})
        result = client.validate_security_key("my-key")
        client._mock_request.assert_called_once()
        assert result["valid"] is True


class TestTenants:
    def test_list_tenants(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"tenants": [{"id": "t1"}], "count": 1})
        result = client.list_tenants()
        client._mock_request.assert_called_once_with("GET", "/tenants")
        assert len(result) == 1

    def test_get_tenant(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"id": "t1", "name": "acme"})
        result = client.get_tenant("t1")
        client._mock_request.assert_called_once_with("GET", "/tenants/t1")
        assert result["name"] == "acme"

    def test_create_tenant(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"id": "t2", "name": "new"})
        result = client.create_tenant("new", plan="pro")
        client._mock_request.assert_called_once()
        assert result["name"] == "new"

    def test_update_tenant(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"id": "t1", "name": "updated"})
        result = client.update_tenant("t1", name="updated")
        client._mock_request.assert_called_once()
        assert result["name"] == "updated"

    def test_delete_tenant(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"deleted": True})
        result = client.delete_tenant("t1")
        client._mock_request.assert_called_once_with("DELETE", "/tenants/t1")
        assert result["deleted"] is True

    def test_tenant_stats(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"users": 10, "workspaces": 3})
        result = client.get_tenant_stats("t1")
        client._mock_request.assert_called_once_with("GET", "/tenants/t1/stats")
        assert result["users"] == 10


class TestProfiles:
    def test_list_profiles(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"profiles": [{"id": "p1"}], "count": 1})
        result = client.list_profiles()
        client._mock_request.assert_called_once_with("GET", "/profiles")
        assert len(result) == 1

    def test_get_profile(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"id": "p1", "name": "default"})
        result = client.get_profile("p1")
        client._mock_request.assert_called_once_with("GET", "/profiles/p1")
        assert result["name"] == "default"

    def test_apply_profile(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"applied": True})
        result = client.apply_profile("p1")
        client._mock_request.assert_called_once()
        assert result["applied"] is True

    def test_active_profile(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"id": "p1"})
        result = client.get_active_profile()
        client._mock_request.assert_called_once_with("GET", "/profiles/active")
        assert result["id"] == "p1"

    def test_recommend_profile(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"profile_id": "p2", "reason": "best match"})
        result = client.recommend_profile()
        client._mock_request.assert_called_once_with("GET", "/profiles/recommend")
        assert result["profile_id"] == "p2"


class TestWorkspaces:
    def test_list_workspaces(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"workspaces": [{"id": "w1"}], "count": 1})
        result = client.list_workspaces()
        client._mock_request.assert_called_once_with("GET", "/workspaces")
        assert len(result) == 1

    def test_get_workspace(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"id": "w1", "name": "team"})
        result = client.get_workspace("w1")
        client._mock_request.assert_called_once_with("GET", "/workspaces/w1")
        assert result["name"] == "team"

    def test_create_workspace(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"id": "w2", "name": "new"})
        result = client.create_workspace("new", description="test")
        client._mock_request.assert_called_once()
        assert result["name"] == "new"

    def test_add_workspace_member(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"added": True})
        result = client.add_workspace_member("w1", "user-1", "admin")
        client._mock_request.assert_called_once()
        assert result["added"] is True
