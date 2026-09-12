"""Tests for the Python SDK client methods."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the SDK package is importable
_sdk_dir = str(Path(__file__).resolve().parent.parent)
if _sdk_dir not in sys.path:
    sys.path.insert(0, _sdk_dir)

from sloughgpt_sdk.client import SloughGPTClient, ChatMessage


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


class TestTrainingAnalytics:
    def test_get_training_analytics(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {
            "total_runs": 5, "avg_quality": 0.75, "convergence_rate": 0.6,
            "quality_trend": [], "method_distribution": {"finetune": 3, "lora": 2},
            "model_distribution": {"gpt2": 5}, "best_run": None, "recent_runs": [], "tag_cloud": {}
        })
        result = client.get_training_analytics()
        client._mock_request.assert_called_once_with("GET", "/settings/training/analytics")
        assert result["total_runs"] == 5
        assert result["avg_quality"] == 0.75


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


class TestGetToken:
    def test_exchange_api_key(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"access_token": "jwt-123", "token_type": "bearer"})
        result = client.get_token("my-api-key")
        client._mock_request.assert_called_once_with("POST", "/auth/token", json={"api_key": "my-api-key"})
        assert result["access_token"] == "jwt-123"


class TestHealth:
    def test_health(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"status": "healthy", "model_loaded": True, "model_type": "gpt2"})
        result = client.health()
        client._mock_request.assert_called_once_with("GET", "/health")
        assert result.status == "healthy"

    def test_liveness(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"alive": True})
        result = client.liveness()
        client._mock_request.assert_called_once_with("GET", "/health/live")
        assert result["alive"] is True

    def test_readiness(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"ready": True})
        result = client.readiness()
        client._mock_request.assert_called_once_with("GET", "/health/ready")
        assert result["ready"] is True

    def test_detailed_health(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"status": "ok", "uptime": 1000})
        result = client.detailed_health()
        client._mock_request.assert_called_once_with("GET", "/health/detailed")
        assert result["uptime"] == 1000

    def test_info(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"name": "sloughgpt", "version": "1.0", "model": {"type": "gpt2", "loaded": True}})
        result = client.info()
        client._mock_request.assert_called_once_with("GET", "/info")
        assert result.version == "1.0"


class TestSettings:
    def test_get_settings(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"generation": {"temperature": 0.7}})
        result = client.get_settings()
        client._mock_request.assert_called_once_with("GET", "/settings")
        assert result["generation"]["temperature"] == 0.7

    def test_get_generation_settings(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"temperature": 0.8, "top_p": 0.9})
        result = client.get_generation_settings()
        client._mock_request.assert_called_once_with("GET", "/settings/generation")
        assert result["temperature"] == 0.8

    def test_update_generation_settings(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"temperature": 0.5})
        result = client.update_generation_settings(temperature=0.5)
        client._mock_request.assert_called_once()
        assert result["temperature"] == 0.5

    def test_get_voice_settings(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"noise_gate_db": -40})
        result = client.get_voice_settings()
        client._mock_request.assert_called_once_with("GET", "/settings/voice")
        assert result["noise_gate_db"] == -40

    def test_update_voice_settings(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"noise_gate_db": -35})
        result = client.update_voice_settings(noise_gate_db=-35)
        client._mock_request.assert_called_once()
        assert result["noise_gate_db"] == -35

    def test_reset_settings(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"reset": True})
        result = client.reset_settings()
        client._mock_request.assert_called_once_with("POST", "/settings/reset")
        assert result["reset"] is True

    def test_get_adaptive_insights(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"exploration_rate": 0.3, "confidence": 0.8})
        result = client.get_adaptive_insights()
        client._mock_request.assert_called_once_with("GET", "/settings/adaptive/insights")
        assert result["confidence"] == 0.8


class TestModels:
    def test_list_models(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"models": [{"model_id": "gpt2", "name": "GPT-2"}], "count": 1})
        result = client.list_models()
        client._mock_request.assert_called_once_with("GET", "/models")
        assert len(result) == 1

    def test_load_model(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"loaded": True, "model_id": "gpt2"})
        result = client.load_model("gpt2")
        client._mock_request.assert_called_once()
        assert result["loaded"] is True

    def test_unload_model(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"unloaded": True})
        result = client.unload_model()
        client._mock_request.assert_called_once_with("POST", "/models/unload")
        assert result["unloaded"] is True

    def test_get_current_model(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"model_id": "gpt2", "loaded": True})
        result = client.get_current_model()
        client._mock_request.assert_called_once_with("GET", "/models/current")
        assert result["model_id"] == "gpt2"

    def test_list_hf_models(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"models": [{"id": "gpt2"}]})
        result = client.list_hf_models(query="gpt", limit=5)
        client._mock_request.assert_called_once()
        assert len(result) == 1


class TestSessions:
    def test_create_session(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"session_id": "s1", "created_at": "2026-09-10"})
        result = client.create_session()
        client._mock_request.assert_called_once_with("POST", "/chat/sessions")
        assert result["session_id"] == "s1"

    def test_list_sessions(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"sessions": [{"session_id": "s1"}]})
        result = client.list_sessions()
        client._mock_request.assert_called_once_with("GET", "/chat/sessions")
        assert len(result) == 1

    def test_get_session(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"session_id": "s1", "messages": []})
        result = client.get_session("s1")
        client._mock_request.assert_called_once_with("GET", "/chat/sessions/s1")
        assert result["session_id"] == "s1"

    def test_delete_session(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"deleted": True})
        result = client.delete_session("s1")
        client._mock_request.assert_called_once_with("DELETE", "/chat/sessions/s1")
        assert result["deleted"] is True

    def test_get_session_messages(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"messages": [{"role": "user", "content": "hi"}]})
        result = client.get_session_messages("s1")
        client._mock_request.assert_called_once()
        assert len(result) == 1


class TestKnowledge:
    def test_list_knowledge(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"items": [{"id": "k1", "content": "test"}]})
        result = client.list_knowledge()
        client._mock_request.assert_called_once_with("GET", "/knowledge")
        assert len(result) == 1

    def test_add_knowledge(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"id": "k2", "content": "new fact"})
        result = client.add_knowledge("new fact", topic="ai")
        client._mock_request.assert_called_once()
        assert result["content"] == "new fact"

    def test_delete_knowledge(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"deleted": True})
        result = client.delete_knowledge("k1")
        client._mock_request.assert_called_once()
        assert result["deleted"] is True

    def test_search_knowledge(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"results": [{"id": "k1", "score": 0.95}]})
        result = client.search_knowledge("machine learning")
        client._mock_request.assert_called_once()
        assert len(result) == 1


class TestDatasets:
    def test_list_datasets(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"datasets": [{"dataset_id": "d1", "name": "train"}], "count": 1})
        result = client.list_datasets()
        client._mock_request.assert_called_once_with("GET", "/datasets")
        assert len(result) == 1

    def test_get_dataset(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"id": "d1", "name": "train", "rows": 100})
        result = client.get_dataset("d1")
        client._mock_request.assert_called_once_with("GET", "/datasets/d1")
        assert result.id == "d1"

    def test_get_dataset_stats(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"rows": 100, "columns": 5, "size_bytes": 1024})
        result = client.get_dataset_stats("d1")
        client._mock_request.assert_called_once_with("GET", "/datasets/d1/stats")
        assert result["rows"] == 100


class TestExperiments:
    def test_create_experiment(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"id": "exp1", "name": "test"})
        result = client.create_experiment("test", description="testing")
        client._mock_request.assert_called_once()
        assert result["name"] == "test"

    def test_list_experiments(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"experiments": [{"id": "exp1"}]})
        result = client.list_experiments()
        client._mock_request.assert_called_once_with("GET", "/experiments")
        assert len(result) == 1

    def test_get_experiment(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"id": "exp1", "name": "test"})
        result = client.get_experiment("exp1")
        client._mock_request.assert_called_once_with("GET", "/experiments/exp1")
        assert result["id"] == "exp1"

    def test_log_metric(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"logged": True})
        result = client.log_metric("exp1", "accuracy", 0.95)
        client._mock_request.assert_called_once()
        assert result["logged"] is True

    def test_log_param(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"logged": True})
        result = client.log_param("exp1", "lr", 0.001)
        client._mock_request.assert_called_once()
        assert result["logged"] is True


class TestRateLimit:
    def test_get_rate_limit_status(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"requests_remaining": 100, "reset_at": "2026-09-10T12:00:00Z"})
        result = client.get_rate_limit_status()
        client._mock_request.assert_called_once_with("GET", "/rate-limit/status")
        assert result["requests_remaining"] == 100

    def test_check_rate_limit(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"limited": False})
        result = client.check_rate_limit()
        client._mock_request.assert_called_once_with("GET", "/rate-limit/check")
        assert result["limited"] is False


class TestAuditLog:
    def test_get_audit_log(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"logs": [{"action": "login", "timestamp": "2026-09-10"}]})
        result = client.get_audit_log()
        client._mock_request.assert_called_once_with("GET", "/security/audit")
        assert len(result) == 1


class TestRegistry:
    def test_list_registry_models(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"models": [{"model_id": "m1"}]})
        result = client.list_registry_models()
        client._mock_request.assert_called_once_with("GET", "/registry/models")
        assert len(result) == 1

    def test_get_registry_model(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"model_id": "m1", "score": 0.9})
        result = client.get_registry_model("m1")
        client._mock_request.assert_called_once_with("GET", "/registry/models/m1")
        assert result["model_id"] == "m1"

    def test_get_registry_best(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"model_id": "best-m", "score": 0.99})
        result = client.get_registry_best()
        client._mock_request.assert_called_once_with("GET", "/registry/best")
        assert result["model_id"] == "best-m"

    def test_get_registry_stats(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"total_models": 5, "avg_score": 0.85})
        result = client.get_registry_stats()
        client._mock_request.assert_called_once_with("GET", "/registry/stats")
        assert result["total_models"] == 5


class TestBenchmark:
    def test_run_benchmark(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"job_id": "b1", "status": "running"})
        result = client.run_benchmark({"model": "gpt2", "dataset": "test"})
        client._mock_request.assert_called_once()
        assert result["status"] == "running"

    def test_get_benchmark_metrics(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"metrics": [{"name": "latency", "value": 50}]})
        result = client.get_benchmark_metrics()
        client._mock_request.assert_called_once_with("GET", "/benchmark/metrics")
        assert len(result) == 1

    def test_get_benchmark_stats(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"total_runs": 10, "avg_latency": 45})
        result = client.get_benchmark_stats()
        client._mock_request.assert_called_once_with("GET", "/benchmark/stats")
        assert result["total_runs"] == 10


class TestTokenizer:
    def test_get_tokenizer_stats(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"vocab_size": 50257, "token_count": 1000})
        result = client.get_tokenizer_stats()
        client._mock_request.assert_called_once_with("GET", "/tokenizer/stats")
        assert result["vocab_size"] == 50257


class TestFeedback:
    def test_record_feedback(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"recorded": True})
        result = client.record_feedback("s1", "msg1", score=5, tags=["helpful"])
        client._mock_request.assert_called_once()
        assert result["recorded"] is True

    def test_get_workflow_status(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"active": True, "pending": 0})
        result = client.get_workflow_status()
        client._mock_request.assert_called_once_with("GET", "/workflow/status")
        assert result["active"] is True


class TestMetrics:
    def test_metrics(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"requests_total": 100, "requests_success": 95})
        result = client.metrics()
        client._mock_request.assert_called_once_with("GET", "/metrics")
        assert result.requests_total == 100


class TestGenerate:
    def test_generate(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"generated_text": "Hello world", "model": "gpt2", "inference_time_ms": 50})
        result = client.generate("Hello", max_new_tokens=50, temperature=0.7)
        client._mock_request.assert_called_once()
        assert result.generated_text == "Hello world"
        assert result.model == "gpt2"

    def test_generate_stream(self, client):
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            'data: {"data": {"token": "Hello"}}',
            'data: {"data": {"token": " world"}}',
            'data: [DONE]',
        ]
        client._mock_request.return_value = mock_resp
        tokens = list(client.generate_stream("Hello"))
        client._mock_request.assert_called_once()
        assert tokens == ["Hello", " world"]

    def test_quick_generate(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"generated_text": "Quick result", "model": "gpt2"})
        result = client.quick_generate("Test")
        client._mock_request.assert_called_once()
        assert result == "Quick result"


class TestChat:
    def test_chat_with_messages(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"choices": [{"message": {"role": "assistant", "content": "Hi there"}}], "model": "gpt2", "inference_time_ms": 30})
        result = client.chat([ChatMessage.user("Hello")])
        client._mock_request.assert_called_once()
        assert result.message.content == "Hi there"

    def test_chat_with_dicts(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"choices": [{"message": {"role": "assistant", "content": "Reply"}}], "model": "gpt2"})
        result = client.chat([{"role": "user", "content": "Hi"}])
        client._mock_request.assert_called_once()
        assert result.message.content == "Reply"

    def test_chat_stream(self, client):
        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = [
            'data: {"data": {"token": "Hi"}}',
            'data: {"data": {"token": " there"}}',
        ]
        client._mock_request.return_value = mock_resp
        tokens = list(client.chat_stream([ChatMessage.user("Hello")]))
        assert tokens == ["Hi", " there"]

    def test_quick_chat(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"choices": [{"message": {"role": "assistant", "content": "Quick reply"}}], "model": "gpt2"})
        result = client.quick_chat("Hello")
        client._mock_request.assert_called_once()
        assert result == "Quick reply"


class TestSessionsExtended:
    def test_save_session_context(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"saved": True})
        result = client.save_session_context("s1", {"key": "value"})
        client._mock_request.assert_called_once()
        assert result["saved"] is True


class TestSouls:
    def test_list_souls(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"souls": [{"name": "default"}], "count": 1})
        result = client.list_souls()
        client._mock_request.assert_called_once_with("GET", "/souls")
        assert len(result) == 1

    def test_get_current_soul(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"name": "default", "active": True})
        result = client.get_current_soul()
        client._mock_request.assert_called_once_with("GET", "/souls/current")
        assert result["name"] == "default"

    def test_switch_soul(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"switched": True, "name": "creative"})
        result = client.switch_soul("creative")
        client._mock_request.assert_called_once()
        assert result["name"] == "creative"

    def test_switch_soul_with_checkpoint(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"switched": True})
        result = client.switch_soul("creative", checkpoint_name="ckpt-1")
        client._mock_request.assert_called_once()
        assert result["switched"] is True


class TestKnowledgeExtended:
    def test_get_knowledge_stats(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"total_items": 100, "topics": 5})
        result = client.get_knowledge_stats()
        client._mock_request.assert_called_once_with("GET", "/knowledge/stats")
        assert result["total_items"] == 100

    def test_get_knowledge_topics(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"topics": ["ai", "math"]})
        result = client.get_knowledge_topics()
        client._mock_request.assert_called_once_with("GET", "/knowledge/topics")
        assert "ai" in result

    def test_ingest_knowledge_url(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"id": "k3", "url": "https://example.com"})
        result = client.ingest_knowledge_url("https://example.com")
        client._mock_request.assert_called_once()
        assert result["url"] == "https://example.com"


class TestTokenizerExtended:
    def test_tokenize(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"tokens": [1, 2, 3], "count": 3})
        result = client.tokenize("Hello world")
        client._mock_request.assert_called_once()
        assert result["tokens"] == [1, 2, 3]

    def test_train_tokenizer(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"vocab_size": 1000, "trained": True})
        result = client.train_tokenizer("training text", vocab_size=1000)
        client._mock_request.assert_called_once()
        assert result["trained"] is True


class TestSystem:
    def test_get_system_metrics(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"cpu_percent": 50, "memory_percent": 60})
        result = client.get_system_metrics()
        client._mock_request.assert_called_once_with("GET", "/system/metrics")
        assert result["cpu_percent"] == 50

    def test_get_system_info(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"platform": "linux", "python": "3.12"})
        result = client.get_system_info()
        client._mock_request.assert_called_once_with("GET", "/system/info")
        assert result["platform"] == "linux"

    def test_get_system_disk(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"total_gb": 500, "used_gb": 200})
        result = client.get_system_disk()
        client._mock_request.assert_called_once_with("GET", "/system/disk")
        assert result["total_gb"] == 500


class TestCompanion:
    def test_get_personalities(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"personalities": [{"name": "friendly"}]})
        result = client.get_personalities()
        client._mock_request.assert_called_once_with("GET", "/personalities")
        assert len(result) == 1

    def test_set_personality(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"personality": "friendly", "set": True})
        result = client.set_personality("friendly")
        client._mock_request.assert_called_once()
        assert result["set"] is True

    def test_get_companion_prompt(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"prompt": "You are helpful"})
        result = client.get_companion_prompt()
        client._mock_request.assert_called_once_with("GET", "/companion/prompt")
        assert "prompt" in result

    def test_list_companion_presets(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"presets": [{"name": "default"}]})
        result = client.list_companion_presets()
        client._mock_request.assert_called_once_with("GET", "/companion/presets")
        assert len(result) == 1


class TestDatasetsExtended:
    def test_import_dataset_local(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"dataset_id": "d2", "name": "local"})
        result = client.import_dataset_local("/path/to/data.csv", name="local")
        client._mock_request.assert_called_once()
        assert result["dataset_id"] == "d2"

    def test_import_dataset_github(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"dataset_id": "d3", "name": "github"})
        result = client.import_dataset_github("https://github.com/user/repo", name="github")
        client._mock_request.assert_called_once()
        assert result["dataset_id"] == "d3"

    def test_import_dataset_url(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"dataset_id": "d4", "name": "url"})
        result = client.import_dataset_url("https://example.com/data.csv", name="url")
        client._mock_request.assert_called_once()
        assert result["dataset_id"] == "d4"


class TestTrainingCore:
    def test_start_training(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"job_id": "t1", "status": "pending"})
        result = client.start_training("gpt2", "dataset-1", epochs=5, batch_size=16)
        client._mock_request.assert_called_once()
        assert result["status"] == "pending"

    def test_get_training_status(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"job_id": "t1", "status": "running", "progress": 0.5})
        result = client.get_training_status("t1")
        client._mock_request.assert_called_once_with("GET", "/training/jobs/t1")
        assert result["progress"] == 0.5

    def test_list_training_jobs(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"jobs": [{"job_id": "t1"}], "count": 1})
        result = client.list_training_jobs()
        client._mock_request.assert_called_once_with("GET", "/training/jobs")
        assert len(result) == 1

    def test_delete_training_job(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"deleted": True})
        result = client.delete_training_job("t1")
        client._mock_request.assert_called_once_with("DELETE", "/training/jobs/t1")
        assert result["deleted"] is True

    def test_stop_training(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"stopped": True})
        result = client.stop_training()
        client._mock_request.assert_called_once_with("POST", "/training/control/stop")
        assert result["stopped"] is True

    def test_pause_training(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"paused": True})
        result = client.pause_training()
        client._mock_request.assert_called_once_with("POST", "/training/control/pause")
        assert result["paused"] is True

    def test_resume_training(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"resumed": True})
        result = client.resume_training()
        client._mock_request.assert_called_once_with("POST", "/training/control/resume")
        assert result["resumed"] is True

    def test_get_training_recovery_stats(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"recoverable": 2, "last_checkpoint": "ckpt-5"})
        result = client.get_training_recovery_stats()
        client._mock_request.assert_called_once_with("GET", "/recovery/stats")
        assert result["recoverable"] == 2

    def test_abandon_recovery(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"abandoned": True})
        result = client.abandon_recovery("t1")
        client._mock_request.assert_called_once_with("DELETE", "/recovery/abandon/t1")
        assert result["abandoned"] is True


class TestAutoTrainExtended:
    def test_start_auto_train(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"started": True})
        result = client.start_auto_train({"model": "gpt2", "dataset": "d1"})
        client._mock_request.assert_called_once()
        assert result["started"] is True

    def test_stop_auto_train(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"stopped": True})
        result = client.stop_auto_train()
        client._mock_request.assert_called_once_with("POST", "/training/stop")
        assert result["stopped"] is True

    def test_list_auto_train_checkpoints(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"checkpoints": [{"name": "ckpt-1"}]})
        result = client.list_auto_train_checkpoints()
        client._mock_request.assert_called_once_with("GET", "/training/checkpoints")
        assert len(result) == 1

    def test_delete_auto_train_checkpoint(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"deleted": True})
        result = client.delete_auto_train_checkpoint("ckpt-1")
        client._mock_request.assert_called_once_with("DELETE", "/training/checkpoints/ckpt-1")
        assert result["deleted"] is True

    def test_load_auto_train_checkpoint(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"loaded": True})
        result = client.load_auto_train_checkpoint("ckpt-1")
        client._mock_request.assert_called_once_with("POST", "/training/checkpoints/ckpt-1/load")
        assert result["loaded"] is True


class TestFeedbackExtended:
    def test_get_feedback_stats(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"total": 50, "avg_score": 4.2})
        result = client.get_feedback_stats()
        client._mock_request.assert_called_once_with("GET", "/feedback/stats/summary")
        assert result["total"] == 50


class TestMetricsPrometheus:
    def test_metrics_prometheus(self, client):
        mock_resp = MagicMock()
        mock_resp.text = "# HELP requests_total Total requests\nrequests_total 100"
        client._mock_request.return_value = mock_resp
        result = client.metrics_prometheus()
        client._mock_request.assert_called_once_with("GET", "/metrics/prometheus")
        assert "requests_total" in result


class TestVQA:
    def test_ask_question(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"answer": "a cat", "confidence": 0.9})
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        with patch("builtins.open", return_value=mock_file):
            result = client.ask_question("/tmp/img.png", "What is this?")
        client._mock_request.assert_called_once()
        assert result["answer"] == "a cat"

    def test_detect_objects(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"objects": [{"label": "cat", "bbox": [0, 0, 100, 100]}]})
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        with patch("builtins.open", return_value=mock_file):
            result = client.detect_objects("/tmp/img.png")
        client._mock_request.assert_called_once()
        assert len(result["objects"]) == 1

    def test_analyze_pdf(self, client):
        client._mock_request.return_value = MagicMock(json=lambda: {"analysis": "This is a report", "pages": 5})
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        with patch("builtins.open", return_value=mock_file):
            result = client.analyze_pdf("/tmp/doc.pdf", "Summarize this")
        client._mock_request.assert_called_once()
        assert result["pages"] == 5


class TestOpenWebUI:
    def test_openwebui_datasets(self, client):
        mock_resp = MagicMock()
        mock_resp.__getitem__ = MagicMock(return_value=[{"id": "d1", "name": "train"}])
        client._mock_request.return_value = mock_resp
        result = client.openwebui_datasets()
        client._mock_request.assert_called_once_with("GET", "/openwebui/datasets")
        assert len(result) == 1

    def test_openwebui_checkpoints(self, client):
        mock_resp = MagicMock()
        mock_resp.__getitem__ = MagicMock(return_value=[{"name": "ckpt-1"}])
        client._mock_request.return_value = mock_resp
        result = client.openwebui_checkpoints()
        client._mock_request.assert_called_once_with("GET", "/openwebui/checkpoints")
        assert len(result) == 1

    def test_openwebui_start_training(self, client):
        mock_resp = MagicMock()
        mock_resp.__getitem__ = lambda self, key: {"started": True, "job_id": "j1"}[key]
        client._mock_request.return_value = mock_resp
        result = client.openwebui_start_training("d1", "finetune")
        client._mock_request.assert_called_once()
        assert result["started"] is True

    def test_openwebui_stop_training(self, client):
        mock_resp = MagicMock()
        mock_resp.__getitem__ = lambda self, key: {"stopped": True}[key]
        client._mock_request.return_value = mock_resp
        result = client.openwebui_stop_training()
        client._mock_request.assert_called_once_with("POST", "/openwebui/training/stop")
        assert result["stopped"] is True

    def test_openwebui_training_status(self, client):
        mock_resp = MagicMock()
        mock_resp.__getitem__ = lambda self, key: {"status": "running", "progress": 0.5}[key]
        client._mock_request.return_value = mock_resp
        result = client.openwebui_training_status()
        client._mock_request.assert_called_once_with("GET", "/openwebui/training/status")
        assert result["status"] == "running"


class TestCloudTraining:
    def test_cloud_training_jobs(self, client):
        mock_resp = MagicMock()
        mock_resp.__getitem__ = MagicMock(return_value=[{"id": "c1", "status": "running"}])
        client._mock_request.return_value = mock_resp
        result = client.cloud_training_jobs(limit=5)
        client._mock_request.assert_called_once()
        assert len(result) == 1

    def test_cloud_training_submit(self, client):
        mock_resp = MagicMock()
        mock_resp.__getitem__ = lambda self, key: {"job_id": "c2", "status": "submitted"}[key]
        client._mock_request.return_value = mock_resp
        result = client.cloud_training_submit("aws", "d1")
        client._mock_request.assert_called_once()
        assert result["status"] == "submitted"

    def test_cloud_training_status(self, client):
        mock_resp = MagicMock()
        mock_resp.__getitem__ = lambda self, key: {"job_id": "c1", "status": "completed"}[key]
        client._mock_request.return_value = mock_resp
        result = client.cloud_training_status("c1")
        client._mock_request.assert_called_once_with("GET", "/cloud-training/c1/status")
        assert result["status"] == "completed"

    def test_cloud_training_cancel(self, client):
        mock_resp = MagicMock()
        mock_resp.__getitem__ = lambda self, key: {"cancelled": True}[key]
        client._mock_request.return_value = mock_resp
        result = client.cloud_training_cancel("c1")
        client._mock_request.assert_called_once_with("POST", "/cloud-training/c1/cancel")
        assert result["cancelled"] is True


class TestPlugins:
    def test_plugins_list(self, client):
        mock_resp = MagicMock()
        mock_resp.__getitem__ = MagicMock(return_value=[{"name": "plugin-a", "enabled": True}])
        client._mock_request.return_value = mock_resp
        result = client.plugins_list()
        client._mock_request.assert_called_once_with("GET", "/plugins")
        assert len(result) == 1

    def test_plugins_enable(self, client):
        mock_resp = MagicMock()
        mock_resp.__getitem__ = lambda self, key: {"enabled": True}[key]
        client._mock_request.return_value = mock_resp
        result = client.plugins_enable("plugin-a")
        client._mock_request.assert_called_once_with("POST", "/plugins/plugin-a/enable")
        assert result["enabled"] is True

    def test_plugins_disable(self, client):
        mock_resp = MagicMock()
        mock_resp.__getitem__ = lambda self, key: {"disabled": True}[key]
        client._mock_request.return_value = mock_resp
        result = client.plugins_disable("plugin-a")
        client._mock_request.assert_called_once_with("POST", "/plugins/plugin-a/disable")
        assert result["disabled"] is True

    def test_plugins_reload(self, client):
        mock_resp = MagicMock()
        mock_resp.__getitem__ = lambda self, key: {"reloaded": True, "count": 3}[key]
        client._mock_request.return_value = mock_resp
        result = client.plugins_reload()
        client._mock_request.assert_called_once_with("POST", "/plugins/reload")
        assert result["reloaded"] is True


# ── Training Analytics ────────────────────────────────────────────────────────


class TestTrainingAnalytics:
    def test_get_training_analytics(self, client):
        client._mock_request.return_value = MagicMock(
            json=lambda: {"total_runs": 10, "avg_loss": 0.5, "by_model": {}}
        )
        result = client.get_training_analytics()
        client._mock_request.assert_called_once_with("GET", "/settings/training/analytics")
        assert result["total_runs"] == 10


# ── Docstore ──────────────────────────────────────────────────────────────────


class TestDocstore:
    def test_list_docstore_docs(self, client):
        client._mock_request.return_value = MagicMock(
            json=lambda: [{"id": "d1", "content": "hello"}, {"id": "d2", "content": "world"}]
        )
        result = client.list_docstore_docs("notes")
        client._mock_request.assert_called_once_with("GET", "/docstore/notes")
        assert len(result) == 2

    def test_get_docstore_doc(self, client):
        client._mock_request.return_value = MagicMock(
            json=lambda: {"id": "d1", "content": "hello", "meta": {}}
        )
        result = client.get_docstore_doc("notes", "d1")
        client._mock_request.assert_called_once_with("GET", "/docstore/notes/d1")
        assert result["id"] == "d1"

    def test_put_docstore_doc(self, client):
        client._mock_request.return_value = MagicMock(
            json=lambda: {"id": "d1", "status": "stored"}
        )
        data = {"content": "hello", "meta": {"tag": "test"}}
        result = client.put_docstore_doc("notes", "d1", data)
        client._mock_request.assert_called_once_with("PUT", "/docstore/notes/d1", json=data)
        assert result["status"] == "stored"

    def test_patch_docstore_doc(self, client):
        client._mock_request.return_value = MagicMock(
            json=lambda: {"id": "d1", "status": "updated"}
        )
        patch_data = {"content": "updated"}
        result = client.patch_docstore_doc("notes", "d1", patch_data)
        client._mock_request.assert_called_once_with("PATCH", "/docstore/notes/d1", json=patch_data)
        assert result["status"] == "updated"

    def test_delete_docstore_doc(self, client):
        client._mock_request.return_value = MagicMock(
            json=lambda: {"deleted": "d1"}
        )
        result = client.delete_docstore_doc("notes", "d1")
        client._mock_request.assert_called_once_with("DELETE", "/docstore/notes/d1")
        assert result["deleted"] == "d1"

    def test_clear_docstore_collection(self, client):
        client._mock_request.return_value = MagicMock(
            json=lambda: {"cleared": 5}
        )
        result = client.clear_docstore_collection("notes")
        client._mock_request.assert_called_once_with("DELETE", "/docstore/notes")
        assert result["cleared"] == 5

    def test_bulk_put_docstore(self, client):
        client._mock_request.return_value = MagicMock(
            json=lambda: {"stored": 3}
        )
        docs = [{"id": "d1"}, {"id": "d2"}, {"id": "d3"}]
        result = client.bulk_put_docstore("notes", docs)
        client._mock_request.assert_called_once_with("POST", "/docstore/notes/bulk", json=docs)
        assert result["stored"] == 3


# ── Collections ───────────────────────────────────────────────────────────────


class TestCollections:
    def test_list_collections(self, client):
        client._mock_request.return_value = MagicMock(
            json=lambda: [{"id": "c1", "name": "web-scrape"}]
        )
        result = client.list_collections()
        client._mock_request.assert_called_once_with("GET", "/collections")
        assert len(result) == 1

    def test_get_collection(self, client):
        client._mock_request.return_value = MagicMock(
            json=lambda: {"id": "c1", "name": "web-scrape", "status": "idle"}
        )
        result = client.get_collection("c1")
        client._mock_request.assert_called_once_with("GET", "/collections/c1")
        assert result["name"] == "web-scrape"

    def test_create_collection(self, client):
        client._mock_request.return_value = MagicMock(
            json=lambda: {"id": "c2", "name": "new-pipeline"}
        )
        result = client.create_collection("new-pipeline", source="rss")
        client._mock_request.assert_called_once_with(
            "POST", "/collections/create", json={"name": "new-pipeline", "source": "rss"}
        )
        assert result["name"] == "new-pipeline"

    def test_delete_collection(self, client):
        client._mock_request.return_value = MagicMock(
            json=lambda: {"deleted": "c1"}
        )
        result = client.delete_collection("c1")
        client._mock_request.assert_called_once_with("DELETE", "/collections/c1")
        assert result["deleted"] == "c1"

    def test_run_collection(self, client):
        client._mock_request.return_value = MagicMock(
            json=lambda: {"status": "running", "job_id": "j1"}
        )
        result = client.run_collection("c1", max_items=100)
        client._mock_request.assert_called_once_with(
            "POST", "/collections/run", json={"pipeline_id": "c1", "max_items": 100}
        )
        assert result["status"] == "running"

    def test_collect_from_collection(self, client):
        client._mock_request.return_value = MagicMock(
            json=lambda: {"collected": 50}
        )
        result = client.collect_from_collection("c1", query="python")
        client._mock_request.assert_called_once_with(
            "POST", "/collections/c1/collect", json={"query": "python"}
        )
        assert result["collected"] == 50

    def test_get_collection_records(self, client):
        client._mock_request.return_value = MagicMock(
            json=lambda: [{"url": "http://example.com", "title": "Example"}]
        )
        result = client.get_collection_records("c1")
        client._mock_request.assert_called_once_with("GET", "/collections/c1/records")
        assert len(result) == 1

    def test_get_collection_stats(self, client):
        client._mock_request.return_value = MagicMock(
            json=lambda: {"total_collections": 5, "active": 2}
        )
        result = client.get_collection_stats()
        client._mock_request.assert_called_once_with("GET", "/collections/stats")
        assert result["total_collections"] == 5
