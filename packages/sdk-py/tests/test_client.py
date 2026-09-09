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
