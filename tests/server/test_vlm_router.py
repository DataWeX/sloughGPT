"""
Tests for the VLM router endpoints.
"""
import pytest
from fastapi.testclient import TestClient

try:
    from apps.api.server.main import app
    client = TestClient(app)
except Exception:
    pytest.skip("Server app not available", allow_module_level=True)


class TestVLMRouter:
    """Tests for /vlm/* endpoints."""

    def test_vlm_status_no_load(self):
        """GET /vlm/status returns vlm_loaded=false without triggering model load."""
        resp = client.get("/vlm/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["vlm_loaded"] is False
        assert "dpo" in data

    def test_dpo_status_idle(self):
        """GET /vlm/dpo/status returns idle state."""
        resp = client.get("/vlm/dpo/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "idle"
        assert data["last_run"] is None

    def test_dpo_no_model(self):
        """POST /vlm/dpo returns 400 when no model loaded."""
        resp = client.post(
            "/vlm/dpo",
            json={"max_pairs": 2},
        )
        assert resp.status_code == 400
        assert "No model loaded" in resp.json()["detail"]

    def test_vlm_generate_no_model(self):
        """POST /vlm/generate returns 400 when VLM not loaded."""
        resp = client.post(
            "/vlm/generate",
            json={"image_base64": "dGVzdA==", "prompt": "test"},
        )
        assert resp.status_code == 400
        assert "VLM not loaded" in resp.json()["detail"]

    def test_vlm_load_no_checkpoint(self):
        """POST /vlm/load returns 500 when checkpoint doesn't exist."""
        resp = client.post(
            "/vlm/load",
            json={"model_dir": "/tmp/nonexistent_vlm_checkpoint"},
        )
        assert resp.status_code == 500


class TestMultimodalDatasetEndpoint:
    """Tests for the VLM dataset creation endpoint."""

    def test_vlm_dataset_no_dir(self):
        """POST /multimodal/vlm-dataset with nonexistent directory returns 400."""
        resp = client.post(
            "/multimodal/vlm-dataset",
            json={
                "name": "test-dataset",
                "image_dir": "/tmp/nonexistent_vlm_test_dir_xyz",
            },
        )
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"].lower()

    def test_vlm_dataset_missing_name(self):
        """POST /multimodal/vlm-dataset without name returns 422."""
        resp = client.post(
            "/multimodal/vlm-dataset",
            json={"image_dir": "/tmp"},
        )
        assert resp.status_code == 422

    def test_models_vlm_load_no_dir(self):
        """POST /models/vlm-load with nonexistent directory returns 400."""
        resp = client.post(
            "/models/vlm-load",
            params={"model_dir": "/tmp/nonexistent_vlm", "model_id": "test"},
        )
        assert resp.status_code == 400

    def test_models_vlm_load_missing_params(self):
        """POST /models/vlm-load without required params returns 422."""
        resp = client.post("/models/vlm-load")
        assert resp.status_code == 422
