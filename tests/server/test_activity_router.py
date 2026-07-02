"""
Tests for the activity router — sensor data, training, prediction, download.
"""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.activity import router, ACTIVITY_NAMES

app = FastAPI()
app.include_router(router)
client = TestClient(app, raise_server_exceptions=False)

DATA_TARGET = "apps.api.server.routers.activity._DATA_DIR"
MODEL_TARGET = "apps.api.server.routers.activity._MODEL"


@pytest.fixture(autouse=True)
def clear_model():
    """Clear global _MODEL before each test to prevent cross-test leakage."""
    import apps.api.server.routers.activity as act_router
    act_router._MODEL = None
    # Also patch _maybe_load_model to always return False and keep _MODEL=None
    with patch("apps.api.server.routers.activity._maybe_load_model", return_value=False):
        yield


@pytest.fixture
def tmp_data_dir(tmp_path: Path):
    with patch(DATA_TARGET, tmp_path / "activity_records"):
        (tmp_path / "activity_records").mkdir(parents=True, exist_ok=True)
        yield tmp_path / "activity_records"


class TestRecordData:
    """POST /activity/data"""

    def test_record_basic(self, tmp_data_dir: Path):
        body = {"data": [[0.1, 0.2, 9.81, 0.01, 0.02, 0.03]]}
        resp = client.post("/activity/data", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == 1
        assert data["samples"] == 1

    def test_record_with_label(self, tmp_data_dir: Path):
        body = {"data": [[0.1, 0.2, 9.81, 0.01, 0.02, 0.03]], "label": 2}
        resp = client.post("/activity/data", json=body)
        assert resp.status_code == 200
        d = np.load(tmp_data_dir / "1.npz")
        assert int(d["label"]) == 2

    def test_record_invalid_shape(self, tmp_data_dir: Path):
        body = {"data": [[0.1, 0.2]]}  # only 2 cols, not 6
        resp = client.post("/activity/data", json=body)
        assert resp.status_code == 400

    def test_record_increments_id(self, tmp_data_dir: Path):
        for i in range(3):
            body = {"data": [[0.1] * 6], "label": 0}
            resp = client.post("/activity/data", json=body)
            assert resp.json()["id"] == i + 1


class TestStatus:
    """GET /activity/status"""

    def test_status_empty(self, tmp_data_dir: Path):
        resp = client.get("/activity/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_recordings"] == 0
        assert data["model_loaded"] is False
        assert data["activities"] == ACTIVITY_NAMES

    def test_status_with_recordings(self, tmp_data_dir: Path):
        client.post("/activity/data", json={"data": [[0.1] * 6], "label": 1})
        client.post("/activity/data", json={"data": [[0.1] * 6]})
        resp = client.get("/activity/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["num_recordings"] == 2
        assert data["num_labels"] == 1


class TestDataset:
    """GET /activity/dataset"""

    def test_list_empty(self, tmp_data_dir: Path):
        resp = client.get("/activity/dataset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0

    def test_list_with_items(self, tmp_data_dir: Path):
        client.post("/activity/data", json={"data": [[0.1] * 6], "label": 3})
        client.post("/activity/data", json={"data": [[0.1] * 6]})
        resp = client.get("/activity/dataset")
        data = resp.json()
        assert data["total"] == 2
        items = data["recordings"]
        assert items[0]["activity"] == "shaking"  # label=3 → 0-indexed, index 3 = shaking
        assert items[1]["activity"] == "unlabeled"


class TestTrain:
    """POST /activity/train"""

    def test_train_too_few_recordings(self, tmp_data_dir: Path):
        body = {"data": [[0.1] * 6], "label": 0}
        client.post("/activity/data", json=body)
        resp = client.post("/activity/train", json={"epochs": 1})
        assert resp.status_code == 400
        assert "5 recordings" in resp.json()["detail"]

    def test_train_too_few_labeled(self, tmp_data_dir: Path):
        for i in range(6):
            client.post("/activity/data", json={"data": [[0.1] * 6]})
        resp = client.post("/activity/train", json={"epochs": 1})
        assert resp.status_code == 400
        assert "labeled" in resp.json()["detail"]


class TestPredict:
    """POST /activity/predict"""

    def test_predict_no_model(self, tmp_data_dir: Path):
        resp = client.post("/activity/predict", json={"data": [[0.1] * 6]})
        assert resp.status_code == 400
        assert "No trained model" in resp.json()["detail"]

    def test_predict_with_mocked_model(self, tmp_data_dir: Path):
        """Test predict endpoint with mock model and patched predict_activity at domain level."""
        def patched_predict(model, arr):
            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]
            probs = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.75], dtype=np.float32)
            cls_id = int(np.argmax(probs))
            return cls_id, ACTIVITY_NAMES[cls_id], probs

        mock_model = MagicMock()

        with patch(MODEL_TARGET, mock_model), \
             patch("domains.activity.predict_activity", patched_predict):
            resp = client.post("/activity/predict", json={"data": [[0.1] * 6]})
            assert resp.status_code == 200
            data = resp.json()
            assert data["activity"] == "cycling"
            assert data["class_id"] == 5
            assert data["confidence"] == 0.75
            assert len(data["probabilities"]) == 6

    def test_predict_invalid_shape(self, tmp_data_dir: Path):
        with patch(MODEL_TARGET, MagicMock()):
            resp = client.post("/activity/predict", json={"data": [[0.1, 0.2]]})
            assert resp.status_code == 400


class TestDelete:
    """DELETE /activity/data"""

    def test_delete_empty(self, tmp_data_dir: Path):
        resp = client.delete("/activity/data")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 0

    def test_delete_with_data(self, tmp_data_dir: Path):
        client.post("/activity/data", json={"data": [[0.1] * 6], "label": 0})
        client.post("/activity/data", json={"data": [[0.1] * 6], "label": 1})
        resp = client.delete("/activity/data")
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 2
        # Verify model cleared
        from apps.api.server.routers.activity import _MODEL
        assert _MODEL is None


class TestTrainStream:
    """POST /activity/train/stream"""

    def test_train_stream_returns_events(self, tmp_data_dir: Path):
        # Seed enough labeled data
        for i in range(6):
            client.post("/activity/data", json={
                "data": [[float(j) for j in range(6)]],
                "label": i,
            })
        resp = client.post("/activity/train/stream", json={"epochs": 2})
        assert resp.status_code == 200
        events = [json.loads(line[6:]) for line in resp.text.strip().split("\n\n") if line.startswith("data:")]
        assert len(events) > 0

    def test_train_stream_too_few(self, tmp_data_dir: Path):
        resp = client.post("/activity/train/stream", json={"epochs": 1})
        assert resp.status_code == 200
        events = [json.loads(line[6:]) for line in resp.text.strip().split("\n\n") if line.startswith("data:")]
        assert len(events) == 1
        assert events[0]["status"] == "error"


class TestModelDownload:
    """GET /activity/model"""

    def test_download_no_model(self, tmp_data_dir: Path):
        with patch("apps.api.server.routers.activity._REPO_ROOT", tmp_data_dir):
            resp = client.get("/activity/model")
            assert resp.status_code == 404

    def test_download_model(self, tmp_data_dir: Path):
        # Create a dummy model.npz
        model_dir = tmp_data_dir / "packages" / "core-py" / "domains" / "activity"
        model_dir.mkdir(parents=True)
        fake = np.array([1, 2, 3])
        np.savez_compressed(str(model_dir / "model.npz"), data=fake)
        with patch("apps.api.server.routers.activity._REPO_ROOT", tmp_data_dir):
            resp = client.get("/activity/model")
            assert resp.status_code == 200
            assert resp.headers["content-type"] == "application/octet-stream"
