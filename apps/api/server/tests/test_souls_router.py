"""Tests for souls router endpoints."""
import pytest

from tests.test_support import get_test_client

client = get_test_client()


class TestListSouls:
    def test_list_souls_returns_success(self):
        resp = client.get("/souls")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["data"], list)

    def test_list_souls_has_meta(self):
        resp = client.get("/souls")
        body = resp.json()
        assert "meta" in body
        assert "current_soul" in body["meta"]


class TestCurrentSoul:
    def test_get_current_soul(self):
        resp = client.get("/souls/current")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"

    def test_current_soul_has_name_or_null(self):
        resp = client.get("/souls/current")
        body = resp.json()
        assert "name" in body["data"]


class TestTraitWeights:
    def test_get_weights(self):
        resp = client.get("/souls/weights")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"

    def test_get_weights_has_structure(self):
        resp = client.get("/souls/weights")
        body = resp.json()
        data = body["data"]
        assert isinstance(data, dict)
        assert "personality" in data
        assert "cognition" in data
        assert "emotion" in data

    def test_save_weights(self):
        resp = client.post("/souls/weights", json={
            "personality": {"warmth": 0.8},
            "cognition": {"reasoning": 0.7},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"

    def test_get_trait_modes(self):
        resp = client.get("/souls/weights/modes")
        assert resp.status_code == 200
        body = resp.json()
        data = body["data"]
        assert "personality" in data
        assert "memory" in data
        assert "style" in data
        assert "task" in data

    def test_trait_modes_have_label_and_confidence(self):
        resp = client.get("/souls/weights/modes")
        data = resp.json()["data"]
        for key in ("personality", "memory", "style", "task"):
            mode = data[key]
            assert "label" in mode
            assert "confidence" in mode


class TestWeightSnapshots:
    def test_list_snapshots(self):
        resp = client.get("/souls/weights/snapshots")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["data"], list)

    def test_save_snapshot(self):
        resp = client.post("/souls/weights/snapshot/test-snap")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"

    def test_load_snapshot(self):
        client.post("/souls/weights/snapshot/load-test")
        resp = client.post("/souls/weights/snapshot/load-test/load")
        assert resp.status_code == 200
        assert resp.json()["data"]["traits_loaded"] >= 0

    def test_delete_snapshot(self):
        client.post("/souls/weights/snapshot/del-test")
        resp = client.delete("/souls/weights/snapshot/del-test")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"]["deleted"] is True


class TestSoulStats:
    def test_stats_returns_dict(self):
        resp = client.get("/souls/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert isinstance(body["data"], dict)


class TestSwitchSoul:
    def test_switch_nonexistent_soul(self):
        resp = client.post("/souls/switch", json={"name": "nonexistent_soul_xyz"})
        assert resp.status_code == 200
        body = resp.json()
        assert "error" in body or body["data"].get("success") is False


class TestSchemaValidation:
    def test_switch_missing_name(self):
        resp = client.post("/souls/switch", json={})
        assert resp.status_code == 422

    def test_save_weights_invalid_body(self):
        resp = client.post("/souls/weights", json="not a dict")
        assert resp.status_code == 422
