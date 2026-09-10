"""Tests for the souls router — personality/soul management endpoints.

Covers all 13 endpoints:
  GET  /souls                          — list_souls
  GET  /souls/current                  — get_current_soul
  GET  /souls/{soul_name}              — get_soul
  POST /souls/switch                   — switch_soul
  POST /souls/chat                     — soul_chat
  GET  /souls/weights                  — get_trait_weights
  POST /souls/weights                  — save_trait_weights
  GET  /souls/weights/modes            — get_trait_modes
  GET  /souls/weights/snapshots        — list_weight_snapshots
  POST /souls/weights/snapshot/{name}  — save_weight_snapshot
  POST /souls/weights/snapshot/{name}/load — load_weight_snapshot
  DELETE /souls/weights/snapshot/{name} — delete_weight_snapshot
  GET  /souls/stats                    — get_soul_stats
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from test_support import get_test_client


def _d(resp):
    j = resp.json()
    return j.get("data", j)


def _mock_soul(name="test-soul", personality=None):
    m = MagicMock()
    m.name = name
    m.path = f"/models/{name}.slo"
    m.description = f"A {name} soul"
    m.personality = personality or {"warmth": 0.8, "creativity": 0.6}
    m.traits = ["analytical", "helpful"]
    m.born_at = "2025-01-01"
    m.training_dataset = "test"
    m.epochs_trained = 10
    m.final_train_loss = 0.5
    m.final_val_loss = 0.6
    m.lineage = ""
    m.base_model = ""
    m.version = "1.0"
    m.size_mb = 1.2
    m.behavior = {}
    m.cognition = {}
    m.emotion = {}
    m.generation_params = {}
    return m


def _mock_manager(souls=None, current=None):
    m = MagicMock()
    m.list_souls.return_value = souls or []
    m.get_current_soul.return_value = current
    m.get_soul.side_effect = lambda name: (
        next((s for s in (souls or [] if souls else m.list_souls.return_value) if s.name == name), None)
    )
    m.switch_soul.return_value = {"success": True, "soul": "test"}
    m.get_trait_weights.return_value = {
        "personality": {"warmth": 0.8},
        "cognition": {"reasoning": 0.7},
        "emotion": {"empathy": 0.9},
    }
    m.get_stats.return_value = {"soul_count": 1, "current_soul": "test"}
    return m


@pytest.fixture(autouse=True)
def _clear_souls_cache():
    """Clear the module-level list_souls cache before each test."""
    import routers.souls as souls_mod
    souls_mod._list_souls_cache = None
    yield
    souls_mod._list_souls_cache = None


class TestListSouls:
    def setup_method(self):
        self.client = get_test_client()

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_list_returns_success(self, mock_get):
        mock_get.return_value = _mock_manager(souls=[_mock_soul()])
        resp = self.client.get("/souls")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        data = _d(resp)
        assert isinstance(data, list)

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_list_has_meta(self, mock_get):
        soul = _mock_soul()
        mock_get.return_value = _mock_manager(souls=[soul], current=soul)
        resp = self.client.get("/souls")
        body = resp.json()
        assert "meta" in body
        assert body["meta"]["current_soul"] == "test-soul"

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_list_empty_souls(self, mock_get):
        mock_get.return_value = _mock_manager(souls=[])
        resp = self.client.get("/souls")
        assert resp.status_code == 200
        assert _d(resp) == []

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_list_soul_has_all_fields(self, mock_get):
        soul = _mock_soul()
        mock_get.return_value = _mock_manager(souls=[soul])
        resp = self.client.get("/souls")
        soul_data = _d(resp)[0]
        for key in ("name", "path", "description", "personality", "traits", "born_at"):
            assert key in soul_data


class TestCurrentSoul:
    def setup_method(self):
        self.client = get_test_client()

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_get_current_soul(self, mock_get):
        soul = _mock_soul()
        mock_get.return_value = _mock_manager(current=soul)
        resp = self.client.get("/souls/current")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_current_soul_has_name(self, mock_get):
        soul = _mock_soul()
        mock_get.return_value = _mock_manager(current=soul)
        resp = self.client.get("/souls/current")
        assert _d(resp)["name"] == "test-soul"

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_no_current_soul(self, mock_get):
        mock_get.return_value = _mock_manager(current=None)
        resp = self.client.get("/souls/current")
        assert resp.status_code == 200
        assert _d(resp)["name"] is None


class TestGetSoul:
    def setup_method(self):
        self.client = get_test_client()

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_get_soul_by_name(self, mock_get):
        soul = _mock_soul("wise-owl")
        mock_get.return_value = _mock_manager(souls=[soul])
        resp = self.client.get("/souls/wise-owl")
        assert resp.status_code == 200
        assert _d(resp)["name"] == "wise-owl"

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_get_nonexistent_soul(self, mock_get):
        mock_get.return_value = _mock_manager(souls=[])
        resp = self.client.get("/souls/nonexistent")
        assert resp.status_code == 404

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_get_soul_has_traits(self, mock_get):
        soul = _mock_soul()
        mock_get.return_value = _mock_manager(souls=[soul])
        resp = self.client.get("/souls/test-soul")
        assert "personality" in _d(resp)
        assert "traits" in _d(resp)


class TestTraitWeights:
    def setup_method(self):
        self.client = get_test_client()

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_get_weights(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = self.client.get("/souls/weights")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_get_weights_structure(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = self.client.get("/souls/weights")
        data = _d(resp)
        assert "personality" in data
        assert "cognition" in data
        assert "emotion" in data

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_save_weights(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = self.client.post(
            "/souls/weights",
            json={
                "personality": {"warmth": 0.8},
                "cognition": {"reasoning": 0.7},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_get_trait_modes(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = self.client.get("/souls/weights/modes")
        assert resp.status_code == 200
        data = _d(resp)
        for key in ("personality", "memory", "style", "task"):
            assert key in data

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_trait_modes_have_structure(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = self.client.get("/souls/weights/modes")
        data = _d(resp)
        for key in ("personality", "memory", "style", "task"):
            mode = data[key]
            assert "label" in mode
            assert "confidence" in mode


class TestWeightSnapshots:
    def setup_method(self):
        self.client = get_test_client()

    @patch("domains.context.managers.get_trait_config")
    def test_list_snapshots(self, mock_get):
        mock_get.return_value = MagicMock(list_snapshots=MagicMock(return_value=["snap1", "snap2"]))
        resp = self.client.get("/souls/weights/snapshots")
        assert resp.status_code == 200
        data = _d(resp)
        assert isinstance(data, list)
        assert "snap1" in data

    @patch("domains.context.managers.get_trait_config")
    def test_save_snapshot(self, mock_get):
        config = MagicMock()
        config.save_snapshot.return_value = "/snapshots/test.json"
        mock_get.return_value = config
        resp = self.client.post("/souls/weights/snapshot/test-snap")
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @patch("domains.context.managers.get_trait_config")
    def test_load_snapshot(self, mock_get):
        config = MagicMock()
        config.load_snapshot.return_value = 5
        mock_get.return_value = config
        resp = self.client.post("/souls/weights/snapshot/load-test/load")
        assert resp.status_code == 200
        assert _d(resp)["traits_loaded"] == 5

    @patch("domains.context.managers.get_trait_config")
    def test_delete_snapshot(self, mock_get):
        config = MagicMock()
        config.delete_snapshot.return_value = True
        mock_get.return_value = config
        resp = self.client.delete("/souls/weights/snapshot/del-test")
        assert resp.status_code == 200
        assert _d(resp)["deleted"] is True


class TestSwitchSoul:
    def setup_method(self):
        self.client = get_test_client()

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_switch_success(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = self.client.post("/souls/switch", json={"name": "new-soul"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_switch_nonexistent_soul(self, mock_get):
        m = _mock_manager()
        m.switch_soul.return_value = {"success": False, "error": "not found"}
        mock_get.return_value = m
        resp = self.client.post("/souls/switch", json={"name": "nonexistent"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"].get("success") is False


class TestSoulStats:
    def setup_method(self):
        self.client = get_test_client()

    @patch("domains.inference.slo_manager.get_slo_manager")
    def test_stats_returns_dict(self, mock_get):
        mock_get.return_value = _mock_manager()
        resp = self.client.get("/souls/stats")
        assert resp.status_code == 200
        data = _d(resp)
        assert isinstance(data, dict)


class TestSchemaValidation:
    def setup_method(self):
        self.client = get_test_client()

    def test_switch_missing_name(self):
        resp = self.client.post("/souls/switch", json={})
        assert resp.status_code == 422

    def test_save_weights_invalid_body(self):
        resp = self.client.post("/souls/weights", json="not a dict")
        assert resp.status_code == 422
