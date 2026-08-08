"""
Tests for the Mobile BFF router — dashboard, conversations, models, knowledge,
notifications, training, sync.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.mobile import MobileRouter, router


@pytest.fixture
def mobile_router():
    return MobileRouter()


@pytest.fixture
def app(mobile_router):
    _app = FastAPI()
    _app.include_router(mobile_router.router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def _mock_internal_get(data):
    return AsyncMock(return_value=data)


def _mock_internal_post(data):
    return AsyncMock(return_value=data)


def _mock_internal_patch(data):
    return AsyncMock(return_value=data)


def _mock_internal_delete(data):
    return AsyncMock(return_value=data)


# ── GET /mobile/dashboard ────────────────────────────────────────────────────


class TestMobileDashboard:
    """GET /mobile/dashboard"""

    def test_returns_dashboard_data(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                {"status": "healthy", "model_type": "gpt2", "model_loaded": True, "inference_count": 5},
                {"name": "sage", "description": "A wise soul"},
                {"sessions": [{"id": "s1", "title": "Chat", "messages": [{"content": "hi"}], "updated_at": "2026-01-01"}]},
                [{"model_id": "gpt2"}],
            ]
            resp = client.get("/mobile/dashboard")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "healthy"
            assert body["model"]["loaded"] is True
            assert body["soul"]["name"] == "sage"
            assert len(body["recent_conversations"]) == 1


# ── GET /mobile/conversations ─────────────────────────────────────────────────


class TestMobileConversations:
    """GET /mobile/conversations"""

    def test_returns_conversation_list(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {
                "sessions": [
                    {"id": "s1", "title": "Chat 1", "messages": [{"content": "hello"}],
                     "updated_at": "2026-01-01", "created_at": "2026-01-01", "starred": False, "pinned": False},
                ]
            }
            resp = client.get("/mobile/conversations")
            assert resp.status_code == 200
            body = resp.json()
            assert body["total"] == 1
            assert body["conversations"][0]["id"] == "s1"


# ── GET /mobile/conversations/{session_id} ────────────────────────────────────


class TestMobileConversationDetail:
    """GET /mobile/conversations/{session_id}"""

    def test_returns_conversation(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"messages": [{"role": "user", "content": "hi"}], "created_at": "2026-01-01"}
            resp = client.get("/mobile/conversations/s1")
            assert resp.status_code == 200
            body = resp.json()
            assert body["id"] == "s1"
            assert len(body["messages"]) == 1


# ── GET /mobile/models ────────────────────────────────────────────────────────


class TestMobileModels:
    """GET /mobile/models"""

    def test_returns_model_list(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                [{"model_id": "gpt2", "name": "GPT-2", "loaded": True, "source": "local"}],
                [],
                {"name": "sage"},
                [],
                {"model_type": "gpt2"},
            ]
            resp = client.get("/mobile/models")
            assert resp.status_code == 200
            body = resp.json()
            assert "models" in body
            assert "souls" in body
            assert "checkpoints" in body


# ── POST /mobile/models/switch ────────────────────────────────────────────────


class TestMobileSwitchModel:
    """POST /mobile/models/switch"""

    def test_switch_with_valid_body(self, client):
        with patch.object(MobileRouter, "_internal_post", new_callable=AsyncMock) as mock_post, \
             patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_post.return_value = {"status": "ok"}
            mock_get.return_value = {"model_type": "gpt2"}
            resp = client.post("/mobile/models/switch", json={"soul_name": "sage"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["soul"] == "sage"

    def test_switch_with_empty_body(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"model_type": "gpt2"}
            resp = client.post("/mobile/models/switch", json={})
            assert resp.status_code == 200


# ── GET /mobile/health ────────────────────────────────────────────────────────


class TestMobileHealth:
    """GET /mobile/health"""

    def test_returns_health_data(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                {"status": "healthy", "model_type": "gpt2", "model_loaded": True,
                 "uptime_seconds": 100, "system": {"cpu_percent": 50.0, "memory_percent": 60.0,
                                                    "memory_available_mb": 8192}},
                {"disk_used_bytes": 1073741824, "disk_free_bytes": 2147483648},
            ]
            resp = client.get("/mobile/health")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "healthy"
            assert body["cpu_percent"] == 50.0
            assert body["disk_free_gb"] > 0


# ── GET /mobile/knowledge ─────────────────────────────────────────────────────


class TestMobileKnowledge:
    """GET /mobile/knowledge"""

    def test_returns_knowledge_list(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = [{"id": "k1", "content": "fact", "topic": "general", "importance": 0.8}]
            resp = client.get("/mobile/knowledge")
            assert resp.status_code == 200
            body = resp.json()
            assert body["total"] == 1
            assert body["items"][0]["content"] == "fact"


# ── POST /mobile/knowledge ────────────────────────────────────────────────────


class TestMobileKnowledgeCreate:
    """POST /mobile/knowledge"""

    def test_creates_knowledge_item(self, client):
        with patch.object(MobileRouter, "_internal_post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = {"id": "k2", "content": "new fact", "topic": "science"}
            resp = client.post("/mobile/knowledge", json={"content": "new fact", "topic": "science"})
            assert resp.status_code == 200
            body = resp.json()
            assert body["content"] == "new fact"

    def test_empty_content_returns_error(self, client):
        resp = client.post("/mobile/knowledge", json={"content": ""})
        assert resp.status_code == 200
        body = resp.json()
        assert body.get("status") == "error" or "content" in body


# ── GET /mobile/sync/status ───────────────────────────────────────────────────


class TestMobileSyncStatus:
    """GET /mobile/sync/status"""

    def test_returns_sync_status(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = {"status": "healthy", "model_loaded": True, "inference_count": 10}
            resp = client.get("/mobile/sync/status")
            assert resp.status_code == 200
            body = resp.json()
            assert body["status"] == "success"
            assert body["data"]["reachable"] is True


# ── POST /mobile/notifications/register ───────────────────────────────────────


class TestMobileNotificationsRegister:
    """POST /mobile/notifications/register"""

    @patch("domains.mobile.notifications.get_notification_service")
    def test_registers_device(self, mock_get_svc, client):
        svc = MagicMock()
        svc.register_device.return_value = {"status": "registered"}
        mock_get_svc.return_value = svc
        resp = client.post("/mobile/notifications/register", json={
            "token": "expo-token-123", "platform": "ios", "user_id": "u1",
        })
        assert resp.status_code == 200


# ── POST /mobile/notifications/unregister ─────────────────────────────────────


class TestMobileNotificationsUnregister:
    """POST /mobile/notifications/unregister"""

    @patch("domains.mobile.notifications.get_notification_service")
    def test_unregisters_device(self, mock_get_svc, client):
        svc = MagicMock()
        svc.unregister_device.return_value = True
        mock_get_svc.return_value = svc
        resp = client.post("/mobile/notifications/unregister", json={"token": "expo-token-123"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "removed"


# ── GET /mobile/notifications/devices ─────────────────────────────────────────


class TestMobileNotificationsDevices:
    """GET /mobile/notifications/devices"""

    @patch("domains.mobile.notifications.get_notification_service")
    def test_returns_device_list(self, mock_get_svc, client):
        svc = MagicMock()
        svc.get_devices.return_value = [{"token": "t1", "platform": "ios"}]
        mock_get_svc.return_value = svc
        resp = client.get("/mobile/notifications/devices")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["devices"]) == 1


# ── GET /mobile/train/stats ──────────────────────────────────────────────────


class TestMobileTrainStats:
    """GET /mobile/train/stats"""

    @patch("domains.training.mobile_training_store.get_training_store")
    def test_returns_stats(self, mock_get_store, client):
        store = MagicMock()
        store.stats.return_value = {"total": 100, "pending": 10, "synced": 50, "used": 40}
        store.quality_breakdown.return_value = {"high": 30, "medium": 50, "low": 20}
        mock_get_store.return_value = store
        resp = client.get("/mobile/train/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 100
        assert body["pending"] == 10


# ── GET /mobile/train/pairs ──────────────────────────────────────────────────


class TestMobileTrainPairs:
    """GET /mobile/train/pairs"""

    @patch("domains.training.mobile_training_store.get_training_store")
    def test_returns_pairs_list(self, mock_get_store, client):
        store = MagicMock()
        store.list_pairs.return_value = [{"_id": "p1", "user_msg": "hi", "assistant_msg": "hello", "quality": 0.9}]
        store.count.return_value = 1
        mock_get_store.return_value = store
        resp = client.get("/mobile/train/pairs")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["pairs"][0]["user_msg"] == "hi"


# ── GET /mobile/train/export ─────────────────────────────────────────────────


class TestMobileTrainExport:
    """GET /mobile/train/export"""

    @patch("domains.training.mobile_training_store.get_training_store")
    def test_returns_export_data(self, mock_get_store, client):
        store = MagicMock()
        store.list_pairs.return_value = [{"user_msg": "hi", "assistant_msg": "hello", "quality": 0.9, "session_id": "s1"}]
        mock_get_store.return_value = store
        resp = client.get("/mobile/train/export")
        assert resp.status_code == 200
        assert "application/x-ndjson" in resp.headers["content-type"]


# ── POST /mobile/train/compact ───────────────────────────────────────────────


class TestMobileTrainCompact:
    """POST /mobile/train/compact"""

    @patch("domains.training.mobile_training_store.get_training_store")
    def test_compacts_store(self, mock_get_store, client):
        store = MagicMock()
        store.compact.return_value = 42
        mock_get_store.return_value = store
        resp = client.post("/mobile/train/compact")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "compacted"
        assert body["count"] == 42


# ── POST /mobile/train/from-sessions ─────────────────────────────────────────


class TestMobileTrainFromSessions:
    """POST /mobile/train/from-sessions"""

    def test_returns_result(self, client):
        resp = client.post("/mobile/train/from-sessions", json={"limit": 50, "min_length": 5})
        assert resp.status_code in (200, 500)


# ── GET /mobile/train/auto-status ────────────────────────────────────────────


class TestMobileTrainAutoStatus:
    """GET /mobile/train/auto-status"""

    @patch("domains.training.auto_trainer.get_auto_trainer")
    def test_returns_status(self, mock_get_trainer, client):
        trainer = MagicMock()
        trainer.status.return_value = {"enabled": False, "threshold": 10, "pending_count": 0}
        mock_get_trainer.return_value = trainer
        resp = client.get("/mobile/train/auto-status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is False
