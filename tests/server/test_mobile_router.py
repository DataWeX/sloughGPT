"""
Tests for the Mobile BFF router — dashboard, conversations, models, knowledge,
notifications, training, sync.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.routers.mobile import MobileRouter
from apps.api.server.infrastructure.exception_handlers import register_all_handlers


@pytest.fixture
def mobile_router():
    return MobileRouter()


@pytest.fixture
def app(mobile_router):
    _app = FastAPI()
    register_all_handlers(_app)
    _app.include_router(mobile_router.router)
    return _app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


def _mock_internal_get(data):
    return AsyncMock(return_value=(data, None))


def _mock_internal_post(data):
    return AsyncMock(return_value=(data, None))


def _mock_internal_patch(data):
    return AsyncMock(return_value=(data, None))


def _mock_internal_delete(data):
    return AsyncMock(return_value=(data, None))


# ── GET /mobile/dashboard ────────────────────────────────────────────────────


class TestMobileDashboard:
    """GET /mobile/dashboard"""

    def test_returns_dashboard_data(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                ({"status": "healthy", "model_type": "gpt2", "model_loaded": True, "inference_count": 5}, None),
                ({"name": "sage", "description": "A wise soul"}, None),
                ({"sessions": [{"id": "s1", "title": "Chat", "messages": [{"content": "hi"}], "updated_at": "2026-01-01"}]}, None),
                ([{"model_id": "gpt2"}], None),
            ]
            resp = client.get("/mobile/dashboard")
            assert resp.status_code == 200
            body = resp.json()["data"]
            assert body["status"] == "healthy"
            assert body["model"]["loaded"] is True
            assert body["soul"]["name"] == "sage"
            assert len(body["recent_conversations"]) == 1


# ── GET /mobile/conversations ─────────────────────────────────────────────────


class TestMobileConversations:
    """GET /mobile/conversations"""

    def test_returns_conversation_list(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = ({
                "sessions": [

                    {"id": "s1", "title": "Chat 1", "messages": [{"content": "hello"}],

                     "updated_at": "2026-01-01", "created_at": "2026-01-01", "starred": False, "pinned": False},

                ]

            }
, None)
            resp = client.get("/mobile/conversations")
            assert resp.status_code == 200
            body = resp.json()["data"]
            assert body["total"] == 1
            assert body["conversations"][0]["id"] == "s1"


# ── GET /mobile/conversations/{session_id} ────────────────────────────────────


class TestMobileConversationDetail:
    """GET /mobile/conversations/{session_id}"""

    def test_returns_conversation(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = ({"messages": [{"role": "user", "content": "hi"}], "created_at": "2026-01-01"}, None)
            resp = client.get("/mobile/conversations/s1")
            assert resp.status_code == 200
            body = resp.json()["data"]
            assert body["id"] == "s1"
            assert len(body["messages"]) == 1


# ── GET /mobile/models ────────────────────────────────────────────────────────


class TestMobileModels:
    """GET /mobile/models"""

    def test_returns_model_list(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                ([{"model_id": "gpt2", "name": "GPT-2", "loaded": True, "source": "local"}], None),
                ([], None),
                ({"name": "sage"}, None),
                ([], None),
                ({"model_type": "gpt2"}, None),
            ]
            resp = client.get("/mobile/models")
            assert resp.status_code == 200
            body = resp.json()["data"]
            assert "models" in body
            assert "souls" in body
            assert "checkpoints" in body


# ── POST /mobile/models/switch ────────────────────────────────────────────────


class TestMobileSwitchModel:
    """POST /mobile/models/switch"""

    def test_switch_with_valid_body(self, client):
        with patch.object(MobileRouter, "_internal_post", new_callable=AsyncMock) as mock_post, \
             patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_post.return_value = ({"status": "ok"}, None)
            mock_get.return_value = ({"model_type": "gpt2"}, None)
            resp = client.post("/mobile/models/switch", json={"soul_name": "sage"})
            assert resp.status_code == 200
            body = resp.json()["data"]
            assert body["soul"] == "sage"

    def test_switch_with_empty_body(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = ({"model_type": "gpt2"}, None)
            resp = client.post("/mobile/models/switch", json={})
            assert resp.status_code == 200


# ── GET /mobile/health ────────────────────────────────────────────────────────


class TestMobileHealth:
    """GET /mobile/health"""

    def test_returns_health_data(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                ({"status": "healthy", "model_type": "gpt2", "model_loaded": True,
                 "uptime_seconds": 100, "system": {"cpu_percent": 50.0, "memory_percent": 60.0,
                                                    "memory_available_mb": 8192}}, None),
                ({"disk_used_bytes": 1073741824, "disk_free_bytes": 2147483648}, None),
            ]
            resp = client.get("/mobile/health")
            assert resp.status_code == 200
            body = resp.json()["data"]
            assert body["status"] == "healthy"
            assert body["cpu_percent"] == 50.0
            assert body["disk_free_gb"] > 0


# ── GET /mobile/knowledge ─────────────────────────────────────────────────────


class TestMobileKnowledge:
    """GET /mobile/knowledge"""

    def test_returns_knowledge_list(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = ([{"id": "k1", "content": "fact", "topic": "general", "importance": 0.8}], None)
            resp = client.get("/mobile/knowledge")
            assert resp.status_code == 200
            body = resp.json()["data"]
            assert body["total"] == 1
            assert body["items"][0]["content"] == "fact"


# ── POST /mobile/knowledge ────────────────────────────────────────────────────


class TestMobileKnowledgeCreate:
    """POST /mobile/knowledge"""

    def test_creates_knowledge_item(self, client):
        with patch.object(MobileRouter, "_internal_post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = ({"id": "k2", "content": "new fact", "topic": "science"}, None)
            resp = client.post("/mobile/knowledge", json={"content": "new fact", "topic": "science"})
            assert resp.status_code == 200
            body = resp.json()["data"]
            assert body["content"] == "new fact"

    def test_empty_content_returns_error(self, client):
        with patch.object(MobileRouter, "_internal_post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = (None, "Failed to create knowledge")
            resp = client.post("/mobile/knowledge", json={"content": ""})
            assert resp.status_code == 400
            body = resp.json()
            assert "error" in body


# ── GET /mobile/sync/status ───────────────────────────────────────────────────


class TestMobileSyncStatus:
    """GET /mobile/sync/status"""

    def test_returns_sync_status(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = ({"status": "healthy", "model_loaded": True, "inference_count": 10}, None)
            resp = client.get("/mobile/sync/status")
            assert resp.status_code == 200
            body = resp.json()["data"]
            assert body["reachable"] is True


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
        body = resp.json()["data"]
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
        body = resp.json()["data"]
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
        body = resp.json()["data"]
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
        body = resp.json()["data"]
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
        body = resp.json()["data"]
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
        body = resp.json()["data"]
        assert body["enabled"] is False


# ── Dashboard edge cases ──────────────────────────────────────────────────────


class TestMobileDashboardEdges:
    """GET /mobile/dashboard — degraded inputs"""

    def test_non_dict_sessions_returns_empty_recent(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                ({"status": "healthy", "model_type": "gpt2", "model_loaded": True, "inference_count": 5}, None),
                ({"name": "sage"}, None),
                ([{"id": "s1"}], None),  # list instead of {"sessions": [...]}
                ([{"model_id": "gpt2"}], None),
            ]
            resp = client.get("/mobile/dashboard")
            assert resp.status_code == 200
            body = resp.json()["data"]
            assert body["recent_conversations"] == []

    def test_missing_health_defaults(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (None, None)
            resp = client.get("/mobile/dashboard")
            assert resp.status_code == 200
            body = resp.json()["data"]
            assert body["status"] == "unknown"
            assert body["model"]["loaded"] is False
            assert body["soul"]["name"] == "Default"

    def test_last_message_uses_newest_session(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = [
                ({"status": "healthy", "model_type": "gpt2", "model_loaded": True, "inference_count": 5}, None),
                ({"name": "sage"}, None),
                ({"sessions": [
                    {"id": "old", "title": "Older", "messages": [{"content": "stale"}], "updated_at": "2026-01-01"},
                    {"id": "new", "title": "Newer", "messages": [{"content": "fresh"}], "updated_at": "2026-06-01"},
                ]}, None),
                ([{"model_id": "gpt2"}], None),
            ]
            body = client.get("/mobile/dashboard").json()["data"]
            assert body["recent_conversations"][0]["id"] == "new"
            assert body["recent_conversations"][0]["last_message"] == "fresh"


# ── Conversations pagination / search ─────────────────────────────────────────


class TestMobileConversationsEdges:
    """GET /mobile/conversations — pagination and search"""

    def _make_session(self, sid, title, content, updated):
        return {"id": sid, "title": title, "messages": [{"content": content}],
                "updated_at": updated, "created_at": updated, "starred": False, "pinned": False}

    def test_paginates(self, client):
        sessions = [self._make_session(f"s{i}", f"Title {i}", "msg", f"2026-01-{i+1:02d}") for i in range(3)]
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = ({"sessions": sessions}, None)
            resp = client.get("/mobile/conversations", params={"page": 2, "per_page": 1})
            body = resp.json()["data"]
            assert body["total"] == 3
            assert len(body["conversations"]) == 1
            assert body["conversations"][0]["id"] == "s1"  # 2nd session (1-indexed page 2)

    def test_search_filters_by_title(self, client):
        sessions = [
            self._make_session("s1", "Meeting Notes", "a", "2026-01-02"),
            self._make_session("s2", "Shopping List", "b", "2026-01-01"),
        ]
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = ({"sessions": sessions}, None)
            resp = client.get("/mobile/conversations", params={"search": "shopping"})
            body = resp.json()["data"]
            assert body["total"] == 1
            assert body["conversations"][0]["id"] == "s2"

    def test_search_matches_message_content(self, client):
        sessions = [self._make_session("s1", "Title", "unique keyword here", "2026-01-02")]
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = ({"sessions": sessions}, None)
            resp = client.get("/mobile/conversations", params={"search": "keyword"})
            assert resp.json()["data"]["total"] == 1

    def test_pagination_validation(self, client):
        resp = client.get("/mobile/conversations", params={"page": 0})
        assert resp.status_code == 422
        resp = client.get("/mobile/conversations", params={"per_page": 101})
        assert resp.status_code == 422


# ── Switch model (with model_id) ──────────────────────────────────────────────


class TestMobileSwitchModelEdges:
    """POST /mobile/models/switch"""

    def test_switch_with_model_id_only(self, client):
        with patch.object(MobileRouter, "_internal_post", new_callable=AsyncMock) as mock_post, \
             patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_post.return_value = ({"status": "ok"}, None)
            mock_get.return_value = ({"model_type": "qwen"}, None)
            resp = client.post("/mobile/models/switch", json={"model_id": "qwen"})
            assert resp.status_code == 200
            body = resp.json()["data"]
            assert body["model"] == "qwen"
            assert body["soul"] == ""

    def test_switch_posts_soul_with_checkpoint(self, client):
        with patch.object(MobileRouter, "_internal_post", new_callable=AsyncMock) as mock_post, \
             patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_post.return_value = ({"status": "ok"}, None)
            mock_get.return_value = ({"model_type": "gpt2"}, None)
            client.post("/mobile/models/switch", json={"soul_name": "sage", "checkpoint_name": "cp1"})
            call = mock_post.call_args_list[-1]
            assert call.args[1] == "/souls/switch"
            assert call.args[2] == {"soul": "sage", "checkpoint_name": "cp1"}


# ── Knowledge update / delete / list edge cases ───────────────────────────────


class TestMobileKnowledgeUpdate:
    """PATCH /mobile/knowledge/{item_id}"""

    def test_updates_item(self, client):
        with patch.object(MobileRouter, "_internal_patch", new_callable=AsyncMock) as mock_patch:
            mock_patch.return_value = ({"id": "k1", "content": "updated", "topic": "new"}, None)
            resp = client.patch("/mobile/knowledge/k1", json={"content": "updated", "topic": "new"})
            assert resp.status_code == 200
            assert resp.json()["data"]["content"] == "updated"
            assert mock_patch.call_args.args[1] == "/knowledge/k1"

    def test_update_failure_returns_error(self, client):
        with patch.object(MobileRouter, "_internal_patch", new_callable=AsyncMock) as mock_patch:
            mock_patch.return_value = (None, "HTTP 500: server error")
            resp = client.patch("/mobile/knowledge/k1", json={"content": "x"})
            assert resp.status_code == 400
            assert "error" in resp.json()

    def test_update_only_sends_provided_fields(self, client):
        with patch.object(MobileRouter, "_internal_patch", new_callable=AsyncMock) as mock_patch:
            mock_patch.return_value = ({}, None)
            client.patch("/mobile/knowledge/k1", json={"topic": "only"})
            body = mock_patch.call_args.args[2]
            assert body == {"topic": "only"}

    def test_update_validation_422(self, client):
        resp = client.patch("/mobile/knowledge/k1", json={"importance": "high"})
        assert resp.status_code == 422


class TestMobileKnowledgeDelete:
    """DELETE /mobile/knowledge/{item_id}"""

    def test_deletes_item(self, client):
        with patch.object(MobileRouter, "_internal_delete", new_callable=AsyncMock) as mock_delete:
            mock_delete.return_value = ({"status": "deleted"}, None)
            resp = client.delete("/mobile/knowledge/k1")
            assert resp.status_code == 200
            body = resp.json()["data"]
            assert body["status"] == "deleted"
            assert body["id"] == "k1"


class TestMobileKnowledgeListEdges:
    """GET /mobile/knowledge — search, topic, pagination"""

    def test_search_uses_search_endpoint(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = ({"results": [{"id": "k1", "content": "match", "topic": "t"}]}, None)
            resp = client.get("/mobile/knowledge", params={"search": "match"})
            assert resp.json()["data"]["total"] == 1
            assert mock_get.call_args.args[1] == "/knowledge/search?query=match"

    def test_topic_filter_applied_after_fetch(self, client):
        items = [
            {"id": "k1", "content": "a", "topic": "science"},
            {"id": "k2", "content": "b", "topic": "sports"},
        ]
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (items, None)
            resp = client.get("/mobile/knowledge", params={"topic": "science"})
            body = resp.json()["data"]
            assert body["total"] == 1
            assert body["items"][0]["id"] == "k1"

    def test_pagination_slices_items(self, client):
        items = [{"id": f"k{i}", "content": f"c{i}", "topic": "t"} for i in range(3)]
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = (items, None)
            resp = client.get("/mobile/knowledge", params={"page": 2, "per_page": 1})
            body = resp.json()["data"]
            assert body["total"] == 3
            assert len(body["items"]) == 1
            assert body["items"][0]["id"] == "k1"

    def test_non_list_items_treated_empty(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = ({"not": "a list"}, None)
            resp = client.get("/mobile/knowledge")
            assert resp.json()["data"]["items"] == []


# ── Sync offline ──────────────────────────────────────────────────────────────


class TestMobileSyncOffline:
    """POST /mobile/sync"""

    def test_syncs_messages_and_returns_counts(self, client):
        with patch.object(MobileRouter, "_internal_post", new_callable=AsyncMock) as mock_post, \
             patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_post.return_value = ({"message": "hi there", "timestamp": 111}, None)
            mock_get.return_value = ({"sessions": [{"id": "s1", "title": "T"}]}, None)
            resp = client.post("/mobile/sync", json={
                "pending_messages": [
                    {"id": "m1", "session_id": "s1", "content": "hello", "timestamp": 100},
                    {"id": "m2", "session_id": "s1", "content": "again", "timestamp": 101},
                ],
            })
            assert resp.status_code == 200
            body = resp.json()["data"]
            assert body["synced_count"] == 2
            assert body["failed_count"] == 0
            assert body["results"][0]["status"] == "sent"
            assert body["results"][0]["assistant_message"]["content"] == "hi there"

    def test_no_response_marks_error(self, client):
        with patch.object(MobileRouter, "_internal_post", new_callable=AsyncMock) as mock_post, \
             patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_post.return_value = (None, "No response")
            mock_get.return_value = ({}, None)
            resp = client.post("/mobile/sync", json={
                "pending_messages": [
                    {"id": "m1", "session_id": "s1", "content": "hello", "timestamp": 100},
                ],
            })
            body = resp.json()["data"]
            assert body["failed_count"] == 1
            assert body["results"][0]["status"] == "error"

    def test_empty_pending_messages(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = ({"sessions": []}, None)
            resp = client.post("/mobile/sync", json={"pending_messages": []})
            body = resp.json()["data"]
            assert body["synced_count"] == 0
            assert body["results"] == []

    def test_sessions_wrapped_in_data_field(self, client):
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = ({"data": [{"id": "s9"}]}, None)
            resp = client.post("/mobile/sync", json={"pending_messages": []})
            sessions = resp.json()["data"]["sessions"]
            assert sessions[0]["id"] == "s9"


# ── Notifications: send / history / cleanup ───────────────────────────────────


class TestMobileNotificationsSend:
    """POST /mobile/notifications/send"""

    @patch("domains.mobile.notifications.get_notification_service")
    def test_sends_notification(self, mock_get_svc, client):
        svc = MagicMock()
        svc.send_notification_async = AsyncMock(return_value={"sent": 3})
        mock_get_svc.return_value = svc
        resp = client.post("/mobile/notifications/send", json={"title": "Hi", "body": "World"})
        assert resp.status_code == 200
        assert resp.json()["data"]["sent"] == 3
        assert svc.send_notification_async.call_args.kwargs["topic"] is None

    @patch("domains.mobile.notifications.get_notification_service")
    def test_sends_with_topic_and_tokens(self, mock_get_svc, client):
        svc = MagicMock()
        svc.send_notification_async = AsyncMock(return_value={"sent": 1})
        mock_get_svc.return_value = svc
        resp = client.post("/mobile/notifications/send",
                           json={"title": "Hi", "body": "World", "topic": "news", "badge": 5})
        assert resp.status_code == 200
        payload = svc.send_notification_async.call_args.kwargs["payload"]
        assert payload.badge == 5
        assert payload.topic == "news"

    def test_missing_title_returns_422(self, client):
        resp = client.post("/mobile/notifications/send", json={"body": "World"})
        assert resp.status_code == 422


class TestMobileNotificationsHistory:
    """GET /mobile/notifications/history"""

    @patch("domains.mobile.notifications.get_notification_service")
    def test_returns_history(self, mock_get_svc, client):
        svc = MagicMock()
        svc.get_history.return_value = [{"title": "Hi", "sent_at": 1}]
        mock_get_svc.return_value = svc
        resp = client.get("/mobile/notifications/history")
        assert resp.status_code == 200
        assert len(resp.json()["data"]["history"]) == 1

    @patch("domains.mobile.notifications.get_notification_service")
    def test_history_limit_validation(self, mock_get_svc, client):
        resp = client.get("/mobile/notifications/history", params={"limit": 500})
        assert resp.status_code == 422


class TestMobileNotificationsCleanup:
    """POST /mobile/notifications/cleanup"""

    @patch("domains.mobile.notifications.get_notification_service")
    def test_cleans_stale_devices(self, mock_get_svc, client):
        svc = MagicMock()
        svc.cleanup_stale.return_value = 4
        mock_get_svc.return_value = svc
        resp = client.post("/mobile/notifications/cleanup")
        assert resp.status_code == 200
        assert resp.json()["data"]["removed"] == 4


class TestMobileNotifyTrainingComplete:
    """POST /mobile/notify/training-complete"""

    @patch("domains.mobile.notifications.get_notification_service")
    def test_notifies_with_loss(self, mock_get_svc, client):
        svc = MagicMock()
        svc.send_notification.return_value = {"sent": 1}
        mock_get_svc.return_value = svc
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = ({"status": "complete", "final_loss": 1.5}, None)
            resp = client.post("/mobile/notify/training-complete")
        assert resp.status_code == 200
        payload = svc.send_notification.call_args.kwargs["payload"]
        assert payload.topic == "training"
        assert "1.5000" in payload.body

    @patch("domains.mobile.notifications.get_notification_service")
    def test_notifies_without_loss(self, mock_get_svc, client):
        svc = MagicMock()
        svc.send_notification.return_value = {"sent": 0}
        mock_get_svc.return_value = svc
        with patch.object(MobileRouter, "_internal_get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = ({"status": "running"}, None)
            resp = client.post("/mobile/notify/training-complete")
        assert resp.status_code == 200
        payload = svc.send_notification.call_args.kwargs["payload"]
        assert "running" in payload.body


# ── Training: pending / session pairs / quality / delete / bulk ───────────────


class TestMobileTrainPending:
    """GET /mobile/train/pending"""

    @patch("domains.training.mobile_training_store.get_training_store")
    def test_returns_pending(self, mock_get_store, client):
        store = MagicMock()
        store.get_pending_pairs.return_value = [{"_id": "p1", "user_msg": "hi", "assistant_msg": "lo"}]
        mock_get_store.return_value = store
        resp = client.get("/mobile/train/pending")
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["count"] == 1
        assert body["pairs"][0]["id"] == "p1"

    @patch("domains.training.mobile_training_store.get_training_store")
    def test_limit_validation(self, mock_get_store, client):
        resp = client.get("/mobile/train/pending", params={"limit": 1000})
        assert resp.status_code == 422


class TestMobileSessionPairs:
    """GET /mobile/train/session/{session_id}"""

    @patch("domains.training.mobile_training_store.get_training_store")
    def test_returns_session_pairs(self, mock_get_store, client):
        store = MagicMock()
        store.get_pairs_by_session.return_value = [
            {"_id": "p1", "user_msg": "hi", "assistant_msg": "lo", "quality": 0.8, "timestamp": 1}
        ]
        mock_get_store.return_value = store
        resp = client.get("/mobile/train/session/s1")
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["session_id"] == "s1"
        assert body["count"] == 1
        assert body["pairs"][0]["quality"] == 0.8


class TestMobileUpdatePairQuality:
    """PATCH /mobile/train/pair/{pair_id}"""

    @patch("domains.training.mobile_training_store.get_training_store")
    def test_updates_quality(self, mock_get_store, client):
        store = MagicMock()
        store.update_quality.return_value = True
        mock_get_store.return_value = store
        resp = client.patch("/mobile/train/pair/p1", json={"quality": 0.9})
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["status"] == "updated"
        assert body["quality"] == 0.9

    @patch("domains.training.mobile_training_store.get_training_store")
    def test_missing_pair_returns_404(self, mock_get_store, client):
        store = MagicMock()
        store.update_quality.return_value = False
        mock_get_store.return_value = store
        resp = client.patch("/mobile/train/pair/p1", json={"quality": 0.9})
        assert resp.status_code == 404

    @patch("domains.training.mobile_training_store.get_training_store")
    def test_quality_validation(self, mock_get_store, client):
        resp = client.patch("/mobile/train/pair/p1", json={"quality": "high"})
        assert resp.status_code == 422


class TestMobileDeletePair:
    """DELETE /mobile/train/pair/{pair_id}"""

    @patch("domains.training.mobile_training_store.get_training_store")
    def test_deletes_pair(self, mock_get_store, client):
        store = MagicMock()
        store.delete_pair.return_value = True
        mock_get_store.return_value = store
        resp = client.delete("/mobile/train/pair/p1")
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "deleted"

    @patch("domains.training.mobile_training_store.get_training_store")
    def test_missing_pair_returns_404(self, mock_get_store, client):
        store = MagicMock()
        store.delete_pair.return_value = False
        mock_get_store.return_value = store
        resp = client.delete("/mobile/train/pair/p1")
        assert resp.status_code == 404


class TestMobileDeleteSynced:
    """DELETE /mobile/train/synced"""

    @patch("domains.training.mobile_training_store.get_training_store")
    def test_deletes_synced(self, mock_get_store, client):
        store = MagicMock()
        store.delete_synced.return_value = 7
        mock_get_store.return_value = store
        resp = client.delete("/mobile/train/synced")
        assert resp.status_code == 200
        body = resp.json()["data"]
        assert body["status"] == "deleted"
        assert body["count"] == 7


class TestMobileDeletePairsBulk:
    """DELETE /mobile/train/pairs/bulk"""

    @patch("domains.training.mobile_training_store.get_training_store")
    def test_deletes_bulk(self, mock_get_store, client):
        store = MagicMock()
        store.delete_pair.side_effect = [True, True, False]
        mock_get_store.return_value = store
        resp = client.delete("/mobile/train/pairs/bulk", params={"ids": ["p1", "p2", "p3"]})
        assert resp.status_code == 200
        assert resp.json()["data"]["count"] == 2

    @patch("domains.training.mobile_training_store.get_training_store")
    def test_requires_ids(self, mock_get_store, client):
        resp = client.delete("/mobile/train/pairs/bulk")
        assert resp.status_code == 422


# ── Training via mobile (subprocess gating) ───────────────────────────────────


class TestMobileTrain:
    """POST /mobile/train"""

    def _pairs(self, n):
        return [{"id": f"p{i}", "user_msg": "hi", "assistant_msg": "hello"} for i in range(n)]

    def test_fewer_than_five_pairs_returns_400(self, client):
        resp = client.post("/mobile/train", json={"pairs": self._pairs(3), "checkpoint": "base"})
        assert resp.status_code == 400

    def test_missing_checkpoint_returns_422(self, client):
        resp = client.post("/mobile/train", json={"pairs": self._pairs(5)})
        assert resp.status_code == 422


# ── Auto-train config update ──────────────────────────────────────────────────


class TestMobileAutoTrainConfig:
    """PATCH /mobile/train/auto-config"""

    @patch("domains.training.auto_trainer.get_auto_trainer")
    def test_updates_threshold(self, mock_get_trainer, client):
        trainer = MagicMock()
        trainer.status.return_value = {"enabled": True, "threshold": 5, "interval_s": 60}
        mock_get_trainer.return_value = trainer
        resp = client.patch("/mobile/train/auto-config", params={"threshold": 5})
        assert resp.status_code == 200
        assert trainer.threshold == 5
        assert resp.json()["data"]["threshold"] == 5

    @patch("domains.training.auto_trainer.get_auto_trainer")
    def test_updates_interval_only(self, mock_get_trainer, client):
        trainer = MagicMock()
        trainer.status.return_value = {"enabled": True, "threshold": 10, "interval_s": 120}
        mock_get_trainer.return_value = trainer
        client.patch("/mobile/train/auto-config", params={"interval_s": 120})
        assert trainer.interval_s == 120

    @patch("domains.training.auto_trainer.get_auto_trainer")
    def test_threshold_bounds_422(self, mock_get_trainer, client):
        assert client.patch("/mobile/train/auto-config", params={"threshold": 0}).status_code == 422
        assert client.patch("/mobile/train/auto-config", params={"threshold": 101}).status_code == 422
        assert client.patch("/mobile/train/auto-config", params={"interval_s": 10}).status_code == 422
