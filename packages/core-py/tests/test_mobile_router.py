"""Tests for the mobile API router (routers/mobile.py).

Covers: training stats, notification history, sync status, device management,
compact, auto-train status, and model switching.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.mobile import MobileRouter


def _app(mr: MobileRouter) -> FastAPI:
    app = FastAPI()
    app.include_router(mr.router)
    return app


# ── Training stats ──


class TestTrainingStats:
    @patch("domains.training.mobile_training_store.get_training_store")
    def test_get_training_stats(self, mock_get_store):
        store = MagicMock()
        store.stats.return_value = {"total": 100, "pending": 5, "synced": 90, "used": 80}
        store.quality_breakdown.return_value = {"good": 70, "bad": 10}
        mock_get_store.return_value = store

        mr = MobileRouter()
        client = TestClient(_app(mr))
        resp = client.get("/mobile/train/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 100
        assert body["pending"] == 5
        assert body["by_quality"]["good"] == 70

    @patch("domains.training.mobile_training_store.get_training_store")
    def test_get_training_stats_empty(self, mock_get_store):
        store = MagicMock()
        store.stats.return_value = {"total": 0, "pending": 0, "synced": 0, "used": 0}
        store.quality_breakdown.return_value = {}
        mock_get_store.return_value = store

        mr = MobileRouter()
        client = TestClient(_app(mr))
        resp = client.get("/mobile/train/stats")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0


# ── Notification history ──


class TestNotificationHistory:
    @patch("domains.mobile.notifications.get_notification_service")
    def test_notification_history(self, mock_get_svc):
        svc = MagicMock()
        svc.get_history.return_value = [
            {"title": "Test", "body": "Hello", "sent_at": 1000},
            {"title": "Test 2", "body": "World", "sent_at": 2000},
        ]
        mock_get_svc.return_value = svc

        mr = MobileRouter()
        client = TestClient(_app(mr))
        resp = client.get("/mobile/notifications/history")
        assert resp.status_code == 200
        body = resp.json()
        # Returns {"history": [...]} dict, not a plain list
        assert "history" in body
        assert len(body["history"]) == 2

    @patch("domains.mobile.notifications.get_notification_service")
    def test_notification_history_with_limit(self, mock_get_svc):
        svc = MagicMock()
        svc.get_history.return_value = [{"title": "Only", "body": "One", "sent_at": 1000}]
        mock_get_svc.return_value = svc

        mr = MobileRouter()
        client = TestClient(_app(mr))
        resp = client.get("/mobile/notifications/history?limit=1")
        assert resp.status_code == 200
        assert len(resp.json()["history"]) == 1


# ── Device list ──


class TestDeviceManagement:
    @patch("domains.mobile.notifications.get_notification_service")
    def test_list_devices(self, mock_get_svc):
        svc = MagicMock()
        svc.get_devices.return_value = [
            {"token": "abc", "platform": "ios", "user_id": "u1"},
            {"token": "def", "platform": "android", "user_id": "u1"},
        ]
        mock_get_svc.return_value = svc

        mr = MobileRouter()
        client = TestClient(_app(mr))
        resp = client.get("/mobile/notifications/devices")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["devices"]) == 2

    @patch("domains.mobile.notifications.get_notification_service")
    def test_list_devices_with_topic_filter(self, mock_get_svc):
        svc = MagicMock()
        svc.get_devices.return_value = [{"token": "abc", "platform": "ios"}]
        mock_get_svc.return_value = svc

        mr = MobileRouter()
        client = TestClient(_app(mr))
        resp = client.get("/mobile/notifications/devices?topic=training")
        assert resp.status_code == 200
        svc.get_devices.assert_called_once_with(topic="training")

    @patch("domains.mobile.notifications.get_notification_service")
    def test_register_device(self, mock_get_svc):
        svc = MagicMock()
        svc.register_device.return_value = {"status": "registered"}
        mock_get_svc.return_value = svc

        mr = MobileRouter()
        client = TestClient(_app(mr))
        resp = client.post("/mobile/notifications/register", json={
            "token": "device_token_123",
            "platform": "ios",
            "user_id": "user1",
            "topics": ["training"],
        })
        assert resp.status_code == 200
        svc.register_device.assert_called_once()

    @patch("domains.mobile.notifications.get_notification_service")
    def test_unregister_device(self, mock_get_svc):
        svc = MagicMock()
        svc.unregister_device.return_value = {"status": "unregistered"}
        mock_get_svc.return_value = svc

        mr = MobileRouter()
        client = TestClient(_app(mr))
        resp = client.post("/mobile/notifications/unregister", json={"token": "old_token"})
        assert resp.status_code == 200
        svc.unregister_device.assert_called_once_with("old_token")


# ── Compact ──


class TestCompact:
    @patch("domains.training.mobile_training_store.get_training_store")
    def test_compact_training_store(self, mock_get_store):
        store = MagicMock()
        store.compact.return_value = 42
        mock_get_store.return_value = store

        mr = MobileRouter()
        client = TestClient(_app(mr))
        resp = client.post("/mobile/train/compact")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "compacted"
        assert body["count"] == 42


# ── Auto-train status ──


class TestAutoTrainStatus:
    @patch("domains.training.auto_trainer.get_auto_trainer")
    def test_get_auto_train_status(self, mock_get_trainer):
        trainer = MagicMock()
        trainer.status.return_value = {
            "enabled": True,
            "threshold": 10,
            "pending_count": 3,
            "last_train": None,
        }
        mock_get_trainer.return_value = trainer

        mr = MobileRouter()
        client = TestClient(_app(mr))
        resp = client.get("/mobile/train/auto-status")
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is True
        assert body["threshold"] == 10

    @patch("domains.training.auto_trainer.get_auto_trainer")
    def test_update_auto_train_config(self, mock_get_trainer):
        trainer = MagicMock()
        trainer.status.return_value = {"enabled": True, "threshold": 20, "interval_s": 120}
        trainer.set_config = MagicMock()
        mock_get_trainer.return_value = trainer

        mr = MobileRouter()
        client = TestClient(_app(mr))
        resp = client.patch("/mobile/train/auto-config?threshold=20&interval_s=120")
        assert resp.status_code == 200


# ── Send notification ──


class TestSendNotification:
    @patch("domains.mobile.notifications.get_notification_service")
    def test_send_notification(self, mock_get_svc):
        svc = MagicMock()
        svc.send_notification.return_value = {"sent": 5, "failed": 0}
        mock_get_svc.return_value = svc

        mr = MobileRouter()
        client = TestClient(_app(mr))
        resp = client.post("/mobile/notifications/send", json={
            "title": "Training Complete",
            "body": "Your model finished training",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["sent"] == 5


# ── Cleanup devices ──


class TestCleanupDevices:
    @patch("domains.mobile.notifications.get_notification_service")
    def test_cleanup_devices(self, mock_get_svc):
        svc = MagicMock()
        svc.cleanup_stale.return_value = 3
        mock_get_svc.return_value = svc

        mr = MobileRouter()
        client = TestClient(_app(mr))
        resp = client.post("/mobile/notifications/cleanup")
        assert resp.status_code == 200
        body = resp.json()
        assert body["removed"] == 3


# ── Pending pairs ──


class TestPendingPairs:
    @patch("domains.training.mobile_training_store.get_training_store")
    def test_get_pending_pairs(self, mock_get_store):
        store = MagicMock()
        store.get_pending_pairs.return_value = [
            {"id": "p1", "user_msg": "Hi", "assistant_msg": "Hello"},
            {"id": "p2", "user_msg": "Bye", "assistant_msg": "Goodbye"},
        ]
        mock_get_store.return_value = store

        mr = MobileRouter()
        client = TestClient(_app(mr))
        resp = client.get("/mobile/train/pending")
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["pairs"]) == 2

    @patch("domains.training.mobile_training_store.get_training_store")
    def test_get_pending_pairs_with_limit(self, mock_get_store):
        store = MagicMock()
        store.get_pending_pairs.return_value = [{"id": "p1"}]
        mock_get_store.return_value = store

        mr = MobileRouter()
        client = TestClient(_app(mr))
        resp = client.get("/mobile/train/pending?limit=1")
        assert resp.status_code == 200
        store.get_pending_pairs.assert_called_once_with(limit=1)


# ── Export training pairs ──


class TestExportTrainingPairs:
    @patch("domains.training.mobile_training_store.get_training_store")
    def test_export_training_pairs(self, mock_get_store):
        store = MagicMock()
        store.list_pairs.return_value = [
            {"id": "p1", "user_msg": "Hi", "assistant_msg": "Hello"},
        ]
        mock_get_store.return_value = store

        mr = MobileRouter()
        client = TestClient(_app(mr))
        resp = client.get("/mobile/train/export")
        assert resp.status_code == 200
        # Returns StreamingResponse with JSONL content
        content = resp.text
        assert "Hi" in content or "user_msg" in content
