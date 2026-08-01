"""Tests for domains/mobile/notifications.py."""

import json

import sys

import pytest

from domains.mobile.notifications import (
    DeviceToken,
    NotificationPayload,
    PushNotificationService,
)


@pytest.fixture
def service(monkeypatch, tmp_path):
    devices = tmp_path / "devices.json"
    history = tmp_path / "history.json"
    monkeypatch.setattr(
        "domains.mobile.notifications._DEVICES_FILE", devices
    )
    monkeypatch.setattr(
        "domains.mobile.notifications._HISTORY_FILE", history
    )
    svc = PushNotificationService()
    return svc


class TestDeviceToken:
    def test_defaults(self):
        token = DeviceToken(token="tok123", platform="ios")
        assert token.user_id == "default"
        assert token.topics == ["chat", "training"]
        assert token.enabled is True
        assert token.created_at > 0
        assert token.last_active > 0

    def test_custom_topics(self):
        token = DeviceToken(
            token="tok", platform="web", topics=["alerts"]
        )
        assert token.topics == ["alerts"]


class TestNotificationPayload:
    def test_defaults(self):
        payload = NotificationPayload(title="t", body="b")
        assert payload.data == {}
        assert payload.sound == "default"
        assert payload.badge is None
        assert payload.topic is None


class TestRegisterDevice:
    def test_register_new(self, service):
        result = service.register_device("abc123", "ios", user_id="u1")
        assert result["status"] == "registered"
        assert "abc123" in service._devices

    def test_register_existing_updates(self, service):
        service.register_device("abc123", "ios", user_id="u1")
        result = service.register_device(
            "abc123", "android", topics=["news"]
        )
        assert result["status"] == "updated"
        device = service._devices["abc123"]
        assert device.platform == "android"
        assert device.topics == ["news"]

    def test_register_persists_to_disk(self, service, tmp_path):
        service.register_device("persist1", "web")
        data = json.loads(
            (tmp_path / "devices.json").read_text()
        )
        assert "persist1" in data

    def test_unregister(self, service):
        service.register_device("abc", "ios")
        assert service.unregister_device("abc") is True
        assert service.unregister_device("abc") is False


class TestGetDevices:
    def test_empty(self, service):
        assert service.get_devices() == []

    def test_filters_disabled(self, service):
        service.register_device("tok1", "ios", topics=["chat"])
        service._devices["tok1"].enabled = False
        assert service.get_devices() == []

    def test_filters_by_topic(self, service):
        service.register_device("tok1", "ios", topics=["chat"])
        service.register_device("tok2", "ios", topics=["news"])
        by_chat = service.get_devices(topic="chat")
        assert len(by_chat) == 1
        assert by_chat[0]["token"].startswith("tok1")
        assert "last_active" in by_chat[0]
        assert "platform" in by_chat[0]
        assert "topics" in by_chat[0]


class TestSendNotification:
    def test_no_recipients(self, service):
        result = service.send_notification(
            NotificationPayload(title="t", body="b")
        )
        assert result == {"status": "no_recipients", "sent": 0}

    def test_sends_to_topic(self, service, monkeypatch):
        service.register_device("tok1", "ios", topics=["chat"])
        service.register_device("tok2", "ios", topics=["news"])

        captured = {}

        class FakeResponse:
            status_code = 200

            def json(self):
                return [{"status": "ok"}]

        class FakeClient:
            def __init__(self, timeout=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, url, json, headers=None):
                captured["url"] = url
                captured["json"] = json
                return FakeResponse()

        fake_httpx = type("FakeHttpx", (), {"Client": FakeClient})
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
        result = service.send_notification(
            NotificationPayload(title="hi", body="yo", topic="chat"),
            topic="chat",
        )
        assert result["status"] == "sent"
        assert result["sent"] == 1
        assert result["total"] == 1
        assert captured["json"][0]["title"] == "hi"
        assert captured["json"][0]["to"] == "tok1"

    def test_logs_when_httpx_missing(self, service, monkeypatch):
        service.register_device("tok1", "ios")
        monkeypatch.delitem("sys.modules", "httpx", raising=False)

        def fake_import(name, *args, **kwargs):
            if name == "httpx":
                raise ImportError("no httpx")
            return __import__(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", fake_import)
        result = service.send_notification(
            NotificationPayload(title="t", body="b")
        )
        assert result["status"] == "sent"
        assert result["sent"] == 1

    def test_errors_recorded(self, service, monkeypatch):
        service.register_device("tok1", "ios")

        class FakeResponse:
            status_code = 500

            def json(self):
                return []

        class FakeClient:
            def __init__(self, timeout=None):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, url, json, headers=None):
                return FakeResponse()

        fake_httpx = type("FakeHttpx", (), {"Client": FakeClient})
        monkeypatch.setitem(sys.modules, "httpx", fake_httpx)
        result = service.send_notification(
            NotificationPayload(title="t", body="b")
        )
        assert result["status"] == "failed"
        assert result["sent"] == 0
        assert "HTTP 500" in result["errors"]

    def test_history_recorded(self, service):
        service.register_device("tok1", "ios")
        service.send_notification(
            NotificationPayload(title="t", body="b")
        )
        history = service.get_history()
        assert len(history) == 1
        assert history[0]["title"] == "t"
        assert history[0]["body"] == "b"


class TestPersistenceAndCleanup:
    def test_history_loaded_from_disk(self, service, tmp_path):
        service.register_device("tok1", "ios")
        service._history.append({"title": "old", "body": "x"})
        service._save_history()
        svc2 = PushNotificationService()
        assert len(svc2.get_history()) == 1
        assert svc2.get_history()[0]["title"] == "old"

    def test_cleanup_stale(self, service):
        service.register_device("tok1", "ios")
        service.register_device("tok2", "ios")
        service._devices["tok1"].last_active = 0.0
        removed = service.cleanup_stale(max_age_seconds=3600)
        assert removed == 1
        assert "tok1" not in service._devices
        assert "tok2" in service._devices

    def test_cleanup_no_stale(self, service):
        service.register_device("tok1", "ios")
        assert service.cleanup_stale() == 0


class TestSingleton:
    def test_singleton(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "domains.mobile.notifications._NOTIFICATIONS_DIR", tmp_path
        )
        monkeypatch.setattr(
            "domains.mobile.notifications._DEVICES_FILE", tmp_path / "devices.json"
        )
        monkeypatch.setattr(
            "domains.mobile.notifications._HISTORY_FILE", tmp_path / "history.json"
        )
        from domains.mobile.notifications import get_notification_service

        monkeypatch.setattr("domains.mobile.notifications._service", None)
        svc1 = get_notification_service()
        svc2 = get_notification_service()
        assert svc1 is svc2
