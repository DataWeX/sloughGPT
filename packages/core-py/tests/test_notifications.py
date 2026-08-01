"""Tests for domains.mobile.notifications: device tokens and Expo push service."""

import sys
import time

import pytest

from domains.mobile.notifications import (
    DeviceToken,
    NotificationPayload,
    PushNotificationService,
    get_notification_service,
)


@pytest.fixture
def service(tmp_path, monkeypatch):
    monkeypatch.setattr("domains.mobile.notifications._DEVICES_FILE", tmp_path / "devices.json")
    monkeypatch.setattr("domains.mobile.notifications._HISTORY_FILE", tmp_path / "history.json")
    return PushNotificationService()


class TestDataclasses:
    def test_device_token_defaults(self):
        d = DeviceToken(token="tok", platform="ios")
        assert d.user_id == "default"
        assert d.topics == ["chat", "training"]
        assert d.enabled is True
        assert isinstance(d.created_at, float)
        assert isinstance(d.last_active, float)

    def test_notification_payload_defaults(self):
        p = NotificationPayload(title="t", body="b")
        assert p.data == {}
        assert p.sound == "default"
        assert p.badge is None
        assert p.topic is None


class TestRegisterUnregister:
    def test_register_new_device(self, service):
        result = service.register_device("tok-123", "ios")
        assert result["status"] == "registered"
        assert result["token"] == "tok-123..."

    def test_register_updates_existing(self, service):
        service.register_device("tok-123", "ios")
        result = service.register_device("tok-123", "android", topics=["training"])
        assert result["status"] == "updated"
        device = service._devices["tok-123"]
        assert device.platform == "android"
        assert device.topics == ["training"]

    def test_register_persists_to_disk(self, service, tmp_path):
        service.register_device("tok-123", "ios", user_id="alice")
        assert (tmp_path / "devices.json").exists()
        content = (tmp_path / "devices.json").read_text()
        assert '"tok-123"' in content
        assert '"alice"' in content

    def test_unregister_existing(self, service):
        service.register_device("tok-123", "ios")
        assert service.unregister_device("tok-123") is True
        assert "tok-123" not in service._devices

    def test_unregister_missing(self, service):
        assert service.unregister_device("nope") is False


class TestGetDevices:
    def test_empty(self, service):
        assert service.get_devices() == []

    def test_lists_registered(self, service):
        service.register_device("tok-123", "ios")
        devices = service.get_devices()
        assert len(devices) == 1
        assert devices[0]["token"] == "tok-123..."
        assert devices[0]["platform"] == "ios"

    def test_filters_by_topic(self, service):
        service.register_device("tok-a", "ios", topics=["chat"])
        service.register_device("tok-b", "android", topics=["training"])
        assert len(service.get_devices(topic="chat")) == 1
        assert len(service.get_devices(topic="training")) == 1
        assert len(service.get_devices(topic="news")) == 0

    def test_skips_disabled(self, service):
        service.register_device("tok-a", "ios")
        service._devices["tok-a"].enabled = False
        assert service.get_devices() == []


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, responses, timeout=None):
        self.responses = list(responses)
        self.posts = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None, headers=None):
        self.posts.append((url, json, headers))
        return self.responses.pop(0)


class FakeHttpx:
    def __init__(self, responses):
        self._client = FakeClient(responses)

    def Client(self, *a, **k):
        return self._client


@pytest.fixture
def fake_httpx(monkeypatch):
    import domains.mobile.notifications as mod

    store = {}

    def make(responses):
        store["client"] = FakeHttpx(responses)
        monkeypatch.setitem(sys.modules, "httpx", store["client"])

    make._store = store
    return make


class TestSendNotification:
    def test_no_recipients(self, service):
        result = service.send_notification(NotificationPayload(title="t", body="b"))
        assert result["status"] == "no_recipients"
        assert result["sent"] == 0

    def test_unknown_token_skipped(self, service):
        service.register_device("tok-a", "ios")
        result = service.send_notification(
            NotificationPayload(title="t", body="b"), tokens=["tok-b"]
        )
        assert result["status"] == "no_recipients"

    def test_httpx_unavailable_logs_and_counts(self, service, monkeypatch):
        service.register_device("tok-a", "ios")
        real_import = __import__

        def blocked(name, *args, **kwargs):
            if name == "httpx":
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", blocked)
        result = service.send_notification(NotificationPayload(title="t", body="b"))
        assert result["status"] == "sent"
        assert result["sent"] == 1
        assert result["total"] == 1

    def test_sends_via_expo_and_counts_ok(self, service, fake_httpx):
        service.register_device("tok-a", "ios")
        service.register_device("tok-b", "android")
        fake_httpx([FakeResponse(200, [{"status": "ok"}, {"status": "ok"}])])
        result = service.send_notification(NotificationPayload(title="t", body="b"))
        assert result["status"] == "sent"
        assert result["sent"] == 2
        assert result["total"] == 2
        client = fake_httpx._store["client"].Client().posts
        assert len(client) == 1
        url, body, headers = client[0]
        assert url == "https://exp.host/--/api/v2/push/send"
        assert body[0]["to"] == "tok-a"
        assert body[0]["title"] == "t"
        assert body[0]["sound"] == "default"
        assert "channelId" not in body[0]

    def test_topic_and_channel_id_in_message(self, service, fake_httpx):
        service.register_device("tok-a", "ios", topics=["news"])
        fake_httpx([FakeResponse(200, [{"status": "ok"}])])
        result = service.send_notification(
            NotificationPayload(title="t", body="b", topic="news", badge=3),
            topic="news",
        )
        assert result["sent"] == 1
        client = fake_httpx._store["client"].Client().posts
        assert client[0][1][0]["badge"] == 3
        assert client[0][1][0]["channelId"] == "news"

    def test_expo_error_status_recorded(self, service, fake_httpx):
        service.register_device("tok-a", "ios")
        fake_httpx([FakeResponse(200, [{"status": "error", "message": "bad token"}])])
        result = service.send_notification(NotificationPayload(title="t", body="b"))
        assert result["status"] == "failed"
        assert result["sent"] == 0
        assert result["errors"] == ["bad token"]

    def test_http_error_recorded(self, service, fake_httpx):
        service.register_device("tok-a", "ios")
        fake_httpx([FakeResponse(500, {})])
        result = service.send_notification(NotificationPayload(title="t", body="b"))
        assert result["errors"] == ["HTTP 500"]

    def test_batches_of_100(self, service, fake_httpx):
        for i in range(150):
            service.register_device(f"tok-{i:03d}", "ios")
        fake_httpx([FakeResponse(200, [{"status": "ok"}] * 100), FakeResponse(200, [{"status": "ok"}] * 50)])
        result = service.send_notification(NotificationPayload(title="t", body="b"))
        assert result["sent"] == 150
        assert len(fake_httpx._store["client"].Client().posts) == 2

    def test_exception_in_send_returns_failed(self, service, monkeypatch):
        service.register_device("tok-a", "ios")
        real_import = __import__

        class Boom(RuntimeError):
            pass

        def blocked(name, *a, **k):
            if name == "httpx":
                raise Boom("no network")
            return real_import(name, *a, **k)

        monkeypatch.setattr("builtins.__import__", blocked)
        result = service.send_notification(NotificationPayload(title="t", body="b"))
        assert result["status"] == "failed"
        assert result["errors"]

    def test_history_recorded(self, service):
        service.register_device("tok-a", "ios")
        service.send_notification(NotificationPayload(title="hello", body="world", topic="chat"))
        history = service.get_history()
        assert len(history) == 1
        assert history[0]["title"] == "hello"
        assert history[0]["topic"] == "chat"


class TestHistoryAndCleanup:
    def test_history_limit(self, service):
        for i in range(5):
            service._history.append({"title": str(i)})
        assert len(service.get_history(limit=2)) == 2
        assert service.get_history(limit=2)[-1]["title"] == "4"

    def test_cleanup_stale(self, service):
        service.register_device("tok-a", "ios")
        service.register_device("tok-b", "ios")
        service._devices["tok-a"].last_active = time.time() - 100
        removed = service.cleanup_stale(max_age_seconds=50)
        assert removed == 1
        assert "tok-a" not in service._devices
        assert "tok-b" in service._devices

    def test_cleanup_none_stale(self, service):
        service.register_device("tok-a", "ios")
        assert service.cleanup_stale(max_age_seconds=0) == 1


class TestPersistence:
    def test_load_roundtrip(self, service, tmp_path, monkeypatch):
        monkeypatch.setattr("domains.mobile.notifications._DEVICES_FILE", tmp_path / "devices.json")
        monkeypatch.setattr("domains.mobile.notifications._HISTORY_FILE", tmp_path / "history.json")
        service.register_device("tok-123", "ios", user_id="alice", topics=["chat"])
        service.send_notification(NotificationPayload(title="t", body="b"))
        reloaded = PushNotificationService()
        assert "tok-123" in reloaded._devices
        assert reloaded._devices["tok-123"].user_id == "alice"
        assert len(reloaded._history) == 1

    def test_load_corrupt_file_ignored(self, tmp_path, monkeypatch):
        (tmp_path / "devices.json").write_text("{not json")
        monkeypatch.setattr("domains.mobile.notifications._DEVICES_FILE", tmp_path / "devices.json")
        monkeypatch.setattr("domains.mobile.notifications._HISTORY_FILE", tmp_path / "history.json")
        service = PushNotificationService()
        assert service._devices == {}


class TestSingleton:
    def test_get_service_singleton(self, monkeypatch):
        monkeypatch.setattr("domains.mobile.notifications._service", None)
        s1 = get_notification_service()
        s2 = get_notification_service()
        assert s1 is s2
        monkeypatch.setattr("domains.mobile.notifications._service", None)
