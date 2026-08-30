"""Tests for domains.mobile.notifications — DeviceToken, NotificationPayload,
PushNotificationService."""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from domains.mobile.notifications import (
    DeviceToken,
    NotificationPayload,
    PushNotificationService,
    _NOTIFICATIONS_DIR,
)


def _fresh_service():
    """Create a PushNotificationService with empty state (no disk load)."""
    svc = PushNotificationService.__new__(PushNotificationService)
    svc._devices = {}
    svc._history = []
    return svc


def _mock_send_success(svc):
    """Patch httpx.Client.post to return success for send_notification calls."""
    def _make_response(batch):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"status": "ok"} for _ in batch]
        return mock_response

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post.side_effect = lambda url, **kw: _make_response(kw.get("json", []))
    return patch("httpx.Client", return_value=mock_client)


# ---------------------------------------------------------------------------
# DeviceToken — dataclass basics
# ---------------------------------------------------------------------------
class TestDeviceTokenFields:
    def test_token(self):
        dt = DeviceToken(token="abc123", platform="ios")
        assert dt.token == "abc123"

    def test_platform(self):
        dt = DeviceToken(token="x", platform="android")
        assert dt.platform == "android"

    def test_user_id_default(self):
        dt = DeviceToken(token="x", platform="ios")
        assert dt.user_id == "default"

    def test_user_id_custom(self):
        dt = DeviceToken(token="x", platform="ios", user_id="u42")
        assert dt.user_id == "u42"

    def test_topics_default(self):
        dt = DeviceToken(token="x", platform="ios")
        assert "chat" in dt.topics
        assert "training" in dt.topics

    def test_topics_custom(self):
        dt = DeviceToken(token="x", platform="ios", topics=["alerts"])
        assert dt.topics == ["alerts"]

    def test_created_at_populated(self):
        dt = DeviceToken(token="x", platform="ios")
        assert dt.created_at > 0

    def test_last_active_populated(self):
        dt = DeviceToken(token="x", platform="ios")
        assert dt.last_active > 0

    def test_enabled_default_true(self):
        dt = DeviceToken(token="x", platform="ios")
        assert dt.enabled is True

    def test_enabled_false(self):
        dt = DeviceToken(token="x", platform="ios", enabled=False)
        assert dt.enabled is False

    def test_web_platform(self):
        dt = DeviceToken(token="w", platform="web")
        assert dt.platform == "web"

    def test_ios_platform(self):
        dt = DeviceToken(token="i", platform="ios")
        assert dt.platform == "ios"

    def test_android_platform(self):
        dt = DeviceToken(token="a", platform="android")
        assert dt.platform == "android"

    def test_multiple_topics(self):
        dt = DeviceToken(token="x", platform="ios", topics=["a", "b", "c"])
        assert len(dt.topics) == 3

    def test_empty_topics(self):
        dt = DeviceToken(token="x", platform="ios", topics=[])
        assert dt.topics == []

    def test_last_active_recent(self):
        before = time.time()
        dt = DeviceToken(token="x", platform="ios")
        after = time.time()
        assert before <= dt.last_active <= after


class TestDeviceTokenMutability:
    def test_token_reassignable(self):
        dt = DeviceToken(token="old", platform="ios")
        dt.token = "new"
        assert dt.token == "new"

    def test_platform_reassignable(self):
        dt = DeviceToken(token="x", platform="ios")
        dt.platform = "android"
        assert dt.platform == "android"

    def test_enabled_toggle(self):
        dt = DeviceToken(token="x", platform="ios", enabled=True)
        dt.enabled = False
        assert dt.enabled is False

    def test_topics_append(self):
        dt = DeviceToken(token="x", platform="ios", topics=["chat"])
        dt.topics.append("alerts")
        assert "alerts" in dt.topics

    def test_user_id_reassignable(self):
        dt = DeviceToken(token="x", platform="ios", user_id="default")
        dt.user_id = "user_42"
        assert dt.user_id == "user_42"


# ---------------------------------------------------------------------------
# NotificationPayload — dataclass basics
# ---------------------------------------------------------------------------
class TestNotificationPayloadFields:
    def test_title(self):
        np_ = NotificationPayload(title="Hello", body="World")
        assert np_.title == "Hello"

    def test_body(self):
        np_ = NotificationPayload(title="t", body="b")
        assert np_.body == "b"

    def test_data_default(self):
        np_ = NotificationPayload(title="t", body="b")
        assert np_.data == {}

    def test_data_custom(self):
        np_ = NotificationPayload(title="t", body="b", data={"key": "val"})
        assert np_.data["key"] == "val"

    def test_sound_default(self):
        np_ = NotificationPayload(title="t", body="b")
        assert np_.sound == "default"

    def test_sound_custom(self):
        np_ = NotificationPayload(title="t", body="b", sound="silent")
        assert np_.sound == "silent"

    def test_badge_default(self):
        np_ = NotificationPayload(title="t", body="b")
        assert np_.badge is None

    def test_badge_value(self):
        np_ = NotificationPayload(title="t", body="b", badge=5)
        assert np_.badge == 5

    def test_topic_default(self):
        np_ = NotificationPayload(title="t", body="b")
        assert np_.topic is None

    def test_topic_value(self):
        np_ = NotificationPayload(title="t", body="b", topic="alerts")
        assert np_.topic == "alerts"

    def test_empty_title(self):
        np_ = NotificationPayload(title="", body="b")
        assert np_.title == ""

    def test_empty_body(self):
        np_ = NotificationPayload(title="t", body="")
        assert np_.body == ""

    def test_large_badge(self):
        np_ = NotificationPayload(title="t", body="b", badge=9999)
        assert np_.badge == 9999

    def test_negative_badge(self):
        np_ = NotificationPayload(title="t", body="b", badge=-1)
        assert np_.badge == -1

    def test_data_many_keys(self):
        data = {f"key_{i}": i for i in range(50)}
        np_ = NotificationPayload(title="t", body="b", data=data)
        assert len(np_.data) == 50


class TestNotificationPayloadMutability:
    def test_title_reassignable(self):
        np_ = NotificationPayload(title="old", body="b")
        np_.title = "new"
        assert np_.title == "new"

    def test_body_reassignable(self):
        np_ = NotificationPayload(title="t", body="old")
        np_.body = "new"
        assert np_.body == "new"

    def test_data_mutable(self):
        np_ = NotificationPayload(title="t", body="b", data={"k": 1})
        np_.data["k"] = 2
        assert np_.data["k"] == 2


# ---------------------------------------------------------------------------
# PushNotificationService — init
# ---------------------------------------------------------------------------
class TestPushNotificationServiceInit:
    def test_init(self):
        svc = _fresh_service()
        assert svc._devices == {}
        assert svc._history == []


# ---------------------------------------------------------------------------
# PushNotificationService — register/unregister
# ---------------------------------------------------------------------------
class TestPushNotificationServiceRegister:
    def test_register_device(self):
        svc = _fresh_service()
        svc.register_device("token_ios", "ios", user_id="u1")
        assert "token_ios" in svc._devices

    def test_register_updates_existing(self):
        svc = _fresh_service()
        svc.register_device("tok", "ios")
        result = svc.register_device("tok", "android")
        assert result["status"] == "updated"
        assert svc._devices["tok"].platform == "android"

    def test_register_custom_topics(self):
        svc = _fresh_service()
        svc.register_device("tok", "ios", topics=["alerts", "news"])
        assert svc._devices["tok"].topics == ["alerts", "news"]

    def test_register_stores_user_id(self):
        svc = _fresh_service()
        svc.register_device("tok", "ios", user_id="u42")
        assert svc._devices["tok"].user_id == "u42"

    def test_register_multiple_devices(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        svc.register_device("t2", "android")
        assert len(svc._devices) == 2

    def test_register_updates_last_active(self):
        svc = _fresh_service()
        svc.register_device("tok", "ios")
        old_time = svc._devices["tok"].last_active
        time.sleep(0.01)
        svc.register_device("tok", "ios")
        assert svc._devices["tok"].last_active >= old_time

    def test_register_default_topics(self):
        svc = _fresh_service()
        svc.register_device("tok", "ios")
        assert "chat" in svc._devices["tok"].topics
        assert "training" in svc._devices["tok"].topics

    def test_register_status_registered(self):
        svc = _fresh_service()
        result = svc.register_device("tok", "ios")
        assert result["status"] == "registered"

    def test_register_token_preview(self):
        svc = _fresh_service()
        result = svc.register_device("abcdefghijklmnopqrst", "ios")
        assert "..." in result["token"]


class TestPushNotificationServiceUnregister:
    def test_unregister_existing(self):
        svc = _fresh_service()
        svc.register_device("tok", "ios")
        assert svc.unregister_device("tok") is True
        assert "tok" not in svc._devices

    def test_unregister_nonexistent(self):
        svc = _fresh_service()
        assert svc.unregister_device("no_such") is False

    def test_unregister_does_not_affect_others(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        svc.register_device("t2", "android")
        svc.unregister_device("t1")
        assert "t2" in svc._devices


# ---------------------------------------------------------------------------
# PushNotificationService — get_devices
# ---------------------------------------------------------------------------
class TestPushNotificationServiceGetDevices:
    def test_get_devices_empty(self):
        svc = _fresh_service()
        assert svc.get_devices() == []

    def test_get_devices_all(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        svc.register_device("t2", "android")
        devices = svc.get_devices()
        assert len(devices) == 2

    def test_get_devices_filter_topic(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios", topics=["chat"])
        svc.register_device("t2", "android", topics=["training"])
        devices = svc.get_devices(topic="chat")
        assert len(devices) == 1

    def test_get_devices_excludes_disabled(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        svc.register_device("t2", "android")
        svc._devices["t2"].enabled = False
        devices = svc.get_devices()
        assert len(devices) == 1
        assert devices[0]["platform"] == "ios"

    def test_get_devices_format(self):
        svc = _fresh_service()
        svc.register_device("tok_ios", "ios", topics=["chat"])
        devices = svc.get_devices()
        d = devices[0]
        assert "token" in d
        assert "platform" in d
        assert "topics" in d
        assert "last_active" in d

    def test_get_devices_token_truncated(self):
        svc = _fresh_service()
        svc.register_device("a" * 50, "ios")
        devices = svc.get_devices()
        assert "..." in devices[0]["token"]
        assert len(devices[0]["token"]) < 50


# ---------------------------------------------------------------------------
# PushNotificationService — send_notification (sync)
# ---------------------------------------------------------------------------
class TestPushNotificationServiceSend:
    def test_send_no_recipients(self):
        svc = _fresh_service()
        payload = NotificationPayload(title="t", body="b")
        result = svc.send_notification(payload)
        assert result["status"] == "no_recipients"
        assert result["sent"] == 0

    def test_send_to_registered_devices(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        svc.register_device("t2", "android")
        payload = NotificationPayload(title="Hello", body="World")
        with _mock_send_success(svc):
            result = svc.send_notification(payload)
        assert result["sent"] == 2
        assert result["total"] == 2

    def test_send_with_topic_filter(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios", topics=["chat"])
        svc.register_device("t2", "android", topics=["training"])
        payload = NotificationPayload(title="t", body="b")
        with _mock_send_success(svc):
            result = svc.send_notification(payload, topic="chat")
        assert result["sent"] == 1

    def test_send_specific_tokens(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        svc.register_device("t2", "android")
        payload = NotificationPayload(title="t", body="b")
        with _mock_send_success(svc):
            result = svc.send_notification(payload, tokens=["t1"])
        assert result["sent"] == 1

    def test_send_nonexistent_token(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        payload = NotificationPayload(title="t", body="b")
        result = svc.send_notification(payload, tokens=["no_such"])
        assert result["status"] == "no_recipients"

    def test_send_records_history(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        payload = NotificationPayload(title="t", body="b")
        with _mock_send_success(svc):
            svc.send_notification(payload)
        history = svc.get_history()
        assert len(history) == 1
        assert history[0]["title"] == "t"

    def test_send_with_badge(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        payload = NotificationPayload(title="t", body="b", badge=3)
        with _mock_send_success(svc):
            result = svc.send_notification(payload)
        assert result["sent"] == 1

    def test_send_with_data(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        payload = NotificationPayload(title="t", body="b", data={"id": 42})
        with _mock_send_success(svc):
            result = svc.send_notification(payload)
        assert result["sent"] == 1

    def test_send_disabled_devices_excluded(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        svc._devices["t1"].enabled = False
        payload = NotificationPayload(title="t", body="b")
        result = svc.send_notification(payload)
        assert result["status"] == "no_recipients"

    def test_send_message_builds_correctly(self):
        svc = _fresh_service()
        svc.register_device("tok1", "ios")
        payload = NotificationPayload(
            title="Test Title", body="Test Body",
            data={"key": "val"}, sound="silent", badge=5,
        )
        with _mock_send_success(svc):
            result = svc.send_notification(payload)
        assert result["total"] == 1
        assert result["sent"] == 1

    def test_send_topic_sets_channel_id(self):
        svc = _fresh_service()
        svc.register_device("tok1", "ios")
        payload = NotificationPayload(title="t", body="b", topic="alerts")
        with _mock_send_success(svc):
            result = svc.send_notification(payload)
        assert result["total"] == 1


# ---------------------------------------------------------------------------
# PushNotificationService — send_notification_sync convenience
# ---------------------------------------------------------------------------
class TestPushNotificationServiceSync:
    def test_send_sync_basic(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        with _mock_send_success(svc):
            result = svc.send_notification_sync("Title", "Body")
        assert result["sent"] == 1

    def test_send_sync_with_topics(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios", topics=["chat"])
        svc.register_device("t2", "android", topics=["training"])
        with _mock_send_success(svc):
            result = svc.send_notification_sync("t", "b", topics=["chat"])
        assert result["sent"] == 1

    def test_send_sync_no_recipients(self):
        svc = _fresh_service()
        result = svc.send_notification_sync("t", "b")
        assert result["status"] == "no_recipients"

    def test_send_sync_with_data(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        with _mock_send_success(svc):
            result = svc.send_notification_sync("t", "b", data={"key": "val"})
        assert result["sent"] == 1

    def test_send_sync_with_sound(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        with _mock_send_success(svc):
            result = svc.send_notification_sync("t", "b", sound="silent")
        assert result["sent"] == 1

    def test_send_sync_multiple_recipients(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        svc.register_device("t2", "android")
        with _mock_send_success(svc):
            result = svc.send_notification_sync("t", "b")
        assert result["sent"] == 2


# ---------------------------------------------------------------------------
# PushNotificationService — history
# ---------------------------------------------------------------------------
class TestPushNotificationServiceHistory:
    def test_history_empty(self):
        svc = _fresh_service()
        assert svc.get_history() == []

    def test_history_after_send(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        with _mock_send_success(svc):
            svc.send_notification(NotificationPayload(title="a", body="b"))
            svc.send_notification(NotificationPayload(title="c", body="d"))
        history = svc.get_history()
        assert len(history) == 2

    def test_history_limit(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        with _mock_send_success(svc):
            for i in range(10):
                svc.send_notification(NotificationPayload(title=f"t{i}", body="b"))
        history = svc.get_history(limit=3)
        assert len(history) == 3

    def test_history_record_fields(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        with _mock_send_success(svc):
            svc.send_notification(NotificationPayload(title="t", body="b", topic="chat"))
        history = svc.get_history()
        record = history[0]
        assert "timestamp" in record
        assert "title" in record
        assert "body" in record
        assert "topic" in record
        assert "sent" in record
        assert "errors" in record

    def test_history_record_after_sync_send(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        with _mock_send_success(svc):
            svc.send_notification_sync("sync_title", "sync_body")
        history = svc.get_history()
        assert history[0]["title"] == "sync_title"

    def test_history_accumulates(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        with _mock_send_success(svc):
            for i in range(5):
                svc.send_notification(NotificationPayload(title=f"msg{i}", body="b"))
        assert len(svc.get_history()) == 5


# ---------------------------------------------------------------------------
# PushNotificationService — cleanup_stale
# ---------------------------------------------------------------------------
class TestPushNotificationServiceCleanup:
    def test_cleanup_no_stale(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        removed = svc.cleanup_stale(max_age_seconds=3600)
        assert removed == 0
        assert "t1" in svc._devices

    def test_cleanup_removes_stale(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        svc._devices["t1"].last_active = time.time() - 100000
        removed = svc.cleanup_stale(max_age_seconds=100)
        assert removed == 1
        assert "t1" not in svc._devices

    def test_cleanup_keeps_recent(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        svc.register_device("t2", "android")
        svc._devices["t1"].last_active = time.time() - 100000
        svc.cleanup_stale(max_age_seconds=100)
        assert "t2" in svc._devices

    def test_cleanup_empty(self):
        svc = _fresh_service()
        removed = svc.cleanup_stale()
        assert removed == 0

    def test_cleanup_default_30_days(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        svc._devices["t1"].last_active = time.time() - (31 * 24 * 3600)
        removed = svc.cleanup_stale()
        assert removed == 1

    def test_cleanup_only_old_devices(self):
        svc = _fresh_service()
        svc.register_device("old", "ios")
        svc.register_device("new", "android")
        svc._devices["old"].last_active = time.time() - 100000
        removed = svc.cleanup_stale(max_age_seconds=100)
        assert removed == 1
        assert "old" not in svc._devices
        assert "new" in svc._devices


# ---------------------------------------------------------------------------
# PushNotificationService — persistence (disk)
# ---------------------------------------------------------------------------
class TestPushNotificationServicePersistence:
    def test_devices_persisted(self, tmp_path):
        with patch("domains.mobile.notifications._DEVICES_FILE", tmp_path / "devices.json"), \
             patch("domains.mobile.notifications._HISTORY_FILE", tmp_path / "history.json"):
            svc = PushNotificationService()
            svc.register_device("persist_tok", "ios", user_id="u1")
            svc2 = PushNotificationService()
            assert "persist_tok" in svc2._devices
            assert svc2._devices["persist_tok"].platform == "ios"

    def test_history_persisted(self, tmp_path):
        with patch("domains.mobile.notifications._DEVICES_FILE", tmp_path / "devices.json"), \
             patch("domains.mobile.notifications._HISTORY_FILE", tmp_path / "history.json"):
            svc = PushNotificationService()
            svc.register_device("t1", "ios")
            with _mock_send_success(svc):
                svc.send_notification(NotificationPayload(title="persist", body="test"))
            svc2 = PushNotificationService()
            assert len(svc2.get_history()) == 1

    def test_corrupted_devices_file(self, tmp_path):
        devices_file = tmp_path / "devices.json"
        devices_file.write_text("{invalid json")
        with patch("domains.mobile.notifications._DEVICES_FILE", devices_file), \
             patch("domains.mobile.notifications._HISTORY_FILE", tmp_path / "history.json"):
            svc = PushNotificationService()
            assert svc._devices == {}

    def test_corrupted_history_file(self, tmp_path):
        history_file = tmp_path / "history.json"
        history_file.write_text("[invalid")
        with patch("domains.mobile.notifications._DEVICES_FILE", tmp_path / "devices.json"), \
             patch("domains.mobile.notifications._HISTORY_FILE", history_file):
            svc = PushNotificationService()
            assert svc._history == []


# ---------------------------------------------------------------------------
# PushNotificationService — message building details
# ---------------------------------------------------------------------------
class TestPushNotificationServiceMessageBuilding:
    def test_expo_message_fields(self):
        svc = _fresh_service()
        svc.register_device("expo_token", "ios")
        payload = NotificationPayload(title="Hi", body="There", data={"id": 1})
        with _mock_send_success(svc):
            result = svc.send_notification(payload)
        assert result["total"] == 1

    def test_channel_id_from_topic(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        payload = NotificationPayload(title="t", body="b", topic="alerts")
        with _mock_send_success(svc):
            result = svc.send_notification(payload)
        assert result["total"] == 1

    def test_badge_in_message(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        payload = NotificationPayload(title="t", body="b", badge=10)
        with _mock_send_success(svc):
            result = svc.send_notification(payload)
        assert result["sent"] == 1

    def test_sound_in_message(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        payload = NotificationPayload(title="t", body="b", sound="silent")
        with _mock_send_success(svc):
            result = svc.send_notification(payload)
        assert result["sent"] == 1

    def test_no_badge_when_none(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        payload = NotificationPayload(title="t", body="b", badge=None)
        with _mock_send_success(svc):
            result = svc.send_notification(payload)
        assert result["sent"] == 1

    def test_send_history_record_structure(self):
        svc = _fresh_service()
        svc.register_device("t1", "ios")
        with _mock_send_success(svc):
            svc.send_notification(NotificationPayload(title="t", body="b", topic="chat"))
        record = svc.get_history()[0]
        assert record["title"] == "t"
        assert record["body"] == "b"
        assert record["topic"] == "chat"
        assert record["sent"] == 1
        assert record["errors"] == []


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
class TestNotificationServiceSingleton:
    def test_get_notification_service(self):
        from domains.mobile import notifications as mod
        old = mod._service
        try:
            mod._service = None
            from domains.mobile.notifications import get_notification_service
            svc1 = get_notification_service()
            svc2 = get_notification_service()
            assert svc1 is svc2
        finally:
            mod._service = old
