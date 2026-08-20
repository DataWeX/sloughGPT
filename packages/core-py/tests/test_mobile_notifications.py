"""Tests for domains.mobile.notifications — DeviceToken, NotificationPayload."""

from domains.mobile.notifications import DeviceToken, NotificationPayload


class TestDeviceToken:
    def test_fields(self):
        dt = DeviceToken(token="abc123", platform="ios", user_id="u1")
        assert dt.token == "abc123"
        assert dt.platform == "ios"
        assert dt.user_id == "u1"
        assert dt.enabled is True

    def test_defaults(self):
        dt = DeviceToken(token="x", platform="android")
        assert dt.user_id == "default"
        assert "chat" in dt.topics
        assert "training" in dt.topics
        assert dt.last_active > 0

    def test_web_platform(self):
        dt = DeviceToken(token="w", platform="web")
        assert dt.platform == "web"


class TestNotificationPayload:
    def test_fields(self):
        np = NotificationPayload(title="Hello", body="World", data={"key": "val"})
        assert np.title == "Hello"
        assert np.body == "World"
        assert np.data["key"] == "val"
        assert np.sound == "default"

    def test_defaults(self):
        np = NotificationPayload(title="t", body="b")
        assert np.badge is None
        assert np.topic is None
