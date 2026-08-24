"""
Push notification service for mobile.

Handles device token registration, notification sending, and topic subscriptions.
Uses Expo Push Notifications (APNs/FCM via Expo) for cross-platform delivery.
"""

import logging
import time
from typing import Optional, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict
import json

logger = logging.getLogger("slo.mobile.notifications")

_NOTIFICATIONS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "mobile_notifications"
_NOTIFICATIONS_DIR.mkdir(parents=True, exist_ok=True)

_DEVICES_FILE = _NOTIFICATIONS_DIR / "devices.json"
_HISTORY_FILE = _NOTIFICATIONS_DIR / "history.json"


@dataclass
class DeviceToken:
    """Registered mobile device."""
    token: str
    platform: str  # "ios" | "android" | "web"
    user_id: str = "default"
    topics: List[str] = field(default_factory=lambda: ["chat", "training"])
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    enabled: bool = True


@dataclass
class NotificationPayload:
    """Push notification payload."""
    title: str
    body: str
    data: Dict[str, Any] = field(default_factory=dict)
    sound: str = "default"
    badge: Optional[int] = None
    topic: Optional[str] = None


class PushNotificationService:
    """
    Manages device tokens and sends push notifications via Expo.

    Notifications are persisted to disk and can be sent via Expo's push API
    or logged for development.
    """

    def __init__(self):
        self._devices: Dict[str, DeviceToken] = {}
        self._history: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        """Load devices and history from disk."""
        try:
            if _DEVICES_FILE.exists():
                data = json.loads(_DEVICES_FILE.read_text())
                for token_str, device_data in data.items():
                    self._devices[token_str] = DeviceToken(**device_data)
        except Exception as e:
            logger.warning("Failed to load devices: %s", e, extra={"tag": "MODEL"})

        try:
            if _HISTORY_FILE.exists():
                self._history = json.loads(_HISTORY_FILE.read_text())
        except Exception as e:
            logger.warning("Failed to load notification history: %s", e, extra={"tag": "MODEL"})

    def _save_devices(self):
        """Persist devices to disk."""
        try:
            data = {k: asdict(v) for k, v in self._devices.items()}
            _DEVICES_FILE.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to save devices: %s", e, extra={"tag": "MODEL"})

    def _save_history(self):
        """Persist notification history to disk."""
        try:
            _HISTORY_FILE.write_text(json.dumps(self._history[-200:], indent=2))
        except Exception as e:
            logger.warning("Failed to save history: %s", e, extra={"tag": "MODEL"})

    def register_device(
        self,
        token: str,
        platform: str,
        user_id: str = "default",
        topics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Register a mobile device for push notifications.

        Args:
            token: Expo push token or FCM/APNs token.
            platform: "ios", "android", or "web".
            user_id: User identifier (default: "default").
            topics: Notification topics to subscribe to.

        Returns:
            Registration status with device info.
        """
        existing = self._devices.get(token)
        if existing:
            existing.last_active = time.time()
            existing.platform = platform
            if topics:
                existing.topics = topics
            self._save_devices()
            return {"status": "updated", "token": token[:20] + "..."}

        device = DeviceToken(
            token=token,
            platform=platform,
            user_id=user_id,
            topics=topics or ["chat", "training"],
        )
        self._devices[token] = device
        self._save_devices()

        logger.info("Registered device: %s... (%s)", token[:20], platform, extra={"tag": "MODEL"})
        return {"status": "registered", "token": token[:20] + "..."}

    def unregister_device(self, token: str) -> bool:
        """Unregister a device."""
        if token in self._devices:
            del self._devices[token]
            self._save_devices()
            return True
        return False

    def get_devices(self, topic: Optional[str] = None) -> List[Dict[str, Any]]:
        """List registered devices, optionally filtered by topic."""
        devices = []
        for device in self._devices.values():
            if not device.enabled:
                continue
            if topic and topic not in device.topics:
                continue
            devices.append({
                "token": device.token[:20] + "...",
                "platform": device.platform,
                "topics": device.topics,
                "last_active": device.last_active,
            })
        return devices

    def send_notification(
        self,
        payload: NotificationPayload,
        tokens: Optional[List[str]] = None,
        topic: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a push notification to registered devices (sync).

        Args:
            payload: Notification title, body, data, sound, badge.
            tokens: Specific tokens to send to (None = all matching topic).
            topic: Filter devices by topic.

        Returns:
            Send result with count of recipients.
        """
        # Determine target devices
        if tokens:
            target_devices = [
                self._devices[t] for t in tokens if t in self._devices
            ]
        else:
            target_devices = [
                d for d in self._devices.values()
                if d.enabled and (not topic or topic in d.topics)
            ]

        if not target_devices:
            return {"status": "no_recipients", "sent": 0}

        # Build Expo push messages
        messages = []
        for device in target_devices:
            msg = {
                "to": device.token,
                "title": payload.title,
                "body": payload.body,
                "sound": payload.sound,
                "data": payload.data,
            }
            if payload.badge is not None:
                msg["badge"] = payload.badge
            if payload.topic:
                msg["channelId"] = payload.topic
            messages.append(msg)

        # Send via Expo Push API
        sent_count = 0
        errors = []

        try:
            import httpx

            # Expo push API accepts batches of up to 100
            for i in range(0, len(messages), 100):
                batch = messages[i:i + 100]
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(
                        "https://exp.host/--/api/v2/push/send",
                        json=batch,
                        headers={"Content-Type": "application/json"},
                    )
                    if resp.status_code == 200:
                        result = resp.json()
                        if isinstance(result, list):
                            for r in result:
                                if r.get("status") == "ok":
                                    sent_count += 1
                                elif r.get("status") == "error":
                                    errors.append(r.get("message", "unknown"))
                    else:
                        errors.append(f"HTTP {resp.status_code}")
        except ImportError:
            # httpx not installed — log the notification
            sent_count = len(messages)
            logger.info("Notification (httpx unavailable, logged): %s → %s devices", payload.title, len(messages), extra={"tag": "MODEL"})
        except Exception as e:
            errors.append(str(e))
            logger.error("Failed to send notifications: %s", e, extra={"tag": "MODEL"})

        # Record in history
        record = {
            "timestamp": time.time(),
            "title": payload.title,
            "body": payload.body,
            "topic": payload.topic,
            "sent": sent_count,
            "errors": errors[:5],
        }
        self._history.append(record)
        self._save_history()

        return {
            "status": "sent" if sent_count > 0 else "failed",
            "sent": sent_count,
            "total": len(messages),
            "errors": errors[:5],
        }

    async def send_notification_async(
        self,
        payload: NotificationPayload,
        tokens: Optional[List[str]] = None,
        topic: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send a push notification to registered devices (async, non-blocking).

        Uses httpx.AsyncClient so the uvicorn event loop is not blocked.
        """
        if tokens:
            target_devices = [
                self._devices[t] for t in tokens if t in self._devices
            ]
        else:
            target_devices = [
                d for d in self._devices.values()
                if d.enabled and (not topic or topic in d.topics)
            ]

        if not target_devices:
            return {"status": "no_recipients", "sent": 0}

        messages = []
        for device in target_devices:
            msg = {
                "to": device.token,
                "title": payload.title,
                "body": payload.body,
                "sound": payload.sound,
                "data": payload.data,
            }
            if payload.badge is not None:
                msg["badge"] = payload.badge
            if payload.topic:
                msg["channelId"] = payload.topic
            messages.append(msg)

        sent_count = 0
        errors = []

        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                for i in range(0, len(messages), 100):
                    batch = messages[i:i + 100]
                    resp = await client.post(
                        "https://exp.host/--/api/v2/push/send",
                        json=batch,
                        headers={"Content-Type": "application/json"},
                    )
                    if resp.status_code == 200:
                        result = resp.json()
                        if isinstance(result, list):
                            for r in result:
                                if r.get("status") == "ok":
                                    sent_count += 1
                                elif r.get("status") == "error":
                                    errors.append(r.get("message", "unknown"))
                    else:
                        errors.append(f"HTTP {resp.status_code}")
        except ImportError:
            sent_count = len(messages)
            logger.info("Notification (httpx unavailable, logged): %s → %s devices", payload.title, len(messages), extra={"tag": "MODEL"})
        except Exception as e:
            errors.append(str(e))
            logger.error("Failed to send notifications: %s", e, extra={"tag": "MODEL"})

        record = {
            "timestamp": time.time(),
            "title": payload.title,
            "body": payload.body,
            "topic": payload.topic,
            "sent": sent_count,
            "errors": errors[:5],
        }
        self._history.append(record)
        self._save_history()

        return {
            "status": "sent" if sent_count > 0 else "failed",
            "sent": sent_count,
            "total": len(messages),
            "errors": errors[:5],
        }

    def send_notification_sync(
        self,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None,
        topics: Optional[List[str]] = None,
        sound: str = "default",
    ) -> Dict[str, Any]:
        """Convenience method — build payload and send to topic-filtered devices."""
        payload = NotificationPayload(
            title=title,
            body=body,
            data=data or {},
            sound=sound,
            topic=topics[0] if topics else None,
        )
        # Filter by each topic (union)
        if topics:
            target_tokens = [
                t for t, d in self._devices.items()
                if d.enabled and any(topic in d.topics for topic in topics)
            ]
            return self.send_notification(payload, tokens=target_tokens)
        return self.send_notification(payload)

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent notification history."""
        return self._history[-limit:]

    def cleanup_stale(self, max_age_seconds: float = 30 * 24 * 3600) -> int:
        """Remove devices inactive for more than max_age_seconds."""
        cutoff = time.time() - max_age_seconds
        stale = [t for t, d in self._devices.items() if d.last_active < cutoff]
        for token in stale:
            del self._devices[token]
        if stale:
            self._save_devices()
        return len(stale)


# Singleton
_service: Optional[PushNotificationService] = None


def get_notification_service() -> PushNotificationService:
    """Get or create the push notification service singleton."""
    global _service
    if _service is None:
        _service = PushNotificationService()
    return _service
