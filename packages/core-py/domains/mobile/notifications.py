"""
Push notification service for mobile.

Handles device token registration, notification sending, and topic subscriptions.
Uses Expo Push Notifications (APNs/FCM via Expo) for cross-platform delivery.
Persistence backed by MogDB.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, List, Dict, Any
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger("slo.mobile.notifications")

NOTIFICATIONS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "mogdb" / "mobile_notifications"

_db = None
_devices_col = None
_history_col = None


def _get_devices_col(db_path: Optional[str] = None):
    global _db, _devices_col
    if _devices_col is not None:
        return _devices_col
    if db_path is None:
        from domains.shared import find_repo_root
        repo = find_repo_root(Path(__file__).resolve())
        db_path = str(repo / "data" / "mogdb" / "mobile_notifications")
    from mogdb import MogDB
    _db = MogDB(db_path)
    _devices_col = _db.collection("devices")
    return _devices_col


def _get_history_col(db_path: Optional[str] = None):
    global _history_col
    if _history_col is not None:
        return _history_col
    if db_path is None:
        from domains.shared import find_repo_root
        repo = find_repo_root(Path(__file__).resolve())
        db_path = str(repo / "data" / "mogdb" / "mobile_notifications")
    from mogdb import MogDB
    if _db is None:
        _db_inst = MogDB(db_path)
    else:
        _db_inst = _db
    _history_col = _db_inst.collection("history")
    return _history_col


def set_mogdb_path(db_path: str) -> None:
    """Override the default MogDB path (used by tests)."""
    global _db, _devices_col, _history_col
    from mogdb import MogDB
    _db = MogDB(db_path)
    _devices_col = _db.collection("devices")
    _history_col = _db.collection("history")


def reset_mogdb() -> None:
    """Reset the module-level MogDB singletons (used by tests)."""
    global _db, _devices_col, _history_col
    _db = None
    _devices_col = None
    _history_col = None


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


def _strip_meta(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Remove MogDB internal fields from a document."""
    return {k: v for k, v in doc.items() if not k.startswith("_")}


class PushNotificationService:
    """
    Manages device tokens and sends push notifications via Expo.

    Notifications are persisted to MogDB and can be sent via Expo's push API
    or logged for development.
    """

    def __init__(self, db_path: Optional[str] = None):
        self._devices: Dict[str, DeviceToken] = {}
        self._history: List[Dict[str, Any]] = []
        self._dev_col = _get_devices_col(db_path)
        self._hist_col = _get_history_col(db_path)
        self._load()

    def _load(self):
        """Load devices and history from MogDB."""
        try:
            for doc in self._dev_col.find({}):
                token = doc["token"]
                self._devices[token] = DeviceToken(**{k: v for k, v in doc.items() if k != "_id"})
        except Exception as e:
            logger.warning("Failed to load devices: %s", e, extra={"tag": "MODEL"})

        try:
            for doc in self._hist_col.find({}):
                self._history.append(_strip_meta(doc))
        except Exception as e:
            logger.warning("Failed to load notification history: %s", e, extra={"tag": "MODEL"})

    def _save_device(self, device: DeviceToken):
        """Persist a single device to MogDB."""
        try:
            data = {
                "_id": device.token,
                "token": device.token,
                "platform": device.platform,
                "user_id": device.user_id,
                "topics": device.topics,
                "created_at": device.created_at,
                "last_active": device.last_active,
                "enabled": device.enabled,
            }
            existing = self._dev_col.find_one({"_id": device.token})
            if existing is not None:
                self._dev_col.update_one({"_id": device.token}, {"$set": data})
            else:
                self._dev_col.insert_one(data)
        except Exception as e:
            logger.warning("Failed to save device: %s", e, extra={"tag": "MODEL"})

    def _delete_device(self, token: str):
        """Remove a device from MogDB."""
        try:
            self._dev_col.delete_one({"_id": token})
        except Exception as e:
            logger.warning("Failed to delete device: %s", e, extra={"tag": "MODEL"})

    def _save_history_record(self, record: Dict[str, Any]):
        """Persist a single history record to MogDB."""
        try:
            data = {"_id": str(record["timestamp"]), **record}
            self._hist_col.insert_one(data)
            # Cap at 200 entries
            all_records = self._hist_col.find({}, sort=[("_id", -1)])
            if len(all_records) > 200:
                stale = all_records[200:]
                self._hist_col.delete_many({"_id": {"$in": [r["_id"] for r in stale]}})
        except Exception as e:
            logger.warning("Failed to save history: %s", e, extra={"tag": "MODEL"})

    def register_device(
        self,
        token: str,
        platform: str,
        user_id: str = "default",
        topics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        existing = self._devices.get(token)
        if existing:
            existing.last_active = time.time()
            existing.platform = platform
            if topics:
                existing.topics = topics
            self._save_device(existing)
            return {"status": "updated", "token": token[:20] + "..."}

        device = DeviceToken(
            token=token,
            platform=platform,
            user_id=user_id,
            topics=topics or ["chat", "training"],
        )
        self._devices[token] = device
        self._save_device(device)

        logger.info("Registered device: %s... (%s)", token[:20], platform, extra={"tag": "MODEL"})
        return {"status": "registered", "token": token[:20] + "..."}

    def unregister_device(self, token: str) -> bool:
        if token in self._devices:
            del self._devices[token]
            self._delete_device(token)
            return True
        return False

    def get_devices(self, topic: Optional[str] = None) -> List[Dict[str, Any]]:
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
        self._save_history_record(record)

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
        self._save_history_record(record)

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
        payload = NotificationPayload(
            title=title,
            body=body,
            data=data or {},
            sound=sound,
            topic=topics[0] if topics else None,
        )
        if topics:
            target_tokens = [
                t for t, d in self._devices.items()
                if d.enabled and any(topic in d.topics for topic in topics)
            ]
            return self.send_notification(payload, tokens=target_tokens)
        return self.send_notification(payload)

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self._history[-limit:]

    def cleanup_stale(self, max_age_seconds: float = 30 * 24 * 3600) -> int:
        cutoff = time.time() - max_age_seconds
        stale = [t for t, d in self._devices.items() if d.last_active < cutoff]
        for token in stale:
            del self._devices[token]
            self._delete_device(token)
        return len(stale)


# Singleton
_service: Optional[PushNotificationService] = None


def get_notification_service() -> PushNotificationService:
    """Get or create the push notification service singleton."""
    global _service
    if _service is None:
        _service = PushNotificationService()
    return _service
