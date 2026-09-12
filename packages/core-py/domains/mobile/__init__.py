"""Backward-compatibility shim — imports from the new ``domain.mobile`` package."""

from domain.mobile import (
    PushNotificationService,
    NotificationPayload,
    DeviceToken,
    get_notification_service,
)

__all__ = [
    "PushNotificationService",
    "NotificationPayload",
    "DeviceToken",
    "get_notification_service",
]
