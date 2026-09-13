"""Backward-compatibility shim — imports from the new ``domain.mobile`` package."""

from domain.mobile import (
    PushNotificationService,
    NotificationPayload,
    DeviceToken,
    get_notification_service,
)

from domain.mobile._internal import notifications  # noqa: F401

__all__ = [
    "PushNotificationService",
    "NotificationPayload",
    "DeviceToken",
    "get_notification_service",
]
