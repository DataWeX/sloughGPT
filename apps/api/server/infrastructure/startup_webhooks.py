"""Startup webhook notifications — startup events delivered via WebhookStore.

Facade over ``training.webhooks.WebhookStore`` (review ⑫-P3 consolidation):
one delivery engine for the whole app — MogDB-persisted registrations, HMAC
signing, exponential backoff, retry queue, dead letters. This module keeps the
historical startup-facing API (``WebhookEvent``, ``WebhookConfig``,
``register_webhook``, ``get_webhook_manager`` -> ``emit`` / ``get_status``)
so ``startup.py`` and the ``health.startup_webhooks`` endpoint are unchanged.

Usage:
    from infrastructure.startup_webhooks import get_webhook_manager, register_webhook

    register_webhook("https://hooks.example.com/startup", events=["startup.complete"])
    # Webhooks are triggered automatically on startup events; startup.* events
    # are also registerable through the REST API (POST /training/webhooks).
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class WebhookEvent(StrEnum):
    """Startup webhook events."""

    STARTUP_START = "startup.start"
    STAGE_COMPLETE = "startup.stage.complete"
    STARTUP_COMPLETE = "startup.complete"
    STARTUP_FAILED = "startup.failed"
    HOOK_FAILED = "startup.hook.failed"


@dataclass
class WebhookConfig:
    """Configuration for a webhook (mapped onto the shared store)."""

    url: str
    events: list[WebhookEvent] = field(default_factory=lambda: list(WebhookEvent))
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 10.0
    enabled: bool = True
    retry_count: int = 3
    retry_delay: float = 1.0  # Historical knob; the store owns backoff timing now

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "events": [e.value if isinstance(e, WebhookEvent) else str(e) for e in self.events],
            "headers": self.headers,
            "timeout": self.timeout,
            "enabled": self.enabled,
            "retry_count": self.retry_count,
        }


def _empty_status() -> dict:
    """The historical health.startup_webhooks shape with no data."""
    return {
        "webhooks": [],
        "recent_deliveries": [],
        "total_deliveries": 0,
        "successful_deliveries": 0,
        "failed_deliveries": 0,
    }


def _store():
    """Lazily fetch the shared store (keeps boot imports light: no httpx/mogdb)."""
    from training.webhooks import get_webhook_store

    return get_webhook_store()


def _is_startup_event(event: str) -> bool:
    return str(event).startswith("startup.")


class WebhookManager:
    """Startup-scoped view over the shared WebhookStore (no local state).

    The store holds registrations and delivery history; this class only
    projects them into the startup-facing API. Behavior notes:

    - ``register`` ignores ``WebhookConfig.enabled=False`` (the store has no
      inactive-registration concept — disable = unregister); ``get_status``
      reports ``enabled`` from the store's ``is_active`` flag.
    - Delivery, retries, HMAC signing, and persistence all live in the store.
    """

    def register(self, config: WebhookConfig) -> None:
        """Register a startup webhook in the shared store."""
        _store().register(
            url=config.url,
            events=[e.value if isinstance(e, WebhookEvent) else str(e) for e in config.events],
            headers=config.headers,
            timeout=config.timeout,
            retry_count=config.retry_count,
        )
        logger.info("Registered webhook: %s", config.url)

    def unregister(self, url: str) -> None:
        """Unregister startup webhooks by URL."""
        store = _store()
        for webhook in store.list():
            if webhook.url == url:
                store.unregister(webhook.id)

    async def emit(self, event: WebhookEvent, data: dict[str, Any] | None = None) -> None:
        """Emit a startup event — delivery happens in the store, fire-and-forget.

        ``notify_event`` schedules each delivery as a background task, so this
        returns immediately even if an endpoint is slow (the boot path relies
        on that).
        """
        from training.webhooks import notify_event

        await notify_event(event.value, data or {})

    def get_status(self) -> dict:
        """Startup-scoped status in the historical ``health.startup_webhooks`` shape.

        Keys and value shapes are pinned by
        ``tests/server/test_webhook_consolidation.py`` against the
        pre-refactor oracle. Degrades to the empty shape if the store is
        unavailable or a projection error occurs.
        """
        try:
            store = _store()
            hooks = [w for w in store.list() if any(_is_startup_event(e) for e in w.events)]
            startup_deliveries = [
                d for d in store.get_deliveries(limit=1000) if _is_startup_event(d.event)
            ]

            recent = []
            for d in startup_deliveries[-10:]:
                webhook = store.get(d.webhook_id)
                recent.append(
                    {
                        "url": webhook.url if webhook else "(removed)",
                        "event": d.event,
                        "success": d.success,
                        "status_code": d.status_code,
                        "error": d.error,
                        "timestamp": d.attempted_at.timestamp(),
                        "duration_ms": round(d.duration_ms, 2),
                    }
                )

            return {
                "webhooks": [
                    {
                        "url": w.url,
                        "events": list(w.events),
                        "headers": dict(w.headers),
                        "timeout": w.timeout,
                        "enabled": w.is_active,
                        "retry_count": w.retry_count,
                    }
                    for w in hooks
                ],
                "recent_deliveries": recent,
                "total_deliveries": len(startup_deliveries),
                "successful_deliveries": sum(1 for d in startup_deliveries if d.success),
                "failed_deliveries": sum(1 for d in startup_deliveries if not d.success),
            }
        except Exception:
            logger.warning("startup webhook status unavailable", exc_info=True)
            return _empty_status()


_global_manager: WebhookManager | None = None


def get_webhook_manager() -> WebhookManager:
    """Get global webhook manager instance."""
    global _global_manager
    if _global_manager is None:
        _global_manager = WebhookManager()
    return _global_manager


def register_webhook(
    url: str,
    events: list[WebhookEvent] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
    retry_count: int = 3,
) -> WebhookConfig:
    """Register a startup webhook (convenience wrapper around the manager)."""
    config = WebhookConfig(
        url=url,
        events=events or list(WebhookEvent),
        headers=headers or {},
        timeout=timeout,
        retry_count=retry_count,
    )
    get_webhook_manager().register(config)
    return config
