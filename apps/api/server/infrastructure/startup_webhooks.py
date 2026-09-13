"""Startup webhook notifications — sends HTTP notifications on startup events.

Usage:
    from infrastructure.startup_webhooks import get_webhook_manager, register_webhook

    manager = get_webhook_manager()
    register_webhook("https://hooks.example.com/startup", events=["startup.complete", "startup.failed"])
    # Webhooks are triggered automatically on startup events
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class WebhookEvent(str, Enum):
    """Startup webhook events."""

    STARTUP_START = "startup.start"
    STAGE_COMPLETE = "startup.stage.complete"
    STARTUP_COMPLETE = "startup.complete"
    STARTUP_FAILED = "startup.failed"
    HOOK_FAILED = "startup.hook.failed"


@dataclass
class WebhookConfig:
    """Configuration for a webhook."""

    url: str
    events: list[WebhookEvent] = field(default_factory=lambda: list(WebhookEvent))
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 10.0
    enabled: bool = True
    retry_count: int = 3
    retry_delay: float = 1.0

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "events": [e.value for e in self.events],
            "headers": self.headers,
            "timeout": self.timeout,
            "enabled": self.enabled,
            "retry_count": self.retry_count,
        }


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""

    url: str
    event: WebhookEvent
    success: bool
    status_code: int | None = None
    error: str | None = None
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "event": self.event.value,
            "success": self.success,
            "status_code": self.status_code,
            "error": self.error,
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 2),
        }


class WebhookManager:
    """Manages and delivers startup webhook notifications."""

    def __init__(self) -> None:
        self._webhooks: list[WebhookConfig] = []
        self._deliveries: list[WebhookDelivery] = []
        self._max_deliveries = 100

    def register(self, config: WebhookConfig) -> None:
        """Register a webhook."""
        self._webhooks.append(config)
        logger.info("Registered webhook: %s", config.url)

    def unregister(self, url: str) -> None:
        """Unregister a webhook by URL."""
        self._webhooks = [w for w in self._webhooks if w.url != url]

    async def emit(self, event: WebhookEvent, data: dict[str, Any] | None = None) -> None:
        """Emit a webhook event to all matching webhooks."""
        payload = {
            "event": event.value,
            "timestamp": time.time(),
            "data": data or {},
        }

        tasks = []
        for webhook in self._webhooks:
            if webhook.enabled and event in webhook.events:
                tasks.append(self._deliver(webhook, payload))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _deliver(self, webhook: WebhookConfig, payload: dict) -> None:
        """Deliver a webhook payload with retries."""
        import httpx

        last_error = None
        for attempt in range(webhook.retry_count):
            start = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=webhook.timeout) as client:
                    response = await client.post(
                        webhook.url,
                        json=payload,
                        headers=webhook.headers,
                    )
                    duration_ms = (time.perf_counter() - start) * 1000

                    delivery = WebhookDelivery(
                        url=webhook.url,
                        event=WebhookEvent(payload["event"]),
                        success=response.status_code < 400,
                        status_code=response.status_code,
                        duration_ms=duration_ms,
                    )
                    self._record_delivery(delivery)

                    if delivery.success:
                        logger.debug("Webhook delivered to %s (%.1fms)", webhook.url, duration_ms)
                        return
                    else:
                        last_error = f"HTTP {response.status_code}"
                        logger.warning(
                            "Webhook delivery failed to %s: %s (attempt %d/%d)",
                            webhook.url,
                            last_error,
                            attempt + 1,
                            webhook.retry_count,
                        )
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                last_error = str(e)
                logger.warning(
                    "Webhook delivery error to %s: %s (attempt %d/%d)",
                    webhook.url,
                    last_error,
                    attempt + 1,
                    webhook.retry_count,
                )

            if attempt < webhook.retry_count - 1:
                await asyncio.sleep(webhook.retry_delay)

        # All retries failed
        delivery = WebhookDelivery(
            url=webhook.url,
            event=WebhookEvent(payload["event"]),
            success=False,
            error=last_error,
        )
        self._record_delivery(delivery)

    def _record_delivery(self, delivery: WebhookDelivery) -> None:
        """Record a webhook delivery."""
        self._deliveries.append(delivery)
        if len(self._deliveries) > self._max_deliveries:
            self._deliveries = self._deliveries[-self._max_deliveries:]

    def get_status(self) -> dict:
        """Get webhook manager status."""
        return {
            "webhooks": [w.to_dict() for w in self._webhooks],
            "recent_deliveries": [d.to_dict() for d in self._deliveries[-10:]],
            "total_deliveries": len(self._deliveries),
            "successful_deliveries": sum(1 for d in self._deliveries if d.success),
            "failed_deliveries": sum(1 for d in self._deliveries if not d.success),
        }


_global_manager: WebhookManager | None = None


def get_webhook_manager() -> WebhookManager:
    """Get the global webhook manager."""
    global _global_manager
    if _global_manager is None:
        _global_manager = WebhookManager()
    return _global_manager


def register_webhook(
    url: str,
    events: list[WebhookEvent] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 10.0,
) -> WebhookConfig:
    """Register a startup webhook."""
    config = WebhookConfig(
        url=url,
        events=events or list(WebhookEvent),
        headers=headers or {},
        timeout=timeout,
    )
    get_webhook_manager().register(config)
    return config
