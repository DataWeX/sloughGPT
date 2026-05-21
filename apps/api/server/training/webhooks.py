"""Webhook notification system for training events.

Allows registering URLs to receive POST notifications when training events occur.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import httpx

logger = logging.getLogger("sloughgpt.webhooks")


@dataclass
class Webhook:
    """A registered webhook endpoint."""

    id: str
    url: str
    events: List[str]  # Event types to receive
    secret: str  # HMAC secret for signature
    created_at: datetime
    is_active: bool = True
    description: str = ""
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""

    id: str
    webhook_id: str
    event: str
    payload: Dict[str, Any]
    status_code: Optional[int] = None
    success: bool = False
    attempted_at: datetime = field(default_factory=datetime.now)
    response_body: Optional[str] = None
    error: Optional[str] = None


class WebhookStore:
    """
    SQLite-backed store for managing registered webhooks.

    Persists across server restarts.
    """

    def __init__(self, db_path: str = "data/webhooks.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS webhooks (
                    id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    events TEXT NOT NULL,
                    secret TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    headers TEXT DEFAULT '{}'
                )
            """)
            conn.commit()
            conn.close()

    def _row_to_webhook(self, row: sqlite3.Row) -> Webhook:
        """Convert a database row to a Webhook object."""
        return Webhook(
            id=row["id"],
            url=row["url"],
            events=json.loads(row["events"]),
            secret=row["secret"],
            description=row["description"],
            is_active=bool(row["is_active"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            headers=json.loads(row["headers"]) if row["headers"] else {},
        )

    def register(
        self,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
        description: str = "",
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """Register a new webhook."""
        webhook_id = hashlib.sha256(f"{url}{time.time()}".encode()).hexdigest()[:16]

        # Generate secret if not provided
        if secret is None:
            secret = hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:32]

        now = datetime.now().isoformat()

        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                """
                INSERT INTO webhooks (id, url, events, secret, description, is_active, created_at, headers)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?)
            """,
                (
                    webhook_id,
                    url,
                    json.dumps(events),
                    secret,
                    description,
                    now,
                    json.dumps(headers or {}),
                ),
            )
            conn.commit()
            conn.close()

        logger.info(f"Registered webhook {webhook_id} for {url}")
        return webhook_id

    def unregister(self, webhook_id: str) -> bool:
        """Unregister a webhook."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.execute("DELETE FROM webhooks WHERE id = ?", (webhook_id,))
            conn.commit()
            deleted = cursor.rowcount > 0
            conn.close()

        if deleted:
            logger.info(f"Unregistered webhook {webhook_id}")
        return deleted

    def get(self, webhook_id: str) -> Optional[Webhook]:
        """Get a webhook by ID."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM webhooks WHERE id = ?", (webhook_id,))
            row = cursor.fetchone()
            conn.close()

        if row:
            return self._row_to_webhook(row)
        return None

    def list(self, event_filter: Optional[str] = None) -> List[Webhook]:
        """List all webhooks, optionally filtered by event."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM webhooks WHERE is_active = 1 ORDER BY created_at DESC"
            )
            rows = cursor.fetchall()
            conn.close()

        webhooks = [self._row_to_webhook(row) for row in rows]

        if event_filter:
            webhooks = [w for w in webhooks if event_filter in w.events]

        return webhooks

    def get_secret(self, webhook_id: str) -> Optional[str]:
        """Get the secret for a webhook (for signing)."""
        webhook = self.get(webhook_id)
        return webhook.secret if webhook else None

    def sign_payload(self, webhook_id: str, payload: str) -> Optional[str]:
        """Generate HMAC signature for payload."""
        secret = self.get_secret(webhook_id)
        if not secret:
            return None

        signature = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

        return f"sha256={signature}"

    async def deliver(
        self,
        webhook_id: str,
        event: str,
        payload: Dict[str, Any],
        timeout: float = 10.0,
        retries: int = 3,
    ) -> WebhookDelivery:
        """Deliver a webhook event to the endpoint."""
        webhook = self.get(webhook_id)
        delivery = WebhookDelivery(
            id=hashlib.sha256(f"{webhook_id}{time.time()}".encode()).hexdigest()[:16],
            webhook_id=webhook_id,
            event=event,
            payload=payload,
        )

        if not webhook or not webhook.is_active:
            delivery.error = "Webhook not found or inactive"
            self._add_delivery(delivery)
            return delivery

        # Check if webhook wants this event
        if event not in webhook.events:
            delivery.error = "Event not subscribed"
            self._add_delivery(delivery)
            return delivery

        # Prepare payload with metadata
        full_payload = {
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "data": payload,
        }

        payload_str = str(full_payload)
        signature = self.sign_payload(webhook_id, payload_str)

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "SloughGPT-Webhook/1.0",
            "X-Webhook-Event": event,
            "X-Webhook-Delivery": delivery.id,
            **webhook.headers,
        }

        if signature:
            headers["X-Webhook-Signature"] = signature

        # Deliver with retries
        last_error = None
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    response = await client.post(
                        webhook.url,
                        content=payload_str,
                        headers=headers,
                    )

                    delivery.status_code = response.status_code
                    delivery.response_body = response.text[:500]
                    delivery.success = 200 <= response.status_code < 300

                    if delivery.success:
                        logger.info(f"Webhook {webhook_id} delivered successfully")
                        break

                    last_error = f"HTTP {response.status_code}"
                    logger.warning(
                        f"Webhook {webhook_id} delivery failed (attempt {attempt + 1}): {last_error}"
                    )

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Webhook {webhook_id} delivery failed (attempt {attempt + 1}): {e}")

            if attempt < retries - 1:
                await asyncio.sleep(2**attempt)  # Exponential backoff

        if not delivery.success:
            delivery.error = last_error
            logger.error(f"Webhook {webhook_id} delivery failed after {retries} attempts")

        self._add_delivery(delivery)
        return delivery

    def _add_delivery(self, delivery: WebhookDelivery) -> None:
        """Add delivery to log, trimming if needed."""
        self.delivery_log.append(delivery)
        if len(self.delivery_log) > self._max_log_size:
            self.delivery_log = self.delivery_log[-self._max_log_size :]

    def get_deliveries(
        self,
        webhook_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[WebhookDelivery]:
        """Get delivery log."""
        deliveries = self.delivery_log

        if webhook_id:
            deliveries = [d for d in deliveries if d.webhook_id == webhook_id]

        return deliveries[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get webhook statistics."""
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM webhooks")
            total_webhooks = cursor.fetchone()["cnt"]
            cursor = conn.execute("SELECT COUNT(*) as cnt FROM webhooks WHERE is_active = 1")
            active_webhooks = cursor.fetchone()["cnt"]
            conn.close()

        total_deliveries = len(self.delivery_log)
        successful = sum(1 for d in self.delivery_log if d.success)
        failed = total_deliveries - successful

        return {
            "total_webhooks": total_webhooks,
            "active_webhooks": active_webhooks,
            "total_deliveries": total_deliveries,
            "successful_deliveries": successful,
            "failed_deliveries": failed,
            "success_rate": f"{(successful / total_deliveries * 100):.1f}%"
            if total_deliveries > 0
            else "N/A",
        }


# Global webhook store
webhook_store = WebhookStore()


def get_webhook_store() -> WebhookStore:
    """Get the global webhook store."""
    return webhook_store


# Event types
TRAINING_EVENTS = [
    "training.started",
    "training.progress",
    "training.completed",
    "training.failed",
    "training.stopped",
]


async def notify_training_event(
    event: str,
    payload: Dict[str, Any],
    sync: bool = False,
) -> List[WebhookDelivery]:
    """Send notification to all matching webhooks."""
    if event not in TRAINING_EVENTS:
        logger.warning(f"Unknown training event: {event}")
        return []

    store = get_webhook_store()
    matching_webhooks = store.list(event_filter=event)

    if not matching_webhooks:
        logger.debug(f"No webhooks registered for event: {event}")
        return []

    logger.info(f"Sending {event} to {len(matching_webhooks)} webhook(s)")

    deliveries = []
    for webhook in matching_webhooks:
        if sync:
            # Synchronous delivery
            delivery = await store.deliver(webhook.id, event, payload, retries=1)
        else:
            # Fire and forget in background
            asyncio.create_task(store.deliver(webhook.id, event, payload))
            delivery = WebhookDelivery(
                id="pending",
                webhook_id=webhook.id,
                event=event,
                payload=payload,
            )
        deliveries.append(delivery)

    return deliveries
