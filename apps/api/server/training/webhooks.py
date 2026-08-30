"""Webhook notification system for training events.

Allows registering URLs to receive POST notifications when training events occur.
Includes persistent retry queue with exponential backoff and dead-letter tracking.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field
from datetime import datetime
import httpx

from mogdb import MogDB
from domains.shared import find_repo_root

logger = logging.getLogger("slo.webhooks")


@dataclass
class Webhook:
    """A registered webhook endpoint."""

    id: str
    url: str
    events: list[str]  # Event types to receive
    secret: str  # HMAC secret for signature
    created_at: datetime
    is_active: bool = True
    description: str = ""
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""

    id: str
    webhook_id: str
    event: str
    payload: dict[str, Any]
    status_code: int | None = None
    success: bool = False
    attempted_at: datetime = field(default_factory=datetime.now)
    response_body: str | None = None
    error: str | None = None
    attempt_count: int = 0
    next_retry_at: datetime | None = None
    dead_letter: bool = False


class WebhookStore:
    """
    MogDB-backed store for managing registered webhooks.

    Persists across server restarts. Delivery records are kept in memory
    (matching the original SQLite behaviour, which never persisted them).
    Failed deliveries are queued for retry with exponential backoff.
    """

    _max_log_size: int = 1000
    _max_retries: int = 3
    _retry_base_delay: float = 2.0  # seconds
    _max_retry_delay: float = 300.0  # 5 minutes
    _dead_letter_max_attempts: int = 5

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            db_path = str(find_repo_root(Path(__file__).resolve()) / "data" / "webhooks.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.delivery_log: list = []
        self._retry_queue: list[dict] = []  # Pending retries
        self._dead_letters: list[dict] = []  # Permanently failed deliveries
        self._db: MogDB | None = None
        self._webhooks = None
        self._retry_worker_task: asyncio.Task | None = None
        try:
            self._db = MogDB(str(self.db_path))
            self._webhooks = self._db.collection("webhooks")
        except Exception:
            logger.warning("WebhookStore: failed to open MogDB at %s, operating in degraded mode", self.db_path)

    @staticmethod
    def _doc_to_webhook(doc: dict[str, Any]) -> Webhook:
        """Convert a stored MogDB document to a Webhook object."""
        return Webhook(
            id=doc["_id"],
            url=doc["url"],
            events=doc["events"],
            secret=doc["secret"],
            description=doc.get("description", ""),
            is_active=bool(doc.get("is_active", True)),
            created_at=datetime.fromisoformat(doc["created_at"]),
            headers=doc.get("headers") or {},
        )

    @property
    def is_available(self) -> bool:
        """Whether the backing store is usable."""
        return self._db is not None and self._webhooks is not None

    def register(
        self,
        url: str,
        events: list[str],
        secret: str | None = None,
        description: str = "",
        headers: dict[str, str] | None = None,
    ) -> str:
        """Register a new webhook."""
        if not self.is_available:
            raise RuntimeError("Webhook store unavailable (MogDB failed to initialise)")

        webhook_id = hashlib.sha256(f"{url}{time.time()}".encode()).hexdigest()[:16]

        # Generate secret if not provided
        if secret is None:
            secret = hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:32]

        now = datetime.now().isoformat()

        with self._lock:
            self._webhooks.insert_one(
                {
                    "_id": webhook_id,
                    "url": url,
                    "events": events,
                    "secret": secret,
                    "description": description,
                    "is_active": True,
                    "created_at": now,
                    "headers": headers or {},
                }
            )

        logger.info("Registered webhook %s for %s", webhook_id, url, extra={"tag": "TRAIN"})
        return webhook_id

    def unregister(self, webhook_id: str) -> bool:
        """Unregister a webhook."""
        if not self.is_available:
            return False
        with self._lock:
            deleted = self._webhooks.delete_one({"_id": webhook_id}) > 0

        if deleted:
            logger.info("Unregistered webhook %s", webhook_id, extra={"tag": "TRAIN"})
        return deleted

    def get(self, webhook_id: str) -> Webhook | None:
        """Get a webhook by ID."""
        if not self.is_available:
            return None
        doc = self._webhooks.find_one({"_id": webhook_id})
        return self._doc_to_webhook(doc) if doc else None

    def list(self, event_filter: str | None = None) -> list[Webhook]:
        """List all webhooks, optionally filtered by event."""
        if not self.is_available:
            return []
        docs = self._webhooks.find(
            {"is_active": True},
            sort=[("created_at", -1)],
        )

        webhooks = [self._doc_to_webhook(d) for d in docs]

        if event_filter:
            webhooks = [w for w in webhooks if event_filter in w.events]

        return webhooks

    def get_secret(self, webhook_id: str) -> str | None:
        """Get the secret for a webhook (for signing)."""
        webhook = self.get(webhook_id)
        return webhook.secret if webhook else None

    def sign_payload(self, webhook_id: str, payload: str) -> str | None:
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
        payload: dict[str, Any],
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
            attempt_count=0,
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

        payload_str = json.dumps(full_payload, default=str)
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
            delivery.attempt_count = attempt + 1
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
                        logger.info("Webhook %s delivered successfully (attempt %d)", webhook_id, attempt + 1, extra={"tag": "TRAIN"})
                        break

                    last_error = f"HTTP {response.status_code}"
                    logger.warning(
                        "Webhook %s delivery failed (attempt %d/%d): %s",
                        webhook_id, attempt + 1, retries, last_error,
                        extra={"tag": "TRAIN"},
                    )

            except Exception as e:
                last_error = str(e)
                logger.warning("Webhook %s delivery failed (attempt %d/%d): %s",
                               webhook_id, attempt + 1, retries, e, extra={"tag": "TRAIN"})

            # Exponential backoff between attempts
            if attempt < retries - 1:
                delay = min(self._retry_base_delay * (2 ** attempt), self._max_retry_delay)
                await asyncio.sleep(delay)

        if not delivery.success:
            delivery.error = last_error
            # Queue for background retry if we haven't exhausted all retries
            if retries < self._dead_letter_max_attempts:
                self._queue_retry(delivery, webhook_id, event, payload, timeout)
            else:
                delivery.dead_letter = True
                self._add_dead_letter(delivery)
                logger.error("Webhook %s moved to dead letter queue after %d attempts", webhook_id, retries, extra={"tag": "TRAIN"})
            logger.error("Webhook %s delivery failed after %d attempts", webhook_id, retries, extra={"tag": "TRAIN"})

        self._add_delivery(delivery)
        return delivery

    def _queue_retry(
        self,
        delivery: WebhookDelivery,
        webhook_id: str,
        event: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> None:
        """Queue a failed delivery for background retry."""
        next_retry_delay = min(
            self._retry_base_delay * (2 ** delivery.attempt_count),
            self._max_retry_delay,
        )
        next_retry_at = datetime.now().timestamp() + next_retry_delay

        retry_entry = {
            "delivery_id": delivery.id,
            "webhook_id": webhook_id,
            "event": event,
            "payload": payload,
            "timeout": timeout,
            "attempt_count": delivery.attempt_count,
            "next_retry_at": next_retry_at,
            "queued_at": datetime.now().isoformat(),
        }

        self._retry_queue.append(retry_entry)
        logger.info(
            "Webhook %s queued for retry in %.1fs (attempt %d/%d)",
            webhook_id, next_retry_delay, delivery.attempt_count, self._dead_letter_max_attempts,
            extra={"tag": "TRAIN"},
        )

    def _add_dead_letter(self, delivery: WebhookDelivery) -> None:
        """Add a permanently failed delivery to the dead letter queue."""
        self._dead_letters.append({
            "delivery_id": delivery.id,
            "webhook_id": delivery.webhook_id,
            "event": delivery.event,
            "error": delivery.error,
            "status_code": delivery.status_code,
            "attempt_count": delivery.attempt_count,
            "dead_lettered_at": datetime.now().isoformat(),
        })
        # Trim dead letter queue
        if len(self._dead_letters) > self._max_log_size:
            self._dead_letters = self._dead_letters[-self._max_log_size:]

    async def process_retry_queue(self) -> int:
        """Process pending retries. Returns number of retries attempted."""
        now = datetime.now().timestamp()
        ready = [r for r in self._retry_queue if r["next_retry_at"] <= now]

        if not ready:
            return 0

        processed = 0
        for entry in ready:
            self._retry_queue.remove(entry)
            attempt_count = entry["attempt_count"]

            if attempt_count >= self._dead_letter_max_attempts:
                # Move to dead letter
                delivery = WebhookDelivery(
                    id=entry["delivery_id"],
                    webhook_id=entry["webhook_id"],
                    event=entry["event"],
                    payload=entry["payload"],
                    attempt_count=attempt_count,
                    dead_letter=True,
                )
                self._add_dead_letter(delivery)
                continue

            # Attempt retry
            try:
                await self.deliver(
                    entry["webhook_id"],
                    entry["event"],
                    entry["payload"],
                    timeout=entry["timeout"],
                    retries=1,  # Single attempt, will re-queue if needed
                )
                processed += 1
            except Exception as e:
                logger.warning("Retry failed for delivery %s: %s", entry["delivery_id"], e, extra={"tag": "TRAIN"})

        return processed

    def get_retry_queue(self) -> list[dict]:
        """Get pending retries."""
        return list(self._retry_queue)

    def get_dead_letters(self, limit: int = 50) -> list[dict]:
        """Get dead-lettered deliveries."""
        return self._dead_letters[-limit:]

    def _add_delivery(self, delivery: WebhookDelivery) -> None:
        """Add delivery to log, trimming if needed."""
        self.delivery_log.append(delivery)
        if len(self.delivery_log) > self._max_log_size:
            self.delivery_log = self.delivery_log[-self._max_log_size :]

    def get_deliveries(
        self,
        webhook_id: str | None = None,
        limit: int = 50,
    ) -> list[WebhookDelivery]:
        """Get delivery log."""
        deliveries = self.delivery_log

        if webhook_id:
            deliveries = [d for d in deliveries if d.webhook_id == webhook_id]

        return deliveries[-limit:]

    def get_stats(self) -> dict[str, Any]:
        """Get webhook statistics."""
        if not self.is_available:
            return {
                "total_webhooks": 0,
                "active_webhooks": 0,
                "total_deliveries": len(self.delivery_log),
                "successful_deliveries": sum(1 for d in self.delivery_log if d.success),
                "failed_deliveries": sum(1 for d in self.delivery_log if not d.success),
                "success_rate": "N/A",
                "pending_retries": len(self._retry_queue),
                "dead_letters": len(self._dead_letters),
            }
        with self._lock:
            docs = self._webhooks.find()
            total_webhooks = len(docs)
            active_webhooks = sum(1 for d in docs if d.get("is_active"))

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
            "pending_retries": len(self._retry_queue),
            "dead_letters": len(self._dead_letters),
        }


# Global webhook store (lazy — initialised on first access to avoid import-time crashes)
_webhook_store: WebhookStore | None = None
_retry_worker_started: bool = False


def get_webhook_store() -> WebhookStore:
    """Get the global webhook store, initialising on first call."""
    global _webhook_store, _retry_worker_started
    if _webhook_store is None:
        _webhook_store = WebhookStore()
    if not _retry_worker_started:
        _retry_worker_started = True
        _start_retry_worker()
    return _webhook_store


def _start_retry_worker() -> None:
    """Start background retry worker if not already running."""
    import asyncio as _asyncio

    async def _retry_loop() -> None:
        """Background loop that processes pending webhook retries."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                store = get_webhook_store()
                await store.process_retry_queue()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug("Webhook retry worker error: %s", e)

    try:
        loop = _asyncio.get_running_loop()
        loop.create_task(_retry_loop())
    except RuntimeError:
        # No event loop running — worker will start when one is
        pass


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
    payload: dict[str, Any],
    sync: bool = False,
) -> list[WebhookDelivery]:
    """Send notification to all matching webhooks."""
    if event not in TRAINING_EVENTS:
        logger.warning("Unknown training event: %s", event, extra={"tag": "TRAIN"})
        return []

    store = get_webhook_store()
    matching_webhooks = store.list(event_filter=event)

    if not matching_webhooks:
        logger.debug("No webhooks registered for event: %s", event)
        return []

    logger.info("Sending %s to %d webhook(s)", event, len(matching_webhooks), extra={"tag": "TRAIN"})

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
