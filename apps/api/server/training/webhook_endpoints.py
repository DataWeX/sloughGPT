"""Webhook notification endpoints for training events."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter
from pydantic import BaseModel
from schemas.common import raise_error

from .webhooks import (
    TRAINING_EVENTS,
    get_webhook_store,
)

logger = logging.getLogger("slo")

router = APIRouter(tags=["training-webhooks"])


@router.get("/training/webhooks")
async def list_webhooks():
    """List all registered webhooks."""
    store = get_webhook_store()
    webhooks = store.list()

    return {
        "webhooks": [
            {
                "id": w.id,
                "url": w.url,
                "events": w.events,
                "description": w.description,
                "is_active": w.is_active,
                "created_at": w.created_at.isoformat(),
            }
            for w in webhooks
        ],
        "available_events": TRAINING_EVENTS,
    }


@router.post("/training/webhooks")
async def register_webhook(
    url: str,
    events: str,
    description: str = "",
    secret: str | None = None,
):
    """Register a new webhook endpoint."""
    try:
        events_list = json.loads(events) if isinstance(events, str) else events
    except json.JSONDecodeError:
        raise_error("Invalid events format. Must be JSON array.", "E_BAD_REQUEST", status_code=400)
    if not url.startswith(("http://", "https://")):
        raise_error("URL must start with http:// or https://", "E_BAD_REQUEST", status_code=400)
    invalid_events = [e for e in events_list if e not in TRAINING_EVENTS]
    if invalid_events:
        raise_error(
            f"Invalid events: {invalid_events}. Available: {TRAINING_EVENTS}",
            "E_BAD_REQUEST",
            status_code=400,
        )
    store = get_webhook_store()
    webhook_id = store.register(
        url=url,
        events=events_list,
        secret=secret,
        description=description,
        headers=None,
    )

    webhook = store.get(webhook_id)

    try:
        from infrastructure.auth import get_audit_logger

        get_audit_logger().log(
            "training.webhook.register",
            resource=url,
            extra={"webhook_id": webhook_id, "events": events_list},
        )
    except Exception as e:
        logger.warning("Audit log failed for webhook registration %s: %s", webhook_id, e)

    return {
        "id": webhook_id,
        "url": url,
        "events": events,
        "secret": webhook.secret if webhook else None,
        "message": "Webhook registered successfully",
    }


# ── Static webhook sub-routes MUST come before /{webhook_id} ──────────


@router.get("/training/webhooks/stats")
async def get_webhook_stats():
    """Get webhook statistics."""
    store = get_webhook_store()
    return store.get_stats()


@router.get("/training/webhooks/retry-queue")
async def get_webhook_retry_queue():
    """Get pending webhook retries."""
    store = get_webhook_store()
    return {"retries": store.get_retry_queue()}


@router.get("/training/webhooks/dead-letters")
async def get_webhook_dead_letters(limit: int = 50):
    """Get dead-lettered webhook deliveries."""
    store = get_webhook_store()
    return {"dead_letters": store.get_dead_letters(limit=limit)}


class TestWebhookRequest(BaseModel):
    url: str


@router.post("/training/webhooks/test")
async def test_webhook(req: TestWebhookRequest):
    """Send a test notification to a URL."""
    store = get_webhook_store()

    webhook_id = store.register(
        url=req.url,
        events=TRAINING_EVENTS,
        description="Temporary test webhook",
    )

    delivery = await store.deliver(
        webhook_id=webhook_id,
        event="training.completed",
        payload={
            "job_id": "test",
            "job_name": "Test Training",
            "status": "completed",
            "message": "This is a test webhook notification",
        },
        retries=1,
    )

    store.unregister(webhook_id)

    return {
        "success": delivery.success,
        "status_code": delivery.status_code,
        "error": delivery.error,
        "response_body": delivery.response_body,
    }


# ── Parameterized webhook routes (after all static sub-routes) ────────


@router.delete("/training/webhooks/{webhook_id}")
async def unregister_webhook(webhook_id: str):
    """Unregister a webhook."""
    store = get_webhook_store()

    if not store.get(webhook_id):
        raise_error("Webhook not found", "E_NOT_FOUND", status_code=404)
    store.unregister(webhook_id)

    try:
        from infrastructure.auth import get_audit_logger

        get_audit_logger().log("training.webhook.delete", resource=webhook_id)
    except Exception as e:
        logger.warning("Audit log failed for webhook deletion %s: %s", webhook_id, e)

    return {"status": "deleted", "webhook_id": webhook_id}


@router.get("/training/webhooks/{webhook_id}")
async def get_webhook(webhook_id: str):
    """Get webhook details (without secret)."""
    store = get_webhook_store()
    webhook = store.get(webhook_id)

    if not webhook:
        raise_error("Webhook not found", "E_NOT_FOUND", status_code=404)
    return {
        "id": webhook.id,
        "url": webhook.url,
        "events": webhook.events,
        "description": webhook.description,
        "is_active": webhook.is_active,
        "created_at": webhook.created_at.isoformat(),
    }


@router.get("/training/webhooks/{webhook_id}/deliveries")
async def get_webhook_deliveries(webhook_id: str, limit: int = 50):
    """Get delivery log for a webhook."""
    store = get_webhook_store()

    if not store.get(webhook_id):
        raise_error("Webhook not found", "E_NOT_FOUND", status_code=404)
    deliveries = store.get_deliveries(webhook_id, limit=limit)

    return {
        "deliveries": [
            {
                "id": d.id,
                "event": d.event,
                "success": d.success,
                "status_code": d.status_code,
                "attempted_at": d.attempted_at.isoformat(),
                "error": d.error,
            }
            for d in deliveries
        ]
    }
