"""Webhook notification endpoints for training events.

Descriptor-projected (card 20261104_151036): nine ``ToolSpec`` contracts →
``create_router`` projections instead of hand-laid decorators. All nine
handlers return bare dicts and the module declares no auth dependency, so
every route is ``envelope=False`` (raw wire) with ``auth_scope="public"``
(brief edge 3 — the hand-built ``Depends`` chain was empty everywhere).
``register_webhook``'s flat scalars ride ``input="query"``: the hand-built
POST published them as query members, not a JSON body. The wiring at the
bottom keeps the hand-built registration order — static sub-routes before
the ``/{webhook_id}`` pair — so ``stats``/``retry-queue``/``dead-letters``/
``test`` can never be shadowed by the parameterized route.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter
from infrastructure.contract import RouteSpec, create_router, params_from_model
from pydantic import BaseModel
from schemas.common import raise_error

from domain.agents import ToolParam, ToolSpec

from .webhooks import (
    TRAINING_EVENTS,
    get_webhook_store,
    known_webhook_events,
)

logger = logging.getLogger("slo")

router = APIRouter(tags=["training-webhooks"])


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
        "available_events": known_webhook_events(),
    }


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
    known_events = known_webhook_events()
    invalid_events = [e for e in events_list if e not in known_events]
    if invalid_events:
        raise_error(
            f"Invalid events: {invalid_events}. Available: {known_events}",
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


async def get_webhook_stats():
    """Get webhook statistics."""
    store = get_webhook_store()
    return store.get_stats()


async def get_webhook_retry_queue():
    """Get pending webhook retries."""
    store = get_webhook_store()
    return {"retries": store.get_retry_queue()}


async def get_webhook_dead_letters(limit: int = 50):
    """Get dead-lettered webhook deliveries."""
    store = get_webhook_store()
    return {"dead_letters": store.get_dead_letters(limit=limit)}


class TestWebhookRequest(BaseModel):
    url: str


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


# ── Descriptor projection: one contract per route (card 20261104_151036) ──
# Registration order is load-bearing and mirrors the hand-built decorators:
# static sub-routes first, the /{webhook_id} pair last.

SPEC_WEBHOOKS_LIST = ToolSpec(
    name="training.webhooks.list",
    description="List all registered webhooks.",
    parameters=[],
    execute=list_webhooks,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_WEBHOOKS_REGISTER = ToolSpec(
    name="training.webhooks.register",
    description="Register a new webhook endpoint.",
    parameters=[
        ToolParam("url", "string", "", True),
        ToolParam("events", "string", "", True),
        ToolParam("description", "string", "", False),
        ToolParam("secret", "string", "", False),
    ],
    execute=register_webhook,
    auth_scope="public",
    idempotent=False,
    version="1",
)

SPEC_WEBHOOKS_STATS = ToolSpec(
    name="training.webhooks.stats",
    description="Get webhook statistics.",
    parameters=[],
    execute=get_webhook_stats,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_WEBHOOKS_RETRY_QUEUE = ToolSpec(
    name="training.webhooks.retry_queue",
    description="Get pending webhook retries.",
    parameters=[],
    execute=get_webhook_retry_queue,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_WEBHOOKS_DEAD_LETTERS = ToolSpec(
    name="training.webhooks.dead_letters",
    description="Get dead-lettered webhook deliveries.",
    parameters=[ToolParam("limit", "integer", "", False)],
    execute=get_webhook_dead_letters,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_WEBHOOKS_TEST = ToolSpec(
    name="training.webhooks.test",
    description="Send a test notification to a URL.",
    parameters=params_from_model(TestWebhookRequest),
    execute=test_webhook,
    auth_scope="public",
    idempotent=False,
    version="1",
)

SPEC_WEBHOOKS_UNREGISTER = ToolSpec(
    name="training.webhooks.unregister",
    description="Unregister a webhook.",
    parameters=[ToolParam("webhook_id", "string", "", True)],
    execute=unregister_webhook,
    auth_scope="public",
    idempotent=False,
    version="1",
)

SPEC_WEBHOOKS_GET = ToolSpec(
    name="training.webhooks.get",
    description="Get webhook details (without secret).",
    parameters=[ToolParam("webhook_id", "string", "", True)],
    execute=get_webhook,
    auth_scope="public",
    idempotent=True,
    version="1",
)

SPEC_WEBHOOKS_DELIVERIES = ToolSpec(
    name="training.webhooks.deliveries",
    description="Get delivery log for a webhook.",
    parameters=[
        ToolParam("webhook_id", "string", "", True),
        ToolParam("limit", "integer", "", False),
    ],
    execute=get_webhook_deliveries,
    auth_scope="public",
    idempotent=True,
    version="1",
)

router.include_router(
    create_router(
        SPEC_WEBHOOKS_LIST,
        RouteSpec(path="/training/webhooks", envelope=False, summary="List Webhooks"),
        list_webhooks,
    )
)
router.include_router(
    create_router(
        SPEC_WEBHOOKS_REGISTER,
        RouteSpec(
            path="/training/webhooks",
            input="query",
            envelope=False,
            summary="Register Webhook",
        ),
        register_webhook,
    )
)
router.include_router(
    create_router(
        SPEC_WEBHOOKS_STATS,
        RouteSpec(path="/training/webhooks/stats", envelope=False, summary="Get Webhook Stats"),
        get_webhook_stats,
    )
)
router.include_router(
    create_router(
        SPEC_WEBHOOKS_RETRY_QUEUE,
        RouteSpec(
            path="/training/webhooks/retry-queue",
            envelope=False,
            summary="Get Webhook Retry Queue",
        ),
        get_webhook_retry_queue,
    )
)
router.include_router(
    create_router(
        SPEC_WEBHOOKS_DEAD_LETTERS,
        RouteSpec(
            path="/training/webhooks/dead-letters",
            envelope=False,
            summary="Get Webhook Dead Letters",
        ),
        get_webhook_dead_letters,
    )
)
router.include_router(
    create_router(
        SPEC_WEBHOOKS_TEST,
        RouteSpec(path="/training/webhooks/test", envelope=False, summary="Test Webhook"),
        test_webhook,
    )
)
router.include_router(
    create_router(
        SPEC_WEBHOOKS_UNREGISTER,
        RouteSpec(
            path="/training/webhooks/{webhook_id}",
            method="DELETE",
            envelope=False,
            summary="Unregister Webhook",
        ),
        unregister_webhook,
    )
)
router.include_router(
    create_router(
        SPEC_WEBHOOKS_GET,
        RouteSpec(
            path="/training/webhooks/{webhook_id}",
            envelope=False,
            summary="Get Webhook",
        ),
        get_webhook,
    )
)
router.include_router(
    create_router(
        SPEC_WEBHOOKS_DELIVERIES,
        RouteSpec(
            path="/training/webhooks/{webhook_id}/deliveries",
            envelope=False,
            summary="Get Webhook Deliveries",
        ),
        get_webhook_deliveries,
    )
)
