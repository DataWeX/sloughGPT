"""Tests for the ⑫-P3 webhook consolidation (kanban card ad9ef322).

``infrastructure/startup_webhooks`` is now a facade over
``training.webhooks.WebhookStore`` — one delivery engine, two views:

1. ``get_status()`` must produce the exact pre-refactor
   ``health.startup_webhooks`` JSON shape (oracle captured from the original
   in-memory manager before the rewrite).
2. ``emit`` must delegate to the store's ``notify_event`` (fire-and-forget).
3. The shared event registry must accept ``startup.*`` at both the emit gate
   and the REST registration gate — startup events are registerable for real
   now instead of being a dead in-memory list.
4. A degraded store must yield the historical empty shape, never an exception.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_DIR = REPO_ROOT / "apps/api/server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from infrastructure.startup_webhooks import (  # noqa: E402
    WebhookConfig,
    WebhookEvent,
    WebhookManager,
    get_webhook_manager,
    register_webhook,
)
from training.webhooks import (  # noqa: E402
    WebhookDelivery,
    WebhookStore,
    known_webhook_events,
    notify_event,
)

from apps.api.server.infrastructure.exception_handlers import (  # noqa: E402
    register_all_handlers,
)
from apps.api.server.training.webhook_endpoints import (  # noqa: E402
    router as webhook_router,
)

# Shapes captured from the pre-refactor WebhookManager.get_status() output.
ORACLE_TOP_KEYS = {
    "webhooks",
    "recent_deliveries",
    "total_deliveries",
    "successful_deliveries",
    "failed_deliveries",
}
ORACLE_HOOK_KEYS = {"url", "events", "headers", "timeout", "enabled", "retry_count"}
ORACLE_DELIVERY_KEYS = {
    "url",
    "event",
    "success",
    "status_code",
    "error",
    "timestamp",
    "duration_ms",
}
EMPTY_SHAPE = {
    "webhooks": [],
    "recent_deliveries": [],
    "total_deliveries": 0,
    "successful_deliveries": 0,
    "failed_deliveries": 0,
}


def _seeded_store(tmp_path: Path) -> WebhookStore:
    """A store with one startup webhook and two delivery records."""
    store = WebhookStore(str(tmp_path / "webhooks.db"))
    store.register(
        url="https://hooks.example.com/startup",
        events=["startup.complete", "startup.failed"],
        headers={"X-A": "b"},
        timeout=7.5,
        retry_count=2,
    )
    [hook] = store.list()
    store._add_delivery(
        WebhookDelivery(
            id="d1",
            webhook_id=hook.id,
            event="startup.complete",
            payload={},
            status_code=200,
            success=True,
            duration_ms=12.5,
        )
    )
    store._add_delivery(
        WebhookDelivery(
            id="d2",
            webhook_id=hook.id,
            event="startup.failed",
            payload={},
            status_code=503,
            success=False,
            error="HTTP 503",
            duration_ms=1000.0,
        )
    )
    return store


class TestGetStatusShape:
    """health.startup_webhooks must be byte-shape-identical pre/post refactor."""

    def test_shape_pinned_against_oracle(self, tmp_path):
        store = _seeded_store(tmp_path)
        with patch("training.webhooks.get_webhook_store", return_value=store):
            status = WebhookManager().get_status()

        assert set(status) == ORACLE_TOP_KEYS

        [hook] = status["webhooks"]
        assert set(hook) == ORACLE_HOOK_KEYS
        assert hook == {
            "url": "https://hooks.example.com/startup",
            "events": ["startup.complete", "startup.failed"],
            "headers": {"X-A": "b"},
            "timeout": 7.5,
            "enabled": True,
            "retry_count": 2,
        }

        assert len(status["recent_deliveries"]) == 2
        first, second = status["recent_deliveries"]
        assert set(first) == ORACLE_DELIVERY_KEYS
        assert first["url"] == "https://hooks.example.com/startup"
        assert first["event"] == "startup.complete"
        assert first["success"] is True
        assert first["status_code"] == 200
        assert first["error"] is None
        assert isinstance(first["timestamp"], float)
        assert first["duration_ms"] == 12.5

        assert second["event"] == "startup.failed"
        assert second["success"] is False
        assert second["status_code"] == 503
        assert second["error"] == "HTTP 503"

        assert status["total_deliveries"] == 2
        assert status["successful_deliveries"] == 1
        assert status["failed_deliveries"] == 1

    def test_empty_store_yields_empty_shape(self, tmp_path):
        store = WebhookStore(str(tmp_path / "webhooks.db"))
        with patch("training.webhooks.get_webhook_store", return_value=store):
            status = WebhookManager().get_status()
        assert status == EMPTY_SHAPE

    def test_degraded_store_yields_empty_shape_not_exception(self, monkeypatch, tmp_path):
        """A dead MogDB must degrade to the empty shape, never raise."""
        store = WebhookStore(str(tmp_path / "webhooks.db"))
        monkeypatch.setattr(WebhookStore, "is_available", property(lambda self: False))
        with patch("training.webhooks.get_webhook_store", return_value=store):
            status = get_webhook_manager().get_status()
        assert status == EMPTY_SHAPE


class TestEmitDelegation:
    """emit must go through the one engine (notify_event), fire-and-forget."""

    def test_emit_delegates_to_notify_event(self):
        mock_notify = AsyncMock(return_value=[])

        async def run():
            with patch("training.webhooks.notify_event", new=mock_notify):
                await get_webhook_manager().emit(WebhookEvent.STARTUP_COMPLETE, {"ok": True})

        asyncio.run(run())
        mock_notify.assert_awaited_once_with("startup.complete", {"ok": True})

    def test_emit_with_no_data_sends_empty_payload(self):
        mock_notify = AsyncMock(return_value=[])

        async def run():
            with patch("training.webhooks.notify_event", new=mock_notify):
                await get_webhook_manager().emit(WebhookEvent.STARTUP_FAILED)

        asyncio.run(run())
        mock_notify.assert_awaited_once_with("startup.failed", {})


class TestEventRegistry:
    """One registry: training + startup events accepted at every gate."""

    def test_known_events_include_all_startup_and_training_events(self):
        known = known_webhook_events()
        assert set(known) >= {e.value for e in WebhookEvent}
        assert "training.completed" in known
        assert len(known) == len(set(known))  # no duplicates

    def test_unknown_event_rejected_before_store_access(self):
        async def run():
            with patch("training.webhooks.get_webhook_store") as mock_store:
                result = await notify_event("nope.nope", {})
                return result, mock_store

        result, mock_store = asyncio.run(run())
        assert result == []
        mock_store.assert_not_called()

    def test_startup_event_passes_gate(self):
        async def run():
            with patch("training.webhooks.get_webhook_store") as mock_store:
                mock_store.return_value.list.return_value = []
                return await notify_event("startup.complete", {})

        assert asyncio.run(run()) == []


class TestRegisterMapping:
    """register/unregister map WebhookConfig onto store fields (no local state)."""

    def test_register_webhook_maps_config_onto_store(self, tmp_path):
        store = WebhookStore(str(tmp_path / "webhooks.db"))
        with patch("training.webhooks.get_webhook_store", return_value=store):
            config = register_webhook(
                "https://hooks.example/y",
                events=[WebhookEvent.STARTUP_COMPLETE],
                timeout=5.0,
                retry_count=4,
            )
            [hook] = store.list()

        assert isinstance(config, WebhookConfig)
        assert hook.url == "https://hooks.example/y"
        assert hook.events == ["startup.complete"]
        assert hook.timeout == 5.0
        assert hook.retry_count == 4

    def test_unregister_by_url(self, tmp_path):
        store = WebhookStore(str(tmp_path / "webhooks.db"))
        manager = WebhookManager()
        with patch("training.webhooks.get_webhook_store", return_value=store):
            manager.register(WebhookConfig(url="https://hooks.example/z"))
            manager.unregister("https://hooks.example/z")
            assert store.list() == []


class TestRestGate:
    """POST /training/webhooks must accept startup.* events (additive)."""

    @pytest.fixture
    def client(self, tmp_path):
        store = WebhookStore(str(tmp_path / "webhooks.db"))
        app = FastAPI()
        register_all_handlers(app)
        app.include_router(webhook_router)
        with patch(
            "apps.api.server.training.webhook_endpoints.get_webhook_store",
            return_value=store,
        ):
            yield TestClient(app, raise_server_exceptions=False)

    def test_register_startup_event_accepted(self, client):
        # main's api-standardized contract: url/events as query params
        # (events is a JSON-encoded string), not a request body.
        resp = client.post(
            "/training/webhooks",
            params={
                "url": "https://hooks.example/startup",
                "events": '["startup.complete"]',
            },
        )
        assert resp.status_code == 200, resp.text
        assert "startup.complete" in resp.json()["events"]

    def test_available_events_grown_additively(self, client):
        resp = client.get("/training/webhooks")
        assert resp.status_code == 200
        available = resp.json()["available_events"]
        assert "startup.complete" in available
        assert "training.completed" in available  # old entries never removed

    def test_bogus_event_still_rejected(self, client):
        resp = client.post(
            "/training/webhooks",
            params={
                "url": "https://hooks.example/x",
                "events": '["not.an.event"]',
            },
        )
        assert resp.status_code == 400
