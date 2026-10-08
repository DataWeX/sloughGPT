"""
Tests for POST /training/webhooks - dual contract (JSON body + query params),
natural-key dedup flag, and input normalization.

Isolated from the production webhook store via a tmp_path WebhookStore.
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from apps.api.server.infrastructure.exception_handlers import register_all_handlers
from apps.api.server.training.webhook_endpoints import router as webhook_router
from apps.api.server.training.webhooks import WebhookStore

URL = "https://hooks.example/journey"
EVENTS = ["training.completed", "training.failed"]


@pytest.fixture
def store(tmp_path):
    return WebhookStore(str(tmp_path / "webhooks.db"))


@pytest.fixture
def client(store):
    app = FastAPI()
    register_all_handlers(app)
    app.include_router(webhook_router)
    with patch(
        "apps.api.server.training.webhook_endpoints.get_webhook_store",
        return_value=store,
    ):
        yield TestClient(app, raise_server_exceptions=False)


class TestRegisterContract:
    """POST /training/webhooks"""

    def test_json_body_register(self, client):
        resp = client.post("/training/webhooks", json={"url": URL, "events": EVENTS})
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"]
        assert body["url"] == URL
        assert body["events"] == EVENTS
        assert body["deduplicated"] is False
        assert body["message"] == "Webhook registered successfully"

    def test_query_params_register(self, client):
        resp = client.post(
            "/training/webhooks",
            params={"url": URL, "events": '["training.completed"]'},
        )
        assert resp.status_code == 200
        assert resp.json()["deduplicated"] is False

    def test_missing_everything_is_400(self, client):
        resp = client.post("/training/webhooks")
        assert resp.status_code == 400

    def test_invalid_scheme_is_400(self, client):
        resp = client.post("/training/webhooks", json={"url": "ftp://x", "events": EVENTS})
        assert resp.status_code == 400

    def test_invalid_event_is_400(self, client):
        resp = client.post("/training/webhooks", json={"url": URL, "events": ["training.nope"]})
        assert resp.status_code == 400

    def test_events_must_be_list_of_strings(self, client):
        resp = client.post("/training/webhooks", json={"url": URL, "events": "training.started"})
        assert resp.status_code == 422


class TestRegisterDedup:
    """Idempotent create on (url, event set)."""

    def test_repeat_body_returns_same_id_and_flag(self, client, store):
        first = client.post("/training/webhooks", json={"url": URL, "events": EVENTS}).json()
        second = client.post("/training/webhooks", json={"url": URL, "events": EVENTS}).json()

        assert second["id"] == first["id"]
        assert second["deduplicated"] is True
        assert second["message"] == "Webhook already registered"
        assert len(store.list()) == 1

    def test_body_then_query_dedupes(self, client, store):
        first = client.post("/training/webhooks", json={"url": URL, "events": EVENTS}).json()
        second = client.post(
            "/training/webhooks",
            params={"url": URL, "events": '["training.completed","training.failed"]'},
        ).json()

        assert second["id"] == first["id"]
        assert second["deduplicated"] is True
        assert len(store.list()) == 1

    def test_duplicate_and_whitespace_events_dedupe(self, client):
        first = client.post(
            "/training/webhooks",
            json={"url": URL, "events": ["training.completed", "training.failed"]},
        ).json()
        second = client.post(
            "/training/webhooks",
            json={
                "url": f"  {URL}  ",
                "events": ["training.failed", "training.completed ", "training.completed"],
            },
        ).json()

        assert second["id"] == first["id"]
        assert second["events"] == ["training.failed", "training.completed"]
        assert second["deduplicated"] is True

    def test_different_events_not_deduped(self, client, store):
        a = client.post(
            "/training/webhooks", json={"url": URL, "events": ["training.started"]}
        ).json()
        b = client.post(
            "/training/webhooks", json={"url": URL, "events": ["training.completed"]}
        ).json()

        assert a["id"] != b["id"]
        assert b["deduplicated"] is False
        assert len(store.list()) == 2
