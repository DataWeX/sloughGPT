"""Webhook registration rejects private/metadata destinations (card 20261108_142919).

"WHERE WE FETCH TO" half of the card: webhook URLs are caller-chosen
destinations fetched server-side — registration is the single seam where
they enter the system, so ``validate_url_not_private`` (the existing SSRF
helper used by routers/datasets.py) gates them there. Registration-time
validation only; delivery-time re-resolution (TOCTOU/DNS rebinding) is
explicitly out of the card's scope.
"""

import pytest
from test_support import get_test_client
from training.webhooks import WebhookStore

from domain.infrastructure._internal.errors import AppError

_EVENTS = ["training.completed"]


def _store(tmp_path) -> WebhookStore:
    return WebhookStore(str(tmp_path / "webhooks.db"))


class TestStoreRegistrationGuard:
    def test_rejects_metadata_service_ip(self, tmp_path):
        """169.254.169.254 — the cloud metadata service — must never be a
        webhook target (the card's required behavior)."""
        store = _store(tmp_path)
        with pytest.raises(AppError) as ei:
            store.register("http://169.254.169.254/latest/meta-data", list(_EVENTS))
        assert ei.value.code == "E_BAD_REQUEST"
        assert ei.value.http_status == 400
        assert store.list() == []  # nothing persisted

    def test_rejects_loopback_ip(self, tmp_path):
        store = _store(tmp_path)
        with pytest.raises(AppError):
            store.register("http://127.0.0.1:9000/hook", list(_EVENTS))
        assert store.list() == []

    def test_rejects_localhost_name(self, tmp_path):
        store = _store(tmp_path)
        with pytest.raises(AppError):
            store.register("http://localhost:9000/hook", list(_EVENTS))

    def test_rejects_rfc1918_address(self, tmp_path):
        store = _store(tmp_path)
        with pytest.raises(AppError):
            store.register("http://10.0.0.5/hook", list(_EVENTS))

    def test_accepts_public_url(self, tmp_path):
        """Public destinations still register (same DNS precedent as the
        datasets import tests, which resolve github.com at test time)."""
        store = _store(tmp_path)
        wid = store.register("https://example.com/hook", list(_EVENTS))
        assert store.get(wid) is not None


class TestEndpointRegistrationGuard:
    """The HTTP projection: POST /training/webhooks → 400 envelope."""

    def test_endpoint_rejects_metadata_ip(self):
        client = get_test_client()
        before = len(client.get("/training/webhooks").json()["webhooks"])
        resp = client.post(
            "/training/webhooks",
            params={
                "url": "http://169.254.169.254/latest/meta-data",
                "events": '["training.completed"]',
            },
        )
        assert resp.status_code == 400
        body = resp.json()
        assert body["code"] == "E_BAD_REQUEST"
        assert "private" in body["error"]
        after = len(client.get("/training/webhooks").json()["webhooks"])
        assert after == before  # rejected before any insert

    def test_endpoint_rejects_loopback_ip(self):
        client = get_test_client()
        resp = client.post(
            "/training/webhooks",
            params={"url": "http://127.0.0.1:9000/hook", "events": '["training.completed"]'},
        )
        assert resp.status_code == 400
        assert resp.json()["code"] == "E_BAD_REQUEST"
