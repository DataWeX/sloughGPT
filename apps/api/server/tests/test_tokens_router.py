"""Tests for the Tokens billing API router."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from infrastructure.auth import require_auth_if_enabled
from infrastructure.exception_handlers import register_app_error_handler

_AUTH_USER = {"id": "user1", "sub": "user1", "tenant_id": "t1"}


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_service():
    return MagicMock()


@pytest.fixture
def app(mock_service):
    from routers.tokens import router

    _app = FastAPI()
    register_app_error_handler(_app)
    _app.include_router(router)
    _app.dependency_overrides[require_auth_if_enabled] = lambda: _AUTH_USER
    with patch("routers.tokens.get_token_billing_service", return_value=mock_service):
        yield _app, mock_service


@pytest.fixture
def client(app):
    return TestClient(app[0])


# ── Balance ──────────────────────────────────────────────────────────────────

class TestBalance:
    def test_get_balance(self, client, mock_service):
        from domains.billing.token_service import Tier, TokenAccount
        account = TokenAccount(user_id="user1", balance=5000, tier=Tier.PRO)
        mock_service.get_balance.return_value = account
        resp = client.get("/tokens/balance")
        assert resp.status_code == 200
        assert resp.json()["data"]["balance"] == 5000

    def test_get_balance_error(self, client, mock_service):
        mock_service.get_balance.side_effect = RuntimeError("DB error")
        resp = client.get("/tokens/balance")
        assert resp.status_code == 500


# ── Usage summary ────────────────────────────────────────────────────────────

class TestUsageSummary:
    def test_get_usage_summary(self, client, mock_service):
        mock_service.get_usage_summary.return_value = {"totalTokens": 1000}
        resp = client.get("/tokens/usage/summary")
        assert resp.status_code == 200
        assert resp.json()["data"]["totalTokens"] == 1000


# ── Usage history ────────────────────────────────────────────────────────────

class TestUsageHistory:
    def test_get_usage_history(self, client, mock_service):
        from domains.billing.token_service import UsageRecord
        record = UsageRecord(
            id="r1", user_id="user1", model="gpt-4",
            input_tokens=100, output_tokens=200, total_tokens=300,
            cost=0.015, timestamp=1000.0,
        )
        mock_service.get_usage_history.return_value = [record]
        resp = client.get("/tokens/usage/history")
        assert resp.status_code == 200
        records = resp.json()["data"]["records"]
        assert len(records) == 1
        assert records[0]["totalTokens"] == 300

    def test_get_usage_history_with_params(self, client, mock_service):
        mock_service.get_usage_history.return_value = []
        resp = client.get("/tokens/usage/history?limit=10&offset=5")
        assert resp.status_code == 200
        mock_service.get_usage_history.assert_called_once_with("user1", limit=10, offset=5)


# ── Topup ────────────────────────────────────────────────────────────────────

class TestTopup:
    def test_topup_success(self, client, mock_service):
        from domains.billing.token_service import Tier, TokenAccount
        account = TokenAccount(user_id="user1", balance=6000, tier=Tier.FREE)
        mock_service.add_credits.return_value = account
        resp = client.post("/tokens/topup", json={"amount": 1000})
        assert resp.status_code == 200
        assert resp.json()["data"]["balance"] == 6000

    def test_topup_invalid_amount(self, client):
        resp = client.post("/tokens/topup", json={"amount": -1})
        assert resp.status_code == 422


# ── Upgrade ──────────────────────────────────────────────────────────────────

class TestUpgrade:
    def test_upgrade_success(self, client, mock_service):
        from domains.billing.token_service import Tier, TokenAccount
        account = TokenAccount(user_id="user1", balance=5000, tier=Tier.PRO)
        mock_service.upgrade_tier.return_value = account
        resp = client.post("/tokens/upgrade", json={"tier": "pro"})
        assert resp.status_code == 200
        assert resp.json()["data"]["tier"] == "pro"

    def test_upgrade_invalid_tier(self, client):
        resp = client.post("/tokens/upgrade", json={"tier": "basic"})
        assert resp.status_code == 400


# ── Check ────────────────────────────────────────────────────────────────────

class TestCheck:
    def test_check_can_afford(self, client, mock_service):
        from domains.billing.token_service import Tier, TokenAccount
        account = TokenAccount(user_id="user1", balance=10000, tier=Tier.PRO)
        mock_service.get_balance.return_value = account
        resp = client.post("/tokens/check", json={
            "model": "gpt-4",
            "input_tokens": 100,
            "output_tokens": 50,
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["canAfford"] is True
        assert data["totalTokens"] == 150

    def test_check_cannot_afford(self, client, mock_service):
        from domains.billing.token_service import Tier, TokenAccount
        account = TokenAccount(user_id="user1", balance=10, tier=Tier.FREE)
        mock_service.get_balance.return_value = account
        resp = client.post("/tokens/check", json={
            "model": "gpt-4",
            "input_tokens": 100,
            "output_tokens": 50,
        })
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["canAfford"] is False
        assert data["balance"] == 10
