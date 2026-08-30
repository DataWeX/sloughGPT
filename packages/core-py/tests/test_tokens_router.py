"""Tests for tokens router - requires FastAPI."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from domains.billing.token_service import get_token_billing_service, Tier
    from apps.api.server.routers.tokens import router
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

if not HAS_FASTAPI:
    pytest.skip("FastAPI not installed", allow_module_level=True)


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as c:
        yield c


class TestGetBalance:
    def test_get_balance(self, client):
        response = client.get("/tokens/balance", headers={"X-User-Id": "test-user"})
        assert response.status_code == 200
        data = response.json()
        assert "balance" in data
        assert "tier" in data

    def test_get_balance_creates_account(self, client):
        response = client.get("/tokens/balance", headers={"X-User-Id": "new-user"})
        assert response.status_code == 200
        data = response.json()
        assert data["balance"] == 500  # Default free tier


class TestUsageSummary:
    def test_usage_summary(self, client):
        response = client.get("/tokens/usage/summary", headers={"X-User-Id": "test-user"})
        assert response.status_code == 200
        data = response.json()
        assert "totalRequests" in data
        assert "totalTokens" in data
        assert "totalCost" in data


class TestUsageHistory:
    def test_usage_history(self, client):
        response = client.get("/tokens/usage/history", headers={"X-User-Id": "test-user"})
        assert response.status_code == 200
        data = response.json()
        assert "records" in data

    def test_usage_history_with_limit(self, client):
        response = client.get("/tokens/usage/history?limit=10", headers={"X-User-Id": "test-user"})
        assert response.status_code == 200


class TestTopUp:
    def test_topup_success(self, client):
        response = client.post("/tokens/topup", json={"amount": 100}, headers={"X-User-Id": "topup-user"})
        assert response.status_code == 200
        data = response.json()
        assert data["balance"] >= 100

    def test_topup_invalid_amount(self, client):
        response = client.post("/tokens/topup", json={"amount": -10}, headers={"X-User-Id": "test-user"})
        assert response.status_code == 400

    def test_topup_exceeds_max(self, client):
        response = client.post("/tokens/topup", json={"amount": 2000000}, headers={"X-User-Id": "test-user"})
        assert response.status_code == 400


class TestUpgrade:
    def test_upgrade_success(self, client):
        response = client.post("/tokens/upgrade", json={"tier": "pro"}, headers={"X-User-Id": "upgrade-user"})
        assert response.status_code == 200
        data = response.json()
        assert data["tier"] == "pro"

    def test_upgrade_invalid_tier(self, client):
        response = client.post("/tokens/upgrade", json={"tier": "invalid"}, headers={"X-User-Id": "test-user"})
        assert response.status_code == 400


class TestCheckTokens:
    def test_check_can_afford(self, client):
        response = client.post("/tokens/check", json={
            "model": "slonet",
            "input_tokens": 10,
            "output_tokens": 20,
        }, headers={"X-User-Id": "check-user"})
        assert response.status_code == 200
        data = response.json()
        assert "canAfford" in data

    def test_check_response_fields(self, client):
        response = client.post("/tokens/check", json={
            "model": "slonet",
            "input_tokens": 10,
            "output_tokens": 20,
        }, headers={"X-User-Id": "check-user2"})
        assert response.status_code == 200
        data = response.json()
        assert "totalTokens" in data
        assert "balance" in data
        assert "dailyRemaining" in data
        assert "monthlyRemaining" in data
