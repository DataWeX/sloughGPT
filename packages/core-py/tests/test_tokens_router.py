"""Tests for tokens router - requires FastAPI."""

import os
import sys
from uuid import uuid4

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from infrastructure.auth import require_auth_if_enabled

    from apps.api.server.routers.tokens import router
    from services.billing._internal.token_service import (  # noqa: F401
        Tier,
        get_token_billing_service,
    )

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

if not HAS_FASTAPI:
    pytest.skip("FastAPI not installed", allow_module_level=True)


@pytest.fixture
def app():
    application = FastAPI()
    application.include_router(router)
    # Give this test a PRIVATE billing account. The router derives the
    # account key from audit_user(auth_user), so without an override every
    # test would share one "anonymous" account and real-service state would
    # leak between them (test_topup_success would fund
    # test_get_balance_creates_account, breaking its balance == 500 assert).
    user_id = f"user-{uuid4().hex[:8]}"
    application.dependency_overrides[require_auth_if_enabled] = lambda: {
        "id": user_id,
        "sub": user_id,
    }
    return application


@pytest.fixture
def client(app):
    with TestClient(app) as c:
        yield c


class TestAuthDisabled:
    """The default configuration: SLO_AUTH_REQUIRED is unset, so
    require_auth_if_enabled returns None rather than a payload."""

    def test_balance_falls_back_to_anonymous(self, app, client):
        # Regression: this used to do auth_user["id"] on None and 500 on
        # EVERY /tokens/* route with auth off (the default).
        app.dependency_overrides.clear()

        response = client.get("/tokens/balance")

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["userId"] == "anonymous"


class TestGetBalance:
    def test_get_balance(self, client):
        response = client.get("/tokens/balance", headers={"X-User-Id": "test-user"})
        assert response.status_code == 200
        data = response.json()["data"]
        assert "balance" in data
        assert "tier" in data

    def test_get_balance_creates_account(self, client):
        response = client.get("/tokens/balance", headers={"X-User-Id": "new-user"})
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["balance"] == 500  # Default free tier


class TestUsageSummary:
    def test_usage_summary(self, client):
        response = client.get("/tokens/usage/summary", headers={"X-User-Id": "test-user"})
        assert response.status_code == 200
        data = response.json()["data"]
        assert "totalRequests" in data
        assert "totalTokens" in data
        assert "totalCost" in data


class TestUsageHistory:
    def test_usage_history(self, client):
        response = client.get("/tokens/usage/history", headers={"X-User-Id": "test-user"})
        assert response.status_code == 200
        data = response.json()["data"]
        assert "records" in data

    def test_usage_history_with_limit(self, client):
        response = client.get("/tokens/usage/history?limit=10", headers={"X-User-Id": "test-user"})
        assert response.status_code == 200


class TestTopUp:
    def test_topup_success(self, client):
        response = client.post(
            "/tokens/topup", json={"amount": 100}, headers={"X-User-Id": "topup-user"}
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["balance"] >= 100

    def test_topup_invalid_amount(self, client):
        response = client.post(
            "/tokens/topup", json={"amount": -10}, headers={"X-User-Id": "test-user"}
        )
        assert response.status_code == 422

    def test_topup_exceeds_max(self, client):
        response = client.post(
            "/tokens/topup", json={"amount": 2000000}, headers={"X-User-Id": "test-user"}
        )
        assert response.status_code == 422


class TestUpgrade:
    def test_upgrade_success(self, client):
        response = client.post(
            "/tokens/upgrade", json={"tier": "pro"}, headers={"X-User-Id": "upgrade-user"}
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert data["tier"] == "pro"

    def test_upgrade_invalid_tier(self, client):
        response = client.post(
            "/tokens/upgrade", json={"tier": "invalid"}, headers={"X-User-Id": "test-user"}
        )
        assert response.status_code == 422


class TestCheckTokens:
    def test_check_can_afford(self, client):
        response = client.post(
            "/tokens/check",
            json={
                "model": "slonet",
                "input_tokens": 10,
                "output_tokens": 20,
            },
            headers={"X-User-Id": "check-user"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert "canAfford" in data

    def test_check_response_fields(self, client):
        response = client.post(
            "/tokens/check",
            json={
                "model": "slonet",
                "input_tokens": 10,
                "output_tokens": 20,
            },
            headers={"X-User-Id": "check-user2"},
        )
        assert response.status_code == 200
        data = response.json()["data"]
        assert "totalTokens" in data
        assert "balance" in data
        assert "dailyRemaining" in data
        assert "monthlyRemaining" in data
