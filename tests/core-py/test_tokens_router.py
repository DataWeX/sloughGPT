"""Tests for tokens router - requires FastAPI."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from fastapi import FastAPI, Request
    from fastapi.testclient import TestClient
    from infrastructure.auth import require_auth_if_enabled

    from apps.api.server.routers.tokens import router
    from domain.billing._internal.token_service import Tier, get_token_billing_service  # noqa: F401

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

if not HAS_FASTAPI:
    pytest.skip("FastAPI not installed", allow_module_level=True)


@pytest.fixture
def client():
    import domain.billing._internal.token_service as billing

    # Fresh singleton per test — process-wide service would leak balances.
    billing._token_billing_service = None

    app = FastAPI()
    app.include_router(router)

    from infrastructure.exception_handlers import register_app_error_handler

    register_app_error_handler(app)

    def _fake_auth(request: Request) -> dict:
        return {"id": request.headers.get("X-User-Id", "anonymous")}

    app.dependency_overrides[require_auth_if_enabled] = _fake_auth
    with TestClient(app) as c:
        yield c


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
        # Field(..., ge=1) → pydantic RequestValidationError → 422
        response = client.post(
            "/tokens/topup", json={"amount": -10}, headers={"X-User-Id": "test-user"}
        )
        assert response.status_code == 422

    def test_topup_exceeds_max(self, client):
        # Field(..., le=1_000_000) → pydantic RequestValidationError → 422
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
        # "basic" passes the pydantic pattern but is not a Tier enum value
        # → handler raise_error(..., status_code=400)
        response = client.post(
            "/tokens/upgrade", json={"tier": "basic"}, headers={"X-User-Id": "test-user"}
        )
        assert response.status_code == 400


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
