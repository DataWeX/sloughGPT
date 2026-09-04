"""Tests for tokens router — token billing API endpoints."""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from fastapi import Depends


@pytest.fixture
def mock_token_service():
    """Create a mock token billing service."""
    mock = MagicMock()
    mock.get_balance.return_value = MagicMock(
        to_dict=lambda: {
            "id": "user-1",
            "tier": "free",
            "balance": 1000,
            "daily_limit": 10000,
            "daily_used": 5000,
            "monthly_limit": 100000,
            "monthly_used": 50000,
        },
        can_afford=lambda n: True,
        balance=1000,
        daily_limit=10000,
        daily_used=5000,
        monthly_limit=100000,
        monthly_used=50000,
    )
    mock.get_usage_summary.return_value = {
        "total_tokens": 55000,
        "by_model": {"gpt-4": 30000, "gpt-3.5": 25000},
    }
    mock.get_usage_history.return_value = [
        MagicMock(to_dict=lambda: {"model": "gpt-4", "tokens": 1000, "timestamp": "2026-09-01T10:00:00Z"}),
    ]
    mock.add_credits.return_value = MagicMock(
        to_dict=lambda: {
            "id": "user-1",
            "tier": "free",
            "balance": 2000,
            "daily_limit": 10000,
            "daily_used": 5000,
            "monthly_limit": 100000,
            "monthly_used": 50000,
        }
    )
    mock.upgrade_tier.return_value = MagicMock(
        to_dict=lambda: {
            "id": "user-1",
            "tier": "pro",
            "balance": 1000,
            "daily_limit": 100000,
            "daily_used": 0,
            "monthly_limit": 1000000,
            "monthly_used": 0,
        }
    )
    return mock


@pytest.fixture
def client(mock_token_service):
    from fastapi import FastAPI
    from infrastructure.exception_handlers import register_app_error_handler
    from routers.tokens import router as tokens_router
    
    app = FastAPI()
    app.include_router(tokens_router)
    register_app_error_handler(app)
    
    # Mock auth dependency - need to override the actual dependency function
    async def mock_auth():
        return {"id": "user-1", "name": "Test User"}
    
    # Override the dependency
    from infrastructure import auth
    app.dependency_overrides[auth.require_auth_if_enabled] = mock_auth
    
    with patch("routers.tokens.get_token_billing_service", return_value=mock_token_service):
        yield TestClient(app)


class TestTokenBalance:
    def test_get_balance_returns_200(self, client):
        resp = client.get("/tokens/balance")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "user-1"
        assert data["balance"] == 1000
        assert data["tier"] == "free"

    def test_get_balance_has_limits(self, client):
        resp = client.get("/tokens/balance")
        data = resp.json()
        assert "daily_limit" in data
        assert "monthly_limit" in data


class TestTokenUsage:
    def test_get_usage_summary(self, client):
        resp = client.get("/tokens/usage/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_tokens" in data
        assert "by_model" in data

    def test_get_usage_history(self, client):
        resp = client.get("/tokens/usage/history")
        assert resp.status_code == 200
        data = resp.json()
        assert "records" in data
        assert len(data["records"]) == 1

    def test_get_usage_history_with_params(self, client):
        resp = client.get("/tokens/usage/history?limit=10&offset=5")
        assert resp.status_code == 200


class TestTokenTopup:
    def test_topup_credits(self, client):
        resp = client.post("/tokens/topup", json={"amount": 1000})
        assert resp.status_code == 200
        data = resp.json()
        assert data["balance"] == 2000

    def test_topup_invalid_amount(self, client):
        resp = client.post("/tokens/topup", json={"amount": -100})
        assert resp.status_code == 422  # Validation error


class TestTokenUpgrade:
    def test_upgrade_tier(self, client):
        resp = client.post("/tokens/upgrade", json={"tier": "pro"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["tier"] == "pro"

    def test_upgrade_invalid_tier(self, client):
        resp = client.post("/tokens/upgrade", json={"tier": "invalid"})
        assert resp.status_code == 422  # Pydantic validation error


class TestTokenCheck:
    def test_check_can_afford(self, client):
        resp = client.post("/tokens/check", json={
            "model": "gpt-4",
            "input_tokens": 100,
            "output_tokens": 200,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["canAfford"] is True
        assert data["totalTokens"] == 300

    def test_check_response_structure(self, client):
        resp = client.post("/tokens/check", json={
            "model": "gpt-4",
            "input_tokens": 100,
            "output_tokens": 200,
        })
        data = resp.json()
        assert "canAfford" in data
        assert "totalTokens" in data
        assert "balance" in data
        assert "dailyRemaining" in data
        assert "monthlyRemaining" in data
