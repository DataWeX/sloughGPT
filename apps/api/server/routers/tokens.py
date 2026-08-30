"""
Token billing API routes.

Endpoints:
- GET /tokens/balance - Get user's token balance and limits
- GET /tokens/usage/summary - Get usage summary
- GET /tokens/usage/history - Get usage history
- POST /tokens/topup - Add credits
- POST /tokens/upgrade - Upgrade tier
- POST /tokens/check - Check if user can afford tokens
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from infrastructure.auth import require_auth_if_enabled
from schemas.common import raise_error
from domains.billing.token_service import (
    get_token_billing_service,
    Tier,
)

router = APIRouter(prefix="/tokens", tags=["tokens"])


class TopUpRequest(BaseModel):
    amount: int = Field(..., ge=1, le=1_000_000, description="Number of tokens to add")


class UpgradeRequest(BaseModel):
    tier: str = Field(..., pattern=r'^(free|basic|pro|enterprise)$', description="Target tier")


class CheckRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=200, description="Model identifier")
    input_tokens: int = Field(..., ge=0, le=10_000_000, description="Expected input tokens")
    output_tokens: int = Field(..., ge=0, le=10_000_000, description="Expected output tokens")


@router.get("/balance")
async def get_balance(auth_user: dict = Depends(require_auth_if_enabled)):
    service = get_token_billing_service()
    account = service.get_balance(auth_user["id"])
    return account.to_dict()


@router.get("/usage/summary")
async def get_usage_summary(auth_user: dict = Depends(require_auth_if_enabled)):
    service = get_token_billing_service()
    return service.get_usage_summary(auth_user["id"])


@router.get("/usage/history")
async def get_usage_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    auth_user: dict = Depends(require_auth_if_enabled),
):
    service = get_token_billing_service()
    records = service.get_usage_history(auth_user["id"], limit=limit, offset=offset)
    return {"records": [r.to_dict() for r in records]}


@router.post("/topup")
async def topup_credits(request: TopUpRequest, auth_user: dict = Depends(require_auth_if_enabled)):
    service = get_token_billing_service()
    account = service.add_credits(auth_user["id"], request.amount)
    return account.to_dict()


@router.post("/upgrade")
async def upgrade_tier(request: UpgradeRequest, auth_user: dict = Depends(require_auth_if_enabled)):
    try:
        tier = Tier(request.tier)
    except ValueError:
        raise_error(
            f"Invalid tier. Must be one of: {', '.join(t.value for t in Tier)}",
            "E_VAL_REQUEST",
            status_code=400,
        )

    service = get_token_billing_service()
    account = service.upgrade_tier(auth_user["id"], tier)
    return account.to_dict()


@router.post("/check")
async def check_tokens(request: CheckRequest, auth_user: dict = Depends(require_auth_if_enabled)):
    service = get_token_billing_service()
    account = service.get_balance(auth_user["id"])
    total_tokens = request.input_tokens + request.output_tokens
    can_afford = account.can_afford(total_tokens)
    return {
        "canAfford": can_afford,
        "totalTokens": total_tokens,
        "balance": account.balance,
        "dailyRemaining": max(0, account.daily_limit - account.daily_used),
        "monthlyRemaining": max(0, account.monthly_limit - account.monthly_used),
    }