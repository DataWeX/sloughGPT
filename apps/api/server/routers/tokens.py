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

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel

from ..infrastructure.auth import get_current_user
from ..billing.token_service import (
    get_token_billing_service,
    Tier,
)

router = APIRouter(prefix="/tokens", tags=["tokens"])


class TopUpRequest(BaseModel):
    amount: int


class UpgradeRequest(BaseModel):
    tier: str


class CheckRequest(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int


@router.get("/balance")
async def get_balance(user=Depends(get_current_user)):
    service = get_token_billing_service()
    account = service.get_balance(user["id"])
    return account.to_dict()


@router.get("/usage/summary")
async def get_usage_summary(user=Depends(get_current_user)):
    service = get_token_billing_service()
    return service.get_usage_summary(user["id"])


@router.get("/usage/history")
async def get_usage_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    user=Depends(get_current_user),
):
    service = get_token_billing_service()
    records = service.get_usage_history(user["id"], limit=limit, offset=offset)
    return {"records": [r.to_dict() for r in records]}


@router.post("/topup")
async def topup_credits(request: TopUpRequest, user=Depends(get_current_user)):
    if request.amount <= 0:
        raise HTTPException(status_code=400, detail="Amount must be positive")
    if request.amount > 1_000_000:
        raise HTTPException(status_code=400, detail="Maximum top-up is 1,000,000 tokens")

    service = get_token_billing_service()
    account = service.add_credits(user["id"], request.amount)
    return account.to_dict()


@router.post("/upgrade")
async def upgrade_tier(request: UpgradeRequest, user=Depends(get_current_user)):
    try:
        tier = Tier(request.tier)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tier. Must be one of: {', '.join(t.value for t in Tier)}",
        )

    service = get_token_billing_service()
    account = service.upgrade_tier(user["id"], tier)
    return account.to_dict()


@router.post("/check")
async def check_tokens(request: CheckRequest, user=Depends(get_current_user)):
    service = get_token_billing_service()
    account = service.get_balance(user["id"])
    total_tokens = request.input_tokens + request.output_tokens
    can_afford = account.can_afford(total_tokens)
    return {
        "canAfford": can_afford,
        "totalTokens": total_tokens,
        "balance": account.balance,
        "dailyRemaining": max(0, account.daily_limit - account.daily_used),
        "monthlyRemaining": max(0, account.monthly_limit - account.monthly_used),
    }