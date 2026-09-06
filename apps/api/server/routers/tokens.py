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

import logging

from domains.billing.token_service import (
    Tier,
    get_token_billing_service,
)
from fastapi import APIRouter, Depends, Query
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel, Field
from schemas.common import raise_error, success_response

logger = logging.getLogger("slo.routers.tokens")

router = APIRouter(prefix="/tokens", tags=["tokens"])


class TopUpRequest(BaseModel):
    amount: int = Field(..., ge=1, le=1_000_000, description="Number of tokens to add")


class UpgradeRequest(BaseModel):
    tier: str = Field(..., pattern=r"^(free|basic|pro|enterprise)$", description="Target tier")


class CheckRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=200, description="Model identifier")
    input_tokens: int = Field(..., ge=0, le=10_000_000, description="Expected input tokens")
    output_tokens: int = Field(..., ge=0, le=10_000_000, description="Expected output tokens")


@router.get("/balance")
async def get_balance(auth_user: dict = Depends(require_auth_if_enabled)):
    try:
        service = get_token_billing_service()
        account = service.get_balance(auth_user["id"])
        return success_response(data=account.to_dict())
    except Exception as e:
        logger.error("Failed to get balance: %s", e, extra={"tag": "TOKENS"})
        raise_error(str(e), "E_TOKENS_BALANCE", status_code=500)


@router.get("/usage/summary")
async def get_usage_summary(auth_user: dict = Depends(require_auth_if_enabled)):
    try:
        service = get_token_billing_service()
        return success_response(data=service.get_usage_summary(auth_user["id"]))
    except Exception as e:
        logger.error("Failed to get usage summary: %s", e, extra={"tag": "TOKENS"})
        raise_error(str(e), "E_TOKENS_USAGE", status_code=500)


@router.get("/usage/history")
async def get_usage_history(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    auth_user: dict = Depends(require_auth_if_enabled),
):
    try:
        service = get_token_billing_service()
        records = service.get_usage_history(auth_user["id"], limit=limit, offset=offset)
        return success_response(data={"records": [r.to_dict() for r in records]})
    except Exception as e:
        logger.error("Failed to get usage history: %s", e, extra={"tag": "TOKENS"})
        raise_error(str(e), "E_TOKENS_HISTORY", status_code=500)


@router.post("/topup")
async def topup_credits(request: TopUpRequest, auth_user: dict = Depends(require_auth_if_enabled)):
    try:
        service = get_token_billing_service()
        account = service.add_credits(auth_user["id"], request.amount)
        return success_response(data=account.to_dict(), message=f"Added {request.amount} credits")
    except Exception as e:
        logger.error("Failed to topup credits: %s", e, extra={"tag": "TOKENS"})
        raise_error(str(e), "E_TOKENS_TOPUP", status_code=500)


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

    try:
        service = get_token_billing_service()
        account = service.upgrade_tier(auth_user["id"], tier)
        return success_response(data=account.to_dict(), message=f"Upgraded to {tier.value}")
    except Exception as e:
        logger.error("Failed to upgrade tier: %s", e, extra={"tag": "TOKENS"})
        raise_error(str(e), "E_TOKENS_UPGRADE", status_code=500)


@router.post("/check")
async def check_tokens(request: CheckRequest, auth_user: dict = Depends(require_auth_if_enabled)):
    try:
        service = get_token_billing_service()
        account = service.get_balance(auth_user["id"])
        total_tokens = request.input_tokens + request.output_tokens
        can_afford = account.can_afford(total_tokens)
        return success_response(
            data={
                "canAfford": can_afford,
                "totalTokens": total_tokens,
                "balance": account.balance,
                "dailyRemaining": max(0, account.daily_limit - account.daily_used),
                "monthlyRemaining": max(0, account.monthly_limit - account.monthly_used),
            }
        )
    except Exception as e:
        logger.error("Failed to check tokens: %s", e, extra={"tag": "TOKENS"})
        raise_error(str(e), "E_TOKENS_CHECK", status_code=500)
