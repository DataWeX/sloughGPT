"""
Token billing system - tracks usage, enforces limits, manages credits.

Architecture:
- TokenAccount: Per-user balance and limits
- UsageTracker: Records every request's token consumption
- TierManager: Subscription tier logic
- CreditManager: Top-up and balance operations
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..core.database import get_db


class Tier(str, Enum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


TIER_LIMITS = {
    Tier.FREE: {"daily": 500, "monthly": 10_000, "price": 0},
    Tier.PRO: {"daily": 10_000, "monthly": 300_000, "price": 20},
    Tier.ENTERPRISE: {"daily": 100_000, "monthly": 3_000_000, "price": 100},
}

MODEL_PRICING = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "default": {"input": 0.001, "output": 0.002},
}


@dataclass
class TokenAccount:
    user_id: str
    balance: int = 0
    tier: Tier = Tier.FREE
    daily_used: int = 0
    monthly_used: int = 0
    daily_limit: int = 0
    monthly_limit: int = 0
    last_daily_reset: float = 0
    last_monthly_reset: float = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self):
        limits = TIER_LIMITS.get(self.tier, TIER_LIMITS[Tier.FREE])
        if self.daily_limit == 0:
            self.daily_limit = limits["daily"]
        if self.monthly_limit == 0:
            self.monthly_limit = limits["monthly"]

    def can_afford(self, tokens: int) -> bool:
        now = time.time()
        self._maybe_reset(now)
        return (
            self.balance >= tokens
            and self.daily_used + tokens <= self.daily_limit
            and self.monthly_used + tokens <= self.monthly_limit
        )

    def deduct(self, tokens: int) -> bool:
        if not self.can_afford(tokens):
            return False
        self.balance -= tokens
        self.daily_used += tokens
        self.monthly_used += tokens
        self.updated_at = time.time()
        return True

    def add_credits(self, amount: int) -> None:
        self.balance += amount
        self.updated_at = time.time()

    def upgrade_tier(self, new_tier: Tier) -> None:
        self.tier = new_tier
        limits = TIER_LIMITS[new_tier]
        self.daily_limit = limits["daily"]
        self.monthly_limit = limits["monthly"]
        self.updated_at = time.time()

    def _maybe_reset(self, now: float) -> None:
        import calendar
        from datetime import datetime

        now_dt = datetime.fromtimestamp(now)
        current_day = now_dt.day
        current_month = now_dt.month

        last_reset_dt = datetime.fromtimestamp(self.last_daily_reset)
        if last_reset_dt.day != current_day:
            self.daily_used = 0
            self.last_daily_reset = now

        if last_reset_dt.month != current_month:
            self.monthly_used = 0
            self.last_monthly_reset = now

    def to_dict(self) -> dict:
        return {
            "userId": self.user_id,
            "balance": self.balance,
            "tier": self.tier.value,
            "dailyUsed": self.daily_used,
            "dailyLimit": self.daily_limit,
            "monthlyUsed": self.monthly_used,
            "monthlyLimit": self.monthly_limit,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }


@dataclass
class UsageRecord:
    id: str
    user_id: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    timestamp: float
    request_id: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "userId": self.user_id,
            "model": self.model,
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
            "totalTokens": self.total_tokens,
            "cost": self.cost,
            "timestamp": self.timestamp,
            "requestId": self.request_id,
        }


class TokenBillingService:
    def __init__(self):
        self._accounts: dict[str, TokenAccount] = {}
        self._usage: list[UsageRecord] = []
        self._load_accounts()

    def _load_accounts(self) -> None:
        db = get_db()
        accounts = db.find("token_accounts", {})
        for acc_data in accounts:
            account = TokenAccount(
                user_id=acc_data["user_id"],
                balance=acc_data.get("balance", 0),
                tier=Tier(acc_data.get("tier", "free")),
                daily_used=acc_data.get("daily_used", 0),
                monthly_used=acc_data.get("monthly_used", 0),
                last_daily_reset=acc_data.get("last_daily_reset", 0),
                last_monthly_reset=acc_data.get("last_monthly_reset", 0),
            )
            self._accounts[account.user_id] = account

    def _save_account(self, account: TokenAccount) -> None:
        db = get_db()
        db.upsert("token_accounts", {"user_id": account.user_id}, account.to_dict())

    def get_or_create_account(self, user_id: str) -> TokenAccount:
        if user_id not in self._accounts:
            account = TokenAccount(
                user_id=user_id,
                balance=500,
                tier=Tier.FREE,
            )
            self._accounts[user_id] = account
            self._save_account(account)
        return self._accounts[user_id]

    def check_and_deduct(self, user_id: str, model: str, input_tokens: int, output_tokens: int, request_id: str = "") -> tuple[bool, str]:
        account = self.get_or_create_account(user_id)
        total_tokens = input_tokens + output_tokens

        if not account.can_afford(total_tokens):
            return False, "Insufficient tokens or limit exceeded"

        if not account.deduct(total_tokens):
            return False, "Failed to deduct tokens"

        pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000

        record = UsageRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost=cost,
            timestamp=time.time(),
            request_id=request_id,
        )
        self._usage.append(record)

        db = get_db()
        db.insert("token_usage", record.to_dict())
        self._save_account(account)

        return True, "OK"

    def add_credits(self, user_id: str, amount: int) -> TokenAccount:
        account = self.get_or_create_account(user_id)
        account.add_credits(amount)
        self._save_account(account)
        return account

    def upgrade_tier(self, user_id: str, tier: Tier) -> TokenAccount:
        account = self.get_or_create_account(user_id)
        account.upgrade_tier(tier)
        self._save_account(account)
        return account

    def get_balance(self, user_id: str) -> TokenAccount:
        return self.get_or_create_account(user_id)

    def get_usage_summary(self, user_id: str) -> dict:
        user_usage = [r for r in self._usage if r.user_id == user_id]

        by_model: dict[str, dict] = {}
        by_day: dict[str, dict] = {}

        for record in user_usage:
            if record.model not in by_model:
                by_model[record.model] = {"requests": 0, "tokens": 0, "cost": 0}
            by_model[record.model]["requests"] += 1
            by_model[record.model]["tokens"] += record.total_tokens
            by_model[record.model]["cost"] += record.cost

            day_key = time.strftime("%Y-%m-%d", time.localtime(record.timestamp))
            if day_key not in by_day:
                by_day[day_key] = {"requests": 0, "tokens": 0}
            by_day[day_key]["requests"] += 1
            by_day[day_key]["tokens"] += record.total_tokens

        return {
            "totalRequests": len(user_usage),
            "totalTokens": sum(r.total_tokens for r in user_usage),
            "totalCost": sum(r.cost for r in user_usage),
            "byModel": by_model,
            "byDay": by_day,
        }

    def get_usage_history(self, user_id: str, limit: int = 50, offset: int = 0) -> list[UsageRecord]:
        user_usage = [r for r in self._usage if r.user_id == user_id]
        user_usage.sort(key=lambda r: r.timestamp, reverse=True)
        return user_usage[offset:offset + limit]


_token_billing_service: Optional[TokenBillingService] = None


def get_token_billing_service() -> TokenBillingService:
    global _token_billing_service
    if _token_billing_service is None:
        _token_billing_service = TokenBillingService()
    return _token_billing_service