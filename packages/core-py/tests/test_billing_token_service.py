"""Tests for TokenAccount — billing dataclass with pure logic."""
from __future__ import annotations

import time

from domains.billing.token_service import MODEL_PRICING, TIER_LIMITS, Tier, TokenAccount, UsageRecord


class TestTokenAccountDefaults:
    def test_default_tier(self):
        acc = TokenAccount(user_id="u1")
        assert acc.tier == Tier.FREE
        assert acc.daily_limit == TIER_LIMITS[Tier.FREE]["daily"]
        assert acc.monthly_limit == TIER_LIMITS[Tier.FREE]["monthly"]

    def test_pro_tier_limits(self):
        acc = TokenAccount(user_id="u1", tier=Tier.PRO)
        assert acc.daily_limit == TIER_LIMITS[Tier.PRO]["daily"]
        assert acc.monthly_limit == TIER_LIMITS[Tier.PRO]["monthly"]

    def test_enterprise_tier_limits(self):
        acc = TokenAccount(user_id="u1", tier=Tier.ENTERPRISE)
        assert acc.daily_limit == TIER_LIMITS[Tier.ENTERPRISE]["daily"]


class TestCanAfford:
    def test_can_afford_with_balance(self):
        acc = TokenAccount(user_id="u1", balance=1000)
        assert acc.can_afford(500) is True

    def test_cannot_afford_exceeds_balance(self):
        acc = TokenAccount(user_id="u1", balance=100)
        assert acc.can_afford(200) is False

    def test_cannot_afford_exceeds_daily_limit(self):
        now = time.time()
        acc = TokenAccount(user_id="u1", balance=100_000, daily_used=490, daily_limit=500, last_daily_reset=now, last_monthly_reset=now)
        assert acc.can_afford(20) is False

    def test_cannot_afford_exceeds_monthly_limit(self):
        now = time.time()
        acc = TokenAccount(user_id="u1", balance=100_000, monthly_used=9_990, monthly_limit=10_000, last_daily_reset=now, last_monthly_reset=now)
        assert acc.can_afford(20) is False


class TestDeduct:
    def test_deduct_success(self):
        acc = TokenAccount(user_id="u1", balance=1000)
        result = acc.deduct(300)
        assert result is True
        assert acc.balance == 700
        assert acc.daily_used == 300

    def test_deduct_failure(self):
        acc = TokenAccount(user_id="u1", balance=100)
        result = acc.deduct(200)
        assert result is False
        assert acc.balance == 100


class TestAddCredits:
    def test_add_credits(self):
        acc = TokenAccount(user_id="u1", balance=100)
        acc.add_credits(500)
        assert acc.balance == 600


class TestUpgradeTier:
    def test_upgrade_to_pro(self):
        acc = TokenAccount(user_id="u1", tier=Tier.FREE)
        acc.upgrade_tier(Tier.PRO)
        assert acc.tier == Tier.PRO
        assert acc.daily_limit == TIER_LIMITS[Tier.PRO]["daily"]

    def test_upgrade_to_enterprise(self):
        acc = TokenAccount(user_id="u1", tier=Tier.PRO)
        acc.upgrade_tier(Tier.ENTERPRISE)
        assert acc.tier == Tier.ENTERPRISE
        assert acc.monthly_limit == TIER_LIMITS[Tier.ENTERPRISE]["monthly"]


class TestToDict:
    def test_to_dict(self):
        acc = TokenAccount(user_id="u1", balance=500, tier=Tier.PRO)
        d = acc.to_dict()
        assert d["userId"] == "u1"
        assert d["balance"] == 500
        assert d["tier"] == "pro"
        assert "dailyUsed" in d
        assert "monthlyLimit" in d


class TestTierLimits:
    def test_all_tiers_have_limits(self):
        for tier in Tier:
            assert tier in TIER_LIMITS
            assert "daily" in TIER_LIMITS[tier]
            assert "monthly" in TIER_LIMITS[tier]
            assert "price" in TIER_LIMITS[tier]


class TestModelPricing:
    def test_all_models_have_pricing(self):
        for model, pricing in MODEL_PRICING.items():
            assert "input" in pricing
            assert "output" in pricing
            assert pricing["input"] >= 0
            assert pricing["output"] >= 0

    def test_default_pricing_exists(self):
        assert "default" in MODEL_PRICING


class TestUsageRecord:
    def test_to_dict(self):
        record = UsageRecord(
            id="r1", user_id="u1", model="gpt-4",
            input_tokens=100, output_tokens=50, total_tokens=150,
            cost=0.01, timestamp=time.time(),
        )
        d = record.to_dict()
        assert d["id"] == "r1"
        assert d["model"] == "gpt-4"
        assert d["totalTokens"] == 150
