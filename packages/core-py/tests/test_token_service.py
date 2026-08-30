"""Comprehensive tests for domains.billing.token_service — pure logic only."""

import pytest
import time
import sys
import os
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from domains.billing.token_service import (
    TokenBillingService,
    TokenAccount,
    UsageRecord,
    Tier,
    TIER_LIMITS,
    MODEL_PRICING,
    get_token_billing_service,
)


# ---------------------------------------------------------------------------
# Tier enum
# ---------------------------------------------------------------------------

class TestTier:
    def test_tier_values(self):
        assert Tier.FREE.value == "free"
        assert Tier.PRO.value == "pro"
        assert Tier.ENTERPRISE.value == "enterprise"

    def test_tier_is_string_enum(self):
        assert isinstance(Tier.FREE, str)
        assert Tier.FREE == "free"

    def test_tier_from_value(self):
        assert Tier("free") is Tier.FREE
        assert Tier("pro") is Tier.PRO
        assert Tier("enterprise") is Tier.ENTERPRISE

    def test_invalid_tier_raises(self):
        with pytest.raises(ValueError):
            Tier("invalid")

    def test_all_tiers_have_limits(self):
        for tier in Tier:
            assert tier in TIER_LIMITS
            limits = TIER_LIMITS[tier]
            assert "daily" in limits
            assert "monthly" in limits
            assert "price" in limits
            assert limits["daily"] > 0
            assert limits["monthly"] > 0
            assert limits["price"] >= 0

    def test_tier_limits_ordering(self):
        assert TIER_LIMITS[Tier.FREE]["daily"] < TIER_LIMITS[Tier.PRO]["daily"]
        assert TIER_LIMITS[Tier.PRO]["daily"] < TIER_LIMITS[Tier.ENTERPRISE]["daily"]
        assert TIER_LIMITS[Tier.FREE]["monthly"] < TIER_LIMITS[Tier.PRO]["monthly"]
        assert TIER_LIMITS[Tier.PRO]["monthly"] < TIER_LIMITS[Tier.ENTERPRISE]["monthly"]


# ---------------------------------------------------------------------------
# MODEL_PRICING
# ---------------------------------------------------------------------------

class TestModelPricing:
    def test_default_pricing_exists(self):
        assert "default" in MODEL_PRICING
        assert "input" in MODEL_PRICING["default"]
        assert "output" in MODEL_PRICING["default"]

    def test_default_pricing_positive(self):
        assert MODEL_PRICING["default"]["input"] > 0
        assert MODEL_PRICING["default"]["output"] > 0

    def test_all_models_have_input_and_output(self):
        for model, pricing in MODEL_PRICING.items():
            assert "input" in pricing, f"{model} missing 'input'"
            assert "output" in pricing, f"{model} missing 'output'"
            assert pricing["input"] >= 0, f"{model} input price negative"
            assert pricing["output"] >= 0, f"{model} output price negative"


# ---------------------------------------------------------------------------
# TokenAccount — dataclass and pure logic
# ---------------------------------------------------------------------------

class TestTokenAccount:
    def test_creation_defaults(self):
        account = TokenAccount(user_id="u1")
        assert account.user_id == "u1"
        assert account.balance == 0
        assert account.tier == Tier.FREE
        assert account.daily_used == 0
        assert account.monthly_used == 0
        assert account.daily_limit == TIER_LIMITS[Tier.FREE]["daily"]
        assert account.monthly_limit == TIER_LIMITS[Tier.FREE]["monthly"]

    def test_creation_with_custom_values(self):
        account = TokenAccount(
            user_id="u2", balance=5000, tier=Tier.PRO,
            daily_used=100, monthly_used=500,
            daily_limit=999, monthly_limit=9999,
        )
        assert account.balance == 5000
        assert account.tier == Tier.PRO
        assert account.daily_limit == 999
        assert account.monthly_limit == 9999

    def test_post_init_auto_fills_limits_when_zero(self):
        account = TokenAccount(user_id="u3", tier=Tier.ENTERPRISE, balance=100)
        assert account.daily_limit == TIER_LIMITS[Tier.ENTERPRISE]["daily"]
        assert account.monthly_limit == TIER_LIMITS[Tier.ENTERPRISE]["monthly"]

    def test_post_init_preserves_custom_limits(self):
        account = TokenAccount(
            user_id="u4", tier=Tier.PRO, balance=100,
            daily_limit=5000, monthly_limit=50000,
        )
        assert account.daily_limit == 5000
        assert account.monthly_limit == 50000

    def test_can_afford_exact_balance(self):
        account = TokenAccount(user_id="u5", balance=100)
        assert account.can_afford(100) is True

    def test_can_afford_within_balance(self):
        account = TokenAccount(user_id="u6", balance=1000)
        assert account.can_afford(500) is True

    def test_cannot_afford_exceeds_balance(self):
        account = TokenAccount(user_id="u7", balance=100)
        assert account.can_afford(101) is False

    def test_cannot_afford_exceeds_daily_limit(self):
        account = TokenAccount(
            user_id="u8", balance=100_000,
            daily_limit=100, monthly_limit=100_000,
        )
        assert account.can_afford(101) is False

    def test_cannot_afford_exceeds_monthly_limit(self):
        account = TokenAccount(
            user_id="u9", balance=100_000,
            daily_limit=100_000, monthly_limit=50,
        )
        assert account.can_afford(51) is False

    def test_can_afford_zero_tokens(self):
        account = TokenAccount(user_id="u10", balance=0)
        assert account.can_afford(0) is True

    def test_deduct_success(self):
        account = TokenAccount(user_id="u11", balance=1000, daily_limit=5000, monthly_limit=50000)
        result = account.deduct(200)
        assert result is True
        assert account.balance == 800
        assert account.daily_used == 200
        assert account.monthly_used == 200

    def test_deduct_failure_insufficient(self):
        account = TokenAccount(user_id="u12", balance=50, daily_limit=5000, monthly_limit=50000)
        result = account.deduct(100)
        assert result is False
        assert account.balance == 50
        assert account.daily_used == 0

    def test_deduct_failure_exceeds_daily(self):
        account = TokenAccount(
            user_id="u13", balance=100_000,
            daily_limit=100, monthly_limit=100_000,
            daily_used=90,
        )
        account.last_daily_reset = time.time()
        result = account.deduct(20)
        assert result is False
        assert account.balance == 100_000

    def test_deduct_failure_exceeds_monthly(self):
        account = TokenAccount(
            user_id="u14", balance=100_000,
            daily_limit=100_000, monthly_limit=100,
            monthly_used=90,
        )
        account.last_daily_reset = time.time()
        result = account.deduct(20)
        assert result is False

    def test_deduct_multiple_times(self):
        account = TokenAccount(user_id="u15", balance=1000, daily_limit=5000, monthly_limit=50000)
        account.deduct(100)
        account.deduct(200)
        account.deduct(50)
        assert account.balance == 650
        assert account.daily_used == 350
        assert account.monthly_used == 350

    def test_add_credits(self):
        account = TokenAccount(user_id="u16", balance=100)
        account.add_credits(200)
        assert account.balance == 300

    def test_add_credits_zero(self):
        account = TokenAccount(user_id="u17", balance=100)
        account.add_credits(0)
        assert account.balance == 100

    def test_add_credits_large_amount(self):
        account = TokenAccount(user_id="u18", balance=0)
        account.add_credits(10_000_000)
        assert account.balance == 10_000_000

    def test_upgrade_tier_free_to_pro(self):
        account = TokenAccount(user_id="u19", balance=100)
        account.upgrade_tier(Tier.PRO)
        assert account.tier == Tier.PRO
        assert account.daily_limit == TIER_LIMITS[Tier.PRO]["daily"]
        assert account.monthly_limit == TIER_LIMITS[Tier.PRO]["monthly"]

    def test_upgrade_tier_pro_to_enterprise(self):
        account = TokenAccount(user_id="u20", balance=100, tier=Tier.PRO)
        account.upgrade_tier(Tier.ENTERPRISE)
        assert account.tier == Tier.ENTERPRISE
        assert account.daily_limit == TIER_LIMITS[Tier.ENTERPRISE]["daily"]
        assert account.monthly_limit == TIER_LIMITS[Tier.ENTERPRISE]["monthly"]

    def test_downgrade_tier(self):
        account = TokenAccount(user_id="u21", balance=100, tier=Tier.ENTERPRISE)
        account.upgrade_tier(Tier.FREE)
        assert account.tier == Tier.FREE
        assert account.daily_limit == TIER_LIMITS[Tier.FREE]["daily"]

    def test_to_dict(self):
        account = TokenAccount(user_id="u22", balance=500, tier=Tier.PRO)
        d = account.to_dict()
        assert d["userId"] == "u22"
        assert d["balance"] == 500
        assert d["tier"] == "pro"
        assert "dailyUsed" in d
        assert "dailyLimit" in d
        assert "monthlyUsed" in d
        assert "monthlyLimit" in d
        assert "createdAt" in d
        assert "updatedAt" in d

    def test_to_dict_free_tier_string(self):
        account = TokenAccount(user_id="u23", balance=0)
        d = account.to_dict()
        assert d["tier"] == "free"

    def test_to_dict_enterprise_tier_string(self):
        account = TokenAccount(user_id="u24", balance=0, tier=Tier.ENTERPRISE)
        d = account.to_dict()
        assert d["tier"] == "enterprise"


# ---------------------------------------------------------------------------
# TokenAccount — daily/monthly reset logic
# ---------------------------------------------------------------------------

class TestTokenAccountResets:
    def test_daily_reset_on_new_day(self):
        account = TokenAccount(
            user_id="r1", balance=10_000,
            daily_used=400, daily_limit=500,
            monthly_used=400, monthly_limit=50_000,
        )
        # Simulate last reset was yesterday
        yesterday = time.time() - 86400
        account.last_daily_reset = yesterday
        account.last_monthly_reset = yesterday

        # After reset, daily_used should be 0
        assert account.can_afford(500) is True
        assert account.daily_used == 0

    def test_monthly_reset_on_new_month(self):
        account = TokenAccount(
            user_id="r2", balance=10_000,
            daily_used=100, daily_limit=500,
            monthly_used=49_000, monthly_limit=50_000,
        )
        # _maybe_reset uses last_daily_reset for BOTH day and month comparisons,
        # so set it to a time in a different month to trigger monthly reset.
        # Use day=1 so that even after monthly reset the daily check passes.
        import calendar
        from datetime import datetime
        now = time.time()
        now_dt = datetime.fromtimestamp(now)
        target_month = now_dt.month - 2
        target_year = now_dt.year
        while target_month <= 0:
            target_month += 12
            target_year -= 1
        import datetime as dt_mod
        target = dt_mod.datetime(target_year, target_month, 1,
                                 now_dt.hour, now_dt.minute, now_dt.second).timestamp()
        account.last_daily_reset = target

        # daily_used resets (different day), monthly_used resets (different month)
        # After reset: daily_used=0, monthly_used=0, so 0+500 <= 500 ✓
        assert account.can_afford(500) is True
        assert account.monthly_used == 0

    def test_no_reset_same_day(self):
        account = TokenAccount(
            user_id="r3", balance=10_000,
            daily_used=400, daily_limit=500,
            monthly_used=400, monthly_limit=50_000,
        )
        account.last_daily_reset = time.time()
        account.last_monthly_reset = time.time()

        assert account.can_afford(100) is True
        assert account.daily_used == 400

    def test_deduct_triggers_daily_reset(self):
        account = TokenAccount(
            user_id="r4", balance=10_000,
            daily_used=490, daily_limit=500,
            monthly_used=490, monthly_limit=50_000,
        )
        yesterday = time.time() - 86400
        account.last_daily_reset = yesterday
        account.last_monthly_reset = yesterday

        result = account.deduct(10)
        assert result is True
        assert account.daily_used == 10
        assert account.balance == 9990


# ---------------------------------------------------------------------------
# UsageRecord
# ---------------------------------------------------------------------------

class TestUsageRecord:
    def test_creation(self):
        record = UsageRecord(
            id="r1", user_id="u1", model="gpt-4",
            input_tokens=10, output_tokens=20,
            total_tokens=30, cost=0.001,
            timestamp=1000.0,
        )
        assert record.id == "r1"
        assert record.user_id == "u1"
        assert record.model == "gpt-4"
        assert record.input_tokens == 10
        assert record.output_tokens == 20
        assert record.total_tokens == 30
        assert record.cost == 0.001
        assert record.timestamp == 1000.0
        assert record.request_id == ""

    def test_creation_with_request_id(self):
        record = UsageRecord(
            id="r2", user_id="u2", model="gpt-3.5-turbo",
            input_tokens=5, output_tokens=15,
            total_tokens=20, cost=0.0001,
            timestamp=2000.0, request_id="req-abc",
        )
        assert record.request_id == "req-abc"

    def test_to_dict(self):
        record = UsageRecord(
            id="r3", user_id="u3", model="claude-3-opus",
            input_tokens=100, output_tokens=200,
            total_tokens=300, cost=0.05,
            timestamp=3000.0, request_id="req-xyz",
        )
        d = record.to_dict()
        assert d["id"] == "r3"
        assert d["userId"] == "u3"
        assert d["model"] == "claude-3-opus"
        assert d["inputTokens"] == 100
        assert d["outputTokens"] == 200
        assert d["totalTokens"] == 300
        assert d["cost"] == 0.05
        assert d["timestamp"] == 3000.0
        assert d["requestId"] == "req-xyz"

    def test_to_dict_empty_request_id(self):
        record = UsageRecord(
            id="r4", user_id="u4", model="default",
            input_tokens=0, output_tokens=0,
            total_tokens=0, cost=0.0,
            timestamp=0.0,
        )
        d = record.to_dict()
        assert d["requestId"] == ""


# ---------------------------------------------------------------------------
# TokenBillingService — uses in-memory DB, no external APIs
# ---------------------------------------------------------------------------

class TestTokenBillingServiceAccounts:
    def test_get_or_create_new_account(self):
        service = TokenBillingService()
        account = service.get_or_create_account("new-user")
        assert account.user_id == "new-user"
        assert account.balance == 500
        assert account.tier == Tier.FREE

    def test_get_or_create_existing_account(self):
        service = TokenBillingService()
        a1 = service.get_or_create_account("existing")
        a1.balance = 9999
        a2 = service.get_or_create_account("existing")
        assert a2.balance == 9999
        assert a1 is a2

    def test_get_or_create_persists_in_db(self):
        service = TokenBillingService()
        service.get_or_create_account("persist-user")
        # New service instance loads from DB
        service2 = TokenBillingService()
        account = service2.get_or_create_account("persist-user")
        assert account.user_id == "persist-user"
        assert account.balance == 500

    def test_get_balance_creates_if_missing(self):
        service = TokenBillingService()
        account = service.get_balance("balance-user")
        assert account.user_id == "balance-user"
        assert account.balance == 500


class TestTokenBillingServiceDeduction:
    def test_check_and_deduct_success(self):
        service = TokenBillingService()
        ok, msg = service.check_and_deduct("user-a", "gpt-4", 100, 50)
        assert ok is True
        assert msg == "OK"

    def test_check_and_deduct_reduces_balance(self):
        service = TokenBillingService()
        service.check_and_deduct("user-b", "gpt-4", 100, 50)
        account = service.get_balance("user-b")
        assert account.balance == 500 - 150

    def test_check_and_deduct_records_usage(self):
        service = TokenBillingService()
        service.check_and_deduct("user-c", "gpt-4", 100, 50, request_id="req-1")
        history = service.get_usage_history("user-c")
        assert len(history) == 1
        assert history[0].model == "gpt-4"
        assert history[0].input_tokens == 100
        assert history[0].output_tokens == 50
        assert history[0].total_tokens == 150
        assert history[0].request_id == "req-1"

    def test_check_and_deduct_insufficient_tokens(self):
        service = TokenBillingService()
        account = service.get_or_create_account("user-d")
        account.balance = 10
        ok, msg = service.check_and_deduct("user-d", "gpt-4", 100, 100)
        assert ok is False
        assert "Insufficient" in msg

    def test_check_and_deduct_zero_tokens(self):
        service = TokenBillingService()
        ok, msg = service.check_and_deduct("user-e", "gpt-4", 0, 0)
        assert ok is True

    def test_check_and_deduct_unknown_model_uses_default(self):
        service = TokenBillingService()
        ok, msg = service.check_and_deduct("user-f", "unknown-model", 100, 100)
        assert ok is True
        history = service.get_usage_history("user-f")
        assert history[0].model == "unknown-model"

    def test_check_and_deduct_cost_calculation_gpt4(self):
        service = TokenBillingService()
        service.upgrade_tier("user-g", Tier.ENTERPRISE)
        service.add_credits("user-g", 100_000)
        service.check_and_deduct("user-g", "gpt-4", 1000, 500)
        # cost = (1000 * 0.03 + 500 * 0.06) / 1000 = (30 + 30) / 1000 = 0.06
        history = service.get_usage_history("user-g")
        assert len(history) == 1
        assert abs(history[0].cost - 0.06) < 1e-9

    def test_check_and_deduct_cost_calculation_gpt35(self):
        service = TokenBillingService()
        service.upgrade_tier("user-h", Tier.ENTERPRISE)
        service.add_credits("user-h", 100_000)
        service.check_and_deduct("user-h", "gpt-3.5-turbo", 2000, 1000)
        # cost = (2000 * 0.0005 + 1000 * 0.0015) / 1000 = (1 + 1.5) / 1000 = 0.0025
        history = service.get_usage_history("user-h")
        assert len(history) == 1
        assert abs(history[0].cost - 0.0025) < 1e-9

    def test_check_and_deduct_cost_calculation_default(self):
        service = TokenBillingService()
        service.upgrade_tier("user-i", Tier.ENTERPRISE)
        service.add_credits("user-i", 100_000)
        service.check_and_deduct("user-i", "custom-model", 500, 500)
        # cost = (500 * 0.001 + 500 * 0.002) / 1000 = (0.5 + 1.0) / 1000 = 0.0015
        history = service.get_usage_history("user-i")
        assert len(history) == 1
        assert abs(history[0].cost - 0.0015) < 1e-9

    def test_check_and_deduct_multiple_requests(self):
        service = TokenBillingService()
        service.check_and_deduct("user-j", "gpt-4", 100, 50)
        service.check_and_deduct("user-j", "gpt-3.5-turbo", 200, 100)
        account = service.get_balance("user-j")
        assert account.balance == 500 - 150 - 300
        history = service.get_usage_history("user-j")
        assert len(history) == 2


class TestTokenBillingServiceCredits:
    def test_add_credits(self):
        service = TokenBillingService()
        account = service.add_credits("credit-user", 1000)
        assert account.balance == 1500

    def test_add_credits_to_existing_balance(self):
        service = TokenBillingService()
        service.add_credits("credit-user2", 500)
        service.add_credits("credit-user2", 300)
        account = service.get_balance("credit-user2")
        assert account.balance == 500 + 500 + 300

    def test_add_credits_zero(self):
        service = TokenBillingService()
        account = service.add_credits("credit-user3", 0)
        assert account.balance == 500


class TestTokenBillingServiceTierUpgrade:
    def test_upgrade_tier(self):
        service = TokenBillingService()
        account = service.upgrade_tier("tier-user", Tier.PRO)
        assert account.tier == Tier.PRO
        assert account.daily_limit == TIER_LIMITS[Tier.PRO]["daily"]

    def test_upgrade_tier_persists(self):
        service = TokenBillingService()
        service.upgrade_tier("tier-user2", Tier.ENTERPRISE)
        service2 = TokenBillingService()
        account = service2.get_balance("tier-user2")
        assert account.tier == Tier.ENTERPRISE


class TestTokenBillingServiceUsageSummary:
    def test_empty_summary(self):
        service = TokenBillingService()
        summary = service.get_usage_summary("no-usage-user")
        assert summary["totalRequests"] == 0
        assert summary["totalTokens"] == 0
        assert summary["totalCost"] == 0
        assert summary["byModel"] == {}
        assert summary["byDay"] == {}

    def test_summary_single_model(self):
        service = TokenBillingService()
        service.check_and_deduct("sum-user", "gpt-4", 100, 200)
        service.check_and_deduct("sum-user", "gpt-4", 50, 100)
        summary = service.get_usage_summary("sum-user")
        assert summary["totalRequests"] == 2
        assert summary["totalTokens"] == 450
        assert "gpt-4" in summary["byModel"]
        assert summary["byModel"]["gpt-4"]["requests"] == 2
        assert summary["byModel"]["gpt-4"]["tokens"] == 450

    def test_summary_multiple_models(self):
        service = TokenBillingService()
        service.check_and_deduct("sum-user2", "gpt-4", 100, 50)
        service.check_and_deduct("sum-user2", "gpt-3.5-turbo", 200, 100)
        summary = service.get_usage_summary("sum-user2")
        assert summary["totalRequests"] == 2
        assert "gpt-4" in summary["byModel"]
        assert "gpt-3.5-turbo" in summary["byModel"]

    def test_summary_by_day(self):
        service = TokenBillingService()
        service.check_and_deduct("sum-user3", "gpt-4", 100, 50)
        summary = service.get_usage_summary("sum-user3")
        today = time.strftime("%Y-%m-%d", time.localtime())
        assert today in summary["byDay"]
        assert summary["byDay"][today]["requests"] == 1
        assert summary["byDay"][today]["tokens"] == 150

    def test_summary_cost_totals(self):
        service = TokenBillingService()
        service.upgrade_tier("sum-user4", Tier.ENTERPRISE)
        service.add_credits("sum-user4", 100_000)
        service.check_and_deduct("sum-user4", "gpt-4", 1000, 500)
        service.check_and_deduct("sum-user4", "gpt-3.5-turbo", 2000, 1000)
        summary = service.get_usage_summary("sum-user4")
        expected_cost = 0.06 + 0.0025
        assert abs(summary["totalCost"] - expected_cost) < 1e-9

    def test_summary_isolation_between_users(self):
        service = TokenBillingService()
        service.check_and_deduct("iso-a", "gpt-4", 100, 50)
        service.check_and_deduct("iso-b", "gpt-4", 200, 100)
        summary_a = service.get_usage_summary("iso-a")
        summary_b = service.get_usage_summary("iso-b")
        assert summary_a["totalTokens"] == 150
        assert summary_b["totalTokens"] == 300


class TestTokenBillingServiceUsageHistory:
    def test_empty_history(self):
        service = TokenBillingService()
        history = service.get_usage_history("no-history-user")
        assert history == []

    def test_history_sorted_by_timestamp_desc(self):
        service = TokenBillingService()
        service.check_and_deduct("hist-user", "gpt-4", 10, 5)
        time.sleep(0.01)
        service.check_and_deduct("hist-user", "gpt-4", 20, 10)
        history = service.get_usage_history("hist-user")
        assert len(history) == 2
        assert history[0].timestamp >= history[1].timestamp

    def test_history_limit(self):
        service = TokenBillingService()
        for i in range(10):
            service.check_and_deduct("limit-user", "gpt-4", 1, 1)
        history = service.get_usage_history("limit-user", limit=3)
        assert len(history) == 3

    def test_history_offset(self):
        service = TokenBillingService()
        for i in range(5):
            service.check_and_deduct("offset-user", "gpt-4", 1, 1)
        all_history = service.get_usage_history("offset-user")
        paginated = service.get_usage_history("offset-user", offset=2)
        assert len(paginated) == 3
        assert paginated[0].id == all_history[2].id

    def test_history_limit_and_offset(self):
        service = TokenBillingService()
        for i in range(10):
            service.check_and_deduct("page-user", "gpt-4", 1, 1)
        page = service.get_usage_history("page-user", limit=2, offset=4)
        assert len(page) == 2

    def test_history_offset_beyond_data(self):
        service = TokenBillingService()
        service.check_and_deduct("far-user", "gpt-4", 10, 5)
        history = service.get_usage_history("far-user", offset=100)
        assert history == []

    def test_history_isolation_between_users(self):
        service = TokenBillingService()
        service.check_and_deduct("h-iso-a", "gpt-4", 10, 5)
        service.check_and_deduct("h-iso-b", "gpt-4", 20, 10)
        assert len(service.get_usage_history("h-iso-a")) == 1
        assert len(service.get_usage_history("h-iso-b")) == 1


class TestTokenBillingServicePersistence:
    def test_accounts_persist_across_instances(self):
        s1 = TokenBillingService()
        s1.add_credits("persist-a", 500)
        s1.upgrade_tier("persist-a", Tier.PRO)

        s2 = TokenBillingService()
        account = s2.get_balance("persist-a")
        assert account.balance == 1000
        assert account.tier == Tier.PRO

    def test_usage_records_not_persisted_in_db(self):
        s1 = TokenBillingService()
        s1.check_and_deduct("persist-b", "gpt-4", 100, 50)
        assert len(s1.get_usage_history("persist-b")) == 1

        # Usage records are in-memory only; new instance starts empty
        s2 = TokenBillingService()
        assert len(s2.get_usage_history("persist-b")) == 0

        # But the account (with deducted balance) IS persisted
        assert s2.get_balance("persist-b").balance < 500


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

class TestSingleton:
    def test_get_token_billing_service_returns_same_instance(self):
        s1 = get_token_billing_service()
        s2 = get_token_billing_service()
        assert s1 is s2

    def test_singleton_is_token_billing_service(self):
        s = get_token_billing_service()
        assert isinstance(s, TokenBillingService)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_deduct_at_exact_daily_limit(self):
        account = TokenAccount(
            user_id="edge1", balance=100_000,
            daily_limit=100, monthly_limit=100_000,
        )
        assert account.deduct(100) is True
        assert account.daily_used == 100
        assert account.can_afford(1) is False

    def test_deduct_at_exact_monthly_limit(self):
        account = TokenAccount(
            user_id="edge2", balance=100_000,
            daily_limit=100_000, monthly_limit=100,
        )
        assert account.deduct(100) is True
        assert account.monthly_used == 100
        assert account.can_afford(1) is False

    def test_large_token_deduction(self):
        account = TokenAccount(
            user_id="edge3", balance=10_000_000,
            daily_limit=10_000_000, monthly_limit=100_000_000,
        )
        assert account.deduct(9_999_999) is True
        assert account.balance == 1

    def test_negative_credit_addition(self):
        account = TokenAccount(user_id="edge4", balance=100)
        account.add_credits(-50)
        assert account.balance == 50

    def test_multiple_users_independent(self):
        service = TokenBillingService()
        service.check_and_deduct("ind-a", "gpt-4", 100, 50)
        service.check_and_deduct("ind-b", "gpt-4", 200, 100)
        assert service.get_balance("ind-a").balance == 350
        assert service.get_balance("ind-b").balance == 200

    def test_empty_model_string_uses_default_pricing(self):
        service = TokenBillingService()
        ok, _ = service.check_and_deduct("edge5", "", 100, 100)
        assert ok is True
        history = service.get_usage_history("edge5")
        assert history[0].model == ""

    def test_updated_at_changes_on_deduct(self):
        account = TokenAccount(user_id="edge6", balance=1000, daily_limit=5000, monthly_limit=50000)
        before = account.updated_at
        time.sleep(0.01)
        account.deduct(100)
        assert account.updated_at >= before

    def test_updated_at_changes_on_add_credits(self):
        account = TokenAccount(user_id="edge7", balance=100)
        before = account.updated_at
        time.sleep(0.01)
        account.add_credits(50)
        assert account.updated_at >= before

    def test_updated_at_changes_on_upgrade_tier(self):
        account = TokenAccount(user_id="edge8", balance=100)
        before = account.updated_at
        time.sleep(0.01)
        account.upgrade_tier(Tier.PRO)
        assert account.updated_at >= before
