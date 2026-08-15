"""Tests for domains.feedback.per_user_lora — per-user LoRA adapter store."""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from domains.feedback.per_user_lora import (
    UserAdapter,
    PerUserLoRAStore,
    get_per_user_lora,
)

import domains.feedback.per_user_lora as mod


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the global singleton before each test."""
    mod._per_user_lora = None
    yield
    mod._per_user_lora = None


@pytest.fixture
def store(tmp_path):
    """Create a PerUserLoRAStore in a temp directory."""
    return PerUserLoRAStore(
        store_path=str(tmp_path),
        adapter_rank=4,
        adapter_alpha=8,
        model_dim=16,
        run_eval=False,
        auto_aggregate_threshold=999,
        auto_prune_threshold=999,
    )


class TestUserAdapterDataclass:
    def test_fields(self):
        a = UserAdapter(
            user_id="u1", W_a=np.zeros((2, 4)), W_b=np.zeros((4, 2)),
            rank=2, alpha=4.0, created_at="1.0", updated_at="2.0",
        )
        assert a.user_id == "u1"
        assert a.rank == 2
        assert a.feedback_count == 0

    def test_feedback_count_default(self):
        a = UserAdapter(
            user_id="x", W_a=np.zeros((1, 1)), W_b=np.zeros((1, 1)),
            rank=1, alpha=1.0, created_at="0", updated_at="0",
        )
        assert a.feedback_count == 0


class TestCreateAdapter:
    def test_creates_new_adapter(self, store):
        adapter = store.create_adapter("user1")
        assert adapter.user_id == "user1"
        assert adapter.W_a.shape == (4, 16)
        assert adapter.W_b.shape == (16, 4)
        assert adapter.rank == 4
        assert adapter.alpha == 8
        assert adapter.feedback_count == 0

    def test_creates_writes_to_disk(self, store):
        store.create_adapter("user1")
        path = store._get_adapter_path("user1")
        assert path.exists()
        data = np.load(path)
        assert "W_a" in data
        assert "W_b" in data

    def test_creates_writes_to_db(self, store):
        store.create_adapter("user1")
        conn = sqlite3.connect(store.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM user_adapters WHERE user_id = 'user1'")
        row = cursor.fetchone()
        conn.close()
        assert row is not None
        assert row[1] == 4  # rank

    def test_cache_hit_skips_disk(self, store):
        a1 = store.create_adapter("user1")
        a2 = store.create_adapter("user1")
        assert a1 is a2

    def test_disk_load_on_second_call(self, store):
        store.create_adapter("user1")
        store._cache.clear()
        a2 = store.create_adapter("user1")
        assert a2.user_id == "user1"
        assert a2.W_a.shape == (4, 16)

    def test_safe_id_slashes(self, store):
        adapter = store.create_adapter("user/with/slashes")
        assert store._get_adapter_path("user/with/slashes").exists()

    def test_safe_id_backslashes(self, store):
        adapter = store.create_adapter("user\\back\\slashes")
        assert adapter.user_id == "user\\back\\slashes"


class TestGetAdapter:
    def test_returns_none_for_missing(self, store):
        assert store.get_adapter("nonexistent") is None

    def test_returns_cached(self, store):
        a1 = store.create_adapter("user1")
        a2 = store.get_adapter("user1")
        assert a1 is a2

    def test_loads_from_disk(self, store):
        store.create_adapter("user1")
        store._cache.clear()
        a = store.get_adapter("user1")
        assert a is not None
        assert a.user_id == "user1"

    def test_returns_none_after_delete(self, store):
        store.create_adapter("user1")
        store.delete_adapter("user1")
        assert store.get_adapter("user1") is None


class TestUpdateAdapter:
    def test_positive_feedback(self, store):
        adapter = store.create_adapter("user1")
        initial_wb = adapter.W_b.copy()
        store.update_adapter("user1", feedback_signal=1.0)
        updated = store.get_adapter("user1")
        assert not np.array_equal(updated.W_b, initial_wb)
        assert updated.feedback_count == 1

    def test_negative_feedback(self, store):
        store.create_adapter("user1")
        store.update_adapter("user1", feedback_signal=-1.0)
        updated = store.get_adapter("user1")
        assert updated.feedback_count == 1

    def test_multiple_updates(self, store):
        store.create_adapter("user1")
        for _ in range(5):
            store.update_adapter("user1", feedback_signal=1.0)
        updated = store.get_adapter("user1")
        assert updated.feedback_count == 5

    def test_clipping(self, store):
        store.create_adapter("user1")
        for _ in range(100):
            store.update_adapter("user1", feedback_signal=10.0, learning_rate=1.0)
        updated = store.get_adapter("user1")
        assert np.all(np.abs(updated.W_b) <= 1.0)
        assert np.all(np.abs(updated.W_a) <= 1.0)

    def test_creates_adapter_if_missing(self, store):
        adapter = store.update_adapter("new_user", feedback_signal=1.0)
        assert adapter.user_id == "new_user"
        assert adapter.feedback_count == 1

    def test_updated_at_changes(self, store):
        store.create_adapter("user1")
        old_updated = store.get_adapter("user1").updated_at
        time.sleep(0.01)
        store.update_adapter("user1", feedback_signal=1.0)
        new_updated = store.get_adapter("user1").updated_at
        assert new_updated >= old_updated


class TestApplyAdapterToLogits:
    def test_applies_adjustment(self, store):
        store.create_adapter("user1")
        # Need to update adapter so W_b is non-zero (initial W_b is zeros)
        store.update_adapter("user1", feedback_signal=1.0)
        logits = np.zeros((1, 16))
        result = store.apply_adapter_to_logits("user1", logits)
        assert result.shape == logits.shape
        assert not np.array_equal(result, logits)

    def test_no_adapter_returns_original(self, store):
        logits = np.ones((1, 16))
        result = store.apply_adapter_to_logits("nonexistent", logits)
        assert np.array_equal(result, logits)

    def test_scale_affects_output(self, store):
        store.create_adapter("user1")
        store.update_adapter("user1", feedback_signal=1.0)
        logits = np.zeros((1, 16))
        r1 = store.apply_adapter_to_logits("user1", logits, scale=1.0)
        r2 = store.apply_adapter_to_logits("user1", logits, scale=2.0)
        assert not np.allclose(r1, r2)


class TestMergeAdapters:
    def test_merges_two_adapters(self, store):
        store.create_adapter("a")
        store.create_adapter("b")
        merged = store.merge_adapters(["a", "b"])
        assert merged["user_count"] == 2
        assert merged["W_a"].shape == (4, 16)
        assert merged["W_b"].shape == (16, 4)

    def test_merge_averages_weights(self, store):
        a1 = store.create_adapter("a")
        a2 = store.create_adapter("b")
        merged = store.merge_adapters(["a", "b"])
        expected_a = (a1.W_a + a2.W_a) / 2
        np.testing.assert_allclose(merged["W_a"], expected_a)

    def test_merge_skips_missing(self, store):
        store.create_adapter("a")
        merged = store.merge_adapters(["a", "nonexistent"])
        assert merged["user_count"] == 1

    def test_merge_empty(self, store):
        merged = store.merge_adapters([])
        assert merged["user_count"] == 0


class TestMergeAll:
    def test_merges_every_adapter(self, store):
        store.create_adapter("a")
        store.create_adapter("b")
        store.create_adapter("c")
        merged = store.merge_all()
        assert merged["user_count"] == 3
        assert merged["W_a"].shape == (4, 16)
        assert merged["W_b"].shape == (16, 4)

    def test_empty_store(self, store):
        merged = store.merge_all()
        assert merged["user_count"] == 0

    def test_matches_explicit_merge(self, store):
        store.create_adapter("a")
        store.create_adapter("b")
        expected = store.merge_adapters(["a", "b"])
        merged = store.merge_all()
        assert merged["user_count"] == expected["user_count"]
        np.testing.assert_allclose(merged["W_a"], expected["W_a"])
        np.testing.assert_allclose(merged["W_b"], expected["W_b"])


class TestGetAllAdapters:
    def test_empty_store(self, store):
        assert store.get_all_adapters() == []

    def test_returns_metadata(self, store):
        store.create_adapter("u1")
        store.create_adapter("u2")
        adapters = store.get_all_adapters()
        assert len(adapters) == 2
        ids = {a["user_id"] for a in adapters}
        assert ids == {"u1", "u2"}

    def test_metadata_fields(self, store):
        store.create_adapter("u1")
        adapters = store.get_all_adapters()
        a = adapters[0]
        assert "user_id" in a
        assert "rank" in a
        assert "feedback_count" in a


class TestGetStats:
    def test_empty_store(self, store):
        stats = store.get_stats()
        assert stats["total_users"] == 0
        assert stats["total_size_bytes"] == 0

    def test_with_adapters(self, store):
        store.create_adapter("u1")
        store.create_adapter("u2")
        stats = store.get_stats()
        assert stats["total_users"] == 2
        assert stats["adapter_rank"] == 4
        assert stats["model_dim"] == 16

    def test_size_calculation(self, store):
        store.create_adapter("u1")
        stats = store.get_stats()
        assert stats["total_size_bytes"] > 0
        assert stats["avg_size_per_user_kb"] > 0


class TestDeleteAdapter:
    def test_deletes_from_disk(self, store):
        store.create_adapter("u1")
        store.delete_adapter("u1")
        assert not store._get_adapter_path("u1").exists()

    def test_deletes_from_db(self, store):
        store.create_adapter("u1")
        store.delete_adapter("u1")
        adapters = store.get_all_adapters()
        assert len(adapters) == 0

    def test_deletes_from_cache(self, store):
        store.create_adapter("u1")
        store.delete_adapter("u1")
        assert "u1" not in store._cache

    def test_delete_nonexistent_is_noop(self, store):
        store.delete_adapter("nope")  # should not raise


class TestGetQualityAdapters:
    def test_filters_by_feedback(self, store):
        store.create_adapter("low")
        store.update_adapter("low", 1.0)
        store.create_adapter("high")
        for _ in range(5):
            store.update_adapter("high", 1.0)
        quality = store.get_quality_adapters(min_feedback_count=3)
        assert len(quality) == 1
        assert quality[0]["user_id"] == "high"

    def test_filters_by_age(self, store):
        store.create_adapter("old")
        for _ in range(5):
            store.update_adapter("old", 1.0)
        # Set updated_at to 60 days ago
        store._cache["old"].updated_at = str(time.time() - 60 * 86400)
        store._update_metadata(store._cache["old"])
        quality = store.get_quality_adapters(min_feedback_count=1, max_age_days=30)
        assert len(quality) == 0


class TestGetQualityReport:
    def test_empty_store(self, store):
        report = store.get_quality_report()
        assert report == {"count": 0, "adapters": []}

    def test_filters_by_feedback(self, store):
        store.create_adapter("low")
        store.update_adapter("low", 1.0)
        store.create_adapter("high")
        for _ in range(5):
            store.update_adapter("high", 1.0)
        report = store.get_quality_report(min_feedback_count=3)
        assert report["count"] == 1
        assert [a["user_id"] for a in report["adapters"]] == ["high"]

    def test_count_matches_adapters(self, store):
        for uid in ["u1", "u2", "u3"]:
            store.create_adapter(uid)
            for _ in range(5):
                store.update_adapter(uid, 1.0)
        report = store.get_quality_report(min_feedback_count=1)
        assert report["count"] == len(report["adapters"]) == 3

    def test_filters_by_age(self, store):
        store.create_adapter("old")
        for _ in range(5):
            store.update_adapter("old", 1.0)
        store._cache["old"].updated_at = str(time.time() - 60 * 86400)
        store._update_metadata(store._cache["old"])
        report = store.get_quality_report(min_feedback_count=1, max_age_days=30)
        assert report["count"] == 0
        assert report["adapters"] == []

    def test_zero_max_age_days_means_no_age_filter(self, store):
        store.create_adapter("u1")
        for _ in range(5):
            store.update_adapter("u1", 1.0)
        report = store.get_quality_report(min_feedback_count=1, max_age_days=0)
        assert report["count"] == 1



class TestPruneLowQuality:
    def test_prunes_old_adapters(self, store):
        store.create_adapter("old")
        store._cache["old"].updated_at = str(time.time() - 60 * 86400)
        store._update_metadata(store._cache["old"])
        deleted = store.prune_low_quality(min_feedback_count=1, max_age_days=30)
        assert "old" in deleted

    def test_keeps_recent_adapters(self, store):
        store.create_adapter("new")
        deleted = store.prune_low_quality(min_feedback_count=0, max_age_days=30)
        assert "new" not in deleted

    def test_prunes_low_feedback(self, store):
        store.create_adapter("low")
        deleted = store.prune_low_quality(min_feedback_count=5, max_age_days=9999)
        assert "low" in deleted


class TestResetUserAdapter:
    def test_resets_weights(self, store):
        store.create_adapter("u1")
        store.update_adapter("u1", 1.0)
        old_wb = store.get_adapter("u1").W_b.copy()
        store.reset_user_adapter("u1")
        new = store.get_adapter("u1")
        assert new.feedback_count == 0
        assert not np.array_equal(new.W_b, old_wb)

    def test_resets_feedback_count(self, store):
        store.create_adapter("u1")
        for _ in range(10):
            store.update_adapter("u1", 1.0)
        store.reset_user_adapter("u1")
        assert store.get_adapter("u1").feedback_count == 0


class TestSingleton:
    def test_get_per_user_lora_returns_same(self, tmp_path):
        a = get_per_user_lora(store_path=str(tmp_path / "a"))
        b = get_per_user_lora(store_path=str(tmp_path / "b"))
        assert a is b
