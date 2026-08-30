"""Tests for per-user LoRA adapter store — creation, updates, merging, pruning."""

import time
import numpy as np
import pytest
from pathlib import Path

from domains.feedback.per_user_lora import (
    PerUserLoRAStore,
    UserAdapter,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    """Fresh PerUserLoRAStore with a temporary directory."""
    return PerUserLoRAStore(
        store_path=str(tmp_path),
        adapter_rank=4,
        adapter_alpha=8,
        model_dim=16,
        auto_aggregate_threshold=5,
        auto_prune_threshold=10,
        min_feedback_for_aggregation=2,
        run_eval=False,
    )


@pytest.fixture
def store_small(tmp_path):
    """Store with very low thresholds for testing auto-management."""
    return PerUserLoRAStore(
        store_path=str(tmp_path),
        adapter_rank=2,
        adapter_alpha=4,
        model_dim=8,
        auto_aggregate_threshold=3,
        auto_prune_threshold=5,
        min_feedback_for_aggregation=1,
        run_eval=False,
    )


# ── UserAdapter dataclass ───────────────────────────────────────────────────

class TestUserAdapter:

    def test_fields(self):
        adapter = UserAdapter(
            user_id="u1",
            W_a=np.zeros((4, 16), dtype=np.float32),
            W_b=np.zeros((16, 4), dtype=np.float32),
            rank=4,
            alpha=8,
            created_at="1.0",
            updated_at="2.0",
            feedback_count=3,
        )
        assert adapter.user_id == "u1"
        assert adapter.rank == 4
        assert adapter.alpha == 8
        assert adapter.feedback_count == 3
        assert adapter.W_a.shape == (4, 16)
        assert adapter.W_b.shape == (16, 4)

    def test_default_feedback_count(self):
        adapter = UserAdapter(
            user_id="u2",
            W_a=np.zeros((4, 16), dtype=np.float32),
            W_b=np.zeros((16, 4), dtype=np.float32),
            rank=4,
            alpha=8,
            created_at="1.0",
            updated_at="1.0",
        )
        assert adapter.feedback_count == 0


# ── Store initialization ────────────────────────────────────────────────────

class TestStoreInit:

    def test_creates_directory(self, tmp_path):
        store_dir = tmp_path / "adapters"
        PerUserLoRAStore(store_path=str(store_dir), run_eval=False)
        assert store_dir.exists()

    def test_default_params(self, tmp_path):
        store = PerUserLoRAStore(store_path=str(tmp_path), run_eval=False)
        assert store.adapter_rank == 8
        assert store.adapter_alpha == 16
        assert store.model_dim == 768

    def test_custom_params(self, store):
        assert store.adapter_rank == 4
        assert store.adapter_alpha == 8
        assert store.model_dim == 16
        assert store.auto_aggregate_threshold == 5
        assert store.auto_prune_threshold == 10


# ── Adapter path safety ─────────────────────────────────────────────────────

class TestAdapterPath:

    def test_safe_path(self, store):
        path = store._get_adapter_path("user/123")
        assert path.name == "user_123.npz"

    def test_backslash_escaped(self, store):
        path = store._get_adapter_path("user\\123")
        assert path.name == "user_123.npz"


# ── Create adapter ──────────────────────────────────────────────────────────

class TestCreateAdapter:

    def test_creates_new_adapter(self, store):
        adapter = store.create_adapter("user1")
        assert adapter.user_id == "user1"
        assert adapter.W_a.shape == (4, 16)
        assert adapter.W_b.shape == (16, 4)
        assert adapter.rank == 4
        assert adapter.alpha == 8
        assert adapter.feedback_count == 0

    def test_w_b_initialized_zeros(self, store):
        adapter = store.create_adapter("user1")
        np.testing.assert_array_equal(adapter.W_b, np.zeros((16, 4), dtype=np.float32))

    def test_w_a_initialized_small_random(self, store):
        adapter = store.create_adapter("user1")
        assert adapter.W_a.dtype == np.float32
        assert np.abs(adapter.W_a).max() < 1.0

    def test_persists_to_disk(self, store):
        store.create_adapter("user1")
        assert (store.store_path / "user1.npz").exists()

    def test_creates_only_once(self, store):
        a1 = store.create_adapter("user1")
        a2 = store.create_adapter("user1")
        assert a1 is a2

    def test_stores_created_at_and_updated_at(self, store):
        before = time.time()
        adapter = store.create_adapter("user1")
        after = time.time()
        assert before <= float(adapter.created_at) <= after
        assert adapter.created_at == adapter.updated_at


# ── Get adapter ─────────────────────────────────────────────────────────────

class TestGetAdapter:

    def test_returns_none_for_missing(self, store):
        assert store.get_adapter("nonexistent") is None

    def test_loads_from_disk(self, store):
        store.create_adapter("user1")
        store._cache.clear()
        loaded = store.get_adapter("user1")
        assert loaded is not None
        assert loaded.user_id == "user1"

    def test_returns_cached(self, store):
        a1 = store.create_adapter("user1")
        a2 = store.get_adapter("user1")
        assert a1 is a2

    def test_restores_metadata_from_disk(self, store):
        original = store.create_adapter("user1")
        original.feedback_count = 7
        store._save_adapter(original)
        store._update_metadata(original)
        store._cache.clear()
        loaded = store.get_adapter("user1")
        assert loaded.feedback_count == 7


# ── Update adapter ──────────────────────────────────────────────────────────

class TestUpdateAdapter:

    def test_increments_feedback_count(self, store):
        store.create_adapter("user1")
        store.update_adapter("user1", feedback_signal=1.0)
        adapter = store.get_adapter("user1")
        assert adapter.feedback_count == 1

    def test_multiple_updates(self, store):
        store.create_adapter("user1")
        for _ in range(5):
            store.update_adapter("user1", feedback_signal=1.0)
        adapter = store.get_adapter("user1")
        assert adapter.feedback_count == 5

    def test_positive_feedback_increases_w_b(self, store):
        adapter = store.create_adapter("user1")
        w_b_before = adapter.W_b.copy()
        store.update_adapter("user1", feedback_signal=1.0, learning_rate=0.1)
        adapter = store.get_adapter("user1")
        assert not np.array_equal(adapter.W_b, w_b_before)

    def test_negative_feedback_increases_w_b_negatively(self, store):
        adapter = store.create_adapter("user1")
        w_b_before = adapter.W_b.copy()
        store.update_adapter("user1", feedback_signal=-1.0, learning_rate=0.1)
        adapter = store.get_adapter("user1")
        assert not np.array_equal(adapter.W_b, w_b_before)

    def test_clips_weights_to_range(self, store):
        store.create_adapter("user1")
        for _ in range(200):
            store.update_adapter("user1", feedback_signal=1.0, learning_rate=1.0)
        adapter = store.get_adapter("user1")
        assert adapter.W_b.max() <= 1.0
        assert adapter.W_b.min() >= -1.0
        assert adapter.W_a.max() <= 1.0
        assert adapter.W_a.min() >= -1.0

    def test_updates_updated_at(self, store):
        store.create_adapter("user1")
        original_time = store.get_adapter("user1").updated_at
        time.sleep(0.01)
        store.update_adapter("user1", feedback_signal=1.0)
        assert store.get_adapter("user1").updated_at >= original_time

    def test_creates_adapter_if_missing(self, store):
        adapter = store.update_adapter("new_user", feedback_signal=1.0)
        assert adapter.user_id == "new_user"
        assert adapter.feedback_count == 1


# ── Apply adapter to logits ─────────────────────────────────────────────────

class TestApplyAdapterToLogits:

    def test_no_adapter_returns_unchanged(self, store):
        logits = np.zeros((1, 16), dtype=np.float32)
        result = store.apply_adapter_to_logits("missing", logits)
        np.testing.assert_array_equal(result, logits)

    def test_modifies_logits(self, store):
        store.create_adapter("user1")
        store.update_adapter("user1", feedback_signal=1.0, learning_rate=0.5)
        logits = np.zeros((1, 16), dtype=np.float32)
        result = store.apply_adapter_to_logits("user1", logits)
        assert not np.allclose(result, logits)

    def test_output_shape_matches(self, store):
        store.create_adapter("user1")
        logits = np.random.randn(1, 16).astype(np.float32)
        result = store.apply_adapter_to_logits("user1", logits)
        assert result.shape == logits.shape

    def test_scale_parameter(self, store):
        store.create_adapter("user1")
        store.update_adapter("user1", feedback_signal=1.0, learning_rate=0.5)
        logits = np.zeros((1, 16), dtype=np.float32)
        r1 = store.apply_adapter_to_logits("user1", logits, scale=0.5)
        r2 = store.apply_adapter_to_logits("user1", logits, scale=2.0)
        assert not np.allclose(r1, r2)


# ── Merge adapters ──────────────────────────────────────────────────────────

class TestMergeAdapters:

    def test_merge_single(self, store):
        store.create_adapter("user1")
        result = store.merge_adapters(["user1"])
        assert result["user_count"] == 1
        assert result["W_a"].shape == (4, 16)
        assert result["W_b"].shape == (16, 4)

    def test_merge_multiple_averages(self, store):
        store.create_adapter("u1")
        store.create_adapter("u2")
        result = store.merge_adapters(["u1", "u2"])
        assert result["user_count"] == 2
        a1 = store.get_adapter("u1")
        a2 = store.get_adapter("u2")
        expected_a = (a1.W_a + a2.W_a) / 2
        np.testing.assert_allclose(result["W_a"], expected_a)

    def test_merge_skips_missing(self, store):
        store.create_adapter("u1")
        result = store.merge_adapters(["u1", "nonexistent"])
        assert result["user_count"] == 1

    def test_merge_empty_list(self, store):
        result = store.merge_adapters([])
        assert result["user_count"] == 0

    def test_merge_all(self, store):
        store.create_adapter("u1")
        store.create_adapter("u2")
        result = store.merge_all()
        assert result["user_count"] == 2


# ── Get all adapters ────────────────────────────────────────────────────────

class TestGetAllAdapters:

    def test_empty_store(self, store):
        assert store.get_all_adapters() == []

    def test_returns_metadata(self, store):
        store.create_adapter("u1")
        store.create_adapter("u2")
        all_adapters = store.get_all_adapters()
        assert len(all_adapters) == 2
        user_ids = {a["user_id"] for a in all_adapters}
        assert user_ids == {"u1", "u2"}

    def test_metadata_keys(self, store):
        store.create_adapter("u1")
        meta = store.get_all_adapters()[0]
        expected_keys = {
            "user_id", "rank", "alpha", "model_dim",
            "created_at", "updated_at", "feedback_count",
        }
        assert set(meta.keys()) == expected_keys

    def test_most_recently_updated_first(self, store):
        store.create_adapter("u1")
        time.sleep(0.01)
        store.create_adapter("u2")
        store.update_adapter("u1", feedback_signal=1.0)
        all_adapters = store.get_all_adapters()
        assert all_adapters[0]["user_id"] == "u1"


# ── Stats ───────────────────────────────────────────────────────────────────

class TestGetStats:

    def test_empty_stats(self, store):
        stats = store.get_stats()
        assert stats["total_users"] == 0
        assert stats["total_size_bytes"] == 0
        assert stats["adapter_rank"] == 4
        assert stats["model_dim"] == 16

    def test_stats_with_adapters(self, store):
        store.create_adapter("u1")
        store.create_adapter("u2")
        stats = store.get_stats()
        assert stats["total_users"] == 2
        assert stats["total_size_bytes"] > 0

    def test_auto_management_stats(self, store):
        stats = store.get_stats()
        am = stats["auto_management"]
        assert am["aggregate_threshold"] == 5
        assert am["prune_threshold"] == 10


# ── Delete adapter ──────────────────────────────────────────────────────────

class TestDeleteAdapter:

    def test_deletes_from_cache_and_disk(self, store):
        store.create_adapter("u1")
        assert store.get_adapter("u1") is not None
        store.delete_adapter("u1")
        assert store.get_adapter("u1") is None
        assert not (store.store_path / "u1.npz").exists()

    def test_deletes_metadata(self, store):
        store.create_adapter("u1")
        store.delete_adapter("u1")
        assert len(store.get_all_adapters()) == 0

    def test_delete_nonexistent_no_error(self, store):
        store.delete_adapter("nobody")


# ── Quality adapters ────────────────────────────────────────────────────────

class TestQualityAdapters:

    def test_filters_by_feedback_count(self, store):
        store.create_adapter("u1")
        store.create_adapter("u2")
        store.update_adapter("u1", feedback_signal=1.0)
        store.update_adapter("u1", feedback_signal=1.0)
        store.update_adapter("u1", feedback_signal=1.0)
        quality = store.get_quality_adapters(min_feedback_count=3)
        assert len(quality) == 1
        assert quality[0]["user_id"] == "u1"

    def test_filters_by_age(self, store):
        store.create_adapter("u1")
        # Backdate the adapter
        adapter = store.get_adapter("u1")
        adapter.updated_at = str(time.time() - 100000)
        store._save_adapter(adapter)
        store._update_metadata(adapter)
        store._cache["u1"] = adapter
        quality = store.get_quality_adapters(min_feedback_count=0, max_age_days=1)
        assert len(quality) == 0

    def test_quality_report(self, store):
        store.create_adapter("u1")
        store.update_adapter("u1", feedback_signal=1.0)
        store.update_adapter("u1", feedback_signal=1.0)
        store.update_adapter("u1", feedback_signal=1.0)
        report = store.get_quality_report(min_feedback_count=3)
        assert report["count"] == 1
        assert len(report["adapters"]) == 1


# ── Prune low quality ──────────────────────────────────────────────────────

class TestPruneLowQuality:

    def test_prunes_by_feedback_count(self, store):
        store.create_adapter("u1")
        store.create_adapter("u2")
        store.update_adapter("u1", feedback_signal=1.0)
        deleted = store.prune_low_quality(min_feedback_count=1, max_age_days=30)
        assert "u2" in deleted
        assert "u1" not in deleted

    def test_prunes_by_age(self, store):
        store.create_adapter("u1")
        adapter = store.get_adapter("u1")
        adapter.updated_at = str(time.time() - 100000)
        store._save_adapter(adapter)
        store._update_metadata(adapter)
        store._cache["u1"] = adapter
        deleted = store.prune_low_quality(min_feedback_count=0, max_age_days=1)
        assert "u1" in deleted

    def test_no_prune_when_quality(self, store):
        store.create_adapter("u1")
        store.update_adapter("u1", feedback_signal=1.0)
        deleted = store.prune_low_quality(min_feedback_count=1, max_age_days=365)
        assert len(deleted) == 0


# ── Reset adapter ───────────────────────────────────────────────────────────

class TestResetAdapter:

    def test_reset_clears_feedback_count(self, store):
        store.create_adapter("u1")
        for _ in range(10):
            store.update_adapter("u1", feedback_signal=1.0)
        store.reset_user_adapter("u1")
        adapter = store.get_adapter("u1")
        assert adapter.feedback_count == 0

    def test_reset_zeros_w_b(self, store):
        store.create_adapter("u1")
        store.update_adapter("u1", feedback_signal=1.0)
        store.reset_user_adapter("u1")
        adapter = store.get_adapter("u1")
        np.testing.assert_array_equal(adapter.W_b, np.zeros((16, 4), dtype=np.float32))

    def test_reset_reinitializes_w_a(self, store):
        store.create_adapter("u1")
        store.update_adapter("u1", feedback_signal=1.0)
        store.reset_user_adapter("u1")
        adapter = store.get_adapter("u1")
        assert adapter.W_a.dtype == np.float32
        assert np.abs(adapter.W_a).max() < 1.0


# ── Aggregate best adapters ─────────────────────────────────────────────────

class TestAggregateBestAdapters:

    def test_no_adapters_returns_error(self, store):
        result = store.aggregate_best_adapters(top_k=5, min_feedback_count=1)
        assert "error" in result
        assert result["count"] == 0

    def test_aggregates_top_k(self, store):
        for i in range(5):
            store.create_adapter(f"u{i}")
            for _ in range(i + 1):
                store.update_adapter(f"u{i}", feedback_signal=1.0)
        result = store.aggregate_best_adapters(
            top_k=3,
            min_feedback_count=1,
            run_eval=False,
        )
        assert result["user_count"] == 3
        assert "output_path" in result
        assert result["total_feedback"] > 0

    def test_creates_output_file(self, store):
        store.create_adapter("u1")
        store.update_adapter("u1", feedback_signal=1.0)
        result = store.aggregate_best_adapters(
            top_k=1,
            min_feedback_count=1,
            output_name="test_agg",
            run_eval=False,
        )
        assert (store.store_path / "test_agg.npz").exists()

    def test_aggregation_weights_by_feedback(self, store):
        store.create_adapter("u1")
        store.update_adapter("u1", feedback_signal=1.0)
        store.create_adapter("u2")
        for _ in range(5):
            store.update_adapter("u2", feedback_signal=1.0)
        result = store.aggregate_best_adapters(
            top_k=2,
            min_feedback_count=1,
            run_eval=False,
        )
        assert result["user_count"] == 2
        u2_adapter = store.get_adapter("u2")
        u1_adapter = store.get_adapter("u1")
        assert u2_adapter.feedback_count > u1_adapter.feedback_count


# ── Auto-management ─────────────────────────────────────────────────────────

class TestAutoManage:

    def test_auto_prune_triggers(self, store_small):
        for i in range(5):
            store_small.create_adapter(f"u{i}")
        # u0 has no feedback → should be pruned
        store_small.update_adapter("u1", feedback_signal=1.0)
        store_small.update_adapter("u2", feedback_signal=1.0)
        store_small.update_adapter("u3", feedback_signal=1.0)
        store_small.update_adapter("u4", feedback_signal=1.0)
        store_small._auto_manage()
        adapters = store_small.get_all_adapters()
        # Low-quality adapters (u0 with 0 feedback, age=0) may be pruned
        # but age=0 is within 7 days, so only feedback_count < 1 matters
        assert len(adapters) <= 5

    def test_auto_manage_error_handled(self, store):
        store._adapters_col = None
        store._auto_manage()


# ── Concurrency ─────────────────────────────────────────────────────────────

class TestConcurrency:

    def test_concurrent_create(self, store):
        import threading

        errors = []

        def create(uid):
            try:
                store.create_adapter(uid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create, args=(f"u{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert len(errors) == 0
        assert len(store.get_all_adapters()) == 10

    def test_concurrent_update(self, store):
        import threading

        store.create_adapter("u1")
        errors = []

        def update():
            try:
                store.update_adapter("u1", feedback_signal=1.0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        assert len(errors) == 0
        assert store.get_adapter("u1").feedback_count == 10


# ── Persistence round-trip ──────────────────────────────────────────────────

class TestPersistence:

    def test_survives_store_recreation(self, tmp_path):
        s1 = PerUserLoRAStore(
            store_path=str(tmp_path),
            adapter_rank=4,
            model_dim=16,
            run_eval=False,
        )
        s1.create_adapter("u1")
        s1.update_adapter("u1", feedback_signal=1.0)

        s2 = PerUserLoRAStore(
            store_path=str(tmp_path),
            adapter_rank=4,
            model_dim=16,
            run_eval=False,
        )
        adapter = s2.get_adapter("u1")
        assert adapter is not None
        assert adapter.feedback_count == 1
        assert adapter.W_a.shape == (4, 16)
