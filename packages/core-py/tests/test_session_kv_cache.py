"""Tests for domains.infrastructure.model_server — SessionKVCache."""

import time
import threading
import pytest
from domains.infrastructure.model_server import SessionKVCache


class TestSessionKVCache:
    def test_init(self):
        skc = SessionKVCache()
        assert skc._max_sessions == 20
        assert skc._ttl == 600.0

    def test_get_empty(self):
        skc = SessionKVCache()
        result, prefix = skc.get("s1", [1, 2, 3])
        assert result is None
        assert prefix == 0

    def test_store_and_get(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2, 3], "pkv_data")
        result, prefix = skc.get("s1", [1, 2, 3, 4])
        assert result == "pkv_data"
        assert prefix == 3

    def test_prefix_mismatch(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2, 3], "pkv_data")
        result, prefix = skc.get("s1", [9, 8, 7])
        assert result is None
        assert prefix == 0

    def test_exact_match(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2, 3], "data")
        result, prefix = skc.get("s1", [1, 2, 3])
        assert result == "data"
        assert prefix == 3

    def test_partial_prefix(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2, 3, 4, 5], "data")
        result, prefix = skc.get("s1", [1, 2, 3])
        assert result == "data"
        assert prefix == 3

    def test_no_prefix(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2, 3], "data")
        result, prefix = skc.get("s1", [4, 5, 6])
        assert result is None
        assert prefix == 0

    def test_multiple_sessions(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2, 3], "data1")
        skc.store("s2", [4, 5, 6], "data2")
        r1, p1 = skc.get("s1", [1, 2, 3, 4])
        r2, p2 = skc.get("s2", [4, 5, 6, 7])
        assert r1 == "data1"
        assert r2 == "data2"
        assert p1 == 3
        assert p2 == 3

    def test_store_overwrite(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2, 3], "old")
        skc.store("s1", [1, 2, 3], "new")
        result, prefix = skc.get("s1", [1, 2, 3, 4])
        assert result == "new"
        assert prefix == 3

    def test_clear(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2, 3], "data")
        skc.clear("s1")
        result, prefix = skc.get("s1", [1, 2, 3])
        assert result is None
        assert prefix == 0

    def test_clear_nonexistent(self):
        skc = SessionKVCache()
        skc.clear("nonexistent")

    def test_size(self):
        skc = SessionKVCache()
        assert skc.size == 0
        skc.store("s1", [1], "d1")
        assert skc.size == 1
        skc.store("s2", [2], "d2")
        assert skc.size == 2

    def test_stats(self):
        skc = SessionKVCache(max_sessions=10, ttl=300.0)
        stats = skc.stats()
        assert stats["entries"] == 0
        assert stats["max_sessions"] == 10
        assert stats["ttl_seconds"] == 300.0

    def test_stats_with_entries(self):
        skc = SessionKVCache()
        skc.store("s1", [1], "d1")
        skc.store("s2", [2], "d2")
        stats = skc.stats()
        assert stats["entries"] == 2

    def test_lru_eviction(self):
        skc = SessionKVCache(max_sessions=2)
        skc.store("s1", [1], "d1")
        skc.store("s2", [2], "d2")
        skc.store("s3", [3], "d3")
        assert skc.size == 2
        r1, _ = skc.get("s1", [1])
        assert r1 is None

    def test_lru_evicts_oldest(self):
        skc = SessionKVCache(max_sessions=3)
        skc.store("s1", [1], "d1")
        time.sleep(0.01)
        skc.store("s2", [2], "d2")
        time.sleep(0.01)
        skc.store("s3", [3], "d3")
        skc.store("s4", [4], "d4")
        assert skc.size == 3
        r1, _ = skc.get("s1", [1])
        assert r1 is None
        r2, _ = skc.get("s2", [2])
        assert r2 == "d2"

    def test_ttl_expiry(self):
        skc = SessionKVCache(ttl=0.01)
        skc.store("s1", [1, 2], "data")
        time.sleep(0.05)
        skc.evict_expired()
        result, prefix = skc.get("s1", [1, 2])
        assert result is None
        assert prefix == 0

    def test_ttl_not_expired(self):
        skc = SessionKVCache(ttl=10.0)
        skc.store("s1", [1, 2], "data")
        result, prefix = skc.get("s1", [1, 2])
        assert result == "data"
        assert prefix == 2

    def test_evict_expired(self):
        skc = SessionKVCache(ttl=0.01)
        skc.store("s1", [1], "d1")
        skc.store("s2", [2], "d2")
        time.sleep(0.05)
        skc.evict_expired()
        assert skc.size == 0

    def test_partial_prefix_only(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2, 3, 4, 5], "data")
        result, prefix = skc.get("s1", [1, 2])
        assert result == "data"
        assert prefix == 2

    def test_first_element_mismatch(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2, 3], "data")
        result, prefix = skc.get("s1", [2, 3, 4])
        assert result is None
        assert prefix == 0

    def test_empty_token_ids(self):
        skc = SessionKVCache()
        skc.store("s1", [], "data")
        result, prefix = skc.get("s1", [1, 2])
        assert result is None
        assert prefix == 0

    def test_empty_current_ids(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2], "data")
        result, prefix = skc.get("s1", [])
        assert result is None
        assert prefix == 0

    def test_large_session_id(self):
        skc = SessionKVCache()
        long_id = "s" * 1000
        skc.store(long_id, [1, 2, 3], "data")
        result, prefix = skc.get(long_id, [1, 2, 3, 4])
        assert result == "data"
        assert prefix == 3

    def test_store_different_tokens_same_session(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2, 3], "data_old")
        skc.store("s1", [1, 2, 3, 4, 5], "data_new")
        result, prefix = skc.get("s1", [1, 2, 3, 4, 5, 6])
        assert result == "data_new"
        assert prefix == 5

    def test_concurrent_get_store(self):
        import threading
        skc = SessionKVCache()
        results = []

        def writer():
            for i in range(50):
                skc.store(f"s{i}", [i], f"d{i}")

        def reader():
            for i in range(50):
                skc.get(f"s{i}", [i])

        threads = [threading.Thread(target=writer) for _ in range(3)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def test_many_sessions_lru(self):
        skc = SessionKVCache(max_sessions=5)
        for i in range(10):
            skc.store(f"s{i}", [i], f"d{i}")
        assert skc.size == 5
        for i in range(5):
            result, _ = skc.get(f"s{i}", [i])
            assert result is None
        for i in range(5, 10):
            result, _ = skc.get(f"s{i}", [i])
            assert result == f"d{i}"

    def test_zero_max_sessions(self):
        skc = SessionKVCache(max_sessions=0)
        with pytest.raises(ValueError):
            skc.store("s1", [1], "data")

    def test_one_max_session(self):
        skc = SessionKVCache(max_sessions=1)
        skc.store("s1", [1], "d1")
        assert skc.size == 1
        skc.store("s2", [2], "d2")
        assert skc.size == 1
        r1, _ = skc.get("s1", [1])
        assert r1 is None
        r2, _ = skc.get("s2", [2])
        assert r2 == "d2"

    def test_get_returns_different_data_per_session(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2], "data_a")
        skc.store("s2", [1, 2], "data_b")
        r1, _ = skc.get("s1", [1, 2])
        r2, _ = skc.get("s2", [1, 2])
        assert r1 == "data_a"
        assert r2 == "data_b"

    def test_prefix_length_one(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2, 3, 4], "data")
        result, prefix = skc.get("s1", [1, 99, 99])
        assert result == "data"
        assert prefix == 1

    def test_store_list_data(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2], [10, 20, 30])
        result, prefix = skc.get("s1", [1, 2, 3])
        assert result == [10, 20, 30]
        assert prefix == 2

    def test_store_dict_data(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2], {"key": "value"})
        result, prefix = skc.get("s1", [1, 2])
        assert result == {"key": "value"}
        assert prefix == 2

    def test_evict_expired_only_removes_stale(self):
        skc = SessionKVCache(ttl=0.01)
        skc.store("s1", [1], "d1")
        time.sleep(0.05)
        skc.store("s2", [2], "d2")
        skc.evict_expired()
        assert skc.size == 1
        r2, _ = skc.get("s2", [2])
        assert r2 == "d2"

    def test_size_thread_safe(self):
        import threading
        skc = SessionKVCache(max_sessions=100)
        errors = []

        def writer(start):
            try:
                for i in range(start, start + 20):
                    skc.store(f"s{i}", [i], f"d{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i * 20,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert skc.size <= 100
        assert len(errors) == 0

    def test_custom_ttl_and_max(self):
        skc = SessionKVCache(max_sessions=5, ttl=1.0)
        assert skc._max_sessions == 5
        assert skc._ttl == 1.0
        for i in range(5):
            skc.store(f"s{i}", [i], f"d{i}")
        assert skc.size == 5

    def test_prefix_with_repeated_tokens(self):
        skc = SessionKVCache()
        skc.store("s1", [5, 5, 5, 5], "data")
        result, prefix = skc.get("s1", [5, 5, 5, 5, 5])
        assert result == "data"
        assert prefix == 4

    def test_prefix_stops_at_mismatch(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2, 3, 4, 5], "data")
        result, prefix = skc.get("s1", [1, 2, 99, 4, 5])
        assert result == "data"
        assert prefix == 2

    def test_long_token_sequence(self):
        skc = SessionKVCache()
        ids = list(range(1000))
        skc.store("s1", ids, "big_data")
        result, prefix = skc.get("s1", ids + [1000, 1001])
        assert result == "big_data"
        assert prefix == 1000

    def test_concurrent_store_get_stress(self):
        skc = SessionKVCache(max_sessions=200)
        errors = []

        def writer(start):
            try:
                for i in range(start, start + 30):
                    skc.store(f"s{i}", [i], f"d{i}")
            except Exception as e:
                errors.append(e)

        def reader(start):
            try:
                for i in range(start, start + 30):
                    skc.get(f"s{i}", [i])
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, args=(i * 30,)))
            threads.append(threading.Thread(target=reader, args=(i * 30,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_clear_one_preserves_others(self):
        skc = SessionKVCache()
        skc.store("s1", [1], "d1")
        skc.store("s2", [2], "d2")
        skc.store("s3", [3], "d3")
        skc.clear("s2")
        assert skc.size == 2
        r1, _ = skc.get("s1", [1])
        r3, _ = skc.get("s3", [3])
        assert r1 == "d1"
        assert r3 == "d3"

    def test_store_updates_timestamp(self):
        skc = SessionKVCache()
        skc.store("s1", [1], "old")
        skc.store("s1", [1, 2], "new")
        result, prefix = skc.get("s1", [1, 2, 3])
        assert result == "new"
        assert prefix == 2

    def test_get_after_clear_returns_none(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2, 3], "data")
        skc.clear("s1")
        result, prefix = skc.get("s1", [1, 2, 3])
        assert result is None
        assert prefix == 0

    def test_many_sessions_fill_and_evict(self):
        skc = SessionKVCache(max_sessions=3)
        for i in range(10):
            skc.store(f"s{i}", [i], f"d{i}")
        assert skc.size == 3
        for i in range(7):
            r, _ = skc.get(f"s{i}", [i])
            assert r is None
        for i in range(7, 10):
            r, _ = skc.get(f"s{i}", [i])
            assert r == f"d{i}"

    def test_stats_after_operations(self):
        skc = SessionKVCache(max_sessions=5, ttl=2.0)
        skc.store("s1", [1], "d1")
        skc.store("s2", [2], "d2")
        skc.clear("s1")
        stats = skc.stats()
        assert stats["entries"] == 1
        assert stats["max_sessions"] == 5
        assert stats["ttl_seconds"] == 2.0

    def test_none_data_stored(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2], None)
        result, prefix = skc.get("s1", [1, 2, 3])
        assert result is None
        assert prefix == 2

    def test_empty_session_id(self):
        skc = SessionKVCache()
        skc.store("", [1, 2], "data")
        result, prefix = skc.get("", [1, 2, 3])
        assert result == "data"
        assert prefix == 2

    def test_non_overlapping_sessions(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2, 3], "data1")
        skc.store("s2", [4, 5, 6], "data2")
        r1, p1 = skc.get("s1", [1, 2, 3])
        r2, p2 = skc.get("s2", [4, 5, 6])
        assert r1 == "data1" and p1 == 3
        assert r2 == "data2" and p2 == 3

    def test_overlapping_different_prefixes(self):
        skc = SessionKVCache()
        skc.store("s1", [1, 2, 3], "data1")
        skc.store("s2", [1, 2, 4], "data2")
        r1, p1 = skc.get("s1", [1, 2, 3, 5])
        r2, p2 = skc.get("s2", [1, 2, 4, 5])
        assert r1 == "data1" and p1 == 3
        assert r2 == "data2" and p2 == 3

    def test_lru_eviction_preserves_recent(self):
        skc = SessionKVCache(max_sessions=3)
        skc.store("s1", [1], "d1")
        time.sleep(0.01)
        skc.store("s2", [2], "d2")
        time.sleep(0.01)
        skc.store("s3", [3], "d3")
        skc.get("s2", [2])
        skc.store("s4", [4], "d4")
        r2, _ = skc.get("s2", [2])
        assert r2 == "d2"
        r1, _ = skc.get("s1", [1])
        assert r1 is None

    def test_large_data_object(self):
        skc = SessionKVCache()
        big_data = list(range(10000))
        skc.store("s1", [1, 2], big_data)
        result, prefix = skc.get("s1", [1, 2, 3])
        assert result == big_data
        assert prefix == 2

    def test_clear_all(self):
        skc = SessionKVCache()
        skc.store("s1", [1], "d1")
        skc.store("s2", [2], "d2")
        removed = skc.clear_all()
        assert removed == 2
        assert skc.size == 0

    def test_clear_all_empty(self):
        skc = SessionKVCache()
        removed = skc.clear_all()
        assert removed == 0

    def test_clear_all_returns_correct_count(self):
        skc = SessionKVCache()
        for i in range(5):
            skc.store(f"s{i}", [i], f"d{i}")
        removed = skc.clear_all()
        assert removed == 5
