"""Tests for domains.infrastructure.model_server — SessionKVCache."""

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
