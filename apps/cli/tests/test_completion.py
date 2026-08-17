"""Tests for apps/cli/src/core/completion.py — TTL cache and fetchers."""
import sys
import os
import time
import pytest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestCompletionCache:
    def test_cache_hit(self):
        from core.completion import CompletionCache
        cache = CompletionCache(ttl_sec=60)
        fetcher = MagicMock(return_value=["a", "b"])
        result = cache.get("key", fetcher)
        assert result == ["a", "b"]
        fetcher.assert_called_once()

    def test_cache_returns_cached_within_ttl(self):
        from core.completion import CompletionCache
        cache = CompletionCache(ttl_sec=60)
        fetcher = MagicMock(return_value=["a"])
        cache.get("key", fetcher)
        cache.get("key", fetcher)
        fetcher.assert_called_once()

    def test_cache_refreshes_after_ttl(self):
        from core.completion import CompletionCache
        cache = CompletionCache(ttl_sec=0.01)
        fetcher = MagicMock(side_effect=[["a"], ["b"]])
        cache.get("key", fetcher)
        time.sleep(0.02)
        result = cache.get("key", fetcher)
        assert result == ["b"]
        assert fetcher.call_count == 2

    def test_cache_stale_on_error(self):
        from core.completion import CompletionCache
        cache = CompletionCache(ttl_sec=0.01)
        fetcher_ok = MagicMock(return_value=["cached"])
        cache.get("key", fetcher_ok)
        time.sleep(0.02)
        fetcher_fail = MagicMock(side_effect=Exception("network error"))
        result = cache.get("key", fetcher_fail)
        assert result == ["cached"]

    def test_cache_empty_on_first_error(self):
        from core.completion import CompletionCache
        cache = CompletionCache(ttl_sec=60)
        fetcher = MagicMock(side_effect=Exception("network error"))
        result = cache.get("key", fetcher)
        assert result == []

    def test_invalidate_single_key(self):
        from core.completion import CompletionCache
        cache = CompletionCache(ttl_sec=60)
        fetcher = MagicMock(return_value=["a"])
        cache.get("key", fetcher)
        cache.invalidate("key")
        cache.get("key", fetcher)
        assert fetcher.call_count == 2

    def test_invalidate_all(self):
        from core.completion import CompletionCache
        cache = CompletionCache(ttl_sec=60)
        f1 = MagicMock(return_value=["a"])
        f2 = MagicMock(return_value=["b"])
        cache.get("k1", f1)
        cache.get("k2", f2)
        cache.invalidate()
        cache.get("k1", f1)
        cache.get("k2", f2)
        assert f1.call_count == 2
        assert f2.call_count == 2


class TestCompletePaths:
    def test_empty_prefix(self, tmp_path):
        from core.completion import complete_paths
        (tmp_path / "testfile.txt").touch()
        result = complete_paths("")
        # Should return items from current dir
        assert isinstance(result, list)

    def test_dot_prefix(self):
        from core.completion import complete_paths
        result = complete_paths(".")
        assert isinstance(result, list)

    def test_nonexistent_dir(self):
        from core.completion import complete_paths
        result = complete_paths("/nonexistent/path/to/dir/")
        assert result == []

    def test_directory_suffix(self, tmp_path):
        from core.completion import complete_paths
        d = tmp_path / "mydir"
        d.mkdir()
        # Test with absolute path ending in /
        result = complete_paths(str(d) + "/")
        assert isinstance(result, list)


class TestCommandCompleters:
    def test_all_keys_are_strings(self):
        from core.completion import COMMAND_COMPLETERS
        for key in COMMAND_COMPLETERS:
            assert isinstance(key, str)

    def test_all_values_are_callables(self):
        from core.completion import COMMAND_COMPLETERS
        for key, fetcher in COMMAND_COMPLETERS.items():
            assert callable(fetcher)

    def test_known_commands_present(self):
        from core.completion import COMMAND_COMPLETERS
        assert "load" in COMMAND_COMPLETERS
        assert "train" in COMMAND_COMPLETERS
        assert "model" in COMMAND_COMPLETERS


class TestGetCompletionsForCommand:
    def test_known_command(self):
        from core.completion import get_completions_for_command
        result = get_completions_for_command("train")
        assert isinstance(result, list)
        assert "status" in result

    def test_unknown_command_returns_paths(self):
        from core.completion import get_completions_for_command
        result = get_completions_for_command("nonexistent")
        assert isinstance(result, list)


class TestGetCache:
    def test_returns_singleton(self):
        from core.completion import get_cache, CompletionCache
        cache = get_cache()
        assert isinstance(cache, CompletionCache)

    def test_same_instance(self):
        from core.completion import get_cache
        assert get_cache() is get_cache()
