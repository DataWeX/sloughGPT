"""Tests for domains.shared — utils, feature_flags, test_framework."""

import json
import time
import pytest
from pathlib import Path

from domains.shared.utils import (
    generate_id,
    hash_string,
    format_size,
    format_time,
    load_json,
    save_json,
    merge_dicts,
    clamp,
    retry,
    Timer,
    Cache,
    RateLimiter,
    validate_config,
    get_timestamp,
    find_available_port,
    find_server_python,
)
from domains.shared.feature_flags import (
    FlagStatus,
    FeatureFlag,
    FeatureFlags,
    is_enabled,
)
from domains.shared.test_framework import (
    TestResult,
    TestSuite,
    TestFramework,
    BenchmarkRunner,
    mark_test,
)


# ===================================================================
# utils.py
# ===================================================================

class TestGenerateId:
    def test_length(self):
        assert len(generate_id()) == 8

    def test_prefix(self):
        result = generate_id("usr_")
        assert result.startswith("usr_")
        assert len(result) == 12

    def test_no_prefix(self):
        result = generate_id()
        assert result.isalnum()
        assert result.isascii()

    def test_uniqueness(self):
        ids = {generate_id() for _ in range(100)}
        assert len(ids) == 100


class TestHashString:
    def test_sha256(self):
        h = hash_string("hello")
        assert len(h) == 64
        assert h == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    def test_md5(self):
        h = hash_string("hello", "md5")
        assert len(h) == 32

    def test_sha1(self):
        h = hash_string("hello", "sha1")
        assert len(h) == 40

    def test_unknown_returns_input(self):
        assert hash_string("abc", "unknown") == "abc"

    def test_empty_string(self):
        h = hash_string("")
        assert len(h) == 64


class TestFormatSize:
    @pytest.mark.parametrize("size,expected", [
        (0, "0.0 B"),
        (512, "512.0 B"),
        (1024, "1.0 KB"),
        (1024 * 1024, "1.0 MB"),
        (1024 ** 3, "1.0 GB"),
        (1024 ** 4, "1.0 TB"),
        (1024 ** 5, "1.0 PB"),
    ])
    def test_units(self, size, expected):
        assert format_size(size) == expected

    def test_partial(self):
        assert format_size(1536) == "1.5 KB"


class TestFormatTime:
    @pytest.mark.parametrize("seconds,expected", [
        (5.0, "5.0s"),
        (65.0, "1.1m"),
        (3700.0, "1.0h"),
        (90000.0, "1.0d"),
    ])
    def test_units(self, seconds, expected):
        assert format_time(seconds) == expected


class TestJsonIO:
    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "test.json")
        data = {"key": [1, 2, 3]}
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_custom_indent(self, tmp_path):
        path = str(tmp_path / "indented.json")
        save_json({"a": 1}, path, indent=4)
        with open(path) as f:
            content = f.read()
        assert "    " in content


class TestMergeDicts:
    def test_empty(self):
        assert merge_dicts() == {}

    def test_single(self):
        assert merge_dicts({"a": 1}) == {"a": 1}

    def test_overwrite(self):
        result = merge_dicts({"a": 1}, {"a": 2})
        assert result == {"a": 2}

    def test_multiple(self):
        result = merge_dicts({"a": 1}, {"b": 2}, {"c": 3})
        assert result == {"a": 1, "b": 2, "c": 3}


class TestClamp:
    @pytest.mark.parametrize("value,min_val,max_val,expected", [
        (5, 0, 10, 5),
        (-1, 0, 10, 0),
        (11, 0, 10, 10),
        (5, 5, 5, 5),
        (0, 0, 0, 0),
    ])
    def test_clamp(self, value, min_val, max_val, expected):
        assert clamp(value, min_val, max_val) == expected


class TestRetry:
    def test_succeeds_first_try(self):
        call_count = 0

        @retry(max_attempts=3, delay=0)
        def func():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert func() == "ok"
        assert call_count == 1

    def test_retries_then_succeeds(self):
        call_count = 0

        @retry(max_attempts=3, delay=0)
        def func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "done"

        assert func() == "done"
        assert call_count == 3

    def test_raises_after_max_attempts(self):
        @retry(max_attempts=2, delay=0)
        def func():
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            func()


class TestTimer:
    def test_context_manager(self):
        with Timer() as t:
            time.sleep(0.01)
        assert t.elapsed is not None
        assert t.elapsed > 0

    def test_start_end_set(self):
        with Timer() as t:
            pass
        assert t.start is not None
        assert t.end is not None
        assert t.end >= t.start


class TestCache:
    def test_get_set(self):
        c = Cache()
        c.set("k", "v")
        assert c.get("k") == "v"

    def test_get_missing(self):
        c = Cache()
        assert c.get("missing") is None

    def test_len(self):
        c = Cache()
        assert len(c) == 0
        c.set("a", 1)
        c.set("b", 2)
        assert len(c) == 2

    def test_eviction(self):
        c = Cache(max_size=3)
        for i in range(5):
            c.set(str(i), i)
        assert len(c) == 3
        assert c.get("0") is None
        assert c.get("1") is None
        assert c.get("4") == 4

    def test_clear(self):
        c = Cache()
        c.set("k", "v")
        c.clear()
        assert len(c) == 0

    def test_overwrite(self):
        c = Cache()
        c.set("k", 1)
        c.set("k", 2)
        assert c.get("k") == 2
        assert len(c) == 1


class TestRateLimiter:
    def test_allows_within_limit(self):
        rl = RateLimiter(max_calls=3, period=1.0)

        @rl
        def fn():
            return "ok"

        assert fn() == "ok"
        assert fn() == "ok"
        assert fn() == "ok"

    def test_raises_on_exceed(self):
        rl = RateLimiter(max_calls=2, period=1.0)

        @rl
        def fn():
            return "ok"

        fn()
        fn()
        with pytest.raises(Exception, match="Rate limit exceeded"):
            fn()


class TestValidateConfig:
    def test_valid(self):
        assert validate_config({"a": 1, "b": 2}, ["a", "b"]) is True

    def test_missing_key(self):
        assert validate_config({"a": 1}, ["a", "b"]) is False

    def test_empty(self):
        assert validate_config({}, []) is True


class TestGetTimestamp:
    def test_returns_string(self):
        assert isinstance(get_timestamp(), str)

    def test_contains_t(self):
        ts = get_timestamp()
        assert "T" in ts


class TestFindAvailablePort:
    def test_finds_port(self):
        port = find_available_port(start_port=20000, max_attempts=100)
        assert isinstance(port, int)
        assert 20000 <= port < 20100


class TestFindServerPython:
    def test_returns_string(self):
        result = find_server_python()
        assert isinstance(result, str)
        assert "python" in result.lower()


# ===================================================================
# feature_flags.py
# ===================================================================

class TestFlagStatus:
    def test_values(self):
        assert FlagStatus.ENABLED.value == "enabled"
        assert FlagStatus.DISABLED.value == "disabled"
        assert FlagStatus.EXPERIMENTAL.value == "experimental"


class TestFeatureFlag:
    def test_auto_env_var(self):
        f = FeatureFlag(name="my_flag", description="test")
        assert f.env_var == "SLO_FF_MY_FLAG"

    def test_custom_env_var(self):
        f = FeatureFlag(name="x", description="", env_var="CUSTOM")
        assert f.env_var == "CUSTOM"

    def test_is_enabled_status_enabled(self):
        f = FeatureFlag(name="t", description="", status=FlagStatus.ENABLED)
        assert f.is_enabled is True

    def test_is_enabled_status_disabled(self):
        f = FeatureFlag(name="t", description="", status=FlagStatus.DISABLED)
        assert f.is_enabled is False

    def test_is_enabled_status_experimental(self):
        f = FeatureFlag(name="t", description="", status=FlagStatus.EXPERIMENTAL)
        assert f.is_enabled is True


class TestFeatureFlagsRegistry:
    def setup_method(self):
        FeatureFlags._flags.clear()

    def teardown_method(self):
        FeatureFlags._flags.clear()
        _register_defaults_for_tests()

    def test_register(self):
        FeatureFlags.register("test_flag", description="desc")
        assert "test_flag" in FeatureFlags._flags

    def test_register_duplicate_returns_existing(self):
        f1 = FeatureFlags.register("dup", description="a")
        f2 = FeatureFlags.register("dup", description="b")
        assert f1 is f2

    def test_is_enabled_unknown(self):
        assert FeatureFlags.is_enabled("nonexistent") is False

    def test_set_status(self):
        FeatureFlags.register("s", status=FlagStatus.DISABLED)
        FeatureFlags.set_status("s", FlagStatus.ENABLED)
        assert FeatureFlags.is_enabled("s") is True

    def test_set_status_unknown_raises(self):
        with pytest.raises(KeyError):
            FeatureFlags.set_status("nope", FlagStatus.ENABLED)

    def test_list_all(self):
        FeatureFlags.register("a", description="A", status=FlagStatus.ENABLED)
        listing = FeatureFlags.list_all()
        assert "a" in listing
        assert listing["a"]["description"] == "A"
        assert listing["a"]["status"] == "enabled"

    def test_save_and_load_config(self, tmp_path):
        FeatureFlags.register("save_test", description="", status=FlagStatus.ENABLED)
        config_path = tmp_path / "flags.json"
        FeatureFlags.save_config(config_path)

        FeatureFlags.set_status("save_test", FlagStatus.DISABLED)
        assert FeatureFlags._flags["save_test"].status == FlagStatus.DISABLED

        FeatureFlags.load_config(config_path)
        assert FeatureFlags._flags["save_test"].status == FlagStatus.ENABLED

    def test_load_nonexistent_config(self, tmp_path):
        FeatureFlags.register("x", status=FlagStatus.ENABLED)
        FeatureFlags.load_config(tmp_path / "nope.json")
        assert FeatureFlags._flags["x"].status == FlagStatus.ENABLED


class TestConvenienceIsEnabled:
    def test_calls_registry(self):
        FeatureFlags._flags.clear()
        FeatureFlags.register("conv", status=FlagStatus.ENABLED)
        assert is_enabled("conv") is True


class TestRegisterDefaults:
    def test_defaults_registered(self):
        _register_defaults_for_tests()
        assert FeatureFlags.is_enabled("slonet_provider") is True
        assert FeatureFlags.is_enabled("native_c_inference") is False
        assert FeatureFlags.is_enabled("multimodal") is True
        assert FeatureFlags.is_enabled("soul_format") is True
        assert FeatureFlags.is_enabled("feature_flags") is True


def _register_defaults_for_tests():
    """Re-register defaults after test teardown clears _flags."""
    FeatureFlags._flags.clear()
    FeatureFlags.register("slonet_provider", status=FlagStatus.ENABLED)
    FeatureFlags.register("native_c_inference", status=FlagStatus.DISABLED)
    FeatureFlags.register("cloud_vector_store", status=FlagStatus.DISABLED)
    FeatureFlags.register("soul_format", status=FlagStatus.ENABLED)
    FeatureFlags.register("soul_manager", status=FlagStatus.ENABLED)
    FeatureFlags.register("slonet_kernels", status=FlagStatus.EXPERIMENTAL)
    FeatureFlags.register("multimodal", status=FlagStatus.ENABLED)
    FeatureFlags.register("cross_attention", status=FlagStatus.ENABLED)
    FeatureFlags.register("kv_cache", status=FlagStatus.ENABLED)
    FeatureFlags.register("session_kv_cache", status=FlagStatus.ENABLED)
    FeatureFlags.register("model_server", status=FlagStatus.ENABLED)
    FeatureFlags.register("model_registry", status=FlagStatus.ENABLED)
    FeatureFlags.register("process_isolation", status=FlagStatus.ENABLED)
    FeatureFlags.register("on_device_training", status=FlagStatus.EXPERIMENTAL)
    FeatureFlags.register("quantization", status=FlagStatus.ENABLED)
    FeatureFlags.register("hf_finetune", status=FlagStatus.DISABLED)
    FeatureFlags.register("vlm", status=FlagStatus.EXPERIMENTAL)
    FeatureFlags.register("dpo", status=FlagStatus.EXPERIMENTAL)
    FeatureFlags.register("context_managers", status=FlagStatus.ENABLED)
    FeatureFlags.register("knowledge_memory", status=FlagStatus.ENABLED)
    FeatureFlags.register("semantic_cache", status=FlagStatus.ENABLED)
    FeatureFlags.register("llm_nlp", status=FlagStatus.EXPERIMENTAL)
    FeatureFlags.register("feature_flags", status=FlagStatus.ENABLED)
    FeatureFlags.register("slonet_provider_tests", status=FlagStatus.ENABLED)
    FeatureFlags.register("slonet_provider_wave_i", status=FlagStatus.ENABLED)
    FeatureFlags.register("slonet_wave_f", status=FlagStatus.ENABLED)


# ===================================================================
# test_framework.py
# ===================================================================

class TestTestResult:
    def test_creation(self):
        r = TestResult(name="t1", status="passed", execution_time=0.1)
        assert r.name == "t1"
        assert r.status == "passed"
        assert r.error_message is None
        assert r.metrics == {}

    def test_with_error(self):
        r = TestResult(name="t2", status="failed", execution_time=0.01, error_message="boom")
        assert r.error_message == "boom"


class TestTestSuite:
    def test_creation(self):
        s = TestSuite(
            name="suite", tests=[], total_tests=0,
            passed_tests=0, failed_tests=0, skipped_tests=0,
            total_execution_time=0.0,
        )
        assert s.name == "suite"
        assert s.coverage_percentage == 0.0


class TestTestFramework:
    def test_register_and_run(self):
        fw = TestFramework("unit")
        fw.register(lambda: None)
        suite = fw.run()
        assert suite.total_tests == 1
        assert suite.passed_tests == 1
        assert suite.failed_tests == 0

    def test_captures_failure(self):
        fw = TestFramework()
        fw.register(lambda: 1 / 0)
        suite = fw.run()
        assert suite.failed_tests == 1
        assert "zero" in suite.tests[0].error_message.lower()

    def test_get_summary(self):
        fw = TestFramework()
        fw.register(lambda: None)
        fw.register(lambda: None)
        suite = fw.run()
        summary = fw.get_summary(suite)
        assert summary["total"] == 2
        assert summary["passed"] == 2
        assert summary["pass_rate"] == 100.0


class TestMarkTest:
    def test_sets_attribute(self):
        @mark_test
        def my_test():
            pass
        assert getattr(my_test, "_is_test", False) is True


class TestBenchmarkRunner:
    def test_run_benchmark(self):
        br = BenchmarkRunner()
        result = br.run_benchmark("fast", lambda: None, iterations=10)
        assert result["iterations"] == 10
        assert result["mean_time"] >= 0
        assert result["min_time"] <= result["max_time"]
        assert len(br.results) == 1
