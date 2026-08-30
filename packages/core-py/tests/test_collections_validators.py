"""Tests for domains.collections.validators — pure logic, no network."""
from __future__ import annotations

import time
import threading

import pytest

from domains.collections.sources import Record
from domains.collections.validators import (
    CallableSource,
    CallableStore,
    CollectorRunner,
    DataEnricher,
    DataValidator,
    EnrichmentRule,
    RateLimiter,
    Schema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r(content: str, **meta) -> Record:
    return Record(content=content, metadata=meta)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_valid_record(self):
        s = Schema(required_fields=["source"])
        valid, err = s.validate(_r("hello", source="test"))
        assert valid is True
        assert err == ""

    def test_missing_required_field(self):
        s = Schema(required_fields=["source"])
        valid, err = s.validate(_r("hello"))
        assert valid is False
        assert "source" in err

    def test_content_too_long(self):
        s = Schema(max_content_length=5)
        valid, err = s.validate(_r("a" * 10))
        assert valid is False
        assert "too long" in err

    def test_content_too_short(self):
        s = Schema(min_content_length=5)
        valid, err = s.validate(_r("ab"))
        assert valid is False
        assert "too short" in err

    def test_field_type_check(self):
        s = Schema(field_types={"count": int})
        valid, err = s.validate(_r("x", count="not_int"))
        assert valid is False
        assert "wrong type" in err

    def test_field_type_check_passes(self):
        s = Schema(field_types={"count": int})
        valid, err = s.validate(_r("x", count=42))
        assert valid is True

    def test_field_validator_passes(self):
        s = Schema(field_validators={"score": lambda v: v > 0})
        valid, err = s.validate(_r("x", score=5))
        assert valid is True

    def test_field_validator_fails(self):
        s = Schema(field_validators={"score": lambda v: v > 0})
        valid, err = s.validate(_r("x", score=-1))
        assert valid is False
        assert "failed validation" in err

    def test_default_limits(self):
        s = Schema()
        valid, _ = s.validate(_r("x"))
        assert valid is True

    def test_optional_field_not_checked(self):
        s = Schema(field_types={"missing": int})
        valid, _ = s.validate(_r("x"))
        assert valid is True

    def test_multiple_required_fields(self):
        s = Schema(required_fields=["a", "b"])
        valid, _ = s.validate(_r("x", a=1, b=2))
        assert valid is True
        valid, err = s.validate(_r("x", a=1))
        assert valid is False
        assert "b" in err


# ---------------------------------------------------------------------------
# DataValidator
# ---------------------------------------------------------------------------

class TestDataValidator:
    def test_valid_increments_stats(self):
        dv = DataValidator(Schema())
        assert dv.validate(_r("hello")) is True
        assert dv.stats["valid"] == 1
        assert dv.stats["invalid"] == 0

    def test_invalid_increments_stats(self):
        dv = DataValidator(Schema(max_content_length=3))
        assert dv.validate(_r("toolong")) is False
        assert dv.stats["invalid"] == 1
        assert dv.stats["errors"]

    def test_validate_all_filters(self):
        dv = DataValidator(Schema(min_content_length=3))
        records = [_r("ab"), _r("abc"), _r("abcd")]
        result = dv.validate_all(records)
        assert len(result) == 2

    def test_reset_stats(self):
        dv = DataValidator(Schema())
        dv.validate(_r("x"))
        dv.reset_stats()
        assert dv.stats == {"valid": 0, "invalid": 0, "errors": {}}

    def test_error_counted(self):
        dv = DataValidator(Schema(max_content_length=3))
        dv.validate(_r("long1"))
        dv.validate(_r("long2"))
        assert dv.stats["errors"]["Content too long: 5 > 3"] == 2


# ---------------------------------------------------------------------------
# EnrichmentRule
# ---------------------------------------------------------------------------

class TestEnrichmentRule:
    def test_static_value(self):
        rule = EnrichmentRule(key="tag", value="v1")
        r = _r("x")
        rule.apply(r)
        assert r.metadata["tag"] == "v1"

    def test_value_fn(self):
        rule = EnrichmentRule(key="len", value_fn=lambda r: len(r.content))
        r = _r("hello")
        rule.apply(r)
        assert r.metadata["len"] == 5

    def test_no_overwrite_existing(self):
        r = _r("x", tag="original")
        rule = EnrichmentRule(key="tag", value="new")
        rule.apply(r)
        assert r.metadata["tag"] == "original"

    def test_overwrite_existing(self):
        r = _r("x", tag="old")
        rule = EnrichmentRule(key="tag", value="new", overwrite=True)
        rule.apply(r)
        assert r.metadata["tag"] == "new"

    def test_new_key_always_set(self):
        r = _r("x")
        rule = EnrichmentRule(key="new_key", value="val")
        rule.apply(r)
        assert r.metadata["new_key"] == "val"


# ---------------------------------------------------------------------------
# DataEnricher
# ---------------------------------------------------------------------------

class TestDataEnricher:
    def test_enrich_adds_field(self):
        de = DataEnricher([EnrichmentRule(key="tag", value="v1")])
        r = _r("x")
        result = de.enrich(r)
        assert result.metadata["tag"] == "v1"
        assert de.stats["enriched"] == 1

    def test_enrich_skip_if_no_change(self):
        r = _r("x", tag="existing")
        de = DataEnricher([EnrichmentRule(key="tag", value="new")])
        de.enrich(r)
        assert de.stats["skipped"] == 1
        assert r.metadata["tag"] == "existing"

    def test_enrich_all(self):
        de = DataEnricher([EnrichmentRule(key="tag", value="v")])
        records = [_r("a"), _r("b")]
        results = de.enrich_all(records)
        assert all(r.metadata["tag"] == "v" for r in results)

    def test_add_rule(self):
        de = DataEnricher()
        result = de.add_rule(EnrichmentRule(key="k", value="v"))
        assert result is de
        r = _r("x")
        de.enrich(r)
        assert r.metadata["k"] == "v"

    def test_reset_stats(self):
        de = DataEnricher([EnrichmentRule(key="k", value="v")])
        de.enrich(_r("x"))
        de.reset_stats()
        assert de.stats == {"enriched": 0, "skipped": 0}

    def test_overwrite_stats(self):
        r = _r("x", k="old")
        de = DataEnricher([EnrichmentRule(key="k", value="new", overwrite=True)])
        de.enrich(r)
        assert de.stats["enriched"] == 1
        assert r.metadata["k"] == "new"


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------

class TestRateLimiter:
    def test_acquire_within_burst(self):
        rl = RateLimiter(max_per_second=10, burst_size=3)
        assert rl.acquire() is True
        assert rl.acquire() is True
        assert rl.acquire() is True
        assert rl.stats["allowed"] == 3

    def test_acquire_exhausts_burst(self):
        rl = RateLimiter(max_per_second=10, burst_size=1)
        assert rl.acquire() is True
        assert rl.acquire() is False
        assert rl.stats["delayed"] == 1

    def test_refill_after_time(self):
        rl = RateLimiter(max_per_second=100, burst_size=1)
        assert rl.acquire() is True
        assert rl.acquire() is False
        time.sleep(0.05)
        assert rl.acquire() is True

    def test_wait_succeeds(self):
        rl = RateLimiter(max_per_second=100, burst_size=1)
        assert rl.acquire() is True
        assert rl.wait(timeout=0.1) is True

    def test_wait_timeout(self):
        rl = RateLimiter(max_per_second=1, burst_size=1)
        rl.acquire()
        assert rl.wait(timeout=0.01) is False

    def test_reset(self):
        rl = RateLimiter(max_per_second=10, burst_size=1)
        rl.acquire()
        rl.reset()
        assert rl.acquire() is True

    def test_stats(self):
        rl = RateLimiter(max_per_second=10, burst_size=1)
        rl.acquire()
        rl.acquire()
        assert rl.stats["allowed"] == 1
        assert rl.stats["delayed"] == 1

    def test_thread_safety(self):
        rl = RateLimiter(max_per_second=1000, burst_size=500)
        results = []
        lock = threading.Lock()
        def acquire_many():
            count = 0
            for _ in range(100):
                if rl.acquire():
                    count += 1
            with lock:
                results.append(count)
        threads = [threading.Thread(target=acquire_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_allowed = sum(results)
        assert total_allowed <= 500


# ---------------------------------------------------------------------------
# CallableSource
# ---------------------------------------------------------------------------

class TestCallableSource:
    def test_read(self):
        def gen():
            yield Record(content="a")
            yield Record(content="b")
        cs = CallableSource(gen)
        records = list(cs.read())
        assert len(records) == 2

    def test_default_name(self):
        def my_func():
            yield Record(content="x")
        cs = CallableSource(my_func)
        assert cs.name == "my_func"

    def test_custom_name(self):
        cs = CallableSource(lambda: iter([]), name="custom")
        assert cs.name == "custom"


# ---------------------------------------------------------------------------
# CallableStore
# ---------------------------------------------------------------------------

class TestCallableStore:
    def test_write(self):
        received = []
        cs = CallableStore(lambda r: received.append(r))
        cs.write(Record(content="x"))
        assert len(received) == 1

    def test_count(self):
        cs = CallableStore(lambda r: None)
        cs.write(Record(content="a"))
        cs.write(Record(content="b"))
        assert cs.count() == 2

    def test_read_all_empty(self):
        cs = CallableStore(lambda r: None)
        assert list(cs.read_all()) == []

    def test_default_name(self):
        def my_fn(r):
            pass
        cs = CallableStore(my_fn)
        assert cs.name == "my_fn"

    def test_custom_name(self):
        cs = CallableStore(lambda r: None, name="cs")
        assert cs.name == "cs"


# ---------------------------------------------------------------------------
# CollectorRunner
# ---------------------------------------------------------------------------

class TestCollectorRunner:
    def _make_collector(self, count: int):
        class FakeCollector:
            def __init__(self, n):
                self.n = n
            def collect(self):
                return self.n
        return FakeCollector(count)

    def test_add_and_list(self):
        cr = CollectorRunner()
        cr.add("a", self._make_collector(5))
        cr.add("b", self._make_collector(3))
        assert sorted(cr.list()) == ["a", "b"]

    def test_run(self):
        cr = CollectorRunner()
        cr.add("a", self._make_collector(5))
        assert cr.run("a") == 5

    def test_run_nonexistent(self):
        cr = CollectorRunner()
        assert cr.run("missing") == 0

    def test_run_stats(self):
        cr = CollectorRunner()
        cr.add("a", self._make_collector(5))
        cr.run("a")
        stats = cr.stats()
        assert stats["a"]["runs"] == 1
        assert stats["a"]["total_collected"] == 5
        assert stats["a"]["last_run"] is not None

    def test_run_error(self):
        class FailCollector:
            def collect(self):
                raise RuntimeError("boom")
        cr = CollectorRunner()
        cr.add("f", FailCollector())
        assert cr.run("f") == 0
        assert cr.stats()["f"]["errors"] == 1

    def test_run_all(self):
        cr = CollectorRunner()
        cr.add("a", self._make_collector(2))
        cr.add("b", self._make_collector(3))
        results = cr.run_all()
        assert results == {"a": 2, "b": 3}

    def test_run_threaded(self):
        cr = CollectorRunner()
        cr.add("a", self._make_collector(10))
        cr.add("b", self._make_collector(20))
        results = cr.run_threaded()
        assert results["a"] == 10
        assert results["b"] == 20

    def test_remove(self):
        cr = CollectorRunner()
        cr.add("a", self._make_collector(1))
        assert cr.remove("a") is True
        assert cr.list() == []

    def test_remove_nonexistent(self):
        cr = CollectorRunner()
        assert cr.remove("missing") is False

    def test_get(self):
        cr = CollectorRunner()
        c = self._make_collector(1)
        cr.add("a", c)
        assert cr.get("a") is c

    def test_get_nonexistent(self):
        cr = CollectorRunner()
        assert cr.get("missing") is None

    def test_accumulate_runs(self):
        cr = CollectorRunner()
        cr.add("a", self._make_collector(1))
        cr.run("a")
        cr.run("a")
        stats = cr.stats()
        assert stats["a"]["runs"] == 2
        assert stats["a"]["total_collected"] == 2

    def test_init(self):
        cr = CollectorRunner()
        assert cr.list() == []
        assert cr.stats() == {}
