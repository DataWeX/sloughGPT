"""Tests for domains.collections.validators — Schema, DataValidator, DataEnricher,
RateLimiter, CallableSource, CallableStore, CollectorRunner.

Covers: schema validation, enrichment rules, rate limiting, callable adapters,
collector orchestration. Uses Record from sources module.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_core_dir = str(Path(__file__).resolve().parents[2])
if _core_dir not in sys.path:
    sys.path.insert(0, _core_dir)

from domains.collections.sources import Record
from domains.collections.validators import (
    Schema,
    DataValidator,
    EnrichmentRule,
    DataEnricher,
    RateLimiter,
    CallableSource,
    CallableStore,
    CollectorRunner,
)


# ── Schema ───────────────────────────────────────────────────────────

class TestSchema:
    def test_valid_record(self):
        schema = Schema(required_fields=["title"], max_content_length=100)
        r = Record(content="hello", metadata={"title": "t"})
        ok, err = schema.validate(r)
        assert ok is True

    def test_content_too_long(self):
        schema = Schema(max_content_length=5)
        r = Record(content="toolongcontent")
        ok, err = schema.validate(r)
        assert ok is False
        assert "too long" in err

    def test_content_too_short(self):
        schema = Schema(min_content_length=10)
        r = Record(content="hi")
        ok, err = schema.validate(r)
        assert ok is False
        assert "too short" in err

    def test_missing_required_field(self):
        schema = Schema(required_fields=["author"])
        r = Record(content="hello", metadata={})
        ok, err = schema.validate(r)
        assert ok is False
        assert "Missing" in err

    def test_wrong_field_type(self):
        schema = Schema(field_types={"count": int})
        r = Record(content="hello", metadata={"count": "not_int"})
        ok, err = schema.validate(r)
        assert ok is False
        assert "wrong type" in err

    def test_custom_validator(self):
        schema = Schema(field_validators={"val": lambda v: v > 0})
        r_ok = Record(content="hello", metadata={"val": 5})
        r_bad = Record(content="hello", metadata={"val": -1})
        assert schema.validate(r_ok)[0] is True
        assert schema.validate(r_bad)[0] is False


# ── DataValidator ────────────────────────────────────────────────────

class TestDataValidator:
    def test_valid(self):
        dv = DataValidator(Schema())
        assert dv.validate(Record(content="ok")) is True
        assert dv.stats["valid"] == 1

    def test_invalid(self):
        dv = DataValidator(Schema(min_content_length=10))
        assert dv.validate(Record(content="short")) is False
        assert dv.stats["invalid"] == 1

    def test_validate_all(self):
        dv = DataValidator(Schema())
        records = [Record(content="a"), Record(content="b")]
        result = dv.validate_all(records)
        assert len(result) == 2

    def test_reset_stats(self):
        dv = DataValidator(Schema())
        dv.validate(Record(content="x"))
        dv.reset_stats()
        assert dv.stats["valid"] == 0


# ── EnrichmentRule ───────────────────────────────────────────────────

class TestEnrichmentRule:
    def test_fixed_value(self):
        rule = EnrichmentRule(key="tag", value="test")
        r = Record(content="hello")
        rule.apply(r)
        assert r.metadata["tag"] == "test"

    def test_no_overwrite(self):
        rule = EnrichmentRule(key="tag", value="new", overwrite=False)
        r = Record(content="hello", metadata={"tag": "existing"})
        rule.apply(r)
        assert r.metadata["tag"] == "existing"

    def test_overwrite(self):
        rule = EnrichmentRule(key="tag", value="new", overwrite=True)
        r = Record(content="hello", metadata={"tag": "old"})
        rule.apply(r)
        assert r.metadata["tag"] == "new"

    def test_value_fn(self):
        rule = EnrichmentRule(key="length", value_fn=lambda r: len(r.content))
        r = Record(content="hello")
        rule.apply(r)
        assert r.metadata["length"] == 5


# ── DataEnricher ─────────────────────────────────────────────────────

class TestDataEnricher:
    def test_enrich(self):
        de = DataEnricher([EnrichmentRule(key="k", value="v")])
        r = Record(content="hello")
        de.enrich(r)
        assert r.metadata["k"] == "v"
        assert de.stats["enriched"] == 1

    def test_enrich_skip(self):
        de = DataEnricher([EnrichmentRule(key="k", value="v", overwrite=False)])
        r = Record(content="hello", metadata={"k": "existing"})
        de.enrich(r)
        assert de.stats["skipped"] == 1

    def test_add_rule(self):
        de = DataEnricher()
        de.add_rule(EnrichmentRule(key="a", value=1))
        assert len(de.rules) == 1

    def test_enrich_all(self):
        de = DataEnricher([EnrichmentRule(key="k", value="v")])
        records = [Record(content="a"), Record(content="b")]
        result = de.enrich_all(records)
        assert len(result) == 2


# ── RateLimiter ──────────────────────────────────────────────────────

class TestRateLimiter:
    def test_acquire_within_burst(self):
        rl = RateLimiter(max_per_second=10, burst_size=3)
        assert rl.acquire() is True
        assert rl.acquire() is True
        assert rl.acquire() is True
        assert rl.stats["allowed"] == 3

    def test_acquire_exceeds_burst(self):
        rl = RateLimiter(max_per_second=1, burst_size=1)
        rl.acquire()
        assert rl.acquire() is False
        assert rl.stats["delayed"] == 1

    def test_refill(self):
        rl = RateLimiter(max_per_second=100, burst_size=1)
        rl.acquire()
        time.sleep(0.05)
        assert rl.acquire() is True

    def test_reset(self):
        rl = RateLimiter(burst_size=1)
        rl.acquire()
        rl.reset()
        assert rl.acquire() is True


# ── CallableSource ───────────────────────────────────────────────────

class TestCallableSource:
    def test_read(self):
        def gen():
            yield Record(content="a")
            yield Record(content="b")
        cs = CallableSource(gen, name="test")
        records = list(cs.read())
        assert len(records) == 2
        assert cs.name == "test"


# ── CallableStore ────────────────────────────────────────────────────

class TestCallableStore:
    def test_write(self):
        stored = []
        cs = CallableStore(lambda r: stored.append(r))
        cs.write(Record(content="hello"))
        assert len(stored) == 1
        assert cs.count() == 1

    def test_read_all(self):
        cs = CallableStore(lambda r: None)
        assert list(cs.read_all()) == []


# ── CollectorRunner ──────────────────────────────────────────────────

class TestCollectorRunner:
    def test_add_and_list(self):
        cr = CollectorRunner()
        mock = type("Mock", (), {"collect": lambda self: 5})()
        cr.add("c1", mock)
        assert "c1" in cr.list()

    def test_remove(self):
        cr = CollectorRunner()
        mock = type("Mock", (), {"collect": lambda self: 5})()
        cr.add("c1", mock)
        assert cr.remove("c1") is True
        assert cr.remove("nonexistent") is False

    def test_run(self):
        cr = CollectorRunner()
        mock = type("Mock", (), {"collect": lambda self: 7})()
        cr.add("c1", mock)
        result = cr.run("c1")
        assert result == 7
        assert cr.stats()["c1"]["runs"] == 1

    def test_run_nonexistent(self):
        cr = CollectorRunner()
        assert cr.run("nope") == 0

    def test_run_all(self):
        cr = CollectorRunner()
        mock = type("Mock", (), {"collect": lambda self: 3})()
        cr.add("c1", mock)
        cr.add("c2", mock)
        results = cr.run_all()
        assert results["c1"] == 3
        assert results["c2"] == 3

    def test_run_threaded(self):
        cr = CollectorRunner()
        mock = type("Mock", (), {"collect": lambda self: 1})()
        cr.add("c1", mock)
        cr.add("c2", mock)
        results = cr.run_threaded()
        assert sum(results.values()) == 2
