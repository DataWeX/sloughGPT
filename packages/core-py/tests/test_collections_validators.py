"""Tests for domains.collections.validators — Schema, DataValidator, EnrichmentRule, DataEnricher; domains.collections.config."""

from domains.collections.validators import (
    Schema, DataValidator, EnrichmentRule, DataEnricher,
)
from domains.collections.sources import Record


class TestSchema:
    def test_valid(self):
        s = Schema(required_fields=["author"], field_types={"author": str})
        r = Record(content="text", metadata={"author": "Alice"})
        valid, err = s.validate(r)
        assert valid is True
        assert err == ""

    def test_missing_required(self):
        s = Schema(required_fields=["author"])
        r = Record(content="text")
        valid, err = s.validate(r)
        assert valid is False
        assert "author" in err

    def test_wrong_type(self):
        s = Schema(field_types={"count": int})
        r = Record(content="text", metadata={"count": "not_int"})
        valid, err = s.validate(r)
        assert valid is False

    def test_too_short(self):
        s = Schema(min_content_length=5)
        r = Record(content="hi")
        valid, err = s.validate(r)
        assert valid is False
        assert "short" in err.lower()

    def test_too_long(self):
        s = Schema(max_content_length=5)
        r = Record(content="hello world")
        valid, err = s.validate(r)
        assert valid is False
        assert "long" in err.lower()

    def test_custom_validator(self):
        s = Schema(field_validators={"age": lambda x: x > 0})
        r = Record(content="text", metadata={"age": 25})
        valid, _ = s.validate(r)
        assert valid is True
        r2 = Record(content="text", metadata={"age": -1})
        valid2, _ = s.validate(r2)
        assert valid2 is False


class TestDataValidator:
    def test_valid(self):
        dv = DataValidator(Schema())
        assert dv.validate(Record(content="hello")) is True
        assert dv.stats["valid"] == 1

    def test_invalid(self):
        dv = DataValidator(Schema(min_content_length=100))
        assert dv.validate(Record(content="short")) is False
        assert dv.stats["invalid"] == 1

    def test_validate_all(self):
        dv = DataValidator(Schema(min_content_length=3))
        records = [Record(content="ok"), Record(content="no"), Record(content="yes")]
        valid = dv.validate_all(records)
        assert len(valid) == 1


class TestEnrichmentRule:
    def test_static_value(self):
        r = EnrichmentRule(key="source", value="test")
        rec = Record(content="text")
        r.apply(rec)
        assert rec.metadata["source"] == "test"

    def test_no_overwrite(self):
        rec = Record(content="text", metadata={"source": "existing"})
        r = EnrichmentRule(key="source", value="new", overwrite=False)
        r.apply(rec)
        assert rec.metadata["source"] == "existing"

    def test_overwrite(self):
        rec = Record(content="text", metadata={"source": "old"})
        r = EnrichmentRule(key="source", value="new", overwrite=True)
        r.apply(rec)
        assert rec.metadata["source"] == "new"


class TestDataEnricher:
    def test_enrich(self):
        de = DataEnricher([EnrichmentRule(key="tag", value="auto")])
        rec = Record(content="text")
        de.enrich(rec)
        assert rec.metadata["tag"] == "auto"
        assert de.stats["enriched"] == 1
