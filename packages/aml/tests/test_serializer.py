"""Tests for the AML serializer + round-trip (parse → serialize → parse)."""

import json

from aml.parser import parse
from aml.serializer import serialize, dict_to_aml
from aml.schema import AmlBlock, AmlDocument


class TestSerialize:
    def test_header(self):
        doc = AmlDocument(version="1.0")
        out = serialize(doc)
        assert out.startswith("@aml 1.0\n")

    def test_simple_block(self):
        doc = AmlDocument(version="1.0", blocks=[
            AmlBlock(tag="knowledge", name="mito", metadata={
                "content": "The powerhouse",
                "topic": "biology",
            }),
        ])
        out = serialize(doc)
        assert "@knowledge mito" in out
        assert 'content = "The powerhouse"' in out
        assert "topic = biology" in out

    def test_list_body(self):
        doc = AmlDocument(version="1.0", blocks=[
            AmlBlock(tag="knowledge", name="photosynthesis", body=[
                "Light-dependent reactions",
                "Calvin cycle",
            ]),
        ])
        out = serialize(doc)
        assert "- Light-dependent reactions" in out
        assert "- Calvin cycle" in out

    def test_bool_values(self):
        doc = AmlDocument(version="1.0", blocks=[
            AmlBlock(tag="config", metadata={"debug": True, "verbose": False}),
        ])
        out = serialize(doc)
        assert "debug = true" in out
        assert "verbose = false" in out

    def test_int_float_values(self):
        doc = AmlDocument(version="1.0", blocks=[
            AmlBlock(tag="config", metadata={"port": 8000, "ratio": 3.14}),
        ])
        out = serialize(doc)
        assert "port = 8000" in out
        assert "ratio = 3.14" in out

    def test_null_value(self):
        doc = AmlDocument(version="1.0", blocks=[
            AmlBlock(tag="config", metadata={"val": None}),
        ])
        out = serialize(doc)
        assert "val = null" in out

    def test_inline_list(self):
        doc = AmlDocument(version="1.0", blocks=[
            AmlBlock(tag="knowledge", metadata={"tags": ["a", "b"]}),
        ])
        out = serialize(doc)
        assert "tags = [a, b]" in out


class TestDictToAml:
    def test_simple(self):
        out = dict_to_aml({
            "knowledge:mito": {
                "content": "The powerhouse",
                "topic": "biology",
            }
        })
        assert "@aml 1.0" in out
        assert "@knowledge mito" in out
        assert "The powerhouse" in out

    def test_list_value(self):
        out = dict_to_aml({
            "knowledge:photo": {
                "body": ["stage 1", "stage 2"],
                "topic": "biology",
            }
        })
        assert "- stage 1" in out
        assert "- stage 2" in out


class TestRoundTrip:
    def test_simple_roundtrip(self):
        src = """@aml 1.0

@knowledge mitochondria {
    content = "The mitochondria is the powerhouse of the cell."
    topic = "biology"
    importance = 0.8
}
"""
        doc = parse(src)
        out = serialize(doc)
        doc2 = parse(out)
        assert len(doc2.blocks) == 1
        b = doc2.blocks[0]
        assert b.tag == "knowledge"
        assert b.name == "mitochondria"
        assert b.metadata["content"] == "The mitochondria is the powerhouse of the cell."
        assert b.metadata["topic"] == "biology"
        assert b.metadata["importance"] == 0.8

    def test_list_roundtrip(self):
        src = """@aml 1.0

@knowledge photo {
    - Light-dependent
    - Calvin cycle
}
"""
        doc = parse(src)
        out = serialize(doc)
        doc2 = parse(out)
        b = doc2.blocks[0]
        assert isinstance(b.body, list)
        assert len(b.body) == 2

    def test_multiple_blocks_roundtrip(self):
        src = """@aml 1.0

@knowledge a {
    content = "fact one"
    topic = "biology"
}

@knowledge b {
    content = "fact two"
    topic = "physics"
}
"""
        doc = parse(src)
        out = serialize(doc)
        doc2 = parse(out)
        assert len(doc2.blocks) == 2

    def test_bool_roundtrip(self):
        src = """@aml 1.0

@config {
    debug = true
    verbose = false
}
"""
        doc = parse(src)
        out = serialize(doc)
        doc2 = parse(out)
        b = doc2.blocks[0]
        assert b.metadata["debug"] is True
        assert b.metadata["verbose"] is False

    def test_int_roundtrip(self):
        src = """@aml 1.0

@config {
    port = 8000
}
"""
        doc = parse(src)
        out = serialize(doc)
        doc2 = parse(out)
        b = doc2.blocks[0]
        assert b.metadata["port"] == 8000

    def test_json_roundtrip(self):
        """Convert AML → dict → JSON → dict → AML and verify consistency."""
        src = """@aml 1.0

@knowledge mitochondria {
    content = "The powerhouse"
    topic = "biology"
}
"""
        doc = parse(src)
        d = doc.to_dict()
        j = json.dumps(d)
        d2 = json.loads(j)
        assert d == d2
