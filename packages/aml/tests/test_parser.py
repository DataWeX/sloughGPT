"""Tests for the AML parser."""

import os
import tempfile

from aml.parser import parse, parse_file, _parse_value, _split_csv
from aml.schema import AmlBlock, AmlDocument


class TestParseValue:
    def test_quoted_string(self):
        assert _parse_value('"hello world"') == "hello world"

    def test_single_quoted(self):
        assert _parse_value("'hello'") == "hello"

    def test_int(self):
        assert _parse_value("42") == 42

    def test_float(self):
        assert _parse_value("3.14") == 3.14

    def test_bool_true(self):
        assert _parse_value("true") is True
        assert _parse_value("yes") is True
        assert _parse_value("on") is True

    def test_bool_false(self):
        assert _parse_value("false") is False
        assert _parse_value("no") is False

    def test_null(self):
        assert _parse_value("null") is None
        assert _parse_value("none") is None

    def test_inline_list(self):
        assert _parse_value("[a, b, c]") == ["a", "b", "c"]

    def test_inline_list_numbers(self):
        assert _parse_value("[1, 2, 3]") == [1, 2, 3]

    def test_inline_list_empty(self):
        assert _parse_value("[]") == []

    def test_inline_list_quoted(self):
        assert _parse_value('["hello world", "foo"]') == ["hello world", "foo"]

    def test_unquoted_string(self):
        assert _parse_value("hello") == "hello"


class TestSplitCsv:
    def test_simple(self):
        assert _split_csv("a, b, c") == ["a", "b", "c"]

    def test_quoted(self):
        assert _split_csv('"a, b", c') == ['"a, b"', "c"]

    def test_empty(self):
        assert _split_csv("") == []


class TestParse:
    def test_empty(self):
        doc = parse("")
        assert doc.version == "1.0"
        assert len(doc.blocks) == 0

    def test_header_only(self):
        doc = parse("@aml 2.0")
        assert doc.version == "2.0"

    def test_simple_block(self):
        doc = parse('@aml 1.0\n\n@knowledge mitochondria {\n    content = "The powerhouse"\n    topic = "biology"\n}')
        assert len(doc.blocks) == 1
        b = doc.blocks[0]
        assert b.tag == "knowledge"
        assert b.name == "mitochondria"
        assert b.metadata["content"] == "The powerhouse"
        assert b.metadata["topic"] == "biology"

    def test_block_no_name(self):
        doc = parse('@config {\n    debug = true\n    port = 8000\n}')
        assert len(doc.blocks) == 1
        b = doc.blocks[0]
        assert b.tag == "config"
        assert b.name is None
        assert b.metadata["debug"] is True
        assert b.metadata["port"] == 8000

    def test_block_with_list(self):
        doc = parse('@aml 1.0\n\n@knowledge mitochondria {\n    tags = ["cell", "energy"]\n}')
        b = doc.blocks[0]
        assert b.metadata["tags"] == ["cell", "energy"]

    def test_block_with_body_items(self):
        src = """@aml 1.0

@knowledge photosynthesis {
    - Light-dependent reactions
    - Calvin cycle
}
"""
        doc = parse(src)
        assert len(doc.blocks) == 1
        b = doc.blocks[0]
        assert b.tag == "knowledge"
        assert b.name == "photosynthesis"
        assert isinstance(b.body, list)
        assert len(b.body) == 2

    def test_multiple_blocks(self):
        src = """@aml 1.0

@knowledge a {
    content = "fact one"
}

@knowledge b {
    content = "fact two"
}
"""
        doc = parse(src)
        assert len(doc.blocks) == 2

    def test_nested_blocks(self):
        src = """@aml 1.0

@chapter biology {
    @section mitochondria {
        content = "The powerhouse"
    }

    @section photosynthesis {
        content = "Plants make food"
    }
}
"""
        doc = parse(src)
        # nested blocks get appended to doc.blocks
        assert len(doc.blocks) == 3

    def test_comments_ignored(self):
        src = """@aml 1.0
# this is a comment

@knowledge a {
    # another comment
    content = "fact"
}
"""
        doc = parse(src)
        assert len(doc.blocks) == 1

    def test_inline_string_body(self):
        doc = parse('@tag name = "hello"')
        assert len(doc.blocks) == 1
        b = doc.blocks[0]
        assert b.body == "hello"

    def test_bool_metadata(self):
        doc = parse('@config {\n    debug = true\n    verbose = false\n}')
        b = doc.blocks[0]
        assert b.metadata["debug"] is True
        assert b.metadata["verbose"] is False

    def test_int_float_metadata(self):
        doc = parse('@config {\n    port = 8000\n    ratio = 3.14\n}')
        b = doc.blocks[0]
        assert b.metadata["port"] == 8000
        assert b.metadata["ratio"] == 3.14


class TestParseFile:
    def test_parse_file(self):
        content = '@aml 1.0\n\n@knowledge test {\n    content = "hello"\n}\n'
        with tempfile.NamedTemporaryFile(suffix=".aml", mode="w", delete=False) as f:
            f.write(content)
            f.flush()
            try:
                doc = parse_file(f.name)
                assert len(doc.blocks) == 1
                assert doc.blocks[0].tag == "knowledge"
            finally:
                os.unlink(f.name)


class TestDocument:
    def test_by_tag(self):
        doc = parse('@aml 1.0\n\n@a { x = 1 }\n@b { y = 2 }\n@a { z = 3 }')
        assert len(doc.by_tag("a")) == 2
        assert len(doc.by_tag("b")) == 1

    def test_by_name(self):
        doc = parse('@aml 1.0\n\n@knowledge mitochondria {\n    content = "hp"\n}')
        b = doc.by_name("mitochondria")
        assert b is not None
        assert b.tag == "knowledge"

    def test_to_dict(self):
        doc = parse('@aml 1.0\n\n@knowledge mito {\n    content = "hp"\n    topic = "bio"\n}')
        d = doc.to_dict()
        assert d["aml"] == "1.0"
        assert "knowledge:mito" in d
        assert d["knowledge:mito"]["content"] == "hp"
