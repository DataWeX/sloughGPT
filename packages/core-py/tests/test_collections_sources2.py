"""Tests for domains.collections.sources — pure logic only."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from domains.collections.sources import (
    ApiSource,
    FileSource,
    GeneratorSource,
    Record,
    SseSource,
    UrlSource,
    WatchSource,
    Source,
)


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

class TestRecord:
    def test_basic_creation(self):
        r = Record(content="hello")
        assert r.content == "hello"
        assert "timestamp" in r.metadata

    def test_timestamp_auto_populated(self):
        r = Record(content="x")
        assert isinstance(r.metadata["timestamp"], float)

    def test_timestamp_not_overwritten(self):
        r = Record(content="x", metadata={"timestamp": 42.0})
        assert r.metadata["timestamp"] == 42.0

    def test_metadata_preserved(self):
        r = Record(content="x", metadata={"a": 1})
        assert r.metadata["a"] == 1

    def test_to_dict(self):
        r = Record(content="x", metadata={"k": "v"})
        d = r.to_dict()
        assert d["content"] == "x"
        assert d["metadata"]["k"] == "v"
        assert "timestamp" in d["metadata"]

    def test_to_dict_returns_copy(self):
        r = Record(content="x", metadata={"k": "v"})
        d = r.to_dict()
        d["metadata"]["k"] = "changed"
        assert r.metadata["k"] == "v"

    def test_empty_content(self):
        r = Record(content="")
        assert r.content == ""

    def test_unicode_content(self):
        r = Record(content="\u00e9\u00e8\u00ea")
        assert r.content == "\u00e9\u00e8\u00ea"


# ---------------------------------------------------------------------------
# Source protocol
# ---------------------------------------------------------------------------

class TestSourceProtocol:
    def test_generator_source_is_source(self):
        gs = GeneratorSource(lambda: [])
        assert isinstance(gs, Source)


# ---------------------------------------------------------------------------
# FileSource
# ---------------------------------------------------------------------------

class TestFileSource:
    def _write(self, tmp_path: Path, name: str, content: str) -> Path:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p

    def test_name_defaults_to_filename(self, tmp_path):
        p = self._write(tmp_path, "data.txt", "line1\n")
        src = FileSource(str(p))
        assert src.name == f"file:{p.name}"

    def test_name_override(self, tmp_path):
        p = self._write(tmp_path, "data.txt", "line1\n")
        src = FileSource(str(p), name="custom")
        assert src.name == "custom"

    def test_nonexistent_file_yields_nothing(self, tmp_path):
        src = FileSource(str(tmp_path / "missing.txt"))
        assert list(src.read()) == []

    def test_text_file(self, tmp_path):
        p = self._write(tmp_path, "data.txt", "alpha\nbeta\n\ngamma\n")
        records = list(FileSource(str(p)).read())
        contents = [r.content for r in records]
        assert contents == ["alpha", "beta", "gamma"]

    def test_text_file_skips_blank_lines(self, tmp_path):
        p = self._write(tmp_path, "data.txt", "a\n\n\nb\n")
        records = list(FileSource(str(p)).read())
        assert len(records) == 2

    def test_text_metadata_has_source_and_line(self, tmp_path):
        p = self._write(tmp_path, "data.txt", "x\n")
        records = list(FileSource(str(p), name="tsrc").read())
        assert records[0].metadata["source"] == "tsrc"
        assert records[0].metadata["line"] == 0

    def test_jsonl_file(self, tmp_path):
        lines = '{"content":"a","tag":1}\n{"content":"b","tag":2}\n'
        p = self._write(tmp_path, "data.jsonl", lines)
        records = list(FileSource(str(p)).read())
        assert len(records) == 2
        assert records[0].content == "a"
        assert records[0].metadata["tag"] == 1
        assert records[0].metadata["line"] == 0

    def test_jsonl_skips_blank_lines(self, tmp_path):
        p = self._write(tmp_path, "data.jsonl", '{"content":"a"}\n\n{"content":"b"}\n')
        records = list(FileSource(str(p)).read())
        assert len(records) == 2

    def test_jsonl_malformed_line_falls_back(self, tmp_path):
        p = self._write(tmp_path, "data.jsonl", 'NOT_JSON\n{"content":"ok"}\n')
        records = list(FileSource(str(p)).read())
        assert len(records) == 2
        assert records[0].content == "NOT_JSON"

    def test_jsonl_non_dict_item(self, tmp_path):
        p = self._write(tmp_path, "data.jsonl", '42\n')
        records = list(FileSource(str(p)).read())
        assert records[0].content == "42"

    def test_json_array_of_strings(self, tmp_path):
        p = self._write(tmp_path, "data.json", '["a", "b", "c"]')
        records = list(FileSource(str(p)).read())
        assert [r.content for r in records] == ["a", "b", "c"]
        assert records[0].metadata["index"] == 0

    def test_json_array_of_dicts(self, tmp_path):
        p = self._write(tmp_path, "data.json", '[{"content":"x","k":1}]')
        records = list(FileSource(str(p)).read())
        assert records[0].content == "x"
        assert records[0].metadata["k"] == 1

    def test_json_single_dict(self, tmp_path):
        p = self._write(tmp_path, "data.json", '{"content":"solo","extra":true}')
        records = list(FileSource(str(p)).read())
        assert len(records) == 1
        assert records[0].content == "solo"
        assert records[0].metadata["extra"] is True

    def test_json_single_dict_no_content_key(self, tmp_path):
        p = self._write(tmp_path, "data.json", '{"text":"hello"}')
        records = list(FileSource(str(p)).read())
        # dict without "content" key yields empty string content
        assert records[0].content == ""
        assert records[0].metadata["text"] == "hello"

    def test_csv_file_with_content_column(self, tmp_path):
        p = self._write(tmp_path, "data.csv", "content,tag\nhello,A\nworld,B\n")
        records = list(FileSource(str(p)).read())
        assert [r.content for r in records] == ["hello", "world"]
        assert records[0].metadata["tag"] == "A"
        assert records[0].metadata["row"] == 0

    def test_csv_file_without_content_column(self, tmp_path):
        p = self._write(tmp_path, "data.csv", "name,age\nAlice,30\n")
        records = list(FileSource(str(p)).read())
        assert len(records) == 1
        # JSON-dumped row becomes content
        data = json.loads(records[0].content)
        assert data["name"] == "Alice"

    def test_unknown_suffix_treated_as_text(self, tmp_path):
        p = self._write(tmp_path, "data.xyz", "line1\nline2\n")
        records = list(FileSource(str(p)).read())
        assert len(records) == 2

    def test_json_single_dict_pops_content(self, tmp_path):
        p = self._write(tmp_path, "data.json", '{"content":"abc","keep":1}')
        records = list(FileSource(str(p)).read())
        assert records[0].content == "abc"
        assert records[0].metadata["keep"] == 1


# ---------------------------------------------------------------------------
# GeneratorSource
# ---------------------------------------------------------------------------

class TestGeneratorSource:
    def test_yields_records_directly(self):
        src = GeneratorSource(lambda: [Record(content="a"), Record(content="b")])
        records = list(src.read())
        assert [r.content for r in records] == ["a", "b"]

    def test_yields_strings_as_records(self):
        src = GeneratorSource(lambda: ["x", "y"])
        records = list(src.read())
        assert records[0].content == "x"
        assert records[0].metadata["source"] == "generator"

    def test_yields_dicts_as_records(self):
        src = GeneratorSource(lambda: [{"content": "d", "extra": 1}])
        records = list(src.read())
        assert records[0].content == "d"
        assert records[0].metadata["extra"] == 1

    def test_yields_dict_without_content(self):
        src = GeneratorSource(lambda: [{"text": "hello"}])
        records = list(src.read())
        # dict without "content" key yields empty string content
        assert records[0].content == ""
        assert records[0].metadata["text"] == "hello"

    def test_empty_generator(self):
        src = GeneratorSource(lambda: [])
        assert list(src.read()) == []

    def test_custom_name(self):
        src = GeneratorSource(lambda: [], name="mygen")
        assert src.name == "mygen"

    def test_mixed_types(self):
        def gen():
            yield Record(content="r")
            yield "s"
            yield {"content": "t", "k": 1}

        src = GeneratorSource(gen)
        records = list(src.read())
        assert len(records) == 3
        assert records[0].content == "r"
        assert records[1].content == "s"
        assert records[2].content == "t"


# ---------------------------------------------------------------------------
# WatchSource
# ---------------------------------------------------------------------------

class TestWatchSource:
    def test_detects_new_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        ws = WatchSource(str(tmp_path))
        records = list(ws.read())
        assert len(records) == 1
        assert records[0].content == "hello"
        assert records[0].metadata["path"] == str(tmp_path / "a.txt")

    def test_skips_unchanged_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        ws = WatchSource(str(tmp_path))
        list(ws.read())  # first read
        records = list(ws.read())  # second read — no change
        assert len(records) == 0

    def test_detects_changes(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("v1")
        ws = WatchSource(str(tmp_path))
        list(ws.read())
        p.write_text("v2")
        import time; time.sleep(0.05)
        records = list(ws.read())
        assert len(records) == 1
        assert records[0].content == "v2"

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "empty.txt").write_text("")
        ws = WatchSource(str(tmp_path))
        records = list(ws.read())
        assert len(records) == 0

    def test_skips_whitespace_only(self, tmp_path):
        (tmp_path / "ws.txt").write_text("   \n  \n")
        ws = WatchSource(str(tmp_path))
        records = list(ws.read())
        assert len(records) == 0

    def test_patterns_filter(self, tmp_path):
        (tmp_path / "a.txt").write_text("txt")
        (tmp_path / "b.md").write_text("md")
        ws = WatchSource(str(tmp_path), patterns=["*.txt"])
        records = list(ws.read())
        assert len(records) == 1
        assert records[0].metadata["path"].endswith("a.txt")

    def test_reset_clears_mtimes(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        ws = WatchSource(str(tmp_path))
        list(ws.read())
        ws.reset()
        records = list(ws.read())
        assert len(records) == 1  # re-detected after reset

    def test_name_default(self, tmp_path):
        ws = WatchSource(str(tmp_path / "sub"))
        assert ws.name == f"watch:{(tmp_path / 'sub').name}"

    def test_name_override(self, tmp_path):
        ws = WatchSource(str(tmp_path), name="mywatch")
        assert ws.name == "mywatch"

    def test_directories_skipped(self, tmp_path):
        (tmp_path / "subdir").mkdir()
        ws = WatchSource(str(tmp_path))
        records = list(ws.read())
        assert len(records) == 0

    def test_poll_interval_stored(self, tmp_path):
        ws = WatchSource(str(tmp_path), poll_interval=5.0)
        assert ws.poll_interval == 5.0

    def test_mtime_tracking(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("v1")
        ws = WatchSource(str(tmp_path))
        list(ws.read())
        assert str(p) in ws._seen_mtimes
        old_mtime = ws._seen_mtimes[str(p)]
        assert old_mtime > 0


# ---------------------------------------------------------------------------
# UrlSource — constructor logic only (no network)
# ---------------------------------------------------------------------------

class TestUrlSource:
    def test_name_default(self):
        src = UrlSource("https://example.com/api/data")
        assert src.name == f"url:https://example.com/api/data"[:60]

    def test_name_override(self):
        src = UrlSource("https://example.com", name="custom")
        assert src.name == "custom"

    def test_timeout_default(self):
        src = UrlSource("https://example.com")
        assert src.timeout == 30

    def test_timeout_override(self):
        src = UrlSource("https://example.com", timeout=10)
        assert src.timeout == 10

    def test_parse_json_list(self):
        src = UrlSource("http://x")
        records = list(src._parse_json('[{"content":"a"},{"content":"b"}]'))
        assert len(records) == 2
        assert records[0].content == "a"

    def test_parse_json_dict(self):
        src = UrlSource("http://x")
        records = list(src._parse_json('{"content":"solo","extra":1}'))
        assert len(records) == 1
        assert records[0].content == "solo"
        assert records[0].metadata["extra"] == 1

    def test_parse_json_dict_no_content(self):
        src = UrlSource("http://x")
        records = list(src._parse_json('{"text":"hi"}'))
        # dict without "content" key yields empty string content
        assert records[0].content == ""
        assert records[0].metadata["text"] == "hi"

    def test_parse_json_single_string_item(self):
        src = UrlSource("http://x")
        records = list(src._parse_json('["just a string"]'))
        assert records[0].content == "just a string"

    def test_parse_json_single_non_dict(self):
        src = UrlSource("http://x")
        records = list(src._parse_json("42"))
        # 42 is not a list or dict, so nothing yielded
        assert records == []


# ---------------------------------------------------------------------------
# SseSource — constructor logic only
# ---------------------------------------------------------------------------

class TestSseSource:
    def test_name_default(self):
        src = SseSource("https://example.com/events")
        assert src.name == "sse:https://example.com/events"[:60]

    def test_name_override(self):
        src = SseSource("https://example.com/events", name="custom")
        assert src.name == "custom"

    def test_timeout_default(self):
        src = SseSource("https://x")
        assert src.timeout == 30


# ---------------------------------------------------------------------------
# ApiSource — constructor + _last_id logic
# ---------------------------------------------------------------------------

class TestApiSource:
    def test_name_default(self):
        src = ApiSource("https://api.example.com/items")
        assert src.name == "api:https://api.example.com/items"[:60]

    def test_name_override(self):
        src = ApiSource("https://x", name="myapi")
        assert src.name == "myapi"

    def test_timeout_default(self):
        src = ApiSource("https://x")
        assert src.timeout == 30

    def test_timeout_override(self):
        src = ApiSource("https://x", timeout=5)
        assert src.timeout == 5

    def test_last_id_initially_none(self):
        src = ApiSource("https://x")
        assert src._last_id is None

    def test_headers_default_empty(self):
        src = ApiSource("https://x")
        assert src.headers == {}

    def test_headers_override(self):
        src = ApiSource("https://x", headers={"Auth": "tok"})
        assert src.headers == {"Auth": "tok"}

    def test_poll_interval(self):
        src = ApiSource("https://x", poll_interval=10.0)
        assert src.poll_interval == 10.0
