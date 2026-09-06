"""Tests for collections.sources — Record, FileSource, UrlSource, etc."""

from __future__ import annotations

import json
import csv
import time
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from domains.collections.sources import (
    Record, FileSource, UrlSource, RssSource, ApiSource, SseSource,
    WatchSource, GeneratorSource, Source,
)


# ── Record ──────────────────────────────────────────────────────────────────


class TestRecord:

    def test_defaults(self):
        r = Record(content="hello")
        assert r.content == "hello"
        assert "timestamp" in r.metadata

    def test_custom_metadata(self):
        r = Record(content="x", metadata={"a": 1})
        assert r.metadata["a"] == 1
        assert "timestamp" in r.metadata

    def test_preserves_timestamp(self):
        r = Record(content="x", metadata={"timestamp": 123})
        assert r.metadata["timestamp"] == 123

    def test_to_dict(self):
        r = Record(content="hi", metadata={"k": "v"})
        d = r.to_dict()
        assert d["content"] == "hi"
        assert d["metadata"]["k"] == "v"


# ── FileSource ──────────────────────────────────────────────────────────────


class TestFileSource:

    def test_name_from_path(self):
        fs = FileSource("/tmp/test.txt")
        assert fs.name == "file:test.txt"

    def test_custom_name(self):
        fs = FileSource("/tmp/test.txt", name="custom")
        assert fs.name == "custom"

    def test_read_nonexistent(self):
        fs = FileSource("/tmp/nonexistent_file_12345.txt")
        assert list(fs.read()) == []

    def test_read_jsonl(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text('{"content": "hello"}\n{"content": "world"}\n')
        fs = FileSource(str(path))
        records = list(fs.read())
        assert len(records) == 2
        assert records[0].content == "hello"
        assert records[1].content == "world"

    def test_read_jsonl_malformed(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"content": "ok"}\nnot json\n{"content": "ok2"}\n')
        fs = FileSource(str(path))
        records = list(fs.read())
        assert len(records) == 3
        assert records[1].content == "not json"

    def test_read_jsonl_empty_lines(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text('\n{"content": "a"}\n\n')
        fs = FileSource(str(path))
        records = list(fs.read())
        assert len(records) == 1

    def test_read_json_list(self, tmp_path):
        path = tmp_path / "list.json"
        path.write_text(json.dumps(["a", "b", "c"]))
        fs = FileSource(str(path))
        records = list(fs.read())
        assert len(records) == 3
        assert records[0].content == "a"

    def test_read_json_dict(self, tmp_path):
        path = tmp_path / "dict.json"
        path.write_text(json.dumps({"content": "single", "extra": True}))
        fs = FileSource(str(path))
        records = list(fs.read())
        assert len(records) == 1
        assert records[0].content == "single"

    def test_read_csv(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("content,label\nhello,1\nworld,2\n")
        fs = FileSource(str(path))
        records = list(fs.read())
        assert len(records) == 2
        assert records[0].content == "hello"
        assert records[0].metadata["label"] == "1"

    def test_read_text(self, tmp_path):
        path = tmp_path / "data.txt"
        path.write_text("line1\nline2\n\nline3\n")
        fs = FileSource(str(path))
        records = list(fs.read())
        assert len(records) == 3
        assert records[0].content == "line1"

    def test_read_json_list_of_dicts(self, tmp_path):
        path = tmp_path / "dicts.json"
        path.write_text(json.dumps([{"content": "a", "k": 1}, {"content": "b"}]))
        fs = FileSource(str(path))
        records = list(fs.read())
        assert len(records) == 2
        assert records[0].metadata["k"] == 1


# ── UrlSource ───────────────────────────────────────────────────────────────


class TestUrlSource:

    def test_name(self):
        us = UrlSource("http://example.com/data")
        assert us.name == "url:http://example.com/data"

    def test_custom_name(self):
        us = UrlSource("http://example.com", name="myapi")
        assert us.name == "myapi"

    def test_read_json_list(self):
        us = UrlSource("http://example.com/api")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps([{"content": "a"}, {"content": "b"}]).encode()
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            records = list(us.read())
        assert len(records) == 2
        assert records[0].content == "a"

    def test_read_json_dict(self):
        us = UrlSource("http://example.com/api")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"content": "single"}).encode()
        mock_resp.headers = {"Content-Type": "application/json"}
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            records = list(us.read())
        assert len(records) == 1
        assert records[0].content == "single"

    def test_read_text(self):
        us = UrlSource("http://example.com/page")
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"line1\nline2"
        mock_resp.headers = {"Content-Type": "text/plain"}
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            records = list(us.read())
        assert len(records) == 2

    def test_read_error(self):
        import urllib.error
        us = UrlSource("http://example.com/fail")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("fail")):
            records = list(us.read())
        assert len(records) == 1
        assert "error" in records[0].metadata


# ── RssSource ───────────────────────────────────────────────────────────────


class TestRssSource:

    def test_init(self):
        rs = RssSource("http://example.com/feed")
        assert rs.name.startswith("rss:")
        assert rs._seen == set()

    def test_reset(self):
        rs = RssSource("http://example.com/feed")
        rs._seen.add("http://seen")
        rs.reset()
        assert rs._seen == set()

    @patch.dict("sys.modules", {"feedparser": MagicMock()})
    def test_read_error(self):
        import urllib.error
        rs = RssSource("http://example.com/feed")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("fail")):
            records = list(rs.read())
        assert records == []


# ── ApiSource ───────────────────────────────────────────────────────────────


class TestApiSource:

    def test_init(self):
        asrc = ApiSource("http://example.com/api")
        assert asrc.name.startswith("api:")
        assert asrc._last_id is None

    def test_read_json(self):
        asrc = ApiSource("http://example.com/api")
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps([
            {"id": "1", "content": "a"},
            {"id": "2", "content": "b"},
        ]).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            records = list(asrc.read())
        assert len(records) == 2
        assert asrc._last_id == "1"

    def test_read_dedup(self):
        asrc = ApiSource("http://example.com/api")
        asrc._last_id = "1"
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps([
            {"id": "1", "content": "a"},
            {"id": "2", "content": "b"},
        ]).encode()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp):
            records = list(asrc.read())
        assert len(records) == 0

    def test_read_error(self):
        import urllib.error
        asrc = ApiSource("http://example.com/api")
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("fail")):
            records = list(asrc.read())
        assert records == []


# ── WatchSource ─────────────────────────────────────────────────────────────


class TestWatchSource:

    def test_init(self):
        ws = WatchSource("/tmp/watch_dir")
        assert ws.name.startswith("watch:")
        assert ws.patterns == ["*"]

    def test_read_new_files(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        ws = WatchSource(str(tmp_path))
        records = list(ws.read())
        assert len(records) == 1
        assert records[0].content == "hello"

    def test_read_no_changes(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        ws = WatchSource(str(tmp_path))
        list(ws.read())
        records = list(ws.read())
        assert len(records) == 0

    def test_reset(self, tmp_path):
        (tmp_path / "a.txt").write_text("hello")
        ws = WatchSource(str(tmp_path))
        list(ws.read())
        ws.reset()
        records = list(ws.read())
        assert len(records) == 1

    def test_patterns(self, tmp_path):
        (tmp_path / "a.txt").write_text("text")
        (tmp_path / "b.json").write_text("{}")
        ws = WatchSource(str(tmp_path), patterns=["*.txt"])
        records = list(ws.read())
        assert len(records) == 1
        assert records[0].content == "text"


# ── GeneratorSource ────────────────────────────────────────────────────────


class TestGeneratorSource:

    def test_read_strings(self):
        gs = GeneratorSource(lambda: ["a", "b", "c"])
        records = list(gs.read())
        assert len(records) == 3
        assert records[0].content == "a"

    def test_read_records(self):
        gs = GeneratorSource(lambda: [Record(content="x"), Record(content="y")])
        records = list(gs.read())
        assert len(records) == 2

    def test_read_dicts(self):
        gs = GeneratorSource(lambda: [{"content": "d", "k": 1}])
        records = list(gs.read())
        assert len(records) == 1
        assert records[0].content == "d"
        assert records[0].metadata["k"] == 1
