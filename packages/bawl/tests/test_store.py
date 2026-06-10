"""Tests for bawl.store"""

import json
import tempfile
from pathlib import Path

from bawl.store import save, load, dumps_json_array, save_json_array
from bawl.parse import parse
from bawl.parse import Page


def test_save_load_roundtrip():
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))
    p = Page(url="https://x.com", title="X", text="content")
    save(p, path=str(tmp))
    pages = list(load(str(tmp)))
    assert len(pages) == 1
    assert pages[0].title == "X"
    assert pages[0].text == "content"
    assert pages[0].url == "https://x.com"
    tmp.unlink()


def test_save_append():
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))
    save(Page(url="a", text="1"), path=str(tmp))
    save(Page(url="b", text="2"), path=str(tmp))
    pages = list(load(str(tmp)))
    assert len(pages) == 2
    tmp.unlink()


def test_save_overwrite():
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))
    save(Page(url="a", text="1"), path=str(tmp))
    save(Page(url="b", text="2"), path=str(tmp), append=False)
    pages = list(load(str(tmp)))
    assert len(pages) == 1
    assert pages[0].text == "2"
    tmp.unlink()


def test_load_bad_lines():
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))
    tmp.write_text("not json\n{}\n")
    pages = list(load(str(tmp)))
    assert len(pages) == 1
    tmp.unlink()


def test_load_empty():
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))
    tmp.write_text("")
    pages = list(load(str(tmp)))
    assert pages == []
    tmp.unlink()


def test_dumps_json_array():
    pages = [
        Page(url="https://a.com", title="A"),
        Page(url="https://b.com", title="B"),
    ]
    out = dumps_json_array(pages)
    data = json.loads(out)
    assert isinstance(data, list)
    assert len(data) == 2


def test_save_json_array_to_file():
    tmp = Path(tempfile.mktemp(suffix=".json"))
    pages = [Page(url="https://x.com", title="X")]
    save_json_array(pages, path=str(tmp))
    data = json.loads(tmp.read_text())
    assert data[0]["title"] == "X"
    tmp.unlink()


def test_save_creates_file():
    tmp = Path(tempfile.mktemp(suffix=".jsonl"))
    page = parse("https://example.com")
    assert page is not None
    save(page, path=str(tmp))
    assert tmp.stat().st_size > 0
    tmp.unlink()
