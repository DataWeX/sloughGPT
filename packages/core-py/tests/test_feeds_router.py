"""
Feeds Router Tests
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_server_dir = str(Path(__file__).resolve().parents[3] / "apps" / "api" / "server")
if _server_dir not in sys.path:
    sys.path.insert(0, _server_dir)

from routers.feeds import FeedsRouter, _filter_notes, _build_rss_xml, _build_json_feed, _md_to_html


@pytest.fixture
def app():
    a = FastAPI()
    a.include_router(FeedsRouter().router)
    return a


def _make_client(app):
    return TestClient(app, raise_server_exceptions=False)


SAMPLE_NOTES = [
    {
        "id": "n1", "title": "First Note", "body": "Body one", "tags": "training,ml",
        "status": "done", "created_at": "2026-09-10T10:00:00Z", "updated_at": "", "sprint": "s1", "gh": "",
    },
    {
        "id": "n2", "title": "Second Note", "body": "Body two", "tags": "infra",
        "status": "open", "created_at": "2026-09-09T10:00:00Z", "updated_at": "", "sprint": "", "gh": "",
    },
    {
        "id": "n3", "title": "Third Note", "body": "Body three", "tags": "training,infra",
        "status": "wip", "created_at": "2026-09-11T10:00:00Z", "updated_at": "", "sprint": "", "gh": "",
    },
]


class TestFilterNotes:
    def test_no_filters(self):
        result = _filter_notes(SAMPLE_NOTES)
        assert len(result) == 3

    def test_filter_by_tag(self):
        result = _filter_notes(SAMPLE_NOTES, tag="training")
        assert len(result) == 2
        assert all("training" in n["tags"].lower() for n in result)

    def test_filter_by_status(self):
        result = _filter_notes(SAMPLE_NOTES, status="done")
        assert len(result) == 1
        assert result[0]["id"] == "n1"

    def test_limit(self):
        result = _filter_notes(SAMPLE_NOTES, limit=2)
        assert len(result) == 2

    def test_sorted_newest_first(self):
        result = _filter_notes(SAMPLE_NOTES)
        assert result[0]["id"] == "n3"  # Sep 11
        assert result[-1]["id"] == "n2"  # Sep 9


class TestBuildRssXml:
    def test_valid_rss(self):
        xml_str = _build_rss_xml(SAMPLE_NOTES)
        root = ET.fromstring(xml_str)
        assert root.tag == "rss"
        assert root.get("version") == "2.0"
        items = root.findall(".//item")
        assert len(items) == 3

    def test_empty_notes(self):
        xml_str = _build_rss_xml([])
        root = ET.fromstring(xml_str)
        items = root.findall(".//item")
        assert len(items) == 0

    def test_item_has_title_and_guid(self):
        xml_str = _build_rss_xml([SAMPLE_NOTES[0]])
        root = ET.fromstring(xml_str)
        item = root.find(".//item")
        assert item.find("title").text == "First Note"
        assert item.find("guid").text == "n1"


class TestBuildJsonFeed:
    def test_valid_feed(self):
        feed = _build_json_feed(SAMPLE_NOTES)
        assert feed["version"] == "https://jsonfeed.org/version/1.1"
        assert len(feed["items"]) == 3

    def test_item_has_required_fields(self):
        feed = _build_json_feed([SAMPLE_NOTES[0]])
        item = feed["items"][0]
        assert item["id"] == "n1"
        assert item["title"] == "First Note"
        assert "tags" in item
        assert "status" in item


class TestMdToHtml:
    def test_h1(self):
        assert "<h1>Title</h1>" in _md_to_html("# Title")

    def test_h2(self):
        assert "<h2>Sub</h2>" in _md_to_html("## Sub")

    def test_bold(self):
        assert "<strong>bold</strong>" in _md_to_html("**bold**")

    def test_italic(self):
        assert "<em>italic</em>" in _md_to_html("*italic*")

    def test_paragraphs(self):
        result = _md_to_html("line1\n\nline2")
        assert "<p>" in result
        assert "</p>" in result


class TestRssFeedEndpoint:
    @patch("routers.feeds._parse_journal", return_value=SAMPLE_NOTES)
    def test_returns_rss(self, _mock_parse, app):
        client = _make_client(app)
        resp = client.get("/feeds/rss.xml")
        assert resp.status_code == 200
        assert "application/rss+xml" in resp.headers["content-type"]
        root = ET.fromstring(resp.text)
        items = root.findall(".//item")
        assert len(items) == 3

    @patch("routers.feeds._parse_journal", return_value=SAMPLE_NOTES)
    def test_rss_with_tag_filter(self, _mock_parse, app):
        client = _make_client(app)
        resp = client.get("/feeds/rss.xml?tag=training")
        assert resp.status_code == 200
        root = ET.fromstring(resp.text)
        items = root.findall(".//item")
        assert len(items) == 2

    @patch("routers.feeds._parse_journal", return_value=SAMPLE_NOTES)
    def test_rss_with_limit(self, _mock_parse, app):
        client = _make_client(app)
        resp = client.get("/feeds/rss.xml?limit=1")
        assert resp.status_code == 200
        root = ET.fromstring(resp.text)
        items = root.findall(".//item")
        assert len(items) == 1


class TestJsonFeedEndpoint:
    @patch("routers.feeds._parse_journal", return_value=SAMPLE_NOTES)
    def test_returns_json_feed(self, _mock_parse, app):
        client = _make_client(app)
        resp = client.get("/feeds/feed.json")
        assert resp.status_code == 200
        assert "application/json" in resp.headers["content-type"]
        feed = resp.json()
        assert feed["version"] == "https://jsonfeed.org/version/1.1"
        assert len(feed["items"]) == 3

    @patch("routers.feeds._parse_journal", return_value=SAMPLE_NOTES)
    def test_json_feed_with_status_filter(self, _mock_parse, app):
        client = _make_client(app)
        resp = client.get("/feeds/feed.json?status=done")
        feed = resp.json()
        assert len(feed["items"]) == 1
        assert feed["items"][0]["status"] == "done"

    @patch("routers.feeds._parse_journal", return_value=[])
    def test_empty_feed(self, _mock_parse, app):
        client = _make_client(app)
        resp = client.get("/feeds/feed.json")
        feed = resp.json()
        assert len(feed["items"]) == 0
