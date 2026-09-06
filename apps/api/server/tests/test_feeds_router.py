"""Tests for feeds router — RSS and JSON Feed generation."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    from fastapi import FastAPI
    from infrastructure.exception_handlers import register_app_error_handler
    from routers.feeds import router as feeds_router

    app = FastAPI()
    app.include_router(feeds_router)
    register_app_error_handler(app)
    return TestClient(app)


@pytest.fixture
def sample_journal(tmp_path):
    """Create a sample journal file for testing."""
    journal = tmp_path / "notes.journal.jsonl"
    entries = [
        {
            "op": "insert",
            "data": {
                "id": "1",
                "title": "Note 1",
                "body": "Body 1",
                "tags": "training,ml",
                "status": "done",
                "created_at": "2026-09-01T10:00:00Z",
            },
        },
        {
            "op": "insert",
            "data": {
                "id": "2",
                "title": "Note 2",
                "body": "Body 2",
                "tags": "infra",
                "status": "open",
                "created_at": "2026-09-02T10:00:00Z",
            },
        },
        {
            "op": "insert",
            "data": {
                "id": "3",
                "title": "Note 3",
                "body": "Body 3",
                "tags": "training",
                "status": "done",
                "created_at": "2026-09-03T10:00:00Z",
            },
        },
    ]
    journal.write_text("\n".join(json.dumps(e) for e in entries))
    return journal


class TestRSSFeed:
    def test_rss_feed_returns_xml(self, client, sample_journal):
        with patch("routers.feeds.NOTES_JOURNAL", sample_journal):
            resp = client.get("/feeds/rss.xml")
        assert resp.status_code == 200
        assert "application/rss+xml" in resp.headers["content-type"]
        assert '<?xml version="1.0"' in resp.text
        assert "<rss" in resp.text
        assert "<channel>" in resp.text

    def test_rss_feed_has_items(self, client, sample_journal):
        with patch("routers.feeds.NOTES_JOURNAL", sample_journal):
            resp = client.get("/feeds/rss.xml")
        assert resp.status_code == 200
        assert "<item>" in resp.text

    def test_rss_feed_filter_by_tag(self, client, sample_journal):
        with patch("routers.feeds.NOTES_JOURNAL", sample_journal):
            resp = client.get("/feeds/rss.xml?tag=training")
        assert resp.status_code == 200
        # Should contain training notes (1 and 3), not infra note (2)
        assert resp.text.count("<item>") == 2

    def test_rss_feed_filter_by_status(self, client, sample_journal):
        with patch("routers.feeds.NOTES_JOURNAL", sample_journal):
            resp = client.get("/feeds/rss.xml?status=done")
        assert resp.status_code == 200
        assert resp.text.count("<item>") == 2

    def test_rss_feed_limit(self, client, sample_journal):
        with patch("routers.feeds.NOTES_JOURNAL", sample_journal):
            resp = client.get("/feeds/rss.xml?limit=1")
        assert resp.status_code == 200
        assert resp.text.count("<item>") == 1


class TestJSONFeed:
    def test_json_feed_returns_json(self, client, sample_journal):
        with patch("routers.feeds.NOTES_JOURNAL", sample_journal):
            resp = client.get("/feeds/feed.json")
        assert resp.status_code == 200
        assert "application/json" in resp.headers["content-type"]
        data = resp.json()
        assert data["version"] == "https://jsonfeed.org/version/1.1"
        assert "items" in data

    def test_json_feed_has_items(self, client, sample_journal):
        with patch("routers.feeds.NOTES_JOURNAL", sample_journal):
            resp = client.get("/feeds/feed.json")
        data = resp.json()
        assert len(data["items"]) == 3

    def test_json_feed_filter_by_tag(self, client, sample_journal):
        with patch("routers.feeds.NOTES_JOURNAL", sample_journal):
            resp = client.get("/feeds/feed.json?tag=infra")
        data = resp.json()
        assert len(data["items"]) == 1
        assert data["items"][0]["title"] == "Note 2"

    def test_json_feed_limit(self, client, sample_journal):
        with patch("routers.feeds.NOTES_JOURNAL", sample_journal):
            resp = client.get("/feeds/feed.json?limit=2")
        data = resp.json()
        assert len(data["items"]) == 2


class TestFeedHelpers:
    def test_parse_journal_missing_file(self):
        from routers.feeds import _parse_journal

        with patch("routers.feeds.NOTES_JOURNAL", Path("/nonexistent")):
            result = _parse_journal()
        assert result == []

    def test_filter_notes_empty(self):
        from routers.feeds import _filter_notes

        result = _filter_notes([], tag="test")
        assert result == []

    def test_md_to_html(self):
        from routers.feeds import _md_to_html

        result = _md_to_html("# Title")
        assert "<h1>Title</h1>" in result

    def test_build_rss_xml_structure(self):
        from routers.feeds import _build_rss_xml

        notes = [
            {
                "id": "1",
                "title": "Test",
                "body": "Body",
                "tags": "tag1",
                "status": "open",
                "created_at": "2026-09-01T00:00:00Z",
            }
        ]
        xml = _build_rss_xml(notes)
        assert "<item>" in xml
        assert "Test" in xml

    def test_build_json_feed_structure(self):
        from routers.feeds import _build_json_feed

        notes = [
            {
                "id": "1",
                "title": "Test",
                "body": "Body",
                "tags": "tag1",
                "status": "open",
                "created_at": "2026-09-01T00:00:00Z",
            }
        ]
        feed = _build_json_feed(notes)
        assert feed["version"] == "https://jsonfeed.org/version/1.1"
        assert len(feed["items"]) == 1
