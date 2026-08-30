"""
Feeds Router — RSS 2.0 and JSON Feed generation for dev notes.

Endpoints:
    GET /feeds/rss.xml          — All notes as RSS 2.0
    GET /feeds/rss.xml?tag=training  — Notes filtered by tag
    GET /feeds/rss.xml?limit=20      — Recent N notes
    GET /feeds/rss.xml?status=done   — Notes filtered by status

    GET /feeds/feed.json        — All notes as JSON Feed
    GET /feeds/feed.json?tag=training  — Notes filtered by tag
    GET /feeds/feed.json?limit=20      — Recent N notes

Generates RSS 2.0 and JSON Feed from the dev notes journal.
"""
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import Response

logger = logging.getLogger(__name__)

import os

_BASE_URL = os.environ.get("SLOUGH_BASE_URL", "http://localhost:8000")

# Notes journal path - resolve from project root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
NOTES_JOURNAL = _REPO_ROOT / ".dev-notes" / "store" / "notes.journal.jsonl"


def _parse_journal() -> list[dict]:
    """Parse the JSONL journal and return all notes."""
    notes = []
    if not NOTES_JOURNAL.exists():
        return notes

    with open(NOTES_JOURNAL, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("op") == "insert":
                    note = entry.get("data", {})
                    notes.append({
                        "id": note.get("id", ""),
                        "title": note.get("title", "Untitled"),
                        "body": note.get("body", ""),
                        "tags": note.get("tags", ""),
                        "status": note.get("status", "open"),
                        "created_at": note.get("created_at", ""),
                        "updated_at": note.get("updated_at", ""),
                        "sprint": note.get("sprint", ""),
                        "gh": note.get("gh", ""),
                    })
            except Exception as e:
                logger.warning("Failed to parse journal line: %s", e)
    return notes


def _filter_notes(
    notes: list[dict],
    tag: Optional[str] = None,
    status: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """Filter notes by tag, status, and limit."""
    filtered = notes

    if tag:
        tag_lower = tag.lower()
        filtered = [
            n for n in filtered
            if tag_lower in n["tags"].lower()
        ]

    if status:
        status_lower = status.lower()
        filtered = [
            n for n in filtered
            if n["status"].lower() == status_lower
        ]

    # Sort by created_at descending (newest first)
    filtered.sort(key=lambda x: x.get("created_at", ""), reverse=True)

    if limit:
        filtered = filtered[:limit]

    return filtered


def _build_rss_xml(notes: list[dict], title: str = "sloughGPT Dev Notes") -> str:
    """Build RSS 2.0 XML from notes."""
    rss = ET.Element("rss", version="2.0")
    channel = ET.SubElement(rss, "channel")

    # Channel metadata
    ET.SubElement(channel, "title").text = title
    ET.SubElement(channel, "link").text = f"{_BASE_URL}/feeds/rss.xml"
    ET.SubElement(channel, "description").text = f"Development notes for sloughGPT — {len(notes)} notes"
    ET.SubElement(channel, "language").text = "en-us"
    ET.SubElement(channel, "lastBuildDate").text = datetime.now(timezone.utc).strftime(
        "%a, %d %b %Y %H:%M:%S +0000"
    )
    ET.SubElement(channel, "generator").text = "sloughGPT feeds router"

    # Add items
    for note in notes:
        item = ET.SubElement(channel, "item")
        ET.SubElement(item, "title").text = note["title"]
        ET.SubElement(item, "guid").text = note["id"]
        ET.SubElement(item, "description").text = note["body"][:500] if note["body"] else ""

        # Pub date
        created = note.get("created_at", "")
        if created:
            try:
                dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                ET.SubElement(item, "pubDate").text = dt.strftime(
                    "%a, %d %b %Y %H:%M:%S +0000"
                )
            except Exception:
                pass

        # Tags as categories
        if note["tags"]:
            for tag in note["tags"].split(","):
                tag = tag.strip()
                if tag:
                    ET.SubElement(item, "category").text = tag

        # Status as custom element
        status_elem = ET.SubElement(item, "status")
        status_elem.text = note["status"]

    # Convert to string
    ET.indent(rss, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(rss, encoding="unicode")


def _build_json_feed(notes: list[dict], title: str = "sloughGPT Dev Notes") -> dict:
    """Build JSON Feed 1.1 spec from notes."""
    feed = {
        "version": "https://jsonfeed.org/version/1.1",
        "title": title,
        "home_page_url": f"{_BASE_URL}/feeds/feed.json",
        "feed_url": f"{_BASE_URL}/feeds/feed.json",
        "description": f"Development notes for sloughGPT — {len(notes)} notes",
        "authors": [{"name": "sloughGPT"}],
        "items": [],
    }

    for note in notes:
        item = {
            "id": note["id"],
            "title": note["title"],
            "content_html": _md_to_html(note["body"]) if note["body"] else "",
            "date_published": note["created_at"],
            "tags": [t.strip() for t in note["tags"].split(",") if t.strip()],
            "status": note["status"],
        }
        feed["items"].append(item)

    return feed


def _md_to_html(text: str) -> str:
    """Minimal markdown to HTML conversion for note bodies."""
    import re
    text = text.strip()
    # Headers
    text = re.sub(r"^### (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.+)$", r"<h2>\1</h2>", text, flags=re.MULTILINE)
    text = re.sub(r"^# (.+)$", r"<h1>\1</h1>", text, flags=re.MULTILINE)
    # Bold
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Italic
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    # Line breaks
    text = text.replace("\n\n", "</p><p>")
    text = f"<p>{text}</p>"
    return text


class FeedsRouter:
    """Routes for /feeds/* endpoints."""

    def __init__(self):
        self.router = APIRouter(prefix="/feeds", tags=["feeds"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/rss.xml", self.rss_feed, methods=["GET"])
        self.router.add_api_route("/feed.json", self.json_feed, methods=["GET"])

    async def rss_feed(
        self,
        tag: Optional[str] = Query(None, description="Filter by tag"),
        status: Optional[str] = Query(None, description="Filter by status (open/wip/done/blocked/review/todo)"),
        limit: Optional[int] = Query(None, description="Limit to N most recent notes"),
    ) -> Response:
        """RSS 2.0 feed of dev notes.

        Filter by tag, status, or limit the number of notes returned.
        """
        notes = _parse_journal()
        filtered = _filter_notes(notes, tag=tag, status=status, limit=limit)

        title = "sloughGPT Dev Notes"
        if tag:
            title = f"sloughGPT Notes — {tag}"
        if status:
            title = f"sloughGPT Notes — {status}"

        xml_content = _build_rss_xml(filtered, title=title)

        return Response(
            content=xml_content,
            media_type="application/rss+xml; charset=utf-8",
            headers={
                "Cache-Control": "public, max-age=300",
            },
        )

    async def json_feed(
        self,
        tag: Optional[str] = Query(None, description="Filter by tag"),
        status: Optional[str] = Query(None, description="Filter by status (open/wip/done/blocked/review/todo)"),
        limit: Optional[int] = Query(None, description="Limit to N most recent notes"),
    ) -> Response:
        """JSON Feed 1.1 of dev notes.

        Filter by tag, status, or limit the number of notes returned.
        """
        notes = _parse_journal()
        filtered = _filter_notes(notes, tag=tag, status=status, limit=limit)

        title = "sloughGPT Dev Notes"
        if tag:
            title = f"sloughGPT Notes — {tag}"
        if status:
            title = f"sloughGPT Notes — {status}"

        feed = _build_json_feed(filtered, title=title)
        content = json.dumps(feed, indent=2, ensure_ascii=False)

        return Response(
            content=content,
            media_type="application/json; charset=utf-8",
            headers={
                "Cache-Control": "public, max-age=300",
            },
        )


router = FeedsRouter().router
