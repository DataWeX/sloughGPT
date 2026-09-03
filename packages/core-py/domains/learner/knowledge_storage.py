"""Knowledge Storage adapter bridging KnowledgeRepository to knowledge module.

Provides a clean interface for the knowledge module to store and retrieve
facts and feed subscriptions using the repository layer instead of raw JSON files.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from domains.infrastructure.entity_repositories import (
    KnowledgeEntry,
    KnowledgeRepository,
    FeedState,
)

logger = logging.getLogger("slo.knowledge.storage")


class KnowledgeStorage:
    """Adapter bridging KnowledgeRepository to knowledge module.

    Replaces direct JSON file I/O with structured repository access.
    """

    def __init__(self, data_dir: str | Path):
        self._repo = KnowledgeRepository(data_dir)

    def add_fact(
        self,
        content: str,
        topic: str = "general",
        source: str = "manual",
        url: str = "",
        importance: float = 0.5,
        tags: list[str] | None = None,
    ) -> str:
        """Add a knowledge fact and return its ID."""
        fact_id = f"fact_{uuid.uuid4().hex[:12]}"
        entry = KnowledgeEntry(
            id=fact_id,
            content=content,
            topic=topic,
            source=source,
            url=url,
            timestamp=time.time(),
            importance=importance,
            tags=tags or [],
        )
        self._repo.save_fact(entry)
        return fact_id

    def get_fact(self, fact_id: str) -> Optional[dict]:
        """Retrieve a knowledge fact by ID."""
        entry = self._repo.get_fact(fact_id)
        return entry.to_dict() if entry else None

    def list_facts(self, topic: str | None = None) -> list[dict]:
        """List all facts, optionally filtered by topic."""
        return [f.to_dict() for f in self._repo.list_facts(topic)]

    def search_facts(self, query: str) -> list[dict]:
        """Search facts by content, topic, or source."""
        return [f.to_dict() for f in self._repo.search_facts(query)]

    def delete_fact(self, fact_id: str) -> bool:
        """Delete a knowledge fact."""
        return self._repo.delete_fact(fact_id)

    def count_facts(self) -> int:
        """Count total facts."""
        return self._repo.count_facts()

    def get_visited(self) -> list[str]:
        """Get list of visited URLs from facts metadata."""
        facts = self._repo.list_facts()
        visited = []
        for fact in facts:
            if fact.url and fact.source == "visited":
                visited.append(fact.url)
        return visited

    def mark_visited(self, url: str) -> None:
        """Mark a URL as visited by creating a fact."""
        fact_id = f"visited_{hash(url) & 0xFFFFFFFF:08x}"
        entry = KnowledgeEntry(
            id=fact_id,
            content=f"Visited: {url}",
            topic="visited",
            source="visited",
            url=url,
            timestamp=time.time(),
            importance=0.1,
        )
        self._repo.save_fact(entry)

    def add_feed(self, url: str, title: str = "", poll_interval: float = 3600.0) -> None:
        """Add an RSS feed subscription."""
        feed = FeedState(
            url=url,
            title=title,
            last_fetched=0.0,
            poll_interval=poll_interval,
            enabled=True,
        )
        self._repo.save_feed(feed)

    def get_feed(self, url: str) -> Optional[dict]:
        """Get feed subscription by URL."""
        feed = self._repo.get_feed(url)
        return feed.to_dict() if feed else None

    def list_feeds(self) -> list[dict]:
        """List all feed subscriptions."""
        return [f.to_dict() for f in self._repo.list_feeds()]

    def update_feed_last_fetched(self, url: str) -> None:
        """Update the last_fetched time for a feed."""
        feed = self._repo.get_feed(url)
        if feed:
            feed.last_fetched = time.time()
            self._repo.save_feed(feed)

    def remove_feed(self, url: str) -> bool:
        """Remove a feed subscription."""
        return self._repo.delete_feed(url)
