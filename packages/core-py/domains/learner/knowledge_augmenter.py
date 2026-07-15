"""Knowledge augmenter — enriches chat with vector-retrieved facts and live web search.

Flow:
  user message → embed → query vector store
    ├─ facts found → inject as [KNOWLEDGE] context
    └─ no facts → trigger live web search → store + re-query

Hooks into the chat pipeline at ``routers/inference.py`` just before generation.
Uses the VectorStore-backed KnowledgeMemory for all retrieval — no keyword extraction.
"""

from __future__ import annotations

import re
import math
import logging
from typing import Optional

from domains.learner.knowledge import (
    get_knowledge_memory,
    get_knowledge_ingestor,
)

logger = logging.getLogger("slo.learner.augmenter")

_QUERY_SIGNALS = [
    "what", "who", "when", "where", "why", "how",
    "explain", "tell me", "define", "describe",
    "latest", "current", "news", "update", "recent",
    "compare", "difference between",
]


_CASUAL_GREETINGS = {"hello", "hi", "hey", "yo", "sup"}
_CASUAL_PATTERNS = ["how are you", "how's it going", "how are things", "what's up", "whats up", "how do you do"]


def _needs_web_search(text: str) -> bool:
    """Check if the message likely needs current/web information."""
    lower = text.lower().strip("?.,! ")
    if lower in _CASUAL_GREETINGS:
        return False
    for pat in _CASUAL_PATTERNS:
        if pat in lower:
            return False
    for signal in _QUERY_SIGNALS:
        if signal in lower:
            return True
    return False


def enrich_with_knowledge(
    user_message: str,
    auto_search: bool = True,
    max_facts: int = 5,
) -> dict:
    """Augment a user message with relevant knowledge facts via vector retrieval.

    Embeds the full user message and queries the VectorStore-backed KnowledgeMemory
    for semantically similar facts. Falls back to live web search when no relevant
    facts exist and the message suggests a need for current information.

    Args:
        user_message: the user's message text
        auto_search: if True, trigger live web search when no relevant facts found
        max_facts: maximum number of fact snippets to return

    Returns:
        dict with:
            facts: list of relevant knowledge strings
            source: "memory", "web", or "none"
            topics: extracted topic keywords
    """
    memory = get_knowledge_memory()

    results = memory.search(user_message, top_k=max_facts)
    if results:
        facts = []
        for r in results:
            content = r.get("content", "")
            if content and len(content) > 20:
                short = content[:300].strip()
                if short not in facts:
                    facts.append(short)
        if facts:
            return {
                "facts": facts[:max_facts],
                "source": "memory",
                "topics": [],
            }

    if auto_search and _needs_web_search(user_message):
        try:
            ingestor = get_knowledge_ingestor()
            ingestor.search_and_ingest(user_message, max_results=2)
            results = memory.search(user_message, top_k=max_facts)
            if results:
                facts = [r["content"][:300].strip() for r in results if r.get("content")]
                return {
                    "facts": facts[:max_facts],
                    "source": "web",
                    "topics": [],
                }
        except Exception as e:
            logger.warning(f"Live web search failed: {e}", extra={"tag": "INF"})

    return {"facts": [], "source": "none", "topics": []}
