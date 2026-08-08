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
_CASUAL_PATTERNS = [
    "how are you",
    "how's it going",
    "how are things",
    "what's up",
    "whats up",
    "how do you do",
    "good morning",
    "good afternoon",
    "good evening",
    "nice to meet you",
]

# Minimum cosine similarity for a fact to be injected into chat context.
# The n-gram embedder produces noisy matches for short generic messages
# (e.g. "Hello!" vs "Paris is the capital of France" = 0.258), so below this
# floor a fact is treated as unrelated and skipped.
MIN_RELEVANCE_SCORE = 0.15

# Minimum character length for a word token to count as content-bearing in the
# topical-overlap gate. Function words ("is", "the", "of", "to", "for") are
# shorter than this, so they are excluded programmatically — no stopword list.
# A fact is only injected when it shares at least one content token with the
# user message, blocking score-passing but topically-unrelated matches such as
# "What color is the sky?" vs the Paris fact (0.224).
_MIN_CONTENT_TOKEN_LEN = 4


def _is_casual_small_talk(text: str) -> bool:
    """True for greetings and small talk that never need knowledge retrieval.

    Args:
        text: raw user message

    Returns:
        True if the message is casual small talk (greetings, pleasantries)

    Side effects:
        - none
    """
    lower = text.lower().strip("?.,! ")
    if not lower:
        return False
    if lower.split()[0] in _CASUAL_GREETINGS:
        return True
    for pat in _CASUAL_PATTERNS:
        if pat in lower:
            return True
    return False


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


def _content_tokens(text: str) -> set:
    """Extract content-bearing word tokens from text.

    Tokens must be at least ``_MIN_CONTENT_TOKEN_LEN`` characters and
    alphanumeric. Shorter tokens are overwhelmingly function words, so this
    filters them algorithmically instead of using a stopword list.

    Args:
        text: raw text

    Returns:
        set of lowercase alphanumeric tokens at least the minimum length

    Side effects:
        - none
    """
    return {
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) >= _MIN_CONTENT_TOKEN_LEN
    }


def _topically_related(query: str, fact: str) -> bool:
    """True when the query and fact share at least one content token.

    A high cosine score alone is not enough for injection — the n-gram
    embedder lets shared function-word n-grams ("is the") inflate unrelated
    matches. Requiring shared content vocabulary filters those without a
    stopword list or stemming table.

    Args:
        query: the user message
        fact: the candidate fact content

    Returns:
        True if both sides have at least one common content token

    Side effects:
        - none
    """
    return bool(_content_tokens(query) & _content_tokens(fact))


def enrich_with_knowledge(
    user_message: str,
    auto_search: bool = True,
    max_facts: int = 5,
    min_score: float = MIN_RELEVANCE_SCORE,
) -> dict:
    """Augment a user message with relevant knowledge facts via vector retrieval.

    Embeds the full user message and queries the VectorStore-backed KnowledgeMemory
    for semantically similar facts. Results below ``min_score`` are discarded, so
    unrelated matches (e.g. greetings that spuriously overlap with a stored fact)
    are never injected. Falls back to live web search when no relevant facts exist
    and the message suggests a need for current information.

    Args:
        user_message: the user's message text
        auto_search: if True, trigger live web search when no relevant facts found
        max_facts: maximum number of fact snippets to return
        min_score: minimum cosine similarity for a fact to be considered relevant

    Returns:
        dict with:
            facts: list of relevant knowledge strings
            source: "memory", "web", or "none"
            topics: extracted topic keywords

    Side effects:
        - may trigger live web search + ingest when auto_search is True
    """
    if _is_casual_small_talk(user_message):
        return {"facts": [], "source": "none", "topics": []}

    memory = get_knowledge_memory()

    def _relevant(results, query):
        facts = []
        for r in results:
            content = r.get("content", "")
            score = r.get("score", 0.0)
            if (
                content
                and len(content) > 20
                and score >= min_score
                and _topically_related(query, content)
            ):
                short = content[:300].strip()
                if short not in facts:
                    facts.append(short)
        return facts

    results = memory.search(user_message, top_k=max_facts)
    facts = _relevant(results, user_message)
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
            facts = _relevant(results, user_message)
            if facts:
                return {
                    "facts": facts[:max_facts],
                    "source": "web",
                    "topics": [],
                }
        except Exception as e:
            logger.warning(f"Live web search failed: {e}", extra={"tag": "INF"})

    return {"facts": [], "source": "none", "topics": []}
