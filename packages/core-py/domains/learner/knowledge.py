"""Knowledge ingestion pipeline: RSS feeds, web search, article scraping, structured memory.

Flow:
  RSS feeds ─┐
  Web search ─┼→ DataFilter (quality/relevance/topic gate) → KnowledgeIngestor
  URLs       ─┘                                          ↓
                                               ┌─────────────────────┐
                                               │ KnowledgeMemory      │
                                               │ (vector store)      │
                                               └─────────┬───────────┘
                                                         ↘
                                               ContinualLearner
                                               (fine-tune on tokens)

Storage backed by VectorStore (InMemory/ChromaDB/Pinecone) instead of JSON files.
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import struct
import hashlib
import logging
import threading
from typing import Optional, Any
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np

logger = logging.getLogger("slo.learner.knowledge")
# Anchor to the repository root so data paths resolve deterministically
# regardless of the process CWD. A CWD-relative path silently reads/writes
# a different directory when the server is launched from anywhere else,
# which previously produced a stray data/ dir and "missing facts" bugs.
_REPO_ROOT = Path(__file__).resolve().parents[4]
KNOWLEDGE_DIR = _REPO_ROOT / "data" / "knowledge"
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)

FEED_STATE_PATH = KNOWLEDGE_DIR / "feeds.json"
VISITED_PATH = KNOWLEDGE_DIR / "visited.json"
ENTRIES_PATH = KNOWLEDGE_DIR / "entries.json"

DEFAULT_FEED_POLL_INTERVAL = 3600  # 1 hour
MAX_ARTICLE_TOKENS = 2048


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeFact:
    """A single knowledge fact."""
    content: str
    topic: str = "general"
    source: str = "manual"
    url: str = ""
    timestamp: float = 0.0
    importance: float = 0.5  # 0-1, auto-estimated


@dataclass
class FeedSubscription:
    url: str
    title: str = ""
    last_fetched: float = 0.0
    poll_interval: float = DEFAULT_FEED_POLL_INTERVAL
    enabled: bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _topic_slug(topic: str) -> str:
    """Normalize a topic string to a filesystem-safe slug."""
    s = topic.lower().strip()
    s = re.sub(r'[^a-z0-9]+', '_', s)
    return s[:64]


# ---------------------------------------------------------------------------
# Document chunking strategies
# ---------------------------------------------------------------------------

def chunk_by_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into fixed-size chunks with optional overlap.

    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap if overlap < chunk_size else end

    return chunks


def chunk_by_paragraph(text: str, max_chunk_size: int = 1000) -> list[str]:
    """Split text by paragraph boundaries (double newlines).

    Merges short paragraphs to avoid too-small chunks.

    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per merged chunk

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    # Split on paragraph boundaries
    paragraphs = re.split(r'\n\s*\n', text.strip())

    chunks = []
    current = ""
    for para in paragraphs:
        para = para.strip()

        if current and len(current) + len(para) + 2 > max_chunk_size:
            if current:
                chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para

    if current:
        chunks.append(current)

    return chunks if chunks else [text.strip()]


def chunk_by_heading(text: str, max_chunk_size: int = 1500) -> list[str]:
    """Split text by markdown-style headings (# ## ###).

    Each heading starts a new chunk. Content under a heading is kept together
    unless it exceeds max_chunk_size.

    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    # Split on heading lines
    lines = text.strip().split('\n')
    sections = []
    current_heading = ""
    current_content = []

    for line in lines:
        if re.match(r'^#{1,3}\s+', line):
            # Save previous section
            if current_heading or current_content:
                content = '\n'.join(current_content).strip()
                if content:
                    sections.append(f"{current_heading}\n{content}" if current_heading else content)
            current_heading = line.strip()
            current_content = []
        else:
            current_content.append(line)

    # Save last section
    if current_heading or current_content:
        content = '\n'.join(current_content).strip()
        if content:
            sections.append(f"{current_heading}\n{content}" if current_heading else content)

    if not sections:
        return chunk_by_fixed_size(text, max_chunk_size)

    # Merge small sections
    chunks = []
    current = ""
    for section in sections:
        if current and len(current) + len(section) + 2 > max_chunk_size:
            chunks.append(current)
            current = section
        else:
            current = f"{current}\n\n{section}" if current else section

    if current:
        chunks.append(current)

    return chunks if chunks else [text.strip()]


def chunk_by_semantic(text: str, max_chunk_size: int = 800, min_chunk_size: int = 100) -> list[str]:
    """Semantic chunking — split at sentence boundaries, group by coherence.

    Tries to keep related sentences together based on:
    - Sentence length (short = likely a transition)
    - Keyword overlap between adjacent sentences

    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk
        min_chunk_size: Minimum characters before forcing a split

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if len(sentences) <= 2:
        return [text.strip()]

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sent = sentences[i]
        current_text = ' '.join(current_chunk)
        current_len = len(current_text)
        sent_len = len(sent)

        # Force split if too long
        if current_len + sent_len > max_chunk_size:
            if current_len >= min_chunk_size:
                chunks.append(current_text)
                current_chunk = [sent]
            else:
                current_chunk.append(sent)
                chunks.append(' '.join(current_chunk))
                current_chunk = []
            continue

        # Check for natural break: short sentence or low keyword overlap
        if sent_len < 40:
            # Short sentence = likely transition — good split point
            if current_len >= min_chunk_size:
                chunks.append(current_text)
                current_chunk = [sent]
                continue

        # Check keyword overlap
        current_words = set(current_text.lower().split())
        sent_words = set(sent.lower().split())
        overlap = len(current_words & sent_words)
        total = len(current_words | sent_words)
        similarity = overlap / max(total, 1)

        if similarity < 0.1 and current_len >= min_chunk_size:
            # Low similarity = topic shift
            chunks.append(current_text)
            current_chunk = [sent]
        else:
            current_chunk.append(sent)

    # Final chunk
    if current_chunk:
        final = ' '.join(current_chunk).strip()
        if final:
            chunks.append(final)

    return chunks if chunks else [text.strip()]


def chunk_text(text: str, strategy: str = "auto", **kwargs) -> list[str]:
    """High-level chunking API with strategy selection.

    Args:
        text: Input text to chunk
        strategy: One of "auto", "fixed", "paragraph", "heading", "semantic"
        **kwargs: Additional arguments passed to the chunking function

    Returns:
        List of text chunks
    """
    strategies = {
        "fixed": chunk_by_fixed_size,
        "paragraph": chunk_by_paragraph,
        "heading": chunk_by_heading,
        "semantic": chunk_by_semantic,
    }

    if strategy == "auto":
        # Auto-select based on content
        if re.search(r'^#{1,3}\s+', text, re.MULTILINE):
            return chunk_by_heading(text, **kwargs)
        elif '\n\n' in text and text.count('\n\n') >= 3:
            return chunk_by_paragraph(text, **kwargs)
        elif len(text) > 2000:
            return chunk_by_semantic(text, **kwargs)
        else:
            return chunk_by_fixed_size(text, **kwargs)

    fn = strategies.get(strategy)
    if fn is None:
        raise ValueError(f"Unknown strategy: {strategy}. Use: {list(strategies.keys())}")
    return fn(text, **kwargs)


def _extract_facts_from_text(text: str) -> list[str]:
    """Extract declarative statements (facts) from text.

    Uses simple heuristics to identify sentences that likely contain
    factual information rather than opinions or questions.

    Args:
        text: Input text to analyze

    Returns:
        List of extracted fact strings
    """
    if not text or len(text.strip()) < 30:
        return []

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    facts = []
    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 20:
            continue

        # Skip questions
        if sent.startswith(('?', 'What', 'How', 'Why', 'When', 'Where', 'Who', 'Is', 'Are', 'Do', 'Does', 'Can', 'Could', 'Would', 'Should')):
            continue

        # Skip imperative (instructions/commands)
        if sent.startswith(('Try', 'Consider', 'Remember', 'Note', 'Make sure', 'Ensure')):
            continue

        # Skip very short or exclamatory
        if len(sent) < 25 or sent.endswith('!'):
            continue

        # Likely factual if it contains declarative patterns
        is_fact = False

        # "X is/are/was/were Y"
        if re.search(r'\b\w+\s+(is|are|was|were)\s+\w+', sent, re.I):
            is_fact = True

        # "X has/have/had Y"
        if re.search(r'\b\w+\s+(has|have|had)\s+\w+', sent, re.I):
            is_fact = True

        # Contains numbers/dates (often factual)
        if re.search(r'\d{4}|\d+\.\d+|\d+%', sent):
            is_fact = True

        # "X can/must/should Y"
        if re.search(r'\b\w+\s+(can|must|should|may|might)\s+\w+', sent, re.I):
            is_fact = True

        if is_fact:
            facts.append(sent)

    return facts


def _extract_topics(text: str, max_topics: int = 5) -> list[str]:
    """Extract likely topic keywords from text via simple TF-like scoring."""
    words = re.findall(r'[a-zA-Z][a-zA-Z-]{2,}', text.lower())
    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "have", "been",
        "some", "them", "than", "what", "when", "who", "will", "with",
        "about", "from", "they", "that", "this", "which", "their",
    }
    counts: dict[str, float] = {}
    for w in words:
        if w not in stopwords and len(w) > 3:
            counts[w] = counts.get(w, 0) + 1.0 + math.log(len(w))
    ranked = sorted(counts.items(), key=lambda x: -x[1])
    return [w for w, _ in ranked[:max_topics]]


def _scrape_article(url: str, timeout: float = 15) -> str:
    """Fetch a URL and extract readable article text."""
    import httpx
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True,
                         headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        import trafilatura
        text = trafilatura.extract(resp.text, include_tables=False, include_images=False,
                                   no_fallback=False)
        if text and len(text.strip()) > 50:
            return text.strip()
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.text, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20)
        return text.strip() or resp.text[:2000]
    except Exception as e:
        logger.warning(f"Article scrape failed {url}: {e}", extra={"tag": "INF"})
        return ""


def _search_ddg(query: str, max_results: int = 5) -> list[dict]:
    """Search DuckDuckGo HTML and return result dicts with title, url, snippet."""
    import httpx
    from urllib.parse import urlparse, unquote
    results: list[dict] = []
    try:
        resp = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            timeout=10,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
        )
        resp.raise_for_status()
        result_blocks = re.finditer(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            resp.text, re.I | re.S
        )
        snippets = list(re.finditer(
            r'class="result__snippet"[^>]*>(.*?)</(?:a|div)>',
            resp.text, re.I | re.S
        ))
        for i, match in enumerate(result_blocks):
            raw_url = match.group(1)
            title = re.sub(r'<[^>]+>', '', match.group(2)).strip()
            url = raw_url
            if "uddg=" in raw_url:
                from urllib.parse import parse_qs, urlparse
                parsed = urlparse(raw_url)
                qs = parse_qs(parsed.query)
                if "uddg" in qs:
                    url = unquote(qs["uddg"][0])
            snippet = ""
            if i < len(snippets):
                snippet = re.sub(r'<[^>]+>', '', snippets[i].group(1)).strip()
            if url and not any(d in url for d in ['duckduckgo.com']):
                results.append({"title": title or snippet[:50], "url": url, "snippet": snippet or title})
                if len(results) >= max_results:
                    break
    except Exception as e:
        logger.warning(f"DDG search failed: {e}", extra={"tag": "INF"})
    return results[:max_results]


# ---------------------------------------------------------------------------
# KnowledgeMemory (VectorStore-backed)
# ---------------------------------------------------------------------------

class KnowledgeMemory:
    """Structured fact storage backed by a VectorStore.

    Facts are embedded via ``EmbeddingService`` and stored as vectors in the store.
    Supports topic-filtered queries, keyword-less semantic search, dedup by
    content hash, and importance scoring (stored as metadata).
    """

    @staticmethod
    def _run_async(coro) -> Any:
        """Run a coroutine to completion (safe from any thread).

        When called from inside a running event loop, running the coroutine on
        that same loop and blocking with ``fut.result()`` deadlocks — the loop
        thread waits on a task only it can schedule. Instead, execute the
        coroutine on a dedicated new event loop in a helper thread so the
        caller never blocks the loop that owns ``coro``'s dependencies.
        """
        import asyncio
        import threading

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

        box: dict = {}

        def _runner() -> None:
            inner = asyncio.new_event_loop()
            try:
                box["result"] = inner.run_until_complete(coro)
            except BaseException as e:  # noqa: BLE001 - propagated to caller
                box["error"] = e
            finally:
                inner.close()

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def __init__(self, vector_store: Optional[Any] = None, load_persisted: bool = True):
        self._lock = threading.Lock()
        self._visited: set[str] = set(self._load_visited()) if load_persisted else set()
        if vector_store is not None:
            self._vector_store = vector_store
        else:
            from domains.inference.vector_store import InMemoryVectorStore
            from domains.infrastructure.embedding_service import get_embedding_service
            dim = get_embedding_service().dimension
            self._vector_store = InMemoryVectorStore(dimension=dim)
            try:
                self._run_async(self._vector_store.connect())
            except Exception:
                pass
        self._embed_fn = None
        self._fact_counter = 0
        if load_persisted:
            self._load_entries()
            self._migrate_from_json_topics()

    def _zero_vec(self) -> list[float]:
        from domains.infrastructure.embedding_service import get_embedding_service
        return [0.0] * get_embedding_service().dimension

    def _get_embedding(self, text: str) -> list[float]:
        if self._embed_fn:
            return self._embed_fn(text)
        from domains.infrastructure.embedding_service import get_embedding_service
        return get_embedding_service().embed(text)

    # ---- persistence -------------------------------------------------------

    def _load_visited(self) -> list[str]:
        if VISITED_PATH.exists():
            try:
                return json.loads(VISITED_PATH.read_text())
            except Exception:
                pass
        return []

    def _save_visited(self):
        VISITED_PATH.write_text(json.dumps(list(self._visited), indent=2))

    def _save_entries(self):
        """Persist all vector entries to JSON for restart survival.

        An emptied store is persisted as ``[]`` when a file already exists,
        so ``clear_all`` survives a restart. The file is only left untouched
        when nothing has ever been persisted (avoids creating an empty file
        for a never-used store).
        """
        try:
            from domains.inference.vector_store import VectorEntry
            entries = getattr(self._vector_store, '_entries', None)
            if not entries and not ENTRIES_PATH.exists():
                return
            data = []
            for eid, entry in (entries or {}).items():
                data.append({
                    "id": entry.id,
                    "vector": entry.vector,
                    "text": entry.text,
                    "metadata": dict(entry.metadata) if entry.metadata else {},
                })
            ENTRIES_PATH.write_text(json.dumps(data))
        except Exception as e:
            logger.warning(f"Failed to save entries: {e}", extra={"tag": "INF"})

    def _load_entries(self):
        """Load persisted entries into vector store on init."""
        if not ENTRIES_PATH.exists():
            return
        try:
            from domains.inference.vector_store import VectorEntry
            data = json.loads(ENTRIES_PATH.read_text())
            if not data:
                return
            # Skip entries with wrong dimension
            expected_dim = self._vector_store.dimension
            entries = []
            skipped = 0
            for d in data:
                vec = d["vector"]
                if len(vec) != expected_dim:
                    skipped += 1
                    continue
                entry = VectorEntry(
                    id=d["id"],
                    vector=vec,
                    text=d["text"],
                    metadata=d.get("metadata", {}),
                )
                entries.append(entry)
            if skipped:
                logger.warning(f"Skipped {skipped} entries with wrong dimension (expected {expected_dim})", extra={"tag": "INF"})
            if entries:
                self._run_async(self._vector_store.upsert(entries))
            if not hasattr(self, '_loaded_log'):
                logger.info(f"Loaded {len(entries)} persisted entries from {ENTRIES_PATH}", extra={"tag": "INF"})
                self._loaded_log = True
        except Exception as e:
            logger.warning(f"Failed to load entries: {e}", extra={"tag": "INF"})

    def _migrate_from_json_topics(self):
        """One-time migration: import old JSON topic files into vector store."""
        legacy_dir = KNOWLEDGE_DIR / "topics"
        if not legacy_dir.exists():
            return
        migration_done = KNOWLEDGE_DIR / ".migrated_to_vector"
        if migration_done.exists():
            return
        migrated = 0
        for f in sorted(legacy_dir.glob("*.json")):
            try:
                facts = json.loads(f.read_text())
                for fact_data in facts:
                    fact = KnowledgeFact(
                        content=fact_data.get("content", ""),
                        topic=fact_data.get("topic", f.stem),
                        source=fact_data.get("source", "legacy"),
                        url=fact_data.get("url", ""),
                        timestamp=fact_data.get("timestamp", 0.0),
                        importance=fact_data.get("importance", 0.5),
                    )
                    if self.add_fact(fact):
                        migrated += 1
            except Exception as e:
                logger.warning(f"Migration failed for {f}: {e}", extra={"tag": "INF"})
        migration_done.write_text(json.dumps({"migrated": migrated, "at": time.time()}))
        logger.info(f"Migrated {migrated} facts from legacy JSON topic files into vector store", extra={"tag": "INF"})

    # ---- storage -----------------------------------------------------------

    def add_fact(self, fact: KnowledgeFact) -> bool:
        """Store a fact. Returns True if new (not duplicate)."""
        content_hash = hashlib.md5(fact.content.encode()).hexdigest()
        with self._lock:
            if content_hash in self._visited:
                return False
            self._visited.add(content_hash)
            self._fact_counter += 1
        import asyncio
        try:
            vec = self._get_embedding(fact.content)
            from domains.inference.vector_store import VectorEntry
            entry = VectorEntry(
                id=f"fact_{self._fact_counter}_{content_hash[:8]}",
                vector=vec,
                text=fact.content,
                metadata={
                    "topic": fact.topic,
                    "source": fact.source,
                    "url": fact.url,
                    "timestamp": fact.timestamp,
                    "importance": fact.importance,
                    "content_hash": content_hash,
                },
            )
            if hasattr(self._vector_store, 'upsert_sync'):
                self._vector_store.upsert_sync([entry])
            else:
                self._run_async(self._vector_store.upsert([entry]))
            self._save_entries()
        except Exception as e:
            logger.warning(f"Vector store upsert failed: {e}", extra={"tag": "INF"})
        self._save_visited()
        return True

    def add_facts(self, facts: list["KnowledgeFact"],
                  vectors: Optional[list[list[float]]] = None) -> int:
        """Store multiple facts, persisting the store once.

        ``add_fact`` rewrites the full persisted store on every call, which
        makes per-fact ingestion O(n) each and O(n·m) for a whole batch. This
        method deduplicates the batch, upserts once, and saves once.

        Args:
            facts: facts to store
            vectors: optional precomputed embeddings aligned with ``facts``;
                when omitted each fact is embedded here

        Returns:
            number of newly stored facts (duplicates excluded)

        Side effects:
            - mutates ``_visited`` / ``_fact_counter``
            - rewrites persisted entries + visited files once per call
        """
        if not facts:
            return 0
        if vectors is not None and len(vectors) != len(facts):
            raise ValueError("vectors must be aligned with facts")

        from domains.inference.vector_store import VectorEntry
        if vectors is None:
            vectors = [self._get_embedding(f.content) for f in facts]

        to_upsert: list[VectorEntry] = []
        with self._lock:
            for i, fact in enumerate(facts):
                content_hash = hashlib.md5(fact.content.encode()).hexdigest()
                if content_hash in self._visited:
                    continue
                self._visited.add(content_hash)
                self._fact_counter += 1
                to_upsert.append(VectorEntry(
                    id=f"fact_{self._fact_counter}_{content_hash[:8]}",
                    vector=vectors[i],
                    text=fact.content,
                    metadata={
                        "topic": fact.topic,
                        "source": fact.source,
                        "url": fact.url,
                        "timestamp": fact.timestamp,
                        "importance": fact.importance,
                        "content_hash": content_hash,
                    },
                ))
        if not to_upsert:
            return 0
        try:
            if hasattr(self._vector_store, 'upsert_sync'):
                self._vector_store.upsert_sync(to_upsert)
            else:
                self._run_async(self._vector_store.upsert(to_upsert))
            self._save_entries()
        except Exception as e:
            logger.warning(f"Vector store batch upsert failed: {e}", extra={"tag": "INF"})
        self._save_visited()
        return len(to_upsert)

    def add_article(self, url: str, title: str, content: str, source: str = "article",
                     chunk_filter: Optional[callable] = None) -> int:
        """Extract topics from article content and store facts. Returns new fact count.

        Args:
            url: source URL
            title: article title
            content: full article text
            source: source type label
            chunk_filter: optional callable(chunk_text, topic) → bool to filter chunks
        """
        topics = _extract_topics(title + " " + content[:500])
        if not topics:
            topics = ["general"]
        now = time.time()
        added = 0
        chunks = [content[i:i+500] for i in range(0, len(content), 500)]
        for topic in topics[:3]:
            for chunk in chunks:
                text = chunk.strip()
                if not text:
                    continue
                if chunk_filter and not chunk_filter(text, topic):
                    continue
                fact = KnowledgeFact(
                    content=text,
                    topic=topic,
                    source=source,
                    url=url,
                    timestamp=now,
                    importance=min(1.0, len(content) / 5000),
                )
                if self.add_fact(fact):
                    added += 1
        return added

    def ingest_from_chat(self, user_message: str, assistant_response: str,
                         max_facts: int = 3) -> list[str]:
        """Extract and store facts from a chat exchange, returning the stored texts.

        Analyzes the assistant's response for declarative statements (facts)
        and stores them in the knowledge base with topic inference.

        Args:
            user_message: The user's message for context
            assistant_response: The assistant's response to extract facts from
            max_facts: Maximum number of facts to extract per exchange

        Returns:
            List of the newly stored fact texts; empty when nothing was
            extracted or everything was already known.

        Side effects:
            - writes new facts into the vector store (persisted).
        """
        facts = _extract_facts_from_text(assistant_response)
        if not facts:
            return []

        # Infer topic from user message
        topics = _extract_topics(user_message)
        topic = topics[0] if topics else "general"

        now = time.time()
        stored = []
        for fact_text in facts[:max_facts]:
            if len(fact_text) < 20:  # skip very short fragments
                continue
            fact = KnowledgeFact(
                content=fact_text,
                topic=topic,
                source="chat",
                timestamp=now,
                importance=0.6,  # slightly above default for chat-sourced
            )
            if self.add_fact(fact):
                stored.append(fact_text)

        if stored:
            logger.info("Auto-ingested %d facts from chat (topic=%s)", len(stored), topic, extra={"tag": "INF"})
        return stored

    def auto_ingest_from_chat(self, user_message: str, assistant_response: str,
                               max_facts: int = 3) -> int:
        """Extract and store facts from a chat exchange.

        Convenience wrapper around ``ingest_from_chat`` returning only the
        number of newly stored facts.

        Args:
            user_message: The user's message for context
            assistant_response: The assistant's response to extract facts from
            max_facts: Maximum number of facts to extract per exchange

        Returns:
            Number of new facts stored
        """
        return len(self.ingest_from_chat(user_message, assistant_response, max_facts))

    # ---- queries -----------------------------------------------------------

    def query(self, topic: str, top_k: int = 10) -> list[dict]:
        """Retrieve facts for a topic, sorted by importance."""
        try:
            if hasattr(self._vector_store, 'count_sync'):
                total = self._vector_store.count_sync()
                results = self._vector_store.query_sync(
                    vector=self._zero_vec(),
                    top_k=total or 1000,
                    filter_metadata={"topic": topic},
                )
            else:
                total = self._run_async(self._vector_store.count())
                results = self._run_async(
                    self._vector_store.query(
                        vector=self._zero_vec(),
                        top_k=total or 1000,
                        filter_metadata={"topic": topic},
                    )
                )
            facts = []
            for r in results:
                fact = {
                    "content": r.text,
                    "topic": r.metadata.get("topic", topic),
                    "source": r.metadata.get("source", ""),
                    "url": r.metadata.get("url", ""),
                    "timestamp": r.metadata.get("timestamp", 0.0),
                    "importance": r.metadata.get("importance", 0.5),
                    "score": r.score,
                }
                facts.append(fact)
            facts.sort(key=lambda f: -f.get("importance", 0.5))
            return facts[:top_k]
        except Exception as e:
            logger.warning(f"Vector store query failed: {e}", extra={"tag": "INF"})
            return []

    def search(self, text: str, top_k: int = 5) -> list[dict]:
        """Semantic search across all facts via vector embedding.

        Args:
            text: search query
            top_k: max results

        Returns:
            list of matching fact dicts, sorted by relevance
        """
        try:
            query_vec = self._get_embedding(text)
            if hasattr(self._vector_store, 'query_sync'):
                results = self._vector_store.query_sync(vector=query_vec, top_k=top_k)
            else:
                results = self._run_async(
                    self._vector_store.query(vector=query_vec, top_k=top_k)
                )
            facts = []
            for r in results:
                fact = {
                    "id": r.id,
                    "content": r.text,
                    "topic": r.metadata.get("topic", "general"),
                    "source": r.metadata.get("source", ""),
                    "url": r.metadata.get("url", ""),
                    "timestamp": r.metadata.get("timestamp", 0.0),
                    "importance": r.metadata.get("importance", 0.5),
                    "score": r.score,
                }
                facts.append(fact)
            return facts
        except Exception as e:
            logger.warning(f"Vector search failed: {e}", extra={"tag": "INF"})
            return []

    def stats(self) -> dict:
        """Return summary statistics."""
        try:
            if hasattr(self._vector_store, 'count_sync'):
                total_facts = self._vector_store.count_sync()
            else:
                total_facts = self._run_async(self._vector_store.count())
        except Exception:
            total_facts = 0
        with self._lock:
            return {
                "topics": max(1, total_facts // 10),
                "total_facts": total_facts,
                "visited_urls": len(self._visited),
            }

    def all_topics(self) -> list[str]:
        return ["general"]

    def get_topic_facts(self, topic: str) -> list[dict]:
        return self.query(topic, top_k=100)

    def list_all(self, top_k: int = 5000) -> list[dict]:
        """Return all stored facts by querying with a zero vector."""
        try:
            if hasattr(self._vector_store, 'query_sync'):
                results = self._vector_store.query_sync(vector=self._zero_vec(), top_k=top_k)
            else:
                results = self._run_async(
                    self._vector_store.query(vector=self._zero_vec(), top_k=top_k)
                )
            facts = []
            for r in results:
                facts.append({
                    "id": r.id,
                    "content": r.text,
                    "topic": r.metadata.get("topic", "general"),
                    "source": r.metadata.get("source", ""),
                    "url": r.metadata.get("url", ""),
                    "timestamp": r.metadata.get("timestamp", 0.0),
                    "importance": r.metadata.get("importance", 0.5),
                    "score": r.score,
                })
            return facts
        except Exception as e:
            logger.warning(f"list_all failed: {e}", extra={"tag": "INF"})
            return []

    def delete_by_id(self, item_id: str) -> bool:
        """Delete a fact by its vector store entry ID."""
        try:
            all_entries = self._run_async(
                self._vector_store.query(vector=self._zero_vec(), top_k=5000)
            )
            removed_hash = None
            for r in all_entries:
                if r.id == item_id:
                    removed_hash = r.metadata.get("content_hash")
                    break
            deleted = self._run_async(self._vector_store.delete([item_id]))
            if deleted and removed_hash:
                with self._lock:
                    self._visited.discard(removed_hash)
                self._save_visited()
                self._save_entries()
            return deleted
        except Exception as e:
            logger.warning(f"delete_by_id failed: {e}", extra={"tag": "INF"})
            return False

    def update_fact(self, item_id: str, content: str,
                    topic: Optional[str] = None,
                    importance: Optional[float] = None) -> bool:
        """Edit an existing fact's text, topic, and/or importance score.

        The vector-store entry keeps its id, so the fact stays discoverable
        and the dedup index migrates from the old content hash to the new one
        (the new text must not already exist as another fact).

        Args:
            item_id: vector-store entry id of the fact to edit.
            content: new fact text; whitespace-only input is rejected.
            topic: optional new topic label; ``None`` keeps the existing one.
            importance: optional importance score in [0, 1]; out-of-range
                values are clamped. ``None`` keeps the existing one.

        Returns:
            True when the fact was updated, False when the id is unknown,
            the new text is empty, or it duplicates another stored fact.

        Side effects:
            - mutates ``_visited`` (hash migration)
            - recomputes the entry embedding and rewrites persisted entries
        """
        text = (content or "").strip()
        if not text:
            return False
        try:
            results = self._run_async(
                self._vector_store.query(vector=self._zero_vec(), top_k=5000)
            )
            target = None
            for r in results:
                if r.id == item_id:
                    target = r
                    break
            if target is None:
                return False
            old_hash = target.metadata.get("content_hash")
            new_hash = hashlib.md5(text.encode()).hexdigest()
            with self._lock:
                if new_hash in self._visited and new_hash != old_hash:
                    return False
                if old_hash and old_hash in self._visited:
                    self._visited.discard(old_hash)
                self._visited.add(new_hash)
            from domains.inference.vector_store import VectorEntry
            meta = dict(target.metadata)
            meta["content_hash"] = new_hash
            if topic is not None:
                meta["topic"] = topic.strip() or meta.get("topic", "general")
            if importance is not None:
                meta["importance"] = min(1.0, max(0.0, float(importance)))
            entry = VectorEntry(
                id=item_id,
                vector=self._get_embedding(text),
                text=text,
                metadata=meta,
            )
            if hasattr(self._vector_store, 'upsert_sync'):
                self._vector_store.upsert_sync([entry])
            else:
                self._run_async(self._vector_store.upsert([entry]))
            self._save_visited()
            self._save_entries()
            return True
        except Exception as e:
            logger.warning(f"update_fact failed: {e}", extra={"tag": "INF"})
            return False

    def clear_all(self) -> int:
        """Delete all stored facts from the vector store. Returns count removed."""
        try:
            all_entries = self._run_async(
                self._vector_store.query(vector=self._zero_vec(), top_k=5000)
            )
            ids = [r.id for r in all_entries]
            if ids:
                self._run_async(self._vector_store.delete(ids))
            with self._lock:
                self._visited.clear()
            self._save_visited()
            self._save_entries()
            return len(ids)
        except Exception as e:
            logger.warning(f"clear_all failed: {e}", extra={"tag": "INF"})
            return 0

    def get_context_string(self, max_items: int = 50) -> str:
        """Format stored facts as a prompt-injection context string."""
        facts = self.list_all(top_k=max_items)
        if not facts:
            return ""
        facts.sort(key=lambda f: -f.get("importance", 0.5))
        lines = ["[KNOWN_FACTS]"]
        for f in facts[:max_items]:
            lines.append(f"- {f['content'][:200]}")
        lines.append("[/KNOWN_FACTS]")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# KnowledgeIngestor
# ---------------------------------------------------------------------------

class KnowledgeIngestor:
    """Multi-source ingestion pipeline.

    Handles:
    - RSS feed polling (background thread)
    - Web search → fetch articles → extract → store
    - Direct URL ingestion
    - Deduplication via visited URL set
    """

    def __init__(self, memory: Optional[KnowledgeMemory] = None, filter_instance=None):
        self.memory = memory or KnowledgeMemory()
        from domains.learner.data_filter import get_data_filter
        self.filter = filter_instance or get_data_filter()
        self._feeds: list[FeedSubscription] = self._load_feeds()
        self._lock = threading.RLock()
        self._running = False
        self._feed_thread: Optional[threading.Thread] = None

    # ---- feeds -------------------------------------------------------------

    def _load_feeds(self) -> list[FeedSubscription]:
        if FEED_STATE_PATH.exists():
            try:
                data = json.loads(FEED_STATE_PATH.read_text())
                return [FeedSubscription(**d) for d in data]
            except Exception as e:
                logger.warning(f"Failed to load feeds: {e}", extra={"tag": "INF"})
        return []

    def _save_feeds(self):
        with self._lock:
            data = [asdict(f) for f in self._feeds]
        FEED_STATE_PATH.write_text(json.dumps(data, indent=2))

    def subscribe_feed(self, url: str, poll_interval: float = DEFAULT_FEED_POLL_INTERVAL) -> bool:
        """Add an RSS/Atom feed to the subscription list."""
        with self._lock:
            if any(f.url == url for f in self._feeds):
                return False
            self._feeds.append(FeedSubscription(
                url=url,
                poll_interval=poll_interval,
                last_fetched=0,
            ))
            self._save_feeds()
        logger.info(f"Subscribed to RSS feed: {url}", extra={"tag": "INF"})
        return True

    def unsubscribe_feed(self, url: str) -> bool:
        with self._lock:
            self._feeds = [f for f in self._feeds if f.url != url]
            self._save_feeds()
        return True

    def list_feeds(self) -> list[dict]:
        with self._lock:
            return [asdict(f) for f in self._feeds]

    def _fetch_feed(self, feed: FeedSubscription) -> list[dict]:
        """Fetch and parse an RSS/Atom feed. Returns list of article dicts."""
        import feedparser
        try:
            parsed = feedparser.parse(feed.url)
            articles = []
            for entry in parsed.entries[:20]:  # max 20 per poll
                url = entry.get("link", "")
                title = entry.get("title", "").strip()
                summary = entry.get("summary", entry.get("description", "")).strip()
                # Strip HTML from summary
                summary = re.sub(r'<[^>]+>', ' ', summary).strip()
                if url and title:
                    articles.append({"url": url, "title": title, "summary": summary})
            return articles
        except Exception as e:
            logger.warning(f"Feed fetch failed for {feed.url}: {e}", extra={"tag": "INF"})
            return []

    def poll_feeds(self, max_articles: int = 10) -> dict:
        """Check all feeds for new articles. Returns {new_articles, rejected, stats}."""
        now = time.time()
        new_articles = 0
        rejected = 0
        with self._lock:
            feeds = list(self._feeds)
        for feed in feeds:
            if not feed.enabled:
                continue
            if now - feed.last_fetched < feed.poll_interval:
                continue
            articles = self._fetch_feed(feed)
            feed.last_fetched = now
            for article in articles[:max_articles]:
                if self._is_visited(article["url"]):
                    continue
                # Fetch full article
                content = _scrape_article(article["url"])
                if not content:
                    content = article.get("summary", article["title"])
                # Run through filter
                passes, reason = self.filter.filter_article(
                    article["url"], article["title"], content,
                    existing_facts=[f.get("content", "") for f in self.memory.search(article["title"], top_k=3)],
                )
                if not passes:
                    rejected += 1
                    self._mark_visited(article["url"])
                    continue
                added = self.memory.add_article(
                    article["url"], article["title"], content, source="rss",
                    chunk_filter=lambda t, topic: self.filter.filter_chunk(t, topic),
                )
                if added > 0:
                    self._mark_visited(article["url"])
                    new_articles += 1
        self._save_feeds()
        return {"new_articles": new_articles, "rejected": rejected, "stats": self.filter.get_stats()}

    # ---- search ------------------------------------------------------------

    def search_and_ingest(self, query: str, max_results: int = 5) -> dict:
        """Search the web, fetch top articles, and store in knowledge memory.

        Returns dict with keys: new_facts, total_tokens, rejected (count), stats.
        """
        results = _search_ddg(query, max_results=max_results)
        total_added = 0
        total_rejected = 0
        for r in results:
            url = r.get("url", "")
            if not url or self._is_visited(url):
                continue
            content = _scrape_article(url)
            if not content:
                content = r.get("snippet", r.get("title", ""))
            # Run through filter
            passes, reason = self.filter.filter_article(
                url, r.get("title", ""), content,
                existing_facts=[f.get("content", "") for f in self.memory.search(r.get("title", ""), top_k=5)],
            )
            if not passes:
                total_rejected += 1
                logger.debug(f"Filter rejected {url[:60]}: {reason}", extra={"tag": "INF"})
                continue
            added = self.memory.add_article(
                url, r.get("title", ""), content, source="search",
                chunk_filter=lambda t, topic: self.filter.filter_chunk(t, topic),
            )
            if added > 0:
                self._mark_visited(url)
                total_added += added
        return {
            "new_facts": total_added,
            "rejected": total_rejected,
            "stats": self.filter.get_stats(),
        }

    def ingest_url(self, url: str) -> dict:
        """Ingest a single URL. Returns {new_facts, title, content_length, rejected, status}."""
        if self._is_visited(url):
            content = _scrape_article(url)
            if not content:
                return {"new_facts": 0, "title": "", "content_length": 0, "rejected": False, "status": "already_visited"}
            # Re-check even if visited — might have been rejected before
        else:
            content = _scrape_article(url)
        if not content or len(content) < 50:
            return {"new_facts": 0, "title": "", "content_length": 0, "rejected": False, "status": "no_content"}
        title = content.split("\n")[0][:100]
        # Run through filter
        passes, reason = self.filter.filter_article(url, title, content)
        if not passes:
            self._mark_visited(url)  # mark visited so we don't retry
            return {"new_facts": 0, "title": title, "content_length": len(content), "rejected": True, "reason": reason, "status": "rejected"}
        added = self.memory.add_article(
            url, title, content, source="direct",
            chunk_filter=lambda t, topic: self.filter.filter_chunk(t, topic),
        )
        self._mark_visited(url)
        return {"new_facts": added, "title": title, "content_length": len(content), "rejected": False, "status": "ok"}

    # ---- visited tracking --------------------------------------------------

    def _is_visited(self, url: str) -> bool:
        h = hashlib.md5(url.encode()).hexdigest()
        return h in self.memory._visited

    def _mark_visited(self, url: str):
        h = hashlib.md5(url.encode()).hexdigest()
        self.memory._visited.add(h)

    # ---- background polling ------------------------------------------------

    def start_background_polling(self, interval: float = 600):
        """Start a daemon thread that polls feeds every `interval` seconds."""
        if self._running:
            return
        self._running = True
        self._feed_thread = threading.Thread(
            target=self._poll_loop, args=(interval,), daemon=True
        )
        self._feed_thread.start()
        logger.info(f"Background feed polling started (interval={interval}s)", extra={"tag": "INF"})

    def stop_background_polling(self):
        self._running = False
        if self._feed_thread and self._feed_thread.is_alive():
            self._feed_thread.join(timeout=5)

    def _poll_loop(self, interval: float):
        while self._running:
            try:
                new = self.poll_feeds(max_articles=5)
                if new.get("new_articles", 0) > 0:
                    logger.info(f"Background poll: {new['new_articles']} new articles ingested", extra={"tag": "INF"})
            except Exception as e:
                logger.warning(f"Background poll error: {e}", extra={"tag": "INF"})
            time.sleep(interval)


# Global singleton
_knowledge_memory: Optional[KnowledgeMemory] = None
_knowledge_ingestor: Optional[KnowledgeIngestor] = None


def get_knowledge_memory() -> KnowledgeMemory:
    global _knowledge_memory
    if _knowledge_memory is None:
        _knowledge_memory = KnowledgeMemory()
    return _knowledge_memory


def get_knowledge_ingestor() -> KnowledgeIngestor:
    global _knowledge_ingestor
    if _knowledge_ingestor is None:
        _knowledge_ingestor = KnowledgeIngestor(memory=get_knowledge_memory())
    return _knowledge_ingestor
