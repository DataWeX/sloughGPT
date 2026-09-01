"""
Practical knowledge operations — the real-world features.

Semantic search across files, duplicate detection, auto-categorization,
knowledge gap analysis, and smart context injection.  All built on top
of the SloNet embedder + InMemoryVectorStore pipeline.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# 1. SEMANTIC FILE SEARCH
# ═══════════════════════════════════════════════════════════════════════

class FileIndex:
    """Index code/documentation files for natural-language search.

    Usage::

        idx = FileIndex()
        idx.index_directory("packages/core-py/domains")
        results = idx.search("how does the embedding work")
        for path, line, score, snippet in results:
            print(f"[{score:.3f}] {path}:{line} — {snippet}")
    """

    IGNORE_DIRS = {
        '.git', '__pycache__', 'node_modules', '.venv', 'venv', '.env',
        'dist', 'build', '.next', '.cache', '.pytest_cache', 'site-packages',
    }
    IGNORE_EXTS = {
        '.pyc', '.pyo', '.so', '.dll', '.dylib', '.exe', '.bin',
        '.min.js', '.min.css', '.map', '.lock', '.DS_Store',
    }
    MAX_FILE_SIZE = 500_000  # 500KB

    def __init__(self, embedder=None):
        from domains.inference.vector_store import InMemoryVectorStore
        self._store = InMemoryVectorStore(dimension=384)
        self._embedder = embedder
        self._file_meta: Dict[str, Dict] = {}  # entry_id → {path, line, chunk_idx}

    def _embed(self, text: str) -> List[float]:
        if self._embedder:
            return self._embedder.embed(text)
        from domains.inference.vector_store import simple_embed
        return simple_embed(text)

    def _chunk_code(self, content: str, filepath: str, max_lines: int = 20) -> List[Tuple[str, int]]:
        """Split code into overlapping line-window chunks."""
        lines = content.split('\n')
        chunks = []
        for i in range(0, len(lines), max_lines // 2):
            window = lines[i:i + max_lines]
            text = '\n'.join(window).strip()
            if len(text) > 10:
                chunks.append((text, i + 1))  # 1-indexed line number
        return chunks

    def index_file(self, filepath: str) -> int:
        """Index a single file. Returns number of chunks indexed."""
        try:
            size = os.path.getsize(filepath)
            if size > self.MAX_FILE_SIZE or size == 0:
                return 0
            with open(filepath, 'r', errors='ignore') as f:
                content = f.read()
        except Exception:
            return 0

        chunks = self._chunk_code(content, filepath)
        if not chunks:
            return 0

        from domains.inference.vector_store import VectorEntry
        entries = []
        for text, line_no in chunks:
            entry_id = f"file_{hashlib.md5(f'{filepath}:{line_no}'.encode()).hexdigest()[:12]}"
            vec = self._embed(text)
            try:
                rel_path = str(Path(filepath).relative_to(Path.cwd()))
            except ValueError:
                rel_path = os.path.basename(filepath)
            entry = VectorEntry(
                id=entry_id,
                vector=vec,
                text=text,
                metadata={"path": rel_path, "line": line_no},
            )
            entries.append(entry)
            self._file_meta[entry_id] = {"path": rel_path, "line": line_no}

        if hasattr(self._store, 'upsert_sync'):
            self._store.upsert_sync(entries)
        return len(chunks)

    def index_directory(self, root: str, extensions: Optional[set] = None) -> Dict[str, int]:
        """Index all matching files in a directory tree.

        Args:
            root: directory path
            extensions: file extensions to include (default: .py, .md, .txt, .ts, .tsx, .json)

        Returns:
            dict with stats: {files_indexed, chunks_total, by_ext: {ext: count}}
        """
        extensions = extensions or {'.py', '.md', '.txt', '.ts', '.tsx', '.json', '.yaml'}
        stats = {"files_indexed": 0, "chunks_total": 0, "by_ext": Counter()}

        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in self.IGNORE_DIRS]
            for fname in filenames:
                ext = Path(fname).suffix
                if ext not in extensions:
                    continue
                fpath = os.path.join(dirpath, fname)
                n = self.index_file(fpath)
                if n > 0:
                    stats["files_indexed"] += 1
                    stats["chunks_total"] += n
                    stats["by_ext"][ext] += n

        logger.info("Indexed %d files, %d chunks", stats["files_indexed"], stats["chunks_total"], extra={"tag": "LEARN"})
        return stats

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Search indexed files with a natural-language query.

        Returns list of {path, line, score, snippet}.
        """
        q_vec = self._embed(query)
        results = self._store.query_sync(q_vec, top_k=top_k)

        out = []
        for r in results:
            meta = self._file_meta.get(r.id, r.metadata)
            out.append({
                "path": meta.get("path", "?"),
                "line": meta.get("line", 0),
                "score": r.score,
                "snippet": r.text[:200],
            })
        return out

    @property
    def file_count(self) -> int:
        return len(set(m["path"] for m in self._file_meta.values()))

    @property
    def chunk_count(self) -> int:
        return len(self._file_meta)


# ═══════════════════════════════════════════════════════════════════════
# 2. DUPLICATE / NEAR-DUPLICATE DETECTION
# ═══════════════════════════════════════════════════════════════════════

class DuplicateDetector:
    """Find semantically similar facts before they pollute the knowledge base.

    Usage::

        dup = DuplicateDetector()
        dup.load_from_store(knowledge_memory._vector_store)

        # Before adding a new fact:
        is_dup, best_match, score = dup.check("Neural networks learn from data")
        if is_dup:
            print(f"Already have this: {best_match} (similarity: {score:.3f})")
    """

    def __init__(self, threshold: float = 0.85):
        self._threshold = threshold
        self._store = None

    def load_from_store(self, store) -> None:
        self._store = store

    def check(self, text: str, embed_fn=None, query_vector=None) -> Tuple[bool, Optional[str], float]:
        """Check if text is a near-duplicate of existing facts.

        Args:
            text: text to check
            embed_fn: optional callable(text) -> embedding vector
            query_vector: optional precomputed embedding for ``text``;
                skips the embed step entirely when provided

        Returns:
            (is_duplicate, best_matching_text, similarity_score)
        """
        if not self._store:
            return False, None, 0.0

        if query_vector is not None:
            q_vec = query_vector
        elif embed_fn:
            q_vec = embed_fn(text)
        else:
            from domains.inference.vector_store import simple_embed
            q_vec = simple_embed(text)

        results = self._store.query_sync(q_vec, top_k=1)
        if not results:
            return False, None, 0.0

        best = results[0]
        if best.score >= self._threshold:
            return True, best.text, best.score
        return False, best.text, best.score

    def find_clusters(self, embed_fn=None, threshold: float = 0.80) -> List[List[Dict]]:
        """Find clusters of similar facts across the entire store.

        Returns list of clusters, each cluster is a list of {id, text, score}.
        """
        if not self._store or not hasattr(self._store, '_entries'):
            return []

        entries = list(self._store._entries.values())
        if len(entries) < 2:
            return []

        visited = set()
        clusters = []

        for i, e1 in enumerate(entries):
            if e1.id in visited:
                continue
            cluster = [{"id": e1.id, "text": e1.text}]
            visited.add(e1.id)

            v1 = np.asarray(e1.vector, dtype=np.float64)
            for j in range(i + 1, len(entries)):
                e2 = entries[j]
                if e2.id in visited:
                    continue
                v2 = np.asarray(e2.vector, dtype=np.float64)
                cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
                if cos >= threshold:
                    cluster.append({"id": e2.id, "text": e2.text, "score": cos})
                    visited.add(e2.id)

            if len(cluster) > 1:
                clusters.append(cluster)

        return sorted(clusters, key=lambda c: len(c), reverse=True)


# ═══════════════════════════════════════════════════════════════════════
# 3. AUTO-CATEGORIZATION
# ═══════════════════════════════════════════════════════════════════════

class AutoCategorizer:
    """Auto-assign topics to incoming knowledge based on existing categories.

    Usage::

        cat = AutoCategorizer()
        cat.load_from_store(knowledge_memory._vector_store)

        topic = cat.categorize("Backpropagation computes gradients through neural layers")
        # → "machine_learning"
    """

    def __init__(self, min_score: float = 0.3):
        self._min_score = min_score
        self._store = None
        self._topic_examples: Dict[str, List[str]] = defaultdict(list)

    def load_from_store(self, store) -> None:
        """Build topic centroids from existing vector store entries."""
        self._store = store
        if not store or not hasattr(store, '_entries'):
            return

        # Group texts by topic
        topic_texts = defaultdict(list)
        for entry in store._entries.values():
            topic = entry.metadata.get("topic", "unknown")
            topic_texts[topic].append(entry.text)

        self._topic_examples = topic_texts

    def _compute_topic_centroid(self, topic: str) -> Optional[np.ndarray]:
        """Compute average embedding for a topic."""
        texts = self._topic_examples.get(topic, [])
        if not texts:
            return None

        from domains.inference.vector_store import simple_embed
        vecs = [simple_embed(t) for t in texts[:20]]  # cap at 20 examples
        return np.mean(vecs, axis=0)

    def categorize(self, text: str, embed_fn=None) -> str:
        """Assign the best topic to a text based on centroid similarity.

        Returns topic string, or 'general' if no good match.
        """
        if not self._topic_examples:
            return "general"

        if embed_fn:
            q_vec = np.asarray(embed_fn(text), dtype=np.float64)
        else:
            from domains.inference.vector_store import simple_embed
            q_vec = np.asarray(simple_embed(text), dtype=np.float64)

        best_topic = "general"
        best_score = 0.0

        for topic in self._topic_examples:
            centroid = self._compute_topic_centroid(topic)
            if centroid is None:
                continue
            cos = float(np.dot(q_vec, centroid) / (np.linalg.norm(q_vec) * np.linalg.norm(centroid) + 1e-10))
            if cos > best_score:
                best_score = cos
                best_topic = topic

        if best_score >= self._min_score:
            return best_topic
        return "general"

    def suggest_topics(self, text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Suggest top-k topics with scores for a text."""
        if not self._topic_examples:
            return []

        from domains.inference.vector_store import simple_embed
        q_vec = np.asarray(simple_embed(text), dtype=np.float64)

        scored = []
        for topic in self._topic_examples:
            centroid = self._compute_topic_centroid(topic)
            if centroid is None:
                continue
            cos = float(np.dot(q_vec, centroid) / (np.linalg.norm(q_vec) * np.linalg.norm(centroid) + 1e-10))
            scored.append((topic, cos))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# ═══════════════════════════════════════════════════════════════════════
# 4. KNOWLEDGE GAP DETECTOR
# ═══════════════════════════════════════════════════════════════════════

class KnowledgeGapDetector:
    """Find what topics are under-represented in your knowledge base.

    Usage::

        gap = KnowledgeGapDetector()
        gap.load_from_store(knowledge_memory._vector_store)

        gaps = gap.find_gaps(seed_topics=["security", "performance", "testing"])
        for topic, coverage, suggestion in gaps:
            print(f"Gap: {topic} — {suggestion}")
    """

    def __init__(self):
        self._store = None
        self._topic_counts: Dict[str, int] = {}

    def load_from_store(self, store) -> None:
        self._store = store
        if not store or not hasattr(store, '_entries'):
            return

        counts = Counter()
        for entry in store._entries.values():
            topic = entry.metadata.get("topic", "unknown")
            counts[topic] += 1
        self._topic_counts = dict(counts)

    def find_gaps(
        self,
        seed_topics: Optional[List[str]] = None,
        expand_with: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Find knowledge gaps.

        Args:
            seed_topics: known topics to check coverage for
            expand_with: additional keywords to expand the search

        Returns:
            list of {topic, count, coverage_pct, suggestion}
        """
        if not self._store or not self._topic_counts:
            return []

        total = sum(self._topic_counts.values())
        if total == 0:
            return []

        # Build topic list
        topics = set(self._topic_counts.keys())
        if seed_topics:
            topics.update(seed_topics)

        gaps = []
        for topic in sorted(topics):
            count = self._topic_counts.get(topic, 0)
            coverage = count / total * 100

            if count == 0:
                suggestion = f"No facts about '{topic}' — ingest articles or add notes"
            elif coverage < 5:
                suggestion = f"Only {count} facts ({coverage:.1f}%) — consider adding more"
            elif coverage > 30:
                suggestion = f"Dominant topic ({count} facts, {coverage:.1f}%) — well covered"
            else:
                continue  # adequately covered

            gaps.append({
                "topic": topic,
                "count": count,
                "coverage_pct": round(coverage, 1),
                "suggestion": suggestion,
            })

        gaps.sort(key=lambda g: g["count"])
        return gaps

    def find_sparse_regions(self, embed_fn=None, n_regions: int = 5) -> List[Dict[str, Any]]:
        """Find embedding-space regions with few facts (potential blind spots).

        Divides the 384-dim space into grid cells and counts facts per cell.
        """
        if not self._store or not hasattr(self._store, '_entries'):
            return []

        entries = list(self._store._entries.values())
        if len(entries) < 10:
            return []

        # Project to first 2 dims for grid analysis
        vecs = np.array([e.vector[:2] for e in entries])
        min_vals = vecs.min(axis=0)
        max_vals = vecs.max(axis=0)
        range_vals = max_vals - min_vals + 1e-10

        # Create 10x10 grid
        grid = defaultdict(int)
        for v in vecs:
            gx = int((v[0] - min_vals[0]) / range_vals[0] * 9)
            gy = int((v[1] - min_vals[1]) / range_vals[1] * 9)
            grid[(gx, gy)] += 1

        # Find empty cells
        sparse = []
        for gx in range(10):
            for gy in range(10):
                if grid[(gx, gy)] == 0:
                    cx = min_vals[0] + (gx + 0.5) / 10 * range_vals[0]
                    cy = min_vals[1] + (gy + 0.5) / 10 * range_vals[1]
                    sparse.append({
                        "grid_cell": (gx, gy),
                        "count": 0,
                        "center": [float(cx), float(cy)],
                        "suggestion": "Empty region — no knowledge indexed here",
                    })

        return sparse[:n_regions]


# ═══════════════════════════════════════════════════════════════════════
# 5. SMART CONTEXT INJECTION
# ═══════════════════════════════════════════════════════════════════════

class SmartContextInjector:
    """Automatically pull relevant knowledge into chat without manual injection.

    Usage::

        injector = SmartContextInjector(knowledge_memory)

        # Before generating a response:
        context = injector.get_context("How do I train a model?", max_chars=500)
        # → "Relevant knowledge:\n- Training requires labeled data...\n- Gradient descent..."
    """

    def __init__(self, knowledge_memory=None, min_score: float = 0.25, max_facts: int = 5):
        self._memory = knowledge_memory
        self._min_score = min_score
        self._max_facts = max_facts

    def get_context(self, user_message: str, max_chars: int = 500) -> str:
        """Get relevant knowledge context for a user message.

        Returns formatted string with relevant facts, or empty string if none found.
        """
        if not self._memory:
            return ""

        results = self._memory.search(user_message, top_k=self._max_facts)
        if not results:
            return ""

        # Filter by score
        relevant = [r for r in results if r.get("score", 0) >= self._min_score]
        if not relevant:
            return ""

        # Build context string
        parts = []
        char_count = 0
        for fact in relevant:
            text = fact["content"]
            if char_count + len(text) > max_chars:
                break
            parts.append(f"- {text}")
            char_count += len(text)

        if not parts:
            return ""

        return "Relevant knowledge:\n" + "\n".join(parts)

    def get_context_for_system(self, user_message: str, system_prompt: str, max_chars: int = 300) -> str:
        """Inject knowledge into system prompt if relevant facts exist."""
        context = self.get_context(user_message, max_chars=max_chars)
        if not context:
            return system_prompt
        return f"{system_prompt}\n\n{context}"

    def should_inject(self, user_message: str) -> bool:
        """Check if knowledge injection would be useful for this message."""
        if not self._memory:
            return False
        results = self._memory.search(user_message, top_k=1)
        if not results:
            return False
        return results[0].get("score", 0) >= self._min_score


# ═══════════════════════════════════════════════════════════════════════
# BULK OPERATIONS
# ═══════════════════════════════════════════════════════════════════════

class BulkProcessor:
    """Process large batches of text efficiently.

    Usage::

        bp = BulkProcessor()
        results = bp.ingest_texts(texts, topic="documentation")
        report = bp.get_report()
    """

    def __init__(self, knowledge_memory=None):
        self._memory = knowledge_memory
        self._added = 0
        self._skipped = 0
        self._errors = 0

    def ingest_texts(
        self,
        texts: List[str],
        topic: str = "imported",
        source: str = "bulk",
        dedup_threshold: float = 0.85,
        progress_callback=None,
    ) -> Dict[str, int]:
        """Ingest a batch of texts with deduplication.

        Args:
            texts: list of text strings
            topic: topic tag
            source: source label
            dedup_threshold: skip if similarity > this
            progress_callback: optional callable(current, total)

        Returns:
            {added, skipped, errors}
        """
        if not self._memory:
            return {"added": 0, "skipped": 0, "errors": len(texts)}

        dup = DuplicateDetector(threshold=dedup_threshold)
        dup.load_from_store(self._memory._vector_store)

        self._added = 0
        self._skipped = 0
        self._errors = 0

        from domains.learner.knowledge import KnowledgeFact

        facts = []
        vectors = []
        for i, text in enumerate(texts):
            if progress_callback:
                progress_callback(i + 1, len(texts))

            if not text or len(text.strip()) < 10:
                self._skipped += 1
                continue

            try:
                vec = self._memory._get_embedding(text)
                is_dup, _, score = dup.check(text, query_vector=vec)
                if is_dup:
                    self._skipped += 1
                    continue

                facts.append(KnowledgeFact(
                    content=text.strip(),
                    topic=topic,
                    source=source,
                    importance=min(1.0, len(text) / 2000),
                ))
                vectors.append(vec)
            except Exception as e:
                logger.warning("Bulk ingest error at %d: %s", i, e, extra={"tag": "LEARN"})
                self._errors += 1

        if facts:
            self._added = self._memory.add_facts(facts, vectors=vectors)
            self._skipped += len(facts) - self._added

        return self.get_report()

    def get_report(self) -> Dict[str, int]:
        return {"added": self._added, "skipped": self._skipped, "errors": self._errors}
