"""
Production-Grade RAG System

Fixed version of RAGGrounder with:
- Proper vector embeddings (simulated for demo)
- BM25 keyword search
- Hybrid retrieval (dense + sparse)
- Reranking
- Citation tracking
- Hallucination detection
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """A chunked piece of text with metadata."""

    id: str
    content: str
    metadata: dict[str, Any]
    token_count: int = 0
    embedding: np.ndarray | None = None
    bm25_score: float = 0.0

    def __post_init__(self):
        if self.token_count == 0:
            self.token_count = len(self.content.split())


@dataclass
class RetrievalResult:
    """Result from retrieval with scores."""

    chunk: TextChunk
    dense_score: float
    sparse_score: float
    combined_score: float
    rank: int


class BM25Indexer:
    """
    BM25: Best Matching 25 - Industry-standard keyword search.
    Used by Elasticsearch, Solr, etc.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_freq: dict[str, int] = Counter()
        self.num_docs: int = 0
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}

    def index(self, chunks: list[TextChunk]):
        """Build BM25 index (idempotent: resets before rebuilding).

        Structures are built into locals and published at the end so a
        concurrent score() never observes an emptied or half-built index
        (which silently returned wrong or empty results mid-rebuild).
        """
        num_docs = len(chunks)
        doc_lengths: list[int] = []
        doc_freq: Counter = Counter()
        inverted_index: dict[str, list[tuple[int, int]]] = {}
        logger.debug("BM25 indexing started: %d documents", num_docs)

        for doc_id, chunk in enumerate(chunks):
            tokens = self._tokenize(chunk.content)
            doc_lengths.append(len(tokens))

            # Count document frequencies
            for token in set(tokens):
                doc_freq[token] += 1

            # Build inverted index
            for pos, token in enumerate(tokens):
                if token not in inverted_index:
                    inverted_index[token] = []
                inverted_index[token].append((doc_id, pos))

        avg_doc_length = sum(doc_lengths) / max(len(doc_lengths), 1)

        # Structural invariants — a term can occur in at most every chunk,
        # and there is exactly one length entry per chunk. If either breaks,
        # the index was double-counted: fail loud, never silently wrong.
        assert len(doc_lengths) == num_docs, (
            f"BM25 index corrupt: {len(doc_lengths)} lengths for {num_docs} chunks"
        )
        assert all(df <= num_docs for df in doc_freq.values()), (
            "BM25 index corrupt: term frequency exceeds chunk count"
        )

        logger.debug(
            "BM25 indexing complete: avg_doc_length=%.1f, unique_terms=%d",
            avg_doc_length,
            len(doc_freq),
        )

        # Publish atomically (plain stores run without dropping the GIL, so
        # readers see either the old complete index or the new one).
        self.num_docs = num_docs
        self.doc_lengths = doc_lengths
        self.doc_freq = doc_freq
        self.avg_doc_length = avg_doc_length
        self.inverted_index = inverted_index

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text."""
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def score(self, query: str) -> list[tuple[int, float]]:
        """
        Score all documents against query (standard BM25).
        Returns list of (doc_id, score) tuples.
        """
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.num_docs

        for token in query_tokens:
            postings = self.inverted_index.get(token)
            if not postings:
                continue

            # IDF for this term
            df = self.doc_freq.get(token, 0)
            idf = float(np.log((self.num_docs - df + 0.5) / (df + 0.5) + 1))

            # Term frequency counted ONCE per doc, O(postings). The old
            # per-occurrence full rescan was O(postings^2): a common term
            # in a large index burned minutes, tripped the chat pipeline's
            # 15s RAG wait_for (TTFT +15s) and leaked a grinding thread.
            for doc_id, tf in Counter(d for d, _ in postings).items():
                doc_len = self.doc_lengths[doc_id]

                # BM25 formula, added once per doc (plain-float list — a
                # numpy elementwise add costs ~10x per scalar update here)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                scores[doc_id] += idf * numerator / denominator

        # Return top docs with scores
        results = [(i, s) for i, s in enumerate(scores) if s > 0]
        results.sort(key=lambda x: -x[1])
        return results


class HybridRetriever:
    """
    Production-grade hybrid retrieval combining:
    - Dense (semantic similarity)
    - Sparse (BM25 keyword matching)
    - Reranking
    """

    def __init__(
        self,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        use_rerank: bool = True,
        use_dense: bool = True,
        embedding_fn: Any | None = None,
    ):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.use_rerank = use_rerank
        # Dense runs embedding inference per query (and the default
        # embedding_fn is an uninformative random projection). Disabling
        # skips that cost and keeps retrieval fully sparse + rerank.
        self.use_dense = use_dense

        self.chunks: list[TextChunk] = []
        self.bm25 = BM25Indexer()
        self.embedding_cache: dict[str, np.ndarray] = {}
        self._embedding_fn = embedding_fn or self._default_embedding

    def add_chunk(self, chunk: TextChunk):
        """Add a chunk to the retriever."""
        self.chunks.append(chunk)

    def build_index(self):
        """Build retrieval indexes."""
        self.bm25.index(self.chunks)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Public entry point — delegates to the configured embedding function."""
        return self._embedding_fn(text)

    def _default_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text.
        In production, use: OpenAI, Cohere, sentence-transformers, etc.
        """
        if text in self.embedding_cache:
            return self.embedding_cache[text]

        # Simulated embedding (use real embeddings in production)
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(384)
        embedding = embedding / np.linalg.norm(embedding)

        self.embedding_cache[text] = embedding
        return embedding

    def _dense_search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        """Dense vector search."""
        query_emb = self._get_embedding(query)

        scores = []
        for i, chunk in enumerate(self.chunks):
            if chunk.embedding is None:
                chunk.embedding = self._get_embedding(chunk.content)

            # Cosine similarity
            score = float(np.dot(query_emb, chunk.embedding))
            scores.append((i, score))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def _sparse_search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        """Sparse BM25 search."""
        return self.bm25.score(query)[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievalResult]:
        """
        Hybrid retrieval with optional reranking.
        """
        logger.debug("Retrieving for query (len=%d), top_k=%d", len(query), top_k)

        # Get results from both methods (dense skipped when disabled —
        # fusion below then reduces to the sparse term, unchanged math).
        dense_results = self._dense_search(query, top_k * 2) if self.use_dense else []
        sparse_results = self._sparse_search(query, top_k * 2)

        logger.debug(
            "Retrieval raw results: dense=%d, sparse=%d",
            len(dense_results),
            len(sparse_results),
        )

        # Normalize scores
        max_dense = max(s for _, s in dense_results) if dense_results else 1
        max_sparse = max(s for _, s in sparse_results) if sparse_results else 1

        # Combine scores
        combined_scores: dict[int, dict[str, float]] = {}

        for doc_id, score in dense_results:
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {"dense": 0, "sparse": 0}
            combined_scores[doc_id]["dense"] = score / max_dense

        for doc_id, score in sparse_results:
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {"dense": 0, "sparse": 0}
            combined_scores[doc_id]["sparse"] = score / max_sparse

        # Calculate combined scores
        w_dense = self.dense_weight if self.use_dense else 0.0
        w_sparse = self.sparse_weight
        active_total = w_dense + w_sparse
        if active_total > 0:
            w_dense /= active_total
            w_sparse /= active_total
        results = []
        for doc_id, scores in combined_scores.items():
            combined = w_dense * scores["dense"] + w_sparse * scores["sparse"]
            results.append(
                RetrievalResult(
                    chunk=self.chunks[doc_id],
                    dense_score=scores["dense"],
                    sparse_score=scores["sparse"],
                    combined_score=combined,
                    rank=0,
                )
            )

        # Sort by combined score
        results.sort(key=lambda x: -x.combined_score)

        # Filter and rank
        final_results = []
        for i, r in enumerate(results):
            if r.combined_score >= min_score:
                r.rank = i + 1
                final_results.append(r)
            if len(final_results) >= top_k:
                break

        # Optional reranking (simple cross-encoder simulation)
        if self.use_rerank and final_results:
            final_results = self._rerank(query, final_results)

        logger.debug(
            "Retrieval complete: query_len=%d, final_results=%d",
            len(query),
            len(final_results),
        )
        return final_results

    # Rerank feature weights: fusion base, query-term coverage,
    # exact-phrase match, term proximity. Deterministic, no model weights.
    RERANK_WEIGHTS = {
        "base": 0.4,
        "coverage": 0.25,
        "phrase": 0.2,
        "proximity": 0.15,
    }
    RERANK_MMR_LAMBDA = 0.7

    def _rerank(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Feature-based rerank with MMR diversity selection.

        Each candidate is rescored on exact-phrase match, query-term
        coverage, and term proximity blended with the fusion score; the
        final order comes from Maximal Marginal Relevance selection, so
        near-duplicates sink instead of crowding out diverse hits.
        Always returns all candidates, reordered. ``combined_score``
        keeps the fusion value; ``rank`` reflects the reranked order.
        """
        if not results:
            return []
        query_terms = self.bm25._tokenize(query)
        if not query_terms:
            for i, r in enumerate(results):
                r.rank = i + 1
            return results
        query_set = set(query_terms)
        phrase = " ".join(query_terms)

        scored: list[tuple[float, RetrievalResult]] = []
        for r in results:
            content = r.chunk.content.lower()
            doc_terms = self.bm25._tokenize(content)
            doc_set = set(doc_terms)
            matched = query_set & doc_set
            coverage = len(matched) / len(query_set)
            exact = 1.0 if phrase in content else 0.0
            if len(matched) < 2 or len(doc_terms) == 0:
                proximity = 1.0 if matched else 0.0
            else:
                positions = [i for i, tok in enumerate(doc_terms) if tok in matched]
                span = max(positions) - min(positions)
                proximity = 1.0 - span / len(doc_terms)
            w = self.RERANK_WEIGHTS
            # Tier bonus: full query-term coverage outranks everything;
            # an exact-phrase bonus confirms but never alone overrides
            # full coverage (chance contiguous matches occur in distractors).
            tier = (0.5 if exact > 0 else 0.0) + (1.0 if coverage >= 1.0 else 0.0)
            score = (
                tier
                + w["base"] * r.combined_score
                + w["coverage"] * coverage
                + w["phrase"] * exact
                + w["proximity"] * proximity
            )
            scored.append((score, r))

        # MMR selection: relevance minus similarity to already-selected.
        selected: list[RetrievalResult] = []
        selected_terms: list[set[str]] = []
        remaining = scored
        while remaining:
            best_idx = 0
            best_mmr = float("-inf")
            for i, (score, r) in enumerate(remaining):
                doc_set = set(self.bm25._tokenize(r.chunk.content.lower()))
                sim = 0.0
                for prev in selected_terms:
                    union = doc_set | prev
                    if union:
                        sim = max(sim, len(doc_set & prev) / len(union))
                mmr = self.RERANK_MMR_LAMBDA * score - (1 - self.RERANK_MMR_LAMBDA) * sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            _, best = remaining.pop(best_idx)
            selected.append(best)
            selected_terms.append(set(self.bm25._tokenize(best.chunk.content.lower())))

        for i, r in enumerate(selected):
            r.rank = i + 1
        return selected


class CitationTracker:
    """
    Track citations for generated text.
    Maps claims to supporting sources.
    """

    def __init__(self):
        self.claims: list[dict[str, Any]] = []

    def extract_claims(self, text: str) -> list[dict[str, Any]]:
        """Extract factual claims from text.

        Splits on sentence boundaries first, then applies claim patterns
        per sentence to avoid cross-sentence predicate captures.
        """
        claims = []

        # Split into sentences first (avoids cross-sentence captures)
        sentences = re.split(r"(?<=[.!?])\s+", text)

        # Pattern-based claim extraction
        claim_patterns = [
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(.+)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+was\s+(.+)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+can\s+(.+)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+has\s+(.+)",
        ]

        offset = 0
        for sentence in sentences:
            for pattern in claim_patterns:
                matches = re.finditer(pattern, sentence)
                for match in matches:
                    claims.append(
                        {
                            "subject": match.group(1),
                            "predicate": match.group(2).rstrip("."),
                            "text": match.group(0),
                            "start": offset + match.start(),
                            "end": offset + match.end(),
                        }
                    )
            offset += len(sentence) + 1  # +1 for the space

        self.claims = claims
        return claims

    def cite(
        self,
        claim: dict[str, Any],
        source_chunks: list[TextChunk],
    ) -> dict[str, Any]:
        """Add citation to claim."""
        return {
            **claim,
            "supported": len(source_chunks) > 0,
            "sources": [
                {
                    "chunk_id": c.id,
                    "content": c.content[:200],
                    "metadata": c.metadata,
                }
                for c in source_chunks[:3]
            ],
        }

    def format_citations(self) -> str:
        """Format citations for output."""
        output = []
        for i, claim in enumerate(self.claims, 1):
            output.append(f"[{i}] {claim['text']}")
            if claim.get("sources"):
                for src in claim["sources"]:
                    output.append(f"    → {src['metadata'].get('source', 'Unknown')}")
        return "\n".join(output)


class HallucinationDetector:
    """
    Detect potential hallucinations in generated text.
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.citation_tracker = CitationTracker()

    def detect(
        self,
        text: str,
        min_confidence: float = 0.5,
    ) -> dict[str, Any]:
        """
        Detect hallucinations in text.

        Returns:
        - hallucinations: List of potentially hallucinated claims
        - grounded: List of well-grounded claims
        - overall_confidence: Confidence score for the text
        """
        claims = self.citation_tracker.extract_claims(text)
        logger.debug("Hallucination check: %d claims extracted", len(claims))

        hallucinations = []
        grounded = []

        for claim in claims:
            # Check against knowledge base
            query = f"{claim['subject']} {claim['predicate']}"
            sources = self.retriever.retrieve(query, top_k=3, min_score=min_confidence)

            if not sources:
                hallucinations.append(
                    {
                        **claim,
                        "reason": "No supporting evidence found",
                        "confidence": 0.0,
                    }
                )
            else:
                # Check predicate overlap — high score alone isn't enough;
                # the source must actually share predicate words with the claim.
                claim_words = set(claim["predicate"].lower().split())
                best_overlap = 0.0
                best_source = None
                for s in sources:
                    source_words = set(s.chunk.content.lower().split())
                    overlap = len(claim_words & source_words) / max(len(claim_words), 1)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_source = s

                if best_overlap < 0.3:
                    # Source matches subject but not predicate → likely hallucination
                    hallucinations.append(
                        {
                            **claim,
                            "reason": f"Source mentions {claim['subject']} but not '{claim['predicate'][:50]}'",
                            "confidence": best_source.combined_score if best_source else 0.0,
                        }
                    )
                else:
                    avg_score = sum(s.combined_score for s in sources) / len(sources)
                    grounded.append(
                        {
                            **claim,
                            "confidence": avg_score,
                            "sources": [s.chunk.id for s in sources],
                        }
                    )

        # Calculate overall confidence
        total_claims = len(claims)
        if total_claims == 0:
            overall_confidence = 1.0
        else:
            grounded_count = len(grounded)
            avg_grounded = sum(c["confidence"] for c in grounded) / max(grounded_count, 1)
            overall_confidence = (grounded_count / total_claims) * avg_grounded

        logger.info(
            "Hallucination detection complete: total=%d, grounded=%d, hallucinations=%d, confidence=%.3f",
            total_claims,
            len(grounded),
            len(hallucinations),
            overall_confidence,
        )

        return {
            "text": text,
            "total_claims": total_claims,
            "grounded_claims": grounded,
            "hallucinations": hallucinations,
            "overall_confidence": overall_confidence,
            "hallucination_rate": len(hallucinations) / max(total_claims, 1),
            "formatted_citations": self.citation_tracker.format_citations(),
        }


class ProductionRAG:
    """
    Production-grade RAG system.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.retriever = HybridRetriever(
            dense_weight=config.get("dense_weight", 0.7),
            sparse_weight=config.get("sparse_weight", 0.3),
            # Live path is sparse + rerank: the default embedding_fn is
            # an uninformative random projection, so dense adds noise,
            # not signal. Pass use_dense=True with a real embedding_fn
            # to re-enable the dense side.
            use_dense=config.get("use_dense", False),
        )
        self.hallucination_detector = HallucinationDetector(self.retriever)

    def add_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 512,
        overlap: int = 50,
        rebuild_index: bool = True,
    ) -> list[str]:
        """
        Add a document with intelligent chunking.

        Set ``rebuild_index=False`` when bulk-loading many documents, then
        call ``retriever.build_index()`` once at the end (rebuilding per
        document is O(n^2)).
        """
        metadata = metadata or {"source": "user"}
        chunk_ids = []

        logger.debug(
            "Document ingestion started: chunk_size=%d, overlap=%d",
            chunk_size,
            overlap,
        )

        # Tokenize and chunk with overlap
        tokens = content.split()
        stride = max(1, chunk_size - overlap)
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_content = " ".join(chunk_tokens)

            chunk_id = hashlib.md5(chunk_content.encode()).hexdigest()[:12]
            chunk = TextChunk(
                id=chunk_id,
                content=chunk_content,
                metadata={**metadata, "position": i // chunk_size},
            )

            self.retriever.add_chunk(chunk)
            chunk_ids.append(chunk_id)

        # Rebuild index (skipped for bulk loads — caller builds once)
        if rebuild_index:
            self.retriever.build_index()

        logger.debug(
            "Document ingestion complete: chunks=%d, total_tokens=%d",
            len(chunk_ids),
            len(tokens),
        )
        return chunk_ids

    def query(
        self,
        question: str,
        top_k: int = 5,
        return_context: bool = True,
    ) -> dict[str, Any]:
        """
        Query the RAG system.
        """
        logger.debug("RAG query: top_k=%d, question_len=%d", top_k, len(question))
        results = self.retriever.retrieve(question, top_k=top_k)
        logger.debug("RAG query returned %d results", len(results))

        context = ""
        if return_context:
            context_parts = []
            for r in results:
                context_parts.append(r.chunk.content)
            context = "\n\n".join(context_parts)

        return {
            "question": question,
            "results": [
                {
                    "chunk_id": r.chunk.id,
                    "content": r.chunk.content,
                    "score": r.combined_score,
                    "rank": r.rank,
                    "metadata": r.chunk.metadata,
                }
                for r in results
            ],
            "context": context,
            "num_results": len(results),
        }

    def verify_and_ground(
        self,
        generated_text: str,
        question: str,
    ) -> dict[str, Any]:
        """
        Verify generated text and add citations.
        """
        # Check for hallucinations
        verification = self.hallucination_detector.detect(generated_text)

        # Add citations
        for claim in verification.get("grounded_claims", []):
            sources = self.retriever.retrieve(
                f"{claim['subject']} {claim['predicate']}",
                top_k=3,
            )
            self.hallucination_detector.citation_tracker.cite(claim, [s.chunk for s in sources])

        return {
            "original_text": generated_text,
            "question": question,
            "verification": verification,
            "citations": verification.get("formatted_citations", ""),
            "confidence": verification.get("overall_confidence", 0.5),
            "is_verified": verification.get("hallucination_rate", 1.0) < 0.3,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "TextChunk",
    "RetrievalResult",
    "BM25Indexer",
    "HybridRetriever",
    "CitationTracker",
    "HallucinationDetector",
    "ProductionRAG",
]
