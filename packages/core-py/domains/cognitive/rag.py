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

import hashlib
import re
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import Counter
import numpy as np


@dataclass
class TextChunk:
    """A chunked piece of text with metadata."""
    id: str
    content: str
    metadata: Dict[str, Any]
    token_count: int = 0
    embedding: Optional[np.ndarray] = None
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
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_freq: Dict[str, int] = Counter()
        self.num_docs: int = 0
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}

    def index(self, chunks: List[TextChunk]):
        """Build BM25 index."""
        self.num_docs = len(chunks)

        for doc_id, chunk in enumerate(chunks):
            tokens = self._tokenize(chunk.content)
            self.doc_lengths.append(len(tokens))

            # Count document frequencies
            for token in set(tokens):
                self.doc_freq[token] += 1

            # Build inverted index
            for pos, token in enumerate(tokens):
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append((doc_id, pos))

        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        return tokens

    def score(self, query: str) -> List[Tuple[int, float]]:
        """
        Score all documents against query.
        Returns list of (doc_id, score) tuples.
        """
        query_tokens = self._tokenize(query)
        scores = np.zeros(self.num_docs)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            # IDF for this term
            df = self.doc_freq.get(token, 0)
            idf = np.log((self.num_docs - df + 0.5) / (df + 0.5) + 1)

            for doc_id, _ in self.inverted_index[token]:
                doc_len = self.doc_lengths[doc_id]
                tf = sum(1 for d, _ in self.inverted_index.get(token, [])
                         if d == doc_id)

                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                score = idf * numerator / denominator

                scores[doc_id] += score

        # Return top docs with scores
        results = [(i, float(scores[i])) for i in range(self.num_docs) if scores[i] > 0]
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
    ):
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.use_rerank = use_rerank

        self.chunks: List[TextChunk] = []
        self.bm25 = BM25Indexer()
        self.embedding_cache: Dict[str, np.ndarray] = {}

    def add_chunk(self, chunk: TextChunk):
        """Add a chunk to the retriever."""
        self.chunks.append(chunk)

    def build_index(self):
        """Build retrieval indexes."""
        self.bm25.index(self.chunks)

    def _get_embedding(self, text: str) -> np.ndarray:
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
    ) -> List[Tuple[int, float]]:
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
    ) -> List[Tuple[int, float]]:
        """Sparse BM25 search."""
        return self.bm25.score(query)[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval with optional reranking.
        """
        # Get results from both methods
        dense_results = self._dense_search(query, top_k * 2)
        sparse_results = self._sparse_search(query, top_k * 2)

        # Normalize scores
        max_dense = max(s for _, s in dense_results) if dense_results else 1
        max_sparse = max(s for _, s in sparse_results) if sparse_results else 1

        # Combine scores
        combined_scores: Dict[int, Dict[str, float]] = {}

        for doc_id, score in dense_results:
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {"dense": 0, "sparse": 0}
            combined_scores[doc_id]["dense"] = score / max_dense

        for doc_id, score in sparse_results:
            if doc_id not in combined_scores:
                combined_scores[doc_id] = {"dense": 0, "sparse": 0}
            combined_scores[doc_id]["sparse"] = score / max_sparse

        # Calculate combined scores
        results = []
        for doc_id, scores in combined_scores.items():
            combined = (
                self.dense_weight * scores["dense"] +
                self.sparse_weight * scores["sparse"]
            )
            results.append(RetrievalResult(
                chunk=self.chunks[doc_id],
                dense_score=scores["dense"],
                sparse_score=scores["sparse"],
                combined_score=combined,
                rank=0,
            ))

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

        return final_results

    def _rerank(
        self,
        query: str,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        """
        Rerank results using cross-encoder style scoring.
        In production, use: sentence-transformers cross-encoder.
        """
        # Simple MMR (Maximal Marginal Relevance) for diversity
        reranked = []
        seen_contexts = set()

        for r in results:
            # Check novelty
            novelty = len(set(r.chunk.content.split()) & seen_contexts) / len(r.chunk.content.split())
            diversity_score = r.combined_score * (1 - novelty * 0.3)

            if diversity_score > 0.1:
                reranked.append(r)
                seen_contexts.update(r.chunk.content.split())

        return reranked[:len(results)]


class CitationTracker:
    """
    Track citations for generated text.
    Maps claims to supporting sources.
    """

    def __init__(self):
        self.claims: List[Dict[str, Any]] = []

    def extract_claims(self, text: str) -> List[Dict[str, Any]]:
        """Extract factual claims from text."""
        claims = []

        # Pattern-based claim extraction
        claim_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(.+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+was\s+(.+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+can\s+(.+)',
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+has\s+(.+)',
        ]

        for pattern in claim_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                claims.append({
                    "subject": match.group(1),
                    "predicate": match.group(2),
                    "text": match.group(0),
                    "start": match.start(),
                    "end": match.end(),
                })

        self.claims = claims
        return claims

    def cite(
        self,
        claim: Dict[str, Any],
        source_chunks: List[TextChunk],
    ) -> Dict[str, Any]:
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
    ) -> Dict[str, Any]:
        """
        Detect hallucinations in text.

        Returns:
        - hallucinations: List of potentially hallucinated claims
        - grounded: List of well-grounded claims
        - overall_confidence: Confidence score for the text
        """
        claims = self.citation_tracker.extract_claims(text)

        hallucinations = []
        grounded = []

        for claim in claims:
            # Check against knowledge base
            query = f"{claim['subject']} {claim['predicate']}"
            sources = self.retriever.retrieve(query, top_k=3, min_score=min_confidence)

            if not sources:
                hallucinations.append({
                    **claim,
                    "reason": "No supporting evidence found",
                    "confidence": 0.0,
                })
            else:
                avg_score = sum(s.combined_score for s in sources) / len(sources)
                grounded.append({
                    **claim,
                    "confidence": avg_score,
                    "sources": [s.chunk.id for s in sources],
                })

        # Calculate overall confidence
        total_claims = len(claims)
        if total_claims == 0:
            overall_confidence = 1.0
        else:
            grounded_count = len(grounded)
            avg_grounded = sum(c["confidence"] for c in grounded) / max(grounded_count, 1)
            overall_confidence = (grounded_count / total_claims) * avg_grounded

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

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.retriever = HybridRetriever(
            dense_weight=config.get("dense_weight", 0.7),
            sparse_weight=config.get("sparse_weight", 0.3),
        )
        self.hallucination_detector = HallucinationDetector(self.retriever)

    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> List[str]:
        """
        Add a document with intelligent chunking.
        """
        metadata = metadata or {"source": "user"}
        chunk_ids = []

        # Tokenize and chunk with overlap
        tokens = content.split()
        stride = max(1, chunk_size - overlap)
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_content = ' '.join(chunk_tokens)

            chunk_id = hashlib.md5(chunk_content.encode()).hexdigest()[:12]
            chunk = TextChunk(
                id=chunk_id,
                content=chunk_content,
                metadata={**metadata, "position": i // chunk_size},
            )

            self.retriever.add_chunk(chunk)
            chunk_ids.append(chunk_id)

        # Rebuild index
        self.retriever.build_index()

        return chunk_ids

    def query(
        self,
        question: str,
        top_k: int = 5,
        return_context: bool = True,
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        """
        results = self.retriever.retrieve(question, top_k=top_k)

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
    ) -> Dict[str, Any]:
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
