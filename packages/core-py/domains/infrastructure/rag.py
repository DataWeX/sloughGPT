"""
RAG System - DEPRECATED

This module is deprecated. Use domains.cognitive.rag instead.

Canonical RAG classes:
- ProductionRAG
- BM25Indexer
- HybridRetriever
- TextChunk
- RetrievalResult

See: domains.cognitive.rag
"""

import warnings
warnings.warn(
    "domains.infrastructure.rag is deprecated. Use domains.cognitive.rag instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export canonical classes for backwards compatibility
from domains.cognitive.rag import (
    ProductionRAG as RAGSystem,
    BM25Indexer as BM25,
    TextChunk,
    RetrievalResult,
    HybridRetriever,
    HallucinationDetector,
)

# Import the rest for backwards compatibility
import json
import sqlite3
import hashlib
import math
import re
from typing import List, Dict, Optional, Any, Callable, Tuple, Union
from pathlib import Path
from collections import defaultdict, Counter


class BM25:
    """BM25 ranking algorithm for better search quality."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths = []
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.corpus_size = 0

    def fit(self, corpus: List[str]):
        """Fit BM25 on corpus."""
        self.corpus_size = len(corpus)
        self.doc_lengths = []
        self.doc_freqs = []

        nd = defaultdict(int)

        for document in corpus:
            self.doc_lengths.append(len(document))
            frequencies = Counter(document.split())
            self.doc_freqs.append(frequencies)

            for term in frequencies:
                nd[term] += 1

        self.avgdl = sum(self.doc_lengths) / max(1, self.corpus_size)

        for term, df in nd.items():
            self.idf[term] = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1)

    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for all documents."""
        scores = [0.0] * self.corpus_size
        query_terms = query.lower().split()

        for i, doc in enumerate(self.doc_freqs):
            doc_len = self.doc_lengths[i]

            for term in query_terms:
                if term not in self.idf:
                    continue

                tf = doc.get(term, 0)
                idf = self.idf[term]

                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(1, self.avgdl))

                scores[i] += idf * (numerator / max(0.001, denominator))

        return scores


class FuzzyMatcher:
    """Fuzzy string matching for better search."""

    @staticmethod
    def levenshtein(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance."""
        if len(s1) < len(s2):
            return FuzzyMatcher.levenshtein(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def similarity(s1: str, s2: str) -> float:
        """Calculate similarity ratio 0-1."""
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        return 1 - FuzzyMatcher.levenshtein(s1.lower(), s2.lower()) / max_len


class QueryExpander:
    """Expand queries with synonyms and related terms."""

    def __init__(self):
        self.synonyms = {
            "ai": ["artificial intelligence", "machine intelligence", "AI"],
            "ml": ["machine learning", "ML", "statistical learning"],
            "dl": ["deep learning", "neural networks", "DL"],
            "nlp": ["natural language processing", "text processing", "NLP"],
            "llm": ["large language model", "language model", "GPT"],
            "rag": ["retrieval augmented", "retrieval-augmented", "RAG"],
            "learn": ["study", "train", "understand", "acquire"],
            "model": ["network", "system", "architecture"],
            "data": ["information", "dataset", "corpus", "examples"],
            "memory": ["storage", "recall", "persistence", "context"],
            "think": ["reason", "process", "analyze", "reasoning"],
        }

    def expand(self, query: str) -> List[str]:
        """Expand query with synonyms."""
        expanded = [query]
        query_lower = query.lower()

        for term, syns in self.synonyms.items():
            if term in query_lower:
                for syn in syns:
                    expanded.append(query_lower.replace(term, syn))

        return list(set(expanded))


class HybridSearchEngine:
    """
    Hybrid Search Engine: Combines semantic and keyword search.

    Combines vector/embedding-based search with keyword-based search
    using a weighted scoring approach.
    """

    def __init__(self, alpha: float = 0.7):
        """
        Args:
            alpha: Weight for semantic search (1-alpha for keyword).
                   Higher = more semantic, lower = more keyword matching.
        """
        self.alpha = alpha

    def search(
        self,
        semantic_search_fn: Callable[[str, int], List[Dict]],
        keyword_search_fn: Callable[[str, int], List[Dict]],
        query: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Execute hybrid search combining semantic and keyword results.

        Args:
            semantic_search_fn: Function that takes (query, top_k) returns semantic results
            keyword_search_fn: Function that takes (query, top_k) returns keyword results
            query: Search query
            top_k: Number of results to return

        Returns:
            Combined and ranked results with scores from both methods
        """
        semantic_results = semantic_search_fn(query, top_k * 2)
        keyword_results = keyword_search_fn(query, top_k * 2)

        doc_scores: Dict[int, Tuple[float, float, float]] = {}

        for result in semantic_results:
            doc_id = result["id"]
            semantic_score = result.get("score", 0.0)
            doc_scores[doc_id] = (semantic_score, 0.0, 0.0)

        for result in keyword_results:
            doc_id = result.get("id", result.get("key"))
            if doc_id is None:
                continue
            keyword_score = result.get("score", 0.0)

            if doc_id in doc_scores:
                sem, kw, _ = doc_scores[doc_id]
                doc_scores[doc_id] = (sem, keyword_score, 0.0)
            else:
                doc_scores[doc_id] = (0.0, keyword_score, 0.0)

        ranked = []
        for doc_id, (semantic_score, keyword_score, _) in doc_scores.items():
            combined = self.alpha * semantic_score + (1 - self.alpha) * keyword_score
            ranked.append((doc_id, combined, semantic_score, keyword_score))

        ranked.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score, sem_score, kw_score in ranked[:top_k]:
            results.append(
                {
                    "id": doc_id,
                    "score": score,
                    "semantic_score": sem_score,
                    "keyword_score": kw_score,
                }
            )

        return results

    def set_alpha(self, alpha: float) -> None:
        """Update the semantic/keyword weight balance."""
        self.alpha = max(0.0, min(1.0, alpha))


class TemporalRankingEngine:
    """
    Temporal Ranking Engine: Weights results by recency.

    Applies exponential decay to older documents, prioritizing
    fresh content while still retaining valuable older knowledge.
    """

    def __init__(self, decay_factor: float = 0.95, time_field: str = "timestamp"):
        """
        Args:
            decay_factor: Daily decay factor (0.95 = 5% decay per day)
            time_field: Metadata field containing timestamp
        """
        self.decay_factor = decay_factor
        self.time_field = time_field

    def rank(self, results: List[Dict], current_time: Optional[float] = None) -> List[Dict]:
        """
        Apply temporal weighting to search results.

        Args:
            results: Base search results
            current_time: Current timestamp (defaults to now)

        Returns:
            Results with temporal weights and adjusted scores
        """
        import time as time_module

        if current_time is None:
            current_time = time_module.time()

        for result in results:
            metadata = result.get("metadata", {})
            doc_time = metadata.get(self.time_field)

            if doc_time:
                temporal_weight, doc_age = self._calculate_weight(doc_time, current_time)
                result["temporal_weight"] = temporal_weight
                result["doc_age_days"] = round(doc_age, 1)
                result["score"] = result.get("score", 1.0) * temporal_weight
            else:
                result["temporal_weight"] = 1.0
                result["doc_age_days"] = 0.0

        ranked = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        return ranked

    def _calculate_weight(self, doc_time: Any, current_time: float) -> Tuple[float, float]:
        """Calculate temporal weight for a document."""
        try:
            if isinstance(doc_time, str):
                from datetime import datetime

                doc_timestamp = datetime.fromisoformat(doc_time).timestamp()
            else:
                doc_timestamp = doc_time

            doc_age_days = (current_time - doc_timestamp) / (24 * 3600)
            temporal_weight = self.decay_factor**doc_age_days
            return temporal_weight, doc_age_days
        except Exception:
            return 1.0, 0.0

    def set_decay_factor(self, decay_factor: float) -> None:
        """Update the decay factor."""
        self.decay_factor = max(0.0, min(1.0, decay_factor))


class PersonalizedRankingEngine:
    """
    Personalized Ranking Engine: Boosts results based on user profile.

    Applies relevance boosting based on user interests, expertise level,
    and content preferences.
    """

    def __init__(
        self, interest_boost: float = 1.5, level_boost: float = 1.3, tag_boost: float = 1.2
    ):
        """
        Args:
            interest_boost: Multiplier for matching interests
            level_boost: Multiplier for matching difficulty level
            tag_boost: Multiplier for matching preferred tags
        """
        self.interest_boost = interest_boost
        self.level_boost = level_boost
        self.tag_boost = tag_boost

    def rank(self, results: List[Dict], user_profile: Dict[str, Any]) -> List[Dict]:
        """
        Apply personalization to search results.

        Args:
            results: Base search results
            user_profile: Dict with 'interests', 'level', 'preferred_tags'

        Returns:
            Results with personalization boosts applied
        """
        interests = user_profile.get("interests", [])
        user_level = user_profile.get("level", "intermediate")
        preferred_tags = user_profile.get("preferred_tags", [])

        for result in results:
            boost_multiplier = 1.0
            boost_reasons = []

            metadata = result.get("metadata", {})
            doc_tags = metadata.get("tags", [])
            doc_topics = metadata.get("topics", [])
            doc_level = metadata.get("level", "intermediate")

            if interests:
                matched_interests = [i for i in interests if i.lower() in (doc_tags + doc_topics)]
                if matched_interests:
                    boost_multiplier *= self.interest_boost
                    boost_reasons.append(f"interest_match:{matched_interests}")

            if preferred_tags and doc_tags:
                matched_tags = [t for t in doc_tags if t in preferred_tags]
                if matched_tags:
                    boost_multiplier *= self.tag_boost
                    boost_reasons.append(f"tag_match:{matched_tags}")

            if doc_level == user_level:
                boost_multiplier *= self.level_boost
                boost_reasons.append(f"level_match:{doc_level}")

            result["personalization"] = {
                "boost_multiplier": boost_multiplier,
                "boost_reasons": boost_reasons,
            }
            result["score"] = result.get("score", 1.0) * boost_multiplier

        ranked = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        return ranked

    def set_boost_factors(
        self,
        interest_boost: Optional[float] = None,
        level_boost: Optional[float] = None,
        tag_boost: Optional[float] = None,
    ) -> None:
        """Update boost factors."""
        if interest_boost is not None:
            self.interest_boost = interest_boost
        if level_boost is not None:
            self.level_boost = level_boost
        if tag_boost is not None:
            self.tag_boost = tag_boost


class RAGSystem:
    """RAG system for knowledge retrieval"""

    def __init__(self, store_path: str = "runs/store/rag_store.db"):
        self.store_path = store_path
        self.documents: List[Dict] = []
        self.max_context_length = 2000

    def add_document(self, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a document to the knowledge base."""
        doc = {"content": content, "metadata": metadata or {}, "id": len(self.documents)}
        self.documents.append(doc)

    def add_training_knowledge(
        self, dataset_path: Union[str, Path], metadata: Optional[Dict] = None
    ) -> int:
        """Add dataset information to knowledge base."""
        path_obj = Path(dataset_path) if not isinstance(dataset_path, Path) else dataset_path
        if not path_obj.exists():
            return 0

        added = 0
        if path_obj.suffix == ".jsonl":
            with open(path_obj, "r") as f:
                for i, line in enumerate(f):
                    if line.strip():
                        try:
                            data = json.loads(line)
                            content = str(data.get("text", data))
                            if content:
                                doc_metadata = {
                                    **(metadata or {}),
                                    "source": "training_dataset",
                                    "dataset_name": path_obj.name,
                                    "line_number": i,
                                }
                                self.add_document(content, doc_metadata)
                                added += 1
                        except json.JSONDecodeError:
                            continue

        return added

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents."""
        query_lower = query.lower()
        results = []

        for doc in self.documents:
            content_lower = doc["content"].lower()
            score = sum(1 for word in query_lower.split() if word in content_lower)
            if score > 0:
                results.append(
                    {"content": doc["content"], "metadata": doc["metadata"], "score": score}
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def search_bm25(self, query: str, top_k: int = 5) -> List[Dict]:
        """BM25-based search for better ranking."""
        if not self.documents:
            return []

        corpus = [doc["content"].lower() for doc in self.documents]

        try:
            bm25 = BM25()
            bm25.fit(corpus)
            scores = bm25.get_scores(query.lower())

            results = []
            for i, score in enumerate(scores):
                if score > 0:
                    results.append(
                        {
                            "content": self.documents[i]["content"],
                            "metadata": self.documents[i]["metadata"],
                            "score": score,
                        }
                    )

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]
        except:
            return self.search(query, top_k)

    def search_fuzzy(self, query: str, top_k: int = 5, threshold: float = 0.6) -> List[Dict]:
        """Fuzzy search for partial matches."""
        if not self.documents:
            return []

        query_lower = query.lower()
        query_words = query_lower.split()

        results = []
        for doc in self.documents:
            content_lower = doc["content"].lower()
            content_words = content_lower.split()

            max_score = 0
            for qw in query_words:
                for cw in content_words:
                    sim = FuzzyMatcher.similarity(qw, cw)
                    if sim >= threshold:
                        max_score = max(max_score, sim)

            if max_score > 0:
                results.append(
                    {"content": doc["content"], "metadata": doc["metadata"], "score": max_score}
                )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def search_expanded(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search with query expansion."""
        expander = QueryExpander()
        expanded_queries = expander.expand(query)

        all_results = []
        seen_content = set()

        for q in expanded_queries:
            results = self.search(q, top_k * 2)
            for r in results:
                content_hash = hash(r["content"][:50])
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_results.append(r)

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def search_hybrid(self, query: str, top_k: int = 5) -> List[Dict]:
        """Combine BM25, fuzzy, and keyword search."""
        bm25_results = self.search_bm25(query, top_k * 2)
        fuzzy_results = self.search_fuzzy(query, top_k * 2)
        keyword_results = self.search(query, top_k * 2)

        combined = {}

        for r in bm25_results:
            key = r["content"][:50]
            combined[key] = {
                "content": r["content"],
                "metadata": r["metadata"],
                "score": r.get("score", 0) * 0.4,
            }

        for r in fuzzy_results:
            key = r["content"][:50]
            if key in combined:
                combined[key]["score"] += r.get("score", 0) * 0.3
            else:
                combined[key] = {
                    "content": r["content"],
                    "metadata": r["metadata"],
                    "score": r.get("score", 0) * 0.3,
                }

        for r in keyword_results:
            key = r["content"][:50]
            if key in combined:
                combined[key]["score"] += r.get("score", 0) * 0.3
            else:
                combined[key] = {
                    "content": r["content"],
                    "metadata": r["metadata"],
                    "score": r.get("score", 0) * 0.3,
                }

        results = list(combined.values())
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_context(self, query: str, max_length: Optional[int] = None) -> str:
        """Get context from retrieved documents."""
        max_length = max_length or self.max_context_length
        results = self.search(query, top_k=10)

        context_parts = []
        current_length = 0

        for result in results:
            content = result["content"]
            if current_length + len(content) <= max_length:
                context_parts.append(content)
                current_length += len(content)

        return "\n\n".join(context_parts)

    def clear(self) -> None:
        """Clear all documents."""
        self.documents = []

    def count(self) -> int:
        """Count total documents."""
        return len(self.documents)


class VectorStore:
    """Simple vector store for embeddings (placeholder)."""

    def __init__(self):
        self.vectors: Dict[str, List[float]] = {}
        self.metadata: Dict[str, Dict] = {}

    def add(self, key: str, vector: List[float], metadata: Optional[Dict] = None) -> None:
        """Add a vector."""
        self.vectors[key] = vector
        self.metadata[key] = metadata or {}

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict]:
        """Search for similar vectors (simple cosine similarity)."""
        results = []

        for key, vector in self.vectors.items():
            similarity = self._cosine_similarity(query_vector, vector)
            results.append(
                {"key": key, "score": similarity, "metadata": self.metadata.get(key, {})}
            )

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)


__all__ = [
    "RAGSystem",
    "VectorStore",
    "ChainOfThoughtRAG",
    "SelfReflectiveRAG",
    "MultiHopRAG",
    "EndicIndex",
    "RAGEngine",
]


class EndicIndex:
    """
    In-memory inverted index for fast keyword search.
    Maps terms to document IDs for O(1) lookup.
    """

    def __init__(self):
        self.index: Dict[str, set] = defaultdict(set)
        self.doc_mapping: Dict[int, Dict] = {}
        self.term_freq: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def add_document(self, doc_id: int, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a document to the index."""
        self.doc_mapping[doc_id] = {"content": content, "metadata": metadata or {}}

        terms = self._tokenize(content)
        for term in terms:
            self.index[term].add(doc_id)
            self.term_freq[doc_id][term] += 1

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into terms."""
        text = text.lower()
        terms = re.findall(r"\b\w+\b", text)
        return terms

    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search for documents matching query."""
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        doc_scores: Dict[int, float] = defaultdict(float)

        num_docs = max(len(self.doc_mapping), 1)

        for term in query_terms:
            matching_docs = self.index.get(term, set())
            if not matching_docs:
                continue

            doc_count = len(matching_docs)
            idf = math.log(num_docs / doc_count) + 1  # Add 1 to avoid negative

            for doc_id in matching_docs:
                tf = self.term_freq[doc_id].get(term, 0)
                if tf > 0:
                    tf_score = 1 + math.log(tf)  # log(tf) + 1
                    tf_idf = tf_score * idf
                    doc_scores[doc_id] += tf_idf

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in ranked[:top_k]:
            doc = self.doc_mapping[doc_id]
            results.append(
                {
                    "id": doc_id,
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "score": score,
                }
            )

        return results

    def clear(self) -> None:
        """Clear the index."""
        self.index.clear()
        self.doc_mapping.clear()
        self.term_freq.clear()


class ChainOfThoughtRAG:
    """
    Chain-of-Thought RAG: Decompose complex queries, retrieve per sub-query, synthesize.

    Flow:
    1. Decompose query into sub-queries
    2. Retrieve context for each sub-query
    3. Synthesize final response from all contexts
    """

    def __init__(self, base_rag: "RAGSystem", decompose_fn: Optional[Callable] = None):
        self.rag = base_rag
        self.decompose_fn = decompose_fn or self._default_decompose

    def _default_decompose(self, query: str) -> List[str]:
        """Default query decomposition using sentence splitting."""
        sentences = re.split(r"[.!?]+", query)
        sub_queries = [s.strip() for s in sentences if s.strip()]

        if len(sub_queries) <= 1:
            keywords = query.split()
            if len(keywords) > 3:
                sub_queries = [
                    " ".join(keywords[: len(keywords) // 2]),
                    " ".join(keywords[len(keywords) // 2 :]),
                ]

        return sub_queries if sub_queries else [query]

    def _synthesize(self, contexts: List[str], original_query: str) -> str:
        """Synthesize response from multiple contexts."""
        if not contexts:
            return "No relevant information found."

        combined = "\n\n".join(contexts)
        return combined

    def retrieve(self, query: str) -> Dict[str, Any]:
        """
        Execute chain-of-thought retrieval.

        Returns:
            Dict with sub_queries, contexts, synthesized_response
        """
        sub_queries = self.decompose_fn(query)

        contexts = []
        for sq in sub_queries:
            context = self.rag.get_context(sq, max_length=1000)
            if context:
                contexts.append(context)

        return {
            "original_query": query,
            "sub_queries": sub_queries,
            "contexts": contexts,
            "synthesized": self._synthesize(contexts, query),
        }


class SelfReflectiveRAG:
    """
    Self-Reflective RAG: Generate, self-critique, refine.

    Flow:
    1. Generate initial response with context
    2. Self-critique the response
    3. If critique found, refine the response
    """

    def __init__(self, base_rag: "RAGSystem", generate_fn: Optional[Callable] = None):
        self.rag = base_rag
        self.generate_fn = generate_fn

    def retrieve(self, query: str, initial_response: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute self-reflective retrieval.

        Returns:
            Dict with initial_response, critique, refined_response
        """
        context = self.rag.get_context(query)

        initial = initial_response
        if not initial and self.generate_fn:
            initial = self.generate_fn(query, context)
        elif not initial:
            initial = f"Based on context: {context[:200]}..."

        critique_query = f"Is this correct and complete? {initial}"
        critique_context = self.rag.get_context(critique_query, max_length=500)

        refined = None
        if critique_context and len(critique_context) > 20:
            refined = f"{initial}\n\nRefinement: {critique_context}"

        return {
            "query": query,
            "context": context,
            "initial_response": initial,
            "critique_context": critique_context,
            "refined_response": refined or initial,
        }


class MultiHopRAG:
    """
    Multi-Hop RAG: Sequential retrieval with context expansion.

    Flow:
    1. First hop: Get initial context
    2. Second hop: Use context to find related info
    3. Combine all contexts
    """

    def __init__(self, base_rag: "RAGSystem", max_hops: int = 2):
        self.rag = base_rag
        self.max_hops = max_hops

    def retrieve(self, query: str) -> Dict[str, Any]:
        """
        Execute multi-hop retrieval.

        Returns:
            Dict with hops, contexts, combined_response
        """
        all_contexts = []
        current_query = query
        hop_results = []

        for hop in range(self.max_hops):
            context = self.rag.get_context(current_query, max_length=800)

            if not context:
                break

            all_contexts.append(context)
            hop_results.append({"hop": hop + 1, "query": current_query, "context": context})

            if hop < self.max_hops - 1:
                key_terms = self._extract_key_terms(context)
                current_query = f"{query} {' '.join(key_terms[:5])}"

        combined = "\n\n---\n\n".join(all_contexts)

        return {
            "original_query": query,
            "hops": hop_results,
            "contexts": all_contexts,
            "combined": combined,
        }

    def _extract_key_terms(self, text: str, n: int = 5) -> List[str]:
        """Extract key terms from text."""
        words = re.findall(r"\b[a-z]{4,}\b", text.lower())
        freq = defaultdict(int)
        for w in words:
            freq[w] += 1
        sorted_terms = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_terms[:n]]


class RAGEngine:
    """
    Complete RAG Engine with all advanced patterns and optimizations.

    Features:
    - Chain-of-Thought, Self-Reflective, Multi-Hop RAG
    - Batch processing
    - SQLite persistence
    - Conversation learning
    """

    def __init__(
        self,
        store_path: Union[str, Path] = "runs/store/rag_store.db",
        enable_persistence: bool = True,
    ):
        self.rag = RAGSystem(store_path)
        self.endic_index = EndicIndex()
        self.enable_persistence = enable_persistence
        self.store_path = store_path

        self.cot_rag = ChainOfThoughtRAG(self.rag)
        self.reflective_rag = SelfReflectiveRAG(self.rag)
        self.multi_hop_rag = MultiHopRAG(self.rag)

        self.hybrid_search_engine = HybridSearchEngine(alpha=0.7)
        self.temporal_ranking_engine = TemporalRankingEngine(decay_factor=0.95)
        self.personalized_ranking_engine = PersonalizedRankingEngine(
            interest_boost=1.5, level_boost=1.3, tag_boost=1.2
        )

        if enable_persistence:
            self._init_db()
            self._load_from_db()

    def _init_db(self) -> None:
        """Initialize SQLite database."""
        Path(self.store_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.store_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT,
                indexed BOOLEAN DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

    def _load_from_db(self) -> None:
        """Load documents from SQLite."""
        try:
            conn = sqlite3.connect(self.store_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, content, metadata FROM documents WHERE indexed = 0")
            for row in cursor.fetchall():
                doc_id, content, metadata_json = row
                metadata = json.loads(metadata_json) if metadata_json else {}
                self.rag.documents.append({"id": doc_id, "content": content, "metadata": metadata})
                self.endic_index.add_document(doc_id, content, metadata)
                cursor.execute("UPDATE documents SET indexed = 1 WHERE id = ?", (doc_id,))
            conn.commit()
            conn.close()
        except Exception:
            pass

    def add_document(self, content: str, metadata: Optional[Dict] = None) -> int:
        """Add a single document."""
        doc_id = len(self.rag.documents)
        self.rag.add_document(content, metadata)
        self.endic_index.add_document(doc_id, content, metadata)

        if self.enable_persistence:
            self._save_to_db(doc_id, content, metadata)

        return doc_id

    def add_batch_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Add multiple documents efficiently using batch processing.

        Args:
            documents: List of {"content": str, "metadata": dict}

        Returns:
            Number of documents added
        """
        added = 0
        start_id = len(self.rag.documents)

        for i, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            if content:
                doc_id = start_id + i
                self.rag.documents.append({"id": doc_id, "content": content, "metadata": metadata})
                self.endic_index.add_document(doc_id, content, metadata)
                added += 1

        if self.enable_persistence and added > 0:
            self._batch_save_to_db(start_id, documents[:added])

        return added

    def _save_to_db(self, doc_id: int, content: str, metadata: Optional[Dict]) -> None:
        """Save single document to SQLite."""
        try:
            conn = sqlite3.connect(self.store_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO documents (id, content, metadata, indexed) VALUES (?, ?, ?, 1)",
                (doc_id, content, json.dumps(metadata)),
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def _batch_save_to_db(self, start_id: int, documents: List[Dict]) -> None:
        """Batch save documents to SQLite."""
        try:
            conn = sqlite3.connect(self.store_path)
            cursor = conn.cursor()
            data = [
                (start_id + i, doc.get("content", ""), json.dumps(doc.get("metadata", {})), 1)
                for i, doc in enumerate(documents)
                if doc.get("content")
            ]
            cursor.executemany(
                "INSERT OR REPLACE INTO documents (id, content, metadata, indexed) VALUES (?, ?, ?, ?)",
                data,
            )
            conn.commit()
            conn.close()
        except Exception:
            pass

    def search(self, query: str, top_k: int = 5, use_index: bool = True) -> List[Dict]:
        """Search using in-memory index for speed."""
        if use_index:
            return self.endic_index.search(query, top_k)
        return self.rag.search(query, top_k)

    def cot_retrieve(self, query: str) -> Dict[str, Any]:
        """Chain-of-Thought retrieval."""
        return self.cot_rag.retrieve(query)

    def reflective_retrieve(
        self, query: str, initial_response: Optional[str] = None
    ) -> Dict[str, Any]:
        """Self-Reflective retrieval."""
        return self.reflective_rag.retrieve(query, initial_response)

    def multi_hop_retrieve(self, query: str, max_hops: int = 2) -> Dict[str, Any]:
        """Multi-Hop retrieval."""
        self.multi_hop_rag.max_hops = max_hops
        return self.multi_hop_rag.retrieve(query)

    def learn_from_interaction(
        self, user_input: str, slo_response: str, feedback: str = "neutral"
    ) -> None:
        """
        Learn from user interaction for continuous improvement.

        Args:
            user_input: User's message
            slo_response: SLO's response
            feedback: "good", "bad", or "neutral"
        """
        if feedback == "good":
            conversation_memory = [
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": slo_response},
                {"role": "feedback", "content": "positive"},
            ]
            self.add_document(
                f"User asked: {user_input}. SLO responded: {slo_response}",
                {"source": "conversation_feedback", "type": "positive_example"},
            )

        elif feedback == "bad":
            self.add_document(
                f"Poor response to: {user_input}. Response was: {slo_response}",
                {"source": "conversation_feedback", "type": "negative_example"},
            )

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        source_counts = defaultdict(int)
        type_counts = defaultdict(int)

        for doc in self.rag.documents:
            meta = doc.get("metadata", {})
            source_counts[meta.get("source", "unknown")] += 1
            type_counts[meta.get("type", "unknown")] += 1

        total = len(self.rag.documents)
        source_pct = (
            {k: round(v / total * 100, 1) for k, v in source_counts.items()} if total > 0 else {}
        )

        return {
            "total_documents": total,
            "by_source": dict(source_counts),
            "by_source_percent": source_pct,
            "by_type": dict(type_counts),
        }

    def clear(self) -> None:
        """Clear all documents."""
        self.rag.clear()
        self.endic_index.clear()

        if self.enable_persistence:
            try:
                conn = sqlite3.connect(self.store_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM documents")
                conn.commit()
                conn.close()
            except Exception:
                pass

    def hybrid_search(
        self, query: str, top_k: int = 10, alpha: float = 0.7, use_index: bool = True
    ) -> List[Dict]:
        """
        Hybrid Search: Combine semantic and keyword search.

        Args:
            query: Search query
            top_k: Number of results
            alpha: Weight for semantic (1-alpha for keyword)
            use_index: Use in-memory index for keyword search

        Returns:
            Combined and ranked results
        """
        if alpha != self.hybrid_search_engine.alpha:
            self.hybrid_search_engine.set_alpha(alpha)

        merged_results = self.hybrid_search_engine.search(
            semantic_search_fn=lambda q, k: self.endic_index.search(q, k),
            keyword_search_fn=lambda q, k: self.rag.search(q, k)
            if use_index
            else self.endic_index.search(q, k),
            query=query,
            top_k=top_k,
        )

        results = []
        for result in merged_results:
            doc_id = result["id"]
            doc = self.rag.documents[doc_id] if doc_id < len(self.rag.documents) else None
            if doc:
                results.append(
                    {
                        "id": doc_id,
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "score": result["score"],
                        "semantic_score": result["semantic_score"],
                        "keyword_score": result["keyword_score"],
                    }
                )

        return results

    def temporal_search(
        self, query: str, top_k: int = 10, decay_factor: float = 0.95, time_field: str = "timestamp"
    ) -> List[Dict]:
        """
        Temporal Weighting: Weight results by recency.

        Args:
            query: Search query
            top_k: Number of results
            decay_factor: Daily decay factor (0.95 = 5% decay per day)
            time_field: Metadata field containing timestamp

        Returns:
            Results with temporal weighting applied
        """
        if decay_factor != self.temporal_ranking_engine.decay_factor:
            self.temporal_ranking_engine.set_decay_factor(decay_factor)
        self.temporal_ranking_engine.time_field = time_field

        base_results = self.endic_index.search(query, top_k * 3)

        ranked = self.temporal_ranking_engine.rank(base_results)

        return ranked[:top_k]

    def personalized_search(
        self,
        query: str,
        user_profile: Dict[str, Any],
        top_k: int = 10,
        interest_boost: float = 1.5,
        level_boost: float = 1.3,
        tag_boost: float = 1.2,
    ) -> List[Dict]:
        """
        Personalized Ranking: Boost results based on user profile.

        Args:
            query: Search query
            user_profile: Dict with 'interests', 'level', 'preferences'
            top_k: Number of results
            interest_boost: Multiplier for matching interests
            level_boost: Multiplier for matching difficulty level
            tag_boost: Multiplier for matching preferred tags

        Returns:
            Results personalized for the user
        """
        if (
            interest_boost != self.personalized_ranking_engine.interest_boost
            or level_boost != self.personalized_ranking_engine.level_boost
            or tag_boost != self.personalized_ranking_engine.tag_boost
        ):
            self.personalized_ranking_engine.set_boost_factors(
                interest_boost=interest_boost, level_boost=level_boost, tag_boost=tag_boost
            )

        base_results = self.endic_index.search(query, top_k * 2)

        ranked = self.personalized_ranking_engine.rank(base_results, user_profile)

        return ranked[:top_k]

    def advanced_search(
        self,
        query: str,
        strategy: str = "hybrid",
        user_profile: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        **kwargs,
    ) -> List[Dict]:
        """
        Unified advanced search interface.

        Args:
            query: Search query
            strategy: "hybrid", "temporal", "personalized", or "adaptive"
            user_profile: User profile for personalized search
            top_k: Number of results
            **kwargs: Additional parameters

        Returns:
            Search results based on strategy
        """
        if strategy == "hybrid":
            return self.hybrid_search(query, top_k, alpha=kwargs.get("alpha", 0.7))

        elif strategy == "temporal":
            return self.temporal_search(query, top_k, decay_factor=kwargs.get("decay_factor", 0.95))

        elif strategy == "personalized":
            if not user_profile:
                user_profile = {"interests": [], "level": "intermediate"}
            return self.personalized_search(query, user_profile, top_k)

        elif strategy == "adaptive":
            results = self.hybrid_search(query, top_k, alpha=0.5)

            if user_profile:
                for result in results:
                    metadata = result.get("metadata", {})
                    interests = user_profile.get("interests", [])
                    if metadata.get("topic") in interests:
                        result["score"] *= 1.2

            return sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

        else:
            return self.search(query, top_k)


class SLOKnowledgeGraph:
    """
    Knowledge Graph for semantic relationship tracking.

    Tracks concepts and their relationships to:
    - Expand queries with related concepts
    - Build deeper understanding of domain
    - Enable graph-based reasoning
    """

    def __init__(self):
        self.concepts: Dict[str, List[float]] = {}
        self.concept_metadata: Dict[str, Dict[str, Any]] = {}
        self.relations: Dict[Tuple[str, str, str], float] = {}
        self.concept_definitions: Dict[str, str] = {}

    def _simple_embedding(self, text: str) -> List[float]:
        """Generate simple embedding from text hash."""
        h = hashlib.sha256(text.encode()).hexdigest()
        vec = [int(h[i : i + 2], 16) / 255.0 for i in range(0, 64, 2)]
        while len(vec) < 32:
            vec.append(0.0)
        return vec[:32]

    def add_concept(self, concept: str, definition: str, metadata: Optional[Dict] = None) -> None:
        """Add a concept to the knowledge graph."""
        embedding = self._simple_embedding(definition)

        self.concepts[concept] = embedding
        self.concept_definitions[concept] = definition
        self.concept_metadata[concept] = metadata or {}
        self.concept_metadata[concept]["definition"] = definition

    def add_relation(self, source: str, relation: str, target: str, weight: float = 1.0) -> None:
        """
        Add a relation between concepts.

        Args:
            source: Source concept
            relation: Type of relation (e.g., "is_a", "related_to", "part_of")
            target: Target concept
            weight: Relation strength (0-1)
        """
        self.relations[(source, relation, target)] = weight

    def expand_query(self, query: str, max_concepts: int = 5) -> str:
        """
        Expand query with related concepts from the graph.

        Args:
            query: Original query
            max_concepts: Maximum related concepts to add

        Returns:
            Expanded query
        """
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        related_concepts = []

        for source, relation, target in self.relations:
            if source in query_terms:
                related_concepts.append(target)
            if target in query_terms:
                related_concepts.append(source)

        related_concepts = related_concepts[:max_concepts]

        expanded = f"{query} {' '.join(related_concepts)}"
        return expanded

    def find_related(self, concept: str, max_results: int = 5) -> List[Dict]:
        """Find related concepts with partial/fuzzy matching."""
        concept_lower = concept.lower()

        matching_concepts = [
            c for c in self.concepts if concept_lower in c.lower() or c.lower() in concept_lower
        ]

        if not matching_concepts and self.concepts:
            matching_concepts = list(self.concepts.keys())[:3]

        related = []

        for match in matching_concepts:
            for (source, relation, target), weight in self.relations.items():
                if source == match:
                    related.append({"concept": target, "relation": relation, "weight": weight})
                elif target == match:
                    related.append({"concept": source, "relation": relation, "weight": weight})

        seen = set()
        unique_related = []
        for r in related:
            if r["concept"] not in seen:
                seen.add(r["concept"])
                unique_related.append(r)

        unique_related.sort(key=lambda x: x["weight"], reverse=True)
        return unique_related[:max_results]

    def get_concept_info(self, concept: str) -> Optional[Dict]:
        """Get full concept information."""
        if concept not in self.concepts:
            return None

        return {
            "concept": concept,
            "definition": self.concept_definitions.get(concept, ""),
            "metadata": self.concept_metadata.get(concept, {}),
            "related": self.find_related(concept),
        }

    def build_from_documents(self, documents: List[Dict], auto_relations: bool = True) -> None:
        """
        Build knowledge graph from documents.

        Args:
            documents: List of {"content": str, "metadata": dict}
            auto_relations: Auto-generate relations based on co-occurrence
        """
        concept_documents: Dict[str, List[str]] = defaultdict(list)

        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            words = content.lower().split()
            unique_words = set(w for w in words if len(w) > 4)

            for word in unique_words:
                concept_documents[word].append(content)

        for concept, contents in concept_documents.items():
            if len(contents) >= 1:
                definition = contents[0][:200]
                self.add_concept(concept, definition)

        if auto_relations:
            concepts = list(self.concepts.keys())
            for i, c1 in enumerate(concepts):
                for c2 in concepts[i + 1 : i + 4]:
                    self.add_relation(c1, "related_to", c2, weight=0.5)

    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        relation_types = defaultdict(int)
        for _, relation, _ in self.relations:
            relation_types[relation] += 1

        return {
            "total_concepts": len(self.concepts),
            "total_relations": len(self.relations),
            "relation_types": dict(relation_types),
        }


__all__ = [
    "RAGSystem",
    "VectorStore",
    "ChainOfThoughtRAG",
    "SelfReflectiveRAG",
    "MultiHopRAG",
    "EndicIndex",
    "RAGEngine",
    "SpacedRepetitionScheduler",
    "SLOKnowledgeGraph",
]
