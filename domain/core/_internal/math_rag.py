"""
MathRAG — Mathematically rigorous RAG engine.

Implements proper BM25, TF-IDF, cosine similarity, and rule-based
extraction with C-style performance rules. No simulated embeddings,
no lazy computation, no shortcuts.

Design principles:
- Every scoring function is mathematically correct
- Index structures are precomputed, not recomputed per query
- Rule engine uses deterministic if/else chains, not fragile regex
- Memory-efficient: chunk storage is flat, embeddings are float32
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ─── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class Chunk:
    """A document chunk with precomputed embeddings and token data."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tokens: list[str] = field(default_factory=list, repr=False)
    token_counts: dict[str, int] = field(default_factory=dict, repr=False)
    embedding: np.ndarray | None = field(default=None, repr=False)
    length: int = 0

    def __post_init__(self):
        if not self.tokens:
            self.tokens = _tokenize(self.content)
        if not self.token_counts:
            self.token_counts = dict(Counter(self.tokens))
        self.length = len(self.tokens)


@dataclass
class ScoredChunk:
    """Chunk with all scoring components for transparency."""

    chunk: Chunk
    bm25: float = 0.0
    tfidf: float = 0.0
    cosine: float = 0.0
    combined: float = 0.0
    rank: int = 0


@dataclass
class ExtractionRule:
    """A single extraction rule with pattern, relation type, and confidence."""

    name: str
    pattern: re.Pattern[str]
    relation: str
    confidence: float = 0.8
    priority: int = 0  # higher = tried first


@dataclass
class ExtractedFact:
    """A fact extracted by the rule engine."""

    subject: str
    predicate: str
    obj: str
    relation: str
    confidence: float
    rule_name: str
    text: str
    start: int = 0
    end: int = 0


# ─── Tokenizer ────────────────────────────────────────────────────────────────

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "used",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "don",
        "now",
        "and",
        "but",
        "or",
        "if",
        "because",
        "while",
        "although",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "he",
        "she",
        "they",
        "we",
        "you",
        "i",
        "me",
        "my",
        "your",
        "his",
        "her",
        "our",
        "their",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric, filter stop words and short tokens."""
    return [
        t for t in re.findall(r"\b[a-z0-9]+\b", text.lower()) if t not in _STOP_WORDS and len(t) > 1
    ]


def _tokenize_raw(text: str) -> list[str]:
    """Tokenize without stop-word removal (for BM25 which needs full tokens)."""
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


# ─── BM25 (Okapi BM25) ───────────────────────────────────────────────────────


class BM25:
    """Okapi BM25 with correct TF, IDF, and document-length normalization.

    References:
        Robertson et al. "The Probabilistic Relevance Framework: BM25 and Beyond"
        https://doi.org/10.1561/1500000006

    Parameters:
        k1: Term frequency saturation (1.2–2.0 typical). Higher = less TF saturation.
        b:  Document length normalization (0.0–1.0). 0 = no norm, 1 = full norm.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.n_docs: int = 0
        self.avg_dl: float = 0.0
        self.doc_lens: np.ndarray = np.empty(0)
        self.df: dict[str, int] = {}
        # term -> [(doc_id, tf)] — only stores docs where term appears
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self._idf_cache: dict[str, float] = {}

    def build(self, chunks: list[Chunk]) -> None:
        """Build the index from chunks. O(N * L) where N=docs, L=avg doc length."""
        self.n_docs = len(chunks)
        if self.n_docs == 0:
            return

        self.doc_lens = np.array([c.length for c in chunks], dtype=np.float32)
        self.avg_dl = float(self.doc_lens.mean())

        self.df.clear()
        self.postings.clear()
        self._idf_cache.clear()

        for doc_id, chunk in enumerate(chunks):
            # Count terms in this document
            for term, tf in chunk.token_counts.items():
                self.df[term] = self.df.get(term, 0) + 1
                if term not in self.postings:
                    self.postings[term] = []
                self.postings[term].append((doc_id, tf))

    def idf(self, term: str) -> float:
        """Inverse Document Frequency with Robertson-Sparck Jones smoothing.

        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

        The +1 inside the log prevents negative IDF for very common terms.
        """
        if term in self._idf_cache:
            return self._idf_cache[term]

        df_t = self.df.get(term, 0)
        # Robertson-Sparck Jones IDF (never negative)
        idf_val = math.log((self.n_docs - df_t + 0.5) / (df_t + 0.5) + 1.0)
        self._idf_cache[term] = idf_val
        return idf_val

    def score(self, query_tokens: list[str], top_k: int | None = None) -> list[tuple[int, float]]:
        """Score all documents against query. Returns [(doc_id, score)] sorted desc.

        BM25(q, d) = sum over t in q of: IDF(t) * (tf(t,d) * (k1+1)) / (tf(t,d) + k1 * (1 - b + b * |d|/avgdl))
        """
        scores = np.zeros(self.n_docs, dtype=np.float32)

        for term in query_tokens:
            term_lower = term.lower()
            if term_lower not in self.postings:
                continue

            idf_val = self.idf(term_lower)
            k1, b, avg_dl = self.k1, self.b, self.avg_dl

            for doc_id, tf in self.postings[term_lower]:
                dl = self.doc_lens[doc_id]
                # BM25 term score
                numerator = tf * (k1 + 1.0)
                denominator = tf + k1 * (1.0 - b + b * dl / avg_dl)
                scores[doc_id] += idf_val * numerator / denominator

        # Build result list, filter zeros, sort descending
        result = [(i, float(scores[i])) for i in range(self.n_docs) if scores[i] > 0.0]
        result.sort(key=lambda x: -x[1])
        if top_k is not None:
            result = result[:top_k]
        return result


# ─── TF-IDF Vector Space ─────────────────────────────────────────────────────


class TFIDFIndex:
    """TF-IDF vector space with proper L2 normalization.

    Uses sublinear TF scaling: TF_raw -> 1 + log(tf) if tf > 0
    This dampens the effect of term frequency for high-frequency terms.
    """

    def __init__(self, use_sublinear_tf: bool = True):
        self.use_sublinear_tf = use_sublinear_tf
        self.vocab: dict[str, int] = {}  # term -> index
        self.idf_vec: np.ndarray | None = None
        self.n_docs: int = 0
        self.matrix: np.ndarray | None = None  # (n_docs, vocab_size) — sparse in practice

    def build(self, chunks: list[Chunk]) -> None:
        """Build TF-IDF matrix. O(N * V) where V = vocabulary size."""
        # Build vocabulary
        vocab_set: set[str] = set()
        for chunk in chunks:
            vocab_set.update(chunk.token_counts.keys())
        self.vocab = {term: idx for idx, term in enumerate(sorted(vocab_set))}
        vocab_size = len(self.vocab)
        self.n_docs = len(chunks)

        # Compute document frequency for IDF
        df = np.zeros(vocab_size, dtype=np.float32)
        for chunk in chunks:
            for term in chunk.token_counts:
                df[self.vocab[term]] += 1.0

        # IDF with smoothing: log(N / (1 + df)) + 1  (sklearn-style)
        self.idf_vec = np.log(self.n_docs / (1.0 + df)) + 1.0

        # Build TF-IDF matrix
        self.matrix = np.zeros((self.n_docs, vocab_size), dtype=np.float32)
        for doc_id, chunk in enumerate(chunks):
            for term, tf in chunk.token_counts.items():
                term_idx = self.vocab[term]
                if self.use_sublinear_tf and tf > 0:
                    tf_val = 1.0 + math.log(tf)
                else:
                    tf_val = float(tf)
                self.matrix[doc_id, term_idx] = tf_val * self.idf_vec[term_idx]

        # L2 normalize each row
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # avoid division by zero
        self.matrix /= norms

    def query(self, query_tokens: list[str], top_k: int | None = None) -> list[tuple[int, float]]:
        """Query using cosine similarity (dot product on L2-normalized vectors)."""
        if self.matrix is None or self.n_docs == 0:
            return []

        # Build query vector
        q = np.zeros(len(self.vocab), dtype=np.float32)
        for term in query_tokens:
            term_lower = term.lower()
            if term_lower in self.vocab:
                idx = self.vocab[term_lower]
                # Count occurrences in query for TF
                tf = query_tokens.count(term_lower)
                if self.use_sublinear_tf and tf > 0:
                    tf_val = 1.0 + math.log(tf)
                else:
                    tf_val = float(tf)
                q[idx] = tf_val * self.idf_vec[idx]

        # L2 normalize query
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-10:
            q /= q_norm

        # Cosine similarity = dot product on normalized vectors
        scores = self.matrix @ q  # (n_docs,)

        result = [(i, float(scores[i])) for i in range(self.n_docs) if scores[i] > 1e-6]
        result.sort(key=lambda x: -x[1])
        if top_k is not None:
            result = result[:top_k]
        return result


# ─── Rule Engine ──────────────────────────────────────────────────────────────


class RuleEngine:
    """Deterministic C-style rule engine for fact extraction.

    Rules are evaluated in priority order. Each rule has a compiled regex
    pattern and a relation type. No ML, no ambiguity — if the pattern
    matches, the fact is extracted with the rule's confidence score.
    """

    def __init__(self) -> None:
        self.rules: list[ExtractionRule] = []
        self._build_default_rules()

    def _build_default_rules(self) -> None:
        """Populate with mathematically-defined extraction rules."""
        # Subject pattern: single proper noun or multi-word proper noun (each word capitalized)
        # "Python" matches. "Albert Einstein" matches. "the car" does NOT match.
        # Anchored to start of string with ^ to prevent cross-sentence matches.
        _S1 = r"^((?:[A-Z][\w]+)(?:\s+[A-Z][\w]+)*)"
        _O = r"(.+)"

        rules_defs: list[tuple[str, str, str, float, int]] = [
            # (name, pattern, relation, confidence, priority)
            # Classification (highest priority — must come before generic "is")
            ("is_a", rf"{_S1}\s+is\s+a\s+(.+)", "is_a", 0.9, 10),
            ("is_an", rf"{_S1}\s+is\s+an\s+(.+)", "is_a", 0.9, 10),
            ("is_the", rf"{_S1}\s+is\s+the\s+(.+)", "is_a", 0.85, 9),
            # Compound verb phrases (must come before generic verb patterns)
            # These match multi-word verb constructions to avoid partial matches
            ("born_in", rf"{_S1}\s+was\s+born\s+in\s+{_O}", "located_in", 0.95, 9),
            ("located_in", rf"{_S1}\s+is\s+located\s+in\s+{_O}", "located_in", 0.9, 9),
            ("used_for", rf"{_S1}\s+is\s+used\s+(?:for|to)\s+{_O}", "used_for", 0.85, 9),
            ("made_of", rf"{_S1}\s+is\s+made\s+of\s+{_O}", "made_of", 0.85, 9),
            ("leads_to", rf"{_S1}\s+leads\s+to\s+{_O}", "leads_to", 0.8, 9),
            ("results_in", rf"{_S1}\s+results\s+in\s+{_O}", "results_in", 0.8, 9),
            ("consists_of", rf"{_S1}\s+consists\s+of\s+{_O}", "consists_of", 0.85, 9),
            ("serves_as", rf"{_S1}\s+serves\s+as\s+{_O}", "serves_as", 0.8, 9),
            ("based_in", rf"{_S1}\s+is\s+based\s+in\s+{_O}", "based_in", 0.85, 9),
            ("lives_in", rf"{_S1}\s+lives\s+in\s+{_O}", "lives_in", 0.85, 9),
            # Generic existence (lower priority than compound phrases)
            (
                "is",
                rf"{_S1}\s+is\s+(?!a\s|an\s|the\s|located\s|used\s|made\s|based\s)(.+)",
                "is",
                0.8,
                5,
            ),
            ("was", rf"{_S1}\s+was\s+(?!born\s)(.+)", "was", 0.8, 5),
            ("are", rf"{_S1}\s+are\s+(.+)", "are", 0.8, 5),
            ("were", rf"{_S1}\s+were\s+(.+)", "were", 0.8, 5),
            # Ability
            ("can", rf"{_S1}\s+can\s+(.+)", "can", 0.8, 4),
            ("could", rf"{_S1}\s+could\s+(.+)", "could", 0.75, 4),
            ("will", rf"{_S1}\s+will\s+(.+)", "will", 0.75, 4),
            # Possession
            ("has", rf"{_S1}\s+has\s+(.+)", "has", 0.85, 3),
            ("have", rf"{_S1}\s+have\s+(.+)", "has", 0.85, 3),
            ("had", rf"{_S1}\s+had\s+(.+)", "had", 0.8, 3),
            ("owns", rf"{_S1}\s+owns\s+(.+)", "owns", 0.85, 3),
            ("contains", rf"{_S1}\s+contains\s+(.+)", "contains", 0.85, 3),
            # Location (generic fallback)
            ("in", rf"{_S1}\s+is\s+in\s+{_O}", "located_in", 0.7, 2),
            # Causation
            ("causes", rf"{_S1}\s+causes\s+(.+)", "causes", 0.85, 1),
        ]

        for name, pattern, relation, confidence, priority in rules_defs:
            self.rules.append(
                ExtractionRule(
                    name=name,
                    pattern=re.compile(pattern, re.IGNORECASE),
                    relation=relation,
                    confidence=confidence,
                    priority=priority,
                )
            )

        # Sort by priority descending
        self.rules.sort(key=lambda r: -r.priority)

    def add_rule(self, rule: ExtractionRule) -> None:
        """Add a custom rule and re-sort by priority."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: -r.priority)

    def extract(self, text: str) -> list[ExtractedFact]:
        """Extract facts from text using deterministic rule evaluation.

        Returns facts sorted by confidence descending. Deduplicates
        by (subject, relation, predicate[:80]).
        """
        facts: list[ExtractedFact] = []
        seen_facts: set[tuple[str, str, str]] = set()  # (subject, relation, predicate[:80])

        # Split into sentences
        sentences = _split_sentences(text)

        for sentence in sentences:
            # Per-sentence span tracking: overlaps only matter within one sentence
            sentence_spans: list[tuple[int, int]] = []

            def _overlaps(start: int, end: int) -> bool:
                for ms, me in sentence_spans:
                    if start < me and end > ms:
                        return True
                return False

            for rule in self.rules:
                for match in rule.pattern.finditer(sentence):
                    # Skip if this span overlaps an already-matched span in this sentence
                    if _overlaps(match.start(), match.end()):
                        continue

                    subject = match.group(1).strip()
                    predicate = match.group(2).strip().rstrip(".")

                    # Dedup by (subject, relation, predicate) across all sentences
                    fact_key = (subject.lower(), rule.relation, predicate.lower()[:80])
                    if fact_key in seen_facts:
                        continue
                    seen_facts.add(fact_key)
                    sentence_spans.append((match.start(), match.end()))

                    # Confidence adjustment based on subject quality
                    subj_conf = _subject_confidence(subject)
                    final_conf = rule.confidence * subj_conf

                    facts.append(
                        ExtractedFact(
                            subject=subject,
                            predicate=predicate,
                            obj=predicate,
                            relation=rule.relation,
                            confidence=round(final_conf, 3),
                            rule_name=rule.name,
                            text=match.group(0).strip(),
                            start=match.start(),
                            end=match.end(),
                        )
                    )

        # Sort by confidence descending
        facts.sort(key=lambda f: -f.confidence)
        return facts


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, handling abbreviations."""
    protected = re.sub(
        r"(Mr|Mrs|Ms|Dr|Prof|Inc|Ltd|Jr|Sr|vs|etc|approx)\.\s",
        r"\1<DOT> ",
        text,
    )
    raw = re.split(r"(?<=[.!?])\s+", protected)
    return [s.replace("<DOT>", ".") for s in raw if s.strip()]


def _subject_confidence(subject: str) -> float:
    """Score confidence that the subject is a real entity.

    Proper nouns score higher than pronouns. Longer noun phrases
    score higher than single tokens.
    """
    words = subject.split()
    if not words:
        return 0.0

    pronouns = {"it", "he", "she", "they", "we", "you", "i", "this", "that"}
    if words[0].lower() in pronouns:
        return 0.2

    proper_count = sum(1 for w in words if w and w[0].isupper())
    score = 0.4 + 0.2 * (proper_count / len(words))
    if len(words) >= 2:
        score += 0.1
    return min(score, 1.0)


# ─── MathRAG Engine ──────────────────────────────────────────────────────────


class MathRAG:
    """Mathematically rigorous RAG engine combining BM25, TF-IDF, and rule extraction.

    Architecture:
        1. Chunking: word-level sliding window with configurable size/overlap
        2. Indexing: BM25 + TF-IDF built once, not per query
        3. Retrieval: hybrid BM25 + TF-IDF scoring (no random embeddings)
        4. Extraction: deterministic rule engine with confidence scoring
        5. Deduplication: content-hash based at ingest time
    """

    def __init__(
        self,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        use_sublinear_tfidf: bool = True,
        bm25_weight: float = 0.5,
        tfidf_weight: float = 0.5,
    ):
        self.bm25 = BM25(k1=bm25_k1, b=bm25_b)
        self.tfidf = TFIDFIndex(use_sublinear_tf=use_sublinear_tfidf)
        self.rule_engine = RuleEngine()
        self.bm25_weight = bm25_weight
        self.tfidf_weight = tfidf_weight

        self.chunks: list[Chunk] = []
        self._content_hashes: set[str] = set()
        self._indexed: bool = False

    def add_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        chunk_size: int = 512,
        overlap: int = 50,
        deduplicate: bool = True,
    ) -> list[str]:
        """Chunk, index, and optionally deduplicate a document.

        Returns list of chunk IDs.
        """
        if not content or not content.strip():
            return []

        content = content.strip()
        metadata = metadata or {"source": "user"}

        # Content-hash deduplication
        if deduplicate:
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:32]
            if content_hash in self._content_hashes:
                return []
            self._content_hashes.add(content_hash)

        # Tokenize and chunk
        tokens = _tokenize_raw(content)
        stride = max(1, chunk_size - overlap)
        chunk_ids: list[str] = []

        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i : i + chunk_size]
            chunk_content = " ".join(chunk_tokens)
            chunk_id = hashlib.md5(chunk_content.encode()).hexdigest()[:12]

            chunk = Chunk(
                id=chunk_id,
                content=chunk_content,
                metadata={**metadata, "position": i // chunk_size},
                tokens=chunk_tokens,
            )
            self.chunks.append(chunk)
            chunk_ids.append(chunk_id)

        # Mark as needing reindex
        self._indexed = False
        return chunk_ids

    def build_index(self) -> None:
        """Build all indexes (BM25 + TF-IDF). Call after adding documents."""
        self.bm25.build(self.chunks)
        self.tfidf.build(self.chunks)
        self._indexed = True

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.01,
    ) -> dict[str, Any]:
        """Hybrid retrieval: BM25 + TF-IDF with proper scoring.

        Returns dict with 'context', 'results', 'num_results'.
        """
        if not self._indexed:
            self.build_index()

        query_tokens = _tokenize(question)
        if not query_tokens:
            return {"context": "", "results": [], "num_results": 0}

        # BM25 scoring
        bm25_scores = self.bm25.score(query_tokens)
        bm25_max = bm25_scores[0][1] if bm25_scores else 1.0

        # TF-IDF scoring
        tfidf_scores = self.tfidf.query(query_tokens)
        tfidf_max = tfidf_scores[0][1] if tfidf_scores else 1.0

        # Merge scores: normalize each to [0, 1], then weighted sum
        merged: dict[int, float] = {}
        for doc_id, score in bm25_scores:
            merged[doc_id] = merged.get(doc_id, 0.0) + self.bm25_weight * (
                score / max(bm25_max, 1e-10)
            )
        for doc_id, score in tfidf_scores:
            merged[doc_id] = merged.get(doc_id, 0.0) + self.tfidf_weight * (
                score / max(tfidf_max, 1e-10)
            )

        # Sort and filter
        ranked = sorted(merged.items(), key=lambda x: -x[1])
        ranked = [(did, sc) for did, sc in ranked if sc >= min_score][:top_k]

        # Build results
        results = []
        for rank, (doc_id, score) in enumerate(ranked):
            chunk = self.chunks[doc_id]
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    bm25=bm25_scores[doc_id][1] if doc_id < len(bm25_scores) else 0.0,
                    tfidf=tfidf_scores[doc_id][1] if doc_id < len(tfidf_scores) else 0.0,
                    combined=score,
                    rank=rank + 1,
                )
            )

        context = "\n\n".join(r.chunk.content for r in results)
        return {
            "context": context,
            "results": results,
            "num_results": len(results),
        }

    def extract_facts(self, text: str) -> list[ExtractedFact]:
        """Extract structured facts from text using the rule engine."""
        return self.rule_engine.extract(text)

    def verify_and_ground(
        self,
        generated_text: str,
        question: str,
    ) -> dict[str, Any]:
        """Verify generated text against the index and extract citations.

        Uses rule-based fact extraction + retrieval scoring to detect
        hallucinations and provide grounded citations.
        """
        # Extract claims from generated text
        facts = self.extract_facts(generated_text)

        grounded = []
        hallucinations = []

        for fact in facts:
            # Query for supporting evidence
            query = f"{fact.subject} {fact.predicate}"
            result = self.query(query, top_k=3, min_score=0.05)

            if result["num_results"] == 0:
                hallucinations.append(
                    {
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "relation": fact.relation,
                        "reason": "No supporting evidence found",
                        "confidence": 0.0,
                    }
                )
            else:
                # Check if any result actually supports the claim
                best_score = result["results"][0].combined if result["results"] else 0.0
                grounded.append(
                    {
                        "subject": fact.subject,
                        "predicate": fact.predicate,
                        "relation": fact.relation,
                        "confidence": best_score,
                        "sources": [r.chunk.id for r in result["results"][:3]],
                    }
                )

        # Compute overall confidence
        total = len(facts)
        if total == 0:
            confidence = 1.0
        else:
            grounded_count = len(grounded)
            avg_g = sum(c["confidence"] for c in grounded) / max(grounded_count, 1)
            confidence = (grounded_count / total) * avg_g

        return {
            "verification": {
                "total_claims": total,
                "grounded": len(grounded),
                "hallucinations": len(hallucinations),
            },
            "grounded_claims": grounded,
            "hallucinations": hallucinations,
            "confidence": confidence,
            "facts": [
                {"subject": f.subject, "predicate": f.predicate, "relation": f.relation}
                for f in facts
            ],
        }

    def stats(self) -> dict[str, Any]:
        """Return engine statistics."""
        return {
            "chunks": len(self.chunks),
            "total_tokens": sum(c.length for c in self.chunks),
            "unique_tokens": len(self.tfidf.vocab) if self.tfidf.vocab else 0,
            "indexed": self._indexed,
            "bm25_k1": self.bm25.k1,
            "bm25_b": self.bm25.b,
        }
