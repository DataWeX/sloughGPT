"""
Tests for Production-Grade RAG System
"""

import pytest

from domain.cognition._internal.rag import (
    BM25Indexer,
    CitationTracker,
    HybridRetriever,
    ProductionRAG,
    TextChunk,
)


class TestBM25:
    """Tests for BM25 indexer."""

    def test_indexing(self):
        """Test BM25 indexing."""
        chunks = [
            TextChunk("1", "Python is a programming language", {}),
            TextChunk("2", "Machine learning is a subset of AI", {}),
            TextChunk("3", "Deep learning uses neural networks", {}),
        ]

        bm25 = BM25Indexer()
        bm25.index(chunks)

        assert bm25.num_docs == 3
        assert bm25.avg_doc_length > 0

    def test_scoring(self):
        """Test BM25 scoring."""
        chunks = [
            TextChunk("1", "Python is a programming language", {}),
            TextChunk("2", "Java is also a programming language", {}),
            TextChunk("3", "Machine learning is useful", {}),
        ]

        bm25 = BM25Indexer()
        bm25.index(chunks)

        scores = bm25.score("programming language")
        assert len(scores) > 0
        assert scores[0][1] > 0  # First result should have positive score


class TestHybridRetriever:
    """Tests for hybrid retriever."""

    def test_add_and_retrieve(self):
        """Test adding chunks and retrieval."""
        retriever = HybridRetriever()

        chunks = [
            TextChunk("1", "Python is a programming language", {"source": "test"}),
            TextChunk("2", "Python is used in ML", {"source": "test"}),
        ]

        for chunk in chunks:
            retriever.add_chunk(chunk)

        retriever.build_index()

        results = retriever.retrieve("Python programming", top_k=2)

        assert len(results) > 0
        assert results[0].combined_score > 0

    def test_hybrid_scoring(self):
        """Test that hybrid scoring combines dense and sparse."""
        retriever = HybridRetriever(dense_weight=0.5, sparse_weight=0.5)

        chunks = [
            TextChunk("1", "The quick brown fox jumps", {"source": "test"}),
            TextChunk("2", "A lazy dog sleeps", {"source": "test"}),
        ]

        for chunk in chunks:
            retriever.add_chunk(chunk)
        retriever.build_index()

        results = retriever.retrieve("fox", top_k=1)

        assert len(results) == 1
        assert results[0].dense_score >= 0
        assert results[0].sparse_score >= 0


class TestCitationTracker:
    """Tests for citation tracker."""

    def test_extract_claims(self):
        """Test claim extraction."""
        tracker = CitationTracker()

        text = "Python is a programming language. Java is also a language."
        claims = tracker.extract_claims(text)

        assert len(claims) >= 1
        assert any("Python" in c["subject"] for c in claims)

    def test_cite(self):
        """Test citation creation."""
        tracker = CitationTracker()

        claim = {
            "subject": "Python",
            "predicate": "is",
            "object": "a programming language",
        }

        chunk = TextChunk("1", "Python is a programming language", {"source": "docs"})
        cited = tracker.cite(claim, [chunk])

        assert cited["supported"]
        assert len(cited["sources"]) == 1


class TestProductionRAG:
    """Tests for production RAG system."""

    def test_add_document(self):
        """Test document addition with chunking."""
        rag = ProductionRAG()

        chunk_ids = rag.add_document(
            "Python is a programming language. " * 100,
            metadata={"source": "test"},
            chunk_size=10,
        )

        assert len(chunk_ids) > 1

    def test_query(self):
        """Test RAG query."""
        rag = ProductionRAG()

        rag.add_document(
            "Python is a programming language developed in the 1990s.",
            metadata={"source": "python.org"},
        )
        rag.add_document(
            "Machine learning is a subset of artificial intelligence.",
            metadata={"source": "ml.org"},
        )

        results = rag.query("What is Python?", top_k=1)

        assert "results" in results
        assert "context" in results
        assert len(results["results"]) >= 1

    def test_verify_and_ground(self):
        """Test verification of generated text."""
        rag = ProductionRAG()

        rag.add_document(
            "Python is a programming language.",
            metadata={"source": "docs"},
        )

        verification = rag.verify_and_ground(
            "Python is a programming language.",
            "What is Python?",
        )

        assert "verification" in verification
        assert "confidence" in verification
        assert verification["confidence"] > 0


class TestBM25ScoreLinear:
    """score() must be O(postings), not O(postings^2).

    The old implementation rescanned the full posting list once per
    occurrence (rag.py:135), so a common term with P postings cost P^2
    work — queries with words like "one"/"the" in a 16k-chunk index
    never finished and were killed by the 15s RAG wait_for (chat TTFT
    ballooned by 15s per request, and the leaked thread kept grinding
    the GIL).
    """

    def test_score_is_linear_in_postings(self):
        """6000 postings must score in well under a second (old: ~seconds)."""
        import time

        bm25 = BM25Indexer()
        chunks = [TextChunk(str(i), f"hot doc{i}", {}) for i in range(6000)]
        bm25.index(chunks)

        t0 = time.monotonic()
        results = bm25.score("hot")
        elapsed = time.monotonic() - t0

        assert len(results) == 6000
        assert elapsed < 1.0, f"score() took {elapsed:.2f}s — O(P^2) regression"

    def test_score_matches_standard_bm25(self):
        """tf must be counted once per doc (standard BM25), not once per occurrence."""
        import math

        bm25 = BM25Indexer()
        chunks = [
            TextChunk("a", "alpha alpha beta", {}),
            TextChunk("b", "alpha gamma", {}),
            TextChunk("c", "delta epsilon", {}),
        ]
        bm25.index(chunks)

        scores = dict(bm25.score("alpha"))

        # Hand-computed standard BM25 (k1=1.5, b=0.75):
        # df(alpha)=2, num_docs=3, lengths=[3,2,2] avg=7/3
        idf = math.log((3 - 2 + 0.5) / (2 + 0.5) + 1)
        avg_dl = 7 / 3

        def expected(tf: float, dl: float) -> float:
            return idf * (tf * 2.5) / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avg_dl))

        assert scores[0] == pytest.approx(expected(2, 3), rel=1e-6)
        assert scores[1] == pytest.approx(expected(1, 2), rel=1e-6)
        assert 2 not in scores  # no "alpha" → not a candidate


class TestAutoIngestBulk:
    """auto_ingest_directory must bulk-load then rebuild ONCE (per-file
    rebuilds are O(n^2): 150 startup files × full index rebuild each)."""

    def test_auto_ingest_rebuilds_index_once(self, monkeypatch, tmp_path):
        import domain.infrastructure._internal.auto_ingest as ai
        from domain.cognition._internal.rag_service import RAGService

        svc = RAGService()
        monkeypatch.setattr(svc, "_save_document", lambda doc: None)
        monkeypatch.setattr(svc, "_extract_kg_claims", lambda content, metadata: None)

        class FakeScanner:
            def __init__(self, root_path=None):
                pass

            def iter_files(self):
                for i in range(5):
                    yield (
                        tmp_path / f"f{i}.py",
                        f"def function_{i}():\n    return {i}\n" + "# " + "x" * 60,
                    )

            def get_file_type(self, path):
                return "py"

        monkeypatch.setattr(ai, "RepoScanner", FakeScanner)

        builds: list[int] = []
        monkeypatch.setattr(
            svc.rag.retriever, "build_index", lambda: builds.append(1)
        )

        ingested = svc.auto_ingest_directory(str(tmp_path), max_files=10)

        assert ingested == 5
        assert len(builds) == 1, f"build_index called {len(builds)}x, want 1 (bulk)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
