"""
Tests for Production-Grade RAG System
"""

import pytest
from domains.cognitive.rag import (
    TextChunk,
    BM25Indexer,
    HybridRetriever,
    CitationTracker,
    ProductionRAG,
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

        assert cited["supported"] == True
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
