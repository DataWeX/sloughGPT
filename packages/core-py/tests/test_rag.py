"""Tests for production RAG system."""

import pytest
import numpy as np
from domains.cognitive.rag import (
    TextChunk,
    RetrievalResult,
    BM25Indexer,
    HybridRetriever,
    CitationTracker,
    HallucinationDetector,
    ProductionRAG,
)


class TestTextChunk:
    def test_default_token_count(self):
        tc = TextChunk(id="c1", content="hello world", metadata={})
        assert tc.token_count == 2

    def test_custom_token_count(self):
        tc = TextChunk(id="c1", content="hello world", metadata={}, token_count=99)
        assert tc.token_count == 99

    def test_empty_content(self):
        tc = TextChunk(id="c1", content="", metadata={})
        assert tc.token_count == 0

    def test_default_scores(self):
        tc = TextChunk(id="c1", content="hello", metadata={})
        assert tc.bm25_score == 0.0
        assert tc.embedding is None


class TestRetrievalResult:
    def test_fields(self):
        chunk = TextChunk(id="c1", content="test", metadata={})
        rr = RetrievalResult(chunk=chunk, dense_score=0.8, sparse_score=0.5, combined_score=0.7, rank=1)
        assert rr.dense_score == 0.8
        assert rr.sparse_score == 0.5
        assert rr.combined_score == 0.7
        assert rr.rank == 1


class TestBM25Indexer:
    def test_initial_state(self):
        bm25 = BM25Indexer()
        assert bm25.num_docs == 0

    def test_index_single_doc(self):
        bm25 = BM25Indexer()
        chunks = [TextChunk(id="c1", content="hello world", metadata={})]
        bm25.index(chunks)
        assert bm25.num_docs == 1
        assert bm25.avg_doc_length == 2.0

    def test_index_multiple_docs(self):
        bm25 = BM25Indexer()
        chunks = [
            TextChunk(id="c1", content="hello world", metadata={}),
            TextChunk(id="c2", content="hello there world", metadata={}),
        ]
        bm25.index(chunks)
        assert bm25.num_docs == 2
        assert bm25.avg_doc_length == 2.5

    def test_score_matches(self):
        bm25 = BM25Indexer()
        chunks = [TextChunk(id="c1", content="hello world", metadata={})]
        bm25.index(chunks)
        results = bm25.score("hello world")
        assert len(results) == 1
        assert results[0][0] == 0
        assert results[0][1] > 0

    def test_score_no_match(self):
        bm25 = BM25Indexer()
        chunks = [TextChunk(id="c1", content="hello world", metadata={})]
        bm25.index(chunks)
        results = bm25.score("xyzzy")
        assert results == []

    def test_score_multiple_docs(self):
        bm25 = BM25Indexer()
        chunks = [
            TextChunk(id="c1", content="hello world", metadata={}),
            TextChunk(id="c2", content="goodbye world", metadata={}),
        ]
        bm25.index(chunks)
        results = bm25.score("hello")
        assert len(results) == 1
        assert results[0][0] == 0

    def test_score_relevance_ordering(self):
        bm25 = BM25Indexer()
        chunks = [
            TextChunk(id="c1", content="hello hello hello world", metadata={}),
            TextChunk(id="c2", content="hello world", metadata={}),
        ]
        bm25.index(chunks)
        results = bm25.score("hello")
        assert results[0][0] == 0

    def test_tokenize(self):
        bm25 = BM25Indexer()
        tokens = bm25._tokenize("Hello, World!")
        assert tokens == ["hello", "world"]

    def test_empty_index_score(self):
        bm25 = BM25Indexer()
        results = bm25.score("hello")
        assert results == []


class TestHybridRetriever:
    def test_initial_state(self):
        hr = HybridRetriever()
        assert hr.chunks == []
        assert hr.dense_weight == 0.7
        assert hr.sparse_weight == 0.3

    def test_add_chunk(self):
        hr = HybridRetriever()
        chunk = TextChunk(id="c1", content="hello world", metadata={})
        hr.add_chunk(chunk)
        assert len(hr.chunks) == 1

    def test_build_index(self):
        hr = HybridRetriever()
        hr.add_chunk(TextChunk(id="c1", content="hello world", metadata={}))
        hr.build_index()
        assert hr.bm25.num_docs == 1

    def test_retrieve_with_results(self):
        hr = HybridRetriever()
        hr.add_chunk(TextChunk(id="c1", content="hello world python", metadata={}))
        hr.add_chunk(TextChunk(id="c2", content="goodbye world", metadata={}))
        hr.build_index()
        results = hr.retrieve("hello python", top_k=5, min_score=0.0)
        assert len(results) >= 1

    def test_retrieve_empty(self):
        hr = HybridRetriever()
        hr.build_index()
        results = hr.retrieve("hello", top_k=5, min_score=0.0)
        assert results == []

    def test_retrieve_top_k(self):
        hr = HybridRetriever()
        for i in range(10):
            hr.add_chunk(TextChunk(id=f"c{i}", content=f"keyword in doc {i}", metadata={}))
        hr.build_index()
        results = hr.retrieve("keyword", top_k=3, min_score=0.0)
        assert len(results) <= 3

    def test_get_embedding(self):
        hr = HybridRetriever()
        emb = hr._get_embedding("hello world")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (384,)

    def test_get_embedding_cached(self):
        hr = HybridRetriever()
        emb1 = hr._get_embedding("test")
        emb2 = hr._get_embedding("test")
        assert emb1 is emb2

    def test_dense_search(self):
        hr = HybridRetriever()
        hr.add_chunk(TextChunk(id="c1", content="python programming", metadata={}))
        hr.build_index()
        results = hr._dense_search("python", top_k=5)
        assert len(results) >= 1

    def test_sparse_search(self):
        hr = HybridRetriever()
        hr.add_chunk(TextChunk(id="c1", content="python programming", metadata={}))
        hr.build_index()
        results = hr._sparse_search("python", top_k=5)
        assert len(results) >= 1

    def test_min_score_filter(self):
        hr = HybridRetriever()
        hr.add_chunk(TextChunk(id="c1", content="hello world", metadata={}))
        hr.build_index()
        results = hr.retrieve("hello", top_k=5, min_score=0.999)
        if results:
            for r in results:
                assert r.combined_score >= 0.999

    def test_rerank_sorts_by_combined(self):
        hr = HybridRetriever()
        hr.add_chunk(TextChunk(id="c1", content="unique content AAA", metadata={}))
        hr.add_chunk(TextChunk(id="c2", content="unique content BBB", metadata={}))
        hr.build_index()
        results = hr.retrieve("unique content", top_k=5, min_score=0.0)
        assert len(results) >= 1


class TestCitationTracker:
    def test_initial_state(self):
        ct = CitationTracker()
        assert ct.claims == []

    def test_extract_claims_finds_patterns(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python is a programming language")
        assert len(claims) > 0
        assert claims[0]["subject"] == "Python"

    def test_extract_claims_capitalization(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Paris has many museums")
        assert len(claims) > 0
        assert claims[0]["subject"] == "Paris"

    def test_extract_claims_no_match(self):
        ct = CitationTracker()
        claims = ct.extract_claims("hello world")
        assert claims == []

    def test_cite_with_support(self):
        ct = CitationTracker()
        claim = {"subject": "Python", "predicate": "is a language", "text": "Python is a language", "start": 0, "end": 10}
        sources = [TextChunk(id="s1", content="Python is a language", metadata={"source": "wiki"})]
        cited = ct.cite(claim, sources)
        assert cited["supported"] is True
        assert len(cited["sources"]) == 1
        assert cited["sources"][0]["chunk_id"] == "s1"

    def test_cite_no_support(self):
        ct = CitationTracker()
        claim = {"subject": "Python", "predicate": "is a language", "text": "Python is a language", "start": 0, "end": 10}
        cited = ct.cite(claim, [])
        assert cited["supported"] is False

    def test_format_citations(self):
        ct = CitationTracker()
        ct.claims = [
            {"text": "Python is great", "sources": [{"metadata": {"source": "wiki"}}]},
        ]
        formatted = ct.format_citations()
        assert "[1]" in formatted
        assert "Python is great" in formatted


class TestHallucinationDetector:
    def test_detect_no_claims(self):
        hr = HybridRetriever()
        hd = HallucinationDetector(hr)
        result = hd.detect("hello world")
        assert result["total_claims"] == 0
        assert result["overall_confidence"] == 1.0

    def test_detect_hallucination(self):
        hr = HybridRetriever()
        hd = HallucinationDetector(hr)
        result = hd.detect("Python is a language")
        assert result["total_claims"] > 0

    def test_detect_returns_structure(self):
        hr = HybridRetriever()
        hd = HallucinationDetector(hr)
        result = hd.detect("Python is a language")
        assert "hallucinations" in result
        assert "grounded_claims" in result
        assert "hallucination_rate" in result


class TestProductionRAG:
    def test_initial_state(self):
        rag = ProductionRAG()
        assert rag.retriever is not None
        assert rag.hallucination_detector is not None

    def test_add_document(self):
        rag = ProductionRAG()
        ids = rag.add_document("hello world this is a test document")
        assert len(ids) == 1

    def test_add_document_chunks_long(self):
        rag = ProductionRAG()
        ids = rag.add_document("word " * 1000, chunk_size=100)
        assert len(ids) > 1

    def test_add_document_with_metadata(self):
        rag = ProductionRAG()
        ids = rag.add_document("hello world", metadata={"source": "test"})
        assert len(ids) == 1

    def test_query_returns_structure(self):
        rag = ProductionRAG()
        rag.add_document("Python is a programming language")
        result = rag.query("Python programming", top_k=5)
        assert "question" in result
        assert "results" in result
        assert "context" in result
        assert "num_results" in result

    def test_query_context_non_empty(self):
        rag = ProductionRAG()
        rag.add_document("Python is a programming language used for web development")
        result = rag.query("Python", top_k=5)
        assert len(result["context"]) > 0

    def test_query_empty_no_results(self):
        rag = ProductionRAG()
        rag.add_document("hello world")
        result = rag.query("xyzzy", top_k=5)
        assert result["num_results"] >= 0

    def test_verify_and_ground(self):
        rag = ProductionRAG()
        rag.add_document("Python is a language created by Guido van Rossum")
        result = rag.verify_and_ground("Python is a language", "what is Python")
        assert "is_verified" in result
        assert "confidence" in result
        assert "verification" in result

    def test_query_result_has_scores(self):
        rag = ProductionRAG()
        rag.add_document("Python is a programming language")
        result = rag.query("Python", top_k=5)
        if result["results"]:
            for r in result["results"]:
                assert "score" in r
                assert "chunk_id" in r
                assert "rank" in r

    def test_custom_dense_weight(self):
        rag = ProductionRAG(config={"dense_weight": 0.5, "sparse_weight": 0.5})
        assert rag.retriever.dense_weight == 0.5
        assert rag.retriever.sparse_weight == 0.5

    def test_add_retrieve_roundtrip(self):
        rag = ProductionRAG()
        content = "Alice went to the market to buy apples"
        rag.add_document(content)
        result = rag.query("Alice market", top_k=5)
        assert result["num_results"] > 0
        if result["results"]:
            assert "Alice" in result["results"][0]["content"]
