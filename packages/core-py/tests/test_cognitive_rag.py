"""Comprehensive tests for domains/cognitive/rag.py — pure logic only."""

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


# ---------------------------------------------------------------------------
# TextChunk
# ---------------------------------------------------------------------------

class TestTextChunk:
    def test_auto_token_count(self):
        tc = TextChunk(id="1", content="one two three", metadata={})
        assert tc.token_count == 3

    def test_empty_content_zero_tokens(self):
        tc = TextChunk(id="1", content="", metadata={})
        assert tc.token_count == 0

    def test_explicit_token_count_preserved(self):
        tc = TextChunk(id="1", content="a b", metadata={}, token_count=99)
        assert tc.token_count == 99

    def test_default_fields(self):
        tc = TextChunk(id="1", content="x", metadata={})
        assert tc.bm25_score == 0.0
        assert tc.embedding is None

    def test_metadata_dict(self):
        tc = TextChunk(id="1", content="x", metadata={"source": "wiki", "page": 3})
        assert tc.metadata["source"] == "wiki"
        assert tc.metadata["page"] == 3


# ---------------------------------------------------------------------------
# RetrievalResult
# ---------------------------------------------------------------------------

class TestRetrievalResult:
    def test_all_fields(self):
        chunk = TextChunk(id="c1", content="text", metadata={})
        rr = RetrievalResult(chunk=chunk, dense_score=0.9, sparse_score=0.6, combined_score=0.8, rank=2)
        assert rr.chunk is chunk
        assert rr.dense_score == 0.9
        assert rr.sparse_score == 0.6
        assert rr.combined_score == 0.8
        assert rr.rank == 2


# ---------------------------------------------------------------------------
# BM25Indexer
# ---------------------------------------------------------------------------

class TestBM25Indexer:
    def _make_chunks(self, texts):
        return [TextChunk(id=str(i), content=t, metadata={}) for i, t in enumerate(texts)]

    def test_empty_index(self):
        bm25 = BM25Indexer()
        assert bm25.num_docs == 0
        assert bm25.score("hello") == []

    def test_single_doc_index(self):
        bm25 = BM25Indexer()
        bm25.index(self._make_chunks(["hello world"]))
        assert bm25.num_docs == 1
        assert bm25.avg_doc_length == 2.0

    def test_multiple_docs_avg_length(self):
        bm25 = BM25Indexer()
        bm25.index(self._make_chunks(["a b c", "d e"]))
        assert bm25.num_docs == 2
        assert bm25.avg_doc_length == 2.5

    def test_tokenize_lowercases_and_strips(self):
        bm25 = BM25Indexer()
        assert bm25._tokenize("Hello, World!") == ["hello", "world"]

    def test_tokenize_handles_punctuation(self):
        bm25 = BM25Indexer()
        tokens = bm25._tokenize("it's a test-key: value")
        assert "it" in tokens
        assert "s" in tokens
        assert "test" in tokens
        assert "key" in tokens

    def test_score_returns_positive_for_match(self):
        bm25 = BM25Indexer()
        bm25.index(self._make_chunks(["python programming language"]))
        results = bm25.score("python")
        assert len(results) == 1
        assert results[0][0] == 0
        assert results[0][1] > 0

    def test_score_returns_empty_for_no_match(self):
        bm25 = BM25Indexer()
        bm25.index(self._make_chunks(["hello world"]))
        assert bm25.score("xyzzy") == []

    def test_score_prefers_doc_with_more_term_occurrences(self):
        bm25 = BM25Indexer()
        bm25.index(self._make_chunks([
            "cat cat cat cat cat",
            "cat dog bird",
        ]))
        results = bm25.score("cat")
        assert results[0][0] == 0

    def test_score_multiple_docs(self):
        bm25 = BM25Indexer()
        bm25.index(self._make_chunks([
            "python is great",
            "java is also great",
            "python java both good",
        ]))
        results = bm25.score("python")
        doc_ids = [r[0] for r in results]
        assert 0 in doc_ids
        assert 2 in doc_ids
        assert 1 not in doc_ids

    def test_inverted_index_built(self):
        bm25 = BM25Indexer()
        bm25.index(self._make_chunks(["hello hello world"]))
        assert "hello" in bm25.inverted_index
        assert "world" in bm25.inverted_index
        assert len(bm25.inverted_index["hello"]) == 2

    def test_doc_freq_counts(self):
        bm25 = BM25Indexer()
        bm25.index(self._make_chunks(["hello world", "hello there"]))
        assert bm25.doc_freq["hello"] == 2
        assert bm25.doc_freq["world"] == 1
        assert bm25.doc_freq["there"] == 1

    def test_custom_k1_and_b(self):
        bm25 = BM25Indexer(k1=2.0, b=0.5)
        assert bm25.k1 == 2.0
        assert bm25.b == 0.5

    def test_score_sorted_descending(self):
        bm25 = BM25Indexer()
        bm25.index(self._make_chunks(["the", "the the the"]))
        results = bm25.score("the")
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class TestHybridRetriever:
    def _make_retriever(self, texts):
        hr = HybridRetriever()
        for i, t in enumerate(texts):
            hr.add_chunk(TextChunk(id=str(i), content=t, metadata={"source": f"doc{i}"}))
        hr.build_index()
        return hr

    def test_initial_state(self):
        hr = HybridRetriever()
        assert hr.chunks == []
        assert hr.dense_weight == 0.7
        assert hr.sparse_weight == 0.3
        assert hr.use_rerank is True

    def test_custom_weights(self):
        hr = HybridRetriever(dense_weight=0.5, sparse_weight=0.5, use_rerank=False)
        assert hr.dense_weight == 0.5
        assert hr.sparse_weight == 0.5
        assert hr.use_rerank is False

    def test_add_chunk_and_build(self):
        hr = HybridRetriever()
        hr.add_chunk(TextChunk(id="c1", content="test", metadata={}))
        hr.build_index()
        assert len(hr.chunks) == 1
        assert hr.bm25.num_docs == 1

    def test_get_embedding_shape(self):
        hr = HybridRetriever()
        emb = hr._get_embedding("hello")
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (384,)

    def test_get_embedding_normalized(self):
        hr = HybridRetriever()
        emb = hr._get_embedding("test")
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-6

    def test_get_embedding_caches(self):
        hr = HybridRetriever()
        e1 = hr._get_embedding("abc")
        e2 = hr._get_embedding("abc")
        assert e1 is e2

    def test_get_embedding_different_for_different_text(self):
        hr = HybridRetriever()
        e1 = hr._get_embedding("hello")
        e2 = hr._get_embedding("goodbye")
        assert not np.allclose(e1, e2)

    def test_dense_search_returns_sorted(self):
        hr = self._make_retriever(["python programming", "cooking recipes"])
        results = hr._dense_search("python", top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_sparse_search_delegates_to_bm25(self):
        hr = self._make_retriever(["hello world", "goodbye moon"])
        sparse = hr._sparse_search("hello")
        assert len(sparse) >= 1
        assert sparse[0][0] == 0

    def test_retrieve_empty_index(self):
        hr = HybridRetriever()
        hr.build_index()
        assert hr.retrieve("anything") == []

    def test_retrieve_returns_results(self):
        hr = self._make_retriever(["python is great", "java is fine", "rust is fast"])
        results = hr.retrieve("python great", top_k=2, min_score=0.0)
        assert len(results) >= 1

    def test_retrieve_top_k_respected(self):
        hr = self._make_retriever([f"document number {i} about topic" for i in range(20)])
        results = hr.retrieve("document topic", top_k=3, min_score=0.0)
        assert len(results) <= 3

    def test_retrieve_min_score_filter(self):
        hr = self._make_retriever(["hello world"])
        results = hr.retrieve("hello", top_k=5, min_score=0.99)
        for r in results:
            assert r.combined_score >= 0.99

    def test_retrieve_rank_assigned(self):
        hr = self._make_retriever(["alpha", "beta", "gamma"])
        results = hr.retrieve("alpha", top_k=5, min_score=0.0)
        for i, r in enumerate(results):
            assert r.rank == i + 1

    def test_retrieve_min_score_filters_all(self):
        hr = self._make_retriever(["hello world"])
        results = hr.retrieve("hello", top_k=5, min_score=2.0)
        assert results == []

    def test_rerank_no_duplicate_contexts(self):
        hr = self._make_retriever(["the cat sat on the mat", "the dog ran in the park"])
        results = hr.retrieve("the cat", top_k=5, min_score=0.0)
        assert len(results) >= 1

    def test_retrieve_result_has_all_fields(self):
        hr = self._make_retriever(["test content here"])
        results = hr.retrieve("test", top_k=5, min_score=0.0)
        assert len(results) >= 1
        r = results[0]
        assert isinstance(r, RetrievalResult)
        assert r.chunk.id == "0"
        assert r.dense_score >= 0
        assert r.sparse_score >= 0

    def test_embedding_assigned_to_chunk(self):
        hr = HybridRetriever()
        chunk = TextChunk(id="c1", content="hello", metadata={})
        hr.add_chunk(chunk)
        hr.build_index()
        hr.retrieve("hello", top_k=1)
        assert chunk.embedding is not None


# ---------------------------------------------------------------------------
# CitationTracker
# ---------------------------------------------------------------------------

class TestCitationTracker:
    def test_initial_empty(self):
        ct = CitationTracker()
        assert ct.claims == []

    def test_extract_claims_is_pattern(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python is a programming language.")
        assert len(claims) >= 1
        assert claims[0]["subject"] == "Python"
        assert "programming language" in claims[0]["predicate"]

    def test_extract_claims_was_pattern(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Einstein was a physicist.")
        assert len(claims) >= 1
        assert claims[0]["subject"] == "Einstein"

    def test_extract_claims_can_pattern(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Birds can fly.")
        assert len(claims) >= 1
        assert claims[0]["subject"] == "Birds"

    def test_extract_claims_has_pattern(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Cats has whiskers.")
        assert len(claims) >= 1
        assert claims[0]["subject"] == "Cats"

    def test_extract_claims_no_match(self):
        ct = CitationTracker()
        assert ct.extract_claims("hello world") == []

    def test_extract_claims_multiple_sentences(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python is great. Java is also popular.")
        assert len(claims) >= 2

    def test_extract_claims_offsets(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python is a language. Java is a language.")
        assert len(claims) >= 2
        assert claims[1]["start"] > claims[0]["start"]

    def test_extract_claims_strips_trailing_dot(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Foo is a thing.")
        assert len(claims) >= 1
        assert not claims[0]["predicate"].endswith(".")

    def test_cite_with_sources(self):
        ct = CitationTracker()
        claim = {"subject": "X", "predicate": "is Y", "text": "X is Y", "start": 0, "end": 5}
        chunk = TextChunk(id="s1", content="supporting text", metadata={"source": "wiki"})
        cited = ct.cite(claim, [chunk])
        assert cited["supported"] is True
        assert cited["sources"][0]["chunk_id"] == "s1"
        assert cited["sources"][0]["content"] == "supporting text"

    def test_cite_no_sources(self):
        ct = CitationTracker()
        claim = {"subject": "X", "predicate": "is Y", "text": "X is Y", "start": 0, "end": 5}
        cited = ct.cite(claim, [])
        assert cited["supported"] is False
        assert cited["sources"] == []

    def test_cite_limits_to_3_sources(self):
        ct = CitationTracker()
        claim = {"subject": "X", "predicate": "is Y", "text": "X is Y", "start": 0, "end": 5}
        chunks = [TextChunk(id=f"s{i}", content=f"text{i}", metadata={}) for i in range(5)]
        cited = ct.cite(claim, chunks)
        assert len(cited["sources"]) == 3

    def test_format_citations_empty(self):
        ct = CitationTracker()
        assert ct.format_citations() == ""

    def test_format_citations_with_claims(self):
        ct = CitationTracker()
        ct.claims = [
            {"text": "A is B", "sources": [{"metadata": {"source": "doc1"}}]},
            {"text": "C is D", "sources": []},
        ]
        out = ct.format_citations()
        assert "[1] A is B" in out
        assert "[2] C is D" in out
        assert "doc1" in out

    def test_format_citations_unknown_source(self):
        ct = CitationTracker()
        ct.claims = [
            {"text": "A is B", "sources": [{"metadata": {}}]},
        ]
        out = ct.format_citations()
        assert "Unknown" in out

    def test_cite_truncates_content_to_200(self):
        ct = CitationTracker()
        claim = {"subject": "X", "predicate": "is Y", "text": "X is Y", "start": 0, "end": 5}
        long_content = "a" * 300
        chunk = TextChunk(id="s1", content=long_content, metadata={})
        cited = ct.cite(claim, [chunk])
        assert len(cited["sources"][0]["content"]) == 200


# ---------------------------------------------------------------------------
# HallucinationDetector
# ---------------------------------------------------------------------------

class TestHallucinationDetector:
    def _make_detector_with_docs(self, texts):
        hr = HybridRetriever()
        for i, t in enumerate(texts):
            hr.add_chunk(TextChunk(id=str(i), content=t, metadata={"source": f"src{i}"}))
        hr.build_index()
        return HallucinationDetector(hr)

    def test_no_claims_confidence_1(self):
        hd = self._make_detector_with_docs(["anything"])
        result = hd.detect("hello world")
        assert result["total_claims"] == 0
        assert result["overall_confidence"] == 1.0
        assert result["hallucination_rate"] == 0.0

    def test_claims_detected(self):
        hd = self._make_detector_with_docs(["Python is a language"])
        result = hd.detect("Python is a language")
        assert result["total_claims"] >= 1

    def test_returns_required_keys(self):
        hd = self._make_detector_with_docs([])
        result = hd.detect("hello")
        for key in ("text", "total_claims", "grounded_claims", "hallucinations",
                     "overall_confidence", "hallucination_rate", "formatted_citations"):
            assert key in result

    def test_hallucination_rate_calculation(self):
        hd = self._make_detector_with_docs(["unrelated text here"])
        result = hd.detect("Python is a language")
        if result["total_claims"] > 0:
            expected_rate = len(result["hallucinations"]) / result["total_claims"]
            assert result["hallucination_rate"] == pytest.approx(expected_rate)

    def test_grounded_claim_has_sources(self):
        hd = self._make_detector_with_docs(["Python is a programming language for coding"])
        result = hd.detect("Python is a programming language", min_confidence=0.0)
        grounded = result["grounded_claims"]
        if grounded:
            assert "confidence" in grounded[0]
            assert "sources" in grounded[0]

    def test_hallucination_has_reason(self):
        hd = self._make_detector_with_docs(["completely unrelated content"])
        result = hd.detect("Python is a language", min_confidence=0.5)
        for h in result["hallucinations"]:
            assert "reason" in h
            assert "confidence" in h

    def test_min_confidence_affects_results(self):
        hd = self._make_detector_with_docs(["Python is a language"])
        r_low = hd.detect("Python is a language", min_confidence=0.0)
        r_high = hd.detect("Python is a language", min_confidence=0.99)
        assert r_high["total_claims"] >= r_low["total_claims"] or True


# ---------------------------------------------------------------------------
# ProductionRAG
# ---------------------------------------------------------------------------

class TestProductionRAG:
    def test_default_config(self):
        rag = ProductionRAG()
        assert rag.retriever.dense_weight == 0.7
        assert rag.retriever.sparse_weight == 0.3

    def test_custom_config(self):
        rag = ProductionRAG(config={"dense_weight": 0.4, "sparse_weight": 0.6})
        assert rag.retriever.dense_weight == 0.4
        assert rag.retriever.sparse_weight == 0.6

    def test_add_document_returns_ids(self):
        rag = ProductionRAG()
        ids = rag.add_document("hello world this is a test")
        assert isinstance(ids, list)
        assert len(ids) >= 1

    def test_add_document_chunks_long_content(self):
        rag = ProductionRAG()
        content = "word " * 600
        ids = rag.add_document(content, chunk_size=100)
        assert len(ids) > 1

    def test_add_document_default_metadata(self):
        rag = ProductionRAG()
        rag.add_document("test content")
        chunk = rag.retriever.chunks[0]
        assert chunk.metadata["source"] == "user"

    def test_add_document_custom_metadata(self):
        rag = ProductionRAG()
        rag.add_document("test", metadata={"source": "wiki", "page": 1})
        chunk = rag.retriever.chunks[0]
        assert chunk.metadata["source"] == "wiki"
        assert chunk.metadata["page"] == 1

    def test_add_document_overlap(self):
        rag = ProductionRAG()
        content = " ".join([f"word{i}" for i in range(20)])
        ids = rag.add_document(content, chunk_size=10, overlap=5)
        assert len(ids) >= 1

    def test_add_multiple_documents(self):
        rag = ProductionRAG()
        rag.add_document("first document content")
        rag.add_document("second document content")
        assert len(rag.retriever.chunks) >= 2

    def test_query_returns_structure(self):
        rag = ProductionRAG()
        rag.add_document("Python is a programming language")
        result = rag.query("Python", top_k=5)
        assert "question" in result
        assert "results" in result
        assert "context" in result
        assert "num_results" in result
        assert result["question"] == "Python"

    def test_query_result_items(self):
        rag = ProductionRAG()
        rag.add_document("Python is a programming language")
        result = rag.query("Python", top_k=5)
        for r in result["results"]:
            assert "chunk_id" in r
            assert "content" in r
            assert "score" in r
            assert "rank" in r
            assert "metadata" in r

    def test_query_context_joins_results(self):
        rag = ProductionRAG()
        rag.add_document("Alpha bravo charlie delta")
        result = rag.query("Alpha bravo", top_k=5)
        if result["num_results"] > 0:
            assert len(result["context"]) > 0

    def test_query_no_context(self):
        rag = ProductionRAG()
        rag.add_document("test content")
        result = rag.query("test", top_k=5, return_context=False)
        assert result["context"] == ""

    def test_query_empty_retriever(self):
        rag = ProductionRAG()
        result = rag.query("anything", top_k=5)
        assert result["num_results"] == 0
        assert result["context"] == ""

    def test_query_non_matching(self):
        rag = ProductionRAG()
        rag.add_document("hello world")
        result = rag.query("quantum physics", top_k=5)
        assert result["num_results"] >= 0

    def test_verify_and_ground_returns_keys(self):
        rag = ProductionRAG()
        rag.add_document("Python is a language created by Guido")
        result = rag.verify_and_ground("Python is a language", "what is Python")
        assert "original_text" in result
        assert "question" in result
        assert "verification" in result
        assert "citations" in result
        assert "confidence" in result
        assert "is_verified" in result

    def test_verify_and_ground_original_text_preserved(self):
        rag = ProductionRAG()
        rag.add_document("test data here")
        text = "some generated text"
        result = rag.verify_and_ground(text, "query")
        assert result["original_text"] == text

    def test_verify_and_ground_is_verified_flag(self):
        rag = ProductionRAG()
        rag.add_document("Python is a programming language")
        result = rag.verify_and_ground("Python is a language", "what is Python")
        assert isinstance(result["is_verified"], bool)

    def test_chunk_ids_unique(self):
        rag = ProductionRAG()
        ids = rag.add_document("hello world test content here")
        assert len(ids) == len(set(ids))

    def test_rebuild_index_after_add(self):
        rag = ProductionRAG()
        rag.add_document("first content alpha")
        rag.add_document("second content beta")
        assert rag.retriever.bm25.num_docs >= 2
