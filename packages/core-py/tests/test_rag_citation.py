"""Tests for domains.cognitive.rag — CitationTracker, TextChunk, BM25Indexer,
HybridRetriever, HallucinationDetector, ProductionRAG."""

import hashlib
import numpy as np
import pytest
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
# TextChunk — dataclass basics
# ---------------------------------------------------------------------------
class TestTextChunkFields:
    def test_basic_fields(self):
        tc = TextChunk(id="c1", content="hello world", metadata={"source": "doc"})
        assert tc.id == "c1"
        assert tc.content == "hello world"
        assert tc.metadata["source"] == "doc"

    def test_token_count_auto(self):
        tc = TextChunk(id="c1", content="one two three four", metadata={})
        assert tc.token_count == 4

    def test_token_count_explicit(self):
        tc = TextChunk(id="c1", content="a b c", metadata={}, token_count=10)
        assert tc.token_count == 10

    def test_embedding_default_none(self):
        tc = TextChunk(id="c1", content="x", metadata={})
        assert tc.embedding is None

    def test_bm25_score_default(self):
        tc = TextChunk(id="c1", content="x", metadata={})
        assert tc.bm25_score == 0.0

    def test_empty_content(self):
        tc = TextChunk(id="c1", content="", metadata={})
        assert tc.token_count == 0

    def test_single_word(self):
        tc = TextChunk(id="c1", content="hello", metadata={})
        assert tc.token_count == 1

    def test_metadata_mutable(self):
        tc = TextChunk(id="c1", content="x", metadata={"k": 1})
        tc.metadata["k"] = 2
        assert tc.metadata["k"] == 2


# ---------------------------------------------------------------------------
# RetrievalResult — dataclass
# ---------------------------------------------------------------------------
class TestRetrievalResult:
    def test_fields(self):
        chunk = TextChunk(id="c1", content="x", metadata={})
        rr = RetrievalResult(chunk=chunk, dense_score=0.8, sparse_score=0.6,
                             combined_score=0.7, rank=1)
        assert rr.chunk.id == "c1"
        assert rr.dense_score == 0.8
        assert rr.sparse_score == 0.6
        assert rr.combined_score == 0.7
        assert rr.rank == 1


# ---------------------------------------------------------------------------
# BM25Indexer — indexing and scoring
# ---------------------------------------------------------------------------
class TestBM25Indexer:
    def test_index_single_doc(self):
        bm25 = BM25Indexer()
        chunks = [TextChunk(id="c1", content="python is great", metadata={})]
        bm25.index(chunks)
        assert bm25.num_docs == 1
        assert bm25.avg_doc_length > 0

    def test_index_multiple_docs(self):
        bm25 = BM25Indexer()
        chunks = [
            TextChunk(id="c1", content="python is a language", metadata={}),
            TextChunk(id="c2", content="java is also a language", metadata={}),
            TextChunk(id="c3", content="rust is systems programming", metadata={}),
        ]
        bm25.index(chunks)
        assert bm25.num_docs == 3

    def test_score_finds_matching_doc(self):
        bm25 = BM25Indexer()
        chunks = [
            TextChunk(id="c1", content="python is a programming language", metadata={}),
            TextChunk(id="c2", content="java is another language", metadata={}),
        ]
        bm25.index(chunks)
        results = bm25.score("python")
        assert len(results) >= 1
        doc_id, score = results[0]
        assert doc_id == 0  # first doc matches best
        assert score > 0

    def test_score_no_match(self):
        bm25 = BM25Indexer()
        chunks = [TextChunk(id="c1", content="hello world", metadata={})]
        bm25.index(chunks)
        results = bm25.score("quantum physics")
        assert len(results) == 0

    def test_score_empty_index(self):
        bm25 = BM25Indexer()
        results = bm25.score("query")
        assert results == []

    def test_tokenization_lowercases(self):
        bm25 = BM25Indexer()
        tokens = bm25._tokenize("Hello World FOO")
        assert tokens == ["hello", "world", "foo"]

    def test_tokenization_splits_on_punctuation(self):
        bm25 = BM25Indexer()
        tokens = bm25._tokenize("hello,world;foo")
        assert "hello" in tokens
        assert "world" in tokens
        assert "foo" in tokens

    def test_score_relevance_ordering(self):
        bm25 = BM25Indexer()
        chunks = [
            TextChunk(id="c1", content="python programming language", metadata={}),
            TextChunk(id="c2", content="cats and dogs are animals", metadata={}),
            TextChunk(id="c3", content="python data science", metadata={}),
        ]
        bm25.index(chunks)
        results = bm25.score("python")
        # Both python docs should score > 0
        assert len(results) >= 2
        scores = [s for _, s in results]
        assert all(s > 0 for s in scores)

    def test_score_multiple_query_tokens(self):
        bm25 = BM25Indexer()
        chunks = [
            TextChunk(id="c1", content="python programming language", metadata={}),
            TextChunk(id="c2", content="java programming language", metadata={}),
        ]
        bm25.index(chunks)
        results = bm25.score("python programming")
        # First doc should score higher (has both terms)
        scores_dict = {doc_id: score for doc_id, score in results}
        assert scores_dict.get(0, 0) > scores_dict.get(1, 0)

    def test_avg_doc_length(self):
        bm25 = BM25Indexer()
        chunks = [
            TextChunk(id="c1", content="a b c", metadata={}),
            TextChunk(id="c2", content="d e f g h", metadata={}),
        ]
        bm25.index(chunks)
        assert bm25.avg_doc_length == 4.0

    def test_inverted_index_populated(self):
        bm25 = BM25Indexer()
        chunks = [TextChunk(id="c1", content="hello world hello", metadata={})]
        bm25.index(chunks)
        assert "hello" in bm25.inverted_index
        assert len(bm25.inverted_index["hello"]) == 2


# ---------------------------------------------------------------------------
# HybridRetriever — dense + sparse retrieval
# ---------------------------------------------------------------------------
class TestHybridRetriever:
    def test_add_and_build(self):
        hr = HybridRetriever()
        hr.add_chunk(TextChunk(id="c1", content="python programming", metadata={}))
        hr.add_chunk(TextChunk(id="c2", content="rust systems", metadata={}))
        hr.build_index()
        assert len(hr.chunks) == 2

    def test_dense_search(self):
        hr = HybridRetriever()
        hr.add_chunk(TextChunk(id="c1", content="python is great", metadata={}))
        hr.add_chunk(TextChunk(id="c2", content="rust is fast", metadata={}))
        hr.build_index()
        results = hr._dense_search("python", top_k=2)
        assert len(results) >= 1

    def test_sparse_search(self):
        hr = HybridRetriever()
        hr.add_chunk(TextChunk(id="c1", content="python programming", metadata={}))
        hr.add_chunk(TextChunk(id="c2", content="rust systems", metadata={}))
        hr.build_index()
        results = hr._sparse_search("python", top_k=2)
        assert len(results) >= 1

    def test_retrieve_hybrid(self):
        hr = HybridRetriever()
        hr.add_chunk(TextChunk(id="c1", content="python is a language", metadata={"source": "doc1"}))
        hr.add_chunk(TextChunk(id="c2", content="rust is fast", metadata={"source": "doc2"}))
        hr.build_index()
        results = hr.retrieve("python language", top_k=2)
        assert len(results) >= 1
        assert isinstance(results[0], RetrievalResult)

    def test_retrieve_top_k(self):
        hr = HybridRetriever()
        for i in range(10):
            hr.add_chunk(TextChunk(id=f"c{i}", content=f"doc number {i} about python", metadata={}))
        hr.build_index()
        results = hr.retrieve("python", top_k=3)
        assert len(results) <= 3

    def test_retrieve_min_score(self):
        hr = HybridRetriever()
        hr.add_chunk(TextChunk(id="c1", content="python", metadata={}))
        hr.build_index()
        results = hr.retrieve("python", top_k=5, min_score=0.99)
        # With high min_score, few or no results
        assert isinstance(results, list)

    def test_retrieve_empty_index(self):
        hr = HybridRetriever()
        results = hr.retrieve("anything", top_k=5)
        assert results == []

    def test_embedding_caching(self):
        hr = HybridRetriever()
        e1 = hr._get_embedding("test text")
        e2 = hr._get_embedding("test text")
        np.testing.assert_array_equal(e1, e2)

    def test_embedding_normalized(self):
        hr = HybridRetriever()
        emb = hr._get_embedding("normalize me")
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-6

    def test_dense_weight_parameter(self):
        hr = HybridRetriever(dense_weight=0.5, sparse_weight=0.5)
        assert hr.dense_weight == 0.5
        assert hr.sparse_weight == 0.5

    def test_rerank_diversity(self):
        hr = HybridRetriever(use_rerank=True)
        hr.add_chunk(TextChunk(id="c1", content="python programming language features", metadata={}))
        hr.add_chunk(TextChunk(id="c2", content="python programming language basics", metadata={}))
        hr.add_chunk(TextChunk(id="c3", content="rust systems programming", metadata={}))
        hr.build_index()
        results = hr.retrieve("python programming", top_k=3)
        assert len(results) >= 1

    def test_no_rerank(self):
        hr = HybridRetriever(use_rerank=False)
        hr.add_chunk(TextChunk(id="c1", content="python programming", metadata={}))
        hr.add_chunk(TextChunk(id="c2", content="rust systems", metadata={}))
        hr.build_index()
        results = hr.retrieve("python", top_k=2)
        assert len(results) >= 1

    def test_retrieve_scores_normalized(self):
        hr = HybridRetriever()
        hr.add_chunk(TextChunk(id="c1", content="python is great", metadata={}))
        hr.add_chunk(TextChunk(id="c2", content="rust is fast", metadata={}))
        hr.build_index()
        results = hr.retrieve("python", top_k=2)
        for r in results:
            assert isinstance(r.dense_score, (int, float))
            assert isinstance(r.sparse_score, (int, float))
            assert r.combined_score >= 0


# ---------------------------------------------------------------------------
# CitationTracker — claim extraction and citation
# ---------------------------------------------------------------------------
class TestCitationTrackerExtractClaims:
    def test_simple_claim(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python is a programming language.")
        assert len(claims) >= 1
        assert claims[0]["subject"] == "Python"
        assert "programming language" in claims[0]["predicate"]

    def test_multiple_claims(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python is great. Java is popular.")
        assert len(claims) == 2

    def test_no_claims(self):
        ct = CitationTracker()
        claims = ct.extract_claims("hello world")
        assert len(claims) == 0

    def test_was_pattern(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python was created in 1991.")
        assert len(claims) >= 1
        assert claims[0]["subject"] == "Python"

    def test_can_pattern(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python can be used for web development.")
        assert len(claims) >= 1

    def test_has_pattern(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python has a large ecosystem.")
        assert len(claims) >= 1

    def test_claim_text_field(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python is a language.")
        assert claims[0]["text"] == "Python is a language."

    def test_claim_start_end_offsets(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python is a language. Java is also one.")
        if len(claims) >= 2:
            assert claims[0]["start"] < claims[0]["end"]
            assert claims[1]["start"] > claims[0]["start"]

    def test_multi_word_subject(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Machine Learning is a field of AI.")
        assert len(claims) >= 1
        assert claims[0]["subject"] == "Machine Learning"

    def test_empty_text(self):
        ct = CitationTracker()
        claims = ct.extract_claims("")
        assert len(claims) == 0

    def test_periods_only(self):
        ct = CitationTracker()
        claims = ct.extract_claims("...")
        assert len(claims) == 0

    def test_store_claims(self):
        ct = CitationTracker()
        ct.extract_claims("Python is a language.")
        assert len(ct.claims) >= 1

    def test_predicate_trailing_dot_stripped(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python is great.")
        assert not claims[0]["predicate"].endswith(".")

    def test_no_cross_sentence_capture(self):
        ct = CitationTracker()
        text = "Python is amazing. It runs fast."
        claims = ct.extract_claims(text)
        # "It runs fast" should not be a claim about "Python"
        subjects = [c["subject"] for c in claims]
        assert "It" not in subjects


class TestCitationTrackerCite:
    def test_cite_single_source(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python is a language.")
        chunk = TextChunk(id="c1", content="Python source", metadata={"source": "doc"})
        cited = ct.cite(claims[0], [chunk])
        assert cited["supported"] is True
        assert len(cited["sources"]) == 1

    def test_cite_no_sources(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python is a language.")
        cited = ct.cite(claims[0], [])
        assert cited["supported"] is False
        assert len(cited["sources"]) == 0

    def test_cite_multiple_sources(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python is a language.")
        chunks = [
            TextChunk(id="c1", content="Python doc 1", metadata={"source": "d1"}),
            TextChunk(id="c2", content="Python doc 2", metadata={"source": "d2"}),
            TextChunk(id="c3", content="Python doc 3", metadata={"source": "d3"}),
            TextChunk(id="c4", content="Python doc 4", metadata={"source": "d4"}),
        ]
        cited = ct.cite(claims[0], chunks)
        assert len(cited["sources"]) == 3  # max 3

    def test_cite_preserves_claim_fields(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python is a language.")
        chunk = TextChunk(id="c1", content="src", metadata={"source": "d"})
        cited = ct.cite(claims[0], [chunk])
        assert "subject" in cited
        assert "predicate" in cited
        assert "text" in cited

    def test_cite_source_truncation(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python is a language.")
        long_content = "x" * 500
        chunk = TextChunk(id="c1", content=long_content, metadata={"source": "d"})
        cited = ct.cite(claims[0], [chunk])
        assert len(cited["sources"][0]["content"]) <= 200


class TestCitationTrackerFormatCitations:
    def test_format_single(self):
        ct = CitationTracker()
        ct.extract_claims("Python is a language.")
        output = ct.format_citations()
        assert "[1]" in output
        assert "Python" in output

    def test_format_multiple(self):
        ct = CitationTracker()
        ct.extract_claims("Python is a language. Java is popular.")
        output = ct.format_citations()
        assert "[1]" in output
        assert "[2]" in output

    def test_format_with_sources(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python is a language.")
        chunk = TextChunk(id="c1", content="src", metadata={"source": "mydoc"})
        cited = ct.cite(claims[0], [chunk])
        assert cited["sources"][0]["metadata"]["source"] == "mydoc"

    def test_format_no_claims(self):
        ct = CitationTracker()
        output = ct.format_citations()
        assert output == ""

    def test_format_unknown_source(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Python is a language.")
        chunk = TextChunk(id="c1", content="src", metadata={})
        cited = ct.cite(claims[0], [chunk])
        assert cited["sources"][0]["metadata"] == {}


# ---------------------------------------------------------------------------
# HallucinationDetector — detection logic
# ---------------------------------------------------------------------------
class TestHallucinationDetector:
    def _make_detector(self, chunks):
        hr = HybridRetriever()
        for c in chunks:
            hr.add_chunk(c)
        hr.build_index()
        return HallucinationDetector(hr)

    def test_no_claims_high_confidence(self):
        hd = self._make_detector([])
        result = hd.detect("hello world no claims here.")
        assert result["total_claims"] == 0
        assert result["overall_confidence"] == 1.0

    def test_grounded_claim(self):
        chunks = [TextChunk(id="c1", content="Python is a programming language", metadata={"source": "wiki"})]
        hd = self._make_detector(chunks)
        result = hd.detect("Python is a programming language.")
        assert result["total_claims"] >= 1

    def test_hallucination_detected(self):
        chunks = [TextChunk(id="c1", content="cats are animals", metadata={"source": "wiki"})]
        hd = self._make_detector(chunks)
        result = hd.detect("Python is a quantum computing framework.")
        assert result["hallucination_rate"] > 0

    def test_result_structure(self):
        hd = self._make_detector([])
        result = hd.detect("Python is great.")
        assert "text" in result
        assert "total_claims" in result
        assert "grounded_claims" in result
        assert "hallucinations" in result
        assert "overall_confidence" in result
        assert "hallucination_rate" in result
        assert "formatted_citations" in result

    def test_hallucination_rate_calculation(self):
        hd = self._make_detector([])
        result = hd.detect("Python is a language. Java is a language.")
        if result["total_claims"] > 0:
            assert 0 <= result["hallucination_rate"] <= 1.0

    def test_overall_confidence_range(self):
        hr = HybridRetriever()
        hr.add_chunk(TextChunk(id="c1", content="Python programming", metadata={}))
        hr.build_index()
        hd = HallucinationDetector(hr)
        result = hd.detect("Python is a language.")
        assert 0 <= result["overall_confidence"] <= 1.0


# ---------------------------------------------------------------------------
# ProductionRAG — end-to-end workflow
# ---------------------------------------------------------------------------
class TestProductionRAG:
    def test_init_default(self):
        rag = ProductionRAG()
        assert rag.retriever is not None
        assert rag.hallucination_detector is not None

    def test_init_custom_config(self):
        rag = ProductionRAG({"dense_weight": 0.5, "sparse_weight": 0.5})
        assert rag.retriever.dense_weight == 0.5

    def test_add_document(self):
        rag = ProductionRAG()
        ids = rag.add_document("Python is a programming language used for many things.")
        assert len(ids) >= 1

    def test_add_document_custom_metadata(self):
        rag = ProductionRAG()
        ids = rag.add_document("test content", metadata={"source": "custom"})
        assert len(ids) >= 1

    def test_add_document_chunking(self):
        rag = ProductionRAG()
        long_text = "word " * 1000
        ids = rag.add_document(long_text, chunk_size=100, overlap=10)
        assert len(ids) > 1

    def test_query_basic(self):
        rag = ProductionRAG()
        rag.add_document("Python is a programming language for data science.")
        result = rag.query("What is Python?")
        assert "question" in result
        assert "results" in result
        assert "context" in result
        assert "num_results" in result

    def test_query_no_results(self):
        rag = ProductionRAG()
        result = rag.query("anything")
        assert result["num_results"] == 0

    def test_query_with_context(self):
        rag = ProductionRAG()
        rag.add_document("Python is great for ML.")
        result = rag.query("Python ML", return_context=True)
        assert isinstance(result["context"], str)

    def test_query_without_context(self):
        rag = ProductionRAG()
        rag.add_document("Python is great for ML.")
        result = rag.query("Python ML", return_context=False)
        assert result["context"] == ""

    def test_query_top_k(self):
        rag = ProductionRAG()
        for i in range(20):
            rag.add_document(f"Document {i} about python programming.")
        result = rag.query("python", top_k=3)
        assert len(result["results"]) <= 3

    def test_verify_and_ground(self):
        rag = ProductionRAG()
        rag.add_document("Python is a programming language for data science.")
        result = rag.verify_and_ground("Python is a programming language.", "What is Python?")
        assert "original_text" in result
        assert "verification" in result
        assert "citations" in result
        assert "confidence" in result
        assert "is_verified" in result

    def test_verify_and_ground_structure(self):
        rag = ProductionRAG()
        rag.add_document("test content for verification.")
        result = rag.verify_and_ground("test content for verification.", "test?")
        assert isinstance(result["is_verified"], bool)
        assert isinstance(result["confidence"], float)

    def test_add_multiple_documents(self):
        rag = ProductionRAG()
        rag.add_document("Python is a language.")
        rag.add_document("Rust is fast.")
        rag.add_document("Go is simple.")
        result = rag.query("programming languages", top_k=5)
        assert result["num_results"] >= 1

    def test_query_results_structure(self):
        rag = ProductionRAG()
        rag.add_document("Python is a language.")
        result = rag.query("Python")
        if result["results"]:
            r = result["results"][0]
            assert "chunk_id" in r
            assert "content" in r
            assert "score" in r
            assert "rank" in r
            assert "metadata" in r

    def test_add_document_short(self):
        rag = ProductionRAG()
        ids = rag.add_document("hi")
        assert len(ids) >= 1

    def test_add_document_overlap(self):
        rag = ProductionRAG()
        text = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
        ids = rag.add_document(text, chunk_size=5, overlap=2)
        assert len(ids) > 1


# ---------------------------------------------------------------------------
# BM25Indexer — edge cases
# ---------------------------------------------------------------------------
class TestBM25EdgeCases:
    def test_single_char_tokens(self):
        bm25 = BM25Indexer()
        chunks = [TextChunk(id="c1", content="a b c", metadata={})]
        bm25.index(chunks)
        results = bm25.score("a")
        assert len(results) >= 1

    def test_repeated_words(self):
        bm25 = BM25Indexer()
        chunks = [TextChunk(id="c1", content="python python python", metadata={})]
        bm25.index(chunks)
        results = bm25.score("python")
        assert len(results) == 1
        assert results[0][1] > 0

    def test_special_characters(self):
        bm25 = BM25Indexer()
        chunks = [TextChunk(id="c1", content="C++ is great!", metadata={})]
        bm25.index(chunks)
        results = bm25.score("great")
        assert len(results) >= 1

    def test_numbers_in_text(self):
        bm25 = BM25Indexer()
        chunks = [TextChunk(id="c1", content="version 2.0 released", metadata={})]
        bm25.index(chunks)
        results = bm25.score("2.0")
        assert len(results) >= 1

    def test_unicode_text(self):
        bm25 = BM25Indexer()
        chunks = [TextChunk(id="c1", content="cafe resume naive", metadata={})]
        bm25.index(chunks)
        results = bm25.score("cafe")
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# HybridRetriever — edge cases
# ---------------------------------------------------------------------------
class TestHybridRetrieverEdgeCases:
    def test_single_chunk(self):
        hr = HybridRetriever()
        hr.add_chunk(TextChunk(id="c1", content="only chunk", metadata={}))
        hr.build_index()
        results = hr.retrieve("only", top_k=5)
        assert len(results) == 1

    def test_many_chunks(self):
        hr = HybridRetriever()
        for i in range(50):
            hr.add_chunk(TextChunk(id=f"c{i}", content=f"chunk number {i} about topic {i % 5}", metadata={}))
        hr.build_index()
        results = hr.retrieve("topic", top_k=10)
        assert len(results) <= 10

    def test_embedding_deterministic(self):
        hr = HybridRetriever()
        e1 = hr._get_embedding("deterministic test")
        e2 = hr._get_embedding("deterministic test")
        np.testing.assert_array_equal(e1, e2)

    def test_embedding_different_texts(self):
        hr = HybridRetriever()
        e1 = hr._get_embedding("text one")
        e2 = hr._get_embedding("text two")
        assert not np.array_equal(e1, e2)

    def test_build_index_updates_bm25(self):
        hr = HybridRetriever()
        hr.add_chunk(TextChunk(id="c1", content="hello", metadata={}))
        hr.build_index()
        assert hr.bm25.num_docs == 1
        hr.add_chunk(TextChunk(id="c2", content="world", metadata={}))
        hr.build_index()
        assert hr.bm25.num_docs == 2
