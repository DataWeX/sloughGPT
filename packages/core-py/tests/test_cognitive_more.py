"""Tests for grounding and RAG modules in the cognitive domain."""

from __future__ import annotations

import pytest
from domains.cognitive.grounding import (
    KnowledgeGrounding,
    CurriculumLearner,
    Document,
    RAGGrounder,
)
from domains.cognitive.rag import (
    TextChunk,
    BM25Indexer,
    CitationTracker,
    HallucinationDetector,
    HybridRetriever,
)


# ── KnowledgeGrounding ────────────────────────────────────────────────────


class TestKnowledgeGrounding:
    def test_add_fact_creates_nodes(self):
        kg = KnowledgeGrounding()
        kg.add_fact("socrates", "is_a", "philosopher")
        assert "socrates" in kg.nodes
        assert "philosopher" in kg.nodes

    def test_query_by_subject(self):
        kg = KnowledgeGrounding()
        kg.add_fact("socrates", "is_a", "philosopher")
        kg.add_fact("socrates", "lives_in", "athens")
        results = kg.query("socrates")
        assert len(results) == 2
        assert "philosopher" in results
        assert "athens" in results

    def test_query_by_subject_and_relation(self):
        kg = KnowledgeGrounding()
        kg.add_fact("socrates", "is_a", "philosopher")
        kg.add_fact("socrates", "lives_in", "athens")
        results = kg.query("socrates", "is_a")
        assert results == ["philosopher"]

    def test_query_unknown_subject(self):
        kg = KnowledgeGrounding()
        assert kg.query("nonexistent") == []

    def test_verify_statement_true(self):
        kg = KnowledgeGrounding()
        kg.add_fact("socrates", "is", "philosopher")
        result = kg.verify_statement("socrates is philosopher")
        assert result["verified"] is True
        assert result["grounded_in_knowledge"] is True

    def test_verify_statement_false(self):
        kg = KnowledgeGrounding()
        kg.add_fact("socrates", "is", "philosopher")
        result = kg.verify_statement("socrates is musician")
        assert result["verified"] is False

    def test_verify_statement_short_string(self):
        kg = KnowledgeGrounding()
        result = kg.verify_statement("hi")
        assert result["verified"] is False

    def test_get_context_for_prompt(self):
        kg = KnowledgeGrounding()
        kg.add_fact("socrates", "is_a", "philosopher")
        kg.add_fact("socrates", "lives_in", "athens")
        context = kg.get_context_for_prompt("socrates")
        assert "socrates" in context.lower()
        assert "philosopher" in context or "athens" in context

    def test_get_context_for_prompt_no_match(self):
        kg = KnowledgeGrounding()
        kg.add_fact("socrates", "is_a", "philosopher")
        context = kg.get_context_for_prompt("plato")
        assert context == ""


# ── CurriculumLearner ────────────────────────────────────────────────────


class TestCurriculumLearner:
    def test_add_example_categorizes_by_difficulty(self):
        cl = CurriculumLearner()
        cl.add_example("easy", 0.1)
        cl.add_example("hard", 0.9)
        assert len(cl.difficulty_levels[1]) == 1  # 0.1*10→level 1
        assert len(cl.difficulty_levels[9]) == 1  # 0.9*10→level 9

    def test_get_batch_bootstrapping_returns_easiest(self):
        cl = CurriculumLearner()
        for i in range(20):
            cl.add_example(f"ex_{i}", 0.05)
        batch = cl.get_batch(5)
        assert len(batch) == 5

    def test_get_batch_returns_requested_size(self):
        cl = CurriculumLearner()
        for i in range(100):
            cl.add_example(f"ex_{i}", 0.05)
        batch = cl.get_batch(10)
        assert len(batch) == 10

    def test_get_batch_returns_all_if_smaller_than_request(self):
        cl = CurriculumLearner()
        cl.add_example("only_one", 0.1)
        batch = cl.get_batch(100)
        assert len(batch) == 1

    def test_update_stage_mastery_on_high_performance(self):
        cl = CurriculumLearner()
        cl.update_stage(0.95)
        assert cl.stage == "mastery"
        assert cl.current_level == 1

    def test_update_stage_progressing_on_moderate_performance(self):
        cl = CurriculumLearner()
        cl.update_stage(0.8)
        assert cl.stage == "progressing"

    def test_update_stage_bootstrapping_on_low_performance(self):
        cl = CurriculumLearner()
        cl.update_stage(0.5)
        assert cl.stage == "bootstrapping"
        assert cl.current_level == 0

    def test_get_batch_progressing_includes_harder_examples(self):
        cl = CurriculumLearner()
        cl.add_example("easy", 0.05)
        cl.add_example("hard", 0.95)
        cl.update_stage(0.8)
        batch = cl.get_batch(10)
        assert len(batch) > 0


# ── RAGGrounder ───────────────────────────────────────────────────────────


class TestRAGGrounder:
    def test_add_text_creates_document(self):
        rag = RAGGrounder()
        doc_id = rag.add_text("Hello world", source="test")
        assert doc_id is not None
        assert doc_id in rag.documents

    def test_add_document_chunks_by_word_count(self):
        rag = RAGGrounder()
        doc = Document(id="d1", content="word " * 1000, source="test")
        rag.add_document(doc, chunk_size=200)
        assert len(rag.chunks) == 5
        assert rag.chunks[0].source == "test"

    def test_retrieve_returns_empty_for_empty_store(self):
        import asyncio
        rag = RAGGrounder()
        results = asyncio.run(rag.retrieve("query"))
        assert results == []

    def test_ground_response_returns_dict(self):
        rag = RAGGrounder()
        rag.add_text("The sky is blue during the day.", source="test")
        result = rag.ground_response("sky", "What color is the sky?")
        assert isinstance(result, dict)


# ── BM25Indexer ───────────────────────────────────────────────────────────


class TestBM25Indexer:
    def test_index_builds_inverted_index(self):
        indexer = BM25Indexer()
        chunks = [
            TextChunk(id="1", content="hello world", metadata={}),
            TextChunk(id="2", content="hello universe", metadata={}),
        ]
        indexer.index(chunks)
        assert "hello" in indexer.inverted_index
        assert "world" in indexer.inverted_index
        assert indexer.num_docs == 2

    def test_index_empty_chunks(self):
        indexer = BM25Indexer()
        indexer.index([])
        assert indexer.num_docs == 0
        assert indexer.avg_doc_length == 0.0

    def test_score_returns_results(self):
        indexer = BM25Indexer()
        chunks = [
            TextChunk(id="1", content="the cat sat on the mat", metadata={}),
            TextChunk(id="2", content="the dog ran in the park", metadata={}),
        ]
        indexer.index(chunks)
        scores = indexer.score("cat")
        assert len(scores) > 0
        # doc 0 should score higher for "cat"
        assert scores[0][0] == 0

    def test_score_empty_query(self):
        indexer = BM25Indexer()
        chunks = [
            TextChunk(id="1", content="hello world", metadata={}),
        ]
        indexer.index(chunks)
        scores = indexer.score("")
        assert scores == []

    def test_score_no_match_returns_empty(self):
        indexer = BM25Indexer()
        chunks = [
            TextChunk(id="1", content="hello world", metadata={}),
        ]
        indexer.index(chunks)
        scores = indexer.score("nonexistentwordxyz")
        assert scores == []

    def test_score_multi_word_query(self):
        indexer = BM25Indexer()
        chunks = [
            TextChunk(id="1", content="machine learning is fun", metadata={}),
            TextChunk(id="2", content="deep learning is cool", metadata={}),
            TextChunk(id="3", content="i like pizza", metadata={}),
        ]
        indexer.index(chunks)
        scores = indexer.score("machine learning")
        assert len(scores) >= 2
        # doc 0 and 1 should both score for "learning"
        doc_ids = [d for d, _ in scores]
        assert 0 in doc_ids
        assert 1 in doc_ids


# ── CitationTracker ───────────────────────────────────────────────────────


class TestCitationTracker:
    def test_extract_claims_finds_named_entities(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Einstein is a physicist")
        assert len(claims) > 0
        assert claims[0]["subject"] == "Einstein"

    def test_extract_claims_multiple_matches(self):
        ct = CitationTracker()
        claims = ct.extract_claims("Einstein is a physicist. Newton was a mathematician.")
        assert len(claims) >= 2

    def test_extract_claims_no_match(self):
        ct = CitationTracker()
        claims = ct.extract_claims("hello world")
        assert claims == []

    def test_cite_with_sources(self):
        ct = CitationTracker()
        claim = {"text": "Einstein is a physicist", "subject": "Einstein", "predicate": "is a physicist"}
        chunks = [TextChunk(id="c1", content="Einstein was a theoretical physicist", metadata={"source": "wiki"})]
        result = ct.cite(claim, chunks)
        assert result["supported"] is True
        assert len(result["sources"]) == 1
        assert result["sources"][0]["chunk_id"] == "c1"

    def test_cite_without_sources(self):
        ct = CitationTracker()
        claim = {"text": "Einstein is a physicist", "subject": "Einstein", "predicate": "is a physicist"}
        result = ct.cite(claim, [])
        assert result["supported"] is False
        assert result["sources"] == []

    def test_format_citations(self):
        ct = CitationTracker()
        ct.extract_claims("Einstein is a physicist")
        formatted = ct.format_citations()
        assert "[1]" in formatted
        assert "Einstein" in formatted

    def test_format_citations_no_sources_does_not_error(self):
        ct = CitationTracker()
        ct.extract_claims("Einstein is a physicist")
        formatted = ct.format_citations()
        assert "Einstein" in formatted


# ── HallucinationDetector ────────────────────────────────────────────────


class TestHallucinationDetector:
    def test_detect_no_claims_returns_high_confidence(self):
        retriever = HybridRetriever()
        detector = HallucinationDetector(retriever)
        result = detector.detect("hello world")
        assert result["total_claims"] == 0
        assert result["overall_confidence"] == 1.0

    def test_detect_returns_expected_structure(self):
        retriever = HybridRetriever()
        ct = TextChunk(id="c1", content="Einstein is a physicist", metadata={})
        retriever.add_chunk(ct)
        retriever.build_index()
        detector = HallucinationDetector(retriever)
        result = detector.detect("Einstein is a physicist")
        assert "total_claims" in result
        assert "hallucinations" in result
        assert "grounded_claims" in result
        assert "overall_confidence" in result


# ── TextChunk ─────────────────────────────────────────────────────────────


class TestTextChunk:
    def test_token_count_default(self):
        chunk = TextChunk(id="1", content="hello world", metadata={})
        assert chunk.token_count == 2

    def test_token_count_preserves_explicit(self):
        chunk = TextChunk(id="1", content="hello world", metadata={}, token_count=99)
        assert chunk.token_count == 99
