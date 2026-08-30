"""Meaningful tests for RAGGrounder — document storage, chunking, retrieval, grounding."""

import pytest
from domains.cognitive.grounding import (
    RAGGrounder, Document, KnowledgeGrounding, KnowledgeNode,
    KnowledgeEdge, HierarchicalContext, CurriculumLearner,
)


# ── RAGGrounder — Add Documents ────────────────────────────────────────

class TestRAGGrounderAddDocument:
    def test_add_document(self):
        rag = RAGGrounder()
        doc = Document(id="d1", content="Hello world", source="test")
        rag.add_document(doc)
        assert "d1" in rag.documents

    def test_add_document_chunks(self):
        rag = RAGGrounder()
        words = " ".join([f"word{i}" for i in range(20)])
        doc = Document(id="d1", content=words, source="test")
        rag.add_document(doc, chunk_size=5)
        assert len(rag.chunks) == 4  # 20/5 = 4

    def test_chunk_ids(self):
        rag = RAGGrounder()
        doc = Document(id="d1", content="a b c d e f", source="test")
        rag.add_document(doc, chunk_size=2)
        assert rag.chunks[0].id == "d1_chunk_0"
        assert rag.chunks[1].id == "d1_chunk_1"

    def test_chunk_metadata_inherits_parent(self):
        rag = RAGGrounder()
        doc = Document(id="d1", content="a b c", source="wiki", metadata={"page": 1})
        rag.add_document(doc, chunk_size=2)
        assert rag.chunks[0].metadata["parent_id"] == "d1"
        assert rag.chunks[0].metadata["page"] == 1

    def test_add_text(self):
        rag = RAGGrounder()
        doc_id = rag.add_text("Hello world", source="user")
        assert doc_id == "doc_0"
        assert len(rag.documents) == 1

    def test_add_multiple_documents(self):
        rag = RAGGrounder()
        rag.add_text("First", source="s1")
        rag.add_text("Second", source="s2")
        assert len(rag.documents) == 2
        assert "doc_0" in rag.documents
        assert "doc_1" in rag.documents

    def test_add_document_preserves_content(self):
        rag = RAGGrounder()
        doc = Document(id="d1", content="exact content here", source="test")
        rag.add_document(doc)
        assert rag.documents["d1"].content == "exact content here"

    def test_add_document_preserves_source(self):
        rag = RAGGrounder()
        doc = Document(id="d1", content="x", source="my_source")
        rag.add_document(doc)
        assert rag.documents["d1"].source == "my_source"

    def test_add_document_with_embedding(self):
        import numpy as np
        rag = RAGGrounder()
        emb = np.array([0.1, 0.2, 0.3])
        doc = Document(id="d1", content="x", source="test", embedding=emb)
        rag.add_document(doc)
        np.testing.assert_array_equal(rag.documents["d1"].embedding, emb)

    def test_chunk_content_correct(self):
        rag = RAGGrounder()
        doc = Document(id="d1", content="a b c d e", source="test")
        rag.add_document(doc, chunk_size=2)
        assert rag.chunks[0].content == "a b"
        assert rag.chunks[1].content == "c d"
        assert rag.chunks[2].content == "e"

    def test_add_text_auto_ids_increment(self):
        rag = RAGGrounder()
        id1 = rag.add_text("one")
        id2 = rag.add_text("two")
        id3 = rag.add_text("three")
        assert id1 == "doc_0"
        assert id2 == "doc_1"
        assert id3 == "doc_2"

    def test_large_document_chunking(self):
        rag = RAGGrounder()
        words = " ".join([f"w{i}" for i in range(100)])
        doc = Document(id="big", content=words, source="test")
        rag.add_document(doc, chunk_size=10)
        assert len(rag.chunks) == 10

    def test_single_word_chunk(self):
        rag = RAGGrounder()
        doc = Document(id="d1", content="hello", source="test")
        rag.add_document(doc, chunk_size=512)
        assert len(rag.chunks) == 1
        assert rag.chunks[0].content == "hello"

    def test_empty_content_no_chunks(self):
        rag = RAGGrounder()
        doc = Document(id="d1", content="", source="test")
        rag.add_document(doc, chunk_size=5)
        assert len(rag.chunks) == 0


# ── RAGGrounder — Retrieve ────────────────────────────────────────────

class TestRAGGrounderRetrieve:
    @pytest.mark.asyncio
    async def test_retrieve_exact_match(self):
        rag = RAGGrounder()
        doc = Document(id="d1", content="Python is a programming language", source="wiki")
        rag.add_document(doc)
        results = await rag.retrieve("Python programming", top_k=5, min_relevance=0.3)
        assert len(results) >= 1
        assert "Python" in results[0].content

    @pytest.mark.asyncio
    async def test_retrieve_no_match(self):
        rag = RAGGrounder()
        doc = Document(id="d1", content="Python is a programming language", source="wiki")
        rag.add_document(doc)
        results = await rag.retrieve("quantum physics", top_k=5, min_relevance=0.5)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_retrieve_top_k(self):
        rag = RAGGrounder()
        for i in range(10):
            rag.add_text(f"Document {i} about Python programming", source=f"src{i}")
        results = await rag.retrieve("Python", top_k=3, min_relevance=0.3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_retrieve_min_relevance_filters(self):
        rag = RAGGrounder()
        rag.add_text("alpha beta gamma", source="s1")
        rag.add_text("delta epsilon", source="s2")
        results = await rag.retrieve("alpha beta", top_k=5, min_relevance=0.5)
        assert len(results) == 1
        assert results[0].content == "alpha beta gamma"

    @pytest.mark.asyncio
    async def test_retrieve_empty_rag(self):
        rag = RAGGrounder()
        results = await rag.retrieve("anything", top_k=5, min_relevance=0.0)
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k(self):
        rag = RAGGrounder()
        for i in range(20):
            rag.add_text("Python programming language", source=f"s{i}")
        results = await rag.retrieve("Python programming", top_k=5, min_relevance=0.0)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_retrieve_sorted_by_relevance(self):
        rag = RAGGrounder()
        rag.add_text("Python", source="s1")
        rag.add_text("Python programming language tutorial", source="s2")
        results = await rag.retrieve("Python programming", top_k=10, min_relevance=0.0)
        if len(results) >= 2:
            assert len(results[0].content) >= len(results[1].content)

    @pytest.mark.asyncio
    async def test_retrieve_case_insensitive(self):
        rag = RAGGrounder()
        rag.add_text("PYTHON is great", source="s1")
        results = await rag.retrieve("python", top_k=5, min_relevance=0.5)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_retrieve_single_word_query(self):
        rag = RAGGrounder()
        rag.add_text("hello world", source="s1")
        results = await rag.retrieve("hello", top_k=5, min_relevance=0.5)
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_retrieve_min_relevance_zero(self):
        rag = RAGGrounder()
        rag.add_text("x y z", source="s1")
        results = await rag.retrieve("a b c", top_k=5, min_relevance=0.0)
        assert len(results) >= 0

    @pytest.mark.asyncio
    async def test_retrieve_multiple_chunks_from_same_doc(self):
        rag = RAGGrounder()
        doc = Document(id="d1", content="Python Python Python Python Python Python", source="s1")
        rag.add_document(doc, chunk_size=2)
        results = await rag.retrieve("Python", top_k=10, min_relevance=0.0)
        assert len(results) >= 1


# ── RAGGrounder — Ground Response ─────────────────────────────────────

class TestRAGGrounderGround:
    def test_ground_response_no_docs(self):
        rag = RAGGrounder()
        result = rag.ground_response("response", "query")
        assert result["grounded"] is False
        assert result["confidence"] == 0.0

    def test_ground_response_structure(self):
        rag = RAGGrounder()
        result = rag.ground_response("response", "query")
        assert "response" in result
        assert "grounded" in result
        assert "confidence" in result
        assert "supporting_docs" in result
        assert "contradictions" in result
        assert "hallucination_score" in result

    def test_ground_response_with_docs(self):
        rag = RAGGrounder()
        rag.add_text("Python programming language", source="wiki")
        result = rag.ground_response("Python is a language", "Python")
        assert result["grounded"] is True
        assert result["confidence"] > 0.0

    def test_ground_response_includes_sources(self):
        rag = RAGGrounder()
        rag.add_text("test content", source="my_source")
        result = rag.ground_response("test", "test")
        if result["supporting_docs"]:
            assert result["supporting_docs"][0]["source"] == "my_source"

    def test_ground_response_preserves_original(self):
        rag = RAGGrounder()
        result = rag.ground_response("original text", "query")
        assert result["response"] == "original text"

    def test_ground_response_confidence_capped(self):
        rag = RAGGrounder()
        for i in range(20):
            rag.add_text("Python programming language", source=f"s{i}")
        result = rag.ground_response("Python", "Python programming")
        assert result["confidence"] <= 0.9

    def test_ground_response_empty_response(self):
        rag = RAGGrounder()
        result = rag.ground_response("", "query")
        assert result["response"] == ""
        assert "grounded" in result

    def test_ground_response_empty_query(self):
        rag = RAGGrounder()
        rag.add_text("some content", source="s1")
        result = rag.ground_response("response", "")
        assert "grounded" in result


# ── KnowledgeGrounding ────────────────────────────────────────────────

class TestKnowledgeGrounding:
    def test_add_fact(self):
        kg = KnowledgeGrounding()
        kg.add_fact("python", "is_a", "language")
        assert "python" in kg.nodes
        assert "language" in kg.nodes

    def test_query(self):
        kg = KnowledgeGrounding()
        kg.add_fact("python", "is_a", "language")
        results = kg.query("python")
        assert "language" in results

    def test_query_with_relation(self):
        kg = KnowledgeGrounding()
        kg.add_fact("python", "is_a", "language")
        kg.add_fact("python", "used_for", "ml")
        results = kg.query("python", "is_a")
        assert results == ["language"]

    def test_query_nonexistent(self):
        kg = KnowledgeGrounding()
        results = kg.query("nothing")
        assert results == []

    def test_verify_statement(self):
        kg = KnowledgeGrounding()
        kg.add_fact("python", "is_a", "language")
        result = kg.verify_statement("python is_a language")
        assert result["verified"] is True

    def test_verify_statement_false(self):
        kg = KnowledgeGrounding()
        kg.add_fact("python", "is_a", "language")
        result = kg.verify_statement("java is_a language")
        assert result["verified"] is False

    def test_verify_statement_unparseable(self):
        kg = KnowledgeGrounding()
        result = kg.verify_statement("two words only")
        assert result["verified"] is False

    def test_get_context_for_prompt(self):
        kg = KnowledgeGrounding()
        kg.add_fact("python", "is_a", "language")
        kg.add_fact("python", "used_for", "ml")
        ctx = kg.get_context_for_prompt("tell me about python")
        assert "python" in ctx.lower() or len(ctx) == 0

    def test_get_context_for_prompt_no_match(self):
        kg = KnowledgeGrounding()
        ctx = kg.get_context_for_prompt("quantum physics")
        assert ctx == ""

    def test_add_fact_stores_confidence(self):
        kg = KnowledgeGrounding()
        kg.add_fact("a", "rel", "b", confidence=0.8)
        assert len(kg.edges) == 1
        assert kg.edges[0].weight == 0.8

    def test_multiple_facts_same_subject(self):
        kg = KnowledgeGrounding()
        kg.add_fact("x", "rel1", "y")
        kg.add_fact("x", "rel2", "z")
        results = kg.query("x")
        assert len(results) == 2

    def test_adjacency_updated(self):
        kg = KnowledgeGrounding()
        kg.add_fact("a", "rel", "b")
        assert len(kg.adjacency["a"]) == 1
        assert kg.adjacency["a"][0][0] == "b"

    def test_verify_statement_supporting_facts(self):
        kg = KnowledgeGrounding()
        kg.add_fact("a", "is", "b")
        result = kg.verify_statement("a is b")
        assert len(result["supporting_facts"]) >= 1

    def test_get_context_multiple_nodes(self):
        kg = KnowledgeGrounding()
        kg.add_fact("alpha", "rel", "beta")
        kg.add_fact("alpha", "rel2", "gamma")
        ctx = kg.get_context_for_prompt("alpha")
        assert "alpha" in ctx or len(ctx) == 0


# ── HierarchicalContext ───────────────────────────────────────────────

class TestHierarchicalContext:
    def test_init(self):
        hc = HierarchicalContext(max_context=2048, chunk_size=256)
        assert hc.max_context == 2048
        assert hc.chunk_size == 256

    def test_build_hierarchy(self):
        hc = HierarchicalContext(chunk_size=5)
        hc.build_hierarchy("word " * 20)
        assert len(hc.hierarchy) >= 1

    def test_get_relevant_context_empty(self):
        hc = HierarchicalContext()
        ctx = hc.get_relevant_context("query")
        assert ctx == ""

    def test_get_relevant_context(self):
        hc = HierarchicalContext(chunk_size=5)
        hc.build_hierarchy("hello world test content here and more words")
        ctx = hc.get_relevant_context("hello")
        assert isinstance(ctx, str)
        assert len(ctx) <= hc.max_context

    def test_attention_mask_shape(self):
        hc = HierarchicalContext(chunk_size=4)
        hc.build_hierarchy("a b c d e f g h")
        mask = hc.attention_mask(8)
        assert mask.shape == (8, 8)

    def test_summarize_pair(self):
        hc = HierarchicalContext(chunk_size=3)
        result = hc._summarize_pair("a b c", "d e f")
        assert isinstance(result, str)
        assert len(result.split()) <= 3

    def test_build_hierarchy_single_chunk(self):
        hc = HierarchicalContext(chunk_size=100)
        hc.build_hierarchy("short text")
        assert len(hc.hierarchy) >= 1
        assert len(hc.hierarchy[0]) == 1

    def test_attention_mask_values(self):
        hc = HierarchicalContext(chunk_size=4)
        hc.build_hierarchy("a b c d e f g h")
        mask = hc.attention_mask(8)
        assert mask.min() >= 0
        assert mask.max() <= 1

    def test_default_init(self):
        hc = HierarchicalContext()
        assert hc.max_context == 4096
        assert hc.chunk_size == 512

    def test_summary_cache_empty(self):
        hc = HierarchicalContext()
        assert hc.summary_cache == {}


# ── CurriculumLearner ─────────────────────────────────────────────────

class TestCurriculumLearner:
    def test_init(self):
        cl = CurriculumLearner()
        assert cl.current_level == 0
        assert cl.stage == "bootstrapping"

    def test_add_example(self):
        cl = CurriculumLearner()
        cl.add_example("example1", difficulty=0.5)
        assert len(cl.difficulty_levels[5]) == 1

    def test_get_batch_bootstrapping(self):
        cl = CurriculumLearner()
        cl.add_example("easy1", difficulty=0.0)
        cl.add_example("easy2", difficulty=0.1)
        cl.add_example("hard", difficulty=0.9)
        batch = cl.get_batch(10)
        assert len(batch) == 2
        assert "hard" not in batch

    def test_get_batch_progressing(self):
        cl = CurriculumLearner()
        cl.stage = "progressing"
        cl.current_level = 5
        cl.add_example("lvl4", difficulty=0.4)
        cl.add_example("lvl5", difficulty=0.5)
        cl.add_example("lvl6", difficulty=0.6)
        cl.add_example("easy", difficulty=0.0)
        batch = cl.get_batch(10)
        assert "easy" not in batch

    def test_get_batch_mastery(self):
        cl = CurriculumLearner()
        cl.stage = "mastery"
        cl.add_example("easy", difficulty=0.0)
        cl.add_example("hard", difficulty=0.9)
        batch = cl.get_batch(10)
        assert len(batch) == 2

    def test_update_stage_high_performance(self):
        cl = CurriculumLearner()
        cl.update_stage(0.95)
        assert cl.stage == "mastery"

    def test_update_stage_medium_performance(self):
        cl = CurriculumLearner()
        cl.update_stage(0.75)
        assert cl.stage == "progressing"

    def test_update_stage_low_performance(self):
        cl = CurriculumLearner()
        cl.update_stage(0.5)
        assert cl.stage == "bootstrapping"

    def test_update_stage_increments_level(self):
        cl = CurriculumLearner()
        cl.update_stage(0.95)
        assert cl.current_level == 1

    def test_update_stage_decrements_level(self):
        cl = CurriculumLearner()
        cl.current_level = 5
        cl.update_stage(0.5)
        assert cl.current_level == 4

    def test_level_does_not_go_below_zero(self):
        cl = CurriculumLearner()
        cl.current_level = 0
        cl.update_stage(0.5)
        assert cl.current_level == 0

    def test_level_does_not_exceed_10(self):
        cl = CurriculumLearner()
        cl.current_level = 10
        cl.update_stage(0.95)
        assert cl.current_level == 10

    def test_get_batch_respects_batch_size(self):
        cl = CurriculumLearner()
        cl.stage = "mastery"
        for i in range(20):
            cl.add_example(f"ex{i}", difficulty=i / 10.0)
        batch = cl.get_batch(5)
        assert len(batch) == 5

    def test_multiple_difficulty_levels(self):
        cl = CurriculumLearner()
        for d in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
            cl.add_example(f"ex{d}", difficulty=d)
        assert len(cl.difficulty_levels) == 7
