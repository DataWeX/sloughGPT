"""Tests for cognitive grounding system."""

import pytest
import numpy as np
from domains.cognitive.grounding import (
    Document,
    RAGGrounder,
    HierarchicalContext,
    KnowledgeNode,
    KnowledgeEdge,
    KnowledgeGrounding,
    CurriculumLearner,
    GroundingOrchestrator,
)


class TestDocument:
    def test_default_metadata(self):
        d = Document(id="d1", content="hello", source="test")
        assert d.metadata == {}
        assert d.embedding is None

    def test_with_metadata(self):
        d = Document(id="d1", content="hello", source="test", metadata={"key": "val"})
        assert d.metadata["key"] == "val"


class TestRAGGrounder:
    def test_initial_state(self):
        rg = RAGGrounder()
        assert rg.documents == {}
        assert rg.chunks == []

    def test_add_text(self):
        rg = RAGGrounder()
        doc_id = rg.add_text("hello world", "user")
        assert doc_id == "doc_0"
        assert len(rg.documents) == 1
        assert len(rg.chunks) == 1

    def test_add_text_default_source(self):
        rg = RAGGrounder()
        rg.add_text("hello")
        assert rg.documents["doc_0"].source == "user"

    def test_add_document_chunks_long_text(self):
        rg = RAGGrounder()
        content = "word " * 600
        doc = Document(id="long", content=content, source="test")
        rg.add_document(doc, chunk_size=100)
        assert len(rg.chunks) == 6
        assert all(c.id.startswith("long_chunk_") for c in rg.chunks)

    def test_add_document_single_chunk(self):
        rg = RAGGrounder()
        doc = Document(id="short", content="hello world", source="test")
        rg.add_document(doc, chunk_size=512)
        assert len(rg.chunks) == 1

    async def test_retrieve_basic(self):
        rg = RAGGrounder()
        rg.add_text("the cat sat on the mat", "user")
        rg.add_text("dogs love to play fetch", "user")
        results = await rg.retrieve("cat mat", top_k=5, min_relevance=0.0)
        assert len(results) == 1
        assert "cat" in results[0].content

    async def test_retrieve_top_k(self):
        rg = RAGGrounder()
        for i in range(10):
            rg.add_text(f"keyword in document number {i}")
        results = await rg.retrieve("keyword", top_k=3, min_relevance=0.0)
        assert len(results) == 3

    async def test_retrieve_no_match(self):
        rg = RAGGrounder()
        rg.add_text("hello world")
        results = await rg.retrieve("xyzzy", top_k=5, min_relevance=0.0)
        assert results == []

    def test_ground_response_no_support(self):
        rg = RAGGrounder()
        g = rg.ground_response("some response text", "some query")
        assert g["grounded"] is False
        assert g["confidence"] == 0.0

    def test_ground_response_with_support(self):
        rg = RAGGrounder()
        rg.add_text("Paris is the capital of France")
        g = rg.ground_response("Paris is great", "Paris capital", include_sources=True)
        if not g["grounded"]:
            g = rg.ground_response("Paris is great", "Paris capital", include_sources=True)
        assert g["confidence"] > 0
        assert len(g["supporting_docs"]) > 0

    def test_multiple_documents(self):
        rg = RAGGrounder()
        rg.add_text("first doc", "src1")
        rg.add_text("second doc", "src2")
        assert len(rg.documents) == 2


class TestHierarchicalContext:
    def test_initial_state(self):
        hc = HierarchicalContext()
        assert hc.hierarchy == []
        assert hc.max_context == 4096
        assert hc.chunk_size == 512

    def test_build_hierarchy_single_chunk(self):
        hc = HierarchicalContext(chunk_size=100)
        hc.build_hierarchy("word " * 50)
        assert len(hc.hierarchy) == 1
        assert len(hc.hierarchy[0]) == 1

    def test_build_hierarchy_multiple_chunks(self):
        hc = HierarchicalContext(chunk_size=10)
        hc.build_hierarchy("word " * 30)
        assert len(hc.hierarchy[0]) == 3

    def test_build_hierarchy_multiple_levels(self):
        hc = HierarchicalContext(chunk_size=5)
        hc.build_hierarchy("word " * 40)
        assert len(hc.hierarchy) > 1
        assert len(hc.hierarchy[-1]) == 1

    def test_get_relevant_context_empty(self):
        hc = HierarchicalContext()
        assert hc.get_relevant_context("query") == ""

    def test_get_relevant_context_non_empty(self):
        hc = HierarchicalContext(chunk_size=100)
        hc.build_hierarchy("hello world " * 10)
        ctx = hc.get_relevant_context("hello")
        assert len(ctx) > 0
        assert len(ctx) <= hc.max_context

    def test_attention_mask_shape(self):
        hc = HierarchicalContext(chunk_size=10)
        hc.build_hierarchy("word " * 20)
        mask = hc.attention_mask(seq_len=20)
        assert mask.shape == (20, 20)
        assert mask.dtype == np.float64

    def test_summarize_pair(self):
        hc = HierarchicalContext(chunk_size=10)
        summary = hc._summarize_pair("a b c d e f g h i j", "k l m n o p q r s t")
        words = summary.split()
        assert len(words) <= 10


class TestKnowledgeNode:
    def test_defaults(self):
        kn = KnowledgeNode(id="k1", label="test", node_type="entity")
        assert kn.properties == {}

    def test_with_properties(self):
        kn = KnowledgeNode(id="k1", label="test", node_type="entity", properties={"color": "red"})
        assert kn.properties["color"] == "red"


class TestKnowledgeEdge:
    def test_default_weight(self):
        ke = KnowledgeEdge(source="a", target="b", relation="is_a")
        assert ke.weight == 1.0

    def test_custom_weight(self):
        ke = KnowledgeEdge(source="a", target="b", relation="is_a", weight=0.5)
        assert ke.weight == 0.5


class TestKnowledgeGrounding:
    def test_initial_state(self):
        kg = KnowledgeGrounding()
        assert kg.nodes == {}
        assert kg.edges == []

    def test_add_fact_creates_nodes(self):
        kg = KnowledgeGrounding()
        kg.add_fact("cat", "is_a", "animal")
        assert "cat" in kg.nodes
        assert "animal" in kg.nodes
        assert len(kg.edges) == 1

    def test_add_fact_multiple_edges(self):
        kg = KnowledgeGrounding()
        kg.add_fact("cat", "is_a", "animal")
        kg.add_fact("cat", "likes", "fish")
        assert len(kg.edges) == 2

    def test_query_all(self):
        kg = KnowledgeGrounding()
        kg.add_fact("cat", "is_a", "animal")
        kg.add_fact("cat", "likes", "fish")
        results = kg.query("cat")
        assert len(results) == 2

    def test_query_by_relation(self):
        kg = KnowledgeGrounding()
        kg.add_fact("cat", "is_a", "animal")
        kg.add_fact("cat", "likes", "fish")
        results = kg.query("cat", "is_a")
        assert results == ["animal"]

    def test_query_nonexistent_subject(self):
        kg = KnowledgeGrounding()
        assert kg.query("nonexistent") == []

    def test_verify_statement_true(self):
        kg = KnowledgeGrounding()
        kg.add_fact("cat", "hunts", "mice")
        result = kg.verify_statement("cat hunts mice")
        assert result["verified"] is True

    def test_verify_statement_false(self):
        kg = KnowledgeGrounding()
        kg.add_fact("cat", "hunts", "mice")
        result = kg.verify_statement("cat hunts dogs")
        assert result["verified"] is False

    def test_verify_statement_unparsable(self):
        kg = KnowledgeGrounding()
        result = kg.verify_statement("hi")
        assert result["verified"] is False

    def test_get_context_for_prompt(self):
        kg = KnowledgeGrounding()
        kg.add_fact("Python", "is_a", "language")
        kg.add_fact("Python", "used_for", "coding")
        ctx = kg.get_context_for_prompt("tell me about Python")
        assert "Python" in ctx
        assert len(ctx) > 0

    def test_get_context_for_prompt_empty(self):
        kg = KnowledgeGrounding()
        ctx = kg.get_context_for_prompt("nothing")
        assert ctx == ""


class TestCurriculumLearner:
    def test_initial_state(self):
        cl = CurriculumLearner()
        assert cl.stage == "bootstrapping"
        assert cl.current_level == 0

    def test_add_example(self):
        cl = CurriculumLearner()
        cl.add_example("easy", 0.2)
        cl.add_example("hard", 0.9)
        assert len(cl.difficulty_levels[2]) == 1
        assert len(cl.difficulty_levels[9]) == 1

    def test_get_batch_bootstrapping(self):
        cl = CurriculumLearner()
        for i in range(20):
            cl.add_example(f"ex_{i}", 0.1)
        batch = cl.get_batch(batch_size=5)
        assert len(batch) == 5

    def test_get_batch_empty_returns_empty(self):
        cl = CurriculumLearner()
        batch = cl.get_batch(batch_size=5)
        assert batch == []

    def test_update_stage_bootstrapping(self):
        cl = CurriculumLearner()
        cl.update_stage(0.5)
        assert cl.stage == "bootstrapping"

    def test_update_stage_progressing(self):
        cl = CurriculumLearner()
        cl.update_stage(0.8)
        assert cl.stage == "progressing"

    def test_update_stage_mastery(self):
        cl = CurriculumLearner()
        cl.update_stage(0.95)
        assert cl.stage == "mastery"
        assert cl.current_level == 1

    def test_update_stage_poor_performance_regresses(self):
        cl = CurriculumLearner()
        cl.current_level = 5
        cl.update_stage(0.3)
        assert cl.current_level == 4


class TestGroundingOrchestrator:
    def test_initial_state(self):
        go = GroundingOrchestrator()
        assert go.rag is not None
        assert go.kg is not None
        assert go.curriculum is not None
        assert go.ewc is None

    def test_add_data(self):
        go = GroundingOrchestrator()
        go.add_data("cats chase mice")
        assert len(go.rag.documents) == 1

    def test_extract_triples(self):
        go = GroundingOrchestrator()
        triples = go._extract_triples("water causes rust")
        assert len(triples) >= 1
        assert triples[0][0] == "water"
        assert triples[0][2] == "rust"

    def test_extract_triples_is_a(self):
        go = GroundingOrchestrator()
        triples = go._extract_triples("Python is a language")
        assert len(triples) >= 1

    def test_extract_triples_no_match(self):
        go = GroundingOrchestrator()
        triples = go._extract_triples("hello world test")
        assert triples == []

    def test_ground_output(self):
        go = GroundingOrchestrator()
        result = go.ground_output("Python is great", "what is Python")
        assert "query" in result
        assert "confidence" in result

    def test_get_knowledge_context(self):
        go = GroundingOrchestrator()
        go.kg.add_fact("Python", "is_a", "language")
        ctx = go.get_knowledge_context("Python")
        assert len(ctx) > 0

    def test_get_curriculum_batch(self):
        go = GroundingOrchestrator()
        for i in range(10):
            go.curriculum.add_example(f"ex_{i}", 0.1)
        batch = go.get_curriculum_batch(batch_size=3)
        assert len(batch) == 3
