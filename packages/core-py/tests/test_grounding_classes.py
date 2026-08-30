"""Tests for domains.cognitive.grounding — HierarchicalContext, CurriculumLearner, RAGGrounder, KnowledgeGrounding, GroundingOrchestrator."""

from domains.cognitive.grounding import (
    HierarchicalContext,
    CurriculumLearner,
    RAGGrounder,
    Document,
    KnowledgeGrounding,
    KnowledgeNode,
    KnowledgeEdge,
    GroundingOrchestrator,
)


class TestHierarchicalContext:
    def test_init(self):
        hc = HierarchicalContext()
        assert hc.max_context == 4096
        assert hc.chunk_size == 512
        assert hc.hierarchy == []

    def test_build_hierarchy(self):
        hc = HierarchicalContext(chunk_size=10)
        text = "word " * 25  # 25 words
        hc.build_hierarchy(text)
        assert len(hc.hierarchy) >= 1
        assert len(hc.hierarchy[0]) == 3  # ceil(25/10) = 3

    def test_get_relevant_context_empty(self):
        hc = HierarchicalContext()
        assert hc.get_relevant_context("query") == ""

    def test_get_relevant_context(self):
        hc = HierarchicalContext(chunk_size=10)
        text = "word " * 25
        hc.build_hierarchy(text)
        ctx = hc.get_relevant_context("word")
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_build_hierarchy_single_chunk(self):
        hc = HierarchicalContext(chunk_size=100)
        text = "hello world"
        hc.build_hierarchy(text)
        assert len(hc.hierarchy) == 1
        assert len(hc.hierarchy[0]) == 1

    def test_build_hierarchy_exact_chunk_size(self):
        hc = HierarchicalContext(chunk_size=5)
        text = "a b c d e"
        hc.build_hierarchy(text)
        assert len(hc.hierarchy[0]) == 1

    def test_hierarchy_depth(self):
        hc = HierarchicalContext(chunk_size=2)
        text = " ".join([f"w{i}" for i in range(16)])
        hc.build_hierarchy(text)
        assert len(hc.hierarchy) >= 2

    def test_context_truncated_to_max(self):
        hc = HierarchicalContext(max_context=20, chunk_size=5)
        text = " ".join([f"word{i}" for i in range(20)])
        hc.build_hierarchy(text)
        ctx = hc.get_relevant_context("word")
        assert len(ctx) <= 20

    def test_attention_mask_shape(self):
        hc = HierarchicalContext(chunk_size=4)
        text = "a b c d e f g h"
        hc.build_hierarchy(text)
        mask = hc.attention_mask(8)
        assert mask.shape == (8, 8)

    def test_attention_mask_values(self):
        hc = HierarchicalContext(chunk_size=4)
        text = "a b c d"
        hc.build_hierarchy(text)
        mask = hc.attention_mask(4)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_summarize_pair(self):
        hc = HierarchicalContext(chunk_size=10)
        result = hc._summarize_pair("hello world", "foo bar")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_hierarchy_empty_text(self):
        hc = HierarchicalContext(chunk_size=10)
        hc.build_hierarchy("")
        assert len(hc.hierarchy) >= 1

    def test_custom_params(self):
        hc = HierarchicalContext(max_context=1024, chunk_size=256)
        assert hc.max_context == 1024
        assert hc.chunk_size == 256


class TestCurriculumLearner:
    def test_init(self):
        cl = CurriculumLearner()
        assert cl.current_level == 0
        assert cl.stage == "bootstrapping"

    def test_add_example(self):
        cl = CurriculumLearner()
        cl.add_example("easy", 0.1)
        cl.add_example("medium", 0.5)
        cl.add_example("hard", 0.9)
        assert len(cl.difficulty_levels) == 3

    def test_get_batch_bootstrapping(self):
        cl = CurriculumLearner()
        cl.add_example("e1", 0.05)
        cl.add_example("e2", 0.15)
        cl.add_example("e3", 0.8)
        batch = cl.get_batch(2)
        assert len(batch) == 2

    def test_update_stage_mastery(self):
        cl = CurriculumLearner()
        cl.update_stage(0.95)
        assert cl.stage == "mastery"

    def test_update_stage_progressing(self):
        cl = CurriculumLearner()
        cl.update_stage(0.8)
        assert cl.stage == "progressing"

    def test_update_stage_bootstrapping(self):
        cl = CurriculumLearner()
        cl.update_stage(0.3)
        assert cl.stage == "bootstrapping"

    def test_difficulty_levels_stored(self):
        cl = CurriculumLearner()
        cl.add_example("ex1", 0.2)
        cl.add_example("ex2", 0.7)
        assert 2 in cl.difficulty_levels
        assert 7 in cl.difficulty_levels

    def test_get_batch_progressing(self):
        cl = CurriculumLearner()
        for i in range(10):
            cl.add_example(f"ex{i}", i / 10.0)
        cl.update_stage(0.8)
        batch = cl.get_batch(5)
        assert len(batch) <= 5

    def test_get_batch_mastery(self):
        cl = CurriculumLearner()
        for i in range(10):
            cl.add_example(f"ex{i}", i / 10.0)
        cl.update_stage(0.95)
        batch = cl.get_batch(5)
        assert len(batch) <= 5

    def test_get_batch_empty(self):
        cl = CurriculumLearner()
        batch = cl.get_batch(5)
        assert batch == []

    def test_mastery_level_increments(self):
        cl = CurriculumLearner()
        cl.update_stage(0.95)
        assert cl.current_level == 1
        cl.update_stage(0.95)
        assert cl.current_level == 2

    def test_bootstrapping_level_decrements(self):
        cl = CurriculumLearner()
        cl.current_level = 5
        cl.update_stage(0.3)
        assert cl.current_level == 4

    def test_level_capped_at_zero(self):
        cl = CurriculumLearner()
        cl.current_level = 0
        cl.update_stage(0.3)
        assert cl.current_level == 0

    def test_level_capped_at_10(self):
        cl = CurriculumLearner()
        cl.current_level = 10
        cl.update_stage(0.95)
        assert cl.current_level == 10

    def test_multiple_adds_same_level(self):
        cl = CurriculumLearner()
        cl.add_example("a", 0.05)
        cl.add_example("b", 0.05)
        assert len(cl.difficulty_levels[0]) == 2


class TestRAGGrounder:
    def test_init(self):
        rag = RAGGrounder()
        assert rag.documents == {}
        assert rag.chunks == []

    def test_add_document(self):
        rag = RAGGrounder()
        doc = Document(id="d1", content="hello world test", source="test")
        rag.add_document(doc, chunk_size=2)
        assert "d1" in rag.documents
        assert len(rag.chunks) > 0

    def test_add_text(self):
        rag = RAGGrounder()
        doc_id = rag.add_text("some text content here")
        assert doc_id.startswith("doc_")
        assert len(rag.chunks) > 0

    def test_chunks_have_parent_metadata(self):
        rag = RAGGrounder()
        doc = Document(id="d1", content="a b c d e f", source="test")
        rag.add_document(doc, chunk_size=2)
        for chunk in rag.chunks:
            assert "parent_id" in chunk.metadata

    def test_retrieve(self):
        import asyncio
        rag = RAGGrounder()
        rag.add_text("machine learning is great", source="test")
        results = asyncio.run(rag.retrieve("machine learning", top_k=5))
        assert len(results) > 0

    def test_retrieve_empty(self):
        import asyncio
        rag = RAGGrounder()
        results = asyncio.run(rag.retrieve("query", top_k=5))
        assert results == []

    def test_ground_response(self):
        rag = RAGGrounder()
        rag.add_text("machine learning deep learning", source="test")
        result = rag.ground_response("response", "machine learning")
        assert "response" in result
        assert "grounded" in result
        assert "confidence" in result

    def test_ground_response_no_docs(self):
        rag = RAGGrounder()
        result = rag.ground_response("response", "query")
        assert result["grounded"] is False
        assert result["confidence"] == 0.0

    def test_document_chunking(self):
        rag = RAGGrounder()
        content = " ".join([f"word{i}" for i in range(20)])
        doc = Document(id="d1", content=content, source="test")
        rag.add_document(doc, chunk_size=5)
        assert len(rag.chunks) == 4

    def test_multiple_documents(self):
        rag = RAGGrounder()
        rag.add_text("first document content", source="s1")
        rag.add_text("second document content", source="s2")
        assert len(rag.documents) == 2


class TestKnowledgeGrounding:
    def test_init(self):
        kg = KnowledgeGrounding()
        assert len(kg.nodes) == 0
        assert len(kg.edges) == 0

    def test_add_fact(self):
        kg = KnowledgeGrounding()
        kg.add_fact("python", "is_a", "language")
        assert "python" in kg.nodes
        assert "language" in kg.nodes
        assert len(kg.edges) == 1

    def test_query(self):
        kg = KnowledgeGrounding()
        kg.add_fact("python", "is_a", "language")
        results = kg.query("python")
        assert "language" in results

    def test_query_with_relation(self):
        kg = KnowledgeGrounding()
        kg.add_fact("python", "is_a", "language")
        kg.add_fact("python", "used_for", "programming")
        results = kg.query("python", relation="is_a")
        assert "language" in results
        assert "programming" not in results

    def test_verify_statement_verified(self):
        kg = KnowledgeGrounding()
        kg.add_fact("python", "is_a", "language")
        result = kg.verify_statement("python is_a language")
        assert result["verified"] is True

    def test_verify_statement_not_verified(self):
        kg = KnowledgeGrounding()
        kg.add_fact("python", "is_a", "language")
        result = kg.verify_statement("java is_a language")
        assert result["verified"] is False

    def test_get_context_for_prompt(self):
        kg = KnowledgeGrounding()
        kg.add_fact("python", "is_a", "language")
        kg.add_fact("python", "used_for", "ml")
        ctx = kg.get_context_for_prompt("tell me about python")
        assert "python" in ctx.lower() or len(ctx) == 0

    def test_multiple_facts(self):
        kg = KnowledgeGrounding()
        kg.add_fact("a", "rel1", "b")
        kg.add_fact("a", "rel2", "c")
        results = kg.query("a")
        assert len(results) == 2

    def test_node_creation(self):
        kg = KnowledgeGrounding()
        kg.add_fact("entity1", "rel", "entity2")
        assert "entity1" in kg.nodes
        assert "entity2" in kg.nodes

    def test_edge_weight(self):
        kg = KnowledgeGrounding()
        kg.add_fact("a", "rel", "b", confidence=0.8)
        assert kg.edges[0].weight == 0.8

    def test_verify_statement_short(self):
        kg = KnowledgeGrounding()
        result = kg.verify_statement("hello")
        assert result["verified"] is False


class TestGroundingOrchestrator:
    def test_init(self):
        go = GroundingOrchestrator()
        assert go.rag is not None
        assert go.kg is not None
        assert go.curriculum is not None

    def test_add_data(self):
        go = GroundingOrchestrator()
        go.add_data("python is a language", source="test")
        assert len(go.rag.documents) >= 1

    def test_ground_output(self):
        go = GroundingOrchestrator()
        go.add_data("python is a language")
        result = go.ground_output("python is a language", "what is python")
        assert "response" in result
        assert "confidence" in result

    def test_ground_output_no_data(self):
        go = GroundingOrchestrator()
        result = go.ground_output("response", "query")
        assert result["confidence"] == 0.5

    def test_get_knowledge_context(self):
        go = GroundingOrchestrator()
        go.kg.add_fact("test", "is_a", "thing")
        ctx = go.get_knowledge_context("test")
        assert isinstance(ctx, str)

    def test_get_curriculum_batch(self):
        go = GroundingOrchestrator()
        go.curriculum.add_example("ex1", 0.1)
        batch = go.get_curriculum_batch(1)
        assert len(batch) <= 1

    def test_extract_triples(self):
        go = GroundingOrchestrator()
        triples = go._extract_triples("python is a language")
        assert len(triples) >= 1

    def test_extract_triples_none_found(self):
        go = GroundingOrchestrator()
        triples = go._extract_triples("hello world")
        assert isinstance(triples, list)

    def test_verify_via_kg(self):
        go = GroundingOrchestrator()
        go.kg.add_fact("x", "rel", "y")
        result = go.ground_output("x rel y", "query")
        assert result["verified"] is True
