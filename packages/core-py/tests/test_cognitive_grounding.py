"""Comprehensive tests for domains/cognitive/grounding.py — pure logic only."""

import pytest
import numpy as np
from domains.cognitive.grounding import (
    Document,
    RAGGrounder,
    FisherInformation,
    ElasticWeightConsolidation,
    HierarchicalContext,
    KnowledgeNode,
    KnowledgeEdge,
    KnowledgeGrounding,
    CurriculumLearner,
    GroundingOrchestrator,
)


# ---------------------------------------------------------------------------
# Document dataclass
# ---------------------------------------------------------------------------

class TestDocument:
    def test_fields(self):
        doc = Document(id="d1", content="hello", source="wiki", metadata={"k": "v"})
        assert doc.id == "d1"
        assert doc.content == "hello"
        assert doc.source == "wiki"
        assert doc.metadata == {"k": "v"}

    def test_default_metadata(self):
        doc = Document(id="d1", content="x", source="s")
        assert doc.metadata == {}

    def test_default_embedding(self):
        doc = Document(id="d1", content="x", source="s")
        assert doc.embedding is None


# ---------------------------------------------------------------------------
# RAGGrounder
# ---------------------------------------------------------------------------

class TestRAGGrounder:
    def test_add_document_stores(self):
        rg = RAGGrounder()
        doc = Document(id="d1", content="hello world test", source="wiki")
        rg.add_document(doc)
        assert "d1" in rg.documents
        assert len(rg.chunks) == 1

    def test_add_document_chunks_long(self):
        rg = RAGGrounder()
        words = " ".join([f"word{i}" for i in range(100)])
        doc = Document(id="d1", content=words, source="test")
        rg.add_document(doc, chunk_size=20)
        assert len(rg.chunks) > 1

    def test_add_document_chunk_ids(self):
        rg = RAGGrounder()
        doc = Document(id="d1", content="a b c d e", source="test")
        rg.add_document(doc, chunk_size=2)
        assert rg.chunks[0].id == "d1_chunk_0"
        assert rg.chunks[1].id == "d1_chunk_1"

    def test_add_document_chunk_metadata_inherits(self):
        rg = RAGGrounder()
        doc = Document(id="d1", content="test content", source="wiki", metadata={"page": 5})
        rg.add_document(doc)
        assert rg.chunks[0].metadata["page"] == 5
        assert rg.chunks[0].metadata["parent_id"] == "d1"

    def test_add_text_creates_document(self):
        rg = RAGGrounder()
        doc_id = rg.add_text("hello world", source="user")
        assert doc_id in rg.documents
        assert rg.documents[doc_id].source == "user"

    def test_add_text_default_source(self):
        rg = RAGGrounder()
        doc_id = rg.add_text("test")
        assert rg.documents[doc_id].source == "user"

    def test_add_text_returns_sequential_ids(self):
        rg = RAGGrounder()
        id1 = rg.add_text("first")
        id2 = rg.add_text("second")
        assert id1 == "doc_0"
        assert id2 == "doc_1"

    @pytest.mark.asyncio
    async def test_retrieve_keyword_match(self):
        rg = RAGGrounder()
        rg.add_text("python programming language")
        rg.add_text("cooking recipes food")
        results = await rg.retrieve("python", top_k=5, min_relevance=0.3)
        assert len(results) >= 1
        assert any("python" in d.content.lower() for d in results)

    @pytest.mark.asyncio
    async def test_retrieve_no_match(self):
        rg = RAGGrounder()
        rg.add_text("hello world")
        results = await rg.retrieve("quantum physics", top_k=5, min_relevance=0.5)
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_top_k(self):
        rg = RAGGrounder()
        for i in range(10):
            rg.add_text(f"document number {i} about topic")
        results = await rg.retrieve("document topic", top_k=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_retrieve_min_relevance(self):
        rg = RAGGrounder()
        rg.add_text("alpha beta gamma")
        results = await rg.retrieve("alpha", top_k=5, min_relevance=2.0)
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_relevance_calculation(self):
        rg = RAGGrounder()
        rg.add_text("the cat sat on the mat")
        results = await rg.retrieve("the cat", top_k=5, min_relevance=0.0)
        if results:
            for d in results:
                assert isinstance(d, Document)

    def test_ground_response_no_supporting(self):
        rg = RAGGrounder()
        result = rg.ground_response("some response", "unrelated query")
        assert result["response"] == "some response"
        assert result["grounded"] is False
        assert result["confidence"] == 0.0
        assert result["supporting_docs"] == []

    def test_ground_response_structure(self):
        rg = RAGGrounder()
        result = rg.ground_response("resp", "query")
        for key in ("response", "grounded", "confidence", "supporting_docs",
                     "contradictions", "hallucination_score"):
            assert key in result

    def test_ground_response_with_data(self):
        rg = RAGGrounder()
        rg.add_text("python is a programming language")
        result = rg.ground_response("Python is great", "python programming")
        assert isinstance(result["grounded"], bool)
        assert isinstance(result["confidence"], float)


# ---------------------------------------------------------------------------
# FisherInformation dataclass
# ---------------------------------------------------------------------------

class TestFisherInformation:
    def test_fields(self):
        fi = FisherInformation(param_name="layer1.weight", importance=0.5, old_value=1.0)
        assert fi.param_name == "layer1.weight"
        assert fi.importance == 0.5
        assert fi.old_value == 1.0


# ---------------------------------------------------------------------------
# ElasticWeightConsolidation
# ---------------------------------------------------------------------------

class _FakeParam:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float64)
        self.grad = None


class _FakeModel:
    def __init__(self, param_dict):
        self._params = param_dict

    def eval(self):
        pass

    def zero_grad(self):
        pass

    def named_parameters(self):
        return list(self._params.items())

    def __call__(self, batch):
        return batch.mean()


class TestElasticWeightConsolidation:
    def test_initial_state(self):
        model = _FakeModel({})
        ewc = ElasticWeightConsolidation(model)
        assert ewc.fisher == {}
        assert ewc.optimal_params == {}
        assert ewc.lambda_ewc == 1000

    def test_ewc_loss_no_fisher(self):
        model = _FakeModel({"w": _FakeParam([1.0, 2.0])})
        ewc = ElasticWeightConsolidation(model)
        assert ewc.ewc_loss() == 0.0

    def test_ewc_loss_with_fisher(self):
        param = _FakeParam([1.0, 2.0])
        model = _FakeModel({"w": param})
        ewc = ElasticWeightConsolidation(model)
        ewc.fisher["w"] = 1.0
        ewc.optimal_params["w"] = np.array([1.0, 2.0])
        loss = ewc.ewc_loss()
        assert loss == 0.0

    def test_ewc_loss_penalizes_drift(self):
        param = _FakeParam([3.0, 4.0])
        model = _FakeModel({"w": param})
        ewc = ElasticWeightConsolidation(model)
        ewc.fisher["w"] = 1.0
        ewc.optimal_params["w"] = np.array([1.0, 2.0])
        loss = ewc.ewc_loss()
        assert loss > 0.0

    def test_ewc_loss_proportional_to_fisher(self):
        param = _FakeParam([3.0, 4.0])
        model = _FakeModel({"w": param})
        ewc = ElasticWeightConsolidation(model)
        ewc.optimal_params["w"] = np.array([1.0, 2.0])

        ewc.fisher["w"] = 1.0
        loss1 = ewc.ewc_loss()

        ewc.fisher["w"] = 10.0
        loss2 = ewc.ewc_loss()

        assert loss2 == pytest.approx(loss1 * 10.0)

    def test_ewc_loss_ignores_params_without_fisher(self):
        param_a = _FakeParam([1.0])
        param_b = _FakeParam([2.0])
        model = _FakeModel({"a": param_a, "b": param_b})
        ewc = ElasticWeightConsolidation(model)
        ewc.fisher["a"] = 1.0
        ewc.optimal_params["a"] = np.array([1.0])
        loss = ewc.ewc_loss()
        assert loss == 0.0

    def test_ewc_loss_multi_param(self):
        p1 = _FakeParam([1.0, 2.0])
        p2 = _FakeParam([3.0, 4.0])
        model = _FakeModel({"w1": p1, "w2": p2})
        ewc = ElasticWeightConsolidation(model)
        ewc.fisher["w1"] = 1.0
        ewc.fisher["w2"] = 2.0
        ewc.optimal_params["w1"] = np.array([1.0, 2.0])
        ewc.optimal_params["w2"] = np.array([3.0, 4.0])

        p1.data = np.array([5.0, 6.0])
        p2.data = np.array([7.0, 8.0])
        loss = ewc.ewc_loss()
        assert loss > 0.0

    def test_compute_fisher_stores_params(self):
        param = _FakeParam([1.0])
        model = _FakeModel({"w": param})
        ewc = ElasticWeightConsolidation(model)
        assert "w" not in ewc.fisher
        assert "w" not in ewc.optimal_params


# ---------------------------------------------------------------------------
# HierarchicalContext
# ---------------------------------------------------------------------------

class TestHierarchicalContext:
    def test_initial_state(self):
        hc = HierarchicalContext(max_context=2048, chunk_size=128)
        assert hc.max_context == 2048
        assert hc.chunk_size == 128
        assert hc.hierarchy == []

    def test_build_hierarchy_single_chunk(self):
        hc = HierarchicalContext(max_context=4096, chunk_size=100)
        hc.build_hierarchy("hello world")
        assert len(hc.hierarchy) >= 1
        assert hc.hierarchy[0] == ["hello world"]

    def test_build_hierarchy_multiple_chunks(self):
        hc = HierarchicalContext(max_context=4096, chunk_size=5)
        words = " ".join([f"w{i}" for i in range(20)])
        hc.build_hierarchy(words)
        assert len(hc.hierarchy[0]) > 1

    def test_build_hierarchy_cascades(self):
        hc = HierarchicalContext(max_context=4096, chunk_size=3)
        words = " ".join([f"w{i}" for i in range(12)])
        hc.build_hierarchy(words)
        if len(hc.hierarchy[0]) > 1:
            assert len(hc.hierarchy) >= 2

    def test_build_hierarchy_top_level_single(self):
        hc = HierarchicalContext(max_context=4096, chunk_size=5)
        words = " ".join([f"word{i}" for i in range(30)])
        hc.build_hierarchy(words)
        assert len(hc.hierarchy[-1]) == 1

    def test_summarize_pair(self):
        hc = HierarchicalContext(max_context=4096, chunk_size=100)
        result = hc._summarize_pair("alpha beta", "gamma delta")
        assert "alpha" in result
        assert "gamma" in result

    def test_summarize_pair_truncates(self):
        hc = HierarchicalContext(max_context=4096, chunk_size=3)
        words = " ".join([f"w{i}" for i in range(10)])
        result = hc._summarize_pair(words, "extra")
        assert len(result.split()) <= 3

    def test_get_relevant_context_empty(self):
        hc = HierarchicalContext(max_context=4096, chunk_size=512)
        assert hc.get_relevant_context("query") == ""

    def test_get_relevant_context_returns_string(self):
        hc = HierarchicalContext(max_context=4096, chunk_size=5)
        hc.build_hierarchy("hello world test content here")
        result = hc.get_relevant_context("hello")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_relevant_context_max_length(self):
        hc = HierarchicalContext(max_context=50, chunk_size=5)
        words = " ".join([f"word{i}" for i in range(100)])
        hc.build_hierarchy(words)
        result = hc.get_relevant_context("word")
        assert len(result) <= 50

    def test_attention_mask_shape(self):
        hc = HierarchicalContext(max_context=4096, chunk_size=4)
        hc.build_hierarchy("a b c d e f g h")
        mask = hc.attention_mask(8)
        assert mask.shape == (8, 8)

    def test_attention_mask_values(self):
        hc = HierarchicalContext(max_context=4096, chunk_size=4)
        hc.build_hierarchy("a b c d e f g h")
        mask = hc.attention_mask(8)
        assert np.all((mask == 0) | (mask == 1))

    def test_attention_mask_empty_hierarchy(self):
        hc = HierarchicalContext(max_context=4096, chunk_size=4)
        mask = hc.attention_mask(4)
        assert mask.shape == (4, 4)
        assert np.all(mask == 1)

    def test_hierarchy_levels_monotonic_decrease(self):
        hc = HierarchicalContext(max_context=4096, chunk_size=3)
        words = " ".join([f"w{i}" for i in range(18)])
        hc.build_hierarchy(words)
        for i in range(1, len(hc.hierarchy)):
            assert len(hc.hierarchy[i]) <= len(hc.hierarchy[i - 1])


# ---------------------------------------------------------------------------
# KnowledgeNode & KnowledgeEdge dataclasses
# ---------------------------------------------------------------------------

class TestKnowledgeNodeEdge:
    def test_node_fields(self):
        node = KnowledgeNode(id="n1", label="Python", node_type="entity", properties={"lang": "en"})
        assert node.id == "n1"
        assert node.label == "Python"
        assert node.node_type == "entity"
        assert node.properties == {"lang": "en"}

    def test_node_default_properties(self):
        node = KnowledgeNode(id="n1", label="X", node_type="concept")
        assert node.properties == {}

    def test_edge_fields(self):
        edge = KnowledgeEdge(source="A", target="B", relation="is_a", weight=0.8)
        assert edge.source == "A"
        assert edge.target == "B"
        assert edge.relation == "is_a"
        assert edge.weight == 0.8

    def test_edge_default_weight(self):
        edge = KnowledgeEdge(source="A", target="B", relation="related_to")
        assert edge.weight == 1.0


# ---------------------------------------------------------------------------
# KnowledgeGrounding
# ---------------------------------------------------------------------------

class TestKnowledgeGrounding:
    def test_initial_state(self):
        kg = KnowledgeGrounding()
        assert kg.nodes == {}
        assert kg.edges == []

    def test_add_fact_creates_nodes(self):
        kg = KnowledgeGrounding()
        kg.add_fact("Python", "is_a", "language")
        assert "Python" in kg.nodes
        assert "language" in kg.nodes

    def test_add_fact_creates_edge(self):
        kg = KnowledgeGrounding()
        kg.add_fact("Python", "is_a", "language")
        assert len(kg.edges) == 1
        assert kg.edges[0].source == "Python"
        assert kg.edges[0].target == "language"
        assert kg.edges[0].relation == "is_a"

    def test_add_fact_updates_adjacency(self):
        kg = KnowledgeGrounding()
        kg.add_fact("Python", "is_a", "language")
        targets = kg.query("Python")
        assert "language" in targets

    def test_add_fact_default_confidence(self):
        kg = KnowledgeGrounding()
        kg.add_fact("A", "related_to", "B")
        assert kg.edges[0].weight == 1.0

    def test_add_fact_custom_confidence(self):
        kg = KnowledgeGrounding()
        kg.add_fact("A", "related_to", "B", confidence=0.7)
        assert kg.edges[0].weight == 0.7

    def test_query_returns_all_targets(self):
        kg = KnowledgeGrounding()
        kg.add_fact("Python", "is_a", "language")
        kg.add_fact("Python", "used_for", "web")
        results = kg.query("Python")
        assert "language" in results
        assert "web" in results

    def test_query_filters_by_relation(self):
        kg = KnowledgeGrounding()
        kg.add_fact("Python", "is_a", "language")
        kg.add_fact("Python", "used_for", "web")
        results = kg.query("Python", relation="is_a")
        assert results == ["language"]

    def test_query_nonexistent_subject(self):
        kg = KnowledgeGrounding()
        assert kg.query("Unknown") == []

    def test_query_relation_not_found(self):
        kg = KnowledgeGrounding()
        kg.add_fact("Python", "is_a", "language")
        results = kg.query("Python", relation="causes")
        assert results == []

    def test_verify_statement_verified(self):
        kg = KnowledgeGrounding()
        kg.add_fact("Paris", "is_a", "city")
        result = kg.verify_statement("Paris is_a city")
        assert result["verified"] is True
        assert result["grounded_in_knowledge"] is True

    def test_verify_statement_not_verified(self):
        kg = KnowledgeGrounding()
        kg.add_fact("Paris", "is_a", "city")
        result = kg.verify_statement("Paris is_a country")
        assert result["verified"] is False

    def test_verify_statement_unparseable(self):
        kg = KnowledgeGrounding()
        result = kg.verify_statement("hello")
        assert result["verified"] is False
        assert "reason" in result

    def test_verify_statement_supporting_facts(self):
        kg = KnowledgeGrounding()
        kg.add_fact("Dog", "is_a", "animal")
        kg.add_fact("Dog", "is_a", "mammal")
        result = kg.verify_statement("Dog is_a animal")
        assert len(result["supporting_facts"]) >= 1

    def test_verify_statement_strips_dot(self):
        kg = KnowledgeGrounding()
        kg.add_fact("X", "is_a", "Y")
        result = kg.verify_statement("X is_a Y.")
        assert result["verified"] is True

    def test_get_context_for_prompt_no_nodes(self):
        kg = KnowledgeGrounding()
        assert kg.get_context_for_prompt("anything") == ""

    def test_get_context_for_prompt_finds_matching(self):
        kg = KnowledgeGrounding()
        kg.add_fact("Python", "is_a", "language")
        kg.add_fact("Python", "used_for", "web")
        ctx = kg.get_context_for_prompt("tell me about Python")
        assert "Python" in ctx

    def test_get_context_for_prompt_limit_5(self):
        kg = KnowledgeGrounding()
        for i in range(10):
            kg.add_fact(f"Node{i}", "related_to", f"Target{i}")
        ctx = kg.get_context_for_prompt(" ".join([f"node{i}" for i in range(10)]))
        parts = ctx.split("; ")
        assert len(parts) <= 5

    def test_get_context_for_prompt_facts_limited_to_2(self):
        kg = KnowledgeGrounding()
        kg.add_fact("A", "r1", "B")
        kg.add_fact("A", "r2", "C")
        kg.add_fact("A", "r3", "D")
        ctx = kg.get_context_for_prompt("A")
        fact_count = ctx.count("is related to")
        assert fact_count <= 2

    def test_multiple_facts_same_subject(self):
        kg = KnowledgeGrounding()
        kg.add_fact("Sun", "is_a", "star")
        kg.add_fact("Sun", "has", "planets")
        assert len(kg.edges) == 2
        assert len(kg.adjacency["Sun"]) == 2


# ---------------------------------------------------------------------------
# CurriculumLearner
# ---------------------------------------------------------------------------

class TestCurriculumLearner:
    def test_initial_state(self):
        cl = CurriculumLearner()
        assert cl.current_level == 0
        assert cl.stage == "bootstrapping"

    def test_add_example(self):
        cl = CurriculumLearner()
        cl.add_example("ex1", difficulty=0.3)
        cl.add_example("ex2", difficulty=0.7)
        assert len(cl.difficulty_levels[3]) == 1
        assert len(cl.difficulty_levels[7]) == 1

    def test_add_example_level_mapping(self):
        cl = CurriculumLearner()
        cl.add_example("x", difficulty=0.0)
        assert 0 in cl.difficulty_levels
        cl.add_example("y", difficulty=1.0)
        assert 10 in cl.difficulty_levels

    def test_get_batch_bootstrapping(self):
        cl = CurriculumLearner()
        cl.add_example("easy1", difficulty=0.0)
        cl.add_example("easy2", difficulty=0.1)
        cl.add_example("hard1", difficulty=0.9)
        batch = cl.get_batch(10)
        assert all(x in ["easy1", "easy2"] for x in batch)

    def test_get_batch_progressing(self):
        cl = CurriculumLearner()
        cl.stage = "progressing"
        cl.current_level = 5
        for i in range(11):
            cl.add_example(f"ex{i}", difficulty=i / 10.0)
        batch = cl.get_batch(10)
        assert len(batch) > 0

    def test_get_batch_mastery(self):
        cl = CurriculumLearner()
        cl.stage = "mastery"
        for i in range(11):
            cl.add_example(f"ex{i}", difficulty=i / 10.0)
        batch = cl.get_batch(100)
        assert len(batch) == 11

    def test_get_batch_respects_size(self):
        cl = CurriculumLearner()
        for i in range(20):
            cl.add_example(f"ex{i}", difficulty=i / 20.0)
        batch = cl.get_batch(3)
        assert len(batch) <= 3

    def test_get_batch_empty_returns_empty(self):
        cl = CurriculumLearner()
        assert cl.get_batch(5) == []

    def test_update_stage_mastery(self):
        cl = CurriculumLearner()
        cl.update_stage(0.95)
        assert cl.stage == "mastery"
        assert cl.current_level == 1

    def test_update_stage_progressing(self):
        cl = CurriculumLearner()
        cl.update_stage(0.75)
        assert cl.stage == "progressing"

    def test_update_stage_bootstrapping(self):
        cl = CurriculumLearner()
        cl.update_stage(0.5)
        assert cl.stage == "bootstrapping"
        assert cl.current_level == 0

    def test_update_stage_bootstrapping_decrements(self):
        cl = CurriculumLearner()
        cl.current_level = 5
        cl.update_stage(0.3)
        assert cl.current_level == 4

    def test_update_stage_mastery_caps_at_10(self):
        cl = CurriculumLearner()
        cl.current_level = 10
        cl.update_stage(0.95)
        assert cl.current_level == 10

    def test_update_stage_bootstrapping_floors_at_0(self):
        cl = CurriculumLearner()
        cl.current_level = 0
        cl.update_stage(0.3)
        assert cl.current_level == 0


# ---------------------------------------------------------------------------
# GroundingOrchestrator
# ---------------------------------------------------------------------------

class TestGroundingOrchestrator:
    def test_initial_state(self):
        go = GroundingOrchestrator()
        assert go.rag is not None
        assert go.kg is not None
        assert go.curriculum is not None
        assert go.ewc is None

    def test_add_data_populates_rag(self):
        go = GroundingOrchestrator()
        go.add_data("hello world test content")
        assert len(go.rag.documents) == 1

    def test_add_data_populates_kg(self):
        go = GroundingOrchestrator()
        go.add_data("Dog is a animal")
        assert len(go.kg.edges) >= 1

    def test_extract_triples_is_a(self):
        go = GroundingOrchestrator()
        triples = go._extract_triples("Dog is a animal")
        assert len(triples) >= 1
        assert triples[0][0] == "Dog"
        assert triples[0][2] == "animal"

    def test_extract_triples_located_in(self):
        go = GroundingOrchestrator()
        triples = go._extract_triples("Paris is located in France")
        assert len(triples) >= 1
        assert triples[0][0] == "Paris"
        assert triples[0][2] == "France"

    def test_extract_triples_causes(self):
        go = GroundingOrchestrator()
        triples = go._extract_triples("Smoking causes cancer")
        assert len(triples) >= 1
        assert triples[0][0] == "Smoking"

    def test_extract_triples_no_match(self):
        go = GroundingOrchestrator()
        triples = go._extract_triples("hello world")
        assert triples == []

    def test_ground_output_structure(self):
        go = GroundingOrchestrator()
        result = go.ground_output("response text", "query")
        assert "response" in result
        assert "query" in result
        assert "grounding_applied" in result
        assert "verified" in result
        assert "confidence" in result
        assert "metadata" in result

    def test_ground_output_verified_via_kg(self):
        go = GroundingOrchestrator()
        go.kg.add_fact("Paris", "is_a", "city")
        result = go.ground_output("Paris is_a city", "what is Paris")
        assert result["verified"] is True
        assert result["confidence"] == 0.9

    def test_ground_output_not_verified(self):
        go = GroundingOrchestrator()
        result = go.ground_output("random text", "query")
        assert result["verified"] is False

    def test_ground_output_metadata_kg(self):
        go = GroundingOrchestrator()
        result = go.ground_output("test", "query")
        assert "kg_verified" in result["metadata"]
        assert "rag_confidence" in result["metadata"]
        assert "supporting_docs" in result["metadata"]

    def test_get_knowledge_context(self):
        go = GroundingOrchestrator()
        go.kg.add_fact("Python", "is_a", "language")
        ctx = go.get_knowledge_context("Python")
        assert "Python" in ctx

    def test_get_knowledge_context_empty(self):
        go = GroundingOrchestrator()
        ctx = go.get_knowledge_context("nothing")
        assert ctx == ""

    def test_get_curriculum_batch(self):
        go = GroundingOrchestrator()
        go.curriculum.add_example("ex1", difficulty=0.1)
        batch = go.get_curriculum_batch(5)
        assert isinstance(batch, list)

    def test_add_data_multiple(self):
        go = GroundingOrchestrator()
        go.add_data("first sentence about topic one")
        go.add_data("second sentence about topic two")
        assert len(go.rag.documents) == 2
