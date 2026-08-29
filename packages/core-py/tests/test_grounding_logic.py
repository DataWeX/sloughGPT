"""Meaningful tests for CurriculumLearner, HierarchicalContext, KnowledgeNode/Edge, FisherInformation."""

import numpy as np
from domains.cognitive.grounding import (
    CurriculumLearner, HierarchicalContext, KnowledgeNode, KnowledgeEdge,
    FisherInformation, KnowledgeGrounding,
)


class TestCurriculumLearner:
    def test_initial_state(self):
        cl = CurriculumLearner()
        assert cl.stage == "bootstrapping"
        assert cl.current_level == 0

    def test_add_example(self):
        cl = CurriculumLearner()
        cl.add_example("example1", difficulty=0.5)
        cl.add_example("example2", difficulty=0.3)
        # Level 5 = 0.5*10, level 3 = 0.3*10
        assert len(cl.difficulty_levels[5]) == 1
        assert len(cl.difficulty_levels[3]) == 1

    def test_get_batch_bootstrapping(self):
        cl = CurriculumLearner()
        cl.add_example("easy1", difficulty=0.0)
        cl.add_example("easy2", difficulty=0.1)
        cl.add_example("hard1", difficulty=0.9)
        batch = cl.get_batch(10)
        # Bootstrapping only returns level 0 + level 1
        assert "hard1" not in batch
        assert "easy1" in batch or "easy2" in batch

    def test_get_batch_progressing(self):
        cl = CurriculumLearner()
        cl.stage = "progressing"
        cl.current_level = 5
        cl.add_example("e1", difficulty=0.4)  # level 4
        cl.add_example("e2", difficulty=0.5)  # level 5
        cl.add_example("e3", difficulty=0.6)  # level 6
        cl.add_example("e4", difficulty=0.0)  # level 0 - should NOT be included
        batch = cl.get_batch(10)
        assert "e4" not in batch

    def test_get_batch_mastery(self):
        cl = CurriculumLearner()
        cl.stage = "mastery"
        cl.add_example("e1", difficulty=0.0)
        cl.add_example("e2", difficulty=0.5)
        cl.add_example("e3", difficulty=0.9)
        batch = cl.get_batch(10)
        assert len(batch) == 3

    def test_get_batch_empty(self):
        cl = CurriculumLearner()
        batch = cl.get_batch(5)
        assert batch == []

    def test_update_stage_high_performance(self):
        cl = CurriculumLearner()
        cl.update_stage(0.95)
        assert cl.stage == "mastery"
        assert cl.current_level == 1

    def test_update_stage_medium_performance(self):
        cl = CurriculumLearner()
        cl.update_stage(0.8)
        assert cl.stage == "progressing"

    def test_update_stage_low_performance(self):
        cl = CurriculumLearner()
        cl.current_level = 5
        cl.update_stage(0.5)
        assert cl.stage == "bootstrapping"
        assert cl.current_level == 4

    def test_update_stage_low_performance_floor(self):
        cl = CurriculumLearner()
        cl.current_level = 0
        cl.update_stage(0.3)
        assert cl.current_level == 0


class TestHierarchicalContext:
    def test_build_hierarchy(self):
        hc = HierarchicalContext(max_context=4096, chunk_size=10)
        text = " ".join([f"word{i}" for i in range(25)])
        hc.build_hierarchy(text)
        assert len(hc.hierarchy) >= 1
        assert len(hc.hierarchy[0]) >= 2

    def test_build_hierarchy_single_chunk(self):
        hc = HierarchicalContext(max_context=4096, chunk_size=100)
        text = "short text"
        hc.build_hierarchy(text)
        assert len(hc.hierarchy[0]) == 1

    def test_get_relevant_context_empty(self):
        hc = HierarchicalContext()
        assert hc.get_relevant_context("query") == ""

    def test_get_relevant_context(self):
        hc = HierarchicalContext(max_context=100, chunk_size=5)
        text = " ".join([f"w{i}" for i in range(30)])
        hc.build_hierarchy(text)
        ctx = hc.get_relevant_context("query")
        assert len(ctx) <= 100
        assert len(ctx) > 0

    def test_attention_mask_shape(self):
        hc = HierarchicalContext(chunk_size=5)
        text = " ".join([f"w{i}" for i in range(20)])
        hc.build_hierarchy(text)
        mask = hc.attention_mask(10)
        assert mask.shape == (10, 10)
        # All zeros in mask should be 0 or 1
        assert set(np.unique(mask)).issubset({0, 1})


class TestKnowledgeNode:
    def test_fields(self):
        n = KnowledgeNode(id="n1", label="Paris", node_type="entity", properties={"pop": 2100000})
        assert n.id == "n1"
        assert n.properties["pop"] == 2100000

    def test_default_properties(self):
        n = KnowledgeNode(id="n1", label="X", node_type="concept")
        assert n.properties == {}


class TestKnowledgeEdge:
    def test_fields(self):
        e = KnowledgeEdge(source="s", target="t", relation="is_a", weight=0.8)
        assert e.source == "s"
        assert e.weight == 0.8


class TestKnowledgeGrounding:
    def test_add_fact_and_query(self):
        kg = KnowledgeGrounding()
        kg.add_fact("Paris", "capital_of", "France")
        results = kg.query("Paris", relation="capital_of")
        assert len(results) == 1
        assert results[0] == "France"

    def test_add_multiple_facts(self):
        kg = KnowledgeGrounding()
        kg.add_fact("Paris", "capital_of", "France")
        kg.add_fact("Berlin", "capital_of", "Germany")
        results = kg.query("Paris")
        assert len(results) == 1

    def test_query_empty(self):
        kg = KnowledgeGrounding()
        results = kg.query("nonexistent")
        assert results == []

    def test_query_creates_nodes(self):
        kg = KnowledgeGrounding()
        kg.add_fact("Paris", "capital_of", "France")
        assert "Paris" in kg.nodes
        assert "France" in kg.nodes

    def test_verify_true(self):
        kg = KnowledgeGrounding()
        kg.add_fact("Paris", "capital_of", "France")
        result = kg.verify_statement("Paris capital_of France")
        assert result["verified"] is True

    def test_verify_false(self):
        kg = KnowledgeGrounding()
        kg.add_fact("Paris", "capital_of", "France")
        result = kg.verify_statement("Paris capital_of Germany")
        assert result["verified"] is False

    def test_get_context(self):
        kg = KnowledgeGrounding()
        kg.add_fact("Paris", "capital_of", "France")
        ctx = kg.get_context_for_prompt("What about Paris")
        assert "Paris" in ctx


class TestFisherInformation:
    def test_fields(self):
        fi = FisherInformation(param_name="layer1.weight", importance=0.95, old_value=0.5)
        assert fi.param_name == "layer1.weight"
        assert fi.importance == 0.95
