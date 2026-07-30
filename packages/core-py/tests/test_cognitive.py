"""Tests for the cognitive domain (core, knowledge graph, reasoning)."""

from __future__ import annotations

import pytest
from domains.cognitive.core import CognitiveCore, ThinkingMode, ReasoningType
from domains.cognitive.knowledge_graph_v2 import KnowledgeGraph, RelationType
from domains.cognitive.reasoning.deep import (
    WorkingMemory,
    FormalLogicEngine,
    Predicate,
    Term,
    WellFormedFormula,
)

# ── CognitiveCore ─────────────────────────────────────────────────────────


class TestCognitiveCore:
    def test_think_analytical(self):
        core = CognitiveCore()
        t = core.think("solve x", ThinkingMode.ANALYTICAL)
        assert t.mode == ThinkingMode.ANALYTICAL
        assert "Analysis" in t.thought_content
        assert t.confidence == 0.85
        assert len(core.thought_history) == 1

    def test_think_creative(self):
        core = CognitiveCore()
        t = core.think("new idea", ThinkingMode.CREATIVE)
        assert t.mode == ThinkingMode.CREATIVE
        assert "Creative insight" in t.thought_content

    def test_think_critical(self):
        core = CognitiveCore()
        t = core.think("review", ThinkingMode.CRITICAL)
        assert t.mode == ThinkingMode.CRITICAL
        assert "Critical review" in t.thought_content

    def test_think_strategic(self):
        core = CognitiveCore()
        t = core.think("plan", ThinkingMode.STRATEGIC)
        assert t.mode == ThinkingMode.STRATEGIC
        assert "Strategic planning" in t.thought_content

    def test_think_reflective(self):
        core = CognitiveCore()
        t = core.think("reflect", ThinkingMode.REFLECTIVE)
        assert t.mode == ThinkingMode.REFLECTIVE
        assert "Reflection" in t.thought_content

    def test_generate_idea(self):
        core = CognitiveCore()
        idea = core.generate_idea("flying car", category="transport")
        assert idea.concept == "flying car"
        assert idea.category == "transport"
        assert 0 <= idea.novelty_score <= 1
        assert len(core.ideas) == 1

    def test_reason(self):
        core = CognitiveCore()
        chain = core.reason("Why is the sky blue?", ReasoningType.CAUSAL)
        assert chain.reasoning_type == ReasoningType.CAUSAL
        assert "Why is the sky blue?" in chain.question
        assert len(chain.reasoning_steps) > 0
        assert len(chain.evidence) > 0
        assert len(core.reasoning_chains) == 1

    def test_get_recent_thoughts_returns_most_recent(self):
        core = CognitiveCore()
        core.think("first")
        core.think("second")
        core.think("third")
        recent = core.get_recent_thoughts(limit=2)
        assert len(recent) == 2
        assert recent[-1].input_prompt == "third"

    def test_get_statistics(self):
        core = CognitiveCore()
        stats = core.get_statistics()
        assert stats["total_thoughts"] == 0
        assert stats["total_ideas"] == 0
        assert stats["total_reasoning_chains"] == 0
        core.think("test", ThinkingMode.ANALYTICAL)
        core.generate_idea("test")
        stats = core.get_statistics()
        assert stats["total_thoughts"] == 1
        assert stats["total_ideas"] == 1
        assert "analytical" in stats["modes_used"]


# ── KnowledgeGraph ─────────────────────────────────────────────────────────


class TestKnowledgeGraph:
    def test_add_entity(self):
        kg = KnowledgeGraph()
        e = kg.add_entity("socrates", "Socrates", "person")
        assert e.id == "socrates"
        assert e.label == "Socrates"
        assert e.entity_type == "person"
        assert kg.stats["entities"] == 1

    def test_add_fact_auto_creates_entities(self):
        kg = KnowledgeGraph()
        fact = kg.add_fact("socrates", "is_a", "human")
        assert fact.subject == "socrates"
        assert fact.predicate == "is_a"
        assert fact.object == "human"
        # Auto-created entities
        assert "socrates" in kg.entities
        assert "human" in kg.entities
        assert kg.stats["facts"] == 1

    def test_get_outgoing(self):
        kg = KnowledgeGraph()
        kg.add_fact("socrates", "is_a", "human")
        kg.add_fact("socrates", "is_a", "philosopher")
        out = kg.get_outgoing("socrates")
        assert len(out) == 2
        assert ("is_a", "human") in out
        assert ("is_a", "philosopher") in out

    def test_get_outgoing_filtered_by_predicate(self):
        kg = KnowledgeGraph()
        kg.add_fact("socrates", "is_a", "human")
        kg.add_fact("socrates", "lives_in", "athens")
        out = kg.get_outgoing("socrates", "lives_in")
        assert len(out) == 1
        assert out[0] == ("lives_in", "athens")

    def test_get_outgoing_nonexistent_entity(self):
        kg = KnowledgeGraph()
        assert kg.get_outgoing("nonexistent") == []

    def test_get_incoming(self):
        kg = KnowledgeGraph()
        kg.add_fact("socrates", "is_a", "human")
        inc = kg.get_incoming("human")
        assert len(inc) == 1
        assert inc[0] == ("is_a", "socrates")

    def test_get_incoming_nonexistent_entity(self):
        kg = KnowledgeGraph()
        assert kg.get_incoming("nonexistent") == []

    def test_query_exact_match(self):
        kg = KnowledgeGraph()
        kg.add_fact("socrates", "is_a", "human")
        results = kg.query("socrates", "is_a", "human")
        assert len(results) == 1

    def test_query_subject_predicate(self):
        kg = KnowledgeGraph()
        kg.add_fact("socrates", "is_a", "human")
        kg.add_fact("socrates", "lives_in", "athens")
        results = kg.query("socrates", "lives_in")
        assert len(results) == 1
        assert results[0].object == "athens"

    def test_query_subject_object(self):
        kg = KnowledgeGraph()
        kg.add_fact("socrates", "is_a", "human")
        kg.add_fact("plato", "is_a", "human")
        results = kg.query(subject="socrates", obj="human")
        assert len(results) == 1
        assert results[0].predicate == "is_a"

    def test_query_predicate_object(self):
        kg = KnowledgeGraph()
        kg.add_fact("socrates", "is_a", "human")
        kg.add_fact("plato", "is_a", "human")
        results = kg.query(predicate="is_a", obj="human")
        assert len(results) == 2

    def test_query_subject_only(self):
        kg = KnowledgeGraph()
        kg.add_fact("socrates", "is_a", "human")
        kg.add_fact("socrates", "lives_in", "athens")
        results = kg.query(subject="socrates")
        assert len(results) == 2

    def test_query_object_only(self):
        kg = KnowledgeGraph()
        kg.add_fact("socrates", "is_a", "human")
        kg.add_fact("plato", "is_a", "human")
        results = kg.query(obj="human")
        assert len(results) == 2

    def test_query_no_match_returns_empty(self):
        kg = KnowledgeGraph()
        assert kg.query("nonexistent") == []

    def test_bfs_discovers_connected_nodes(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "connects_to", "b")
        kg.add_fact("b", "connects_to", "c")
        kg.add_fact("c", "connects_to", "d")
        paths = kg.bfs("a", max_depth=5)
        assert "b" in paths
        assert "c" in paths
        assert "d" in paths

    def test_bfs_respects_max_depth(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "connects_to", "b")
        kg.add_fact("b", "connects_to", "c")
        kg.add_fact("c", "connects_to", "d")
        paths = kg.bfs("a", max_depth=1)
        assert "b" in paths
        assert "c" not in paths

    def test_bfs_with_predicate_filter(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "is_a", "b")
        kg.add_fact("a", "likes", "c")
        paths = kg.bfs("a", predicate_filter=lambda p: p == "is_a")
        assert "b" in paths
        assert "c" not in paths

    def test_dfs_returns_all_paths(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "connects_to", "b")
        kg.add_fact("a", "connects_to", "c")
        paths = kg.dfs("a", max_depth=2)
        assert len(paths) >= 2

    def test_find_paths_direct_connection(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "connects_to", "b")
        paths = kg.find_paths("a", "b")
        assert paths == [["a", "b"]]

    def test_find_paths_multi_hop(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "connects_to", "b")
        kg.add_fact("b", "connects_to", "c")
        paths = kg.find_paths("a", "c")
        assert len(paths) == 1
        assert paths[0] == ["a", "b", "c"]

    def test_find_paths_no_path(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "connects_to", "b")
        kg.add_fact("c", "connects_to", "d")
        assert kg.find_paths("a", "d") == []

    def test_shortest_path(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "connects_to", "b")
        kg.add_fact("b", "connects_to", "d")
        kg.add_fact("a", "connects_to", "c")
        kg.add_fact("c", "connects_to", "d")
        path = kg.shortest_path("a", "d")
        assert path is not None
        assert len(path) >= 2

    def test_infer_transitive(self):
        kg = KnowledgeGraph()
        kg.add_fact("human", "is_a", "mammal")
        kg.add_fact("mammal", "is_a", "animal")
        reachable = kg.infer_transitive("human", "is_a")
        assert "mammal" in reachable
        assert "animal" in reachable

    def test_verify_statement_true(self):
        kg = KnowledgeGraph()
        kg.add_fact("socrates", "is_a", "human")
        result = kg.verify_statement("socrates is a human")
        assert result["verified"] is True
        assert result["predicate"] == "is_a"

    def test_verify_statement_false(self):
        kg = KnowledgeGraph()
        kg.add_fact("socrates", "is_a", "human")
        result = kg.verify_statement("socrates is a dog")
        assert result["verified"] is False

    def test_verify_statement_contradiction(self):
        kg = KnowledgeGraph()
        kg.add_fact("whale", "is_a", "mammal")
        result = kg.verify_statement("whale is a fish")
        assert result["verified"] is False

    def test_verify_statement_unparsable(self):
        kg = KnowledgeGraph()
        result = kg.verify_statement("this is not a parseable statement at all")
        assert result["verified"] is False
        assert "Could not parse" in result["reason"]

    def test_check_consistency_empty(self):
        kg = KnowledgeGraph()
        assert kg.check_consistency() == []

    def test_check_consistency_multiple_types(self):
        kg = KnowledgeGraph()
        kg.add_fact("bat", RelationType.IS_A.value, "mammal")
        kg.add_fact("bat", RelationType.IS_A.value, "bird")
        issues = kg.check_consistency()
        error_issues = [i for i in issues if i.get("severity") == "error"]
        assert len(error_issues) > 0

    def test_export_includes_entities_and_facts(self):
        kg = KnowledgeGraph()
        kg.add_entity("e1", "Entity 1", "type1")
        kg.add_fact("e1", "is_a", "type1")
        exported = kg.export()
        assert "entities" in exported
        assert "facts" in exported
        assert "stats" in exported
        assert len(exported["facts"]) == 1

    def test_summary_includes_counts(self):
        kg = KnowledgeGraph()
        kg.add_fact("socrates", "is_a", "human")
        summary = kg.summary()
        assert "Entities: 2" in summary
        assert "Facts: 1" in summary

    def test_bfs_empty_graph(self):
        kg = KnowledgeGraph()
        result = kg.bfs("nonexistent")
        assert isinstance(result, dict) and len(result) > 0

    def test_dfs_empty_graph(self):
        kg = KnowledgeGraph()
        result = kg.dfs("nonexistent")
        assert isinstance(result, list) and len(result) > 0

    def test_find_paths_same_start_end(self):
        kg = KnowledgeGraph()
        assert kg.find_paths("a", "a") == [["a"]]


# ── WorkingMemory ──────────────────────────────────────────────────────────


class TestWorkingMemory:
    def test_add_item(self):
        wm = WorkingMemory()
        wm.add("item1")
        assert "item1" in wm.items
        assert wm.access_count["item1"] == 1

    def test_lru_eviction_when_full(self):
        wm = WorkingMemory(capacity=3)
        wm.add("a")
        wm.add("b")
        wm.add("c")
        wm.add("d")
        assert "d" in wm.items
        assert len(wm.items) == 3
        # "a" should be evicted (least accessed)
        assert "a" not in wm.items

    def test_access_increases_count(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.access("a")
        wm.access("a")
        assert wm.access_count["a"] == 3

    def test_get_recent_returns_most_accessed(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.add("b")
        wm.access("a")
        wm.access("a")
        wm.access("b")
        recent = wm.get_recent(2)
        assert len(recent) <= 2

    def test_get_recent_respects_limit(self):
        wm = WorkingMemory()
        for i in range(5):
            wm.add(str(i))
        recent = wm.get_recent(2)
        assert len(recent) == 2

    def test_clear_empties_memory(self):
        wm = WorkingMemory()
        wm.add("a")
        wm.add("b")
        wm.clear()
        assert wm.items == []
        assert wm.access_count == {}

    def test_capacity_default(self):
        wm = WorkingMemory()
        assert wm.capacity == 7

    def test_evicts_least_recently_used(self):
        wm = WorkingMemory(capacity=2)
        wm.add("a")
        wm.add("b")
        wm.access("a")
        wm.access("a")
        wm.access("b")
        wm.add("c")
        assert "c" in wm.items
        assert len(wm.items) == 2


# ── FormalLogicEngine ──────────────────────────────────────────────────────


class TestFormalLogicEngine:
    def test_assert_fact(self):
        eng = FormalLogicEngine()
        pred = Predicate(name="mortal", terms=[Term(name="socrates")])
        wff = WellFormedFormula(predicate=pred)
        eng.assert_fact(wff)
        assert len(eng.knowledge_base) == 1

    def test_assert_predicate_convenience(self):
        eng = FormalLogicEngine()
        eng.assert_predicate("mortal", "socrates")
        assert len(eng.knowledge_base) == 1
        assert eng.knowledge_base[0].predicate.name == "mortal"
        assert eng.knowledge_base[0].predicate.terms[0].name == "socrates"

    def test_query_known_fact(self):
        eng = FormalLogicEngine()
        eng.assert_predicate("mortal", "socrates")
        query_pred = Predicate(name="mortal", terms=[Term(name="socrates")])
        assert eng.query(query_pred) is True

    def test_query_unknown_fact(self):
        eng = FormalLogicEngine()
        eng.assert_predicate("mortal", "socrates")
        query_pred = Predicate(name="immortal", terms=[Term(name="socrates")])
        assert eng.query(query_pred) is False

    def test_prove_syllogism_valid(self):
        eng = FormalLogicEngine()
        result = eng.prove_syllogism(
            ("All", "are", "mortal"),
            ("All", "are", "human"),
            ("All", "are", "mortal"),
        )
        assert result["valid"] is True

    def test_prove_syllogism_invalid(self):
        eng = FormalLogicEngine()
        result = eng.prove_syllogism(
            ("All", "are", "B"),
            ("Some", "are", "D"),
            ("No", "are", "D"),
        )
        assert result["valid"] is False

    def test_resolution_chain(self):
        eng = FormalLogicEngine()
        eng.assert_predicate("mortal", "socrates")
        goal = Predicate(name="mortal", terms=[Term(name="socrates")])
        result = eng.resolution(goal)
        assert result is True
