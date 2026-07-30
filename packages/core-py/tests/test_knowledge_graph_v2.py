"""Tests for production knowledge graph."""

import pytest
from domains.cognitive.knowledge_graph_v2 import (
    Entity,
    Fact,
    RelationType,
    KnowledgeGraph,
)


class TestEntity:
    def test_defaults(self):
        e = Entity(id="e1", label="test", entity_type="concept")
        assert e.properties == {}
        assert e.aliases == set()
        assert e.confidence == 1.0

    def test_with_properties_and_aliases(self):
        e = Entity(id="e1", label="test", entity_type="concept",
                   properties={"color": "red"}, aliases={"t"})
        assert e.properties["color"] == "red"
        assert "t" in e.aliases

    def test_hash_based_on_id(self):
        e1 = Entity(id="e1", label="a", entity_type="t")
        e2 = Entity(id="e2", label="b", entity_type="t")
        assert hash(e1) != hash(e2)


class TestFact:
    def test_defaults(self):
        f = Fact(subject="s", predicate="p", object="o")
        assert f.confidence == 1.0
        assert f.source == "unknown"
        assert f.verified is False

    def test_repr(self):
        f = Fact(subject="s", predicate="p", object="o")
        assert repr(f) == "(s, p, o)"

    def test_custom_values(self):
        f = Fact(subject="s", predicate="p", object="o", confidence=0.5, source="wiki", verified=True)
        assert f.confidence == 0.5
        assert f.source == "wiki"
        assert f.verified is True


class TestRelationType:
    def test_values(self):
        assert RelationType.IS_A.value == "rdf:type"
        assert RelationType.PART_OF.value == "part_of"
        assert RelationType.CAUSES.value == "causes"
        assert RelationType.SIMILAR_TO.value == "similar_to"
        assert RelationType.OPPOSITE_OF.value == "opposite_of"

    def test_members_count(self):
        assert len(RelationType) == 9


class TestKnowledgeGraph:
    def test_initial_state(self):
        kg = KnowledgeGraph()
        assert kg.entities == {}
        assert kg.facts == {}
        assert kg.stats["entities"] == 0
        assert kg.stats["facts"] == 0

    def test_add_entity(self):
        kg = KnowledgeGraph()
        e = kg.add_entity("python", "Python", "language")
        assert e.id == "python"
        assert kg.entities["python"].label == "Python"
        assert kg.stats["entities"] == 1

    def test_add_entity_with_properties(self):
        kg = KnowledgeGraph()
        kg.add_entity("python", "Python", "language", properties={"creator": "Guido"})
        assert kg.entities["python"].properties["creator"] == "Guido"

    def test_add_entity_with_aliases(self):
        kg = KnowledgeGraph()
        kg.add_entity("python", "Python", "language", aliases=["py", "CPython"])
        assert "py" in kg.entities["python"].aliases

    def test_add_fact_creates_entities(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "is_a", "animal")
        assert "cat" in kg.entities
        assert "animal" in kg.entities
        assert kg.stats["facts"] == 1

    def test_add_fact_custom_confidence(self):
        kg = KnowledgeGraph()
        f = kg.add_fact("cat", "is_a", "animal", confidence=0.9, source="expert")
        assert f.confidence == 0.9
        assert f.source == "expert"

    def test_add_duplicate_fact_does_not_duplicate(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "is_a", "animal")
        kg.add_fact("cat", "is_a", "animal")
        assert kg.stats["facts"] == 1

    def test_get_outgoing_all(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "is_a", "animal")
        kg.add_fact("cat", "likes", "fish")
        outgoing = kg.get_outgoing("cat")
        assert len(outgoing) == 2

    def test_get_outgoing_filtered(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "is_a", "animal")
        kg.add_fact("cat", "likes", "fish")
        outgoing = kg.get_outgoing("cat", "is_a")
        assert outgoing == [("is_a", "animal")]

    def test_get_outgoing_nonexistent(self):
        kg = KnowledgeGraph()
        assert kg.get_outgoing("nonexistent") == []

    def test_get_incoming_all(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "is_a", "animal")
        kg.add_fact("dog", "is_a", "animal")
        incoming = kg.get_incoming("animal")
        assert len(incoming) == 2

    def test_get_incoming_filtered(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "is_a", "animal")
        incoming = kg.get_incoming("animal", "is_a")
        assert incoming == [("is_a", "cat")]

    def test_get_incoming_nonexistent(self):
        kg = KnowledgeGraph()
        assert kg.get_incoming("nonexistent") == []

    def test_query_subject_predicate_object_exact(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "is_a", "animal")
        results = kg.query(subject="cat", predicate="is_a", obj="animal")
        assert len(results) == 1

    def test_query_subject_predicate(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "is_a", "animal")
        kg.add_fact("cat", "likes", "fish")
        results = kg.query(subject="cat", predicate="is_a")
        assert len(results) == 1

    def test_query_subject_object(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "is_a", "animal")
        kg.add_fact("cat", "likes", "fish")
        results = kg.query(subject="cat", obj="fish")
        assert len(results) == 1

    def test_query_predicate_object(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "is_a", "animal")
        kg.add_fact("dog", "is_a", "animal")
        results = kg.query(predicate="is_a", obj="animal")
        assert len(results) == 2

    def test_query_subject_all(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "is_a", "animal")
        kg.add_fact("cat", "likes", "fish")
        results = kg.query(subject="cat")
        assert len(results) == 2

    def test_query_object_all(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "is_a", "animal")
        kg.add_fact("dog", "is_a", "animal")
        results = kg.query(obj="animal")
        assert len(results) == 2

    def test_query_no_match(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "is_a", "animal")
        assert kg.query(subject="dog") == []

    def test_bfs_basic(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "related_to", "b")
        kg.add_fact("b", "related_to", "c")
        paths = kg.bfs("a", max_depth=3)
        assert "a" in paths
        assert "b" in paths
        assert "c" in paths

    def test_bfs_with_predicate_filter(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "related_to", "b")
        kg.add_fact("a", "causes", "c")
        paths = kg.bfs("a", predicate_filter=lambda p: p == "related_to", max_depth=3)
        assert "b" in paths
        assert "c" not in paths

    def test_dfs_basic(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "related_to", "b")
        kg.add_fact("b", "related_to", "c")
        paths = kg.dfs("a", max_depth=3)
        assert len(paths) >= 1

    def test_dfs_with_predicate_filter(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "related_to", "b")
        kg.add_fact("a", "causes", "c")
        paths = kg.dfs("a", predicate_filter=lambda p: p == "related_to", max_depth=3)
        assert len(paths) >= 1

    def test_find_paths_direct(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "related_to", "b")
        paths = kg.find_paths("a", "b")
        assert len(paths) == 1
        assert paths[0] == ["a", "b"]

    def test_find_paths_indirect(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "related_to", "b")
        kg.add_fact("b", "related_to", "c")
        paths = kg.find_paths("a", "c", max_length=5)
        assert len(paths) == 1
        assert "a" in paths[0]
        assert "c" in paths[0]

    def test_find_paths_no_path(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "related_to", "b")
        kg.add_fact("c", "related_to", "d")
        paths = kg.find_paths("a", "d", max_length=5)
        assert paths == []

    def test_find_paths_same_entity(self):
        kg = KnowledgeGraph()
        paths = kg.find_paths("a", "a")
        assert paths == [["a"]]

    def test_shortest_path(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "related_to", "b")
        kg.add_fact("b", "related_to", "c")
        kg.add_fact("a", "related_to", "c")
        path = kg.shortest_path("a", "c")
        assert path == ["a", "c"]

    def test_shortest_path_no_path(self):
        kg = KnowledgeGraph()
        assert kg.shortest_path("a", "z") is None

    def test_infer_transitive(self):
        kg = KnowledgeGraph()
        kg.add_fact("human", "is_a", "mammal", source="wiki")
        kg.add_fact("mammal", "is_a", "animal", source="wiki")
        reachable = kg.infer_transitive("human", "is_a", max_depth=5)
        assert "mammal" in reachable
        assert "animal" in reachable

    def test_infer_transitive_no_results(self):
        kg = KnowledgeGraph()
        reachable = kg.infer_transitive("unknown", "is_a")
        assert reachable == set()

    def test_verify_statement_true(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "is_a", "animal", source="wiki")
        result = kg.verify_statement("cat is a animal")
        assert result["verified"] is True
        assert result["confidence"] == 1.0

    def test_verify_statement_false(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "is_a", "animal", source="wiki")
        result = kg.verify_statement("cat is a vegetable")
        assert result["verified"] is False

    def test_verify_statement_contradiction(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "is_a", "animal", source="wiki")
        kg.add_fact("cat", "is_a", "mineral", source="other")
        result = kg.verify_statement("cat is a animal")
        assert result["verified"] is True

    def test_verify_statement_unparsable(self):
        kg = KnowledgeGraph()
        result = kg.verify_statement("hello world")
        assert result["verified"] is False
        assert "Could not parse" in result["reason"]

    def test_check_consistency_empty(self):
        kg = KnowledgeGraph()
        issues = kg.check_consistency()
        assert issues == []

    def test_check_consistency_multiple_types(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "rdf:type", "animal")
        kg.add_fact("cat", "rdf:type", "mineral")
        issues = kg.check_consistency()
        type_issues = [i for i in issues if i["type"] == "multiple_types"]
        assert len(type_issues) >= 1

    def test_check_consistency_no_issues(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "rdf:type", "animal")
        issues = kg.check_consistency()
        type_issues = [i for i in issues if i["type"] == "multiple_types"]
        assert len(type_issues) == 0

    def test_export(self):
        kg = KnowledgeGraph()
        kg.add_entity("cat", "Cat", "animal")
        kg.add_fact("cat", "is_a", "animal")
        exported = kg.export()
        assert "entities" in exported
        assert "facts" in exported
        assert "stats" in exported
        assert exported["stats"]["entities"] == 2
        assert exported["stats"]["facts"] == 1

    def test_export_fact_fields(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "is_a", "animal", confidence=0.9, source="wiki")
        exported = kg.export()
        fact = exported["facts"][0]
        assert fact["subject"] == "cat"
        assert fact["predicate"] == "is_a"
        assert fact["object"] == "animal"
        assert fact["confidence"] == 0.9

    def test_summary(self):
        kg = KnowledgeGraph()
        kg.add_fact("cat", "is_a", "animal")
        summary = kg.summary()
        assert "Entities" in summary
        assert "Facts" in summary
        assert "Top Relations" in summary

    def test_summary_empty(self):
        kg = KnowledgeGraph()
        summary = kg.summary()
        assert "Entities: 0" in summary
        assert "Facts: 0" in summary

    def test_large_graph(self):
        kg = KnowledgeGraph()
        for i in range(100):
            kg.add_fact(f"entity_{i}", "related_to", f"entity_{(i+1)%100}")
        assert kg.stats["entities"] == 100
        assert kg.stats["facts"] == 100

    def test_find_paths_uses_predicate_filter(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "related_to", "b")
        kg.add_fact("a", "causes", "c")
        kg.add_fact("c", "related_to", "d")
        paths = kg.find_paths("a", "b", predicate_filter=lambda p: p == "related_to")
        assert len(paths) >= 1

    def test_shortest_path_length(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "related_to", "b")
        kg.add_fact("b", "related_to", "c")
        kg.add_fact("a", "related_to", "c")
        path = kg.shortest_path("a", "c")
        assert len(path) == 2
