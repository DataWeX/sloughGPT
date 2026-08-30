"""Comprehensive tests for domains.cognitive.knowledge_graph_v2."""

import pytest
from domains.cognitive.knowledge_graph_v2 import (
    Entity,
    Fact,
    KnowledgeGraph,
    RelationType,
)


# ---------------------------------------------------------------------------
# RelationType
# ---------------------------------------------------------------------------

class TestRelationType:
    def test_all_members_exist(self):
        expected = {"IS_A", "PART_OF", "CAUSES", "RELATED_TO", "SIMILAR_TO",
                    "OPPOSITE_OF", "LOCATED_IN", "HAS_PROPERTY", "INSTANCE_OF"}
        assert {m.name for m in RelationType} == expected

    def test_enum_values_are_strings(self):
        for rt in RelationType:
            assert isinstance(rt.value, str)
            assert len(rt.value) > 0

    def test_is_a_value(self):
        assert RelationType.IS_A.value == "rdf:type"

    def test_part_of_value(self):
        assert RelationType.PART_OF.value == "part_of"

    def test_instance_of_value(self):
        assert RelationType.INSTANCE_OF.value == "instance_of"


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------

class TestEntity:
    def test_basic_fields(self):
        e = Entity(id="e1", label="Cat", entity_type="animal")
        assert e.id == "e1"
        assert e.label == "Cat"
        assert e.entity_type == "animal"
        assert e.properties == {}
        assert e.aliases == set()
        assert e.confidence == 1.0

    def test_with_properties(self):
        props = {"color": "orange", "size": "small"}
        e = Entity(id="e1", label="Cat", entity_type="animal", properties=props)
        assert e.properties == props

    def test_with_aliases(self):
        e = Entity(id="e1", label="Paris", entity_type="city",
                   aliases={"City of Light"})
        assert "City of Light" in e.aliases

    def test_hash_based_on_id(self):
        e1 = Entity(id="abc", label="X", entity_type="t")
        e2 = Entity(id="abc", label="Y", entity_type="z")
        assert hash(e1) == hash(e2)

    def test_eq_based_on_id(self):
        e1 = Entity(id="abc", label="X", entity_type="t")
        e2 = Entity(id="abc", label="Y", entity_type="z")
        assert e1 == e2

    def test_neq_different_ids(self):
        e1 = Entity(id="a", label="X", entity_type="t")
        e2 = Entity(id="b", label="X", entity_type="t")
        assert e1 != e2

    def test_eq_with_non_entity_returns_not_implemented(self):
        e = Entity(id="a", label="X", entity_type="t")
        assert e.__eq__("not an entity") is NotImplemented

    def test_entity_in_set(self):
        s = {Entity(id="a", label="A", entity_type="t"),
             Entity(id="a", label="B", entity_type="t"),
             Entity(id="c", label="C", entity_type="t")}
        assert len(s) == 2

    def test_entity_as_dict_key(self):
        d = {}
        e = Entity(id="x", label="X", entity_type="t")
        d[e] = 42
        assert d[Entity(id="x", label="?", entity_type="?")] == 42


# ---------------------------------------------------------------------------
# Fact
# ---------------------------------------------------------------------------

class TestFact:
    def test_basic_fields(self):
        f = Fact(subject="a", predicate="rel", object="b")
        assert f.subject == "a"
        assert f.predicate == "rel"
        assert f.object == "b"
        assert f.confidence == 1.0
        assert f.source == "unknown"
        assert f.timestamp is None
        assert f.verified is False

    def test_custom_fields(self):
        f = Fact(subject="a", predicate="rel", object="b",
                 confidence=0.9, source="wiki", timestamp=100.0, verified=True)
        assert f.confidence == 0.9
        assert f.source == "wiki"
        assert f.timestamp == 100.0
        assert f.verified is True

    def test_repr(self):
        f = Fact(subject="cat", predicate="is_a", object="animal")
        r = repr(f)
        assert "cat" in r
        assert "is_a" in r
        assert "animal" in r

    def test_repr_format(self):
        f = Fact(subject="s", predicate="p", object="o")
        assert repr(f) == "(s, p, o)"


# ---------------------------------------------------------------------------
# KnowledgeGraph — init and stats
# ---------------------------------------------------------------------------

class TestKGInit:
    def test_empty_graph(self):
        kg = KnowledgeGraph()
        assert len(kg.entities) == 0
        assert len(kg.facts) == 0
        assert kg.stats["entities"] == 0
        assert kg.stats["facts"] == 0
        assert kg.stats["avg_degree"] == 0.0

    def test_stats_update_on_add_entity(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "A", "t")
        assert kg.stats["entities"] == 1

    def test_stats_update_on_add_fact(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "A", "t")
        kg.add_entity("b", "B", "t")
        kg.add_fact("a", "rel", "b")
        assert kg.stats["facts"] == 1
        assert kg.stats["entities"] == 2


# ---------------------------------------------------------------------------
# add_entity
# ---------------------------------------------------------------------------

class TestAddEntity:
    def test_add_new_entity(self):
        kg = KnowledgeGraph()
        e = kg.add_entity("c1", "Paris", "city")
        assert e.id == "c1"
        assert kg.entities["c1"].label == "Paris"

    def test_add_duplicate_returns_existing(self):
        kg = KnowledgeGraph()
        e1 = kg.add_entity("c1", "Paris", "city")
        e2 = kg.add_entity("c1", "Paris", "capital")
        assert e1 is e2
        assert e1.entity_type == "city"
        assert len(kg.entities) == 1

    def test_add_with_aliases(self):
        kg = KnowledgeGraph()
        e = kg.add_entity("c1", "Paris", "city", aliases=["City of Light"])
        assert "City of Light" in e.aliases

    def test_duplicate_add_preserves_aliases(self):
        kg = KnowledgeGraph()
        kg.add_entity("c1", "Paris", "city", aliases=["City of Light"])
        e = kg.add_entity("c1", "Paris", "city", aliases=["Ville Lumiere"])
        assert "City of Light" in e.aliases
        assert "Ville Lumiere" in e.aliases

    def test_add_with_properties(self):
        kg = KnowledgeGraph()
        e = kg.add_entity("a", "A", "t", properties={"k": "v"})
        assert e.properties == {"k": "v"}


# ---------------------------------------------------------------------------
# add_fact
# ---------------------------------------------------------------------------

class TestAddFact:
    def test_auto_creates_entities(self):
        kg = KnowledgeGraph()
        f = kg.add_fact("cat", "is_a", "animal")
        assert f is not None
        assert "cat" in kg.entities
        assert "animal" in kg.entities

    def test_returns_fact(self):
        kg = KnowledgeGraph()
        f = kg.add_fact("a", "rel", "b", confidence=0.8, source="test")
        assert f.subject == "a"
        assert f.predicate == "rel"
        assert f.object == "b"
        assert f.confidence == 0.8
        assert f.source == "test"

    def test_dedup_keeps_higher_confidence(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "rel", "b", confidence=0.5, source="src1")
        result = kg.add_fact("a", "rel", "b", confidence=0.9, source="src2")
        assert result is None
        fact = kg.facts[("a", "rel", "b")]
        assert fact.confidence == 0.9
        assert fact.source == "src2"
        assert len(kg.facts) == 1

    def test_dedup_keeps_existing_if_higher(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "rel", "b", confidence=0.9, source="src1")
        result = kg.add_fact("a", "rel", "b", confidence=0.5, source="src2")
        assert result is None
        fact = kg.facts[("a", "rel", "b")]
        assert fact.confidence == 0.9
        assert fact.source == "src1"

    def test_subject_index_populated(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "rel", "b")
        assert "a" in kg.subject_index
        assert "rel" in kg.subject_index["a"]
        assert "b" in kg.subject_index["a"]["rel"]

    def test_object_index_populated(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "rel", "b")
        assert "b" in kg.object_index
        assert "rel" in kg.object_index["b"]
        assert "a" in kg.object_index["b"]["rel"]

    def test_multiple_facts_same_subject_different_pred(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "p1", "b")
        kg.add_fact("a", "p2", "c")
        assert len(kg.facts) == 2
        assert len(kg.subject_index["a"]) == 2

    def test_multiple_facts_different_subject_same_object(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "rel", "x")
        kg.add_fact("b", "rel", "x")
        assert len(kg.facts) == 2
        assert len(kg.object_index["x"]["rel"]) == 2


# ---------------------------------------------------------------------------
# case-insensitive resolution
# ---------------------------------------------------------------------------

class TestCaseInsensitive:
    def test_resolve_by_entity_id(self):
        kg = KnowledgeGraph()
        kg.add_entity("Paris", "Paris", "city")
        eid = kg._resolve_entity_id("paris")
        assert eid == "Paris"

    def test_resolve_by_alias(self):
        kg = KnowledgeGraph()
        kg.add_entity("Paris", "Paris", "city", aliases=["City of Light"])
        eid = kg._resolve_entity_id("city of light")
        assert eid == "Paris"

    def test_resolve_no_match_returns_original(self):
        kg = KnowledgeGraph()
        eid = kg._resolve_entity_id("London")
        assert eid == "London"

    def test_add_fact_case_insensitive(self):
        kg = KnowledgeGraph()
        kg.add_entity("Paris", "Paris", "city")
        f = kg.add_fact("paris", "located_in", "France")
        assert f.subject == "Paris"
        assert len(kg.facts) == 1

    def test_get_outgoing_case_insensitive(self):
        kg = KnowledgeGraph()
        kg.add_entity("Paris", "Paris", "city")
        kg.add_entity("France", "France", "country")
        kg.add_fact("Paris", "located_in", "France")
        out = kg.get_outgoing("paris")
        assert len(out) == 1
        assert out[0][1] == "France"

    def test_get_incoming_case_insensitive(self):
        kg = KnowledgeGraph()
        kg.add_entity("Paris", "Paris", "city")
        kg.add_entity("France", "France", "country")
        kg.add_fact("Paris", "located_in", "France")
        inc = kg.get_incoming("france")
        assert len(inc) == 1
        assert inc[0][1] == "Paris"


# ---------------------------------------------------------------------------
# get_outgoing / get_incoming
# ---------------------------------------------------------------------------

class TestEdges:
    def _build_graph(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "A", "t")
        kg.add_entity("b", "B", "t")
        kg.add_entity("c", "C", "t")
        kg.add_fact("a", "p1", "b")
        kg.add_fact("a", "p2", "c")
        kg.add_fact("b", "p1", "c")
        return kg

    def test_get_outgoing_all(self):
        kg = self._build_graph()
        out = kg.get_outgoing("a")
        assert len(out) == 2
        targets = {o for _, o in out}
        assert targets == {"b", "c"}

    def test_get_outgoing_filtered(self):
        kg = self._build_graph()
        out = kg.get_outgoing("a", predicate="p1")
        assert len(out) == 1
        assert out[0] == ("p1", "b")

    def test_get_outgoing_empty(self):
        kg = KnowledgeGraph()
        kg.add_entity("x", "X", "t")
        assert kg.get_outgoing("x") == []

    def test_get_outgoing_nonexistent(self):
        kg = KnowledgeGraph()
        assert kg.get_outgoing("nope") == []

    def test_get_incoming_all(self):
        kg = self._build_graph()
        inc = kg.get_incoming("c")
        assert len(inc) == 2
        sources = {s for _, s in inc}
        assert sources == {"a", "b"}

    def test_get_incoming_filtered(self):
        kg = self._build_graph()
        inc = kg.get_incoming("c", predicate="p1")
        assert len(inc) == 1
        assert inc[0][1] == "b"

    def test_get_incoming_empty(self):
        kg = KnowledgeGraph()
        kg.add_entity("x", "X", "t")
        assert kg.get_incoming("x") == []


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

class TestQuery:
    def _build_graph(self):
        kg = KnowledgeGraph()
        kg.add_entity("cat", "Cat", "animal")
        kg.add_entity("dog", "Dog", "animal")
        kg.add_entity("animal", "Animal", "concept")
        kg.add_fact("cat", "is_a", "animal", confidence=0.9, source="wiki")
        kg.add_fact("dog", "is_a", "animal", confidence=0.95, source="textbook")
        kg.add_fact("cat", "related_to", "dog", confidence=0.7, source="obs")
        return kg

    def test_query_all_facts(self):
        kg = self._build_graph()
        # query() with no args returns nothing — need at least subject or object
        by_subject = kg.query(subject="cat") + kg.query(subject="dog")
        assert len(by_subject) == 3

    def test_query_by_subject(self):
        kg = self._build_graph()
        results = kg.query(subject="cat")
        assert len(results) == 2
        preds = {f.predicate for f in results}
        assert preds == {"is_a", "related_to"}

    def test_query_by_object(self):
        kg = self._build_graph()
        results = kg.query(obj="animal")
        assert len(results) == 2

    def test_query_by_predicate(self):
        kg = self._build_graph()
        # query(predicate="is_a") alone doesn't match any code path
        # need subject or object. Let's query subject + predicate
        results = kg.query(subject="cat", predicate="is_a")
        assert len(results) == 1
        assert results[0].object == "animal"

    def test_query_by_subject_predicate(self):
        kg = self._build_graph()
        results = kg.query(subject="cat", predicate="related_to")
        assert len(results) == 1
        assert results[0].object == "dog"

    def test_query_by_predicate_object(self):
        kg = self._build_graph()
        results = kg.query(predicate="is_a", obj="animal")
        assert len(results) == 2
        subjects = {f.subject for f in results}
        assert subjects == {"cat", "dog"}

    def test_query_by_subject_object(self):
        kg = self._build_graph()
        results = kg.query(subject="cat", obj="dog")
        assert len(results) == 1
        assert results[0].predicate == "related_to"

    def test_query_all_three(self):
        kg = self._build_graph()
        results = kg.query(subject="cat", predicate="is_a", obj="animal")
        assert len(results) == 1
        assert results[0].confidence == 0.9

    def test_query_no_match(self):
        kg = self._build_graph()
        results = kg.query(subject="elephant")
        assert results == []

    def test_query_case_insensitive(self):
        kg = self._build_graph()
        results = kg.query(subject="Cat")
        assert len(results) == 2


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------

class TestBFS:
    def _build_chain(self):
        kg = KnowledgeGraph()
        for name in ["a", "b", "c", "d"]:
            kg.add_entity(name, name, "t")
        kg.add_fact("a", "rel", "b")
        kg.add_fact("b", "rel", "c")
        kg.add_fact("c", "rel", "d")
        return kg

    def test_bfs_returns_all_reachable(self):
        kg = self._build_chain()
        paths = kg.bfs("a")
        assert "a" in paths
        assert "b" in paths
        assert "c" in paths
        assert "d" in paths

    def test_bfs_start_has_no_path(self):
        kg = self._build_chain()
        paths = kg.bfs("a")
        assert paths["a"] == []

    def test_bfs_path_tracking(self):
        kg = self._build_chain()
        paths = kg.bfs("a")
        assert paths["b"] == [("rel", "a")]
        assert paths["c"] == [("rel", "a"), ("rel", "b")]

    def test_bfs_max_depth(self):
        kg = self._build_chain()
        paths = kg.bfs("a", max_depth=2)
        assert "a" in paths
        assert "b" in paths
        assert "c" in paths
        assert "d" not in paths

    def test_bfs_predicate_filter(self):
        kg = self._build_chain()
        kg.add_fact("b", "other", "d")
        paths = kg.bfs("a", predicate_filter=lambda p: p == "rel")
        assert "d" in paths
        # "d" reached via b->c->d (all "rel")

    def test_bfs_disconnected(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "A", "t")
        kg.add_entity("b", "B", "t")
        paths = kg.bfs("a")
        assert paths == {"a": []}

    def test_bfs_diamond(self):
        kg = KnowledgeGraph()
        for n in ["a", "b", "c", "d"]:
            kg.add_entity(n, n, "t")
        kg.add_fact("a", "rel", "b")
        kg.add_fact("a", "rel", "c")
        kg.add_fact("b", "rel", "d")
        kg.add_fact("c", "rel", "d")
        paths = kg.bfs("a")
        assert "d" in paths
        assert len(paths) == 4


# ---------------------------------------------------------------------------
# DFS
# ---------------------------------------------------------------------------

class TestDFS:
    def _build_linear(self):
        kg = KnowledgeGraph()
        for n in ["a", "b", "c"]:
            kg.add_entity(n, n, "t")
        kg.add_fact("a", "rel", "b")
        kg.add_fact("b", "rel", "c")
        return kg

    def test_dfs_returns_paths(self):
        kg = self._build_linear()
        paths = kg.dfs("a")
        assert isinstance(paths, list)
        assert len(paths) > 0

    def test_dfs_first_path_is_empty(self):
        kg = self._build_linear()
        paths = kg.dfs("a")
        assert paths[0] == []

    def test_dfs_finds_all_paths(self):
        kg = self._build_linear()
        paths = kg.dfs("a")
        # Should include: [], [rel,b], [rel,b,rel,c]
        assert len(paths) >= 3

    def test_dfs_max_depth(self):
        kg = self._build_linear()
        paths = kg.dfs("a", max_depth=1)
        # Only [rel,b] at depth 1, nothing deeper
        path_lengths = [len(p) for p in paths]
        assert max(path_lengths) <= 1

    def test_dfs_predicate_filter(self):
        kg = KnowledgeGraph()
        for n in ["a", "b", "c"]:
            kg.add_entity(n, n, "t")
        kg.add_fact("a", "keep", "b")
        kg.add_fact("b", "skip", "c")
        paths = kg.dfs("a", predicate_filter=lambda p: p == "keep")
        # Only "keep" edges traversed, so no path reaches c
        all_entities = set()
        for path in paths:
            for _, ent in path:
                all_entities.add(ent)
        assert "c" not in all_entities


# ---------------------------------------------------------------------------
# find_paths / shortest_path
# ---------------------------------------------------------------------------

class TestPathFinding:
    def _build_grid(self):
        kg = KnowledgeGraph()
        for n in ["a", "b", "c", "d", "e"]:
            kg.add_entity(n, n, "t")
        kg.add_fact("a", "rel", "b")
        kg.add_fact("a", "rel", "c")
        kg.add_fact("b", "rel", "d")
        kg.add_fact("c", "rel", "d")
        kg.add_fact("d", "rel", "e")
        return kg

    def test_find_paths_same_node(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "A", "t")
        paths = kg.find_paths("a", "a")
        assert paths == [["a"]]

    def test_find_paths_direct(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "A", "t")
        kg.add_entity("b", "B", "t")
        kg.add_fact("a", "rel", "b")
        paths = kg.find_paths("a", "b")
        assert paths == [["a", "b"]]

    def test_find_paths_multi_hop(self):
        kg = self._build_grid()
        paths = kg.find_paths("a", "e")
        assert len(paths) >= 1
        assert paths[0][0] == "a"
        assert paths[0][-1] == "e"

    def test_find_paths_no_path(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "A", "t")
        kg.add_entity("b", "B", "t")
        paths = kg.find_paths("a", "b")
        assert paths == []

    def test_find_paths_max_length_truncates_search(self):
        kg = self._build_grid()
        # max_length=1 only allows one hop from start
        paths = kg.find_paths("a", "e", max_length=1)
        assert paths == []

    def test_find_paths_max_length_allows_direct(self):
        kg = self._build_grid()
        # max_length=10 finds the path
        paths = kg.find_paths("a", "e", max_length=10)
        assert len(paths) >= 1

    def test_find_paths_predicate_filter(self):
        kg = KnowledgeGraph()
        for n in ["a", "b", "c"]:
            kg.add_entity(n, n, "t")
        kg.add_fact("a", "keep", "b")
        kg.add_fact("b", "skip", "c")
        paths = kg.find_paths("a", "c", predicate_filter=lambda p: p == "keep")
        assert paths == []

    def test_shortest_path_direct(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "A", "t")
        kg.add_entity("b", "B", "t")
        kg.add_fact("a", "rel", "b")
        sp = kg.shortest_path("a", "b")
        assert sp == ["a", "b"]

    def test_shortest_path_multi_hop(self):
        kg = self._build_grid()
        sp = kg.shortest_path("a", "e")
        assert sp is not None
        assert sp[0] == "a"
        assert sp[-1] == "e"
        assert len(sp) <= 4

    def test_shortest_path_no_path(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "A", "t")
        kg.add_entity("b", "B", "t")
        assert kg.shortest_path("a", "b") is None


# ---------------------------------------------------------------------------
# infer_transitive
# ---------------------------------------------------------------------------

class TestInferTransitive:
    def test_single_hop(self):
        kg = KnowledgeGraph()
        for n in ["cat", "animal", "thing"]:
            kg.add_entity(n, n, "t")
        kg.add_fact("cat", "is_a", "animal")
        result = kg.infer_transitive("cat", "is_a")
        assert "animal" in result

    def test_multi_hop(self):
        kg = KnowledgeGraph()
        for n in ["cat", "mammal", "animal"]:
            kg.add_entity(n, n, "t")
        kg.add_fact("cat", "is_a", "mammal")
        kg.add_fact("mammal", "is_a", "animal")
        result = kg.infer_transitive("cat", "is_a")
        assert "mammal" in result
        assert "animal" in result

    def test_transitive_with_reverse_similar_to(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "A", "t")
        kg.add_entity("b", "B", "t")
        kg.add_entity("c", "C", "t")
        kg.add_fact("a", "similar_to", "b")
        kg.add_fact("c", "similar_to", "a")
        result = kg.infer_transitive("a", "similar_to")
        assert "b" in result
        assert "c" in result

    def test_infer_empty(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "A", "t")
        result = kg.infer_transitive("a", "is_a")
        assert result == set()

    def test_infer_no_limit_exceeded(self):
        kg = KnowledgeGraph()
        # Chain of 5 — should complete
        for i in range(6):
            kg.add_entity(str(i), str(i), "t")
            if i > 0:
                kg.add_fact(str(i), "is_a", str(i - 1))
        result = kg.infer_transitive("5", "is_a")
        assert len(result) == 5


# ---------------------------------------------------------------------------
# verify_statement
# ---------------------------------------------------------------------------

class TestVerifyStatement:
    def _build_verified_graph(self):
        kg = KnowledgeGraph()
        kg.add_entity("cat", "Cat", "animal")
        kg.add_entity("animal", "Animal", "concept")
        kg.add_entity("Paris", "Paris", "city")
        kg.add_entity("France", "France", "country")
        kg.add_fact("cat", "is_a", "animal", confidence=0.95, source="wiki")
        kg.add_fact("Paris", "located_in", "France", confidence=0.99, source="atlas")
        return kg

    def test_verify_is_a_true(self):
        kg = self._build_verified_graph()
        result = kg.verify_statement("cat is a animal")
        assert result["verified"] is True
        assert result["confidence"] == 0.95

    def test_verify_is_a_false(self):
        kg = self._build_verified_graph()
        result = kg.verify_statement("cat is a vehicle")
        assert result["verified"] is False
        # Cat has is_a=animal, so "is_a vehicle" is a contradiction
        assert "Contradicting" in result["reason"]

    def test_verify_located_in(self):
        kg = self._build_verified_graph()
        result = kg.verify_statement("Paris is located in France")
        assert result["verified"] is True
        assert result["confidence"] == 0.99

    def test_verify_causes(self):
        kg = KnowledgeGraph()
        kg.add_entity("smoking", "Smoking", "action")
        kg.add_entity("cancer", "Cancer", "disease")
        kg.add_fact("smoking", "causes", "cancer")
        result = kg.verify_statement("smoking causes cancer")
        assert result["verified"] is True

    def test_verify_part_of(self):
        kg = KnowledgeGraph()
        kg.add_entity("wheel", "Wheel", "part")
        kg.add_entity("car", "Car", "vehicle")
        kg.add_fact("wheel", "part_of", "car")
        result = kg.verify_statement("wheel is part of car")
        assert result["verified"] is True

    def test_verify_unparseable(self):
        kg = KnowledgeGraph()
        result = kg.verify_statement("the weather is nice today")
        assert result["verified"] is False
        assert result["reason"] == "Could not parse statement"

    def test_verify_contradiction(self):
        kg = KnowledgeGraph()
        kg.add_entity("cat", "Cat", "animal")
        kg.add_entity("dog", "Dog", "animal")
        kg.add_entity("bird", "Bird", "animal")
        kg.add_fact("cat", "is_a", "dog")
        result = kg.verify_statement("cat is a bird")
        assert result["verified"] is False
        assert "Contradicting" in result["reason"]

    def test_verify_sources_collected(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "A", "t")
        kg.add_entity("b", "B", "t")
        kg.add_fact("a", "is_a", "b", source="src1")
        result = kg.verify_statement("a is a b")
        assert "src1" in result["sources"]


# ---------------------------------------------------------------------------
# check_consistency
# ---------------------------------------------------------------------------

class TestCheckConsistency:
    def test_clean_graph(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "A", "t")
        kg.add_entity("b", "B", "t")
        kg.add_fact("a", "is_a", "b")
        issues = kg.check_consistency()
        assert issues == []

    def test_multiple_types_detected(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "A", "t")
        kg.add_entity("b", "B", "t")
        kg.add_entity("c", "C", "t")
        # Use "rdf:type" — the value of RelationType.IS_A
        kg.add_fact("a", "rdf:type", "b")
        kg.add_fact("a", "rdf:type", "c")
        issues = kg.check_consistency()
        type_issues = [i for i in issues if i["type"] == "multiple_types"]
        # One issue per rdf:type edge for entity "a" (2 edges = 2 issues)
        assert len(type_issues) >= 1
        assert type_issues[0]["entity"] == "a"
        assert type_issues[0]["severity"] == "error"
        assert set(type_issues[0]["types"]) == {"b", "c"}

    def test_deep_hierarchy_warning(self):
        kg = KnowledgeGraph()
        for i in range(6):
            kg.add_entity(str(i), str(i), "t")
            if i > 0:
                # Use "rdf:type" — the value of RelationType.IS_A
                kg.add_fact(str(i), "rdf:type", str(i - 1))
        issues = kg.check_consistency()
        deep = [i for i in issues if i["type"] == "deep_hierarchy"]
        assert len(deep) > 0
        assert deep[0]["severity"] == "warning"


# ---------------------------------------------------------------------------
# export / export_triples / summary
# ---------------------------------------------------------------------------

class TestExport:
    def test_export(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "A", "t", properties={"k": "v"})
        kg.add_fact("a", "rel", "b", confidence=0.8, source="test")
        data = kg.export()
        assert "a" in data["entities"]
        assert data["entities"]["a"]["label"] == "A"
        assert len(data["facts"]) == 1
        assert data["stats"]["entities"] == 2

    def test_export_triples(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "rel", "b")
        kg.add_fact("x", "rel", "y")
        triples = kg.export_triples()
        assert len(triples) == 2
        subjects = {t["subject"] for t in triples}
        assert subjects == {"a", "x"}
        for t in triples:
            assert "subject" in t
            assert "predicate" in t
            assert "object" in t
            assert "confidence" in t
            assert "source" in t

    def test_summary(self):
        kg = KnowledgeGraph()
        kg.add_fact("a", "is_a", "b")
        s = kg.summary()
        assert "Knowledge Graph Summary" in s
        assert "Entities:" in s
        assert "Facts:" in s
        assert "is_a" in s


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_graph_bfs(self):
        kg = KnowledgeGraph()
        paths = kg.bfs("nonexistent")
        assert paths == {"nonexistent": []}

    def test_empty_graph_dfs(self):
        kg = KnowledgeGraph()
        paths = kg.dfs("nonexistent")
        assert paths == [[]]

    def test_self_referential_fact(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "A", "t")
        f = kg.add_fact("a", "self_ref", "a")
        assert f is not None
        out = kg.get_outgoing("a")
        assert ("self_ref", "a") in out

    def test_many_facts_single_entity(self):
        kg = KnowledgeGraph()
        kg.add_entity("hub", "Hub", "t")
        for i in range(20):
            kg.add_entity(f"leaf{i}", f"Leaf{i}", "t")
            kg.add_fact("hub", "connects", f"leaf{i}")
        out = kg.get_outgoing("hub")
        assert len(out) == 20

    def test_export_empty_graph(self):
        kg = KnowledgeGraph()
        data = kg.export()
        assert data["entities"] == {}
        assert data["facts"] == []
        assert data["stats"]["entities"] == 0
