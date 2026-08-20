"""Meaningful tests for KnowledgeGraph — case-insensitive lookup, dedup, indexing, BFS, DFS, paths, infer, verify, consistency."""

from domains.cognitive.knowledge_graph_v2 import KnowledgeGraph, RelationType, Entity, Fact


class TestAddEntity:
    def test_add_entity(self):
        g = KnowledgeGraph()
        e = g.add_entity("paris", "Paris", "city", properties={"population": 2161000})
        assert e.id == "paris"
        assert e.label == "Paris"
        assert g.entities["paris"].properties["population"] == 2161000

    def test_add_entity_with_aliases(self):
        g = KnowledgeGraph()
        g.add_entity("paris", "Paris", "city", aliases=["city of light"])
        assert "city of light" in g.entities["paris"].aliases

    def test_add_entity_duplicate_returns_existing(self):
        g = KnowledgeGraph()
        e1 = g.add_entity("paris", "Paris", "city")
        e2 = g.add_entity("paris", "Paris", "metropolis")
        assert e1 is e2

    def test_add_entity_case_insensitive(self):
        g = KnowledgeGraph()
        g.add_entity("Paris", "Paris", "city")
        e2 = g.add_entity("paris", "Paris", "city")
        assert e2.id == "Paris"

    def test_add_entity_aliases_merge(self):
        g = KnowledgeGraph()
        g.add_entity("Paris", "Paris", "city", aliases=["city of light"])
        g.add_entity("Paris", "Paris", "city", aliases=["ville lumiere"])
        aliases = g.entities["Paris"].aliases
        assert "city of light" in aliases
        assert "ville lumiere" in aliases


class TestAddFact:
    def test_add_fact_creates_entities(self):
        g = KnowledgeGraph()
        f = g.add_fact("Paris", "capital_of", "France")
        assert "Paris" in g.entities
        assert "France" in g.entities
        assert f is not None

    def test_add_fact_stores_fact(self):
        g = KnowledgeGraph()
        g.add_fact("Paris", "capital_of", "France", confidence=0.9, source="wiki")
        key = ("Paris", "capital_of", "France")
        assert key in g.facts
        assert g.facts[key].confidence == 0.9

    def test_add_fact_dedup_keeps_higher_confidence(self):
        g = KnowledgeGraph()
        g.add_fact("Paris", "capital_of", "France", confidence=0.5, source="guess")
        result = g.add_fact("Paris", "capital_of", "France", confidence=0.95, source="wiki")
        assert result is None  # Duplicate, no-op
        key = ("Paris", "capital_of", "France")
        assert g.facts[key].confidence == 0.95
        assert g.facts[key].source == "wiki"

    def test_add_fact_dedup_keeps_lower_confidence(self):
        g = KnowledgeGraph()
        g.add_fact("Paris", "capital_of", "France", confidence=0.9)
        g.add_fact("Paris", "capital_of", "France", confidence=0.3)
        assert g.facts[("Paris", "capital_of", "France")].confidence == 0.9

    def test_add_fact_updates_indices(self):
        g = KnowledgeGraph()
        g.add_fact("Paris", "capital_of", "France")
        assert "Paris" in g.subject_index
        assert "capital_of" in g.subject_index["Paris"]
        assert "France" in g.subject_index["Paris"]["capital_of"]
        assert "France" in g.object_index
        assert "capital_of" in g.object_index["France"]
        assert "Paris" in g.object_index["France"]["capital_of"]

    def test_add_fact_multiple_objects_same_predicate(self):
        g = KnowledgeGraph()
        g.add_fact("France", "borders", "Germany")
        g.add_fact("France", "borders", "Spain")
        targets = g.subject_index["France"]["borders"]
        assert "Germany" in targets
        assert "Spain" in targets


class TestGetOutgoing:
    def test_get_outgoing(self):
        g = KnowledgeGraph()
        g.add_fact("Paris", "capital_of", "France")
        g.add_fact("Paris", "located_in", "Europe")
        edges = g.get_outgoing("Paris")
        assert len(edges) == 2
        preds = {p for p, _ in edges}
        assert "capital_of" in preds

    def test_get_outgoing_filtered(self):
        g = KnowledgeGraph()
        g.add_fact("Paris", "capital_of", "France")
        g.add_fact("Paris", "located_in", "Europe")
        edges = g.get_outgoing("Paris", predicate="capital_of")
        assert len(edges) == 1
        assert edges[0][1] == "France"

    def test_get_outgoing_missing_entity(self):
        g = KnowledgeGraph()
        assert g.get_outgoing("nonexistent") == []


class TestGetIncoming:
    def test_get_incoming(self):
        g = KnowledgeGraph()
        g.add_fact("Paris", "capital_of", "France")
        edges = g.get_incoming("France")
        assert len(edges) == 1
        assert edges[0][1] == "Paris"

    def test_get_incoming_filtered(self):
        g = KnowledgeGraph()
        g.add_fact("Paris", "capital_of", "France")
        g.add_fact("Berlin", "capital_of", "Germany")
        g.add_fact("Germany", "borders", "France")
        edges = g.get_incoming("France", predicate="capital_of")
        assert len(edges) == 1


class TestQuery:
    def test_query_by_subject_and_predicate(self):
        g = KnowledgeGraph()
        g.add_fact("Paris", "capital_of", "France")
        results = g.query(subject="Paris", predicate="capital_of")
        assert len(results) == 1
        assert results[0].object == "France"

    def test_query_by_subject_only(self):
        g = KnowledgeGraph()
        g.add_fact("Paris", "capital_of", "France")
        g.add_fact("Paris", "located_in", "Europe")
        results = g.query(subject="Paris")
        assert len(results) == 2

    def test_query_by_predicate_and_object(self):
        g = KnowledgeGraph()
        g.add_fact("Paris", "capital_of", "France")
        g.add_fact("Berlin", "capital_of", "Germany")
        results = g.query(predicate="capital_of", obj="France")
        assert len(results) == 1
        assert results[0].subject == "Paris"

    def test_query_no_match(self):
        g = KnowledgeGraph()
        g.add_fact("Paris", "capital_of", "France")
        results = g.query(subject="London")
        assert len(results) == 0

    def test_query_by_object_only(self):
        g = KnowledgeGraph()
        g.add_fact("Paris", "capital_of", "France")
        g.add_fact("Lyon", "located_in", "France")
        results = g.query(obj="France")
        assert len(results) == 2


class TestBFS:
    def test_bfs_linear(self):
        g = KnowledgeGraph()
        g.add_fact("A", "leads_to", "B")
        g.add_fact("B", "leads_to", "C")
        paths = g.bfs("A")
        assert "B" in paths
        assert "C" in paths
        assert paths["B"] == [("leads_to", "A")]

    def test_bfs_max_depth(self):
        g = KnowledgeGraph()
        g.add_fact("A", "leads_to", "B")
        g.add_fact("B", "leads_to", "C")
        g.add_fact("C", "leads_to", "D")
        paths = g.bfs("A", max_depth=2)
        assert "D" not in paths

    def test_bfs_predicate_filter(self):
        g = KnowledgeGraph()
        g.add_fact("A", "likes", "B")
        g.add_fact("A", "dislikes", "C")
        g.add_fact("B", "likes", "D")
        paths = g.bfs("A", predicate_filter=lambda p: p == "likes")
        assert "B" in paths
        assert "C" not in paths

    def test_bfs_no_outgoing(self):
        g = KnowledgeGraph()
        g.add_entity("X", "X", "thing")
        paths = g.bfs("X")
        assert paths == {"X": []}


class TestDFS:
    def test_dfs_paths(self):
        g = KnowledgeGraph()
        g.add_fact("A", "leads_to", "B")
        g.add_fact("B", "leads_to", "C")
        paths = g.dfs("A")
        assert len(paths) >= 3  # [A], [A->B], [A->B->C]

    def test_dfs_max_depth(self):
        g = KnowledgeGraph()
        g.add_fact("A", "leads_to", "B")
        g.add_fact("B", "leads_to", "C")
        g.add_fact("C", "leads_to", "D")
        paths = g.dfs("A", max_depth=1)
        for p in paths:
            assert len(p) <= 1


class TestFindPaths:
    def test_find_paths_direct(self):
        g = KnowledgeGraph()
        g.add_fact("A", "leads_to", "B")
        paths = g.find_paths("A", "B")
        assert paths == [["A", "B"]]

    def test_find_paths_indirect(self):
        g = KnowledgeGraph()
        g.add_fact("A", "leads_to", "B")
        g.add_fact("B", "leads_to", "C")
        paths = g.find_paths("A", "C")
        assert len(paths) == 1
        assert paths[0] == ["A", "B", "C"]

    def test_find_paths_none(self):
        g = KnowledgeGraph()
        g.add_entity("A", "A", "thing")
        g.add_entity("B", "B", "thing")
        assert g.find_paths("A", "B") == []

    def test_find_paths_same_entity(self):
        g = KnowledgeGraph()
        g.add_entity("A", "A", "thing")
        paths = g.find_paths("A", "A")
        assert paths == [["A"]]

    def test_find_paths_with_filter(self):
        g = KnowledgeGraph()
        g.add_fact("A", "likes", "B")
        g.add_fact("B", "hates", "C")
        g.add_fact("B", "likes", "D")
        paths = g.find_paths("A", "D", predicate_filter=lambda p: p == "likes")
        assert len(paths) == 1

    def test_shortest_path(self):
        g = KnowledgeGraph()
        g.add_fact("A", "leads_to", "B")
        g.add_fact("A", "leads_to", "C")
        g.add_fact("C", "leads_to", "B")
        sp = g.shortest_path("A", "B")
        assert sp == ["A", "B"]

    def test_shortest_path_none(self):
        g = KnowledgeGraph()
        g.add_entity("A", "A", "thing")
        g.add_entity("B", "B", "thing")
        assert g.shortest_path("A", "B") is None


class TestInferTransitive:
    def test_infer_chain(self):
        g = KnowledgeGraph()
        g.add_fact("Human", "is_a", "Mammal")
        g.add_fact("Mammal", "is_a", "Animal")
        g.add_fact("Animal", "is_a", "LivingThing")
        reachable = g.infer_transitive("Human", "is_a")
        assert "Mammal" in reachable
        assert "Animal" in reachable
        assert "LivingThing" in reachable

    def test_infer_empty(self):
        g = KnowledgeGraph()
        g.add_entity("X", "X", "thing")
        assert g.infer_transitive("X", "is_a") == set()


class TestVerifyStatement:
    def test_verify_true_and_false(self):
        g = KnowledgeGraph()
        g.add_fact("Paris", "rdf:type", "City")
        result_true = g.verify_statement("Paris is a City")
        assert result_true["verified"] is True
        # Now verify a different statement that is NOT in the graph
        result_false = g.verify_statement("Paris is a Country")
        assert result_false["verified"] is False

    def test_verify_contradiction(self):
        g = KnowledgeGraph()
        g.add_fact("Paris", "rdf:type", "City")
        g.add_fact("Paris", "rdf:type", "Country")
        # Both triples exist so "Paris is a Country" is verified
        result = g.verify_statement("Paris is a Country")
        assert result["verified"] is True

    def test_verify_unparseable(self):
        g = KnowledgeGraph()
        result = g.verify_statement("the quick brown fox jumps")
        assert result["verified"] is False
        assert result["reason"] == "Could not parse statement"

    def test_verify_located_in(self):
        g = KnowledgeGraph()
        g.add_fact("Eiffel Tower", "located_in", "Paris")
        result = g.verify_statement("Eiffel Tower is located in Paris")
        assert result["verified"] is True


class TestCheckConsistency:
    def test_consistency_clean(self):
        g = KnowledgeGraph()
        g.add_fact("Paris", "capital_of", "France")
        issues = g.check_consistency()
        assert issues == []

    def test_consistency_multiple_types(self):
        g = KnowledgeGraph()
        g.add_fact("Paris", "rdf:type", "City")
        g.add_fact("Paris", "rdf:type", "Country")
        issues = g.check_consistency()
        types_issues = [i for i in issues if i["type"] == "multiple_types"]
        assert len(types_issues) >= 1
        assert types_issues[0]["severity"] == "error"


class TestExport:
    def test_export(self):
        g = KnowledgeGraph()
        g.add_fact("Paris", "capital_of", "France", confidence=0.9, source="wiki")
        data = g.export()
        assert "Paris" in data["entities"]
        assert len(data["facts"]) == 1
        assert data["stats"]["entities"] == 2
        assert data["stats"]["facts"] == 1
