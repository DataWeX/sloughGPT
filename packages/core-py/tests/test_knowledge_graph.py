"""Tests for domains.cognitive.knowledge_graph_v2 — RelationType, Entity, Fact, KnowledgeGraph."""

from domains.cognitive.knowledge_graph_v2 import (
    RelationType, Entity, Fact, KnowledgeGraph,
)


class TestRelationType:
    def test_all_members(self):
        assert len(RelationType) >= 7

    def test_values(self):
        assert RelationType.IS_A.value == "rdf:type"
        assert RelationType.PART_OF.value == "part_of"
        assert RelationType.CAUSES.value == "causes"


class TestEntity:
    def test_fields(self):
        e = Entity(id="e1", label="cat", entity_type="animal")
        assert e.id == "e1"
        assert e.label == "cat"

    def test_hash_and_eq(self):
        e1 = Entity(id="e1", label="cat", entity_type="animal")
        e2 = Entity(id="e1", label="dog", entity_type="animal")
        assert e1 == e2
        assert hash(e1) == hash(e2)

    def test_neq(self):
        e1 = Entity(id="e1", label="cat", entity_type="animal")
        e2 = Entity(id="e2", label="cat", entity_type="animal")
        assert e1 != e2


class TestFact:
    def test_fields(self):
        f = Fact(subject="cat", predicate="is_a", object="animal")
        assert f.subject == "cat"
        assert f.predicate == "is_a"

    def test_repr(self):
        f = Fact(subject="cat", predicate="is_a", object="animal")
        assert "cat" in repr(f)


class TestKnowledgeGraph:
    def test_init(self):
        kg = KnowledgeGraph()
        assert len(kg.entities) == 0

    def test_add_entity(self):
        kg = KnowledgeGraph()
        kg.add_entity("cat", "Cat", "animal")
        assert "cat" in kg.entities

    def test_add_fact(self):
        kg = KnowledgeGraph()
        kg.add_entity("cat", "Cat", "animal")
        kg.add_entity("dog", "Dog", "animal")
        kg.add_fact("cat", "related_to", "dog")
        assert len(kg.facts) >= 1

    def test_get_outgoing(self):
        kg = KnowledgeGraph()
        kg.add_entity("cat", "Cat", "animal")
        kg.add_entity("dog", "Dog", "animal")
        kg.add_fact("cat", "related_to", "dog")
        outgoing = kg.get_outgoing("cat")
        assert len(outgoing) >= 1

    def test_get_incoming(self):
        kg = KnowledgeGraph()
        kg.add_entity("cat", "Cat", "animal")
        kg.add_entity("dog", "Dog", "animal")
        kg.add_fact("cat", "related_to", "dog")
        incoming = kg.get_incoming("dog")
        assert len(incoming) >= 1

    def test_query(self):
        kg = KnowledgeGraph()
        kg.add_entity("cat", "Cat", "animal")
        kg.add_entity("dog", "Dog", "animal")
        kg.add_fact("cat", "related_to", "dog")
        results = kg.query("cat")
        assert len(results) >= 1

    def test_bfs(self):
        kg = KnowledgeGraph()
        kg.add_entity("a", "A", "type")
        kg.add_entity("b", "B", "type")
        kg.add_entity("c", "C", "type")
        kg.add_fact("a", "related_to", "b")
        kg.add_fact("b", "related_to", "c")
        result = kg.bfs("a")
        assert isinstance(result, dict)
        assert "a" in result
        assert "b" in result
