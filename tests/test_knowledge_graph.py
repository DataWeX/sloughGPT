"""
Tests for Production-Grade Knowledge Graph
"""

import pytest
from domains.cognitive.knowledge_graph_v2 import (
    Entity,
    Fact,
    RelationType,
    KnowledgeGraph,
)


class TestKnowledgeGraph:
    """Tests for production knowledge graph."""

    def test_add_entity(self):
        """Test adding entities."""
        kg = KnowledgeGraph()

        entity = kg.add_entity(
            "python",
            "Python",
            "programming_language",
            properties={"year": 1991},
        )

        assert entity.id == "python"
        assert entity.label == "Python"
        assert entity.entity_type == "programming_language"
        assert kg.entities["python"].properties["year"] == 1991

    def test_add_fact(self):
        """Test adding facts."""
        kg = KnowledgeGraph()

        kg.add_entity("human", "Human", "mammal")
        kg.add_entity("mortal", "Mortal", "concept")

        fact = kg.add_fact("human", "is_a", "mortal", confidence=1.0)

        assert fact.subject == "human"
        assert fact.predicate == "is_a"
        assert fact.object == "mortal"
        assert fact.confidence == 1.0

    def test_query_exact(self):
        """Test exact fact query."""
        kg = KnowledgeGraph()

        kg.add_fact("socrates", "is_a", "human")
        kg.add_fact("socrates", "is_a", "mortal")

        facts = kg.query(subject="socrates", predicate="is_a", obj="human")
        assert len(facts) == 1
        assert facts[0].object == "human"

    def test_query_subject(self):
        """Test querying by subject only."""
        kg = KnowledgeGraph()

        kg.add_fact("socrates", "is_a", "human")
        kg.add_fact("socrates", "is_a", "mortal")

        facts = kg.query(subject="socrates")
        assert len(facts) == 2

    def test_get_outgoing(self):
        """Test getting outgoing edges."""
        kg = KnowledgeGraph()

        kg.add_fact("socrates", "is_a", "human")
        kg.add_fact("socrates", "is_a", "mortal")

        outgoing = kg.get_outgoing("socrates")
        assert len(outgoing) == 2
        assert ("is_a", "human") in outgoing

    def test_get_incoming(self):
        """Test getting incoming edges."""
        kg = KnowledgeGraph()

        kg.add_fact("socrates", "is_a", "human")
        kg.add_fact("plato", "is_a", "human")

        incoming = kg.get_incoming("human")
        assert len(incoming) == 2

    def test_bfs(self):
        """Test BFS traversal."""
        kg = KnowledgeGraph()

        # Human -> Mammal -> Animal
        kg.add_fact("human", "is_a", "mammal")
        kg.add_fact("mammal", "is_a", "animal")

        paths = kg.bfs("human", max_depth=3)

        assert "human" in paths
        assert "mammal" in paths
        assert "animal" in paths
        assert paths["mammal"] == [("is_a", "human")]
        assert paths["animal"] == [("is_a", "human"), ("is_a", "mammal")]

    def test_dfs(self):
        """Test DFS traversal."""
        kg = KnowledgeGraph()

        kg.add_fact("a", "relates_to", "b")
        kg.add_fact("b", "relates_to", "c")

        paths = kg.dfs("a", max_depth=3)
        assert len(paths) > 0

    def test_shortest_path(self):
        """Test shortest path finding."""
        kg = KnowledgeGraph()

        kg.add_fact("socrates", "is_a", "human")
        kg.add_fact("human", "is_a", "mortal")
        kg.add_fact("human", "is_a", "animal")

        path = kg.shortest_path("socrates", "mortal")
        assert path is not None
        assert "socrates" in path
        assert "mortal" in path

    def test_infer_transitive(self):
        """Test transitive inference."""
        kg = KnowledgeGraph()

        kg.add_fact("human", "is_a", "mammal")
        kg.add_fact("mammal", "is_a", "animal")
        kg.add_fact("animal", "is_a", "organism")

        reachable = kg.infer_transitive("human", "is_a", max_depth=5)

        assert "mammal" in reachable
        assert "animal" in reachable
        assert "organism" in reachable

    def test_verify_statement(self):
        """Test statement verification."""
        kg = KnowledgeGraph()

        kg.add_fact("socrates", "is_a", "human")
        kg.add_fact("socrates", "is_a", "mortal")

        result = kg.verify_statement("socrates is a human")
        assert result["verified"] == True
        assert result["confidence"] == 1.0

        result = kg.verify_statement("socrates is a stone")
        assert result["verified"] == False

    def test_consistency_check(self):
        """Test consistency checking."""
        kg = KnowledgeGraph()

        kg.add_fact("human", "is_a", "mammal")
        kg.add_fact("mammal", "is_a", "animal")

        issues = kg.check_consistency()
        assert isinstance(issues, list)

    def test_export(self):
        """Test graph export."""
        kg = KnowledgeGraph()

        kg.add_entity("test", "Test", "type")
        kg.add_fact("test", "relates_to", "other")

        export = kg.export()

        assert "entities" in export
        assert "facts" in export
        assert "stats" in export
        assert export["stats"]["entities"] == 2

    def test_summary(self):
        """Test summary generation."""
        kg = KnowledgeGraph()

        kg.add_entity("a", "A", "type")
        kg.add_entity("b", "B", "type")
        kg.add_fact("a", "relates_to", "b")

        summary = kg.summary()
        assert "Knowledge Graph Summary" in summary
        assert "Entities: 2" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
