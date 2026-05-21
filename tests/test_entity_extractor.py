"""Tests for entity_extractor — relationship and entity extraction from chat."""
import pytest
from domains.learner.entity_extractor import (
    extract_entities,
    extract_relationships,
    extract_facts_from_conversation,
)


class TestExtractEntities:
    def test_extracts_capitalized_names(self):
        text = "Alice and Bob went to Paris."
        entities = extract_entities(text)
        assert "Alice" in entities
        assert "Bob" in entities

    def test_extracts_multi_word_entities(self):
        text = "I work at Microsoft Corporation."
        entities = extract_entities(text)
        assert "Microsoft Corporation" in entities

    def test_ignores_stop_words(self):
        text = "The cat sat on the mat."
        entities = extract_entities(text)
        assert all(e.lower() not in ("the", "a", "an") for e in entities)

    def test_ignores_greetings(self):
        text = "Hello world. Hi there."
        entities = extract_entities(text)
        assert "Hello" not in entities
        assert "Hi" not in entities

    def test_empty_text(self):
        assert extract_entities("") == []

    def test_no_entities(self):
        assert extract_entities("this is just a normal sentence") == []


class TestExtractRelationships:
    def test_is_a_relationship(self):
        rels = extract_relationships("Python is a programming language.")
        assert any((s, r, o) == ("Python", "is_a", "programming language") for s, r, o in rels)

    def test_likes_relationship(self):
        rels = extract_relationships("Alice likes pizza.")
        assert any("likes" in (s, r, o) for s, r, o in rels)

    def test_has_relationship(self):
        rels = extract_relationships("Bob has a car.")
        assert any("has" in (s, r, o) for s, r, o in rels)

    def test_possessive_relationship(self):
        rels = extract_relationships("Alice's cat is fluffy.")
        assert any("possesses" in r for _, r, _ in rels)

    def test_empty_text(self):
        assert extract_relationships("") == []


class TestExtractFacts:
    def test_extracts_relationship_facts(self):
        facts = extract_facts_from_conversation(
            "What is Python?", "Python is a programming language."
        )
        assert any("Python is a" in f for f in facts)

    def test_extracts_entity_facts(self):
        facts = extract_facts_from_conversation(
            "Tell me about Alice.", "Alice is a software engineer."
        )
        assert any("Entity Alice" in f for f in facts)

    def test_empty_messages(self):
        assert extract_facts_from_conversation("", "") == []
