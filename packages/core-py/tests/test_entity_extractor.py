"""Tests for domains.learner.entity_extractor — pure extraction functions."""

from domains.learner.entity_extractor import (
    _is_valid_entity, extract_entities, extract_relationships,
    extract_facts_from_conversation,
)


class TestIsValidEntity:
    def test_valid(self):
        assert _is_valid_entity("Python") is True

    def test_stop_word(self):
        assert _is_valid_entity("the") is False

    def test_too_short(self):
        assert _is_valid_entity("a") is False

    def test_punctuation(self):
        assert _is_valid_entity("hello.") is False


class TestExtractEntities:
    def test_empty(self):
        assert extract_entities("") == []

    def test_multi_word(self):
        entities = extract_entities("I visited New York City last summer.")
        assert "New York City" in entities

    def test_single_entity(self):
        entities = extract_entities("Python is a great language.")
        assert "Python" in entities

    def test_false_entities_excluded(self):
        entities = extract_entities("Hello, how are you today?")
        assert "Hello" not in entities


class TestExtractRelationships:
    def test_is_a(self):
        triples = extract_relationships("A dog is a mammal.")
        assert any(r[1] == "is_a" for r in triples)

    def test_likes(self):
        triples = extract_relationships("Alice likes pizza.")
        assert any(r[1] == "likes" for r in triples)

    def test_empty(self):
        assert extract_relationships("just random text no patterns") == []


class TestExtractFactsFromConversation:
    def test_empty(self):
        assert extract_facts_from_conversation("", "") == []

    def test_with_relationship(self):
        facts = extract_facts_from_conversation("What is a dog?", "A dog is a mammal.")
        assert any("dog" in f.lower() and "mammal" in f.lower() for f in facts)

    def test_deduplication(self):
        facts = extract_facts_from_conversation("I like cats", "Cats are great pets.")
        assert len(facts) == len(set(facts))
