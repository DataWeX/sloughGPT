"""Tests for domains.cognitive.rag — TextChunk, RetrievalResult; domains.cognitive.grounding — Document, FisherInformation, KnowledgeNode, KnowledgeEdge."""

import numpy as np
from domains.cognitive.rag import TextChunk, RetrievalResult
from domains.cognitive.grounding import (
    Document, FisherInformation, KnowledgeNode, KnowledgeEdge,
)


class TestTextChunk:
    def test_fields(self):
        tc = TextChunk(id="c1", content="hello world", metadata={}, token_count=2)
        assert tc.id == "c1"
        assert tc.token_count == 2

    def test_auto_token_count(self):
        tc = TextChunk(id="c1", content="hello world foo bar", metadata={})
        assert tc.token_count == 4

    def test_with_embedding(self):
        emb = np.array([0.1, 0.2])
        tc = TextChunk(id="c1", content="x", metadata={}, embedding=emb)
        assert tc.embedding is not None


class TestRetrievalResult:
    def test_fields(self):
        tc = TextChunk(id="c1", content="x", metadata={})
        rr = RetrievalResult(chunk=tc, dense_score=0.9, sparse_score=0.8, combined_score=0.85, rank=1)
        assert rr.dense_score == 0.9
        assert rr.rank == 1


class TestDocument:
    def test_fields(self):
        d = Document(id="d1", content="text", source="file")
        assert d.id == "d1"
        assert d.source == "file"
        assert d.metadata == {}


class TestFisherInformation:
    def test_fields(self):
        fi = FisherInformation(param_name="W", importance=0.5, old_value=0.3)
        assert fi.param_name == "W"
        assert fi.importance == 0.5


class TestKnowledgeNode:
    def test_fields(self):
        kn = KnowledgeNode(id="n1", label="cat", node_type="entity")
        assert kn.id == "n1"
        assert kn.node_type == "entity"
        assert kn.properties == {}


class TestKnowledgeEdge:
    def test_fields(self):
        ke = KnowledgeEdge(source="n1", target="n2", relation="is_a", weight=0.8)
        assert ke.source == "n1"
        assert ke.relation == "is_a"
        assert ke.weight == 0.8

    def test_defaults(self):
        ke = KnowledgeEdge(source="a", target="b", relation="related_to")
        assert ke.weight == 1.0
