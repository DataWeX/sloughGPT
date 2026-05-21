"""
Tests for Vector Store Integration - Pinecone
"""

import pytest
import numpy as np
from domains.inference.vector_store import (
    PineconeVectorStore,
    VectorEntry,
    QueryResult,
    simple_embed,
)


class TestPineconeVectorStore:
    """Tests for Pinecone vector store."""

    def test_initialization(self):
        """Test store initialization."""
        store = PineconeVectorStore(
            api_key="test-key",
            index_name="test-index",
            dimension=768,
        )
        assert store.api_key == "test-key"
        assert store.index_name == "test-index"
        assert store.dimension == 768

    def test_initialization_with_env(self, monkeypatch):
        """Test store initialization from environment."""
        monkeypatch.setenv("PINECONE_API_KEY", "env-key")
        store = PineconeVectorStore(index_name="env-index")
        assert store.api_key == "env-key"


class TestSimpleEmbed:
    """Tests for simple embedding function."""

    def test_embed_dimension(self):
        """Test embedding dimension."""
        vec = simple_embed("test text", dimension=384)
        assert len(vec) == 384

    def test_embed_normalized(self):
        """Test embedding is normalized."""
        vec = simple_embed("test text")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01

    def test_embed_consistent(self):
        """Test same text gives same embedding."""
        vec1 = simple_embed("hello world")
        vec2 = simple_embed("hello world")
        assert np.allclose(vec1, vec2)

    def test_embed_different_texts(self):
        """Test different texts give different embeddings."""
        vec1 = simple_embed("python programming")
        vec2 = simple_embed("machine learning")
        assert not np.allclose(vec1, vec2)


class TestVectorEntry:
    """Tests for VectorEntry dataclass."""

    def test_vector_entry_creation(self):
        """Test VectorEntry creation."""
        entry = VectorEntry(
            id="test_id",
            vector=[0.1, 0.2, 0.3],
            text="Test text",
            metadata={"key": "value"},
        )

        assert entry.id == "test_id"
        assert entry.vector == [0.1, 0.2, 0.3]
        assert entry.text == "Test text"
        assert entry.metadata == {"key": "value"}

    def test_vector_entry_default_metadata(self):
        """Test default metadata."""
        entry = VectorEntry(id="1", vector=[0.1], text="test")
        assert entry.metadata == {}


class TestQueryResult:
    """Tests for QueryResult dataclass."""

    def test_query_result_creation(self):
        """Test QueryResult creation."""
        result = QueryResult(
            id="test_id",
            score=0.95,
            text="Result text",
            metadata={"source": "test"},
        )

        assert result.id == "test_id"
        assert result.score == 0.95
        assert result.text == "Result text"
        assert result.metadata == {"source": "test"}
