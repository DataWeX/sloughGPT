"""Tests for domains/inference/embeddings.py."""

import sys

import numpy as np
import pytest

from domains.inference.embeddings import (
    BatchEmbedder,
    Embedder,
    EmbeddingProvider,
    EmbeddingResult,
    InMemoryEmbedder,
    OpenAIEmbedder,
    create_embedder,
)


class TestEmbeddingProvider:
    def test_enum_values(self):
        assert EmbeddingProvider.SENTENCE_TRANSFORMERS.value == "sentence_transformers"
        assert EmbeddingProvider.OPENAI.value == "openai"
        assert EmbeddingProvider.HUGGINGFACE.value == "huggingface"
        assert EmbeddingProvider.IN_MEMORY.value == "in_memory"

    def test_enum_is_enum(self):
        assert EmbeddingProvider.OPENAI == EmbeddingProvider("openai")


class TestEmbeddingResult:
    def test_fields(self):
        result = EmbeddingResult(embedding=[0.1], model="m", dimension=1, token_count=5)
        assert result.embedding == [0.1]
        assert result.model == "m"
        assert result.dimension == 1
        assert result.token_count == 5

    def test_token_count_default(self):
        result = EmbeddingResult(embedding=[], model="m", dimension=0)
        assert result.token_count is None


class TestInMemoryEmbedder:
    def test_single_string(self):
        embedder = InMemoryEmbedder(dimension=16)
        vectors = embedder.embed("hello world")
        assert isinstance(vectors, list)
        assert len(vectors) == 1
        assert len(vectors[0]) == 16

    def test_list_of_strings(self):
        embedder = InMemoryEmbedder(dimension=16)
        vectors = embedder.embed(["hello", "world", "foo bar baz"])
        assert len(vectors) == 3
        assert all(len(v) == 16 for v in vectors)

    def test_output_unit_norm(self):
        embedder = InMemoryEmbedder(dimension=64)
        vector = embedder.embed("some meaningful text here")[0]
        norm = np.linalg.norm(vector)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_deterministic(self):
        embedder = InMemoryEmbedder(dimension=64)
        assert embedder.embed("repeat me") == embedder.embed("repeat me")

    def test_empty_text_returns_zero_vector(self):
        embedder = InMemoryEmbedder(dimension=16)
        vector = embedder.embed("")[0]
        assert vector == [0.0] * 16

    def test_words_beyond_dimension_ignored(self):
        embedder = InMemoryEmbedder(dimension=4)
        vector = embedder.embed("a b c d e f")[0]
        assert len(vector) == 4

    def test_get_dimension(self):
        embedder = InMemoryEmbedder(dimension=123)
        assert embedder.get_dimension() == 123

    def test_get_model_name(self):
        assert InMemoryEmbedder().get_model_name() == "in_memory"


class TestOpenAIEmbedder:
    def test_requires_api_key(self):
        with pytest.raises(ValueError):
            OpenAIEmbedder(api_key=None, model="text-embedding-3-small")

    def test_no_openai_library_raises_import_error(self):
        sys.modules.pop("openai", None)
        with pytest.raises(ImportError):
            OpenAIEmbedder(api_key="sk-test")

    def test_dimensions_known_models(self):
        assert OpenAIEmbedder.DIMENSIONS["text-embedding-ada-002"] == 1536
        assert OpenAIEmbedder.DIMENSIONS["text-embedding-3-small"] == 1536
        assert OpenAIEmbedder.DIMENSIONS["text-embedding-3-large"] == 3072

    def test_embed_with_mock_client(self, monkeypatch):
        class FakeItem:
            embedding = [0.1, 0.2, 0.3]

        class FakeData:
            data = [FakeItem(), FakeItem()]

        class FakeEmbeddings:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return FakeData()

        class FakeClient:
            def __init__(self, api_key):
                self.embeddings = FakeEmbeddings()

        monkeypatch.setitem(sys.modules, "openai", type("openai", (), {"OpenAI": FakeClient}))
        embedder = OpenAIEmbedder(api_key="sk-test", model="text-embedding-3-small")
        result = embedder.embed(["one", "two"])
        assert result == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        assert embedder.client.embeddings.kwargs["model"] == "text-embedding-3-small"
        assert embedder.client.embeddings.kwargs["dimensions"] == 1536

    def test_embed_ada_passes_no_dimensions(self, monkeypatch):
        class FakeItem:
            embedding = [0.1]

        class FakeData:
            data = [FakeItem()]

        class FakeEmbeddings:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return FakeData()

        class FakeClient:
            def __init__(self, api_key):
                self.embeddings = FakeEmbeddings()

        monkeypatch.setitem(sys.modules, "openai", type("openai", (), {"OpenAI": FakeClient}))
        embedder = OpenAIEmbedder(api_key="sk-test", model="text-embedding-ada-002")
        embedder.embed("hello")
        assert embedder.client.embeddings.kwargs["dimensions"] is None

    def test_unknown_model_defaults_dimension(self, monkeypatch):
        class FakeClient:
            def __init__(self, api_key):
                self.embeddings = None

        monkeypatch.setitem(sys.modules, "openai", type("openai", (), {"OpenAI": FakeClient}))
        embedder = OpenAIEmbedder(api_key="sk-test", model="some-custom-model")
        assert embedder.get_dimension() == 1536

    def test_custom_dimensions(self, monkeypatch):
        class FakeClient:
            def __init__(self, api_key):
                self.embeddings = None

        monkeypatch.setitem(sys.modules, "openai", type("openai", (), {"OpenAI": FakeClient}))
        embedder = OpenAIEmbedder(api_key="sk-test", model="text-embedding-3-large", dimensions=512)
        assert embedder.get_dimension() == 512

    def test_get_model_name(self, monkeypatch):
        class FakeClient:
            def __init__(self, api_key):
                self.embeddings = None

        monkeypatch.setitem(sys.modules, "openai", type("openai", (), {"OpenAI": FakeClient}))
        embedder = OpenAIEmbedder(api_key="sk-test", model="text-embedding-3-large")
        assert embedder.get_model_name() == "text-embedding-3-large"


class TestEmbedder:
    def test_default_is_in_memory_impl(self):
        embedder = Embedder()
        assert isinstance(embedder._impl, InMemoryEmbedder)

    def test_in_memory_provider(self):
        embedder = Embedder(provider="in_memory", dimension=8)
        assert isinstance(embedder._impl, InMemoryEmbedder)
        assert embedder.get_dimension() == 8

    def test_openai_provider_without_key_raises(self):
        with pytest.raises(ValueError):
            Embedder(provider="openai", api_key=None)

    def test_embed_and_embed_single(self):
        embedder = Embedder(provider="in_memory", dimension=8)
        vectors = embedder.embed("hello world")
        assert len(vectors) == 1
        assert embedder.embed_single("hello world") == vectors[0]

    def test_call_magic(self):
        embedder = Embedder(provider="in_memory", dimension=8)
        assert embedder("hello") == embedder.embed("hello")

    def test_get_model_name(self):
        embedder = Embedder(provider="in_memory")
        assert embedder.get_model_name() == "in_memory"


class TestBatchEmbedder:
    class CountingEmbedder:
        def __init__(self):
            self.calls = 0

        def embed(self, texts):
            self.calls += 1
            return [[float(ord(t[0]))] for t in texts]

    def test_batches_inputs(self):
        counting = self.CountingEmbedder()
        batch = BatchEmbedder(embedder=counting, batch_size=2)
        batch.embed(["a", "b", "c", "d", "e"])
        assert counting.calls == 3

    def test_returns_original_order(self):
        counting = self.CountingEmbedder()
        batch = BatchEmbedder(embedder=counting, batch_size=2)
        assert batch.embed(["a", "b", "c"]) == [[97.0], [98.0], [99.0]]

    def test_caches_repeated_texts(self):
        counting = self.CountingEmbedder()
        batch = BatchEmbedder(embedder=counting, batch_size=2)
        first = batch.embed(["hello"])
        second = batch.embed(["hello"])
        assert first == second
        assert counting.calls == 1

    def test_partial_cache_hits(self):
        counting = self.CountingEmbedder()
        batch = BatchEmbedder(embedder=counting, batch_size=2)
        batch.embed(["a"])
        result = batch.embed(["a", "b"])
        assert result == [[97.0], [98.0]]
        assert counting.calls == 2

    def test_clear_cache(self):
        counting = self.CountingEmbedder()
        batch = BatchEmbedder(embedder=counting, batch_size=2)
        batch.embed(["hello"])
        batch.clear_cache()
        batch.embed(["hello"])
        assert counting.calls == 2


class TestCreateEmbedder:
    def test_returns_base_impl(self):
        from domains.inference.embeddings import BaseEmbedder

        embedder = create_embedder(provider="in_memory", dimension=16)
        assert isinstance(embedder, BaseEmbedder)
        assert embedder.get_dimension() == 16
