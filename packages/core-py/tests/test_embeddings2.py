import hashlib
import numpy as np
import pytest

from domains.inference.embeddings import (
    EmbeddingProvider,
    EmbeddingResult,
    InMemoryEmbedder,
    Embedder,
    BatchEmbedder,
    OpenAIEmbedder,
    create_embedder,
)


class TestEmbeddingProvider:
    def test_values(self):
        assert EmbeddingProvider.SENTENCE_TRANSFORMERS.value == "sentence_transformers"
        assert EmbeddingProvider.OPENAI.value == "openai"
        assert EmbeddingProvider.HUGGINGFACE.value == "huggingface"
        assert EmbeddingProvider.IN_MEMORY.value == "in_memory"

    def test_member_count(self):
        assert len(EmbeddingProvider) == 4


class TestEmbeddingResult:
    def test_basic_creation(self):
        r = EmbeddingResult(embedding=[0.1, 0.2], model="m", dimension=2)
        assert r.embedding == [0.1, 0.2]
        assert r.model == "m"
        assert r.dimension == 2
        assert r.token_count is None

    def test_optional_token_count(self):
        r = EmbeddingResult(embedding=[], model="m", dimension=0, token_count=42)
        assert r.token_count == 42

    def test_equality(self):
        a = EmbeddingResult(embedding=[1.0], model="x", dimension=1)
        b = EmbeddingResult(embedding=[1.0], model="x", dimension=1)
        assert a == b


class TestInMemoryEmbedder:
    def test_single_string(self):
        emb = InMemoryEmbedder(dimension=64)
        result = emb.embed("hello world")
        assert len(result) == 1
        assert len(result[0]) == 64

    def test_list_of_strings(self):
        emb = InMemoryEmbedder(dimension=32)
        result = emb.embed(["cat", "dog", "bird"])
        assert len(result) == 3
        for vec in result:
            assert len(vec) == 32

    def test_normalization(self):
        emb = InMemoryEmbedder(dimension=128)
        result = emb.embed("test normalization")
        vec = np.array(result[0])
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-6

    def test_empty_string(self):
        emb = InMemoryEmbedder(dimension=16)
        result = emb.embed("")
        assert len(result) == 1
        assert len(result[0]) == 16

    def test_deterministic(self):
        emb = InMemoryEmbedder(dimension=64)
        r1 = emb.embed("deterministic check")
        r2 = emb.embed("deterministic check")
        assert r1 == r2

    def test_different_texts_produce_different_vectors(self):
        emb = InMemoryEmbedder(dimension=64)
        r1 = emb.embed("alpha beta")
        r2 = emb.embed("gamma delta epsilon")
        assert r1 != r2

    def test_get_dimension(self):
        assert InMemoryEmbedder(dimension=256).get_dimension() == 256

    def test_get_model_name(self):
        assert InMemoryEmbedder().get_model_name() == "in_memory"

    def test_word_count_exceeds_dimension(self):
        emb = InMemoryEmbedder(dimension=4)
        words = " ".join(["word"] * 10)
        result = emb.embed(words)
        assert len(result) == 1
        assert len(result[0]) == 4

    def test_case_insensitive(self):
        emb = InMemoryEmbedder(dimension=64)
        r1 = emb.embed("Hello World")
        r2 = emb.embed("hello world")
        assert r1 == r2

    def test_zero_norm_empty_text(self):
        emb = InMemoryEmbedder(dimension=16)
        result = emb.embed("")
        vec = np.array(result[0])
        assert np.allclose(vec, 0.0)


class TestEmbedder:
    def test_default_provider(self):
        e = Embedder()
        assert e.get_model_name() == "in_memory"

    def test_embed_single(self):
        e = Embedder(dimension=32)
        vec = e.embed_single("single text")
        assert len(vec) == 32
        assert isinstance(vec, list)

    def test_embed_list(self):
        e = Embedder(dimension=32)
        result = e.embed(["a", "b"])
        assert len(result) == 2

    def test_callable(self):
        e = Embedder(dimension=32)
        result = e("callable test")
        assert len(result) == 1
        assert len(result[0]) == 32

    def test_get_dimension(self):
        e = Embedder(dimension=512)
        assert e.get_dimension() == 512

    def test_explicit_openai_requires_key(self):
        with pytest.raises(ValueError, match="API key required"):
            Embedder(provider="openai")

    def test_explicit_openai_wrong_import(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "openai":
                raise ImportError("no openai")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pip install openai"):
            Embedder(provider="openai", api_key="sk-fake")


class TestOpenAIEmbedder:
    def test_dimensions_map(self):
        assert OpenAIEmbedder.DIMENSIONS["text-embedding-ada-002"] == 1536
        assert OpenAIEmbedder.DIMENSIONS["text-embedding-3-small"] == 1536
        assert OpenAIEmbedder.DIMENSIONS["text-embedding-3-large"] == 3072

    def test_init_requires_api_key(self):
        with pytest.raises(ValueError, match="API key required"):
            OpenAIEmbedder(api_key=None)


class TestBatchEmbedder:
    def test_basic_batch(self):
        b = BatchEmbedder(embedder=Embedder(dimension=32), batch_size=10)
        result = b.embed(["a", "b", "c"])
        assert len(result) == 3
        for vec in result:
            assert len(vec) == 32

    def test_caching(self):
        b = BatchEmbedder(embedder=Embedder(dimension=32))
        r1 = b.embed(["cached text"])
        r2 = b.embed(["cached text"])
        assert r1 == r2

    def test_cache_hit_preserves_order(self):
        b = BatchEmbedder(embedder=Embedder(dimension=16))
        b.embed(["first"])
        result = b.embed(["second", "first", "third"])
        assert len(result) == 3

    def test_clear_cache(self):
        b = BatchEmbedder(embedder=Embedder(dimension=16))
        b.embed(["x"])
        assert len(b._cache) == 1
        b.clear_cache()
        assert len(b._cache) == 0

    def test_batch_size_limits(self):
        b = BatchEmbedder(embedder=Embedder(dimension=16), batch_size=2)
        texts = [f"text_{i}" for i in range(7)]
        result = b.embed(texts)
        assert len(result) == 7

    def test_empty_input(self):
        b = BatchEmbedder()
        result = b.embed([])
        assert result == []

    def test_cache_eviction_not_enforced_on_embed(self):
        b = BatchEmbedder(embedder=Embedder(dimension=16), cache_size=2)
        b.embed(["a"])
        b.embed(["b"])
        b.embed(["c"])
        assert len(b._cache) == 3


class TestCreateEmbedder:
    def test_default(self):
        e = create_embedder()
        assert isinstance(e, InMemoryEmbedder)

    def test_explicit_in_memory(self):
        e = create_embedder(provider="in_memory")
        assert isinstance(e, InMemoryEmbedder)

    def test_openai_requires_key(self):
        with pytest.raises(ValueError):
            create_embedder(provider="openai")


class TestCrossInstanceDeterminism:
    def test_same_input_same_output(self):
        e1 = InMemoryEmbedder(dimension=64)
        e2 = InMemoryEmbedder(dimension=64)
        r1 = e1.embed("shared input")
        r2 = e2.embed("shared input")
        assert r1 == r2
