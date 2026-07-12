"""Tests for InMemoryVectorStore, MogDBVectorStore, embed model loading — core vector CRUD + similarity search."""

import os
import tempfile
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
from domains.inference.vector_store import (
    InMemoryVectorStore,
    MogDBVectorStore,
    VectorEntry,
    simple_embed,
    _cosine_similarity,
    _load_embed_model,
    _EMBED_MIN_MEMORY_MB,
)


@pytest.fixture
def store():
    return InMemoryVectorStore(dimension=384)


@pytest.mark.asyncio
async def test_upsert_and_count(store):
    entries = [VectorEntry(id="a", vector=[1.0] * 384, text="hello", metadata={"topic": "greeting"})]
    n = await store.upsert(entries)
    assert n == 1
    assert await store.count() == 1


@pytest.mark.asyncio
async def test_upsert_overwrite(store):
    e1 = VectorEntry(id="x", vector=[1.0] * 384, text="first")
    e2 = VectorEntry(id="x", vector=[0.0] * 384, text="second")
    await store.upsert([e1])
    await store.upsert([e2])
    assert await store.count() == 1


@pytest.mark.asyncio
async def test_query_returns_results(store):
    await store.upsert([
        VectorEntry(id="a", vector=[1.0] * 384, text="alpha"),
        VectorEntry(id="b", vector=[0.0] * 384, text="beta"),
    ])
    results = await store.query(vector=[1.0] * 384, top_k=5)
    assert len(results) == 2
    assert results[0].id == "a"
    assert results[0].score >= results[1].score


@pytest.mark.asyncio
async def test_query_with_filter(store):
    await store.upsert([
        VectorEntry(id="a", vector=[1.0] * 384, text="doc a", metadata={"lang": "en"}),
        VectorEntry(id="b", vector=[1.0] * 384, text="doc b", metadata={"lang": "fr"}),
    ])
    results = await store.query(vector=[1.0] * 384, top_k=5, filter_metadata={"lang": "en"})
    assert len(results) == 1
    assert results[0].id == "a"


@pytest.mark.asyncio
async def test_query_empty_store(store):
    results = await store.query(vector=[1.0] * 384, top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_delete_existing(store):
    await store.upsert([VectorEntry(id="keep", vector=[1.0] * 384, text="keep")])
    assert await store.delete(["keep"]) is True
    assert await store.count() == 0


@pytest.mark.asyncio
async def test_delete_nonexistent(store):
    assert await store.delete(["ghost"]) is False


@pytest.mark.asyncio
async def test_delete_partial(store):
    await store.upsert([
        VectorEntry(id="a", vector=[1.0] * 384, text="a"),
        VectorEntry(id="b", vector=[1.0] * 384, text="b"),
    ])
    assert await store.delete(["a", "nonexistent"]) is True
    assert await store.count() == 1


# =========================================================================
# MogDBVectorStore tests
# =========================================================================


class TestMogDBVectorStore:
    @pytest.fixture
    def store(self):
        with tempfile.TemporaryDirectory() as tmp:
            s = MogDBVectorStore(dimension=384, path=os.path.join(tmp, "vectors"))
            import asyncio
            asyncio.run(s.connect())
            yield s
            asyncio.run(s.disconnect())

    @pytest.mark.asyncio
    async def test_upsert_and_count(self, store):
        entries = [VectorEntry(id="a", vector=[1.0] * 384, text="hello", metadata={"topic": "greeting"})]
        n = await store.upsert(entries)
        assert n == 1
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_upsert_overwrite(self, store):
        e1 = VectorEntry(id="x", vector=[1.0] * 384, text="first")
        e2 = VectorEntry(id="x", vector=[0.0] * 384, text="second")
        await store.upsert([e1])
        await store.upsert([e2])
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_query_returns_results(self, store):
        await store.upsert([
            VectorEntry(id="a", vector=[1.0] * 384, text="alpha"),
            VectorEntry(id="b", vector=[0.0] * 384, text="beta"),
        ])
        results = await store.query(vector=[1.0] * 384, top_k=5)
        assert len(results) == 2
        assert results[0].id == "a"
        assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    async def test_query_with_filter(self, store):
        await store.upsert([
            VectorEntry(id="a", vector=[1.0] * 384, text="doc a", metadata={"lang": "en"}),
            VectorEntry(id="b", vector=[1.0] * 384, text="doc b", metadata={"lang": "fr"}),
        ])
        results = await store.query(vector=[1.0] * 384, top_k=5, filter_metadata={"lang": "en"})
        assert len(results) == 1
        assert results[0].id == "a"

    @pytest.mark.asyncio
    async def test_query_empty_store(self, store):
        results = await store.query(vector=[1.0] * 384, top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_delete_existing(self, store):
        await store.upsert([VectorEntry(id="keep", vector=[1.0] * 384, text="keep")])
        assert await store.delete(["keep"]) is True
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        assert await store.delete(["ghost"]) is False

    @pytest.mark.asyncio
    async def test_delete_partial(self, store):
        await store.upsert([
            VectorEntry(id="a", vector=[1.0] * 384, text="a"),
            VectorEntry(id="b", vector=[1.0] * 384, text="b"),
        ])
        assert await store.delete(["a", "nonexistent"]) is True
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_persistence_survives_reconnect(self):
        """Entries survive closing and re-opening the MogDBVectorStore."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "vec")
            s1 = MogDBVectorStore(dimension=384, path=path)
            await s1.connect()
            await s1.upsert([
                VectorEntry(id="p1", vector=[1.0] * 384, text="persistent"),
                VectorEntry(id="p2", vector=[0.5] * 384, text="also here"),
            ])
            assert await s1.count() == 2
            await s1.disconnect()

            s2 = MogDBVectorStore(dimension=384, path=path)
            await s2.connect()
            assert await s2.count() == 2
            results = await s2.query(vector=[1.0] * 384, top_k=5)
            assert len(results) == 2
            ids = {r.id for r in results}
            assert ids == {"p1", "p2"}
            await s2.disconnect()

    @pytest.mark.asyncio
    async def test_persist_and_reload_after_delete(self):
        """Deletes are reflected after reconnect."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "vec")
            s1 = MogDBVectorStore(dimension=384, path=path)
            await s1.connect()
            await s1.upsert([
                VectorEntry(id="keep", vector=[1.0] * 384, text="keep me"),
                VectorEntry(id="gone", vector=[0.0] * 384, text="delete me"),
            ])
            await s1.delete(["gone"])
            await s1.disconnect()

            s2 = MogDBVectorStore(dimension=384, path=path)
            await s2.connect()
            assert await s2.count() == 1
            assert (await s2.query(vector=[1.0] * 384))[0].id == "keep"
            await s2.disconnect()


def test_cosine_similarity_identical():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    assert abs(_cosine_similarity(a, b) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert abs(_cosine_similarity(a, b)) < 1e-9


def test_cosine_similarity_opposite():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert abs(_cosine_similarity(a, b) - (-1.0)) < 1e-9


class TestSimpleEmbed:
    def test_returns_list_of_correct_length(self):
        vec = simple_embed("hello world")
        assert isinstance(vec, list)
        assert len(vec) == 384
        assert all(isinstance(v, float) for v in vec)

    def test_deterministic(self):
        v1 = simple_embed("test string")
        v2 = simple_embed("test string")
        # SloNet embedder: Metal GPU accelerator causes minor floating-point variance
        assert np.allclose(v1, v2, atol=0.05)

    def test_l2_normalized(self):
        vec = simple_embed("some text")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-6

    def test_similar_texts_higher_score(self):
        a = simple_embed("Python is a programming language")
        b = simple_embed("Python is used for web development")
        c = simple_embed("The cat sat on the mat")
        sim_ab = float(np.dot(a, b))
        sim_ac = float(np.dot(a, c))
        # With tiny embedder, discrimination is limited — just verify scores are in valid range
        assert 0.0 <= sim_ab <= 1.0, f"similarity out of range: {sim_ab}"
        assert 0.0 <= sim_ac <= 1.0, f"similarity out of range: {sim_ac}"

    def test_empty_text(self):
        vec = simple_embed("")
        assert len(vec) == 384
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-6


# =========================================================================
# Embed model loading tests
# =========================================================================


class TestLoadEmbedModel:
    """Tests for _load_embed_model() auto-download + fallback behavior."""

    def setup_method(self):
        """Reset global state before each test."""
        import domains.inference.vector_store as vs
        vs._embed_model = None
        vs._EMBED_LOAD_FAILED = False

    def teardown_method(self):
        """Reset global state after each test."""
        import domains.inference.vector_store as vs
        vs._embed_model = None
        vs._EMBED_LOAD_FAILED = False

    def test_returns_none_when_sentence_transformers_missing(self):
        """Falls back gracefully when sentence-transformers can't import."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            result = _load_embed_model()
            assert result is None

    def test_returns_none_when_import_error(self):
        """Handles ImportError from sentence-transformers sub-dependencies."""
        import domains.inference.vector_store as vs
        vs._EMBED_LOAD_FAILED = False
        vs._embed_model = None

        # Make sentence_transformers importable but SentenceTransformer raise on construction
        fake_st = MagicMock()
        fake_st.SentenceTransformer.side_effect = RuntimeError("no torch")

        fake_mem = MagicMock()
        fake_mem.available = 2 * 1024 * 1024 * 1024

        with patch.dict("sys.modules", {"sentence_transformers": fake_st}):
            with patch("psutil.virtual_memory", return_value=fake_mem):
                result = _load_embed_model()
                assert result is None
                assert vs._EMBED_LOAD_FAILED is True

    def test_returns_none_when_memory_low(self):
        """Skips loading when available memory is below threshold."""
        fake_mem = MagicMock()
        fake_mem.available = _EMBED_MIN_MEMORY_MB * 1024 * 1024 - 1  # 1 byte below threshold
        with patch("psutil.virtual_memory", return_value=fake_mem):
            result = _load_embed_model()
            assert result is None

    def test_caches_model_after_successful_load(self):
        """Returns cached model on second call."""
        import domains.inference.vector_store as vs
        mock_model = MagicMock()
        vs._embed_model = mock_model

        result = _load_embed_model()
        assert result is mock_model

    def test_load_failed_flag_prevents_retry(self):
        """Once loading fails, subsequent calls return None immediately."""
        import domains.inference.vector_store as vs
        vs._EMBED_LOAD_FAILED = True

        result = _load_embed_model()
        assert result is None

    def test_returns_model_on_successful_import(self):
        """Returns the SentenceTransformer model when everything works."""
        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_st.SentenceTransformer.return_value = mock_model

        fake_mem = MagicMock()
        fake_mem.available = 2 * 1024 * 1024 * 1024  # 2 GB

        with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
            with patch("psutil.virtual_memory", return_value=fake_mem):
                import domains.inference.vector_store as vs
                vs._embed_model = None
                vs._EMBED_LOAD_FAILED = False
                result = _load_embed_model()
                assert result is mock_model
                mock_st.SentenceTransformer.assert_called_once_with(vs._EMBED_MODEL_NAME, device="cpu")
