"""Tests for ChromaDBVectorStore — connect, upsert, query, delete, count.

Chromadb is not a dependency; ``connect()`` is exercised via import mocking.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from domains.inference.vector_store import VectorEntry
from domains.inference.vector_stores.chromadb_store import ChromaDBVectorStore


class TestConstruction:

    def test_defaults(self):
        store = ChromaDBVectorStore()
        assert store.persist_directory == "data/vector_store"
        assert store.collection_name == "sloughgpt"
        assert store.client is None
        assert store.collection is None

    def test_custom_config(self):
        store = ChromaDBVectorStore(persist_directory="/tmp/x", collection_name="col")
        assert store.persist_directory == "/tmp/x"
        assert store.collection_name == "col"


class TestConnect:

    @pytest.mark.asyncio
    async def test_connect_success(self):
        store = ChromaDBVectorStore()
        fake_client = MagicMock()
        fake_collection = MagicMock()

        chromadb = MagicMock()
        chromadb.PersistentClient.return_value = fake_client
        fake_client.get_or_create_collection.return_value = fake_collection
        settings = MagicMock()

        with patch.dict(
            "sys.modules",
            {"chromadb": chromadb, "chromadb.config": MagicMock(Settings=settings)},
        ):
            result = await store.connect()

        assert result is True
        assert store.client is fake_client
        assert store.collection is fake_collection

    @pytest.mark.asyncio
    async def test_connect_import_error(self):
        store = ChromaDBVectorStore()
        with patch.dict("sys.modules", {"chromadb": None}):
            with pytest.raises(ImportError):
                await store.connect()

    @pytest.mark.asyncio
    async def test_connect_generic_failure_returns_false(self):
        store = ChromaDBVectorStore()
        chromadb = MagicMock()
        chromadb.PersistentClient.side_effect = RuntimeError("boom")
        with patch.dict("sys.modules", {"chromadb": chromadb, "chromadb.config": MagicMock()}):
            result = await store.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect(self):
        store = ChromaDBVectorStore()
        store.client = MagicMock()
        store.collection = MagicMock()
        await store.disconnect()
        assert store.client is None
        assert store.collection is None


class TestOperations:

    @pytest.mark.asyncio
    async def test_upsert_requires_connection(self):
        store = ChromaDBVectorStore()
        with pytest.raises(RuntimeError, match="Not connected"):
            await store.upsert([VectorEntry(id="a", vector=[1.0], text="t")])

    @pytest.mark.asyncio
    async def test_upsert(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        entries = [
            VectorEntry(id="a", vector=[1.0, 2.0], text="one", metadata={"k": 1}),
            VectorEntry(id="b", vector=[3.0, 4.0], text="two"),
        ]
        n = await store.upsert(entries)
        assert n == 2
        store.collection.upsert.assert_called_once()
        kwargs = store.collection.upsert.call_args.kwargs
        assert kwargs["ids"] == ["a", "b"]
        assert kwargs["embeddings"] == [[1.0, 2.0], [3.0, 4.0]]
        assert kwargs["documents"] == ["one", "two"]
        assert kwargs["metadatas"] == [{"k": 1}, {}]

    @pytest.mark.asyncio
    async def test_upsert_auto_ids(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        await store.upsert([VectorEntry(id="", vector=[1.0], text="t")])
        assert store.collection.upsert.call_args.kwargs["ids"] == ["entry_0"]

    @pytest.mark.asyncio
    async def test_query_requires_connection(self):
        store = ChromaDBVectorStore()
        with pytest.raises(RuntimeError, match="Not connected"):
            await store.query([1.0, 2.0])

    @pytest.mark.asyncio
    async def test_query(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.query.return_value = {
            "ids": [["a", "b"]],
            "distances": [[0.1, 0.5]],
            "documents": [["one", "two"]],
            "metadatas": [[{"k": 1}, {"k": 2}]],
        }
        results = await store.query([1.0, 2.0], top_k=2)
        assert len(results) == 2
        assert results[0].id == "a"
        assert results[0].score == pytest.approx(0.1)
        assert results[0].text == "one"
        assert results[0].metadata == {"k": 1}
        store.collection.query.assert_called_once_with(
            query_embeddings=[[1.0, 2.0]], n_results=2, where=None
        )

    @pytest.mark.asyncio
    async def test_query_passes_filter(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.query.return_value = {"ids": [], "distances": [], "documents": [], "metadatas": []}
        await store.query([1.0], filter_metadata={"k": 1})
        assert store.collection.query.call_args.kwargs["where"] == {"k": 1}

    @pytest.mark.asyncio
    async def test_query_missing_optional_fields(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.query.return_value = {"ids": [["a"]], "distances": [], "documents": [], "metadatas": []}
        results = await store.query([1.0])
        assert len(results) == 1
        assert results[0].id == "a"
        assert results[0].score == 0.0
        assert results[0].text == ""
        assert results[0].metadata == {}

    @pytest.mark.asyncio
    async def test_delete_requires_connection(self):
        store = ChromaDBVectorStore()
        assert await store.delete(["a"]) is False

    @pytest.mark.asyncio
    async def test_delete(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        assert await store.delete(["a", "b"]) is True
        store.collection.delete.assert_called_once_with(ids=["a", "b"])

    @pytest.mark.asyncio
    async def test_count_requires_connection(self):
        store = ChromaDBVectorStore()
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_count(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.count.return_value = 7
        assert await store.count() == 7
