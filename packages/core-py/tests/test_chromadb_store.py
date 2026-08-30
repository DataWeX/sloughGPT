"""Tests for ChromaDBVectorStore — connect, upsert, query, delete, count.

Chromadb is not a dependency; ``connect()`` is exercised via import mocking.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from domains.inference.vector_store import VectorEntry, QueryResult
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

    def test_persist_directory_preserved(self):
        store = ChromaDBVectorStore(persist_directory="/custom/path")
        assert store.persist_directory == "/custom/path"

    def test_collection_name_preserved(self):
        store = ChromaDBVectorStore(collection_name="my_collection")
        assert store.collection_name == "my_collection"

    def test_client_starts_none(self):
        store = ChromaDBVectorStore()
        assert store.client is None

    def test_collection_starts_none(self):
        store = ChromaDBVectorStore()
        assert store.collection is None

    def test_empty_strings(self):
        store = ChromaDBVectorStore(persist_directory="", collection_name="")
        assert store.persist_directory == ""
        assert store.collection_name == ""


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

    @pytest.mark.asyncio
    async def test_disconnect_when_already_none(self):
        store = ChromaDBVectorStore()
        await store.disconnect()
        assert store.client is None
        assert store.collection is None

    @pytest.mark.asyncio
    async def test_connect_uses_correct_path(self):
        store = ChromaDBVectorStore(persist_directory="/test/dir")
        fake_client = MagicMock()
        chromadb = MagicMock()
        chromadb.PersistentClient.return_value = fake_client
        fake_client.get_or_create_collection.return_value = MagicMock()

        with patch.dict(
            "sys.modules",
            {"chromadb": chromadb, "chromadb.config": MagicMock()},
        ):
            await store.connect()

        chromadb.PersistentClient.assert_called_once_with(path="/test/dir")

    @pytest.mark.asyncio
    async def test_connect_uses_correct_collection_name(self):
        store = ChromaDBVectorStore(collection_name="test_coll")
        fake_client = MagicMock()
        chromadb = MagicMock()
        chromadb.PersistentClient.return_value = fake_client
        fake_client.get_or_create_collection.return_value = MagicMock()

        with patch.dict(
            "sys.modules",
            {"chromadb": chromadb, "chromadb.config": MagicMock()},
        ):
            await store.connect()

        fake_client.get_or_create_collection.assert_called_once_with(name="test_coll")


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

    @pytest.mark.asyncio
    async def test_upsert_single_entry(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        entries = [VectorEntry(id="x", vector=[1.0, 2.0, 3.0], text="hello")]
        n = await store.upsert(entries)
        assert n == 1
        kwargs = store.collection.upsert.call_args.kwargs
        assert kwargs["ids"] == ["x"]
        assert kwargs["embeddings"] == [[1.0, 2.0, 3.0]]

    @pytest.mark.asyncio
    async def test_upsert_multiple_entries_count(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        entries = [VectorEntry(id=f"e{i}", vector=[float(i)], text=f"t{i}") for i in range(10)]
        n = await store.upsert(entries)
        assert n == 10

    @pytest.mark.asyncio
    async def test_query_empty_results(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.query.return_value = {
            "ids": [[]],
            "distances": [[]],
            "documents": [[]],
            "metadatas": [[]],
        }
        results = await store.query([1.0, 2.0])
        assert results == []

    @pytest.mark.asyncio
    async def test_query_single_result(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.query.return_value = {
            "ids": [["only"]],
            "distances": [[0.05]],
            "documents": [["solo result"]],
            "metadatas": [[{"score": 1}]],
        }
        results = await store.query([1.0])
        assert len(results) == 1
        assert results[0].id == "only"
        assert results[0].text == "solo result"
        assert results[0].metadata == {"score": 1}

    @pytest.mark.asyncio
    async def test_query_distances_missing(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.query.return_value = {
            "ids": [["a", "b"]],
            "documents": [["one", "two"]],
            "metadatas": [[{"k": 1}, {"k": 2}]],
        }
        results = await store.query([1.0])
        # No "distances" key → score defaults to 0.0
        assert results[0].score == 0.0

    @pytest.mark.asyncio
    async def test_query_documents_missing(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.query.return_value = {
            "ids": [["a"]],
            "distances": [[0.1]],
            "metadatas": [[{"k": 1}]],
        }
        results = await store.query([1.0])
        assert results[0].text == ""

    @pytest.mark.asyncio
    async def test_query_metadatas_missing(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.query.return_value = {
            "ids": [["a"]],
            "distances": [[0.1]],
            "documents": [["text"]],
        }
        results = await store.query([1.0])
        assert results[0].metadata == {}

    @pytest.mark.asyncio
    async def test_delete_empty_list(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        assert await store.delete([]) is True
        store.collection.delete.assert_called_once_with(ids=[])

    @pytest.mark.asyncio
    async def test_delete_single_id(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        assert await store.delete(["only_one"]) is True
        store.collection.delete.assert_called_once_with(ids=["only_one"])

    @pytest.mark.asyncio
    async def test_count_zero(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.count.return_value = 0
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_count_large_number(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.count.return_value = 1000000
        assert await store.count() == 1000000

    @pytest.mark.asyncio
    async def test_upsert_auto_ids_multiple(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        entries = [
            VectorEntry(id="", vector=[1.0], text="a"),
            VectorEntry(id="", vector=[2.0], text="b"),
            VectorEntry(id="custom", vector=[3.0], text="c"),
        ]
        await store.upsert(entries)
        ids = store.collection.upsert.call_args.kwargs["ids"]
        assert ids[0] == "entry_0"
        assert ids[1] == "entry_1"
        assert ids[2] == "custom"

    @pytest.mark.asyncio
    async def test_query_top_k_respected(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.query.return_value = {
            "ids": [["a", "b", "c", "d", "e"]],
            "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]],
            "documents": [["a", "b", "c", "d", "e"]],
            "metadatas": [[{}, {}, {}, {}, {}]],
        }
        results = await store.query([1.0], top_k=3)
        store.collection.query.assert_called_once_with(
            query_embeddings=[[1.0]], n_results=3, where=None
        )

    @pytest.mark.asyncio
    async def test_connect_then_operations(self):
        """Integration-style: connect, upsert, query, count, delete, disconnect."""
        store = ChromaDBVectorStore()
        fake_client = MagicMock()
        fake_collection = MagicMock()
        chromadb = MagicMock()
        chromadb.PersistentClient.return_value = fake_client
        fake_client.get_or_create_collection.return_value = fake_collection

        with patch.dict(
            "sys.modules",
            {"chromadb": chromadb, "chromadb.config": MagicMock()},
        ):
            assert await store.connect() is True

        # Upsert
        n = await store.upsert([VectorEntry(id="a", vector=[1.0], text="hello")])
        assert n == 1

        # Count
        fake_collection.count.return_value = 1
        assert await store.count() == 1

        # Query
        fake_collection.query.return_value = {
            "ids": [["a"]],
            "distances": [[0.0]],
            "documents": [["hello"]],
            "metadatas": [[{}]],
        }
        results = await store.query([1.0])
        assert len(results) == 1

        # Delete
        assert await store.delete(["a"]) is True

        # Disconnect
        await store.disconnect()
        assert store.client is None
        assert store.collection is None


class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_upsert_empty_list(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        n = await store.upsert([])
        assert n == 0

    @pytest.mark.asyncio
    async def test_upsert_preserves_metadata_keys(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        entries = [VectorEntry(id="a", vector=[1.0], text="t", metadata={"source": "test", "page": 5})]
        await store.upsert(entries)
        kwargs = store.collection.upsert.call_args.kwargs
        assert kwargs["metadatas"] == [{"source": "test", "page": 5}]

    @pytest.mark.asyncio
    async def test_query_returns_query_result_type(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.query.return_value = {
            "ids": [["a"]],
            "distances": [[0.5]],
            "documents": [["text"]],
            "metadatas": [[{}]],
        }
        results = await store.query([1.0])
        from domains.inference.vector_store import QueryResult
        assert isinstance(results[0], QueryResult)

    @pytest.mark.asyncio
    async def test_query_score_is_float(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.query.return_value = {
            "ids": [["a"]],
            "distances": [[0.123]],
            "documents": [["t"]],
            "metadatas": [[{}]],
        }
        results = await store.query([1.0])
        assert isinstance(results[0].score, float)

    @pytest.mark.asyncio
    async def test_disconnect_idempotent(self):
        store = ChromaDBVectorStore()
        store.client = MagicMock()
        store.collection = MagicMock()
        await store.disconnect()
        await store.disconnect()
        assert store.client is None
        assert store.collection is None

    @pytest.mark.asyncio
    async def test_connect_returns_bool(self):
        store = ChromaDBVectorStore()
        fake_client = MagicMock()
        chromadb = MagicMock()
        chromadb.PersistentClient.return_value = fake_client
        fake_client.get_or_create_collection.return_value = MagicMock()
        with patch.dict("sys.modules", {"chromadb": chromadb, "chromadb.config": MagicMock()}):
            result = await store.connect()
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_upsert_large_vector(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        vec = list(range(1536))
        entries = [VectorEntry(id="big", vector=vec, text="large embedding")]
        n = await store.upsert(entries)
        assert n == 1
        kwargs = store.collection.upsert.call_args.kwargs
        assert kwargs["embeddings"] == [vec]

    @pytest.mark.asyncio
    async def test_query_multiple_results_ordered_by_distance(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.query.return_value = {
            "ids": [["a", "b", "c"]],
            "distances": [[0.1, 0.5, 0.3]],
            "documents": [["first", "second", "third"]],
            "metadatas": [[{}, {}, {}]],
        }
        results = await store.query([1.0], top_k=3)
        assert results[0].score <= results[1].score
        assert results[1].score >= results[2].score

    @pytest.mark.asyncio
    async def test_delete_multiple_ids(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        ids = [f"id_{i}" for i in range(20)]
        assert await store.delete(ids) is True
        store.collection.delete.assert_called_once_with(ids=ids)

    @pytest.mark.asyncio
    async def test_count_after_upsert(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.count.return_value = 5
        assert await store.count() == 5

    @pytest.mark.asyncio
    async def test_upsert_entries_with_empty_metadata(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        entries = [VectorEntry(id="a", vector=[1.0], text="t", metadata={})]
        await store.upsert(entries)
        kwargs = store.collection.upsert.call_args.kwargs
        assert kwargs["metadatas"] == [{}]

    @pytest.mark.asyncio
    async def test_query_with_complex_filter(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.query.return_value = {
            "ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]
        }
        filt = {"$and": [{"type": "doc"}, {"year": 2024}]}
        await store.query([1.0], filter_metadata=filt)
        assert store.collection.query.call_args.kwargs["where"] == filt

    @pytest.mark.asyncio
    async def test_store_inherits_vector_store_interface(self):
        from domains.inference.vector_store import VectorStore
        store = ChromaDBVectorStore()
        assert isinstance(store, VectorStore)

    @pytest.mark.asyncio
    async def test_default_config_values(self):
        store = ChromaDBVectorStore()
        assert store.persist_directory == "data/vector_store"
        assert store.collection_name == "sloughgpt"


class TestChromaDBExtended:

    @pytest.mark.asyncio
    async def test_upsert_returns_int(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        n = await store.upsert([VectorEntry(id="a", vector=[1.0], text="t")])
        assert isinstance(n, int)

    @pytest.mark.asyncio
    async def test_query_returns_list(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.query.return_value = {
            "ids": [["a"]], "distances": [[0.1]],
            "documents": [["text"]], "metadatas": [[{}]],
        }
        results = await store.query([1.0])
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_delete_returns_bool(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        result = await store.delete(["a"])
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_count_returns_int(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.count.return_value = 42
        result = await store.count()
        assert isinstance(result, int)

    @pytest.mark.asyncio
    async def test_connect_false_on_error(self):
        store = ChromaDBVectorStore()
        chromadb = MagicMock()
        chromadb.PersistentClient.side_effect = ValueError("bad config")
        with patch.dict("sys.modules", {"chromadb": chromadb, "chromadb.config": MagicMock()}):
            result = await store.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_upsert_many_entries_performance(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        entries = [VectorEntry(id=f"e{i}", vector=[float(i)] * 10, text=f"text{i}") for i in range(100)]
        n = await store.upsert(entries)
        assert n == 100

    @pytest.mark.asyncio
    async def test_query_with_all_fields(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.query.return_value = {
            "ids": [["a", "b"]],
            "distances": [[0.1, 0.9]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"k": 1}, {"k": 2}]],
        }
        results = await store.query([1.0, 2.0], top_k=2, filter_metadata={"type": "test"})
        assert len(results) == 2
        store.collection.query.assert_called_once_with(
            query_embeddings=[[1.0, 2.0]], n_results=2, where={"type": "test"}
        )

    @pytest.mark.asyncio
    async def test_query_default_top_k(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.query.return_value = {
            "ids": [[]], "distances": [[]],
            "documents": [[]], "metadatas": [[]],
        }
        await store.query([1.0])
        store.collection.query.assert_called_once_with(
            query_embeddings=[[1.0]], n_results=5, where=None
        )

    @pytest.mark.asyncio
    async def test_connect_and_disconnect_cycle(self):
        store = ChromaDBVectorStore()
        fake_client = MagicMock()
        chromadb = MagicMock()
        chromadb.PersistentClient.return_value = fake_client
        fake_client.get_or_create_collection.return_value = MagicMock()
        with patch.dict("sys.modules", {"chromadb": chromadb, "chromadb.config": MagicMock()}):
            await store.connect()
        assert store.client is fake_client
        await store.disconnect()
        assert store.client is None
        assert store.collection is None

    @pytest.mark.asyncio
    async def test_upsert_then_count(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.count.return_value = 5
        await store.upsert([VectorEntry(id="a", vector=[1.0], text="t")])
        count = await store.count()
        assert count == 5

    @pytest.mark.asyncio
    async def test_delete_nonexistent_id(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        result = await store.delete(["nonexistent_id"])
        assert result is True
        store.collection.delete.assert_called_once_with(ids=["nonexistent_id"])

    @pytest.mark.asyncio
    async def test_upsert_vector_preserves_type(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        vec = [1.0, 2.0, 3.0]
        await store.upsert([VectorEntry(id="a", vector=vec, text="t")])
        kwargs = store.collection.upsert.call_args.kwargs
        assert kwargs["embeddings"] == [vec]

    @pytest.mark.asyncio
    async def test_query_result_id_type(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        store.collection.query.return_value = {
            "ids": [["my_id"]],
            "distances": [[0.5]],
            "documents": [["text"]],
            "metadatas": [[{}]],
        }
        results = await store.query([1.0])
        assert results[0].id == "my_id"
        assert isinstance(results[0].id, str)

    @pytest.mark.asyncio
    async def test_upsert_empty_vector(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        await store.upsert([VectorEntry(id="a", vector=[], text="t")])
        kwargs = store.collection.upsert.call_args.kwargs
        assert kwargs["embeddings"] == [[]]

    @pytest.mark.asyncio
    async def test_query_result_metadata_types(self):
        store = ChromaDBVectorStore()
        store.collection = MagicMock()
        meta = {"str_key": "val", "int_key": 42, "float_key": 3.14}
        store.collection.query.return_value = {
            "ids": [["a"]],
            "distances": [[0.1]],
            "documents": [["text"]],
            "metadatas": [[meta]],
        }
        results = await store.query([1.0])
        assert results[0].metadata == meta

    @pytest.mark.asyncio
    async def test_disconnect_clears_all(self):
        store = ChromaDBVectorStore()
        store.client = MagicMock()
        store.collection = MagicMock()
        await store.disconnect()
        assert store.client is None
        assert store.collection is None

    @pytest.mark.asyncio
    async def test_connect_custom_path(self):
        store = ChromaDBVectorStore(persist_directory="/custom/path", collection_name="custom")
        fake_client = MagicMock()
        chromadb = MagicMock()
        chromadb.PersistentClient.return_value = fake_client
        fake_client.get_or_create_collection.return_value = MagicMock()
        with patch.dict("sys.modules", {"chromadb": chromadb, "chromadb.config": MagicMock()}):
            result = await store.connect()
        assert result is True
        chromadb.PersistentClient.assert_called_once_with(path="/custom/path")
        fake_client.get_or_create_collection.assert_called_once_with(name="custom")
