"""Tests for PineconeVectorStore — config, connect, upsert, query, delete, count.

Pinecone is not a dependency; ``connect()`` is exercised via import mocking.
"""

import pytest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from domains.inference.vector_store import VectorEntry
from domains.inference.vector_stores.pinecone_store import PineconeVectorStore


class TestConstruction:

    def test_defaults(self):
        store = PineconeVectorStore()
        assert store.api_key is None
        assert store.index_name == "sloughgpt"
        assert store.environment == "us-east-1"
        assert store.dimension == 768
        assert store.metric == "cosine"
        assert store.index is None

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("PINECONE_API_KEY", "secret")
        store = PineconeVectorStore()
        assert store.api_key == "secret"

    def test_explicit_api_key_wins(self, monkeypatch):
        monkeypatch.setenv("PINECONE_API_KEY", "env-secret")
        store = PineconeVectorStore(api_key="explicit")
        assert store.api_key == "explicit"


class TestConnect:

    @pytest.mark.asyncio
    async def test_connect_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("PINECONE_API_KEY", raising=False)
        store = PineconeVectorStore(api_key=None)
        with patch.dict("sys.modules", {"pinecone": MagicMock()}):
            result = await store.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_existing_index(self):
        store = PineconeVectorStore(api_key="k")
        client = MagicMock()
        client.list_indexes.return_value = [SimpleNamespace(name="sloughgpt")]
        index = MagicMock()
        client.Index.return_value = index
        pinecone = MagicMock(Pinecone=MagicMock(return_value=client))
        with patch.dict("sys.modules", {"pinecone": pinecone}):
            result = await store.connect()

        assert result is True
        assert store.client is client
        assert store.index is index
        client.create_index.assert_not_called()

    @pytest.mark.asyncio
    async def test_connect_creates_serverless_index(self):
        store = PineconeVectorStore(api_key="k", environment="us-east-1")
        client = MagicMock()
        client.list_indexes.return_value = []
        index = MagicMock()
        client.Index.return_value = index
        pinecone = MagicMock(Pinecone=MagicMock(return_value=client))
        serverless_spec = MagicMock()
        pod_spec = MagicMock()

        def spec_factory(**kwargs):
            return serverless_spec

        pinecone.ServerlessSpec.side_effect = spec_factory
        pinecone.PodSpec = pod_spec

        with patch.dict("sys.modules", {"pinecone": pinecone}):
            result = await store.connect()

        assert result is True
        client.create_index.assert_called_once()
        args = client.create_index.call_args.kwargs
        assert args["name"] == "sloughgpt"
        assert args["dimension"] == 768
        assert args["metric"] == "cosine"
        assert args["spec"] is serverless_spec
        assert store.serverless_spec is serverless_spec

    @pytest.mark.asyncio
    async def test_connect_creates_pod_index(self):
        store = PineconeVectorStore(api_key="k", environment="eu-central-1")
        client = MagicMock()
        client.list_indexes.return_value = []
        index = MagicMock()
        client.Index.return_value = index
        pinecone = MagicMock(Pinecone=MagicMock(return_value=client))
        pod_spec = MagicMock()
        pinecone.ServerlessSpec.side_effect = AssertionError("should not be used")
        pinecone.PodSpec.return_value = pod_spec

        with patch.dict("sys.modules", {"pinecone": pinecone}):
            result = await store.connect()

        assert result is True
        assert store.pod_spec is pod_spec
        assert client.create_index.call_args.kwargs["spec"] is pod_spec

    @pytest.mark.asyncio
    async def test_connect_import_error(self):
        store = PineconeVectorStore(api_key="k")
        with patch.dict("sys.modules", {"pinecone": None}):
            with pytest.raises(ImportError):
                await store.connect()

    @pytest.mark.asyncio
    async def test_connect_generic_failure_returns_false(self):
        store = PineconeVectorStore(api_key="k")
        pinecone = MagicMock(Pinecone=MagicMock(side_effect=RuntimeError("boom")))
        with patch.dict("sys.modules", {"pinecone": pinecone}):
            result = await store.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_disconnect(self):
        store = PineconeVectorStore(api_key="k")
        store.index = MagicMock()
        await store.disconnect()
        assert store.index is None


class TestOperations:

    @pytest.mark.asyncio
    async def test_upsert_requires_connection(self):
        store = PineconeVectorStore(api_key="k")
        with pytest.raises(RuntimeError, match="Not connected"):
            await store.upsert([VectorEntry(id="a", vector=[1.0], text="t")])

    @pytest.mark.asyncio
    async def test_upsert(self):
        store = PineconeVectorStore(api_key="k")
        store.index = MagicMock()
        entries = [
            VectorEntry(id="a", vector=[1.0, 2.0], text="one", metadata={"k": 1}),
            VectorEntry(id="b", vector=[3.0, 4.0], text="two"),
        ]
        n = await store.upsert(entries)
        assert n == 2
        vectors = store.index.upsert.call_args.kwargs["vectors"]
        assert vectors[0]["id"] == "a"
        assert vectors[0]["values"] == [1.0, 2.0]
        assert vectors[0]["metadata"] == {"text": "one", "k": 1}
        assert vectors[1]["metadata"] == {"text": "two"}

    @pytest.mark.asyncio
    async def test_query_requires_connection(self):
        store = PineconeVectorStore(api_key="k")
        with pytest.raises(RuntimeError, match="Not connected"):
            await store.query([1.0, 2.0])

    @pytest.mark.asyncio
    async def test_query(self):
        store = PineconeVectorStore(api_key="k")
        store.index = MagicMock()
        store.index.query.return_value = {
            "matches": [
                {"id": "a", "score": 0.9, "metadata": {"text": "one", "k": 1}},
                {"id": "b", "score": 0.4, "metadata": {"text": "two"}},
            ]
        }
        results = await store.query([1.0, 2.0], top_k=2)
        assert len(results) == 2
        assert results[0].id == "a"
        assert results[0].score == pytest.approx(0.9)
        assert results[0].text == "one"
        assert results[0].metadata == {"k": 1}
        store.index.query.assert_called_once_with(
            vector=[1.0, 2.0], top_k=2, include_metadata=True
        )

    @pytest.mark.asyncio
    async def test_query_passes_filter(self):
        store = PineconeVectorStore(api_key="k")
        store.index = MagicMock()
        store.index.query.return_value = {"matches": []}
        await store.query([1.0], filter_metadata={"k": 1})
        assert store.index.query.call_args.kwargs["filter"] == {"k": 1}

    @pytest.mark.asyncio
    async def test_query_empty_matches(self):
        store = PineconeVectorStore(api_key="k")
        store.index = MagicMock()
        store.index.query.return_value = {}
        assert await store.query([1.0]) == []

    @pytest.mark.asyncio
    async def test_delete_requires_connection(self):
        store = PineconeVectorStore(api_key="k")
        assert await store.delete(["a"]) is False

    @pytest.mark.asyncio
    async def test_delete(self):
        store = PineconeVectorStore(api_key="k")
        store.index = MagicMock()
        assert await store.delete(["a"]) is True
        store.index.delete.assert_called_once_with(ids=["a"])

    @pytest.mark.asyncio
    async def test_count_requires_connection(self):
        store = PineconeVectorStore(api_key="k")
        assert await store.count() == 0

    @pytest.mark.asyncio
    async def test_count(self):
        store = PineconeVectorStore(api_key="k")
        store.index = MagicMock()
        store.index.describe_index_stats.return_value = {"total_vector_count": 42}
        assert await store.count() == 42
