"""Tests for domains/inference/vector_stores/ package + create_vector_store factory.

External providers (ChromaDB, Pinecone) are faked via ``sys.modules`` injection
since their real clients are heavy and unavailable in the test environment.
"""

import os
import sys
import types

import pytest

from domains.inference.vector_store import (
    InMemoryVectorStore,
    MogDBVectorStore,
    VectorEntry,
    create_vector_store,
)
from domains.inference.vector_stores import (
    ChromaDBVectorStore,
    PineconeVectorStore,
)
from domains.inference.vector_stores import chromadb_store, pinecone_store


def test_package_re_exports():
    assert ChromaDBVectorStore is chromadb_store.ChromaDBVectorStore
    assert PineconeVectorStore is pinecone_store.PineconeVectorStore


# =========================================================================
# Fake provider modules
# =========================================================================


class FakeCollection:
    def __init__(self):
        self.upsert_calls = []
        self.delete_calls = []
        self.query_result = None
        self.count_result = 3

    def upsert(self, **kwargs):
        self.upsert_calls.append(kwargs)

    def query(self, **kwargs):
        return self.query_result

    def delete(self, ids=None, **kwargs):
        self.delete_calls.append(ids)

    def count(self):
        return self.count_result


class FakeChromaClient:
    def __init__(self, path):
        self.path = path
        self.collection = FakeCollection()

    def get_or_create_collection(self, name):
        return self.collection


def _install_fake_chromadb(monkeypatch, client_factory):
    chromadb_mod = types.ModuleType("chromadb")
    chromadb_mod.PersistentClient = client_factory
    config_mod = types.ModuleType("chromadb.config")
    config_mod.Settings = type("Settings", (), {})
    monkeypatch.setitem(sys.modules, "chromadb", chromadb_mod)
    monkeypatch.setitem(sys.modules, "chromadb.config", config_mod)


class FakeIndex:
    def __init__(self):
        self.upsert_calls = []
        self.delete_calls = []
        self.query_result = None
        self.stats = {"total_vector_count": 7}

    def upsert(self, vectors=None, **kwargs):
        self.upsert_calls.append(vectors)

    def query(self, **kwargs):
        return self.query_result

    def delete(self, ids=None, **kwargs):
        self.delete_calls.append(ids)

    def describe_index_stats(self):
        return self.stats


class FakePineconeClient:
    def __init__(self, api_key=None, existing_indexes=None):
        self.api_key = api_key
        self._existing = existing_indexes or []
        self._indexes = {}
        self.created = []

    def list_indexes(self):
        return [type("Idx", (), {"name": n})() for n in self._existing]

    def create_index(self, name, dimension, metric, spec):
        self.created.append(
            {"name": name, "dimension": dimension, "metric": metric, "spec": spec}
        )

    def Index(self, name):
        if name not in self._indexes:
            self._indexes[name] = FakeIndex()
        return self._indexes[name]


def _install_fake_pinecone(monkeypatch, client_factory):
    pinecone_mod = types.ModuleType("pinecone")
    pinecone_mod.Pinecone = client_factory
    pinecone_mod.ServerlessSpec = (
        lambda cloud, region: {"kind": "serverless", "cloud": cloud, "region": region}
    )
    pinecone_mod.PodSpec = lambda environment, replicas, shards: {
        "kind": "pod",
        "environment": environment,
    }
    monkeypatch.setitem(sys.modules, "pinecone", pinecone_mod)


class FakeMogCollection:
    def find(self):
        return []

    def update_one(self, *a, **k):
        return None

    def insert_one(self, *a, **k):
        return None

    def delete_one(self, *a, **k):
        return None


class FakeMogDB:
    def __init__(self, path):
        self.path = path

    def collection(self, name):
        return FakeMogCollection()

    def close(self):
        pass


def _install_fake_mogdb(monkeypatch):
    mogdb_mod = types.ModuleType("mogdb")
    mogdb_mod.MogDB = FakeMogDB
    monkeypatch.setitem(sys.modules, "mogdb", mogdb_mod)


# =========================================================================
# ChromaDBVectorStore tests
# =========================================================================


class TestChromaDBVectorStore:
    def _store(self):
        return ChromaDBVectorStore(
            persist_directory="/tmp/chroma-test",
            collection_name="test_col",
        )

    @pytest.mark.asyncio
    async def test_connect_success(self, monkeypatch):
        _install_fake_chromadb(monkeypatch, FakeChromaClient)
        store = self._store()
        assert await store.connect() is True
        assert isinstance(store.client, FakeChromaClient)
        assert store.collection is store.client.collection

    @pytest.mark.asyncio
    async def test_connect_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "chromadb", None)
        store = self._store()
        with pytest.raises(ImportError, match="pip install chromadb"):
            await store.connect()

    @pytest.mark.asyncio
    async def test_connect_generic_exception(self, monkeypatch):
        class BrokenClient:
            def __init__(self, path):
                raise OSError("disk full")

        _install_fake_chromadb(monkeypatch, BrokenClient)
        store = self._store()
        assert await store.connect() is False

    @pytest.mark.asyncio
    async def test_disconnect(self, monkeypatch):
        _install_fake_chromadb(monkeypatch, FakeChromaClient)
        store = self._store()
        await store.connect()
        await store.disconnect()
        assert store.client is None
        assert store.collection is None

    @pytest.mark.asyncio
    async def test_upsert_not_connected(self):
        store = self._store()
        with pytest.raises(RuntimeError, match="Not connected"):
            await store.upsert([VectorEntry(id="a", vector=[1.0], text="t")])

    @pytest.mark.asyncio
    async def test_upsert_connected(self, monkeypatch):
        _install_fake_chromadb(monkeypatch, FakeChromaClient)
        store = self._store()
        await store.connect()
        n = await store.upsert(
            [
                VectorEntry(id="a", vector=[1.0, 0.0], text="alpha", metadata={"k": "v"}),
                VectorEntry(id=None, vector=[0.0, 1.0], text="beta"),
            ]
        )
        assert n == 2
        call = store.collection.upsert_calls[-1]
        assert call["ids"] == ["a", "entry_1"]
        assert call["embeddings"] == [[1.0, 0.0], [0.0, 1.0]]
        assert call["documents"] == ["alpha", "beta"]
        assert call["metadatas"] == [{"k": "v"}, {}]

    @pytest.mark.asyncio
    async def test_query_not_connected(self):
        store = self._store()
        with pytest.raises(RuntimeError, match="Not connected"):
            await store.query([1.0])

    @pytest.mark.asyncio
    async def test_query_full_results(self, monkeypatch):
        _install_fake_chromadb(monkeypatch, FakeChromaClient)
        store = self._store()
        await store.connect()
        store.collection.query_result = {
            "ids": [["id1", "id2"]],
            "distances": [[0.1, 0.2]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"a": 1}, {"a": 2}]],
        }
        results = await store.query([1.0, 0.0], top_k=5)
        assert len(results) == 2
        assert results[0].id == "id1"
        assert results[0].score == 0.1
        assert results[0].text == "doc1"
        assert results[0].metadata == {"a": 1}

    @pytest.mark.asyncio
    async def test_query_minimal_results(self, monkeypatch):
        _install_fake_chromadb(monkeypatch, FakeChromaClient)
        store = self._store()
        await store.connect()
        store.collection.query_result = {"ids": [["id1"]]}
        results = await store.query([1.0])
        assert len(results) == 1
        assert results[0].id == "id1"
        assert results[0].score == 0.0
        assert results[0].text == ""
        assert results[0].metadata == {}

    @pytest.mark.asyncio
    async def test_query_no_results(self, monkeypatch):
        _install_fake_chromadb(monkeypatch, FakeChromaClient)
        store = self._store()
        await store.connect()
        store.collection.query_result = {"ids": [[]]}
        assert await store.query([1.0]) == []

    @pytest.mark.asyncio
    async def test_delete_not_connected(self):
        assert await self._store().delete(["a"]) is False

    @pytest.mark.asyncio
    async def test_delete_connected(self, monkeypatch):
        _install_fake_chromadb(monkeypatch, FakeChromaClient)
        store = self._store()
        await store.connect()
        assert await store.delete(["a", "b"]) is True
        assert store.collection.delete_calls == [["a", "b"]]

    @pytest.mark.asyncio
    async def test_count_not_connected(self):
        assert await self._store().count() == 0

    @pytest.mark.asyncio
    async def test_count_connected(self, monkeypatch):
        _install_fake_chromadb(monkeypatch, FakeChromaClient)
        store = self._store()
        await store.connect()
        assert await store.count() == 3


# =========================================================================
# PineconeVectorStore tests
# =========================================================================


class TestPineconeVectorStore:
    def test_init_uses_env_api_key(self, monkeypatch):
        monkeypatch.setenv("PINECONE_API_KEY", "envkey")
        store = PineconeVectorStore()
        assert store.api_key == "envkey"
        assert store.index_name == "sloughgpt"

    @pytest.mark.asyncio
    async def test_connect_missing_key_returns_false(self, monkeypatch):
        _install_fake_pinecone(monkeypatch, FakePineconeClient)
        store = PineconeVectorStore(api_key=None, index_name="idx")
        assert await store.connect() is False

    @pytest.mark.asyncio
    async def test_connect_existing_index(self, monkeypatch):
        client = FakePineconeClient(existing_indexes=["idx"])
        _install_fake_pinecone(monkeypatch, lambda api_key: client)
        store = PineconeVectorStore(api_key="k", index_name="idx")
        assert await store.connect() is True
        assert client.created == []
        assert store.index is client._indexes["idx"]

    @pytest.mark.asyncio
    async def test_connect_creates_serverless_index(self, monkeypatch):
        client = FakePineconeClient()
        _install_fake_pinecone(monkeypatch, lambda api_key: client)
        store = PineconeVectorStore(api_key="k", index_name="idx", environment="us-east-1")
        assert await store.connect() is True
        created = client.created[-1]
        assert created["name"] == "idx"
        assert created["dimension"] == 768
        assert created["spec"]["kind"] == "serverless"

    @pytest.mark.asyncio
    async def test_connect_creates_pod_index(self, monkeypatch):
        client = FakePineconeClient()
        _install_fake_pinecone(monkeypatch, lambda api_key: client)
        store = PineconeVectorStore(api_key="k", index_name="idx", environment="custom-env")
        assert await store.connect() is True
        assert client.created[-1]["spec"]["kind"] == "pod"

    @pytest.mark.asyncio
    async def test_connect_import_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pinecone", None)
        store = PineconeVectorStore(api_key="k")
        with pytest.raises(ImportError, match="pip install pinecone-client"):
            await store.connect()

    @pytest.mark.asyncio
    async def test_connect_generic_exception(self, monkeypatch):
        def boom(api_key):
            raise OSError("timeout")

        _install_fake_pinecone(monkeypatch, boom)
        store = PineconeVectorStore(api_key="k")
        assert await store.connect() is False

    @pytest.mark.asyncio
    async def test_disconnect(self, monkeypatch):
        _install_fake_pinecone(monkeypatch, FakePineconeClient)
        store = PineconeVectorStore(api_key="k", index_name="idx")
        await store.connect()
        await store.disconnect()
        assert store.index is None

    @pytest.mark.asyncio
    async def test_upsert_not_connected(self):
        store = PineconeVectorStore(api_key="k")
        with pytest.raises(RuntimeError, match="Not connected"):
            await store.upsert([VectorEntry(id="a", vector=[1.0], text="t")])

    @pytest.mark.asyncio
    async def test_upsert_connected(self, monkeypatch):
        _install_fake_pinecone(monkeypatch, FakePineconeClient)
        store = PineconeVectorStore(api_key="k", index_name="idx")
        await store.connect()
        n = await store.upsert(
            [
                VectorEntry(id="a", vector=[1.0, 0.0], text="alpha", metadata={"k": "v"}),
                VectorEntry(id="b", vector=[0.0, 1.0], text="beta"),
            ]
        )
        assert n == 2
        vectors = store.index.upsert_calls[-1]
        assert vectors[0] == {"id": "a", "values": [1.0, 0.0], "metadata": {"text": "alpha", "k": "v"}}
        assert vectors[1] == {"id": "b", "values": [0.0, 1.0], "metadata": {"text": "beta"}}

    @pytest.mark.asyncio
    async def test_query_not_connected(self):
        store = PineconeVectorStore(api_key="k")
        with pytest.raises(RuntimeError, match="Not connected"):
            await store.query([1.0])

    @pytest.mark.asyncio
    async def test_query_matches(self, monkeypatch):
        _install_fake_pinecone(monkeypatch, FakePineconeClient)
        store = PineconeVectorStore(api_key="k", index_name="idx")
        await store.connect()
        store.index.query_result = {
            "matches": [
                {"id": "m1", "score": 0.9, "metadata": {"text": "hello", "lang": "en"}},
                {"id": "m2", "score": 0.8},
            ]
        }
        results = await store.query([1.0], top_k=5, filter_metadata={"lang": "en"})
        assert len(results) == 2
        assert results[0].id == "m1"
        assert results[0].score == 0.9
        assert results[0].text == "hello"
        assert results[0].metadata == {"lang": "en"}
        assert results[1].text == ""

    @pytest.mark.asyncio
    async def test_query_no_matches(self, monkeypatch):
        _install_fake_pinecone(monkeypatch, FakePineconeClient)
        store = PineconeVectorStore(api_key="k", index_name="idx")
        await store.connect()
        store.index.query_result = {}
        assert await store.query([1.0]) == []

    @pytest.mark.asyncio
    async def test_delete_not_connected(self):
        assert await PineconeVectorStore(api_key="k").delete(["a"]) is False

    @pytest.mark.asyncio
    async def test_delete_connected(self, monkeypatch):
        _install_fake_pinecone(monkeypatch, FakePineconeClient)
        store = PineconeVectorStore(api_key="k", index_name="idx")
        await store.connect()
        assert await store.delete(["a"]) is True
        assert store.index.delete_calls == [["a"]]

    @pytest.mark.asyncio
    async def test_count_not_connected(self):
        assert await PineconeVectorStore(api_key="k").count() == 0

    @pytest.mark.asyncio
    async def test_count_connected(self, monkeypatch):
        _install_fake_pinecone(monkeypatch, FakePineconeClient)
        store = PineconeVectorStore(api_key="k", index_name="idx")
        await store.connect()
        assert await store.count() == 7


# =========================================================================
# create_vector_store factory tests
# =========================================================================


class TestCreateVectorStore:
    @pytest.mark.asyncio
    async def test_in_memory_default(self):
        store = await create_vector_store("in_memory")
        assert isinstance(store, InMemoryVectorStore)
        assert store.dimension == 384

    @pytest.mark.asyncio
    async def test_in_memory_aliases(self):
        for alias in ("memory", "local"):
            store = await create_vector_store(alias, dimension=128)
            assert isinstance(store, InMemoryVectorStore)
            assert store.dimension == 128

    @pytest.mark.asyncio
    async def test_in_memory_none_provider(self):
        store = await create_vector_store(None)
        assert isinstance(store, InMemoryVectorStore)

    @pytest.mark.asyncio
    async def test_mogdb_aliases(self, monkeypatch):
        _install_fake_mogdb(monkeypatch)
        for alias in ("mogdb", "persist", "persistent"):
            store = await create_vector_store(alias, dimension=384, path="/tmp/mogdb-test")
            assert isinstance(store, MogDBVectorStore)
            await store.disconnect()

    @pytest.mark.asyncio
    async def test_chromadb(self, monkeypatch):
        _install_fake_chromadb(monkeypatch, FakeChromaClient)
        store = await create_vector_store("chromadb")
        assert isinstance(store, ChromaDBVectorStore)
        assert store.collection is not None

    @pytest.mark.asyncio
    async def test_pinecone(self, monkeypatch):
        _install_fake_pinecone(monkeypatch, FakePineconeClient)
        store = await create_vector_store(
            "pinecone", api_key="k", index="custom", dimension=512
        )
        assert isinstance(store, PineconeVectorStore)
        assert store.index_name == "custom"
        assert store.dimension == 512
        assert store.index is not None

    @pytest.mark.asyncio
    async def test_pinecone_connect_failure(self, monkeypatch):
        def exploding(api_key):
            raise RuntimeError("pinecone unavailable")

        _install_fake_pinecone(monkeypatch, exploding)
        with pytest.raises(RuntimeError, match="Pinecone connection failed"):
            await create_vector_store("pinecone", api_key="k")

    @pytest.mark.asyncio
    async def test_unknown_provider(self):
        with pytest.raises(NotImplementedError, match="not implemented"):
            await create_vector_store("weaviate")
