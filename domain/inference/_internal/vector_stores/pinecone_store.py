"""Pinecone-backed ``VectorStore`` implementation."""

from __future__ import annotations

import logging
import os
from typing import Any

from domain.inference._internal.vector_store import QueryResult, VectorEntry, VectorStore

logger = logging.getLogger("slo.inference.vector_stores.pinecone")


class PineconeVectorStore(VectorStore):
    """Pinecone vector store with serverless and pod support."""

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str = "sloughgpt",
        environment: str = "us-east-1",
        dimension: int = 768,
        metric: str = "cosine",
        host: str | None = None,
    ):
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name
        self.environment = environment
        self.dimension = dimension
        self.metric = metric
        self.host = host
        self.index = None
        self.serverless_spec = None
        self.pod_spec = None

    async def connect(self) -> bool:
        try:
            from pinecone import Pinecone, PodSpec, ServerlessSpec

            if not self.api_key:
                raise ValueError("PINECONE_API_KEY is required")

            self.client = Pinecone(api_key=self.api_key)

            if self.index_name not in [idx.name for idx in self.client.list_indexes()]:
                if self.environment in ["us-east-1", "us-west-2", "eu-west-1"]:
                    self.serverless_spec = ServerlessSpec(
                        cloud=self.environment.split("-")[0].upper(),
                        region=self.environment,
                    )
                    self.client.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric=self.metric,
                        spec=self.serverless_spec,
                    )
                else:
                    self.pod_spec = PodSpec(
                        environment=self.environment,
                        replicas=1,
                        shards=1,
                    )
                    self.client.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric=self.metric,
                        spec=self.pod_spec,
                    )

            self.index = self.client.Index(self.index_name)
            return True
        except ImportError:
            raise ImportError("pip install pinecone-client") from None
        except Exception as e:
            logger.error("Pinecone connection failed: %s", e, extra={"tag": "INF"})
            return False

    async def disconnect(self) -> None:
        self.index = None

    async def upsert(self, entries: list[VectorEntry]) -> int:
        if not self.index:
            raise RuntimeError("Not connected to Pinecone")

        vectors = []
        for entry in entries:
            vectors.append(
                {
                    "id": entry.id,
                    "values": entry.vector,
                    "metadata": {
                        "text": entry.text,
                        **entry.metadata,
                    },
                }
            )

        self.index.upsert(vectors=vectors)
        return len(entries)

    async def query(
        self,
        vector: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[QueryResult]:
        if not self.index:
            raise RuntimeError("Not connected to Pinecone")

        kwargs: dict[str, Any] = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": True,
        }
        if filter_metadata:
            kwargs["filter"] = filter_metadata

        response = self.index.query(**kwargs)

        out: list[QueryResult] = []
        if response and response.get("matches"):
            for match in response["matches"]:
                metadata = match.get("metadata") or {}
                out.append(
                    QueryResult(
                        id=match["id"],
                        score=match["score"],
                        text=metadata.pop("text", ""),
                        metadata=dict(metadata),
                    )
                )
        return out

    async def delete(self, ids: list[str]) -> bool:
        if not self.index:
            return False
        self.index.delete(ids=ids)
        return True

    async def count(self) -> int:
        if not self.index:
            return 0
        stats = self.index.describe_index_stats()
        return stats.get("total_vector_count", 0)
