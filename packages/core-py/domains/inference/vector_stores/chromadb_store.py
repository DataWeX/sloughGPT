"""ChromaDB-backed ``VectorStore`` implementation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from domains.inference.vector_store import VectorEntry, VectorStore, QueryResult

logger = logging.getLogger("man.inference.vector_stores.chromadb")


class ChromaDBVectorStore(VectorStore):
    """ChromaDB vector store with persistent storage."""

    def __init__(
        self,
        persist_directory: str = "data/vector_store",
        collection_name: str = "sloughgpt",
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None

    async def connect(self) -> bool:
        try:
            import chromadb
            from chromadb.config import Settings

            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(name=self.collection_name)
            return True
        except ImportError:
            raise ImportError("pip install chromadb")
        except Exception as e:
            logger.error("ChromaDB connection failed: %s", e, extra={"tag": "INF"})
            return False

    async def disconnect(self) -> None:
        self.collection = None
        self.client = None

    async def upsert(self, entries: List[VectorEntry]) -> int:
        if not self.collection:
            raise RuntimeError("Not connected to ChromaDB")
        ids = [e.id or f"entry_{i}" for i, e in enumerate(entries)]
        embeddings = [list(e.vector) for e in entries]
        documents = [e.text for e in entries]
        metadatas = [e.metadata or {} for e in entries]
        self.collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )
        return len(entries)

    async def query(
        self,
        vector: List[float],
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[QueryResult]:
        if not self.collection:
            raise RuntimeError("Not connected to ChromaDB")
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filter_metadata,
        )
        out: List[QueryResult] = []
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                out.append(
                    QueryResult(
                        id=doc_id,
                        score=float(results["distances"][0][i]) if results.get("distances") else 0.0,
                        text=results["documents"][0][i] if results.get("documents") else "",
                        metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
                    )
                )
        return out

    async def delete(self, ids: List[str]) -> bool:
        if not self.collection:
            return False
        self.collection.delete(ids=ids)
        return True

    async def count(self) -> int:
        if not self.collection:
            return 0
        return self.collection.count()
