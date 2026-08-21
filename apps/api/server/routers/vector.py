"""
Vector Store Router - Embedding/vector operations
"""
import logging
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional, List

logger = logging.getLogger(__name__)

from schemas.common import raise_error, success_response, classify_and_raise, safe_audit_log


class VectorStoreConfig(BaseModel):
    provider: str = "in_memory"
    dimension: Optional[int] = 384


class UpsertRequest(BaseModel):
    texts: List[str]
    ids: Optional[List[str]] = None
    embeddings: Optional[List[List[float]]] = None
    metadata: Optional[List[dict]] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class VectorRouter:
    def __init__(self):
        self._vector_store = None
        self._vector_store_type = "in_memory"
        self.router = APIRouter(prefix="/vector", tags=["vector"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("/init", self.init_vector_store, methods=["POST"])
        self.router.add_api_route("/stats", self.get_stats, methods=["GET"])
        self.router.add_api_route("/upsert", self.upsert_vectors, methods=["POST"])
        self.router.add_api_route("/search", self.search_vectors, methods=["POST"])
        self.router.add_api_route("/ingest/status", self.ingest_status, methods=["GET"])

    async def get_vector_store(self) -> dict:
        """get_vector_store."""
        if self._vector_store is None:
            try:
                from domains.inference.vector_store import create_vector_store
                kwargs = {"dimension": 384}
                if self._vector_store_type == "chromadb":
                    kwargs["persist_directory"] = "data/vector_store"
                self._vector_store = await create_vector_store(provider=self._vector_store_type, **kwargs)
            except Exception as e:
                logging.getLogger(__name__).debug("Vector store init failed: %s", e)
        return self._vector_store

    async def init_vector_store(self, config: VectorStoreConfig) -> dict:
        """init_vector_store."""
        self._vector_store_type = config.provider or "chromadb"
        try:
            from domains.inference.vector_store import create_vector_store
            kwargs = {"dimension": config.dimension}
            if self._vector_store_type == "chromadb":
                kwargs["persist_directory"] = "data/vector_store"
            self._vector_store = await create_vector_store(provider=self._vector_store_type, **kwargs)
            try:
                from routers.inference import set_vector_store_ref
                set_vector_store_ref(self._vector_store)
            except Exception as e:
                logger.warning("Failed to set vector store ref on inference router: %s", e)
            safe_audit_log("vector.init", resource=self._vector_store_type, detail=f"provider={self._vector_store_type} dimension={config.dimension}")
            return success_response(data={"status": "connected", "provider": self._vector_store_type})
        except ImportError:
            self._vector_store_type = "in_memory"
            self._vector_store = await create_vector_store(provider="in_memory", dimension=config.dimension)
            try:
                from routers.inference import set_vector_store_ref
                set_vector_store_ref(self._vector_store)
            except Exception as e:
                logger.warning("Failed to set vector store ref on inference router: %s", e)
            safe_audit_log("vector.init", resource="in_memory", detail=f"provider=in_memory dimension={config.dimension}")
            return success_response(data={"status": "connected", "provider": "in_memory", "note": "chromadb not installed, using in-memory store"})
        except Exception as e:
            logger.warning("Vector store init failed: %s", e)
            classify_and_raise(e, source="vector")

    async def get_stats(self) -> dict:
        """get_stats."""
        store = await self.get_vector_store()
        if not store:
            return success_response(data={"provider": self._vector_store_type, "count": 0})
        return success_response(data={"provider": self._vector_store_type, "count": await store.count()})

    async def upsert_vectors(self, request: UpsertRequest) -> dict:
        """upsert_vectors."""
        store = await self.get_vector_store()
        if not store:
            raise_error("Vector store not connected", "E_INFRA_STARTUP", status_code=500)
        from domains.inference.vector_store import VectorEntry, simple_embed
        entries = []
        for i, text in enumerate(request.texts):
            embedding = request.embeddings[i] if request.embeddings and i < len(request.embeddings) else simple_embed(text)
            entry = VectorEntry(
                id=request.ids[i] if request.ids and i < len(request.ids) else None,
                text=text,
                vector=embedding,
                metadata=request.metadata[i] if request.metadata and i < len(request.metadata) else {}
            )
            entries.append(entry)
        count = await store.upsert(entries)
        safe_audit_log("vector.upsert", resource="vector_store", detail=f"count={count}")
        return success_response(data={"status": "upserted", "count": count})

    async def search_vectors(self, request: SearchRequest) -> dict:
        """search_vectors."""
        store = await self.get_vector_store()
        if not store:
            return success_response(data={"results": []})
        from domains.inference.vector_store import simple_embed
        query_embedding = simple_embed(request.query)
        results = await store.query(query_embedding, top_k=request.top_k)
        return success_response(data={"results": [{"text": r.text, "score": r.score, "id": r.id} for r in results]})

    async def ingest_status(self) -> dict:
        """ingest_status."""
        return success_response(data={"status": "ready"})


router = VectorRouter().router