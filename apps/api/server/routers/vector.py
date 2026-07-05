"""
Vector Store Router - Embedding/vector operations
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter(prefix="/vector", tags=["vector"])

_vector_store = None
_vector_store_type = "in_memory"


class VectorStoreConfig(BaseModel):
    provider: str = "in_memory"
    dimension: Optional[int] = 768


class UpsertRequest(BaseModel):
    texts: List[str]
    ids: Optional[List[str]] = None
    embeddings: Optional[List[List[float]]] = None
    metadata: Optional[List[dict]] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


async def get_vector_store():
    global _vector_store
    if _vector_store is None:
        try:
            from domains.inference.vector_store import create_vector_store, VectorStoreType
            kwargs = {"dimension": 384}
            if _vector_store_type == "chromadb":
                kwargs["persist_directory"] = "data/vector_store"
            _vector_store = await create_vector_store(provider=_vector_store_type, **kwargs)
        except Exception as e:
            logging.getLogger(__name__).debug("Vector store init failed: %s", e)
    return _vector_store


@router.post("/init")
async def init_vector_store(config: VectorStoreConfig):
    global _vector_store, _vector_store_type
    _vector_store_type = config.provider or "chromadb"
    try:
        from domains.inference.vector_store import create_vector_store
        kwargs = {"dimension": 384}
        if _vector_store_type == "chromadb":
            kwargs["persist_directory"] = "data/vector_store"
        _vector_store = await create_vector_store(provider=_vector_store_type, **kwargs)
        # Connect to ContextCore
        try:
            from routers.inference import set_vector_store_ref
            set_vector_store_ref(_vector_store)
        except Exception:
            pass
        return {"status": "connected", "provider": _vector_store_type}
    except ImportError:
        _vector_store_type = "in_memory"
        _vector_store = await create_vector_store(provider="in_memory", dimension=384)
        try:
            from routers.inference import set_vector_store_ref
            set_vector_store_ref(_vector_store)
        except Exception:
            pass
        return {"status": "connected", "provider": "in_memory", "note": "chromadb not installed, using in-memory store"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    store = await get_vector_store()
    if not store:
        return {"provider": _vector_store_type, "count": 0}
    return {"provider": _vector_store_type, "count": await store.count()}


@router.post("/upsert")
async def upsert_vectors(request: UpsertRequest):
    store = await get_vector_store()
    if not store:
        raise HTTPException(status_code=500, detail="Vector store not connected")
    from domains.inference.vector_store import VectorEntry, simple_embed
    entries = []
    for i, text in enumerate(request.texts):
        embedding = request.embeddings[i] if request.embeddings and i < len(request.embeddings) else simple_embed(text)
        entry = VectorEntry(
            id=request.ids[i] if request.ids else None,
            text=text,
            vector=embedding,
            metadata=request.metadata[i] if request.metadata else {}
        )
        entries.append(entry)
    count = await store.upsert(entries)
    return {"status": "upserted", "count": count}


@router.post("/search")
async def search_vectors(request: SearchRequest):
    store = await get_vector_store()
    if not store:
        return {"results": []}
    from domains.inference.vector_store import simple_embed
    query_embedding = simple_embed(request.query)
    results = await store.query(query_embedding, top_k=request.top_k)
    return {"results": [{"text": r.text, "score": r.score, "id": r.id} for r in results]}


@router.get("/ingest/status")
async def ingest_status():
    return {"status": "ready"}
