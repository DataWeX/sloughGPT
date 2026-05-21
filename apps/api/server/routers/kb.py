"""
Knowledge Router - Knowledge base management backed by KnowledgeMemory.

All operations use the vector-store-backed KnowledgeMemory (the same store
used by entity_extractor, soul engine prompt injection, and chat enrichment).
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import time

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


class KnowledgeItemOut(BaseModel):
    id: str
    content: str
    topic: str = "general"
    source: str = ""
    url: str = ""
    timestamp: float = 0.0
    importance: float = 0.5
    score: float = 0.0


class KnowledgeCreate(BaseModel):
    content: str = Field(..., min_length=1)
    topic: str = "injected"
    source: str = "manual"
    importance: float = 0.7


class KnowledgeBatchItem(BaseModel):
    content: str
    source: str = "injected"
    tags: List[str] = []


class KnowledgeBatchRequest(BaseModel):
    items: List[KnowledgeBatchItem]


def _get_memory():
    from domains.learner.knowledge import get_knowledge_memory, KnowledgeFact
    return get_knowledge_memory()


def _fact_from_entry(entry: dict) -> KnowledgeItemOut:
    return KnowledgeItemOut(
        id=entry.get("id", ""),
        content=entry.get("content", ""),
        topic=entry.get("topic", "general"),
        source=entry.get("source", ""),
        url=entry.get("url", ""),
        timestamp=entry.get("timestamp", 0.0),
        importance=entry.get("importance", 0.5),
        score=entry.get("score", 0.0),
    )


@router.get("", response_model=List[KnowledgeItemOut])
def list_knowledge():
    memory = _get_memory()
    entries = memory.list_all()
    return [_fact_from_entry(e) for e in entries]


@router.post("")
def add_knowledge(req: KnowledgeCreate):
    from domains.learner.knowledge import KnowledgeFact
    memory = _get_memory()
    fact = KnowledgeFact(
        content=req.content,
        topic=req.topic,
        source=req.source,
        importance=req.importance,
    )
    memory.add_fact(fact)
    return {"status": "stored", "content": req.content}


@router.post("/batch")
def batch_ingest(req: KnowledgeBatchRequest):
    """Store multiple knowledge items in KnowledgeMemory."""
    from domains.learner.knowledge import KnowledgeFact
    memory = _get_memory()
    stored = 0
    for item in req.items:
        fact = KnowledgeFact(
            content=item.content,
            topic="injected",
            source=item.source,
            timestamp=time.time(),
            importance=0.7,
        )
        if memory.add_fact(fact):
            stored += 1
    return {"stored": stored}


@router.get("/search")
def search_knowledge(query: str = ""):
    memory = _get_memory()
    results = memory.search(query, top_k=20) if query else []
    return {
        "results": [_fact_from_entry(r) for r in results],
        "count": len(results),
    }


@router.delete("/{item_id}")
def delete_knowledge(item_id: str):
    memory = _get_memory()
    if memory.delete_by_id(item_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Item not found")


@router.get("/context")
def get_context():
    memory = _get_memory()
    context = memory.get_context_string(max_items=50)
    all_facts = memory.list_all()
    return {"context": context, "count": len(all_facts)}
