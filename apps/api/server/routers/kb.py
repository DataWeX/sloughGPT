"""
Knowledge Router - Knowledge base management backed by KnowledgeMemory.

All operations use the vector-store-backed KnowledgeMemory (the same store
used by entity_extractor, soul engine prompt injection, and chat enrichment).
"""
from fastapi import APIRouter, HTTPException, Query
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
    topic: str = "general"
    source: str = "manual"
    importance: float = 0.7
    auto_tag: bool = False


class KnowledgeUpdate(BaseModel):
    content: Optional[str] = None
    topic: Optional[str] = None
    importance: Optional[float] = None


class KnowledgeBatchItem(BaseModel):
    content: str
    source: str = "injected"
    tags: List[str] = []


class KnowledgeBatchRequest(BaseModel):
    items: List[KnowledgeBatchItem]


class BatchDeleteRequest(BaseModel):
    ids: List[str]


class SuggestTopicRequest(BaseModel):
    content: str = Field(..., min_length=1)


class UrlIngestRequest(BaseModel):
    url: str = Field(..., min_length=1)
    source: str = "direct"


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


def _auto_tag(content: str) -> str:
    """
    Auto-detect the best topic for content using simple TF scoring.
    Returns one of the known topics or the best keyword match.
    """
    known_topics = ["general", "code", "docs", "reference", "persona", "science", "tech", "business"]
    words = content.lower().split()
    scores = {t: 0 for t in known_topics}
    code_indicators = ["function", "def ", "class ", "import ", "const ", "var ", "return", "=>", "```"]
    doc_indicators = ["documentation", "guide", "tutorial", "how to", "reference", "manual"]
    science_indicators = ["study", "research", "experiment", "data", "analysis", "hypothesis", "theory"]
    tech_indicators = ["software", "hardware", "api", "server", "database", "cloud", "deploy"]
    business_indicators = ["revenue", "market", "strategy", "customer", "product", "growth", "startup"]
    persona_indicators = ["personality", "trait", "voice", "tone", "character", "style", "soul"]

    # Score based on keyword matches in content
    content_lower = content.lower()
    for w in words:
        w_clean = w.strip(".,!?;:'\"()[]{}")
        if w_clean in tech_indicators: scores["tech"] += 2
        if w_clean in science_indicators: scores["science"] += 2
        if w_clean in business_indicators: scores["business"] += 2
        if w_clean in persona_indicators: scores["persona"] += 2
        if w_clean in doc_indicators: scores["docs"] += 2
        if w_clean in code_indicators: scores["code"] += 2

    # Multi-word indicators
    for ind in code_indicators:
        if ind in content_lower: scores["code"] += 3
    for ind in doc_indicators:
        if ind in content_lower: scores["docs"] += 2
    for ind in science_indicators:
        if ind in content_lower: scores["science"] += 3
    for ind in tech_indicators:
        if ind in content_lower: scores["tech"] += 3
    for ind in business_indicators:
        if ind in content_lower: scores["business"] += 3
    for ind in persona_indicators:
        if ind in content_lower: scores["persona"] += 3

    # If content has code blocks, strongly favor code
    if "```" in content or content_lower.count("function") > 1:
        scores["code"] += 5

    # If content is very short, check for topic words
    if len(words) < 10:
        for w in words:
            w_clean = w.strip(".,!?;:'\"()[]{}")
            for t in known_topics[1:]:
                if w_clean == t:
                    scores[t] += 5

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


@router.get("", response_model=List[KnowledgeItemOut])
def list_knowledge(limit: int = Query(200, ge=1, le=5000), offset: int = Query(0, ge=0)):
    """List knowledge items with optional pagination."""
    memory = _get_memory()
    entries = memory.list_all(top_k=limit + offset)
    entries = entries[offset:offset + limit]
    return [_fact_from_entry(e) for e in entries]


@router.post("")
def add_knowledge(req: KnowledgeCreate):
    from domains.learner.knowledge import KnowledgeFact
    memory = _get_memory()
    topic = req.topic if not req.auto_tag else _auto_tag(req.content)
    fact = KnowledgeFact(
        content=req.content,
        topic=topic,
        source=req.source,
        importance=req.importance,
    )
    memory.add_fact(fact)
    return {"status": "stored", "content": req.content, "topic": topic}


@router.patch("/{item_id}")
def update_knowledge(item_id: str, req: KnowledgeUpdate):
    """Update a knowledge item's content, topic, or importance."""
    memory = _get_memory()
    all_items = memory.list_all(top_k=5000)
    target = None
    for item in all_items:
        if item.get("id") == item_id:
            target = item
            break
    if not target:
        raise HTTPException(status_code=404, detail="Item not found")

    # Build updated fact
    from domains.learner.knowledge import KnowledgeFact
    new_fact = KnowledgeFact(
        content=req.content if req.content is not None else target["content"],
        topic=req.topic if req.topic is not None else target.get("topic", "general"),
        source=target.get("source", "manual"),
        url=target.get("url", ""),
        timestamp=target.get("timestamp", time.time()),
        importance=req.importance if req.importance is not None else target.get("importance", 0.5),
    )

    # Delete old, add new
    memory.delete_by_id(item_id)
    ok = memory.add_fact(new_fact)
    return {"status": "updated" if ok else "stored"}


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


@router.get("/stats")
def knowledge_stats():
    """Return knowledge base statistics."""
    memory = _get_memory()
    all_items = memory.list_all(top_k=5000)
    topics: dict[str, int] = {}
    sources: dict[str, int] = {}
    total = len(all_items)
    for item in all_items:
        t = item.get("topic", "general")
        topics[t] = topics.get(t, 0) + 1
        s = item.get("source", "unknown")
        sources[s] = sources.get(s, 0) + 1
    avg_importance = sum(item.get("importance", 0.5) for item in all_items) / max(total, 1)
    return {
        "total_items": total,
        "topics": topics,
        "topic_count": len(topics),
        "sources": sources,
        "avg_importance": round(avg_importance, 3),
        "searchable": True,
    }


@router.get("/topics")
def list_topics():
    """List all unique topics with item counts."""
    memory = _get_memory()
    all_items = memory.list_all(top_k=5000)
    topics: dict[str, int] = {}
    for item in all_items:
        t = item.get("topic", "general")
        topics[t] = topics.get(t, 0) + 1
    sorted_topics = sorted(topics.items(), key=lambda x: -x[1])
    return {
        "topics": [{"name": t, "count": c} for t, c in sorted_topics],
        "total": len(topics),
    }


@router.post("/ingest-url")
def ingest_url(req: UrlIngestRequest):
    """Ingest a URL into the knowledge base."""
    try:
        from domains.learner.knowledge import get_knowledge_ingestor
        ingestor = get_knowledge_ingestor()
        result = ingestor.ingest_url(req.url)
        return {
            "status": result.get("status", "ok"),
            "new_facts": result.get("new_facts", 0),
            "title": result.get("title", ""),
            "content_length": result.get("content_length", 0),
            "rejected": result.get("rejected", False),
            "reason": result.get("reason"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@router.post("/batch-delete")
def batch_delete_knowledge(req: BatchDeleteRequest):
    """Delete multiple knowledge items by ID."""
    memory = _get_memory()
    deleted = 0
    for item_id in req.ids:
        if memory.delete_by_id(item_id):
            deleted += 1
    return {"deleted": deleted}


@router.post("/suggest-topic")
def suggest_topic(req: SuggestTopicRequest):
    """Return the best auto-detected topic for content without storing."""
    topic = _auto_tag(req.content)
    return {"topic": topic, "confidence": "high" if topic != "general" else "low"}


@router.delete("/{item_id}")
def delete_knowledge(item_id: str):
    memory = _get_memory()
    if memory.delete_by_id(item_id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Item not found")


@router.post("/train-adapter")
def train_knowledge_adapter_route():
    """Train a LoRA adapter on all knowledge facts to bake them into model weights."""
    from domains.infrastructure.knowledge_weight_integrator import train_knowledge_adapter, get_adapter_status
    memory = _get_memory()
    facts = memory.list_all(top_k=5000)

    result = train_knowledge_adapter(
        knowledge_facts=facts,
        num_epochs=2,
    )

    status = get_adapter_status()
    return {**result, "adapter_status": status}


@router.get("/adapter-status")
def knowledge_adapter_status():
    """Return status of the knowledge weight adapter."""
    from domains.infrastructure.knowledge_weight_integrator import get_adapter_status
    return get_adapter_status()


@router.get("/{item_id}/related")
def related_knowledge(item_id: str, top_k: int = Query(6, ge=1, le=20)):
    """Return semantically related knowledge items, excluding the current one."""
    memory = _get_memory()
    all_items = memory.list_all(top_k=5000)
    target = None
    for item in all_items:
        if item.get("id") == item_id:
            target = item
            break
    if not target:
        raise HTTPException(status_code=404, detail="Item not found")
    results = memory.search(target.get("content", ""), top_k=top_k + 1)
    related = [r for r in results if r.get("id") != item_id][:top_k]
    return {"items": [_fact_from_entry(r) for r in related], "count": len(related)}


@router.get("/context")
def get_context():
    memory = _get_memory()
    context = memory.get_context_string(max_items=50)
    all_facts = memory.list_all()
    return {"context": context, "count": len(all_facts)}
