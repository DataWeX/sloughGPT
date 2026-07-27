"""
Knowledge Router - Knowledge base management backed by KnowledgeMemory.

All operations use the vector-store-backed KnowledgeMemory (the same store
used by entity_extractor, soul engine prompt injection, and chat enrichment).
"""
import json
import re
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional, List
import time

from schemas.common import success_response

import urllib.parse

_BLOCKED_SCHEMES = {"file", "ftp", "data", "javascript", "vbscript"}
_ALLOWED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"}  # blocked for SSRF

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

    # Auto-label semantic category if not set
    label = ""
    try:
        from domains.infrastructure.truth_labeler import get_truth_labeler
        labeler = get_truth_labeler()
        lr = labeler.label(req.content)
        label = lr.label
    except Exception:
        pass

    fact = KnowledgeFact(
        content=req.content,
        topic=topic,
        source=req.source,
        importance=req.importance,
    )
    memory.add_fact(fact)
    return success_response(data={"status": "stored", "content": req.content, "topic": topic, "label": label})


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
    return success_response(data={"status": "updated" if ok else "stored"})


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
    return success_response(data={"stored": stored})


@router.get("/search")
def search_knowledge(query: str = ""):
    memory = _get_memory()
    results = memory.search(query, top_k=20) if query else []
    return success_response(data={
        "results": [_fact_from_entry(r) for r in results],
        "count": len(results),
    })


@router.get("/stats")
def knowledge_stats():
    """Return knowledge base statistics.

    Uses batched iteration to avoid loading all 5000 entries into memory.
    Fetches in chunks of 200 and aggregates topic/source counts incrementally.
    """
    memory = _get_memory()
    topics: dict[str, int] = {}
    sources: dict[str, int] = {}
    total = 0
    importance_sum = 0.0

    batch_size = 200
    offset = 0
    while True:
        batch = memory.list_all(top_k=batch_size)
        if not batch:
            break
        for item in batch:
            t = item.get("topic", "general")
            topics[t] = topics.get(t, 0) + 1
            s = item.get("source", "unknown")
            sources[s] = sources.get(s, 0) + 1
            importance_sum += item.get("importance", 0.5)
            total += 1
        if len(batch) < batch_size:
            break
        offset += batch_size

    avg_importance = importance_sum / max(total, 1)
    return success_response(data={
        "total_items": total,
        "topics": topics,
        "topic_count": len(topics),
        "sources": sources,
        "avg_importance": round(avg_importance, 3),
        "searchable": True,
    })


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
    return success_response(data={
        "topics": [{"name": t, "count": c} for t, c in sorted_topics],
        "total": len(topics),
    })


@router.post("/ingest-url")
def ingest_url(req: UrlIngestRequest):
    """Ingest a URL into the knowledge base."""
    try:
        parsed = urllib.parse.urlparse(req.url)
        if parsed.scheme.lower() in _BLOCKED_SCHEMES:
            raise HTTPException(status_code=400, detail=f"URL scheme '{parsed.scheme}' not allowed")
        if parsed.hostname and parsed.hostname in _ALLOWED_HOSTS:
            raise HTTPException(status_code=400, detail="Internal host URLs not allowed")
        if not parsed.scheme or parsed.scheme.lower() not in ("http", "https"):
            raise HTTPException(status_code=400, detail="Only HTTP/HTTPS URLs allowed")
        from domains.learner.knowledge import get_knowledge_ingestor
        ingestor = get_knowledge_ingestor()
        result = ingestor.ingest_url(req.url)
        return success_response(data={
            "status": result.get("status", "ok"),
            "new_facts": result.get("new_facts", 0),
            "title": result.get("title", ""),
            "content_length": result.get("content_length", 0),
            "rejected": result.get("rejected", False),
            "reason": result.get("reason"),
        })
    except HTTPException:
        raise
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
    return success_response(data={"deleted": deleted})


@router.post("/suggest-topic")
def suggest_topic(req: SuggestTopicRequest):
    """Return the best auto-detected topic for content without storing."""
    topic = _auto_tag(req.content)
    return success_response(data={"topic": topic, "confidence": "high" if topic != "general" else "low"})


@router.delete("/{item_id}")
def delete_knowledge(item_id: str):
    memory = _get_memory()
    if memory.delete_by_id(item_id):
        return success_response(data={"status": "deleted"})
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
    return success_response(data={**result, "adapter_status": status})


@router.get("/adapter-status")
def knowledge_adapter_status():
    """Return status of the knowledge weight adapter."""
    from domains.infrastructure.knowledge_weight_integrator import get_adapter_status
    return success_response(data=get_adapter_status())


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
    return success_response(data={"items": [_fact_from_entry(r) for r in related], "count": len(related)})


@router.get("/context")
def get_context():
    memory = _get_memory()
    context = memory.get_context_string(max_items=50)
    all_facts = memory.list_all()
    return success_response(data={"context": context, "count": len(all_facts)})


@router.post("/ingest-file")
async def ingest_file(
    file: UploadFile = File(...),
    topic: str = Form("imported"),
    chunk_size: int = Form(500),
    overlap: int = Form(50),
):
    """Import a textbook or document file as knowledge facts.

    Supports .txt, .md, and .json (array of strings) files.
    Large files are split into overlapping chunks of ``chunk_size``
    characters with ``overlap`` character overlap between chunks.
    Each chunk is stored as a separate knowledge fact.
    """
    if chunk_size <= overlap:
        raise HTTPException(status_code=400, detail="chunk_size must exceed overlap")
    if chunk_size < 100 or chunk_size > 10000:
        raise HTTPException(status_code=400, detail="chunk_size must be 100–10000")

    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    if file.filename and file.filename.endswith(".json"):
        import json as _json
        try:
            items = _json.loads(text)
            if isinstance(items, list):
                chunks = [str(item) for item in items if isinstance(item, str)]
            elif isinstance(items, dict):
                chunks = [str(v) for v in items.values() if isinstance(v, str)]
            else:
                raise ValueError("JSON must be an array of strings or a dict of strings")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON file: {e}")
    else:
        chunks = _chunk_text(text, chunk_size, overlap)

    from domains.learner.knowledge import KnowledgeFact
    memory = _get_memory()

    def _store_chunks():
        stored = 0
        for chunk in chunks:
            fact = KnowledgeFact(
                content=chunk,
                topic=topic,
                source=f"file:{file.filename or 'unknown'}",
                importance=min(1.0, len(chunk) / 2000),
            )
            if memory.add_fact(fact):
                stored += 1
        return stored

    import asyncio
    stored = await asyncio.to_thread(_store_chunks)

    return success_response(data={
        "status": "imported",
        "stored": stored,
        "total_chunks": len(chunks),
        "topic": topic,
        "filename": file.filename or "unknown",
        "file_size": len(raw),
    })


def _lines_for_chars(lines: list[str], target_chars: int, start: int) -> int:
    """Count how many lines from ``start`` backward cover ``target_chars``."""
    count = 0
    chars = 0
    i = start
    while i >= 0 and chars < target_chars:
        chars += len(lines[i]) + 1
        count += 1
        i -= 1
    return count


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks, respecting paragraph boundaries.

    Splits on double newlines first (paragraph breaks), then merges
    small paragraphs up to chunk_size. Oversized paragraphs are split
    at sentence boundaries. Returns chunks >= 20 characters.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks: list[str] = []
    buffer = ""

    for para in paragraphs:
        if len(para) <= chunk_size - len(buffer):
            buffer = f"{buffer}\n\n{para}" if buffer else para
        else:
            if buffer and len(buffer) >= 20:
                chunks.append(buffer)
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                sub = ""
                for sent in sentences:
                    if len(sub) + len(sent) + 1 <= chunk_size:
                        sub = f"{sub} {sent}" if sub else sent
                    else:
                        if sub and len(sub) >= 20:
                            chunks.append(sub)
                        sub = sent
                if sub and len(sub) >= 20:
                    buffer = sub
                else:
                    buffer = ""
            else:
                buffer = para

    if buffer and len(buffer) >= 20:
        chunks.append(buffer)

    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append(f"...{tail}\n\n{chunks[i]}")
        chunks = overlapped

    return chunks


# ═══════════════════════════════════════════════════════════════════════
# Practical knowledge operations
# ═══════════════════════════════════════════════════════════════════════


class FileSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    path: str = "."
    extensions: Optional[List[str]] = None
    top_k: int = 10


class DuplicateCheckRequest(BaseModel):
    content: str = Field(..., min_length=1)
    threshold: float = 0.85


class CategorizeRequest(BaseModel):
    content: str = Field(..., min_length=1)


class BulkIngestRequest(BaseModel):
    items: List[str]
    topic: str = "imported"
    source: str = "bulk"
    dedup_threshold: float = 0.85


@router.post("/search-files")
async def search_files(req: FileSearchRequest):
    """Semantic search across codebase files.

    Indexes files in the given path and returns results ranked by
    natural-language relevance.
    """
    from domains.learner.knowledge_ops import FileIndex

    idx = FileIndex()
    extensions = set(req.extensions) if req.extensions else None
    stats = idx.index_directory(req.path, extensions=extensions)
    results = idx.search(req.query, top_k=req.top_k)

    return success_response(data={
        "results": results,
        "indexed_files": stats["files_indexed"],
        "indexed_chunks": stats["chunks_total"],
    })


@router.post("/check-duplicate")
async def check_duplicate(req: DuplicateCheckRequest):
    """Check if content is a near-duplicate of existing knowledge.

    Returns whether it's a duplicate, the best match, and similarity score.
    """
    from domains.learner.knowledge_ops import DuplicateDetector

    memory = _get_memory()
    dup = DuplicateDetector(threshold=req.threshold)
    dup.load_from_store(memory._vector_store)

    is_dup, best_match, score = dup.check(req.content, embed_fn=memory._get_embedding)

    return success_response(data={
        "is_duplicate": is_dup,
        "best_match": best_match,
        "score": score,
        "threshold": req.threshold,
    })


@router.post("/categorize")
async def categorize_knowledge(req: CategorizeRequest):
    """Auto-assign a topic to content based on existing knowledge categories."""
    from domains.learner.knowledge_ops import AutoCategorizer

    memory = _get_memory()
    cat = AutoCategorizer()
    cat.load_from_store(memory._vector_store)

    topic = cat.categorize(req.content, embed_fn=memory._get_embedding)
    suggestions = cat.suggest_topics(req.content, top_k=3)

    return {
        "topic": topic,
        "suggestions": [{"topic": t, "score": round(s, 4)} for t, s in suggestions],
    }


@router.get("/gaps")
async def knowledge_gaps():
    """Find under-represented topics and knowledge gaps."""
    from domains.learner.knowledge_ops import KnowledgeGapDetector

    memory = _get_memory()
    gap = KnowledgeGapDetector()
    gap.load_from_store(memory._vector_store)

    gaps = gap.find_gaps()

    return {
        "gaps": gaps,
        "total_facts": memory._fact_counter,
        "topics": list(gap._topic_counts.keys()),
    }


@router.post("/bulk-ingest")
async def bulk_ingest(req: BulkIngestRequest):
    """Bulk ingest texts with automatic deduplication.

    Skips near-duplicate content and reports added/skipped/errors.
    """
    from domains.learner.knowledge_ops import BulkProcessor

    memory = _get_memory()
    bp = BulkProcessor(memory)
    report = bp.ingest_texts(
        req.items,
        topic=req.topic,
        source=req.source,
        dedup_threshold=req.dedup_threshold,
    )

    return {
        "status": "completed",
        **report,
    }


@router.post("/train-embedder")
async def train_embedder_endpoint():
    """Train the SloNet text embedder on all knowledge + dataset texts.

    Collects texts from the knowledge base, ingested files, and datasets,
    then trains a contrastive embedder. Returns training stats.
    """
    import asyncio

    def _train():
        from pathlib import Path
        from domains.inference.slo_embedder import train_embedder

        REPO = Path(__file__).resolve().parents[4]
        texts = []

        # 1. Knowledge entries
        kb_path = REPO / "data" / "knowledge" / "entries.json"
        if kb_path.exists():
            with open(kb_path) as f:
                entries = json.load(f)
            for e in entries:
                t = e.get("text", "")
                if len(t) > 20:
                    texts.append(t)

        # 2. Ingested files
        ingested_dir = REPO / "data" / "ingested"
        if ingested_dir.exists():
            for fp in ingested_dir.glob("*.txt"):
                try:
                    raw = fp.read_text(errors="ignore")
                    chunks = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 40]
                    texts.extend(chunks[:500])
                except Exception:
                    pass

        # 3. Dataset files
        datasets_dir = REPO / "datasets"
        if datasets_dir.exists():
            for fp in datasets_dir.glob("*.txt"):
                try:
                    raw = fp.read_text(errors="ignore")
                    chunks = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 40]
                    texts.extend(chunks[:500])
                except Exception:
                    pass

        # Deduplicate
        seen = set()
        unique = [t for t in texts if hash(t[:200]) not in seen and not seen.add(hash(t[:200]))]
        texts = unique[:500]

        if len(texts) < 10:
            return {"status": "error", "message": f"Only {len(texts)} texts found. Need at least 10."}

        result = train_embedder(
            texts, epochs=15, lr=5e-4, batch_size=32,
            embed_dim=64, vocab_size=1024, max_seq_len=32,
            n_heads=2, n_layers=1,
        )
        return {
            "status": "trained",
            "texts_used": len(texts),
            "epochs": result["epochs"],
            "final_loss": result["final_loss"],
            "save_path": result["save_path"],
        }

    return await asyncio.to_thread(_train)


@router.get("/embedder-status")
async def embedder_status():
    """Check if a trained embedder checkpoint exists."""
    from domains.inference.slo_embedder import _EMBEDDER_PATH, SloTextEmbedder

    exists = _EMBEDDER_PATH.exists()
    info = None
    if exists:
        emb = SloTextEmbedder.load()
        if emb:
            info = {
                "embed_dim": emb.embed_dim,
                "vocab_size": len(emb.vocab),
                "path": str(_EMBEDDER_PATH),
            }

    return {
        "trained": exists,
        "info": info,
    }


# ── Spaced repetition for knowledge review ────────────────────────────

_spaced_rep_scheduler = None

def _get_spaced_rep():
    global _spaced_rep_scheduler
    if _spaced_rep_scheduler is None:
        from domains.infrastructure.spaced_repetition_engine import SpacedRepetitionScheduler
        _spaced_rep_scheduler = SpacedRepetitionScheduler()
    return _spaced_rep_scheduler


@router.get("/reviews/due")
def get_due_reviews():
    """Get knowledge items that are due for review."""
    scheduler = _get_spaced_rep()
    due_ids = scheduler.get_due_reviews()
    stats = scheduler.get_review_stats()
    return success_response(data={"due_ids": due_ids, "stats": stats})


@router.post("/reviews/{item_id}/schedule")
def schedule_review(item_id: str, performance: float = Query(0.8, ge=0.0, le=1.0)):
    """Record a review performance and schedule next review."""
    scheduler = _get_spaced_rep()
    next_time = scheduler.schedule_review(item_id, performance)
    return success_response(data={"item_id": item_id, "next_review": next_time})


# ── Knowledge labeler ─────────────────────────────────────────────────

@router.get("/label")
def label_text(text: str = Query(..., min_length=1)):
    """Classify text into semantic category (factual, procedural, etc.)."""
    from domains.infrastructure.truth_labeler import get_truth_labeler
    labeler = get_truth_labeler()
    result = labeler.label(text)
    return success_response(data=result.to_dict())
