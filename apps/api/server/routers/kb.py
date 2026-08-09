"""
Knowledge Router - Knowledge base management backed by KnowledgeMemory.

All operations use the vector-store-backed KnowledgeMemory (the same store
used by entity_extractor, soul engine prompt injection, and chat enrichment).
"""
import json
import re
import asyncio
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional, List
import time

from schemas.common import success_response

import urllib.parse


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
    content: str = Field(..., min_length=1, max_length=10000)
    topic: str = Field(default="general", max_length=100)
    source: str = Field(default="manual", max_length=100)
    importance: float = Field(default=0.7, ge=0.0, le=1.0)
    auto_tag: bool = False


class KnowledgeUpdate(BaseModel):
    content: Optional[str] = Field(default=None, max_length=10000)
    topic: Optional[str] = Field(default=None, max_length=100)
    importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class KnowledgeBatchItem(BaseModel):
    content: str = Field(max_length=10000)
    source: str = Field(default="injected", max_length=100)
    tags: List[str] = Field(default=[], max_length=20)


class KnowledgeBatchRequest(BaseModel):
    items: List[KnowledgeBatchItem] = Field(max_length=100)


class BatchDeleteRequest(BaseModel):
    ids: List[str] = Field(max_length=100)


class SuggestTopicRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)


class UrlIngestRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2000)
    source: str = Field(default="direct", max_length=100)


class FileSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    path: str = Field(default=".", max_length=500)
    extensions: Optional[List[str]] = Field(default=None, max_length=20)
    top_k: int = Field(default=10, ge=1, le=100)


class DuplicateCheckRequest(BaseModel):
    content: str = Field(..., min_length=1)
    threshold: float = 0.85


class CategorizeRequest(BaseModel):
    content: str = Field(..., min_length=1)


class BulkIngestRequest(BaseModel):
    items: List[str] = Field(max_length=500)
    topic: str = Field(default="imported", max_length=100)
    source: str = Field(default="bulk", max_length=100)
    dedup_threshold: float = Field(default=0.85, ge=0.0, le=1.0)


class KBRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/knowledge", tags=["knowledge"])
        self._BLOCKED_SCHEMES = {"file", "ftp", "data", "javascript", "vbscript"}
        self._ALLOWED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"}
        self._spaced_rep_scheduler = None
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("", self.list_knowledge, methods=["GET"], response_model=List[KnowledgeItemOut])
        self.router.add_api_route("", self.add_knowledge, methods=["POST"])
        self.router.add_api_route("/{item_id}", self.update_knowledge, methods=["PATCH"])
        self.router.add_api_route("/batch", self.batch_ingest, methods=["POST"])
        self.router.add_api_route("/search", self.search_knowledge, methods=["GET"])
        self.router.add_api_route("/stats", self.knowledge_stats, methods=["GET"])
        self.router.add_api_route("/topics", self.list_topics, methods=["GET"])
        self.router.add_api_route("/ingest-url", self.ingest_url, methods=["POST"])
        self.router.add_api_route("/batch-delete", self.batch_delete_knowledge, methods=["POST"])
        self.router.add_api_route("/suggest-topic", self.suggest_topic, methods=["POST"])
        self.router.add_api_route("/{item_id}", self.delete_knowledge, methods=["DELETE"])
        self.router.add_api_route("/train-adapter", self.train_knowledge_adapter_route, methods=["POST"])
        self.router.add_api_route("/adapter-status", self.knowledge_adapter_status, methods=["GET"])
        self.router.add_api_route("/{item_id}/related", self.related_knowledge, methods=["GET"])
        self.router.add_api_route("/context", self.get_context, methods=["GET"])
        self.router.add_api_route("/ingest-file", self.ingest_file, methods=["POST"])
        self.router.add_api_route("/search-files", self.search_files, methods=["POST"])
        self.router.add_api_route("/check-duplicate", self.check_duplicate, methods=["POST"])
        self.router.add_api_route("/categorize", self.categorize_knowledge, methods=["POST"])
        self.router.add_api_route("/gaps", self.knowledge_gaps, methods=["GET"])
        self.router.add_api_route("/bulk-ingest", self.bulk_ingest, methods=["POST"])
        self.router.add_api_route("/train-embedder", self.train_embedder_endpoint, methods=["POST"])
        self.router.add_api_route("/embedder-status", self.embedder_status, methods=["GET"])
        self.router.add_api_route("/reviews/due", self.get_due_reviews, methods=["GET"])
        self.router.add_api_route("/reviews/{item_id}/schedule", self.schedule_review, methods=["POST"])
        self.router.add_api_route("/label", self.label_text, methods=["GET"])

    def _get_memory(self):
        from domains.learner.knowledge import get_knowledge_memory, KnowledgeFact
        return get_knowledge_memory()

    def _fact_from_entry(self, entry: dict) -> KnowledgeItemOut:
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

    def _auto_tag(self, content: str) -> str:
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

    def list_knowledge(self, limit: int = Query(200, ge=1, le=5000), offset: int = Query(0, ge=0)):
        """List knowledge items with optional pagination."""
        memory = self._get_memory()
        entries = memory.list_all(top_k=limit + offset)
        entries = entries[offset:offset + limit]
        return [self._fact_from_entry(e) for e in entries]

    def add_knowledge(self, req: KnowledgeCreate):
        from domains.learner.knowledge import KnowledgeFact
        memory = self._get_memory()
        topic = req.topic if not req.auto_tag else self._auto_tag(req.content)

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
        is_new = memory.add_fact(fact)
        import hashlib
        content_hash = hashlib.md5(req.content.encode()).hexdigest()
        item_id = f"fact_{memory._fact_counter}_{content_hash[:8]}"
        try:
            from infrastructure.auth import get_audit_logger
            get_audit_logger().log(
                "knowledge.add",
                resource=topic or "general",
                detail="stored" if is_new else "duplicate",
                extra={"source": req.source or ""},
            )
        except Exception:
            pass
        return success_response(data={"status": "stored" if is_new else "duplicate", "id": item_id, "content": req.content, "topic": topic, "label": label})

    def update_knowledge(self, item_id: str, req: KnowledgeUpdate):
        """Update a knowledge item's content, topic, or importance."""
        memory = self._get_memory()
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
        try:
            from infrastructure.auth import get_audit_logger
            get_audit_logger().log("knowledge.update", resource=item_id, detail="updated" if ok else "stored")
        except Exception:
            pass
        return success_response(data={"status": "updated" if ok else "stored"})

    def batch_ingest(self, req: KnowledgeBatchRequest):
        """Store multiple knowledge items in KnowledgeMemory."""
        from domains.learner.knowledge import KnowledgeFact
        memory = self._get_memory()
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
        try:
            from infrastructure.auth import get_audit_logger
            get_audit_logger().log("knowledge.add", resource="batch", detail=f"stored={stored}")
        except Exception:
            pass
        return success_response(data={"stored": stored})

    def search_knowledge(self, query: str = ""):
        memory = self._get_memory()
        results = memory.search(query, top_k=20) if query else []
        return success_response(data={
            "results": [self._fact_from_entry(r) for r in results],
            "count": len(results),
        })

    def knowledge_stats(self):
        """Return knowledge base statistics.

        Aggregates topic/source counts from the full store in a single pass.
        """
        memory = self._get_memory()
        topics: dict[str, int] = {}
        sources: dict[str, int] = {}
        total = 0
        importance_sum = 0.0

        for item in memory.list_all(top_k=5000):
            t = item.get("topic", "general")
            topics[t] = topics.get(t, 0) + 1
            s = item.get("source", "unknown")
            sources[s] = sources.get(s, 0) + 1
            importance_sum += item.get("importance", 0.5)
            total += 1

        avg_importance = importance_sum / max(total, 1)
        return success_response(data={
            "total_items": total,
            "topics": topics,
            "topic_count": len(topics),
            "sources": sources,
            "avg_importance": round(avg_importance, 3),
            "searchable": True,
        })

    def list_topics(self):
        """List all unique topics with item counts."""
        memory = self._get_memory()
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

    def ingest_url(self, req: UrlIngestRequest):
        """Ingest a URL into the knowledge base."""
        try:
            parsed = urllib.parse.urlparse(req.url)
            if parsed.scheme.lower() in self._BLOCKED_SCHEMES:
                raise HTTPException(status_code=400, detail=f"URL scheme '{parsed.scheme}' not allowed")
            if parsed.hostname and parsed.hostname in self._ALLOWED_HOSTS:
                raise HTTPException(status_code=400, detail="Internal host URLs not allowed")
            if not parsed.scheme or parsed.scheme.lower() not in ("http", "https"):
                raise HTTPException(status_code=400, detail="Only HTTP/HTTPS URLs allowed")
            from domains.learner.knowledge import get_knowledge_ingestor
            ingestor = get_knowledge_ingestor()
            result = ingestor.ingest_url(req.url)
            try:
                from infrastructure.auth import get_audit_logger
                get_audit_logger().log(
                    "knowledge.add",
                    resource=req.url,
                    detail="url",
                    extra={"new_facts": result.get("new_facts", 0), "rejected": result.get("rejected", False)},
                )
            except Exception:
                pass
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
            from domains.infrastructure.errors import classify_exception, emit_error_event
            err = classify_exception(e)
            emit_error_event(err, source="kb_ingest")
            raise HTTPException(status_code=err.http_status, detail=err.user_message)

    def batch_delete_knowledge(self, req: BatchDeleteRequest):
        """Delete multiple knowledge items by ID."""
        memory = self._get_memory()
        deleted = 0
        for item_id in req.ids:
            if memory.delete_by_id(item_id):
                deleted += 1
        try:
            from infrastructure.auth import get_audit_logger
            get_audit_logger().log("knowledge.batch.delete", resource="batch", detail=f"deleted={deleted}")
        except Exception:
            pass
        return success_response(data={"deleted": deleted})

    def suggest_topic(self, req: SuggestTopicRequest):
        """Return the best auto-detected topic for content without storing."""
        topic = self._auto_tag(req.content)
        return success_response(data={"topic": topic, "confidence": "high" if topic != "general" else "low"})

    def delete_knowledge(self, item_id: str):
        memory = self._get_memory()
        if memory.delete_by_id(item_id):
            try:
                from infrastructure.auth import get_audit_logger
                get_audit_logger().log("knowledge.delete", resource=item_id)
            except Exception:
                pass
            return success_response(data={"status": "deleted"})
        raise HTTPException(status_code=404, detail="Item not found")

    def train_knowledge_adapter_route(self):
        """Train a LoRA adapter on all knowledge facts to bake them into model weights."""
        from domains.infrastructure.knowledge_weight_integrator import train_knowledge_adapter, get_adapter_status
        memory = self._get_memory()
        facts = memory.list_all(top_k=5000)

        result = train_knowledge_adapter(
            knowledge_facts=facts,
            num_epochs=2,
        )

        status = get_adapter_status()
        try:
            from infrastructure.auth import get_audit_logger
            get_audit_logger().log(
                "knowledge.train",
                resource="adapter",
                extra={"facts": len(facts), "status": result.get("status", "")},
            )
        except Exception:
            pass
        return success_response(data={**result, "adapter_status": status})

    def knowledge_adapter_status(self):
        """Return status of the knowledge weight adapter."""
        from domains.infrastructure.knowledge_weight_integrator import get_adapter_status
        return success_response(data=get_adapter_status())

    def related_knowledge(self, item_id: str, top_k: int = Query(6, ge=1, le=20)):
        """Return semantically related knowledge items, excluding the current one."""
        memory = self._get_memory()
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
        return success_response(data={"items": [self._fact_from_entry(r) for r in related], "count": len(related)})

    def get_context(self):
        memory = self._get_memory()
        context = memory.get_context_string(max_items=50)
        all_facts = memory.list_all()
        return success_response(data={"context": context, "count": len(all_facts)})

    async def ingest_file(
        self,
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
            chunks = self._chunk_text(text, chunk_size, overlap)

        from domains.learner.knowledge import KnowledgeFact
        memory = self._get_memory()

        def _store_chunks():
            facts = [
                KnowledgeFact(
                    content=chunk,
                    topic=topic,
                    source=f"file:{file.filename or 'unknown'}",
                    importance=min(1.0, len(chunk) / 2000),
                )
                for chunk in chunks
            ]
            return memory.add_facts(facts)

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

    def _lines_for_chars(self, lines: list[str], target_chars: int, start: int) -> int:
        """Count how many lines from ``start`` backward cover ``target_chars``."""
        count = 0
        chars = 0
        i = start
        while i >= 0 and chars < target_chars:
            chars += len(lines[i]) + 1
            count += 1
            i -= 1
        return count

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
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

    async def search_files(self, req: FileSearchRequest):
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

    async def check_duplicate(self, req: DuplicateCheckRequest):
        """Check if content is a near-duplicate of existing knowledge.

        Returns whether it's a duplicate, the best match, and similarity score.
        """
        from domains.learner.knowledge_ops import DuplicateDetector

        memory = self._get_memory()
        dup = DuplicateDetector(threshold=req.threshold)
        dup.load_from_store(memory._vector_store)

        is_dup, best_match, score = dup.check(req.content, embed_fn=memory._get_embedding)

        return success_response(data={
            "is_duplicate": is_dup,
            "best_match": best_match,
            "score": score,
            "threshold": req.threshold,
        })

    async def categorize_knowledge(self, req: CategorizeRequest):
        """Auto-assign a topic to content based on existing knowledge categories."""
        from domains.learner.knowledge_ops import AutoCategorizer

        memory = self._get_memory()
        cat = AutoCategorizer()
        cat.load_from_store(memory._vector_store)

        topic = cat.categorize(req.content, embed_fn=memory._get_embedding)
        suggestions = cat.suggest_topics(req.content, top_k=3)

        return {
            "topic": topic,
            "suggestions": [{"topic": t, "score": round(s, 4)} for t, s in suggestions],
        }

    async def knowledge_gaps(self):
        """Find under-represented topics and knowledge gaps."""
        from domains.learner.knowledge_ops import KnowledgeGapDetector

        memory = self._get_memory()
        gap = KnowledgeGapDetector()
        gap.load_from_store(memory._vector_store)

        gaps = gap.find_gaps()

        return {
            "gaps": gaps,
            "total_facts": memory._fact_counter,
            "topics": list(gap._topic_counts.keys()),
        }

    async def bulk_ingest(self, req: BulkIngestRequest):
        """Bulk ingest texts with automatic deduplication.

        Skips near-duplicate content and reports added/skipped/errors.
        The synchronous ingest runs in a thread pool so the event loop is not
        blocked for other requests.
        """
        from domains.learner.knowledge_ops import BulkProcessor

        memory = self._get_memory()
        bp = BulkProcessor(memory)
        report = await asyncio.to_thread(
            bp.ingest_texts,
            req.items,
            topic=req.topic,
            source=req.source,
            dedup_threshold=req.dedup_threshold,
        )

        try:
            from infrastructure.auth import get_audit_logger
            get_audit_logger().log(
                "knowledge.add",
                resource=req.topic or "bulk",
                detail="bulk",
                extra={"added": report.get("added", 0), "skipped": report.get("skipped", 0)},
            )
        except Exception:
            pass
        return {
            "status": "completed",
            **report,
        }

    async def train_embedder_endpoint(self):
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

        result = await asyncio.to_thread(_train)
        try:
            from infrastructure.auth import get_audit_logger
            get_audit_logger().log(
                "knowledge.train",
                resource="embedder",
                detail=result.get("status", "ok"),
                extra={"texts_used": result.get("texts_used", 0)},
            )
        except Exception:
            pass
        return result

    async def embedder_status(self):
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

    def _get_spaced_rep(self):
        if self._spaced_rep_scheduler is None:
            from domains.infrastructure.spaced_repetition_engine import SpacedRepetitionScheduler
            self._spaced_rep_scheduler = SpacedRepetitionScheduler()
        return self._spaced_rep_scheduler

    def get_due_reviews(self):
        """Get knowledge items that are due for review."""
        scheduler = self._get_spaced_rep()
        due_ids = scheduler.get_due_reviews()
        stats = scheduler.get_review_stats()
        return success_response(data={"due_ids": due_ids, "stats": stats})

    def schedule_review(self, item_id: str, performance: float = Query(0.8, ge=0.0, le=1.0)):
        """Record a review performance and schedule next review."""
        scheduler = self._get_spaced_rep()
        next_time = scheduler.schedule_review(item_id, performance)
        return success_response(data={"item_id": item_id, "next_review": next_time})

    def label_text(self, text: str = Query(..., min_length=1)):
        """Classify text into semantic category (factual, procedural, etc.)."""
        from domains.infrastructure.truth_labeler import get_truth_labeler
        labeler = get_truth_labeler()
        result = labeler.label(text)
        return success_response(data=result.to_dict())


router = KBRouter().router
