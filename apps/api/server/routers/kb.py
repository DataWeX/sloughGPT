"""
Knowledge Router - Knowledge base management backed by KnowledgeMemory.

All operations use the vector-store-backed KnowledgeMemory (the same store
used by entity_extractor, soul engine prompt injection, and chat enrichment).
"""
import json
import logging
import re
import asyncio
from fastapi import APIRouter, Depends, Query, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Optional, List
import time

logger = logging.getLogger(__name__)

from infrastructure.auth import require_auth_if_enabled
from schemas.common import success_response, raise_error, classify_and_raise, safe_audit_log
from domains.infrastructure.errors import AppError
from infrastructure.ssrf import is_private_ip as _is_private_ip

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


class RAGIngestRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=100000)
    source: str = Field(default="user", max_length=200)
    topic: str = Field(default="general", max_length=100)
    chunk_size: int = Field(default=512, ge=64, le=4096)


class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class RAGVerifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    question: str = Field(..., min_length=1, max_length=2000)


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

        # Production RAG endpoints
        self.router.add_api_route("/rag/ingest", self.rag_ingest, methods=["POST"])
        self.router.add_api_route("/rag/query", self.rag_query, methods=["POST"])
        self.router.add_api_route("/rag/verify", self.rag_verify, methods=["POST"])
        self.router.add_api_route("/rag/documents", self.rag_list_documents, methods=["GET"])
        self.router.add_api_route("/rag/clear", self.rag_clear, methods=["POST"])
        self.router.add_api_route("/rag/stats", self.rag_stats, methods=["GET"])
        self.router.add_api_route("/kg/sync", self.kg_sync_to_rag, methods=["POST"])
        self.router.add_api_route("/kg/pipeline-stats", self.kg_pipeline_stats, methods=["GET"])

    def _get_memory(self):
        from domains.learner.knowledge import get_knowledge_memory
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

    def list_knowledge(self, limit: int = Query(200, ge=1, le=5000), offset: int = Query(0, ge=0)) -> dict:
        """List knowledge items with optional pagination."""
        try:
            memory = self._get_memory()
            entries = memory.list_all(top_k=limit + offset)
            entries = entries[offset:offset + limit]
            return [self._fact_from_entry(e) for e in entries]

        except Exception as e:
            classify_and_raise(e, source="kb.list_knowledge")
    def add_knowledge(self, req: KnowledgeCreate, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Store a new knowledge fact in KnowledgeMemory."""
        try:
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
            except Exception as exc:
                logger.debug("Truth labeler unavailable: %s", exc)
            logger.debug("Suppressed exception in %s", __name__, exc_info=True)

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

            # Auto-ingest into production RAG for grounding verification
            if is_new:
                try:
                    from domains.cognitive.rag_service import get_rag_service
                    rag_svc = get_rag_service()
                    rag_svc.add_document(
                        content=req.content,
                        metadata={"source": req.source or "knowledge", "topic": topic or "general", "item_id": item_id},
                    )
                except Exception as e:
                    logger.warning("RAG auto-ingest failed for knowledge add (topic=%s): %s", topic, e, exc_info=True)

            safe_audit_log(
                "knowledge.add",
                resource=topic or "general",
                detail="stored" if is_new else "duplicate",
                source=req.source or "",
            )
            return success_response(data={"status": "stored" if is_new else "duplicate", "id": item_id, "content": req.content, "topic": topic, "label": label})

        except Exception as e:
            classify_and_raise(e, source="kb.add_knowledge")
    def update_knowledge(self, item_id: str, req: KnowledgeUpdate, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Update a knowledge item's content, topic, or importance."""
        try:
            memory = self._get_memory()
            all_items = memory.list_all(top_k=5000)
            target = None
            for item in all_items:
                if item.get("id") == item_id:
                    target = item
                    break
            if not target:
                raise_error("Item not found", "E_NOT_FOUND", status_code=404)

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
            safe_audit_log("knowledge.update", resource=item_id, detail="updated" if ok else "stored")
            return success_response(data={"status": "updated" if ok else "stored"})

        except Exception as e:
            classify_and_raise(e, source="kb.update_knowledge")
    def batch_ingest(self, req: KnowledgeBatchRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Store multiple knowledge items in KnowledgeMemory."""
        try:
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
            safe_audit_log("knowledge.add", resource="batch", detail=f"stored={stored}")
            return success_response(data={"stored": stored})

        except Exception as e:
            classify_and_raise(e, source="kb.batch_ingest")
    def search_knowledge(self, query: str = "") -> dict:
        """Search knowledge items by content."""
        try:
            memory = self._get_memory()
            results = memory.search(query, top_k=20) if query else []
            return success_response(data={
                "results": [self._fact_from_entry(r) for r in results],
                "count": len(results),
            })

        except Exception as e:
            classify_and_raise(e, source="kb.search_knowledge")
    def knowledge_stats(self) -> dict:
        """Return knowledge base statistics.

        Aggregates topic/source counts from the full store in a single pass.
        """
        try:
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

        except Exception as e:
            classify_and_raise(e, source="kb.knowledge_stats")
    def list_topics(self) -> dict:
        """List all unique topics with item counts."""
        try:
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

        except Exception as e:
            classify_and_raise(e, source="kb.list_topics")
    def ingest_url(self, req: UrlIngestRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Ingest a URL into the knowledge base."""
        try:
            parsed = urllib.parse.urlparse(req.url)
            if parsed.scheme.lower() in self._BLOCKED_SCHEMES:
                raise_error(f"URL scheme '{parsed.scheme}' not allowed", "E_BAD_REQUEST", status_code=400)
            if not parsed.scheme or parsed.scheme.lower() not in ("http", "https"):
                raise_error("Only HTTP/HTTPS URLs allowed", "E_BAD_REQUEST", status_code=400)
            if parsed.hostname and _is_private_ip(parsed.hostname):
                raise_error("Internal/private host URLs not allowed", "E_BAD_REQUEST", status_code=400)
            from domains.learner.knowledge import get_knowledge_ingestor
            ingestor = get_knowledge_ingestor()
            result = ingestor.ingest_url(req.url)

            # Auto-ingest extracted content into production RAG
            if result.get("new_facts", 0) > 0:
                try:
                    from domains.cognitive.rag_service import get_rag_service
                    rag_svc = get_rag_service()
                    # Re-fetch the facts we just stored to get their content
                    memory = self._get_memory()
                    recent = memory.list_all(top_k=result.get("new_facts", 5))
                    for item in recent:
                        if item.get("source", "").startswith("url:"):
                            rag_svc.add_document(
                                content=item.get("content", ""),
                                metadata={"source": req.url, "topic": item.get("topic", "web")},
                            )
                except Exception as e:
                    logger.warning("RAG auto-ingest failed for URL ingest (url=%s): %s", req.url, e, exc_info=True)

            safe_audit_log(
                "knowledge.add",
                resource=req.url,
                detail="url",
                new_facts=result.get("new_facts", 0), rejected=result.get("rejected", False),
            )
            return success_response(data={
                "status": result.get("status", "ok"),
                "new_facts": result.get("new_facts", 0),
                "title": result.get("title", ""),
                "content_length": result.get("content_length", 0),
                "rejected": result.get("rejected", False),
                "reason": result.get("reason"),
            })
        except AppError as e:
            classify_and_raise(e, source="kb.ingest_url")
        except Exception as e:
            logger.warning("KB ingest failed: %s", e)
            classify_and_raise(e, source="kb_ingest")

    def batch_delete_knowledge(self, req: BatchDeleteRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Delete multiple knowledge items by ID."""
        try:
            memory = self._get_memory()
            deleted = 0
            for item_id in req.ids:
                if memory.delete_by_id(item_id):
                    deleted += 1
            safe_audit_log("knowledge.batch.delete", resource="batch", detail=f"deleted={deleted}")
            return success_response(data={"deleted": deleted})

        except Exception as e:
            classify_and_raise(e, source="kb.batch_delete_knowledge")
    def suggest_topic(self, req: SuggestTopicRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Return the best auto-detected topic for content without storing."""
        try:
            topic = self._auto_tag(req.content)
            return success_response(data={"topic": topic, "confidence": "high" if topic != "general" else "low"})

        except Exception as e:
            classify_and_raise(e, source="kb.suggest_topic")
    def delete_knowledge(self, item_id: str, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Delete a single knowledge item by ID."""
        try:
            memory = self._get_memory()
            if memory.delete_by_id(item_id):
                safe_audit_log("knowledge.delete", resource=item_id)
                return success_response(data={"status": "deleted"})
            raise_error("Item not found", "E_NOT_FOUND", status_code=404)

        except Exception as e:
            classify_and_raise(e, source="kb.delete_knowledge")
    def train_knowledge_adapter_route(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        """Train a LoRA adapter on all knowledge facts to bake them into model weights."""
        try:
            import time as _time
            from domains.infrastructure.knowledge_weight_integrator import train_knowledge_adapter, get_adapter_status
            memory = self._get_memory()
            facts = memory.list_all(top_k=5000)

            _t0 = _time.monotonic()
            result = train_knowledge_adapter(
                knowledge_facts=facts,
                num_epochs=2,
            )
            _elapsed_ms = (_time.monotonic() - _t0) * 1000

            status = get_adapter_status()
            safe_audit_log(
                "knowledge.train",
                resource="adapter",
                detail=f"elapsed={_elapsed_ms:.0f}ms",
                facts=len(facts), status=result.get("status", ""),
            )
            return success_response(data={**result, "adapter_status": status, "elapsed_ms": round(_elapsed_ms, 1)})

        except Exception as e:
            classify_and_raise(e, source="kb.train_knowledge_adapter_route")
    def knowledge_adapter_status(self) -> dict:
        """Return status of the knowledge weight adapter."""
        try:
            from domains.infrastructure.knowledge_weight_integrator import get_adapter_status
            return success_response(data=get_adapter_status())

        except Exception as e:
            classify_and_raise(e, source="kb.knowledge_adapter_status")
    def related_knowledge(self, item_id: str, top_k: int = Query(6, ge=1, le=20)) -> dict:
        """Return semantically related knowledge items, excluding the current one."""
        try:
            memory = self._get_memory()
            all_items = memory.list_all(top_k=5000)
            target = None
            for item in all_items:
                if item.get("id") == item_id:
                    target = item
                    break
            if not target:
                raise_error("Item not found", "E_NOT_FOUND", status_code=404)
            results = memory.search(target.get("content", ""), top_k=top_k + 1)
            related = [r for r in results if r.get("id") != item_id][:top_k]
            return success_response(data={"items": [self._fact_from_entry(r) for r in related], "count": len(related)})

        except Exception as e:
            classify_and_raise(e, source="kb.related_knowledge")
    def get_context(self) -> dict:
        """Return the full knowledge context string for injection into prompts."""
        try:
            memory = self._get_memory()
            context = memory.get_context_string(max_items=50)
            all_facts = memory.list_all()
            return success_response(data={"context": context, "count": len(all_facts)})

        except Exception as e:
            classify_and_raise(e, source="kb.get_context")
    async def ingest_file(
        self,
        file: UploadFile = File(...),
        topic: str = Form("imported"),
        chunk_size: int = Form(500),
        overlap: int = Form(50),
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Import a textbook or document file as knowledge facts.

        Supports .txt, .md, and .json (array of strings) files.
        Large files are split into overlapping chunks of ``chunk_size``
        characters with ``overlap`` character overlap between chunks.
        Each chunk is stored as a separate knowledge fact.
        """
        if chunk_size <= overlap:
            raise_error("chunk_size must exceed overlap", "E_BAD_REQUEST", status_code=400)
        if chunk_size < 100 or chunk_size > 10000:
            raise_error("chunk_size must be 100–10000", "E_BAD_REQUEST", status_code=400)

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
                raise_error(f"Invalid JSON file: {e}", "E_BAD_REQUEST", status_code=400)
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

        # Auto-ingest into production RAG
        try:
            from domains.cognitive.rag_service import get_rag_service
            rag_svc = get_rag_service()

            def _ingest_rag():
                for chunk in chunks:
                    rag_svc.add_document(
                        content=chunk,
                        metadata={"source": f"file:{file.filename or 'unknown'}", "topic": topic},
                    )

            await asyncio.to_thread(_ingest_rag)
        except Exception as e:
            logger.warning("RAG auto-ingest failed for file ingest (file=%s): %s", file.filename, e, exc_info=True)

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

    async def search_files(self, req: FileSearchRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """Semantic search across codebase files.

            Indexes files in the given path and returns results ranked by
            natural-language relevance.
            """
            from pathlib import Path as _P
            from domains.learner.knowledge_ops import FileIndex

            search_path = _P(req.path).resolve()
            _allowed_bases = [_P.home(), _P.cwd(), _P("/tmp")]
            if not any(str(search_path).startswith(str(b)) for b in _allowed_bases if b.exists()):
                raise_error(f"Path not allowed: {req.path}", "E_BAD_REQUEST", status_code=400)
            if not search_path.exists():
                raise_error(f"Path not found: {req.path}", "E_BAD_REQUEST", status_code=400)

            idx = FileIndex()
            extensions = set(req.extensions) if req.extensions else None

            def _index_and_search():
                stats = idx.index_directory(req.path, extensions=extensions)
                results = idx.search(req.query, top_k=req.top_k)
                return stats, results

            stats, results = await asyncio.to_thread(_index_and_search)

            return success_response(data={
                "results": results,
                "indexed_files": stats["files_indexed"],
                "indexed_chunks": stats["chunks_total"],
            })

        except Exception as e:
            classify_and_raise(e, source="kb.search_files")
    async def check_duplicate(self, req: DuplicateCheckRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """Check if content is a near-duplicate of existing knowledge.

            Returns whether it's a duplicate, the best match, and similarity score.
            """
            from domains.learner.knowledge_ops import DuplicateDetector

            memory = self._get_memory()
            dup = DuplicateDetector(threshold=req.threshold)

            def _check_dup():
                dup.load_from_store(memory._vector_store)
                return dup.check(req.content, embed_fn=memory._get_embedding)

            is_dup, best_match, score = await asyncio.to_thread(_check_dup)

            return success_response(data={
                "is_duplicate": is_dup,
                "best_match": best_match,
                "score": score,
                "threshold": req.threshold,
            })

        except Exception as e:
            classify_and_raise(e, source="kb.check_duplicate")
    async def categorize_knowledge(self, req: CategorizeRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """Auto-assign a topic to content based on existing knowledge categories."""
            from domains.learner.knowledge_ops import AutoCategorizer

            memory = self._get_memory()
            cat = AutoCategorizer()

            def _categorize():
                cat.load_from_store(memory._vector_store)
                topic = cat.categorize(req.content, embed_fn=memory._get_embedding)
                suggestions = cat.suggest_topics(req.content, top_k=3)
                return topic, suggestions

            topic, suggestions = await asyncio.to_thread(_categorize)

            return success_response(data={
                "topic": topic,
                "suggestions": [{"topic": t, "score": round(s, 4)} for t, s in suggestions],
            })

        except Exception as e:
            classify_and_raise(e, source="kb.categorize_knowledge")
    async def knowledge_gaps(self) -> dict:
        try:
            """Find under-represented topics and knowledge gaps."""
            from domains.learner.knowledge_ops import KnowledgeGapDetector

            memory = self._get_memory()
            gap = KnowledgeGapDetector()

            def _find_gaps():
                gap.load_from_store(memory._vector_store)
                return gap.find_gaps(), gap._topic_counts

            gaps, topic_counts = await asyncio.to_thread(_find_gaps)

            return success_response(data={
                "gaps": gaps,
                "total_facts": memory._fact_counter,
                "topics": list(topic_counts.keys()),
            })

        except Exception as e:
            classify_and_raise(e, source="kb.knowledge_gaps")
    async def bulk_ingest(self, req: BulkIngestRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
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

            safe_audit_log(
                "knowledge.add",
                resource=req.topic or "bulk",
                detail="bulk",
                added=report.get("added", 0), skipped=report.get("skipped", 0),
            )
            return success_response(data={
                "status": "completed",
                **report,
            })

        except Exception as e:
            classify_and_raise(e, source="kb.bulk_ingest")
    async def train_embedder_endpoint(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
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
                        except Exception as exc:
                            logger.debug("File read failed during embedder training: %s: %s", fp.name, exc)
                        logger.debug("Suppressed exception in %s", __name__, exc_info=True)

                # 3. Dataset files
                datasets_dir = REPO / "datasets"
                if datasets_dir.exists():
                    for fp in datasets_dir.glob("*.txt"):
                        try:
                            raw = fp.read_text(errors="ignore")
                            chunks = [p.strip() for p in raw.split("\n\n") if len(p.strip()) > 40]
                            texts.extend(chunks[:500])
                        except Exception as exc:
                            logger.debug("File read failed during embedder training: %s: %s", fp.name, exc)
                        logger.debug("Suppressed exception in %s", __name__, exc_info=True)

                # Deduplicate
                seen = set()
                unique = [t for t in texts if hash(t[:200]) not in seen and not seen.add(hash(t[:200]))]
                texts = unique[:500]

                if len(texts) < 10:
                    raise_error(f"Only {len(texts)} texts found. Need at least 10.", code="E_VAL_FIELD")

                result = train_embedder(
                    texts, epochs=15, lr=5e-4, batch_size=32,
                    embed_dim=64, vocab_size=1024, max_seq_len=32,
                    n_heads=2, n_layers=1,
                )
                return success_response(data={
                    "status": "trained",
                    "texts_used": len(texts),
                    "epochs": result["epochs"],
                    "final_loss": result["final_loss"],
                    "save_path": result["save_path"],
                })

            result = await asyncio.to_thread(_train)
            safe_audit_log("knowledge.train", resource="embedder", detail=result.get("status", "ok"), texts_used=result.get("texts_used", 0))
            return success_response(data=result)

        except Exception as e:
            classify_and_raise(e, source="kb.train_embedder_endpoint")
    async def embedder_status(self) -> dict:
        try:
            """Check if a trained embedder checkpoint exists."""
            from domains.inference.slo_embedder import _EMBEDDER_PATH, SloTextEmbedder

            exists = await asyncio.to_thread(lambda: _EMBEDDER_PATH.exists())
            info = None
            if exists:
                emb = await asyncio.to_thread(SloTextEmbedder.load)
                if emb:
                    info = {
                        "embed_dim": emb.embed_dim,
                        "vocab_size": len(emb.vocab),
                        "path": str(_EMBEDDER_PATH),
                    }

            return success_response(data={
                "trained": exists,
                "info": info,
            })

        except Exception as e:
            classify_and_raise(e, source="kb.embedder_status")
    def _get_spaced_rep(self):
        if self._spaced_rep_scheduler is None:
            from domains.infrastructure.spaced_repetition_engine import SpacedRepetitionScheduler
            self._spaced_rep_scheduler = SpacedRepetitionScheduler()
        return self._spaced_rep_scheduler

    def get_due_reviews(self) -> dict:
        try:
            """Get knowledge items that are due for review."""
            scheduler = self._get_spaced_rep()
            due_ids = scheduler.get_due_reviews()
            stats = scheduler.get_review_stats()
            return success_response(data={"due_ids": due_ids, "stats": stats})

        except Exception as e:
            classify_and_raise(e, source="kb.get_due_reviews")
    def schedule_review(self, item_id: str, performance: float = Query(0.8, ge=0.0, le=1.0), auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """Record a review performance and schedule next review."""
            scheduler = self._get_spaced_rep()
            next_time = scheduler.schedule_review(item_id, performance)
            return success_response(data={"item_id": item_id, "next_review": next_time})

        except Exception as e:
            classify_and_raise(e, source="kb.schedule_review")
    def label_text(self, text: str = Query(..., min_length=1)) -> dict:
        try:
            """Classify text into semantic category (factual, procedural, etc.)."""
            from domains.infrastructure.truth_labeler import get_truth_labeler
            labeler = get_truth_labeler()
            result = labeler.label(text)
            return success_response(data=result.to_dict())

        # ── Production RAG Endpoints ───────────────────────────────────────────

        except Exception as e:
            classify_and_raise(e, source="kb.label_text")
    def rag_ingest(self, req: RAGIngestRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """Ingest a document into the production RAG index."""
            import time as _time
            _t0 = _time.monotonic()
            from domains.cognitive.rag_service import get_rag_service
            rag_svc = get_rag_service()
            chunk_ids = rag_svc.add_document(
                content=req.content,
                metadata={"source": req.source, "topic": req.topic},
                chunk_size=req.chunk_size,
            )
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            safe_audit_log("knowledge.rag_ingest", resource=req.source or "unknown", detail=f"chunks={len(chunk_ids)} elapsed={_elapsed_ms:.0f}ms")
            return success_response(data={
                "chunk_ids": chunk_ids,
                "num_chunks": len(chunk_ids),
                "stats": rag_svc.stats(),
            })

        except Exception as e:
            classify_and_raise(e, source="kb.rag_ingest")
    def rag_query(self, req: RAGQueryRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """Query the production RAG index for relevant context."""
            import time as _time
            _t0 = _time.monotonic()
            from domains.cognitive.rag_service import get_rag_service
            rag_svc = get_rag_service()
            result = rag_svc.query(req.question, top_k=req.top_k)
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            logger.info("RAG query in %.1fms (top_k=%d)", _elapsed_ms, req.top_k)
            return success_response(data=result)

        except Exception as e:
            classify_and_raise(e, source="kb.rag_query")
    def rag_verify(self, req: RAGVerifyRequest, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """Verify generated text against the RAG index for hallucinations."""
            import time as _time
            _t0 = _time.monotonic()
            from domains.cognitive.rag_service import get_rag_service
            rag_svc = get_rag_service()
            result = rag_svc.verify_and_ground(req.text, req.question)
            _elapsed_ms = (_time.monotonic() - _t0) * 1000
            logger.info("RAG verify in %.1fms", _elapsed_ms)
            return success_response(data=result)

        except Exception as e:
            classify_and_raise(e, source="kb.rag_verify")
    async def rag_list_documents(self) -> dict:
        try:
            """List all documents in the RAG index (metadata only)."""
            from domains.cognitive.rag_service import get_rag_service, is_rag_service_ready
            if not is_rag_service_ready():
                return success_response(data={"documents": [], "stats": {}, "ready": False})
            rag_svc = get_rag_service()
            return success_response(data={
                "documents": rag_svc.list_documents(),
                "stats": rag_svc.stats(),
                "ready": True,
            })

        except Exception as e:
            classify_and_raise(e, source="kb.rag_list_documents")
    def rag_clear(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """Clear the entire RAG index and persisted documents."""
            from domains.cognitive.rag_service import get_rag_service
            import time
            rag_svc = get_rag_service()
            _t0 = time.monotonic()
            count = rag_svc.clear()
            _elapsed_ms = (time.monotonic() - _t0) * 1000
            safe_audit_log("knowledge.rag_clear", resource="rag", detail=f"cleared={count} elapsed={_elapsed_ms:.0f}ms")
            return success_response(data={"cleared": count, "elapsed_ms": round(_elapsed_ms, 1)})

        except Exception as e:
            classify_and_raise(e, source="kb.rag_clear")
    async def rag_stats(self) -> dict:
        try:
            """Retrieve statistics for the production RAG index."""
            from domains.cognitive.rag_service import get_rag_service, is_rag_service_ready
            if not is_rag_service_ready():
                return success_response(data={"ready": False, "total_chunks": 0, "total_documents": 0})
            rag_svc = get_rag_service()
            return success_response(data={**rag_svc.stats(), "ready": True})

        except Exception as e:
            classify_and_raise(e, source="kb.rag_stats")
    def kg_sync_to_rag(self, auth_user: dict = Depends(require_auth_if_enabled)) -> dict:
        try:
            """Sync all KG triples into the RAG index via the training pipeline."""
            from domains.cognitive.rag_service import KGTrainingPipeline, get_rag_service
            rag_svc = get_rag_service()
            pipeline = KGTrainingPipeline(rag_service=rag_svc)
            result = pipeline.sync_kg_to_rag()
            return success_response(data=result)

        except Exception as e:
            classify_and_raise(e, source="kb.kg_sync_to_rag")
    def kg_pipeline_stats(self) -> dict:
        try:
            """Return KG → RAG pipeline queue stats."""
            from domains.cognitive.rag_service import KGTrainingPipeline
            pipeline = KGTrainingPipeline()
            return success_response(data=pipeline.stats())


        except Exception as e:
            classify_and_raise(e, source="kb.kg_pipeline_stats")
router = KBRouter().router