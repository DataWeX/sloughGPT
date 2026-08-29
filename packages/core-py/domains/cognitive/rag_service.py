"""
RAG Service — Production RAG with real embeddings and document persistence.

Bridges the production-grade ``ProductionRAG`` from ``cognitive/rag.py`` into
the chat pipeline by:

1. Replacing simulated random embeddings with the project's n-gram TF-IDF
   embedder (zero downloads, works on CPU).
2. Persisting ingested documents to disk so they survive server restarts.
3. Providing a singleton accessor for the chat router.
"""

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from domains.cognitive.rag import ProductionRAG
from domains.inference.vector_store import simple_embed
from domains.shared import find_repo_root

logger = logging.getLogger("slo.rag_service")

_DATA_DIR = find_repo_root(Path(__file__).resolve()) / "data" / "rag_store"
_DOCUMENTS_FILE = _DATA_DIR / "documents.jsonl"


class ProductionRAGWithRealEmbeddings(ProductionRAG):
    """ProductionRAG subclass that uses the project's real embedder."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.retriever._get_embedding = self._real_embed

    @staticmethod
    def _real_embed(text: str) -> np.ndarray:
        """Get a real 384-dim embedding via the project's n-gram TF-IDF embedder."""
        vec = simple_embed(text, dimension=384)
        return np.asarray(vec, dtype=np.float32)


class RAGService:
    """High-level RAG service with document persistence, query, and KG integration.

    Thread-safe singleton via ``get_rag_service()``. Persists documents to
    ``data/rag_store/documents.jsonl`` so they survive server restarts.

    Attributes:
        rag: Underlying ProductionRAG instance with real embeddings.
        _documents: In-memory list of persisted document dicts.
        _kg: Lazily-created KnowledgeGraph for entity/fact extraction.
        _lock: Thread lock for document list mutations.
    """

    def __init__(self) -> None:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.rag = ProductionRAGWithRealEmbeddings()
        self._documents: List[Dict[str, Any]] = []
        self._kg = None
        self._lock = threading.Lock()
        self._load_documents()

    def _load_documents(self) -> None:
        """Load persisted documents from the JSONL file into memory and RAG index."""
        if not _DOCUMENTS_FILE.exists():
            return
        try:
            with open(_DOCUMENTS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    doc = json.loads(line)
                    content = doc.get("content")
                    if not content:
                        continue
                    self._documents.append(doc)
                    self.rag.add_document(
                        content=content,
                        metadata=doc.get("metadata", {}),
                        chunk_size=doc.get("chunk_size", 512),
                        overlap=doc.get("overlap", 50),
                    )
            logger.info("Loaded %d documents from RAG store", len(self._documents))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to load RAG documents: %s", e)

    def _save_document(self, doc: Dict[str, Any]) -> None:
        """Append a single document to the JSONL persistence file."""
        try:
            with open(_DOCUMENTS_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("Failed to persist RAG document: %s", e)

    def _ensure_kg(self) -> None:
        """Lazily initialize the knowledge graph on first access."""
        if self._kg is None:
            from domains.cognitive.knowledge_graph_v2 import KnowledgeGraph
            self._kg = KnowledgeGraph()

    def _extract_kg_claims(self, content: str, metadata: Dict[str, Any]) -> None:
        """Extract entity claims from text and add them to the knowledge graph."""
        self._ensure_kg()
        try:
            detector = self.rag.hallucination_detector
            claims = detector.citation_tracker.extract_claims(content)
        except AttributeError:
            logger.debug("Hallucination detector not available for KG extraction")
            return
        source = metadata.get("source", "rag")
        for claim in claims:
            obj_text = content[claim["start"]:claim["end"]][:200]
            self._kg.add_fact(
                subject=claim["subject"],
                predicate=claim["predicate"],
                obj=obj_text,
                confidence=0.8,
                source=source,
            )

    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> List[str]:
        """Ingest a document into the RAG index, persist it, and extract KG facts.

        Args:
            content: Raw text to ingest.
            metadata: Optional metadata dict (source, topic, etc.).
            chunk_size: Max tokens per chunk.
            overlap: Overlap between adjacent chunks.

        Returns:
            List of chunk IDs created.
        """
        if not content or not content.strip():
            return []

        metadata = metadata or {"source": "user"}
        chunk_ids = self.rag.add_document(
            content=content,
            metadata=metadata,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        doc_record = {
            "content": content,
            "metadata": metadata,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "chunk_ids": chunk_ids,
            "added_at": time.time(),
        }
        with self._lock:
            self._documents.append(doc_record)
        self._save_document(doc_record)

        self._extract_kg_claims(content, metadata)

        logger.info(
            "Ingested document (%d chars → %d chunks) into RAG index",
            len(content),
            len(chunk_ids),
        )
        return chunk_ids

    def query(
        self,
        question: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Query the RAG index for relevant context.

        Args:
            question: User question to search for.
            top_k: Maximum number of results.

        Returns:
            Dict with 'context' (concatenated text), 'results' (ranked list), 'num_results'.
        """
        import time as _time
        t0 = _time.monotonic()
        try:
            result = self.rag.query(question, top_k=top_k, return_context=True)
            elapsed_ms = (_time.monotonic() - t0) * 1000
            logger.info("rag_service: query complete", extra={
                "question_len": len(question), "top_k": top_k,
                "num_results": result.get("num_results", 0),
                "elapsed_ms": round(elapsed_ms, 1),
            })
            return result
        except Exception as e:
            elapsed_ms = (_time.monotonic() - t0) * 1000
            logger.error("rag_service: query failed", extra={
                "question_len": len(question), "top_k": top_k,
                "error": str(e), "elapsed_ms": round(elapsed_ms, 1),
            })
            raise

    def verify_and_ground(
        self,
        generated_text: str,
        question: str,
    ) -> Dict[str, Any]:
        """Verify generated text against the RAG index and add citations.

        Args:
            generated_text: The model's response to verify.
            question: The original user question.

        Returns:
            Dict with 'verification', 'citations', 'confidence', 'is_verified'.
        """
        import time as _time
        t0 = _time.monotonic()
        try:
            result = self.rag.verify_and_ground(generated_text, question)
            elapsed_ms = (_time.monotonic() - t0) * 1000
            logger.info("rag_service: verify_and_ground complete", extra={
                "is_verified": result.get("is_verified", False),
                "confidence": result.get("confidence", 0),
                "num_citations": len(result.get("citations", [])),
                "elapsed_ms": round(elapsed_ms, 1),
            })
            return result
        except Exception as e:
            elapsed_ms = (_time.monotonic() - t0) * 1000
            logger.error("rag_service: verify_and_ground failed", extra={
                "error": str(e), "elapsed_ms": round(elapsed_ms, 1),
            })
            raise

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all ingested documents (metadata only, no content)."""
        with self._lock:
            return [
                {
                    "metadata": doc.get("metadata", {}),
                    "chunk_size": doc.get("chunk_size", 512),
                    "num_chunks": len(doc.get("chunk_ids", [])),
                    "added_at": doc.get("added_at", 0),
                }
                for doc in self._documents
            ]

    def clear(self) -> int:
        """Clear the entire RAG index and persisted documents."""
        with self._lock:
            count = len(self._documents)
            self._documents.clear()
        self.rag = ProductionRAGWithRealEmbeddings()
        try:
            if _DOCUMENTS_FILE.exists():
                _DOCUMENTS_FILE.unlink()
        except OSError as e:
            logger.warning("Failed to delete RAG persistence file: %s", e)
        logger.info("Cleared RAG index (%d documents removed)", count)
        return count

    def stats(self) -> Dict[str, Any]:
        """Return RAG index statistics."""
        return {
            "total_documents": len(self._documents),
            "total_chunks": len(self.rag.retriever.chunks),
            "index_size": len(self.rag.retriever.bm25.inverted_index),
        }

    def auto_ingest_directory(self, root_path: str, max_files: int = 200) -> int:
        """Scan a directory and ingest code/docs into RAG."""
        try:
            from domains.infrastructure.auto_ingest import RepoScanner
        except ImportError:
            logger.debug("RepoScanner unavailable, skipping auto-ingest")
            return 0

        scanner = RepoScanner(root_path=root_path)
        ingested = 0
        root_resolved = Path(root_path).resolve()

        for path, content in scanner.iter_files():
            if ingested >= max_files:
                break
            try:
                if len(content.strip()) < 50:
                    continue
                try:
                    rel = str(path.relative_to(root_resolved))
                except ValueError:
                    rel = str(path)
                self.add_document(
                    content=content[:8000],
                    metadata={
                        "source": "auto-ingest",
                        "file_path": rel,
                        "file_type": scanner.get_file_type(path),
                    },
                )
                ingested += 1
            except Exception as e:
                logger.debug("auto-ingest file %s failed: %s", path, e)

        if ingested > 0:
            logger.info("Auto-ingested %d files into RAG from %s", ingested, root_path)
        return ingested

    def kg_stats(self) -> Dict[str, Any]:
        """Return knowledge graph statistics."""
        if self._kg is None:
            return {"entities": 0, "facts": 0, "avg_degree": 0.0}
        return {
            "entities": len(self._kg.entities),
            "facts": len(self._kg.facts),
            "avg_degree": self._kg.stats.get("avg_degree", 0.0),
        }

    def kg_query(
        self,
        subject: str = "",
        predicate: str = "",
        obj: str = "",
    ) -> List[Dict[str, Any]]:
        """Query the knowledge graph for facts matching the given pattern."""
        if self._kg is None:
            return []
        results = []
        for fact in self._kg.facts.values():
            if subject and subject.lower() not in fact.subject.lower():
                continue
            if predicate and predicate.lower() not in fact.predicate.lower():
                continue
            if obj and obj.lower() not in fact.object.lower():
                continue
            results.append({
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object": fact.object,
                "confidence": fact.confidence,
                "source": fact.source,
            })
        return results


# ---------------------------------------------------------------------------
# KG → Training Data Pipeline (via Pugqeep TaskQueue)
# ---------------------------------------------------------------------------

class KGTrainingPipeline:
    """Pipeline that exports KG triples → embeds → stores in RAG index.

    Uses Pugqeep TaskQueue for async batch processing. Each triple becomes
    a task that gets embedded and indexed.

    Attributes:
        _rag: RAGService instance for document ingestion.
        _queue: Lazily-initialized TaskQueue (None until first use).
    """

    _REQUIRED_TRIPLE_KEYS = {"subject", "predicate", "object"}

    def __init__(self, rag_service: Optional[RAGService] = None) -> None:
        self._rag = rag_service or get_rag_service()
        self._queue = None

    def _get_queue(self):
        """Lazy-init Pugqeep TaskQueue with persistent storage.

        Returns:
            TaskQueue instance, created on first call.
        """
        if self._queue is None:
            from domains.infrastructure.pugqeep.task_queue import TaskQueue
            storage = Path("data/kg_pipeline")
            try:
                self._queue = TaskQueue(name="kg-training", storage_dir=storage)
            except TypeError:
                self._queue = TaskQueue(name="kg-training")
        return self._queue

    def _validate_triple(self, triple: Dict[str, Any]) -> bool:
        """Check a triple dict has required keys with non-empty string values.

        Args:
            triple: Dict to validate.

        Returns:
            True if valid, False otherwise.
        """
        if not isinstance(triple, dict):
            return False
        for key in self._REQUIRED_TRIPLE_KEYS:
            val = triple.get(key)
            if not isinstance(val, str) or not val.strip():
                return False
        return True

    def submit_triples(self, triples: List[Dict[str, Any]]) -> int:
        """Submit KG triples as batch tasks to the queue.

        Args:
            triples: List of dicts, each with required keys
                ``subject``, ``predicate``, ``object`` (strings), and optional
                ``confidence`` (float, default 0.8) and ``source`` (str, default "kg").

        Returns:
            Number of tasks successfully submitted.

        Raises:
            ValueError: If triples list is empty.
        """
        if not triples:
            raise ValueError("triples list must not be empty")

        from domains.infrastructure.pugqeep.task_queue import Task, TaskPriority

        queue = self._get_queue()
        submitted = 0
        for t in triples:
            if not self._validate_triple(t):
                logger.warning("KG pipeline: skipping invalid triple %r", t)
                continue
            content = f"{t['subject']} {t['predicate']} {t['object']}"
            task = Task(
                name="kg_embed_index",
                data={
                    "content": content,
                    "subject": t["subject"],
                    "predicate": t["predicate"],
                    "object": t["object"],
                    "confidence": float(t.get("confidence", 0.8)),
                    "source": str(t.get("source", "kg")),
                },
                priority=TaskPriority.NORMAL,
            )
            queue.submit(task)
            submitted += 1

        logger.info("KG pipeline: submitted %d/%d triples", submitted, len(triples))
        return submitted

    def process_batch(self, max_tasks: int = 50) -> Dict[str, Any]:
        """Process pending tasks from the queue — embed and index into RAG.

        Args:
            max_tasks: Maximum tasks to process in this batch (must be > 0).

        Returns:
            Dict with keys: processed, failed, remaining.
        """
        if max_tasks <= 0:
            return {"processed": 0, "failed": 0, "remaining": len(self._get_queue()._pending)}

        queue = self._get_queue()
        processed = 0
        failed = 0

        while processed < max_tasks:
            task = queue.next()
            if task is None:
                break

            try:
                data = task.data
                content = data["content"]
                metadata = {
                    "source": data.get("source", "kg"),
                    "subject": data["subject"],
                    "predicate": data["predicate"],
                    "object": data["object"],
                    "confidence": data.get("confidence", 0.8),
                    "kg_triple": True,
                }
                self._rag.add_document(content=content, metadata=metadata)
                queue.complete(task.id, result={"indexed": True})
                processed += 1
            except Exception as e:
                queue.fail(task.id, error=str(e))
                failed += 1
                logger.warning("KG pipeline task %s failed: %s", task.id, e)

        remaining = len(queue._pending)
        logger.info(
            "KG pipeline batch: processed=%d failed=%d remaining=%d",
            processed, failed, remaining,
        )
        return {
            "processed": processed,
            "failed": failed,
            "remaining": remaining,
        }

    def sync_kg_to_rag(self, kg=None) -> Dict[str, Any]:
        """Full pipeline: export all KG triples → submit → process → RAG index.

        Args:
            kg: KnowledgeGraph instance. If None, uses the RAG service's
                internal KG (set via ``RAGService._kg`` during document ingestion).

        Returns:
            Dict with keys: total_triples, processed, failed.
        """
        if kg is None:
            if not hasattr(self._rag, '_kg') or self._rag._kg is None:
                logger.info("KG pipeline: no knowledge graph available")
                return {"total_triples": 0, "processed": 0, "failed": 0}
            kg = self._rag._kg

        triples = kg.export_triples()
        if not triples:
            return {"total_triples": 0, "processed": 0, "failed": 0}

        self.submit_triples(triples)
        result = self.process_batch(max_tasks=len(triples))
        result["total_triples"] = len(triples)
        return result

    def stats(self) -> Dict[str, Any]:
        """Return pipeline queue stats.

        Returns:
            Dict with keys: pending, running, completed.
        """
        queue = self._get_queue()
        return {
            "pending": len(queue._pending),
            "running": len(queue._running),
            "completed": len(queue._completed),
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_rag_service: Optional[RAGService] = None
_rag_service_lock = threading.Lock()


def is_rag_service_ready() -> bool:
    """Return True if the RAG service singleton has been initialized.

    Non-blocking check — does NOT create the service if absent.
    """
    return _rag_service is not None


def get_rag_service() -> RAGService:
    """Get or create the singleton RAGService.

    Thread-safe: the first caller initializes the instance while
    subsequent callers wait on a lock instead of creating duplicates.
    """
    global _rag_service
    if _rag_service is not None:
        return _rag_service
    with _rag_service_lock:
        if _rag_service is not None:
            return _rag_service
        _rag_service = RAGService()
    return _rag_service
