"""
Context Core - Unified multi-layer context management for AI models.
Manages: session context, long-term memory, RAG retrieval, context managers.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
import hashlib
import logging
import threading

logger = logging.getLogger("slo.infrastructure.context_core")

# Lazy import to avoid heavy deps at module load
def simple_embed(text: str) -> List[float]:
    """Embed text using sentence-transformers (cached model singleton)."""
    from domains.inference.vector_store import simple_embed as vs_embed
    return vs_embed(text)


@dataclass
class ContextLayer:
    """A single layer of context."""
    layer_type: str  # "session" | "memory" | "rag" | "system"
    content: str
    tokens: int
    source: str
    timestamp: str
    priority: float = 1.0


@dataclass
class ContextFrame:
    """Complete context frame sent to model."""
    id: str
    system_prompt: str
    layers: List[ContextLayer]
    total_tokens: int
    max_tokens: int
    created_at: str

    def to_prompt(self) -> str:
        """Render as model input string."""
        parts = [self.system_prompt]
        for layer in sorted(self.layers, key=lambda x: -x.priority):
            parts.append(f"\n[{layer.layer_type.upper()}] {layer.content}")
        return "\n\n".join(parts)


class ContextCore:
    """
    Multi-layer context management system.

    Layers (bottom to top):
    1. system - system prompt, agent instructions
    2. session - current conversation messages
    3. memory - episodic/semantic memory from past conversations
    4. rag - retrieved documents from vector store
    """

    DEFAULT_SYSTEM = """You are SloughGPT, a helpful AI assistant.
You have access to conversation history and retrieved context.
Be concise, accurate, and helpful."""

    _AUTO_MEMORY_TOP_K = 5

    def __init__(
        self,
        max_tokens: int = 2048,
        memory_enabled: bool = True,
        rag_enabled: bool = True,
        personality_manager: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
        style_manager: Optional[Any] = None,
        task_manager: Optional[Any] = None,
    ):
        self.max_tokens = max_tokens

        # Session context
        self.session_messages: List[Dict[str, str]] = []
        self.session_id: Optional[str] = None

        # Working memory (Miller's law: 7 +/- 2 items)
        self.memory_enabled = memory_enabled
        self.working_capacity = 7
        self.working_memory: List[Dict] = []

        # Context frame history
        self.frame_history: List[ContextFrame] = []

        # System prompt
        self.system_prompt = self.DEFAULT_SYSTEM

        # Long-term memory (in-memory, persisted separately)
        self.episodic_memory: Dict[str, List[Dict]] = {}
        self.semantic_memory: Dict[str, Dict] = {}
        self.sensory_buffer: List[Dict] = []

        # RAG
        self.rag_enabled = rag_enabled
        self.rag_top_k = 3
        self.rag_max_chars = 500
        self._vector_store = None
        self._embedding_fn = None

        # Context managers (injected, not imported — avoid circular deps)
        self._personality = personality_manager
        self._memory = memory_manager
        self._style = style_manager
        self._task = task_manager

    def set_managers(
        self,
        personality: Optional[Any] = None,
        memory: Optional[Any] = None,
        style: Optional[Any] = None,
        task: Optional[Any] = None,
    ) -> None:
        """Inject context managers after construction."""
        if personality:
            self._personality = personality
        if memory:
            self._memory = memory
        if style:
            self._style = style
        if task:
            self._task = task

    def _apply_managers(self, query: str = "") -> Dict[str, Any]:
        """Build manager-generated context modifications.

        Returns dict with optional keys: system_extra, working_capacity, etc.
        """
        mods: Dict[str, Any] = {"system_extra": ""}

        if self._personality:
            mods["system_extra"] += self._personality.apply(self.system_prompt)

        if self._memory:
            mods["working_capacity"] = self._memory.working_capacity

        if self._style:
            mods["system_extra"] += self._style.apply(self.system_prompt)

        if self._task:
            mods["system_extra"] += self._task.apply(self.system_prompt)

        return mods

    def set_vector_store(self, store, embedding_fn=None) -> None:
        """Set vector store and embedding function for RAG."""
        self._vector_store = store
        self._embedding_fn = embedding_fn or simple_embed

    def set_rag_config(self, top_k: int = 3, max_chars: int = 500) -> None:
        """Configure RAG retrieval."""
        self.rag_top_k = top_k
        self.rag_max_chars = max_chars

    def set_system_prompt(self, prompt: str) -> None:
        """Set system prompt."""
        self.system_prompt = prompt
        self._add_sensory(f"System prompt updated: {len(prompt)} chars")

    def set_session_id(self, session_id: str) -> None:
        """Set current session."""
        self.session_id = session_id
        if session_id not in self.episodic_memory:
            self.episodic_memory[session_id] = []

    def add_message(self, role: str, content: str) -> None:
        """Add message to session."""
        self.session_messages.append({"role": role, "content": content})
        self._add_sensory(f"User message: {content[:50]}...")
        self._to_working({"role": role, "content": content})

    def add_response(self, content: str, model: str = "gpt2") -> None:
        """Add assistant response."""
        self.session_messages.append({"role": "assistant", "content": content})
        self._add_sensory(f"Response: {content[:50]}...")
        self._to_working({"role": "assistant", "content": content, "model": model})

    def _add_sensory(self, data: Any) -> None:
        """Add to sensory buffer."""
        self.sensory_buffer.append({
            "data": data,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self.sensory_buffer) > 100:
            self.sensory_buffer = self.sensory_buffer[-50:]

    def _to_working(self, item: Dict) -> None:
        """Move item to working memory."""
        cap = self._memory.working_capacity if self._memory else self.working_capacity
        if len(self.working_memory) >= cap:
            evicted = self.working_memory.pop(0)
            self._consolidate_episode(evicted)
        self.working_memory.append(item)

    def _consolidate_episode(self, item: Dict) -> None:
        """Consolidate to episodic memory."""
        if self.session_id:
            self.episodic_memory[self.session_id].append({
                "content": item,
                "timestamp": datetime.now().isoformat(),
                "importance": 1.0,
            })

    def store_fact(self, key: str, value: Any) -> None:
        """Store in semantic memory."""
        if key in self.semantic_memory:
            self.semantic_memory[key]["value"] = value
            self.semantic_memory[key]["strength"] += 0.1
        else:
            self.semantic_memory[key] = {
                "value": value,
                "strength": 1.0,
                "created": datetime.now().isoformat(),
                "accessed": datetime.now().isoformat(),
            }
        self._add_sensory(f"Stored fact: {key}")

    def recall_fact(self, key: str) -> Optional[Any]:
        """Recall from semantic memory."""
        if key in self.semantic_memory:
            self.semantic_memory[key]["accessed"] = datetime.now().isoformat()
            return self.semantic_memory[key]["value"]
        return None

    def search_semantic(self, query: str, limit: int = 5) -> List[Dict]:
        """Search semantic memory."""
        query_lower = query.lower()
        results = []
        for key, val in self.semantic_memory.items():
            if query_lower in key.lower() or query_lower in str(val.get("value", "")).lower():
                results.append({"key": key, **val})
        return sorted(results, key=lambda x: x.get("strength", 0), reverse=True)[:limit]

    def get_episodic_context(self, query: str = "", limit: int = 3) -> str:
        """Get episodic context for query."""
        if not self.memory_enabled or not self.session_id:
            return ""

        episodes = self.episodic_memory.get(self.session_id, [])
        if not episodes:
            return ""

        if query:
            query_lower = query.lower()
            scored = []
            for ep in episodes:
                content = str(ep.get("content", {}))
                score = sum(1 for word in query_lower.split() if word in content.lower())
                scored.append((score, ep))
            scored.sort(reverse=True)
            episodes = [ep for _, ep in scored[:limit]]
        else:
            episodes = episodes[-limit:]

        parts = []
        for ep in episodes:
            content = ep.get("content", {})
            if isinstance(content, dict):
                role = content.get("role", "unknown")
                text = content.get("content", "")
                parts.append(f"[{role}]: {text[:200]}")
            else:
                parts.append(str(content)[:200])

        return "\n".join(parts)

    def _auto_ingest(self) -> None:
        """Trigger repo auto-ingestion if vector store is empty."""
        import os
        import threading
        provider = os.environ.get("MAN_VECTOR_STORE", "chromadb")
        def _do():
            try:
                from domains.infrastructure.auto_ingest import AutoIngester
                import asyncio
                ingester = AutoIngester(provider=provider)
                asyncio.run(ingester.ingest())
            except Exception as e:
                logger.warning("context_core: auto-ingestion failed", extra={
                    "provider": provider, "error": str(e),
                })
        threading.Thread(target=_do, daemon=True).start()

    async def get_rag_context(self, query: str) -> str:
        """Get RAG context from vector store."""
        if not self.rag_enabled:
            return ""

        if self._vector_store is None:
            # Query KnowledgeMemory through the relevance-gated augmenter — the
            # same score floor + topical-overlap filter the chat loop applies —
            # so the RAG layer never injects spurious matches into the frame.
            import asyncio
            try:
                from domains.learner.knowledge_augmenter import enrich_with_knowledge
                def _query_aug():
                    result = enrich_with_knowledge(query, auto_search=False, max_facts=self.rag_top_k)
                    return result.get("facts", [])
                facts = await asyncio.to_thread(_query_aug)
                if facts:
                    return "\n".join(f"[Knowledge] {f}" for f in facts)
            except Exception as e:
                logger.debug("context_core: knowledge augmenter query failed", extra={
                    "error": str(e),
                })
            # Auto-ingest if empty, then fallback to semantic memory
            if not hasattr(self, '_auto_ingest_triggered'):
                self._auto_ingest_triggered = True
                self._auto_ingest()

            facts = self.search_semantic(query, limit=self.rag_top_k)
            if facts:
                parts = [f"Related: {f['key']} = {f['value']}" for f in facts]
                return "\n".join(parts)
            return ""

        try:
            # Generate embedding for query
            if self._embedding_fn:
                query_vec = self._embedding_fn(query)
            else:
                query_vec = simple_embed(query)

            results = await self._vector_store.query(query_vec, top_k=self.rag_top_k)

            if not results:
                return ""

            parts = []
            for r in results:
                text = r.text[:self.rag_max_chars] if len(r.text) > self.rag_max_chars else r.text
                parts.append(f"[Doc: {r.id}] {text}")

            return "\n".join(parts)
        except Exception as e:
            logger.warning("Vector store RAG query failed, falling back to semantic memory: %s", e)
            # Fallback to semantic memory
            facts = self.search_semantic(query, limit=self.rag_top_k)
            if facts:
                parts = [f"Related: {f['key']} = {f['value']}" for f in facts]
                return "\n".join(parts)
            return ""

    async def get_auto_memory_context(self, query: str = "", limit: int = 5) -> str:
        """
        Retrieve personal facts from the auto-memory layer.

        Surfaces facts learned from past conversations (``MemoryService``.
        ``retrieve``) as ``[Memory] <fact>`` lines. Unlike the RAG layer, the
        auto-memory read path has no topical-overlap gate, so personal
        statements ("the user prefers Zed") reach context even when they do
        not overlap the query topically.

        Args:
            query: the lookup text (typically the user message).
            limit: maximum number of facts to surface.

        Returns:
            Multi-line string of ``[Memory] ...`` facts; "" when the memory
            service is disabled, empty, or errors (fail closed).

        Side effects:
            - none; read-only.
        """
        if not query:
            return ""
        try:
            import asyncio
            from domains.memory.memory_service import get_memory_service

            def _retrieve() -> List[Dict[str, Any]]:
                return get_memory_service().retrieve(query, limit)

            facts = await asyncio.to_thread(_retrieve)
            lines = [f"[Memory] {f.get('content')}" for f in facts if f.get("content")]
            return "\n".join(lines)
        except Exception as e:
            logger.debug("Auto-memory retrieve failed: %s", e)
            return ""

    async def build_context_frame(
        self,
        include_rag: bool = True,
        include_memory: bool = True,
        query: str = "",
    ) -> ContextFrame:
        """Build a complete context frame, applying context managers."""
        # Apply managers
        manager_mods = self._apply_managers(query)
        system_prompt = self.system_prompt + manager_mods.get("system_extra", "")

        frame_id = hashlib.md5(f"{datetime.now().isoformat()}{query}".encode()).hexdigest()[:12]
        layers: List[ContextLayer] = []
        used_tokens = self._estimate_tokens(system_prompt)

        # Layer 1: Session messages
        session_content = "\n".join(
            f"[{m['role']}]: {m['content']}" for m in self.session_messages[-10:]
        )
        if session_content:
            layer = ContextLayer(
                layer_type="session",
                content=session_content,
                tokens=self._estimate_tokens(session_content),
                source="current_session",
                timestamp=datetime.now().isoformat(),
                priority=1.0,
            )
            layers.append(layer)
            used_tokens += layer.tokens

        # Layer 2: Memory
        if include_memory and self.memory_enabled:
            memory_content = self.get_episodic_context(query)
            auto_memory = await self.get_auto_memory_context(
                query, limit=self._AUTO_MEMORY_TOP_K
            )
            if auto_memory:
                memory_content = (
                    f"{memory_content}\n{auto_memory}".strip()
                    if memory_content else auto_memory
                )
            if memory_content:
                layer = ContextLayer(
                    layer_type="memory",
                    content=memory_content,
                    tokens=self._estimate_tokens(memory_content),
                    source="episodic_store+auto_memory" if auto_memory else "episodic_store",
                    timestamp=datetime.now().isoformat(),
                    priority=0.8,
                )
                if used_tokens + layer.tokens <= self.max_tokens:
                    layers.append(layer)
                    used_tokens += layer.tokens

        # Layer 3: RAG
        if include_rag and self.rag_enabled:
            rag_content = await self.get_rag_context(query)
            if rag_content:
                layer = ContextLayer(
                    layer_type="rag",
                    content=rag_content,
                    tokens=self._estimate_tokens(rag_content),
                    source="vector_store",
                    timestamp=datetime.now().isoformat(),
                    priority=0.7,
                )
                if used_tokens + layer.tokens <= self.max_tokens:
                    layers.append(layer)
                    used_tokens += layer.tokens

        frame = ContextFrame(
            id=frame_id,
            system_prompt=system_prompt,
            layers=layers,
            total_tokens=used_tokens,
            max_tokens=self.max_tokens,
            created_at=datetime.now().isoformat(),
        )

        self.frame_history.append(frame)
        if len(self.frame_history) > 50:
            self.frame_history = self.frame_history[-50:]

        return frame

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 chars)."""
        return max(1, len(text) // 4)

    def get_context_inspector(self) -> Dict[str, Any]:
        """Get full context state for UI inspection."""
        return {
            "system_prompt": self.system_prompt,
            "session_messages": self.session_messages[-10:],
            "working_memory": self.working_memory,
            "semantic_keys": list(self.semantic_memory.keys()),
            "episodic_count": sum(len(v) for v in self.episodic_memory.values()),
            "sensory_buffer_size": len(self.sensory_buffer),
            "frame_history_size": len(self.frame_history),
            "last_frame": asdict(self.frame_history[-1]) if self.frame_history else None,
        }

    def export_memory(self) -> Dict[str, Any]:
        """Export all memory for persistence."""
        return {
            "semantic": self.semantic_memory,
            "episodic": self.episodic_memory,
            "sensory": self.sensory_buffer[-100:],
        }

    def import_memory(self, data: Dict) -> None:
        """Import memory from persistence."""
        if "semantic" in data:
            self.semantic_memory = data["semantic"]
        if "episodic" in data:
            self.episodic_memory = data["episodic"]
        if "sensory" in data:
            self.sensory_buffer = data["sensory"]

    def reset_session(self) -> None:
        """Reset session but keep memory."""
        self.session_messages = []
        self.working_memory = []
        self.session_id = None

    def reset_all(self) -> None:
        """Reset everything."""
        self.session_messages = []
        self.working_memory = []
        self.episodic_memory = {}
        self.semantic_memory = {}
        self.sensory_buffer = []
        self.frame_history = []
        self.session_id = None


# Global instance
_context_core: Optional[ContextCore] = None
_context_core_lock = threading.Lock()


def get_context_core() -> ContextCore:
    global _context_core
    if _context_core is None:
        with _context_core_lock:
            if _context_core is None:
                from domains.context.managers import (
                    PersonalityManager, MemoryManager,
                    StyleManager, TaskManager,
                )
                _context_core = ContextCore(
                    personality_manager=PersonalityManager(),
                    memory_manager=MemoryManager(),
                    style_manager=StyleManager(),
                    task_manager=TaskManager(),
                )
                # Auto-select vector store from env vars
                import os
                vs_provider = os.environ.get("MAN_VECTOR_STORE", "")
                if vs_provider:
                    try:
                        from domains.inference.vector_store import create_vector_store
                        kwargs = {}
                        if vs_provider == "pinecone":
                            api_key = os.environ.get("MAN_PINECONE_API_KEY", "")
                            index_name = os.environ.get("MAN_PINECONE_INDEX", "sloughgpt")
                            if not api_key:
                                vs_provider = ""
                            else:
                                kwargs = {"api_key": api_key, "index_name": index_name}
                        elif vs_provider == "chromadb":
                            persist_dir = os.environ.get("MAN_CHROMADB_DIR", "data/vector_store")
                            kwargs = {"persist_directory": persist_dir}
                        if vs_provider:
                            import asyncio
                            loop = asyncio.new_event_loop()
                            try:
                                store = loop.run_until_complete(
                                    create_vector_store(provider=vs_provider, **kwargs)
                                )
                            finally:
                                loop.close()
                            _context_core.set_vector_store(store)
                            logger.info("Vector store auto-configured: %s", vs_provider, extra={"tag": "INFRA"})
                    except Exception as e:
                        logger.warning("Failed to auto-configure vector store %s: %s", vs_provider, e, extra={"tag": "INFRA"})
    return _context_core


def reset_context_core() -> None:
    global _context_core
    with _context_core_lock:
        _context_core = None
