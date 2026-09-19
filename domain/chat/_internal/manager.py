"""ChatManager — app-layer facade over the chat subsystems.

One entry point for everything chat, so routers and the gateway call into a
single place instead of reaching across layers. Thin by design: every piece
of real work is delegated to its existing owner, this class only wires them.

Delegation map (no logic duplicated here):

- messages      → ``ChatDomain.respond`` (``domain/chat/_internal/domain.py``)
- conversations → ``SessionCore`` (``domain/infrastructure/_internal/session_core.py``)
- queues        → provider's ``SloNetServer`` / ``PriorityRequestQueue``
- operations    → ``CancelManager`` (``domain/infrastructure/_internal/cancel_manager.py``)
- model tooling → provider registry (``domain/models/_internal/provider.py``)
- settings      → ``PersistentSettings`` (``domain/settings/_internal/persistent.py``)
- availability  → provider presence + server session stats
- engine        → ``SloNetChatProvider`` / ``InferenceEngine`` behind the provider
- training      → ``SloChatTrainer`` adapter only (never a new loop here)

Speed notes: ``__slots__``, constructor injection, lazy imports of heavy
subsystems, provider resolved once and cached (re-resolved on unload),
streaming yields tokens with zero buffering, per-request work is O(messages).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator
from typing import Any

from domain.chat._internal.domain import ChatDomain, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


def _load_sessions():
    from domain.infrastructure._internal.session_core import SessionCore

    return SessionCore


def _load_cancel():
    from domain.infrastructure._internal.cancel_manager import OpType, get_cancel_manager

    return OpType, get_cancel_manager()


def _resolve_provider(name: str):
    from domain.models._internal.provider import get_provider

    return get_provider(name)


class ChatManager:
    """Facade for chat conversations, queues, operations, and availability.

    Args:
        domain: ChatDomain instance (messages + logging). Created if omitted.
        sessions: SessionCore class (static API). Resolved lazily if omitted.
        provider: Ready provider instance. When omitted, resolved from the
            registry by ``provider_name`` on first use and cached.
        provider_name: Registry key, ``"default"`` normally.
    """

    __slots__ = (
        "_domain",
        "_sessions",
        "_cancel",
        "_optype",
        "_provider",
        "_provider_name",
        "_session_ops",
        "_ops_lock",
    )

    def __init__(
        self,
        domain: ChatDomain | None = None,
        sessions: Any | None = None,
        cancel_manager: Any | None = None,
        provider: Any | None = None,
        provider_name: str = "default",
    ) -> None:
        self._domain = domain or ChatDomain()
        self._sessions = sessions
        self._provider = provider
        self._provider_name = provider_name
        self._session_ops: dict[str, set[str]] = {}
        self._ops_lock = threading.Lock()
        if cancel_manager is None:
            optype, mgr = _load_cancel()
            self._optype = optype
            self._cancel = mgr
        else:
            self._optype = None
            self._cancel = cancel_manager

    # ── internals (cold paths) ──────────────────────────────────────────

    def _session_store(self):
        if self._sessions is None:
            self._sessions = _load_sessions()
        return self._sessions

    def _active_provider(self):
        if self._provider is not None:
            return self._provider
        provider = _resolve_provider(self._provider_name)
        if provider is not None:
            self._provider = provider  # cache; cleared on unload via note below
        return provider

    def note_unloaded(self) -> None:
        """Drop the cached provider (call on model unload/switch)."""
        self._provider = None

    def _track(self, session_id: str, label: str):
        """Register a cancellable op; returns (op_id, stop_event)."""
        stop = threading.Event()
        try:
            op_id = self._cancel.register(
                op_type=self._optype.INFERENCE,
                label=label,
                cancel_fn=stop.set,
                meta={"session_id": session_id},
            )
            self._cancel.start(op_id)
        except Exception as exc:  # observability must never break inference
            logger.debug("CancelManager register skipped: %s", exc)
            return "", stop
        with self._ops_lock:
            self._session_ops.setdefault(session_id, set()).add(op_id)
        return op_id, stop

    def _untrack(self, session_id: str, op_id: str, error: str | None = None) -> None:
        try:
            if op_id:
                self._cancel.finish(op_id, error=error)
        except Exception as exc:
            logger.debug("CancelManager finish skipped: %s", exc)
        if op_id:
            with self._ops_lock:
                ops = self._session_ops.get(session_id)
                if ops:
                    ops.discard(op_id)
                    if not ops:
                        self._session_ops.pop(session_id, None)

    def _history(self, session_id: str) -> list[dict[str, str]]:
        try:
            return self._session_store().get_messages(session_id)
        except Exception as exc:
            logger.debug("Session history unavailable for %s: %s", session_id, exc)
            return []

    def _store(self, session_id: str, messages: list[dict], reply: str) -> None:
        try:
            self._session_store().store_context(
                session_id, [*messages, {"role": "assistant", "content": reply}]
            )
        except Exception as exc:
            logger.debug("Session store skipped for %s: %s", session_id, exc)

    # ── public API ──────────────────────────────────────────────────────

    async def respond(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt2",
        system_prompt: str = "",
        temperature: float = 0.8,
        max_tokens: int = 256,
        session_id: str = "default",
        user_id: str = "default",
    ) -> ChatResponse:
        """Non-streaming reply. Single path via ``ChatDomain.respond``."""
        op_id, _stop = self._track(session_id, f"chat:{session_id}")
        try:
            resp = await self._domain.respond(
                messages=messages,
                model=model,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                session_id=session_id,
                user_id=user_id,
            )
            self._store(session_id, messages, resp.text)
            return resp
        except Exception as exc:
            logger.error("ChatManager.respond failed: %s", exc, exc_info=True)
            self._untrack(session_id, op_id, error=str(exc))
            raise
        else:
            self._untrack(session_id, op_id)

    async def stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
        session_id: str = "default",
    ) -> AsyncIterator[str]:
        """Token stream with zero buffering. Honors ``cancel_session``."""
        provider = self._active_provider()
        if provider is None:
            yield "[Error: No provider available]"
            return
        op_id, stop = self._track(session_id, f"chat-stream:{session_id}")
        try:
            async for token in provider.chat_stream(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
                cancel_event=stop,
                session_id=session_id,
            ):
                yield token
        finally:
            self._untrack(session_id, op_id)

    async def regenerate(
        self,
        session_id: str,
        messages: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Re-run the last turn from stored history (or supplied messages)."""
        msgs = list(messages) if messages else self._history(session_id)
        if not msgs:
            return ChatResponse(
                text="[no history]",
                session_id=session_id,
                tokens_generated=0,
                duration_ms=0,
            )
        return await self.respond(msgs, session_id=session_id, **kwargs)

    def cancel_session(self, session_id: str) -> int:
        """Cancel in-flight ops for one session. Returns count cancelled."""
        with self._ops_lock:
            op_ids = sorted(self._session_ops.get(session_id, set()))
        cancelled = 0
        for op_id in op_ids:
            try:
                if self._cancel.cancel(op_id):
                    cancelled += 1
            except Exception as exc:
                logger.debug("Cancel failed for %s: %s", op_id, exc)
        return cancelled

    def active_sessions(self) -> list[str]:
        """Sessions with in-flight ops."""
        with self._ops_lock:
            return sorted(self._session_ops)

    def history(self, session_id: str) -> list[dict[str, str]]:
        """Stored conversation for a session (no inference)."""
        return self._history(session_id)

    def addons(self) -> dict[str, Any]:
        """Chat addons: registered message processors + provider capabilities.

        Read-only registry probe (Vision/Knowledge/ToolUse/Personality/Style
        processors live in ``domain/models/_internal/provider.py`` and run
        inside the provider — nothing is duplicated here).
        """
        try:
            from domain.models._internal.provider import list_processors

            processors = sorted(list_processors())
        except Exception as exc:
            logger.debug("Processor registry unavailable: %s", exc)
            processors = []
        capabilities: dict[str, Any] = {}
        provider = self._active_provider()
        if provider is not None:
            try:
                caps = provider.capabilities
            except Exception as exc:
                logger.debug("Provider capabilities unavailable: %s", exc)
                caps = None
            if caps is not None:
                for key in ("chat", "streaming", "embedding", "vision", "functions"):
                    capabilities[key] = bool(getattr(caps, key, False))
        return {"processors": processors, "capabilities": capabilities}

    def apply_settings(self, user_id: str | None = None, **overrides: Any) -> dict[str, Any]:
        """Cold path: merge persisted generation defaults with overrides."""
        defaults: dict[str, Any] = {}
        try:
            from domain.settings._internal.persistent import get_settings

            settings = get_settings(user_id)
            gen = getattr(settings, "generation", None)
            for key in ("model", "temperature", "max_tokens", "top_p", "system_prompt"):
                value = getattr(gen, key, None)
                if value is not None:
                    defaults[key] = value
        except Exception as exc:
            logger.debug("Settings unavailable, using overrides only: %s", exc)
        defaults.update(overrides)
        return defaults

    def health(self) -> dict[str, Any]:
        """Cheap availability probe. Never loads a model."""
        provider = self._active_provider()
        if provider is None:
            return {"available": False, "model_id": None, "active_sessions": 0}
        info: dict[str, Any] = {
            "available": True,
            "model_id": getattr(provider, "model_id", getattr(provider, "_model_id", None)),
            "active_sessions": len(self.active_sessions()),
        }
        server = getattr(provider, "get_server", lambda: None)()
        stats = getattr(server, "session_stats", None)
        if callable(stats):
            try:
                info["server_sessions"] = stats()
            except Exception as exc:
                logger.debug("Server session stats skipped: %s", exc)
        return info

    def attach_adapter(self, adapter_path: str, merge: bool = False) -> dict[str, Any]:
        """Chat model tooling: attach a LoRA adapter to the active provider.

        Delegates to ``SloNetChatProvider.apply_adapter`` when the provider
        supports it — no weight logic duplicated here. Cold path only.
        """
        provider = self._active_provider()
        if provider is None:
            return {"attached": False, "error": "No provider available"}
        apply = getattr(provider, "apply_adapter", None)
        if not callable(apply):
            return {"attached": False, "error": "Provider does not support adapters"}
        try:
            result = apply(adapter_path, merge=merge)
        except Exception as exc:
            logger.warning("Adapter attach failed: %s", exc)
            return {"attached": False, "error": str(exc)}
        if isinstance(result, dict):
            return {"attached": True, **result}
        return {"attached": True, "result": result}


# ── singleton (mirrors get_chat_domain) ──────────────────────────────────

_chat_manager: ChatManager | None = None
_manager_lock = threading.Lock()


def get_chat_manager() -> ChatManager:
    """Global ChatManager instance."""
    global _chat_manager
    if _chat_manager is None:
        with _manager_lock:
            if _chat_manager is None:
                _chat_manager = ChatManager()
    return _chat_manager


def reset_chat_manager() -> None:
    """Drop the global instance (tests / model switch)."""
    global _chat_manager
    with _manager_lock:
        _chat_manager = None


__all__ = [
    "ChatManager",
    "ChatRequest",
    "ChatResponse",
    "get_chat_manager",
    "reset_chat_manager",
]
