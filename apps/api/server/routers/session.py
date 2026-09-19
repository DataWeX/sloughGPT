"""
Session Router - Conversation state management.

Handles session context, messages, and inspector.
Chat inference lives in chat.py.
"""

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, Depends
from infrastructure.auth import require_auth_if_enabled
from pydantic import BaseModel
from schemas.common import classify_and_raise, endpoint, safe_audit_log, success_response

logger = logging.getLogger(__name__)


class SessionContext(BaseModel):
    system_prompt: str | None = None
    knowledge: list[str] | None = None
    messages: list[dict[str, Any]] | None = []


class SessionRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/session", tags=["session"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route(
            "/{session_id}/context", self.set_session_context, methods=["POST"]
        )
        self.router.add_api_route(
            "/{session_id}/messages", self.get_session_messages, methods=["GET"]
        )
        self.router.add_api_route(
            "/{session_id}/inspector", self.get_session_inspector, methods=["GET"]
        )

    @staticmethod
    def get_router_session_context(session_id: str) -> list[dict[str, Any]] | None:
        """Legacy helper – forwards to ``SessionCore``.

        Existing code (e.g., ``main.py``) still calls this function. It now
        proxies to the unified ``SessionCore`` implementation.
        """
        from domain.infrastructure.session_core import SessionCore

        return SessionCore.get_messages(session_id)

    @endpoint("session.set_session_context")
    async def set_session_context(
        self,
        session_id: str,
        ctx: SessionContext,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        try:
            """Set session context (messages stored for regeneration)."""
            if ctx.messages:
                from domain.infrastructure.session_core import SessionCore

                result = SessionCore.store_context(session_id, ctx.messages)
                safe_audit_log(
                    "session.context_store",
                    resource=session_id,
                    detail=f"messages={len(ctx.messages)}",
                )
                return success_response(data=result)
            return success_response(
                data={"session_id": session_id, "message_count": 0}, message="stored"
            )

        except Exception as e:
            classify_and_raise(e, source="session.set_session_context")

    @endpoint("session.get_session_messages")
    async def get_session_messages(self, session_id: str) -> dict:
        """Return stored conversation messages for a session.

        Used by the UI to load a chat history.
        """
        from domain.infrastructure.session_core import SessionCore

        try:
            msgs = SessionCore.get_messages(session_id)
            return success_response(data={"session_id": session_id, "messages": msgs})
        except Exception as e:
            logger.warning("Get session messages failed: %s", e)
            classify_and_raise(e, source="session_get_messages")

    @endpoint("session.get_session_inspector")
    async def get_session_inspector(self, session_id: str) -> dict:
        """Return aggregated context state for the UI context inspector.

        Combines session context, knowledge stats, trait weights, manager
        modes, and workspace memory into a single response. Fetches all
        sources concurrently for low latency.
        """
        _inspector_start = time.time()
        try:

            def _fetch_messages():
                from domain.infrastructure.session_core import SessionCore

                return SessionCore.get_messages(session_id)

            def _fetch_feedback():
                from domain.feedback import get_message_feedback

                fb = get_message_feedback()
                return fb.get_stats()

            def _fetch_knowledge():
                knowledge = {"total_facts": 0, "topics": []}
                try:
                    from domain.learner import get_knowledge_memory

                    km = get_knowledge_memory()
                    knowledge["total_facts"] = km.stats().get("total_facts", 0)
                    knowledge["topics"] = [t[0] for t in km.all_topics()[:10]]
                except Exception as e:
                    logger.debug("Knowledge memory unavailable in session detail: %s", e)
                return knowledge

            def _fetch_traits():
                traits, modes = {}, {}
                try:
                    from domain.context import (
                        MemoryManager,
                        PersonalityManager,
                        StyleManager,
                        TaskManager,
                        get_trait_config,
                    )

                    config = get_trait_config()
                    traits = config.all()
                    modes = {
                        "personality": PersonalityManager(config).get_mode(),
                        "memory": MemoryManager(config).get_mode(),
                        "style": StyleManager(config).get_mode(),
                        "task": TaskManager(config).get_mode(),
                    }
                except Exception as e:
                    logger.debug("Trait config unavailable in session detail: %s", e)
                return traits, modes

            def _fetch_workspace():
                workspace = {"working_memory": [], "semantic_keys": [], "episodic_count": 0}
                try:
                    from domain.infrastructure.context_core import get_context_core

                    cc = get_context_core()
                    insp = cc.get_context_inspector()
                    workspace = {
                        "working_memory": insp.get("working_memory", []),
                        "semantic_keys": insp.get("semantic_keys", []),
                        "episodic_count": insp.get("episodic_count", 0),
                        "sensory_buffer_size": insp.get("sensory_buffer_size", 0),
                        "system_prompt": insp.get("system_prompt", "")[:300],
                    }
                except Exception as e:
                    logger.debug("Context core unavailable in session detail: %s", e)
                return workspace

            msgs, fb_stats, knowledge, (traits, modes), workspace = await asyncio.gather(
                asyncio.to_thread(_fetch_messages),
                asyncio.to_thread(_fetch_feedback),
                asyncio.to_thread(_fetch_knowledge),
                asyncio.to_thread(_fetch_traits),
                asyncio.to_thread(_fetch_workspace),
            )
            _elapsed_ms = round((time.time() - _inspector_start) * 1000)

            return success_response(
                data={
                    "session": {
                        "id": session_id,
                        "message_count": len(msgs),
                        "messages": msgs[-10:],
                    },
                    "knowledge": knowledge,
                    "traits": traits,
                    "modes": modes,
                    "feedback": {
                        "total": fb_stats.get("feedback_total", 0),
                        "thumbs_up": fb_stats.get("thumbs_up", 0),
                        "thumbs_down": fb_stats.get("thumbs_down", 0),
                    },
                    "workspace": workspace,
                    "elapsed_ms": _elapsed_ms,
                }
            )

        except Exception as e:
            logger.warning("Session inspector failed: %s", e)
            classify_and_raise(e, source="session_inspector")


router = SessionRouter().router
