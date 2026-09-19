"""
Chat Router - Inference, streaming, and generation.

Handles all model interaction: respond, stream, regenerate, cancel.
Session state lives in session.py.
"""

import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from infrastructure.auth import require_auth_if_enabled
from infrastructure.sse_fallback import sse_error, sse_token
from infrastructure.sse_fallback import sse_event as _sse_event
from pydantic import BaseModel
from schemas.common import classify_and_raise, endpoint, safe_audit_log, success_response

from config import ServerConfig

logger = logging.getLogger(__name__)

cfg = ServerConfig.from_env()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str = "gpt2"
    system_prompt: str = ""
    temperature: float = 0.8
    max_tokens: int = 256
    session_id: str = "default"


class ChatRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/chat", tags=["chat"])
        self._init_sse_helpers()
        self._register_routes()

    def _init_sse_helpers(self):
        self._sse_event = _sse_event
        self._sse_token = sse_token
        self._sse_error = sse_error

    def _register_routes(self):
        self.router.add_api_route("", self.respond, methods=["POST"])
        self.router.add_api_route("/stream", self.stream, methods=["POST"])
        self.router.add_api_route("/{session_id}/regenerate", self.regenerate, methods=["POST"])
        self.router.add_api_route("/{session_id}/cancel", self.cancel, methods=["POST"])
        self.router.add_api_route("/active", self.active_sessions, methods=["GET"])
        self.router.add_api_route("/health", self.health, methods=["GET"])
        self.router.add_api_route("/addons", self.addons, methods=["GET"])

    @endpoint("chat.respond")
    async def respond(
        self,
        request: ChatRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Non-streaming chat response."""
        try:
            from domain.chat import get_chat_manager

            manager = get_chat_manager()
            resp = await manager.respond(
                messages=[m.model_dump() for m in request.messages],
                model=request.model,
                system_prompt=request.system_prompt,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                session_id=request.session_id,
            )
            return success_response(
                data={
                    "text": resp.text,
                    "session_id": resp.session_id,
                    "tokens_generated": resp.tokens_generated,
                    "duration_ms": resp.duration_ms,
                }
            )
        except Exception as e:
            classify_and_raise(e, source="chat.respond")

    @endpoint("chat.stream")
    async def stream(
        self,
        request: ChatRequest,
        http_request: Request,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> StreamingResponse:
        """Streaming chat response (SSE)."""
        try:
            from domain.chat import get_chat_manager

            manager = get_chat_manager()

            async def generate() -> AsyncIterator[str]:
                _start = time.time()
                _token_count = 0
                _first_token_ms = None
                try:
                    async for token in manager.stream(
                        messages=[m.model_dump() for m in request.messages],
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        session_id=request.session_id,
                    ):
                        if await http_request.is_disconnected():
                            return
                        if token:
                            if _first_token_ms is None:
                                _first_token_ms = (time.time() - _start) * 1000
                            _token_count += 1
                            yield self._sse_token("chat", token)
                        else:
                            yield ": heartbeat\n\n"
                    yield self._sse_token("chat", "", done=True)
                    _elapsed_ms = round((time.time() - _start) * 1000)
                    safe_audit_log(
                        "chat.stream",
                        resource=request.session_id,
                        detail=f"tokens={_token_count} elapsed={_elapsed_ms}ms",
                    )
                except Exception as e:
                    logger.error("Stream error: %s", e, exc_info=True)
                    yield self._sse_error("chat", "STREAM", str(e))

            return StreamingResponse(generate(), media_type="text/event-stream")
        except Exception as e:
            classify_and_raise(e, source="chat.stream")

    @endpoint("chat.regenerate")
    async def regenerate(
        self,
        session_id: str,
        request: Request,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> StreamingResponse:
        """Regenerate last assistant response for a session."""
        try:
            from domain.chat import get_chat_manager

            manager = get_chat_manager()

            async def generate() -> AsyncIterator[str]:
                _start = time.time()
                _token_count = 0
                try:
                    async for token in manager.stream(
                        messages=[],  # uses stored history
                        session_id=session_id,
                    ):
                        if await request.is_disconnected():
                            return
                        if token:
                            _token_count += 1
                            yield self._sse_token("chat", token)
                    yield self._sse_token("chat", "", done=True)
                    _elapsed_ms = round((time.time() - _start) * 1000)
                    safe_audit_log(
                        "chat.regenerate",
                        resource=session_id,
                        detail=f"tokens={_token_count} elapsed={_elapsed_ms}ms",
                    )
                except Exception as e:
                    logger.error("Regenerate error: %s", e, exc_info=True)
                    yield self._sse_error("chat", "REGENERATE", str(e))

            return StreamingResponse(generate(), media_type="text/event-stream")
        except Exception as e:
            classify_and_raise(e, source="chat.regenerate")

    @endpoint("chat.cancel")
    async def cancel(
        self,
        session_id: str,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """Cancel in-flight operations for a session."""
        try:
            from domain.chat import get_chat_manager

            manager = get_chat_manager()
            cancelled = manager.cancel_session(session_id)
            return success_response(data={"cancelled": cancelled})
        except Exception as e:
            classify_and_raise(e, source="chat.cancel")

    @endpoint("chat.active_sessions")
    async def active_sessions(
        self,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """List sessions with in-flight operations."""
        try:
            from domain.chat import get_chat_manager

            manager = get_chat_manager()
            sessions = manager.active_sessions()
            return success_response(data={"sessions": sessions})
        except Exception as e:
            classify_and_raise(e, source="chat.active_sessions")

    @endpoint("chat.health")
    async def health(self) -> dict:
        """Check chat provider availability."""
        try:
            from domain.chat import get_chat_manager

            manager = get_chat_manager()
            info = manager.health()
            return success_response(data=info)
        except Exception as e:
            classify_and_raise(e, source="chat.health")

    @endpoint("chat.addons")
    async def addons(self) -> dict:
        """List chat processors and capabilities."""
        try:
            from domain.chat import get_chat_manager

            manager = get_chat_manager()
            info = manager.addons()
            return success_response(data=info)
        except Exception as e:
            classify_and_raise(e, source="chat.addons")


router = ChatRouter().router
