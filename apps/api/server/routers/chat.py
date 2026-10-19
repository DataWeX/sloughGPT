"""
Chat Router - the prompting surface: respond, stream, regenerate, cancel.

Owns POST /chat and POST /chat/stream (single registration owner — the parity
gate fails on any duplicate path+method). Those two delegate to the inference
kernel through its public facades (`handle_chat` / `handle_chat_stream`) so
model-readiness gates, enrichment phases, reconnect replay and the SSE
framing stay identical for every caller (card 13c37a23, porting the settled
fix 86c6cff8c from feat/chat-owns-prompting).
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
from schemas.common import classify_and_raise, endpoint, safe_audit_log, success_response

from config import ServerConfig
from routers.inference import ChatRequest, ChatResponse, handle_chat, handle_chat_stream

logger = logging.getLogger(__name__)

cfg = ServerConfig.from_env()


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

    async def _generate(
        self,
        request: ChatRequest,
        *,
        stream: bool,
        http_request: Request | None = None,
        auth_user: dict | None = None,
    ) -> ChatResponse | StreamingResponse:
        """Single chat-generation seam.

        POST /chat and POST /chat/stream are two projections of one generation
        path: both funnels through here, and `stream` is a mode rather than a
        second, parallel handler. Prompting surface lives in this router;
        readiness/circuit-breaker gates and the response shape are whatever the
        chosen backend owns, and they are identical for every caller precisely
        because there is only one place that decides.

        Keeping this seam single is what makes the generation backend swappable
        without touching either endpoint's contract.

        Generation and state are split deliberately: the inference kernel owns
        generation (readiness/circuit-breaker gates, KV cache), while chat owns
        its own state — the cancellation registry and conversation history —
        through the manager's public entry rather than its internals. The two
        domains hold orthogonal state, so they compose.
        """
        if stream:
            # Streaming still delegates wholesale to the inference kernel's SSE
            # facade. That returns a fully-formed StreamingResponse, so chat
            # never sees the raw tokens and cannot persist history without
            # parsing SSE frames — brittle. Doing it properly needs an inference
            # facade yielding raw tokens, the shape ChatManager.stream() already
            # uses. Carded separately.
            return await handle_chat_stream(request, http_request, auth_user)

        from domain.chat import get_chat_manager

        manager = get_chat_manager()
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        async def _gen() -> ChatResponse:
            return await handle_chat(request)

        return await manager.generate_tracked(
            request.session_id or "default",
            messages,
            _gen,
            # inference's ChatResponse carries `message`; domain/chat's
            # identically-named type carries `text`. Project down explicitly
            # rather than relying on the shared name.
            text_of=lambda resp: resp.message,
        )

    @endpoint("chat.respond")
    async def respond(
        self,
        request: ChatRequest,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> ChatResponse:
        """Non-streaming chat response."""
        return await self._generate(request, stream=False, auth_user=auth_user)

    @endpoint("chat.stream")
    async def stream(
        self,
        request: ChatRequest,
        http_request: Request,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> StreamingResponse:
        """Streaming chat response (SSE) — the same generation path, streamed."""
        return await self._generate(
            request, stream=True, http_request=http_request, auth_user=auth_user
        )

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
                    from domain.infrastructure.session_core import SessionCore

                    msgs = SessionCore.get_messages(session_id)
                    if not msgs:
                        yield self._sse_error(
                            "chat",
                            "REGENERATE",
                            "No session context found",
                            code="E_VAL_REQUEST",
                            http_status=400,
                        )
                        return

                    from domain.models import get_provider

                    provider = get_provider("default")
                    if provider is None:
                        yield self._sse_error(
                            "chat",
                            "REGENERATE",
                            "Model not loaded",
                            code="E_INFRA_REGISTRY",
                            http_status=503,
                        )
                        return

                    yield self._sse_event(
                        "chat", "REGENERATE", "thinking", data={}, message="Regenerating..."
                    )

                    async for token in manager.stream(
                        messages=msgs,
                        session_id=session_id,
                    ):
                        if await request.is_disconnected():
                            return
                        if token:
                            _token_count += 1
                            yield self._sse_token("chat", token)
                    yield self._sse_token(
                        "chat", "", done=True, meta={"usage_tokens": manager.last_usage()}
                    )
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
