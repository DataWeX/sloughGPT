"""
Chat Router - the prompting surface: stream, regenerate, cancel.

Owns POST /chat (single registration owner — the parity gate fails on any
duplicate path+method). It reads the engine through its port
(`domain.inference.engine.get_chat_engine`) — never by importing the inference
router — so model-readiness gates, enrichment phases and the token stream stay
kernel-owned and identical for every caller (card 13c37a23). Chat reads the
engine's output (tokens + details) as data and shapes the response for the
frontend.

POST /chat STREAMS. That is not a mode and not an option: the engine's one
method yields raw tokens, so streaming is the endpoint's only state. There is
deliberately no POST /chat/stream sibling and no `stream` flag — a second URL
would re-advertise streaming as a choice, which is the exact coupling the port
removed. A caller that wants a complete reply folds the stream at *its* edge:
the web client reads until the `done: true` frame, and the mobile BFF folds
in-process through `handle_chat`.

Session state lives in session.py.
"""

import logging
import threading
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from infrastructure.auth import require_auth_if_enabled
from infrastructure.sse_fallback import sse_error, sse_token
from infrastructure.sse_fallback import sse_event as _sse_event
from schemas.chat import ChatRequest
from schemas.common import classify_and_raise, endpoint, safe_audit_log, success_response

from config import ServerConfig
from domain.inference.engine import EnginePhase, TurnResult, get_chat_engine

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
        self.router.add_api_route("", self.stream, methods=["POST"])
        self.router.add_api_route("/{session_id}/regenerate", self.regenerate, methods=["POST"])
        self.router.add_api_route("/{session_id}/cancel", self.cancel, methods=["POST"])
        self.router.add_api_route("/active", self.active_sessions, methods=["GET"])
        self.router.add_api_route("/health", self.health, methods=["GET"])
        self.router.add_api_route("/addons", self.addons, methods=["GET"])

    async def _stream_tokens(
        self,
        request: ChatRequest,
        cancel_event: threading.Event | None = None,
    ) -> AsyncIterator:
        """The single path: the engine's raw-token stream.

        Streaming is baked in — this is how chat responds, not a mode and not a
        parameter. The engine yields raw ``str`` tokens followed by a final
        ``TurnResult``. POST /chat frames them as they arrive; a caller wanting
        a complete payload folds them at its own edge. Gates and cache
        (readiness/circuit-breaker, KV) live in the
        engine's own domain — ``domain/generate``; chat owns its own state — the
        cancellation registry and conversation history — through the manager's
        public entry.

        ``cancel_event`` is the manager's stop flag for this session, delivered
        by the tracked wrapper that owns the registry; it rides the turn down to
        the provider's token loop. Chat never polls it itself — it only forwards
        the channel it was handed.
        """
        engine = get_chat_engine()
        turn_req = request.to_turn(cancel_event=cancel_event)
        async for item in engine.stream(turn_req):
            yield item

    @endpoint("chat.stream")
    async def stream(
        self,
        request: ChatRequest,
        http_request: Request,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> StreamingResponse:
        """POST /chat — chat's one entry point. Streams, because it is the only state.

        There is no `stream` flag and no /chat/stream sibling: the engine yields
        raw tokens, so this route frames them as they arrive and always has.
        A caller wanting a complete reply folds the stream at its own edge (read
        until the `done: true` frame), which is why the mobile BFF folds
        in-process through ``handle_chat`` instead of asking for a second shape.
        """
        from domain.chat import get_chat_manager

        manager = get_chat_manager()
        session_id = request.session_id or "default"
        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        async def generate():
            _start = time.time()
            _token_count = 0
            _closing: TurnResult | None = None
            _stop: threading.Event | None = None

            def _factory(stop: threading.Event):
                # Capture the registry's stop flag off the same transaction
                # that registers it: the terminal frame reports cancelled vs
                # finalized truthfully, and cancel stays chat's state.
                nonlocal _stop
                _stop = stop
                return self._stream_tokens(request, stop)

            try:
                async for item in manager.generate_stream_tracked(
                    session_id,
                    messages,
                    _factory,
                    text_of=lambda resp: resp.text,
                ):
                    if await http_request.is_disconnected():
                        return
                    if isinstance(item, EnginePhase):
                        # Engine stage boundary → lifecycle frame, phase and
                        # status straight from the envelope contract.
                        yield self._sse_event(
                            "chat",
                            item.phase,
                            item.status,
                            data=item.data,
                            message=item.message,
                        )
                    elif isinstance(item, str):
                        if item:
                            _token_count += 1
                            yield self._sse_token("chat", item, status="streaming")
                    else:
                        # Final TurnResult: detail frames now (they belong to
                        # the finalizing wrap-up); the terminal rides after
                        # the loop — the manager stores history when it
                        # resumes, and finalized must be last on the wire.
                        _closing = item
                        if item.consciousness:
                            yield self._sse_event(
                                "chat",
                                "CONSCIOUSNESS",
                                "complete",
                                data=item.consciousness,
                                message="Reflection updated",
                            )
                        if item.memory:
                            yield self._sse_event(
                                "chat",
                                "MEMORY",
                                "success",
                                data=item.memory,
                                message="New fact remembered",
                            )
                if _closing is not None:
                    _cancelled = _stop is not None and _stop.is_set()
                    yield self._sse_event(
                        "chat",
                        "CANCELLED" if _cancelled else "FINALIZED",
                        "cancelled" if _cancelled else "finalized",
                        data={},
                        meta={
                            "usage_tokens": _closing.usage,
                            "model": _closing.model,
                            "finish_reason": _closing.finish_reason,
                            "elapsed_ms": _closing.elapsed_ms,
                        },
                        message="",
                    )
                _elapsed_ms = round((time.time() - _start) * 1000)
                safe_audit_log(
                    "chat.stream",
                    resource=session_id,
                    detail=f"tokens={_token_count} elapsed={_elapsed_ms}ms",
                )
            except Exception as e:
                logger.error("Chat stream error: %s", e, exc_info=True)
                yield self._sse_error("chat", "STREAM", str(e))

        return StreamingResponse(generate(), media_type="text/event-stream")

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
                        "chat", "REGENERATE", "generating", data={}, message="Regenerating..."
                    )

                    async for token in manager.stream(
                        messages=msgs,
                        session_id=session_id,
                    ):
                        if await request.is_disconnected():
                            return
                        if token:
                            _token_count += 1
                            yield self._sse_token("chat", token, status="streaming")
                    yield self._sse_event(
                        "chat",
                        "FINALIZED",
                        "finalized",
                        data={},
                        meta={"usage_tokens": manager.last_usage()},
                        message="",
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
