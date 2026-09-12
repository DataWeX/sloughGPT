"""Tools Router — everyday tools (writing, translate, rewrite, ...).

Exposes declarative tool metadata and an SSE generation endpoint that
renders a prompt from the core tools domain and streams tokens through
the default inference provider.
"""

import logging
import threading
from collections.abc import AsyncIterator

from domains.infrastructure.cancel_manager import OpType, get_cancel_manager
from domains.models.provider import get_provider
from domains.tools import get_tools_engine
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from infrastructure.auth import require_auth_if_enabled
from infrastructure.sse_fallback import sse_error, sse_token
from pydantic import BaseModel, Field
from schemas.common import endpoint, success_response

logger = logging.getLogger("slo.tools")

_tools_engine = get_tools_engine()


class ToolGenerateRequest(BaseModel):
    """Payload for POST /tools/{tool_id}/generate."""

    payload: dict = Field(default_factory=dict)
    max_tokens: int | None = Field(default=None, ge=16, le=4000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)


class ToolsRouter:
    def __init__(self):
        self.router = APIRouter(prefix="/tools", tags=["tools"])
        self._register_routes()

    def _register_routes(self):
        self.router.add_api_route("", self.list_tools, methods=["GET"])
        self.router.add_api_route(
            "/{tool_id}/generate",
            self.generate_tool,
            methods=["POST"],
        )

    @endpoint("tools.list")
    async def list_tools(
        self,
        request: Request,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> dict:
        """List all available tools with their UI metadata."""
        tools = _tools_engine.list_tools()
        return success_response(data={"tools": tools})

    @endpoint("tools.generate")
    async def generate_tool(
        self,
        tool_id: str,
        req: ToolGenerateRequest,
        request: Request,
        auth_user: dict = Depends(require_auth_if_enabled),
    ) -> StreamingResponse:
        """Stream a tool execution through the inference provider."""
        profile = _tools_engine.get_profile(tool_id)
        if profile is None:
            async def unknown_stream() -> AsyncIterator[str]:
                """unknown_stream."""
                yield sse_error(
                    "tools",
                    "IDLE",
                    f"Unknown tool: {tool_id}",
                    code="E_NOT_FOUND",
                    http_status=404,
                )

            return StreamingResponse(unknown_stream(), media_type="text/event-stream")

        corr_id = request.scope.get("correlation_id", "-")
        logger.info(
            "Tools generate corr=%s tool=%s payload_keys=%s",
            corr_id,
            tool_id,
            sorted(req.payload.keys()),
        )

        async def generate() -> AsyncIterator[str]:
            """generate."""
            try:
                prompt = _tools_engine.render_prompt(tool_id, req.payload)
            except Exception as e:
                logger.debug("Tool prompt render failed: %s", e)
                yield sse_error(
                    "tools",
                    "IDLE",
                    "Could not render prompt",
                    code="E_VAL_REQUEST",
                    http_status=400,
                )
                return

            provider = get_provider("default")
            if provider is None:
                yield sse_error(
                    "tools",
                    "IDLE",
                    "No provider available — load a model first",
                    code="E_INFRA_REGISTRY",
                    http_status=500,
                )
                return

            cancel_event = threading.Event()
            mgr = get_cancel_manager()
            op_id = mgr.register(
                OpType.INFERENCE,
                f"tools:{tool_id}",
                cancel_fn=lambda: cancel_event.set(),
            )
            mgr.start(op_id)

            provider_messages = [{"role": "user", "content": prompt}]
            max_tokens = req.max_tokens or profile.max_tokens
            temperature = req.temperature or 0.8

            try:
                async for token in provider.chat_stream(
                    provider_messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    cancel_event=cancel_event,
                ):
                    if cancel_event.is_set() or await request.is_disconnected():
                        mgr.finish(op_id)
                        return
                    if token:
                        yield sse_token("tools", token)
                yield sse_token("tools", "", done=True)
            except Exception as e:
                logger.warning(
                    "Tools stream interrupted tool=%s error=%s",
                    tool_id,
                    e,
                    exc_info=True,
                )
                yield sse_error(
                    "tools",
                    "STREAMING",
                    str(e) or "Generation failed",
                    code="E_INFRA_GENERATION",
                    http_status=500,
                )
            finally:
                mgr.finish(op_id)

        return StreamingResponse(generate(), media_type="text/event-stream")


router = ToolsRouter().router
