"""Chat-surface wire contracts.

These are the request/response shapes for the conversation surface (text +
audio). They live in a neutral home so **neither router imports the other**:
`routers/chat.py` (the response/frontend handler) and `routers/inference.py`
(the inference engine) both project these same contracts.

Historically these lived in ``routers/inference.py``, which forced the chat
router to import a sibling router (``from routers.inference import ...``) — a
layering violation. They now live here; ``routers/inference.py`` re-exports
them for backwards compatibility with existing importers (mobile BFF, tests).
"""

from typing import Any

from pydantic import BaseModel, Field

from domain.inference.engine import TurnRequest


class Message(BaseModel):
    role: str = Field(..., pattern=r"^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=100000)


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str = "qwen2.5-0.5b-instruct"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=128, ge=1, le=2048)
    max_new_tokens: int | None = Field(
        default=None, ge=1, le=2048, description="Alias for max_tokens (frontend compat)"
    )
    top_p: float = Field(default=0.85, ge=0.0, le=1.0)
    top_k: int = Field(default=40, ge=0, le=500)
    repetition_penalty: float = Field(default=1.15, ge=0.5, le=2.0)
    session_id: str | None = None
    user_id: str | None = None
    system_prompt: str | None = None
    knowledge: list[str] | None = None
    images: list[str] | None = Field(default=None, description="Base64 encoded images")
    use_context_core: bool = Field(
        default=True, description="Use ContextCore for multi-layer context"
    )
    use_rag: bool = Field(
        default=True, description="Use production RAG for document-grounded responses"
    )
    agent_id: str | None = Field(default=None, description="Agent ID for role-play instructions")

    def model_post_init(self, __context: Any) -> None:
        """Resolve max_new_tokens alias into max_tokens."""
        if self.max_new_tokens is not None and self.max_tokens == 128:
            object.__setattr__(self, "max_tokens", self.max_new_tokens)

    def to_turn(self) -> TurnRequest:
        """Translate the wire contract into the engine's domain input.

        This is the edge translation the port depends on: the wire type stays a
        pydantic ``ChatRequest`` (HTTP/JSON coupling), the engine consumes a
        domain-native ``TurnRequest``. The dependency points *down* (app → domain),
        so neither side imports the other's layer.
        """
        return TurnRequest(
            messages=[{"role": m.role, "content": m.content} for m in self.messages],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            session_id=self.session_id or "default",
            user_id=self.user_id or "default",
            system_prompt=self.system_prompt,
            knowledge=self.knowledge,
            images=self.images,
            agent_id=self.agent_id,
            use_context_core=self.use_context_core,
            use_rag=self.use_rag,
            model_fields_set=set(self.model_fields_set),
        )


class ChatResponse(BaseModel):
    """Chat completion result.

    The detail fields (``model``, ``usage``, ``elapsed_ms``, ``finish_reason``)
    surface what the inference engine already computes but historically
    discarded — the chat handler reads them off the engine's output rather than
    reaching into the engine. They are optional and additive for wire
    compatibility.
    """

    message: str
    session_id: str
    done: bool = True
    model: str | None = None
    usage: dict | None = None
    elapsed_ms: float | None = None
    finish_reason: str | None = None


class ChatControlRequest(BaseModel):
    session_id: str
    action: str = Field(..., pattern=r"^(cancel|approve|context)$")
    tool_name: str | None = None
    approved: bool | None = None
    context: str | None = None
