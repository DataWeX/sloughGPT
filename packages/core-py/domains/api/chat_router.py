"""
Chat API - Clean domain-based endpoints

Simple, focused: use domains directly
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]
    model: str = "gpt2"
    system_prompt: str = ""
    temperature: float = 0.8
    max_tokens: int = 256
    session_id: Optional[str] = None
    user_id: Optional[str] = None


class ChatResponse(BaseModel):
    text: str
    session_id: str
    done: bool = True
    tokens_generated: int = 0


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Chat with the model."""
    try:
        # Use chat domain
        from domains.chat.domain import get_chat_domain
        chat_domain = get_chat_domain()
        
        result = await chat_domain.respond(
            messages=req.messages,
            model=req.model,
            system_prompt=req.system_prompt,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            session_id=req.session_id or "default",
            user_id=req.user_id or "default",
        )
        
        return ChatResponse(
            text=result.text,
            session_id=result.session_id,
            tokens_generated=result.tokens_generated,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


__all__ = ["router"]