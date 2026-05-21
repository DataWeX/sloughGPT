"""
Clean Domain API - Simple endpoints using domain architecture
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/api", tags=["domain"])


# ============ Chat Endpoint ============

class ChatRequest(BaseModel):
    messages: List[dict]
    model: str = "gpt2"
    system_prompt: str = ""
    temperature: float = 0.8
    max_tokens: int = 256
    session_id: Optional[str] = None
    user_id: Optional[str] = None


@router.post("/chat")
async def chat(req: ChatRequest):
    """Simple chat using domain."""
    from domains import get_chat_domain
    
    domain = get_chat_domain()
    result = await domain.respond(
        messages=req.messages,
        model=req.model,
        system_prompt=req.system_prompt,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
        session_id=req.session_id or "default",
        user_id=req.user_id or "default",
    )
    
    return {
        "text": result.text,
        "session_id": result.session_id,
        "done": result.done,
        "tokens_generated": result.tokens_generated,
    }


# ============ Companion Endpoints ============

@router.get("/companion")
async def get_companion():
    """Get companion info."""
    from domains import get_companion
    
    comp = get_companion()
    return comp.to_dict()


@router.post("/companion/{personality_id}")
async def set_personality(personality_id: str):
    """Set personality."""
    from domains import get_companion
    
    comp = get_companion()
    if hasattr(comp, 'set_personality'):
        comp.set_personality(personality_id)
    elif hasattr(comp, 'set_persona'):
        comp.set_persona(personality_id)
    
    return {"status": "ok", "personality": personality_id}


@router.get("/companion/presets")
async def list_presets():
    """List available personalities."""
    from domains import get_companion
    
    comp = get_companion()
    if hasattr(comp, 'traits'):
        return {
            "presets": [
                {"id": k, "name": v.get("name", k), "description": v.get("description", "")}
                for k, v in comp.traits.items()
            ]
        }
    return {"presets": []}


# ============ Benchmark Endpoints ============

@router.get("/benchmark/stats")
async def benchmark_stats():
    """Get benchmark stats."""
    from domains import get_benchmark_domain
    
    domain = get_benchmark_domain()
    return domain.get_stats()


@router.get("/benchmark/quality")
async def benchmark_quality(limit: int = 50):
    """Get quality metrics."""
    from domains import get_benchmark_domain
    
    domain = get_benchmark_domain()
    return domain.evaluate_latest(limit=limit)


# ============ Response Logging ============

@router.get("/responses")
async def get_responses(limit: int = 20):
    """Get recent logged responses."""
    from domains import get_chat_domain
    
    domain = get_chat_domain()
    return {"responses": domain.get_recent_responses(limit=limit)}


__all__ = ["router"]