"""
Chat Domain - Clean domain-based chat logic

Simple, focused: receive message → generate response → log
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json
import time
from pathlib import Path


@dataclass
class ChatRequest:
    messages: List[Dict[str, str]]
    model: str = "gpt2"
    system_prompt: str = ""
    temperature: float = 0.8
    max_tokens: int = 256
    session_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass  
class ChatResponse:
    text: str
    session_id: str
    done: bool = True
    tokens_generated: int = 0
    duration_ms: int = 0


class ChatDomain:
    """
    Clean chat domain - handles chat generation and logging.
    
    Usage:
        chat = ChatDomain()
        
        # Generate response
        response = chat.respond(
            messages=[{"role": "user", "content": "Hello!"}],
            model="gpt2"
        )
        
        # Returns response and logs automatically
    """
    
    def __init__(self, log_dir: str = "data/response_logs", engine: Optional[Any] = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._engine = engine
    
    async def respond(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt2",
        system_prompt: str = "",
        temperature: float = 0.8,
        max_tokens: int = 256,
        session_id: str = "default",
        user_id: str = "default",
    ) -> ChatResponse:
        """Generate chat response."""
        start_time = time.perf_counter()
        
        # Get last user message
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break
        
        # Generate response (placeholder - uses external model)
        text = await self._generate(
            user_msg=user_msg,
            system_prompt=system_prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
        )
        
        duration_ms = int((time.perf_counter() - start_time) * 1000)
        tokens = len(text.split()) if text else 0
        
        # Log response
        self._log(
            user_message=user_msg,
            assistant_response=text or "",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            session_id=session_id,
            user_id=user_id,
            tokens_generated=tokens,
            duration_ms=duration_ms,
        )
        
        return ChatResponse(
            text=text or "[no response]",
            session_id=session_id,
            tokens_generated=tokens,
            duration_ms=duration_ms,
        )
    
    def set_engine(self, engine) -> None:
        """Set the inference engine to reuse (avoids loading a fresh model per call)."""
        self._engine = engine

    async def _generate(
        self,
        user_msg: str,
        system_prompt: str,
        model: str,
        temperature: float,
        max_tokens: int,
        messages: List[Dict[str, str]] = None,
    ) -> str:
        """Generate response using provider pipeline or core inference engine."""
        # Try the provider pipeline first (supports loaded HF models like TinyLlama)
        try:
            from domains.models.provider import get_provider
            provider = get_provider("default")
            if provider is not None:
                msgs = messages or [{"role": "user", "content": user_msg}]
                result = await provider.chat(msgs, max_tokens=max_tokens, temperature=temperature)
                return result or ""
        except Exception:
            pass

        # Fallback: use core inference engine
        engine = self._engine
        if engine is None:
            try:
                from domains.inference.engine import create_engine
                engine = create_engine()
                self._engine = engine
            except Exception as e:
                return f"[Error: {str(e)}]"
        try:
            full_prompt = self._build_prompt(system_prompt, messages or [], user_msg)
            response = engine.generate_single(
                prompt=full_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            return response
        except Exception as e:
            return f"[Error: {str(e)}]"

    @staticmethod
    def _build_prompt(system_prompt: str, messages: List[Dict[str, str]], user_msg: str) -> str:
        """Build a prompt that preserves full conversation context."""
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        # Add all prior messages (excluding the last user message to avoid duplication)
        cutoff = len(messages) - 1
        for i, m in enumerate(messages):
            if i >= cutoff:
                break
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        # Add the last user message
        parts.append(f"User: {user_msg}")
        parts.append("Assistant:")
        return "\n".join(parts)
    
    def _log(
        self,
        user_message: str,
        assistant_response: str,
        model: str,
        temperature: float,
        max_tokens: int,
        session_id: str,
        user_id: str,
        tokens_generated: int,
        duration_ms: int,
    ) -> None:
        """Log response to file."""
        import datetime
        
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_message": user_message[:500],
            "assistant_response": assistant_response[:1000],
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "session_id": session_id,
            "user_id": user_id,
            "tokens_generated": tokens_generated,
            "duration_ms": duration_ms,
        }
        
        log_file = self.log_dir / f"responses_{datetime.datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def get_recent_responses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent logged responses."""
        responses = []
        log_file = self.log_dir / f"responses_{time.strftime('%Y%m%d')}.jsonl"
        
        if not log_file.exists():
            return responses
        
        with open(log_file) as f:
            for line in f:
                try:
                    responses.append(json.loads(line))
                except:
                    continue
        
        return responses[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get response statistics."""
        responses = self.get_recent_responses(100)
        
        if not responses:
            return {"total": 0}
        
        total = len(responses)
        avg_tokens = sum(r.get("tokens_generated", 0) for r in responses) / total
        avg_duration = sum(r.get("duration_ms", 0) for r in responses) / total
        models = set(r.get("model") for r in responses)
        
        return {
            "total": total,
            "avg_tokens": round(avg_tokens, 1),
            "avg_duration_ms": round(avg_duration, 1),
            "unique_models": list(models),
        }


# Global instance  
_chat_domain: Optional[ChatDomain] = None


def get_chat_domain() -> ChatDomain:
    """Get global chat domain instance."""
    global _chat_domain
    if _chat_domain is None:
        _chat_domain = ChatDomain()
    return _chat_domain


__all__ = [
    "ChatRequest",
    "ChatResponse", 
    "ChatDomain",
    "get_chat_domain",
]