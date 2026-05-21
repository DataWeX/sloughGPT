"""
SloughGPT SDK Models
Data models for API requests and responses.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Iterator, Union


@dataclass
class GenerateRequest:
    """Request model for text generation."""
    prompt: str
    max_new_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    do_sample: bool = True
    repetition_penalty: Optional[float] = 1.0
    num_beams: int = 1
    early_stopping: bool = False
    personality: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        data = {"prompt": self.prompt}
        if self.max_new_tokens is not None:
            data["max_new_tokens"] = self.max_new_tokens
        if self.temperature is not None:
            data["temperature"] = self.temperature
        if self.top_k is not None:
            data["top_k"] = self.top_k
        if self.top_p is not None:
            data["top_p"] = self.top_p
        if self.do_sample is not None:
            data["do_sample"] = self.do_sample
        if self.repetition_penalty is not None:
            data["repetition_penalty"] = self.repetition_penalty
        if self.num_beams is not None:
            data["num_beams"] = self.num_beams
        if self.early_stopping:
            data["early_stopping"] = self.early_stopping
        if self.personality:
            data["personality"] = self.personality
        return data


@dataclass
class ChatMessage:
    """A single message in a chat conversation."""
    role: str
    content: str
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {"role": self.role, "content": self.content}
        if self.name:
            data["name"] = self.name
        return data
    
    @classmethod
    def user(cls, content: str) -> "ChatMessage":
        """Create a user message."""
        return cls(role="user", content=content)
    
    @classmethod
    def assistant(cls, content: str) -> "ChatMessage":
        """Create an assistant message."""
        return cls(role="assistant", content=content)
    
    @classmethod
    def system(cls, content: str) -> "ChatMessage":
        """Create a system message."""
        return cls(role="system", content=content)


@dataclass
class ChatRequest:
    """Request model for chat completions (``POST /chat`` and ``POST /chat/stream``)."""
    messages: List[ChatMessage]
    model: Optional[str] = "gpt2"
    temperature: Optional[float] = 0.8
    max_new_tokens: Optional[int] = 100
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    stream: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        data = {
            "messages": [m.to_dict() if isinstance(m, ChatMessage) else m for m in self.messages],
        }
        if self.model:
            data["model"] = self.model
        if self.temperature is not None:
            data["temperature"] = self.temperature
        if self.max_new_tokens is not None:
            data["max_new_tokens"] = self.max_new_tokens
        if self.top_p is not None:
            data["top_p"] = self.top_p
        if self.top_k is not None:
            data["top_k"] = self.top_k
        return data


@dataclass
class BatchRequest:
    """Request model for batch text generation."""
    prompts: List[str]
    max_new_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        return {
            "prompts": self.prompts,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }


@dataclass
class GenerationResult:
    """Result from text generation."""
    generated_text: str
    prompt: str
    model: Optional[str] = None
    tokens_generated: Optional[int] = None
    inference_time_ms: Optional[float] = None
    raw_response: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_response(cls, response: Dict[str, Any], prompt: str) -> "GenerationResult":
        """Create from API response."""
        text = response.get("generated_text") or response.get("text") or response.get("response", "")
        return cls(
            generated_text=text,
            prompt=prompt,
            model=response.get("model"),
            tokens_generated=response.get("tokens_generated"),
            inference_time_ms=response.get("inference_time_ms"),
            raw_response=response,
        )


@dataclass
class ChatResult:
    """Result from chat completion."""
    message: ChatMessage
    model: Optional[str] = None
    tokens_generated: Optional[int] = None
    raw_response: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_response(cls, response: Dict[str, Any]) -> "ChatResult":
        """Create from API response (SloughGPT ``POST /chat`` or OpenAI-style ``choices``)."""
        content = ""
        if response.get("choices"):
            choices = response.get("choices") or []
            choice = choices[0] if choices else {}
            msg_data = choice.get("message", {})
            content = msg_data.get("content", "") or ""
        elif "text" in response:
            content = response.get("text") or ""
        message = ChatMessage(role="assistant", content=content)
        return cls(
            message=message,
            model=response.get("model"),
            tokens_generated=response.get("tokens_generated"),
            raw_response=response,
        )


@dataclass
class BatchResult:
    """Result from batch generation."""
    results: List[GenerationResult]
    total_prompts: int
    successful: int
    failed: int
    total_time_ms: Optional[float] = None
    
    @classmethod
    def from_response(cls, response: Dict[str, Any], prompts: List[str]) -> "BatchResult":
        """Create from API response."""
        raw_results = response.get("results", response.get("responses", []))
        results = []
        for i, r in enumerate(raw_results):
            prompt = prompts[i] if i < len(prompts) else ""
            results.append(GenerationResult.from_response(r, prompt))
        
        return cls(
            results=results,
            total_prompts=len(prompts),
            successful=len(results),
            failed=len(prompts) - len(results),
            total_time_ms=response.get("total_time_ms"),
        )


@dataclass
class ModelInfo:
    """Information about an available model."""
    id: str
    name: Optional[str] = None
    source: Optional[str] = None
    description: Optional[str] = None
    size_mb: Optional[float] = None
    parameters: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelInfo":
        """Create from dictionary."""
        return cls(
            id=data.get("id", data.get("name", "")),
            name=data.get("name"),
            source=data.get("source"),
            description=data.get("description"),
            size_mb=data.get("size_mb") or data.get("size"),
            parameters=data.get("parameters"),
            tags=data.get("tags", []),
            raw=data,
        )


@dataclass
class DatasetInfo:
    """Information about an available dataset."""
    id: str
    name: Optional[str] = None
    source: Optional[str] = None
    size_mb: Optional[float] = None
    num_samples: Optional[int] = None
    description: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetInfo":
        """Create from dictionary."""
        return cls(
            id=data.get("id", data.get("name", "")),
            name=data.get("name"),
            source=data.get("source"),
            size_mb=data.get("size_mb"),
            num_samples=data.get("num_samples"),
            description=data.get("description"),
            raw=data,
        )


@dataclass
class HealthStatus:
    """Health check status."""
    status: str
    version: Optional[str] = None
    model_loaded: bool = False
    model_name: Optional[str] = None
    device: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
    
    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.status in ("ok", "healthy", "alive")
    
    @classmethod
    def from_response(cls, response: Dict[str, Any]) -> "HealthStatus":
        """Create from API response."""
        return cls(
            status=response.get("status", "unknown"),
            version=response.get("version"),
            model_loaded=response.get("model_loaded", False),
            model_name=response.get("model_name"),
            device=response.get("device"),
            raw=response,
        )


@dataclass
class SystemInfo:
    """System information."""
    version: Optional[str] = None
    pytorch_version: Optional[str] = None
    cuda_available: bool = False
    cuda: Optional[Dict[str, Any]] = None
    platform: Optional[str] = None
    python_version: Optional[str] = None
    cpu_count: Optional[int] = None
    memory_total: Optional[int] = None
    raw: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_response(cls, response: Dict[str, Any]) -> "SystemInfo":
        """Create from API response."""
        return cls(
            version=response.get("version"),
            pytorch_version=response.get("pytorch_version"),
            cuda_available=response.get("cuda_available", False),
            cuda=response.get("cuda"),
            platform=response.get("platform"),
            python_version=response.get("python_version"),
            cpu_count=response.get("cpu_count"),
            memory_total=response.get("memory_total"),
            raw=response,
        )


@dataclass
class MetricsData:
    """API metrics."""
    requests_total: int = 0
    requests_success: int = 0
    requests_failed: int = 0
    avg_response_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    active_connections: int = 0
    raw: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_response(cls, response: Dict[str, Any]) -> "MetricsData":
        """Create from API response."""
        return cls(
            requests_total=response.get("requests_total", 0),
            requests_success=response.get("requests_success", 0),
            requests_failed=response.get("requests_failed", 0),
            avg_response_time_ms=response.get("avg_response_time_ms", 0.0),
            cache_hits=response.get("cache_hits", 0),
            cache_misses=response.get("cache_misses", 0),
            active_connections=response.get("active_connections", 0),
            raw=response,
        )
