"""
SloughGPT Model Registry
Centralized model management and tracking.
"""

import time
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import threading


class ModelStatus(Enum):
    """Model status."""
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    DEPRECATED = "deprecated"


class ModelTag(Enum):
    """Model tags."""
    LATEST = "latest"
    STABLE = "stable"
    EXPERIMENTAL = "experimental"
    DEPRECATED = "deprecated"
    CPU = "cpu"
    GPU = "gpu"


@dataclass
class ModelMetrics:
    """Metrics for a model."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_latency_ms: float = 0
    avg_latency_ms: float = 0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0
    memory_usage_mb: float = 0
    last_used: Optional[float] = None
    
    def record_request(self, latency_ms: float, tokens: int = 0, success: bool = True):
        """Record a request."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.total_tokens += tokens
        self.total_latency_ms += latency_ms
        self.avg_latency_ms = self.total_latency_ms / self.total_requests
        self.min_latency_ms = min(self.min_latency_ms, latency_ms)
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.last_used = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_tokens": self.total_tokens,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2) if self.min_latency_ms != float('inf') else 0,
            "max_latency_ms": round(self.max_latency_ms, 2),
            "memory_usage_mb": round(self.memory_usage_mb, 2),
            "last_used": self.last_used,
            "success_rate": round(self.successful_requests / self.total_requests * 100, 2) if self.total_requests > 0 else 0,
        }


@dataclass
class ModelInfo:
    """Information about a registered model."""
    id: str
    name: str
    version: str
    path: str
    description: str = ""
    size_mb: float = 0
    parameters: int = 0
    framework: str = "pytorch"
    status: ModelStatus = ModelStatus.LOADING
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "path": self.path,
            "description": self.description,
            "size_mb": self.size_mb,
            "parameters": self.parameters,
            "framework": self.framework,
            "status": self.status.value,
            "tags": self.tags,
            "metadata": self.metadata,
            "metrics": self.metrics.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "config": self.config,
        }


class ModelRegistry:
    """
    Centralized model registry.
    
    Example:
    
    ```python
    from sloughgpt_sdk.registry import ModelRegistry, ModelStatus, ModelTag
    
    registry = ModelRegistry()
    
    # Register a model
    registry.register(
        id="gpt2-large",
        name="GPT-2 Large",
        version="1.0",
        path="/models/gpt2-large",
        tags=[ModelTag.STABLE, ModelTag.GPU],
        config={"max_length": 1024}
    )
    
    # Get best model for task
    model = registry.get_best_model(criteria="latency")
    
    # Record usage
    registry.record_request("gpt2-large", latency_ms=50)
    
    # Get all models
    models = registry.list_models(status=ModelStatus.READY)
    ```
    """
    
    def __init__(self, storage_path: str = "./.model_registry.json"):
        """Initialize registry."""
        self._storage_path = storage_path
        self._models: Dict[str, ModelInfo] = {}
        self._lock = threading.RLock()
        self._load()
    
    def _load(self):
        """Load registry from storage."""
        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)
                for model_data in data.get("models", []):
                    model_data["status"] = ModelStatus(model_data.get("status", "loading"))
                    metrics_data = model_data.get("metrics", {})
                    metrics_data.pop("success_rate", None)
                    model_data["metrics"] = ModelMetrics(**metrics_data)
                    self._models[model_data["id"]] = ModelInfo(**model_data)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    def _save(self):
        """Save registry to storage."""
        with self._lock:
            data = {"models": [m.to_dict() for m in self._models.values()]}
            with open(self._storage_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
    
    def register(
        self,
        id: str,
        name: str,
        version: str,
        path: str,
        description: str = "",
        size_mb: float = 0,
        parameters: int = 0,
        framework: str = "pytorch",
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ModelInfo:
        """
        Register a new model.
        
        Args:
            id: Unique model ID
            name: Human-readable name
            version: Model version
            path: Path to model files
            description: Model description
            size_mb: Model size in MB
            parameters: Number of parameters
            framework: ML framework (pytorch, tensorflow, etc.)
            tags: Model tags (latest, stable, etc.)
            metadata: Additional metadata
            config: Model configuration
        
        Returns:
            Registered ModelInfo
        """
        tags = tags or [ModelTag.STABLE.value]
        metadata = metadata or {}
        config = config or {}
        
        model = ModelInfo(
            id=id,
            name=name,
            version=version,
            path=path,
            description=description,
            size_mb=size_mb,
            parameters=parameters,
            framework=framework,
            status=ModelStatus.READY,
            tags=tags,
            metadata=metadata,
            config=config,
        )
        
        with self._lock:
            self._models[id] = model
            self._save()
        
        return model
    
    def unregister(self, model_id: str) -> bool:
        """Unregister a model."""
        with self._lock:
            if model_id in self._models:
                del self._models[model_id]
                self._save()
                return True
        return False
    
    def get(self, model_id: str) -> Optional[ModelInfo]:
        """Get model by ID."""
        return self._models.get(model_id)
    
    def list_models(
        self,
        status: Optional[ModelStatus] = None,
        tags: Optional[List[str]] = None,
        framework: Optional[str] = None,
    ) -> List[ModelInfo]:
        """
        List models with optional filters.
        
        Args:
            status: Filter by status
            tags: Filter by tags (any match)
            framework: Filter by framework
        
        Returns:
            List of matching models
        """
        models = list(self._models.values())
        
        if status:
            models = [m for m in models if m.status == status]
        
        if tags:
            models = [m for m in models if any(t in m.tags for t in tags)]
        
        if framework:
            models = [m for m in models if m.framework == framework]
        
        return sorted(models, key=lambda m: m.updated_at, reverse=True)
    
    def update_status(self, model_id: str, status: ModelStatus) -> bool:
        """Update model status."""
        model = self._models.get(model_id)
        if model:
            model.status = status
            model.updated_at = time.time()
            self._save()
            return True
        return False
    
    def update_metrics(self, model_id: str, memory_mb: float) -> bool:
        """Update model memory metrics."""
        model = self._models.get(model_id)
        if model:
            model.metrics.memory_usage_mb = memory_mb
            self._save()
            return True
        return False
    
    def record_request(
        self,
        model_id: str,
        latency_ms: float,
        tokens: int = 0,
        success: bool = True,
    ) -> bool:
        """Record a request for a model."""
        model = self._models.get(model_id)
        if model:
            model.metrics.record_request(latency_ms, tokens, success)
            model.updated_at = time.time()
            self._save()
            return True
        return False
    
    def get_metrics(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model metrics."""
        model = self._models.get(model_id)
        return model.metrics.to_dict() if model else None
    
    def get_best_model(
        self,
        criteria: str = "latency",
        tags: Optional[List[str]] = None,
    ) -> Optional[ModelInfo]:
        """
        Get best model based on criteria.
        
        Args:
            criteria: Selection criteria (latency, throughput, memory)
            tags: Filter by tags
        
        Returns:
            Best model or None
        """
        models = self.list_models(status=ModelStatus.READY, tags=tags)
        
        if not models:
            return None
        
        if criteria == "latency":
            return min(models, key=lambda m: m.metrics.avg_latency_ms)
        elif criteria == "throughput":
            return max(models, key=lambda m: m.metrics.total_requests)
        elif criteria == "memory":
            return min(models, key=lambda m: m.metrics.memory_usage_mb)
        elif criteria == "success_rate":
            return max(
                models,
                key=lambda m: m.metrics.successful_requests / max(m.metrics.total_requests, 1)
            )
        
        return models[0]
    
    def get_latest(self, name: str) -> Optional[ModelInfo]:
        """Get latest version of a model by name."""
        models = [m for m in self._models.values() if m.name == name]
        if not models:
            return None
        return max(models, key=lambda m: m.version)
    
    def tag_model(self, model_id: str, tag: str) -> bool:
        """Add tag to model."""
        model = self._models.get(model_id)
        if model and tag not in model.tags:
            model.tags.append(tag)
            model.updated_at = time.time()
            self._save()
            return True
        return False
    
    def untag_model(self, model_id: str, tag: str) -> bool:
        """Remove tag from model."""
        model = self._models.get(model_id)
        if model and tag in model.tags:
            model.tags.remove(tag)
            model.updated_at = time.time()
            self._save()
            return True
        return False
    
    def deprecate(self, model_id: str, replacement: Optional[str] = None) -> bool:
        """Deprecate a model."""
        model = self._models.get(model_id)
        if model:
            model.status = ModelStatus.DEPRECATED
            if ModelTag.DEPRECATED.value not in model.tags:
                model.tags.append(ModelTag.DEPRECATED.value)
            if replacement:
                model.metadata["replacement"] = replacement
            model.updated_at = time.time()
            self._save()
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total = len(self._models)
        by_status = defaultdict(int)
        by_framework = defaultdict(int)
        total_requests = 0
        total_tokens = 0
        
        for model in self._models.values():
            by_status[model.status.value] += 1
            by_framework[model.framework] += 1
            total_requests += model.metrics.total_requests
            total_tokens += model.metrics.total_tokens
        
        return {
            "total_models": total,
            "by_status": dict(by_status),
            "by_framework": dict(by_framework),
            "total_requests": total_requests,
            "total_tokens": total_tokens,
        }
    
    def export_config(self) -> Dict[str, Any]:
        """Export registry configuration."""
        return {
            "models": [m.to_dict() for m in self._models.values()],
            "exported_at": time.time(),
        }
    
    def import_config(self, config: Dict[str, Any]) -> int:
        """Import registry configuration."""
        count = 0
        for model_data in config.get("models", []):
            try:
                model_data["status"] = ModelStatus(model_data.get("status", "loading"))
                model_data["metrics"] = ModelMetrics(**model_data.get("metrics", {}))
                model = ModelInfo(**model_data)
                self._models[model.id] = model
                count += 1
            except Exception:
                pass
        
        self._save()
        return count


class ModelSelector:
    """
    Intelligent model selection based on request requirements.
    
    Example:
    
    ```python
    selector = ModelSelector(registry)
    
    # Select model for task
    model = selector.select(
        task="chat",
        prefer_fast=True,
        max_latency_ms=100,
    )
    
    # Or let it decide automatically
    model = selector.auto_select(task="generation")
    ```
    """
    
    def __init__(self, registry: ModelRegistry):
        """Initialize selector."""
        self._registry = registry
    
    def select(
        self,
        task: Optional[str] = None,
        prefer_fast: bool = False,
        prefer_cheap: bool = False,
        max_latency_ms: Optional[float] = None,
        max_memory_mb: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[ModelInfo]:
        """
        Select best model for requirements.
        
        Args:
            task: Task type (chat, generate, embed, etc.)
            prefer_fast: Prefer low latency
            prefer_cheap: Prefer low memory
            max_latency_ms: Maximum acceptable latency
            max_memory_mb: Maximum memory usage
            tags: Preferred tags
        
        Returns:
            Selected model or None
        """
        models = self._registry.list_models(status=ModelStatus.READY)
        
        if tags:
            models = [m for m in models if any(t in m.tags for t in tags)]
        
        if max_latency_ms:
            models = [
                m for m in models
                if m.metrics.avg_latency_ms <= max_latency_ms or m.metrics.total_requests == 0
            ]
        
        if max_memory_mb:
            models = [m for m in models if m.metrics.memory_usage_mb <= max_memory_mb]
        
        if not models:
            return None
        
        if prefer_fast:
            return min(models, key=lambda m: m.metrics.avg_latency_ms)
        elif prefer_cheap:
            return min(models, key=lambda m: m.metrics.memory_usage_mb)
        elif task == "chat":
            return self._registry.get_best_model("latency", tags=None)
        elif task == "generation":
            return self._registry.get_best_model("throughput", tags=None)
        elif task == "embedding":
            return self._registry.get_best_model("memory", tags=None)
        
        return models[0]
    
    def auto_select(self, task: str) -> Optional[ModelInfo]:
        """Auto-select model based on task."""
        task_defaults = {
            "chat": {"prefer_fast": True},
            "generate": {"prefer_fast": True},
            "complete": {"prefer_fast": True},
            "embed": {"prefer_cheap": True},
            "analyze": {"prefer_fast": True},
            "summarize": {"prefer_fast": True},
        }
        
        kwargs = task_defaults.get(task, {})
        return self.select(task=task, **kwargs)
