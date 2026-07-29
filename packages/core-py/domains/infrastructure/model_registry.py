"""
Model registry — composable registry for serving multiple models.

Provides ``ModelRegistry``, a thread-safe registry that wraps ``ModelServer``
instances. The registry acts as the single frontend for all model operations:
loading, unloading, generation, listing, and health checking.

Usage::

    registry = get_model_registry()
    registry.load("gpt2", model, tokenizer)
    result = await registry.generate("gpt2", "Hello", max_new_tokens=50)
    metrics = registry.get_metrics()
"""

import asyncio
import logging
import time
from threading import Lock
from typing import Any, Optional, Union, get_type_hints

from .model_server import ModelServer, ModelStatus

logger = logging.getLogger("slo.infrastructure.model_registry")

DEFAULT_MODEL_ID = "default"


class ModelRegistry:
    """Thread-safe registry for composable model serving.

    Supports multiple named model instances, each with its own concurrency
    semaphore, circuit breaker, and metrics collection.
    """

    def __init__(self) -> None:
        self._servers: dict[str, Union[ModelServer, Any]] = {}
        self._lock = Lock()
        self._default_id: Optional[str] = None

    # --- Registration ---

    def register(
        self,
        model_id: str,
        model: Any,
        tokenizer: Any,
        make_default: bool = False,
        max_concurrent: Optional[int] = None,
        generate_timeout: float = 120.0,
        enable_circuit_breaker: bool = True,
        process_guard: Optional[Any] = None,
    ) -> ModelServer:
        """Register a model with the registry.

        Args:
            model_id: Unique identifier for this model instance.
            model: The HuggingFace model object.
            tokenizer: The corresponding tokenizer.
            make_default: If True, this model becomes the default for
                unqualified ``generate()`` calls.
            max_concurrent: Maximum concurrent generation requests.
            generate_timeout: Per-generation timeout in seconds.
            enable_circuit_breaker: Whether to enable circuit breaker.
            process_guard: Optional ``ProcessGuard`` for subprocess isolation.
                When set, ``_generate_sync()`` delegates to the guard and
                crash/restart callbacks are wired to the circuit breaker.

        Returns:
            The created ModelServer instance.
        """
        server = ModelServer(
            model=model,
            tokenizer=tokenizer,
            model_id=model_id,
            max_concurrent=max_concurrent,
            generate_timeout=generate_timeout,
            enable_circuit_breaker=enable_circuit_breaker,
            process_guard=process_guard,
        )
        with self._lock:
            old = self._servers.get(model_id)
            self._servers[model_id] = server
            if make_default or self._default_id is None:
                self._default_id = model_id

        # Clean up old server if replacing
        if old is not None and old is not server:
            old.set_status(ModelStatus.UNLOADED)
            logger.info("ModelRegistry: replaced model '%s'", model_id, extra={"tag": "MODEL"})

        logger.info(
            "ModelRegistry: registered '%s' (device=%s, timeout=%ss, concurrent=%s)",
            model_id, server._device, generate_timeout, max_concurrent if max_concurrent is not None else "default",
            extra={"tag": "MODEL"},
        )
        return server

    def register_engine(
        self,
        engine_id: str,
        provider: Any,
        make_default: bool = False,
    ) -> None:
        """Register an inference engine provider.

        The provider is expected to implement ``get_metrics_snapshot()``
        and ``metadata``.

        Args:
            engine_id: Unique identifier for this engine.
            provider: The provider instance (e.g. ``SloNetChatProvider``).
            make_default: If True, this becomes the default.
        """
        with self._lock:
            self._servers[engine_id] = provider  # type: ignore[assignment]
            if make_default or self._default_id is None:
                self._default_id = engine_id
        logger.info(
            "ModelRegistry: registered engine '%s' (make_default=%s)",
            engine_id, make_default,
            extra={"tag": "MODEL"},
        )

    def unregister(self, model_id: str) -> bool:
        """Unregister a model and clean up resources."""
        with self._lock:
            server = self._servers.pop(model_id, None)
            if server is None:
                return False
            server.set_status(ModelStatus.UNLOADED)
            if self._default_id == model_id:
                self._default_id = next(iter(self._servers)) if self._servers else None
        logger.info("ModelRegistry: unregistered '%s'", model_id, extra={"tag": "MODEL"})
        return True

    # --- Access ---

    def get(self, model_id: Optional[str] = None) -> Optional[ModelServer]:
        """Get a ModelServer by ID, or the default if not specified."""
        with self._lock:
            mid = model_id or self._default_id
            if mid is None:
                return None
            return self._servers.get(mid)

    def list_models(self) -> list[dict]:
        """List all registered models with their status and metrics."""
        with self._lock:
            servers = list(self._servers.items())
        result = []
        for mid, server in servers:
            entry = server.get_metrics_snapshot()
            entry["model_id"] = mid
            entry["is_default"] = mid == self._default_id
            result.append(entry)
        return result

    @property
    def default_id(self) -> Optional[str]:
        with self._lock:
            return self._default_id

    @default_id.setter
    def default_id(self, model_id: str) -> None:
        with self._lock:
            if model_id in self._servers:
                self._default_id = model_id

    # --- Generation ---

    async def generate(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Generate text using the specified or default model.

        Args:
            prompt: Input text.
            model_id: Model to use (uses default if None).
            **kwargs: Passed through to ModelServer.generate().

        Returns:
            Generation result dict with ``text`` and ``tokens_generated``.
        """
        server = self.get(model_id)
        if server is None:
            raise RuntimeError(
                f"No model registered" + (f" (requested '{model_id}')" if model_id else "")
            )
        return await server.generate(prompt, **kwargs)

    # --- Health ---

    def health_summary(self) -> dict:
        """Return a summary suitable for the /health endpoint."""
        models = self.list_models()
        loaded_count = sum(
            1 for m in models if m.get("status") in ("ready", "degraded")
        )
        degraded = any(m.get("status") == "degraded" for m in models)
        errors = [m for m in models if m.get("status") == "error"]
        return {
            "models_loaded": loaded_count,
            "models_registered": len(models),
            "healthy": loaded_count > 0 and not degraded,
            "degraded": degraded,
            "has_errors": len(errors) > 0,
            "default_model": self.default_id,
            "models": models,
        }

    def reset_metrics(self) -> None:
        """Reset all metrics counters."""
        servers = self.list_models()
        for s in servers:
            server = self.get(s["model_id"])
            if server:
                server.metrics = type(server.metrics)()
                server.set_status(ModelStatus.READY)


_registry: Optional[ModelRegistry] = None
_registry_lock = Lock()


def get_model_registry() -> ModelRegistry:
    """Get (or create) the singleton ModelRegistry."""
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ModelRegistry()
    return _registry
