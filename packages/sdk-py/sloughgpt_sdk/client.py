"""
SloughGPT SDK Client
Main client for interacting with the SloughGPT API.
"""

from __future__ import annotations

import sys
import os
import importlib.util

import requests
from typing import List, Optional, Dict, Any, Iterator, Union, TYPE_CHECKING
import time
import json

if TYPE_CHECKING:
    from models import (
        GenerateRequest,
        GenerationResult,
        ChatRequest,
        ChatMessage,
        ChatResult,
        BatchRequest,
        BatchResult,
        ModelInfo,
        DatasetInfo,
        HealthStatus,
        SystemInfo,
        MetricsData,
    )
else:
    _models_spec = importlib.util.spec_from_file_location("models", os.path.join(os.path.dirname(__file__), "models.py"))
    models = importlib.util.module_from_spec(_models_spec)
    _models_spec.loader.exec_module(models)

    GenerateRequest = models.GenerateRequest
    GenerateResult = models.GenerationResult
    ChatRequest = models.ChatRequest
    ChatMessage = models.ChatMessage
    ChatResult = models.ChatResult
    BatchRequest = models.BatchRequest
    BatchResult = models.BatchResult
    ModelInfo = models.ModelInfo
    DatasetInfo = models.DatasetInfo
    HealthStatus = models.HealthStatus
    SystemInfo = models.SystemInfo
    MetricsData = models.MetricsData


def _build_training_start_payload(
    model_name: str,
    dataset_id: str,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """JSON body for POST /training/start (server ``TrainingRequest``).

    Extra keyword arguments are merged into the payload as-is (e.g. ``max_steps``,
    ``n_embed`` … ``block_size``, ``log_interval``, ``eval_interval``, ``dropout``,
    ``weight_decay``, ``gradient_accumulation_steps``, ``max_grad_norm``,
    ``use_mixed_precision``, ``mixed_precision_dtype``, ``warmup_steps``, ``min_lr``,
    ``scheduler``, ``use_lora``, ``lora_rank``, ``lora_alpha``, ``checkpoint_dir``,
    ``checkpoint_interval``, ``save_best_only``, ``max_checkpoints``, ``device``).

    Trainer ``step_*.pt`` files on the server include ``stoi`` / ``itos`` / ``chars``;
    see ``docs/policies/CONTRIBUTING.md`` (*Checkpoint vocabulary*).
    """
    opts = dict(kwargs)
    name = opts.pop("name", f"{model_name}-training")
    manifest_uri = opts.pop("manifest_uri", None)
    dataset_ref = opts.pop("dataset_ref", None)
    payload: Dict[str, Any] = {
        "name": name,
        "model": model_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    }
    if manifest_uri is not None:
        payload["manifest_uri"] = manifest_uri
    elif dataset_ref is not None:
        payload["dataset_ref"] = dataset_ref
    else:
        payload["dataset"] = dataset_id
    payload.update(opts)
    return payload


def _coerce_training_jobs_list(data: Any) -> List[Dict[str, Any]]:
    """``GET /training/jobs`` returns a JSON array; some gateways may wrap ``{jobs: [...]}``."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        jobs = data.get("jobs")
        if isinstance(jobs, list):
            return jobs
    return []


class SloughGPTClient:
    """
    Python client for the SloughGPT API.
    
    Example usage:
    
    ```python
    from sloughgpt_sdk import SloughGPTClient
    
    client = SloughGPTClient(base_url="http://localhost:8000")
    
    # Check health
    health = client.health()
    print(f"Status: {health.status}")
    
    # Generate text
    result = client.generate("Hello, how are you?")
    print(result.generated_text)
    
    # Chat
    chat_result = client.chat([
        ChatMessage.user("Hello!"),
        ChatMessage.assistant("Hi! How can I help you?"),
        ChatMessage.user("Tell me a joke"),
    ])
    print(chat_result.message.content)
    
    # Stream generation
    for token in client.generate_stream("Once upon a time"):
        print(token, end="", flush=True)
    ```
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        verify_ssl: bool = True,
        headers: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the SloughGPT client.
        
        Args:
            base_url: Base URL of the SloughGPT API server.
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
            verify_ssl: Whether to verify SSL certificates.
            headers: Additional headers to include with requests.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        
        self._headers = headers or {}
        if api_key:
            self._headers["X-API-Key"] = api_key
        
        self._session = requests.Session()
        self._session.headers.update(self._headers)
    
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> requests.Response:
        """Make an HTTP request to the API."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("verify", self.verify_ssl)
        
        response = self._session.request(method, url, **kwargs)
        response.raise_for_status()
        return response
    
    # Health & Status
    
    def health(self) -> HealthStatus:
        """Check API health status."""
        response = self._request("GET", "/health")
        return HealthStatus.from_response(response.json())
    
    def liveness(self) -> Dict[str, Any]:
        """Check if the server is alive (liveness probe)."""
        response = self._request("GET", "/health/live")
        return response.json()
    
    def readiness(self) -> Dict[str, Any]:
        """Check if the server is ready (readiness probe)."""
        response = self._request("GET", "/health/ready")
        return response.json()
    
    def info(self) -> SystemInfo:
        """Get detailed system information."""
        response = self._request("GET", "/info")
        return SystemInfo.from_response(response.json())
    
    # Text Generation
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The input prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (0-2).
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling parameter.
            **kwargs: Additional generation parameters.
        
        Returns:
            GenerationResult with generated text.
        """
        request = GenerateRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            **kwargs
        )
        
        start_time = time.time()
        response = self._request("POST", "/generate", json=request.to_dict())
        elapsed_ms = (time.time() - start_time) * 1000
        
        result = GenerationResult.from_response(response.json(), prompt)
        if result.inference_time_ms is None:
            result.inference_time_ms = elapsed_ms
        return result
    
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        **kwargs
    ) -> Iterator[str]:
        """
        Generate text with streaming response.
        
        Args:
            prompt: The input prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            **kwargs: Additional generation parameters.
        
        Yields:
            Generated tokens as they arrive.
        """
        request = GenerateRequest(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )
        
        response = self._request(
            "POST",
            "/generate/stream",
            json=request.to_dict(),
            stream=True
        )
        
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                data = line[5:].strip()
                if data and data != "[DONE]":
                    try:
                        yield data
                    except json.JSONDecodeError:
                        yield data
    
    # Chat Completions
    
    def chat(
        self,
        messages: Union[List[ChatMessage], List[Dict[str, str]]],
        model: Optional[str] = None,
        temperature: float = 0.8,
        max_new_tokens: int = 100,
        **kwargs
    ) -> ChatResult:
        """
        Generate a chat completion.
        
        Args:
            messages: List of chat messages.
            model: Model to use (optional).
            temperature: Sampling temperature.
            max_new_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters.
        
        Returns:
            ChatResult with assistant's response.
        """
        chat_messages = []
        for m in messages:
            if isinstance(m, ChatMessage):
                chat_messages.append(m)
            elif isinstance(m, dict):
                chat_messages.append(ChatMessage(
                    role=m.get("role", "user"),
                    content=m.get("content", "")
                ))
        
        request = ChatRequest(
            messages=chat_messages,
            model=model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        
        response = self._request("POST", "/chat", json=request.to_dict())
        data = response.json()
        err = data.get("error")
        if isinstance(err, str) and err.strip() and not str(data.get("text") or "").strip():
            from .exceptions import SloughGPTError

            raise SloughGPTError(err)
        return ChatResult.from_response(data)
    
    def chat_stream(
        self,
        messages: Union[List[ChatMessage], List[Dict[str, str]]],
        **kwargs
    ) -> Iterator[str]:
        """
        Generate a chat completion with streaming.
        
        Args:
            messages: List of chat messages.
            **kwargs: Additional parameters.
        
        Yields:
            Generated tokens as they arrive.
        """
        chat_messages = []
        for m in messages:
            if isinstance(m, ChatMessage):
                chat_messages.append(m)
            elif isinstance(m, dict):
                chat_messages.append(ChatMessage(
                    role=m.get("role", "user"),
                    content=m.get("content", "")
                ))
        
        request = ChatRequest(messages=chat_messages, **kwargs)
        
        response = self._request(
            "POST",
            "/chat/stream",
            json=request.to_dict(),
            stream=True
        )
        
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data:"):
                raw = line[5:].strip()
                if raw and raw != "[DONE]":
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("error"):
                        break
                    tok = obj.get("token")
                    if tok:
                        yield tok
    
    # Batch Processing
    
    def batch_generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 0.8,
    ) -> BatchResult:
        """
        Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts.
            max_new_tokens: Maximum tokens per generation.
            temperature: Sampling temperature.
        
        Returns:
            BatchResult with all generated texts.
        """
        request = BatchRequest(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        response = self._request("POST", "/generate/batch", json=request.to_dict())
        return BatchResult.from_response(response.json(), prompts)
    
    # Models
    
    def list_models(self) -> List[ModelInfo]:
        """List available models."""
        response = self._request("GET", "/models")
        data = response.json()
        models = data.get("models", data) if isinstance(data, dict) else data
        return [ModelInfo.from_dict(m) for m in models]
    
    def get_model(self, model_id: str) -> ModelInfo:
        """Get information about a specific model."""
        response = self._request("GET", f"/models/{model_id}")
        return ModelInfo.from_dict(response.json())
    
    def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a model into memory."""
        response = self._request("POST", "/models/load", json={"model_id": model_id})
        return response.json()
    
    def list_hf_models(self, query: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """List available HuggingFace models."""
        params = {"limit": limit}
        if query:
            params["q"] = query
        
        response = self._request("GET", "/models/hf", params=params)
        return response.json().get("models", [])
    
    # Datasets
    
    def list_datasets(self) -> List[DatasetInfo]:
        """List available datasets."""
        response = self._request("GET", "/datasets")
        data = response.json()
        datasets = data.get("datasets", data) if isinstance(data, dict) else data
        return [DatasetInfo.from_dict(d) for d in datasets]
    
    def get_dataset(self, dataset_id: str) -> DatasetInfo:
        """Get information about a specific dataset."""
        response = self._request("GET", f"/datasets/{dataset_id}")
        return DatasetInfo.from_dict(response.json())
    
    # Metrics
    
    def metrics(self) -> MetricsData:
        """Get API metrics."""
        response = self._request("GET", "/metrics")
        return MetricsData.from_response(response.json())
    
    def metrics_prometheus(self) -> str:
        """Get metrics in Prometheus text exposition format."""
        response = self._request("GET", "/metrics/prometheus")
        return response.text
    
    # Authentication
    
    def create_token(self, api_key: str) -> Dict[str, Any]:
        """Exchange API key for JWT (POST /auth/token)."""
        response = self._request("POST", "/auth/token", json={"api_key": api_key})
        return response.json()

    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Deprecated alias: server uses API keys, not username/password.

        If ``password`` looks like an API key (non-empty), it is sent as ``api_key``.
        Otherwise ``username`` is sent as the API key (for backwards compatibility).
        """
        api_key = password if password else username
        return self.create_token(api_key)
    
    def refresh_token(self, access_token: str) -> Dict[str, Any]:
        """Refresh JWT (POST /auth/refresh with Bearer token)."""
        response = self._request(
            "POST",
            "/auth/refresh",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        return response.json()
    
    # Training
    
    def start_training(
        self,
        model_name: str,
        dataset_id: str,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Start a training job (POST /training/start).

        Args:
            model_name: Passed as JSON ``model``.
            dataset_id: Folder name under ``datasets/<id>/input.txt`` when not using manifest/ref.
            epochs: Training epochs.
            batch_size: Batch size.
            learning_rate: Learning rate.
            **kwargs: Optional ``name``, ``manifest_uri``, ``dataset_ref``, model dims
                (``n_embed`` … ``block_size``), ``max_steps``, ``log_interval``,
                ``eval_interval``, and trainer options (``dropout``, ``weight_decay``,
                ``use_mixed_precision``, ``mixed_precision_dtype``, ``scheduler``, LoRA,
                checkpoint fields, ``device``, etc.). Provide exactly one corpus selector:
                default ``dataset``, or ``manifest_uri`` / ``dataset_ref``.

        Returns:
            Training job information.

        Server-side ``step_*.pt`` checkpoints include ``stoi`` / ``itos`` / ``chars`` for
        char LM eval; see CONTRIBUTING (*Checkpoint vocabulary*).
        """
        payload = _build_training_start_payload(
            model_name,
            dataset_id,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs,
        )
        response = self._request("POST", "/training/start", json=payload)
        return response.json()

    def resolve_training(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Dry-run corpus resolver (POST /train/resolve).

        Does not write checkpoints. After ``POST /train`` or ``POST /training/start``,
        trainer ``step_*.pt`` embeds ``stoi`` / ``itos`` / ``chars``; see CONTRIBUTING
        (*Checkpoint vocabulary*).
        """
        response = self._request("POST", "/train/resolve", json=body)
        return response.json()

    def infer_v1(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """SloughGPT Standard v1 inference (POST /v1/infer)."""
        hdrs = {**dict(self._session.headers), "X-SloughGPT-Standard": "1"}
        response = self._request("POST", "/v1/infer", json=body, headers=hdrs)
        return response.json()
    
    def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """Get training job status (GET /training/jobs/{job_id}).

        Completed jobs may include ``checkpoint``; native ``step_*.pt`` embeds
        ``stoi`` / ``itos`` / ``chars`` — CONTRIBUTING (*Checkpoint vocabulary*).
        """
        response = self._request("GET", f"/training/jobs/{job_id}")
        return response.json()
    
    def list_training_jobs(self) -> List[Dict[str, Any]]:
        """List all training jobs (GET /training/jobs).

        Same ``checkpoint`` / ``step_*.pt`` semantics as :meth:`get_training_status`.
        """
        response = self._request("GET", "/training/jobs")
        return _coerce_training_jobs_list(response.json())
    
    # Simple Tracking (UX-friendly)
    
    def track(self, name: str = "default") -> "SimpleTracker":
        """
        Start a simple tracking session.
        
        Simple usage:
        
        ```python
        tracker = client.track("my-training")
        tracker.log("accuracy", 0.95)
        tracker.log("loss", 0.05)
        tracker.finish()
        ```
        """
        return SimpleTracker(self, name)
    
    def log(self, metric: str, value: float, step: Optional[int] = None):
        """
        Quick log a metric value.
        
        ```python
        client.log("accuracy", 0.95)
        client.log("loss", 0.05)
        ```
        """
        self._request("POST", "/metrics/log", json={
            "metric": metric,
            "value": value,
            "step": step
        })
    
    # Legacy Experiments (for advanced users)
    
    def create_experiment(
        self,
        name: str,
        description: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new experiment (advanced)."""
        payload = {"name": name, "description": description, **kwargs}
        response = self._request("POST", "/experiments", json=payload)
        return response.json()
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments (advanced)."""
        response = self._request("GET", "/experiments")
        return response.json().get("experiments", [])
    
    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment details (advanced)."""
        response = self._request("GET", f"/experiments/{experiment_id}")
        return response.json()
    
    def log_metric(
        self,
        experiment_id: str,
        metric_name: str,
        value: float,
        step: Optional[int] = None
    ) -> Dict[str, Any]:
        """Log a metric to an experiment (advanced)."""
        payload = {"metric": metric_name, "value": value}
        if step is not None:
            payload["step"] = step
        response = self._request(
            "POST",
            f"/experiments/{experiment_id}/log_metric",
            json=payload
        )
        return response.json()
    
    def log_param(
        self,
        experiment_id: str,
        param_name: str,
        value: Any
    ) -> Dict[str, Any]:
        """Log a parameter to an experiment (advanced)."""
        payload = {"param": param_name, "value": value}
        response = self._request(
            "POST",
            f"/experiments/{experiment_id}/log_param",
            json=payload
        )
        return response.json()
    
    # Rate Limiting
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get rate limit status."""
        response = self._request("GET", "/rate-limit/status")
        return response.json()
    
    def check_rate_limit(self) -> Dict[str, Any]:
        """Check if rate limit would allow request."""
        response = self._request("GET", "/rate-limit/check")
        return response.json()
    
    # Personalities
    
    def get_personalities(self) -> List[Dict[str, Any]]:
        """Get available personalities."""
        response = self._request("GET", "/personalities")
        return response.json().get("personalities", [])
    
    def set_personality(self, personality: str) -> Dict[str, Any]:
        """Set the current personality."""
        response = self._request("POST", "/personalities", json={"personality": personality})
        return response.json()
    
    # Context Manager
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self._session.close()
    
    # Convenience Methods
    
    def quick_generate(self, prompt: str) -> str:
        """Quick generation with default settings."""
        return self.generate(prompt).generated_text
    
    def quick_chat(self, user_message: str) -> str:
        """Quick chat with a single user message."""
        result = self.chat([ChatMessage.user(user_message)])
        return result.message.content
    
    # Model Registry
    
    def register_model(
        self,
        model_id: str,
        name: str,
        version: str,
        path: str,
        description: str = "",
        size_mb: float = 0,
        parameters: int = 0,
        framework: str = "pytorch",
        tags: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Register a model with the server registry."""
        payload = {
            "id": model_id,
            "name": name,
            "version": version,
            "path": path,
            "description": description,
            "size_mb": size_mb,
            "parameters": parameters,
            "framework": framework,
            "tags": tags or [],
            **kwargs
        }
        response = self._request("POST", "/registry/models", json=payload)
        return response.json()
    
    def list_registered_models(
        self,
        status: Optional[str] = None,
        tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List models in the server registry."""
        params = {}
        if status:
            params["status"] = status
        if tag:
            params["tag"] = tag
        response = self._request("GET", "/registry/models", params=params)
        return response.json().get("models", [])
    
    def get_registered_model(self, model_id: str) -> Dict[str, Any]:
        """Get a model from the server registry."""
        response = self._request("GET", f"/registry/models/{model_id}")
        return response.json()
    
    def unregister_model(self, model_id: str) -> Dict[str, Any]:
        """Unregister a model from the server registry."""
        response = self._request("DELETE", f"/registry/models/{model_id}")
        return response.json()
    
    def record_to_registry(
        self,
        model_id: str,
        latency_ms: float,
        tokens: int = 0,
        success: bool = True
    ) -> Dict[str, Any]:
        """Record a request to the server registry."""
        payload = {
            "latency_ms": latency_ms,
            "tokens": tokens,
            "success": success
        }
        response = self._request("POST", f"/registry/models/{model_id}/record", json=payload)
        return response.json()
    
    def get_registry_metrics(self, model_id: str) -> Dict[str, Any]:
        """Get model metrics from the server registry."""
        response = self._request("GET", f"/registry/models/{model_id}/metrics")
        return response.json()
    
    def get_best_registered_model(
        self,
        criteria: str = "latency",
        tag: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get the best model from the server registry."""
        params = {"criteria": criteria}
        if tag:
            params["tag"] = tag
        response = self._request("GET", "/registry/best", params=params)
        data = response.json()
        if "error" in data:
            return None
        return data
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics from the server."""
        response = self._request("GET", "/registry/stats")
        return response.json()


class SimpleTracker:
    """
    Simple context manager for tracking metrics.
    
    Example:
    
    ```python
    tracker = client.track("training-v1")
    
    for epoch in range(10):
        acc = train()
        tracker.log("accuracy", acc)
    
    tracker.finish()
    ```
    """
    
    def __init__(self, client: "SloughGPTClient", name: str):
        """Initialize tracker."""
        self._client = client
        self._name = name
        self._step = 0
    
    def log(self, metric: str, value: float):
        """Log a metric."""
        self._client.log(metric, value, self._step)
    
    def next_step(self):
        """Move to next step."""
        self._step += 1
    
    def finish(self):
        """Finish tracking."""
        pass  # Could add summary call here
    
    def __enter__(self):
        """Enter context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.finish()


class AsyncSloughGPTClient:
    """
    Async Python client for the SloughGPT API.
    
    Example usage:
    
    ```python
    import asyncio
    from sloughgpt_sdk import AsyncSloughGPTClient
    
    async def main():
        async with AsyncSloughGPTClient() as client:
            result = await client.generate("Hello!")
            print(result.generated_text)
    
    asyncio.run(main())
    ```
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: int = 30,
        verify_ssl: bool = True,
        headers: Optional[Dict[str, str]] = None,
    ):
        """Initialize the async client."""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._headers = headers or {}
        if api_key:
            self._headers["X-API-Key"] = api_key
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an async HTTP request."""
        import httpx

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        kwargs.setdefault("timeout", self.timeout)
        extra_headers = kwargs.pop("extra_headers", None)
        merged = {**self._headers, **(extra_headers or {})}

        async with httpx.AsyncClient(verify=self.verify_ssl, headers=merged) as client:
            response = await client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
    
    async def health(self) -> HealthStatus:
        """Check API health."""
        data = await self._request("GET", "/health")
        return HealthStatus.from_response(data)
    
    async def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text."""
        from .models import GenerateRequest
        
        request = GenerateRequest(prompt=prompt, **kwargs)
        data = await self._request("POST", "/generate", json=request.to_dict())
        return GenerationResult.from_response(data, prompt)
    
    async def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResult:
        """Generate chat completion."""
        body = {
            "messages": [m.to_dict() if isinstance(m, ChatMessage) else m for m in messages],
            **kwargs,
        }
        data = await self._request("POST", "/chat", json=body)
        err = data.get("error")
        if isinstance(err, str) and err.strip() and not str(data.get("text") or "").strip():
            from .exceptions import SloughGPTError

            raise SloughGPTError(err)
        return ChatResult.from_response(data)
    
    async def list_models(self) -> List[ModelInfo]:
        """List available models."""
        data = await self._request("GET", "/models")
        models = data.get("models", data) if isinstance(data, dict) else data
        return [ModelInfo.from_dict(m) for m in models]
    
    async def metrics(self) -> MetricsData:
        """Get API metrics."""
        data = await self._request("GET", "/metrics")
        return MetricsData.from_response(data)
    
    async def start_training(
        self,
        model_name: str,
        dataset_id: str,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Start a training job (POST /training/start).

        See synchronous :meth:`start_training` for ``**kwargs`` (corpus selectors,
        loop/model fields, and trainer hyperparameters forwarded by
        :func:`_build_training_start_payload`). On-disk ``step_*.pt`` bundles include
        charset maps; see CONTRIBUTING (*Checkpoint vocabulary*).
        """
        payload = _build_training_start_payload(
            model_name,
            dataset_id,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs,
        )
        return await self._request("POST", "/training/start", json=payload)

    async def resolve_training(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Dry-run corpus resolver (POST /train/resolve).

        Does not write checkpoints. After ``POST /train`` or ``POST /training/start``,
        trainer ``step_*.pt`` embeds ``stoi`` / ``itos`` / ``chars``; see CONTRIBUTING
        (*Checkpoint vocabulary*).
        """
        return await self._request("POST", "/train/resolve", json=body)

    async def infer_v1(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """SloughGPT Standard v1 inference (POST /v1/infer)."""
        return await self._request(
            "POST",
            "/v1/infer",
            json=body,
            extra_headers={"X-SloughGPT-Standard": "1"},
        )
    
    async def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """Get training job status (GET /training/jobs/{job_id}).

        Completed jobs may include ``checkpoint``; native ``step_*.pt`` embeds
        ``stoi`` / ``itos`` / ``chars`` — CONTRIBUTING (*Checkpoint vocabulary*).
        """
        return await self._request("GET", f"/training/jobs/{job_id}")
    
    async def list_training_jobs(self) -> List[Dict[str, Any]]:
        """List all training jobs (GET /training/jobs).

        Same ``checkpoint`` / ``step_*.pt`` semantics as :meth:`get_training_status`.
        """
        data = await self._request("GET", "/training/jobs")
        return _coerce_training_jobs_list(data)
    
    async def create_experiment(self, name: str, description: str = "", **kwargs) -> Dict[str, Any]:
        """Create a new experiment."""
        payload = {"name": name, "description": description, **kwargs}
        return await self._request("POST", "/experiments", json=payload)
    
    async def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        data = await self._request("GET", "/experiments")
        return data.get("experiments", [])
    
    async def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment details."""
        return await self._request("GET", f"/experiments/{experiment_id}")
    
    async def log_metric(self, experiment_id: str, metric_name: str, value: float, **kwargs) -> Dict[str, Any]:
        """Log a metric to an experiment."""
        payload = {"metric": metric_name, "value": value, **kwargs}
        return await self._request("POST", f"/experiments/{experiment_id}/log_metric", json=payload)
    
    async def __aenter__(self):
        """Enter async context manager."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context manager."""
        pass
