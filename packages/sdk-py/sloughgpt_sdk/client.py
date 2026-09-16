"""
SloughGPT SDK Client
Main client for interacting with the SloughGPT API.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import requests

if TYPE_CHECKING:
    from models import (
        ChatMessage,
        ChatRequest,
        ChatResult,
        DatasetInfo,
        GenerateRequest,
        GenerationResult,
        HealthStatus,
        MetricsData,
        ModelInfo,
        SystemInfo,
    )
else:
    _models_spec = importlib.util.spec_from_file_location(
        "models", os.path.join(os.path.dirname(__file__), "models.py")
    )
    models = importlib.util.module_from_spec(_models_spec)
    _models_spec.loader.exec_module(models)

    GenerateRequest = models.GenerateRequest
    GenerationResult = models.GenerationResult
    ChatRequest = models.ChatRequest
    ChatMessage = models.ChatMessage
    ChatResult = models.ChatResult
    ModelInfo = models.ModelInfo
    DatasetInfo = models.DatasetInfo
    HealthStatus = models.HealthStatus
    SystemInfo = models.SystemInfo
    MetricsData = models.MetricsData


def _unwrap_response(data: Any) -> Any:
    """Unwrap the StandardResponse envelope ``{"status": "success", "data": ...}``.

    Returns the payload verbatim when the response is not enveloped
    (e.g. a bare list or dict), keeping the SDK tolerant of both shapes.

    Args:
        data: raw JSON decoded from the API response.

    Returns:
        The inner ``data`` payload when enveloped, otherwise ``data`` unchanged.
    """
    if isinstance(data, dict) and data.get("status") == "success" and "data" in data:
        return data["data"]
    return data


def _build_training_start_payload(
    model_name: str,
    dataset_id: str,
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build JSON body for POST /training/start."""
    opts = dict(kwargs)
    name = opts.pop("name", f"{model_name}-training")
    manifest_uri = opts.pop("manifest_uri", None)
    dataset_ref = opts.pop("dataset_ref", None)
    payload: dict[str, Any] = {
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


def _coerce_training_jobs_list(data: Any) -> list[dict[str, Any]]:
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
    health = client.health()
    result = client.generate("Hello")
    chat_result = client.chat([ChatMessage.user("Hi!")])
    for token in client.generate_stream("Once upon a time"):
        print(token, end="", flush=True)
    ```
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: int = 30,
        verify_ssl: bool = True,
        headers: dict[str, str] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._headers = headers or {}
        if api_key:
            self._headers["X-API-Key"] = api_key
        self._session = requests.Session()
        self._session.headers.update(self._headers)

    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("verify", self.verify_ssl)
        response = self._session.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    # ============ Auth ============

    def get_token(self, api_key: str) -> dict[str, Any]:
        """Exchange an API key for a JWT access token (``POST /auth/token``)."""
        response = self._request("POST", "/auth/token", json={"api_key": api_key})
        return _unwrap_response(response.json())

    # ============ Health & Status ============

    def health(self) -> HealthStatus:
        """Check API health status."""
        response = self._request("GET", "/health")
        return HealthStatus.from_response(_unwrap_response(response.json()))

    def liveness(self) -> dict[str, Any]:
        """Check if the server is alive."""
        response = self._request("GET", "/health/live")
        return _unwrap_response(response.json())

    def readiness(self) -> dict[str, Any]:
        """Check if the server is ready."""
        response = self._request("GET", "/health/ready")
        return _unwrap_response(response.json())

    def detailed_health(self) -> dict[str, Any]:
        """Get detailed health info."""
        response = self._request("GET", "/health/detailed")
        return _unwrap_response(response.json())

    def info(self) -> SystemInfo:
        """Get detailed system information."""
        response = self._request("GET", "/info")
        return SystemInfo.from_response(_unwrap_response(response.json()))

    # ============ Text Generation ============

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        **kwargs,
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
            **kwargs,
        )
        start_time = time.time()
        response = self._request("POST", "/inference/generate", json=request.to_dict())
        elapsed_ms = (time.time() - start_time) * 1000
        result = GenerationResult.from_response(response.json(), prompt)
        if result.inference_time_ms is None:
            result.inference_time_ms = elapsed_ms
        return result

    def generate_stream(
        self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.8, **kwargs
    ) -> Iterator[str]:
        """
        Generate text with streaming response.

        Yields:
            Generated tokens as they arrive.
        """
        request = GenerateRequest(
            prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature, **kwargs
        )
        response = self._request(
            "POST", "/inference/generate/stream", json=request.to_dict(), stream=True
        )
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                raw = line[5:].strip()
                if raw and raw != "[DONE]":
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    payload = obj.get("data") or {}
                    if not isinstance(payload, dict):
                        continue
                    if obj.get("status") == "error" or payload.get("error"):
                        break
                    tok = payload.get("token")
                    if tok:
                        yield tok

    # ============ Chat Completions ============

    def chat(
        self,
        messages: list[ChatMessage] | list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.8,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> ChatResult:
        """Generate a chat completion."""
        chat_messages = []
        for m in messages:
            if isinstance(m, ChatMessage):
                chat_messages.append(m)
            elif isinstance(m, dict):
                chat_messages.append(
                    ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
                )
        request = ChatRequest(
            messages=chat_messages,
            model=model,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        response = self._request("POST", "/chat", json=request.to_dict())
        data = response.json()
        err = data.get("error")
        if isinstance(err, str) and err.strip() and not str(data.get("message") or "").strip():
            from .exceptions import SloughGPTError

            raise SloughGPTError(err)
        return ChatResult.from_response(data)

    def chat_stream(
        self, messages: list[ChatMessage] | list[dict[str, str]], **kwargs
    ) -> Iterator[str]:
        """Generate a chat completion with streaming."""
        chat_messages = []
        for m in messages:
            if isinstance(m, ChatMessage):
                chat_messages.append(m)
            elif isinstance(m, dict):
                chat_messages.append(
                    ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
                )
        request = ChatRequest(messages=chat_messages, **kwargs)
        response = self._request("POST", "/chat/stream", json=request.to_dict(), stream=True)
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
                    if not isinstance(obj, dict):
                        continue
                    payload = obj.get("data") or {}
                    if not isinstance(payload, dict):
                        continue
                    if obj.get("status") == "error" or payload.get("error"):
                        break
                    tok = payload.get("token")
                    if tok:
                        yield tok

    # ============ Models ============

    def list_models(self) -> list[ModelInfo]:
        """List available models."""
        response = self._request("GET", "/models")
        data = _unwrap_response(response.json())
        models_list = data.get("models", data) if isinstance(data, dict) else data
        if not isinstance(models_list, list):
            models_list = []
        return [ModelInfo.from_dict(m) for m in models_list]

    def load_model(self, model_id: str) -> dict[str, Any]:
        """Load a model into memory."""
        response = self._request("POST", "/models/load", json={"model_id": model_id})
        return response.json()

    def unload_model(self) -> dict[str, Any]:
        """Unload the current model."""
        response = self._request("POST", "/models/unload")
        return response.json()

    def get_current_model(self) -> dict[str, Any]:
        """Get current loaded model info."""
        response = self._request("GET", "/models/current")
        return response.json()

    def list_hf_models(self, query: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        """List available HuggingFace models."""
        params = {"limit": limit}
        if query:
            params["q"] = query
        response = self._request("GET", "/models/hf", params=params)
        return response.json().get("models", [])

    # ============ Sessions ============

    def create_session(self) -> dict[str, Any]:
        """Create a new chat session."""
        response = self._request("POST", "/chat/sessions")
        return response.json()

    def list_sessions(self) -> list[dict[str, Any]]:
        """List chat sessions."""
        response = self._request("GET", "/chat/sessions")
        data = response.json()
        return data.get("sessions", data) if isinstance(data, dict) else data

    def get_session(self, session_id: str) -> dict[str, Any]:
        """Get session details."""
        response = self._request("GET", f"/chat/sessions/{session_id}")
        return response.json()

    def delete_session(self, session_id: str) -> dict[str, Any]:
        """Delete a session."""
        response = self._request("DELETE", f"/chat/sessions/{session_id}")
        return response.json()

    def save_session_context(self, session_id: str, context: dict[str, Any]) -> dict[str, Any]:
        """Store regeneration context for a session."""
        response = self._request("POST", f"/session/{session_id}/context", json=context)
        return response.json()

    def get_session_messages(self, session_id: str) -> list[dict[str, Any]]:
        """Get stored context messages for a session."""
        response = self._request("GET", f"/session/{session_id}/messages")
        data = response.json()
        return data.get("messages", data) if isinstance(data, dict) else data

    # ============ Souls ============

    def list_souls(self) -> list[dict[str, Any]]:
        """List available souls."""
        response = self._request("GET", "/souls")
        data = _unwrap_response(response.json())
        return data.get("souls", data) if isinstance(data, dict) else data

    def get_current_soul(self) -> dict[str, Any]:
        """Get the current active soul."""
        response = self._request("GET", "/souls/current")
        return response.json()

    def switch_soul(self, name: str, checkpoint_name: str | None = None) -> dict[str, Any]:
        """Switch to a soul by name, optionally loading a checkpoint."""
        body: dict[str, Any] = {"name": name}
        if checkpoint_name:
            body["checkpoint_name"] = checkpoint_name
        response = self._request("POST", "/souls/switch", json=body)
        return response.json()

    # ============ Knowledge ============

    def list_knowledge(self) -> list[dict[str, Any]]:
        """List knowledge items."""
        response = self._request("GET", "/knowledge")
        data = _unwrap_response(response.json())
        return data.get("items", data) if isinstance(data, dict) else data

    def add_knowledge(self, content: str, topic: str | None = None) -> dict[str, Any]:
        """Add a knowledge item."""
        body: dict[str, Any] = {"content": content}
        if topic:
            body["topic"] = topic
        response = self._request("POST", "/knowledge", json=body)
        return response.json()

    def delete_knowledge(self, item_id: str) -> dict[str, Any]:
        """Delete a knowledge item."""
        response = self._request("DELETE", f"/knowledge/{item_id}")
        return response.json()

    def search_knowledge(self, query: str) -> list[dict[str, Any]]:
        """Search knowledge items."""
        response = self._request("GET", "/knowledge/search", params={"q": query})
        data = response.json()
        return data.get("results", data) if isinstance(data, dict) else data

    def get_knowledge_stats(self) -> dict[str, Any]:
        """Get knowledge base statistics."""
        response = self._request("GET", "/knowledge/stats")
        return response.json()

    def get_knowledge_topics(self) -> list[str]:
        """Get distinct knowledge topics."""
        response = self._request("GET", "/knowledge/topics")
        data = response.json()
        return data.get("topics", data) if isinstance(data, dict) else data

    def ingest_knowledge_url(self, url: str) -> dict[str, Any]:
        """Ingest a URL into the knowledge base."""
        response = self._request("POST", "/knowledge/ingest-url", json={"url": url})
        return response.json()

    # ============ Tokenizer ============

    def get_tokenizer_stats(self) -> dict[str, Any]:
        """Get tokenizer statistics."""
        response = self._request("GET", "/tokenizer/stats")
        return response.json()

    def tokenize(self, text: str) -> dict[str, Any]:
        """Tokenize text."""
        response = self._request("POST", "/tokenizer/tokenize", json={"text": text})
        return response.json()

    def train_tokenizer(self, text: str, vocab_size: int | None = None) -> dict[str, Any]:
        """Train the tokenizer on text."""
        body: dict[str, Any] = {"text": text}
        if vocab_size:
            body["vocab_size"] = vocab_size
        response = self._request("POST", "/tokenizer/train", json=body)
        return response.json()

    # ============ System ============

    def get_system_metrics(self) -> dict[str, Any]:
        """Get system metrics (CPU, memory, disk, GPU)."""
        response = self._request("GET", "/system/metrics")
        return response.json()

    def get_system_info(self) -> dict[str, Any]:
        """Get system information."""
        response = self._request("GET", "/system/info")
        return response.json()

    def get_system_disk(self) -> dict[str, Any]:
        """Get disk usage information."""
        response = self._request("GET", "/system/disk")
        return response.json()

    # ============ Companion / Personality ============

    def get_personalities(self) -> list[dict[str, Any]]:
        """Get available personalities."""
        response = self._request("GET", "/personalities")
        return response.json().get("personalities", [])

    def set_personality(self, personality: str) -> dict[str, Any]:
        """Set the current personality via companion."""
        response = self._request(
            "POST", "/companion/personality", json={"personality": personality}
        )
        return response.json()

    def get_companion_prompt(self) -> dict[str, Any]:
        """Get the current companion system prompt."""
        response = self._request("GET", "/companion/prompt")
        return response.json()

    def list_companion_presets(self) -> list[dict[str, Any]]:
        """List available companion presets."""
        response = self._request("GET", "/companion/presets")
        data = response.json()
        return data.get("presets", data) if isinstance(data, dict) else data

    # ============ Datasets ============

    def list_datasets(self) -> list[DatasetInfo]:
        """List available datasets."""
        response = self._request("GET", "/datasets")
        data = response.json()
        datasets = data.get("datasets", data) if isinstance(data, dict) else data
        return [DatasetInfo.from_dict(d) for d in datasets]

    def get_dataset(self, dataset_id: str) -> DatasetInfo:
        """Get information about a specific dataset."""
        response = self._request("GET", f"/datasets/{dataset_id}")
        return DatasetInfo.from_dict(response.json())

    def get_dataset_stats(self, dataset_id: str) -> dict[str, Any]:
        """Get dataset statistics."""
        response = self._request("GET", f"/datasets/{dataset_id}/stats")
        return response.json()

    def import_dataset_local(self, path: str, name: str | None = None) -> dict[str, Any]:
        """Import a local file or directory as a dataset."""
        body: dict[str, Any] = {"path": path}
        if name:
            body["name"] = name
        response = self._request("POST", "/datasets/import/local", json=body)
        return response.json()

    def import_dataset_github(self, repo: str, name: str | None = None) -> dict[str, Any]:
        """Import a GitHub repository as a dataset."""
        body: dict[str, Any] = {"repo": repo}
        if name:
            body["name"] = name
        response = self._request("POST", "/datasets/import/github", json=body)
        return response.json()

    def import_dataset_url(self, url: str, name: str | None = None) -> dict[str, Any]:
        """Import a URL as a dataset."""
        body: dict[str, Any] = {"url": url}
        if name:
            body["name"] = name
        response = self._request("POST", "/datasets/import/url", json=body)
        return response.json()

    # ============ Metrics ============

    def metrics(self) -> MetricsData:
        """Get API metrics."""
        response = self._request("GET", "/metrics")
        return MetricsData.from_response(_unwrap_response(response.json()))

    def metrics_prometheus(self) -> str:
        """Get metrics in Prometheus text exposition format."""
        response = self._request("GET", "/metrics/prometheus")
        return response.text

    # ============ Training ============

    def start_training(
        self,
        model_name: str,
        dataset_id: str,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        **kwargs,
    ) -> dict[str, Any]:
        """Start a training job (POST /training/start)."""
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

    def get_training_status(self, job_id: str) -> dict[str, Any]:
        """Get training job status."""
        response = self._request("GET", f"/training/jobs/{job_id}")
        return response.json()

    def list_training_jobs(self) -> list[dict[str, Any]]:
        """List all training jobs."""
        response = self._request("GET", "/training/jobs")
        return _coerce_training_jobs_list(response.json())

    def delete_training_job(self, job_id: str) -> dict[str, Any]:
        """Delete a training job."""
        response = self._request("DELETE", f"/training/jobs/{job_id}")
        return response.json()

    def stop_training(self) -> dict[str, Any]:
        """Stop the current training run."""
        response = self._request("POST", "/training/control/stop")
        return response.json()

    def pause_training(self) -> dict[str, Any]:
        """Pause the current training run."""
        response = self._request("POST", "/training/control/pause")
        return response.json()

    def resume_training(self) -> dict[str, Any]:
        """Resume the current training run."""
        response = self._request("POST", "/training/control/resume")
        return response.json()

    def get_training_recovery_stats(self) -> dict[str, Any]:
        """Get training recovery statistics."""
        response = self._request("GET", "/recovery/stats")
        return response.json()

    def abandon_recovery(self, job_id: str) -> dict[str, Any]:
        """Abandon a recoverable training job."""
        response = self._request("DELETE", f"/recovery/abandon/{job_id}")
        return response.json()

    # ============ Auto-Train ============

    def start_auto_train(self, config: dict[str, Any]) -> dict[str, Any]:
        """Start auto-training."""
        response = self._request("POST", "/training/start", json=config)
        return response.json()

    def stop_auto_train(self) -> dict[str, Any]:
        """Stop auto-training."""
        response = self._request("POST", "/training/stop")
        return response.json()

    def list_auto_train_checkpoints(self) -> list[dict[str, Any]]:
        """List auto-training checkpoints."""
        response = self._request("GET", "/training/checkpoints")
        data = response.json()
        return data.get("checkpoints", data) if isinstance(data, dict) else data

    def delete_auto_train_checkpoint(self, name: str) -> dict[str, Any]:
        """Delete an auto-training checkpoint."""
        response = self._request("DELETE", f"/training/checkpoints/{name}")
        return response.json()

    def load_auto_train_checkpoint(self, name: str) -> dict[str, Any]:
        """Load an auto-training checkpoint."""
        response = self._request("POST", f"/training/checkpoints/{name}/load")
        return response.json()

    # ============ Feedback ============

    def record_feedback(
        self, session_id: str, message_id: str, score: int, tags: list[str] | None = None
    ) -> dict[str, Any]:
        """Record feedback for a message (triggers workflow)."""
        body: dict[str, Any] = {"session_id": session_id, "message_id": message_id, "score": score}
        if tags:
            body["tags"] = tags
        response = self._request("POST", "/feedback/workflow-record", json=body)
        return response.json()

    def get_feedback_stats(self) -> dict[str, Any]:
        """Get feedback statistics."""
        response = self._request("GET", "/feedback/stats/summary")
        return response.json()

    # ============ Workflow ============

    def get_workflow_status(self) -> dict[str, Any]:
        """Get the feedback workflow status."""
        response = self._request("GET", "/workflow/status")
        return response.json()

    # ============ Experiments ============

    def create_experiment(self, name: str, description: str = "", **kwargs) -> dict[str, Any]:
        """Create a new experiment."""
        payload = {"name": name, "description": description, **kwargs}
        response = self._request("POST", "/experiments", json=payload)
        return response.json()

    def list_experiments(self) -> list[dict[str, Any]]:
        """List all experiments."""
        response = self._request("GET", "/experiments")
        return response.json().get("experiments", [])

    def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Get experiment details."""
        response = self._request("GET", f"/experiments/{experiment_id}")
        return response.json()

    def log_metric(
        self, experiment_id: str, metric_name: str, value: float, step: int | None = None
    ) -> dict[str, Any]:
        """Log a metric to an experiment."""
        payload: dict[str, Any] = {"metric": metric_name, "value": value}
        if step is not None:
            payload["step"] = step
        response = self._request("POST", f"/experiments/{experiment_id}/log_metric", json=payload)
        return response.json()

    def log_param(self, experiment_id: str, param_name: str, value: Any) -> dict[str, Any]:
        """Log a parameter to an experiment."""
        payload = {"param": param_name, "value": value}
        response = self._request("POST", f"/experiments/{experiment_id}/log_param", json=payload)
        return response.json()

    # ============ Rate Limit ============

    def get_rate_limit_status(self) -> dict[str, Any]:
        """Get rate limit status."""
        response = self._request("GET", "/rate-limit/status")
        return response.json()

    def check_rate_limit(self) -> dict[str, Any]:
        """Check if a request would be rate limited."""
        response = self._request("GET", "/rate-limit/check")
        return response.json()

    # ============ Security ============

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Get the security audit log."""
        response = self._request("GET", "/security/audit")
        data = _unwrap_response(response.json())
        return data if isinstance(data, list) else data.get("logs", data)

    def get_security_keys(self) -> list[dict[str, Any]]:
        """List registered security/API keys."""
        response = self._request("GET", "/security/keys")
        data = _unwrap_response(response.json())
        return data if isinstance(data, list) else data.get("keys", data)

    def create_security_key(
        self, name: str, scopes: list[str] | None = None, expires_in_days: int | None = None
    ) -> dict[str, Any]:
        """Create a new API key."""
        body: dict[str, Any] = {"name": name}
        if scopes:
            body["scopes"] = scopes
        if expires_in_days is not None:
            body["expires_in_days"] = expires_in_days
        response = self._request("POST", "/security/keys", json=body)
        return _unwrap_response(response.json())

    def get_security_key(self, key_id: str) -> dict[str, Any]:
        """Get a specific API key by ID."""
        response = self._request("GET", f"/security/keys/{key_id}")
        return _unwrap_response(response.json())

    def delete_security_key(self, key_id: str) -> dict[str, Any]:
        """Delete/revoke an API key."""
        response = self._request("DELETE", f"/security/keys/{key_id}")
        return _unwrap_response(response.json())

    def rotate_security_key(self, key_id: str) -> dict[str, Any]:
        """Rotate an API key (generates new secret, invalidates old)."""
        response = self._request("POST", f"/security/keys/{key_id}/rotate")
        return _unwrap_response(response.json())

    def validate_security_key(self, key: str) -> dict[str, Any]:
        """Validate an API key."""
        response = self._request("POST", "/security/keys/validate", json={"key": key})
        return _unwrap_response(response.json())

    # ============ Registry ============

    def list_tenants(self) -> list[dict[str, Any]]:
        """List all tenants."""
        response = self._request("GET", "/tenants")
        data = _unwrap_response(response.json())
        return data if isinstance(data, list) else data.get("tenants", data)

    def get_tenant(self, tenant_id: str) -> dict[str, Any]:
        """Get a tenant by ID."""
        response = self._request("GET", f"/tenants/{tenant_id}")
        return _unwrap_response(response.json())

    def create_tenant(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Create a new tenant."""
        body = {"name": name, **kwargs}
        response = self._request("POST", "/tenants", json=body)
        return _unwrap_response(response.json())

    def update_tenant(self, tenant_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update a tenant."""
        response = self._request("PUT", f"/tenants/{tenant_id}", json=kwargs)
        return _unwrap_response(response.json())

    def delete_tenant(self, tenant_id: str) -> dict[str, Any]:
        """Delete a tenant."""
        response = self._request("DELETE", f"/tenants/{tenant_id}")
        return _unwrap_response(response.json())

    def get_tenant_stats(self, tenant_id: str) -> dict[str, Any]:
        """Get tenant statistics."""
        response = self._request("GET", f"/tenants/{tenant_id}/stats")
        return _unwrap_response(response.json())

    # ============ Profiles ============

    def list_profiles(self) -> list[dict[str, Any]]:
        """List all available profiles."""
        response = self._request("GET", "/profiles")
        data = _unwrap_response(response.json())
        return data if isinstance(data, list) else data.get("profiles", data)

    def get_profile(self, profile_id: str) -> dict[str, Any]:
        """Get a profile by ID."""
        response = self._request("GET", f"/profiles/{profile_id}")
        return _unwrap_response(response.json())

    def apply_profile(self, profile_id: str) -> dict[str, Any]:
        """Apply a profile to current settings."""
        response = self._request("POST", "/profiles/apply", json={"profile_id": profile_id})
        return _unwrap_response(response.json())

    def get_active_profile(self) -> dict[str, Any]:
        """Get the currently active profile."""
        response = self._request("GET", "/profiles/active")
        return _unwrap_response(response.json())

    def recommend_profile(self) -> dict[str, Any]:
        """Get profile recommendation based on usage patterns."""
        response = self._request("GET", "/profiles/recommend")
        return _unwrap_response(response.json())

    # ============ Workspaces ============

    def list_workspaces(self) -> list[dict[str, Any]]:
        """List all workspaces."""
        response = self._request("GET", "/workspaces")
        data = _unwrap_response(response.json())
        return data if isinstance(data, list) else data.get("workspaces", data)

    def get_workspace(self, workspace_id: str) -> dict[str, Any]:
        """Get a workspace by ID."""
        response = self._request("GET", f"/workspaces/{workspace_id}")
        return _unwrap_response(response.json())

    def create_workspace(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Create a new workspace."""
        body = {"name": name, **kwargs}
        response = self._request("POST", "/workspaces", json=body)
        return _unwrap_response(response.json())

    def add_workspace_member(
        self, workspace_id: str, user_id: str, role: str = "member"
    ) -> dict[str, Any]:
        """Add a member to a workspace."""
        body = {"user_id": user_id, "role": role}
        response = self._request("POST", f"/workspaces/{workspace_id}/members", json=body)
        return _unwrap_response(response.json())

    def list_registry_models(self) -> list[dict[str, Any]]:
        """List models registered in the live model registry."""
        response = self._request("GET", "/registry/models")
        data = _unwrap_response(response.json())
        return data.get("models", data)

    def get_registry_model(self, model_id: str) -> dict[str, Any]:
        """Get a single registered model's details."""
        response = self._request("GET", f"/registry/models/{model_id}")
        data = _unwrap_response(response.json())
        return data if isinstance(data, dict) else data.get("data", data)

    def get_registry_best(self) -> dict[str, Any]:
        """Get best performing model by live registry metrics."""
        response = self._request("GET", "/registry/best")
        return _unwrap_response(response.json())

    def get_registry_stats(self) -> dict[str, Any]:
        """Get live model registry statistics."""
        response = self._request("GET", "/registry/stats")
        return _unwrap_response(response.json())

    # ============ Auto-train ============

    def get_auto_train_status(self) -> dict[str, Any]:
        """Get auto-trainer status and configuration."""
        response = self._request("GET", "/settings/training/auto-train/status")
        return _unwrap_response(response.json())

    def update_auto_train_config(
        self,
        threshold: int | None = None,
        interval_s: int | None = None,
    ) -> dict[str, Any]:
        """Update auto-trainer configuration at runtime."""
        params: dict[str, Any] = {}
        if threshold is not None:
            params["threshold"] = threshold
        if interval_s is not None:
            params["interval_s"] = interval_s
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"/settings/training/auto-train/config{f'?{qs}' if qs else ''}"
        response = self._request("PATCH", url)
        return _unwrap_response(response.json())

    def get_training_analytics(self) -> dict[str, Any]:
        """Get aggregated training analytics for charts and summaries."""
        response = self._request("GET", "/settings/training/analytics")
        return _unwrap_response(response.json())

    # ============ Docstore ============

    def list_docstore_docs(self, collection: str) -> list[dict[str, Any]]:
        """List all documents in a docstore collection."""
        response = self._request("GET", f"/docstore/{collection}")
        return _unwrap_response(response.json())

    def get_docstore_doc(self, collection: str, doc_id: str) -> dict[str, Any]:
        """Get a single document from a docstore collection."""
        response = self._request("GET", f"/docstore/{collection}/{doc_id}")
        return _unwrap_response(response.json())

    def put_docstore_doc(
        self, collection: str, doc_id: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Put (create/update) a document in a docstore collection."""
        response = self._request("PUT", f"/docstore/{collection}/{doc_id}", json=data)
        return _unwrap_response(response.json())

    def patch_docstore_doc(
        self, collection: str, doc_id: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Patch (partial update) a document in a docstore collection."""
        response = self._request("PATCH", f"/docstore/{collection}/{doc_id}", json=data)
        return _unwrap_response(response.json())

    def delete_docstore_doc(self, collection: str, doc_id: str) -> dict[str, Any]:
        """Delete a document from a docstore collection."""
        response = self._request("DELETE", f"/docstore/{collection}/{doc_id}")
        return _unwrap_response(response.json())

    def clear_docstore_collection(self, collection: str) -> dict[str, Any]:
        """Clear all documents from a docstore collection."""
        response = self._request("DELETE", f"/docstore/{collection}")
        return _unwrap_response(response.json())

    def bulk_put_docstore(self, collection: str, docs: list[dict[str, Any]]) -> dict[str, Any]:
        """Bulk put documents into a docstore collection."""
        response = self._request("POST", f"/docstore/{collection}/bulk", json=docs)
        return _unwrap_response(response.json())

    # ============ Collections ============

    def list_collections(self) -> list[dict[str, Any]]:
        """List all collections/pipelines."""
        response = self._request("GET", "/collections")
        return _unwrap_response(response.json())

    def get_collection(self, collection_id: str) -> dict[str, Any]:
        """Get a single collection/pipeline."""
        response = self._request("GET", f"/collections/{collection_id}")
        return _unwrap_response(response.json())

    def create_collection(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Create a new collection/pipeline."""
        response = self._request("POST", "/collections/create", json={"name": name, **kwargs})
        return _unwrap_response(response.json())

    def delete_collection(self, collection_id: str) -> dict[str, Any]:
        """Delete a collection/pipeline."""
        response = self._request("DELETE", f"/collections/{collection_id}")
        return _unwrap_response(response.json())

    def run_collection(self, collection_id: str, **kwargs: Any) -> dict[str, Any]:
        """Run a collection/pipeline."""
        response = self._request(
            "POST", "/collections/run", json={"pipeline_id": collection_id, **kwargs}
        )
        return _unwrap_response(response.json())

    def collect_from_collection(self, collection_id: str, **kwargs: Any) -> dict[str, Any]:
        """Collect data from a collection/pipeline."""
        response = self._request("POST", f"/collections/{collection_id}/collect", json=kwargs)
        return _unwrap_response(response.json())

    def get_collection_records(self, collection_id: str) -> list[dict[str, Any]]:
        """Get records from a collection/pipeline."""
        response = self._request("GET", f"/collections/{collection_id}/records")
        return _unwrap_response(response.json())

    def get_collection_stats(self) -> dict[str, Any]:
        """Get collection/pipeline stats."""
        response = self._request("GET", "/collections/stats")
        return _unwrap_response(response.json())

    # ============ Benchmark ============

    def run_benchmark(self, config: dict[str, Any]) -> dict[str, Any]:
        """Run a benchmark."""
        response = self._request("POST", "/benchmark/run", json=config)
        return response.json()

    def get_benchmark_metrics(self) -> list[dict[str, Any]]:
        """Get benchmark metrics."""
        response = self._request("GET", "/benchmark/metrics")
        data = response.json()
        return data if isinstance(data, list) else data.get("metrics", data)

    def get_benchmark_stats(self) -> dict[str, Any]:
        """Get benchmark statistics."""
        response = self._request("GET", "/benchmark/stats")
        return response.json()

    # ============ Settings ============

    def get_settings(self) -> dict[str, Any]:
        """Get all user settings."""
        response = self._request("GET", "/settings")
        return response.json()

    def get_generation_settings(self) -> dict[str, Any]:
        """Get generation settings."""
        response = self._request("GET", "/settings/generation")
        return response.json()

    def update_generation_settings(self, **kwargs) -> dict[str, Any]:
        """Update generation settings."""
        response = self._request("PATCH", "/settings/generation", json=kwargs)
        return response.json()

    def get_voice_settings(self) -> dict[str, Any]:
        """Get voice settings."""
        response = self._request("GET", "/settings/voice")
        return response.json()

    def update_voice_settings(self, **kwargs) -> dict[str, Any]:
        """Update voice settings."""
        response = self._request("PATCH", "/settings/voice", json=kwargs)
        return response.json()

    def reset_settings(self) -> dict[str, Any]:
        """Reset all settings to defaults."""
        response = self._request("POST", "/settings/reset")
        return response.json()

    def get_adaptive_insights(self) -> dict[str, Any]:
        """Get adaptive training insights."""
        response = self._request("GET", "/settings/adaptive/insights")
        return response.json()

    def export_training_history(self, format: str = "json", limit: int = 0) -> dict[str, Any]:
        """Export training history as JSON or CSV."""
        response = self._request(
            "GET", f"/settings/training/history/export?format={format}&limit={limit}"
        )
        return response.json()

    def generate_model_card(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Generate a model card from training metadata."""
        params = {"name": name, **kwargs}
        response = self._request("POST", "/settings/model-card", json=params)
        return response.json()

    def get_dashboard_summary(self) -> dict[str, Any]:
        """Get quick system summary from dashboard."""
        response = self._request("GET", "/dashboard/summary")
        return response.json()

    def compare_training_runs(self, run_a: str, run_b: str) -> dict[str, Any]:
        """Compare two training runs side by side."""
        response = self._request("GET", f"/settings/training/compare?run_a={run_a}&run_b={run_b}")
        return response.json()

    def get_batch_training_status(self) -> dict[str, Any]:
        """Get status of all training jobs."""
        response = self._request("GET", "/settings/training/batch-status")
        return response.json()

    def list_training_presets(self) -> dict[str, Any]:
        """List all available training presets."""
        response = self._request("GET", "/settings/training/presets")
        return response.json()

    def get_training_preset(self, name: str) -> dict[str, Any]:
        """Get a specific training preset."""
        response = self._request("GET", f"/settings/training/presets/{name}")
        return response.json()

    def apply_training_preset(self, name: str) -> dict[str, Any]:
        """Apply a training preset to current settings."""
        response = self._request("POST", f"/settings/training/presets/{name}/apply")
        return response.json()

    def get_training_run(self, run_id: str) -> dict[str, Any]:
        """Get a single training run by ID."""
        response = self._request("GET", f"/settings/training/runs/{run_id}")
        return response.json()

    def delete_training_run(self, run_id: str) -> dict[str, Any]:
        """Delete a specific training run by ID."""
        response = self._request("DELETE", f"/settings/training/runs/{run_id}")
        return response.json()

    def filter_training_runs(
        self,
        model: str | None = None,
        method: str | None = None,
        converged: bool | None = None,
        min_quality: float | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Filter training runs by model, method, convergence, or quality."""
        params = {"limit": limit}
        if model:
            params["model"] = model
        if method:
            params["method"] = method
        if converged is not None:
            params["converged"] = str(converged).lower()
        if min_quality is not None:
            params["min_quality"] = min_quality
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        response = self._request("GET", f"/settings/training/runs?{qs}")
        return response.json()

    def clear_training_history(self) -> dict[str, Any]:
        """Clear all training history."""
        response = self._request("POST", "/settings/training/history/clear")
        return response.json()

    def add_run_tag(self, run_id: str, tag: str) -> dict[str, Any]:
        """Add a tag to a training run."""
        response = self._request("POST", f"/settings/training/runs/{run_id}/tags?tag={tag}")
        return response.json()

    def remove_run_tag(self, run_id: str, tag: str) -> dict[str, Any]:
        """Remove a tag from a training run."""
        response = self._request("DELETE", f"/settings/training/runs/{run_id}/tags/{tag}")
        return response.json()

    def set_run_notes(self, run_id: str, notes: str) -> dict[str, Any]:
        """Set notes on a training run."""
        response = self._request("PUT", f"/settings/training/runs/{run_id}/notes?notes={notes}")
        return response.json()

    def get_all_tags(self) -> dict[str, Any]:
        """Get all unique tags across all training runs."""
        response = self._request("GET", "/settings/training/tags")
        return response.json()

    def get_runs_by_tag(self, tag: str) -> dict[str, Any]:
        """Get all training runs with a specific tag."""
        response = self._request("GET", f"/settings/training/tags/{tag}")
        return response.json()

    def export_training_run(self, run_id: str, format: str = "json") -> dict[str, Any]:
        """Export a single training run as JSON or YAML."""
        response = self._request("GET", f"/settings/training/runs/{run_id}/export?format={format}")
        return response.json()

    def toggle_bookmark(self, run_id: str) -> dict[str, Any]:
        """Toggle bookmark status on a training run."""
        response = self._request("POST", f"/settings/training/runs/{run_id}/bookmark")
        return response.json()

    def get_bookmarked_runs(self) -> dict[str, Any]:
        """Get all bookmarked training runs."""
        response = self._request("GET", "/settings/training/bookmarks")
        return response.json()

    def duplicate_training_run(self, run_id: str, new_run_id: str = "") -> dict[str, Any]:
        """Duplicate a training run with a new ID."""
        params = f"?new_run_id={new_run_id}" if new_run_id else ""
        response = self._request("POST", f"/settings/training/runs/{run_id}/duplicate{params}")
        return response.json()

    def bulk_delete_runs(self, run_ids: list[str]) -> dict[str, Any]:
        """Delete multiple training runs."""
        ids_str = ",".join(run_ids)
        response = self._request("POST", f"/settings/training/runs/bulk/delete?run_ids={ids_str}")
        return response.json()

    def bulk_add_tag(self, run_ids: list[str], tag: str) -> dict[str, Any]:
        """Add a tag to multiple training runs."""
        ids_str = ",".join(run_ids)
        response = self._request(
            "POST", f"/settings/training/runs/bulk/tag?run_ids={ids_str}&tag={tag}"
        )
        return response.json()

    def bulk_bookmark(self, run_ids: list[str], bookmarked: bool = True) -> dict[str, Any]:
        """Set bookmark status on multiple training runs."""
        ids_str = ",".join(run_ids)
        response = self._request(
            "POST",
            f"/settings/training/runs/bulk/bookmark?run_ids={ids_str}&bookmarked={str(bookmarked).lower()}",
        )
        return response.json()

    # ============ VQA ============

    def ask_question(self, image_path: str, question: str) -> dict[str, Any]:
        """Ask a question about an image (VQA)."""
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, "image/png")}
            data = {"question": question}
            response = self._request("POST", "/multimodal/ask", files=files, data=data)
            return response.json()

    def detect_objects(self, image_path: str) -> dict[str, Any]:
        """Detect objects in an image."""
        with open(image_path, "rb") as f:
            files = {"file": (image_path, f, "image/png")}
            response = self._request("POST", "/multimodal/detect", files=files)
            return response.json()

    def analyze_pdf(
        self, pdf_path: str, question: str = "Analyze this document."
    ) -> dict[str, Any]:
        """Analyze a PDF document."""
        with open(pdf_path, "rb") as f:
            files = {"file": (pdf_path, f, "application/pdf")}
            data = {"question": question}
            response = self._request("POST", "/multimodal/pdf/upload", files=files, data=data)
            return response.json()

    # ============ Context Manager ============

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._session.close()

    # ============ Convenience Methods ============

    def quick_generate(self, prompt: str) -> str:
        """Quick generation with default settings."""
        return self.generate(prompt).generated_text

    def quick_chat(self, user_message: str) -> str:
        """Quick chat with a single user message."""
        result = self.chat([ChatMessage.user(user_message)])
        return result.message.content

    # ============ OpenWebUI Integration ============

    def openwebui_datasets(self) -> list:
        """List datasets for OpenWebUI panel."""
        return self._request("GET", "/openwebui/datasets")["datasets"]

    def openwebui_checkpoints(self) -> list:
        """List checkpoints for OpenWebUI panel."""
        return self._request("GET", "/openwebui/checkpoints")["checkpoints"]

    def openwebui_start_training(self, dataset_id: str, method: str = "finetune") -> dict:
        """Start training from OpenWebUI panel."""
        return self._request(
            "POST", "/openwebui/training/start", json={"dataset_id": dataset_id, "method": method}
        )

    def openwebui_stop_training(self) -> dict:
        """Stop training from OpenWebUI panel."""
        return self._request("POST", "/openwebui/training/stop")

    def openwebui_training_status(self) -> dict:
        """Get training status for OpenWebUI panel."""
        return self._request("GET", "/openwebui/training/status")

    # ============ Cloud Training ============

    def cloud_training_jobs(self, limit: int = 10) -> list:
        """List cloud training jobs."""
        return self._request("GET", f"/cloud-training/jobs?limit={limit}")["jobs"]

    def cloud_training_submit(self, provider: str = "local", dataset_id: str = "") -> dict:
        """Submit a cloud training job."""
        return self._request(
            "POST", "/cloud-training/submit", json={"provider": provider, "dataset_id": dataset_id}
        )

    def cloud_training_status(self, job_id: str) -> dict:
        """Get cloud training job status."""
        return self._request("GET", f"/cloud-training/{job_id}/status")

    def cloud_training_cancel(self, job_id: str) -> dict:
        """Cancel a cloud training job."""
        return self._request("POST", f"/cloud-training/{job_id}/cancel")

    # ============ Plugins ============

    def plugins_list(self) -> list:
        """List loaded plugins."""
        return self._request("GET", "/plugins")["plugins"]

    def plugins_enable(self, plugin_name: str) -> dict:
        """Enable a plugin."""
        return self._request("POST", f"/plugins/{plugin_name}/enable")

    def plugins_disable(self, plugin_name: str) -> dict:
        """Disable a plugin."""
        return self._request("POST", f"/plugins/{plugin_name}/disable")

    def plugins_reload(self) -> dict:
        """Reload plugins."""
        return self._request("POST", "/plugins/reload")


class SimpleTracker:
    """Simple context manager for tracking metrics."""

    def __init__(self, client: SloughGPTClient, name: str):
        self._client = client
        self._name = name
        self._step = 0

    def log(self, metric: str, value: float):
        pass

    def next_step(self):
        self._step += 1

    def finish(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


class AsyncSloughGPTClient:
    """
    Async Python client for the SloughGPT API.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: int = 30,
        verify_ssl: bool = True,
        headers: dict[str, str] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._headers = headers or {}
        if api_key:
            self._headers["X-API-Key"] = api_key

    async def _request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        import httpx

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        kwargs.setdefault("timeout", self.timeout)
        extra_headers = kwargs.pop("extra_headers", None)
        merged = {**self._headers, **(extra_headers or {})}
        async with httpx.AsyncClient(verify=self.verify_ssl, headers=merged) as client:
            response = await client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()

    async def get_token(self, api_key: str) -> dict[str, Any]:
        """Exchange an API key for a JWT access token (``POST /auth/token``)."""
        data = await self._request("POST", "/auth/token", json={"api_key": api_key})
        return _unwrap_response(data)

    async def health(self) -> HealthStatus:
        data = await self._request("GET", "/health")
        return HealthStatus.from_response(_unwrap_response(data))

    async def generate(self, prompt: str, **kwargs) -> GenerationResult:
        from .models import GenerateRequest

        request = GenerateRequest(prompt=prompt, **kwargs)
        data = await self._request("POST", "/inference/generate", json=request.to_dict())
        return GenerationResult.from_response(data, prompt)

    async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResult:
        body = {
            "messages": [m.to_dict() if isinstance(m, ChatMessage) else m for m in messages],
            **kwargs,
        }
        data = await self._request("POST", "/chat", json=body)
        err = data.get("error")
        if isinstance(err, str) and err.strip() and not str(data.get("message") or "").strip():
            from .exceptions import SloughGPTError

            raise SloughGPTError(err)
        return ChatResult.from_response(data)

    async def list_models(self) -> list[ModelInfo]:
        data = await self._request("GET", "/models")
        data = _unwrap_response(data)
        models_list = data.get("models", data) if isinstance(data, dict) else data
        if not isinstance(models_list, list):
            models_list = []
        return [ModelInfo.from_dict(m) for m in models_list]

    async def list_souls(self) -> list[dict[str, Any]]:
        data = await self._request("GET", "/souls")
        data = _unwrap_response(data)
        return data.get("souls", data) if isinstance(data, dict) else data

    async def switch_soul(self, name: str, checkpoint_name: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"name": name}
        if checkpoint_name:
            body["checkpoint_name"] = checkpoint_name
        return await self._request("POST", "/souls/switch", json=body)

    async def list_knowledge(self) -> list[dict[str, Any]]:
        data = await self._request("GET", "/knowledge")
        data = _unwrap_response(data)
        return data.get("items", data) if isinstance(data, dict) else data

    async def add_knowledge(self, content: str, topic: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"content": content}
        if topic:
            body["topic"] = topic
        return await self._request("POST", "/knowledge", json=body)

    async def search_knowledge(self, query: str) -> list[dict[str, Any]]:
        data = await self._request("GET", "/knowledge/search", params={"q": query})
        return data.get("results", data) if isinstance(data, dict) else data

    async def get_system_metrics(self) -> dict[str, Any]:
        return await self._request("GET", "/system/metrics")

    async def metrics(self) -> MetricsData:
        data = await self._request("GET", "/metrics")
        return MetricsData.from_response(_unwrap_response(data))

    async def get_workflow_status(self) -> dict[str, Any]:
        return await self._request("GET", "/workflow/status")

    async def record_feedback(
        self, session_id: str, message_id: str, score: int, tags: list[str] | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"session_id": session_id, "message_id": message_id, "score": score}
        if tags:
            body["tags"] = tags
        return await self._request("POST", "/feedback/workflow-record", json=body)

    async def start_training(
        self,
        model_name: str,
        dataset_id: str,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        **kwargs: Any,
    ) -> dict[str, Any]:
        payload = _build_training_start_payload(
            model_name,
            dataset_id,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs,
        )
        return await self._request("POST", "/training/start", json=payload)

    async def get_training_status(self, job_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/training/jobs/{job_id}")

    async def list_training_jobs(self) -> list[dict[str, Any]]:
        data = await self._request("GET", "/training/jobs")
        return _coerce_training_jobs_list(data)

    async def create_experiment(self, name: str, description: str = "", **kwargs) -> dict[str, Any]:
        payload = {"name": name, "description": description, **kwargs}
        return await self._request("POST", "/experiments", json=payload)

    async def list_experiments(self) -> list[dict[str, Any]]:
        data = await self._request("GET", "/experiments")
        return data.get("experiments", [])

    async def get_experiment(self, experiment_id: str) -> dict[str, Any]:
        return await self._request("GET", f"/experiments/{experiment_id}")

    async def log_metric(
        self, experiment_id: str, metric_name: str, value: float, **kwargs
    ) -> dict[str, Any]:
        payload = {"metric": metric_name, "value": value, **kwargs}
        return await self._request("POST", f"/experiments/{experiment_id}/log_metric", json=payload)

    async def get_tokenizer_stats(self) -> dict[str, Any]:
        return await self._request("GET", "/tokenizer/stats")

    async def list_auto_train_checkpoints(self) -> list[dict[str, Any]]:
        data = await self._request("GET", "/training/checkpoints")
        return data.get("checkpoints", data) if isinstance(data, dict) else data

    async def get_security_keys(self) -> list[dict[str, Any]]:
        """List registered security/API keys."""
        data = _unwrap_response(await self._request("GET", "/security/keys"))
        return data if isinstance(data, list) else data.get("keys", data)

    async def create_security_key(
        self, name: str, scopes: list[str] | None = None, expires_in_days: int | None = None
    ) -> dict[str, Any]:
        """Create a new API key."""
        body: dict[str, Any] = {"name": name}
        if scopes:
            body["scopes"] = scopes
        if expires_in_days is not None:
            body["expires_in_days"] = expires_in_days
        return _unwrap_response(await self._request("POST", "/security/keys", json=body))

    async def get_security_key(self, key_id: str) -> dict[str, Any]:
        """Get a specific API key by ID."""
        return _unwrap_response(await self._request("GET", f"/security/keys/{key_id}"))

    async def delete_security_key(self, key_id: str) -> dict[str, Any]:
        """Delete/revoke an API key."""
        return _unwrap_response(await self._request("DELETE", f"/security/keys/{key_id}"))

    async def rotate_security_key(self, key_id: str) -> dict[str, Any]:
        """Rotate an API key (generates new secret, invalidates old)."""
        return _unwrap_response(await self._request("POST", f"/security/keys/{key_id}/rotate"))

    async def validate_security_key(self, key: str) -> dict[str, Any]:
        """Validate an API key."""
        return _unwrap_response(
            await self._request("POST", "/security/keys/validate", json={"key": key})
        )

    async def list_registry_models(self) -> list[dict[str, Any]]:
        """List models registered in the live model registry."""
        data = _unwrap_response(await self._request("GET", "/registry/models"))
        return data.get("models", data)

    async def get_registry_model(self, model_id: str) -> dict[str, Any]:
        """Get a single registered model's details."""
        data = _unwrap_response(await self._request("GET", f"/registry/models/{model_id}"))
        return data if isinstance(data, dict) else data.get("data", data)

    async def get_registry_best(self) -> dict[str, Any]:
        """Get best performing model by live registry metrics."""
        return _unwrap_response(await self._request("GET", "/registry/best"))

    async def get_registry_stats(self) -> dict[str, Any]:
        """Get live model registry statistics."""
        return _unwrap_response(await self._request("GET", "/registry/stats"))

    # ── Tenants ─────────────────────────────────────────────

    async def list_tenants(self) -> list[dict[str, Any]]:
        """List all tenants."""
        data = _unwrap_response(await self._request("GET", "/tenants"))
        return data if isinstance(data, list) else data.get("tenants", data)

    async def get_tenant(self, tenant_id: str) -> dict[str, Any]:
        """Get a tenant by ID."""
        return _unwrap_response(await self._request("GET", f"/tenants/{tenant_id}"))

    async def create_tenant(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Create a new tenant."""
        body = {"name": name, **kwargs}
        return _unwrap_response(await self._request("POST", "/tenants", json=body))

    async def update_tenant(self, tenant_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update a tenant."""
        return _unwrap_response(await self._request("PUT", f"/tenants/{tenant_id}", json=kwargs))

    async def delete_tenant(self, tenant_id: str) -> dict[str, Any]:
        """Delete a tenant."""
        return _unwrap_response(await self._request("DELETE", f"/tenants/{tenant_id}"))

    async def get_tenant_stats(self, tenant_id: str) -> dict[str, Any]:
        """Get tenant statistics."""
        return _unwrap_response(await self._request("GET", f"/tenants/{tenant_id}/stats"))

    # ── Profiles ────────────────────────────────────────────

    async def list_profiles(self) -> list[dict[str, Any]]:
        """List all available profiles."""
        data = _unwrap_response(await self._request("GET", "/profiles"))
        return data if isinstance(data, list) else data.get("profiles", data)

    async def get_profile(self, profile_id: str) -> dict[str, Any]:
        """Get a profile by ID."""
        return _unwrap_response(await self._request("GET", f"/profiles/{profile_id}"))

    async def apply_profile(self, profile_id: str) -> dict[str, Any]:
        """Apply a profile to current settings."""
        return _unwrap_response(
            await self._request("POST", "/profiles/apply", json={"profile_id": profile_id})
        )

    async def get_active_profile(self) -> dict[str, Any]:
        """Get the currently active profile."""
        return _unwrap_response(await self._request("GET", "/profiles/active"))

    async def recommend_profile(self) -> dict[str, Any]:
        """Get profile recommendation based on usage patterns."""
        return _unwrap_response(await self._request("GET", "/profiles/recommend"))

    # ── Workspaces ──────────────────────────────────────────

    async def list_workspaces(self) -> list[dict[str, Any]]:
        """List all workspaces."""
        data = _unwrap_response(await self._request("GET", "/workspaces"))
        return data if isinstance(data, list) else data.get("workspaces", data)

    async def get_workspace(self, workspace_id: str) -> dict[str, Any]:
        """Get a workspace by ID."""
        return _unwrap_response(await self._request("GET", f"/workspaces/{workspace_id}"))

    async def create_workspace(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Create a new workspace."""
        body = {"name": name, **kwargs}
        return _unwrap_response(await self._request("POST", "/workspaces", json=body))

    async def add_workspace_member(
        self, workspace_id: str, user_id: str, role: str = "member"
    ) -> dict[str, Any]:
        """Add a member to a workspace."""
        body = {"user_id": user_id, "role": role}
        return _unwrap_response(
            await self._request("POST", f"/workspaces/{workspace_id}/members", json=body)
        )

    # ── Auto-train ─────────────────────────────────────────────

    async def get_auto_train_status(self) -> dict[str, Any]:
        """Get auto-trainer status and configuration."""
        return _unwrap_response(await self._request("GET", "/settings/training/auto-train/status"))

    async def update_auto_train_config(
        self,
        threshold: int | None = None,
        interval_s: int | None = None,
    ) -> dict[str, Any]:
        """Update auto-trainer configuration at runtime."""
        params: dict[str, Any] = {}
        if threshold is not None:
            params["threshold"] = threshold
        if interval_s is not None:
            params["interval_s"] = interval_s
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"/settings/training/auto-train/config{f'?{qs}' if qs else ''}"
        return _unwrap_response(await self._request("PATCH", url))

    async def get_training_analytics(self) -> dict[str, Any]:
        """Get aggregated training analytics for charts and summaries."""
        return _unwrap_response(await self._request("GET", "/settings/training/analytics"))

    # ── Docstore ────────────────────────────────────────────────────────

    async def list_docstore_docs(self, collection: str) -> list[dict[str, Any]]:
        """List all documents in a docstore collection."""
        return _unwrap_response(await self._request("GET", f"/docstore/{collection}"))

    async def get_docstore_doc(self, collection: str, doc_id: str) -> dict[str, Any]:
        """Get a single document from a docstore collection."""
        return _unwrap_response(await self._request("GET", f"/docstore/{collection}/{doc_id}"))

    async def put_docstore_doc(
        self, collection: str, doc_id: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Put (create/update) a document in a docstore collection."""
        return _unwrap_response(
            await self._request("PUT", f"/docstore/{collection}/{doc_id}", json=data)
        )

    async def patch_docstore_doc(
        self, collection: str, doc_id: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Patch (partial update) a document in a docstore collection."""
        return _unwrap_response(
            await self._request("PATCH", f"/docstore/{collection}/{doc_id}", json=data)
        )

    async def delete_docstore_doc(self, collection: str, doc_id: str) -> dict[str, Any]:
        """Delete a document from a docstore collection."""
        return _unwrap_response(await self._request("DELETE", f"/docstore/{collection}/{doc_id}"))

    async def clear_docstore_collection(self, collection: str) -> dict[str, Any]:
        """Clear all documents from a docstore collection."""
        return _unwrap_response(await self._request("DELETE", f"/docstore/{collection}"))

    async def bulk_put_docstore(
        self, collection: str, docs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Bulk put documents into a docstore collection."""
        return _unwrap_response(
            await self._request("POST", f"/docstore/{collection}/bulk", json=docs)
        )

    # ── Collections ─────────────────────────────────────────────────────

    async def list_collections(self) -> list[dict[str, Any]]:
        """List all collections/pipelines."""
        return _unwrap_response(await self._request("GET", "/collections"))

    async def get_collection(self, collection_id: str) -> dict[str, Any]:
        """Get a single collection/pipeline."""
        return _unwrap_response(await self._request("GET", f"/collections/{collection_id}"))

    async def create_collection(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Create a new collection/pipeline."""
        return _unwrap_response(
            await self._request("POST", "/collections/create", json={"name": name, **kwargs})
        )

    async def delete_collection(self, collection_id: str) -> dict[str, Any]:
        """Delete a collection/pipeline."""
        return _unwrap_response(await self._request("DELETE", f"/collections/{collection_id}"))

    async def run_collection(self, collection_id: str, **kwargs: Any) -> dict[str, Any]:
        """Run a collection/pipeline."""
        return _unwrap_response(
            await self._request(
                "POST", "/collections/run", json={"pipeline_id": collection_id, **kwargs}
            )
        )

    async def collect_from_collection(self, collection_id: str, **kwargs: Any) -> dict[str, Any]:
        """Collect data from a collection/pipeline."""
        return _unwrap_response(
            await self._request("POST", f"/collections/{collection_id}/collect", json=kwargs)
        )

    async def get_collection_records(self, collection_id: str) -> list[dict[str, Any]]:
        """Get records from a collection/pipeline."""
        return _unwrap_response(await self._request("GET", f"/collections/{collection_id}/records"))

    async def get_collection_stats(self) -> dict[str, Any]:
        """Get collection/pipeline stats."""
        return _unwrap_response(await self._request("GET", "/collections/stats"))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
