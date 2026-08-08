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
) -> Dict[str, Any]:
    """Build JSON body for POST /training/start."""
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
        api_key: Optional[str] = None,
        timeout: int = 30,
        verify_ssl: bool = True,
        headers: Optional[Dict[str, str]] = None,
    ):
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
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        kwargs.setdefault("timeout", self.timeout)
        kwargs.setdefault("verify", self.verify_ssl)
        response = self._session.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    # ============ Health & Status ============

    def health(self) -> HealthStatus:
        """Check API health status."""
        response = self._request("GET", "/health")
        return HealthStatus.from_response(response.json())

    def liveness(self) -> Dict[str, Any]:
        """Check if the server is alive."""
        response = self._request("GET", "/health/live")
        return response.json()

    def readiness(self) -> Dict[str, Any]:
        """Check if the server is ready."""
        response = self._request("GET", "/health/ready")
        return response.json()

    def detailed_health(self) -> Dict[str, Any]:
        """Get detailed health info."""
        response = self._request("GET", "/health/detailed")
        return response.json()

    def info(self) -> SystemInfo:
        """Get detailed system information."""
        response = self._request("GET", "/info")
        return SystemInfo.from_response(response.json())

    # ============ Text Generation ============

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
        response = self._request("POST", "/inference/generate", json=request.to_dict())
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
            "/inference/generate/stream",
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

    # ============ Chat Completions ============

    def chat(
        self,
        messages: Union[List[ChatMessage], List[Dict[str, str]]],
        model: Optional[str] = None,
        temperature: float = 0.8,
        max_new_tokens: int = 100,
        **kwargs
    ) -> ChatResult:
        """Generate a chat completion."""
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
        """Generate a chat completion with streaming."""
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
                    if obj.get("error"):
                        break
                    tok = obj.get("token")
                    if tok:
                        yield tok

    # ============ Models ============

    def list_models(self) -> List[ModelInfo]:
        """List available models."""
        response = self._request("GET", "/models")
        data = response.json()
        models_list = data.get("models", data) if isinstance(data, dict) else data
        return [ModelInfo.from_dict(m) for m in models_list]

    def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a model into memory."""
        response = self._request("POST", "/models/load", json={"model_id": model_id})
        return response.json()

    def unload_model(self) -> Dict[str, Any]:
        """Unload the current model."""
        response = self._request("POST", "/models/unload")
        return response.json()

    def get_current_model(self) -> Dict[str, Any]:
        """Get current loaded model info."""
        response = self._request("GET", "/models/current")
        return response.json()

    def list_hf_models(self, query: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """List available HuggingFace models."""
        params = {"limit": limit}
        if query:
            params["q"] = query
        response = self._request("GET", "/models/hf", params=params)
        return response.json().get("models", [])

    # ============ Sessions ============

    def create_session(self) -> Dict[str, Any]:
        """Create a new chat session."""
        response = self._request("POST", "/chat/sessions")
        return response.json()

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List chat sessions."""
        response = self._request("GET", "/chat/sessions")
        data = response.json()
        return data.get("sessions", data) if isinstance(data, dict) else data

    def get_session(self, session_id: str) -> Dict[str, Any]:
        """Get session details."""
        response = self._request("GET", f"/chat/sessions/{session_id}")
        return response.json()

    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """Delete a session."""
        response = self._request("DELETE", f"/chat/sessions/{session_id}")
        return response.json()

    def save_session_context(self, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Store regeneration context for a session."""
        response = self._request("POST", f"/session/{session_id}/context", json=context)
        return response.json()

    def get_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """Get stored context messages for a session."""
        response = self._request("GET", f"/session/{session_id}/messages")
        data = response.json()
        return data.get("messages", data) if isinstance(data, dict) else data

    # ============ Souls ============

    def list_souls(self) -> List[Dict[str, Any]]:
        """List available souls."""
        response = self._request("GET", "/souls")
        data = response.json()
        return data.get("souls", data) if isinstance(data, dict) else data

    def get_current_soul(self) -> Dict[str, Any]:
        """Get the current active soul."""
        response = self._request("GET", "/souls/current")
        return response.json()

    def switch_soul(self, name: str, checkpoint_name: Optional[str] = None) -> Dict[str, Any]:
        """Switch to a soul by name, optionally loading a checkpoint."""
        body: Dict[str, Any] = {}
        if checkpoint_name:
            body["checkpoint_name"] = checkpoint_name
        response = self._request("POST", f"/souls/switch/{name}", json=body)
        return response.json()

    # ============ Knowledge ============

    def list_knowledge(self) -> List[Dict[str, Any]]:
        """List knowledge items."""
        response = self._request("GET", "/knowledge")
        data = response.json()
        return data.get("items", data) if isinstance(data, dict) else data

    def add_knowledge(self, content: str, topic: Optional[str] = None) -> Dict[str, Any]:
        """Add a knowledge item."""
        body: Dict[str, Any] = {"content": content}
        if topic:
            body["topic"] = topic
        response = self._request("POST", "/knowledge", json=body)
        return response.json()

    def delete_knowledge(self, item_id: str) -> Dict[str, Any]:
        """Delete a knowledge item."""
        response = self._request("DELETE", f"/knowledge/{item_id}")
        return response.json()

    def search_knowledge(self, query: str) -> List[Dict[str, Any]]:
        """Search knowledge items."""
        response = self._request("GET", f"/knowledge/search", params={"q": query})
        data = response.json()
        return data.get("results", data) if isinstance(data, dict) else data

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        response = self._request("GET", "/knowledge/stats")
        return response.json()

    def get_knowledge_topics(self) -> List[str]:
        """Get distinct knowledge topics."""
        response = self._request("GET", "/knowledge/topics")
        data = response.json()
        return data.get("topics", data) if isinstance(data, dict) else data

    def ingest_knowledge_url(self, url: str) -> Dict[str, Any]:
        """Ingest a URL into the knowledge base."""
        response = self._request("POST", "/knowledge/ingest-url", json={"url": url})
        return response.json()

    # ============ Tokenizer ============

    def get_tokenizer_stats(self) -> Dict[str, Any]:
        """Get tokenizer statistics."""
        response = self._request("GET", "/tokenizer/stats")
        return response.json()

    def tokenize(self, text: str) -> Dict[str, Any]:
        """Tokenize text."""
        response = self._request("POST", "/tokenizer/tokenize", json={"text": text})
        return response.json()

    def train_tokenizer(self, text: str, vocab_size: Optional[int] = None) -> Dict[str, Any]:
        """Train the tokenizer on text."""
        body: Dict[str, Any] = {"text": text}
        if vocab_size:
            body["vocab_size"] = vocab_size
        response = self._request("POST", "/tokenizer/train", json=body)
        return response.json()

    # ============ System ============

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system metrics (CPU, memory, disk, GPU)."""
        response = self._request("GET", "/system/metrics")
        return response.json()

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        response = self._request("GET", "/system/info")
        return response.json()

    def get_system_disk(self) -> Dict[str, Any]:
        """Get disk usage information."""
        response = self._request("GET", "/system/disk")
        return response.json()

    # ============ Companion / Personality ============

    def get_personalities(self) -> List[Dict[str, Any]]:
        """Get available personalities."""
        response = self._request("GET", "/personalities")
        return response.json().get("personalities", [])

    def set_personality(self, personality: str) -> Dict[str, Any]:
        """Set the current personality via companion."""
        response = self._request("POST", "/companion/personality", json={"personality": personality})
        return response.json()

    def get_companion_prompt(self) -> Dict[str, Any]:
        """Get the current companion system prompt."""
        response = self._request("GET", "/companion/prompt")
        return response.json()

    def list_companion_presets(self) -> List[Dict[str, Any]]:
        """List available companion presets."""
        response = self._request("GET", "/companion/presets")
        data = response.json()
        return data.get("presets", data) if isinstance(data, dict) else data

    # ============ Datasets ============

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

    def get_dataset_stats(self, dataset_id: str) -> Dict[str, Any]:
        """Get dataset statistics."""
        response = self._request("GET", f"/datasets/{dataset_id}/stats")
        return response.json()

    def import_dataset_local(self, path: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Import a local file or directory as a dataset."""
        body: Dict[str, Any] = {"path": path}
        if name:
            body["name"] = name
        response = self._request("POST", "/datasets/import/local", json=body)
        return response.json()

    def import_dataset_github(self, repo: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Import a GitHub repository as a dataset."""
        body: Dict[str, Any] = {"repo": repo}
        if name:
            body["name"] = name
        response = self._request("POST", "/datasets/import/github", json=body)
        return response.json()

    def import_dataset_url(self, url: str, name: Optional[str] = None) -> Dict[str, Any]:
        """Import a URL as a dataset."""
        body: Dict[str, Any] = {"url": url}
        if name:
            body["name"] = name
        response = self._request("POST", "/datasets/import/url", json=body)
        return response.json()

    # ============ Metrics ============

    def metrics(self) -> MetricsData:
        """Get API metrics."""
        response = self._request("GET", "/metrics")
        return MetricsData.from_response(response.json())

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
        **kwargs
    ) -> Dict[str, Any]:
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

    def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """Get training job status."""
        response = self._request("GET", f"/training/jobs/{job_id}")
        return response.json()

    def list_training_jobs(self) -> List[Dict[str, Any]]:
        """List all training jobs."""
        response = self._request("GET", "/training/jobs")
        return _coerce_training_jobs_list(response.json())

    def delete_training_job(self, job_id: str) -> Dict[str, Any]:
        """Delete a training job."""
        response = self._request("DELETE", f"/training/jobs/{job_id}")
        return response.json()

    def stop_training(self) -> Dict[str, Any]:
        """Stop the current training run."""
        response = self._request("POST", "/training/control/stop")
        return response.json()

    def pause_training(self) -> Dict[str, Any]:
        """Pause the current training run."""
        response = self._request("POST", "/training/control/pause")
        return response.json()

    def resume_training(self) -> Dict[str, Any]:
        """Resume the current training run."""
        response = self._request("POST", "/training/control/resume")
        return response.json()

    def get_training_recovery_stats(self) -> Dict[str, Any]:
        """Get training recovery statistics."""
        response = self._request("GET", "/recovery/stats")
        return response.json()

    def abandon_recovery(self, job_id: str) -> Dict[str, Any]:
        """Abandon a recoverable training job."""
        response = self._request("DELETE", f"/recovery/abandon/{job_id}")
        return response.json()

    # ============ Auto-Train ============

    def start_auto_train(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start auto-training."""
        response = self._request("POST", "/auto-train/start", json=config)
        return response.json()

    def stop_auto_train(self) -> Dict[str, Any]:
        """Stop auto-training."""
        response = self._request("POST", "/auto-train/stop")
        return response.json()

    def get_auto_train_status(self) -> Dict[str, Any]:
        """Get auto-training status."""
        response = self._request("GET", "/auto-train/status")
        return response.json()

    def list_auto_train_checkpoints(self) -> List[Dict[str, Any]]:
        """List auto-training checkpoints."""
        response = self._request("GET", "/auto-train/checkpoints")
        data = response.json()
        return data.get("checkpoints", data) if isinstance(data, dict) else data

    def delete_auto_train_checkpoint(self, name: str) -> Dict[str, Any]:
        """Delete an auto-training checkpoint."""
        response = self._request("DELETE", f"/auto-train/checkpoints/{name}")
        return response.json()

    def load_auto_train_checkpoint(self, name: str) -> Dict[str, Any]:
        """Load an auto-training checkpoint."""
        response = self._request("POST", f"/auto-train/checkpoints/{name}/load")
        return response.json()

    # ============ Feedback ============

    def record_feedback(self, session_id: str, message_id: str, score: int, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Record feedback for a message (triggers workflow)."""
        body: Dict[str, Any] = {"session_id": session_id, "message_id": message_id, "score": score}
        if tags:
            body["tags"] = tags
        response = self._request("POST", "/feedback/workflow-record", json=body)
        return response.json()

    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        response = self._request("GET", "/feedback/stats/summary")
        return response.json()

    # ============ Workflow ============

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get the feedback workflow status."""
        response = self._request("GET", "/workflow/status")
        return response.json()

    # ============ Experiments ============

    def create_experiment(
        self,
        name: str,
        description: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new experiment."""
        payload = {"name": name, "description": description, **kwargs}
        response = self._request("POST", "/experiments", json=payload)
        return response.json()

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments."""
        response = self._request("GET", "/experiments")
        return response.json().get("experiments", [])

    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment details."""
        response = self._request("GET", f"/experiments/{experiment_id}")
        return response.json()

    def log_metric(
        self,
        experiment_id: str,
        metric_name: str,
        value: float,
        step: Optional[int] = None
    ) -> Dict[str, Any]:
        """Log a metric to an experiment."""
        payload: Dict[str, Any] = {"metric": metric_name, "value": value}
        if step is not None:
            payload["step"] = step
        response = self._request("POST", f"/experiments/{experiment_id}/log_metric", json=payload)
        return response.json()

    def log_param(
        self,
        experiment_id: str,
        param_name: str,
        value: Any
    ) -> Dict[str, Any]:
        """Log a parameter to an experiment."""
        payload = {"param": param_name, "value": value}
        response = self._request("POST", f"/experiments/{experiment_id}/log_param", json=payload)
        return response.json()

    # ============ Rate Limit ============

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get rate limit status."""
        response = self._request("GET", "/rate-limit/status")
        return response.json()

    def check_rate_limit(self) -> Dict[str, Any]:
        """Check if a request would be rate limited."""
        response = self._request("GET", "/rate-limit/check")
        return response.json()

    # ============ Security ============

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get the security audit log."""
        response = self._request("GET", "/security/audit")
        data = _unwrap_response(response.json())
        return data if isinstance(data, list) else data.get("logs", data)

    def get_security_keys(self) -> List[Dict[str, Any]]:
        """List registered security/API keys."""
        response = self._request("GET", "/security/keys")
        data = _unwrap_response(response.json())
        return data if isinstance(data, list) else data.get("keys", data)

    # ============ Registry ============

    def list_registry_models(self) -> List[Dict[str, Any]]:
        """List models registered in the live model registry."""
        response = self._request("GET", "/registry/models")
        data = _unwrap_response(response.json())
        return data.get("models", data)

    def get_registry_model(self, model_id: str) -> Dict[str, Any]:
        """Get a single registered model's details."""
        response = self._request("GET", f"/registry/models/{model_id}")
        data = _unwrap_response(response.json())
        return data if isinstance(data, dict) else data.get("data", data)

    def get_registry_best(self) -> Dict[str, Any]:
        """Get best performing model by live registry metrics."""
        response = self._request("GET", "/registry/best")
        return _unwrap_response(response.json())

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get live model registry statistics."""
        response = self._request("GET", "/registry/stats")
        return _unwrap_response(response.json())

    # ============ Benchmark ============

    def run_benchmark(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a benchmark."""
        response = self._request("POST", "/benchmark/run", json=config)
        return response.json()

    def get_benchmark_metrics(self) -> List[Dict[str, Any]]:
        """Get benchmark metrics."""
        response = self._request("GET", "/benchmark/metrics")
        data = response.json()
        return data if isinstance(data, list) else data.get("metrics", data)

    def get_benchmark_stats(self) -> Dict[str, Any]:
        """Get benchmark statistics."""
        response = self._request("GET", "/benchmark/stats")
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


class SimpleTracker:
    """Simple context manager for tracking metrics."""

    def __init__(self, client: "SloughGPTClient", name: str):
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
        api_key: Optional[str] = None,
        timeout: int = 30,
        verify_ssl: bool = True,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self._headers = headers or {}
        if api_key:
            self._headers["X-API-Key"] = api_key

    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
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
        data = await self._request("GET", "/health")
        return HealthStatus.from_response(data)

    async def generate(self, prompt: str, **kwargs) -> GenerationResult:
        from .models import GenerateRequest
        request = GenerateRequest(prompt=prompt, **kwargs)
        data = await self._request("POST", "/inference/generate", json=request.to_dict())
        return GenerationResult.from_response(data, prompt)

    async def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResult:
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
        data = await self._request("GET", "/models")
        models_list = data.get("models", data) if isinstance(data, dict) else data
        return [ModelInfo.from_dict(m) for m in models_list]

    async def list_souls(self) -> List[Dict[str, Any]]:
        data = await self._request("GET", "/souls")
        return data.get("souls", data) if isinstance(data, dict) else data

    async def switch_soul(self, name: str, checkpoint_name: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if checkpoint_name:
            body["checkpoint_name"] = checkpoint_name
        return await self._request("POST", f"/souls/switch/{name}", json=body)

    async def list_knowledge(self) -> List[Dict[str, Any]]:
        data = await self._request("GET", "/knowledge")
        return data.get("items", data) if isinstance(data, dict) else data

    async def add_knowledge(self, content: str, topic: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"content": content}
        if topic:
            body["topic"] = topic
        return await self._request("POST", "/knowledge", json=body)

    async def search_knowledge(self, query: str) -> List[Dict[str, Any]]:
        data = await self._request("GET", "/knowledge/search", params={"q": query})
        return data.get("results", data) if isinstance(data, dict) else data

    async def get_system_metrics(self) -> Dict[str, Any]:
        return await self._request("GET", "/system/metrics")

    async def metrics(self) -> MetricsData:
        data = await self._request("GET", "/metrics")
        return MetricsData.from_response(data)

    async def get_workflow_status(self) -> Dict[str, Any]:
        return await self._request("GET", "/workflow/status")

    async def record_feedback(self, session_id: str, message_id: str, score: int, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"session_id": session_id, "message_id": message_id, "score": score}
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
    ) -> Dict[str, Any]:
        payload = _build_training_start_payload(
            model_name,
            dataset_id,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            **kwargs,
        )
        return await self._request("POST", "/training/start", json=payload)

    async def get_training_status(self, job_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/training/jobs/{job_id}")

    async def list_training_jobs(self) -> List[Dict[str, Any]]:
        data = await self._request("GET", "/training/jobs")
        return _coerce_training_jobs_list(data)

    async def create_experiment(self, name: str, description: str = "", **kwargs) -> Dict[str, Any]:
        payload = {"name": name, "description": description, **kwargs}
        return await self._request("POST", "/experiments", json=payload)

    async def list_experiments(self) -> List[Dict[str, Any]]:
        data = await self._request("GET", "/experiments")
        return data.get("experiments", [])

    async def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/experiments/{experiment_id}")

    async def log_metric(self, experiment_id: str, metric_name: str, value: float, **kwargs) -> Dict[str, Any]:
        payload = {"metric": metric_name, "value": value, **kwargs}
        return await self._request("POST", f"/experiments/{experiment_id}/log_metric", json=payload)

    async def get_tokenizer_stats(self) -> Dict[str, Any]:
        return await self._request("GET", "/tokenizer/stats")

    async def list_auto_train_checkpoints(self) -> List[Dict[str, Any]]:
        data = await self._request("GET", "/auto-train/checkpoints")
        return data.get("checkpoints", data) if isinstance(data, dict) else data

    async def get_security_keys(self) -> List[Dict[str, Any]]:
        """List registered security/API keys."""
        data = _unwrap_response(await self._request("GET", "/security/keys"))
        return data if isinstance(data, list) else data.get("keys", data)

    async def list_registry_models(self) -> List[Dict[str, Any]]:
        """List models registered in the live model registry."""
        data = _unwrap_response(await self._request("GET", "/registry/models"))
        return data.get("models", data)

    async def get_registry_model(self, model_id: str) -> Dict[str, Any]:
        """Get a single registered model's details."""
        data = _unwrap_response(await self._request("GET", f"/registry/models/{model_id}"))
        return data if isinstance(data, dict) else data.get("data", data)

    async def get_registry_best(self) -> Dict[str, Any]:
        """Get best performing model by live registry metrics."""
        return _unwrap_response(await self._request("GET", "/registry/best"))

    async def get_registry_stats(self) -> Dict[str, Any]:
        """Get live model registry statistics."""
        return _unwrap_response(await self._request("GET", "/registry/stats"))

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
