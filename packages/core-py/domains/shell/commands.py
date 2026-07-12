"""
ShellCommands — delegates to existing domains for all operations.

Each command maps to real backend endpoints or domain functions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("man.shell.commands")

_REPO_ROOT = Path(__file__).resolve().parents[4]

API_BASE = "http://localhost:8000"


def _api_get(path: str) -> dict[str, Any] | list:
    import requests
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=5)
        if r.status_code == 200:
            return r.json()
        return {"error": f"HTTP {r.status_code}", "detail": r.text[:200]}
    except Exception as e:
        return {"error": str(e)}


def _api_post(path: str, data: dict | None = None) -> dict[str, Any] | list:
    import requests
    try:
        r = requests.post(f"{API_BASE}{path}", json=data or {}, timeout=120)
        if r.status_code in (200, 201):
            return r.json()
        return {"error": f"HTTP {r.status_code}", "detail": r.text[:200]}
    except Exception as e:
        return {"error": str(e)}


async def _api_post_async(path: str, data: dict | None = None) -> dict[str, Any] | list:
    """Async version of _api_post using httpx with connection pooling."""
    import httpx
    try:
        async with httpx.AsyncClient(base_url=API_BASE, timeout=120.0) as client:
            r = await client.post(path, json=data or {})
            if r.status_code in (200, 201):
                return r.json()
            return {"error": f"HTTP {r.status_code}", "detail": r.text[:200]}
    except Exception as e:
        return {"error": str(e)}


def _api_delete(path: str) -> dict[str, Any]:
    import requests
    try:
        r = requests.delete(f"{API_BASE}{path}", timeout=5)
        return r.json() if r.status_code == 200 else {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}


class ShellCommands:
    """All commands are thin wrappers delegating to real domains."""

    @staticmethod
    def ps() -> list[dict[str, Any]]:
        """List running training jobs (processes)."""
        result = _api_get("/training/jobs")
        if isinstance(result, list):
            return result
        return []

    @staticmethod
    def kill(job_id: str) -> dict[str, Any]:
        """Stop a training job."""
        return _api_post(f"/training/jobs/{job_id}/stop")

    @staticmethod
    def models() -> list[dict[str, Any]]:
        """List available models."""
        result = _api_get("/models")
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("models", result.get("models", []))
        return []

    @staticmethod
    def load_model(model_name: str) -> dict[str, Any]:
        """Load a model by name."""
        import requests
        try:
            r = requests.post(
                f"{API_BASE}/models/load",
                json={"model_id": model_name, "device": "cpu"},
                timeout=600,
            )
            if r.status_code in (200, 201):
                return r.json()
            return {"error": f"HTTP {r.status_code}", "detail": r.text[:200]}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def unload_model() -> dict[str, Any]:
        """Unload the current model."""
        return _api_post("/models/unload")

    @staticmethod
    def souls() -> list[dict[str, Any]]:
        """List available souls."""
        result = _api_get("/souls")
        if isinstance(result, dict):
            return result.get("souls", [])
        return []

    @staticmethod
    def switch_soul(name: str) -> dict[str, Any]:
        """Switch to a soul."""
        return _api_post("/souls/switch", {"name": name})

    @staticmethod
    def current_soul() -> dict[str, Any]:
        """Get the current soul."""
        result = _api_get("/souls/current")
        if isinstance(result, dict):
            return result
        return {"name": "unknown"}

    @staticmethod
    def health() -> dict[str, Any]:
        """System health check."""
        result = _api_get("/health")
        if isinstance(result, dict):
            return result
        return {"status": "unknown"}

    @staticmethod
    def health_detailed() -> dict[str, Any]:
        """Detailed system health."""
        result = _api_get("/health/detailed")
        if isinstance(result, dict):
            return result
        return {"status": "unknown"}

    @staticmethod
    def datasets() -> list[dict[str, Any]]:
        """List available datasets."""
        result = _api_get("/datasets")
        if isinstance(result, dict):
            return result.get("datasets", [])
        return []

    @staticmethod
    def list_knowledge(query: str = "") -> list[dict[str, Any]]:
        """List knowledge base entries."""
        if query:
            result = _api_get(f"/knowledge/search?query={query}")
            if isinstance(result, dict):
                return result.get("results", [])
        else:
            result = _api_get("/knowledge")
            if isinstance(result, list):
                return result
        return []

    @staticmethod
    def add_knowledge(content: str, topic: str = "shell") -> dict[str, Any]:
        """Add a fact to the knowledge base."""
        return _api_post("/knowledge", {"content": content, "topic": topic, "source": "shell"})

    @staticmethod
    def knowledge_stats() -> dict[str, Any]:
        """Knowledge base statistics."""
        result = _api_get("/knowledge/stats")
        if isinstance(result, dict):
            return result
        return {"total_items": 0}

    @staticmethod
    def checkpoints() -> list[dict[str, Any]]:
        """List saved checkpoints."""
        result = _api_get("/auto-train/checkpoints")
        if isinstance(result, dict):
            return result.get("checkpoints", [])
        return []

    @staticmethod
    def load_checkpoint(name: str) -> dict[str, Any]:
        """Load a checkpoint."""
        return _api_post(f"/auto-train/checkpoints/{name}/load")

    @staticmethod
    def delete_checkpoint(name: str) -> dict[str, Any]:
        """Delete a checkpoint."""
        return _api_delete(f"/auto-train/checkpoints/{name}")

    @staticmethod
    def finetuned_models() -> list[dict[str, Any]]:
        """List fine-tuned models."""
        result = _api_get("/training/finetuned-models")
        if isinstance(result, dict):
            return result.get("models", [])
        return []

    @staticmethod
    def load_finetuned(name: str) -> dict[str, Any]:
        """Load a fine-tuned model."""
        return _api_post(f"/training/finetuned-models/{name}/load")

    @staticmethod
    def delete_finetuned(name: str) -> dict[str, Any]:
        """Delete a fine-tuned model."""
        return _api_delete(f"/training/finetuned-models/{name}")

    # ── Training ─────────────────────────────────────────────────────────

    @staticmethod
    def train_quick(dataset: str, name: str = "") -> dict[str, Any]:
        """Start quick training on a dataset."""
        return _api_post("/training/quick", {"dataset": dataset, "name": name or None})

    @staticmethod
    def train_auto(soul_name: str = "", teacher: str = "gpt2", epochs: int = 10,
                   source_text: str = "", dataset_id: str = "") -> dict[str, Any]:
        """Start auto-train (SloNet student learning)."""
        return _api_post("/auto-train/start", {
            "soul_name": soul_name, "teacher_model": teacher,
            "epochs": epochs, "source_text": source_text, "dataset_id": dataset_id,
        })

    @staticmethod
    def train_distill(dataset: str, teacher: str = "gpt2", name: str = "",
                      temperature: float = 2.0, epochs: int = 5) -> dict[str, Any]:
        """Start knowledge distillation."""
        return _api_post("/training/distill", {
            "dataset": dataset, "teacher_model": teacher, "name": name or None,
            "temperature": temperature, "epochs": epochs,
        })

    @staticmethod
    def train_hf(model: str, dataset: str, name: str = "", epochs: int = 3,
                 use_lora: bool = True) -> dict[str, Any]:
        """Start HuggingFace fine-tuning."""
        return _api_post("/training/hf-start", {
            "model": model, "dataset": dataset, "name": name or None,
            "epochs": epochs, "use_lora": use_lora,
        })

    @staticmethod
    def train_status() -> list[dict[str, Any]]:
        """List all training jobs with status."""
        result = _api_get("/training/jobs")
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return result.get("jobs", [])
        return []

    @staticmethod
    def train_stop(job_id: str) -> dict[str, Any]:
        """Stop a running training job."""
        return _api_post(f"/training/jobs/{job_id}/stop")

    @staticmethod
    def train_delete(job_id: str) -> dict[str, Any]:
        """Delete a training job record."""
        return _api_delete(f"/training/jobs/{job_id}")

    @staticmethod
    def generate(prompt: str, max_tokens: int = 100) -> dict[str, Any]:
        """Generate text via the inference endpoint."""
        return _api_post("/inference/generate", {
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
        })

    @staticmethod
    async def generate_async(prompt: str, max_tokens: int = 100) -> dict[str, Any]:
        """Async generate via httpx — non-blocking, connection-pooled."""
        return await _api_post_async("/inference/generate", {
            "prompt": prompt,
            "max_new_tokens": max_tokens,
            "temperature": 0.7,
        })

    @staticmethod
    def chat(messages: list[dict[str, str]]) -> dict[str, Any]:
        """Send a chat message."""
        return _api_post("/chat", {"messages": messages})

    @staticmethod
    def system_metrics() -> dict[str, Any]:
        """System metrics (CPU, memory, disk)."""
        result = _api_get("/system/metrics")
        if isinstance(result, dict):
            return result
        return {}

    @staticmethod
    def tokenizer_stats() -> dict[str, Any]:
        """Tokenizer statistics."""
        result = _api_get("/tokenizer/stats")
        if isinstance(result, dict):
            return result
        return {}
