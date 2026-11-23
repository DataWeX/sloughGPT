"""LLMBrain — token-space decision brain (Stage B of the World–Model Bridge).

Stage A (card 7ff49d20) put weight-space brains behind the ``BRAIN_KINDS``
registry: numpy tensors updated in-world by backprop. Stage B adds the tier
its module docstring reserved — **token-space**: the observation is
serialized into a prompt, a real model answers, and the answer is parsed
back into decision gates.

The seam contract is unchanged: ``LLMBrain`` honours
``forward``/``update``/``to_dict``/``from_dict`` exactly like the other four
kinds, so ``SimBaby``, ``spawn_child``, serialization and the CLI need no
knowledge of which tier is deciding. What changes:

* **No flat tensors.** The model's weights live outside the world, so the
  brain has no ``W``/``b`` the genome could carry. Like ``slonet`` it
  inherits WHOLE through ``brain_from_dict`` in ``SimBaby.spawn_child`` and
  is refused by ``Genome.from_baby``/``Genome.apply_to`` (fail loud, never
  a silent half-copy of tensors that are not there).
* **``update`` is a no-op by design.** The world's reward signal cannot
  reach frozen model weights; for non-genome brains learning lives where
  Stage A already put it — the baby's episodic memory (memotype) and
  generational selection. A reward-aware prompt is a documented follow-up,
  not a hidden side effect.
* **The model sits behind a client — real backends only.** ``ModelClient``
  has two backends: ``local`` (the served ``InferenceEngine`` over its
  length-prefixed socket) and ``api`` (any OpenAI-compatible HTTP
  endpoint). There is deliberately **no mock** (user ruling 2026-11-21:
  "no mock anywhere, real engine in tests"): tests talk to a real engine
  (already-running or spawned — see ``engine_port`` in
  ``test_world_llm_brain``) or to a dead port for the transport-failure
  path, and answer parsing is exercised as the pure function it is.
  ``make_client`` is the single construction seam; a process-wide default
  set once at CLI start (``set_default_client``) keeps the registry
  factory a plain ``(input_dim, output_dim)`` constructor like every
  other kind.

Fail-safe: a dead client must steer neutral, not kill the tick — transport
errors and unparseable answers both fall back to gates of 0.5 (the same
contract ``_FALLBACK_GATE`` states in ``model_brain``).
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from typing import Any

import numpy as np

logger = logging.getLogger("slo.world.llm_brain")

#: Neutral gate — a dead brain must steer neutral instead of killing the tick.
_FALLBACK_GATE = 0.5

#: Floats extracted from a model answer before ``output_dim`` is enforced.
#: Generous because the model may preface its numbers with prose.
_FLOAT_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


class ModelClient:
    """Sync prompt-in/text-out interface behind every LLM brain decision.

    The world core is synchronous by design (AGENTS.md execution
    philosophy), so ``complete`` blocks: parallelism comes from the worker
    pool fanning out across independent runs, never from async in here.
    """

    #: Backend name — recorded in logs and chosen by ``make_client``.
    name = "base"

    def complete(self, prompt: str, *, timeout: float = 30.0) -> str:
        """Return the model's answer text for ``prompt``.

        Args:
            prompt: the serialized decision request.
            timeout: seconds before the backend may fail.

        Returns:
            The model's raw answer text (may be prose; the brain parses it).
        """
        raise NotImplementedError


class LocalEngineClient(ModelClient):
    """Backend over the served ``InferenceEngine`` TCP protocol.

    Reuses ``InferenceClient``'s thread-safe length-prefixed transport
    (``_send_and_recv``) with a ``generate`` frame — the same wire the API
    server rides — so the world asks the local model exactly what the rest
    of the product asks it. The socket opens lazily on first use, which
    keeps forked worker processes from inheriting a parent's connection.
    """

    name = "local"

    def __init__(self, host: str = "127.0.0.1", port: int = 9100):
        """Args:
        host: engine host.
        port: engine port (``ENGINE_READY port=...``).
        """
        self.host = host
        self.port = int(port)
        self._client: Any = None

    def complete(self, prompt: str, *, timeout: float = 30.0) -> str:
        """Send one ``generate`` frame and return the result text.

        Args:
            prompt: the serialized decision request.
            timeout: seconds to wait for the engine's reply.

        Returns:
            The model's answer text.

        Raises:
            RuntimeError: if the engine answers with an error frame or no
                frame at all (the brain converts this to neutral gates).
        """
        if self._client is None:
            from domain.infrastructure._internal.inference_client import InferenceClient

            self._client = InferenceClient(host=self.host, port=self.port)
        # temperature=0.0 → greedy argmax in the engine's sampler
        # (slonet._sample_token), so one observation always yields the
        # same gates — the determinism tests depend on it.
        response = self._client._send_and_recv(
            {
                "type": "generate",
                "id": uuid.uuid4().hex,
                "messages": [{"role": "user", "content": prompt}],
                "params": {"max_new_tokens": 64, "temperature": 0.0},
            },
            timeout=timeout,
        )
        if not response or response.get("type") != "result":
            kind = None if response is None else response.get("type")
            raise RuntimeError(f"inference engine answered {kind!r} (expected 'result')")
        return str(response.get("text", ""))


class ExternalApiClient(ModelClient):
    """Backend over any OpenAI-compatible ``/chat/completions`` HTTP endpoint.

    Temperature defaults to 0 so a given endpoint answers a given prompt
    reproducibly — the parity tests rely on the backend being a pure
    function of the prompt wherever the provider supports it.
    """

    name = "api"

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str = "sloughgpt",
        default_timeout: float = 30.0,
    ):
        """Args:
        base_url: API root, e.g. ``http://127.0.0.1:8000/v1`` (the
            ``/chat/completions`` suffix is appended).
        api_key: bearer token, if the endpoint wants one.
        model: model id sent in the payload.
        default_timeout: fallback timeout when the caller passes none.
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.default_timeout = float(default_timeout)

    def complete(self, prompt: str, *, timeout: float | None = None) -> str:
        """POST one chat completion and return the message content.

        Args:
            prompt: the serialized decision request.
            timeout: seconds; defaults to ``default_timeout``.

        Returns:
            The model's answer text.

        Raises:
            requests.HTTPError / requests.Timeout: on transport failure
                (the brain converts these to neutral gates).
        """
        import requests

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 64,
            "temperature": 0.0,
            "stream": False,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=self.default_timeout if timeout is None else timeout,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])


def make_client(backend: str = "local", url: str | None = None) -> ModelClient:
    """Construct a model client by backend name — the one construction seam.

    Args:
        backend: ``local`` | ``api`` (no mock — real models only).
        url: ``host:port`` for ``local`` (default ``127.0.0.1:9100``);
            base URL for ``api`` (default ``$SLO_WORLD_API_BASE`` or
            ``http://127.0.0.1:8000/v1``).

    Returns:
        A ready ``ModelClient``.

    Raises:
        ValueError: on an unknown backend name (fail loud — including
            ``"mock"``, removed by user ruling 2026-11-21).
    """
    if backend == "local":
        host, sep, port = (url or "127.0.0.1:9100").partition(":")
        return LocalEngineClient(host=host or "127.0.0.1", port=int(port) if sep and port else 9100)
    if backend == "api":
        base = url or os.environ.get("SLO_WORLD_API_BASE", "http://127.0.0.1:8000/v1")
        return ExternalApiClient(base, api_key=os.environ.get("SLO_WORLD_API_KEY"))
    raise ValueError(f"unknown model backend {backend!r} (known: local, api)")


_default_client: ModelClient | None = None


def get_default_client() -> ModelClient:
    """Return the process-wide model client (the real local engine until configured).

    Not lazily re-created per call: the CLI sets the default once before
    any brain constructs, and worker processes inherit that choice through
    fork (or rebuild it from their own configuration).
    """
    global _default_client
    if _default_client is None:
        _default_client = LocalEngineClient()
    return _default_client


def set_default_client(client: ModelClient) -> None:
    """Install the process-wide model client (CLI start, tests)."""
    global _default_client
    _default_client = client


def reset_default_client() -> None:
    """Drop the process-wide client so the next use starts from the local engine."""
    global _default_client
    _default_client = None


class LLMBrain:
    """Token-space decision brain: observation → prompt → model → gates.

    Drop-in behind the same ``forward``/``update``/``to_dict``/``from_dict``
    seam as the weight-space kinds. ``forward`` renders the perception
    features into a compact prompt, asks the model client, and parses the
    answer into ``output_dim`` sigmoid-range gates; ``update`` is a no-op
    (see module docstring); serialization carries identity only (dims —
    like ``RandomBrain``), because the model's weights are not the world's
    to copy.
    """

    #: Serialization marker — lets ``brain_from_dict`` rebuild the right class.
    kind = "llm"

    def __init__(self, input_dim: int, output_dim: int, hidden_units: int = 0, client: Any = None):
        """Args:
        input_dim: perception feature count.
        output_dim: decision gate count.
        hidden_units: accepted for registry-factory parity and ignored —
            token space has no projection tensor.
        client: model backend; defaults to the process-wide client.
        """
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_units = 0  # weight-space concept; dropped by design
        self.client = client if client is not None else get_default_client()
        self.calls = 0  # observability: decisions asked of the model

    def _render_prompt(self, x: np.ndarray) -> str:
        """Serialize perception features into the decision prompt.

        Deterministic and compact: fixed4-decimal rendering so the same
        observation always produces byte-identical prompt text (the
        determinism tests and seeded parity depend on it).

        Args:
            x: perception feature vector.

        Returns:
            The prompt string.
        """
        arr = np.asarray(x, dtype=np.float32).ravel()
        observation = ",".join(f"{float(v):.4f}" for v in arr)
        return (
            f"Decide an agent's action gates. Reply with exactly {self.output_dim} numbers "
            f"in [0,1], comma-separated, no other text. observation=[{observation}]"
        )

    def _parse_gates(self, text: str) -> np.ndarray:
        """Extract ``output_dim`` gates in [0, 1] from model answer text.

        Missing values stay neutral (0.5); out-of-range values clip. Any
        unparseable answer yields an all-neutral vector — a brain that
        cannot read its model must steer straight, not crash the tick.

        Args:
            text: raw model answer.

        Returns:
            float32 vector of length ``output_dim``.
        """
        values = [float(m) for m in _FLOAT_RE.findall(text or "")]
        gates = np.full(self.output_dim, _FALLBACK_GATE, dtype=np.float32)
        for i, value in enumerate(values[: self.output_dim]):
            gates[i] = min(1.0, max(0.0, value))
        return gates

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Decide: serialize ``x``, ask the model client, parse the gates.

        Args:
            x: perception features (same contract as ``Perceptron.forward``).

        Returns:
            float32 gates in [0, 1]; neutral 0.5 on client or parse failure.
        """
        prompt = self._render_prompt(x)
        self.calls += 1
        try:
            text = self.client.complete(prompt)
        except Exception as exc:  # noqa: BLE001 — any backend failure steers neutral
            logger.warning(
                "LLMBrain: model client %s failed, steering neutral: %s", self.client.name, exc
            )
            return np.full(self.output_dim, _FALLBACK_GATE, dtype=np.float32)
        return self._parse_gates(text)

    def update(self, x: np.ndarray, error: np.ndarray, lr: float = 0.01):
        """No-op by design — frozen model weights cannot absorb the reward.

        The reward still lands where non-genome brains keep learning: the
        baby's episodic memory (memotype, consolidated by ``spawn_child``)
        and generational selection over who survives to breed.

        Args:
            x: ignored.
            error: ignored.
            lr: ignored.
        """

    def to_dict(self) -> dict[str, Any]:
        """Serialize identity only — dims, no model weights (RandomBrain parity).

        Returns:
            JSON-safe dict with the ``kind`` tag and both dims.
        """
        return {"kind": self.kind, "input_dim": self.input_dim, "output_dim": self.output_dim}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LLMBrain:
        """Rebuild from :meth:`to_dict` output against the default client.

        Args:
            data: serialized brain dict.

        Returns:
            A new LLMBrain with the stored dims.
        """
        return cls(int(data["input_dim"]), int(data["output_dim"]))
