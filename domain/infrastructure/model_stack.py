"""One-time model base with stacked knowledge/datasets/caches.

Model file is immutable — initialized once, never deleted. Knowledge stores,
datasets and ephemeral caches are stacked as overlay layers referenced by path,
not copied into the model directory. Stack is workspace-scoped.

Phase 8 Just-Cache builds on this: cache `external/<name>` entries are
stack layers, `ModelBase` stays at `~/.cache/sloughgpt/models/<sha>`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelBase:
    """Immutable model handle — never deleted."""

    model_id: str
    path: Path
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def ensure_readonly(self) -> None:
        """Best-effort chmod 444 — prevents accidental delete."""
        try:
            if self.path.exists():
                self.path.chmod(0o444)
        except Exception as e:
            logger.debug("ensure_readonly %s: %s", self.path, e)


@dataclass
class StackLayer:
    kind: str  # "knowledge" | "dataset" | "cache"
    name: str
    path: Path
    workspace_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelStack:
    """One-time model + overlay stacks.

    Usage:
        stack = ModelStack(ModelBase("gpt2", Path("~/.cache/models/gpt2.soul")))
        stack.push(StackLayer("knowledge", "facts", Path("~/.cache/external/facts")))
        # inference uses base + layers via context, never mutates base file
    """

    def __init__(self, base: ModelBase) -> None:
        self.base = base
        self.base.ensure_readonly()
        self._layers: list[StackLayer] = []
        self._by_name: dict[str, StackLayer] = {}

    # ── Layers ────────────────────────────────────────────────────────
    def push(self, layer: StackLayer) -> None:
        if layer.name in self._by_name:
            raise ValueError(f"layer '{layer.name}' already stacked — use replace()")
        if not layer.path.exists():
            logger.warning("Stack layer path missing: %s", layer.path)
        self._layers.append(layer)
        self._by_name[layer.name] = layer

    def replace(self, layer: StackLayer) -> None:
        if layer.name in self._by_name:
            idx = next(i for i, l in enumerate(self._layers) if l.name == layer.name)
            self._layers[idx] = layer
        else:
            self._layers.append(layer)
        self._by_name[layer.name] = layer

    def remove(self, name: str) -> bool:
        if name not in self._by_name:
            return False
        self._layers = [l for l in self._layers if l.name != name]
        del self._by_name[name]
        return True

    def clear(self, kind: str | None = None) -> int:
        before = len(self._layers)
        if kind is None:
            self._layers.clear()
            self._by_name.clear()
        else:
            self._layers = [l for l in self._layers if l.kind != kind]
            self._by_name = {l.name: l for l in self._layers}
        return before - len(self._layers)

    def layers(self, kind: str | None = None, workspace_id: str | None = None) -> list[StackLayer]:
        out = self._layers
        if kind is not None:
            out = [l for l in out if l.kind == kind]
        if workspace_id is not None:
            out = [l for l in out if l.workspace_id in ("", workspace_id)]
        return list(out)

    # ── Context for inference ─────────────────────────────────────────
    def inference_context(self, workspace_id: str = "") -> dict[str, Any]:
        """Build context dict for inference — base + stacked layers."""
        return {
            "model_id": self.base.model_id,
            "model_path": str(self.base.path),
            "model_sha": self.base.sha256,
            "knowledge": [str(l.path) for l in self.layers("knowledge", workspace_id)],
            "datasets": [str(l.path) for l in self.layers("dataset", workspace_id)],
            "caches": [str(l.path) for l in self.layers("cache", workspace_id)],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": {
                "model_id": self.base.model_id,
                "path": str(self.base.path),
                "sha": self.base.sha256,
            },
            "layers": [
                {
                    "kind": l.kind,
                    "name": l.name,
                    "path": str(l.path),
                    "workspace_id": l.workspace_id,
                }
                for l in self._layers
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelStack:
        base = ModelBase(
            model_id=data["base"]["model_id"],
            path=Path(data["base"]["path"]),
            sha256=data["base"].get("sha"),
        )
        stack = cls(base)
        for l in data.get("layers", []):
            stack.push(
                StackLayer(
                    kind=l["kind"],
                    name=l["name"],
                    path=Path(l["path"]),
                    workspace_id=l.get("workspace_id", ""),
                )
            )
        return stack

    # ── Persistence (per-workspace) ───────────────────────────────────
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> ModelStack | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return cls.from_dict(data)
        except Exception as e:
            logger.warning("ModelStack load %s: %s", path, e)
            return None
