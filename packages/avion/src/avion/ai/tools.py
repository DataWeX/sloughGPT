"""Product-tool seam — use outside tools without importing them.

Avion never imports product code (no domain.*, no apps.*, no heavy
libraries at module level). Instead the product side injects a tiny
async callable; Avion normalizes its result into an observation.

Product wiring is one line::

    from avion.ai.tools import make_executor
    executor = make_executor(
        lambda tool, args: runner.execute(tool, args, exec_context))
    agent = Agent(model=..., tools=registry, executor=executor)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolSpec:
    """Describes one outside tool for models and validation."""

    name: str
    description: str = ""
    params: dict[str, str] = field(default_factory=dict)
    requires_approval: bool = False

    @classmethod
    def from_definition(cls, definition: Any) -> ToolSpec:
        """Convert a product ToolDefinition (duck-typed, never imported)."""
        params = getattr(definition, "parameters", {}) or {}
        flat = {k: str(v) for k, v in params.items()} if isinstance(params, dict) else {}
        return cls(
            name=str(getattr(definition, "name", "")),
            description=str(getattr(definition, "description", "")),
            params=flat,
            requires_approval=bool(getattr(definition, "requires_approval", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "params": self.params,
            "requires_approval": self.requires_approval,
        }


@runtime_checkable
class ToolExecutor(Protocol):
    """Runs an outside tool. Returns text, or a result dict."""

    async def run(self, tool: str, args: dict[str, Any]) -> Any:
        """Execute tool with args. Dicts use success/error keys."""
        ...


def normalize_result(result: Any, max_chars: int = 2000) -> tuple[str, bool]:
    """Turn an executor result into (observation, ok)."""
    if isinstance(result, dict):
        if result.get("success") is False or "error" in result:
            detail = result.get("error", result)
            return f"tool failed: {detail}"[:max_chars], False
        for key in ("result", "output", "text", "data"):
            if key in result:
                return str(result[key])[:max_chars], True
        return str(result)[:max_chars], True
    return str(result)[:max_chars], True


def make_executor(
    run_fn: Callable[[str, dict[str, Any]], Awaitable[Any]],
) -> ToolExecutor:
    """Wrap a bare async (tool, args) callable as a ToolExecutor."""

    class _FnExecutor:
        async def run(self, tool: str, args: dict[str, Any]) -> Any:
            return await run_fn(tool, args)

    return _FnExecutor()


class ToolRegistry:
    """Named tool specs shown to models and checked before execution."""

    def __init__(self, tools: list[ToolSpec] | None = None):
        self._tools: dict[str, ToolSpec] = {}
        for tool in tools or []:
            self.add(tool)

    def __len__(self) -> int:
        return len(self._tools)

    def add(self, tool: ToolSpec) -> None:
        self._tools[tool.name] = tool

    def remove(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> str:
        """Human-readable tool list for model prompts."""
        if not self._tools:
            return "(no outside tools)"
        lines = []
        for name in self.names:
            spec = self._tools[name]
            params = ", ".join(f"{k}: {v}" for k, v in spec.params.items()) or "none"
            gate = " [needs approval]" if spec.requires_approval else ""
            lines.append(f"- {name}: {spec.description} Params({params}){gate}")
        return "\n".join(lines)
