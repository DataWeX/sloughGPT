"""
Tool Registry — lightweight, callable tools for chat.

Each tool has:
- name, description, parameters (JSON schema)
- execute() callable
- pattern (regex) for detecting intent in user messages

Usage:
    registry = ToolRegistry()
    result = await registry.execute("calculator", {"expression": "2+2"})
    tools = registry.list_tools()  # for /chat/tools endpoint
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import math
import re
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger("slo.tools")


@dataclass
class ToolParam:
    name: str
    type: str
    description: str
    required: bool = False


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: List[ToolParam]
    execute: Callable[..., Coroutine[Any, Any, Dict[str, Any]]]
    pattern: Optional[re.Pattern] = None
    requires_approval: bool = False


@dataclass
class ToolResult:
    success: bool
    output: str
    error: Optional[str] = None
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Registry of tools accessible from chat."""

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.register(
            ToolSpec(
                name="calculator",
                description="Evaluate a mathematical expression",
                parameters=[
                    ToolParam("expression", "string", "The mathematical expression to evaluate, e.g. 2+2 or sqrt(144)", required=True),
                ],
                execute=self._run_calculator,
                pattern=re.compile(r"^(?:calculate|calc|math|what is|what's)\s+(.+)$", re.IGNORECASE),
            )
        )
        self.register(
            ToolSpec(
                name="current_time",
                description="Get the current date and time",
                parameters=[
                    ToolParam("timezone", "string", "Timezone (optional, default local)", required=False),
                ],
                execute=self._run_current_time,
                pattern=re.compile(r"^(?:time|date|what time|what date|what's the time|current time)\s*$", re.IGNORECASE),
            )
        )
        self.register(
            ToolSpec(
                name="web_search",
                description="Search the web for information",
                parameters=[
                    ToolParam("query", "string", "What to search for", required=True),
                    ToolParam("num_results", "number", "Number of results to return (default 3)", required=False),
                ],
                execute=self._run_web_search,
                pattern=re.compile(r"^(?:search|look up|find|google|web search)\s+(.+)$", re.IGNORECASE),
                requires_approval=True,
            )
        )
        self.register(
            ToolSpec(
                name="run_code",
                description="Execute Python or shell code safely",
                parameters=[
                    ToolParam("language", "string", "Code language: python or bash", required=True),
                    ToolParam("code", "string", "The code to execute", required=True),
                ],
                execute=self._run_code,
                pattern=re.compile(r"^```(\w+)?\n(.+?)\n```$", re.DOTALL),
                requires_approval=True,
            )
        )

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec
        logger.info("Registered tool: %s", spec.name, extra={"tag": "MODEL"})

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": [{"name": p.name, "type": p.type, "description": p.description, "required": p.required} for p in t.parameters],
                "requires_approval": t.requires_approval,
            }
            for t in self._tools.values()
        ]

    def detect_tool_intent(self, text: str) -> Optional[tuple[str, Dict[str, Any]]]:
        """Detect if a user message is requesting a tool.

        Returns (tool_name, args) if a pattern matches, else None.
        """
        for name, spec in self._tools.items():
            if spec.pattern:
                m = spec.pattern.search(text.strip())
                if m:
                    if name == "calculator":
                        return name, {"expression": m.group(1).strip()}
                    elif name == "current_time":
                        return name, {}
                    elif name == "web_search":
                        return name, {"query": m.group(1).strip(), "num_results": 3}
                    elif name == "run_code":
                        lang = m.group(1) or "python"
                        return name, {"language": lang, "code": m.group(2).strip()}
        return None

    async def execute(self, name: str, args: Dict[str, Any]) -> ToolResult:
        """Execute a tool by name with given args."""
        spec = self._tools.get(name)
        if not spec:
            return ToolResult(success=False, output="", error=f"Unknown tool: {name}")
        start = asyncio.get_event_loop().time()
        try:
            result = await spec.execute(**args)
            elapsed = (asyncio.get_event_loop().time() - start) * 1000
            if result.get("error"):
                return ToolResult(success=False, output=result.get("output", ""), error=result["error"], duration_ms=elapsed, metadata=result)
            return ToolResult(success=True, output=result.get("output", ""), duration_ms=elapsed, metadata=result)
        except Exception as e:
            elapsed = (asyncio.get_event_loop().time() - start) * 1000
            return ToolResult(success=False, output="", error=str(e), duration_ms=elapsed)

    # ── Tool Implementations ────────────────────────────────────────

    async def _run_calculator(self, expression: str) -> Dict[str, Any]:
        """Evaluate a math expression safely."""
        allowed = set("0123456789+-*/.()% ,")
        cleaned = "".join(c for c in expression if c in allowed or c.isalpha())
        cleaned = cleaned.strip()
        if not cleaned:
            return {"output": "", "error": "Empty expression"}
        blocked = {"__import__", "eval", "exec", "open", "os.", "import"}
        if any(b in cleaned for b in blocked):
            return {"output": "", "error": "Blocked function detected"}
        safe_globals = {"__builtins__": {k: __builtins__[k] for k in ["abs", "int", "float", "str", "round", "min", "max", "sum", "len", "range", "list", "dict", "tuple", "bool", "pow"]}}
        safe_globals["__builtins__"].update({"math": math, "sqrt": math.sqrt, "pi": math.pi, "e": math.e, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log, "log10": math.log10, "ceil": math.ceil, "floor": math.floor, "factorial": math.factorial})
        try:
            result = eval(cleaned, safe_globals, {})
            return {"output": str(result)}
        except Exception as e:
            return {"output": "", "error": str(e)}

    async def _run_current_time(self, timezone: str = "") -> Dict[str, Any]:
        now = datetime.datetime.now()
        return {"output": now.strftime("%Y-%m-%d %H:%M:%S (%A, %B %d, %Y)")}

    async def _run_web_search(self, query: str, num_results: int = 3) -> Dict[str, Any]:
        """Search the web via web_search tool."""
        try:
            from web_search import web_search as ws
            results = await ws(query, num_results=num_results)
            if isinstance(results, list):
                formatted = []
                for r in results[:num_results]:
                    title = r.get("title", "Untitled")
                    url = r.get("url", "")
                    snippet = r.get("snippet", "") or r.get("content", "")
                    formatted.append(f"- **{title}**\n  {snippet}\n  {url}")
                return {"output": "\n\n".join(formatted) if formatted else "No results found."}
            return {"output": str(results)}
        except ImportError:
            return {"output": "Web search not available. Try installing `web_search` package or use a different tool."}

    async def _run_code(self, language: str, code: str) -> Dict[str, Any]:
        """Execute code in a subprocess with timeout."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py" if language == "python" else ".sh", delete=False) as f:
            f.write(code)
            f.flush()
            fname = f.name
        try:
            cmd = ["python3", fname] if language == "python" else ["bash", fname]
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=15)
            out = stdout.decode().strip() if stdout else ""
            err = stderr.decode().strip() if stderr else ""
            if proc.returncode != 0:
                return {"output": out, "error": err or f"Exit code {proc.returncode}"}
            return {"output": out, "error": err if err else None}
        except asyncio.TimeoutError:
            return {"output": "", "error": "Execution timed out (15s limit)"}
        except Exception as e:
            return {"output": "", "error": str(e)}
        finally:
            try:
                import os
                os.unlink(fname)
            except Exception:
                pass


_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
