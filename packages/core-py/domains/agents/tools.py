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
        self.register(
            ToolSpec(
                name="file_read",
                description="Read contents of a file",
                parameters=[
                    ToolParam("path", "string", "File path to read", required=True),
                ],
                execute=self._run_file_read,
                pattern=re.compile(r"^(?:read|open|show|cat|view)\s+(?:file\s+)?(.+)$", re.IGNORECASE),
            )
        )
        self.register(
            ToolSpec(
                name="knowledge_retrieval",
                description="Search knowledge base for relevant information",
                parameters=[
                    ToolParam("query", "string", "What to search for in the knowledge base", required=True),
                ],
                execute=self._run_knowledge_retrieval,
                pattern=re.compile(r"^(?:knowledge|recall|remember|look up in docs?)\s+(.+)$", re.IGNORECASE),
            )
        )
        self.register(
            ToolSpec(
                name="image_analysis",
                description="Analyze an image and describe its contents",
                parameters=[
                    ToolParam("image_path", "string", "Path or URL of the image to analyze", required=True),
                ],
                execute=self._run_image_analysis,
                pattern=re.compile(r"^(?:analyze|describe|what's in|look at)\s+(?:image|photo|picture)\s+(.+)$", re.IGNORECASE),
            )
        )
        self.register(
            ToolSpec(
                name="data_analysis",
                description="Analyze a dataset and compute statistics",
                parameters=[
                    ToolParam("data_path", "string", "Path to the data file (CSV, JSON, JSONL)", required=True),
                    ToolParam("operation", "string", "Analysis to perform: summary, statistics, columns, head, or a custom expression", required=False),
                ],
                execute=self._run_data_analysis,
                pattern=re.compile(r"^(?:analyze|stats|summarize|describe)\s+(?:data|dataset|csv|json)\s+(.+)$", re.IGNORECASE),
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
                    elif name == "file_read":
                        return name, {"path": m.group(1).strip()}
                    elif name == "knowledge_retrieval":
                        return name, {"query": m.group(1).strip()}
                    elif name == "image_analysis":
                        return name, {"image_path": m.group(1).strip()}
                    elif name == "data_analysis":
                        return name, {"data_path": m.group(1).strip(), "operation": "summary"}
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
        """Evaluate a math expression safely using AST validation."""
        import ast
        cleaned = expression.strip()
        if not cleaned:
            return {"output": "", "error": "Empty expression"}
        try:
            tree = ast.parse(cleaned, mode='eval')
            allowed_names = {"math", "sqrt", "pi", "e", "sin", "cos", "tan",
                             "log", "log10", "ceil", "floor", "factorial"}
            allowed_nodes = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                             ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
                             ast.USub, ast.UAdd, ast.Call, ast.Name, ast.Load)
            for node in ast.walk(tree):
                if not isinstance(node, allowed_nodes):
                    return {"output": "", "error": f"Disallowed expression: {type(node).__name__}"}
                if isinstance(node, ast.Name) and node.id not in allowed_names:
                    return {"output": "", "error": f"Unknown name: {node.id}"}
                if isinstance(node, ast.Call):
                    if not (isinstance(node.func, ast.Name) and node.func.id in allowed_names):
                        return {"output": "", "error": "Disallowed function call"}
            safe_env = {"__builtins__": {}, "math": math, "sqrt": math.sqrt, "pi": math.pi,
                        "e": math.e, "sin": math.sin, "cos": math.cos, "tan": math.tan,
                        "log": math.log, "log10": math.log10, "ceil": math.ceil, "floor": math.floor,
                        "factorial": math.factorial}
            result = eval(compile(tree, '<calc>', 'eval'), safe_env)
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

    async def _run_file_read(self, path: str) -> Dict[str, Any]:
        """Read a file from the workspace."""
        import os
        try:
            resolved = os.path.abspath(path)
            if not os.path.exists(resolved):
                return {"output": "", "error": f"File not found: {path}"}
            if os.path.getsize(resolved) > 1_000_000:
                return {"output": "", "error": "File too large (>1MB limit)"}
            with open(resolved, "r", errors="replace") as f:
                content = f.read()
            return {"output": content}
        except Exception as e:
            return {"output": "", "error": str(e)}

    async def _run_knowledge_retrieval(self, query: str) -> Dict[str, Any]:
        """Search the knowledge base (docs/*.md, README, etc.)."""
        import os
        import glob
        try:
            workspace = os.environ.get("WORKSPACE_ROOT", ".")
            patterns = ["docs/**/*.md", "README*", "*.md", "CHANGELOG*"]
            files = []
            for p in patterns:
                files.extend(glob.glob(os.path.join(workspace, p), recursive=True))
            matches = []
            query_lower = query.lower()
            for fp in files[:50]:
                try:
                    with open(fp, "r", errors="replace") as f:
                        text = f.read(8192)
                    if query_lower in text.lower():
                        snippet = text[:500].strip()
                        matches.append(f"**{os.path.relpath(fp, workspace)}**:\n{snippet}")
                except Exception:
                    continue
            if matches:
                return {"output": f"Found {len(matches)} relevant documents:\n\n" + "\n\n---\n\n".join(matches[:5])}
            return {"output": f"No documents found matching '{query}'."}
        except Exception as e:
            return {"output": "", "error": str(e)}

    async def _run_image_analysis(self, image_path: str) -> Dict[str, Any]:
        """Analyze an image using VisionCNN — embedding, caption, and object detection."""
        import os
        try:
            if not os.path.exists(image_path):
                return {"output": "", "error": f"Image not found: {image_path}"}
            size = os.path.getsize(image_path)
            ext = os.path.splitext(image_path)[1].lower()

            try:
                from domains.multimodal.vision import VisionCNN
                vision = VisionCNN()
                caption_result = vision.caption(image_path)
                objects = vision.detect(image_path)
                embedding = vision.get_embedding(image_path)

                lines = [
                    f"Image: {os.path.basename(image_path)}",
                    f"Format: {ext or 'unknown'}  |  Size: {size:,} bytes",
                    f"Caption: {caption_result.text}",
                    f"Confidence: {caption_result.confidence:.1%}",
                ]
                if caption_result.tags:
                    lines.append(f"Tags: {', '.join(caption_result.tags)}")
                if objects:
                    lines.append("Objects:")
                    for obj in objects:
                        lines.append(f"  - {obj.label} ({obj.confidence:.1%})")
                lines.append(f"Embedding: {len(embedding)}-dim, L2={float((embedding**2).sum()**0.5):.3f}")
                return {"output": "\n".join(lines)}
            except ImportError:
                return {"output": f"Image: {os.path.basename(image_path)}\nFormat: {ext}\nSize: {size:,} bytes\n\n(Vision model not available)"}
            except Exception as ve:
                return {"output": f"Image: {os.path.basename(image_path)}\nFormat: {ext}\nSize: {size:,} bytes\n\nVision error: {ve}"}
        except Exception as e:
            return {"output": "", "error": str(e)}

    async def _run_data_analysis(self, data_path: str, operation: str = "summary") -> Dict[str, Any]:
        """Analyze a CSV/JSON/JSONL data file."""
        import os
        import json
        try:
            if not os.path.exists(data_path):
                return {"output": "", "error": f"Data file not found: {data_path}"}
            with open(data_path, "r", errors="replace") as f:
                raw = f.read(500_000)
            ext = os.path.splitext(data_path)[1].lower()
            if ext == ".csv":
                lines = raw.strip().split("\n")
                headers = lines[0].split(",") if lines else []
                return {"output": f"CSV: {len(lines)-1} rows, {len(headers)} columns\nHeaders: {', '.join(headers)}\nFirst row: {lines[1] if len(lines) > 1 else '(empty)'}"}
            elif ext in (".json",):
                data = json.loads(raw)
                if isinstance(data, list):
                    return {"output": f"JSON array: {len(data)} items\nFirst item keys: {list(data[0].keys()) if data and isinstance(data[0], dict) else '(not dict)'}"}
                return {"output": f"JSON object: {len(data)} keys\nKeys: {', '.join(list(data.keys())[:20])}"}
            elif ext == ".jsonl":
                lines = raw.strip().split("\n")
                sample = json.loads(lines[0]) if lines else {}
                keys = list(sample.keys()) if isinstance(sample, dict) else []
                return {"output": f"JSONL: {len(lines)} lines\nSample keys: {', '.join(keys)}\nFirst line preview: {lines[0][:300]}"}
            else:
                lines = raw.strip().split("\n")
                return {"output": f"Plain text: {len(lines)} lines, {len(raw)} chars\nFirst line: {lines[0][:200] if lines else '(empty)'}"}
        except Exception as e:
            return {"output": "", "error": str(e)}


_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
