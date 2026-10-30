"""
Agentic Infrastructure - Safe internal tool system.

This module provides the core agent framework:
- Agent: Main orchestration class for agentic AI
- ToolRunner: Internal tool execution (NOT exposed to frontend)
- Security: Sandboxing and safety guards
- AgentContext: Session and state management

Architecture:
    LLM (Agent) → Agent.execute_internal() → ToolRunner → Tools
                                      ↓
                              (tools never exposed via API)
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger("slo.agents")


# ============ Security & Safety ============


@dataclass
class SecurityConfig:
    """Security configuration for tool execution."""

    max_execution_time: int = 30
    max_memory_mb: int = 512
    max_file_size_mb: int = 100
    allow_network: bool = False
    allowed_directories: list[str] = field(default_factory=list)
    blocked_patterns: list[str] = field(default_factory=list)
    rate_limit_per_minute: int = 60


class SecurityBoundary:
    """Security boundaries for safe execution."""

    BLOCKED_PATTERNS = [
        r"import\s+os\s*.*remove",
        r"import\s+shutil\s*.*rmtree",
        r"subprocess\s*\..*shell\s*=\s*True",
        r"__import__\s*\(\s*['\"]os['\"]",
        r"eval\s*\(",
        r"exec\s*\(",
        r"open\s*\([^)]*\br\/w",
        r"pickle",
        r"marshal",
        r"compile\s*\(",
    ]

    ALLOWED_DIRS = [
        "data",
        "models",
        "datasets",
        "temp",
    ]

    def __init__(self, config: SecurityConfig | None = None):
        self.config = config or SecurityConfig()
        self._blocked_re = [re.compile(p) for p in self.BLOCKED_PATTERNS]

    def is_allowed(self, code: str) -> tuple[bool, str]:
        """Check if code is safe to execute."""
        for pattern in self._blocked_re:
            if pattern.search(code):
                return False, f"Blocked pattern: {pattern.pattern}"
        return True, ""

    @contextmanager
    def resource_limit(self, tool: str):
        """Context manager for resource limits."""
        try:
            yield
        finally:
            pass


# ============ Tool System ============


class ToolCapability(Enum):
    """Tool capability levels."""

    CODE_EXECUTION = "code_execution"
    FILE_READ = "file_read"
    FILE_SEARCH = "file_search"
    WEB_SEARCH = "web_search"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    IMAGE_ANALYSIS = "image_analysis"
    DATA_ANALYSIS = "data_analysis"
    CITATION = "citation"


@dataclass
class ToolDefinition:
    """Definition of a tool for the agent."""

    name: str
    description: str
    parameters: dict[str, Any]
    capability: ToolCapability
    requires_approval: bool = False


@dataclass
class ToolExecutionContext:
    """Context for tool execution."""

    session_id: str
    user_id: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolRunner:
    """
    Internal tool runner - NOT exposed to API.

    This is the core infrastructure that agent uses internally.
    """

    def __init__(self, security: SecurityBoundary | None = None):
        self.security = security or SecurityBoundary()
        self._executed_count = 0
        self._last_reset = asyncio.get_event_loop().time()

    async def execute(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        """Execute tool internally (called by agent, not API)."""
        # Rate limit check
        if not self._check_rate_limit():
            return {"error": "Rate limit exceeded", "success": False}

        # Route to correct tool
        if tool_name == ToolCapability.CODE_EXECUTION.value:
            return await self._run_code(args, context)
        elif tool_name == ToolCapability.FILE_READ.value:
            return await self._run_file_read(args, context)
        elif tool_name == ToolCapability.FILE_SEARCH.value:
            return await self._run_file_search(args, context)
        elif tool_name == ToolCapability.WEB_SEARCH.value:
            return await self._run_web_search(args, context)
        elif tool_name == ToolCapability.CITATION.value:
            return await self._run_citation(args, context)
        elif tool_name == ToolCapability.KNOWLEDGE_RETRIEVAL.value:
            return await self._run_knowledge_retrieval(args, context)
        elif tool_name == ToolCapability.IMAGE_ANALYSIS.value:
            return await self._run_image_analysis(args, context)
        elif tool_name == ToolCapability.DATA_ANALYSIS.value:
            return await self._run_data_analysis(args, context)
        else:
            return {"error": f"Unknown tool: {tool_name}", "success": False}

    def _check_rate_limit(self) -> bool:
        """Check rate limits."""
        now = asyncio.get_event_loop().time()
        if now - self._last_reset > 60:
            self._executed_count = 0
            self._last_reset = now

        if self._executed_count >= self.security.config.rate_limit_per_minute:
            return False
        self._executed_count += 1
        return True

    async def _run_code(
        self,
        args: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        """Execute code internally."""
        code = args.get("code", "")
        language = args.get("language", "python")

        # Security check
        safe, msg = self.security.is_allowed(code)
        if not safe:
            return {"error": msg, "success": False}

        # Execute in subprocess with timeout
        try:
            result = await asyncio.wait_for(
                self._execute_subprocess(code, language),
                timeout=self.security.config.max_execution_time,
            )
            return {"success": True, **result}
        except TimeoutError:
            return {"error": "Execution timed out", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _execute_subprocess(
        self,
        code: str,
        language: str,
    ) -> dict[str, Any]:
        """Execute code in subprocess."""
        suffix = ".py" if language == "python" else ".js"

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=suffix,
            delete=False,
        ) as f:
            f.write(code)
            path = f.name

        cmd = ["python3", path] if language == "python" else ["node", path]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await proc.communicate()
            return {
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
                "returncode": proc.returncode,
            }
        finally:
            os.unlink(path)

    async def _run_file_search(
        self,
        args: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        """Search files internally."""
        query = args.get("query", "")
        path = args.get("path", ".")
        limit = args.get("limit", 20)

        if not query:
            return {"error": "query required", "success": False}

        try:
            files = await self._search_files(query, path, limit)
            return {"success": True, "files": files, "count": len(files)}
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _search_files(
        self,
        query: str,
        path: str,
        limit: int,
    ) -> list[str]:
        """Search files using grep.

        No ``--max-count``: GNU grep treats ``-m 0`` as "stop after 0
        matches", which silently made every search return zero files.
        ``--exclude-dir`` keeps .git/node_modules noise (and speed) out.
        """
        proc = await asyncio.create_subprocess_exec(
            "grep",
            "-r",
            "-l",
            "--exclude-dir=.git",
            "--exclude-dir=node_modules",
            "--exclude-dir=__pycache__",
            query,
            path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, _ = await proc.communicate()
        files = stdout.decode().strip().split("\n")
        return [f for f in files if f][:limit]

    async def _run_web_search(
        self,
        args: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        """Search web using libraries."""
        query = args.get("query", "")

        if not query:
            return {"error": "query required", "success": False}

        try:
            results = await self._search_web(query)
            return {"success": True, "results": results, "count": len(results)}
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _search_web(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, str]]:
        """Search web using httpx + BeautifulSoup."""
        import httpx
        from bs4 import BeautifulSoup

        url = "https://html.duckduckgo.com/html/"
        data = {"q": query}

        async with httpx.AsyncClient() as client:
            response = await client.post(url, data=data, timeout=15)
            soup = BeautifulSoup(response.text, "html.parser")

            results = []
            for result in soup.select(".result")[:limit]:
                title = result.select_one(".result__title")
                snippet = result.select_one(".result__snippet")

                results.append(
                    {
                        "title": title.get_text(strip=True) if title else "",
                        "snippet": snippet.get_text(strip=True) if snippet else "",
                    }
                )

            return results

    async def _run_citation(
        self,
        args: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        """Generate citations internally."""
        text = args.get("text", "")
        sources = args.get("sources", [])

        if not text:
            return {"error": "text required", "success": False}

        citations = self._generate_citations(text, sources)
        return {"success": True, "citations": citations, "count": len(citations)}

    def _generate_citations(
        self,
        text: str,
        sources: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Generate citation objects."""
        words = set(text.lower().split())
        citations = []

        for i, source in enumerate(sources):
            source_text = source.get("text", "").lower()
            source_words = set(source_text.split())
            overlap = words & source_words

            if overlap:
                citations.append(
                    {
                        "id": f"source_{i}",
                        "text": source.get("text", "")[:200],
                        "url": source.get("url", ""),
                        "relevance": len(overlap) / len(source_words) if source_words else 0,
                    }
                )

        return sorted(citations, key=lambda x: x["relevance"], reverse=True)

    async def _run_file_read(
        self,
        args: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        """Read a file's contents."""
        path = args.get("path", "")
        if not path:
            return {"error": "path required", "success": False}
        try:
            if not os.path.exists(path):
                return {"error": f"File not found: {path}", "success": False}
            if os.path.getsize(path) > 1_000_000:
                return {"error": "File too large (>1MB)", "success": False}
            with open(path, errors="replace") as f:
                content = f.read()
            return {"success": True, "content": content}
        except Exception as e:
            return {"error": str(e), "success": False}

    def _query_knowledge_engine(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Query the canonical KnowledgeEngine RAG store.

        Returns normalized fact dicts; empty list on any failure so callers
        can degrade to file-based fallback instead of erroring the tool.
        """
        try:
            from domain.knowledge import get_knowledge_engine

            result = get_knowledge_engine().query(query, top_k=limit)
            if not result.success or not result.data:
                return []
            matches = []
            for fact in result.data:
                matches.append(
                    {
                        "content": fact.get("content", ""),
                        "topic": fact.get("topic", "general"),
                        "source": fact.get("source") or fact.get("url") or fact.get("path") or "",
                        "score": round(float(fact.get("score", 0.0)), 4),
                    }
                )
            return matches
        except Exception as e:
            logger.debug("Knowledge engine query failed, falling back: %s", e)
            return []

    async def _run_knowledge_retrieval(
        self,
        args: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        """Search knowledge base for relevant information."""
        query = args.get("query", "")
        if not query:
            return {"error": "query required", "success": False}
        rag_matches = self._query_knowledge_engine(query)
        if rag_matches:
            return {
                "success": True,
                "matches": rag_matches,
                "count": len(rag_matches),
                "engine": "rag",
            }
        try:
            import glob as globmod

            workspace = os.environ.get("WORKSPACE_ROOT", ".")
            patterns = ["docs/**/*.md", "README*", "*.md"]
            files = []
            for p in patterns:
                files.extend(globmod.glob(os.path.join(workspace, p), recursive=True))
            matches = []
            query_lower = query.lower()
            for fp in files[:50]:
                try:
                    with open(fp, errors="replace") as f:
                        text = f.read(8192)
                    if query_lower in text.lower():
                        matches.append(
                            {
                                "file": os.path.relpath(fp, workspace),
                                "snippet": text[:500].strip(),
                            }
                        )
                except Exception:
                    continue
            return {
                "success": True,
                "matches": matches,
                "count": len(matches),
                "engine": "files",
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _run_image_analysis(
        self,
        args: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        """Analyze an image using VisionCNN — embedding, caption, and object detection."""
        image_path = args.get("image_path", "")
        if not image_path:
            return {"error": "image_path required", "success": False}
        try:
            if not os.path.exists(image_path):
                return {"error": f"Image not found: {image_path}", "success": False}

            size = os.path.getsize(image_path)
            ext = os.path.splitext(image_path)[1].lower()

            try:
                from domain.multimodal._internal.vision import VisionCNN

                vision = VisionCNN()
                caption_result = vision.caption(image_path)
                objects = vision.detect(image_path)
                embedding = vision.get_embedding(image_path)

                return {
                    "success": True,
                    "format": ext,
                    "size_bytes": size,
                    "caption": caption_result.text,
                    "confidence": round(caption_result.confidence, 3),
                    "tags": caption_result.tags,
                    "objects": [
                        {"label": obj.label, "confidence": round(obj.confidence, 3)}
                        for obj in objects
                    ],
                    "embedding_dim": len(embedding),
                    "embedding_stats": {
                        "mean": round(float(embedding.mean()), 4),
                        "std": round(float(embedding.std()), 4),
                        "l2_norm": round(float((embedding**2).sum() ** 0.5), 4),
                    },
                }
            except ImportError:
                return {
                    "success": True,
                    "format": ext,
                    "size_bytes": size,
                    "note": "Vision model not available — install SloNet dependencies",
                }
            except Exception as ve:
                logger.warning("VisionCNN analysis failed, falling back to metadata: %s", ve)
                return {
                    "success": True,
                    "format": ext,
                    "size_bytes": size,
                    "note": f"Vision analysis error: {ve}",
                }
        except Exception as e:
            return {"error": str(e), "success": False}

    async def _run_data_analysis(
        self,
        args: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        """Analyze a data file (CSV/JSON/JSONL)."""
        data_path = args.get("data_path", "")
        if not data_path:
            return {"error": "data_path required", "success": False}
        try:
            if not os.path.exists(data_path):
                return {"error": f"Data file not found: {data_path}", "success": False}
            with open(data_path, errors="replace") as f:
                raw = f.read(500_000)
            ext = os.path.splitext(data_path)[1].lower()
            if ext == ".csv":
                lines = raw.strip().split("\n")
                headers = lines[0].split(",") if lines else []
                return {
                    "success": True,
                    "rows": len(lines) - 1,
                    "columns": len(headers),
                    "headers": headers,
                }
            elif ext in (".json", ".jsonl"):
                import json

                data = (
                    json.loads(raw)
                    if ext == ".json"
                    else [json.loads(line) for line in raw.strip().split("\n") if line]
                )
                if isinstance(data, list):
                    keys = list(data[0].keys()) if data and isinstance(data[0], dict) else []
                    return {"success": True, "items": len(data), "keys": keys}
                return {"success": True, "keys": list(data.keys())}
            return {"success": True, "lines": len(raw.strip().split("\n")), "chars": len(raw)}
        except Exception as e:
            return {"error": str(e), "success": False}


# ============ Agent ============


@dataclass
class AgentConfig:
    """Configuration for agent."""

    tools: list[ToolCapability] = field(
        default_factory=lambda: [
            ToolCapability.CODE_EXECUTION,
            ToolCapability.FILE_READ,
            ToolCapability.WEB_SEARCH,
            ToolCapability.KNOWLEDGE_RETRIEVAL,
        ]
    )
    security: SecurityConfig | None = None
    max_iterations: int = 10
    timeout: int = 120
    instructions: str = ""


class Agent:
    """
    Main agent class - orchestrates tool usage internally.

    The agent uses ToolRunner internally, tools are NOT
    exposed via API. Frontend sends requests to agent,
    agent decides which tools to use.

    When an ``inference_fn`` is provided, ``_plan_execution`` uses LLM-based
    reasoning to plan tool usage. Falls back to keyword matching otherwise.
    """

    def __init__(self, config: AgentConfig | None = None, inference_fn: Callable | None = None):
        self.config = config or AgentConfig()
        security = SecurityBoundary(self.config.security)
        self._runner = ToolRunner(security)
        self._inference_fn = inference_fn
        self._sessions: dict[str, dict[str, Any]] = {}

    def set_inference_fn(self, fn: Callable) -> None:
        """Set or replace the inference function for LLM planning."""
        self._inference_fn = fn

    async def execute(
        self,
        user_request: str,
        session_id: str,
        user_id: str = "default",
    ) -> dict[str, Any]:
        """
        Execute agent on user request.

        This is the main entry point - frontend calls this,
        NOT tools directly.

        Consumes ``execute_stream`` — one loop implementation for both the
        dict API and the SSE API.
        """
        tools_used: list[dict[str, Any]] = []
        response = ""
        async for event in self.execute_stream(user_request, session_id, user_id):
            kind = event.get("event")
            if kind == "tool_result":
                tools_used.append({"tool": event["tool"], "result": event["result"]})
            elif kind == "final":
                response = event.get("text", "")
            elif kind == "error":
                raise RuntimeError(event.get("error", "agent run failed"))
        if not response:
            response = self._compose_response(user_request, tools_used)
        return {
            "response": response,
            "tools_used": tools_used,
            "session_id": session_id,
        }

    async def execute_stream(
        self,
        user_request: str,
        session_id: str,
        user_id: str = "default",
    ):
        """Yield step events for one bounded model⇄tool loop run.

        Events: ``run_start`` → (``tool_start`` → ``tool_result``)* →
        ``final`` → ``run_end``; ``error`` on failure. The loop lets the
        model observe tool results before deciding the next step, and every
        run is recorded in the shared run store.
        """
        from domain.agents._internal.run_history import get_agent_run_store

        context = self._get_session(session_id, user_id)
        store = None
        run_id = None
        try:
            store = get_agent_run_store()
            run_id = store.start(user_request, session_id)
        except Exception as e:
            logger.debug("Run persistence unavailable: %s", e, extra={"tag": "AGENT"})
            store = None

        yield {"event": "run_start", "run_id": run_id, "session_id": session_id}

        observations: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        final_text = ""
        budget = max(1, int(self.config.max_iterations))
        try:
            for step in range(budget):
                plan = await self._plan_next(user_request, context, observations)
                if not plan:
                    break
                for tool_name, args in plan:
                    yield {
                        "event": "tool_start",
                        "tool": tool_name,
                        "args": args,
                        "step": step,
                        "run_id": run_id,
                    }
                    try:
                        result = await self._runner.execute(tool_name, args, context)
                    except Exception as e:
                        result = {"success": False, "error": str(e)}
                    summary = _result_summary(result)
                    observations.append({"tool": tool_name, "args": args, "summary": summary})
                    tool_results.append({"tool": tool_name, "result": result})
                    if store is not None and run_id is not None:
                        try:
                            store.append_log(run_id, f"step {step}: {tool_name} → {summary[:180]}")
                        except Exception as e:
                            logger.debug("append_log failed: %s", e, extra={"tag": "AGENT"})
                    yield {
                        "event": "tool_result",
                        "tool": tool_name,
                        "result": result,
                        "step": step,
                        "run_id": run_id,
                    }
            final_text = await self._synthesize(user_request, context, observations, tool_results)
            yield {"event": "final", "text": final_text, "run_id": run_id}
        except Exception as e:
            if store is not None and run_id is not None:
                try:
                    store.fail(run_id, str(e))
                except Exception:
                    pass
            yield {"event": "error", "error": str(e), "run_id": run_id}
            raise
        else:
            if store is not None and run_id is not None:
                try:
                    store.complete(run_id, response=final_text)
                except Exception as e:
                    logger.debug("complete failed: %s", e, extra={"tag": "AGENT"})
        yield {"event": "run_end", "run_id": run_id, "session_id": session_id}

    async def _plan_next(
        self,
        request: str,
        context: ToolExecutionContext,
        observations: list[dict[str, Any]],
    ) -> list[tuple[str, dict[str, Any]]]:
        """Decide the next tools given what has run so far (bounded loop).

        With inference the model re-plans after every batch of observations
        (``[]`` = enough information, synthesize). The bootstrap model plans
        ``[]`` even for explicit tool requests (0/5 selection live
        2026-10-30), so on step 0 an empty model plan falls back to the
        keyword matcher — conservative patterns only (quoted searches,
        code blocks) — until SFT tool-use training lands.
        """
        if self._inference_fn:
            plan = await self._plan_with_llm(request, observations)
            if plan:
                return plan
            if not observations:
                return self._plan_with_keywords(request)
            return []
        if observations:
            return []
        return self._plan_with_keywords(request)

    def _generate_response(self, request: str, context: ToolExecutionContext) -> str:
        """Generate a response via LLM when no tools are needed."""
        try:
            agent_instructions = getattr(self.config, "instructions", "")
            system = (
                f"You are an AI assistant.\n{agent_instructions}"
                if agent_instructions
                else "You are an AI assistant."
            )
            prompt = f"{system}\n\nUser: {request}\nAssistant:"
            result = self._inference_fn(prompt)
            if isinstance(result, dict) and "text" in result:
                return result["text"]
            if isinstance(result, str):
                return result
            return str(result)
        except Exception as e:
            logger.warning("Inference failed: %s", e, extra={"tag": "MODEL"})
            return self._compose_response(request, [])

    def _get_session(
        self,
        session_id: str,
        user_id: str,
    ) -> ToolExecutionContext:
        """Get or create session context."""
        import time

        return ToolExecutionContext(
            session_id=f"{user_id}:{session_id}",
            user_id=user_id,
            timestamp=time.time(),
        )

    async def _plan_execution(
        self,
        request: str,
        context: ToolExecutionContext,
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Plan which tools to use.

        Compatibility shim — the loop uses ``_plan_next`` (one planner path).
        """
        return await self._plan_next(request, context, [])

    def _plan_with_keywords(
        self,
        request: str,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Plan tools using simple keyword matching."""
        plan = []
        lower = request.lower()

        if "code" in lower or "execute" in lower or "run" in lower:
            code_match = re.search(r"```(\w+)?\n(.+?)```", request, re.DOTALL)
            if code_match:
                plan.append(
                    (
                        ToolCapability.CODE_EXECUTION.value,
                        {"code": code_match.group(2), "language": "python"},
                    )
                )

        if "search" in lower or "find" in lower or "match" in lower:
            # Any quoted span works as the query — the earlier pattern only
            # fired for `search for "x"` and missed `search for files matching "x"`.
            query = re.search(r"[\"'](.+?)[\"']", request)
            if query:
                plan.append(
                    (
                        ToolCapability.FILE_SEARCH.value,
                        {"query": query.group(1)},
                    )
                )

        if "web" in lower or "online" in lower or "internet" in lower or "browser" in lower:
            query = re.search(r"['\"](.+?)['\"]", lower)
            if query:
                plan.append(
                    (
                        ToolCapability.WEB_SEARCH.value,
                        {"query": query.group(1)},
                    )
                )

        if "cite" in lower or "citation" in lower or "source" in lower:
            cite_match = re.search(r"(?:cite|citation)\s+(?:your\s+)?(?:sources?\s+)?(.+)$", lower)
            if cite_match:
                plan.append(
                    (
                        ToolCapability.CITATION.value,
                        {"text": cite_match.group(1).strip()},
                    )
                )

        if "read" in lower or "open" in lower or "show" in lower:
            path_match = re.search(r"(?:read|open|show)\s+(?:file\s+)?['\"](.+?)['\"]", lower)
            if path_match:
                plan.append(
                    (
                        ToolCapability.FILE_READ.value,
                        {"path": path_match.group(1)},
                    )
                )

        if "knowledge" in lower or "recall" in lower:
            query_match = re.search(r"(?:knowledge|recall)\s+(.+)$", lower)
            if query_match:
                plan.append(
                    (
                        ToolCapability.KNOWLEDGE_RETRIEVAL.value,
                        {"query": query_match.group(1).strip()},
                    )
                )

        if "analyze" in lower or "data" in lower or "stats" in lower:
            data_match = re.search(r"(?:analyze|stats|data)\s+(.+)$", lower)
            if data_match:
                plan.append(
                    (
                        ToolCapability.DATA_ANALYSIS.value,
                        {"data_path": data_match.group(1).strip()},
                    )
                )

        if "image" in lower or "photo" in lower or "picture" in lower:
            img_match = re.search(r"(?:image|photo|picture)\s+(.+)$", lower)
            if img_match:
                plan.append(
                    (
                        ToolCapability.IMAGE_ANALYSIS.value,
                        {"image_path": img_match.group(1).strip()},
                    )
                )

        return plan

    async def _plan_with_llm(
        self,
        request: str,
        observations: list[dict[str, Any]] | None = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        """Plan (or re-plan) tools using LLM reasoning.

        With ``observations`` the model sees what already ran and decides the
        next step — ``[]`` means "enough information, synthesize now".
        """
        allowed = [t.value for t in self.config.tools] or [c.value for c in ToolCapability]
        # The 0.5B bootstrap model treats a bare "decide which tools" question
        # as chit-chat and refuses/loops (live journey 2026-10-30). A tiny
        # format example + strict format line raises JSON compliance without
        # any training — the SFT lever stays carded for real quality.
        prompt = (
            "You are a tool-calling assistant. Pick tools for the request.\n"
            f"Available tools: {', '.join(allowed)}\n"
            "Respond with ONLY one JSON array, no prose. Examples:\n"
            '  [{"tool": "file_search", "args": {"query": "run_history"}}]\n'
            "  []\n"
            "Use [] when no tool is needed.\n"
            f"User request: {request}\n"
        )
        if observations:
            prompt += (
                "You already ran tools. Observations:\n"
                f"{json.dumps(observations, ensure_ascii=False)[:4000]}\n"
                "Decide the NEXT step now: another JSON array of tools to run, "
                "or [] if you have enough information to answer.\n"
            )
        prompt += (
            "Return a JSON array of {'tool': str, 'args': dict} objects, "
            "or an empty array [] if no tools are needed.\n"
            "Return ONLY the JSON, no other text."
        )
        try:
            resp = await asyncio.to_thread(self._inference_fn, prompt)
            text = resp.get("text", "") if isinstance(resp, dict) else str(resp)
            text = text.strip()
            # Extract JSON array
            match = re.search(r"\[.*?\]", text, re.DOTALL)
            if match:
                span = match.group()
                try:
                    plan = json.loads(span)
                except json.JSONDecodeError:
                    # The 0.5B model often emits single-quoted Python-style
                    # dicts — literal_eval is safe (no exec) and recovers it.
                    plan = ast.literal_eval(span)
            else:
                return []
            return [
                (item["tool"], item.get("args", {}))
                for item in plan
                if isinstance(item, dict) and "tool" in item
            ]
        except Exception as e:
            logger.warning("LLM planning failed: %s", e, extra={"tag": "MODEL"})
            # Never keyword-replan mid-loop: it would re-run the same tools.
            if observations:
                return []
            return self._plan_with_keywords(request)

    async def _synthesize(
        self,
        request: str,
        context: ToolExecutionContext,
        observations: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
    ) -> str:
        """Produce the final answer: model reads observations, compose is fallback."""
        if self._inference_fn:
            try:
                if observations:
                    system = getattr(self.config, "instructions", "") or "You are an AI assistant."
                    prompt = (
                        f"{system}\n\n"
                        f"User: {request}\n"
                        "Tool observations:\n"
                        f"{json.dumps(observations, ensure_ascii=False)[:6000]}\n"
                        "Write the final answer for the user based on these observations. "
                        "One answer only: never repeat a phrase or sentence, and do not "
                        "mention your tools or limits."
                    )
                    resp = await asyncio.to_thread(self._inference_fn, prompt)
                    text = resp.get("text", "") if isinstance(resp, dict) else str(resp)
                    if text.strip():
                        from domain.agents._internal.run_history import looks_like_refusal

                        # Bootstrap-model symptoms: canned refusals and
                        # phrase loops make bad training/UX — compose is honest.
                        if looks_like_refusal(text):
                            logger.debug("Synthesis refusal, composing", extra={"tag": "MODEL"})
                        elif _looks_degenerate(text):
                            logger.debug("Synthesis loop, composing", extra={"tag": "MODEL"})
                        else:
                            return text.strip()
                else:
                    return await asyncio.to_thread(self._generate_response, request, context)
            except Exception as e:
                logger.warning("Synthesis failed: %s", e, extra={"tag": "MODEL"})
        return self._compose_response(request, tool_results)

    def _compose_response(
        self,
        request: str,
        tool_results: list[dict[str, Any]],
    ) -> str:
        """Compose final response from tool results."""
        outputs = []

        for tr in tool_results:
            result = tr.get("result", {})
            if result.get("success"):
                if "stdout" in result:
                    outputs.append(result["stdout"])
                elif "files" in result:
                    outputs.append(f"Found {result['count']} files")
                elif "results" in result:
                    outputs.append(f"Found {result['count']} results")
                elif "citations" in result:
                    outputs.append(f"Generated {result['count']} citations")
            else:
                outputs.append(f"Error: {result.get('error', 'Unknown')}")

        return "\n".join(outputs) if outputs else "No results"


# ============ Singleton Instances ============


def _result_summary(result: Any) -> str:
    """One-line human/model summary of a tool result (loop observation)."""
    if not isinstance(result, dict):
        return str(result)[:300]
    if not result.get("success", True):
        return f"error: {result.get('error', 'unknown')}"
    for key in ("stdout", "answer", "text", "results", "files", "citations"):
        if key in result:
            value = result[key]
            if isinstance(value, str):
                return value[:300]
            return f"{key}: {str(value)[:300]}"
    return str(result)[:300]


def _looks_degenerate(text: str) -> bool:
    """True when output loops — the 0.5B repeats a phrase 3x in a row.

    Checked from long chunks down to bigrams: live runs showed both
    sentence-level loops and tails like "artificial intelligence, ..."
    three times over. A legit answer never repeats any chunk exactly 3x
    consecutively, so short answers simply can't trip it.
    """
    words = text.split()
    if len(words) < 8:
        return False
    for n in (6, 5, 4, 3, 2):
        chunk_len = 3 * n
        for i in range(len(words) - chunk_len + 1):
            chunk = words[i : i + n]
            if words[i + n : i + 2 * n] == chunk and words[i + 2 * n : i + chunk_len] == chunk:
                return True
    return False


_agent: Agent | None = None
_runner: ToolRunner | None = None


def get_agent() -> Agent:
    """Get or create agent singleton."""
    global _agent
    if _agent is None:
        _agent = Agent()
    return _agent


def get_runner() -> ToolRunner:
    """Get or create tool runner singleton."""
    global _runner
    if _runner is None:
        _runner = ToolRunner()
    return _runner


__all__ = [
    "Agent",
    "AgentConfig",
    "ToolRunner",
    "ToolCapability",
    "ToolDefinition",
    "ToolExecutionContext",
    "SecurityConfig",
    "SecurityBoundary",
    "get_agent",
    "get_runner",
    "MultiAgentOrchestrator",  # noqa: F822
    "SpecializedAgent",  # noqa: F822
    "AgentTask",  # noqa: F822
    "TaskStatus",  # noqa: F822
    "get_orchestrator",  # noqa: F822
    "reset_orchestrator",  # noqa: F822
]


# Lazy import multi-agent components
def __getattr__(name: str) -> Any:
    if name in (
        "MultiAgentOrchestrator",
        "SpecializedAgent",
        "AgentTask",
        "TaskStatus",
        "get_orchestrator",
        "reset_orchestrator",
    ):
        from .multi import (  # noqa: F401
            AgentTask,
            MultiAgentOrchestrator,
            SpecializedAgent,
            TaskStatus,
            get_orchestrator,
            reset_orchestrator,
        )

        globals().update(locals())
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
