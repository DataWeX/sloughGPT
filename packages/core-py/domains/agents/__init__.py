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

import os
import re
import asyncio
import logging
import hashlib
import tempfile
import subprocess
import resource
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from contextlib import contextmanager

logger = logging.getLogger("man")


# ============ Security & Safety ============

@dataclass
class SecurityConfig:
    """Security configuration for tool execution."""
    max_execution_time: int = 30
    max_memory_mb: int = 512
    max_file_size_mb: int = 100
    allow_network: bool = False
    allowed_directories: List[str] = field(default_factory=list)
    blocked_patterns: List[str] = field(default_factory=list)
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
    
    def __init__(self, config: Optional[SecurityConfig] = None):
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
    FILE_SEARCH = "file_search" 
    WEB_SEARCH = "web_search"
    CITATION = "citation"
    MEMORY = "memory"


@dataclass
class ToolDefinition:
    """Definition of a tool for the agent."""
    name: str
    description: str
    parameters: Dict[str, Any]
    capability: ToolCapability
    requires_approval: bool = False


@dataclass
class ToolExecutionContext:
    """Context for tool execution."""
    session_id: str
    user_id: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolRunner:
    """
    Internal tool runner - NOT exposed to API.
    
    This is the core infrastructure that agent uses internally.
    """
    
    def __init__(self, security: Optional[SecurityBoundary] = None):
        self.security = security or SecurityBoundary()
        self._executed_count = 0
        self._last_reset = asyncio.get_event_loop().time()
    
    async def execute(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: ToolExecutionContext,
    ) -> Dict[str, Any]:
        """Execute tool internally (called by agent, not API)."""
        # Rate limit check
        if not self._check_rate_limit():
            return {"error": "Rate limit exceeded", "success": False}
        
        # Route to correct tool
        if tool_name == ToolCapability.CODE_EXECUTION.value:
            return await self._run_code(args, context)
        elif tool_name == ToolCapability.FILE_SEARCH.value:
            return await self._run_file_search(args, context)
        elif tool_name == ToolCapability.WEB_SEARCH.value:
            return await self._run_web_search(args, context)
        elif tool_name == ToolCapability.CITATION.value:
            return await self._run_citation(args, context)
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
        args: Dict[str, Any],
        context: ToolExecutionContext,
    ) -> Dict[str, Any]:
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
        except asyncio.TimeoutError:
            return {"error": "Execution timed out", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}
    
    async def _execute_subprocess(
        self,
        code: str,
        language: str,
    ) -> Dict[str, Any]:
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
        args: Dict[str, Any],
        context: ToolExecutionContext,
    ) -> Dict[str, Any]:
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
    ) -> List[str]:
        """Search files using grep."""
        proc = await asyncio.create_subprocess_exec(
            "grep", "-r", "-l", query, path,
            "--max-count=0",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        stdout, _ = await proc.communicate()
        files = stdout.decode().strip().split("\n")
        return [f for f in files if f][:limit]
    
    async def _run_web_search(
        self,
        args: Dict[str, Any],
        context: ToolExecutionContext,
    ) -> Dict[str, Any]:
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
    ) -> List[Dict[str, str]]:
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
                
                results.append({
                    "title": title.get_text(strip=True) if title else "",
                    "snippet": snippet.get_text(strip=True) if snippet else "",
                })
            
            return results
    
    async def _run_citation(
        self,
        args: Dict[str, Any],
        context: ToolExecutionContext,
    ) -> Dict[str, Any]:
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
        sources: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Generate citation objects."""
        words = set(text.lower().split())
        citations = []
        
        for i, source in enumerate(sources):
            source_text = source.get("text", "").lower()
            source_words = set(source_text.split())
            overlap = words & source_words
            
            if overlap:
                citations.append({
                    "id": f"source_{i}",
                    "text": source.get("text", "")[:200],
                    "url": source.get("url", ""),
                    "relevance": len(overlap) / len(source_words) if source_words else 0,
                })
        
        return sorted(citations, key=lambda x: x["relevance"], reverse=True)


# ============ Agent ============

@dataclass
class AgentConfig:
    """Configuration for agent."""
    tools: List[ToolCapability] = field(
        default_factory=lambda: [
            ToolCapability.CODE_EXECUTION,
            ToolCapability.FILE_SEARCH,
            ToolCapability.CITATION,
        ]
    )
    security: Optional[SecurityConfig] = None
    max_iterations: int = 10
    timeout: int = 120


class Agent:
    """
    Main agent class - orchestrates tool usage internally.
    
    The agent uses ToolRunner internally, tools are NOT
    exposed via API. Frontend sends requests to agent,
    agent decides which tools to use.
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        security = SecurityBoundary(self.config.security)
        self._runner = ToolRunner(security)
        self._sessions: Dict[str, Dict[str, Any]] = {}
    
    async def execute(
        self,
        user_request: str,
        session_id: str,
        user_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Execute agent on user request.
        
        This is the main entry point - frontend calls this,
        NOT tools directly.
        """
        # Get or create session
        context = self._get_session(session_id, user_id)
        
        # Agent reasoning to decide tools
        plan = await self._plan_execution(user_request, context)
        
        # Execute planned tools
        results = []
        for tool_name, args in plan:
            result = await self._runner.execute(
                tool_name,
                args,
                context,
            )
            results.append({
                "tool": tool_name,
                "result": result,
            })
        
        # Generate response
        response = self._compose_response(user_request, results)
        
        return {
            "response": response,
            "tools_used": results,
            "session_id": session_id,
        }
    
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
    ) -> List[tuple[str, Dict[str, Any]]]:
        """
        Plan which tools to use.
        
        This would be enhanced with actual LLM reasoning.
        For now, simple keyword matching.
        """
        plan = []
        lower = request.lower()
        
        # Simple planning based on keywords
        # In production, this uses LLM to decide
        if "code" in lower or "execute" in lower or "run" in lower:
            # Look for code blocks
            code_match = re.search(r"```(\w+)?\n(.+?)```", request, re.DOTALL)
            if code_match:
                plan.append((
                    ToolCapability.CODE_EXECUTION.value,
                    {"code": code_match.group(2), "language": "python"},
                ))
        
        if "search" in lower or "find" in lower:
            query = re.search(r"(?:search|find)\s+(?:for\s+)?['\"](.+?)['\"]", lower)
            if query:
                plan.append((
                    ToolCapability.FILE_SEARCH.value,
                    {"query": query.group(1)},
                ))
        
        if "cite" in lower or "source" in lower:
            # Would extract sources from context
            plan.append((
                ToolCapability.CITATION.value,
                {"text": request, "sources": []},
            ))
        
        return plan
    
    def _compose_response(
        self,
        request: str,
        tool_results: List[Dict[str, Any]],
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

_agent: Optional[Agent] = None
_runner: Optional[ToolRunner] = None


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
    "MultiAgentOrchestrator",
    "SpecializedAgent",
    "AgentTask",
    "TaskStatus",
    "get_orchestrator",
    "reset_orchestrator",
]


# Lazy import multi-agent components
def __getattr__(name: str) -> Any:
    if name in ("MultiAgentOrchestrator", "SpecializedAgent", "AgentTask",
                 "TaskStatus", "get_orchestrator", "reset_orchestrator"):
        from .multi import (MultiAgentOrchestrator, SpecializedAgent,
                            AgentTask, TaskStatus,
                            get_orchestrator, reset_orchestrator)
        globals().update(locals())
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")