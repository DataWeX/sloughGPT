"""
AgentEngine — reasoning loop with tool calling on SloNet/SloEngine.

Architecture:
  1. User prompt → system prompt with tool descriptions
  2. Model generates a response (text or tool call)
  3. If tool call → execute tool → feed result back to model
  4. Model generates final response → return to user

Tools are registered with a name, description, parameter schema, and callable.
The model can invoke them by outputting structured text like:

  TOOL_CALL: tool_name
  args: {"key": "value"}

The agent loop handles routing these back through the model.
"""

from __future__ import annotations

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("man.agent")


@dataclass
class Tool:
    """A tool the agent can call."""
    name: str
    description: str
    fn: Callable
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {},
    })


@dataclass
class AgentRun:
    """Record of a single agent invocation."""
    prompt: str
    response: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[Dict[str, str]] = field(default_factory=list)
    elapsed_ms: float = 0.0
    error: Optional[str] = None


class AgentEngine:
    """Reasoning loop with tool calling.

    Wraps any generate()-compatible engine (SloEngine, HF model, etc.)
    and adds an agentic loop: generate → parse tool call → execute → repeat.
    """

    def __init__(self, engine, tools: Optional[List[Tool]] = None):
        self._engine = engine
        self._tools: Dict[str, Tool] = {}
        self._max_steps = 6
        self._session_memory: List[Dict[str, str]] = []

        if tools:
            for t in tools:
                self.register_tool(t)

    # ------------------------------------------------------------------
    # Tool registry
    # ------------------------------------------------------------------

    def register_tool(self, tool: Tool) -> None:
        """Register a tool the agent can call."""
        self._tools[tool.name] = tool
        logger.info(f"Agent tool registered: {tool.name}")

    def unregister_tool(self, name: str) -> None:
        self._tools.pop(name, None)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    # ------------------------------------------------------------------
    # Agent loop
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Build system prompt describing available tools."""
        lines = [
            "You are an AI agent with access to tools.",
            "When you need to use a tool, respond with exactly:",
            "TOOL_CALL: tool_name",
            'args: {"key": "value"}',
            "",
            "Available tools:",
        ]
        for t in self._tools.values():
            lines.append(f"  - {t.name}: {t.description}")
        lines.append("")
        lines.append("After receiving the tool result, respond naturally to the user.")
        return "\n".join(lines)

    def run(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7) -> AgentRun:
        """Execute the agent loop: generate → tool call → generate → ..."""
        start = time.perf_counter()
        run = AgentRun(prompt=prompt)

        # Build context with session memory
        memory_text = ""
        if self._session_memory:
            memory_text = "\n".join(
                f"{m['role']}: {m['content']}" for m in self._session_memory[-6:]
            ) + "\n"

        system = self._build_system_prompt()
        full_prompt = f"{system}\n\n{memory_text}User: {prompt}\nAssistant:"

        current_prompt = full_prompt
        step = 0

        while step < self._max_steps:
            step += 1
            response = self._engine.generate(
                current_prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                include_reasoning=False,
            )

            # Clean response
            for prefix in ["[SOUL_REASONING]", "[REASONING_CHAIN]", "System:", "You are"]:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()

            # Check for tool call
            if "TOOL_CALL:" in response:
                try:
                    tool_name, args_str = self._parse_tool_call(response)
                    run.tool_calls.append({"tool": tool_name, "args": args_str})

                    tool = self._tools.get(tool_name)
                    if tool is None:
                        result = f"Error: tool '{tool_name}' not found"
                    else:
                        try:
                            args = json.loads(args_str) if args_str else {}
                            result = tool.fn(**args)
                            result = str(result)
                        except Exception as e:
                            result = f"Error calling {tool_name}: {e}"

                    run.steps.append({"tool": tool_name, "args": args_str, "result": result[:200]})
                    current_prompt = f"{current_prompt}\n{response}\nTOOL_RESULT: {result}\nAssistant:"
                except Exception as e:
                    run.error = str(e)
                    break
            else:
                # No tool call — this is the final response
                run.response = response
                break

        self._session_memory.append({"role": "user", "content": prompt})
        self._session_memory.append({"role": "assistant", "content": run.response})
        run.elapsed_ms = (time.perf_counter() - start) * 1000

        if step >= self._max_steps and not run.response:
            run.response = "[Agent reached max steps without final response]"

        return run

    def _parse_tool_call(self, text: str) -> tuple:
        """Extract tool name and args from model output."""
        lines = text.strip().split("\n")
        tool_line = ""
        args_str = "{}"
        for i, line in enumerate(lines):
            if line.startswith("TOOL_CALL:"):
                tool_line = line.replace("TOOL_CALL:", "").strip()
                # Look for args on next line
                for j in range(i + 1, min(i + 3, len(lines))):
                    if lines[j].startswith("args:"):
                        args_str = lines[j].replace("args:", "").strip()
                        break
                break
        return tool_line, args_str

    def reset_memory(self) -> None:
        """Clear session memory."""
        self._session_memory = []

    def status(self) -> Dict[str, Any]:
        return {
            "tools": self.list_tools(),
            "memory_size": len(self._session_memory),
            "max_steps": self._max_steps,
        }
