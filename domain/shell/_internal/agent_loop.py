"""
AgentLoop — interactive agent loop for the shell REPL.

Plans shell commands via LLM, executes with user approval, loops until done.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("slo.shell.agent")

# Tool definitions the LLM can plan to use
SHELL_TOOLS = [
    {
        "name": "run_command",
        "description": "Execute a shell command and return its output",
        "parameters": {
            "command": "The shell command to execute",
            "explanation": "Why this command is needed",
        },
    },
    {
        "name": "read_file",
        "description": "Read a file's contents",
        "parameters": {
            "path": "File path to read",
            "explanation": "Why this file needs to be read",
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file (creates or overwrites)",
        "parameters": {
            "path": "File path to write",
            "content": "Content to write",
            "explanation": "Why this file needs to be written",
        },
    },
    {
        "name": "edit_file",
        "description": "Replace a section in a file",
        "parameters": {
            "path": "File path to edit",
            "old_text": "Text to find and replace",
            "new_text": "Replacement text",
            "explanation": "Why this edit is needed",
        },
    },
    {
        "name": "search_code",
        "description": "Search for patterns in the codebase",
        "parameters": {
            "pattern": "Regex pattern to search for",
            "path": "Directory to search in (default: current)",
            "explanation": "What we're looking for",
        },
    },
    {
        "name": "answer",
        "description": "Final answer — task is complete",
        "parameters": {
            "message": "The final answer or summary",
        },
    },
]

TOOL_SCHEMAS = json.dumps(SHELL_TOOLS, indent=2)

SYSTEM_PROMPT = """You are an AI coding agent running in a shell. You can execute commands, read/write files, and search code.

You have access to these tools:
{tools}

RULES:
1. Think step-by-step about what needs to be done.
2. Use tools one at a time. Each tool call is a JSON object.
3. After each tool result, decide the next step.
4. When the task is complete, use the "answer" tool.
5. Always explain what you're doing and why.
6. For destructive operations (rm, overwrite, etc.), be explicit.
7. If unsure, ask the user for clarification using "answer".

Respond with EXACTLY ONE JSON object per turn:
{{"tool": "tool_name", "args": {{"key": "value", ...}}, "thinking": "optional thought"}}

Or for final answer:
{{"tool": "answer", "args": {{"message": "your summary"}}}}
"""


class AgentLoop:
    """Interactive agent loop with tool approval.

    Usage:
        loop = AgentLoop(repl, generate_fn)
        loop.run("fix the bug in main.py")
    """

    def __init__(
        self,
        repl: Any,
        generate_fn: Callable[[str], str],
        max_iterations: int = 15,
        auto_approve: bool = False,
    ):
        self.repl = repl
        self.generate_fn = generate_fn
        self.max_iterations = max_iterations
        self.auto_approve = auto_approve
        self._history: list[dict] = []
        self._iteration = 0

    def run(self, user_request: str) -> str:
        """Run the agent loop. Returns the final answer."""
        self._history = []
        self._iteration = 0

        system = SYSTEM_PROMPT.replace("{tools}", TOOL_SCHEMAS)
        self._history.append({"role": "system", "content": system})
        self._history.append({"role": "user", "content": user_request})

        self.repl._print(f"\n  {_C_BOLD}Agent loop started{_C_RESET}")
        self.repl._print(f"  Request: {user_request}")
        self.repl._print(f"  Max iterations: {self.max_iterations}")
        self.repl._print(
            f"  Type {_C_YELLOW}yes{_C_RESET} to approve, {_C_YELLOW}no{_C_RESET} to skip, {_C_YELLOW}quit{_C_RESET} to abort\n"
        )

        final_answer = ""

        while self._iteration < self.max_iterations:
            self._iteration += 1
            self.repl._print(
                f"  {_C_DIM}── iteration {self._iteration}/{self.max_iterations} ──{_C_RESET}"
            )

            # Get LLM plan
            plan = self._get_plan()
            if plan is None:
                self.repl._print(f"  {_C_RED}Failed to get plan from LLM{_C_RESET}")
                break

            tool = plan.get("tool", "")
            args = plan.get("args", {})
            thinking = plan.get("thinking", "")

            if thinking:
                self.repl._print(f"  {_C_CYAN}Thinking:{_C_RESET} {thinking}")

            # Handle final answer
            if tool == "answer":
                final_answer = args.get("message", "Done.")
                self.repl._print(f"\n  {_C_GREEN}Agent complete{_C_RESET}")
                self.repl._print(f"  {final_answer}\n")
                break

            # Show the planned action
            self._show_plan(tool, args)

            # Get approval
            approved = self._get_approval(tool, args)
            if approved is None:
                self.repl._print(f"  {_C_DIM}Agent aborted by user{_C_RESET}")
                break

            # Execute
            result = (
                self._execute_tool(tool, args) if approved else {"success": False, "skipped": True}
            )

            # Feed result back
            result_str = json.dumps(result, default=str)
            self._history.append({"role": "assistant", "content": json.dumps(plan)})
            self._history.append({"role": "user", "content": f"Tool result:\n{result_str}"})

            # Show result
            if approved:
                output = result.get("output", result.get("error", str(result)))
                if output:
                    # Truncate long output
                    lines = output.strip().split("\n")
                    if len(lines) > 30:
                        truncated = (
                            "\n".join(lines[:15])
                            + f"\n  {_C_DIM}... ({len(lines) - 30} lines omitted) ...{_C_RESET}\n"
                            + "\n".join(lines[-15:])
                        )
                        self.repl._print(f"  Output ({len(lines)} lines):\n{truncated}")
                    else:
                        self.repl._print(f"  Output:\n{output}")
            else:
                self.repl._print(f"  {_C_DIM}Skipped{_C_RESET}")

            self.repl._print("")

        if self._iteration >= self.max_iterations and not final_answer:
            self.repl._print(f"  {_C_YELLOW}Max iterations reached{_C_RESET}")
            final_answer = "Max iterations reached."

        return final_answer

    def _get_plan(self) -> dict | None:
        """Get the LLM's next plan."""
        prompt = "\n".join(m["content"] for m in self._history)
        try:
            response = self.generate_fn(prompt)
        except Exception as e:
            logger.warning("LLM generation failed: %s", e)
            return None

        if not response:
            return None

        # Parse JSON from response
        response = response.strip()
        # Try to find JSON object in the response
        match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Try entire response as JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Fallback: treat as a command
        return {
            "tool": "run_command",
            "args": {"command": response},
            "thinking": "Treating response as command",
        }

    def _show_plan(self, tool: str, args: dict) -> None:
        """Display the planned action."""
        color = _C_YELLOW if tool in ("write_file", "edit_file") else _C_CYAN
        self.repl._print(f"  {color}Plan: {tool}{_C_RESET}")

        if tool == "run_command":
            self.repl._print(f"    Command: {args.get('command', '?')}")
        elif tool == "read_file":
            self.repl._print(f"    File: {args.get('path', '?')}")
        elif tool == "write_file":
            path = args.get("path", "?")
            content = args.get("content", "")
            lines = content.count("\n") + 1
            self.repl._print(f"    File: {path} ({lines} lines)")
        elif tool == "edit_file":
            self.repl._print(f"    File: {args.get('path', '?')}")
            old = args.get("old_text", "")[:60]
            new = args.get("new_text", "")[:60]
            self.repl._print(f"    Find:    {old}...")
            self.repl._print(f"    Replace: {new}...")
        elif tool == "search_code":
            self.repl._print(f"    Pattern: {args.get('pattern', '?')}")
            self.repl._print(f"    Path:    {args.get('path', '.')}")

    def _get_approval(self, tool: str, args: dict) -> bool | None:
        """Ask user for approval. Returns True/False/None (abort)."""
        if self.auto_approve:
            return True

        # Safe tools auto-approve
        if tool in ("read_file", "search_code", "answer"):
            return True

        try:
            self.repl.io.write(f"  {_C_YELLOW}Approve?{_C_RESET} [y/n/quit]: ", end="")
            answer = self.repl.io.read("").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None

        if answer in ("quit", "q", "abort"):
            return None
        if answer in ("y", "yes", "ok", "approve"):
            return True
        return False

    def _execute_tool(self, tool: str, args: dict) -> dict:
        """Execute a tool and return the result."""
        try:
            if tool == "run_command":
                return self._exec_command(args.get("command", ""))
            elif tool == "read_file":
                return self._exec_read_file(args.get("path", ""))
            elif tool == "write_file":
                return self._exec_write_file(args.get("path", ""), args.get("content", ""))
            elif tool == "edit_file":
                return self._exec_edit_file(
                    args.get("path", ""),
                    args.get("old_text", ""),
                    args.get("new_text", ""),
                )
            elif tool == "search_code":
                return self._exec_search(args.get("pattern", ""), args.get("path", "."))
            else:
                return {"success": False, "error": f"Unknown tool: {tool}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _exec_command(self, command: str) -> dict:
        """Execute a shell command via the REPL."""
        if not command.strip():
            return {"success": False, "error": "Empty command"}

        # Use the REPL's execute for proper capture
        output, exit_code = self.repl.execute(command)
        return {
            "success": exit_code == 0,
            "output": output,
            "exit_code": exit_code,
        }

    def _exec_read_file(self, path: str) -> dict:
        """Read a file."""
        from pathlib import Path

        p = Path(path).expanduser()
        if not p.exists():
            return {"success": False, "error": f"File not found: {path}"}
        if not p.is_file():
            return {"success": False, "error": f"Not a file: {path}"}
        if p.stat().st_size > 1_000_000:
            return {"success": False, "error": f"File too large ({p.stat().st_size} bytes)"}
        content = p.read_text(errors="replace")
        return {"success": True, "output": content}

    def _exec_write_file(self, path: str, content: str) -> dict:
        """Write content to a file."""
        from pathlib import Path

        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        lines = content.count("\n") + 1
        return {"success": True, "output": f"Wrote {lines} lines to {path}"}

    def _exec_edit_file(self, path: str, old_text: str, new_text: str) -> dict:
        """Replace text in a file."""
        from pathlib import Path

        p = Path(path).expanduser()
        if not p.exists():
            return {"success": False, "error": f"File not found: {path}"}
        content = p.read_text(errors="replace")
        if old_text not in content:
            return {"success": False, "error": f"Text not found in {path}"}
        count = content.count(old_text)
        new_content = content.replace(old_text, new_text)
        p.write_text(new_content)
        return {"success": True, "output": f"Replaced {count} occurrence(s) in {path}"}

    def _exec_search(self, pattern: str, path: str) -> dict:
        """Search for patterns in files."""
        import subprocess

        try:
            result = subprocess.run(
                [
                    "grep",
                    "-rn",
                    "--include=*.py",
                    "--include=*.ts",
                    "--include=*.tsx",
                    "--include=*.js",
                    "--include=*.jsx",
                    "--include=*.md",
                    pattern,
                    path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout
            if not output:
                return {"success": True, "output": "No matches found"}
            lines = output.strip().split("\n")
            if len(lines) > 50:
                output = "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more matches)"
            return {"success": True, "output": output}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Search timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# Color constants (local to avoid circular import with repl.py)
import os as _os

_NO_COLOR = _os.environ.get("NO_COLOR")
if _NO_COLOR:
    _C_CYAN = _C_GREEN = _C_RED = _C_YELLOW = _C_DIM = _C_BOLD = _C_RESET = ""
else:
    _C_CYAN = "\033[36m"
    _C_GREEN = "\033[32m"
    _C_RED = "\033[31m"
    _C_YELLOW = "\033[33m"
    _C_DIM = "\033[2m"
    _C_BOLD = "\033[1m"
    _C_RESET = "\033[0m"
