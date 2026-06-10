"""
Multi-agent system — specialized agents collaborating on a goal.

Simple sequential orchestration:
  1. Orchestrator receives a goal
  2. LLM plans subtasks
  3. Each subtask assigned to the right agent
  4. Agents execute in sequence, passing context
  5. Final response composed and returned

Usage:
    from domains.agents.multi import MultiAgentOrchestrator, get_orchestrator

    orch = get_orchestrator()
    result = orch.execute("research transformers and write a summary")
    print(result["response"])
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import ToolCapability
from ..shell.commands import ShellCommands


# ── Agent definitions ─────────────────────────────────────────────────

@dataclass
class SpecializedAgent:
    """A specialized agent with a role and system prompt."""

    name: str
    role: str
    system_prompt: str
    tools: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "description": self.system_prompt[:80] + "...",
        }


DEFAULT_AGENTS: Dict[str, SpecializedAgent] = {
    "researcher": SpecializedAgent(
        name="Researcher",
        role="research and gather information",
        system_prompt=(
            "You are a research agent. Your job is to find information, "
            "analyze it, and present clear findings. Be thorough and cite "
            "key facts. Output structured notes, not a finished article."
        ),
        tools=["web_search", "memory"],
    ),
    "writer": SpecializedAgent(
        name="Writer",
        role="write and compose content",
        system_prompt=(
            "You are a writing agent. Your job is to take research notes "
            "and turn them into clear, well-structured content. Write in "
            "a professional tone suitable for the given audience."
        ),
        tools=["memory"],
    ),
    "coder": SpecializedAgent(
        name="Coder",
        role="write and analyze code",
        system_prompt=(
            "You are a coding agent. Your job is to write, review, and "
            "explain code. Output working code with brief explanations. "
            "Prefer Python unless told otherwise."
        ),
        tools=["code_execution", "file_search"],
    ),
    "critic": SpecializedAgent(
        name="Critic",
        role="review and improve work",
        system_prompt=(
            "You are a critic agent. Your job is to review work from "
            "other agents, identify gaps, suggest improvements, and "
            "flag errors. Be constructive and specific."
        ),
        tools=["memory"],
    ),
}


# ── Task model ────────────────────────────────────────────────────────


class TaskStatus:
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentTask:
    """A subtask assigned to a specific agent."""

    id: str
    description: str
    assigned_agent: str
    context: str = ""
    result: str = ""
    status: str = TaskStatus.PENDING
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "agent": self.assigned_agent,
            "status": self.status,
            "result_preview": self.result[:100] if self.result else "",
        }


# ── Multi-agent orchestrator ──────────────────────────────────────────

class MultiAgentOrchestrator:
    """Orchestrates multiple specialized agents to accomplish a goal.

    Strategy: plan → sequential execution with context passing.
    """

    def __init__(self, agents: Optional[Dict[str, SpecializedAgent]] = None):
        self.agents = agents or dict(DEFAULT_AGENTS)
        self._cmds = ShellCommands

    def list_agents(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.agents.values()]

    def get_agent(self, name: str) -> Optional[SpecializedAgent]:
        return self.agents.get(name)

    def execute(self, goal: str, context: str = "") -> Dict[str, Any]:
        """Execute a goal using multi-agent orchestration.

        Steps:
          1. Plan subtasks (LLM)
          2. For each subtask, pick the best agent
          3. Run each agent sequentially with accumulated context
          4. Compose final response
        """
        tasks = self._plan(goal, context)
        if not tasks:
            return {"response": "Could not plan this goal.", "tasks": []}

        for task in tasks:
            task.status = TaskStatus.IN_PROGRESS
            try:
                result = self._run_agent(task, goal, context)
                task.result = result
                task.status = TaskStatus.COMPLETED
                context += f"\n\n[{task.assigned_agent} result]\n{result}"
            except Exception as e:
                task.error = str(e)
                task.status = TaskStatus.FAILED

        final = self._compose(goal, tasks)
        return {"response": final, "tasks": [t.to_dict() for t in tasks]}

    def _plan(self, goal: str, context: str) -> List[AgentTask]:
        """Use LLM to plan subtasks."""
        agent_names = ", ".join(self.agents.keys())
        prompt = (
            f"Goal: {goal}\n"
            f"Available agents: {agent_names}\n"
            f"Break this goal into 1-3 sequential subtasks. "
            f"For each, specify the agent to use and what to do.\n"
            f"Format: JSON array of {{'id': str, 'description': str, 'agent': str}}\n"
            f"Return ONLY the JSON, no other text."
        )
        resp = self._generate(prompt)

        try:
            plan = json.loads(resp)
            if not isinstance(plan, list):
                plan = [plan]
        except (json.JSONDecodeError, TypeError):
            # Fallback: extract JSON from response
            match = re.search(r"\[.*?\]", resp, re.DOTALL)
            if match:
                try:
                    plan = json.loads(match.group())
                except json.JSONDecodeError:
                    return self._simple_plan(goal)
            else:
                return self._simple_plan(goal)

        tasks = []
        for i, item in enumerate(plan):
            agent_name = item.get("agent", "researcher")
            if agent_name not in self.agents:
                agent_name = "researcher"
            tasks.append(AgentTask(
                id=item.get("id", str(i + 1)),
                description=item.get("description", item.get("task", f"Step {i + 1}")),
                assigned_agent=agent_name,
            ))
        return tasks or self._simple_plan(goal)

    def _simple_plan(self, goal: str) -> List[AgentTask]:
        """Fallback plan when LLM fails."""
        return [
            AgentTask(id="1", description=f"Research: {goal}", assigned_agent="researcher"),
            AgentTask(id="2", description=f"Write: synthesize findings", assigned_agent="writer"),
        ]

    def _run_agent(self, task: AgentTask, goal: str, context: str) -> str:
        """Run a single agent on its task."""
        agent = self.agents.get(task.assigned_agent)
        if not agent:
            return f"[No agent '{task.assigned_agent}' available]"

        prev = f"Previous work:\n{context}" if context else ""
        prompt = (
            f"{agent.system_prompt}\n\n"
            f"Goal: {goal}\n"
            f"Your task: {task.description}\n"
            f"{prev}\n"
            f"Output your work below:"
        )
        result = self._generate(prompt, max_tokens=300)
        return result.strip()

    def _compose(self, goal: str, tasks: List[AgentTask]) -> str:
        """Compose final output from all task results."""
        completed = [t for t in tasks if t.status == TaskStatus.COMPLETED]
        if not completed:
            return "All agents failed."

        lines = [f"# Result: {goal}", ""]
        for t in completed:
            agent = self.agents.get(t.assigned_agent)
            name = agent.name if agent else t.assigned_agent
            lines.append(f"## {name}: {t.description}")
            lines.append(t.result)
            lines.append("")

        # Ask LLM to write a summary
        summary_prompt = (
            f"Synthesize the following agent outputs into a cohesive response "
            f"for the user's goal: {goal}\n\n"
            + "\n".join(lines)
            + "\n\nFinal response:"
        )
        summary = self._generate(summary_prompt, max_tokens=400)
        return summary.strip()

    def _generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate text via inference API."""
        result = self._cmds.generate(prompt, max_tokens=max_tokens)
        if isinstance(result, dict) and "text" in result:
            return result["text"]
        if isinstance(result, dict) and "error" in result:
            return f"[LLM error: {result['error']}]"
        return str(result)


# ── Singleton ─────────────────────────────────────────────────────────

_orchestrator: Optional[MultiAgentOrchestrator] = None


def get_orchestrator() -> MultiAgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = MultiAgentOrchestrator()
    return _orchestrator


def reset_orchestrator() -> None:
    global _orchestrator
    _orchestrator = None
