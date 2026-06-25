"""
Multi-agent system — specialized agents collaborating on a goal.

Parallel-aware orchestration:
  1. Orchestrator receives a goal
  2. LLM plans subtasks with dependency info
  3. Tasks grouped by dependency level (independent → parallel)
  4. Each level's tasks run in parallel via ThreadPoolExecutor
  5. Sequential levels feed context forward
  6. Final response composed and returned

Usage:
    from domains.agents.multi import MultiAgentOrchestrator, get_orchestrator

    orch = get_orchestrator()
    result = orch.execute("research transformers and write a summary")
    print(result["response"])
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import logging

from . import ToolCapability
from ..shell.commands import ShellCommands

logger = logging.getLogger("man.agents.multi")


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
    depends_on: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "agent": self.assigned_agent,
            "status": self.status,
            "result_preview": self.result[:100] if self.result else "",
            "depends_on": self.depends_on,
        }


# ── Multi-agent orchestrator ──────────────────────────────────────────

class MultiAgentOrchestrator:
    """Orchestrates multiple specialized agents to accomplish a goal.

    Tasks without dependencies run in parallel; dependent tasks wait
    for their upstream results before executing. This reduces total
    wall-clock time for goals that decompose into independent subtasks.
    """

    def __init__(self, agents: Optional[Dict[str, SpecializedAgent]] = None):
        self.agents = agents or dict(DEFAULT_AGENTS)
        self._cmds = ShellCommands

    def list_agents(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self.agents.values()]

    def get_agent(self, name: str) -> Optional[SpecializedAgent]:
        return self.agents.get(name)

    def execute(self, goal: str, context: str = "") -> Dict[str, Any]:
        """Execute a goal using parallel-aware multi-agent orchestration.

        Steps:
          1. Plan subtasks with dependency info (LLM)
          2. Group tasks into parallel batches by dependency level
          3. Run each batch in parallel via ThreadPoolExecutor
          4. Pass context from dependencies forward
          5. Compose final response from all results
        """
        tasks = self._plan(goal, context)
        if not tasks:
            return {"response": "Could not plan this goal.", "tasks": []}

        task_map = {t.id: t for t in tasks}
        levels = self._compute_levels(tasks)
        results_ctx: Dict[str, str] = {}

        for level_idx, task_ids in enumerate(levels):
            logger.info(
                "Executing level %d (%d tasks in parallel)",
                level_idx, len(task_ids),
            )
            with ThreadPoolExecutor(max_workers=len(task_ids)) as pool:
                future_map = {}
                for tid in task_ids:
                    task = task_map[tid]
                    task.status = TaskStatus.IN_PROGRESS
                    dep_context = self._build_dep_context(task, task_map, results_ctx)
                    future = pool.submit(self._run_agent, task, goal, dep_context)
                    future_map[future] = task

                for future in as_completed(future_map):
                    task = future_map[future]
                    try:
                        result = future.result()
                        task.result = result
                        task.status = TaskStatus.COMPLETED
                        results_ctx[task.id] = result
                    except Exception as e:
                        error = str(e)
                        task.error = error
                        task.status = TaskStatus.FAILED
                        results_ctx[task.id] = f"[error: {error}]"

        final = self._compose(goal, tasks)
        return {"response": final, "tasks": [t.to_dict() for t in tasks]}

    def _plan(self, goal: str, context: str) -> List[AgentTask]:
        """Use LLM to plan subtasks with dependency info."""
        agent_names = ", ".join(self.agents.keys())
        prompt = (
            f"Goal: {goal}\n"
            f"Available agents: {agent_names}\n"
            f"Break this goal into 1-4 subtasks. "
            f"Tasks that can run in parallel go in the same level "
            f"(depends_on is empty for independent tasks). "
            f"Tasks that depend on earlier results set depends_on.\n"
            f"Format: JSON array of "
            f"{{'id': str, 'description': str, 'agent': str, "
            f"'depends_on': list[str]}}\n"
            f"Return ONLY the JSON, no other text."
        )
        resp = self._generate(prompt)

        try:
            plan = json.loads(resp)
            if not isinstance(plan, list):
                plan = [plan]
        except (json.JSONDecodeError, TypeError):
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
            deps = item.get("depends_on") or []
            if isinstance(deps, str):
                deps = [deps]
            tasks.append(AgentTask(
                id=item.get("id", str(i + 1)),
                description=item.get("description", item.get("task", f"Step {i + 1}")),
                assigned_agent=agent_name,
                depends_on=[d for d in deps if d],
            ))
        return tasks or self._simple_plan(goal)

    def _simple_plan(self, goal: str) -> List[AgentTask]:
        """Fallback plan when LLM fails — independent research + write."""
        return [
            AgentTask(id="1", description=f"Research: {goal}", assigned_agent="researcher", depends_on=[]),
            AgentTask(id="2", description=f"Write: synthesize findings", assigned_agent="writer", depends_on=["1"]),
        ]

    def _compute_levels(self, tasks: List[AgentTask]) -> List[List[str]]:
        """Topological sort into parallel levels by dependency.

        Returns [[level_0_ids], [level_1_ids], ...] where each level's
        tasks have no remaining unmet dependencies.
        """
        remaining: Set[str] = {t.id for t in tasks}
        completed: Set[str] = set()
        task_map = {t.id: t for t in tasks}
        levels: List[List[str]] = []

        while remaining:
            level = [
                tid for tid in remaining
                if all(dep in completed for dep in task_map[tid].depends_on)
            ]
            if not level:
                # Circular or broken dependency — run remaining anyway
                level = list(remaining)
            levels.append(level)
            for tid in level:
                completed.add(tid)
                remaining.discard(tid)

        return levels

    def _build_dep_context(
        self,
        task: AgentTask,
        task_map: Dict[str, AgentTask],
        results_ctx: Dict[str, str],
    ) -> str:
        """Build context string from completed dependencies."""
        parts = []
        for dep_id in task.depends_on:
            dep_task = task_map.get(dep_id)
            if dep_task and dep_id in results_ctx:
                name = dep_task.assigned_agent
                parts.append(f"[{name} ({dep_task.description})]\n{results_ctx[dep_id]}")
        return "\n\n".join(parts)

    def _run_agent(self, task: AgentTask, goal: str, dep_context: str) -> str:
        """Run a single agent on its task."""
        agent = self.agents.get(task.assigned_agent)
        if not agent:
            return f"[No agent '{task.assigned_agent}' available]"

        prev = f"Dependency context:\n{dep_context}" if dep_context else ""
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
