"""Agentic Infrastructure - Safe internal tool system.

Canonical implementation lives in :mod:`domain.agents._internal.agents`; this
package re-exports it so ``from domain.agents._internal import X`` and
``from domain.agents._internal.agents import X`` yield the exact same objects.
The lazy multi-agent imports are handled by ``agents.py.__getattr__``.
"""

from domain.agents._internal.agents import (  # noqa: F401
    Agent,
    AgentConfig,
    SecurityBoundary,
    SecurityConfig,
    ToolCapability,
    ToolDefinition,
    ToolExecutionContext,
    ToolRunner,
    get_agent,
    get_runner,
)

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


def __getattr__(name: str):
    return getattr(__import__("domain.agents._internal.agents", fromlist=[name]), name)
