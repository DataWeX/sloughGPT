"""agents — Agentic infrastructure for safe internal tool execution.

Public API:
    SecurityConfig, SecurityBoundary, ToolCapability, ToolDefinition
    ToolExecutionContext, ToolRunner, AgentConfig, Agent
    ToolSpec, ToolParam, ToolRegistry, get_agent, get_runner, get_agent_system,
    get_tool_registry
    AgentLoop, Event, ToolCall, ProtocolError, decode, encode_tool_block,
    validate_args
    make_provider_infer, resolve_provider, render_tool_contract
"""

from __future__ import annotations

from domain.agents._internal.agents import (
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
from domain.agents._internal.system import get_agent_system
from domain.agents._internal.tools import ToolParam, ToolRegistry, ToolSpec, get_tool_registry

_LAZY_IMPORTS = {
    # orchestrator + run history (heavier graphs)
    "MultiAgentOrchestrator": ("._internal.multi", "MultiAgentOrchestrator"),
    "get_agent_run_store": ("._internal.run_history", "get_agent_run_store"),
    # canonical loop + protocol
    "AgentLoop": ("._internal.loop", "AgentLoop"),
    "Event": ("._internal.loop", "Event"),
    "validate_args": ("._internal.loop", "validate_args"),
    "ToolCall": ("._internal.protocol", "ToolCall"),
    "ProtocolError": ("._internal.protocol", "ProtocolError"),
    "decode": ("._internal.protocol", "decode"),
    "encode_tool_block": ("._internal.protocol", "encode_tool_block"),
    # provider-backed InferFn (Qwen/SmolLM/SloNet → AgentLoop seam)
    "make_provider_infer": ("._internal.provider_infer", "make_provider_infer"),
    "resolve_provider": ("._internal.provider_infer", "resolve_provider"),
    "render_tool_contract": ("._internal.provider_infer", "render_tool_contract"),
    "NO_PROVIDER_MSG": ("._internal.provider_infer", "NO_PROVIDER_MSG"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, package=__name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SecurityConfig",
    "SecurityBoundary",
    "ToolCapability",
    "ToolDefinition",
    "ToolExecutionContext",
    "ToolRunner",
    "AgentConfig",
    "Agent",
    "get_agent",
    "get_runner",
    "get_agent_system",
    "get_tool_registry",
    "ToolSpec",
    "ToolParam",
    "ToolRegistry",
    "MultiAgentOrchestrator",
    "get_agent_run_store",
    "AgentLoop",
    "Event",
    "validate_args",
    "ToolCall",
    "ProtocolError",
    "decode",
    "encode_tool_block",
    "make_provider_infer",
    "resolve_provider",
    "render_tool_contract",
    "NO_PROVIDER_MSG",
]
