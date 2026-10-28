"""Provider-backed InferFn — bridge the live model provider to AgentLoop.

The loop's ``infer`` seam is ``(messages, tool_specs) -> str`` (canonical
text out). This adapter wraps any registered ``ModelProvider`` (the
autoloaded model — Qwen 2.5, SmolLM, SloNet — behind ``get_provider``) into
that seam, doing the two things a raw provider cannot:

1. **Teaching the protocol** — injects the canonical ```tool_call contract
   into the system message, so a base/chat model without native
   function-calling learns the fence the parser expects (same
   prompt-injection approach as ``ToolUseProcessor`` for the legacy marker).
2. **Translating tool results** — provider prompt renderers (e.g.
   ``SloTransformer._messages_to_prompt``) only understand
   system/user/assistant and *silently drop* anything else; the loop's
   ``tool``-role messages are re-attributed as user messages so the model
   actually sees its own tool outputs.

Usage:
    infer = make_provider_infer()            # best registered provider
    infer = make_provider_infer(my_provider)  # explicit
    async for event in AgentLoop().run(task, infer=infer, tools=registry):
        ...
"""

from __future__ import annotations

from typing import Any

from domain.agents._internal.tools import ToolSpec

PROTOCOL_HEADER = (
    "You can call tools. To call a tool, output a fenced JSON block and nothing else:\n"
    '```tool_call\n{"name": "<tool>", "args": {"<param>": <value>}}\n```\n'
    'Multiple calls in one block: {"tool_calls": [{"name": ..., "args": ...}, ...]}\n'
    "When you are done, answer in plain prose with no tool block."
)

NO_PROVIDER_MSG = "no model provider registered (load a model first)"


def render_tool_contract(specs: list[ToolSpec]) -> str:
    """Render the ToolSpec list into the prompt block that teaches the protocol.

    Returns "" for an empty spec list — no tools, no contract, no prompt
    pollution.
    """
    if not specs:
        return ""
    lines = [PROTOCOL_HEADER, "Available tools:"]
    for s in specs:
        params = ", ".join(
            f"{p.name}: {p.type}" + ("" if p.required else " (optional)") for p in s.parameters
        )
        lines.append(f"- {s.name}: {s.description}" + (f" [{params}]" if params else ""))
    return "\n".join(lines)


def _prepare_messages(messages: list[dict[str, Any]], tool_contract: str) -> list[dict[str, Any]]:
    """Inject the contract into the system message; re-attribute tool results.

    ``tool``-role messages become ``user`` messages prefixed with the tool
    name — providers without a tool role would otherwise drop them, hiding
    every tool result from the model.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        if role == "tool":
            name = m.get("name", "tool")
            out.append(
                {"role": "user", "content": f"[tool result: {name}]\n{m.get('content', '')}"}
            )
        else:
            out.append(m)
    if tool_contract:
        if out and out[0].get("role") == "system":
            out[0] = {
                "role": "system",
                "content": f"{out[0].get('content', '')}\n\n{tool_contract}",
            }
        else:
            out.insert(0, {"role": "system", "content": tool_contract})
    return out


def resolve_provider(provider: Any = None) -> Any:
    """Explicit → ``slonet-native`` → ``default`` → first registered → None."""
    if provider is not None:
        return provider
    from domain.models._internal.provider.registry import get_provider, list_providers

    for name in ("slonet-native", "default"):
        p = get_provider(name)
        if p is not None:
            return p
    names = list_providers()
    return get_provider(names[0]) if names else None


def make_provider_infer(
    provider: Any = None, *, max_tokens: int = 512, temperature: float = 0.7
) -> Any:
    """Wrap a registered provider into the loop's ``InferFn`` signature.

    Raises :class:`LookupError` immediately when no provider can be resolved
    — fail fast at wiring time, not at the first model call.
    """
    resolved = resolve_provider(provider)
    if resolved is None:
        raise LookupError(NO_PROVIDER_MSG)

    async def infer(messages: list[dict[str, Any]], tool_specs: list[ToolSpec]) -> str:
        prepared = _prepare_messages(messages, render_tool_contract(tool_specs))
        text: str = await resolved.chat(prepared, max_tokens=max_tokens, temperature=temperature)
        return text

    return infer
