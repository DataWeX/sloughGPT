"""Tests for the provider-backed InferFn adapter (AgentLoop ↔ live model).

Covers:
- render_tool_contract — protocol teaching block, empty-spec short-circuit
- _prepare_messages — tool-role re-attribution, system injection
- resolve_provider — resolution order (explicit → slonet-native → default → first)
- make_provider_infer — fail-fast on no provider, message shaping, kwargs
- end-to-end: AgentLoop driven by a fake ModelProvider through the adapter
- facade exports (merged __getattr__)
"""

from __future__ import annotations

import asyncio

import pytest

from domain.agents._internal.provider_infer import (
    NO_PROVIDER_MSG,
    _prepare_messages,
    make_provider_infer,
    render_tool_contract,
    resolve_provider,
)
from domain.agents._internal.tools import ToolParam, ToolRegistry, ToolSpec
from domain.models._internal.provider import registry as reg
from domain.models._internal.provider.registry import register_provider


@pytest.fixture(autouse=True)
def _clean_providers():
    """Snapshot + clear the provider registry so resolution tests are hermetic."""
    saved = dict(reg._providers)
    reg.clear_providers()
    yield
    reg.clear_providers()
    reg._providers.update(saved)


def _spec(name: str = "calculator") -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Evaluate a mathematical expression",
        parameters=[
            ToolParam("expression", "string", "The expression", required=True),
            ToolParam("digits", "integer", "Rounding digits", required=False),
        ],
        execute=None,  # type: ignore[arg-type] - never called in these tests
    )


async def _collect(agen):
    return [e async for e in agen]


class TestRenderToolContract:
    def test_lists_tools_with_params_and_required(self):
        out = render_tool_contract([_spec()])
        assert "```tool_call" in out
        assert "- calculator: Evaluate a mathematical expression" in out
        assert "expression: string" in out
        assert "digits: integer (optional)" in out

    def test_empty_specs_returns_empty(self):
        assert render_tool_contract([]) == ""


class TestPrepareMessages:
    def test_tool_role_becomes_attributed_user(self):
        msgs = [
            {"role": "assistant", "content": "calling"},
            {"role": "tool", "name": "calculator", "content": "4"},
        ]
        out = _prepare_messages(msgs, "")
        assert out[1]["role"] == "user"
        assert out[1]["content"] == "[tool result: calculator]\n4"

    def test_contract_appended_to_existing_system(self):
        msgs = [{"role": "system", "content": "You are terse."}]
        out = _prepare_messages(msgs, "CONTRACT")
        assert out[0]["role"] == "system"
        assert out[0]["content"] == "You are terse.\n\nCONTRACT"

    def test_contract_prepended_when_no_system(self):
        out = _prepare_messages([{"role": "user", "content": "hi"}], "CONTRACT")
        assert out[0] == {"role": "system", "content": "CONTRACT"}
        assert out[1]["role"] == "user"

    def test_no_contract_leaves_messages_untouched(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert _prepare_messages(msgs, "") == msgs


class TestResolveProvider:
    def test_none_when_empty(self):
        assert resolve_provider() is None

    def test_explicit_wins(self):
        sentinel = object()
        register_provider("whatever", sentinel)
        assert resolve_provider(sentinel) is sentinel

    def test_slonet_native_preferred_over_default(self):
        native, default = object(), object()
        register_provider("default", default)
        register_provider("slonet-native", native)
        assert resolve_provider() is native

    def test_default_over_first_registered(self):
        default, other = object(), object()
        register_provider("other", other)
        register_provider("default", default)
        assert resolve_provider() is default

    def test_first_registered_fallback(self):
        p = object()
        register_provider("only", p)
        assert resolve_provider() is p


class TestMakeProviderInfer:
    def test_raises_without_provider(self):
        with pytest.raises(LookupError) as exc:
            make_provider_infer()
        assert str(exc.value) == NO_PROVIDER_MSG

    def test_shapes_messages_and_returns_provider_text(self):
        seen: list[list[dict]] = []

        class FakeProvider:
            async def chat(self, messages, **kwargs):
                seen.append(messages)
                return '```tool_call\n{"name": "calculator", "args": {}}\n```'

        infer = make_provider_infer(FakeProvider())
        text = asyncio.run(
            infer(
                [
                    {"role": "system", "content": "base"},
                    {"role": "user", "content": "2+2?"},
                ],
                [_spec()],
            )
        )
        assert "```tool_call" in text
        msgs = seen[0]
        # contract injected into system, tool spec listed
        assert msgs[0]["role"] == "system"
        assert "calculator" in msgs[0]["content"]
        assert msgs[1] == {"role": "user", "content": "2+2?"}

    def test_generation_kwargs_forwarded(self):
        captured: dict = []

        class FakeProvider:
            async def chat(self, messages, **kwargs):
                captured.append(kwargs)
                return "done"

        infer = make_provider_infer(FakeProvider(), max_tokens=128, temperature=0.1)
        asyncio.run(infer([{"role": "user", "content": "x"}], []))
        assert captured[0]["max_tokens"] == 128
        assert captured[0]["temperature"] == 0.1


class TestLoopEndToEnd:
    def test_full_tool_round_trip_through_adapter(self):
        """Qwen-shaped fake: emits a tool call, sees the result, answers."""
        from domain.agents import AgentLoop, decode, encode_tool_block  # facade path

        class FakeProvider:
            def __init__(self):
                self.calls = 0
                self.tool_result_seen = False

            async def chat(self, messages, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    block = '{"name": "calculator", "args": {"expression": "2+2"}}'
                    return f"```tool_call\n{block}\n```"
                # second call must carry the re-attributed tool result
                self.tool_result_seen = any(
                    m.get("role") == "user" and "[tool result: calculator]" in m.get("content", "")
                    for m in messages
                )
                return "The answer is 4."

        provider = FakeProvider()
        registry = ToolRegistry()  # real calculator
        events = asyncio.run(
            _collect(
                AgentLoop().run("what is 2+2?", infer=make_provider_infer(provider), tools=registry)
            )
        )
        kinds = [e.kind for e in events]
        assert kinds == ["plan", "tool_call", "tool_result", "final"]
        assert events[-1].payload["text"] == "The answer is 4."
        assert provider.tool_result_seen, "adapter dropped the tool result from the prompt"

        tool_result = next(e for e in events if e.kind == "tool_result")
        assert tool_result.payload["success"] is True
        assert tool_result.payload["output"], "real calculator produced no output"

        # decode/encode symmetry on the same canonical block
        calls, residual = decode(
            '```tool_call\n{"name": "calculator", "args": {"expression": "2+2"}}\n```'
        )
        assert [c.name for c in calls] == ["calculator"]
        assert residual == ""
        assert encode_tool_block(calls).startswith("```tool_call")


class TestToolSubset:
    def test_subset_keeps_only_named(self):
        registry = ToolRegistry()
        sub = registry.subset(["calculator"])
        assert sub.get("calculator") is not None
        assert sub.get("current_time") is None

    def test_subset_unknown_raises(self):
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="unknown tool: nope"):
            registry.subset(["nope"])

    def test_empty_defaults_registry(self):
        registry = ToolRegistry(register_defaults=False)
        assert registry.list_tools() == []


def test_facade_exports():
    """The merged facade must expose loop, protocol and adapter names."""
    import domain.agents as A

    for name in (
        "AgentLoop",
        "Event",
        "ToolCall",
        "decode",
        "encode_tool_block",
        "validate_args",
        "make_provider_infer",
        "resolve_provider",
        "render_tool_contract",
        "NO_PROVIDER_MSG",
        "ToolRegistry",
    ):
        assert hasattr(A, name), f"facade missing {name}"
