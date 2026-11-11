# Agent core: role contracts and tool boundaries

Agent definitions are persisted in `domain/agents/_internal/system.py` and executed by the shared `Agent` loop in `domain/agents/_internal/agents.py`. A definition is not just a name and prompt: it has a structured role contract and an explicit tool allowlist.

## Role contract

Definitions may specify:

- `role`: stable specialization identifier (for example, `implementation_engineer`).
- `objective`: the outcome the agent is assigned to achieve.
- `scope`: what it may and may not do.
- `success_criteria`: observable checks for a good result.
- `instructions`: additional behavioral guidance.
- `tools`: the only `ToolCapability` values the agent may execute.

The role fields are rendered into runtime instructions for planning and response synthesis. Built-in profiles have these fields. Older stored built-ins receive missing built-in metadata at read time; user edits are preserved and the stored JSON is not rewritten by this compatibility behavior.

## Tool authorization invariant

`tools` is an allowlist, not a hint. The planner sees only that list, and `Agent._execute_allowed_tool` checks it again immediately before dispatch to `ToolRunner`. A model-produced tool call outside the allowlist returns `E_TOOL_NOT_ALLOWED` without invoking the tool. Empty or invalid allowlists grant no tools; they never expand to all registered capabilities. Unknown legacy tool names are logged and ignored. The legacy `memory` name maps to the existing read-only `knowledge_retrieval` capability; unsupported `terminal` does not map to code execution.

Tool allowlists are not a substitute for user authentication or tool-specific approval. Sensitive capabilities must continue to enforce their own guardrails at the execution boundary.

## Current Vallerian foundation and limits

The shared loop records run history and supports bounded model/tool iterations. These changes make role intent and per-agent capabilities explicit and fail-closed. They do **not** yet add durable task leases/claims, cross-run semantic memory, multi-worker coordination, or an app publication workflow. Do not describe those as implemented until their storage, ownership, recovery, and concurrency behavior is built and tested.

## Verification

Core regression tests cover built-in role completeness, invalid tool configuration, role-contract injection, and denial before tool dispatch. Run the focused suites with:

```sh
./scripts/python -m pytest tests/core-py/test_agents_system.py tests/core-py/test_agent_core.py apps/api/server/tests/test_agents_router.py -q
```

