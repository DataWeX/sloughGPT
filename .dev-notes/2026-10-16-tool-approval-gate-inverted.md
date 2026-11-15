---
id: 20261016_tool_approval_gate_inverted
title: "Chat tool-approval gate is inverted and dead: run_code/web_search never execute, harmless tools prompt instead"
status: todo
date: 2026-10-16
tags: [web, chat, tools, security, sse, backend]
---

# Chat tool-approval gate is inverted and dead

Found 2026-10-16 while reconciling the chat-stream SSE callback bag against
what the backend actually emits (web session, branch `feat/display-layer`,
commit `e6f8f5c7d`). **Not fixed** — it is a backend change on another
session's branch and the correct behaviour is a product decision.

## Trace (all verified, `/chat/stream` path)

**Backend** `apps/api/server/routers/inference.py:2107`:

```python
if spec and spec.requires_approval:
    yield _sse_event("chat", "TOOL_APPROVAL", "pending", ...)   # emits, falls through
else:
    yield _sse_event("chat", "TOOL", "working", ...)
    result = await tool_reg.execute(tool_name, tool_args)       # execution lives HERE
```

### 1. Approval-gated tools never execute

`requires_approval=True` → emit `TOOL_APPROVAL/pending` → leave the
if/else → `_mark("agent")` → continue. `tool_reg.execute` is in the `else`,
so it is never reached. The only `tool_reg.execute` call in the file is
`:2123`, inside that `else`.

The two gated tools are `run_code` and `web_search`
(`domain/agents/_internal/tools.py:138` and `:151`). **Both are dead.**

### 2. Approval never reaches an execution path anyway

`chatController.approveTool` → POST control → `set_chat_control(...)`
(`inference.py:3469`). The single consumer is the token loop at `:2560`,
which on `action == "approve"` only `yield`s a `CONTROL/approved` frame —
a notification. **No `tool_reg.execute` follows.** So approving, if it were
ever offered, would emit an event and do nothing.

### 3. The client prompts for the wrong tools

Client dispatch arms (`lib/stream-chat-response.ts`) are
`thinking / MEMORY / TOOL / CONTROL / CONSCIOUSNESS` + the
`status === 'error'` catch-all. There is **no `TOOL_APPROVAL` branch**, and
`status=pending` matches none of them → the frame is dropped.

Meanwhile the *non*-gated path emits `phase=TOOL, status=working` → maps to
`executing` → `lib/chat-store.ts:583`
`if (event.status === 'executing' && event.args)` →
`set({ pendingToolApproval: ... })`.

Net effect: **the user is prompted to approve tools that need no approval,
and is never prompted for `run_code` or `web_search`.**

## Security read

No hole: the dangerous tools do not run at all, and ungated tools are the
ones that already run without approval. This is an inverted-UX /
dead-functionality defect, not an un-gated-execution vulnerability.

## Decision needed (product)

How should approval work? Candidate shapes:

- **(a) Block the stream** — emit `TOOL_APPROVAL/pending`, then `await`
  `get_chat_control(session_id)` before `tool_reg.execute`. Requires the
  stream to park on a control poll (timeout policy needed).
- **(b) Defer the tool** — queue the tool, generate a placeholder, execute
  on the approval frame, inject the result into the next turn.
- **(c) Drop the gate** — remove `requires_approval` if tool sandboxing is
  the real control. Simplest; but `run_code` executing arbitrary Python/shell
  likely wants a gate.

Fixing the client alone makes it *worse*: a prompt that approves nothing.
The backend must land first, then add the `TOOL_APPROVAL` dispatch arm.

## Also related (smaller)

- `apps/api/server/routers/infer.py:179` defines a third `self._sse_event`
  duplicating `domain/api/_internal/sse_envelope.sse_event`. Swapping it to
  the import is **wire-identical** (`SSEEnvelope.to_dict` omits `id` when
  null) and has only 2 call sites (`:662`, `:678`). Untouched because
  `infer.py` sits on another session's branch.
