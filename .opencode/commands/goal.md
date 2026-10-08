---
description: One-shot goal run — plan once, then build/test/fix/merge with zero mid-run stops
agent: build
---

One-shot execution of this goal: $ARGUMENTS

Run the whole thing without stopping, under the AGENTS.md "One-shot execution" rules:

1. **Log it first** — add the kanban card (SOP step 1), then branch `feat/<name>` (config/doc-only work may stay on main).
2. **Plan once, up front** — write a 3–6 step plan before the first edit. Ask at most one clarifying question here. After the first edit, questions are soft check-ins only: use them to report an update or request genuinely missing information — never for approval, never to ask whether to continue; otherwise resolve ambiguity yourself.
3. **Conservative defaults** — from the first edit onward, take the safest reasonable option for every unspecified detail and record it. No check-ins, no "should I continue?".
4. **Tests are not a stop point** — on failures, group by root cause, fix the code (not the test, unless the test itself is wrong), re-run until green. Only a product decision or deleting user data may stop you — then state the fix you'd apply and wait.
5. **Full gate** — lint, typecheck, tests; run the relevant `scripts/benchmark_*` if this touches training/inference/quantization/architecture.
6. **Land it** — merge to main only when green; if unrecoverable, revert the branch and explain why.
7. **Manager-style report** — the reader is an engineering manager: lead with the verdict (what shipped, did it work), then the major and specifically-decided details — design choices, ASSUMPTIONS, test/bench numbers, open risks. Omit working noise: musings, raw shell commands, tool transcripts. Include a detail only if it's relevant to the decision or an interesting engineering read, and keep even those collapsed/appendix-style.
