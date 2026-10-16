---
description: One-shot goal run — classify build vs build-well, plan once, then execute with zero mid-run stops
agent: build
---

One-shot execution of this goal: $ARGUMENTS

Run the whole thing without stopping, under the AGENTS.md "One-shot execution" rules:

0. **Classify the goal before planning — most goals are NOT builds.** Decide one class and state it in the first line of the plan:
   - **build** — new capability that does not exist yet. Deliverable: working feature + tests.
   - **build-well** — quality work on something that already exists: audit→fix, conformance to the design system, de-duplication, hardening, de-ad-hoc-ing, spec-truth-up. Deliverable: the thing meets a _named standard_, with a gate that keeps it there. Fixing-for-the-sake-of-green is failure; every fix cites the standard it serves.
   - **land** — integration/merge/release work. Deliverable: merged-with-evidence or explicitly blocked.
     For build-well and land, name the **standard** up front (the design-system doc, spec section, contract test, invariant, or user decision) before writing the plan. If no standard can be named from the goal text, that is the one clarifying question — ask it in step 1 and nothing else.
1. **Log it first** — add the kanban card (SOP step 1), then branch: `feat/<name>` for build, `fix/<name>` or `test/<name>` for build-well, `chore/<name>` for land (config/doc-only work may stay on main).
2. **Plan once, up front** — write a 3–6 step plan before the first edit. The plan shape follows the class: build plans end in a feature; build-well plans end in a **gate** (test/ratchet/baseline) that makes the fixed defect unable to silently return. Ask at most one clarifying question here. After the first edit, questions are soft check-ins only: use them to report an update or request genuinely missing information — never for approval, never to ask whether to continue; otherwise resolve ambiguity yourself.
3. **Conservative defaults** — from the first edit onward, take the safest reasonable option for every unspecified detail and record it. No check-ins, no "should I continue?".
4. **Tests are not a stop point** — on failures, group by root cause, fix the code (not the test, unless the test itself is wrong), re-run until green. Only a product decision or deleting user data may stop you — then state the fix you'd apply and wait.
5. **Full gate** — lint, typecheck, tests; for build-well also **mutation-check the new gate** (break the thing it guards, confirm it fails, restore) — an unverified gate is a false sense of safety. Run the relevant `scripts/benchmark_*` if this touches training/inference/quantization/architecture.
6. **Land it** — merge to main only when green; if unrecoverable, revert the branch and explain why. For build-well, "green" includes the new gate passing AND failing correctly when mutated.
7. **Manager-style report** — the reader is an engineering manager: lead with the verdict (what shipped, did it work), then the major and specifically-decided details — design choices, ASSUMPTIONS, test/bench numbers, open risks. For build-well, the verdict is measured against the named standard: state it ("meets DESIGN_SYSTEM.md §type-scale; 3 raw buttons removed, ratchet baseline shrunk"), never just "fixed". Omit working noise: musings, raw shell commands, tool transcripts. Include a detail only if it's relevant to the decision or an interesting engineering read, and keep even those collapsed/appendix-style.
