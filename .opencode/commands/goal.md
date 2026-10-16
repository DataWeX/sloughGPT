---
description: Autonomous goal loop — classify, then act→verify→re-assess until the definition of done is met (Claude-Code-style)
agent: build
---

Autonomous execution of this goal: $ARGUMENTS

Work like Claude Code's own loop: **act, verify, re-assess, repeat — until done**, under the AGENTS.md "One-shot execution" rules. Do not stop after a single pass; stop only when the loop's exit conditions are met.

0. **Classify the goal — most goals are NOT builds.** Decide one class and state it in the first line of the plan:
   - **build** — new capability that does not exist yet.
   - **build-well** — quality work on existing surface: audit→fix, conformance, de-duplication, hardening, spec-truth-up.
   - **land** — integration/merge/release work.
     Then state the **definition of done** for this goal — the observable evidence that ends the loop:
   - build: feature works in a real walk + tests green.
   - build-well: a **named standard** (design-system doc, spec section, contract test, invariant, or user decision) is met, AND a gate keeps it met. If no standard can be named from the goal text, that is the one clarifying question — ask it before the loop starts, and nothing else.
   - land: merged with quoted evidence, or explicitly blocked with the blocker named.
1. **Log it first** — add the kanban card (SOP step 1), then branch: `feat/<name>` for build, `fix/<name>` or `test/<name>` for build-well, `chore/<name>` for land (config/doc-only work may stay on main).
2. **Plan once, up front** — a 3–6 step sketch of the likely path. It is a sketch: each loop iteration may re-plan freely when reality disagrees. Ask at most one clarifying question before the first edit (spent in step 0 if needed). After the first edit, questions are soft check-ins only: report an update or request genuinely missing information — never for approval, never to ask whether to continue.
3. **The loop — each iteration:** (a) **re-read the actual state** (git status, failing tests, the file itself — never trust memory of it); (b) **pick the single highest-value next action** toward the definition of done; (c) **do it** with conservative defaults for every unspecified detail — safest reasonable option, recorded; (d) **verify** it worked (run what it touched); (e) if the plan sketch no longer matches reality, re-draw it. Then ask: is the definition of done met? If no, next iteration.
4. **Failures are not exit conditions** — on failures, group by root cause, fix the code (not the test, unless the test itself is wrong), re-run until green. Only a product decision or deleting user data may exit — then state the fix you'd apply and wait.
5. **Full gate, every loop** — lint, typecheck, tests before calling done; for build-well also **mutation-check the new gate** (break the thing it guards, confirm it fails, restore) — an unverified gate is a false sense of safety. Run the relevant `scripts/benchmark_*` if this touches training/inference/quantization/architecture.
6. **Exit conditions — the loop ends only on:**
   - **DONE**: the definition of done is verified by evidence (step 0), and the full gate is green → land it (merge only when green; if unrecoverable, revert the branch and explain why);
   - **BLOCKED**: a product decision or user-data deletion is required → state the exact decision and the fix you'd apply, then wait;
   - **STUCK**: the same failure repeats with no state change across two iterations → stop, report the loop, the root-cause hypothesis, and what you'd try next. Never spin.
7. **Manager-style report** — the reader is an engineering manager: lead with the verdict (goal achieved? evidence?), then the major and specifically-decided details — design choices, ASSUMPTIONS, test/bench numbers, open risks. For build-well, the verdict is measured against the named standard ("meets DESIGN_SYSTEM.md §type-scale; ratchet baseline shrunk"), never just "fixed". Mention any plan deviation from step 2 and why reality won. Omit working noise: musings, raw shell commands, tool transcripts. Include a detail only if it's relevant to the decision or an interesting engineering read, and keep even those collapsed/appendix-style.
