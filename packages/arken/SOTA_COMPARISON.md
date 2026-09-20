# Arken vs Top Agent Architectures — Student Study Notes

> Renamed: `packages/voyager` → `packages/arken` (brand Arken).
> V1 scope: minimal autoclicker (find, click, type). `packages/voyager/`
> is now a thin re-export shim. Old AI/vision/backends modules removed.

Goal: learn how the best computer-use agents work, then match the most
basic functioning version in our Voyager. No code changes yet, just comparison.

Our Voyager lives in `packages/voyager/src/voyager/`.
Key files: `ai/agent.py`, `ai/models.py`, `ai/learning.py`, `core/session.py`.

---

## 1. What I studied

### A. ReAct (Yao et al. 2022) — the base loop

The simplest idea that everything else copies:

1. Thought — model talks to itself ("I need to click Start").
2. Action — model calls a tool with args (click at x,y).
3. Observation — tool result goes back into context.

Repeat until done. The transcript IS the state. The model is stateless,
our code (the harness) owns the loop.

Modern form: assistant turn = `text` (thought) + `tool_use` (action).
Driver appends full turn back to messages, runs tool, appends `tool_result`,
calls model again.

### B. Anthropic "Building Effective Agents" — workflows vs agents

- Workflows = fixed code path (prompt chaining, orchestrator-workers, evaluator-optimizer).
- Agents = model directs its own steps, uses tools, checks ground truth each step.

Big lesson: harness owns the bottom layer, model owns the top layer.

- Model decides: which tool, what args, when to stop.
- Harness enforces: step cap, timeout, token/dollar cap, no-progress detect,
  retry, error shaping, persistence, streaming, cancel.

Prompt cannot enforce a budget. Only code can.

### C. OpenAI CUA / Operator — pixel-only computer use

Loop: Perception → Reasoning → Action.

- Perception: screenshot added to context each turn.
- Reasoning: chain-of-thought over current + past screenshots + actions.
- Action: virtual mouse/keyboard until done or user input needed.
- Self-corrects when UI changes. Asks human for login/CAPTCHA.
- OSWorld score 38.1% vs human 72.4%. More steps allowed = better score.

### D. Anthropic Computer Use — tool-shaped version of same loop

Loop: screenshot → reason → act → fresh screenshot → repeat.

- Enable `computer_toolset_20260801`, model emits tool requests.
- Fixed vocab: screenshot, cursor_position, left_click, right_click,
  double_click, mouse_move, left_click_drag, scroll, type, key, hold_key.
- Our code runs actions in a sandbox (Docker + virtual display).
- Limits: latency (screenshot→reason→act is slow), prompt injection via
  screen content, coordinate errors.
- Safety: minimal-privilege VM, domain allowlist, human confirm for
  consequential actions, turn/action budgets.

### E. Voyager paper (Minecraft, Wang et al. 2023) — name confusion

Our package shares a name but not the design. That Voyager = lifelong
learner with three parts:

1. Automatic curriculum — proposes next task based on progress.
2. Skill library — stores working code skills, reuses them.
3. Iterative prompting — refines actions with environment feedback + self-verify.

We have none of these three. Worth borrowing for phase 2.

### F. Plan-and-execute vs reactive

- ReAct: one model call per step. Good for unknown step counts.
- Plan-and-execute: one plan call (JSON steps), then execute steps with
  cheap executor, re-plan on failure. Cheaper for predictable tasks.

---

## 2. What our Voyager has today

| Part                                    | File                                              | Status                                                             |
| --------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------ |
| Perceive-act loop                       | `ai/agent.py:Agent.run()`                         | Yes, observe→decide→act→learn, max 50 steps                        |
| Action vocab                            | `ai/models.py:ActionType`                         | 14 actions, close to Anthropic vocab + navigate/wait/done/fail     |
| Model interface                         | `ai/models.py:VisionModel`                        | Protocol with predict_action, good shape                           |
| Test models                             | `ai/models.py` Echo, RuleBased, Composite         | Only mocks, no real LLM                                            |
| Experience store                        | `ai/learning.py:ExperienceBuffer`                 | In-memory list, save/load JSON, no embeddings                      |
| Feedback                                | `ai/learning.py:FeedbackLoop`                     | Rating + correction + pattern count, no training signal            |
| Journey session                         | `core/session.py:Voyager`                         | goto/find/click/fill + task runner + reporter, separate from Agent |
| Backends                                | `backends/` 7 total                               | Playwright, Selenium, CDP, API, CLI, Appium, Desktop               |
| Waits, tabs, replay, network mock, perf | `waits/`, `tabs/`, `events/`, `network/`, `perf/` | Present, not wired into Agent loop                                 |
| Budgets                                 | `ai/agent.py:AgentConfig`                         | max_steps + step_timeout (unused) only                             |
| Transcript persist                      | none                                              | Trajectory in memory, no JSONL resume                              |
| Tool schema docs                        | none                                              | No per-tool description/examples for model                         |
| Verifier                                | none                                              | DONE/FAIL self-reported, no ground-truth check                     |
| Skill library                           | none                                              | No reusable skill files                                            |
| Safety                                  | none                                              | No sandbox, allowlist, or confirm gate                             |

Tests: 201 passing, 14 skipped. Solid base, mocks only.

---

## 3. Side-by-side: basic functioning

| SOTA basic                                                     | Us today                                                             | Gap                                                              |
| -------------------------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------- |
| ReAct thought+action+obs appended each turn                    | We store Step(action, obs, reward) but no thought text               | Add `reasoning` to transcript, keep full history                 |
| Harness enforces step cap + deadline + no-progress + goal tool | Only max_steps, no deadline, no repeat-call detect, DONE is implicit | Add 4-part stop: done/fail + cap + deadline + repeat hash        |
| Tool definitions with docs + examples + strict args            | ActionType enum only, no descriptions                                | Write tool schema table, validate args, typed errors             |
| Screenshot + a11y tree each step, prune old images             | Screenshot each step, a11y best-effort                               | Keep latest screenshot, prune older to text narration            |
| Verifier checks ground truth (tool result, page state)         | Reward hardcoded (click=+0.1, done=+1)                               | Replace with real check: element found, URL, text present        |
| JSONL transcript, resume after crash                           | In-memory Trajectory                                                 | Append each Step to JSONL before reply                           |
| Sandbox + allowlist + confirm                                  | Direct backend calls                                                 | Add confirm hook + domain allowlist, even if permissive at first |
| Skill reuse                                                    | ExperienceBuffer list                                                | Later: save successful Trajectory as named skill file            |

---

## 4. Minimal match (done, in `ai/`)

1. **Stop rules** ✅ — `AgentConfig.deadline_ms` + `no_progress_limit`
   - `max_steps`; `AgentResult.stop_reason` tells which fired first.
2. **Tool cards** ✅ — `ACTION_CARDS` + `validate_action()` +
   `tool_cards_text()`; invalid actions return typed errors as
   observations instead of executing.
3. **Thought in loop** — `Action.reasoning` persisted per Step (in
   trajectory + JSONL).
4. **Real reward** ✅ — `ai/verifier.py`: `GoalCheck`
   (url_contains, text_present/absent, element_present) checked against
   the live page on DONE. Pass → reward 1.0 + success; fail → reward
   0.0 + `stop_reason="unverified"`. No goal → old behavior.
5. **One transcript file** ✅ — `output_dir/<task>.jsonl`, one line per
   Step, flushed before the next model call.

Explicitly later (not MVP): real LLM backend, skill library + curriculum
(Voyager-paper style), network sandbox, audio/keyframe observation.

---

## 5. Open questions for build

- Model backend for v1: sloughGPT InferenceEngine or external API first?
- Merge `Voyager` (journey session) + `Agent` (computer-use loop) into one
  entry, or keep separate?
- Where do verifier checks live: in task def, or separate Verifier class?

No code written until these are answered.
