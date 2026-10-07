# Response Modes Protocol — a headless interface and its displays

> **Status:** design, not implemented. Card `9db02516`.
> **Thesis:** the **baby is the learner**; learning is optimized through
> **structured inputs and data points**; a response is a **parsed, well-formed
> tree** (grammar + shared head + mode branch), not a flat container; this
> document is a **protocol for handling the different modes of response**; and
> the **interface** is a headless programmable shell while WebUI, CLI and TUI
> are **displays** of it.
> **Doctrine:** [TRANSPORT_PROJECTIONS.md](TRANSPORT_PROJECTIONS.md) —
> capability → contract → transport — and the _Endpoint & Transport Rule_ in
> the root [AGENTS.md](../AGENTS.md).
> **Related:** [API.md](API.md) (SSE envelope, state read-model),
> [PRODUCT_ENGINEERING.md](PRODUCT_ENGINEERING.md), [UX_FLOWS.md](UX_FLOWS.md),
> [design/DESIGN_SYSTEM.md](design/DESIGN_SYSTEM.md).

---

## 1. Two words: interface and display

Most projects use these interchangeably. This spec does not, because they are
different jobs, and conflating them is how a UI ends up owning the logic.

### 1.1 Interface

The **thin, headless layer for interacting with the core stack** — a
programmable shell, a mini-SDK of predefined, well-ported APIs that activates
model-stack controls and logic **without rendering anything**.

- You write minimal commands; you never re-write application logic just to
  make the app do something.
- It is a `cmd`-app view onto the app's underlay, with the real targets up in
  the application layer.
- "Programmable" means it is driven the same way by a human at a terminal, a
  model via tool-calls, a script, or CI — the shell does not care who types.

### 1.2 Display

**WebUI, CLI, TUI, mobile — what renders.** Not "the frontend", not "the
interface". They are displays: they draw state, they do not define what the
app can do.

The subtlest case is the terminal, because both words live there: the verbs
you type (`slough probe …`, `slough stream …`) are **interface**; the coloured
table printed back is **display**. Same window, two different jobs.

### 1.3 What a display is, physically — the paper wrapper

A display is the small paper wrapper around the doughnut: you hold the
doughnut through it, the wrapper is not where the doughnut lives, and grease
soaks through — your hands get soiled, just not much.

| Rule                       | Meaning                                                          | Failure mode it forbids                                          |
| -------------------------- | ---------------------------------------------------------------- | ---------------------------------------------------------------- |
| **Holds, doesn't contain** | It frames state; it is never where state lives                   | State trapped in a component; closing the tab loses the work     |
| **Disposable**             | Any display can be discarded and re-rendered; none is privileged | One blessed UI; "we can't remove that button, logic lives in it" |
| **Thin**                   | Adds no material — no logic, no writes                           | The UI becomes load-bearing (plastic, not paper)                 |
| **Permeable**              | The core's real signals pass through it                          | A sealed abstraction that hides the system behind a pretty face  |
| **Attenuator, not filter** | It dampens the noise; it does not airbrush it                    | See below                                                        |

The attenuation rule is the important one, and it has exactly two forbidden
extremes:

- **Perfect insulation** — no residue at all: the display hides what is
  actually happening, and you go blind to the machine while looking at it.
- **No wrapper** — hands fully greasy: the raw stack is dumped on the user.
  That is the text-channel problem again.

The target is the phrase itself: **soiled, just not much.** Honest signal,
reduced friction, and the human never stops knowing there is a real machine
under the paper. So: **commands flow in unfiltered; state flows out
attenuated.**

### 1.4 Scope of this vocabulary

The words land here first. Existing docs that say "frontend" or "interface"
are not mass-renamed — they migrate when next touched (grandfathering, same
rule as routers).

---

## 2. The problem: the chat channel conflates the two

The default web chat box is a **display pretending to be an interface**: a
window that emits prose. It cannot express state, it offers the model no
controls to operate, and everything the model does has to be squeezed into a
paragraph. It is response-only, so it cannot support learning
(emit → get corrected → change the next emit).

So we are not adding features to the chat box. We are separating the two
words: declare what the app can do, drive it headlessly, and let displays
render. Two mistakes already retired elsewhere in this codebase, both of
which this split forbids:

- **Hacking behavior into one UI.** The next surface rebuilds it, semantics
  drift per surface, the logic gets retested N times. That is the
  endpoint-per-feature belt.
- **A new endpoint per feature.** The contract is declared once; everything
  else is projected from it.

If the interface is headless and declared, the app can be driven with nothing
on screen — and swapping a display never changes what the app can do.

---

## 3. The learner: the baby

The learner is a small model — a baby. It babbles before it is useful, it is
cheap enough to let talk constantly, it is honest about not knowing, and it
learns the way anything does: **emit → get corrected → change the next emit**.

What follows from that:

- **Every response must be a data point.** Free prose gives the loop nothing
  structured to optimize on. Structured, it can be compared across time and
  the correction's effect can be measured.
- **Correction must cost the human almost nothing.** The human is the
  caregiver; if replying costs more than a glance, the loop dies in a week.
- **Error is expected data, not failure.** The metric is not first-try
  correctness (§8).

---

## 4. The response tree: one grammar, many modes

**This is the core of the protocol.** A response is not a bag of fields in a
flat container — it is a **parsed, well-structured tree**: a typed structure
where a fixed `head` is declared once and shared by every response, and the
`mode` branch selects which payload subtree exists. Structure is syntax:
either it parses into a known tree, or it is rejected.

The tree is also the interface's **return vocabulary**: commands go in, trees
come out, displays walk them.

### 4.1 The tree

```
Response                     grammar: response.v1
├── meta
│   ├── grammar  "response.v1"   ← version of the SHAPE; breaking change = new grammar
│   ├── id       rsp_01H…        ← replies, acks and repairs point at this node
│   ├── turn_id  trn_01H…
│   └── ts       1791179543.658
│
├── head                         ← declared ONCE, shared by every mode
│   ├── question                 ┐
│   ├── topic                    │ fixed order: question → topic → state →
│   ├── state {                  │ confidence → reason → intent → motive
│   │   ├── observed             │ (the data-point core; §8 diffs this node)
│   │   ├── expected             │
│   │   └── surprise             │
│   ├── confidence  0.35         │ stated — and treated as a claim (§8.1)
│   ├── reason                   │
│   ├── intent                   │
│   └── motive                   ┘
│
├── mode                         ← the branch point (sum type)
│   think | babble | probe | answer | act | repair | ack
│
└── payload                      ← typed subtree; exists only for the chosen mode
    ├── probe  → { offer, anchor, budget }
    ├── act    → { target, change, effect }
    ├── repair → { rephrases: rsp_id, miss_n }
    ├── ack    → { delta: state, corrects: rsp_id }
    └── think | babble | answer → { }        (head already carries them)
```

**Tree, not container** — three properties that fall out:

| Property                                      | What it buys                                                                                                                 |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| **`head` is a shared node type**              | Declared once, referenced from every branch — never copied. Same doctrine as descriptors, applied to data.                   |
| **`mode` is a sum type, not a string field**  | The payload subtree either exists and type-checks, or the response does not parse. No untyped `payload` bag.                 |
| **`grammar: response.v1` versions the shape** | Adding a mode is compatible; changing `head` is a new grammar, and old displays fail loudly instead of misrendering quietly. |

**Projection.** The tree is expressed as JSON Schema with `$defs` + `$ref` —
one definition per node type, _derived_ into whatever view is needed (wire
JSON, TS types, CLI help, docs). Same rule as `ToolSpec`: `parameters` stored,
`params` derived, never stored twice.

**Return types plug into the same graph.** A descriptor's declared `result`
is a _reference_ to node types in this tree — a command that returns `state`
returns the same `state` node the probes carry. That is the mapping of return
types to responses: commands declare their returns as nodes, never as ad-hoc
response shapes.

**Validation happens once, at the writer.** The tree is parsed and validated
at the single writer (§6): either it is a well-formed tree or it never leaves
the core. Displays never validate — they walk.

### 4.2 The response modes

| Mode         | What it is                                                           | Floor              | Persisted in `messages[]` |
| ------------ | -------------------------------------------------------------------- | ------------------ | ------------------------- |
| **`think`**  | Thinking out loud — cognition we may overhear, not a claim           | No (stream)        | **No**                    |
| **`babble`** | Practicing/exploring with no claim attached — the baby testing sound | No, quota-bound    | **No**                    |
| **`probe`**  | A question shaped so a one-word human reply is a complete correction | **Requests** floor | **No**, until answered    |
| **`answer`** | A committed response to an actual question                           | Held floor         | **Yes**                   |
| **`act`**    | Operating the interface — changing a control, mutating surface state | Held floor         | Yes, as the effect it had |
| **`repair`** | Rephrasing after being misunderstood — same `id`, one retry          | Requests floor     | **No**                    |
| **`ack`**    | Received the correction, showing the resulting `state` delta         | No                 | **No**                    |

And the human's reply modes, which are part of the same protocol:

| Human mode                  | Meaning                                                       |
| --------------------------- | ------------------------------------------------------------- |
| **answer**                  | A correction — commits the probe + reply pair                 |
| **grant / defer / dismiss** | Floor control; all three first-class, _dismiss_ must be cheap |
| **silence**                 | Also legal — a baby is not punished for being ignored         |

### 4.3 Persistence invariant

> **Transcript = things we committed to. Probes and babble = things still in
> flight.**

Ephemerality is a property of **branches**: the `think`, `babble` and
unanswered-`probe` subtrees never enter `messages[]`. A `probe` _with_ the
human's reply commits as a pair — that pair is the real teaching moment.
Anything else lets unearned text pile into history and corrupt later context.

---

## 5. The protocol (handling the modes)

The tree says what a response means (§4). This section says how the modes are
exchanged — ordering, floor, budgets, repair, delivery.

### 5.1 Floor control

- A mode that wants the human **requests** the floor; it never seizes it.
- Requests queue and surface glanceably. Interrupting is the one failure mode
  that trains the human to ignore the channel — after that, the loop is dead
  regardless of how good the probes are.
- The human's grant / defer / dismiss are responses too (§4.2).

### 5.2 Budgets — the playpen fence

Bounded, because that is what makes it safe to leave running:

| Bound                                   | Purpose                                          |
| --------------------------------------- | ------------------------------------------------ |
| Probe/babble quota per session          | Stops wandering; keeps the channel worth reading |
| Topic scope tied to the current surface | Keeps joint attention real                       |
| Cost per response (tokens/time)         | Keeps a small model's talking affordable         |
| Cooldown between floor requests         | Ritual pacing, no machine-gun questioning        |

Budgets are **server-issued and server-counted**; displays render them, they
never compute them (§6).

### 5.3 Repair

Misunderstood correction → `repair` once (same `id`, rephrased). Second miss →
drop and record the miss. Never escalate silently.

### 5.4 Delivery

Responses ride the **existing** stream described in [API.md](API.md). The
transport's own envelope (`{stream, phase, status, data}`) is untouched —
**the response tree rides inside `data`**. Modes map onto a `phase` value on
an already-existing stream; they are **not** a new per-feature endpoint.
Reconnection (resume on last event id) and terminal `status: "error"` come for
free. Ephemeral branches are tagged so a reconnect does not resurrect
dismissed or superseded responses.

---

## 6. The interface: a programmable shell over the contracts

```
core stack — controls, logic, state          (single writer: ServerState)
  └─ descriptors — ToolSpec {name, params (JSON Schema), result,
  │                          auth_scope, idempotent, version}   ← declared once
       └─ INTERFACE — headless:  commands in
       │     (shell verb · HTTP route — same descriptor, no rendering)
       └─ RESPONSE TREE — modes out  (§4: grammar + shared head + mode branch)
            └─ DISPLAYS — walk the tree and draw it
                  (WebUI · CLI · TUI · mobile — paper wrapper, §1.3)
```

**Interface = command side. Response tree = result side. Display = rendering.**

Rules that fall out:

1. **The interface is projected, not written.** Shell subcommand, HTTP route,
   SSE frame and typed TS helper are all projections of one descriptor entry.
   Hand-writing a second API surface means drift from the contract — the belt
   being retired.
2. **Commands map 1:1 to descriptor `name` + `params`.** Nothing the shell
   can do is undeclared; nothing declared is unreachable headlessly.
3. **Displays read, never write.** Floor and budget state have exactly one
   writer (the `ServerState` read-model rule). A display that writes has
   stopped being paper (§1.3).
4. **Tool tiers stay uncrossed.** The shell is operator/internal tooling. The
   model gets only what is declared in its own registry (AGENTS tier _a_,
   internal/model-facing) — it never inherits raw shell access, and external
   site-utility tooling never enters this path.
5. **The interface never renders; the display never declares.** Each side's
   one forbidden act.
6. **The tree is parsed once, at the writer; displays walk it.** A display
   renders node kinds through a registry (`mode → renderer`) with an
   attenuated generic fallback for an unknown mode (§1.3) — never a
   per-display switch on strings, never display-side validation.

### 6.1 Enabling gap

`build_router(spec)` — **not implemented yet**: the contract half exists, the
route-emission half does not (AGENTS _Endpoint & Transport Rule_). Until it
exists, every projection is hand-built, so it is the prerequisite in §11, not
a nice-to-have.

### 6.2 Projections

| Projection                        | Consumer              | Note                                                                                                                                                       |
| --------------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| descriptor → model tool-call      | the model             | Internal/model-facing (tier _a_)                                                                                                                           |
| descriptor → shell subcommand     | operator, CI, scripts | The headless path; zero rendering                                                                                                                          |
| descriptor → HTTP route           | remote clients        | `build_router(spec)` output                                                                                                                                |
| descriptor → SSE `phase` frame    | live clients          | Transport, not contract                                                                                                                                    |
| descriptor → typed TS helper      | the web display       | Via `http-client.ts` — single source of truth, no raw `fetch()`                                                                                            |
| response tree → display renderers | humans                | WebUI banner/panel (`useBannerStore` + `<GlobalBanner />`, one global banner, deduped by `key`), TUI pane, mobile card — via the node registry (§6 rule 6) |
| descriptor → docs / tests         | humans, CI            | Derived from the descriptor, never hand-copied                                                                                                             |

Renderers may differ in look ([design/DESIGN_SYSTEM.md](design/DESIGN_SYSTEM.md))
and in which modes they surface — a terminal has no use for some of them — but
they may not differ in meaning, and they may not write.

---

## 7. What the baby's environment requires of the protocol

These are the rules that come from how a new learner actually learns; they
set the constants in §5 rather than forming a separate feature.

| Property of a learning environment        | Rule it produces                                                                                  |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Speech is equipment, not transmission** | `think`/`babble` are never graded and never persisted                                             |
| **Joint attention**                       | Every `probe` is scoped to `topic` and to the surface it is on — no naked questions               |
| **Correction is short and local**         | The reply affordance must accept a word, a ✓/✗, or a pick — `answer` is complete at one keystroke |
| **Mixed initiative**                      | Floor control alternates; the human leads sometimes                                               |
| **Bounded playpen**                       | Budgets in §5.2 are the safety mechanism, not permissions                                         |
| **Mostly ungraded**                       | Only `answer` pairs are recorded as teaching moments                                              |
| **Repair, not silence**                   | One retry, then drop (§5.3)                                                                       |
| **Ritual + variation**                    | Stable pacing and message shape; content varies                                                   |

---

## 8. The optimization loop

The reason for insisting on structure (§4.1): every response is a data point,
so learning can be optimized on evidence instead of impression.

Primary metric:

> **response → human correction → did the next `state` change?**

- Flat delta = feedback is not being absorbed, however polite the answers look.
- Compare the `state` node and the `confidence` field across the pair, not
  prose.

### 8.1 Confidence is a claim, not a measurement

A baby is a poor judge of its own confidence — it shows up as hesitation,
repetition, topic switching rather than as a reliable number. So `confidence`
is recorded and **not trusted**: it is one of the fields compared against what
actually happened, to learn whether the model is miscalibrated. Never let
stated confidence gate display behavior by itself.

---

## 9. Where it hooks (extend, don't duplicate)

| Existing                                                       | Role here                                                                                      |
| -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `ToolSpec` (`domain/agents/_internal/tools.py`)                | Descriptor type; its `result` field references nodes of the response tree                      |
| `soul.personality.curiosity` (`domain/core/_internal/soul.py`) | The existing _disposition_ trait — how readily it probes                                       |
| `domain/core/_internal/prompt_builder.py`                      | Where the learner stance is written into the prompt (it already branches on `curiosity > 0.7`) |
| `domain/context/_internal/managers.py`                         | Already weights `curiosity` among traits                                                       |
| Stream envelope + `streamSSE` ([API.md](API.md))               | Delivery (§5.4) — tree rides inside `data`                                                     |
| `ServerState` read-model (one writer, `AtomicRef`)             | Floor + budget state; parses/validates the tree (§4.1, §6)                                     |
| `apps/web/lib/sse-client.ts`                                   | Existing display-side client                                                                   |
| `useBannerStore` / `<GlobalBanner />`                          | Floor signal in the web display                                                                |

---

## 10. Invariants and non-goals

**Must not change:**

- The persistence invariant (§4.3) — ephemeral branches stay out of history.
- One writer for floor/budget state — displays read only; the interface never
  renders, the display never writes (§6).
- **Parse at the writer, walk in the display** (§6 rule 6) — no display-side
  validation, no per-display switches on mode strings.
- The tool-tier boundary — internal/model-facing only for the model; the
  shell stays operator tooling; the external site-utility tier never enters
  this path.
- One global banner, deduped by `key` — no per-page banners.
- No new endpoint per surface — descriptors only, projections derive from them.
- The transport envelope in [API.md](API.md) — the tree rides inside `data`;
  delivery semantics are not part of this change.

**Non-goals:**

- Not a second chat implementation — modes are a distinct node kind, not chat
  turns in costume.
- Not autonomous unbounded questioning — quotas bound it by design.
- Not long-term memory of every response — only `answer` pairs commit.
- Not an unbounded graph: node _types_ may be shared, but each response
  **instance** is a bounded tree — fixed depth, no cycles.
- Not a mass rename of "frontend"/"interface" across existing docs (§1.4).

---

## 11. Phases

1. **Prerequisite — projection machinery.** `build_router(spec)`: contract
   half exists, route emission does not (§6.1). Until it lands, every
   projection is hand-written and drifts.
2. **Declare the response tree** — grammar, shared `head`, mode branches —
   as `ToolSpec`-shaped descriptors with JSON Schema `$defs`/`$ref`; views
   (wire JSON, TS types, CLI help, docs) are derived, never stored twice.
3. **Give the baby the tool** so responses are structured by construction
   (no client-side text parsing, ever).
4. **Transport** — modes as `phase` values on an existing stream, tree in
   `data`; ephemeral tagging; floor + budget in the read-model with a single
   writer that parses.
5. **The interface** — shell-subcommand projection; a small demo command set,
   headless: commands in, trees out, zero rendering.
6. **First display** — walks the tree through the node registry, cheapest
   possible reply affordance, reads only.
7. **Metric** — record response → correction → `state` delta (§8).
8. **Second display** — proving the contract is genuinely display-independent.
   Phase 8 is the acceptance test for §1.3 and §6.

---

## 12. Open questions

1. **Who types into the shell first** — the model (tool-calls; recommended,
   since the baby is the learner and the interface exists for it to explore),
   you as operator, or tests/CI? All three are projections of the same
   descriptor, so this picks the demo, not the contract.
2. **Which display is first**, and does the first shell command set cover the
   same operations that display needs? The headless path should be exercisable
   before any pixels exist.
3. **What reply affordance will the human really use** — ✓/✗,
   pick-from-three, or free text? If it costs more than a glance, the loop
   dies (§3).
4. **Which modes does each display render?** A terminal and a browser should
   not be forced to show the same set (§6.2).
