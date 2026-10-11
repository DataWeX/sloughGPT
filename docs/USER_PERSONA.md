# User Persona

> **Revised 2026-10-11.** Authority order changed from Alex → Maya → Jon to
> **Jon → Maya → Alex**, and the design principles rewritten.
>
> **Reason:** the previous ordering was never audited against the actual
> operator. It defined Alex as "runs a Mac at home" while the real deployment
> is a Linux host with an NVIDIA GTX 1050, conda environments, systemd user
> services, and manual tuning of the matmul CPU/GPU dispatch threshold. The
> document also contradicted itself: Alex was declared "comfortable with the
> terminal" while design principle 2 forbade the word "LoRA". That is the
> signature of a persona written as fiction and never tested against a real
> session.
>
> A wrong persona is worse than none, because it actively misdirects. Alex's
> rules would have buried the `.soul` path Jon needs to `rsync`, hidden the
> eval detail Maya needs as a CI gate, and renamed terms so that support
> threads and documentation stop resolving. See **Evidence** below for the
> grounding that replaces the guesswork.

---

## Primary: Jon the Infra Builder

**Role:** Owns his model destination, repo, and compute. Builds infra for
himself, frequently on shared hardware.

**Demographics:** 27-42, platform/infra engineer, runs own GPU box or shared
cluster, builds model **destination + repo + infra** for himself on **shared
hardware**.

**What they want:**

- A **model destination & repo** he controls (push/pull `.soul`/`.slnc`, version on his infra, not ours)
- To build **infra for himself** on **shared hardware / platform infra** — `ModelRegistry` + `ProcessGuard` + `TrainingExecutor` + `~/.cache` that survives `data/` deletes, with **compute** isolation (per-workspace `TrainingExecutor`, `get_training_executor().submit`)
- **No magic delete** of the model file when he stacks datasets/knowledge/caches — copy-on-write, `ModelBase` `chmod 444`
- **Competition/compute** visibility: `benchmark/score` programmable + `system/executor` + `metrics` to tune his box

**What they don't want:**

- Vendor-locked model store that decimates his `.soul` on cache clear
- A UI that hides the model path, registry, or executor logs
- `data/` vs `.cache` duality — wants single `Just-Cache` he can `rsync`

**How they interact:**

- 50% infra/repo (`registry`, `model-stack`, `executor`, `cache_tags`), 30% train/bench, 20% API
- Lives in `domain/infrastructure` + `apps/api/server/routers/system.py` + `training/executor`
- Expects `docs/PRODUCT_ENGINEERING.md:42` CCGT — infra is a domain, not plumbing
- Reads benchmark and profiling output directly; tunes thresholds against
  measured crossover points rather than documented defaults

**Success criteria:**

- "I built my repo `models/jon-llama.soul`, set destination `~/.cache/sloughgpt/models`, stacked 3 caches, train still sees model" → success
- "Adding a dataset deleted my model file" → failure
- "My shared hardware shows `executor` + `inference` + `cache` in one `system` view" → success
- "I can't `scp` my model repo off the box" → failure

---

## Secondary: Maya the Platform Builder

**Demographics:** 24-38, codes daily (TS/Python), ships AI features in apps, runs evals in CI, reads HF docs, self-hosts models.

**What they want:**

- A model platform they can **program**: one-time `ModelBase` + stacked `knowledge/dataset/cache` layers that never delete the `.soul` file
- **APIs + SDK** for `Chat/Train/Datasets/Models/Souls/Knowledge/Tools/Benchmark` — not clicks
- **Weighted, programmable bench** like OSBench — `config/bench_weights.yaml` `logic: "0.3*norm_latency + 0.25*recall … | if recall<0.8 then 0"` to gate releases
- To version **model + tokenizer + contract** as one unit, compare checkpoints, export `INT8/INT4`

**What they don't want:**

- A black-box chat UI that hides metrics, logs, or model files
- Magic that deletes the model when a dataset is removed
- Separate `data/` vs `.cache/` confusion — one Just-Cache source of truth

**How they interact:**

- 40% API/SDK, 30% eval/bench, 20% train, 10% chat sanity-check
- Reads `docs/PRODUCT_ENGINEERING.md` CCGT (Cognitive/Core/Gateway/Training) — expects each feature to ship all four
- Wants `benchmark/score` and `model-stack` programmable, not just `benchmark/run`

**Success criteria:**

- "I init the model once, stack 5 knowledge stores, train, bench 87.5, ship — model file untouched" → success
- "I have to re-download the model after adding a dataset" → failure
- "I can gate CI on `bench score > 80`" → success
- "Benchmark is just a perplexity number" → failure

---

## Tertiary (future segment, non-authoritative): Alex the Hobbyist AI Tinkerer

> **Not the current operator.** Alex is a hypothetical downstream consumer of
> this product. His constraints are recorded for future planning and **must
> not override the primary or secondary surfaces.** Where a decision is
> contested, Jon and Maya win; a future Alex-facing layer is additive, sitting
> on top of the same vocabulary and the same exposed paths — never a
> substitution for them.

**Role:** Non-technical owner — model is _theirs_, not a layer stack.

**Demographics:** 28-45, works in tech (but not ML), runs a Mac at home, comfortable with terminal but not a data scientist.

**What they want:**

- A personal AI they can train on their own data (Shakespeare, their journal, a niche hobby forum)
- The AI should feel like _theirs_ — different personality from ChatGPT, trained on stuff they care about
- Simple workflows: pick data → click train → use it
- To understand what's happening without needing a PhD

**What they don't want:**

- 15-parameter configuration forms presented without defaults
- To feel stupid when using their own tool
- To be handed raw `final_loss=1.2345` with no interpretation attached

**How they interact:**

- 80% chat, 10% check if training worked, 10% tweak settings when curious
- They'll read a tooltip or two, but won't read documentation
- They want to _see_ results (loss went down, personality changed) as well as numbers

**Success criteria:**

- "I trained an AI on my notes and it talks like me" → success
- "It took 3 clicks to train" → success
- "I had to read a README to understand the training page" → failure

---

## Evidence (why the order changed)

A persona that governs design must be falsifiable, so here is what the
repository actually demonstrates about its operator. Each item is checkable:

| Observation                                                                                                     | Implies                                       |
| --------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| Host runs Linux with an NVIDIA GTX 1050 (3072 MiB), CuPy 14.2.0                                                 | Not a Mac-at-home user; owns real GPU compute |
| Work proceeds through conda envs, `PYTHONNOUSERSITE=1`, pinned interpreters                                     | Comfortable in a terminal at engineer depth   |
| The matmul CPU/GPU dispatch threshold was re-measured against the hardware rather than accepted from a constant | Tunes against measurement, not documentation  |
| Infra work spans systemd user services, Buildroot images, a custom kernel, and v86 browser Linux                | Platform/infra engineer — Jon's definition    |
| The standing instruction is to run benchmarks after every training/inference change                             | Eval-gated workflow — Maya's definition       |

The earlier document described none of these. It is retained below only as a
future market segment.

---

## Feature Visibility Matrix

Authority runs **left to right**: Jon and Maya are binding, Alex is advisory.

| Feature                                                                    | Jon (Infra)                     | Maya (Builder)                 | Alex (future, advisory) |
| -------------------------------------------------------------------------- | ------------------------------- | ------------------------------ | ----------------------- |
| **Chat**                                                                   | ✅ Core (API)                   | ✅ Core (via API)              | ✅ Core                 |
| **Soul/personality switcher**                                              | ✅ Core (API)                   | ✅ Core (API)                  | ✅ Core                 |
| **Dataset selector**                                                       | ✅ Core (SDK)                   | ✅ Core (SDK)                  | ✅ Core                 |
| **Train button**                                                           | ✅ API `/training` + `executor` | ✅ API `/training`             | ✅ Core                 |
| **Loss chart**                                                             | ✅ API + bench                  | ✅ API + bench                 | ✅ After training       |
| **Checkpoint catalog**                                                     | ✅ API                          | ✅ API                         | ✅ After training       |
| **Eval results** (coherence, repetition)                                   | ✅ Weighted score + compute     | ✅ Weighted score 0-100        | ✅ Plain verdict        |
| **Benchmark weighted** (`bench_weights.yaml` + `POST /benchmark/score`)    | ✅ Core (compute gate)          | ✅ Core (CI gate)              | ❌ Hidden               |
| **ModelStack** (`POST /model-stack/push`) — one-time model, stacked layers | ✅ Core (destination + repo)    | ✅ Core (never delete `.soul`) | ❌ Hidden               |
| **Just-Cache** (`~/.cache/sloughgpt/external`)                             | ✅ Core (rsync)                 | ✅ Core                        | ❌ Hidden               |
| **Model destination/repo** (push/pull `.soul`/`.slnc`)                     | ✅ Core (own infra)             | 🔧 Power                       | ❌ Hidden               |
| **Shared hardware / platform infra** (`/system`, `executor`, `registry`)   | ✅ Core                         | ⚠️ Advanced                    | ❌ Hidden               |
| **Compute isolation** (`TrainingExecutor`, `ProcessGuard`)                 | ✅ Core                         | ⚠️ Advanced                    | ❌ Hidden               |
| **Config save/load**                                                       | ✅ Core                         | ✅ Core                        | 🔧 Power user           |
| **Dataset stats**                                                          | ✅ Core                         | ✅ Core                        | 🔧 Power user           |
| **Checkpoint comparison**                                                  | ✅ Core                         | ✅ Core                        | 🔧 Power user           |
| **RL / GRPO**                                                              | ⚠️ Advanced (preset)            | ⚠️ Advanced (preset)           | ❌ Hidden               |
| **KL coefficient**                                                         | ⚠️ Advanced                     | ⚠️ Advanced                    | ❌ Hidden               |
| **LoRA rank/alpha**                                                        | ⚠️ Advanced                     | ⚠️ Advanced                    | ❌ Hidden (slider)      |
| **Gradient accumulation**                                                  | ⚠️ Advanced                     | ⚠️ Advanced                    | ❌ Hidden               |
| **Reward function mode**                                                   | ⚠️ Advanced                     | ⚠️ Advanced                    | ❌ Hidden               |
| **Warmup steps**                                                           | ⚠️ Advanced                     | ⚠️ Advanced                    | ❌ Hidden               |
| **Max sequence length**                                                    | ⚠️ Advanced                     | ⚠️ Advanced                    | ❌ Hidden               |
| **Learning rate**                                                          | ✅ Tunable                      | ✅ Tunable                     | ⚠️ Advanced only        |
| **Batch size**                                                             | ✅ Tunable                      | ✅ Tunable                     | ⚠️ Advanced only        |

---

## Design Principles (derived from persona)

The supersession matters: principles 2 and 4 of the earlier revision were
**substitutions** — they replaced standard terminology with private euphemisms
("training quality" for LoRA, "personality strength" for reward mode). They are
withdrawn below and replaced with **layering**, because terminology is a
lookup key. `LoRA` resolves to a global body of documentation, troubleshooting,
and community knowledge; `training quality` resolves to nothing. A renamed
concept severs the user from the entire support graph around it, and the
renaming buys no simplicity — the underlying parameter is still there, just
unfindable.

1. **One-click training** — Default settings should work for 90% of cases. No form filling required. _(unchanged)_
2. **Standard terms, explained — never renamed** — Use the real vocabulary ("LoRA rank", "GRPO", "KL coefficient") as the primary label, each paired with a plain-language gloss on the same surface: _"LoRA rank — how much new style the adapter can learn (64)"_. Defaults do the work, not euphemisms. Euphemistic renaming is prohibited: it breaks searchability, documentation parity, and error triage.
3. **Progressive disclosure is additive, not substitutive** — Show results first (did it learn?), details on click (loss chart, metrics), expert settings behind a toggle. The summary layer sits _above_ the technical layer; it never replaces it.
4. **Name the outcome AND the mechanism** — _"Shakespeare training complete — adapter saved (12 MB), coherence +8%"_, not `final_loss=1.2345` alone. Both halves ship: the outcome for the glance, the mechanism for the trace.
5. **Visual, not numerical** — Loss chart with down arrow = good. Number "1.2345" is meaningless alone. _(unchanged, but it complements principle 4 — it does not replace the number, it supplements it)_
6. **Paths and logs are never hidden** — Model destination, registry entries, executor logs, and checkpoint locations stay visible or one click away. This is a hard constraint from the primary persona and outranks any readability preference.
7. **The casual surface is additive** — Any future Alex-facing simplification layers on top of the same exposed paths and the same vocabulary. It must not fork the information model or introduce a second naming scheme.
8. **Vocabulary is a shared namespace** — API field names, schema keys, error text, CLI output, and benchmark verdicts use canonical terms verbatim. Users paste error text into search engines; renaming it breaks triage. Documentation-parity gates (`scripts/check_docs_api_parity.py`) detect this class of drift and must stay at zero findings.
9. **Transparency on demand** — Dig into settings when curious, without ever being required to. _(unchanged)_
