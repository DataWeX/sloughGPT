# User Persona

## Primary: Alex the Hobbyist AI Tinkerer

**Role:** Non-technical owner — model is _theirs_, not a layer stack.

**Demographics:** 28-45, works in tech (but not ML), runs a Mac at home, comfortable with terminal but not a data scientist.

**Demographics:** 28-45, works in tech (but not ML), runs a Mac at home, comfortable with terminal but not a data scientist.

**What they want:**

- A personal AI they can train on their own data (Shakespeare, their journal, a niche hobby forum)
- The AI should feel like _theirs_ — different personality from ChatGPT, trained on stuff they care about
- Simple workflows: pick data → click train → use it
- To understand what's happening without needing a PhD

**What they don't want:**

- To learn what KL divergence, LoRA rank, or GRPO means
- 15-parameter configuration forms
- Jargon in the UI ("RLHF", "nucleus sampling", "gradient accumulation")
- To feel stupid when using their own tool

**How they interact:**

- 80% chat, 10% check if training worked, 10% tweak settings when curious
- They'll read a tooltip or two, but won't read documentation
- They want to _see_ results (loss went down, personality changed) not numbers (loss=1.2345)

**Success criteria:**

- "I trained an AI on my notes and it talks like me" → success
- "I don't know what half these settings do" → failure
- "It took 3 clicks to train" → success
- "I had to read a README to understand the training page" → failure

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

## Feature Visibility Matrix

| Feature                                                                    | Alex (Hobbyist)                   | Maya (Builder)                 |
| -------------------------------------------------------------------------- | --------------------------------- | ------------------------------ |
| **Chat**                                                                   | ✅ Core                           | ✅ Core (via API)              |
| **Soul/personality switcher**                                              | ✅ Core                           | ✅ Core (API)                  |
| **Dataset selector**                                                       | ✅ Core                           | ✅ Core (SDK)                  |
| **Train button**                                                           | ✅ Core                           | ✅ API `/training`             |
| **Loss chart**                                                             | ✅ After training                 | ✅ API + bench                 |
| **Checkpoint catalog**                                                     | ✅ After training                 | ✅ API                         |
| **Eval results** (coherence, repetition)                                   | ✅ After training (plain verdict) | ✅ Weighted score 0-100        |
| **Benchmark weighted** (`bench_weights.yaml` + `POST /benchmark/score`)    | ❌ Hidden                         | ✅ Core (CI gate)              |
| **ModelStack** (`POST /model-stack/push`) — one-time model, stacked layers | ❌ Hidden                         | ✅ Core (never delete `.soul`) |
| **Just-Cache** (`~/.cache/sloughgpt/external`)                             | ❌ Hidden                         | ✅ Core                        |
| **Config save/load**                                                       | 🔧 Power user                     | ✅ Core                        |
| **Dataset stats**                                                          | 🔧 Power user                     | ✅ Core                        |
| **Checkpoint comparison**                                                  | 🔧 Power user                     | ✅ Core                        |
| **RL / GRPO**                                                              | ❌ Hidden                         | ⚠️ Advanced (preset)           |
| **KL coefficient**                                                         | ❌ Hidden                         | ⚠️ Advanced                    |
| **LoRA rank/alpha**                                                        | ❌ Hidden (slider)                | ⚠️ Advanced                    |
| **Gradient accumulation**                                                  | ❌ Hidden                         | ⚠️ Advanced                    |
| **Reward function mode**                                                   | ❌ Hidden                         | ⚠️ Advanced                    |
| **Warmup steps**                                                           | ❌ Hidden                         | ⚠️ Advanced                    |
| **Max sequence length**                                                    | ❌ Hidden                         | ⚠️ Advanced                    |
| **Learning rate**                                                          | ⚠️ Advanced only                  | ✅ Tunable                     |
| **Batch size**                                                             | ⚠️ Advanced only                  | ✅ Tunable                     |

---

## Design Principles (derived from persona)

1. **One-click training** — Default settings should work for 90% of cases. No form filling required.
2. **Jargon-free UI** — Never say "LoRA", "GRPO", "KL coefficient" in the UI. Say "training quality", "personality strength", "learning style".
3. **Progressive disclosure** — Show results first (did it learn?), details on click (loss chart, metrics), expert settings behind a toggle.
4. **Plain language** — "Training complete! Your AI now knows Shakespeare" not "Fine-tune job hf_job_3 completed with final_loss=1.2345"
5. **Visual, not numerical** — Loss chart with down arrow = good. Number "1.2345" = meaningless.
6. **Transparency on demand** — Alex can dig into settings if curious, but shouldn't have to.
