# User Persona

## Primary: Alex the Hobbyist AI Tinkerer

**Demographics:** 28-45, works in tech (but not ML), runs a Mac at home, comfortable with terminal but not a data scientist.

**What they want:**
- A personal AI they can train on their own data (Shakespeare, their journal, a niche hobby forum)
- The AI should feel like *theirs* — different personality from ChatGPT, trained on stuff they care about
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
- They want to *see* results (loss went down, personality changed) not numbers (loss=1.2345)

**Success criteria:**
- "I trained an AI on my notes and it talks like me" → success
- "I don't know what half these settings do" → failure
- "It took 3 clicks to train" → success
- "I had to read a README to understand the training page" → failure

---

## Feature Visibility Matrix

| Feature | Alex needs to see? | Why / why not |
|---------|-------------------|---------------|
| **Chat** | ✅ Core | This is why they're here |
| **Soul/personality switcher** | ✅ Core | "My AI has a personality" — this is the magic |
| **Dataset selector** | ✅ Core | They need to pick what to train on |
| **Train button** | ✅ Core | One-click start |
| **Loss chart** | ✅ After training | They want to see "it learned something" |
| **Checkpoint catalog** | ✅ After training | They want to switch between trained versions |
| **Eval results** (coherence, repetition) | ✅ After training | "Is it any good?" — plain language verdict |
| **Config save/load** | 🔧 Power user | Nice to have, not blocking |
| **Dataset stats** | 🔧 Power user | Helps pick the right dataset |
| **Checkpoint comparison** | 🔧 Power user | When they have multiple trained versions |
| **RL / GRPO** | ❌ Hidden | ML researcher territory — hide behind "advanced" toggle or remove from UI |
| **KL coefficient** | ❌ Hidden | Meaningless to hobbyists |
| **LoRA rank/alpha** | ❌ Hidden | Auto-set, expose as "training quality" slider only |
| **Gradient accumulation** | ❌ Hidden | Implementation detail |
| **Reward function mode** | ❌ Hidden | If RL is exposed at all, preset it |
| **Warmup steps** | ❌ Hidden | Auto-set based on dataset size |
| **Max sequence length** | ❌ Hidden | Auto-set, rarely needs changing |
| **Learning rate** | ⚠️ Advanced only | Only show in "expert mode" |
| **Batch size** | ⚠️ Advanced only | Only show in "expert mode" |

---

## Design Principles (derived from persona)

1. **One-click training** — Default settings should work for 90% of cases. No form filling required.
2. **Jargon-free UI** — Never say "LoRA", "GRPO", "KL coefficient" in the UI. Say "training quality", "personality strength", "learning style".
3. **Progressive disclosure** — Show results first (did it learn?), details on click (loss chart, metrics), expert settings behind a toggle.
4. **Plain language** — "Training complete! Your AI now knows Shakespeare" not "Fine-tune job hf_job_3 completed with final_loss=1.2345"
5. **Visual, not numerical** — Loss chart with down arrow = good. Number "1.2345" = meaningless.
6. **Transparency on demand** — Alex can dig into settings if curious, but shouldn't have to.
