---
id: 20261114_113921_rfe-consciousness-in-the-model-weights-build-out-personality
status: todo
board: 20261114_113921_rfe-consciousness-in-the-model-weights-build-out-personality
title: "RFE: consciousness in the model weights — build out personality engine & train it (bigger base model near-term)"
tags: ["research", "consciousness", "model"]
sprint: ""
gh: null
---

# RFE: consciousness in the model weights — build out personality engine & train it (bigger base model near-term)

USER-VISION (2026-11-14, surfaced by card 20261114_092948): consciousness features should influence a layer of the model itself and always be present/perceived — NOT via prompt injection. Injection was probe-proven to hijack open questions on the 0.5B regardless of position/framing/directives (8 prompt variants), so injection is the wrong layer. USER'S ROUTING: (a) NEAR-TERM = bigger base model — 0.5B (Qwen2.5-0.5B-Instruct, 0.49B params) lacks the instruction-following strength to treat in-context self-narrative as background; larger instruct models obey directives the 0.5B cannot. (b) LATER (if we can) = build out the personality engine and train it into the weights (LoRA/fine-tune on self-report pairs) — the pieces probably already exist SCATTERED across the fragmented codebase (archaeology pass first: consciousness trainer routes + pairs collection + LoRA config lora_rank=4 + training_enabled flag + domain/training infra), per the no-duplicate-training-loops rule — adapter into SloughGPTTrainer, never a new loop. (c) Retire the prompt-injection path (ConsciousnessProcessor) once weights carry the cognition. Eval harness: reuse this card's probe methodology (ELI5/2+2/Hello × flag on/off × ctx/rag).
