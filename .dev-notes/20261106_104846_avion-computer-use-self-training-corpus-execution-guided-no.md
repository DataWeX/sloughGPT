---
id: 20261106_104846_avion-computer-use-self-training-corpus-execution-guided-no
status: todo
board: 20261106_104846_avion-computer-use-self-training-corpus-execution-guided-no
title: "avion: computer-use self-training corpus — execution-guided, no third-party data"
tags: ["avion", "training", "sloNet", "area:training"]
sprint: ""
gh: null
---

# avion: computer-use self-training corpus — execution-guided, no third-party data

Agreed shape (this session): improve sloNet for computer use by training on our OWN trajectories, filtered by an oracle that is not the model. No scraped screenshot-to-action sets (OSWorld/AITW/SeeClick class) — they bake in someone else's UI vocabulary.

Plan:
1. Freeze and version the contract: lettered prompt + axtree observation spec -> one hash, stamped into every sample. Provenance is the invariant; unhashed corpora silently mix prompt formats.
2. Trajectory recorder in the existing ChoiceAgent.run loop -> JSONL per step: contract_hash, task, ledger (letter to label), raw_reply, executed observation, goal_pre, goal_post, stop_reason. Append-only, no behaviour change.
3. Admissibility filter lives at RECORDING time, not in the trainer: keep a step only if the verifier flipped false->true AND the reply resolved via the ledger AND the run was not pre-held-satisfied. ALREADY_SATISFIED is the same oracle, already built.
4. Preference pairs for free from the negative side: a pick that errored vs the pick that replaced it.
5. Dataset adapter feeding the EXISTING SloughGPTTrainer / SloChatTrainer mode. No fourth training loop (core rule). Loss masked to the answer span only, never the echo.
6. Gate on task success rate over a fixed exercise set with fixed start states, not loss. Reuse the own-tokens gap benchmark (49e170a39).

OPEN QUESTIONS before code:
- Base checkpoint: user says NOT externally trained, so one of ours. Which .soul? (checkpoints/qt_1793814797.soul, data/learner/continual.soul, or the own-docs corpus run?)
- Do exercises come only from this app's pages, or is a seeded sandbox UI allowed? Changes what 'no third-party data' means.
- Full finetune or LoRA.

Known failure mode to design against: coverage. Self-play explores one page; without a curriculum spanning ux/ journeys the model gets good at one exercise, and gets good at picking goals it already satisfies.
