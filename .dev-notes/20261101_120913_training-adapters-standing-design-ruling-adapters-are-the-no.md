---
id: 20261101_120913_training-adapters-standing-design-ruling-adapters-are-the-no
status: done
board: 20261101_120913_training-adapters-standing-design-ruling-adapters-are-the-no
title: "Training adapters: standing design ruling (adapters are the noun; BatchArray seam; automatic composition gate via AML)"
tags: ["training", "adapters", "design"]
sprint: ""
gh: null
---

# Training adapters: standing design ruling (adapters are the noun; BatchArray seam; automatic composition gate via AML)

Supersedes parked card 63116e83 (gone from board). ADAPTERS ARE THE NOUN: per-training-task artifacts (text-pretrain, tts-espeak, image-caption, chat-sft), never per-source. Sources = 4 closed config kinds (local/http/kaggle/shell operator-tier only); config names components, never contains them; new source = one YAML line, zero rebuild. TRAINER SEAM: BatchArray protocol (re-iterable, sized, shufflable — map-style Dataset contract, NOT a one-shot iterator: epochs+shuffle require collection semantics); every adapter implements it; adapters never import the loop; exactly ONE seam on SloughGPTTrainer. COMPOSITION GATE: AUTOMATIC, not on-demand — three layers: (a) fail-fast at registry boot (broken adapter never registers), (b) static parametrized pytest over the whole registry asserting declared in/out contracts compose (test_command_parity.py shape), (c) one dynamic synthetic smoke: tiny fixture batch through the real AML pipeline (dedup -> quality -> snapshot -> BatchArray) proving runtime shape. AML cruncher pipeline is the canonical flow passing through the gate (dogfood). 100% registry coverage from day one, no grandfather baseline; rides affected-closure loop AND every full-CI run (seam gate, Test Rule clause 4); never requires real credentials/datasets — synthetic only; ONE parametrized gate, not per-adapter tests. SCOPE: weights training only (knowledge store out per parked ruling); espeak_pairs grandfathered until BatchArray byte-parity; spine builds on the FIRST real training task, not before; sources get their 4 kinds at that same trigger; modality payloads live inside the batch, not the contract. AML import->clean->snapshot design rides here.
