---
id: 20261102_032256_fixchat-meta-weight-saturation-drives-generation-into-repeti
status: review
board: 20261102_032256_fixchat-meta-weight-saturation-drives-generation-into-repeti
title: "fix(chat): meta-weight saturation drives generation into repetition loop"
tags: ["cleanup", "chat", "generation", "feedback", "root-cause", "area:web"]
sprint: ""
gh: null
---

# fix(chat): meta-weight saturation drives generation into repetition loop

ROOT CAUSE (proven on live data): data/feedback.db user_meta_weights for 'default' accumulated 473 thumbs_up / 10 thumbs_down since Aug 16 -> temperature_boost +9.26, top_k_boost -450, repetition_boost -9.26. get_adjustment() clamped straight to each param's hard range, pinning temperature 1.5 / top_k 5 / repetition_penalty 0.80 (<1.0 = rewards repeats). Web chat sends only temperature+max_new_tokens, so top_k/top_p/repetition_penalty took the nudge -> 5 candidate tokens + pro-repetition penalty -> guaranteed loop until the max_new_tokens cap.

EVIDENCE: session chat_j7ya9c3b = exactly 200 tokens ending mid-loop vs good turn chat_9avguc1i (110 tokens); warning 'extreme values temp=1.50 top_k=5' logged 16:49:22 = same second as the looping POST /chat (req 795bf387).

SHIPPED (branch fix/meta-weight-band, commit 472565281, 1 commit off origin/main): _SAFE_BAND tight neutral+/-band window applied BEFORE the hard clamp + repetition_penalty floor 0.8 -> 1.0. Measured: temp 1.5->1.1, rep 0.80->1.0, top_k 5->10.

SCOPE CORRECTION (2026-11-01): an _GEN_SAFE guard at the _apply_meta_weights seam was drafted then DROPPED -- origin/main already fixed that seam on Oct 23 (3384d08dd, card 1e13f836, _META_WEIGHT_BOUNDS + drift-gated test_meta_weights_clamp.py) with a different, schema-not-generation concern. Landing a second bounds dict there contradicted a landed contract. The residual duplication is carded: 20261101_182428_refactorinference-reconcile-the-two-meta-weight-bound-sets-a.

GATES: 211 passed (incl. pre-existing test_meta_weights_clamp.py 4), ruff clean, benchmark_meta_weights hit p50 3.82us / miss p50 56.17us -- no regression.

(Recreated on the canonical board 2026-11-01: the original row only existed in the abandoned branch fix/meta-weight-saturation's committed .kanban/board.jsonl.)
